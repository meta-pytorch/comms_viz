// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! No-DOM merged-trace artifact for inline-annotated JSON: splice the row
//! walk's edits into the original document bytes.
//!
//! The serde-tree pipeline existed to serialize the merged-trace artifact —
//! on a 10240-rank JSON job it was ~90% of the analyze window's accounted
//! CPU (tree_parse + tree_walk + tree_ser + convert), and its ~6-15x
//! document-size allocation footprint is what forces the memory gate to
//! throttle concurrency on small-budget hosts. But the inline
//! (no-reassign) walk only ever makes four kinds of edit:
//!
//!   - drop noise events (cleanup's two retain passes)      → skip spans
//!   - reassign an overlapping graphed kernel's stream      → rewrite the
//!     `tid` and `args.stream` scalars
//!   - clamp a small ts overlap                             → rewrite `ts`
//!   - append synthetic `thread_name` metadata events       → emit at the end
//!
//! So: the row walk (`cudagraph::apply_walks`) reports which rows it
//! touched; one structural scan of the pristine document records every
//! element's byte span, with the `ts`/`tid`/`args.stream` value spans
//! extracted only for the touched rows ([`scan_sparse`]; the verify oracle
//! uses the full [`scan`]); and [`splice`] emits the artifact by copying
//! original bytes — minified: whitespace outside strings is dropped
//! ([`copy_min`], SIMD on AVX-512 hosts), else the producer's
//! pretty-printing made the artifact ~13% larger than the compact tree
//! serialization — and substituting only the touched scalars. A plan that
//! touches nothing skips the scan and splice outright ([`minify_gzip`]).
//! Element order, key order and number formatting of untouched events are
//! preserved (the tree path re-serialized with sorted keys — semantically
//! identical JSON, different bytes; byte-parity is retired, comms_viz-vs-
//! comms_viz determinism holds because the output is a pure function of the
//! input bytes).
//!
//! The pkl side channel (reassigning) adds two more edit classes, both
//! spliceable:
//!
//!   - retarget a graphed kernel to its annotation's stream → rewrite
//!     `tid`, rebuild the `args` object (annotation keys upserted as
//!     strings, `args.stream` overwritten with the stream number) — the
//!     [`SplicePlan::retagged`] rows, args via [`SplicePlan::ann_args`];
//!   - retarget the matching `ac2g` flow events                → rewrite
//!     `tid` only ([`SplicePlan::retagged_ac2g`]).
//!
//! Only the recording merge (arbitrary `args` rewrites from a second
//! trace) keeps the tree path. Any scan/shape surprise falls back to the
//! tree path per rank — this module must never be the reason a rank
//! fails.
//!
//! Row alignment invariant: the JSON rows scan pushes one row per OBJECT
//! element of the winning `traceEvents` array (non-objects are skipped,
//! duplicate `traceEvents` keys resolve last-wins). The scanner applies
//! the same two rules, and [`splice`] cross-checks its object count
//! against the walk's row count before touching anything.

use anyhow::Result;
use anyhow::bail;

/// One `traceEvents` element. Non-objects carry only the span.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct EvSpan {
    /// Element bytes `[start, end)`, whitespace-trimmed.
    pub start: u32,
    pub end: u32,
    pub is_object: bool,
    /// Object has at least one key (drives comma placement on inserts).
    pub has_keys: bool,
    /// `ts` value span + "integer literal" (no `.`/`e`, numeric).
    pub ts: Option<(u32, u32, bool)>,
    /// `tid` value span.
    pub tid: Option<(u32, u32)>,
    /// `args` object's `{` index, when args is an object.
    pub args_open: Option<u32>,
    pub args_has_keys: bool,
    /// `args.stream` value span.
    pub stream: Option<(u32, u32)>,
}

pub struct DocSpans {
    pub events: Vec<EvSpan>,
    /// `[` of the winning traceEvents array.
    pub arr_open: u32,
    /// `]` of the winning traceEvents array.
    pub arr_close: u32,
    pub n_objects: u32,
}

/// Full per-element spans of the winning traceEvents array, captured by the
/// walked JSON parse (`json::parse_bytes_walked_bounds`) while it traverses
/// those exact bytes anyway. The capture fills every field `scan_event`
/// would, so the splice consumes it directly — the second structural pass
/// (scan_event re-walking each edited event, ~1.1 s/rank of skip_value/
/// read_key on the 09-02 24k-rank profile) is gone entirely.
///
/// One deliberate divergence from `scan_event`: the parse matches keys
/// unescaped and types the ts literal by its parsed token (huge integers
/// overflow to f64), both of which follow the serde tree — the scanner's
/// raw-byte versions were the odd ones out.
pub struct ElemSpans {
    /// One span per element, whitespace-trimmed, document order.
    pub events: Vec<EvSpan>,
    pub arr_open: u32,
    pub arr_close: u32,
    /// Object-element count (the splice's row-alignment cross-check).
    pub n_objects: u32,
}

impl From<ElemSpans> for DocSpans {
    fn from(e: ElemSpans) -> Self {
        DocSpans {
            events: e.events,
            arr_open: e.arr_open,
            arr_close: e.arr_close,
            n_objects: e.n_objects,
        }
    }
}

/// Rendered annotation args for the reassign splice: gnid → (key, value)
/// pairs already `python_str`-rendered and folded across the gnid's
/// annotation list (later dicts win), `stream` excluded — the retag
/// writes it as a number, mirroring the tree twin's final overwrite.
pub type RenderedAnnArgs = rustc_hash::FxHashMap<i64, Vec<(String, String)>>;

/// The row walk's artifact-relevant edits (see `cudagraph::WalkOutcome`).
pub struct SplicePlan<'a> {
    pub keep: &'a [bool],
    /// Row indices reassigned to `overlap_stream`, ascending.
    pub moved: &'a [u32],
    /// (row index, new ts), ascending by row.
    pub fixed: &'a [(u32, f64)],
    pub appended: &'a [crate::cudagraph::MetaRow],
    pub overlap_stream: i64,
    /// Reassign-mode graphed rows: (row, gnid, final stream), ascending.
    pub retagged: &'a [(u32, i64, i64)],
    /// Reassign-mode ac2g partners: (row, final stream), ascending.
    pub retagged_ac2g: &'a [(u32, i64)],
    pub ann_args: &'a RenderedAnnArgs,
}

// ---------------------------------------------------------------------------
// Structural scan.
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn skip_ws(d: &[u8], mut i: usize) -> usize {
    while i < d.len() && matches!(d[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

/// `i` at the opening quote; returns the index just past the closing quote.
/// memchr2 jumps over string content instead of walking it byte by byte —
/// the scalar loop here was the analyzer's single hottest instruction path
/// (skip_string+skip_value = 22% of shard cycles on the 08-16 profile).
pub(crate) fn skip_string(d: &[u8], i: usize) -> Result<usize> {
    debug_assert_eq!(d[i], b'"');
    let mut j = i + 1;
    while j < d.len() {
        match memchr::memchr2(b'"', b'\\', &d[j..]) {
            Some(k) if d[j + k] == b'"' => return Ok(j + k + 1),
            Some(k) => j += k + 2, // step past the escape pair
            None => break,
        }
    }
    bail!("unterminated string at {i}");
}

/// `i` at the first byte of a value; returns the index just past it.
pub(crate) fn skip_value(d: &[u8], i: usize) -> Result<usize> {
    match d.get(i) {
        Some(b'"') => skip_string(d, i),
        Some(b'{') | Some(b'[') => {
            let mut depth = 0usize;
            let mut j = i;
            while j < d.len() {
                match d[j] {
                    b'"' => j = skip_string(d, j)?,
                    b'{' | b'[' => {
                        depth += 1;
                        j += 1;
                    }
                    b'}' | b']' => {
                        depth -= 1;
                        j += 1;
                        if depth == 0 {
                            return Ok(j);
                        }
                    }
                    _ => j += 1,
                }
            }
            bail!("unterminated container at {i}");
        }
        Some(_) => {
            // number / true / false / null: run to a structural delimiter.
            let mut j = i;
            while j < d.len() && !matches!(d[j], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
            {
                j += 1;
            }
            if j == i {
                bail!("empty value at {i}");
            }
            Ok(j)
        }
        None => bail!("eof at value start"),
    }
}

/// Read the key at `i` (opening quote) into the doc slice; return
/// (key bytes, index past the `:`).
pub(crate) fn read_key(d: &[u8], i: usize) -> Result<(&[u8], usize)> {
    let end = skip_string(d, i)?;
    let key = &d[i + 1..end - 1];
    let j = skip_ws(d, end);
    if d.get(j) != Some(&b':') {
        bail!("expected ':' after key at {j}");
    }
    Ok((key, skip_ws(d, j + 1)))
}

/// Is this number value an integer literal (serde parses it as i64/u64)?
/// Strings/containers/keywords → false, matching the tree's
/// `v.is_i64() || v.is_u64()`.
fn int_literal(v: &[u8]) -> bool {
    if v.is_empty() || !(v[0] == b'-' || v[0].is_ascii_digit()) {
        return false;
    }
    v.iter().all(|&b| b == b'-' || b.is_ascii_digit())
}

/// Scan one event object starting at `i` (its `{`); fill `ev`.
/// Duplicate keys: last occurrence wins, same as the serde tree.
fn scan_event(d: &[u8], i: usize, ev: &mut EvSpan) -> Result<usize> {
    let mut j = skip_ws(d, i + 1);
    loop {
        match d.get(j) {
            Some(b'}') => return Ok(j + 1),
            Some(b'"') => {}
            _ => bail!("expected key or '}}' at {j}"),
        }
        ev.has_keys = true;
        let (key, vi) = read_key(d, j)?;
        let ve = skip_value(d, vi)?;
        match key {
            b"ts" => ev.ts = Some((vi as u32, ve as u32, int_literal(&d[vi..ve]))),
            b"tid" => ev.tid = Some((vi as u32, ve as u32)),
            b"args" => {
                if d[vi] == b'{' {
                    ev.args_open = Some(vi as u32);
                    ev.args_has_keys = false;
                    ev.stream = None;
                    let mut aj = skip_ws(d, vi + 1);
                    loop {
                        match d.get(aj) {
                            Some(b'}') => break,
                            Some(b'"') => {}
                            _ => bail!("expected args key or '}}' at {aj}"),
                        }
                        ev.args_has_keys = true;
                        let (ak, avi) = read_key(d, aj)?;
                        let ave = skip_value(d, avi)?;
                        if ak == b"stream" {
                            ev.stream = Some((avi as u32, ave as u32));
                        }
                        aj = skip_ws(d, ave);
                        if d.get(aj) == Some(&b',') {
                            aj = skip_ws(d, aj + 1);
                        }
                    }
                } else {
                    // non-object args: treated as absent, same as the tape.
                    ev.args_open = None;
                    ev.stream = None;
                }
            }
            _ => {}
        }
        j = skip_ws(d, ve);
        if d.get(j) == Some(&b',') {
            j = skip_ws(d, j + 1);
        }
    }
}

/// Scan the elements of a `traceEvents` array whose `[` is at `arr_open`;
/// returns (element spans, index of the `]`, object count).
///
/// `detail`: sorted object-row indices whose inner value spans
/// (`ts`/`tid`/`args.stream`, `has_keys`) the caller will consult — the
/// plan's moved/fixed rows. Rows outside the set get the cheap boundary
/// walk (`skip_value`): the splice only ever copies them whole. `None`
/// details every row (the verify oracle's shape).
fn scan_events_array(
    doc: &[u8],
    arr_open: usize,
    detail: Option<&[u32]>,
) -> Result<(Vec<EvSpan>, usize, u32)> {
    let mut events = Vec::new();
    let mut n_objects = 0u32;
    let mut di = 0usize;
    let mut j = skip_ws(doc, arr_open + 1);
    loop {
        match doc.get(j) {
            Some(b']') => return Ok((events, j, n_objects)),
            Some(_) => {}
            None => bail!("unterminated traceEvents array at {arr_open}"),
        }
        let mut ev = EvSpan {
            start: j as u32,
            ..Default::default()
        };
        let end = if doc[j] == b'{' {
            ev.is_object = true;
            let row = n_objects;
            n_objects += 1;
            let detailed = match detail {
                None => true,
                Some(rows) => {
                    while rows.get(di).is_some_and(|&r| r < row) {
                        di += 1;
                    }
                    rows.get(di) == Some(&row)
                }
            };
            if detailed {
                scan_event(doc, j, &mut ev)?
            } else {
                skip_value(doc, j)?
            }
        } else {
            skip_value(doc, j)?
        };
        ev.end = end as u32;
        events.push(ev);
        j = skip_ws(doc, end);
        if doc.get(j) == Some(&b',') {
            j = skip_ws(doc, j + 1);
        }
    }
}

/// Structural scan: locate the winning (last) root-level `traceEvents`
/// array and record every element's spans. The array's elements are
/// scanned in the same pass that finds it — the old shape byte-walked the
/// whole array once just to find its `]` (skip_value on ~99% of the
/// document), then walked it again for spans. Last-wins on duplicate root
/// keys, like the tape flavor; a malformed earlier duplicate now fails the
/// scan instead of being skipped, which lands in the per-rank tree
/// fallback (same artifact, slower) rather than changing any output.
pub fn scan(doc: &[u8]) -> Result<DocSpans> {
    scan_impl(doc, None)
}

/// [`scan`] detailing only the given sorted object-row indices — the rows
/// the splice plan actually edits. Other rows carry only their boundary
/// span, which is all a whole-event copy needs.
pub fn scan_sparse(doc: &[u8], detail: &[u32]) -> Result<DocSpans> {
    scan_impl(doc, Some(detail))
}

fn scan_impl(doc: &[u8], detail: Option<&[u32]>) -> Result<DocSpans> {
    if doc.len() > u32::MAX as usize {
        bail!("document too large for span indices");
    }
    let mut i = skip_ws(doc, 0);
    if doc.get(i) != Some(&b'{') {
        bail!("root is not an object");
    }
    i = skip_ws(doc, i + 1);
    let mut found: Option<(usize, usize, Vec<EvSpan>, u32)> = None;
    loop {
        match doc.get(i) {
            Some(b'}') => break,
            Some(b'"') => {}
            _ => bail!("expected root key or '}}' at {i}"),
        }
        let (key, vi) = read_key(doc, i)?;
        let ve = if key == b"traceEvents" && doc.get(vi) == Some(&b'[') {
            let (events, arr_close, n_objects) = scan_events_array(doc, vi, detail)?;
            found = Some((vi, arr_close, events, n_objects));
            arr_close + 1
        } else {
            skip_value(doc, vi)?
        };
        i = skip_ws(doc, ve);
        if doc.get(i) == Some(&b',') {
            i = skip_ws(doc, i + 1);
        }
    }
    let Some((arr_open, arr_close, events, n_objects)) = found else {
        bail!("no traceEvents array at the root");
    };
    Ok(DocSpans {
        events,
        arr_open: arr_open as u32,
        arr_close: arr_close as u32,
        n_objects,
    })
}

// ---------------------------------------------------------------------------
// Splice.
// ---------------------------------------------------------------------------

/// serde_json's float formatting (ryu); the fix pass's non-integer ts
/// literals must match the tree serializer's bytes.
fn push_f64(out: &mut Vec<u8>, v: f64) {
    let mut buf = ryu::Buffer::new();
    out.extend_from_slice(buf.format_finite(v).as_bytes());
}

/// One scalar substitution inside a kept event: replace `[start,end)`
/// (zero-width = insertion) with `bytes`.
struct Sub {
    start: u32,
    end: u32,
    bytes: Vec<u8>,
}

use super::minify::copy_min;

/// JSON string literal (quoted, escaped) — serde_json's escaping, so a
/// spliced annotation value reparses to exactly the tree twin's
/// `Value::String`.
fn push_json_str(out: &mut Vec<u8>, s: &str) -> Result<()> {
    serde_json::to_writer(&mut *out, s)?;
    Ok(())
}

/// Re-emit one retagged event's `args` object (`open` at its `{`):
/// original pairs copy through minified unless an annotation upsert
/// collides (value replaced in place), `stream` values become the retag
/// stream number, and leftover upserts + a missing `stream` append at the
/// end. Mirrors the tree twin's final map state — byte ORDER differs, and
/// reparse equality is the contract, same as the rest of the splice.
fn rebuild_args(
    doc: &[u8],
    open: usize,
    pairs: &[(String, String)],
    stream_val: i64,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(64 + 32 * pairs.len());
    out.push(b'{');
    let mut used = vec![false; pairs.len()];
    let mut saw_stream = false;
    let mut wrote = false;
    let mut j = skip_ws(doc, open + 1);
    loop {
        match doc.get(j) {
            Some(b'}') => break,
            Some(b'"') => {}
            _ => bail!("expected args key or '}}' at {j}"),
        }
        let (key, vi) = read_key(doc, j)?;
        let ve = skip_value(doc, vi)?;
        if wrote {
            out.push(b',');
        }
        wrote = true;
        // Key + ':' verbatim; the raw key bytes keep their escapes, so a
        // pair-list key that only matches after unescaping misses here and
        // appends below instead — later duplicate, same reparse winner.
        copy_min(&mut out, &doc[j..vi]);
        if key == b"stream" {
            saw_stream = true;
            out.extend_from_slice(stream_val.to_string().as_bytes());
        } else if let Some(p) = pairs.iter().position(|(k, _)| k.as_bytes() == key) {
            used[p] = true;
            push_json_str(&mut out, &pairs[p].1)?;
        } else {
            copy_min(&mut out, &doc[vi..ve]);
        }
        j = skip_ws(doc, ve);
        if doc.get(j) == Some(&b',') {
            j = skip_ws(doc, j + 1);
        }
    }
    for (p, (k, v)) in pairs.iter().enumerate() {
        if used[p] {
            continue;
        }
        if wrote {
            out.push(b',');
        }
        wrote = true;
        push_json_str(&mut out, k)?;
        out.push(b':');
        push_json_str(&mut out, v)?;
    }
    if !saw_stream {
        if wrote {
            out.push(b',');
        }
        out.extend_from_slice(format!("\"stream\":{stream_val}").as_bytes());
    }
    out.push(b'}');
    Ok(out)
}

/// Emit the merged-trace document: original bytes with the plan's edits.
/// The caller gzips (`tree::gzip_artifact`).
pub fn splice(doc: &[u8], spans: &DocSpans, plan: &SplicePlan) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(doc.len() + 96 * plan.appended.len() + 64);
    splice_into(doc, spans, plan, &mut out)?;
    Ok(out)
}

/// [`splice`] into a caller-owned buffer (the pooled-buffer paths).
fn splice_into(doc: &[u8], spans: &DocSpans, plan: &SplicePlan, out: &mut Vec<u8>) -> Result<()> {
    if plan.keep.len() != spans.n_objects as usize {
        bail!(
            "row/span mismatch: {} rows vs {} scanned objects",
            plan.keep.len(),
            spans.n_objects
        );
    }
    out.reserve(doc.len() + 96 * plan.appended.len() + 64);
    copy_min(out, &doc[..spans.arr_open as usize + 1]);

    let mut moved = plan.moved.iter().peekable();
    let mut fixed = plan.fixed.iter().peekable();
    let mut retag = plan.retagged.iter().peekable();
    let mut retag_ac2g = plan.retagged_ac2g.iter().peekable();
    let mut subs: Vec<Sub> = Vec::new();
    let mut row = 0u32;
    let mut emitted = false;
    for ev in &spans.events {
        let (s, e) = (ev.start as usize, ev.end as usize);
        if !ev.is_object {
            if emitted {
                out.push(b',');
            }
            copy_min(out, &doc[s..e]);
            emitted = true;
            continue;
        }
        let r = row;
        row += 1;
        let is_moved = moved.next_if(|&&m| m == r).is_some();
        let fix = fixed.next_if(|&&(f, _)| f == r).map(|&(_, ts)| ts);
        let rt = retag.next_if(|&&(m, _, _)| m == r).copied();
        let rt_ac2g = retag_ac2g.next_if(|&&(m, _)| m == r).copied();
        if !plan.keep[r as usize] {
            continue;
        }
        if emitted {
            out.push(b',');
        }
        emitted = true;
        if !is_moved && fix.is_none() && rt.is_none() && rt_ac2g.is_none() {
            copy_min(out, &doc[s..e]);
            continue;
        }
        subs.clear();
        // tid target: the move pass runs AFTER annotate, so an overlap
        // move overrides a retag (tree twin: annotate writes tid, move
        // overwrites it).
        let tid_target: Option<i64> = if is_moved {
            Some(plan.overlap_stream)
        } else if let Some((_, _, s)) = rt {
            Some(s)
        } else {
            rt_ac2g.map(|(_, s)| s)
        };
        if let Some(t) = tid_target {
            let n = t.to_string();
            match ev.tid {
                Some((a, b)) => subs.push(Sub {
                    start: a,
                    end: b,
                    bytes: n.as_bytes().to_vec(),
                }),
                None => {
                    // Tree twin: obj.insert("tid", …) — add the key.
                    let mut bytes = format!("\"tid\":{n}").into_bytes();
                    if ev.has_keys {
                        bytes.push(b',');
                    }
                    subs.push(Sub {
                        start: ev.start + 1,
                        end: ev.start + 1,
                        bytes,
                    });
                }
            }
        }
        // args writes only when args is an object (tree twin:
        // `get_mut("args").and_then(as_object_mut)`). A retagged row
        // rebuilds the whole object — annotation upserts can collide with
        // existing keys; a moved-only row keeps the scalar substitution.
        if let Some((_, gnid, rt_stream)) = rt {
            let stream_val = if is_moved {
                plan.overlap_stream
            } else {
                rt_stream
            };
            if let Some(ao) = ev.args_open {
                let ae = skip_value(doc, ao as usize)?;
                static NO_PAIRS: &[(String, String)] = &[];
                let pairs = plan
                    .ann_args
                    .get(&gnid)
                    .map(|v| v.as_slice())
                    .unwrap_or(NO_PAIRS);
                subs.push(Sub {
                    start: ao,
                    end: ae as u32,
                    bytes: rebuild_args(doc, ao as usize, pairs, stream_val)?,
                });
            }
        } else if is_moved {
            let n = plan.overlap_stream.to_string();
            if let Some(ao) = ev.args_open {
                match ev.stream {
                    Some((a, b)) => subs.push(Sub {
                        start: a,
                        end: b,
                        bytes: n.as_bytes().to_vec(),
                    }),
                    None => {
                        let mut bytes = format!("\"stream\":{n}").into_bytes();
                        if ev.args_has_keys {
                            bytes.push(b',');
                        }
                        subs.push(Sub {
                            start: ao + 1,
                            end: ao + 1,
                            bytes,
                        });
                    }
                }
            }
        }
        if let Some(new_ts) = fix {
            let was_int = ev.ts.map(|(_, _, ii)| ii).unwrap_or(false);
            let mut bytes = Vec::new();
            if was_int && new_ts.fract() == 0.0 {
                bytes.extend_from_slice((new_ts as i64).to_string().as_bytes());
            } else {
                push_f64(&mut bytes, new_ts);
            }
            match ev.ts {
                Some((a, b, _)) => subs.push(Sub {
                    start: a,
                    end: b,
                    bytes,
                }),
                None => {
                    let mut kb = b"\"ts\":".to_vec();
                    kb.extend_from_slice(&bytes);
                    if ev.has_keys {
                        kb.push(b',');
                    }
                    subs.push(Sub {
                        start: ev.start + 1,
                        end: ev.start + 1,
                        bytes: kb,
                    });
                }
            }
        }
        subs.sort_by_key(|x| x.start);
        let mut cur = s;
        for sub in &subs {
            copy_min(out, &doc[cur..sub.start as usize]);
            out.extend_from_slice(&sub.bytes);
            cur = sub.end as usize;
        }
        copy_min(out, &doc[cur..e]);
    }
    // Synthetic thread_name events, ascending tid (the walk sorted them);
    // serde-sorted key order to match the tree twin's bytes for these.
    for m in plan.appended {
        if emitted {
            out.push(b',');
        }
        emitted = true;
        out.extend_from_slice(
            format!(
                "{{\"args\":{{\"name\":\"stream {t}\"}},\"name\":\"thread_name\",\
                 \"ph\":\"M\",\"pid\":{p},\"tid\":{t}}}",
                t = m.tid,
                p = m.pid
            )
            .as_bytes(),
        );
    }
    copy_min(out, &doc[spans.arr_close as usize..]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Pooled artifact entry points (one worker thread = one document buffer).
// ---------------------------------------------------------------------------

/// Splice-path artifact production stats, printed with the shard's stage
/// breakdown: ranks whose walk plan had no edits (artifact = the minified
/// document, no scan/splice) vs ranks that needed the full scan + splice.
pub static NOOP_RANKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static EDITED_RANKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// NOTE on buffer lifetimes: the raw artifact document and the gzip scratch
// are deliberately TRANSIENT (per rank), not per-thread pools. Retaining
// them (~83 MB/thread) halved this host's minor faults but tipped a
// memory-tight host's PSI gate into backoff collapse (202 GB budget,
// 160 threads) and ran 1.6x SLOWER. Peak RSS,
// not allocator churn, is the binding constraint on the small hosts.

/// Splice + gzip; only the exact-size gz artifact leaves the worker.
pub fn splice_gzip(doc: &[u8], spans: &DocSpans, plan: &SplicePlan) -> Result<Vec<u8>> {
    let raw = splice(doc, spans, plan)?;
    super::tree::gzip_artifact(&raw)
}

/// No-edit fast path: a walk plan that keeps every row and moves/fixes/
/// appends nothing splices to exactly the minified document, so the scan
/// and the per-event splice are skipped outright.
pub fn minify_gzip(doc: &[u8]) -> Result<Vec<u8>> {
    let mut raw = Vec::with_capacity(doc.len() + 64);
    copy_min(&mut raw, doc);
    super::tree::gzip_artifact(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_ann() -> &'static RenderedAnnArgs {
        static E: std::sync::OnceLock<RenderedAnnArgs> = std::sync::OnceLock::new();
        E.get_or_init(Default::default)
    }

    fn ev_json(doc: &str) -> DocSpans {
        scan(doc.as_bytes()).expect("scan")
    }

    #[test]
    fn scan_finds_spans() {
        let doc = r#"{"schemaVersion":1,"traceEvents":[
            {"ph":"X","cat":"kernel","ts":100,"dur":5,"tid":7,"args":{"stream":7,"graph node id":3}},
            {"ph":"X","cat":"kernel","ts":1.5e2,"dur":5,"tid":8,"args":{}},
            42,
            {"name":"a\"b\\","args":"notobj"}
        ],"other":{"x":[1,2]}}"#;
        let s = ev_json(doc);
        assert_eq!(s.events.len(), 4);
        assert_eq!(s.n_objects, 3);
        let e0 = &s.events[0];
        let (a, b, ii) = e0.ts.unwrap();
        assert_eq!(&doc[a as usize..b as usize], "100");
        assert!(ii);
        let (sa, sb) = e0.stream.unwrap();
        assert_eq!(&doc[sa as usize..sb as usize], "7");
        let e1 = &s.events[1];
        assert!(!e1.ts.unwrap().2, "1.5e2 is not an int literal");
        assert!(e1.args_open.is_some());
        assert!(!e1.args_has_keys);
        assert!(e1.stream.is_none());
        assert!(!s.events[2].is_object);
        let e3 = &s.events[3];
        assert!(e3.args_open.is_none(), "non-object args is absent");
    }

    #[test]
    fn scan_last_trace_events_wins() {
        let doc = r#"{"traceEvents":[{"a":1}],"traceEvents":[{"b":2},{"c":3}]}"#;
        let s = ev_json(doc);
        assert_eq!(s.events.len(), 2);
    }

    #[test]
    fn scan_rejects_non_object_root() {
        assert!(scan(b"[1,2]").is_err());
        assert!(scan(b"{\"a\":1}").is_err(), "no traceEvents");
        assert!(scan(b"{\"traceEvents\":7}").is_err());
    }

    /// End-to-end oracle: real tape parse + real walk, splice vs the
    /// serde-tree twin, compared as parsed values.
    fn oracle(doc: &str) {
        // Tree side.
        let mut tree = super::super::tree::parse_tree(doc.as_bytes()).expect("tree parse");
        super::super::tree::inline_tree(&mut tree, crate::cudagraph::DEFAULT_STREAM);
        let tree_bytes =
            super::super::tree::serialize_tree(&tree, false, doc.len()).expect("tree ser");

        // Splice side.
        let spans = scan(doc.as_bytes()).expect("scan");
        let parsed = super::super::json::parse_bytes_walked(doc.as_bytes()).expect("scan parse");
        let digests = crate::annotations::AnnotationDigests::default();
        let (_view, plan) = crate::rows::convert_parsed_walked_view_plan(
            parsed,
            &digests,
            crate::cudagraph::DEFAULT_STREAM,
            false,
        )
        .expect("walk");
        let splice_plan = SplicePlan {
            keep: &plan.keep,
            moved: &plan.moved,
            fixed: &plan.fixed,
            appended: &plan.appended,
            overlap_stream: crate::cudagraph::DEFAULT_STREAM + 1,
            retagged: &[],
            retagged_ac2g: &[],
            ann_args: empty_ann(),
        };
        let spliced = splice(doc.as_bytes(), &spans, &splice_plan).expect("splice");

        // Capture path: parse-captured element spans must splice to the
        // exact same bytes as the full structural scan.
        let (_, bounds) =
            super::super::json::parse_bytes_walked_bounds(doc.as_bytes()).expect("bounds parse");
        let bspans: DocSpans = bounds.expect("bounds captured").into();
        let bspliced = splice(doc.as_bytes(), &bspans, &splice_plan).expect("bounds splice");
        assert_eq!(spliced, bspliced, "capture vs scan splice bytes diverged");

        let a: serde_json::Value = serde_json::from_slice(&tree_bytes).expect("tree bytes json");
        let b: serde_json::Value = serde_json::from_slice(&spliced).expect("spliced json");
        assert_eq!(a, b, "tree vs spliced artifact diverged");
    }

    #[test]
    fn oracle_move_fix_cleanup_append() {
        // Exercises every edit class at once:
        // - two overlapping graphed kernels on the default stream → move
        // - two graphed work events overlapping by <1us on one stream → fix
        // - orphan gpu_user_annotation + ac2g f + stale M row → cleanup drops
        // - kernels on tids 7/8 with no thread_name rows → appended synthetics
        let doc = r#"{"schemaVersion":1,"traceEvents":[
 {"ph":"M","name":"thread_name","pid":9,"tid":7,"args":{"name":"stream 7"}},
 {"ph":"M","name":"thread_name","pid":9,"tid":99,"args":{"name":"gone"}},
 {"ph":"X","cat":"kernel","name":"k1","ts":100,"dur":10,"tid":7,"args":{"stream":7,"graph node id":3}},
 {"ph":"X","cat":"kernel","name":"k2","ts":105,"dur":10,"tid":7,"args":{"stream":7,"graph node id":4}},
 {"ph":"X","cat":"kernel","name":"k3","ts":120.25,"dur":4,"tid":8,"args":{"stream":8,"graph node id":5}},
 {"ph":"X","cat":"kernel","name":"k4","ts":124.0,"dur":4,"tid":8,"args":{"stream":8,"graph node id":6}},
 {"ph":"X","cat":"kernel","name":"k5","ts":127.5,"dur":4,"tid":8,"args":{"stream":8,"graph node id":7}},
 {"ph":"X","cat":"gpu_user_annotation","name":"gua","ts":1,"dur":1,"tid":55},
 {"ph":"f","cat":"ac2g","ts":2,"tid":55},
 {"ph":"f","cat":"ac2g","ts":100,"tid":7},
 {"ph":"X","cat":"cpu_op","name":"aten::mm","ts":90,"dur":3,"tid":300}
],"deviceProperties":[{"id":0}]}"#;
        oracle(doc);
    }

    #[test]
    fn oracle_no_edits_is_verbatim_events() {
        let doc = r#"{"traceEvents":[
 {"ph":"X","cat":"kernel","name":"k","ts":10,"dur":1,"tid":3,"args":{"stream":3}},
 {"ph":"M","name":"thread_name","pid":1,"tid":3,"args":{"name":"stream 3"}}
],"traceName":"t"}"#;
        oracle(doc);
    }

    #[test]
    fn oracle_delete_first_and_last() {
        // First and last elements are cleanup victims: comma handling.
        let doc = r#"{"traceEvents":[
 {"ph":"f","cat":"ac2g","ts":2,"tid":55},
 {"ph":"X","cat":"kernel","name":"k","ts":10,"dur":1,"tid":3,"args":{"stream":3}},
 {"ph":"M","name":"thread_name","pid":1,"tid":3,"args":{"name":"stream 3"}},
 {"ph":"X","cat":"gpu_user_annotation","name":"g","ts":1,"dur":1,"tid":55}
]}"#;
        oracle(doc);
    }

    #[test]
    fn oracle_everything_deleted_appends_into_empty() {
        let doc = r#"{"traceEvents":[
 {"ph":"f","cat":"ac2g","ts":2,"tid":55}
]}"#;
        oracle(doc);
    }

    #[test]
    fn oracle_move_without_stream_key_and_missing_tid_types() {
        // k2 overlaps k1 on the default stream but its args has no
        // "stream" key → insertion path; float ts fix preserved as float.
        let doc = r#"{"traceEvents":[
 {"ph":"X","cat":"kernel","name":"k1","ts":100.5,"dur":10,"tid":7,"args":{"graph node id":3}},
 {"ph":"X","cat":"kernel","name":"k2","ts":105,"dur":10,"tid":7,"args":{"graph node id":4}},
 {"ph":"X","cat":"kernel","name":"k3","ts":109.75,"dur":2,"tid":7,"args":{"graph node id":5}}
]}"#;
        oracle(doc);
    }

    /// Reassign-mode oracle: pkl annotations applied by the real row walk,
    /// splice vs the serde-tree twin (`annotate_tree`), compared as parsed
    /// values — the same contract the --verify-walks artifact oracle
    /// asserts in production.
    fn oracle_reassign(doc: &str, anns: serde_json::Value) {
        let full: crate::annotations::Annotations = serde_json::from_value(anns).expect("anns");

        // Tree side.
        let mut tree = super::super::tree::parse_tree(doc.as_bytes()).expect("tree parse");
        super::super::tree::annotate_tree(&mut tree, &full, crate::cudagraph::DEFAULT_STREAM);
        let tree_bytes =
            super::super::tree::serialize_tree(&tree, false, doc.len()).expect("tree ser");

        // Splice side.
        let spans = scan(doc.as_bytes()).expect("scan");
        let parsed = super::super::json::parse_bytes_walked(doc.as_bytes()).expect("scan parse");
        let digests = super::super::pkl::digests_from_annotations(&full);
        let (_view, plan) = crate::rows::convert_parsed_walked_view_plan(
            parsed,
            &digests,
            crate::cudagraph::DEFAULT_STREAM,
            true,
        )
        .expect("walk");
        let wanted: rustc_hash::FxHashSet<i64> = plan.retagged.iter().map(|&(_, g, _)| g).collect();
        let ann_args = super::super::pkl::render_ann_args(&full, &wanted);
        let sp = SplicePlan {
            keep: &plan.keep,
            moved: &plan.moved,
            fixed: &plan.fixed,
            appended: &plan.appended,
            overlap_stream: crate::cudagraph::DEFAULT_STREAM + 1,
            retagged: &plan.retagged,
            retagged_ac2g: &plan.retagged_ac2g,
            ann_args: &ann_args,
        };
        let spliced = splice(doc.as_bytes(), &spans, &sp).expect("splice");

        // Capture path: parse-captured spans must splice byte-identically.
        let (_, bounds) =
            super::super::json::parse_bytes_walked_bounds(doc.as_bytes()).expect("bounds parse");
        let bspans: DocSpans = bounds.expect("bounds captured").into();
        let bspliced = splice(doc.as_bytes(), &bspans, &sp).expect("bounds splice");
        assert_eq!(spliced, bspliced, "capture vs scan splice bytes diverged");

        let a: serde_json::Value = serde_json::from_slice(&tree_bytes).expect("tree bytes json");
        let b: serde_json::Value = serde_json::from_slice(&spliced).expect("spliced json");
        assert_eq!(a, b, "tree vs reassign-spliced artifact diverged");
    }

    #[test]
    fn oracle_reassign_full_matrix() {
        // Every reassign edit class at once:
        // - k1 (gnid 3): annotation with numeric stream, arg upserts that
        //   collide ("graph node id") and that are new ("Process Group
        //   Name", numeric value rendering)
        // - k2 (gnid 4): string stream coercion, no extra keys
        // - k3 (gnid 5): two annotation dicts (later wins), no stream →
        //   default; escaped quote in an upsert value
        // - k4 (gnid 6): NO annotation entry → bare retag to the default
        //   stream; its args has no "stream" key → insertion path
        // - ac2g f partner of k1's ORIGINAL (tid,ts) → tid-only retag
        // - k5/k6 (gnids 7/8): annotated onto the default stream and
        //   overlapping → the move pass retargets k6 AFTER the annotate
        //   retag (move wins)
        // - orphan gpu_user_annotation → cleanup drop
        // - work tids without thread_name metadata → appended synthetics
        let doc = r#"{"schemaVersion":1,"traceEvents":[
 {"ph":"M","name":"thread_name","pid":9,"tid":20,"args":{"name":"stream 20"}},
 {"ph":"X","cat":"kernel","name":"k1","ts":100,"dur":5,"tid":3,"args":{"stream":3,"graph node id":3}},
 {"ph":"f","cat":"ac2g","ts":100,"tid":3},
 {"ph":"X","cat":"kernel","name":"k2","ts":200,"dur":5,"tid":3,"args":{"stream":3,"graph node id":4}},
 {"ph":"X","cat":"kernel","name":"k3","ts":300,"dur":5,"tid":3,"args":{"stream":3,"graph node id":5}},
 {"ph":"X","cat":"kernel","name":"k4","ts":400,"dur":5,"tid":3,"args":{"graph node id":6}},
 {"ph":"X","cat":"kernel","name":"k5","ts":500,"dur":10,"tid":3,"args":{"stream":3,"graph node id":7}},
 {"ph":"X","cat":"kernel","name":"k6","ts":505,"dur":10,"tid":3,"args":{"stream":3,"graph node id":8}},
 {"ph":"X","cat":"gpu_user_annotation","name":"gua","ts":1,"dur":1,"tid":55}
]}"#;
        oracle_reassign(
            doc,
            serde_json::json!({
                "3": [{"stream": 20, "graph node id": 42, "Process Group Name": 7.5}],
                "4": [{"stream": "21"}],
                "5": [{"note": "first"}, {"note": "a\"b\\c"}],
                "7": [{}],
                "8": [{}]
            }),
        );
    }

    #[test]
    fn oracle_reassign_fix_after_retag() {
        // Two graphed work events annotated onto one stream, overlapping by
        // <1us: the fix pass clamps the second's float ts on the same rows
        // the retag rewrote.
        let doc = r#"{"traceEvents":[
 {"ph":"X","cat":"kernel","name":"a","ts":100.25,"dur":4,"tid":3,"args":{"stream":3,"graph node id":11}},
 {"ph":"X","cat":"kernel","name":"b","ts":103.75,"dur":4,"tid":3,"args":{"stream":3,"graph node id":12}}
]}"#;
        oracle_reassign(
            doc,
            serde_json::json!({
                "11": [{"stream": 31}],
                "12": [{"stream": 31}]
            }),
        );
    }

    #[test]
    fn copy_min_strips_only_outside_strings() {
        let mut out = Vec::new();
        copy_min(&mut out, b"{ \"a b\": \"c \\\" d\" ,\n\t\"n\": 1.5 }");
        assert_eq!(out, b"{\"a b\":\"c \\\" d\",\"n\":1.5}");
    }

    #[test]
    fn sparse_scan_splices_identically() {
        // Detailing only the edited rows must not change a single output
        // byte relative to the full scan.
        let doc = br#"{ "traceEvents": [
 { "ph": "X", "name": "k0", "ts": 10, "dur": 1, "tid": 3, "args": { "stream": 3 } },
 { "ph": "X", "name": "k1", "ts": 20, "dur": 1, "tid": 3, "args": { "stream": 3 } },
 { "ph": "X", "name": "k2", "ts": 30.5, "dur": 1, "tid": 3, "args": { "stream": 3 } },
 { "ph": "X", "name": "k3", "ts": 40, "dur": 1, "tid": 3 }
] }"#;
        let plan = SplicePlan {
            keep: &[true, false, true, true],
            moved: &[3],
            fixed: &[(2, 31.25)],
            appended: &[],
            overlap_stream: 8,
            retagged: &[],
            retagged_ac2g: &[],
            ann_args: empty_ann(),
        };
        let full = splice(doc, &scan(doc).unwrap(), &plan).unwrap();
        let sparse = splice(doc, &scan_sparse(doc, &[2, 3]).unwrap(), &plan).unwrap();
        assert_eq!(full, sparse);
    }

    #[test]
    fn minified_doc_equals_no_edit_splice() {
        // The no-edit fast path must be byte-identical to a splice whose
        // plan keeps every row and edits nothing.
        let doc = br#"{ "traceEvents": [
 { "ph": "X", "name": "k 1", "ts": 10, "dur": 1, "tid": 3, "args": { "stream": 3 } },
 "stray string",
 { "ph": "M", "name": "process_name" }
], "traceName": "a b" }"#;
        let spans = scan(doc).unwrap();
        let plan = SplicePlan {
            keep: &[true; 2],
            moved: &[],
            fixed: &[],
            appended: &[],
            overlap_stream: 8,
            retagged: &[],
            retagged_ac2g: &[],
            ann_args: empty_ann(),
        };
        let spliced = splice(doc, &spans, &plan).unwrap();
        let mut minified = Vec::new();
        copy_min(&mut minified, doc);
        assert_eq!(spliced, minified);
    }

    #[test]
    fn splice_output_has_no_interstring_whitespace() {
        let doc = r#"{ "traceEvents": [
 { "ph": "X", "cat": "kernel", "name": "k 1", "ts": 10, "dur": 1, "tid": 3, "args": { "stream": 3 } }
], "traceName": "a b" }"#;
        let spans = scan(doc.as_bytes()).unwrap();
        let plan = SplicePlan {
            keep: &[true],
            moved: &[],
            fixed: &[],
            appended: &[],
            overlap_stream: 8,
            retagged: &[],
            retagged_ac2g: &[],
            ann_args: empty_ann(),
        };
        let out = splice(doc.as_bytes(), &spans, &plan).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(
            s,
            "{\"traceEvents\":[{\"ph\":\"X\",\"cat\":\"kernel\",\"name\":\"k 1\",\
             \"ts\":10,\"dur\":1,\"tid\":3,\"args\":{\"stream\":3}}],\"traceName\":\"a b\"}"
        );
    }

    /// Parse-captured spans must be field-identical to the structural
    /// scan's, for every element.
    fn assert_capture_matches_scan(doc: &[u8]) {
        let (_, bounds) = super::super::json::parse_bytes_walked_bounds(doc).expect("parse");
        let caps: DocSpans = bounds.expect("captured").into();
        let full = scan(doc).expect("scan");
        assert_eq!(caps.arr_open, full.arr_open);
        assert_eq!(caps.arr_close, full.arr_close);
        assert_eq!(caps.n_objects, full.n_objects);
        assert_eq!(caps.events, full.events);
    }

    #[test]
    fn bounds_capture_matches_scan() {
        // Non-objects, duplicate traceEvents (last wins), trailing root key.
        assert_capture_matches_scan(
            br#"{"traceEvents":[{"a":1}],"traceEvents":[
 { "ph": "X", "name": "k0", "ts": 10, "dur": 1, "tid": 3, "args": { "stream": 3 } },
 42, "stray",
 { "ph": "M", "name": "process_name" }
], "traceName": "t"}"#,
        );
        // Duplicate keys inside one event (last wins at every level), empty
        // objects, float/string ts, args without stream, args replaced by a
        // non-object (stale args_has_keys, like the scanner), nested
        // containers inside unmatched keys.
        assert_capture_matches_scan(
            br#"{"other":{"x":[1,{"ts":9}]},"traceEvents":[
 {},
 { "ts": 1, "ts": 2.5, "tid": 3, "tid": 4, "args": { "stream": 1 }, "args": { "a": [1,2], "stream": 7 } },
 { "ts": "not a number", "args": {} },
 { "args": { "a": 1 }, "args": 7, "dur": 2 },
 { "ts": 18446744073709551615, "args": { "stream": { "weird": true } } }
]}"#,
        );
    }

    #[test]
    fn bounds_absent_without_events_array() {
        let (_, bounds) =
            super::super::json::parse_bytes_walked_bounds(br#"{"traceEvents":7}"#).expect("parse");
        assert!(bounds.is_none(), "non-array traceEvents captures nothing");
        let (_, bounds) =
            super::super::json::parse_bytes_walked_bounds(br#"{"a":1}"#).expect("parse");
        assert!(bounds.is_none(), "no traceEvents captures nothing");
    }

    #[test]
    fn splice_rejects_row_mismatch() {
        let doc = r#"{"traceEvents":[{"a":1}]}"#;
        let spans = scan(doc.as_bytes()).unwrap();
        let plan = SplicePlan {
            keep: &[true, true],
            moved: &[],
            fixed: &[],
            appended: &[],
            overlap_stream: 8,
            retagged: &[],
            retagged_ac2g: &[],
            ann_args: empty_ann(),
        };
        assert!(splice(doc.as_bytes(), &spans, &plan).is_err());
    }
}

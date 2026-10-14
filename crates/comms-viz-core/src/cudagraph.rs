// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Tree-free CUDA-graph walk passes over the converter's row representation.
//!
//! This is the row-path CUDA-graph repair (annotate → cleanup → move → fix),
//! applied directly to the parsed rows; it serves every rank that needs no
//! recording merge. The serde-tree twin lives in `legacy/tree_walks.rs` —
//! building its `serde_json::Value` tree costs ~2 core-s per 231K-event
//! trace, which is why this path exists — and `--verify-walks` runs both
//! side by side and asserts identical converted columns.
//!
//! Exactness invariants (what the column equality rests on):
//! - the ac2g reindex key is (tid,ts) as f64 BIT PATTERNS built from the
//!   ORIGINAL values (`tid_bits`/`ts_bits`);
//! - cleanup/move/fix read tids via `as_i64().unwrap_or(0)` semantics
//!   (float/missing tids collapse to 0 in the SETS, unlike the bit keys);
//! - annotation values are stringified with serde_json `Number::to_string`
//!   (`python_str`), then re-coerced by the converter rules (`stoll` for
//!   the int columns);
//! - appended `thread_name` metadata rows join the row stream at the end
//!   (post-walk serialized order).

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use serde_json::Value;

use crate::annotations::AnnotationDigests;
use crate::rows::Span;
use crate::rows::arena_push;

/// The CUDA default stream id the walks assign unannotated graphed work to
/// (and `default_stream + 1` for overlap moves).
pub const DEFAULT_STREAM: i64 = 7;

/// Raw (pre-coercion) walk-relevant view of one parsed event, parse order.
#[derive(Clone, Default)]
pub struct RawEvent {
    /// `event_field_i64(ev, "tid")`: as_i64 else 0 (floats/missing → 0).
    pub tid_i: i64,
    /// Original tid as f64 bits (None if missing/non-numeric).
    pub tid_bits: Option<u64>,
    /// Original ts as f64 bits (None if missing/non-numeric).
    pub ts_bits: Option<u64>,
    /// `event_field_f64(ev, "ts")`: as_f64 else 0.0 — move/fix domain.
    pub ts_f: f64,
    /// `event_field_f64(ev, "dur")`.
    pub dur_f: f64,
    /// `event_field_i64(ev, "pid")`.
    pub pid_i: i64,
    pub ph_is_m: bool,
    pub ph_is_f: bool,
    pub cat_is_ac2g: bool,
    pub cat_is_work: bool,
    pub cat_is_gua: bool,
    pub cat_is_kernel: bool,
    pub is_thread_name_meta: bool,
    /// M/thread_name event whose args.name starts with "stream ".
    pub meta_stream_name: bool,
    /// args["graph node id"] as_i64 else 0.
    pub gnid: i64,
}

/// Python-style value stringification with serde_json `Number::to_string`
/// semantics, NOT CPython `str()` — the converted column bytes depend on
/// exactly this formatting.
pub(crate) fn python_str(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(_) | Value::Object(_) => v.to_string(),
    }
}

/// Lenient i64 coercion for annotation `stream` fields.
pub(crate) fn coerce_to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.parse::<i64>().ok(),
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => None,
    }
}

fn f64_key(f: f64) -> u64 {
    if f.is_nan() {
        f64::NAN.to_bits()
    } else {
        f.to_bits()
    }
}

/// Columns the walk can touch, borrowed from the converter's staging rows.
/// (Split out so `rows.rs` can hand us mutable access without exposing
/// its whole `Rows` internals.)
pub struct WalkColumns<'a> {
    /// Append-only string arena backing the span columns; annotation writes
    /// append their stringified values here.
    pub arena: &'a mut String,
    pub ts: &'a mut Vec<Option<f64>>,
    pub stream: &'a mut Vec<Option<i64>>,
    pub pg_desc: &'a mut Vec<Option<Span>>,
    pub pg_name: &'a mut Vec<Option<Span>>,
}

/// A synthesized `thread_name` metadata row (appended post-walk).
pub struct MetaRow {
    pub pid: i64,
    /// The work tid the synthesized row names (`"stream {tid}"`).
    pub tid: i64,
}

/// Last value appended to the arena for one annotation column, with its span.
///
/// Graph ids are unique per kernel launch, but the VALUES they carry repeat:
/// one process group per stream, a handful of dtypes. Appending a fresh copy
/// per row would rewrite the same bytes tens of thousands of times AND give
/// every row a different span for identical content, forcing every
/// downstream "same values as the previous row?" test to chase two arena
/// strings — a cache miss each, measured at 6.4% of analyzer cycles in
/// `process_view`. Reusing the previous span when the bytes match keeps the
/// column values exactly the same while making span equality answer that
/// question.
#[derive(Default)]
struct LastPush {
    text: String,
    span: Option<Span>,
}

impl LastPush {
    #[inline]
    fn push(&mut self, arena: &mut String, v: &str) -> Span {
        if let Some(sp) = self.span {
            if self.text == v {
                return sp;
            }
        }
        let sp = arena_push(arena, v);
        self.text.clear();
        self.text.push_str(v);
        self.span = Some(sp);
        sp
    }
}

pub struct WalkOutcome {
    /// keep[i]: row i survives cleanup.
    pub keep: Vec<bool>,
    /// Appended metadata rows (converter view: cat None, name
    /// "thread_name", pid = gpu_pid, everything else missing).
    pub appended: Vec<MetaRow>,
    /// Rows the move pass reassigned to `overlap_stream` (tree twin
    /// rewrites `tid` + `args.stream`). Parse order, ascending.
    pub moved: Vec<u32>,
    /// Rows the fix pass clamped, with the new `ts` (tree twin rewrites
    /// `ts`, int-preserving). Parse order within each stream bucket.
    pub fixed: Vec<(u32, f64)>,
    /// Reassign mode: graphed rows the annotate pass retargeted, as
    /// (row, gnid, final stream), ascending. The tree twin rewrites `tid`,
    /// upserts each annotation key into `args` as a string, then
    /// overwrites `args.stream` with the stream number.
    pub retagged: Vec<(u32, i64, i64)>,
    /// Reassign mode: their ac2g `f` partner rows, as (row, final
    /// stream), ascending, deduped last-wins like the tree twin's
    /// sequential application. `tid` only — no `args` writes.
    pub retagged_ac2g: Vec<(u32, i64)>,
}

/// Apply annotate(+cleanup)→move→fix to the rows. `reassign` false is the
/// inline-annotated variant (`walk_passes_no_reassign`: cleanup+move+fix,
/// no tid/stream reassignment, no annotation application).
pub fn apply_walks(
    cols: &mut WalkColumns<'_>,
    raw: &mut [RawEvent],
    annotations: &AnnotationDigests,
    default_stream: i64,
    reassign: bool,
) -> WalkOutcome {
    let n = raw.len();
    let overlap_stream = default_stream + 1;
    let mut retagged: Vec<(u32, i64, i64)> = Vec::new();
    let mut retagged_ac2g: Vec<(u32, i64)> = Vec::new();

    // ---- annotate (walk_annotate) ----
    if reassign {
        // ac2g index on ORIGINAL (tid,ts) bit keys.
        let mut ac2g_index: HashMap<(Option<u64>, Option<u64>), Vec<usize>> = HashMap::default();
        for (i, r) in raw.iter().enumerate() {
            if r.cat_is_ac2g && r.ph_is_f {
                ac2g_index
                    .entry((r.tid_bits, r.ts_bits))
                    .or_default()
                    .push(i);
            }
        }
        // Collect (graphed-event) updates against the ORIGINAL keys.
        let mut ac2g_updates: Vec<(usize, i64)> = Vec::new();
        let mut tid_updates: Vec<(usize, i64)> = Vec::new();
        // See `LastPush`: repeated annotation values share one arena copy.
        let (mut m_pgd, mut m_pgn) = (LastPush::default(), LastPush::default());
        for (i, r) in raw.iter().enumerate() {
            let gnid = r.gnid;
            if gnid == 0 {
                continue;
            }
            let mut stream_id: Option<i64> = None;
            if let Some(ann_list) = annotations.get(&gnid) {
                // PG/stream fields use the same python_str conversions as
                // the tree walk. rows::walk_compact applies the compact
                // communication metadata in this same annotation order.
                for ann in ann_list {
                    if let Some(s) = &ann.pg_desc {
                        cols.pg_desc[i] = Some(m_pgd.push(cols.arena, s));
                    }
                    if let Some(s) = &ann.pg_name {
                        cols.pg_name[i] = Some(m_pgn.push(cols.arena, s));
                    }
                    if let Some(s) = ann.stream {
                        stream_id = Some(s);
                    }
                }
            }
            let final_stream = stream_id.unwrap_or(default_stream);
            let old_key = (r.tid_bits, r.ts_bits);
            tid_updates.push((i, final_stream));
            retagged.push((i as u32, gnid, final_stream));
            // args["stream"] overwritten as INT after the string updates.
            cols.stream[i] = Some(final_stream);
            if let Some(list) = ac2g_index.get(&old_key) {
                for &j in list {
                    ac2g_updates.push((j, final_stream));
                }
            }
        }
        for (i, t) in tid_updates {
            raw[i].tid_i = t;
            raw[i].tid_bits = Some(f64_key(t as f64));
        }
        // Dedup for the splice plan BEFORE the sequential application: a
        // row hit by several graphed events keeps the LAST stream, exactly
        // what the loop below leaves in `raw`.
        {
            let mut last: HashMap<usize, i64> = HashMap::default();
            for &(j, t) in &ac2g_updates {
                last.insert(j, t);
            }
            retagged_ac2g = last.into_iter().map(|(j, t)| (j as u32, t)).collect();
            retagged_ac2g.sort_unstable_by_key(|&(j, _)| j);
        }
        for (j, t) in ac2g_updates {
            raw[j].tid_i = t;
            raw[j].tid_bits = Some(f64_key(t as f64));
        }
    }

    // ---- cleanup (cleanup_trace_metadata) — runs in BOTH modes ----
    let mut keep = vec![true; n];
    {
        // Pass 1: tids_with_work (as_i64-or-0 semantics).
        let mut tids_with_work: HashSet<i64> = HashSet::default();
        for r in raw.iter() {
            if r.cat_is_work {
                tids_with_work.insert(r.tid_i);
            }
        }
        // Filter 1: noise events.
        for (i, r) in raw.iter().enumerate() {
            if r.cat_is_gua && !tids_with_work.contains(&r.tid_i) {
                keep[i] = false;
            }
            if r.cat_is_ac2g && r.ph_is_f && !tids_with_work.contains(&r.tid_i) {
                keep[i] = false;
            }
        }
        // all_tids over surviving non-M rows.
        let mut all_tids: HashSet<i64> = HashSet::default();
        for (i, r) in raw.iter().enumerate() {
            if keep[i] && !r.ph_is_m {
                all_tids.insert(r.tid_i);
            }
        }
        // gpu_pid: first surviving M/thread_name with args.name "stream *".
        let mut gpu_pid: i64 = 0;
        for (i, r) in raw.iter().enumerate() {
            if keep[i] && r.is_thread_name_meta && r.meta_stream_name {
                gpu_pid = r.pid_i;
                break;
            }
        }
        // Filter 2: M rows with unknown tids.
        for (i, r) in raw.iter().enumerate() {
            if keep[i] && r.ph_is_m && !all_tids.contains(&r.tid_i) {
                keep[i] = false;
            }
        }
        // existing thread_name tids among survivors.
        let mut existing: HashSet<i64> = HashSet::default();
        for (i, r) in raw.iter().enumerate() {
            if keep[i] && r.is_thread_name_meta {
                existing.insert(r.tid_i);
            }
        }
        let mut missing: Vec<i64> = tids_with_work.difference(&existing).copied().collect();
        missing.sort();
        let appended: Vec<MetaRow> = missing
            .iter()
            .map(|&tid| MetaRow { pid: gpu_pid, tid })
            .collect();

        // ---- move (walk_move) over surviving rows ----
        let mut moved: Vec<u32> = Vec::new();
        {
            let mut indices: Vec<usize> = (0..n)
                .filter(|&i| {
                    keep[i]
                        && raw[i].cat_is_kernel
                        && raw[i].tid_i == default_stream
                        && raw[i].gnid != 0
                })
                .collect();
            indices.sort_by(|&a, &b| {
                raw[a]
                    .ts_f
                    .partial_cmp(&raw[b].ts_f)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut prev_end = 0.0f64;
            for i in indices {
                let ts = raw[i].ts_f;
                let dur = raw[i].dur_f;
                if ts < prev_end {
                    raw[i].tid_i = overlap_stream;
                    cols.stream[i] = Some(overlap_stream);
                    moved.push(i as u32);
                } else {
                    prev_end = ts + dur;
                }
            }
            moved.sort_unstable();
        }

        // ---- fix (walk_fix, max_adjust_us = 1.0) over surviving rows ----
        let mut fixed: Vec<(u32, f64)> = Vec::new();
        {
            let max_adjust_us = 1.0f64;
            let mut per_stream: HashMap<i64, Vec<usize>> = HashMap::default();
            for (i, r) in raw.iter().enumerate() {
                if keep[i] && r.cat_is_work && r.gnid != 0 {
                    per_stream.entry(r.tid_i).or_default().push(i);
                }
            }
            for (_tid, mut indices) in per_stream {
                indices.sort_by(|&a, &b| {
                    raw[a]
                        .ts_f
                        .partial_cmp(&raw[b].ts_f)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut prev_end = 0.0f64;
                for i in indices {
                    let ts = raw[i].ts_f;
                    let dur = raw[i].dur_f;
                    let mut effective_ts = ts;
                    if ts < prev_end {
                        let overlap = prev_end - ts;
                        if overlap <= max_adjust_us {
                            cols.ts[i] = Some(prev_end);
                            effective_ts = prev_end;
                            fixed.push((i as u32, prev_end));
                        }
                    }
                    prev_end = effective_ts + dur;
                }
            }
            fixed.sort_unstable_by_key(|&(i, _)| i);
        }

        WalkOutcome {
            keep,
            appended,
            moved,
            fixed,
            retagged,
            retagged_ac2g,
        }
    }
}

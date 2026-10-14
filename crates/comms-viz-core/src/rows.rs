// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Format-independent row model between the trace parsers and the analysis:
//! the `Span` string arena, the columnar `Rows` staging, the post-walk
//! `ConvertedColumns`/`WalkedView` shapes and their thread-local shell
//! pools, the gunzip helpers, the global sort permutation, the walk
//! plumbing, and the pftrace → rows bridge. The JSON parsers live in
//! `legacy/json.rs`.
//!
//! The contract the analysis — and comms_viz-vs-comms_viz artifact determinism —
//! depends on:
//! - one stable global sort by (ts asc, dur==0 desc, dur desc), each key
//!   nulls-first, parse order breaking ties — the tie order is part of the
//!   pinned artifact bytes;
//! - normalization fills over the eight columns: ts/dur→0.0,
//!   stream/grid→-1, pg_desc/pg_name→"" (via the accessors); cat/name stay
//!   null — downstream drops cat-less rows, so the null is load-bearing.

use anyhow::Context;
use anyhow::Result;

/// Byte range of one string value inside a `Rows`/`ConvertedColumns` string
/// arena. Spans stay valid across appends because the arena is append-only.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Span {
    off: u64,
    len: u32,
}

/// Append `s` to an arena, returning its span.
#[inline]
pub(crate) fn arena_push(arena: &mut String, s: &str) -> Span {
    let off = arena.len();
    arena.push_str(s);
    arena_span(arena, off)
}

/// Span of the arena tail appended since `off` was captured. Offsets are
/// 64-bit because large annotated Kineto traces can build arenas over 4 GiB;
/// individual strings remain bounded by `u32`.
#[inline]
pub(crate) fn arena_span(arena: &str, off: usize) -> Span {
    assert!(
        arena.len() - off <= u32::MAX as usize,
        "individual string exceeds 4 GiB"
    );
    Span {
        off: off as u64,
        len: (arena.len() - off) as u32,
    }
}

/// Columnar staging for parsed events, in parse order. String values live in
/// one shared append-only `arena`; the string columns hold spans, eliminating
/// the per-row heap Strings that dominated allocator and DRAM traffic here.
///
/// Core analysis columns plus sparse communication metadata and the clock origin.
/// Walk-facing per-event fields (tid/pid/ph/gnid/...) live in the parallel
/// `RawEvent` buffer instead.
#[derive(Default)]
pub(crate) struct Rows {
    pub(crate) arena: String,
    pub(crate) comm: rustc_hash::FxHashMap<usize, crate::comm::CommMetadata>,
    pub(crate) base_ns: Option<u64>,
    pub(crate) cat: Vec<Option<Span>>,
    pub(crate) name: Vec<Option<Span>>,
    pub(crate) ts: Vec<Option<f64>>,
    pub(crate) dur: Vec<Option<f64>>,
    pub(crate) stream: Vec<Option<i64>>,
    pub(crate) grid: Vec<Option<i64>>,
    pub(crate) pg_desc: Vec<Option<Span>>,
    pub(crate) pg_name: Vec<Option<Span>>,
    pub(crate) ph: Vec<Option<Span>>,
    pub(crate) collective_name: Vec<Option<Span>>,
    pub(crate) in_nelems: Vec<Option<i64>>,
    pub(crate) out_nelems: Vec<Option<i64>>,
    pub(crate) group_size: Vec<Option<i64>>,
    pub(crate) counter_value: Vec<Option<f64>>,
}

impl Rows {
    fn len(&self) -> usize {
        self.ts.len()
    }

    #[inline]
    pub(crate) fn get(&self, sp: Span) -> &str {
        &self.arena[sp.off as usize..sp.off as usize + sp.len as usize]
    }
}

impl Rows {
    /// Heap bytes pinned by this shell's capacities (see `pool_bytes`).
    fn bytes(&self) -> u64 {
        fn elem<T>(_: &[T]) -> usize {
            size_of::<T>()
        }
        let mut b = self.arena.capacity() as u64;
        b += (self.comm.capacity() * size_of::<(usize, crate::comm::CommMetadata)>()) as u64;
        b += self
            .comm
            .values()
            .flat_map(|m| m.0.iter().flatten())
            .map(|s| s.capacity() as u64)
            .sum::<u64>();
        macro_rules! cols {
            ($($f:ident),*) => { $( b += (self.$f.capacity() * elem(&self.$f)) as u64; )* };
        }
        cols!(
            cat,
            name,
            ts,
            dur,
            stream,
            grid,
            pg_desc,
            pg_name,
            ph,
            collective_name,
            in_nelems,
            out_nelems,
            group_size,
            counter_value
        );
        b
    }

    /// Reset for reuse from the thread-local shell pool: keep capacities
    /// (avoids per-rank mmap/fault churn), ensure the targets are covered.
    pub(crate) fn recycle(&mut self, n: usize, arena_bytes: usize) {
        self.comm.clear();
        self.base_ns = None;
        self.arena.clear();
        self.arena.reserve(arena_bytes);
        macro_rules! rec {
            ($($f:ident),*) => { $( self.$f.clear(); self.$f.reserve(n); )* };
        }
        rec!(
            cat,
            name,
            ts,
            dur,
            stream,
            grid,
            pg_desc,
            pg_name,
            ph,
            collective_name,
            in_nelems,
            out_nelems,
            group_size,
            counter_value
        );
    }
}

/// C++ `stoll` prefix parse on raw bytes — the coercion the trace's
/// decimal-string values follow: optional whitespace/sign then ≥1 ASCII
/// digit; no digits or overflow → NULL. Byte input because the pftrace wire
/// walker matches values without validating UTF-8 first (the parse itself
/// only ever consumes ASCII).
pub fn stoll_prefix_bytes(b: &[u8]) -> Option<i64> {
    let mut i = 0;
    while i < b.len() && (b[i] as char).is_whitespace() {
        i += 1;
    }
    let neg = if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        let neg = b[i] == b'-';
        i += 1;
        neg
    } else {
        false
    };
    let start = i;
    let mut acc: i128 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        acc = acc * 10 + (b[i] - b'0') as i128;
        if acc > (i64::MAX as i128) + 1 {
            return None; // overflow → std::out_of_range → NULL
        }
        i += 1;
    }
    if i == start {
        return None; // no digits → std::invalid_argument → NULL
    }
    let v = if neg { -acc } else { acc };
    if v < i64::MIN as i128 || v > i64::MAX as i128 {
        return None;
    }
    Some(v as i64)
}

/// Decompress gzip (possibly multi-member) or pass through raw bytes.
pub fn maybe_gunzip(input: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let n = maybe_gunzip_into(input, &mut out)?;
    out.truncate(n);
    Ok(out)
}

/// Offset of the raw deflate stream inside a single-member gzip file
/// (RFC 1952 header: fixed 10 bytes + optional FEXTRA/FNAME/FCOMMENT/FHCRC).
fn gzip_deflate_offset(b: &[u8]) -> Option<usize> {
    if b.len() < 18 || b[0] != 0x1f || b[1] != 0x8b || b[2] != 8 {
        return None;
    }
    let flg = b[3];
    let mut off = 10usize;
    if flg & 0x04 != 0 {
        // FEXTRA
        if b.len() < off + 2 {
            return None;
        }
        let xlen = u16::from_le_bytes([b[off], b[off + 1]]) as usize;
        off += 2 + xlen;
    }
    if flg & 0x08 != 0 {
        // FNAME (NUL-terminated)
        off += b.get(off..)?.iter().position(|&c| c == 0)? + 1;
    }
    if flg & 0x10 != 0 {
        // FCOMMENT
        off += b.get(off..)?.iter().position(|&c| c == 0)? + 1;
    }
    if flg & 0x02 != 0 {
        off += 2; // FHCRC
    }
    if off + 8 >= b.len() {
        return None;
    }
    Some(off)
}

/// Decompress into a caller-owned buffer, returning the filled length.
/// The buffer is kept at its high-water length across calls (contents beyond
/// the returned length are garbage): reusing one initialized buffer per
/// worker thread avoids both a ~38MB mmap/munmap round trip per rank (which
/// serialized the rank loop on the kernel mmap lock at high thread counts)
/// and read_to_end's re-zeroing of spare capacity.
///
/// Single-member gzip goes through libdeflate (~3x faster than zlib
/// inflate); multi-member or otherwise unusual streams fall back to
/// flate2's MultiGzDecoder, byte-identical output either way.
pub fn maybe_gunzip_into(input: &[u8], out: &mut Vec<u8>) -> Result<usize> {
    if input.len() < 18 || input[0] != 0x1f || input[1] != 0x8b {
        out.clear();
        out.extend_from_slice(input);
        return Ok(input.len());
    }
    // ISIZE trailer: uncompressed size mod 2^32 (exact for our <4GB traces).
    let isize_hint = u32::from_le_bytes(input[input.len() - 4..].try_into().unwrap()) as usize;
    let mut need = isize_hint.max(64);
    // Raw-deflate on the header-stripped stream skips the CRC32 pass (~2.5%
    // of rank-loop cycles); the deflate structure itself still validates,
    // and any oddity falls through to the checking decoders below.
    let deflate_off = gzip_deflate_offset(input);
    for _ in 0..3 {
        if out.len() < need {
            out.resize(need, 0);
        }
        let res = match deflate_off {
            Some(off) => libdeflater::Decompressor::new().deflate_decompress(&input[off..], out),
            None => libdeflater::Decompressor::new().gzip_decompress(input, out),
        };
        match res {
            // Raw-deflate can't see a second gzip member: only trust it when
            // the output length matches the (last-member) ISIZE trailer,
            // which multi-member concatenations won't. Mismatch → flate2.
            Ok(n) => {
                if deflate_off.is_none() || n as u32 == isize_hint as u32 {
                    return Ok(n);
                }
                break;
            }
            Err(libdeflater::DecompressionError::InsufficientSpace) => {
                need = need.saturating_mul(2);
                continue;
            }
            // BadData here can just mean a multi-member stream (libdeflate
            // rejects trailing members) — let flate2 decide.
            Err(_) => break,
        }
    }
    use std::io::Read;
    let mut fallback = Vec::with_capacity(input.len() * 4);
    let mut decoder = flate2::read::MultiGzDecoder::new(input);
    decoder
        .read_to_end(&mut fallback)
        .context("gzip decompress failed")?;
    let n = fallback.len();
    if out.len() < n {
        out.resize(n, 0);
    }
    out[..n].copy_from_slice(&fallback);
    Ok(n)
}

struct SortKey {
    ts: Option<f64>,
    dur: Option<f64>,
}

/// Multi-key comparator: (ts asc, __zero_dur desc, dur desc), each key
/// nulls-first.
fn arrow_cmp(a: &SortKey, b: &SortKey) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    // ts ascending, nulls first
    match (a.ts, b.ts) {
        (None, None) => {}
        (None, Some(_)) => return Less,
        (Some(_), None) => return Greater,
        (Some(x), Some(y)) => {
            // total_cmp matches arrow's f64 ordering for non-NaN data.
            match x.total_cmp(&y) {
                Equal => {}
                o => return o,
            }
        }
    }
    // __zero_dur = (dur == 0.0), descending (true first), nulls first
    let az = a.dur.map(|d| d == 0.0);
    let bz = b.dur.map(|d| d == 0.0);
    match (az, bz) {
        (None, None) => {}
        (None, Some(_)) => return Less,
        (Some(_), None) => return Greater,
        (Some(x), Some(y)) => match y.cmp(&x) {
            Equal => {}
            o => return o,
        },
    }
    // dur descending, nulls first
    match (a.dur, b.dur) {
        (None, None) => Equal,
        (None, Some(_)) => Less,
        (Some(_), None) => Greater,
        (Some(x), Some(y)) => y.total_cmp(&x),
    }
}

/// Compute the global emit permutation: one stable sort of the parse
/// indices by `arrow_cmp` (parse order breaks ties).
fn sort_permutation(keys: &[SortKey]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..keys.len() as u32).collect();
    idx.sort_by(|&x, &y| arrow_cmp(&keys[x as usize], &keys[y as usize]));
    idx
}

fn gather_opt<T: Copy>(v: &[Option<T>], perm: &[u32]) -> Vec<Option<T>> {
    perm.iter().map(|&i| v[i as usize]).collect()
}

/// Post-sort, normalized column view holding exactly what the analysis
/// consumes. Null semantics preserved where downstream behavior depends on
/// them (cat/name); pg_desc/pg_name keep the empty-string fill via the
/// accessors. String columns are spans into the arena moved out of the
/// staging `Rows`.
pub struct ConvertedColumns {
    pub base_ns: Option<u64>,
    comm: rustc_hash::FxHashMap<usize, crate::comm::CommMetadata>,
    arena: String,
    cat: Vec<Option<Span>>,
    name: Vec<Option<Span>>,
    pub ts: Vec<f64>,
    pub dur: Vec<f64>,
    pub stream: Vec<i64>,
    pub grid: Vec<i64>,
    pg_desc: Vec<Option<Span>>,
    pg_name: Vec<Option<Span>>,
    ph: Vec<Option<Span>>,
    collective_name: Vec<Option<Span>>,
    pub in_nelems: Vec<i64>,
    pub out_nelems: Vec<i64>,
    pub group_size: Vec<i64>,
    pub counter_value: Vec<Option<f64>>,
}

impl ConvertedColumns {
    pub fn comm(&self, i: usize) -> Option<&crate::comm::CommMetadata> {
        self.comm.get(&i)
    }
    #[inline]
    fn s(&self, sp: Span) -> &str {
        &self.arena[sp.off as usize..sp.off as usize + sp.len as usize]
    }

    #[inline]
    pub fn cat(&self, i: usize) -> Option<&str> {
        self.cat[i].map(|sp| self.s(sp))
    }

    #[inline]
    pub fn name(&self, i: usize) -> Option<&str> {
        self.name[i].map(|sp| self.s(sp))
    }

    /// Empty string where the row had no value (the old `unwrap_or_default`).
    #[inline]
    pub fn pg_desc(&self, i: usize) -> &str {
        self.pg_desc[i].map_or("", |sp| self.s(sp))
    }

    /// Empty string where the row had no value (the old `unwrap_or_default`).
    #[inline]
    pub fn pg_name(&self, i: usize) -> &str {
        self.pg_name[i].map_or("", |sp| self.s(sp))
    }

    #[inline]
    pub fn ph(&self, i: usize) -> &str {
        self.ph[i].map_or("", |sp| self.s(sp))
    }

    #[inline]
    pub fn collective_name(&self, i: usize) -> &str {
        self.collective_name[i].map_or("", |sp| self.s(sp))
    }
}

/// Fast-path converter tail: sort + normalize into `ConvertedColumns`.
pub(crate) fn finish_columns(rows: &mut Rows) -> ConvertedColumns {
    let n = rows.len();
    let keys: Vec<SortKey> = (0..n)
        .map(|i| SortKey {
            ts: rows.ts[i],
            dur: rows.dur[i],
        })
        .collect();
    let perm = sort_permutation(&keys);
    ConvertedColumns {
        base_ns: rows.base_ns,
        comm: perm
            .iter()
            .enumerate()
            .filter_map(|(j, i)| rows.comm.remove(&(*i as usize)).map(|m| (j, m)))
            .collect(),
        cat: gather_opt(&rows.cat, &perm),
        name: gather_opt(&rows.name, &perm),
        ts: perm
            .iter()
            .map(|&i| rows.ts[i as usize].unwrap_or(0.0))
            .collect(),
        dur: perm
            .iter()
            .map(|&i| rows.dur[i as usize].unwrap_or(0.0))
            .collect(),
        stream: perm
            .iter()
            .map(|&i| rows.stream[i as usize].unwrap_or(-1))
            .collect(),
        grid: perm
            .iter()
            .map(|&i| rows.grid[i as usize].unwrap_or(-1))
            .collect(),
        pg_desc: gather_opt(&rows.pg_desc, &perm),
        pg_name: gather_opt(&rows.pg_name, &perm),
        ph: gather_opt(&rows.ph, &perm),
        collective_name: gather_opt(&rows.collective_name, &perm),
        in_nelems: perm
            .iter()
            .map(|&i| rows.in_nelems[i as usize].unwrap_or(-1))
            .collect(),
        out_nelems: perm
            .iter()
            .map(|&i| rows.out_nelems[i as usize].unwrap_or(-1))
            .collect(),
        group_size: perm
            .iter()
            .map(|&i| rows.group_size[i as usize].unwrap_or(-1))
            .collect(),
        counter_value: gather_opt(&rows.counter_value, &perm),
        arena: std::mem::take(&mut rows.arena),
    }
}

// ---------------------------------------------------------------------------
// Tree-free walk path (see cudagraph.rs).
// ---------------------------------------------------------------------------

fn compact_and_append<T: Clone>(
    v: &mut Vec<Option<T>>,
    keep: &[bool],
    n_meta: usize,
    meta: Option<T>,
) {
    let mut w = 0usize;
    for (i, &k) in keep.iter().enumerate().take(v.len()) {
        if k {
            v.swap(w, i);
            w += 1;
        }
    }
    v.truncate(w);
    for _ in 0..n_meta {
        v.push(meta.clone());
    }
}

// Row/raw-event shells, thread-local and reused across ranks: zero
// big-allocation churn in the rank loop.
thread_local! {
    static SHELL_POOL: std::cell::RefCell<
        Option<(Rows, Vec<crate::cudagraph::RawEvent>)>,
    > = const { std::cell::RefCell::new(None) };
}

/// How much slack a pooled buffer may carry before it is released instead of
/// retained for the next rank.
///
/// Pools are sized to the rank that built them, so retaining unconditionally
/// pins every worker at the largest rank it has ever seen — measured at
/// 1.6-2.0GB per thread, i.e. 180-220GB of one analysis process's RSS at
/// 24k ranks, for the whole run. That footprint is what turns ordinary host tenancy into direct
/// reclaim (measured: sys CPU 4.4% -> 25.6%, wall 1.8x). Releasing above 4x the
/// current rank keeps the steady state at ~4x a typical rank while leaving
/// the common case (successive ranks of similar size) allocation-free — the
/// regrowth this pooling exists to avoid only happens across a >4x step.
pub const RETAIN_FACTOR: usize = 4;

/// Take the thread-local row/raw shells (empty defaults on first use).
pub(crate) fn pool_take_shells() -> (Rows, Vec<crate::cudagraph::RawEvent>) {
    SHELL_POOL
        .with(|cell| cell.borrow_mut().take())
        .unwrap_or_default()
}

/// Drop this thread's retained shells outright. The rank loop calls this
/// while memory-gated: retained pools (1.6-2.0GB/thread on 24k-rank runs) are
/// exactly the footprint the gate is defending against, so under pressure
/// re-allocation is the cheaper side of the trade.
pub fn pool_clear() {
    SHELL_POOL.with(|cell| cell.borrow_mut().take());
}

/// Bytes this thread's retained shells pin (arena + column + raw-event
/// capacities). This is the worker's standing footprint between ranks — the
/// gate charges it to the next rank the thread runs (memgate footprint
/// samples are same-thread deltas, which cannot see capacity allocated by an
/// earlier rank).
pub fn pool_bytes() -> u64 {
    SHELL_POOL.with(|cell| {
        cell.borrow().as_ref().map_or(0, |(rows, raws)| {
            rows.bytes() + (raws.capacity() * size_of::<crate::cudagraph::RawEvent>()) as u64
        })
    })
}

/// Return the row/raw shells to the thread-local pool unless they are sized
/// for a much larger rank than the one just handled (see `RETAIN_FACTOR`).
/// `raws` is one entry per parsed event, so its capacity/len ratio is the
/// pool's slack; the arena and the column vectors scale with the same count.
pub(crate) fn pool_shells(rows: Rows, raws: Vec<crate::cudagraph::RawEvent>) {
    if raws.capacity() <= RETAIN_FACTOR * raws.len().max(1) {
        SHELL_POOL.with(|cell| *cell.borrow_mut() = Some((rows, raws)));
    }
}

pub struct ParsedRows {
    pub(crate) rows: Rows,
    pub(crate) raws: Vec<crate::cudagraph::RawEvent>,
}

/// Distinct nonzero live graph ids (value >> 32), same set the byte-scan
/// (`legacy::pkl::extract_live_graph_ids_from_raw`) yields on trace JSON: every
/// `"graph node id"` occurrence lives in a traceEvent's args.
pub fn live_gids(parsed: &ParsedRows) -> std::collections::HashSet<i64> {
    // Built with FxHash (one insert attempt per graphed row, millions per
    // rank) and handed back as the std set the callers take; the ids are
    // sorted downstream, so the hasher cannot affect any output.
    let mut fx: rustc_hash::FxHashSet<i64> = rustc_hash::FxHashSet::default();
    for r in &parsed.raws {
        if r.gnid != 0 {
            fx.insert(((r.gnid as u64) >> 32) as i64);
        }
    }
    fx.into_iter().collect()
}

/// The walk's artifact-relevant edits in parse (document) order — what the
/// splice writer (`legacy::splice`) needs to rewrite the original document
/// without the serde tree. Field semantics: `cudagraph::WalkOutcome`.
pub struct WalkPlan {
    pub keep: Vec<bool>,
    pub moved: Vec<u32>,
    pub fixed: Vec<(u32, f64)>,
    pub appended: Vec<crate::cudagraph::MetaRow>,
    pub retagged: Vec<(u32, i64, i64)>,
    pub retagged_ac2g: Vec<(u32, i64)>,
}

/// Shared phase-2 head: walk passes + compaction.
fn walk_compact(
    parsed: ParsedRows,
    annotations: &crate::annotations::AnnotationDigests,
    default_stream: i64,
    reassign: bool,
) -> (Rows, Vec<crate::cudagraph::RawEvent>, WalkPlan) {
    let ParsedRows { mut rows, mut raws } = parsed;

    if reassign {
        for (&i, m) in &mut rows.comm {
            if let Some(anns) = annotations.get(&raws[i].gnid) {
                for ann in anns {
                    m.overlay(&ann.comm);
                }
            }
        }
    }
    let outcome = {
        let mut wcols = crate::cudagraph::WalkColumns {
            arena: &mut rows.arena,
            ts: &mut rows.ts,
            stream: &mut rows.stream,
            pg_desc: &mut rows.pg_desc,
            pg_name: &mut rows.pg_name,
        };
        crate::cudagraph::apply_walks(&mut wcols, &mut raws, annotations, default_stream, reassign)
    };

    // Compact survivors + append synthesized thread_name metadata rows
    // (cat None, name "thread_name"; all else missing — the meta pid is
    // tree-path-only, its column is dead here).
    let keep = &outcome.keep;
    let n_meta = outcome.appended.len();
    // One arena string shared by every appended meta row's name span.
    let meta_name = if n_meta > 0 {
        Some(arena_push(&mut rows.arena, "thread_name"))
    } else {
        None
    };
    rows.comm = keep
        .iter()
        .enumerate()
        .filter(|(_, k)| **k)
        .enumerate()
        .filter_map(|(j, (i, _))| rows.comm.remove(&i).map(|m| (j, m)))
        .collect();
    compact_and_append(&mut rows.cat, keep, n_meta, None);
    compact_and_append(&mut rows.name, keep, n_meta, meta_name);
    compact_and_append(&mut rows.ts, keep, n_meta, None);
    compact_and_append(&mut rows.dur, keep, n_meta, None);
    compact_and_append(&mut rows.stream, keep, n_meta, None);
    compact_and_append(&mut rows.grid, keep, n_meta, None);
    compact_and_append(&mut rows.pg_desc, keep, n_meta, None);
    compact_and_append(&mut rows.pg_name, keep, n_meta, None);
    compact_and_append(&mut rows.ph, keep, n_meta, None);
    compact_and_append(&mut rows.collective_name, keep, n_meta, None);
    compact_and_append(&mut rows.in_nelems, keep, n_meta, None);
    compact_and_append(&mut rows.out_nelems, keep, n_meta, None);
    compact_and_append(&mut rows.group_size, keep, n_meta, None);
    compact_and_append(&mut rows.counter_value, keep, n_meta, None);

    let plan = WalkPlan {
        keep: outcome.keep,
        moved: outcome.moved,
        fixed: outcome.fixed,
        appended: outcome.appended,
        retagged: outcome.retagged,
        retagged_ac2g: outcome.retagged_ac2g,
    };
    (rows, raws, plan)
}

/// Phase 2 (materialized): walk head + gather into `ConvertedColumns`.
/// Kept for --verify-walks (bitwise column comparison needs both sides
/// materialized) and any caller of the gathered representation.
pub fn convert_parsed_walked(
    parsed: ParsedRows,
    annotations: &crate::annotations::AnnotationDigests,
    default_stream: i64,
    reassign: bool,
) -> Result<ConvertedColumns> {
    convert_parsed_walked_plan(parsed, annotations, default_stream, reassign).map(|(c, _)| c)
}

/// `convert_parsed_walked` + the walk's artifact plan (splice writer input).
pub fn convert_parsed_walked_plan(
    parsed: ParsedRows,
    annotations: &crate::annotations::AnnotationDigests,
    default_stream: i64,
    reassign: bool,
) -> Result<(ConvertedColumns, WalkPlan)> {
    let (mut rows, raws, plan) = walk_compact(parsed, annotations, default_stream, reassign);
    let cols = finish_columns(&mut rows);
    // Return the (now arena-less) column shells + raw-event buffer to the
    // thread-local pool: ~55-60MB of per-rank Vec churn otherwise re-faults
    // pages while every worker is active.
    SHELL_POOL.with(|cell| *cell.borrow_mut() = Some((rows, raws)));
    Ok((cols, plan))
}

/// Lean view over the post-walk rows in frame (globally sorted) order —
/// the fused-analysis input. Same 8 logical columns as `ConvertedColumns`
/// with identical null-fill semantics, but nothing is gathered: accessors
/// indirect through the sort permutation. Call `recycle()` when analysis is
/// done to return the shells to the thread-local pool.
pub struct WalkedView {
    rows: Rows,
    raws: Vec<crate::cudagraph::RawEvent>,
    perm: Vec<u32>,
}

impl WalkedView {
    pub fn base_ns(&self) -> Option<u64> {
        self.rows.base_ns
    }
    pub fn comm(&self, j: usize) -> Option<&crate::comm::CommMetadata> {
        self.rows.comm.get(&self.r(j))
    }
    #[inline]
    fn r(&self, j: usize) -> usize {
        self.perm[j] as usize
    }

    pub fn len(&self) -> usize {
        self.perm.len()
    }

    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    #[inline]
    pub fn cat(&self, j: usize) -> Option<&str> {
        self.rows.cat[self.r(j)].map(|sp| self.rows.get(sp))
    }

    #[inline]
    pub fn name(&self, j: usize) -> Option<&str> {
        self.rows.name[self.r(j)].map(|sp| self.rows.get(sp))
    }

    #[inline]
    pub fn ts(&self, j: usize) -> f64 {
        self.rows.ts[self.r(j)].unwrap_or(0.0)
    }

    #[inline]
    pub fn dur(&self, j: usize) -> f64 {
        self.rows.dur[self.r(j)].unwrap_or(0.0)
    }

    #[inline]
    pub fn stream(&self, j: usize) -> i64 {
        self.rows.stream[self.r(j)].unwrap_or(-1)
    }

    #[inline]
    pub fn grid(&self, j: usize) -> i64 {
        self.rows.grid[self.r(j)].unwrap_or(-1)
    }

    /// Empty string where the row had no value (the old `unwrap_or_default`).
    #[inline]
    pub fn pg_desc(&self, j: usize) -> &str {
        self.rows.pg_desc[self.r(j)].map_or("", |sp| self.rows.get(sp))
    }

    /// Empty string where the row had no value (the old `unwrap_or_default`).
    #[inline]
    pub fn pg_name(&self, j: usize) -> &str {
        self.rows.pg_name[self.r(j)].map_or("", |sp| self.rows.get(sp))
    }

    #[inline]
    pub fn ph(&self, j: usize) -> &str {
        self.rows.ph[self.r(j)].map_or("", |sp| self.rows.get(sp))
    }

    #[inline]
    pub fn collective_name(&self, j: usize) -> &str {
        self.rows.collective_name[self.r(j)].map_or("", |sp| self.rows.get(sp))
    }

    #[inline]
    pub fn in_nelems(&self, j: usize) -> i64 {
        self.rows.in_nelems[self.r(j)].unwrap_or(-1)
    }

    #[inline]
    pub fn out_nelems(&self, j: usize) -> i64 {
        self.rows.out_nelems[self.r(j)].unwrap_or(-1)
    }

    #[inline]
    pub fn group_size(&self, j: usize) -> i64 {
        self.rows.group_size[self.r(j)].unwrap_or(-1)
    }

    #[inline]
    pub fn counter_value(&self, j: usize) -> Option<f64> {
        self.rows.counter_value[self.r(j)]
    }

    /// Arena spans of the (pg_desc, pg_name) pair. Equal spans mean equal
    /// bytes, so a consumer that only needs "same pair as the previous row?"
    /// can answer with two integer compares instead of two string compares
    /// (`analysis::process_view`; the walk gives every row of a graph id
    /// the same spans, so the fast answer is the common one).
    #[inline]
    pub fn pg_spans(&self, j: usize) -> (Option<Span>, Option<Span>) {
        let r = self.r(j);
        (self.rows.pg_desc[r], self.rows.pg_name[r])
    }

    /// Return the shells (arena included; cleared on next recycle) to the
    /// thread-local pool.
    pub fn recycle(self) {
        pool_shells(self.rows, self.raws);
    }
}

/// Phase 2 (fused): walk head + sort permutation, NO gather — the analysis
/// consumes rows through the view in frame order. The perm and the per-row
/// values are exactly what `finish_columns` would have gathered.
pub fn convert_parsed_walked_view(
    parsed: ParsedRows,
    annotations: &crate::annotations::AnnotationDigests,
    default_stream: i64,
    reassign: bool,
) -> Result<WalkedView> {
    convert_parsed_walked_view_plan(parsed, annotations, default_stream, reassign).map(|(v, _)| v)
}

/// `convert_parsed_walked_view` + the walk's artifact plan (splice input).
pub fn convert_parsed_walked_view_plan(
    parsed: ParsedRows,
    annotations: &crate::annotations::AnnotationDigests,
    default_stream: i64,
    reassign: bool,
) -> Result<(WalkedView, WalkPlan)> {
    let (rows, raws, plan) = {
        let _t = crate::timing::Timer::new(crate::timing::Stage::Walk);
        walk_compact(parsed, annotations, default_stream, reassign)
    };
    let _t = crate::timing::Timer::new(crate::timing::Stage::Sort);
    let n = rows.len();
    let keys: Vec<SortKey> = (0..n)
        .map(|i| SortKey {
            ts: rows.ts[i],
            dur: rows.dur[i],
        })
        .collect();
    let perm = sort_permutation(&keys);
    Ok((WalkedView { rows, raws, perm }, plan))
}

// pftrace → converter rows bridge.
// ---------------------------------------------------------------------------

/// Build the converter's row representation from a decoded perfetto trace —
/// the pftrace counterpart of the JSON tape walk
/// (`legacy::json::parse_bytes_walked`), feeding the same walks and the same
/// analysis untouched.
///
/// Value semantics mirror the JSON twin field-for-field:
/// - timestamps: the twin writes µs with exactly three decimals, i.e. the
///   exact value `delta_ns / 1000`. `(delta_ns as f64) / 1000.0` IS the
///   correctly-rounded double of that decimal (delta_ns < 2^53 is exact and
///   IEEE division rounds once), so it equals what simd-json would have
///   parsed from the twin's literal — column value and walk-facing raw
///   (`ts_f`/`ts_bits`) alike, exactly like the JSON tape walk.
/// - `cat`: GPU stages map to the twin's category spellings (stage names are
///   exporter format constants); CPU slices carry their interned
///   `event_categories` string verbatim, and slices the exporter emits with
///   NO category (its overhead / cuda_sync groups) get a null `cat` — the
///   analysis drops cat-less rows, which is exactly what those rows are in
///   the twin's analysis too (`overhead`/`cuda_sync` feed no metric).
/// - `stream`: the resolver's logical stream when the stage carries one
///   (annotated collectives), else the original CUDA stream — the twin's
///   `args.stream` on the same rows.
/// - `tid`: CPU rows use the thread id; GPU rows use the logical lane id
///   recovered from the hardware-queue name (`"[0061] HSDP_SHARD"` → 61,
///   with the pre-fix `"stream 7"` spelling as fallback) — the twin's
///   display tid. The lane drives the walk's move/fix passes.
/// - annotations (`pg_desc`/`pg_name`/`collective_name`): from the
///   stage's inline extra_data / the cpu op's debug annotations. Interned
///   values share one arena span per distinct string, which is what the
///   analysis' span-equality memo wants.
///
/// Parse order is packet order (the exporter batches CPU groups before GPU
/// stages). The twin interleaves differently, and tied rows keep parse
/// order, so byte-identical artifacts across formats are out of scope by
/// construction; VALUES are what this bridge preserves.
pub fn convert_pftrace_rows(tr: &crate::trace::pftrace::PfTrace) -> Result<ParsedRows> {
    use crate::trace::pftrace::NO_NAME;
    use crate::trace::pftrace::PfCat;

    let n = tr.events.len();
    let (mut rows, mut raws) = SHELL_POOL
        .with(|cell| cell.borrow_mut().take())
        .unwrap_or_default();
    // Arena: one copy per DISTINCT string (names + annotation values), so the
    // hint is modest; shortfall grows by doubling like the tape path.
    rows.recycle(n, 64 * 1024);
    raws.clear();
    raws.reserve(n);

    let base = tr.base_ns.unwrap_or(0);

    // Logical lane id per hardware queue, from the queue NAME (a format
    // property of this exporter: the fixed generation writes
    // "[<zero-padded lane>] <label>", the original wrote "stream <N>" or a
    // bare resolver label). -1 when unrecoverable.
    let lanes: Vec<i64> = tr
        .hw_queues
        .iter()
        .map(|(name_id, _iid)| crate::trace::pftrace::logical_lane(tr.name(*name_id)).unwrap_or(-1))
        .collect();

    // One arena span per distinct interned string id, per logical column
    // group (values shared across columns would still compare equal by
    // bytes; sharing per-id keeps the span-equality fast path hot).
    let mut span_of: rustc_hash::FxHashMap<u32, Span> = rustc_hash::FxHashMap::default();
    macro_rules! span_for {
        ($rows:expr, $id:expr) => {{
            let id: u32 = $id;
            if id == NO_NAME {
                None
            } else {
                Some(match span_of.get(&id) {
                    Some(sp) => *sp,
                    None => {
                        let sp = arena_push(&mut $rows.arena, tr.name(id));
                        span_of.insert(id, sp);
                        sp
                    }
                })
            }
        }};
    }
    // GPU category spellings, pushed once (format constants of the twin).
    let mut cat_kernel: Option<Span> = None;
    let mut cat_memcpy: Option<Span> = None;
    let mut cat_memset: Option<Span> = None;
    let mut cat_gua: Option<Span> = None;
    let mut cat_counter: Option<Span> = None;
    let mut ph_counter: Option<Span> = None;

    rows.base_ns = tr.base_ns;
    for (event_idx, e) in tr.events.iter().enumerate() {
        let raw_ts = (e.ts_ns.saturating_sub(base)) as f64 / 1000.0;
        let raw_dur = e.dur_ns as f64 / 1000.0;

        // cat column + walk booleans.
        let (cat_span, cat_is_work, cat_is_kernel, cat_is_gua);
        match e.cat {
            PfCat::Kernel => {
                if cat_kernel.is_none() {
                    cat_kernel = Some(arena_push(&mut rows.arena, "kernel"));
                }
                cat_span = cat_kernel;
                (cat_is_work, cat_is_kernel, cat_is_gua) = (true, true, false);
            }
            PfCat::Memcpy => {
                if cat_memcpy.is_none() {
                    cat_memcpy = Some(arena_push(&mut rows.arena, "gpu_memcpy"));
                }
                cat_span = cat_memcpy;
                (cat_is_work, cat_is_kernel, cat_is_gua) = (true, false, false);
            }
            PfCat::Memset => {
                if cat_memset.is_none() {
                    cat_memset = Some(arena_push(&mut rows.arena, "gpu_memset"));
                }
                cat_span = cat_memset;
                (cat_is_work, cat_is_kernel, cat_is_gua) = (true, false, false);
            }
            PfCat::GpuAnnotation => {
                if cat_gua.is_none() {
                    cat_gua = Some(arena_push(&mut rows.arena, "gpu_user_annotation"));
                }
                cat_span = cat_gua;
                (cat_is_work, cat_is_kernel, cat_is_gua) = (false, false, true);
            }
            PfCat::GpuCounter => {
                if cat_counter.is_none() {
                    cat_counter = Some(arena_push(&mut rows.arena, "gpu_counter"));
                }
                cat_span = cat_counter;
                (cat_is_work, cat_is_kernel, cat_is_gua) = (false, false, false);
            }
            // GPU stage with an unrecognized name: carry the stage name as
            // the category (no twin spelling exists; analysis treats it as
            // any other non-work cat). CPU slices: the interned category
            // string, None when the slice has none.
            PfCat::GpuOther | PfCat::CpuSlice => {
                let id = if matches!(e.cat, PfCat::CpuSlice) {
                    e.cat_name
                } else {
                    e.name
                };
                cat_span = span_for!(rows, id);
                (cat_is_work, cat_is_kernel, cat_is_gua) = (false, false, false);
            }
        }

        // tid/pid (walk-facing).
        let (tid_i, pid_i) = match e.cat {
            PfCat::CpuSlice => match tr.threads.get(e.track as usize) {
                Some(t) => (t.tid, t.pid),
                None => (0, 0),
            },
            _ => {
                let lane = lanes.get(e.track as usize).copied().unwrap_or(-1);
                (if lane >= 0 { lane } else { 0 }, e.gpu_id.max(0))
            }
        };

        // stream column: logical when assigned, else the original stream.
        let stream_col = match e.cat {
            PfCat::CpuSlice => None,
            _ if e.logical_stream >= 0 => Some(e.logical_stream),
            _ if e.stream >= 0 => Some(e.stream),
            _ => None,
        };

        if let Some(m) = tr.comm.get(&event_idx) {
            rows.comm.insert(rows.ts.len(), m.clone());
        }
        rows.cat.push(cat_span);
        rows.name.push(span_for!(rows, e.name));
        rows.ts.push(Some(raw_ts));
        rows.dur.push(Some(raw_dur));
        rows.stream.push(stream_col);
        rows.grid
            .push(if e.grid0 >= 0 { Some(e.grid0) } else { None });
        rows.pg_desc.push(span_for!(rows, e.pg_desc));
        rows.pg_name.push(span_for!(rows, e.pg_name));
        rows.ph.push(if matches!(e.cat, PfCat::GpuCounter) {
            if ph_counter.is_none() {
                ph_counter = Some(arena_push(&mut rows.arena, "C"));
            }
            ph_counter
        } else {
            None
        });
        rows.collective_name
            .push(span_for!(rows, e.collective_name));
        rows.in_nelems
            .push((e.in_nelems >= 0).then_some(e.in_nelems));
        rows.out_nelems
            .push((e.out_nelems >= 0).then_some(e.out_nelems));
        rows.group_size
            .push((e.group_size >= 0).then_some(e.group_size));
        rows.counter_value.push(e.counter_value);

        raws.push(crate::cudagraph::RawEvent {
            tid_i,
            tid_bits: Some((tid_i as f64).to_bits()),
            ts_bits: Some(raw_ts.to_bits()),
            ts_f: raw_ts,
            dur_f: raw_dur,
            pid_i,
            ph_is_m: false,
            ph_is_f: false,
            cat_is_ac2g: false,
            cat_is_work,
            cat_is_gua,
            cat_is_kernel,
            // No metadata rows exist in a pftrace; the walk synthesizes the
            // thread_name rows it needs (they are analysis-invisible).
            is_thread_name_meta: false,
            meta_stream_name: false,
            gnid: e.gnid,
        });
    }

    Ok(ParsedRows { rows, raws })
}

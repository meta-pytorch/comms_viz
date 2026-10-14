// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! The perfetto native (`.pftrace`) trace format.
//!
//! Decodes the packet stream written by PyTorch's Cuspy profiler backend
//! (formerly the CUPTI monitor) with a hand-rolled protobuf wire-format walker (no proto codegen, no new
//! dependencies).
//!
//! Field numbers below are format constants of perfetto's TracePacket schema
//! plus the exporter's declared extension fields. Nothing here encodes
//! assumptions about a particular job, model, or run — names, queues, gids,
//! streams, and categories are all discovered from the file.
//!
//! Event families in this exporter:
//!   - CPU ops → TrackEvent TYPE_SLICE_BEGIN/END pairs on thread tracks
//!     (names interned; debug annotations carry Ev Idx / External id /
//!     correlation / nccl PG metadata);
//!   - GPU work → GpuRenderStageEvent packets (stage = Kernel / Memcpy /
//!     Memset / Annotation; hw queue = the annotation-resolver-named track;
//!     extra_data carries stream id / correlation / graph id / graph node id;
//!     extensions carry kernel name and launch config incl. grid);
//!   - GPU counters → GpuCounterEvent packets (counter samples used by the
//!     diagnostics timeline);
//!   - track naming → TrackDescriptor (process/threads) + interned GPU
//!     specifications (queues/stages).
//!
//! ## Status: ingestion live (fixed-exporter generation)
//!
//! Three exporter-side gaps used to gate ingestion; all three are closed in
//! the current exporter generation:
//!   1. CPU-slice categories — now emitted (`TrackEvent.category_iids` +
//!      `InternedData.event_categories`), decoded here.
//!   2. Time base — now emitted (a `ClockSnapshot` first packet whose clocks
//!      both carry `baseTimeNanoseconds`), decoded here; timestamps land in
//!      the same session frame as the JSON twin's columns.
//!   3. PG/comms metadata — now emitted INLINE on GPU collective stages as
//!      string `extra_data` (Process Group Description/Name, dtype, Group
//!      size, msg nelems, logical stream), decoded here. This is also why
//!      [`SELF_ANNOTATED`] is true: a fixed-generation pftrace needs no
//!      side-channel annotation pickles.
//!
//! Traces from PRE-fix exporters still decode, degraded exactly as their
//! content dictates: no categories → CPU rows arrive cat-less and analysis
//! drops them (their cuda* totals read zero); no ClockSnapshot →
//! timestamps sit in an absolute-ns frame; no inline PG extras → PG columns
//! stay empty. Nothing guesses at missing content.
//!
//! Known twin divergences (values, not plumbing): ac2g flow rows and
//! `ph:"M"` metadata rows do not exist in a pftrace. Both are analysis-
//! invisible (flow rows are stream −1 / non-work; meta rows are cat-less),
//! but they ARE rows in the JSON parse stream, so parse order — and with it
//! tied-row order — differs across formats ⇒ cross-format artifacts can
//! differ in float low bits and tied-row order. Byte-identical artifacts
//! across formats are out of scope by construction; matching VALUES are the
//! contract.

use anyhow::Result;
use anyhow::anyhow;
use rustc_hash::FxHashMap;

// --- TracePacket field numbers (perfetto trace_packet.proto). ---
const TP_CLOCK_SNAPSHOT: u64 = 6; // ClockSnapshot{clocks=1: Clock{clock_id=1, timestamp=2}}
const TP_TIMESTAMP: u64 = 8;
const TP_SEQ_ID: u64 = 10;
const TP_TRACK_EVENT: u64 = 11;
const TP_INTERNED: u64 = 12;
const TP_SEQ_FLAGS: u64 = 13;
const TP_GPU_COUNTER: u64 = 52;
const TP_GPU_RENDER_STAGE: u64 = 53;
const TP_TRACK_DESCRIPTOR: u64 = 60;

const SEQ_INCREMENTAL_STATE_CLEARED: u64 = 1;

// --- TrackEvent (track_event.proto). ---
const TE_CATEGORY_IIDS: u64 = 3; // repeated NON-PACKED uint64 → event_categories
const TE_DEBUG_ANN: u64 = 4;
const TE_TYPE: u64 = 9;
const TE_NAME_IID: u64 = 10;
const TE_TRACK_UUID: u64 = 11;
const TE_NAME: u64 = 23;
/// Exporter extension: correlation id linking a runtime launch slice to the
/// GPU event with the same `correlation` extra.
const TE_EXT_CORRELATION: u64 = 3000;

const TE_TYPE_BEGIN: u64 = 1;
const TE_TYPE_END: u64 = 2;

// --- TrackDescriptor / ProcessDescriptor / ThreadDescriptor. ---
const TD_UUID: u64 = 1;
const TD_PROCESS: u64 = 3;
const TD_THREAD: u64 = 4;
const PD_PID: u64 = 1;
const PD_NAME: u64 = 6;
const THD_PID: u64 = 1;
const THD_TID: u64 = 2;
const THD_NAME: u64 = 5;

// --- InternedData. ---
const ID_EVENT_CATEGORIES: u64 = 1; // {iid=1, name=2}
const ID_EVENT_NAMES: u64 = 2; // {iid=1, name=2}
const ID_GPU_SPECS: u64 = 24; // {iid=1, name=2} (stages + hw queues)
const ID_GPU_COUNTER_DESCRIPTORS: u64 = 47;
const ID_EXT_KERNEL_NAMES: u64 = 1000; // {iid=1, name=3}

// --- GpuCounterEvent / GpuCounterDescriptor. ---
const GCE_DESCRIPTOR: u64 = 1;
const GCE_COUNTER: u64 = 2;
const GCE_GPU_ID: u64 = 3;
const GCE_DESCRIPTOR_IID: u64 = 4;
const GCD_SPEC: u64 = 1;
const GCS_ID: u64 = 1;
const GCS_NAME: u64 = 2;
const GC_ID: u64 = 1;
const GC_INT_VALUE: u64 = 2;
const GC_DOUBLE_VALUE: u64 = 3;
const IGCD_IID: u64 = 1;
const IGCD_DESCRIPTOR: u64 = 2;
const IGCD_GPU_ID: u64 = 3;

// --- GpuRenderStageEvent (+ exporter extensions). ---
const RS_EVENT_ID: u64 = 1; // == correlation id
const RS_DURATION: u64 = 2; // ns
const RS_EXTRA_DATA: u64 = 6; // {name=1, value=2}
const RS_GPU_ID: u64 = 11;
const RS_HW_QUEUE_IID: u64 = 13;
const RS_STAGE_IID: u64 = 14;
const RS_EXT_KERNEL_NAME_IID: u64 = 19; // → InternedData[1000]
const RS_EXT_EVENT_NAME_IID: u64 = 20; // → InternedData.event_names
const RS_EXT_LAUNCH: u64 = 21; // {grid=1{x,y,z}, block=2{...}, ann=3{iid,val}}

// --- DebugAnnotation (subset). ---
const DA_NAME: u64 = 10; // inline string name (this exporter interns none)
const DA_INT: u64 = 4;
const DA_STRING: u64 = 6;

/// Event category, derived from the packet family + interned stage name.
/// CPU slices have NO category in the pftrace (gap vs JSON `cat`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PfCat {
    CpuSlice,
    Kernel,
    Memcpy,
    Memset,
    GpuAnnotation,
    GpuCounter,
    /// GPU stage not named Kernel/Memcpy/Memset/Annotation.
    GpuOther,
}

pub const NO_NAME: u32 = u32::MAX;

/// One decoded event (CPU slice or GPU stage event), parse order.
pub struct PfEvent {
    pub cat: PfCat,
    /// CPU slices: interned category string (`cpu_op` / `cuda_runtime` / …)
    /// from `TrackEvent.category_iids`; NO_NAME when the slice carries no
    /// category (the exporter's overhead / cuda_sync groups). GPU rows:
    /// NO_NAME — their category is the stage kind in `cat`.
    pub cat_name: u32,
    /// Index into `PfTrace::names` (NO_NAME when absent).
    pub name: u32,
    /// Absolute timestamp, ns.
    pub ts_ns: u64,
    pub dur_ns: u64,
    /// CPU: index into `threads`; GPU: index into `hw_queues`.
    pub track: u32,
    /// GPU: original CUDA stream id from extra_data ("stream id"); -1 CPU.
    pub stream: i64,
    /// GPU: resolver-assigned logical stream from the "stream" extra (only
    /// present on annotated collective stages); -1 otherwise. This is the
    /// value the JSON twin exposes as `args.stream` on annotated rows.
    pub logical_stream: i64,
    /// GPU device id (`gpu_id` field / "device id" extra) — the JSON twin's
    /// pid on GPU rows. CPU rows: -1 (pid comes from the thread descriptor).
    pub gpu_id: i64,
    /// grid.x from the launch-config extension (kernels; -1 otherwise).
    pub grid0: i64,
    /// args["graph node id"] equivalent, in the JSON twin's COMBINED
    /// `(graph_id << 32) | node_id` encoding; 0 when absent. Reconstructed
    /// from the separate "graph id"/"graph node id" extras when the exporter
    /// emits them split (see `combine_gnid`).
    pub gnid: i64,
    /// correlation id (GPU event_id / extra, CPU annotation/ext; -1 absent).
    pub correlation: i64,
    /// Inline annotation values (interned; NO_NAME when absent): the four
    /// converter columns the analysis reads. On GPU collective stages these
    /// arrive as string extra_data (exporter generations with the comms-
    /// metadata fix); on nccl CPU ops as debug annotations.
    pub pg_desc: u32,
    pub pg_name: u32,
    pub dtype: u32,
    pub collective_name: u32,
    pub in_nelems: i64,
    pub out_nelems: i64,
    pub group_size: i64,
    pub counter_value: Option<f64>,
}

pub struct PfThread {
    pub pid: i64,
    pub tid: i64,
    pub name: Option<u32>,
}

#[derive(Default)]
pub struct PfTrace {
    pub comm: FxHashMap<usize, crate::comm::CommMetadata>,
    pub events: Vec<PfEvent>,
    /// Unified string table (interned event/kernel/spec names).
    pub names: Vec<String>,
    pub threads: Vec<PfThread>,
    /// (name index, interned iid) per hw queue, discovery order.
    pub hw_queues: Vec<(u32, u64)>,
    pub process: Option<(i64, u32)>,
    pub n_packets: usize,
    pub n_counter_packets: usize,
    pub unmatched_begin: usize,
    pub unmatched_end: usize,
    /// CPU-slice debug-annotation name → occurrence count (discovery only).
    /// Debug-annotation name → count, keyed by raw wire bytes (counting
    /// must not allocate per annotation); names that are invalid UTF-8
    /// are dropped by consumers, matching the old from_utf8-then-count.
    pub cpu_ann_counts: FxHashMap<Vec<u8>, u64>,
    /// `baseTimeNanoseconds` equivalent, from the ClockSnapshot packet the
    /// fixed exporter emits first (REALTIME == BOOTTIME == base). None on
    /// pre-fix traces, which carry no time base at all.
    pub base_ns: Option<u64>,
}

impl PfTrace {
    pub fn name(&self, id: u32) -> &str {
        if id == NO_NAME {
            ""
        } else {
            &self.names[id as usize]
        }
    }

    /// Distinct nonzero live graph ids (gnid >> 32), same convention as
    /// `rows::live_gids` on the JSON path (`gnid` is stored combined).
    pub fn live_gids(&self) -> std::collections::HashSet<i64> {
        let mut out = std::collections::HashSet::new();
        for e in &self.events {
            if e.gnid != 0 {
                out.insert(((e.gnid as u64) >> 32) as i64);
            }
        }
        out
    }
}

// --- wire-format primitives ---

/// One- and two-byte fast paths inline into the field-walk loops (field
/// keys, wiretypes, iids and most lengths fit in ≤2 bytes — 8.7% of shard
/// on-CPU time was this decode); the general loop
/// stays out of line.
#[inline(always)]
fn varint(b: &[u8], i: &mut usize) -> Result<u64> {
    if let Some(&x) = b.get(*i) {
        if x < 0x80 {
            *i += 1;
            return Ok(x as u64);
        }
        if let Some(&y) = b.get(*i + 1) {
            if y < 0x80 {
                *i += 2;
                return Ok(((y as u64) << 7) | (x & 0x7f) as u64);
            }
        }
    }
    varint_multi(b, i)
}

// NOT #[cold]: timestamps and track uuids are 8-9 byte varints, so the
// general loop is a common case — it is only kept out of line so the ≤2
// byte path inlines small.
fn varint_multi(b: &[u8], i: &mut usize) -> Result<u64> {
    let mut r: u64 = 0;
    let mut s = 0u32;
    loop {
        let x = *b.get(*i).ok_or_else(|| anyhow!("truncated varint"))?;
        *i += 1;
        r |= ((x & 0x7f) as u64) << s;
        if x & 0x80 == 0 {
            return Ok(r);
        }
        s += 7;
        if s >= 64 {
            return Err(anyhow!("varint overflow"));
        }
    }
}

/// Iterate one message's (field, wiretype, payload-slice-or-varint) triples.
struct Fields<'a> {
    b: &'a [u8],
    i: usize,
}

enum Val<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed64(u64),
    Fixed32,
}

impl<'a> Fields<'a> {
    fn new(b: &'a [u8]) -> Self {
        Fields { b, i: 0 }
    }

    #[inline(always)]
    fn next_field(&mut self) -> Result<Option<(u64, Val<'a>)>> {
        if self.i >= self.b.len() {
            return Ok(None);
        }
        let key = varint(self.b, &mut self.i)?;
        let (f, wt) = (key >> 3, key & 7);
        let v = match wt {
            0 => Val::Varint(varint(self.b, &mut self.i)?),
            1 => {
                let bytes = self
                    .b
                    .get(self.i..self.i + 8)
                    .ok_or_else(|| anyhow!("truncated fixed64"))?;
                self.i += 8;
                Val::Fixed64(u64::from_le_bytes(
                    bytes.try_into().expect("fixed64 has eight bytes"),
                ))
            }
            2 => {
                let n = varint(self.b, &mut self.i)? as usize;
                let s = self
                    .b
                    .get(self.i..self.i + n)
                    .ok_or_else(|| anyhow!("truncated bytes"))?;
                self.i += n;
                Val::Bytes(s)
            }
            5 => {
                self.b
                    .get(self.i..self.i + 4)
                    .ok_or_else(|| anyhow!("truncated fixed32"))?;
                self.i += 4;
                Val::Fixed32
            }
            _ => return Err(anyhow!("unsupported wiretype {wt}")),
        };
        Ok(Some((f, v)))
    }
}

// --- per-sequence interning state ---

#[derive(Default)]
struct SeqState {
    event_names: FxHashMap<u64, u32>,
    /// category iid → interned name (InternedData.event_categories).
    categories: FxHashMap<u64, u32>,
    kernel_names: FxHashMap<u64, u32>,
    /// gpu spec iid → (name id, is-known-stage marker resolved lazily)
    specs: FxHashMap<u64, u32>,
    counter_descriptors: FxHashMap<u64, CounterDescriptor>,
    inline_counter_descriptor: CounterDescriptor,
    /// hw queue iid → index into PfTrace::hw_queues
    queues: FxHashMap<u64, u32>,
}

#[derive(Default)]
struct CounterDescriptor {
    gpu_id: Option<i64>,
    names: FxHashMap<u64, u32>,
}

/// One buffered TrackEvent, pending timestamp-ordered slice pairing.
///
/// BEGIN/END packets cannot be paired in packet order: the fixed exporter
/// batches each slice group as ALL its BEGINs followed by ALL its ENDs, and
/// groups from different kinds interleave per track, so packet order is not
/// timestamp order (measured: per-track BEGIN ts non-monotonic). Pairing is
/// therefore done at EOF, per track, over the timestamp-sorted event list
/// with a stack simulation — which is also correct for the original
/// exporter's interleaved, properly nested emission. Equal-ts ties keep
/// arrival order (stable sort); ENDs carry no payload, so any pairing of
/// equal-ts ENDs yields identical slices.
struct PendingTe {
    ts: u64,
    is_end: bool,
    name: u32,
    cat_name: u32,
    correlation: i64,
    pg_desc: u32,
    pg_name: u32,
    dtype: u32,
    collective_name: u32,
    in_nelems: i64,
    out_nelems: i64,
    group_size: i64,
}

/// Byte-keyed interner: the dedup lookup runs on the raw wire bytes, so
/// UTF-8 validation happens once per UNIQUE string instead of once per
/// occurrence (8.1% of shard on-CPU time was `from_utf8` re-validating
/// names the table already held). Returns None for
/// invalid UTF-8 — exactly the strings the old `from_utf8(s).ok()?` +
/// intern path dropped.
fn intern_bytes(
    names: &mut Vec<String>,
    dedup: &mut FxHashMap<Vec<u8>, u32>,
    s: &[u8],
) -> Option<u32> {
    if let Some(&id) = dedup.get(s) {
        return Some(id);
    }
    let valid = std::str::from_utf8(s).ok()?;
    let id = names.len() as u32;
    names.push(valid.to_string());
    dedup.insert(s.to_vec(), id);
    Some(id)
}

fn intern(names: &mut Vec<String>, dedup: &mut FxHashMap<Vec<u8>, u32>, s: &str) -> u32 {
    // &str input is valid by construction — intern_bytes cannot refuse it.
    intern_bytes(names, dedup, s.as_bytes()).expect("valid utf8")
}

/// C++-`stoll`-ish prefix parse for extra_data values (they are decimal
/// strings in this exporter). Operates on the raw wire bytes: stoll only
/// ever consumes an ASCII prefix, so UTF-8 validation adds nothing.
fn parse_i64(s: &[u8]) -> Option<i64> {
    crate::rows::stoll_prefix_bytes(s)
}

/// Decode a decompressed pftrace byte stream.
pub fn read_trace(bytes: &[u8]) -> Result<PfTrace> {
    let mut tr = PfTrace::default();
    let mut dedup: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    let mut seqs: FxHashMap<u64, SeqState> = FxHashMap::default();
    let mut track_threads: FxHashMap<u64, u32> = FxHashMap::default();
    // Per-track pending slice events (see `PendingTe`), tracks in first-seen
    // order so the emitted event order is deterministic.
    let mut pending: FxHashMap<u64, Vec<PendingTe>> = FxHashMap::default();
    let mut track_order: Vec<u64> = Vec::new();

    let mut top = Fields::new(bytes);
    while let Some((f, v)) = top.next_field()? {
        let Val::Bytes(pkt) = v else {
            continue;
        };
        if f != 1 {
            continue; // Trace has only `packet = 1`
        }
        tr.n_packets += 1;

        // First pass over the packet: locate ts/seq/flags and payload.
        let mut ts: u64 = 0;
        let mut seq_id: u64 = 0;
        let mut track_event: Option<&[u8]> = None;
        let mut render_stage: Option<&[u8]> = None;
        let mut gpu_counter: Option<&[u8]> = None;
        let mut interned: Option<&[u8]> = None;
        let mut track_desc: Option<&[u8]> = None;
        let mut clock_snapshot: Option<&[u8]> = None;
        let mut flags: u64 = 0;
        let mut fs = Fields::new(pkt);
        while let Some((f2, v2)) = fs.next_field()? {
            match (f2, v2) {
                (TP_TIMESTAMP, Val::Varint(x)) => ts = x,
                (TP_SEQ_ID, Val::Varint(x)) => seq_id = x,
                (TP_SEQ_FLAGS, Val::Varint(x)) => flags = x,
                (TP_TRACK_EVENT, Val::Bytes(b)) => track_event = Some(b),
                (TP_GPU_RENDER_STAGE, Val::Bytes(b)) => render_stage = Some(b),
                (TP_INTERNED, Val::Bytes(b)) => interned = Some(b),
                (TP_TRACK_DESCRIPTOR, Val::Bytes(b)) => track_desc = Some(b),
                (TP_CLOCK_SNAPSHOT, Val::Bytes(b)) => clock_snapshot = Some(b),
                (TP_GPU_COUNTER, Val::Bytes(b)) => {
                    tr.n_counter_packets += 1;
                    gpu_counter = Some(b);
                }
                _ => {}
            }
        }

        // ClockSnapshot: the fixed exporter's time base. Both clocks
        // (REALTIME=1, BOOTTIME=6) carry base_ns; any clock's timestamp is
        // the base. First snapshot wins (exactly one is emitted).
        if let Some(cs) = clock_snapshot {
            if tr.base_ns.is_none() {
                let mut css = Fields::new(cs);
                while let Some((cf, cv)) = css.next_field()? {
                    if let (1, Val::Bytes(clk)) = (cf, cv) {
                        let mut ks = Fields::new(clk);
                        while let Some((kf, kv)) = ks.next_field()? {
                            if let (2, Val::Varint(t)) = (kf, kv) {
                                tr.base_ns = Some(t);
                            }
                        }
                    }
                }
            }
        }

        let seq = seqs.entry(seq_id).or_default();
        if flags & SEQ_INCREMENTAL_STATE_CLEARED != 0 {
            seq.event_names.clear();
            seq.categories.clear();
            seq.kernel_names.clear();
            seq.specs.clear();
            seq.counter_descriptors.clear();
            seq.inline_counter_descriptor = CounterDescriptor::default();
        }

        if let Some(b) = interned {
            decode_interned(b, seq, &mut tr.names, &mut dedup)?;
        }
        if let Some(b) = track_desc {
            decode_track_descriptor(b, &mut tr, &mut dedup, &mut track_threads)?;
        }
        if let Some(b) = track_event {
            decode_track_event(
                b,
                ts,
                seq,
                &mut tr,
                &mut dedup,
                &mut pending,
                &mut track_order,
            )?;
        }
        if let Some(b) = render_stage {
            decode_render_stage(b, ts, seq, &mut tr, &mut dedup)?;
        }
        if let Some(b) = gpu_counter {
            decode_gpu_counter(b, ts, seq, &mut tr, &mut dedup)?;
        }
    }

    // Pair the buffered slice events: per track, timestamp order (stable for
    // ties), stack simulation. CPU slices therefore land in tr.events after
    // the GPU stages, tracks in first-seen order — a deterministic order,
    // which is all downstream requires (parse order is already
    // cross-format-divergent by construction).
    for uuid in track_order {
        let mut evs = pending.remove(&uuid).unwrap_or_default();
        evs.sort_by_key(|e| e.ts);
        let track = track_threads.get(&uuid).copied().unwrap_or(u32::MAX);
        let mut stack: Vec<PendingTe> = Vec::new();
        for e in evs {
            if !e.is_end {
                stack.push(e);
                continue;
            }
            match stack.pop() {
                Some(open) => {
                    tr.events.push(PfEvent {
                        cat: PfCat::CpuSlice,
                        cat_name: open.cat_name,
                        name: open.name,
                        ts_ns: open.ts,
                        dur_ns: e.ts.saturating_sub(open.ts),
                        track,
                        stream: -1,
                        logical_stream: -1,
                        gpu_id: -1,
                        grid0: -1,
                        gnid: 0,
                        correlation: if open.correlation != -1 {
                            open.correlation
                        } else {
                            e.correlation
                        },
                        pg_desc: open.pg_desc,
                        pg_name: open.pg_name,
                        dtype: open.dtype,
                        collective_name: open.collective_name,
                        in_nelems: open.in_nelems,
                        out_nelems: open.out_nelems,
                        group_size: open.group_size,
                        counter_value: None,
                    });
                }
                None => tr.unmatched_end += 1,
            }
        }
        tr.unmatched_begin += stack.len();
    }
    Ok(tr)
}

/// Head bytes to read from a trace file when probing for its time base,
/// and (smaller — inflate is the cost driver at large rank counts) how many
/// decoded bytes of a `.gz` head to scan. The fixed exporter emits the
/// ClockSnapshot as the FIRST packet (~40 bytes in), so both caps carry
/// orders-of-magnitude margin; a deeper-than-cap snapshot degrades to "no
/// base", i.e. an un-zoomed deep link, never a wrong one.
const BASE_NS_HEAD: usize = 64 * 1024;
const BASE_NS_HEAD_DECODED: usize = 4 * 1024;

/// Scan the head of a decompressed pftrace stream for the exporter's
/// ClockSnapshot time base (first snapshot wins, [`read_trace`]'s rule).
/// Truncated input is expected — callers feed a bounded head — so a wire
/// error just ends the scan. `None` = no snapshot seen: a pre-fix trace,
/// whose timestamps are already absolute (base 0).
pub fn read_base_ns(bytes: &[u8]) -> Option<u64> {
    let mut top = Fields::new(bytes);
    while let Ok(Some((f, v))) = top.next_field() {
        let Val::Bytes(pkt) = v else { continue };
        if f != 1 {
            continue;
        }
        let mut fs = Fields::new(pkt);
        while let Ok(Some((f2, v2))) = fs.next_field() {
            let (TP_CLOCK_SNAPSHOT, Val::Bytes(cs)) = (f2, v2) else {
                continue;
            };
            // Same value resolution as read_trace: the last Clock.timestamp
            // of the snapshot (the exporter pins REALTIME == BOOTTIME ==
            // base, so they agree; faithfulness matters because THIS value
            // must equal the base the converter rebased the columns by).
            let mut base = None;
            let mut css = Fields::new(cs);
            while let Ok(Some((cf, cv))) = css.next_field() {
                if let (1, Val::Bytes(clk)) = (cf, cv) {
                    let mut ks = Fields::new(clk);
                    while let Ok(Some((kf, kv))) = ks.next_field() {
                        if let (2, Val::Varint(t)) = (kf, kv) {
                            base = Some(t);
                        }
                    }
                }
            }
            if base.is_some() {
                return base;
            }
        }
    }
    None
}

/// [`read_base_ns`] over a `.pftrace`/`.pftrace.gz` file, reading only the
/// head — never the trace body. IO/decode failures read as "no snapshot".
pub fn read_base_ns_path(path: &std::path::Path) -> Option<u64> {
    use std::io::Read;
    let f = std::fs::File::open(path).ok()?;
    let mut head = Vec::new();
    f.take(BASE_NS_HEAD as u64).read_to_end(&mut head).ok()?;
    if head.starts_with(&[0x1f, 0x8b]) {
        let mut out = vec![0u8; BASE_NS_HEAD_DECODED];
        let mut dec = flate2::read::MultiGzDecoder::new(&head[..]);
        let mut n = 0;
        // A truncated member errors once the buffered input runs dry;
        // whatever decoded before that is the head we wanted.
        loop {
            match dec.read(&mut out[n..]) {
                Ok(0) => break,
                Ok(k) => {
                    n += k;
                    if n == out.len() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        read_base_ns(&out[..n])
    } else {
        read_base_ns(&head)
    }
}

fn decode_interned(
    b: &[u8],
    seq: &mut SeqState,
    names: &mut Vec<String>,
    dedup: &mut FxHashMap<Vec<u8>, u32>,
) -> Result<()> {
    let mut fs = Fields::new(b);
    while let Some((f, v)) = fs.next_field()? {
        let Val::Bytes(entry) = v else { continue };
        if f == ID_GPU_COUNTER_DESCRIPTORS {
            let mut iid = 0;
            let mut gpu_id = None;
            let mut descriptor = None;
            let mut es = Fields::new(entry);
            while let Some((f2, v2)) = es.next_field()? {
                match (f2, v2) {
                    (IGCD_IID, Val::Varint(value)) => iid = value,
                    (IGCD_DESCRIPTOR, Val::Bytes(value)) => descriptor = Some(value),
                    (IGCD_GPU_ID, Val::Varint(value)) => gpu_id = Some(value as i64),
                    _ => {}
                }
            }
            if let Some(descriptor) = descriptor {
                let mut decoded = decode_counter_descriptor(descriptor, names, dedup)?;
                decoded.gpu_id = gpu_id;
                seq.counter_descriptors.insert(iid, decoded);
            }
            continue;
        }
        let (name_field, table): (u64, u8) = match f {
            ID_EVENT_NAMES => (2, 0),
            ID_GPU_SPECS => (2, 1),
            ID_EXT_KERNEL_NAMES => (3, 2),
            ID_EVENT_CATEGORIES => (2, 3),
            _ => continue,
        };
        let mut iid: u64 = 0;
        let mut nm: Option<&str> = None;
        let mut es = Fields::new(entry);
        while let Some((f2, v2)) = es.next_field()? {
            match (f2, v2) {
                (1, Val::Varint(x)) => iid = x,
                (fnum, Val::Bytes(s)) if fnum == name_field => {
                    nm = std::str::from_utf8(s).ok();
                }
                _ => {}
            }
        }
        if let Some(nm) = nm {
            let id = intern(names, dedup, nm);
            match table {
                0 => {
                    seq.event_names.insert(iid, id);
                }
                1 => {
                    seq.specs.insert(iid, id);
                }
                3 => {
                    seq.categories.insert(iid, id);
                }
                _ => {
                    seq.kernel_names.insert(iid, id);
                }
            }
        }
    }
    Ok(())
}

fn decode_counter_descriptor(
    b: &[u8],
    names: &mut Vec<String>,
    dedup: &mut FxHashMap<Vec<u8>, u32>,
) -> Result<CounterDescriptor> {
    let mut descriptor = CounterDescriptor::default();
    let mut fs = Fields::new(b);
    while let Some((f, v)) = fs.next_field()? {
        let (GCD_SPEC, Val::Bytes(spec)) = (f, v) else {
            continue;
        };
        let mut counter_id = None;
        let mut name = None;
        let mut fields = Fields::new(spec);
        while let Some((field, value)) = fields.next_field()? {
            match (field, value) {
                (GCS_ID, Val::Varint(value)) => counter_id = Some(value),
                (GCS_NAME, Val::Bytes(value)) => name = intern_bytes(names, dedup, value),
                _ => {}
            }
        }
        if let (Some(counter_id), Some(name)) = (counter_id, name) {
            descriptor.names.insert(counter_id, name);
        }
    }
    Ok(descriptor)
}

fn decode_gpu_counter(
    b: &[u8],
    ts: u64,
    seq: &mut SeqState,
    tr: &mut PfTrace,
    dedup: &mut FxHashMap<Vec<u8>, u32>,
) -> Result<()> {
    let mut descriptor = None;
    let mut descriptor_iid = None;
    let mut gpu_id = None;
    let mut counters = Vec::new();
    let mut fs = Fields::new(b);
    while let Some((f, v)) = fs.next_field()? {
        match (f, v) {
            (GCE_DESCRIPTOR, Val::Bytes(value)) => descriptor = Some(value),
            (GCE_COUNTER, Val::Bytes(value)) => counters.push(value),
            (GCE_GPU_ID, Val::Varint(value)) => gpu_id = Some(value as i64),
            (GCE_DESCRIPTOR_IID, Val::Varint(value)) => descriptor_iid = Some(value),
            _ => {}
        }
    }
    if let Some(descriptor) = descriptor {
        seq.inline_counter_descriptor =
            decode_counter_descriptor(descriptor, &mut tr.names, dedup)?;
    }
    let descriptor = descriptor_iid
        .and_then(|iid| seq.counter_descriptors.get(&iid))
        .unwrap_or(&seq.inline_counter_descriptor);
    let gpu_id = descriptor.gpu_id.or(gpu_id).unwrap_or(-1);

    for counter in counters {
        let mut counter_id = None;
        let mut value = None;
        let mut fields = Fields::new(counter);
        while let Some((field, raw)) = fields.next_field()? {
            match (field, raw) {
                (GC_ID, Val::Varint(raw)) => counter_id = Some(raw),
                (GC_INT_VALUE, Val::Varint(raw)) => value = Some(raw as i64 as f64),
                (GC_DOUBLE_VALUE, Val::Fixed64(raw)) => value = Some(f64::from_bits(raw)),
                _ => {}
            }
        }
        let Some((counter_id, value)) = counter_id.zip(value) else {
            continue;
        };
        let name = descriptor
            .names
            .get(&counter_id)
            .copied()
            .unwrap_or_else(|| intern(&mut tr.names, dedup, &format!("GPU counter {counter_id}")));
        tr.events.push(PfEvent {
            cat: PfCat::GpuCounter,
            cat_name: NO_NAME,
            name,
            ts_ns: ts,
            dur_ns: 0,
            track: u32::MAX,
            stream: -1,
            logical_stream: -1,
            gpu_id,
            grid0: -1,
            gnid: 0,
            correlation: -1,
            pg_desc: NO_NAME,
            pg_name: NO_NAME,
            dtype: NO_NAME,
            collective_name: NO_NAME,
            in_nelems: -1,
            out_nelems: -1,
            group_size: -1,
            counter_value: Some(value),
        });
    }
    Ok(())
}

fn decode_track_descriptor(
    b: &[u8],
    tr: &mut PfTrace,
    dedup: &mut FxHashMap<Vec<u8>, u32>,
    track_threads: &mut FxHashMap<u64, u32>,
) -> Result<()> {
    let mut uuid: u64 = 0;
    let mut thread: Option<PfThread> = None;
    let mut process: Option<(i64, u32)> = None;
    let mut fs = Fields::new(b);
    while let Some((f, v)) = fs.next_field()? {
        match (f, v) {
            (TD_UUID, Val::Varint(x)) => uuid = x,
            (TD_THREAD, Val::Bytes(tb)) => {
                let (mut pid, mut tid, mut nm) = (0i64, 0i64, None);
                let mut ts_ = Fields::new(tb);
                while let Some((f2, v2)) = ts_.next_field()? {
                    match (f2, v2) {
                        (THD_PID, Val::Varint(x)) => pid = x as i64,
                        (THD_TID, Val::Varint(x)) => tid = x as i64,
                        (THD_NAME, Val::Bytes(s)) => {
                            nm = std::str::from_utf8(s)
                                .ok()
                                .map(|s| intern(&mut tr.names, dedup, s));
                        }
                        _ => {}
                    }
                }
                thread = Some(PfThread { pid, tid, name: nm });
            }
            (TD_PROCESS, Val::Bytes(pb)) => {
                let (mut pid, mut nm) = (0i64, NO_NAME);
                let mut ps = Fields::new(pb);
                while let Some((f2, v2)) = ps.next_field()? {
                    match (f2, v2) {
                        (PD_PID, Val::Varint(x)) => pid = x as i64,
                        (PD_NAME, Val::Bytes(s)) => {
                            if let Ok(s) = std::str::from_utf8(s) {
                                nm = intern(&mut tr.names, dedup, s);
                            }
                        }
                        _ => {}
                    }
                }
                process = Some((pid, nm));
            }
            _ => {}
        }
    }
    if let Some(t) = thread {
        track_threads.insert(uuid, tr.threads.len() as u32);
        tr.threads.push(t);
    }
    if let Some(p) = process {
        tr.process = Some(p);
    }
    Ok(())
}

fn decode_track_event(
    b: &[u8],
    ts: u64,
    seq: &mut SeqState,
    tr: &mut PfTrace,
    dedup: &mut FxHashMap<Vec<u8>, u32>,
    pending: &mut FxHashMap<u64, Vec<PendingTe>>,
    track_order: &mut Vec<u64>,
) -> Result<()> {
    let mut ty: u64 = 0;
    let mut name: u32 = NO_NAME;
    let mut cat_name: u32 = NO_NAME;
    let mut track_uuid: u64 = 0;
    let mut corr: i64 = -1;
    // nccl cpu_ops carry the PG metadata as debug annotations on the BEGIN.
    let (mut pg_desc, mut pg_name) = (NO_NAME, NO_NAME);
    let (mut dtype, mut coll) = (NO_NAME, NO_NAME);
    let (mut in_nelems, mut out_nelems, mut group_size) = (-1, -1, -1);
    let mut fs = Fields::new(b);
    while let Some((f, v)) = fs.next_field()? {
        match (f, v) {
            (TE_TYPE, Val::Varint(x)) => ty = x,
            (TE_CATEGORY_IIDS, Val::Varint(x)) => {
                // At most one category per slice in this exporter.
                cat_name = seq.categories.get(&x).copied().unwrap_or(NO_NAME);
            }
            (TE_NAME_IID, Val::Varint(x)) => {
                name = seq.event_names.get(&x).copied().unwrap_or(NO_NAME);
            }
            (TE_NAME, Val::Bytes(s)) => {
                // Not used by this exporter; kept for generality.
                if let Ok(s) = std::str::from_utf8(s) {
                    let id = tr.names.len() as u32;
                    tr.names.push(s.to_string());
                    name = id;
                }
            }
            (TE_TRACK_UUID, Val::Varint(x)) => track_uuid = x,
            (TE_EXT_CORRELATION, Val::Bytes(eb)) => {
                let mut es = Fields::new(eb);
                while let Some((f2, v2)) = es.next_field()? {
                    if let (1, Val::Varint(x)) = (f2, v2) {
                        corr = x as i64;
                    }
                }
            }
            (TE_DEBUG_ANN, Val::Bytes(ab)) => {
                // Names/values are matched and interned as raw bytes:
                // byte literals need no UTF-8 validation, and intern_bytes
                // validates once per unique string (returning NO_NAME for
                // invalid UTF-8, like the old from_utf8().ok() path).
                let mut nm: Option<&[u8]> = None;
                let mut ival: Option<i64> = None;
                let mut sval: Option<&[u8]> = None;
                let mut as_ = Fields::new(ab);
                while let Some((f2, v2)) = as_.next_field()? {
                    match (f2, v2) {
                        (DA_NAME, Val::Bytes(s)) => nm = Some(s),
                        (DA_INT, Val::Varint(x)) => ival = Some(x as i64),
                        (DA_STRING, Val::Bytes(s)) => sval = Some(s),
                        _ => {}
                    }
                }
                if let Some(nm) = nm {
                    let mut istr =
                        |v: &[u8]| intern_bytes(&mut tr.names, dedup, v).unwrap_or(NO_NAME);
                    match (nm, sval) {
                        (b"correlation", _) => {
                            if let Some(x) = ival {
                                corr = x;
                            }
                        }
                        // The four converter columns, exactly as the JSON
                        // twin's args carry them on nccl cpu_ops.
                        (b"Process Group Description", Some(v)) => pg_desc = istr(v),
                        (b"Process Group Name", Some(v)) => pg_name = istr(v),
                        (b"dtype", Some(v)) => dtype = istr(v),
                        (b"Collective name", Some(v)) => coll = istr(v),
                        _ => {}
                    }
                    let numeric = || ival.or_else(|| sval.and_then(parse_i64)).unwrap_or(-1);
                    match nm {
                        b"In msg nelems" => in_nelems = numeric(),
                        b"Out msg nelems" => out_nelems = numeric(),
                        b"Group size" => group_size = numeric(),
                        _ => {}
                    }
                    // Two-probe count: no per-annotation String alloc.
                    if let Some(c) = tr.cpu_ann_counts.get_mut(nm) {
                        *c += 1;
                    } else {
                        tr.cpu_ann_counts.insert(nm.to_vec(), 1);
                    }
                }
            }
            _ => {}
        }
    }

    if ty == TE_TYPE_BEGIN || ty == TE_TYPE_END {
        let slot = pending.entry(track_uuid).or_insert_with(|| {
            track_order.push(track_uuid);
            Vec::new()
        });
        slot.push(PendingTe {
            ts,
            is_end: ty == TE_TYPE_END,
            name,
            cat_name,
            correlation: corr,
            pg_desc,
            pg_name,
            dtype,
            collective_name: coll,
            in_nelems,
            out_nelems,
            group_size,
        });
    }
    Ok(())
}

/// JSON-twin combined graph-node id. The original exporter emitted the
/// combined `(graph_id << 32) | node` value under "graph node id"; the fixed
/// exporter splits it into "graph id" + "graph node id" (raw node). Values
/// already ≥ 2^32 are combined and pass through; split values are recombined.
/// Discovered per event from the file's own extras — no job assumptions.
fn combine_gnid(gid: i64, gnid: i64) -> i64 {
    if gnid > 0 && gnid < (1i64 << 32) && gid > 0 {
        (gid << 32) | gnid
    } else {
        gnid
    }
}

fn decode_render_stage(
    b: &[u8],
    ts: u64,
    seq: &mut SeqState,
    tr: &mut PfTrace,
    dedup: &mut FxHashMap<Vec<u8>, u32>,
) -> Result<()> {
    let mut dur: u64 = 0;
    let mut event_id: i64 = -1;
    let mut hwq_iid: u64 = 0;
    let mut stage_iid: u64 = 0;
    let mut kname: u32 = NO_NAME;
    let mut ename: u32 = NO_NAME;
    let mut grid0: i64 = -1;
    let mut stream: i64 = -1;
    let mut logical_stream: i64 = -1;
    let mut gpu_id: i64 = -1;
    let mut gid: i64 = 0;
    let mut gnid: i64 = 0;
    let mut corr: i64 = -1;
    let (mut pg_desc, mut pg_name) = (NO_NAME, NO_NAME);
    let (mut dtype, mut coll) = (NO_NAME, NO_NAME);
    let (mut in_nelems, mut out_nelems, mut group_size) = (-1, -1, -1);

    let mut comm_values: [Option<&[u8]>; 9] = [None; 9];
    let mut fs = Fields::new(b);
    while let Some((f, v)) = fs.next_field()? {
        match (f, v) {
            (RS_EVENT_ID, Val::Varint(x)) => event_id = x as i64,
            (RS_DURATION, Val::Varint(x)) => dur = x,
            (RS_GPU_ID, Val::Varint(x)) => gpu_id = x as i64,
            (RS_HW_QUEUE_IID, Val::Varint(x)) => hwq_iid = x,
            (RS_STAGE_IID, Val::Varint(x)) => stage_iid = x,
            (RS_EXT_KERNEL_NAME_IID, Val::Varint(x)) => {
                kname = seq.kernel_names.get(&x).copied().unwrap_or(NO_NAME);
            }
            (RS_EXT_EVENT_NAME_IID, Val::Varint(x)) => {
                ename = seq.event_names.get(&x).copied().unwrap_or(NO_NAME);
            }
            (RS_EXT_LAUNCH, Val::Bytes(lb)) => {
                let mut ls = Fields::new(lb);
                while let Some((f2, v2)) = ls.next_field()? {
                    if let (1, Val::Bytes(gb)) = (f2, v2) {
                        // grid dims {x=1, y=2, z=3}: FIRST-element semantics
                        // of the JSON converter's `grid` column.
                        let mut gs = Fields::new(gb);
                        while let Some((f3, v3)) = gs.next_field()? {
                            if let (1, Val::Varint(x)) = (f3, v3) {
                                grid0 = x as i64;
                            }
                        }
                    }
                }
            }
            (RS_EXTRA_DATA, Val::Bytes(eb)) => {
                // Byte-literal name match + bytes-prefix stoll: this runs
                // per extra_data entry per GPU event, and the names are a
                // small fixed set — UTF-8 validation buys nothing here
                // (stoll only ever consumes an ASCII prefix).
                let mut nm: Option<&[u8]> = None;
                let mut val: Option<&[u8]> = None;
                let mut es = Fields::new(eb);
                while let Some((f2, v2)) = es.next_field()? {
                    match (f2, v2) {
                        (1, Val::Bytes(s)) => nm = Some(s),
                        (2, Val::Bytes(s)) => val = Some(s),
                        _ => {}
                    }
                }
                let mut istr = |v: Option<&[u8]>| {
                    v.and_then(|v| intern_bytes(&mut tr.names, dedup, v))
                        .unwrap_or(NO_NAME)
                };
                if let (Some(key), Some(value)) = (nm, val) {
                    if let Some(idx) = std::str::from_utf8(key)
                        .ok()
                        .and_then(crate::comm::CommMetadata::key_index)
                    {
                        comm_values[idx] = Some(value);
                    }
                }
                match nm {
                    Some(b"stream id") => stream = val.and_then(parse_i64).unwrap_or(-1),
                    Some(b"stream") => logical_stream = val.and_then(parse_i64).unwrap_or(-1),
                    Some(b"device id") => gpu_id = val.and_then(parse_i64).unwrap_or(-1),
                    Some(b"graph id") => gid = val.and_then(parse_i64).unwrap_or(0),
                    Some(b"graph node id") => gnid = val.and_then(parse_i64).unwrap_or(0),
                    Some(b"correlation") => corr = val.and_then(parse_i64).unwrap_or(-1),
                    // Inline comms metadata (exporter generations with the
                    // GPU-collective fix) — the converter columns.
                    Some(b"Process Group Description") => pg_desc = istr(val),
                    Some(b"Process Group Name") => pg_name = istr(val),
                    Some(b"dtype") => dtype = istr(val),
                    Some(b"Collective name") => coll = istr(val),
                    Some(b"In msg nelems") => in_nelems = val.and_then(parse_i64).unwrap_or(-1),
                    Some(b"Out msg nelems") => out_nelems = val.and_then(parse_i64).unwrap_or(-1),
                    Some(b"Group size") => group_size = val.and_then(parse_i64).unwrap_or(-1),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    // Stage name → category (stage names are exporter constants; resolved
    // from the file's own interned table, not assumed by id).
    let stage_name_id = seq.specs.get(&stage_iid).copied().unwrap_or(NO_NAME);
    let stage_name = tr.name(stage_name_id);
    let cat = match stage_name {
        "Kernel" => PfCat::Kernel,
        "Memcpy" => PfCat::Memcpy,
        "Memset" => PfCat::Memset,
        "Annotation" => PfCat::GpuAnnotation,
        _ => PfCat::GpuOther,
    };
    // JSON-name equivalence: kernels use the CUPTI kernel name; annotations
    // use the event name; memcpy/memset rows are named by their stage
    // ("Memcpy"/"Memset" — matches the JSON twin's names for this exporter).
    let name = match cat {
        PfCat::Kernel => {
            if kname != NO_NAME {
                kname
            } else {
                ename
            }
        }
        PfCat::GpuAnnotation => ename,
        PfCat::Memcpy | PfCat::Memset | PfCat::GpuOther => stage_name_id,
        PfCat::CpuSlice | PfCat::GpuCounter => unreachable!(),
    };

    // hw queue registry (discovery order).
    let queue = match seq.queues.get(&hwq_iid) {
        Some(&q) => q,
        None => {
            let name_id = seq
                .specs
                .get(&hwq_iid)
                .copied()
                .unwrap_or_else(|| intern(&mut tr.names, dedup, &format!("queue {hwq_iid}")));
            let q = tr.hw_queues.len() as u32;
            tr.hw_queues.push((name_id, hwq_iid));
            seq.queues.insert(hwq_iid, q);
            q
        }
    };

    let annotated = coll != NO_NAME
        && pg_name != NO_NAME
        && crate::kernel_types::is_annotated_collective(tr.name(coll), tr.name(pg_name));
    if crate::comm::is_comm(tr.name(name)) || annotated {
        let mut comm = crate::comm::CommMetadata::default();
        for (dst, value) in comm.0.iter_mut().zip(comm_values) {
            *dst = value
                .and_then(|v| std::str::from_utf8(v).ok())
                .map(str::to_owned);
        }
        // Use the same combined node representation as the JSON twin.
        if gnid != 0 || gid != 0 {
            comm.set(
                "graph node id",
                (combine_gnid(gid, gnid) as u64).to_string(),
            );
        }
        tr.comm.insert(tr.events.len(), comm);
    }
    tr.events.push(PfEvent {
        cat,
        cat_name: NO_NAME,
        name,
        ts_ns: ts,
        dur_ns: dur,
        track: queue,
        stream,
        logical_stream,
        gpu_id,
        grid0,
        gnid: combine_gnid(gid, gnid),
        correlation: if event_id != -1 { event_id } else { corr },
        pg_desc,
        pg_name,
        dtype,
        collective_name: coll,
        in_nelems,
        out_nelems,
        group_size,
        counter_value: None,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Format contract (see `crate::trace`).
// ---------------------------------------------------------------------------

/// A fixed-generation perfetto trace carries its CUDA-graph annotations
/// inline: the comms metadata rides the GPU collective stages as string
/// extra_data, and the lanes are already resolver-assigned. There is no
/// side channel to consult and the walk must not reassign.
///
/// The ORIGINAL exporter generation
/// deliberately omitted the PG values, expecting the consumer to run the
/// annotation resolver — those traces analyze with empty PG columns under
/// this flag rather than joining pickles. That trade is intentional: no
/// production pftrace+pkl job exists, the fixed generation is what ships,
/// and joining pickles onto a format that now carries its own values would
/// re-create exactly the intertwining this seam removed.
pub const SELF_ANNOTATED: bool = true;
// --- Wire details verified against the fixed exporter -----------------------
//
// Read from PyTorch's Cuspy encoder (`torch/csrc/profiler/cuspy/cuspy_pftrace.cpp`
// and `torch/profiler/_cuspy/trace.py`, formerly `monitor_pftrace.cpp` and
// `monitor_trace.py`), with
// field numbers cross-checked against the vendored perfetto v56.1
// amalgamation.
// NOT yet decoded below — these are the constants the row bridge needs.
//
// Categories (the gap that used to block ingestion — now emitted):
//   TrackEvent.category_iids = field 3, repeated NON-PACKED uint64 (wire
//     type 0, one tag+varint each); at most one per slice in practice; absent
//     means "no category".
//   InternedData.event_categories = field 1, EventCategory{iid=1, name=2};
//     iid == index in the exporter's category table + 1.
//   Strings are the verbatim Kineto `cat` for X-phase CPU slices (`cpu_op`,
//     `user_annotation`, `python_function`, …) plus literal `cuda_runtime` /
//     `cuda_driver`. GPU work never appears here — it is render stages,
//     identified by stage_iid → `Kernel`/`Memcpy`/`Memset`/`Annotation`.
//   Residual: `overhead` and `cuda_sync` slices are emitted with NO category
//     (the exporter passes cats=None), so they would arrive cat-less and be
//     dropped by `analysis::prepare`, unlike the JSON path where they are
//     present. Recoverable from the `cuda_sync_kind` annotation and the
//     Overhead track respectively, if it ever matters.
//
// Time base (also now emitted):
//   TracePacket.clock_snapshot = field 6, the FIRST packet in the stream.
//   ClockSnapshot.clocks = field 1 (repeated Clock); Clock.clock_id = 1
//     (uint32), Clock.timestamp = 2 (uint64). BOOTTIME=6 and REALTIME=1 are
//     both present and both equal `base_ns`, so the snapshot is an identity
//     mapping whose only content is `base_ns` itself — which is also that
//     packet's own TracePacket.timestamp (field 8).
//   Reconstruction: `json_ts_us == (packet.timestamp - base_ns) / 1000.0`.
//     No timestamp_clock_id and no trace_packet_defaults are ever emitted.
//
// GpuRenderStageEvent.context = field 5 (uint64) is not read. gpu_id
// (field 11) and the legacy "device id" extra both identify the device.
//
// Changed shape (would break a name-parsing reader):
//   Hardware-queue names are now `"[<zero-padded lane id>] <label>"`, e.g.
//   `"[0026674] stream 26674"` or `"[0000007] DP"`, where the pad width is
//   per-trace. The bracketed id is the LOGICAL lane; the physical CUDA stream
//   stays in the `"stream id"` extra_data (they differ for graphed ops).
//
// Still absent by design:
//   ac2g flow events. TrackEvent.flow_ids (47) / terminating_flow_ids (48)
//   are never emitted — `add_group(..., flow=None)` is the only call shape.
//   CPU→GPU linkage stays ext 3000 → GpuCorrelation
//   .render_stage_submission_event_ids (field 1) joined to
//   GpuRenderStageEvent.event_id (field 1), and is 1:N for `cudaGraphLaunch`
//   (one correlation shared across every replayed kernel of a graph).
//   If flows are ever added, note flow_ids is FIXED64, not varint.

/// Parse a decompressed pftrace into converter rows (see
/// `rows::convert_pftrace_rows` for the value-mapping contract).
pub fn parse_rows(decompressed: &mut [u8]) -> Result<crate::rows::ParsedRows> {
    let t = read_trace(decompressed)?;
    crate::rows::convert_pftrace_rows(&t)
}

/// Logical lane encoded by the exporter in a hardware-queue name.
pub(crate) fn logical_lane(name: &str) -> Option<i64> {
    if let Some(rest) = name.strip_prefix('[') {
        let end = rest.find(']')?;
        rest[..end].parse().ok()
    } else {
        name.strip_prefix("stream ")?.trim().parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal wire-format writer for the test fixture.
    fn vint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
    }
    fn field_varint(out: &mut Vec<u8>, f: u64, v: u64) {
        vint(out, f << 3);
        vint(out, v);
    }
    fn field_bytes(out: &mut Vec<u8>, f: u64, b: &[u8]) {
        vint(out, (f << 3) | 2);
        vint(out, b.len() as u64);
        out.extend_from_slice(b);
    }
    fn field_fixed64(out: &mut Vec<u8>, f: u64, value: u64) {
        vint(out, (f << 3) | 1);
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn field_str(out: &mut Vec<u8>, f: u64, s: &str) {
        field_bytes(out, f, s.as_bytes());
    }
    fn packet(trace: &mut Vec<u8>, body: &[u8]) {
        field_bytes(trace, 1, body);
    }

    fn build_fixture() -> Vec<u8> {
        let mut trace = Vec::new();

        // Track descriptors: process + one thread track (uuid 2).
        let mut td = Vec::new();
        field_varint(&mut td, TD_UUID, 1);
        let mut proc_ = Vec::new();
        field_varint(&mut proc_, PD_PID, 42);
        field_str(&mut proc_, PD_NAME, "python3");
        field_bytes(&mut td, TD_PROCESS, &proc_);
        let mut pkt = Vec::new();
        field_bytes(&mut pkt, TP_TRACK_DESCRIPTOR, &td);
        packet(&mut trace, &pkt);

        let mut td = Vec::new();
        field_varint(&mut td, TD_UUID, 2);
        let mut th = Vec::new();
        field_varint(&mut th, THD_PID, 42);
        field_varint(&mut th, THD_TID, 43);
        field_str(&mut th, THD_NAME, "t43");
        field_bytes(&mut td, TD_THREAD, &th);
        let mut pkt = Vec::new();
        field_bytes(&mut pkt, TP_TRACK_DESCRIPTOR, &td);
        packet(&mut trace, &pkt);

        // Interned data on sequence 1 (state-cleared first packet).
        let mut interned = Vec::new();
        let mut en = Vec::new();
        field_varint(&mut en, 1, 7);
        field_str(&mut en, 2, "aten::empty");
        field_bytes(&mut interned, ID_EVENT_NAMES, &en);
        for (iid, name) in [(1u64, "Kernel"), (100u64, "stream 7")] {
            let mut sp = Vec::new();
            field_varint(&mut sp, 1, iid);
            field_str(&mut sp, 2, name);
            field_bytes(&mut interned, ID_GPU_SPECS, &sp);
        }
        let mut kn = Vec::new();
        field_varint(&mut kn, 1, 9);
        field_str(&mut kn, 3, "myKernel(int)");
        field_bytes(&mut interned, ID_EXT_KERNEL_NAMES, &kn);
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_varint(&mut pkt, TP_SEQ_FLAGS, SEQ_INCREMENTAL_STATE_CLEARED);
        field_bytes(&mut pkt, TP_INTERNED, &interned);
        packet(&mut trace, &pkt);

        // CPU slice: BEGIN(ts=1000) / END(ts=1500) on track 2.
        let mut te = Vec::new();
        field_varint(&mut te, TE_TYPE, TE_TYPE_BEGIN);
        field_varint(&mut te, TE_NAME_IID, 7);
        field_varint(&mut te, TE_TRACK_UUID, 2);
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_TIMESTAMP, 1000);
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_bytes(&mut pkt, TP_TRACK_EVENT, &te);
        packet(&mut trace, &pkt);

        let mut te = Vec::new();
        field_varint(&mut te, TE_TYPE, TE_TYPE_END);
        field_varint(&mut te, TE_TRACK_UUID, 2);
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_TIMESTAMP, 1500);
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_bytes(&mut pkt, TP_TRACK_EVENT, &te);
        packet(&mut trace, &pkt);

        // GPU kernel on queue "stream 7", gnid = 5<<32, grid.x = 32.
        let mut rs = Vec::new();
        field_varint(&mut rs, RS_EVENT_ID, 77);
        field_varint(&mut rs, RS_DURATION, 250);
        field_varint(&mut rs, RS_HW_QUEUE_IID, 100);
        field_varint(&mut rs, RS_STAGE_IID, 1);
        field_varint(&mut rs, RS_EXT_KERNEL_NAME_IID, 9);
        let mut launch = Vec::new();
        let mut grid = Vec::new();
        field_varint(&mut grid, 1, 32);
        field_varint(&mut grid, 2, 1);
        field_varint(&mut grid, 3, 1);
        field_bytes(&mut launch, 1, &grid);
        field_bytes(&mut rs, RS_EXT_LAUNCH, &launch);
        for (k, v) in [("stream id", "7"), ("graph node id", "21474836481")] {
            let mut ex = Vec::new();
            field_str(&mut ex, 1, k);
            field_str(&mut ex, 2, v);
            field_bytes(&mut rs, RS_EXTRA_DATA, &ex);
        }
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_TIMESTAMP, 2000);
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_bytes(&mut pkt, TP_GPU_RENDER_STAGE, &rs);
        packet(&mut trace, &pkt);

        trace
    }

    #[test]
    fn decode_fixture() {
        let tr = read_trace(&build_fixture()).unwrap();
        assert_eq!(tr.n_packets, 6);
        assert_eq!(tr.events.len(), 2);
        assert_eq!(tr.unmatched_begin, 0);
        assert_eq!(tr.unmatched_end, 0);

        // Slice pairing happens at EOF, so CPU slices follow GPU stages in
        // the event list (a deterministic order; parse order is already
        // cross-format-divergent by construction).
        let cpu = &tr.events[1];
        assert_eq!(cpu.cat, PfCat::CpuSlice);
        assert_eq!(tr.name(cpu.name), "aten::empty");
        assert_eq!((cpu.ts_ns, cpu.dur_ns), (1000, 500));
        assert_eq!(cpu.cat_name, NO_NAME); // fixture predates categories
        let th = &tr.threads[cpu.track as usize];
        assert_eq!((th.pid, th.tid), (42, 43));
        assert_eq!(tr.name(th.name.unwrap()), "t43");
        assert_eq!(tr.base_ns, None); // no ClockSnapshot in the old fixture

        let k = &tr.events[0];
        assert_eq!(k.cat, PfCat::Kernel);
        assert_eq!(tr.name(k.name), "myKernel(int)");
        assert_eq!((k.ts_ns, k.dur_ns), (2000, 250));
        assert_eq!(k.stream, 7);
        assert_eq!(k.grid0, 32);
        assert_eq!(k.gnid, (5i64 << 32) + 1);
        assert_eq!(k.correlation, 77);
        let (qname, qiid) = tr.hw_queues[k.track as usize];
        assert_eq!(tr.name(qname), "stream 7");
        assert_eq!(qiid, 100);

        let gids: Vec<i64> = tr.live_gids().into_iter().collect();
        assert_eq!(gids, vec![5]);

        assert_eq!(tr.process.as_ref().map(|(p, _)| *p), Some(42));
    }

    #[test]
    fn decodes_interned_gpu_counters() {
        let mut trace = Vec::new();

        let mut spec = Vec::new();
        field_varint(&mut spec, GCS_ID, 7);
        field_str(&mut spec, GCS_NAME, "SM Active (%)");
        let mut descriptor = Vec::new();
        field_bytes(&mut descriptor, GCD_SPEC, &spec);
        let mut interned_descriptor = Vec::new();
        field_varint(&mut interned_descriptor, IGCD_IID, 11);
        field_bytes(&mut interned_descriptor, IGCD_DESCRIPTOR, &descriptor);
        field_varint(&mut interned_descriptor, IGCD_GPU_ID, 3);
        let mut interned = Vec::new();
        field_bytes(
            &mut interned,
            ID_GPU_COUNTER_DESCRIPTORS,
            &interned_descriptor,
        );
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_varint(&mut pkt, TP_SEQ_FLAGS, SEQ_INCREMENTAL_STATE_CLEARED);
        field_bytes(&mut pkt, TP_INTERNED, &interned);
        packet(&mut trace, &pkt);

        let mut counter = Vec::new();
        field_varint(&mut counter, GC_ID, 7);
        field_fixed64(&mut counter, GC_DOUBLE_VALUE, 42.5f64.to_bits());
        let mut event = Vec::new();
        field_varint(&mut event, GCE_DESCRIPTOR_IID, 11);
        field_bytes(&mut event, GCE_COUNTER, &counter);
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_TIMESTAMP, 12_000);
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_bytes(&mut pkt, TP_GPU_COUNTER, &event);
        packet(&mut trace, &pkt);

        let parsed = read_trace(&trace).expect("decode counter fixture");
        assert_eq!(parsed.n_counter_packets, 1);
        assert_eq!(parsed.events.len(), 1);
        let sample = &parsed.events[0];
        assert_eq!(sample.cat, PfCat::GpuCounter);
        assert_eq!(parsed.name(sample.name), "SM Active (%)");
        assert_eq!(sample.gpu_id, 3);
        assert_eq!(sample.ts_ns, 12_000);
        assert_eq!(sample.counter_value, Some(42.5));

        let rows = crate::rows::convert_pftrace_rows(&parsed).expect("convert counter row");
        assert_eq!(rows.rows.counter_value, vec![Some(42.5)]);
        assert!(rows.rows.ph[0].is_some());
    }

    /// The fixed-exporter generation: ClockSnapshot time base, CPU
    /// categories, SPLIT graph id / node id extras, inline PG metadata,
    /// and the batched all-BEGINs-then-all-ENDs slice emission.
    fn build_fixed_gen_fixture() -> Vec<u8> {
        let mut trace = Vec::new();

        // ClockSnapshot first packet: BOOTTIME + REALTIME both at base.
        let base: u64 = 1_000_000_000_000;
        let mut cs = Vec::new();
        for clock_id in [6u64, 1] {
            let mut clk = Vec::new();
            field_varint(&mut clk, 1, clock_id);
            field_varint(&mut clk, 2, base);
            field_bytes(&mut cs, 1, &clk);
        }
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_TIMESTAMP, base);
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_bytes(&mut pkt, TP_CLOCK_SNAPSHOT, &cs);
        packet(&mut trace, &pkt);

        // Thread track.
        let mut td = Vec::new();
        field_varint(&mut td, TD_UUID, 2);
        let mut th = Vec::new();
        field_varint(&mut th, THD_PID, 42);
        field_varint(&mut th, THD_TID, 43);
        field_bytes(&mut td, TD_THREAD, &th);
        let mut pkt = Vec::new();
        field_bytes(&mut pkt, TP_TRACK_DESCRIPTOR, &td);
        packet(&mut trace, &pkt);

        // Interned: names, categories, bracketed hw queue, stage.
        let mut interned = Vec::new();
        for (iid, name) in [(7u64, "cudaLaunchKernel"), (8u64, "aten::mm")] {
            let mut en = Vec::new();
            field_varint(&mut en, 1, iid);
            field_str(&mut en, 2, name);
            field_bytes(&mut interned, ID_EVENT_NAMES, &en);
        }
        for (iid, name) in [(1u64, "cuda_runtime"), (2u64, "cpu_op")] {
            let mut ec = Vec::new();
            field_varint(&mut ec, 1, iid);
            field_str(&mut ec, 2, name);
            field_bytes(&mut interned, ID_EVENT_CATEGORIES, &ec);
        }
        for (iid, name) in [(1u64, "Kernel"), (100u64, "[0061] HSDP_SHARD")] {
            let mut sp = Vec::new();
            field_varint(&mut sp, 1, iid);
            field_str(&mut sp, 2, name);
            field_bytes(&mut interned, ID_GPU_SPECS, &sp);
        }
        let mut kn = Vec::new();
        field_varint(&mut kn, 1, 9);
        field_str(&mut kn, 3, "ncclDevKernel_AllGather(void*)");
        field_bytes(&mut interned, ID_EXT_KERNEL_NAMES, &kn);
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_varint(&mut pkt, TP_SEQ_FLAGS, SEQ_INCREMENTAL_STATE_CLEARED);
        field_bytes(&mut pkt, TP_INTERNED, &interned);
        packet(&mut trace, &pkt);

        // Two nested CPU slices emitted BATCHED (both BEGINs, then both
        // ENDs) and OUT of timestamp order across the batch boundary —
        // the fixed exporter's group emission. Slices:
        //   aten::mm        [base+1000, base+5000)  cat cpu_op
        //   cudaLaunchKernel[base+2000, base+3000)  cat cuda_runtime (nested)
        let emit_te = |trace: &mut Vec<u8>, ts: u64, ty: u64, name_iid: u64, cat_iid: u64| {
            let mut te = Vec::new();
            field_varint(&mut te, TE_TYPE, ty);
            if ty == TE_TYPE_BEGIN {
                field_varint(&mut te, TE_CATEGORY_IIDS, cat_iid);
                field_varint(&mut te, TE_NAME_IID, name_iid);
            }
            field_varint(&mut te, TE_TRACK_UUID, 2);
            let mut pkt = Vec::new();
            field_varint(&mut pkt, TP_TIMESTAMP, ts);
            field_varint(&mut pkt, TP_SEQ_ID, 1);
            field_bytes(&mut pkt, TP_TRACK_EVENT, &te);
            packet(trace, &pkt);
        };
        // cpu_op group: B, E
        emit_te(&mut trace, base + 1000, TE_TYPE_BEGIN, 8, 2);
        emit_te(&mut trace, base + 5000, TE_TYPE_END, 0, 0);
        // runtime group afterwards, nested INSIDE the first slice: B, E
        emit_te(&mut trace, base + 2000, TE_TYPE_BEGIN, 7, 1);
        emit_te(&mut trace, base + 3000, TE_TYPE_END, 0, 0);

        // Annotated collective kernel: split graph id/node id + PG extras +
        // logical stream.
        let mut rs = Vec::new();
        field_varint(&mut rs, RS_EVENT_ID, 501);
        field_varint(&mut rs, RS_DURATION, 4_500);
        field_varint(&mut rs, RS_HW_QUEUE_IID, 100);
        field_varint(&mut rs, RS_STAGE_IID, 1);
        field_varint(&mut rs, RS_EXT_KERNEL_NAME_IID, 9);
        for (k, v) in [
            ("device id", "3"),
            ("stream id", "7"),
            ("stream", "61"),
            ("graph id", "5"),
            ("graph node id", "12"),
            ("Process Group Description", "HSDP_SHARD"),
            ("Process Group Name", "abc123"),
            ("dtype", "bfloat16"),
            ("In msg nelems", "1024"),
            ("Out msg nelems", "512"),
            ("Group size", "4"),
        ] {
            let mut ex = Vec::new();
            field_str(&mut ex, 1, k);
            field_str(&mut ex, 2, v);
            field_bytes(&mut rs, RS_EXTRA_DATA, &ex);
        }
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_TIMESTAMP, base + 10_000);
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_bytes(&mut pkt, TP_GPU_RENDER_STAGE, &rs);
        packet(&mut trace, &pkt);

        trace
    }

    #[test]
    fn decode_fixed_gen() {
        let tr = read_trace(&build_fixed_gen_fixture()).unwrap();
        assert_eq!(tr.base_ns, Some(1_000_000_000_000));
        assert_eq!(tr.events.len(), 3);
        assert_eq!((tr.unmatched_begin, tr.unmatched_end), (0, 0));

        // GPU stage first (slice pairing happens at EOF).
        let k = &tr.events[0];
        assert_eq!(k.cat, PfCat::Kernel);
        assert_eq!(tr.name(k.name), "ncclDevKernel_AllGather(void*)");
        // Split gnid recombined into the JSON twin's encoding.
        assert_eq!(k.gnid, (5i64 << 32) | 12);
        assert_eq!((k.stream, k.logical_stream, k.gpu_id), (7, 61, 3));
        assert_eq!(tr.name(k.pg_desc), "HSDP_SHARD");
        assert_eq!(tr.name(k.pg_name), "abc123");
        assert_eq!(tr.name(k.dtype), "bfloat16");
        assert_eq!(k.collective_name, NO_NAME);
        assert_eq!((k.in_nelems, k.out_nelems, k.group_size), (1024, 512, 4));
        assert_eq!(tr.live_gids().into_iter().collect::<Vec<_>>(), vec![5]);

        // Batched slices paired by timestamp order, not packet order: the
        // nested runtime slice must NOT swallow the outer op's span.
        let by_name: Vec<(&str, &str, u64, u64)> = tr.events[1..]
            .iter()
            .map(|e| {
                (
                    tr.name(e.name),
                    tr.name(e.cat_name),
                    e.ts_ns - tr.base_ns.unwrap(),
                    e.dur_ns,
                )
            })
            .collect();
        assert!(by_name.contains(&("cudaLaunchKernel", "cuda_runtime", 2000, 1000)));
        assert!(by_name.contains(&("aten::mm", "cpu_op", 1000, 4000)));
    }

    #[test]
    fn base_ns_head_scan() {
        let bytes = build_fixed_gen_fixture();
        // Matches the full decode, and survives a head cut mid-stream (the
        // snapshot is the first packet).
        assert_eq!(read_base_ns(&bytes), read_trace(&bytes).unwrap().base_ns);
        assert_eq!(read_base_ns(&bytes), Some(1_000_000_000_000));
        assert_eq!(
            read_base_ns(&bytes[..bytes.len() / 2]),
            Some(1_000_000_000_000)
        );
        // Pre-fix generation: no snapshot anywhere → no base.
        assert_eq!(read_base_ns(&build_fixture()), None);
        assert_eq!(read_base_ns(&[]), None);
    }

    #[test]
    fn base_ns_from_file_head() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let bytes = build_fixed_gen_fixture();

        let raw = dir.path().join("000001.merged.pftrace");
        std::fs::write(&raw, &bytes).unwrap();
        assert_eq!(read_base_ns_path(&raw), Some(1_000_000_000_000));

        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&bytes).unwrap();
        let gz = dir.path().join("000002.merged.pftrace.gz");
        std::fs::write(&gz, enc.finish().unwrap()).unwrap();
        assert_eq!(read_base_ns_path(&gz), Some(1_000_000_000_000));

        assert_eq!(read_base_ns_path(&dir.path().join("missing.pftrace")), None);
    }
}

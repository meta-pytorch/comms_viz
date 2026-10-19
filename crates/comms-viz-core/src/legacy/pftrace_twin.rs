// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Chrome-JSON → perfetto `.pftrace` writer: synthesize the trace the
//! fixed-generation pftrace exporter WOULD have written for a rank whose
//! production output was Kineto chrome-JSON (+ side-channel annotation
//! pickles).
//!
//! Purpose: scale-test the pftrace ingestion path on real jobs that predate
//! the pftrace write pipeline. The output is structurally faithful to the
//! fixed exporter (verified against its output, wire-dumped 2026-07-29):
//!
//!   - packet phases: ClockSnapshot → TrackDescriptors → one InternedData
//!     packet (incremental-state-cleared) → all CPU TrackEvents → all
//!     GpuRenderStageEvents, everything on one sequence;
//!   - CPU slices as BEGIN/END pairs batched per track (all BEGINs then all
//!     ENDs — the pairing trap the reader handles by ts-sorted stack);
//!   - interned categories / event names / gpu specs / kernel names, hw
//!     queues named `"[<zero-padded lane>] <label>"` with per-trace pad
//!     width, stages named `Kernel`/`Memcpy`/`Memset`/`Annotation`;
//!   - `graph id` / `graph node id` extras SPLIT (the JSON twin carries the
//!     combined `(gid<<32)|node` encoding);
//!   - PG/comms metadata inline: the pftrace format is SELF-ANNOTATED, so
//!     this writer bakes in exactly the `walk_annotate` pass (annotation kv
//!     as string extra_data, lanes reassigned to the annotation stream or
//!     `default_stream`). The reader-side walk then replays cleanup + move +
//!     fix, exactly as it does for a native pftrace — so analysis values
//!     match the JSON+pkl path by construction.
//!
//! Content the JSON twin cannot supply is omitted rather than invented
//! (GpuCounterEvents, CUPTI channel/roofline extras); content the pftrace
//! format does not represent is dropped and counted (ac2g flows, `ph:"M"`
//! metadata rows, instant events, string-pid pseudo-tracks). Memcpy/Memset
//! rows are named by their stage — the pftrace route's naming — so their
//! Kineto direction suffix (`"Memcpy DtoH (…)"`) lives only in the byte
//! stream they came from; kernel-type classification is prefix-based and
//! unaffected.
//!
//! Nothing here hardcodes a job: names, lanes, categories, pids, the time
//! base, and the annotation content are all discovered from the rank's own
//! trace + pickles.

use std::collections::HashMap;

use anyhow::Result;
use anyhow::anyhow;
use serde_json::Value;

use crate::annotations::Annotations;
use crate::cudagraph::coerce_to_i64;
use crate::cudagraph::python_str;

// TracePacket / TrackEvent / TrackDescriptor / InternedData /
// GpuRenderStageEvent field numbers — the same constants the reader
// (`crate::trace::pftrace`) decodes; see that module for provenance.
const TP_CLOCK_SNAPSHOT: u64 = 6;
const TP_TIMESTAMP: u64 = 8;
const TP_SEQ_ID: u64 = 10;
const TP_TRACK_EVENT: u64 = 11;
const TP_INTERNED: u64 = 12;
const TP_SEQ_FLAGS: u64 = 13;
const TP_GPU_RENDER_STAGE: u64 = 53;
const TP_TRACK_DESCRIPTOR: u64 = 60;

const SEQ_INCREMENTAL_STATE_CLEARED: u64 = 1;
const SEQ_NEEDS_INCREMENTAL_STATE: u64 = 2;

const TE_CATEGORY_IIDS: u64 = 3;
const TE_DEBUG_ANN: u64 = 4;
const TE_TYPE: u64 = 9;
const TE_NAME_IID: u64 = 10;
const TE_TRACK_UUID: u64 = 11;
const TE_TYPE_BEGIN: u64 = 1;
const TE_TYPE_END: u64 = 2;

const TD_UUID: u64 = 1;
const TD_PROCESS: u64 = 3;
const TD_THREAD: u64 = 4;
const PD_PID: u64 = 1;
const PD_NAME: u64 = 6;
const THD_PID: u64 = 1;
const THD_TID: u64 = 2;
const THD_NAME: u64 = 5;

const ID_EVENT_CATEGORIES: u64 = 1;
const ID_EVENT_NAMES: u64 = 2;
const ID_GPU_SPECS: u64 = 24;
const ID_EXT_KERNEL_NAMES: u64 = 1000;

const RS_EVENT_ID: u64 = 1;
const RS_DURATION: u64 = 2;
const RS_CONTEXT: u64 = 5;
const RS_EXTRA_DATA: u64 = 6;
const RS_GPU_ID: u64 = 11;
const RS_HW_QUEUE_IID: u64 = 13;
const RS_STAGE_IID: u64 = 14;
const RS_EXT_KERNEL_NAME_IID: u64 = 19;
const RS_EXT_EVENT_NAME_IID: u64 = 20;
const RS_EXT_LAUNCH: u64 = 21;

const DA_NAME: u64 = 10;
const DA_INT: u64 = 4;
const DA_STRING: u64 = 6;

/// What one conversion dropped, kept, and verified.
#[derive(Default, Debug, Clone)]
pub struct WriteStats {
    pub cpu_slices: usize,
    pub gpu_stages: usize,
    pub annotated_rows: usize,
    pub dropped_flows: usize,
    pub dropped_meta: usize,
    pub dropped_other: usize,
    /// Rows whose pid or tid is not an integer (`"Spans"` pseudo-tracks).
    pub dropped_string_track: usize,
    /// CPU slices whose interval crosses a sibling's (would mis-pair in the
    /// reader's ts-sorted stack reconstruction). Counted, not repaired.
    pub nest_violations: usize,
    /// µs literals that were integers (the twin contract says 3-decimal
    /// floats; the fix-pass int-preserving branch is unrepresentable here).
    pub int_ts_literals: usize,
    pub raw_bytes: usize,
}

// --- wire primitives -------------------------------------------------------

#[inline]
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

#[inline]
fn field_varint(out: &mut Vec<u8>, f: u64, v: u64) {
    vint(out, f << 3);
    vint(out, v);
}

/// Signed varint (two's complement, like proto int32/int64) — negative pids.
#[inline]
fn field_int(out: &mut Vec<u8>, f: u64, v: i64) {
    field_varint(out, f, v as u64);
}

#[inline]
fn field_bytes(out: &mut Vec<u8>, f: u64, b: &[u8]) {
    vint(out, (f << 3) | 2);
    vint(out, b.len() as u64);
    out.extend_from_slice(b);
}

#[inline]
fn field_str(out: &mut Vec<u8>, f: u64, s: &str) {
    field_bytes(out, f, s.as_bytes());
}

fn packet(trace: &mut Vec<u8>, body: &[u8]) {
    field_bytes(trace, 1, body);
}

// --- helpers over the parsed tree ------------------------------------------

fn ev_str<'a>(ev: &'a Value, key: &str) -> Option<&'a str> {
    ev.get(key).and_then(|v| v.as_str())
}

fn ev_i64(ev: &Value, key: &str) -> Option<i64> {
    ev.get(key).and_then(|v| v.as_i64())
}

fn ev_f64(ev: &Value, key: &str) -> Option<f64> {
    ev.get(key).and_then(|v| v.as_f64())
}

fn arg<'a>(ev: &'a Value, key: &str) -> Option<&'a Value> {
    ev.get("args").and_then(|a| a.get(key))
}

/// µs → ns against the trace's own base. Exact for the magnitudes a session
/// produces (delta < 2^53 ns; one correctly-rounded multiply stays within
/// ±0.25 ns of the true integer, so `round` recovers it).
#[inline]
fn ns_of(base: u64, us: f64) -> u64 {
    let d = (us * 1000.0).round() as i64;
    (base as i64).saturating_add(d).max(0) as u64
}

/// String interner assigning 1-based iids in first-insertion order.
#[derive(Default)]
struct Interner {
    ids: HashMap<String, u64>,
    order: Vec<String>,
    first_iid: u64,
}

impl Interner {
    fn with_first(first_iid: u64) -> Self {
        Interner {
            first_iid,
            ..Default::default()
        }
    }

    fn get(&mut self, s: &str) -> u64 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = self.first_iid + self.order.len() as u64;
        self.ids.insert(s.to_string(), id);
        self.order.push(s.to_string());
        id
    }

    fn lookup(&self, s: &str) -> Option<u64> {
        self.ids.get(s).copied()
    }
}

/// GPU stage kinds — format constants shared with the reader.
fn stage_of(cat: &str) -> Option<&'static str> {
    match cat {
        "kernel" => Some("Kernel"),
        "gpu_memcpy" => Some("Memcpy"),
        "gpu_memset" => Some("Memset"),
        "gpu_user_annotation" => Some("Annotation"),
        _ => None,
    }
}

/// CPU groups the exporter emits WITHOUT a category (cats=None): they arrive
/// cat-less and analysis drops them — a format property, not a job one.
fn catless_cpu_group(cat: Option<&str>) -> bool {
    matches!(cat, None | Some("overhead") | Some("cuda_sync"))
}

/// The GPU-row arg keys consumed by canonical fields/extras rather than
/// mirrored verbatim, plus the Kineto-JSON-only decorations the pftrace
/// exporter does not emit.
fn gpu_arg_handled(key: &str) -> bool {
    matches!(
        key,
        "device"
            | "context"
            | "stream"
            | "correlation"
            | "queued"
            | "bytes"
            | "graph id"
            | "graph node id"
            | "grid"
            | "block"
            | "External id"
            | "registers per thread"
            | "shared memory"
            | "blocks per SM"
            | "warps per SM"
            | "est. achieved occupancy %"
            | "memory bandwidth (GB/s)"
    )
}

/// Convert one rank's parsed chrome-JSON tree (+ its side-channel
/// annotations) into uncompressed pftrace bytes.
///
/// `annotations` and `default_stream` replicate `tree_walks::walk_annotate`'s
/// per-row effect (kv → args, lane → annotation stream or `default_stream`)
/// WITHOUT the cleanup pass: cleanup + move + fix run at read time for a
/// self-annotated format, so baking them in would double-apply them.
pub fn tree_to_pftrace(
    root: &Value,
    annotations: &Annotations,
    default_stream: i64,
) -> Result<(Vec<u8>, WriteStats)> {
    let mut st = WriteStats::default();
    let events = root
        .get("traceEvents")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("no traceEvents array"))?;
    let base: u64 = root
        .get("baseTimeNanoseconds")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // arch per device, from the trace's own deviceProperties.
    let mut arch_of: HashMap<i64, String> = HashMap::new();
    if let Some(devs) = root.get("deviceProperties").and_then(|v| v.as_array()) {
        for d in devs {
            if let (Some(id), Some(maj), Some(min)) = (
                ev_i64(d, "id"),
                ev_i64(d, "computeMajor"),
                ev_i64(d, "computeMinor"),
            ) {
                arch_of.insert(id, format!("sm_{maj}{min}"));
            }
        }
    }

    // ---- pass 1: metadata + classification --------------------------------
    let mut process_names: HashMap<i64, String> = HashMap::new(); // pid → name
    let mut thread_names: HashMap<(i64, i64), String> = HashMap::new(); // (pid,tid) → name
    // CPU slice indices per (pid,tid) track, first-seen track order.
    let mut cpu_tracks: Vec<(i64, i64)> = Vec::new();
    let mut cpu_by_track: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    // GPU rows: (event idx, lane, gnid, ann list index or none).
    struct GpuRow {
        idx: usize,
        lane: i64,
        gnid: i64,
        annotated: bool,
        final_stream: Option<i64>,
    }
    let mut gpu_rows: Vec<GpuRow> = Vec::new();

    for (i, ev) in events.iter().enumerate() {
        let ph = ev_str(ev, "ph").unwrap_or("");
        match ph {
            "M" => {
                st.dropped_meta += 1;
                let name = ev_str(ev, "name").unwrap_or("");
                let pid = ev_i64(ev, "pid");
                let tid = ev_i64(ev, "tid");
                let arg_name = arg(ev, "name").and_then(|v| v.as_str());
                if let (Some(pid), Some(nm)) = (pid, arg_name) {
                    match name {
                        "process_name" => {
                            process_names.entry(pid).or_insert_with(|| nm.to_string());
                        }
                        "thread_name" => {
                            if let Some(tid) = tid {
                                thread_names
                                    .entry((pid, tid))
                                    .or_insert_with(|| nm.to_string());
                            }
                        }
                        _ => {}
                    }
                }
                continue;
            }
            "s" | "f" => {
                st.dropped_flows += 1;
                continue;
            }
            "X" => {}
            _ => {
                st.dropped_other += 1;
                continue;
            }
        }
        let (Some(pid), Some(tid)) = (ev_i64(ev, "pid"), ev_i64(ev, "tid")) else {
            st.dropped_string_track += 1;
            continue;
        };
        if ev
            .get("ts")
            .map(|v| v.is_i64() || v.is_u64())
            .unwrap_or(false)
        {
            st.int_ts_literals += 1;
        }
        let cat = ev_str(ev, "cat");
        if let Some(_stage) = cat.and_then(stage_of) {
            // GPU row: replicate walk_annotate's lane assignment.
            let gnid = arg(ev, "graph node id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let (lane, annotated, final_stream) = if gnid != 0 {
                let mut stream_id: Option<i64> = None;
                let ann_list = annotations.get(&gnid);
                if let Some(list) = ann_list {
                    for ann in list {
                        if let Some(s) = ann.get("stream").and_then(coerce_to_i64) {
                            stream_id = Some(s);
                        }
                    }
                }
                let fs = stream_id.unwrap_or(default_stream);
                (fs, ann_list.is_some(), Some(fs))
            } else {
                (tid, false, None)
            };
            if annotated {
                st.annotated_rows += 1;
            }
            gpu_rows.push(GpuRow {
                idx: i,
                lane,
                gnid,
                annotated,
                final_stream,
            });
        } else {
            let key = (pid, tid);
            cpu_by_track
                .entry(key)
                .or_insert_with(|| {
                    cpu_tracks.push(key);
                    Vec::new()
                })
                .push(i);
        }
    }

    // The process descriptor: the pid owning the most CPU slices (the
    // trainer process; other pids — overhead's -1 — stay thread-only, like
    // the exporter's own CUPTI helper tracks).
    let main_pid: Option<i64> = {
        let mut per_pid: HashMap<i64, usize> = HashMap::new();
        for ((pid, _), v) in &cpu_by_track {
            *per_pid.entry(*pid).or_insert(0) += v.len();
        }
        per_pid
            .into_iter()
            .max_by_key(|&(pid, n)| (n, std::cmp::Reverse(pid)))
            .map(|(pid, _)| pid)
    };
    let main_pname = main_pid.and_then(|p| process_names.get(&p).cloned());

    // ---- lanes → hw queues -------------------------------------------------
    // Label preference per lane: the trace's own GPU thread_name → dominant
    // PG description among the lane's annotated rows → dominant annotation
    // name → "stream <lane>". Cosmetic (the reader parses only the bracketed
    // lane id), but keeps files looking like the exporter's.
    let mut lane_order: Vec<i64> = Vec::new();
    let mut lane_seen: HashMap<i64, usize> = HashMap::new();
    let mut lane_pgdesc: HashMap<i64, HashMap<String, usize>> = HashMap::new();
    let mut lane_annname: HashMap<i64, HashMap<String, usize>> = HashMap::new();
    for r in &gpu_rows {
        if let std::collections::hash_map::Entry::Vacant(e) = lane_seen.entry(r.lane) {
            e.insert(lane_order.len());
            lane_order.push(r.lane);
        }
        if r.annotated {
            if let Some(list) = annotations.get(&r.gnid) {
                for ann in list {
                    if let Some(d) = ann
                        .get("Process Group Description")
                        .and_then(|v| v.as_str())
                    {
                        *lane_pgdesc
                            .entry(r.lane)
                            .or_default()
                            .entry(d.to_string())
                            .or_insert(0) += 1;
                    }
                    if let Some(n) = ann.get("name").and_then(|v| v.as_str()) {
                        *lane_annname
                            .entry(r.lane)
                            .or_default()
                            .entry(n.to_string())
                            .or_insert(0) += 1;
                    }
                }
            }
        }
    }
    let gpu_pid: i64 = gpu_rows
        .first()
        .and_then(|r| ev_i64(&events[r.idx], "pid"))
        .unwrap_or(0);
    let pad = lane_order
        .iter()
        .map(|l| format!("{l}").len())
        .max()
        .unwrap_or(1);
    let top = |m: Option<&HashMap<String, usize>>| -> Option<String> {
        m.and_then(|m| {
            m.iter()
                .max_by_key(|(k, v)| (**v, std::cmp::Reverse(k.as_str())))
                .map(|(k, _)| k.clone())
        })
    };
    let lane_label = |lane: i64| -> String {
        if let Some(nm) = thread_names.get(&(gpu_pid, lane)) {
            return nm.trim().to_string();
        }
        if let Some(d) = top(lane_pgdesc.get(&lane)) {
            return d;
        }
        if let Some(n) = top(lane_annname.get(&lane)) {
            return n;
        }
        format!("stream {lane}")
    };

    // ---- interned tables ---------------------------------------------------
    let mut categories = Interner::with_first(1);
    let mut event_names = Interner::with_first(1);
    let mut kernel_names = Interner::with_first(1);
    let mut specs = Interner::with_first(1); // stages 1.., queues appended at 100+
    for &s in &["Kernel", "Memcpy", "Memset", "Annotation"] {
        specs.get(s);
    }
    let mut queue_iids: HashMap<i64, u64> = HashMap::new();
    let mut queue_names: Vec<(u64, String)> = Vec::new();
    for (qi, &lane) in lane_order.iter().enumerate() {
        let iid = 100 + qi as u64;
        queue_iids.insert(lane, iid);
        queue_names.push((iid, format!("[{lane:0pad$}] {}", lane_label(lane))));
    }

    // Pre-intern in emission order: CPU names/categories, then GPU names.
    for key in &cpu_tracks {
        for &i in &cpu_by_track[key] {
            let ev = &events[i];
            if let Some(nm) = ev_str(ev, "name") {
                event_names.get(nm);
            }
            let cat = ev_str(ev, "cat");
            if !catless_cpu_group(cat) {
                categories.get(cat.unwrap());
            }
        }
    }
    for r in &gpu_rows {
        let ev = &events[r.idx];
        let cat = ev_str(ev, "cat").unwrap_or("");
        match cat {
            "kernel" => {
                if let Some(nm) = ev_str(ev, "name") {
                    kernel_names.get(nm);
                }
            }
            "gpu_user_annotation" => {
                if let Some(nm) = ev_str(ev, "name") {
                    event_names.get(nm);
                }
            }
            _ => {}
        }
        // Annotation event names ride the shared event-name table (the
        // resolver's names, e.g. "all_gather"), referenced by RS field 20.
        if r.annotated {
            if let Some(list) = annotations.get(&r.gnid) {
                for ann in list {
                    if let Some(n) = ann.get("name").and_then(|v| v.as_str()) {
                        event_names.get(n);
                    }
                }
            }
        }
    }

    // ---- emission ----------------------------------------------------------
    let mut trace: Vec<u8> = Vec::with_capacity(events.len() * 96);

    // 1. ClockSnapshot: BOOTTIME(6) + REALTIME(1), both at base — emitted
    //    only when the trace has a base to declare, like the fixed exporter.
    if base > 0 {
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
    }

    // 2. Track descriptors: process, then one thread per CPU track.
    let mut next_uuid: u64 = 1;
    if let Some(pid) = main_pid {
        let mut td = Vec::new();
        field_varint(&mut td, TD_UUID, next_uuid);
        let mut proc_ = Vec::new();
        field_int(&mut proc_, PD_PID, pid);
        if let Some(nm) = &main_pname {
            field_str(&mut proc_, PD_NAME, nm);
        }
        field_bytes(&mut td, TD_PROCESS, &proc_);
        let mut pkt = Vec::new();
        field_bytes(&mut pkt, TP_TRACK_DESCRIPTOR, &td);
        packet(&mut trace, &pkt);
        next_uuid += 1;
    }
    let mut track_uuid: HashMap<(i64, i64), u64> = HashMap::new();
    for &(pid, tid) in &cpu_tracks {
        let uuid = next_uuid;
        next_uuid += 1;
        track_uuid.insert((pid, tid), uuid);
        let mut td = Vec::new();
        field_varint(&mut td, TD_UUID, uuid);
        let mut th = Vec::new();
        field_int(&mut th, THD_PID, pid);
        field_int(&mut th, THD_TID, tid);
        if let Some(nm) = thread_names.get(&(pid, tid)) {
            field_str(&mut th, THD_NAME, nm);
        }
        field_bytes(&mut td, TD_THREAD, &th);
        let mut pkt = Vec::new();
        field_bytes(&mut pkt, TP_TRACK_DESCRIPTOR, &td);
        packet(&mut trace, &pkt);
    }

    // 3. One InternedData packet, incremental state cleared.
    {
        let mut interned = Vec::new();
        let entry = |out: &mut Vec<u8>, table: u64, iid: u64, name_field: u64, nm: &str| {
            let mut e = Vec::new();
            field_varint(&mut e, 1, iid);
            field_str(&mut e, name_field, nm);
            field_bytes(out, table, &e);
        };
        for (i, nm) in categories.order.iter().enumerate() {
            entry(&mut interned, ID_EVENT_CATEGORIES, 1 + i as u64, 2, nm);
        }
        for (i, nm) in event_names.order.iter().enumerate() {
            entry(&mut interned, ID_EVENT_NAMES, 1 + i as u64, 2, nm);
        }
        for (i, nm) in specs.order.iter().enumerate() {
            entry(&mut interned, ID_GPU_SPECS, 1 + i as u64, 2, nm);
        }
        for (iid, nm) in &queue_names {
            entry(&mut interned, ID_GPU_SPECS, *iid, 2, nm);
        }
        for (i, nm) in kernel_names.order.iter().enumerate() {
            entry(&mut interned, ID_EXT_KERNEL_NAMES, 1 + i as u64, 3, nm);
        }
        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_varint(&mut pkt, TP_SEQ_FLAGS, SEQ_INCREMENTAL_STATE_CLEARED);
        field_bytes(&mut pkt, TP_INTERNED, &interned);
        packet(&mut trace, &pkt);
    }

    // 4. CPU slices: per track, all BEGINs (ts asc, outer-first at ties)
    //    then all ENDs (end-ts asc) — the exporter's batched group emission,
    //    reconstructed exactly by the reader's ts-sorted stack.
    for key in &cpu_tracks {
        let uuid = track_uuid[key];
        let idxs = &cpu_by_track[key];
        let mut slices: Vec<(u64, u64, usize)> = idxs
            .iter()
            .map(|&i| {
                let ev = &events[i];
                let ts = ev_f64(ev, "ts").unwrap_or(0.0);
                let dur = ev_f64(ev, "dur").unwrap_or(0.0);
                let b = ns_of(base, ts);
                let e = b + (dur * 1000.0).round().max(0.0) as u64;
                (b, e, i)
            })
            .collect();
        slices.sort_by_key(|&(b, e, i)| (b, std::cmp::Reverse(e), i));
        // Nesting check: a slice must close before its enclosing one.
        {
            let mut stack: Vec<u64> = Vec::new();
            for &(b, e, _) in &slices {
                while stack.last().map(|&t| b >= t).unwrap_or(false) {
                    stack.pop();
                }
                if stack.last().map(|&t| e > t).unwrap_or(false) {
                    st.nest_violations += 1;
                }
                stack.push(e);
            }
        }
        for &(b, _, i) in &slices {
            let ev = &events[i];
            let mut te = Vec::new();
            field_varint(&mut te, TE_TYPE, TE_TYPE_BEGIN);
            let cat = ev_str(ev, "cat");
            if !catless_cpu_group(cat) {
                if let Some(iid) = categories.lookup(cat.unwrap()) {
                    field_varint(&mut te, TE_CATEGORY_IIDS, iid);
                }
            }
            if let Some(nm) = ev_str(ev, "name") {
                if let Some(iid) = event_names.lookup(nm) {
                    field_varint(&mut te, TE_NAME_IID, iid);
                }
            }
            field_varint(&mut te, TE_TRACK_UUID, uuid);
            if let Some(args) = ev.get("args").and_then(|a| a.as_object()) {
                for (k, v) in args {
                    let mut da = Vec::new();
                    field_str(&mut da, DA_NAME, k);
                    match v.as_i64() {
                        Some(x) => field_int(&mut da, DA_INT, x),
                        None => field_str(&mut da, DA_STRING, &python_str(v)),
                    }
                    field_bytes(&mut te, TE_DEBUG_ANN, &da);
                }
            }
            let mut pkt = Vec::new();
            field_varint(&mut pkt, TP_TIMESTAMP, b);
            field_varint(&mut pkt, TP_SEQ_ID, 1);
            field_varint(&mut pkt, TP_SEQ_FLAGS, SEQ_NEEDS_INCREMENTAL_STATE);
            field_bytes(&mut pkt, TP_TRACK_EVENT, &te);
            packet(&mut trace, &pkt);
        }
        let mut ends: Vec<u64> = slices.iter().map(|&(_, e, _)| e).collect();
        ends.sort_unstable();
        for e in ends {
            let mut te = Vec::new();
            field_varint(&mut te, TE_TYPE, TE_TYPE_END);
            field_varint(&mut te, TE_TRACK_UUID, uuid);
            let mut pkt = Vec::new();
            field_varint(&mut pkt, TP_TIMESTAMP, e);
            field_varint(&mut pkt, TP_SEQ_ID, 1);
            field_varint(&mut pkt, TP_SEQ_FLAGS, SEQ_NEEDS_INCREMENTAL_STATE);
            field_bytes(&mut pkt, TP_TRACK_EVENT, &te);
            packet(&mut trace, &pkt);
        }
        st.cpu_slices += slices.len();
    }

    // 5. GPU render stages, in row order.
    for r in &gpu_rows {
        let ev = &events[r.idx];
        let cat = ev_str(ev, "cat").unwrap_or("");
        let stage = stage_of(cat).expect("classified as GPU above");
        let stage_iid = specs.lookup(stage).unwrap();
        let device = ev_i64(ev, "pid").unwrap_or(0);
        let ts = ev_f64(ev, "ts").unwrap_or(0.0);
        let dur = ev_f64(ev, "dur").unwrap_or(0.0);

        let mut rs = Vec::new();
        let correlation = arg(ev, "correlation").and_then(|v| v.as_i64());
        field_varint(&mut rs, RS_EVENT_ID, correlation.unwrap_or(0).max(0) as u64);
        field_varint(&mut rs, RS_DURATION, (dur * 1000.0).round().max(0.0) as u64);
        field_int(&mut rs, RS_GPU_ID, device);
        field_varint(&mut rs, RS_HW_QUEUE_IID, queue_iids[&r.lane]);
        field_varint(&mut rs, RS_STAGE_IID, stage_iid);
        if let Some(ctx) = arg(ev, "context").and_then(|v| v.as_i64()) {
            field_int(&mut rs, RS_CONTEXT, ctx);
        }
        // Event name: the resolver's annotation name for annotated rows,
        // the row's own name for Annotation stages.
        let ename: Option<&str> = if cat == "gpu_user_annotation" {
            ev_str(ev, "name")
        } else if r.annotated {
            annotations.get(&r.gnid).and_then(|l| {
                l.iter()
                    .rev()
                    .find_map(|a| a.get("name").and_then(|v| v.as_str()))
            })
        } else {
            None
        };
        if let Some(nm) = ename {
            if let Some(iid) = event_names.lookup(nm) {
                field_varint(&mut rs, RS_EXT_EVENT_NAME_IID, iid);
            }
        }
        if cat == "kernel" {
            if let Some(nm) = ev_str(ev, "name") {
                if let Some(iid) = kernel_names.lookup(nm) {
                    field_varint(&mut rs, RS_EXT_KERNEL_NAME_IID, iid);
                }
            }
            // Launch config: grid + block, straight from the row's args.
            let dims = |key: &str| -> Option<[u64; 3]> {
                let a = arg(ev, key)?.as_array()?;
                let mut d = [1u64; 3];
                for (i, v) in a.iter().take(3).enumerate() {
                    d[i] = v.as_u64().unwrap_or(1);
                }
                Some(d)
            };
            let (grid, block) = (dims("grid"), dims("block"));
            if grid.is_some() || block.is_some() {
                let mut launch = Vec::new();
                for (f, d) in [(1u64, grid), (2u64, block)] {
                    if let Some(d) = d {
                        let mut g = Vec::new();
                        field_varint(&mut g, 1, d[0]);
                        field_varint(&mut g, 2, d[1]);
                        field_varint(&mut g, 3, d[2]);
                        field_bytes(&mut launch, f, &g);
                    }
                }
                field_bytes(&mut rs, RS_EXT_LAUNCH, &launch);
            }
        }

        // extra_data, exporter-shaped: canonical CUPTI extras first, then
        // graph ids (split), then the annotation / mirrored args sorted.
        let extra = |rs: &mut Vec<u8>, k: &str, v: &str| {
            let mut ex = Vec::new();
            field_str(&mut ex, 1, k);
            field_str(&mut ex, 2, v);
            field_bytes(rs, RS_EXTRA_DATA, &ex);
        };
        extra(&mut rs, "device id", &device.to_string());
        if let Some(ctx) = arg(ev, "context").and_then(|v| v.as_i64()) {
            extra(&mut rs, "context id", &ctx.to_string());
        }
        if let Some(s) = arg(ev, "stream").and_then(|v| v.as_i64()) {
            extra(&mut rs, "stream id", &s.to_string());
        }
        if let Some(c) = correlation {
            extra(&mut rs, "correlation", &c.to_string());
        }
        if let Some(q) = arg(ev, "queued") {
            extra(&mut rs, "queued", &python_str(q));
        }
        if let Some(b) = arg(ev, "bytes") {
            extra(&mut rs, "bytes", &python_str(b));
        }
        if let Some(a) = arch_of.get(&device) {
            extra(&mut rs, "arch", a);
        }
        if let Some(pid) = main_pid {
            extra(&mut rs, "process_id", &pid.to_string());
        }
        if let Some(nm) = &main_pname {
            extra(&mut rs, "process_name", nm);
        }
        if r.gnid != 0 {
            // Split encoding, losslessly: the reader recombines via
            // `combine_gnid` ((gid<<32)|node for 0<node<2^32 and gid>0);
            // values that would not survive the round trip (node-low==0, or
            // an uncombined id) are emitted combined/verbatim instead.
            let gid = (r.gnid as u64 >> 32) as i64;
            let node = (r.gnid as u64 & 0xFFFF_FFFF) as i64;
            if gid > 0 && node > 0 {
                extra(&mut rs, "graph id", &gid.to_string());
                extra(&mut rs, "graph node id", &node.to_string());
            } else {
                if let Some(g) = arg(ev, "graph id").and_then(|v| v.as_i64()) {
                    if g != gid || node != 0 {
                        // graph id only when it cannot re-combine wrongly:
                        // combine_gnid touches only 0<gnid<2^32 with gid>0.
                        if r.gnid >= 1i64 << 32 || g <= 0 {
                            extra(&mut rs, "graph id", &g.to_string());
                        }
                    } else {
                        extra(&mut rs, "graph id", &g.to_string());
                    }
                }
                extra(&mut rs, "graph node id", &r.gnid.to_string());
            }
        }
        // Annotation kv (walk_annotate's arg updates), then the row's own
        // unhandled args (pre-existing PG metadata on non-graphed nccl
        // rows), sorted; annotation "stream" is overwritten by the final
        // integer lane exactly like args["stream"] in the walk.
        let mut kv: std::collections::BTreeMap<String, String> = Default::default();
        if let Some(args) = ev.get("args").and_then(|a| a.as_object()) {
            for (k, v) in args {
                if !gpu_arg_handled(k) {
                    kv.insert(k.clone(), python_str(v));
                }
            }
        }
        if r.annotated {
            if let Some(list) = annotations.get(&r.gnid) {
                for ann in list {
                    for (k, v) in ann {
                        kv.insert(k.clone(), python_str(v));
                    }
                }
            }
        }
        if let Some(fs) = r.final_stream {
            kv.insert("stream".to_string(), fs.to_string());
        }
        for (k, v) in &kv {
            extra(&mut rs, k, v);
        }

        let mut pkt = Vec::new();
        field_varint(&mut pkt, TP_TIMESTAMP, ns_of(base, ts));
        field_varint(&mut pkt, TP_SEQ_ID, 1);
        field_varint(&mut pkt, TP_SEQ_FLAGS, SEQ_NEEDS_INCREMENTAL_STATE);
        field_bytes(&mut pkt, TP_GPU_RENDER_STAGE, &rs);
        packet(&mut trace, &pkt);
        st.gpu_stages += 1;
    }

    st.raw_bytes = trace.len();
    Ok((trace, st))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::pftrace::NO_NAME;
    use crate::trace::pftrace::PfCat;
    use crate::trace::pftrace::read_trace;

    fn ann(entries: &[(&str, Value)]) -> std::collections::HashMap<String, Value> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn fixture_tree() -> Value {
        serde_json::json!({
            "baseTimeNanoseconds": 1_000_000_000_000u64,
            "deviceProperties": [{"id": 0, "computeMajor": 9, "computeMinor": 0}],
            "traceEvents": [
                {"ph": "M", "name": "process_name", "pid": 42, "tid": 0,
                 "args": {"name": "python3"}},
                {"ph": "M", "name": "thread_name", "pid": 42, "tid": 43,
                 "args": {"name": "pt_main_thread"}},
                {"ph": "M", "name": "thread_name", "pid": 0, "tid": 7,
                 "args": {"name": "stream 7 "}},
                // Nested CPU slices (outer cpu_op, inner cuda_runtime).
                {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "pid": 42, "tid": 43,
                 "ts": 1000.000, "dur": 4.000,
                 "args": {"Ev Idx": 5, "External id": 77}},
                {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                 "pid": 42, "tid": 43, "ts": 1001.000, "dur": 1.000,
                 "args": {"cbid": 211, "correlation": 501}},
                // Catless group.
                {"ph": "X", "cat": "overhead", "name": "Activity Buffer Request",
                 "pid": -1, "tid": 0, "ts": 1010.500, "dur": 0.250},
                // Graphed, annotated collective kernel (combined gnid).
                {"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllGather(void*)",
                 "pid": 0, "tid": 7000, "ts": 2000.123, "dur": 4.500,
                 "args": {"device": 0, "context": 1, "stream": 7000,
                          "correlation": 501, "queued": 0,
                          "grid": [32, 1, 1], "block": [128, 1, 1],
                          "registers per thread": 96,
                          "graph id": 5, "graph node id": (5i64<<32) | 12}},
                // Graphed, unannotated kernel → default stream lane.
                {"ph": "X", "cat": "kernel", "name": "eltwise(float*)",
                 "pid": 0, "tid": 7000, "ts": 2010.000, "dur": 1.000,
                 "args": {"device": 0, "context": 1, "stream": 7000,
                          "correlation": 502, "grid": [4, 1, 1], "block": [64, 1, 1],
                          "graph id": 5, "graph node id": (5i64<<32) | 13}},
                // Non-graph memcpy on its capture stream.
                {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH (Device -> Pinned)",
                 "pid": 0, "tid": 7, "ts": 2020.000, "dur": 0.500,
                 "args": {"device": 0, "context": 1, "stream": 7,
                          "correlation": 503, "bytes": 4096,
                          "memory bandwidth (GB/s)": 8.19}},
                // ac2g flow + instant rows: dropped.
                {"ph": "f", "id": 1, "pid": 0, "tid": 7, "ts": 2020.0, "cat": "ac2g",
                 "name": "ac2g", "bp": "e"},
                {"ph": "i", "name": "Iteration Start", "pid": "Traces",
                 "tid": "Trace X", "ts": 900.0, "s": "g"},
                // String-pid pseudo span: dropped.
                {"ph": "X", "cat": "Trace", "name": "PyTorch Profiler (0)",
                 "pid": "Spans", "tid": "PyTorch Profiler", "ts": 900.0,
                 "dur": 2000.0, "args": {"Op count": 0}},
            ]
        })
    }

    fn fixture_annotations() -> Annotations {
        let mut a = Annotations::default();
        a.insert(
            (5i64 << 32) | 12,
            vec![ann(&[
                ("name", Value::from("all_gather")),
                ("stream", Value::from(61)),
                ("Group size", Value::from(64)),
                ("Process Group Description", Value::from("HSDP_SHARD")),
                ("Process Group Name", Value::from("abc123")),
                ("Process Group Ranks", serde_json::json!([0, 2, 4])),
                ("dtype", Value::from("uint8")),
            ])],
        );
        a
    }

    #[test]
    fn roundtrip_through_the_reader() {
        let tree = fixture_tree();
        let (bytes, st) = tree_to_pftrace(&tree, &fixture_annotations(), 7).unwrap();
        assert_eq!(st.cpu_slices, 3);
        assert_eq!(st.gpu_stages, 3);
        assert_eq!(st.annotated_rows, 1);
        assert_eq!(st.dropped_flows, 1);
        assert_eq!(st.dropped_string_track, 1);
        assert_eq!(st.nest_violations, 0);

        let tr = read_trace(&bytes).unwrap();
        assert_eq!(tr.base_ns, Some(1_000_000_000_000));
        assert_eq!((tr.unmatched_begin, tr.unmatched_end), (0, 0));
        // 3 GPU + 3 CPU rows.
        assert_eq!(tr.events.len(), 6);

        // Annotated collective: lane 61, split gnid recombined, PG inline.
        let k = tr
            .events
            .iter()
            .find(|e| tr.name(e.name).starts_with("ncclDevKernel"))
            .unwrap();
        assert_eq!(k.cat, PfCat::Kernel);
        assert_eq!(k.gnid, (5i64 << 32) | 12);
        assert_eq!((k.stream, k.logical_stream, k.gpu_id), (7000, 61, 0));
        assert_eq!(tr.name(k.pg_desc), "HSDP_SHARD");
        assert_eq!(tr.name(k.pg_name), "abc123");
        assert_eq!(tr.name(k.dtype), "uint8");
        assert_eq!(k.grid0, 32);
        let (qname, _) = tr.hw_queues[k.track as usize];
        assert_eq!(tr.name(qname), "[61] HSDP_SHARD");
        // µs round trip: 2000.123 µs → exact.
        assert_eq!(k.ts_ns - tr.base_ns.unwrap(), 2_000_123);
        assert_eq!(k.dur_ns, 4_500);

        // Unannotated graphed kernel → default lane 7 (shared with the
        // capture-stream memcpy's lane? no — memcpy tid is 7 → same lane id).
        let e = tr
            .events
            .iter()
            .find(|e| tr.name(e.name).starts_with("eltwise"))
            .unwrap();
        assert_eq!(e.logical_stream, 7);
        assert_eq!(e.gnid, (5i64 << 32) | 13);

        // Memcpy: named by stage, original stream, no logical stream.
        let m = tr.events.iter().find(|e| e.cat == PfCat::Memcpy).unwrap();
        assert_eq!(tr.name(m.name), "Memcpy");
        assert_eq!((m.stream, m.logical_stream), (7, -1));

        // CPU slices: exact spans, categories, nesting preserved.
        let mm = tr
            .events
            .iter()
            .find(|e| tr.name(e.name) == "aten::mm")
            .unwrap();
        assert_eq!(tr.name(mm.cat_name), "cpu_op");
        assert_eq!(mm.ts_ns - tr.base_ns.unwrap(), 1_000_000);
        assert_eq!(mm.dur_ns, 4_000);
        let lk = tr
            .events
            .iter()
            .find(|e| tr.name(e.name) == "cudaLaunchKernel")
            .unwrap();
        assert_eq!(lk.correlation, 501);
        assert_eq!(lk.dur_ns, 1_000);
        let ovh = tr
            .events
            .iter()
            .find(|e| {
                e.cat == PfCat::CpuSlice
                    && e.name != NO_NAME
                    && tr.name(e.name).starts_with("Activity")
            })
            .unwrap();
        assert_eq!(ovh.cat_name, NO_NAME); // catless group
        let th = &tr.threads[ovh.track as usize];
        assert_eq!((th.pid, th.tid), (-1, 0));

        // Thread names arrived verbatim.
        let mt = &tr.threads[mm.track as usize];
        assert_eq!((mt.pid, mt.tid), (42, 43));
        assert_eq!(tr.name(mt.name.unwrap()), "pt_main_thread");
        assert_eq!(
            tr.process,
            Some((42, {
                // interned somewhere in the table; resolve by value
                tr.names.iter().position(|n| n == "python3").unwrap() as u32
            }))
        );

        assert_eq!(tr.live_gids().into_iter().collect::<Vec<_>>(), vec![5]);
    }

    #[test]
    fn node_low_zero_gnid_survives_combined() {
        let mut tree = fixture_tree();
        // gnid = 248<<32 (node low bits zero) — split would read back as 0.
        let gnid = 248i64 << 32;
        tree["traceEvents"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "ph": "X", "cat": "kernel", "name": "agk(void*)",
                "pid": 0, "tid": 7000, "ts": 3000.000, "dur": 1.000,
                "args": {"device": 0, "stream": 7000, "correlation": 900,
                         "graph id": 248, "graph node id": gnid}
            }));
        let (bytes, _) = tree_to_pftrace(&tree, &Annotations::default(), 7).unwrap();
        let tr = read_trace(&bytes).unwrap();
        let k = tr
            .events
            .iter()
            .find(|e| tr.name(e.name).starts_with("agk"))
            .unwrap();
        assert_eq!(k.gnid, gnid);
        let mut gids: Vec<i64> = tr.live_gids().into_iter().collect();
        gids.sort_unstable();
        assert_eq!(gids, vec![5, 248]);
    }
}

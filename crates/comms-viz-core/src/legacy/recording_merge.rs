// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! The CUDA-graph recording-step merge.
//!
//! This is the dominant cost of the recording-present path
//! (`tree::recording_merge_tree`): it classifies the comm streams of both the
//! recording and the replay trace, matches their kernels (by count, by
//! name+duration, and by name+order), reassigns the replay kernels' stream/tid
//! to a canonical scheme, copies recording metadata onto the matched replay
//! kernels (setdefault semantics — the replay keeps its own NCCL PG metadata),
//! regroups NCCL kernels by Process Group, and remaps the non-kernel event tids
//! to follow the new stream assignment. It operates on the replay
//! `serde_json::Value` tree in place.

use std::collections::HashMap;
use std::collections::HashSet;

use serde_json::Map;
use serde_json::Value;

/// Lightweight kernel record. `event_idx` indexes into the owning trace's
/// `traceEvents` array; `ts`/`dur`/`stream`/`name` are copied out so the
/// sorting + matching passes don't keep a borrow on the events array. The cat
/// is not stored — Step 5's per-type counting reads it from the recording
/// events array directly via `event_idx`.
#[derive(Clone)]
struct Kernel {
    event_idx: usize,
    name: String,
    ts: f64,
    dur: f64,
    stream: i64,
}

/// Per-stream classification of kernels.
#[derive(Default)]
struct StreamInfo {
    nccl_ag: Vec<Kernel>,
    nccl_rs: Vec<Kernel>,
    nccl_rs_fsdp: Vec<Kernel>,
    nccl_ar: Vec<Kernel>,
    nccl_sendrecv: Vec<Kernel>,
    dtensor_ag: Vec<Kernel>,
    dtensor_rs: Vec<Kernel>,
    compute: Vec<Kernel>,
}

/// Kernel-name classification.
fn classify_kernel(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("nccl") {
        if n.contains("allgather") {
            return "nccl_ag";
        }
        if n.contains("reducescatter") {
            if n.contains("_f32_") {
                return "nccl_rs_fsdp";
            }
            return "nccl_rs";
        }
        if n.contains("allreduce") {
            return "nccl_ar";
        }
        if n.contains("sendrecv") {
            return "nccl_sendrecv";
        }
        return "nccl_other";
    }
    if n.contains("_all_gather") {
        return "dtensor_ag";
    }
    if n.contains("_reduce_scatter") {
        return "dtensor_rs";
    }
    "compute"
}

fn f64_field(ev: &Value, key: &str) -> Option<f64> {
    ev.get(key).and_then(|v| v.as_f64())
}

/// `args["stream"]` as i64 (traces carry both int and float literals). None
/// when the field is absent or null.
fn arg_stream_i64(ev: &Value) -> Option<i64> {
    ev.get("args")
        .and_then(|a| a.get("stream"))
        .and_then(value_to_i64)
}

fn value_to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        _ => None,
    }
}

fn cat_of(ev: &Value) -> &str {
    ev.get("cat").and_then(|v| v.as_str()).unwrap_or("")
}

fn name_of(ev: &Value) -> &str {
    ev.get("name").and_then(|v| v.as_str()).unwrap_or("")
}

/// Group "kernel"/"Kernel" events by `args.stream`, skipping events with no
/// stream/ts/dur. Kernels are classified into the per-stream buckets. Events
/// whose stream is missing OR JSON null are skipped; `stream == 0` is kept.
fn extract_kernels(events: &[Value]) -> HashMap<i64, StreamInfo> {
    let mut streams: HashMap<i64, StreamInfo> = HashMap::new();
    for (idx, event) in events.iter().enumerate() {
        let cat = cat_of(event);
        if cat != "kernel" && cat != "Kernel" {
            continue;
        }
        let Some(stream) = arg_stream_i64(event) else {
            continue;
        };
        let (Some(ts), Some(dur)) = (f64_field(event, "ts"), f64_field(event, "dur")) else {
            continue;
        };
        let name = name_of(event).to_string();
        let kind = classify_kernel(&name);
        let kernel = Kernel {
            event_idx: idx,
            name,
            ts,
            dur,
            stream,
        };
        let si = streams.entry(stream).or_default();
        match kind {
            "nccl_ag" => si.nccl_ag.push(kernel),
            "nccl_rs" => si.nccl_rs.push(kernel),
            "nccl_rs_fsdp" => si.nccl_rs_fsdp.push(kernel),
            "nccl_ar" => si.nccl_ar.push(kernel),
            "nccl_sendrecv" => si.nccl_sendrecv.push(kernel),
            "dtensor_ag" => si.dtensor_ag.push(kernel),
            "dtensor_rs" => si.dtensor_rs.push(kernel),
            _ => si.compute.push(kernel),
        }
    }
    streams
}

fn sort_by_ts(kernels: &mut [Kernel]) {
    kernels.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal));
}

/// True if `[ts, ts+dur)` overlaps any compute kernel. O(n) scan,
/// short-circuiting on the first hit.
fn has_compute_overlap(k: &Kernel, all_compute: &[Kernel]) -> bool {
    let start = k.ts;
    let end = k.ts + k.dur;
    all_compute
        .iter()
        .any(|c| start < c.ts + c.dur && c.ts < end)
}

/// Classification groups produced by `classify_comm_streams`.
struct CommClassified {
    fsdp_allgather: Vec<Kernel>,
    tp: Vec<Kernel>,
    fsdp_reduce_scatter: Vec<Kernel>,
    pp_p2p: Vec<Kernel>,
    others: Vec<Kernel>,
}

/// Classify comm kernels into FSDP/TP/PP/other groups. NOTE: `streams` is a
/// HashMap, so `values()` iterates in arbitrary order; the result lists are
/// sorted by ts at the end, so the iteration order does not affect the final
/// grouping (only membership + overlap classification, which is
/// deterministic).
fn classify_comm_streams(streams: &HashMap<i64, StreamInfo>) -> CommClassified {
    let mut all_compute: Vec<Kernel> = Vec::new();
    for si in streams.values() {
        all_compute.extend(si.compute.iter().cloned());
    }
    sort_by_ts(&mut all_compute);

    let mut out = CommClassified {
        fsdp_allgather: Vec::new(),
        tp: Vec::new(),
        fsdp_reduce_scatter: Vec::new(),
        pp_p2p: Vec::new(),
        others: Vec::new(),
    };
    for si in streams.values() {
        for ag in &si.nccl_ag {
            if has_compute_overlap(ag, &all_compute) {
                out.fsdp_allgather.push(ag.clone());
            } else {
                out.tp.push(ag.clone());
            }
        }
        for rs in si.nccl_rs.iter().chain(si.nccl_rs_fsdp.iter()) {
            if has_compute_overlap(rs, &all_compute) {
                out.fsdp_reduce_scatter.push(rs.clone());
            } else {
                out.tp.push(rs.clone());
            }
        }
        out.pp_p2p.extend(si.nccl_sendrecv.iter().cloned());
        out.others.extend(si.nccl_ar.iter().cloned());
    }
    sort_by_ts(&mut out.fsdp_allgather);
    sort_by_ts(&mut out.tp);
    sort_by_ts(&mut out.fsdp_reduce_scatter);
    sort_by_ts(&mut out.pp_p2p);
    sort_by_ts(&mut out.others);
    out
}

/// Match by count: pair replay[i] with recording[i] in ts order, up to min
/// length. Returns (rec_event_idx, rep_kernel) pairs — we carry the
/// recording event_idx so Step 5 can copy its args.
fn match_by_count(replay: &[Kernel], recording: &[Kernel]) -> Vec<(usize, Kernel)> {
    let mut rec = recording.to_vec();
    let mut rep = replay.to_vec();
    sort_by_ts(&mut rec);
    sort_by_ts(&mut rep);
    rep.into_iter()
        .enumerate()
        .filter_map(|(i, rep_k)| rec.get(i).map(|rec_k| (rec_k.event_idx, rep_k)))
        .collect()
}

/// Match by name + duration: group recording by name (ts order), for each
/// replay kernel (ts order) pick the same-name recording kernel with the
/// smallest relative duration diff, consume it.
fn match_by_name_and_duration(replay: &[Kernel], recording: &[Kernel]) -> Vec<(usize, Kernel)> {
    let mut rec_by_name: HashMap<String, Vec<Kernel>> = HashMap::new();
    let mut rec = recording.to_vec();
    sort_by_ts(&mut rec);
    for k in rec {
        rec_by_name.entry(k.name.clone()).or_default().push(k);
    }
    let mut rep = replay.to_vec();
    sort_by_ts(&mut rep);
    let mut matches = Vec::new();
    for rep_k in rep {
        let Some(candidates) = rec_by_name.get_mut(&rep_k.name) else {
            continue;
        };
        if candidates.is_empty() {
            continue;
        }
        let mut best_i = 0usize;
        let mut best_diff = f64::INFINITY;
        for (i, rec_k) in candidates.iter().enumerate() {
            let diff = (rec_k.dur - rep_k.dur).abs() / rep_k.dur.max(1.0);
            if diff < best_diff {
                best_diff = diff;
                best_i = i;
            }
        }
        let rec_k = candidates.remove(best_i);
        matches.push((rec_k.event_idx, rep_k));
    }
    matches
}

/// Match by name + order: group recording by name (ts order); for each replay
/// kernel (ts order) consume the first same-name recording kernel.
fn match_kernels_by_name(replay: &[Kernel], recording: &[Kernel]) -> Vec<(usize, Kernel)> {
    let mut rec_by_name: HashMap<String, std::collections::VecDeque<Kernel>> = HashMap::new();
    let mut rec = recording.to_vec();
    sort_by_ts(&mut rec);
    for k in rec {
        rec_by_name.entry(k.name.clone()).or_default().push_back(k);
    }
    let mut rep = replay.to_vec();
    sort_by_ts(&mut rep);
    let mut matches = Vec::new();
    for rep_k in rep {
        if let Some(q) = rec_by_name.get_mut(&rep_k.name) {
            if let Some(rec_k) = q.pop_front() {
                matches.push((rec_k.event_idx, rep_k));
            }
        }
    }
    matches
}

/// Greedy 2-coloring by ts so overlapping kernels land on different lists;
/// prefer stream1.
fn split_overlapping_kernels(kernels: &[Kernel]) -> (Vec<Kernel>, Vec<Kernel>) {
    if kernels.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut sorted = kernels.to_vec();
    sort_by_ts(&mut sorted);
    let mut s1: Vec<Kernel> = Vec::new();
    let mut s2: Vec<Kernel> = Vec::new();
    for k in sorted {
        let ov1 = s1.last().is_some_and(|l| k.ts < l.ts + l.dur);
        let ov2 = s2.last().is_some_and(|l| k.ts < l.ts + l.dur);
        if !ov1 {
            s1.push(k);
        } else if !ov2 {
            s2.push(k);
        } else {
            s1.push(k);
        }
    }
    (s1, s2)
}

/// First unused candidate stream, else the first unused id ≥ `fallback`.
fn get_unique_stream(candidates: &[i64], mut fallback: i64, used: &mut HashSet<i64>) -> i64 {
    for &s in candidates {
        if !used.contains(&s) {
            used.insert(s);
            return s;
        }
    }
    while used.contains(&fallback) {
        fallback += 1;
    }
    used.insert(fallback);
    fallback
}

/// Count stream occurrences and return the keys sorted by descending count.
/// Stable sort: ties keep first-seen (insertion) order — the tie order feeds
/// the canonical stream assignment, so it is part of the artifact bytes.
fn streams_by_descending_count(counts: &[(i64, i64)]) -> Vec<i64> {
    // `counts` is in insertion order (stream, count). Stable sort by -count.
    let mut v: Vec<(i64, i64)> = counts.to_vec();
    v.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
    v.into_iter().map(|(s, _)| s).collect()
}

/// Build an ordered (stream, count) tally over the given kernels' streams,
/// preserving first-seen order (the tie-break order for
/// `streams_by_descending_count`).
fn tally_streams<'a>(kernels: impl Iterator<Item = &'a Kernel>) -> Vec<(i64, i64)> {
    let mut order: Vec<i64> = Vec::new();
    let mut counts: HashMap<i64, i64> = HashMap::new();
    for k in kernels {
        let e = counts.entry(k.stream).or_insert_with(|| {
            order.push(k.stream);
            0
        });
        *e += 1;
    }
    order.into_iter().map(|s| (s, counts[&s])).collect()
}

/// Tally `args.stream` over events of a given cat, preserving first-seen order.
fn tally_event_streams(events: &[Value], cat: &str) -> Vec<(i64, i64)> {
    let mut order: Vec<i64> = Vec::new();
    let mut counts: HashMap<i64, i64> = HashMap::new();
    for ev in events {
        if cat_of(ev) != cat {
            continue;
        }
        if let Some(s) = arg_stream_i64(ev) {
            let e = counts.entry(s).or_insert_with(|| {
                order.push(s);
                0
            });
            *e += 1;
        }
    }
    order.into_iter().map(|s| (s, counts[&s])).collect()
}

/// Result of the full merge — match counts feeding the sparse-vs-full
/// self-check test.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MergeCounts {
    pub(crate) rec_events: usize,
    pub(crate) rep_events: usize,
    pub(crate) total_matches: usize,
    pub(crate) meta_compute: usize,
    pub(crate) meta_comm: usize,
    pub(crate) meta_memcpy: usize,
    pub(crate) meta_memset: usize,
    pub(crate) tid_remapped: usize,
}

/// Sort key for PG regroup: TP first, then CP, then the rest alphabetically.
/// Sorted as (bucket, name) with a stable sort.
fn pg_sort_bucket(pg: &str) -> u8 {
    let l = pg.to_lowercase();
    if l.contains("_tp_") || l.ends_with("_tp") {
        return 0;
    }
    if l.contains("_cp_") || l.ends_with("_cp") {
        return 1;
    }
    2
}

/// Categories whose recording events are match-relevant. Every other recording
/// event is dropped (replaced by `Value::Null`) by the streaming path — the
/// merge only ever reads recording events of these cats (via `cat_of` filters
/// in `extract_kernels`, `tally_event_streams`, `collect_mem_kernels`, and the
/// Step-5 metadata copy, which only touches *matched* kernels).
pub(crate) const RECORDING_KEEP_CATS: &[&str] = &["kernel", "Kernel", "gpu_memcpy", "gpu_memset"];

/// Decide whether a recording event must be retained for the merge.
///
/// True iff its `cat` is in `RECORDING_KEEP_CATS`. Used by the streaming reader
/// to drop the bulk of the recording (cpu_op / ac2g / annotation / metadata
/// events — the recording is ~3x the replay, almost all of which is these) so
/// the recording never fully materialises as a `serde_json::Value` tree.
pub(crate) fn recording_event_is_relevant(ev: &Value) -> bool {
    RECORDING_KEEP_CATS.contains(&cat_of(ev))
}

/// The full recording-step merge. Mutates `replay` in place.
///
/// Takes OWNERSHIP of a recording events array — which, on the streaming
/// path, is a SPARSE array where every event
/// that is not a kernel/memcpy/memset has been replaced by `Value::Null`. The
/// original event indices are preserved (so the `event_idx` carried by matched
/// recording kernels still indexes correctly), but the heavy non-kernel event
/// bodies were never held in memory. Behaviourally identical to passing the
/// full recording tree: every recording read in this function filters by `cat`,
/// and a `Null` event reports `cat == ""`, so it is skipped exactly as a
/// dropped event would be.
pub(crate) fn merge_cudagraph_metadata_from_rec_events(
    rec_events_ref: Vec<Value>,
    replay: &mut Value,
) -> MergeCounts {
    // Extract kernels from both traces. The recording kernels carry their own
    // event_idx into the recording events array (used to copy args in Step 5).
    let mut counts = MergeCounts {
        rec_events: rec_events_ref.len(),
        ..Default::default()
    };

    let rec_streams = extract_kernels(&rec_events_ref);

    // Pull the replay events array out to own + mutate it; reinsert at the end.
    let mut rep_events: Vec<Value> = replay
        .get_mut("traceEvents")
        .and_then(|v| v.as_array_mut())
        .map(std::mem::take)
        .unwrap_or_default();
    counts.rep_events = rep_events.len();

    let rep_streams = extract_kernels(&rep_events);

    // Steps 1 & 2: classify comm streams.
    let rep_classified = classify_comm_streams(&rep_streams);
    let rec_classified = classify_comm_streams(&rec_streams);

    // Step 3: match comm groups. (rec_event_idx, rep_kernel) pairs.
    let mut all_matches: Vec<(usize, Kernel)> = Vec::new();
    all_matches.extend(match_by_count(
        &rep_classified.fsdp_allgather,
        &rec_classified.fsdp_allgather,
    ));
    all_matches.extend(match_by_count(&rep_classified.tp, &rec_classified.tp));
    all_matches.extend(match_by_count(
        &rep_classified.fsdp_reduce_scatter,
        &rec_classified.fsdp_reduce_scatter,
    ));

    // PP P2P: match by count, then classify send/recv from recording args.
    let p2p_matches = match_by_count(&rep_classified.pp_p2p, &rec_classified.pp_p2p);
    let mut pp_send_indices: HashSet<usize> = HashSet::new();
    let mut pp_recv_indices: HashSet<usize> = HashSet::new();
    for (rec_idx, rep_k) in &p2p_matches {
        let collective = rec_events_ref
            .get(*rec_idx)
            .and_then(|e| e.get("args"))
            .and_then(|a| a.get("Collective name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();
        if collective.contains("send") {
            pp_send_indices.insert(rep_k.event_idx);
        } else if collective.contains("recv") {
            pp_recv_indices.insert(rep_k.event_idx);
        }
    }
    all_matches.extend(p2p_matches);

    // Others: match by name and duration.
    all_matches.extend(match_by_name_and_duration(
        &rep_classified.others,
        &rec_classified.others,
    ));

    // Step 3.5: compute + dtensor kernels, matched by name + order.
    let mut rep_compute: Vec<Kernel> = Vec::new();
    let mut rec_compute: Vec<Kernel> = Vec::new();
    for si in rep_streams.values() {
        rep_compute.extend(si.compute.iter().cloned());
        rep_compute.extend(si.dtensor_ag.iter().cloned());
        rep_compute.extend(si.dtensor_rs.iter().cloned());
    }
    for si in rec_streams.values() {
        rec_compute.extend(si.compute.iter().cloned());
        rec_compute.extend(si.dtensor_ag.iter().cloned());
        rec_compute.extend(si.dtensor_rs.iter().cloned());
    }
    sort_by_ts(&mut rep_compute);
    sort_by_ts(&mut rec_compute);
    all_matches.extend(match_kernels_by_name(&rep_compute, &rec_compute));

    // Step 3.6/3.7: memcpy + memset, matched by name + order.
    let rep_memcpy = collect_mem_kernels(&rep_events, "gpu_memcpy");
    let rec_memcpy = collect_mem_kernels(&rec_events_ref, "gpu_memcpy");
    all_matches.extend(match_kernels_by_name(&rep_memcpy, &rec_memcpy));

    let rep_memset = collect_mem_kernels(&rep_events, "gpu_memset");
    let rec_memset = collect_mem_kernels(&rec_events_ref, "gpu_memset");
    all_matches.extend(match_kernels_by_name(&rep_memset, &rec_memset));

    counts.total_matches = all_matches.len();

    // Step 4: assign canonical streams from the recording candidates.
    let mut used: HashSet<i64> = HashSet::new();

    let compute_candidates = streams_by_descending_count(&tally_streams(
        rec_streams.values().flat_map(|s| &s.compute),
    ));
    let compute_stream = get_unique_stream(&compute_candidates, 7, &mut used);

    let (rec_fsdp_ag1, rec_fsdp_ag2) = split_overlapping_kernels(&rec_classified.fsdp_allgather);
    let fsdp_ag1 = get_unique_stream(&streams_of(&rec_fsdp_ag1), 20, &mut used);
    let fsdp_ag2 = get_unique_stream(&streams_of(&rec_fsdp_ag2), 21, &mut used);

    let (rec_fsdp_rs1, rec_fsdp_rs2) =
        split_overlapping_kernels(&rec_classified.fsdp_reduce_scatter);
    let fsdp_rs1 = get_unique_stream(&streams_of(&rec_fsdp_rs1), 22, &mut used);
    let fsdp_rs2 = get_unique_stream(&streams_of(&rec_fsdp_rs2), 23, &mut used);

    let (rec_tp1, rec_tp2) = split_overlapping_kernels(&rec_classified.tp);
    let tp1 = get_unique_stream(&streams_of(&rec_tp1), 24, &mut used);
    let tp2 = get_unique_stream(&streams_of(&rec_tp2), 27, &mut used);

    let other_stream = get_unique_stream(&streams_of(&rec_classified.others), 25, &mut used);

    let memset_candidates =
        streams_by_descending_count(&tally_event_streams(&rec_events_ref, "gpu_memset"));
    let memset_stream = get_unique_stream(&memset_candidates, 26, &mut used);

    let memcpy_candidates =
        streams_by_descending_count(&tally_event_streams(&rec_events_ref, "gpu_memcpy"));
    let memcpy_stream = get_unique_stream(&memcpy_candidates, 70, &mut used);

    let p2p_cands = streams_of(&rec_classified.pp_p2p);
    let pp_send_stream = get_unique_stream(&p2p_cands, 28, &mut used);
    let pp_recv_stream = get_unique_stream(&p2p_cands, 29, &mut used);

    // Split replay groups for the per-event stream assignment.
    let (rep_fsdp_ag1, rep_fsdp_ag2) = split_overlapping_kernels(&rep_classified.fsdp_allgather);
    let (rep_fsdp_rs1, rep_fsdp_rs2) =
        split_overlapping_kernels(&rep_classified.fsdp_reduce_scatter);
    let (rep_tp1, rep_tp2) = split_overlapping_kernels(&rep_classified.tp);

    let idx_set = |ks: &[Kernel]| -> HashSet<usize> { ks.iter().map(|k| k.event_idx).collect() };
    let fsdp_ag1_idx = idx_set(&rep_fsdp_ag1);
    let fsdp_ag2_idx = idx_set(&rep_fsdp_ag2);
    let fsdp_rs1_idx = idx_set(&rep_fsdp_rs1);
    let fsdp_rs2_idx = idx_set(&rep_fsdp_rs2);
    let tp1_idx = idx_set(&rep_tp1);
    let tp2_idx = idx_set(&rep_tp2);
    let other_idx = idx_set(&rep_classified.others);

    // Unmatched P2P kernels default to send stream.
    let pp_all_idx = idx_set(&rep_classified.pp_p2p);
    for i in &pp_all_idx {
        if !pp_send_indices.contains(i) && !pp_recv_indices.contains(i) {
            pp_send_indices.insert(*i);
        }
    }

    // Compute indices: compute + dtensor_ag + dtensor_rs across replay streams.
    let mut compute_indices: HashSet<usize> = HashSet::new();
    for si in rep_streams.values() {
        for k in si
            .compute
            .iter()
            .chain(&si.dtensor_ag)
            .chain(&si.dtensor_rs)
        {
            compute_indices.insert(k.event_idx);
        }
    }

    // Pre-scan: original args.stream per event (for the Step 5.5 remap votes).
    let mut orig_streams: HashMap<usize, i64> = HashMap::new();
    for (idx, ev) in rep_events.iter().enumerate() {
        if let Some(s) = arg_stream_i64(ev) {
            orig_streams.insert(idx, s);
        }
    }

    // Apply stream assignments (Step 4 loop).
    for (idx, event) in rep_events.iter_mut().enumerate() {
        let cat = cat_of(event).to_string();
        if !matches!(
            cat.as_str(),
            "kernel" | "Kernel" | "gpu_memcpy" | "gpu_memset"
        ) {
            continue;
        }
        ensure_args(event);
        let original_stream = arg_stream_i64(event);

        let (new_stream, comm_type): (i64, &str) = if cat == "gpu_memset" {
            (memset_stream, "memset")
        } else if cat == "gpu_memcpy" {
            (memcpy_stream, "memcpy")
        } else if pp_send_indices.contains(&idx) {
            (pp_send_stream, "pp_p2p_send")
        } else if pp_recv_indices.contains(&idx) {
            (pp_recv_stream, "pp_p2p_recv")
        } else if compute_indices.contains(&idx) {
            (compute_stream, "compute")
        } else if fsdp_ag1_idx.contains(&idx) {
            (fsdp_ag1, "fsdp_allgather_1")
        } else if fsdp_ag2_idx.contains(&idx) {
            (fsdp_ag2, "fsdp_allgather_2")
        } else if fsdp_rs1_idx.contains(&idx) {
            (fsdp_rs1, "fsdp_reduce_scatter_1")
        } else if fsdp_rs2_idx.contains(&idx) {
            (fsdp_rs2, "fsdp_reduce_scatter_2")
        } else if tp1_idx.contains(&idx) {
            (tp1, "tp_1")
        } else if tp2_idx.contains(&idx) {
            (tp2, "tp_2")
        } else if other_idx.contains(&idx) {
            (other_stream, "other")
        } else {
            (compute_stream, "unclassified")
        };

        set_arg_i64(event, "stream", new_stream);
        set_arg_str(event, "comm_type", comm_type);
        // tid follows the new stream only when it equaled the original
        // stream; otherwise it is left untouched.
        if let Some(orig) = original_stream {
            if event.get("tid").and_then(value_to_i64) == Some(orig) {
                event["tid"] = Value::from(new_stream);
            }
        }
    }

    // Step 5: copy recording metadata onto matched replay kernels.
    for (rec_idx, rep_k) in &all_matches {
        let rec_args = rec_events_ref
            .get(*rec_idx)
            .and_then(|e| e.get("args"))
            .and_then(|a| a.as_object())
            .cloned()
            .unwrap_or_default();
        let rec_cat = rec_events_ref
            .get(*rec_idx)
            .map(|e| cat_of(e).to_string())
            .unwrap_or_default();
        let event = &mut rep_events[rep_k.event_idx];
        ensure_args(event);
        if let Some(args) = event.get_mut("args").and_then(|a| a.as_object_mut()) {
            for (key, value) in &rec_args {
                if key == "stream" || key == "queued" || key == "correlation" {
                    continue;
                }
                args.entry(key.clone()).or_insert_with(|| value.clone());
            }
            args.insert(
                "metadata_source".to_string(),
                Value::String("cudagraph_recording".to_string()),
            );
        }
        if compute_indices.contains(&rep_k.event_idx) {
            counts.meta_compute += 1;
        } else if rec_cat == "gpu_memcpy" {
            counts.meta_memcpy += 1;
        } else if rec_cat == "gpu_memset" {
            counts.meta_memset += 1;
        } else {
            counts.meta_comm += 1;
        }
    }

    // Step 5.2: regroup NCCL kernels by Process Group Name.
    regroup_by_pg(
        &mut rep_events,
        compute_stream,
        memset_stream,
        memcpy_stream,
    );

    // Step 5.5 + 5.6: remap non-kernel event tids, then merge annotations.
    counts.tid_remapped = remap_event_tids(&mut rep_events, &orig_streams);
    let rep_events = merge_overlapping_annotations(rep_events);

    // Reinsert the mutated events array.
    if let Some(obj) = replay.as_object_mut() {
        obj.insert("traceEvents".to_string(), Value::Array(rep_events));
    }

    counts
}

fn streams_of(kernels: &[Kernel]) -> Vec<i64> {
    kernels.iter().map(|k| k.stream).collect()
}

/// Collect memcpy/memset kernels (Step 3.6/3.7 style: requires ts AND dur, uses
/// `args.stream` or 0).
fn collect_mem_kernels(events: &[Value], cat: &str) -> Vec<Kernel> {
    let mut out: Vec<Kernel> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| cat_of(e) == cat)
        .filter_map(|(idx, e)| {
            let ts = f64_field(e, "ts")?;
            let dur = f64_field(e, "dur")?;
            let stream = arg_stream_i64(e).unwrap_or(0);
            Some(Kernel {
                event_idx: idx,
                name: name_of(e).to_string(),
                ts,
                dur,
                stream,
            })
        })
        .collect();
    sort_by_ts(&mut out);
    out
}

fn ensure_args(event: &mut Value) {
    if let Some(obj) = event.as_object_mut() {
        if !obj.contains_key("args") {
            obj.insert("args".to_string(), Value::Object(Map::new()));
        }
    }
}

fn set_arg_i64(event: &mut Value, key: &str, val: i64) {
    if let Some(args) = event.get_mut("args").and_then(|a| a.as_object_mut()) {
        args.insert(key.to_string(), Value::from(val));
    }
}

fn set_arg_str(event: &mut Value, key: &str, val: &str) {
    if let Some(args) = event.get_mut("args").and_then(|a| a.as_object_mut()) {
        args.insert(key.to_string(), Value::String(val.to_string()));
    }
}

/// Step 5.2 — regroup NCCL kernels by Process Group Name.
fn regroup_by_pg(events: &mut [Value], compute: i64, memset: i64, memcpy: i64) {
    // Build PG -> [(event_idx)], preserving first-seen PG order.
    let mut pg_order: Vec<String> = Vec::new();
    let mut by_pg: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, ev) in events.iter().enumerate() {
        let cat = cat_of(ev);
        if cat != "kernel" && cat != "Kernel" {
            continue;
        }
        if !name_of(ev).to_lowercase().contains("nccl") {
            continue;
        }
        let pg = ev
            .get("args")
            .and_then(|a| a.get("Process Group Name"))
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        by_pg.entry(pg.clone()).or_insert_with(|| {
            pg_order.push(pg.clone());
            Vec::new()
        });
        by_pg.get_mut(&pg).expect("pg key just inserted").push(idx);
    }
    if by_pg.is_empty() {
        return;
    }

    let mut pg_used: HashSet<i64> = [compute, memset, memcpy].into_iter().collect();
    let mut pg_next: i64 = 20;

    // Sort PGs: (bucket, name). Stable so equal buckets stay alphabetical via
    // the secondary key.
    let mut pgs: Vec<String> = pg_order.clone();
    pgs.sort_by(|a, b| {
        pg_sort_bucket(a)
            .cmp(&pg_sort_bucket(b))
            .then_with(|| a.cmp(b))
    });

    for pg in pgs {
        let items = &by_pg[&pg];
        let kernels: Vec<Kernel> = items
            .iter()
            .map(|&idx| {
                let ev = &events[idx];
                Kernel {
                    event_idx: idx,
                    name: name_of(ev).to_string(),
                    ts: f64_field(ev, "ts").unwrap_or(0.0),
                    dur: f64_field(ev, "dur").unwrap_or(0.0),
                    stream: arg_stream_i64(ev).unwrap_or(0),
                }
            })
            .collect();
        let (s1_list, s2_list) = split_overlapping_kernels(&kernels);

        while pg_used.contains(&pg_next) {
            pg_next += 1;
        }
        let s1 = pg_next;
        pg_used.insert(s1);
        pg_next += 1;

        let s2 = if !s2_list.is_empty() {
            while pg_used.contains(&pg_next) {
                pg_next += 1;
            }
            let v = pg_next;
            pg_used.insert(v);
            pg_next += 1;
            Some(v)
        } else {
            None
        };

        let s1_idx: HashSet<usize> = s1_list.iter().map(|k| k.event_idx).collect();
        for &idx in items {
            let target = if s1_idx.contains(&idx) { Some(s1) } else { s2 };
            if let Some(t) = target {
                set_arg_i64(&mut events[idx], "stream", t);
                // Only numeric tids are rewritten; anything else is left
                // untouched.
                if events[idx]
                    .get("tid")
                    .map(|v| v.is_number())
                    .unwrap_or(false)
                {
                    events[idx]["tid"] = Value::from(t);
                }
            }
        }
    }
}

/// Matched against annotation names with case and underscores dropped, so every
/// producer's spelling counts: `all_gather`, `_allgather_base`,
/// `allgather_into_tensor_coalesced`, `allGatherSingleImpl`, and likewise for
/// the other collectives.
const NCCL_ANN_PATTERNS: &[&str] = &["allgather", "reducescatter", "allreduce", "alltoall"];

fn is_nccl_annotation(name: &str) -> bool {
    let normalized: String = name
        .chars()
        .filter(|c| *c != '_')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    NCCL_ANN_PATTERNS.iter().any(|p| normalized.contains(p))
}

/// Step 5.5 — remap non-kernel event tids by majority vote, with
/// per-annotation NCCL overlap matching. Drops stale `thread_name` metadata for
/// remapped streams. Returns number of tids remapped. Mutates `events` in place
/// AND removes the stale thread_name events (returns the kept set via retain).
fn remap_event_tids(events: &mut Vec<Value>, orig_streams: &HashMap<usize, i64>) -> usize {
    // Build remap votes: old args.stream -> {new args.stream: count}.
    let mut votes: HashMap<i64, HashMap<i64, i64>> = HashMap::new();
    // Track first-seen order of new-stream keys so max-by-count ties resolve
    // to the first-encountered max — the tie rule decides remapped tids in
    // the artifact.
    let mut vote_order: HashMap<i64, Vec<i64>> = HashMap::new();
    for (idx, event) in events.iter().enumerate() {
        let cat = cat_of(event);
        if !matches!(cat, "kernel" | "Kernel" | "gpu_memcpy" | "gpu_memset") {
            continue;
        }
        let (Some(&old), Some(new)) = (orig_streams.get(&idx), arg_stream_i64(event)) else {
            continue;
        };
        let inner = votes.entry(old).or_default();
        let ord = vote_order.entry(old).or_default();
        inner.entry(new).and_modify(|c| *c += 1).or_insert_with(|| {
            ord.push(new);
            1
        });
    }
    let mut stream_remap: HashMap<i64, i64> = HashMap::new();
    for (old, inner) in &votes {
        // max by count, first-seen on ties.
        let order = &vote_order[old];
        let mut best = order[0];
        let mut best_c = inner[&best];
        for &k in order.iter().skip(1) {
            let c = inner[&k];
            if c > best_c {
                best_c = c;
                best = k;
            }
        }
        stream_remap.insert(*old, best);
    }

    // Build NCCL index: old tid (orig stream) -> [(ts, dur, new stream)].
    let mut nccl_idx: HashMap<i64, Vec<(f64, f64, i64)>> = HashMap::new();
    for (idx, event) in events.iter().enumerate() {
        let cat = cat_of(event);
        if cat != "kernel" && cat != "Kernel" {
            continue;
        }
        if !name_of(event).to_lowercase().contains("nccl") {
            continue;
        }
        let (Some(&old), Some(new)) = (orig_streams.get(&idx), arg_stream_i64(event)) else {
            continue;
        };
        let ts = f64_field(event, "ts").unwrap_or(0.0);
        let dur = f64_field(event, "dur").unwrap_or(0.0);
        nccl_idx.entry(old).or_default().push((ts, dur, new));
    }

    let find_nccl_stream = |old_tid: i64, ts: f64, dur: f64| -> Option<i64> {
        let list = nccl_idx.get(&old_tid)?;
        let ann_end = ts + dur;
        let mut best_stream = None;
        let mut best_overlap = 0.0_f64;
        for &(k_ts, k_dur, k_stream) in list {
            let overlap = ann_end.min(k_ts + k_dur) - ts.max(k_ts);
            if overlap > best_overlap {
                best_overlap = overlap;
                best_stream = Some(k_stream);
            }
        }
        best_stream
    };

    let mut tid_remapped = 0usize;
    let mut kept: Vec<Value> = Vec::with_capacity(events.len());
    for event in std::mem::take(events) {
        let cat = cat_of(&event);
        if matches!(cat, "kernel" | "Kernel" | "gpu_memcpy" | "gpu_memset") {
            kept.push(event);
            continue;
        }
        let Some(tid) = event.get("tid").and_then(|v| v.as_f64()) else {
            kept.push(event);
            continue;
        };
        let tid_int = tid as i64;
        let Some(&default_new) = stream_remap.get(&tid_int) else {
            kept.push(event);
            continue;
        };

        // Drop stale thread_name metadata for remapped streams.
        if event.get("ph").and_then(|v| v.as_str()) == Some("M")
            && name_of(&event) == "thread_name"
            && default_new != tid_int
        {
            continue;
        }

        let new_tid = if cat == "gpu_user_annotation" {
            if is_nccl_annotation(name_of(&event)) {
                let ts = f64_field(&event, "ts").unwrap_or(0.0);
                let dur = f64_field(&event, "dur").unwrap_or(0.0);
                find_nccl_stream(tid_int, ts, dur).unwrap_or(default_new)
            } else {
                default_new
            }
        } else {
            default_new
        };

        let mut event = event;
        if new_tid != tid_int {
            event["tid"] = Value::from(new_tid);
            tid_remapped += 1;
        }
        kept.push(event);
    }
    *events = kept;
    tid_remapped
}

/// Split out `gpu_user_annotation` ph=="X" events, group by (name, tid) in
/// first-seen order, merge overlapping intervals, then return
/// `other_events + merged`.
fn merge_overlapping_annotations(events: Vec<Value>) -> Vec<Value> {
    let mut annotations: Vec<Value> = Vec::new();
    let mut others: Vec<Value> = Vec::new();
    for ev in events {
        if cat_of(&ev) == "gpu_user_annotation"
            && ev.get("ph").and_then(|v| v.as_str()) == Some("X")
        {
            annotations.push(ev);
        } else {
            others.push(ev);
        }
    }
    if annotations.is_empty() {
        // No annotations: return the event list unchanged, in its
        // original order.
        others.extend(annotations);
        return others;
    }

    // Group by (name, tid) preserving first-seen order. Key uses `value_repr`
    // string forms so numerically-equal int/float tids group together (tids
    // are ints post-merge).
    let mut order: Vec<(String, String)> = Vec::new();
    let mut groups: HashMap<(String, String), Vec<Value>> = HashMap::new();
    for ann in annotations {
        let name = ann
            .get("name")
            .map(value_repr)
            .unwrap_or_else(|| "None".to_string());
        let tid = ann
            .get("tid")
            .map(value_repr)
            .unwrap_or_else(|| "None".to_string());
        let key = (name, tid);
        groups.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            Vec::new()
        });
        groups.get_mut(&key).expect("group just inserted").push(ann);
    }

    let mut merged: Vec<Value> = Vec::new();
    for key in order {
        let mut group = groups.remove(&key).expect("key from order present");
        group.sort_by(|a, b| {
            let ta = f64_field(a, "ts").unwrap_or(0.0);
            let tb = f64_field(b, "ts").unwrap_or(0.0);
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut current = group[0].clone();
        for ann in group.into_iter().skip(1) {
            let cur_ts = f64_field(&current, "ts").unwrap_or(0.0);
            let cur_dur = f64_field(&current, "dur").unwrap_or(0.0);
            let cur_end = cur_ts + cur_dur;
            let ann_ts = f64_field(&ann, "ts").unwrap_or(0.0);
            let ann_dur = f64_field(&ann, "dur").unwrap_or(0.0);
            if ann_ts <= cur_end {
                let new_end = cur_end.max(ann_ts + ann_dur);
                current["dur"] = Value::from(new_end - cur_ts);
            } else {
                merged.push(current);
                current = ann.clone();
            }
        }
        merged.push(current);
    }

    others.extend(merged);
    others
}

/// String form of a JSON value matching Python `repr`-equivalence for use as a
/// grouping key (numbers compare by value, strings by content).
fn value_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            // Normalise 1 vs 1.0 so int/float tids group together like Python.
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 {
                    (f as i64).to_string()
                } else {
                    f.to_string()
                }
            } else {
                n.to_string()
            }
        }
        Value::String(s) => format!("s:{s}"),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full-tree reference form of the merge: clones the recording's events
    /// array and delegates to the sparse core. Kept test-only as the
    /// cross-check reference for the sparse path.
    fn merge_cudagraph_metadata(recording: &Value, replay: &mut Value) -> MergeCounts {
        let rec_events_ref: Vec<Value> = recording
            .get("traceEvents")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        merge_cudagraph_metadata_from_rec_events(rec_events_ref, replay)
    }

    #[test]
    fn nccl_annotation_name_variants() {
        for name in [
            "nccl:all_gather",
            "nccl:_allgather_base",
            "nccl:allgather_into_tensor_coalesced",
            "allGatherSingleImpl",
            "nccl:_reduce_scatter_base",
            "reduceScatterSingleImpl",
            "nccl:allreduce",
            "all_reduce",
            "nccl:all_to_all",
            "nccl:all_to_allv",
        ] {
            assert!(is_nccl_annotation(name), "{name}");
        }
        for name in ["nccl:broadcast", "aten::mm", "nccl:send"] {
            assert!(!is_nccl_annotation(name), "{name}");
        }
    }

    #[test]
    fn classify_kernel_buckets() {
        assert_eq!(classify_kernel("ncclDevKernel_AllGather"), "nccl_ag");
        assert_eq!(classify_kernel("ncclReduceScatter_f32_x"), "nccl_rs_fsdp");
        assert_eq!(classify_kernel("ncclReduceScatter_bf16"), "nccl_rs");
        assert_eq!(classify_kernel("ncclAllReduce"), "nccl_ar");
        assert_eq!(classify_kernel("ncclSendRecv"), "nccl_sendrecv");
        assert_eq!(classify_kernel("some_all_gather_thing"), "dtensor_ag");
        assert_eq!(classify_kernel("gemm_kernel"), "compute");
    }

    #[test]
    fn split_alternates_overlapping() {
        let mk = |idx, ts, dur| Kernel {
            event_idx: idx,
            name: "k".into(),
            ts,
            dur,
            stream: 0,
        };
        // Two overlapping kernels -> different streams.
        let (s1, s2) = split_overlapping_kernels(&[mk(0, 0.0, 10.0), mk(1, 5.0, 10.0)]);
        assert_eq!(s1.len(), 1, "first overlapping kernel on stream1");
        assert_eq!(s2.len(), 1, "second overlapping kernel on stream2");
    }

    #[test]
    fn match_by_count_pairs_min_len() {
        let mk = |idx, ts| Kernel {
            event_idx: idx,
            name: "k".into(),
            ts,
            dur: 1.0,
            stream: 0,
        };
        let rep = vec![mk(10, 1.0), mk(11, 2.0), mk(12, 3.0)];
        let rec = vec![mk(0, 5.0), mk(1, 6.0)];
        let m = match_by_count(&rep, &rec);
        assert_eq!(m.len(), 2, "min(3,2)=2 matches");
        // ts-order: rep[10]<->rec[0], rep[11]<->rec[1].
        assert_eq!(m[0].0, 0);
        assert_eq!(m[0].1.event_idx, 10);
    }

    #[test]
    fn full_merge_smoke() {
        // Recording: one compute kernel on stream 99 with extra metadata.
        let recording = serde_json::json!({
            "traceEvents": [
                {"cat": "kernel", "name": "gemm", "ts": 0, "dur": 10, "tid": 99,
                 "args": {"stream": 99, "Foo": "bar"}},
            ]
        });
        // Replay: same compute kernel on stream 5; expect remap to the
        // recording's dominant compute stream (99) — `get_unique_stream`'s
        // 7 is only the exhausted-candidates fallback — and the "Foo"
        // metadata copied via setdefault.
        let mut replay = serde_json::json!({
            "traceEvents": [
                {"cat": "kernel", "name": "gemm", "ts": 0, "dur": 10, "tid": 5,
                 "args": {"stream": 5}},
            ]
        });
        let counts = merge_cudagraph_metadata(&recording, &mut replay);
        assert_eq!(counts.rep_events, 1);
        assert_eq!(counts.total_matches, 1, "one compute kernel matched");
        assert_eq!(counts.meta_compute, 1);
        let ev = &replay["traceEvents"][0];
        assert_eq!(
            ev["args"]["stream"], 99,
            "compute keeps recording's dominant stream"
        );
        assert_eq!(ev["tid"], 99, "tid follows stream when it equaled orig");
        assert_eq!(ev["args"]["Foo"], "bar", "recording metadata copied");
        assert_eq!(ev["args"]["comm_type"], "compute");
        assert_eq!(ev["args"]["metadata_source"], "cudagraph_recording");
    }

    /// EQUIVALENCE GATE: sparse reader vs full parse. Merging against a
    /// recording whose non-kernel events have been dropped to `Value::Null`
    /// (what the streaming reader produces) MUST yield byte-identical replay
    /// output to merging against the full recording tree. The merge only reads
    /// kernel/memcpy/memset recording events, so the Null placeholders are
    /// invisible to it.
    #[test]
    fn sparse_recording_matches_full_recording() {
        // A recording with a mix of relevant (kernel/memcpy/memset) and
        // irrelevant (cpu_op / ac2g / annotation / metadata) events.
        let make_recording = || {
            serde_json::json!({
                "traceEvents": [
                    {"cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 5, "tid": 1},
                    {"cat": "kernel", "name": "gemm", "ts": 0, "dur": 10, "tid": 99,
                     "args": {"stream": 99, "Foo": "bar"}},
                    {"cat": "ac2g", "name": "link", "ph": "f", "ts": 1, "tid": 2},
                    {"cat": "kernel", "name": "ncclAllReduce", "ts": 20, "dur": 8, "tid": 50,
                     "args": {"stream": 50, "Process Group Name": "REC_PG"}},
                    {"cat": "gpu_memcpy", "name": "Memcpy DtoD", "ts": 30, "dur": 3,
                     "args": {"stream": 12}},
                    {"ph": "M", "name": "thread_name", "tid": 99, "pid": 1,
                     "args": {"name": "stream 99"}},
                ]
            })
        };
        let make_replay = || {
            serde_json::json!({
                "traceEvents": [
                    {"cat": "kernel", "name": "gemm", "ts": 0, "dur": 10, "tid": 5,
                     "args": {"stream": 5}},
                    {"cat": "kernel", "name": "ncclAllReduce", "ts": 20, "dur": 8, "tid": 6,
                     "args": {"stream": 6, "Process Group Name": "REPLAY_PG"}},
                    {"cat": "gpu_memcpy", "name": "Memcpy DtoD", "ts": 30, "dur": 3, "tid": 7,
                     "args": {"stream": 7}},
                    {"cat": "gpu_user_annotation", "name": "all_reduce", "ph": "X",
                     "ts": 20, "dur": 8, "tid": 6},
                ]
            })
        };

        // Full-recording path.
        let recording = make_recording();
        let mut replay_full = make_replay();
        let counts_full = merge_cudagraph_metadata(&recording, &mut replay_full);

        // Sparse-recording path: replace non-kernel events with Null, preserving
        // indices (exactly what `recording_read::read_recording_sparse` produces).
        let rec_events: Vec<Value> = recording["traceEvents"].as_array().unwrap().clone();
        let sparse: Vec<Value> = rec_events
            .into_iter()
            .map(|ev| {
                if recording_event_is_relevant(&ev) {
                    ev
                } else {
                    Value::Null
                }
            })
            .collect();
        let mut replay_sparse = make_replay();
        let counts_sparse = merge_cudagraph_metadata_from_rec_events(sparse, &mut replay_sparse);

        assert_eq!(
            counts_full, counts_sparse,
            "sparse recording must produce identical match/metadata counts"
        );
        assert_eq!(
            replay_full, replay_sparse,
            "sparse recording must produce identical merged replay output"
        );
        // Sanity: the merge actually did something (matched + copied metadata).
        assert!(
            counts_full.total_matches >= 3,
            "expected the kernels to match"
        );
        assert_eq!(
            replay_full["traceEvents"][0]["args"]["Foo"], "bar",
            "recording metadata was copied through the sparse path"
        );
    }

    #[test]
    fn setdefault_preserves_replay_metadata() {
        let recording = serde_json::json!({
            "traceEvents": [
                {"cat": "kernel", "name": "ncclAllReduce", "ts": 0, "dur": 10, "tid": 50,
                 "args": {"stream": 50, "Process Group Name": "REC_PG"}},
            ]
        });
        let mut replay = serde_json::json!({
            "traceEvents": [
                {"cat": "kernel", "name": "ncclAllReduce", "ts": 0, "dur": 10, "tid": 3,
                 "args": {"stream": 3, "Process Group Name": "REPLAY_PG"}},
            ]
        });
        merge_cudagraph_metadata(&recording, &mut replay);
        // setdefault must NOT clobber the replay's own PG name.
        assert_eq!(
            replay["traceEvents"][0]["args"]["Process Group Name"], "REPLAY_PG",
            "replay PG metadata preserved by setdefault"
        );
    }
}

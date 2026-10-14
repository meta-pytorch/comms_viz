// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! In-place mutating walks over a parsed Kineto trace.
//!
//! The passes all operate on one `serde_json::Value` tree and are fused
//! by `walk_passes`, so nothing is re-parsed or re-serialized between
//! them.
//!
//! Each walk returns a count of the events it touched — internal
//! bookkeeping only.

use std::collections::HashMap;
use std::collections::HashSet;

use serde_json::Map;
use serde_json::Value;

use crate::annotations::Annotations;

/// Categories counted as "work" events.
const WORK_CATEGORIES: &[&str] = &["kernel", "gpu_memcpy", "gpu_memset"];

fn as_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn as_i64(v: &Value) -> i64 {
    v.as_i64().unwrap_or(0)
}

fn event_field_f64(ev: &Value, key: &str) -> f64 {
    ev.get(key).map(as_f64).unwrap_or(0.0)
}

fn event_field_i64(ev: &Value, key: &str) -> i64 {
    ev.get(key).map(as_i64).unwrap_or(0)
}

fn graph_node_id(ev: &Value) -> i64 {
    ev.get("args")
        .and_then(|a| a.get("graph node id"))
        .map(as_i64)
        .unwrap_or(0)
}

/// Stable hashable key for an event's `ts`: numerically equal int and
/// float literals (`1000` vs `1000.0`) must map to the same key, while
/// missing/null/non-numeric `ts` stays distinct from real `0.0` and
/// from each other.
///
/// CRITICAL: a previous version used `.unwrap_or(0)` here. That
/// collapsed every missing/float/null `ts` in the ac2g index to a
/// single bucket `0`. On a real 380K-event trace we observed one
/// bucket with 55K entries; the subsequent annotate loop became
/// O(graphed × bucket) ≈ 3.5 × 10^9 pushes and hung in
/// `RawVec::grow_one` for 10+ minutes. Always use `Option<u64>`.
fn ts_bits(ev: &Value) -> Option<u64> {
    ev.get("ts").and_then(|v| v.as_f64()).map(|f| {
        if f.is_nan() {
            f64::NAN.to_bits()
        } else {
            f.to_bits()
        }
    })
}

/// Same idea for tid — used in the ac2g key. Missing/non-numeric → None.
fn tid_bits(ev: &Value) -> Option<u64> {
    ev.get("tid").and_then(|v| v.as_f64()).map(|f| {
        if f.is_nan() {
            f64::NAN.to_bits()
        } else {
            f.to_bits()
        }
    })
}

/// Move graphed default-stream kernels that overlap the previous kept
/// kernel onto `overlap_stream` (rewrites both `tid` and `args.stream`);
/// non-overlapping kernels advance the high-water mark.
pub(crate) fn walk_move(root: &mut Value, default_stream: i64, overlap_stream: i64) -> usize {
    let Some(events) = root.get_mut("traceEvents").and_then(|v| v.as_array_mut()) else {
        return 0;
    };

    let mut indices: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.get("cat").and_then(|v| v.as_str()) == Some("kernel")
                && event_field_i64(e, "tid") == default_stream
                && graph_node_id(e) != 0
        })
        .map(|(i, _)| i)
        .collect();
    indices.sort_by(|&a, &b| {
        let ta = event_field_f64(&events[a], "ts");
        let tb = event_field_f64(&events[b], "ts");
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut moved = 0_usize;
    let mut prev_end: f64 = 0.0;
    for idx in indices {
        let ts = event_field_f64(&events[idx], "ts");
        let dur = event_field_f64(&events[idx], "dur");
        if ts < prev_end {
            if let Some(obj) = events[idx].as_object_mut() {
                obj.insert("tid".to_string(), Value::from(overlap_stream));
                if let Some(args) = obj.get_mut("args").and_then(|a| a.as_object_mut()) {
                    args.insert("stream".to_string(), Value::from(overlap_stream));
                }
            }
            moved += 1;
        } else {
            prev_end = ts + dur;
        }
    }
    moved
}

/// Clamp small per-stream timestamp overlaps: a graphed work event that
/// starts before the previous event's end is pushed forward to that end,
/// but only when the overlap is ≤ `max_adjust_us`.
pub(crate) fn walk_fix(root: &mut Value, max_adjust_us: f64) -> usize {
    let Some(events) = root.get_mut("traceEvents").and_then(|v| v.as_array_mut()) else {
        return 0;
    };

    let mut per_stream: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, ev) in events.iter().enumerate() {
        let cat = ev.get("cat").and_then(|v| v.as_str()).unwrap_or("");
        if !WORK_CATEGORIES.contains(&cat) {
            continue;
        }
        if graph_node_id(ev) == 0 {
            continue;
        }
        let tid = event_field_i64(ev, "tid");
        per_stream.entry(tid).or_default().push(i);
    }

    let mut adjusted = 0_usize;
    for (_tid, mut indices) in per_stream {
        indices.sort_by(|&a, &b| {
            let ta = event_field_f64(&events[a], "ts");
            let tb = event_field_f64(&events[b], "ts");
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut prev_end: f64 = 0.0;
        for idx in indices {
            let ts = event_field_f64(&events[idx], "ts");
            let dur = event_field_f64(&events[idx], "dur");
            let mut effective_ts = ts;
            if ts < prev_end {
                let overlap = prev_end - ts;
                if overlap <= max_adjust_us {
                    // Preserve the literal's numeric type: if the
                    // existing `ts` was an int and prev_end is integral,
                    // stay int; otherwise float. An integer ts literal
                    // stays an integer literal in the serialized artifact.
                    let was_int = events[idx]
                        .get("ts")
                        .map(|v| v.is_i64() || v.is_u64())
                        .unwrap_or(false);
                    let new_val: Value = if was_int && prev_end.fract() == 0.0 {
                        Value::from(prev_end as i64)
                    } else {
                        Value::from(prev_end)
                    };
                    if let Some(obj) = events[idx].as_object_mut() {
                        obj.insert("ts".to_string(), new_val);
                    }
                    effective_ts = prev_end;
                    adjusted += 1;
                }
            }
            prev_end = effective_ts + dur;
        }
    }
    adjusted
}

/// Python `str(value)` semantics applied to a `serde_json::Value`.
/// Used when annotate copies pickle annotation values into args.
fn python_str(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(_) | Value::Object(_) => v.to_string(),
    }
}

/// Coerce an annotation `stream` field to int — the pickles carry it as
/// an int, float, or numeric string.
fn coerce_to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.parse::<i64>().ok(),
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => None,
    }
}

/// Annotate graphed events from the pickle annotations, reassign their
/// stream/tid, then clean up trace metadata.
pub(crate) fn walk_annotate(
    root: &mut Value,
    annotations: &Annotations,
    default_stream: i64,
) -> usize {
    let Some(events) = root.get_mut("traceEvents").and_then(|v| v.as_array_mut()) else {
        return 0;
    };

    // Build ac2g_f_index: (tid, ts_bits) -> [indices].
    //
    // Key uses `ts_bits` (f64 bit pattern) rather than `i64` because
    // `ts` can be either int (parsed from JSON like `"ts":1000`) or
    // float (e.g. after `walk_fix` writes `prev_end`), and `(tid, 1000)`
    // must land in the same bucket as `(tid, 1000.0)` — a
    // numeric-equivalence-preserving key. Using `i64` here collapses
    // every float-ts event to `(tid, 0)`, which on real 12K traces
    // creates one bucket with 55K ac2g entries; the subsequent update
    // loop becomes O(graphed × bucket) ≈ 10^9 and hangs.
    let mut ac2g_index: HashMap<(Option<u64>, Option<u64>), Vec<usize>> = HashMap::new();
    for (i, ev) in events.iter().enumerate() {
        if ev.get("cat").and_then(|v| v.as_str()) == Some("ac2g")
            && ev.get("ph").and_then(|v| v.as_str()) == Some("f")
        {
            let key = (tid_bits(ev), ts_bits(ev));
            ac2g_index.entry(key).or_default().push(i);
        }
    }

    // First pass: collect updates (cannot mutate inside a &events iter).
    struct Update {
        idx: usize,
        new_tid: i64,
        arg_updates: Vec<(String, String)>,
    }
    let mut updates: Vec<Update> = Vec::new();
    let mut ac2g_updates: Vec<(usize, i64)> = Vec::new();
    let mut annotated = 0_usize;

    for (i, ev) in events.iter().enumerate() {
        let gnid = match ev
            .get("args")
            .and_then(|a| a.get("graph node id"))
            .and_then(|n| n.as_i64())
        {
            Some(g) if g != 0 => g,
            _ => continue,
        };

        let mut stream_id: Option<i64> = None;
        let mut arg_updates: Vec<(String, String)> = Vec::new();
        if let Some(ann_list) = annotations.get(&gnid) {
            for ann in ann_list {
                for (k, v) in ann {
                    arg_updates.push((k.clone(), python_str(v)));
                }
                if let Some(s_val) = ann.get("stream").and_then(coerce_to_i64) {
                    stream_id = Some(s_val);
                }
            }
            annotated += 1;
        }

        let final_stream = stream_id.unwrap_or(default_stream);
        let old_key = (tid_bits(ev), ts_bits(ev));
        updates.push(Update {
            idx: i,
            new_tid: final_stream,
            arg_updates,
        });
        if let Some(ac2g_idx_list) = ac2g_index.get(&old_key) {
            for &j in ac2g_idx_list {
                ac2g_updates.push((j, final_stream));
            }
        }
    }

    // Apply updates.
    for upd in updates {
        if let Some(obj) = events[upd.idx].as_object_mut() {
            obj.insert("tid".to_string(), Value::from(upd.new_tid));
            if let Some(args) = obj.get_mut("args").and_then(|a| a.as_object_mut()) {
                for (k, v) in upd.arg_updates {
                    args.insert(k, Value::String(v));
                }
                args.insert("stream".to_string(), Value::from(upd.new_tid));
            }
        }
    }
    for (idx, new_tid) in ac2g_updates {
        if let Some(obj) = events[idx].as_object_mut() {
            obj.insert("tid".to_string(), Value::from(new_tid));
        }
    }

    cleanup_trace_metadata(root);
    annotated
}

/// Drop noise events and stale metadata, then append synthetic
/// `thread_name` entries for work streams that lack one.
fn cleanup_trace_metadata(root: &mut Value) {
    let Some(events) = root.get_mut("traceEvents").and_then(|v| v.as_array_mut()) else {
        return;
    };

    // Pass 1: tids_with_work.
    let mut tids_with_work: HashSet<i64> = HashSet::new();
    for ev in events.iter() {
        let cat = ev.get("cat").and_then(|v| v.as_str()).unwrap_or("");
        if WORK_CATEGORIES.contains(&cat) {
            tids_with_work.insert(event_field_i64(ev, "tid"));
        }
    }

    // Filter 1: drop noise events.
    events.retain(|ev| {
        let cat = ev.get("cat").and_then(|v| v.as_str()).unwrap_or("");
        let tid = event_field_i64(ev, "tid");
        match cat {
            "gpu_user_annotation" => tids_with_work.contains(&tid),
            "ac2g" if ev.get("ph").and_then(|v| v.as_str()) == Some("f") => {
                tids_with_work.contains(&tid)
            }
            _ => true,
        }
    });

    // All tids carried by non-metadata (ph != "M") events.
    let mut all_tids: HashSet<i64> = HashSet::new();
    for ev in events.iter() {
        if ev.get("ph").and_then(|v| v.as_str()) != Some("M") {
            all_tids.insert(event_field_i64(ev, "tid"));
        }
    }

    // Find gpu_pid (first thread_name whose name starts with "stream ").
    let mut gpu_pid: i64 = 0;
    for ev in events.iter() {
        if ev.get("ph").and_then(|v| v.as_str()) == Some("M")
            && ev.get("name").and_then(|v| v.as_str()) == Some("thread_name")
        {
            let name = ev
                .get("args")
                .and_then(|a| a.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if name.starts_with("stream ") {
                gpu_pid = event_field_i64(ev, "pid");
                break;
            }
        }
    }

    // Filter 2: drop metadata events whose tid has no events.
    events.retain(|ev| {
        if ev.get("ph").and_then(|v| v.as_str()) != Some("M") {
            return true;
        }
        all_tids.contains(&event_field_i64(ev, "tid"))
    });

    // existing_thread_names.
    let mut existing_thread_names: HashSet<i64> = HashSet::new();
    for ev in events.iter() {
        if ev.get("ph").and_then(|v| v.as_str()) == Some("M")
            && ev.get("name").and_then(|v| v.as_str()) == Some("thread_name")
        {
            existing_thread_names.insert(event_field_i64(ev, "tid"));
        }
    }

    // Append synthetic thread_name events in ascending tid order — the
    // append order is part of the artifact bytes.
    let mut missing: Vec<i64> = tids_with_work
        .difference(&existing_thread_names)
        .copied()
        .collect();
    missing.sort();
    for tid in missing {
        let mut args = Map::new();
        args.insert("name".to_string(), Value::String(format!("stream {}", tid)));
        let mut ev = Map::new();
        ev.insert("ph".to_string(), Value::String("M".to_string()));
        ev.insert("pid".to_string(), Value::from(gpu_pid));
        ev.insert("tid".to_string(), Value::from(tid));
        ev.insert("name".to_string(), Value::String("thread_name".to_string()));
        ev.insert("args".to_string(), Value::Object(args));
        events.push(Value::Object(ev));
    }
}

/// Fused pipeline in the fixed post-merge order: annotate → move → fix.
pub(crate) fn walk_passes(
    root: &mut Value,
    annotations: &Annotations,
    default_stream: i64,
) -> (usize, usize, usize) {
    let n_a = walk_annotate(root, annotations, default_stream);
    let n_m = walk_move(root, default_stream, default_stream + 1);
    let n_t = walk_fix(root, 1.0);
    (n_a, n_m, n_t)
}

/// Inline-annotated overlap-only pipeline: the no-reassign variant for
/// self-annotated traces, same pass order. The producer already injected
/// annotation fields and reassigned tids, so the annotate/tid-reassign
/// pass MUST be skipped — running it would clobber the producer's tids
/// back to `default_stream`. Only the noise-cleanup + global overlap-move
/// + timestamp-clamp passes run, operating on the producer-assigned tids.
///
/// Returns `(0, n_moved, n_ts_fixed)` — nothing is annotated in this
/// mode, so the annotate count is always 0.
///
/// Ordering note: noise cleanup (build `tids_with_work`, drop orphan
/// `gpu_user_annotation`/`ac2g`, drop stale `M`, append synthetic
/// `thread_name`) runs BEFORE overlap detection — cleanup, then move,
/// then fix. The order decides which events the overlap passes see, so
/// it is part of the artifact bytes.
pub(crate) fn walk_passes_no_reassign(
    root: &mut Value,
    default_stream: i64,
) -> (usize, usize, usize) {
    cleanup_trace_metadata(root);
    let n_m = walk_move(root, default_stream, default_stream + 1);
    let n_t = walk_fix(root, 1.0);
    (0, n_m, n_t)
}

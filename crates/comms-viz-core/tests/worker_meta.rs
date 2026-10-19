// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Per-worker `_meta` artifact contract: one completed heartbeat doc per
//! worker, a
//! `_meta/failures/worker_{i}.json` doc when a worker's ranks errored, and
//! RunSummary counts that agree with those docs. Worlds this small get one
//! worker per rank. The final `_meta/status.json` keeps
//! a fixed key set — part of the schema tooling reads — filled with
//! truthful observed counts, never an all-succeeded summary.

use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;

use comms_viz_core::pipeline::RunConfig;
use comms_viz_core::pipeline::run;

const T0_US: i64 = 1_700_000_000_000_000;
const T1_US: i64 = 1_700_000_001_500_000; // dur 1.5s

fn pg_config() -> serde_json::Value {
    serde_json::json!([{"pg_name": "0", "pg_desc": "default_pg", "backend_config": "cuda:nccl",
                        "pg_size": 8, "ranks": [0, 1, 2, 3, 4, 5, 6, 7]}])
}

/// A trace whose head names `rank` but whose body is cut off mid-event.
fn write_truncated_trace(step_dir: &Path, rank: i64) {
    let body = format!(
        r#"{{"distributedInfo": {{"rank": {rank}, "world_size": 8}}, "traceEvents": [{{"ph": "X","#
    );
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(body.as_bytes()).unwrap();
    std::fs::write(
        step_dir.join(format!("{rank:06}.1700000000000.pt.trace.json.gz")),
        enc.finish().unwrap(),
    )
    .unwrap();
}

/// Minimal Kineto-shaped trace for one rank (flat layout filename).
fn write_trace(step_dir: &Path, rank: i64) {
    let events = serde_json::json!([
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "pid": 3, "tid": 1,
         "ts": T0_US, "dur": 10, "args": {"External id": 1}},
        {"ph": "X", "cat": "kernel", "name": "triton_poi_fused_elementwise_88",
         "pid": 3, "tid": 7, "ts": T0_US + 5, "dur": 42,
         "args": {"stream": 7, "correlation": 1, "grid": [1, 1, 1]}},
    ]);
    let trace = serde_json::json!({
        "schemaVersion": 1,
        "distributedInfo": {"rank": rank, "world_size": 8, "pg_config": pg_config()},
        "traceEvents": events,
        "traceName": format!("/tmp/rank{rank}.json"),
        "displayTimeUnit": "ms",
    });
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(trace.to_string().as_bytes()).unwrap();
    let bytes = enc.finish().unwrap();
    std::fs::create_dir_all(step_dir).unwrap();
    std::fs::write(
        step_dir.join(format!("{rank:06}.1700000000000.pt.trace.json.gz")),
        bytes,
    )
    .unwrap();
}

fn base_cfg(trace_root: PathBuf, out_root: PathBuf, world: i64) -> RunConfig {
    RunConfig {
        trace_root,
        source_trace_root: None,
        out_root,
        run_id: "test-run".into(),
        run_name: "test-job".into(),
        world_size: Some(world),
        trace_format: comms_viz_core::trace::Format::Json,
        annotation_mode: comms_viz_core::annotations::AnnotationMode::Disabled,
        recording_step: None,
        write_repaired_traces: false,
        threads: 2,
        started_at_us: T0_US,
        completed_at_us: T1_US,
        verbose: false,
        verify_walks: false,
    }
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// The wall clock as the CLI stamps started_at (µs epoch).
fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
}

/// round(x, 1) as the heartbeat/phase builders apply it.
fn round1(us: i64) -> f64 {
    (us as f64 / 1e6 * 10.0).round() / 10.0
}

#[test]
fn per_worker_heartbeats_and_counts() {
    let tmp = tempfile::tempdir().unwrap();
    let step = tmp.path().join("step_000000000106");
    // world 6 → one worker per rank; rank 3 missing → skip.
    for r in [0, 1, 2, 4, 5] {
        write_trace(&step, r);
    }
    let out = tmp.path().join("out");
    let summary = run(&base_cfg(step, out.clone(), 6)).unwrap();

    assert_eq!(summary.processed, 5);
    assert_eq!(summary.skipped, 1);
    assert_eq!(summary.errors, 0);
    assert_eq!(summary.skipped_ranks, vec![3]);
    assert!(summary.errored_ranks.is_empty());

    let meta = out.join("tree").join("test-run").join("_meta");
    // One completed heartbeat per worker, pinned schema.
    let hb3_path = meta.join("heartbeats").join("worker_3.json");
    for w in 0..6 {
        let hb = read_json(&meta.join("heartbeats").join(format!("worker_{w}.json")));
        assert_eq!(hb["worker_index"], w);
        assert_eq!(hb["assigned"], 1);
        assert_eq!(hb["state"], "completed");
        assert_eq!(hb["rank_lo"], w);
        assert_eq!(hb["rank_hi"], w + 1);
    }
    // Exact bytes for the worker with the skip (sorted key order, like
    // every metadata doc).
    assert_eq!(
        std::fs::read_to_string(&hb3_path).unwrap(),
        "{\"assigned\":1,\"duration_s\":1.5,\"errors\":0,\"processed\":0,\
         \"rank_hi\":4,\"rank_lo\":3,\"skipped\":1,\
         \"started_at\":1700000000000000,\"state\":\"completed\",\
         \"updated_at\":1700000001500000,\"worker_index\":3}"
    );
    assert_eq!(read_json(&meta.join("run.json"))["run_name"], "test-job");
    // No failures dir when nothing errored.
    assert!(!meta.join("failures").exists());

    // Final status.json: pinned key set/order, truthful values — the
    // skipped rank shows as processed < total, all workers completed,
    // elapsed from the run's started/completed pair.
    let status_path = meta.join("status.json");
    let status = read_json(&status_path);
    assert_eq!(status["ranks"]["processed"], 5);
    assert_eq!(status["ranks"]["total"], 6);
    assert_eq!(status["workers_done"], 6);
    assert_eq!(status["shards"]["completed"], 6);
    assert_eq!(status["shards"]["failed"], 0);
    assert_eq!(status["elapsed_s"], 1.5);
    // Exact bytes around the phases dict (sorted key order; phases content
    // is pinned by its own surface, not here).
    let status_text = std::fs::read_to_string(&status_path).unwrap();
    assert!(status_text.starts_with("{\"elapsed_s\":1.5,\"phase\":\"completed\",\"phases\":{"));
    assert!(status_text.ends_with(
        "\"ranks\":{\"processed\":5,\"total\":6},\"run_id\":\"test-run\",\
         \"shard_count\":6,\"shards\":{\"completed\":6,\"failed\":0,\
         \"in_flight\":0,\"pending\":0},\"version\":2,\
         \"workers_active\":0,\"workers_done\":6}"
    ));
}

#[test]
fn failure_doc_on_rank_error() {
    let tmp = tempfile::tempdir().unwrap();
    let step = tmp.path().join("step_000000000106");
    write_trace(&step, 0);
    // rank 1: truncated body → rank error (worker_1's whole range).
    write_truncated_trace(&step, 1);
    let out = tmp.path().join("out");
    let summary = run(&base_cfg(step, out.clone(), 2)).unwrap();

    assert_eq!(summary.processed, 1);
    assert_eq!(summary.skipped, 0);
    assert_eq!(summary.errors, 1);
    assert_eq!(summary.errored_ranks, vec![1]);

    let meta = out.join("tree").join("test-run").join("_meta");
    let hb1 = read_json(&meta.join("heartbeats").join("worker_1.json"));
    assert_eq!(hb1["processed"], 0);
    assert_eq!(hb1["errors"], 1);

    // Failure-doc shape: rank-level errors surface in the traceback field
    // (rank errors are caught, so this doc is where they land).
    let fail_path = meta.join("failures").join("worker_1.json");
    let fail = read_json(&fail_path);
    assert_eq!(fail["worker_index"], 1);
    assert_eq!(fail["exception_type"], "RankProcessingError");
    assert!(fail["traceback"].as_str().unwrap().starts_with("rank 1: "));
    assert_eq!(fail["timestamp"], T1_US);
    // indent-2 pretty layout, sorted key order.
    assert!(std::fs::read_to_string(&fail_path)
        .unwrap()
        .starts_with("{\n  \"exception_type\": \"RankProcessingError\",\n  \"timestamp\": 1700000001500000,\n  \"traceback\": \"rank 1: "));
    // Healthy worker: heartbeat only.
    assert!(!meta.join("failures").join("worker_0.json").exists());

    // status.json: the rank error shows as processed < total; the worker
    // itself completed (rank errors are caught), so no failed shard.
    let status = read_json(&meta.join("status.json"));
    assert_eq!(status["ranks"]["processed"], 1);
    assert_eq!(status["ranks"]["total"], 2);
    assert_eq!(status["workers_done"], 2);
    assert_eq!(status["shards"]["failed"], 0);
    assert_eq!(status["elapsed_s"], 1.5);
}

/// Default path (no --completed-at-us): completed_at_us stays 0 through the
/// run and every consuming doc is stamped at its own write time — after the
/// run's work, never the pre-run instant.
#[test]
fn unparameterized_completion_is_stamped_at_write_time() {
    let tmp = tempfile::tempdir().unwrap();
    let step = tmp.path().join("step_000000000106");
    write_trace(&step, 0);
    // rank 1: truncated body → rank error → worker_1 failure doc too.
    write_truncated_trace(&step, 1);
    let out = tmp.path().join("out");
    let mut cfg = base_cfg(step, out.clone(), 2);
    let t0 = now_us();
    cfg.started_at_us = t0; // what the CLI resolves at process start
    cfg.completed_at_us = 0;
    run(&cfg).unwrap();
    let t1 = now_us();
    let in_window = |ts: i64| ts > t0 && ts <= t1;

    // run.json + status.json: one resolved instant per emit_metadata call.
    let meta = out.join("tree").join("test-run").join("_meta");
    let run_doc = read_json(&meta.join("run.json"));
    assert_eq!(run_doc["started_at"], t0);
    let completed = run_doc["completed_at"].as_i64().unwrap();
    assert!(
        in_window(completed),
        "completed_at {completed} not in ({t0}, {t1}]"
    );
    let status = read_json(&meta.join("status.json"));
    // Real recorded boundaries: strictly chained (each phase closes where
    // the next opens), all inside the run window, aggregation closing at
    // the same resolved instant run.json carries.
    let mut prev_end: Option<i64> = None;
    for name in ["claim", "assign", "processing", "aggregation"] {
        let phase = &status["phases"][name];
        let s = phase["started_at"].as_i64().unwrap();
        let c = phase["completed_at"].as_i64().unwrap();
        assert!(
            t0 <= s && s <= c && c <= t1,
            "phase {name}: [{s}, {c}] outside [{t0}, {t1}]"
        );
        if let Some(p) = prev_end {
            assert_eq!(s, p, "phase {name} not chained to its predecessor");
        }
        prev_end = Some(c);
        assert_eq!(
            phase["duration_s"].as_f64().unwrap(),
            (c - s) as f64 / 1e6,
            "phase {name}"
        );
    }
    assert_eq!(prev_end.unwrap(), completed);

    // Heartbeats: each write stamps its own updated_at; duration_s is the
    // run's elapsed time at that write.
    for w in 0..2 {
        let hb = read_json(&meta.join("heartbeats").join(format!("worker_{w}.json")));
        let updated = hb["updated_at"].as_i64().unwrap();
        assert!(
            in_window(updated),
            "worker {w} updated_at {updated} not in ({t0}, {t1}]"
        );
        assert_eq!(hb["started_at"], t0);
        assert_eq!(hb["duration_s"].as_f64().unwrap(), round1(updated - t0));
    }
    let fail = read_json(&meta.join("failures").join("worker_1.json"));
    assert!(in_window(fail["timestamp"].as_i64().unwrap()));
}

#[test]
fn source_trace_root_flows_from_config_to_artifacts() {
    let tmp = tempfile::tempdir().unwrap();
    let step = tmp.path().join("step_000000000106");
    write_trace(&step, 0);
    let out = tmp.path().join("out");

    // The artifacts record the source URI instead of the local copy.
    let uri = "s3://bucket/o/profiling/traces/step_000000000106";
    let mut cfg = base_cfg(step, out.clone(), 1);
    cfg.source_trace_root = Some(uri.into());
    run(&cfg).unwrap();

    let config = read_json(
        &out.join("tree")
            .join("test-run")
            .join("_meta")
            .join("config.json"),
    );
    assert_eq!(config["trace_root"], uri);
    let metadata = read_json(&out.join("tree").join("test-run").join("metadata.json"));
    assert_eq!(metadata["config"]["trace_folder_path"], uri);
}

fn status_path(out: &Path) -> PathBuf {
    out.join("tree")
        .join("test-run")
        .join("_meta")
        .join("status.json")
}

#[test]
fn status_phases_pinned_deterministic() {
    // Replay-pinned timestamps (explicit --started-at-us/--completed-at-us):
    // phases are a pure function of the bounds — the interval is attributed
    // to processing, zero-length claim/assign/aggregation at its edges — so
    // replayed runs produce byte-identical status.json.
    let tmp = tempfile::tempdir().unwrap();
    let step = tmp.path().join("step_000000000106");
    for r in [0, 1] {
        write_trace(&step, r);
    }
    let out_a = tmp.path().join("a");
    run(&base_cfg(step.clone(), out_a.clone(), 2)).unwrap();
    let out_b = tmp.path().join("b");
    run(&base_cfg(step, out_b.clone(), 2)).unwrap();
    let text = std::fs::read_to_string(status_path(&out_a)).unwrap();
    assert_eq!(text, std::fs::read_to_string(status_path(&out_b)).unwrap());

    let status: serde_json::Value = serde_json::from_str(&text).unwrap();
    let phases = status["phases"].as_object().unwrap();
    assert_eq!(phases.len(), 4);
    for (name, s, c) in [
        ("claim", T0_US, T0_US),
        ("assign", T0_US, T0_US),
        ("processing", T0_US, T1_US),
        ("aggregation", T1_US, T1_US),
    ] {
        assert_eq!(phases[name]["started_at"], s, "{name}");
        assert_eq!(phases[name]["completed_at"], c, "{name}");
        assert_eq!(phases[name]["status"], "completed", "{name}");
    }
    assert_eq!(phases["processing"]["duration_s"], 1.5);
    // Sorted key order (like every metadata doc) — readers key by name,
    // not position.
    let order: Vec<usize> = [
        "\"aggregation\"",
        "\"assign\"",
        "\"claim\"",
        "\"processing\"",
    ]
    .iter()
    .map(|k| text.find(*k).unwrap())
    .collect();
    assert!(order.windows(2).all(|w| w[0] < w[1]), "order: {order:?}");
}

#[test]
fn status_phases_live_monotonic() {
    // Live completion time (0 = write time): phases carry the recorded
    // claim → assign → processing → aggregation boundaries — monotonic,
    // inside the run's wall window, with real processing/aggregation spans.
    let now_us = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    };
    let tmp = tempfile::tempdir().unwrap();
    let step = tmp.path().join("step_000000000106");
    for r in [0, 1] {
        write_trace(&step, r);
    }
    let out = tmp.path().join("out");
    let mut cfg = base_cfg(step, out.clone(), 2);
    cfg.completed_at_us = 0;
    let before = now_us();
    run(&cfg).unwrap();
    let after = now_us();

    let status = read_json(&status_path(&out));
    let phases = status["phases"].as_object().unwrap();
    assert_eq!(phases.len(), 4);
    let mut prev = before;
    for name in ["claim", "assign", "processing", "aggregation"] {
        let s = phases[name]["started_at"].as_i64().unwrap();
        let c = phases[name]["completed_at"].as_i64().unwrap();
        assert!(prev <= s && s <= c, "{name}: {prev} <= {s} <= {c}");
        assert_eq!(phases[name]["status"], "completed", "{name}");
        prev = c;
    }
    assert!(prev <= after);
    // The rank loop and the collect+fold both take real time.
    assert!(phases["processing"]["duration_s"].as_f64().unwrap() > 0.0);
    assert!(phases["aggregation"]["duration_s"].as_f64().unwrap() > 0.0);
}

#[test]
fn capture_clock_origins_without_trace_artifacts() {
    let tmp = tempfile::tempdir().unwrap();
    let step = tmp.path().join("step_000000000177");
    std::fs::create_dir(&step).unwrap();
    for (rank, origin) in [
        (0, Some(1_753_000_000_000_000_123_u64)),
        (2, Some(1_753_000_000_000_500_987)),
        (3, None),
    ] {
        let trace = serde_json::json!({"baseTimeNanoseconds":origin,
            "distributedInfo":{"rank":rank,"world_size":4,"pg_config":pg_config()},"traceEvents":[
            {"ph":"X","cat":"kernel","name":"ncclKernel_SendRecv","ts":100,"dur":2,"args":{"stream":7}}
        ]});
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(trace.to_string().as_bytes()).unwrap();
        std::fs::write(
            step.join(format!("{rank:06}.1700000000000.pt.trace.json.gz")),
            enc.finish().unwrap(),
        )
        .unwrap();
    }
    let out = tmp.path().join("out");
    let summary = run(&base_cfg(step, out.clone(), 4)).unwrap();
    assert_eq!(
        (summary.processed, summary.skipped, summary.errors),
        (3, 1, 0)
    );
    assert!(!out.join("tree/test-run/traces").exists());
    let metadata = read_json(&out.join("tree/test-run/metadata.json"));
    let capture = &metadata["capture"];
    assert_eq!(capture["run_name"], "test-job");
    assert_eq!(capture["step"], 177);
    assert_eq!(
        capture["rank_clock_origins_ns"],
        serde_json::json!({
            "0":"1753000000000000123", "2":"1753000000000500987", "3":null
        })
    );
}

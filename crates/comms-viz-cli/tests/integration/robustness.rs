// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Property and edge-case tests around the snapshot suite: byte
//! determinism across runs and thread counts, the exact scope of the known
//! merge.rs instability, and the CLI's failure/degradation behavior on bad
//! inputs.

use crate::harness::Format;
use crate::harness::{self};

/// The pinned-stamp invocation must be byte-deterministic — same tree from
/// repeat runs AND across --threads (rank order is the reference ordering,
/// not completion order). cgp_json's merged traces are the known exception
/// (see Fixture::merged_stable) and are excluded for it only.
#[test]
fn analysis_is_deterministic_across_runs_and_threads() {
    for fx in harness::FIXTURES {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_a = tmp.path().join("a");
        let out_b = tmp.path().join("b");
        harness::analyze(fx, &out_a, tmp.path(), &["--threads", "2"]);
        harness::analyze(fx, &out_b, tmp.path(), &["--threads", "5"]);
        harness::assert_trees_identical(&out_a, &out_b, |rel| {
            !fx.merged_stable && rel.starts_with("tree/") && rel.contains("/traces/")
        });
    }
}

/// Documents the exact blast radius of the open merge.rs equal-ts
/// hash-order issue: cgp_json's merged-trace artifacts may flip between
/// runs, but every analysis artifact stays byte-identical. If this test
/// starts failing on the analysis side the instability has spread; if
/// merged traces become stable, flip Fixture::merged_stable for cgp_json
/// and let the manifest snapshot pin their bytes too.
#[test]
fn cgp_instability_is_confined_to_merged_traces() {
    let fx = harness::fixture("cgp_json");
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_a = tmp.path().join("a");
    let out_b = tmp.path().join("b");
    harness::analyze(fx, &out_a, tmp.path(), &[]);
    harness::analyze(fx, &out_b, tmp.path(), &[]);
    harness::assert_trees_identical(&out_a, &out_b, |rel| {
        rel.starts_with("tree/") && rel.contains("/traces/")
    });
}

/// A missing or empty trace root is a clean error, not a hang or a panic.
#[test]
fn missing_trace_root_fails_cleanly() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = harness::comms_viz(
        tmp.path(),
        &[
            "--trace-root",
            &tmp.path().join("does-not-exist").display().to_string(),
            "--out",
            &tmp.path().join("out").display().to_string(),
        ],
    );
    assert!(!out.status.success(), "expected failure on a missing root");
    let stderr = harness::stderr_str(&out);
    assert!(
        stderr.contains("trace files found under"),
        "unexpected error shape:\n{stderr}"
    );
}

/// Forcing the wrong --trace-format finds no traces and fails the run.
#[test]
fn wrong_trace_format_fails() {
    let fx = harness::fixture("synth_json");
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = harness::comms_viz(
        tmp.path(),
        &[
            "--trace-root",
            &harness::fixture_root(fx).display().to_string(),
            "--out",
            &tmp.path().join("out").display().to_string(),
            "--trace-format",
            "pftrace",
        ],
    );
    assert!(
        !out.status.success(),
        "expected failure on the wrong format"
    );
    let stderr = harness::stderr_str(&out);
    assert!(
        stderr.contains("no pftrace trace files found under"),
        "unexpected error shape:\n{stderr}"
    );
}

/// A dir holding both formats needs an explicit --trace-format.
#[test]
fn mixed_formats_need_trace_format() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("mixed");
    std::fs::create_dir(&root).expect("mkdir");
    for name in ["synth_json", "synth_pftrace"] {
        let src = harness::fixture_root(harness::fixture(name));
        let first = std::fs::read_dir(&src)
            .expect("fixture dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .min()
            .expect("fixture file");
        std::fs::copy(&first, root.join(first.file_name().expect("name"))).expect("copy");
    }
    let out = harness::comms_viz(
        tmp.path(),
        &[
            "--trace-root",
            &root.display().to_string(),
            "--out",
            &tmp.path().join("out").display().to_string(),
        ],
    );
    assert!(!out.status.success(), "expected failure on mixed formats");
    let stderr = harness::stderr_str(&out);
    assert!(
        stderr.contains("pass --trace-format json|pftrace"),
        "unexpected error shape:\n{stderr}"
    );
}

/// `--trace-format auto` must land on the majority format for both
/// vendored layouts (the fixtures are single-format dirs, so auto must
/// match the explicit flag byte-for-byte).
#[test]
fn auto_format_detection_matches_explicit() {
    for name in ["synth_json", "synth_pftrace"] {
        let fx = harness::fixture(name);
        let explicit = match fx.format {
            Format::Json => "json",
            Format::Pftrace => "pftrace",
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_auto = tmp.path().join("auto");
        let out_explicit = tmp.path().join("explicit");
        harness::analyze(fx, &out_auto, tmp.path(), &[]);
        harness::analyze(fx, &out_explicit, tmp.path(), &["--trace-format", explicit]);
        harness::assert_trees_identical(&out_auto, &out_explicit, |_| false);
    }
}

/// The local CLI requires an output directory and rejects positional jobs.
#[test]
fn cli_grammar_guards() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // No trace root, output tree, or report destination.
    let out = harness::comms_viz::<&str>(tmp.path(), &[]);
    assert!(!out.status.success());
    assert!(
        harness::stderr_str(&out).contains("--out is required"),
        "usage bail changed:\n{}",
        harness::stderr_str(&out)
    );
    // Remote job names are no longer part of the CLI grammar.
    let out = harness::comms_viz(tmp.path(), &["some-job", "--trace-root", "/tmp/x"]);
    assert!(!out.status.success());
    assert!(
        harness::stderr_str(&out).contains("unexpected argument"),
        "conflict diagnostics changed:\n{}",
        harness::stderr_str(&out)
    );
    // --report without --out falls into the same usage bail.
    let out = harness::comms_viz(tmp.path(), &["--report", "/tmp/x"]);
    assert!(!out.status.success());
    assert!(harness::stderr_str(&out).contains("--out is required"));
}

/// Re-analyzing into the same `--out` never mixes runs: the default id
/// moves to the next free name, an explicit id needs `--overwrite`.
#[test]
fn existing_run_id_is_not_reused() {
    let fx = harness::fixture("synth_json");
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("out");
    harness::analyze_default(fx, &out, tmp.path(), &[]);
    harness::analyze_default(fx, &out, tmp.path(), &[]);
    assert!(out.join("tree/local-run-2").is_dir());

    let rerun = harness::comms_viz(
        tmp.path(),
        &[
            "--trace-root",
            &harness::fixture_root(fx).display().to_string(),
            "--out",
            &out.display().to_string(),
            "--run-id",
            "local-run",
        ],
    );
    assert!(!rerun.status.success());
    assert!(harness::stderr_str(&rerun).contains("pass --overwrite"));

    harness::analyze_default(
        fx,
        &out,
        tmp.path(),
        &["--run-id", "local-run", "--overwrite"],
    );
    assert!(!out.join("tree/local-run-3").exists());
}

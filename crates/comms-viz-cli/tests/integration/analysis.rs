// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Analysis-output snapshots: one full direct-mode run per vendored
//! fixture × format, with the whole artifact tree pinned.
//!
//! Per fixture:
//!  - `<name>__run_summary`      final stderr counters (ranks, PGs)
//!  - `<name>__artifacts`        sorted path+size+sha256 manifest (bytes)
//!  - `<name>__breakdown`        the rank breakdown table (parquet, rendered
//!    as its page-facing CSV text)
//!  - `<name>__per_pg_members`   every per-group member payload, decoded
//!    (page-visible view), concatenated
//!  - `<name>__per_pg_parquet`   schema/shape of every per-PG parquet
//!  - `<name>__flat_docs`        the 10 flat analyzer docs
//!  - `<name>__meta`             MANIFEST + _meta docs (rev-filtered)
//!  - `<name>__merged_trace_r0`  decoded rank-0 merged-trace summary
//!    (JSON-format fixtures with stable merge)

use std::path::Path;

use crate::harness::Fixture;
use crate::harness::Format;
use crate::harness::{self};

fn snapshot_fixture(fx: &Fixture) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_dir = tmp.path().join("out");
    let run = harness::analyze(fx, &out_dir, tmp.path(), &[]);

    let settings = harness::settings();
    settings.bind(|| {
        insta::assert_snapshot!(
            format!("{}__run_summary", fx.name),
            harness::run_summary(&run)
        );
        insta::assert_snapshot!(
            format!("{}__artifacts", fx.name),
            harness::artifact_manifest(&out_dir, fx.merged_stable)
        );

        let pipeline_a = out_dir.join("tree/local-run/results/pipeline_a");
        insta::assert_snapshot!(
            format!("{}__breakdown", fx.name),
            harness::parquet_csv(&pipeline_a.join("breakdown.parquet"))
        );
        insta::assert_snapshot!(
            format!("{}__per_pg_members", fx.name),
            harness::concat_members(&pipeline_a.join("per_pg"))
        );
        insta::assert_snapshot!(
            format!("{}__per_pg_parquet", fx.name),
            harness::parquet_schemas(&pipeline_a.join("per_pg"))
        );
        insta::assert_snapshot!(
            format!("{}__flat_docs", fx.name),
            harness::concat_dir(&out_dir.join("flat"), |n| n.ends_with(".json"))
        );

        let meta = out_dir.join("tree/local-run/_meta");
        let mut meta_dump = harness::concat_dir(&meta, |n| n == "MANIFEST" || n.ends_with(".json"));
        meta_dump.push_str(&harness::concat_dir(&meta.join("heartbeats"), |_| true));
        insta::assert_snapshot!(format!("{}__meta", fx.name), meta_dump);

        if fx.format == Format::Json && fx.merged_stable {
            insta::assert_snapshot!(
                format!("{}__merged_trace_r0", fx.name),
                harness::merged_trace_summary(
                    &out_dir.join("tree/local-run/traces/000000.merged.pt.trace.json.gz")
                )
            );
        }
    });
}

#[test]
fn synth_json_analysis() {
    snapshot_fixture(harness::fixture("synth_json"));
}

#[test]
fn synth_pftrace_analysis() {
    snapshot_fixture(harness::fixture("synth_pftrace"));
}

#[test]
fn inline_json_analysis() {
    snapshot_fixture(harness::fixture("inline_json"));
}

#[test]
fn inline_pftrace_analysis() {
    snapshot_fixture(harness::fixture("inline_pftrace"));
}

#[test]
fn cgp_json_analysis() {
    snapshot_fixture(harness::fixture("cgp_json"));
}

#[test]
fn cgp_norec_json_analysis() {
    snapshot_fixture(harness::fixture("cgp_norec_json"));
}

#[test]
fn cgp_pftrace_analysis() {
    snapshot_fixture(harness::fixture("cgp_pftrace"));
}

/// The JSON fixture and its converter-derived pftrace twin must agree on
/// the process-group structure they discover (values differ slightly —
/// the pftrace representation stores integer-ns timestamps and drops
/// non-analysis event classes — but the PG axes and artifact shape are
/// format-invariant).
#[test]
fn format_twins_agree_on_pg_structure() {
    for (a, b) in [
        ("synth_json", "synth_pftrace"),
        ("inline_json", "inline_pftrace"),
        ("cgp_json", "cgp_pftrace"),
    ] {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_a = tmp.path().join("a");
        let out_b = tmp.path().join("b");
        harness::analyze(harness::fixture(a), &out_a, tmp.path(), &[]);
        harness::analyze(harness::fixture(b), &out_b, tmp.path(), &[]);
        let listing = |out: &Path| -> Vec<String> {
            let mut v: Vec<String> =
                std::fs::read_dir(out.join("tree/local-run/results/pipeline_a/per_pg"))
                    .expect("per_pg")
                    .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
                    .collect();
            v.sort();
            v
        };
        assert_eq!(
            listing(&out_a),
            listing(&out_b),
            "per-PG artifact sets diverge between {a} and {b}"
        );
    }
}

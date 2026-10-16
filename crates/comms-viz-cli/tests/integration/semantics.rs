// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Behavior checks on `testdata/traces/mesh`, a 16-rank pp4 x dp_shard2 x
//! tp2 (ep2) step from gen_mesh_traces.py with known faults (its
//! `ground_truth.json`): rank 5 straggles, node 3 (ranks 12-15) has slow
//! links and low SM Active, rank 2 lost the PG metadata of half its NCCL
//! kernels, and expert compute launches unannotated HybridEP-style kernels.
//! Unlike the snapshot suites these assert what the reports conclude.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;

use serde_json::Value;

use crate::harness;

const ROOT: &str = "mesh/step_000000000010";

/// Analyze the mesh fixture with both reports; returns the out dir and the
/// structured summary.
fn analyze(tmp: &Path) -> (PathBuf, Value, String) {
    let out = tmp.join("out");
    let run = harness::comms_viz(
        tmp,
        &[
            "--trace-root",
            &harness::traces_dir().join(ROOT).display().to_string(),
            "--out",
            &out.display().to_string(),
            "--run-name",
            "mesh",
            "--threads",
            "4",
        ],
    );
    assert!(run.status.success(), "{}", harness::stderr_str(&run));
    let dir = std::fs::read_dir(out.join("structured_report"))
        .expect("structured report dir")
        .next()
        .expect("one report")
        .expect("entry")
        .path();
    let summary = serde_json::from_str(
        &std::fs::read_to_string(dir.join("summary.json")).expect("summary.json"),
    )
    .expect("summary parses");
    (out, summary, harness::stderr_str(&run))
}

fn per_pg_dir(out: &Path) -> PathBuf {
    out.join("tree/local-run/results/pipeline_a/per_pg")
}

fn axis<'a>(summary: &'a Value, name: &str) -> &'a Value {
    summary["axes"]
        .as_array()
        .expect("axes")
        .iter()
        .find(|a| a["axis"] == name)
        .unwrap_or_else(|| panic!("no axis {name}"))
}

#[test]
fn groups_follow_the_device_mesh() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (out, summary, _) = analyze(tmp.path());
    let pg_map: Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("tree/local-run/pg_map.json")).expect("pg_map"),
    )
    .expect("pg_map parses");
    let ranks = |axis: &str| -> BTreeSet<Vec<i64>> {
        pg_map[axis]["pg_ranks_map"]
            .as_array()
            .unwrap_or_else(|| panic!("no {axis} in pg_map"))
            .iter()
            .map(|g| {
                g.as_array()
                    .unwrap()
                    .iter()
                    .map(|r| r.as_i64().unwrap())
                    .collect()
            })
            .collect()
    };
    let pairs = |stride: i64, step: i64| -> BTreeSet<Vec<i64>> {
        (0..16)
            .filter(|r| r % step < stride)
            .map(|r| vec![r, r + stride])
            .collect()
    };
    assert_eq!(ranks("tp"), pairs(1, 2));
    assert_eq!(ranks("dp_shard"), pairs(2, 4));
    assert_eq!(ranks("ep"), pairs(2, 4), "ep borrows dp_shard at cp=1");
    assert_eq!(
        ranks("pp"),
        (0..4).map(|r| vec![r, r + 4, r + 8, r + 12]).collect()
    );
    // split_group gives every stage pair of a round one name; each pair is
    // its own group, named by its lowest rank.
    let mut stems: BTreeMap<String, usize> = BTreeMap::new();
    for entry in std::fs::read_dir(per_pg_dir(&out)).expect("per_pg") {
        let name = entry
            .expect("entry")
            .file_name()
            .to_string_lossy()
            .into_owned();
        if let Some(stem) = name.strip_suffix(".dpg") {
            let axis = stem.rsplit_once('_').map_or(stem, |(a, _)| a);
            *stems.entry(axis.to_string()).or_default() += 1;
        }
    }
    let truth: Value = serde_json::from_str(
        &std::fs::read_to_string(harness::traces_dir().join("mesh/ground_truth.json"))
            .expect("ground truth"),
    )
    .expect("ground truth parses");
    for (key, axis_name) in [
        ("tp", "tp"),
        ("dp_shard", "dp_shard"),
        ("ep", "ep"),
        ("pp", "pp"),
        ("pp_p2p_0", "pp_p2p_round_0"),
        ("pp_p2p_1", "pp_p2p_round_1"),
        ("pp_p2p_2", "pp_p2p_round_2"),
        ("pp_p2p_3", "pp_p2p_round_3"),
    ] {
        assert_eq!(
            axis(&summary, axis_name)["groups"],
            truth["groups"][key],
            "{axis_name} group count"
        );
    }
}

#[test]
fn summary_finds_the_injected_faults() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (_, s, stderr) = analyze(tmp.path());

    let top = &s["stragglers"]["top"][0];
    assert_eq!(top["rank"], 5, "{:#}", s["stragglers"]);
    assert!(top["z"].as_f64().unwrap() >= 4.0, "{top}");
    assert_eq!(s["stragglers"]["suspects"], 1);

    let slow: BTreeSet<i64> = s["ranks_in_slow_groups"]["top"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["rank"].as_i64().unwrap())
        .collect();
    assert_eq!(slow, (12..16).collect(), "{:#}", s["ranks_in_slow_groups"]);
    assert!(
        s["host_outliers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|h| { h["host"] == "node00003" && h["metric"] == "SM Active (%) mean low" }),
        "{:#}",
        s["host_outliers"]
    );

    let q = &s["data_quality"];
    assert_eq!(q["ranks_losing_pg_metadata"], 1);
    assert_eq!(q["worst_ranks_losing_pg_metadata"][0]["rank"], 2);
    assert!(
        q["other_comm_kernels_without_pg_metadata"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        stderr.contains("warning: 1 of 16 ranks have NCCL kernels without process-group metadata")
    );
    assert!(s["warnings"][0].as_str().unwrap().contains("ranks 2)"));
    // The unannotated HybridEP-style kernels are on no group.
    assert!(
        !s["axes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["axis"] == "other")
    );
}

#[test]
fn p2p_and_unsequenced_calls_stay_out_of_cross_rank_stats() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (_, s, _) = analyze(tmp.path());
    for round in 0..4 {
        let a = axis(&s, &format!("pp_p2p_round_{round}"));
        assert_eq!(a["matched_collectives"], 0, "{a:#}");
        assert!(a["unmatchable_kernels"]["p2p"].as_u64().unwrap() > 0);
        assert!(a["start_spread_us"].is_null());
    }
    assert!(axis(&s, "tp")["matched_collectives"].as_u64().unwrap() > 0);
}

/// The page's JavaScript matching (`<cohort-logic>` in static_page.html, run
/// under node) and the structured report's Rust port agree on every group:
/// same matched collectives per axis, same last arrivals per rank.
#[test]
fn page_and_report_match_the_same_collectives() {
    if std::process::Command::new("node")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping: node not found");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let (out, s, _) = analyze(tmp.path());

    let page = std::fs::read_to_string(
        harness::workspace_root().join("crates/comms-viz-cli/src/static_page.html"),
    )
    .expect("page source");
    let region = |tag: &str| -> &str {
        let open = format!("// <{tag}>");
        let close = format!("// </{tag}>");
        let start = page.find(&open).unwrap_or_else(|| panic!("no {open}"));
        &page[start..page.find(&close).unwrap_or_else(|| panic!("no {close}"))]
    };
    let axes: Vec<String> = s["axes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["axis"].as_str().unwrap().to_string())
        .collect();
    let mut groups = Vec::new();
    for entry in std::fs::read_dir(per_pg_dir(&out)).expect("per_pg") {
        let path = entry.expect("entry").path();
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        if path.extension().is_none_or(|e| e != "dpg") || stem == "diagnostics_0" {
            continue;
        }
        let axis = axes
            .iter()
            .filter(|a| stem.starts_with(&format!("{a}_")))
            .max_by_key(|a| a.len())
            .expect("stem has an axis")
            .clone();
        let (names, cols) =
            comms_viz_core::dpg::decode_member(&std::fs::read(&path).expect("member"));
        groups.push(serde_json::json!({"axis": axis, "names": names, "cols": cols}));
    }
    let input = tmp.path().join("groups.json");
    std::fs::write(&input, serde_json::to_string(&groups).unwrap()).unwrap();
    let script = format!(
        r#"{events}
{cohorts}
const groups = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const matched = {{}}, last = {{}};
for (const g of groups) {{
  const evs = eventsFromTable(g.names, (c, r) => g.cols[c][r], g.cols[0].length);
  for (const c of cohortsOf(evs).list.filter(cohortComplete)) {{
    matched[g.axis] = (matched[g.axis] || 0) + 1;
    last[c.lat.rank] = (last[c.lat.rank] || 0) + 1;
  }}
}}
console.log(JSON.stringify({{ matched, last }}));
"#,
        events = region("events-from-table"),
        cohorts = region("cohort-logic"),
    );
    let script_path = tmp.path().join("cohorts.js");
    std::fs::write(&script_path, script).unwrap();
    let run = std::process::Command::new("node")
        .arg(&script_path)
        .arg(&input)
        .output()
        .expect("run node");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let js: Value = serde_json::from_slice(&run.stdout).expect("node output");

    for a in s["axes"].as_array().unwrap() {
        let name = a["axis"].as_str().unwrap();
        assert_eq!(
            js["matched"][name].as_u64().unwrap_or(0),
            a["matched_collectives"].as_u64().unwrap(),
            "matched collectives on {name}"
        );
    }
    let rust_last: BTreeMap<String, u64> = s["stragglers"]["top"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["last"].as_u64().unwrap() > 0)
        .map(|r| (r["rank"].to_string(), r["last"].as_u64().unwrap()))
        .collect();
    let js_last: BTreeMap<String, u64> = js["last"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_u64().unwrap()))
        .collect();
    assert_eq!(js_last, rust_last, "last arrivals per rank");
}

/// A rerun replaces the structured report instead of leaving files from the
/// earlier version behind.
#[test]
fn structured_report_rewrite_leaves_no_stale_files() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (out, _, _) = analyze(tmp.path());
    let report = std::fs::read_dir(out.join("structured_report"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::write(report.join("stale.csv"), "old").unwrap();
    let rerun = harness::comms_viz(tmp.path(), &["--out", &out.display().to_string()]);
    assert!(rerun.status.success(), "{}", harness::stderr_str(&rerun));
    let mut files: Vec<String> = std::fs::read_dir(&report)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    assert_eq!(files, ["README.md", "summary.json"]);
}

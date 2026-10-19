// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Converter (`--convert-pftrace`) contract tests: the vendored pftrace
//! twins in testdata/traces must stay exactly what the current converter
//! produces from the vendored JSON fixtures, and conversion itself must be
//! deterministic. If a converter change is intentional, regenerate the
//! twins (commands in testdata/traces/README.md) and re-review the
//! affected snapshots.

use std::path::Path;

use crate::harness;

/// (json fixture dir, vendored twin dir, converter annotation flag) —
/// matching the twins' recorded generation commands.
const TWINS: &[(&str, &str, &str)] = &[
    (
        "synth/step_000000000106",
        "synth-pftrace/step_000000000106",
        "--annotations=disabled",
    ),
    (
        "inline/step_000000000106",
        "inline-pftrace/step_000000000106",
        "--annotations=disabled",
    ),
    (
        "cgp/step_000000000106",
        "cgp-pftrace/step_000000000106",
        "--annotations=pkl",
    ),
];

fn convert(json_rel: &str, flag: &str, dest: &Path, home: &Path) {
    let root = harness::traces_dir().join(json_rel);
    let out = harness::comms_viz(
        home,
        &[
            "--trace-root",
            &root.display().to_string(),
            "--convert-pftrace",
            &dest.display().to_string(),
            flag,
            "--threads",
            "4",
        ],
    );
    assert!(
        out.status.success(),
        "--convert-pftrace failed for {json_rel}:\n{}",
        harness::stderr_str(&out)
    );
}

fn sorted_files(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
fn vendored_pftrace_twins_match_current_converter() {
    for &(json_rel, twin_rel, flag) in TWINS {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dest = tmp.path().join("converted");
        convert(json_rel, flag, &dest, tmp.path());

        let vendored = harness::traces_dir().join(twin_rel);
        assert_eq!(
            sorted_files(&dest),
            sorted_files(&vendored),
            "converter output file set diverged from the vendored twins at \
             testdata/traces/{twin_rel}"
        );
        for name in sorted_files(&dest) {
            let fresh = std::fs::read(dest.join(&name)).expect("fresh twin");
            let pinned = std::fs::read(vendored.join(&name)).expect("vendored twin");
            assert!(
                fresh == pinned,
                "converter output for {name} no longer matches the vendored twin \
                 testdata/traces/{twin_rel}/{name} ({}B sha {} vs {}B sha {}).\n\
                 If the converter change is intentional, regenerate the twins per \
                 testdata/traces/README.md and re-review the pftrace snapshots.",
                fresh.len(),
                harness::sha256_16(&fresh),
                pinned.len(),
                harness::sha256_16(&pinned),
            );
        }
    }
}

#[test]
fn converter_is_deterministic() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (json_rel, _, flag) = TWINS[0];
    let a = tmp.path().join("a");
    let b = tmp.path().join("b");
    convert(json_rel, flag, &a, tmp.path());
    convert(json_rel, flag, &b, tmp.path());
    harness::assert_trees_identical(&a, &b, |_| false);
}

/// The converter's row accounting (its faithfulness self-report) is part
/// of the contract; pin it for the annotated (cgp) conversion.
#[test]
fn converter_row_accounting_snapshot() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dest = tmp.path().join("converted");
    let root = harness::traces_dir().join("cgp/step_000000000106");
    let out = harness::comms_viz(
        tmp.path(),
        &[
            "--trace-root",
            &root.display().to_string(),
            "--convert-pftrace",
            &dest.display().to_string(),
            "--annotations=pkl",
            "--threads",
            "4",
        ],
    );
    assert!(out.status.success(), "{}", harness::stderr_str(&out));
    let stderr = harness::stderr_str(&out);
    let rows = stderr
        .lines()
        .find(|l| l.starts_with("comms_viz-convert: rows "))
        .unwrap_or_else(|| panic!("no row accounting in stderr:\n{stderr}"));
    harness::settings().bind(|| {
        insta::assert_snapshot!("cgp_convert__row_accounting", rows);
    });
}

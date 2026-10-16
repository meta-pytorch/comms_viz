// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Static report page snapshots, driven end-to-end through the binary:
//! direct-mode analysis produces the out tree, then `--out … --report …`
//! builds the page HTML and `.pgdata` sidecar locally.
//!
//! Snapshots pin the parsed payload island, which is where run content
//! lives; template changes are left to review rather than a byte snapshot.

use std::path::Path;
use std::path::PathBuf;

use crate::harness;

/// The rendered page stem, discovered from the dest dir's sole `.html`:
/// the name carries the run's job/step (`publish::page_stem`), which the
/// fixture docs decide — tests assert against the discovery, not a
/// hardcoded spelling.
fn rendered_stem(dest: &Path) -> String {
    let mut htmls: Vec<String> = std::fs::read_dir(dest)
        .expect("dest dir")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.file_name()
                .to_string_lossy()
                .strip_suffix(".html")
                .map(str::to_string)
        })
        .collect();
    assert_eq!(
        htmls.len(),
        1,
        "expected exactly one rendered page: {htmls:?}"
    );
    htmls.pop().expect("one stem")
}

/// Analyze `fx` then render its page; returns (dest dir, html).
fn render(fx: &harness::Fixture, tmp: &Path, extra: &[&str]) -> (PathBuf, String) {
    let out_dir = tmp.join(format!("out-{}", fx.name));
    harness::analyze(fx, &out_dir, tmp, extra);
    let dest = tmp.join(format!("page-{}", fx.name));
    let out = harness::comms_viz(
        tmp,
        &[
            "--out",
            &out_dir.display().to_string(),
            "--report",
            &dest.display().to_string(),
        ],
    );
    assert!(
        out.status.success(),
        "--report failed for {}:\n{}",
        fx.name,
        harness::stderr_str(&out)
    );
    let stem = rendered_stem(&dest);
    // Local renders keep the run id verbatim and carry the fixture's
    // job/step in the name.
    assert!(
        stem.starts_with("comms_viz-local-run"),
        "unexpected page stem {stem}"
    );
    let html =
        std::fs::read_to_string(dest.join(format!("{stem}.html"))).expect("rendered page html");
    (dest, html)
}

/// The `<script id="…" type="application/json">` island bodies the page
/// boots from.
fn island(html: &str, id: &str) -> String {
    let tag = format!(r#"id="{id}""#);
    let start = html.find(&tag).unwrap_or_else(|| panic!("no {id} island"));
    let body = &html[start + html[start..].find('>').expect("tag close") + 1..];
    body[..body.find("</script>").expect("island close")].to_string()
}

fn payload(html: &str) -> serde_json::Value {
    serde_json::from_str(&island(html, "comms_viz-data")).expect("payload island parses")
}

/// Invariants every rendered page must satisfy, whatever the fixture.
fn assert_page_wellformed(dest: &Path, html: &str, p: &serde_json::Value) {
    for placeholder in [
        "__COMMS_VIZ_DATA__",
        "__COMMS_VIZ_TITLE__",
        "__COMMS_VIZ_PGMAP__",
    ] {
        assert!(!html.contains(placeholder), "unreplaced {placeholder}");
    }
    // Script-safety: no literal "<" may survive inside the payload island.
    assert!(!island(html, "comms_viz-data").contains('<'));
    assert_eq!(p["run_id"], "local-run", "run id must be kept verbatim");
    // Tiny fixtures fit one sidecar object.
    assert!(p["data_shards"].is_null());
    // data_name is null exactly when the sidecar is empty (a run whose
    // page artifacts all inline). The sink still creates the
    // shard-0 file up front, so "empty" on disk means a 0-byte .pgdata.
    let data_name = format!("{}.pgdata", rendered_stem(dest));
    let pgdata = dest.join(&data_name);
    match &p["data_name"] {
        serde_json::Value::Null => assert_eq!(
            pgdata.metadata().map(|m| m.len()).unwrap_or(0),
            0,
            "empty sidecar advertised, but the .pgdata file has bytes"
        ),
        name => {
            assert_eq!(name, &data_name);
            assert!(
                pgdata.metadata().map(|m| m.len()).unwrap_or(0) > 0,
                "payload names a sidecar that wasn't written"
            );
        }
    }
    // Cross-pipeline agreement: every sidecar group stem must resolve in
    // the pg_map island (same axis, gid among its pg_ids).
    let pgmap: serde_json::Value =
        serde_json::from_str(&island(html, "comms_viz-pgmap")).expect("pgmap island parses");
    assert!(
        pgmap.get("t").is_none(),
        "fixture pgmap island crossed GZ_THRESHOLD; teach this check gz64"
    );
    let key_of_alias = |alias: &str| -> Option<&str> {
        p["pg_summary"]
            .as_array()?
            .iter()
            .find(|s| s["alias"] == alias)
            .and_then(|s| s["key"].as_str())
    };
    for entry in p["pg_index"].as_array().into_iter().flatten() {
        let prefix = entry[0].as_str().expect("pg_index prefix");
        // Per-rank diagnostics ride in the sidecar but are not a PG axis.
        if prefix == "diagnostics" {
            continue;
        }
        for pair in entry[1].as_array().expect("pg_index pairs").chunks(2) {
            let gid = match &pair[0] {
                serde_json::Value::String(id) => id.clone(),
                id => id.to_string(),
            };
            let key = key_of_alias(prefix)
                .unwrap_or_else(|| panic!("sidecar axis {prefix} missing from pg_summary"));
            let ids = pgmap[key]["pg_ids"].as_array().unwrap_or_else(|| {
                panic!("sidecar axis {prefix} ({key}) missing from the pg_map island")
            });
            assert!(
                ids.iter().any(|id| id.as_str() == Some(gid.as_str())),
                "sidecar stem {prefix}_{gid} not in pg_map {key} pg_ids {ids:?}"
            );
        }
    }
}

#[test]
fn synth_page_renders_and_snapshots() {
    let fx = harness::fixture("synth_json");
    let tmp = tempfile::tempdir().expect("tempdir");
    let (dest, html) = render(fx, tmp.path(), &[]);
    let p = payload(&html);
    assert_page_wellformed(&dest, &html, &p);

    let settings = harness::settings();
    settings.bind(|| {
        // The payload island again, pretty-printed — the reviewable form.
        insta::assert_snapshot!(
            "synth_json__page_payload",
            serde_json::to_string_pretty(&p).expect("payload pretty")
        );
        // The lazy pg_map island (raw JSON at fixture scale).
        insta::assert_snapshot!(
            "synth_json__page_pgmap",
            serde_json::to_string_pretty(
                &serde_json::from_str::<serde_json::Value>(&island(&html, "comms_viz-pgmap"))
                    .expect("pgmap island parses")
            )
            .expect("pgmap pretty")
        );
    });
}

#[test]
fn cgp_pftrace_page_payload_snapshots() {
    let fx = harness::fixture("cgp_pftrace");
    let tmp = tempfile::tempdir().expect("tempdir");
    let (dest, html) = render(fx, tmp.path(), &[]);
    let p = payload(&html);
    assert_page_wellformed(&dest, &html, &p);
    harness::settings().bind(|| {
        insta::assert_snapshot!(
            "cgp_pftrace__page_payload",
            serde_json::to_string_pretty(&p).expect("payload pretty")
        );
    });
}

/// Capture metadata survives into the page payload.
#[test]
fn inline_page_keeps_capture_metadata() {
    let fx = harness::fixture("inline_json");
    let tmp = tempfile::tempdir().expect("tempdir");
    let (dest, html) = render(fx, tmp.path(), &[]);
    let p = payload(&html);
    assert_page_wellformed(&dest, &html, &p);
    assert_eq!(p["metadata"]["capture"]["step"], 106);
    assert_eq!(
        p["metadata"]["capture"]["rank_clock_origins_ns"]
            .as_object()
            .unwrap()
            .len(),
        4
    );
    harness::settings().bind(|| {
        insta::assert_snapshot!(
            "inline_json__page_payload",
            serde_json::to_string_pretty(&p).expect("payload pretty")
        );
    });
}

/// Without `--write-repaired-traces` no traces are written and the page
/// still renders.
#[test]
fn page_renders_without_repaired_traces() {
    let fx = harness::fixture("inline_json");
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_dir = tmp.path().join("out");
    harness::analyze_default(fx, &out_dir, tmp.path(), &[]);
    assert!(
        !out_dir.join("tree/local-run/traces").exists(),
        "traces are written only with --write-repaired-traces"
    );
    let dest = tmp.path().join("page");
    let out = harness::comms_viz(
        tmp.path(),
        &[
            "--out",
            &out_dir.display().to_string(),
            "--report",
            &dest.display().to_string(),
        ],
    );
    assert!(out.status.success(), "{}", harness::stderr_str(&out));
    let html =
        std::fs::read_to_string(dest.join(format!("{}.html", rendered_stem(&dest)))).expect("html");
    let p = payload(&html);
    assert_page_wellformed(&dest, &html, &p);
}

/// Rendering twice from the same tree produces byte-identical artifacts
/// modulo the generated_at_s stamp (redacted before comparing).
#[test]
fn page_render_is_deterministic() {
    let fx = harness::fixture("synth_json");
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_dir = tmp.path().join("out");
    harness::analyze(fx, &out_dir, tmp.path(), &[]);
    let mut pages = Vec::new();
    let re = regex::Regex::new(r#""generated_at_s":\s*\d+"#).expect("re");
    for i in 0..2 {
        let dest = tmp.path().join(format!("page-{i}"));
        let out = harness::comms_viz(
            tmp.path(),
            &[
                "--out",
                &out_dir.display().to_string(),
                "--report",
                &dest.display().to_string(),
            ],
        );
        assert!(out.status.success(), "{}", harness::stderr_str(&out));
        let stem = rendered_stem(&dest);
        let html = std::fs::read_to_string(dest.join(format!("{stem}.html"))).expect("html");
        pages.push((
            re.replace_all(&html, r#""generated_at_s":0"#).into_owned(),
            std::fs::read(dest.join(format!("{stem}.pgdata"))).expect("pgdata"),
        ));
    }
    assert_eq!(pages[0].0, pages[1].0, "page html not deterministic");
    assert_eq!(pages[0].1, pages[1].1, "pgdata sidecar not deterministic");
}

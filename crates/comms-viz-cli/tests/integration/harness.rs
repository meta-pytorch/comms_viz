// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Shared plumbing for the integration suite: the fixture table,
//! hermetic binary invocation, and the snapshot builders (artifact
//! manifest, merged-trace summary, parquet schema dump).

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Output;

use sha2::Digest;
use sha2::Sha256;

/// Pinned metadata stamps: with both set, every emitted doc (run/status/
/// heartbeats/flat) is byte-deterministic — verified for all six configs by
/// robustness::analysis_is_deterministic_across_runs_and_threads.
pub const STARTED_AT_US: &str = "1700000000000000";
pub const COMPLETED_AT_US: &str = "1700000001000000";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Json,
    Pftrace,
}

pub struct Fixture {
    /// Snapshot-name prefix.
    pub name: &'static str,
    /// Step dir under testdata/traces (the --trace-root).
    pub root: &'static str,
    pub world: &'static str,
    /// The battery-recorded analysis args for this fixture family.
    pub extra: &'static [&'static str],
    pub format: Format,
    /// false only for cgp_json: the CUDA-graph recording merge iterates
    /// stream HashMaps, so equal-ts kernels from different streams keep
    /// hash-order-dependent relative order and the merged-trace artifact
    /// bytes flip between runs (known open issue in comms-viz-core/src/
    /// merge.rs). Analysis artifacts stay byte-stable — only
    /// tree/<run>/traces/* is affected, and the manifest snapshot marks
    /// those entries instead of hashing them.
    pub merged_stable: bool,
}

/// One entry per vendored fixture × format. The pftrace
/// twins run with the same knobs except cgp's, where the converter already
/// baked the pkl annotations inline (the pftrace representation is
/// self-annotated, so no side channel exists to configure).
pub const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "synth_json",
        root: "synth/step_000000000106",
        world: "8",
        extra: &["--annotations=disabled"],
        format: Format::Json,
        merged_stable: true,
    },
    Fixture {
        name: "synth_pftrace",
        root: "synth-pftrace/step_000000000106",
        world: "8",
        extra: &["--annotations=disabled"],
        format: Format::Pftrace,
        merged_stable: true,
    },
    Fixture {
        name: "inline_json",
        root: "inline/step_000000000106",
        world: "4",
        extra: &["--annotations=disabled"],
        format: Format::Json,
        merged_stable: true,
    },
    Fixture {
        name: "inline_pftrace",
        root: "inline-pftrace/step_000000000106",
        world: "4",
        extra: &["--annotations=disabled"],
        format: Format::Pftrace,
        merged_stable: true,
    },
    Fixture {
        name: "cgp_json",
        root: "cgp/step_000000000106",
        world: "4",
        extra: &["--annotations=pkl"],
        format: Format::Json,
        merged_stable: false,
    },
    // cgp without the recording step dir: the pkl side channel resolves,
    // no recording merge applies, and the merged-trace artifact comes from
    // the reassigning splice writer instead of the serde tree. Splice output is a pure function of the input
    // bytes, so merged traces are stable here, unlike cgp_json's.
    Fixture {
        name: "cgp_norec_json",
        root: "cgp-norec/step_000000000106",
        world: "4",
        extra: &["--annotations=pkl"],
        format: Format::Json,
        merged_stable: true,
    },
    Fixture {
        name: "cgp_pftrace",
        root: "cgp-pftrace/step_000000000106",
        world: "4",
        extra: &[],
        format: Format::Pftrace,
        merged_stable: true,
    },
];

pub fn fixture(name: &str) -> &'static Fixture {
    FIXTURES.iter().find(|f| f.name == name).expect("fixture")
}

pub fn workspace_root() -> PathBuf {
    // crates/comms-viz-cli → workspace root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

pub fn traces_dir() -> PathBuf {
    workspace_root().join("testdata/traces")
}

pub fn fixture_root(fx: &Fixture) -> PathBuf {
    traces_dir().join(fx.root)
}

/// Hermetic comms_viz invocation with a private home and deterministic stderr.
pub fn comms_viz_cmd(home: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_comms_viz"));
    cmd.env("HOME", home);
    cmd
}

/// Run the compiled comms_viz binary hermetically (see [`comms_viz_cmd`]).
pub fn comms_viz<S: AsRef<std::ffi::OsStr>>(home: &Path, args: &[S]) -> Output {
    comms_viz_cmd(home)
        .args(args)
        .output()
        .expect("spawn comms_viz")
}

pub fn stderr_str(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Direct-mode analysis of a fixture with fully pinned determinism knobs.
/// `--source-trace-root` records a checkout-independent provenance token in
/// the artifacts (metadata docs embed it; the real testdata path differs
/// per machine). Writes the repaired traces so the snapshots pin them.
/// Panics on failure with the captured stderr.
pub fn analyze(fx: &Fixture, out_dir: &Path, home: &Path, extra: &[&str]) -> Output {
    let mut extra = extra.to_vec();
    extra.push("--write-repaired-traces");
    analyze_default(fx, out_dir, home, &extra)
}

/// [`analyze`] with the CLI defaults (no repaired traces).
pub fn analyze_default(fx: &Fixture, out_dir: &Path, home: &Path, extra: &[&str]) -> Output {
    let root = fixture_root(fx);
    let mut args: Vec<String> = vec![
        "--trace-root".into(),
        root.display().to_string(),
        "--out".into(),
        out_dir.display().to_string(),
        "--world-size".into(),
        fx.world.into(),
        "--started-at-us".into(),
        STARTED_AT_US.into(),
        "--completed-at-us".into(),
        COMPLETED_AT_US.into(),
        "--source-trace-root".into(),
        format!("/fixtures/{}", fx.root),
    ];
    args.extend(fx.extra.iter().map(|s| s.to_string()));
    args.extend(extra.iter().map(|s| s.to_string()));
    // Bounded default parallelism for CI; tests that probe thread-count
    // invariance pass their own --threads via `extra`.
    if !extra.contains(&"--threads") {
        args.extend(["--threads".to_string(), "4".to_string()]);
    }
    let out = comms_viz(home, &args);
    assert!(
        out.status.success(),
        "comms_viz --trace-root failed for {}:\n{}",
        fx.name,
        stderr_str(&out)
    );
    out
}

/// The final direct-mode stderr summary, e.g.
/// `comms_viz: world_size=8 processed=8 skipped=0 errors=0 pgs=[…] wall=1.2s`
/// with the volatile wall time stripped.
pub fn run_summary(out: &Output) -> String {
    let err = stderr_str(out);
    let line = err
        .lines()
        .rev()
        .find(|l| l.starts_with("comms_viz: world_size="))
        .unwrap_or_else(|| panic!("no run summary in stderr:\n{err}"));
    match line.rfind(" wall=") {
        Some(i) => line[..i].to_string(),
        None => line.to_string(),
    }
}

pub fn sha256_16(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Sorted `relpath  <size>B  sha256:<prefix>` listing of the out tree
/// minus the reports — the byte-level pin for every artifact, including the binary ones
/// (parquet, merged-trace gz) that have no readable content snapshot.
/// When `merged_stable` is false the tree/<run>/traces entries get a
/// known-issue marker instead of size+hash (see Fixture::merged_stable).
pub fn artifact_manifest(out_dir: &Path, merged_stable: bool) -> String {
    let mut files = Vec::new();
    collect_files(out_dir, &mut files);
    let mut lines = Vec::new();
    let re_rev = regex::Regex::new(r#""rev":\s*"[^"]+""#).expect("re");
    for path in files {
        let rel = path
            .strip_prefix(out_dir)
            .expect("under out dir")
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        if is_report(&rel) {
            continue;
        }
        let unstable_merged =
            !merged_stable && rel.starts_with("tree/") && rel.contains("/traces/");
        if unstable_merged {
            lines.push(format!(
                "{rel}  [bytes unstable run-to-run: merge.rs equal-ts stream hash-order (known issue)]"
            ));
        } else {
            let bytes = std::fs::read(&path).expect("read artifact");
            // _meta/config.json embeds the engine rev, which changes with
            // every commit and whose width depends on the build
            // environment; size and hash it with the rev normalized to pin
            // every other byte without snapshot churn.
            let (tag, hashed) = if rel.ends_with("_meta/config.json") {
                let text = String::from_utf8_lossy(&bytes);
                (
                    "sha256(rev-normalized)",
                    re_rev
                        .replace(&text, r#""rev": "REV""#)
                        .into_owned()
                        .into_bytes(),
                )
            } else {
                ("sha256", bytes)
            };
            lines.push(format!(
                "{rel}  {}B  {tag}:{}",
                hashed.len(),
                sha256_16(&hashed)
            ));
        }
    }
    lines.sort();
    lines.join("\n")
}

/// The default `--report` / `--structured-report` dirs inside `--out`;
/// the page tests pin them, and the page carries a render timestamp.
fn is_report(rel: &str) -> bool {
    rel.starts_with("report/") || rel.starts_with("structured_report/")
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read_dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_files(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// Render a parquet artifact as its page-facing CSV text (the same
/// rendering the publisher ships to the page), for a readable content
/// snapshot; bytes are pinned separately by the manifest hash.
pub fn parquet_csv(path: &Path) -> String {
    comms_viz_core::emit::parquet_to_csv_text(path)
        .unwrap_or_else(|e| panic!("render {}: {e}", path.display()))
}

/// `===== <name>` concatenation of every per-group member payload
/// (`per_pg/*.dpg`) under `dir`, decoded to the page-visible view (DPG1
/// columns render as the doubles the page's decoder yields; raw-CSV
/// fallback payloads pass through), sorted by file name.
pub fn concat_members(dir: &Path) -> String {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".dpg"))
        .collect();
    names.sort();
    let mut out = String::new();
    for n in &names {
        out.push_str(&format!("===== {n}\n"));
        let payload = std::fs::read(dir.join(n)).expect("read member");
        out.push_str(&comms_viz_core::dpg::member_to_text(&payload));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

/// `===== <name>` concatenation of every file directly under `dir` whose
/// name passes `keep`, sorted by file name.
pub fn concat_dir(dir: &Path, keep: impl Fn(&str) -> bool) -> String {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .filter(|n| keep(n))
        .collect();
    names.sort();
    let mut out = String::new();
    for n in &names {
        out.push_str(&format!("===== {n}\n"));
        out.push_str(&String::from_utf8_lossy(
            &std::fs::read(dir.join(n)).expect("read"),
        ));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

/// Readable summary of a merged Chrome-JSON trace artifact: top-level doc
/// shape plus event histograms — the human-reviewable complement to the
/// manifest's byte hash.
pub fn merged_trace_summary(path: &Path) -> String {
    use std::io::Read;
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(f)
        .read_to_end(&mut raw)
        .expect("gunzip merged trace");
    let doc: serde_json::Value = serde_json::from_slice(&raw).expect("parse merged trace");
    let obj = doc.as_object().expect("trace object");
    let mut out = String::new();
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    out.push_str(&format!(
        "top-level keys: {}\n",
        keys.iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    for k in [
        "schemaVersion",
        "distributedInfo",
        "traceName",
        "displayTimeUnit",
    ] {
        if let Some(v) = obj.get(k) {
            out.push_str(&format!("{k} = {v}\n"));
        }
    }
    let events = obj["traceEvents"].as_array().expect("traceEvents");
    out.push_str(&format!("traceEvents: {}\n", events.len()));
    let mut by_ph: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_cat: BTreeMap<String, usize> = BTreeMap::new();
    for e in events {
        let ph = e.get("ph").and_then(|v| v.as_str()).unwrap_or("?");
        *by_ph.entry(ph.to_string()).or_default() += 1;
        let cat = e.get("cat").and_then(|v| v.as_str()).unwrap_or("(none)");
        *by_cat.entry(cat.to_string()).or_default() += 1;
    }
    out.push_str("events by ph:\n");
    for (k, v) in &by_ph {
        out.push_str(&format!("  {k}: {v}\n"));
    }
    out.push_str("events by cat:\n");
    for (k, v) in &by_cat {
        out.push_str(&format!("  {k}: {v}\n"));
    }
    out
}

/// Schema + shape of every per-PG parquet: readable review surface for the
/// column contract (bytes are pinned by the manifest hash).
pub fn parquet_schemas(per_pg_dir: &Path) -> String {
    use polars::prelude::*;
    let mut names: Vec<String> = std::fs::read_dir(per_pg_dir)
        .expect("read per_pg")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".parquet"))
        .collect();
    names.sort();
    let mut out = String::new();
    for n in &names {
        let f = std::fs::File::open(per_pg_dir.join(n)).expect("open parquet");
        let df = ParquetReader::new(f).finish().expect("decode parquet");
        out.push_str(&format!("===== {n}: {} rows\n", df.height()));
        for col in df.columns() {
            out.push_str(&format!("  {}: {}\n", col.name(), col.dtype()));
        }
    }
    out
}

/// Insta settings shared by every snapshot: strips the two pieces of
/// output that legitimately change without a behavior change —
/// the engine git rev baked into _meta/config.json (and re-serialized into
/// the page payload) and the page's render wall-clock stamp.
pub fn settings() -> insta::Settings {
    let mut s = insta::Settings::clone_current();
    s.set_prepend_module_to_snapshot(false);
    s.add_filter(r#""rev":\s*"[^"]+""#, r#""rev": "[GIT_REV]""#);
    s.add_filter(r#""generated_at_s":\s*\d+"#, r#""generated_at_s": "[TS]""#);
    s
}

/// Recursive byte-compare of two out trees; `skip` filters rel paths.
pub fn assert_trees_identical(a: &Path, b: &Path, skip: impl Fn(&str) -> bool) {
    let rel_set = |root: &Path| -> BTreeMap<String, PathBuf> {
        let mut files = Vec::new();
        collect_files(root, &mut files);
        files
            .into_iter()
            .map(|p| {
                (
                    p.strip_prefix(root)
                        .expect("under root")
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy())
                        .collect::<Vec<_>>()
                        .join("/"),
                    p,
                )
            })
            .filter(|(rel, _)| !is_report(rel) && !skip(rel))
            .collect()
    };
    let fa = rel_set(a);
    let fb = rel_set(b);
    assert_eq!(
        fa.keys().collect::<Vec<_>>(),
        fb.keys().collect::<Vec<_>>(),
        "tree file sets differ"
    );
    for (rel, pa) in &fa {
        let ba = std::fs::read(pa).expect("read a");
        let bb = std::fs::read(&fb[rel]).expect("read b");
        assert!(
            ba == bb,
            "byte mismatch at {rel} ({}B vs {}B, sha {} vs {})",
            ba.len(),
            bb.len(),
            sha256_16(&ba),
            sha256_16(&bb)
        );
    }
}

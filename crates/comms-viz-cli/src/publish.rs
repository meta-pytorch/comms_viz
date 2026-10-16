// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Local static-report rendering from a completed comms_viz artifact tree.

use std::collections::HashSet;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;

pub struct RunData {
    pub run_id: String,
    pub run_json: serde_json::Value,
    pub config_json: serde_json::Value,
    pub status_json: serde_json::Value,
    pub metadata_json: Option<serde_json::Value>,
    pub pg_map_json: Option<serde_json::Value>,
    pub breakdown_csv: String,
    pub per_pg_warm: Vec<(String, Vec<u8>)>,
    pub per_pg_all: Vec<(String, u64, PathBuf)>,
    pub rank_map_json: Option<String>,
    /// Where `--write-repaired-traces` wrote the run's traces, if it did.
    pub repaired_traces_dir: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
struct ValidatedTree {
    run_id: String,
    run_root: PathBuf,
    result_dir: String,
    warnings: Vec<String>,
}

fn validate_out_dir(out_dir: &Path, select_run_id: Option<&str>) -> Result<ValidatedTree> {
    let tree_dir = out_dir.join("tree");
    if !tree_dir.is_dir() {
        bail!("{} has no tree/ directory", out_dir.display());
    }
    let mut candidates: Vec<String> = std::fs::read_dir(&tree_dir)?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|entry| entry.path().is_dir())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    candidates.sort();
    let run_id = match select_run_id {
        Some(id) if candidates.iter().any(|candidate| candidate == id) => id.to_string(),
        Some(id) => bail!(
            "run id {id} not found under {} (available: {})",
            tree_dir.display(),
            candidates.join(", ")
        ),
        None if candidates.len() == 1 => candidates.remove(0),
        None if candidates.is_empty() => bail!("no run directories under {}", tree_dir.display()),
        None => bail!(
            "multiple runs under {} -- pass --run-id to select one of: {}",
            tree_dir.display(),
            candidates.join(", ")
        ),
    };
    let run_root = tree_dir.join(&run_id);
    let manifest_path = run_root.join("_meta/MANIFEST");
    let manifest = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("missing {}", manifest_path.display()))?;
    if manifest.trim().is_empty() {
        bail!("{} is empty", manifest_path.display());
    }
    for relative in manifest.lines().filter(|line| !line.is_empty()) {
        if !run_root.join(relative).is_file() {
            bail!("MANIFEST references missing artifact {relative}");
        }
    }

    let run_path = run_root.join("_meta/run.json");
    let run_doc: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&run_path)
            .with_context(|| format!("missing {}", run_path.display()))?,
    )
    .with_context(|| format!("unparsable {}", run_path.display()))?;
    if run_doc.get("run_id").and_then(|value| value.as_str()) != Some(&run_id) {
        bail!("run.json run_id does not match tree directory {run_id}");
    }

    let mut warnings = Vec::new();
    if run_doc.get("status").and_then(|value| value.as_i64()) != Some(4) {
        warnings.push("run is not marked completed".to_string());
    }
    let result_dir = std::fs::read_to_string(run_root.join("_meta/config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| {
            value
                .get("result_dir")
                .and_then(|dir| dir.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| comms_viz_core::emit::RESULT_DIR.to_string());

    Ok(ValidatedTree {
        run_id,
        run_root,
        result_dir,
        warnings,
    })
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("missing {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("unparsable {}", path.display()))
}

fn read_json_opt(path: &Path) -> Option<serde_json::Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
}

fn axis_of(stem: &str) -> &str {
    match stem.rsplit_once('_') {
        Some((axis, id)) if !id.is_empty() && id.bytes().all(|byte| byte.is_ascii_digit()) => axis,
        _ => stem,
    }
}

fn layout_key(stem: &str) -> (String, u64, String) {
    let id = stem
        .rsplit_once('_')
        .and_then(|(_, id)| id.parse().ok())
        .unwrap_or(u64::MAX);
    (axis_of(stem).to_string(), id, stem.to_string())
}

fn default_group_idx(
    groups: &[(String, u64, PathBuf)],
    pg_map: &Option<serde_json::Value>,
    metadata: &Option<serde_json::Value>,
) -> Option<usize> {
    let mut aliases = HashSet::new();
    if let Some(map) = pg_map.as_ref().and_then(serde_json::Value::as_object) {
        let info = metadata
            .as_ref()
            .and_then(|value| value.get("pg_info"))
            .and_then(|value| value.get("groups"));
        for (key, value) in map {
            let group = info.and_then(|groups| groups.get(key));
            let size = group
                .and_then(|group| group.get("size"))
                .and_then(serde_json::Value::as_i64)
                .or_else(|| {
                    value
                        .get("pg_ranks_map")
                        .and_then(|ranks| ranks.get(0))
                        .and_then(serde_json::Value::as_array)
                        .map(|ranks| ranks.len() as i64)
                })
                .unwrap_or(0);
            if size > 1 {
                aliases.insert(
                    group
                        .and_then(|group| group.get("alias"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or(key)
                        .to_string(),
                );
            }
        }
    }
    let candidates = || {
        (0..groups.len())
            .filter(|index| !comms_viz_core::analysis::is_diagnostics_member(&groups[*index].0))
    };
    candidates()
        .filter(|index| aliases.contains(axis_of(&groups[*index].0)))
        .min_by_key(|index| layout_key(&groups[*index].0))
        .or_else(|| candidates().min_by_key(|index| layout_key(&groups[*index].0)))
}

fn load_run_data(tree: &ValidatedTree) -> Result<RunData> {
    let root = &tree.run_root;
    let run_json = read_json(&root.join("_meta/run.json"))?;
    let config_json = read_json(&root.join("_meta/config.json"))?;
    let status_json = read_json(&root.join("_meta/status.json"))?;
    let metadata_json = read_json_opt(&root.join("metadata.json"));
    let pg_map_json = read_json_opt(&root.join("pg_map.json"));
    let breakdown_path = root
        .join(&tree.result_dir)
        .join("pipeline_a/breakdown.parquet");
    let breakdown_csv = comms_viz_core::emit::parquet_to_csv_text(&breakdown_path)
        .with_context(|| format!("load {}", breakdown_path.display()))?;

    let per_pg_dir = root.join(&tree.result_dir).join("pipeline_a/per_pg");
    let mut per_pg_all = Vec::new();
    if per_pg_dir.is_dir() {
        for entry in std::fs::read_dir(&per_pg_dir)?.collect::<std::io::Result<Vec<_>>>()? {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(stem) = name.strip_suffix(".dpg") {
                per_pg_all.push((stem.to_string(), entry.metadata()?.len(), entry.path()));
            }
        }
        per_pg_all.sort_by_key(|(stem, _, _)| layout_key(stem));
    }
    let per_pg_warm = default_group_idx(&per_pg_all, &pg_map_json, &metadata_json)
        .map(|index| -> Result<(String, Vec<u8>)> {
            let (stem, _, path) = &per_pg_all[index];
            Ok((stem.clone(), std::fs::read(path)?))
        })
        .transpose()?
        .into_iter()
        .collect();

    // The analyzed ranks with their trace files and any known hosts.
    let rank_map_json = std::fs::read_to_string(root.join("rank_map.json")).ok();
    let repaired_traces_dir = std::fs::canonicalize(root.join("traces"))
        .ok()
        .map(|p| format!("{}/", p.display()));

    let mut warnings = tree.warnings.clone();
    let diagnostics = crate::diagnostics::read(&per_pg_all)?;
    warnings.extend(crate::diagnostics::coverage_warnings(
        &status_json,
        &diagnostics,
    ));
    warnings.extend(crate::diagnostics::step_warnings(&diagnostics));
    warnings.extend(crate::diagnostics::annotation_warnings(&diagnostics));
    warnings.extend(crate::diagnostics::graph_annotation_warnings(&diagnostics));

    Ok(RunData {
        run_id: tree.run_id.clone(),
        run_json,
        config_json,
        status_json,
        metadata_json,
        pg_map_json,
        breakdown_csv,
        per_pg_warm,
        per_pg_all,
        rank_map_json,
        repaired_traces_dir,
        warnings,
    })
}

#[derive(Debug, PartialEq)]
pub struct PgIndexEntry {
    pub stem: String,
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Default)]
pub struct Sidecar {
    pub artifacts: Vec<(&'static str, u64, u64)>,
    pub groups: Vec<PgIndexEntry>,
    pub shard_lens: Vec<u64>,
}

impl Sidecar {
    pub fn is_empty(&self) -> bool {
        self.artifacts.is_empty() && self.groups.is_empty()
    }
}

const SHARD_TARGET: u64 = 64 << 20;

fn gzip_member(data: &[u8]) -> Result<Vec<u8>> {
    let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::new(6).unwrap());
    let mut output = vec![0; compressor.gzip_compress_bound(data.len())];
    let len = compressor
        .gzip_compress(data, &mut output)
        .map_err(|error| anyhow::anyhow!("gzip member: {error:?}"))?;
    output.truncate(len);
    Ok(output)
}

fn write_sidecar(
    dest: &Path,
    data_name: &str,
    artifacts: &[(&'static str, &str)],
    groups: &[(String, u64, PathBuf)],
) -> Result<Sidecar> {
    let mut sidecar = Sidecar::default();
    let mut shard = 0usize;
    let mut shard_len = 0u64;
    let mut offset = 0u64;
    let open = |index: usize| -> Result<std::io::BufWriter<std::fs::File>> {
        let name = if index == 0 {
            data_name.to_string()
        } else {
            format!("{data_name}.{index}")
        };
        Ok(std::io::BufWriter::new(std::fs::File::create(
            dest.join(name),
        )?))
    };
    let mut writer = open(0)?;

    for (key, text) in artifacts {
        let member = gzip_member(text.as_bytes())?;
        writer.write_all(&member)?;
        sidecar.artifacts.push((key, offset, member.len() as u64));
        offset += member.len() as u64;
        shard_len += member.len() as u64;
    }
    for (stem, _, path) in groups {
        let member = gzip_member(&std::fs::read(path)?)?;
        if shard_len > 0 && shard_len + member.len() as u64 > SHARD_TARGET {
            writer.flush()?;
            sidecar.shard_lens.push(shard_len);
            shard += 1;
            shard_len = 0;
            writer = open(shard)?;
        }
        writer.write_all(&member)?;
        sidecar.groups.push(PgIndexEntry {
            stem: stem.clone(),
            offset,
            len: member.len() as u64,
        });
        offset += member.len() as u64;
        shard_len += member.len() as u64;
    }
    writer.flush()?;
    sidecar.shard_lens.push(shard_len);
    Ok(sidecar)
}

fn path_slug(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character
            } else {
                '-'
            }
        })
        .collect()
}

fn page_stem(data: &RunData) -> String {
    let mut parts = vec!["comms_viz".to_string(), path_slug(&data.run_id)];
    if let Some(name) = data
        .run_json
        .get("run_name")
        .and_then(serde_json::Value::as_str)
        .filter(|name| !name.is_empty())
    {
        parts.push(path_slug(name));
    }
    let trace_path = data
        .metadata_json
        .as_ref()
        .and_then(|metadata| {
            metadata
                .get("trace_prefix")
                .or_else(|| metadata.get("config")?.get("trace_folder_path"))
        })
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            data.config_json
                .get("trace_root")
                .and_then(serde_json::Value::as_str)
        });
    if let Some(step) = trace_path
        .and_then(|path| path.trim_end_matches('/').rsplit('/').next())
        .and_then(|leaf| leaf.strip_prefix("step_"))
        .and_then(|step| step.parse::<u64>().ok())
    {
        parts.push(format!("step_{step}"));
    }
    parts.join("-")
}

/// Where to write a run's reports; at least one is set.
#[derive(Clone, Copy)]
pub struct Targets<'a> {
    /// HTML report directory.
    pub report: Option<&'a Path>,
    /// Structured report directory (Markdown, CSV and JSON).
    pub structured: Option<&'a Path>,
}

pub fn render_local(out_dir: &Path, run_id: Option<&str>, targets: Targets<'_>) -> Result<()> {
    let tree = validate_out_dir(out_dir, run_id)?;
    let data = load_run_data(&tree)?;
    let stem = page_stem(&data);
    render_page(&data, &stem, targets)
}

fn render_page(data: &RunData, stem: &str, targets: Targets<'_>) -> Result<()> {
    for warning in &data.warnings {
        eprintln!("comms_viz: warning: {warning}");
    }
    if let Some(structured) = targets.structured {
        crate::structured_report::write_structured_report(data, stem, structured)?;
    }
    let Some(dest) = targets.report else {
        return Ok(());
    };
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    let data_name = format!("{stem}.pgdata");
    let embeds = crate::static_page::plan_embeds(data)?;
    let sidecar = write_sidecar(
        dest,
        &data_name,
        &embeds.sidecar_artifacts(data),
        &data.per_pg_all,
    )?;
    let html = crate::static_page::render(data, &embeds, &sidecar, &data_name)?;
    let html_path = dest.join(format!("{stem}.html"));
    std::fs::write(&html_path, html).with_context(|| format!("write {}", html_path.display()))?;
    eprintln!("comms_viz: rendered {}", html_path.display());
    Ok(())
}

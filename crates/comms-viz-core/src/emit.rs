// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Artifact emission — per-worker partial writes, the aggregation fold,
//! and staging promotion, all in-process against a local output directory.
//!
//! Any worker split of the same rank-ordered rows yields byte-identical
//! final artifacts (identical column sets per partial; concat in worker
//! order == rank order).
//!
//! The final parquets are produced by the aggregation round trip
//! (write_parquet → read_parquet → concat → write_parquet) over the worker
//! partials. All tabular artifacts are typed parquet; per-group data lands
//! directly as `.dpg` sidecar member payloads (see dpg), the form the
//! report page actually loads.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use polars::prelude::*;
use serde_json::Value;
use serde_json::json;

use crate::analysis::BreakdownValue;
use crate::analysis::DIAGNOSTICS_PG;
use crate::analysis::pg_group_id;
use crate::legacy::json::TraceEnvironment;
use crate::trace::ProcessGroupConfig;

/// One rank's processing output (order of `pg_dfs` = rank_info order).
pub struct RankResult {
    pub base_ns: Option<u64>,
    pub rank: i64,
    pub breakdown_row: Vec<(&'static str, BreakdownValue)>,
    pub pg_dfs: Vec<(String, DataFrame)>,
    pub process_groups: Vec<ProcessGroupConfig>,
    pub environment: TraceEnvironment,
}

/// Result directory under `tree/{run}/`, recorded in `_meta/config.json`
/// so the publisher also reads trees written under another name.
pub const RESULT_DIR: &str = "results";

/// Parameterized run metadata (timestamps/uuids/hosts are inputs so the
/// snapshot suite can pin output bytes deterministically).
pub struct RunMeta {
    pub run_id: String,
    /// Descriptive run name; empty when unnamed.
    pub run_name: String,
    pub world_size: i64,
    pub trace_root: String,
    /// Trace-source provenance recorded in the artifacts: the SUBMITTED
    /// trace root, which outlives the run; a staged `trace_root` deleted at
    /// run end would leave dangling provenance.
    /// None → record `trace_root` itself.
    pub source_trace_root: Option<String>,
    pub started_at_us: i64,
    /// 0 → resolved per document at write time (`resolve_write_ts_us`).
    pub completed_at_us: i64,
    pub worker_count: i64,
    /// Extra run-config entries to embed in config.json.
    pub config_extra: Vec<(String, Value)>,
}

impl RunMeta {
    /// The trace root the artifacts record as the source of truth
    /// (metadata.json trace_folder_path, _meta/config.json trace_root);
    /// file access always uses
    /// `trace_root`.
    fn recorded_trace_root(&self) -> &str {
        self.source_trace_root
            .as_deref()
            .unwrap_or(&self.trace_root)
    }
}

/// Engine provenance stamp, embedded in `_meta/config.json` under the
/// `engine` key — how comms_viz-produced runs are identified and badged.
pub fn engine_stamp() -> Value {
    json!({
        "name": "comms_viz",
        "version": env!("CARGO_PKG_VERSION"),
        "rev": env!("COMMS_VIZ_GIT_REV"),
    })
}

/// Compact JSON for the metadata documents (string-keyed `Value`s — cannot
/// fail). `Value` maps emit in sorted key order; key order is not part of
/// the artifact contract (readers parse, the page sorts for display).
fn dumps(v: &Value) -> String {
    serde_json::to_string(v).expect("string-keyed JSON emission cannot fail")
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes).with_context(|| format!("write {path:?}"))
}

/// polars `df.write_csv()` with default settings — the text form the
/// page's CSV parsers consume (only ever rendered in memory; no CSV
/// artifact lands on disk).
pub fn write_csv_bytes(df: &mut DataFrame) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    CsvWriter::new(&mut buf).finish(df)?;
    Ok(buf)
}

/// Read a parquet artifact and render it as CSV text — the publisher
/// feeds this to the page, whose breakdown parser speaks CSV text.
pub fn parquet_to_csv_text(path: &Path) -> Result<String> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    // UFCS: polars' prelude now exports its own `PolarsContext::with_context`.
    let mut df = anyhow::Context::with_context(ParquetReader::new(f).finish(), || {
        format!("decode parquet {}", path.display())
    })?;
    let bytes = write_csv_bytes(&mut df)?;
    String::from_utf8(bytes).context("parquet rendered non-utf8 csv")
}

/// polars `df.write_parquet(buf)` (zstd, statistics on), with the row
/// groups encoded+compressed in parallel.
///
/// The stock `ParquetWriter::finish` runs essentially single-threaded on
/// large frames: its "parallel" flag only CONSTRUCTS page iterators —
/// polars-parquet encodes pages lazily on iterator `next()` and compresses
/// on `Compressor::advance()`, both of which happen as the one FileWriter
/// thread drains the row groups in order. In the aggregation fold that
/// pinned one core for ~16s on the fattest PG key while the machine sat
/// 74% idle. Here every row group is materialized
/// (encode + zstd) on the polars pool first, then written in index order.
/// Chunk boundaries (`chunk_df_for_writing`, 512² rows), page splits, and
/// write order are exactly the serial writer's, so the output bytes are
/// identical by construction — the snapshot suite pins this.
fn write_parquet_bytes(df: &mut DataFrame) -> Result<Vec<u8>> {
    use polars_io::parquet::write::get_encodings;
    use polars_parquet::write::CompressedPage;
    use polars_parquet::write::Compressor;
    use polars_parquet::write::Version;
    use polars_parquet::write::WriteOptions as PqWriteOptions;
    use polars_parquet::write::array_to_columns;
    use polars_parquet::write::to_parquet_schema;
    use rayon::prelude::*;

    // min/max/null-count on, distinct off, no binary truncation — the
    // stock writer's defaults.
    let statistics = StatisticsOptions::default();
    let options = PqWriteOptions {
        statistics,
        compression: ParquetCompression::Zstd(None).into(),
        version: Version::V1,
        data_page_size: None,
    };

    // ParquetWriter::finish's default row-group size.
    let chunked = polars_core::frame::chunk_df_for_writing(df, 512 * 512)?;

    // The same per-field mapping schema_to_arrow_checked applies (we have
    // no Object columns, the only case it rejects).
    let arrow_schema: ArrowSchema = chunked
        .schema()
        .iter_fields()
        .map(|f| {
            let f = f
                .dtype()
                .to_arrow_field(f.name().clone(), CompatLevel::newest());
            (f.name.clone(), f)
        })
        .collect();
    let encodings = get_encodings(&arrow_schema);
    let parquet_schema = to_parquet_schema(&arrow_schema)?;
    let fields = parquet_schema.fields().to_vec();

    // Materialize each row group's compressed pages in parallel (batch
    // order preserved by the indexed collect).
    let batches: Vec<_> = chunked
        .iter_chunks(CompatLevel::newest(), false)
        .filter(|b| !b.is_empty())
        .collect();
    let heights: Vec<u64> = batches.iter().map(|b| b.len() as u64).collect();
    let row_groups: Vec<Vec<Vec<CompressedPage>>> =
        polars_core::runtime::THREAD_POOL.install(|| {
            batches
                .into_par_iter()
                .map(|batch| {
                    batch
                        .columns()
                        .iter()
                        .zip(fields.iter())
                        .zip(encodings.iter())
                        .map(|((array, type_), col_encodings)| {
                            let leaves =
                                array_to_columns(array, type_.clone(), options, col_encodings)
                                    .map_err(|e| anyhow::anyhow!("parquet encode: {e}"))?;
                            leaves
                                .into_iter()
                                .map(|pages| {
                                    Compressor::new_from_vec(
                                        pages.map(|r| {
                                            r.map_err(|e| {
                                            polars_parquet::read::ParquetError::FeatureNotSupported(
                                                format!("reraised in comms_viz: {e}"),
                                            )
                                        })
                                        }),
                                        options.compression,
                                        vec![],
                                    )
                                    .collect::<std::result::Result<Vec<CompressedPage>, _>>()
                                    .map_err(|e| anyhow::anyhow!("parquet compress: {e}"))
                                })
                                .collect::<Result<Vec<_>>>()
                        })
                        .try_fold(Vec::new(), |mut acc, cols| {
                            acc.extend(cols?);
                            Ok::<_, anyhow::Error>(acc)
                        })
                })
                .collect::<Result<Vec<_>>>()
        })?;

    let mut buf = Vec::new();
    {
        let mut batched = ParquetWriter::new(&mut buf)
            .with_compression(ParquetCompression::Zstd(None))
            .with_statistics(statistics)
            .batched(chunked.schema())?;
        for (rg, height) in row_groups.iter().zip(heights) {
            batched
                .write_row_group(height, rg)
                .map_err(|e| anyhow::anyhow!("parquet write: {e}"))?;
        }
        batched.finish()?;
    }
    Ok(buf)
}

fn read_parquet_bytes(bytes: &[u8]) -> Result<DataFrame> {
    Ok(ParquetReader::new(Cursor::new(bytes.to_vec())).finish()?)
}

/// Eager vertical concat (vstack chain, no rechunk); schemas are
/// identical across our inputs by construction. Eager vstack avoids the
/// lazy planner's per-input overhead at 14K+ inputs.
fn concat_vertical_relaxed(dfs: Vec<DataFrame>) -> Result<DataFrame> {
    let mut it = dfs.into_iter();
    let mut out = it.next().ok_or_else(|| anyhow::anyhow!("empty concat"))?;
    for df in it {
        out.vstack_mut(&df)?;
    }
    // One chunk per input frame otherwise: the diagnostics member gets one
    // per rank, and encoding ~20k chunks overflowed a rayon worker's stack
    // (and took ~18 s at 20k ranks).
    out.rechunk_mut_par();
    Ok(out)
}

/// Vertical concat of breakdown frames whose columns can differ (per-axis
/// columns depend on each rank's PGs): columns become the sorted union, with
/// nulls where a frame lacks one.
fn concat_breakdown(dfs: Vec<DataFrame>) -> Result<DataFrame> {
    let mut names: Vec<PlSmallStr> = dfs
        .iter()
        .flat_map(|df| df.get_column_names_owned())
        .collect();
    names.sort_unstable();
    names.dedup();
    let aligned = dfs
        .into_iter()
        .map(|mut df| {
            let height = df.height();
            for name in &names {
                if df.column(name).is_err() {
                    let dtype = if name == "rank" {
                        DataType::Int64
                    } else {
                        DataType::Float64
                    };
                    df.with_column(Column::full_null(name.clone(), height, &dtype))?;
                }
            }
            Ok(df.select(names.iter().cloned())?)
        })
        .collect::<Result<Vec<_>>>()?;
    concat_vertical_relaxed(aligned)
}

pub struct EmitSummary {
    pub ranks_processed: usize,
    pub pg_keys: Vec<String>,
    pub manifest_rels: Vec<String>,
    pub process_groups: Vec<ProcessGroupConfig>,
}

/// One worker's partial artifacts, folded by the aggregator.
pub struct WorkerPartial {
    /// breakdown.parquet bytes; empty = the worker produced no rows.
    breakdown: Vec<u8>,
    pg_keys: Vec<String>,
    pg_bytes: HashMap<String, Vec<u8>>,
    process_groups: Vec<ProcessGroupConfig>,
}

/// Build the breakdown table from per-rank rows: columns are the
/// alphabetically sorted union of row keys — a pinned column order the
/// page's category rail preserves; `rank` is i64, every total is f64 (the
/// failure sentinel's Int(-1) lands as -1.0). A key a row lacks (e.g. a
/// sentinel row on a run with extra PG axes) is null. The fixed dtypes
/// also make every partial's schema identical by construction, so the
/// aggregator's vstack can never hit a dtype cliff.
fn rows_to_df(rows: &[Vec<(&str, BreakdownValue)>]) -> Result<DataFrame> {
    let mut keys: Vec<&str> = Vec::new();
    for row in rows {
        for (k, _) in row {
            if !keys.contains(k) {
                keys.push(k);
            }
        }
    }
    keys.sort_unstable();
    let find = |row: &[(&str, BreakdownValue)], k: &str| -> Option<BreakdownValue> {
        row.iter().find(|(rk, _)| *rk == k).map(|(_, v)| *v)
    };
    let cols: Vec<Column> = keys
        .iter()
        .map(|k| {
            if *k == "rank" {
                let vals: Vec<Option<i64>> = rows
                    .iter()
                    .map(|row| {
                        find(row, k).map(|v| match v {
                            BreakdownValue::Int(i) => i,
                            BreakdownValue::Float(f) => f as i64,
                        })
                    })
                    .collect();
                Column::new((*k).into(), vals)
            } else {
                let vals: Vec<Option<f64>> = rows
                    .iter()
                    .map(|row| {
                        find(row, k).map(|v| match v {
                            BreakdownValue::Int(i) => i as f64,
                            BreakdownValue::Float(f) => f,
                        })
                    })
                    .collect();
                Column::new((*k).into(), vals)
            }
        })
        .collect();
    Ok(DataFrame::new_infer_height(cols)?)
}

/// Front-loaded per-worker rank ranges: the remainder ranks go to the
/// first workers, one extra each.
pub fn worker_rank_ranges(world: i64, w_count: i64) -> Vec<(i64, i64)> {
    let base = world / w_count;
    let remainder = world % w_count;
    let mut out = Vec::with_capacity(w_count as usize);
    let mut lo = 0i64;
    for w in 0..w_count {
        let size = if w < remainder { base + 1 } else { base };
        out.push((lo, lo + size));
        lo += size;
    }
    out
}

/// Build + write the worker partial files.
///
/// With `meta_ctx`, each worker's completed heartbeat (and failure doc when
/// its ranks errored) is written right after its partial files, so a
/// heartbeat never lands before the partial it attests to.
pub fn write_worker_partials(
    combined_dir: &Path,
    results: &mut [Option<RankResult>],
    w_count: i64,
    meta_ctx: Option<&WorkerMetaCtx>,
) -> Result<Vec<WorkerPartial>> {
    let world = results.len() as i64;
    let w_count = w_count.max(1);
    use rayon::prelude::*;

    // Split results into per-worker mutable slices (front-loaded ranges,
    // worker_rank_ranges semantics).
    let mut slices: Vec<&mut [Option<RankResult>]> = Vec::with_capacity(w_count as usize);
    let mut rank_los: Vec<i64> = Vec::with_capacity(w_count as usize);
    {
        let base = world / w_count;
        let remainder = (world % w_count) as usize;
        let mut rest = &mut *results;
        let mut lo = 0i64;
        for w in 0..w_count as usize {
            let size = if w < remainder { base + 1 } else { base } as usize;
            let take = size.min(rest.len());
            let (head, tail) = rest.split_at_mut(take);
            slices.push(head);
            rank_los.push(lo);
            lo += take as i64;
            rest = tail;
        }
    }

    slices
        .into_par_iter()
        .enumerate()
        .map(|(w, slice)| {
            let lo = rank_los[w];
            let hi = lo + slice.len() as i64;
            let partial = write_single_worker_partial(combined_dir, w, slice)?;
            if let Some(ctx) = meta_ctx {
                let mut processed = 0i64;
                let mut errs: Vec<(i64, String)> = Vec::new();
                for (i, r) in slice.iter().enumerate() {
                    let rank = lo + i as i64;
                    if r.is_some() {
                        processed += 1;
                    } else if let Some(msg) = ctx.rank_errors.get(&rank) {
                        errs.push((rank, msg.clone()));
                    }
                }
                let skipped = (hi - lo) - processed - errs.len() as i64;
                write_worker_heartbeat(
                    &ctx.meta_dir,
                    w as i64,
                    lo,
                    hi,
                    processed,
                    skipped,
                    errs.len() as i64,
                    ctx.updated_at_us,
                    ctx.started_at_us,
                )?;
                if !errs.is_empty() {
                    write_worker_failure(
                        &ctx.meta_dir,
                        w as i64,
                        "RankProcessingError",
                        &rank_errors_traceback(&errs),
                        ctx.updated_at_us,
                    )?;
                }
            }
            Ok(partial)
        })
        .collect::<Result<Vec<_>>>()
}

/// One worker's partial (breakdown + per_pg parquets).
fn write_single_worker_partial(
    combined_dir: &Path,
    w: usize,
    slice: &mut [Option<RankResult>],
) -> Result<WorkerPartial> {
    let partial_dir = combined_dir.join(format!("worker_{w}"));
    let clocks: serde_json::Map<String, Value> = slice
        .iter()
        .flatten()
        .map(|r| (r.rank.to_string(), json!(r.base_ns.map(|v| v.to_string()))))
        .collect();
    write_file(
        &partial_dir.join("rank_clocks.json"),
        dumps(&Value::Object(clocks)).as_bytes(),
    )?;
    let breakdown_rows: Vec<Vec<(&str, BreakdownValue)>> = slice
        .iter()
        .flatten()
        .map(|r| r.breakdown_row.clone())
        .collect();
    let breakdown = if breakdown_rows.is_empty() {
        Vec::new()
    } else {
        write_parquet_bytes(&mut rows_to_df(&breakdown_rows)?)?
    };
    if !breakdown.is_empty() {
        write_file(&partial_dir.join("breakdown.parquet"), &breakdown)?;
    }
    let mut pg_order: Vec<String> = Vec::new();
    let mut by_key: HashMap<String, Vec<DataFrame>> = HashMap::new();
    for r in slice.iter_mut().flatten() {
        for (key, df) in r.pg_dfs.drain(..) {
            if !by_key.contains_key(&key) {
                pg_order.push(key.clone());
            }
            by_key.entry(key).or_default().push(df);
        }
    }
    let mut pg_bytes = HashMap::new();
    for key in &pg_order {
        let dfs = by_key.remove(key).unwrap();
        let mut combined = concat_vertical_relaxed(dfs)?;
        if combined.height() == 0 {
            continue;
        }
        let bytes = write_parquet_bytes(&mut combined)?;
        write_file(
            &partial_dir.join("per_pg").join(format!("{key}.parquet")),
            &bytes,
        )?;
        pg_bytes.insert(key.clone(), bytes);
    }
    let process_groups = merge_process_group_configs(
        slice
            .iter()
            .flatten()
            .flat_map(|result| &result.process_groups),
    )?;
    Ok(WorkerPartial {
        breakdown,
        pg_keys: pg_order,
        pg_bytes,
        process_groups,
    })
}

/// Rename process groups whose name covers more ranks than the group's size
/// to `{name}-r{lowest rank}`, in each rank's group inventory, table keys and
/// `pg_name` column. c10d's `split_group` names every subgroup of one split
/// the same (e.g. TorchTitan's PP p2p groups); kept as one group, their
/// collectives would be matched across subgroups.
fn disambiguate_shared_group_names(results: &mut [Option<RankResult>]) -> Result<()> {
    // Per name: every rank any rank lists, and the largest reported size.
    let mut spans: HashMap<String, (std::collections::BTreeSet<i64>, i64)> = HashMap::new();
    for group in results.iter().flatten().flat_map(|r| &r.process_groups) {
        let (ranks, size) = spans.entry(group.pg_name.clone()).or_default();
        ranks.extend(&group.ranks);
        *size = (*size).max(group.pg_size);
    }
    spans.retain(|_, (ranks, size)| *size > 0 && ranks.len() > *size as usize);
    if spans.is_empty() {
        return Ok(());
    }
    for result in results.iter_mut().flatten() {
        let mut renames: Vec<(String, String)> = Vec::new();
        for group in &mut result.process_groups {
            let Some(&lowest) = group.ranks.iter().min() else {
                continue;
            };
            if spans.contains_key(&group.pg_name) {
                let name = format!("{}-r{lowest}", group.pg_name);
                renames.push((std::mem::replace(&mut group.pg_name, name.clone()), name));
            }
        }
        for (key, df) in &mut result.pg_dfs {
            if key == DIAGNOSTICS_PG {
                continue;
            }
            for (old, new) in &renames {
                let Some(axis) = key.strip_suffix(&format!("_{}", pg_group_id(old))) else {
                    continue;
                };
                *key = format!("{axis}_{}", pg_group_id(new));
                df.with_column(Column::new(
                    "pg_name".into(),
                    vec![new.as_str(); df.height()],
                ))?;
            }
        }
    }
    Ok(())
}

/// Emit the full artifact tree for a completed run. `results` is indexed
/// by rank; `None` = skipped rank (missing trace / timeout / error) — it
/// contributes no rows.
pub fn emit_run(
    out_root: &Path,
    meta: &RunMeta,
    results: &mut [Option<RankResult>],
    meta_ctx: Option<&WorkerMetaCtx>,
) -> Result<EmitSummary> {
    let t_emit = std::time::Instant::now();
    disambiguate_shared_group_names(results)?;
    let combined_dir = out_root
        .join("tree")
        .join(&meta.run_id)
        .join("_staging")
        .join("_combined");
    let partials = write_worker_partials(&combined_dir, results, meta.worker_count, meta_ctx)?;
    eprintln!(
        "emit: worker partials {:.1}s",
        t_emit.elapsed().as_secs_f64()
    );
    aggregate_partials(out_root, meta, &partials)
}

/// A worker partial with its parquet payloads decoded.
struct DecodedPartial {
    breakdown_df: Option<DataFrame>,
    pg_keys: Vec<String>,
    pg_dfs: HashMap<String, DataFrame>,
    process_groups: Vec<ProcessGroupConfig>,
}

fn decode_partial(p: &WorkerPartial) -> Result<DecodedPartial> {
    let breakdown_df = if p.breakdown.is_empty() {
        None
    } else {
        Some(read_parquet_bytes(&p.breakdown)?)
    };
    let mut pg_dfs = HashMap::new();
    for k in &p.pg_keys {
        pg_dfs.insert(k.clone(), read_parquet_bytes(&p.pg_bytes[k])?);
    }
    Ok(DecodedPartial {
        breakdown_df,
        pg_keys: p.pg_keys.clone(),
        pg_dfs,
        process_groups: p.process_groups.clone(),
    })
}

/// Aggregation + promotion + MANIFEST over the worker partials.
fn aggregate_partials(
    out_root: &Path,
    meta: &RunMeta,
    partials: &[WorkerPartial],
) -> Result<EmitSummary> {
    use rayon::prelude::*;
    let decoded: Vec<DecodedPartial> = partials
        .par_iter()
        .map(decode_partial)
        .collect::<Result<Vec<_>>>()?;
    aggregate_decoded(out_root, meta, &decoded)
}

fn aggregate_decoded(
    out_root: &Path,
    meta: &RunMeta,
    partials: &[DecodedPartial],
) -> Result<EmitSummary> {
    let run_root = out_root.join("tree").join(&meta.run_id);
    let staging = run_root.join("_staging");
    let final_staging = staging.join(RESULT_DIR).join("pipeline_a");
    use rayon::prelude::*;
    let t_agg = std::time::Instant::now();

    // All fold outputs funnel through a small writer crew: rayon keeps
    // serializing in parallel, but only FILE_WRITERS threads ever sit in
    // write() — 64 threads queueing on btrfs's delalloc spin lock write
    // no faster than 4, they only burn CPU spinning (51.4% of aggregator
    // on-CPU time).
    let files = crate::writer_pool::WriterPool::new(4, 256 * 1024 * 1024);

    // ---- Aggregation fold over the worker partials ----
    let ranks_processed;
    {
        let dfs: Vec<DataFrame> = partials
            .iter()
            .filter_map(|p| p.breakdown_df.clone())
            .collect::<Vec<_>>();
        if !dfs.is_empty() {
            let mut combined = concat_breakdown(dfs)?;
            ranks_processed = combined.height();
            let out = write_parquet_bytes(&mut combined)?;
            files.submit(final_staging.join("breakdown.parquet"), out, false);
        } else {
            ranks_processed = 0;
        }
    }
    eprintln!("emit: rank-table {:.2}s", t_agg.elapsed().as_secs_f64());

    // Per-PG merge: union of keys across workers, sorted; per key concat the
    // worker parquets in worker order. Keys are independent → parallel.
    let mut pg_keys: Vec<String> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for p in partials {
            for k in &p.pg_keys {
                if seen.insert(k.clone()) {
                    pg_keys.push(k.clone());
                }
            }
        }
        pg_keys.sort();
    }
    let members_direct_dir = run_root.join(RESULT_DIR).join("pipeline_a");
    if !pg_keys.is_empty() {
        std::fs::create_dir_all(members_direct_dir.join("per_pg"))?;
    }
    let uploaded: Vec<Option<String>> = pg_keys
        .par_iter()
        .map(|pg_key| -> Result<Option<String>> {
            let dfs: Vec<DataFrame> = partials
                .iter()
                .filter_map(|p| p.pg_dfs.get(pg_key))
                .filter(|&df| df.height() > 0)
                .cloned()
                .collect();
            if dfs.is_empty() {
                return Ok(None);
            }
            let t_key = std::time::Instant::now();
            let mut combined = concat_vertical_relaxed(dfs)?;
            if combined.height() == 0 {
                return Ok(None);
            }
            let t_concat = t_key.elapsed().as_secs_f64();
            let rows = combined.height();

            // The encode itself parallelizes across row groups inside
            // write_parquet_bytes (the 07-25 rayon::join overlap A/B that
            // measured a wash predates that: back then BOTH sides drained
            // lazily on one thread each, so the join only added
            // contention). Row group layout still matches the single-frame
            // write, so output stays byte-deterministic.
            let t = std::time::Instant::now();
            let bytes = write_parquet_bytes(&mut combined)?;
            files.submit(
                final_staging
                    .join("per_pg")
                    .join(format!("{pg_key}.parquet")),
                bytes,
                false,
            );
            let t_parquet = t.elapsed().as_secs_f64();

            let t = std::time::Instant::now();
            let csv = write_csv_bytes(&mut combined)?;
            let payload = match std::str::from_utf8(&csv)
                .ok()
                .and_then(crate::dpg::encode_member)
            {
                Some((encoded, false)) => encoded,
                _ => csv,
            };
            files.submit(
                members_direct_dir
                    .join("per_pg")
                    .join(format!("{pg_key}.dpg")),
                payload,
                false,
            );
            let t_members = t.elapsed().as_secs_f64();
            eprintln!(
                "emit:   pg {pg_key}: rows={rows} concat={t_concat:.2}s \
                 parquet={t_parquet:.2}s group-members={t_members:.2}s"
            );
            Ok(Some(pg_key.clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut uploaded_pg_keys: Vec<String> = uploaded.into_iter().flatten().collect();
    uploaded_pg_keys.sort();
    // Drain before the promotion walk below scans these dirs (and before
    // the timing print, so the fold time stays honest).
    files.finish()?;
    eprintln!(
        "emit: aggregate+group-members {:.1}s",
        t_agg.elapsed().as_secs_f64()
    );
    let t_promo = std::time::Instant::now();

    // ---- Promotion: copy staging/{RESULT_DIR} -> final,
    // MANIFEST last (list order: LIFO dir queue, files in ls order). ----
    let mut manifest_rels: Vec<String> = Vec::new();
    let prefix = staging.join(RESULT_DIR);
    let staging_root = staging.clone();
    let mut queue: Vec<PathBuf> = vec![prefix];
    while let Some(cur) = queue.pop() {
        let Ok(rd) = std::fs::read_dir(&cur) else {
            continue;
        };
        let mut entries: Vec<_> = rd.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let path = e.path();
            if path.is_dir() {
                queue.push(path);
            } else {
                let rel = path
                    .strip_prefix(&staging_root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                manifest_rels.push(rel);
            }
        }
    }
    for rel in &manifest_rels {
        let src = staging.join(rel);
        let dst = run_root.join(rel);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&src, &dst)?;
    }
    write_file(
        &run_root.join("_meta").join("MANIFEST"),
        manifest_rels.join("\n").as_bytes(),
    )?;
    eprintln!("emit: promotion {:.2}s", t_promo.elapsed().as_secs_f64());

    Ok(EmitSummary {
        ranks_processed,
        pg_keys: uploaded_pg_keys
            .into_iter()
            .filter(|key| key != DIAGNOSTICS_PG)
            .collect(),
        manifest_rels,
        process_groups: merge_process_group_configs(
            partials.iter().flat_map(|partial| &partial.process_groups),
        )?,
    })
}

/// Union by `pg_name`, keeping first-seen (trace) order.
fn merge_process_group_configs<'a>(
    configs: impl Iterator<Item = &'a ProcessGroupConfig>,
) -> Result<Vec<ProcessGroupConfig>> {
    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut merged: Vec<ProcessGroupConfig> = Vec::new();
    for config in configs {
        if config.pg_name.is_empty() {
            continue;
        }
        let Some(&i) = index.get(config.pg_name.as_str()) else {
            index.insert(&config.pg_name, merged.len());
            merged.push(config.clone());
            continue;
        };
        let previous = &mut merged[i];
        anyhow::ensure!(
            previous.pg_desc.is_empty()
                || config.pg_desc.is_empty()
                || previous.pg_desc == config.pg_desc,
            "conflicting descriptions for process group {}",
            config.pg_name
        );
        anyhow::ensure!(
            previous.backend_config.is_empty()
                || config.backend_config.is_empty()
                || previous.backend_config == config.backend_config,
            "conflicting backends for process group {}",
            config.pg_name
        );
        anyhow::ensure!(
            previous.pg_size <= 0 || config.pg_size <= 0 || previous.pg_size == config.pg_size,
            "conflicting sizes for process group {}",
            config.pg_name
        );
        if previous.pg_desc.is_empty() {
            previous.pg_desc.clone_from(&config.pg_desc);
        }
        if previous.backend_config.is_empty() {
            previous.backend_config.clone_from(&config.backend_config);
        }
        if previous.pg_size <= 0 {
            previous.pg_size = config.pg_size;
        }
        previous.ranks.extend_from_slice(&config.ranks);
    }
    for group in &mut merged {
        group.ranks.sort_unstable();
        group.ranks.dedup();
        if group.pg_size <= 0 {
            group.pg_size = group.ranks.len() as i64;
        }
        anyhow::ensure!(
            group.ranks.len() <= group.pg_size as usize,
            "process group {} has {} ranks but reports size {}",
            group.pg_name,
            group.ranks.len(),
            group.pg_size
        );
    }
    Ok(merged)
}

fn trace_pg_axes(configs: &[ProcessGroupConfig]) -> BTreeMap<String, Vec<&ProcessGroupConfig>> {
    crate::analysis::process_group_axes(configs)
}

fn trace_pg_info_json(configs: &[ProcessGroupConfig]) -> Value {
    let axes = trace_pg_axes(configs);
    let groups = axes
        .iter()
        .map(|(axis, configs)| {
            let sizes = configs
                .iter()
                .map(|config| config.pg_size)
                .collect::<std::collections::BTreeSet<_>>();
            let size = sizes.iter().next_back().copied().unwrap_or(0);
            let descriptions = configs
                .iter()
                .map(|config| config.pg_desc.as_str())
                .collect::<std::collections::BTreeSet<_>>();
            let backends = configs
                .iter()
                .map(|config| config.backend_config.as_str())
                .collect::<std::collections::BTreeSet<_>>();
            (
                axis.clone(),
                json!({
                    "alias": axis,
                    "count": configs.len(),
                    "size": size,
                    "pg_sizes": sizes,
                    "pg_descs": descriptions,
                    "backend_configs": backends,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    json!({
        "valid_pgs": axes.keys().collect::<Vec<_>>(),
        "shuffle_piggyback": null,
        "replica_size": null,
        "groups": groups,
    })
}

/// `tree/{run}/pg_map.json`: member ranks of every PG, per axis.
fn pg_map_json(configs: &[ProcessGroupConfig]) -> String {
    let groups = trace_pg_axes(configs)
        .into_iter()
        .map(|(axis, configs)| {
            (
                axis,
                json!({
                    "pg_ranks_map": configs.iter().map(|config| &config.ranks).collect::<Vec<_>>(),
                    "pg_names": configs.iter().map(|config| &config.pg_name).collect::<Vec<_>>(),
                    "pg_ids": configs.iter().map(|config| crate::analysis::pg_group_id(&config.pg_name)).collect::<Vec<_>>(),
                    "pg_descs": configs.iter().map(|config| &config.pg_desc).collect::<Vec<_>>(),
                    "backend_configs": configs.iter().map(|config| &config.backend_config).collect::<Vec<_>>(),
                    "pg_sizes": configs.iter().map(|config| config.pg_size).collect::<Vec<_>>(),
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    dumps(&Value::Object(groups))
}

// ---------------------------------------------------------------------------
// Metadata documents
// ---------------------------------------------------------------------------

/// Resolve a parameterized end-of-run timestamp: 0 (not replayed) → the wall
/// clock at the call site, so each document is stamped when it is actually
/// written (run.json's completed_at after staging promotion, each
/// heartbeat's updated_at at its own write). An explicit nonzero value
/// (deterministic replay, see `RunMeta`) wins unchanged.
pub fn resolve_write_ts_us(ts_us: i64) -> i64 {
    if ts_us != 0 {
        return ts_us;
    }
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// `tree/{run}/_meta/run.json` (final, status 4). `completed_at_us` is the
/// resolved end-of-run stamp — `emit_metadata` resolves it once so run.json
/// and status.json carry the same instant.
pub fn run_json(meta: &RunMeta, completed_at_us: i64) -> String {
    dumps(&json!({
        "run_id": &*meta.run_id,
        "run_name": &*meta.run_name,
        "total_ranks": meta.world_size,
        "status": 4,
        "started_at": meta.started_at_us,
        "completed_at": completed_at_us,
        "error": null,
    }))
}

/// Observed outcome counts for the final status.json. comms_viz fills it with
/// truthful counts — a skipped rank shows as processed < total, a dead
/// worker as shards.failed > 0 — rather than an all-succeeded summary.
/// Readers key by name and tolerate partial counts.
#[derive(Debug, Clone, Copy)]
pub struct StatusCounts {
    /// Ranks that produced a breakdown row (skips + errors excluded).
    pub ranks_processed: i64,
    /// Workers that completed their partial emission.
    pub workers_done: i64,
}

/// `tree/{run}/_meta/status.json` (final).
pub fn final_status_json(
    meta: &RunMeta,
    phases: &serde_json::Map<String, Value>,
    counts: &StatusCounts,
    completed_at_us: i64,
) -> String {
    // elapsed_s: from the caller's resolved end-of-run stamp — the same
    // instant run.json and the phases entries carry (no separate clock) —
    // rounded to one decimal.
    let elapsed_s = ((completed_at_us - meta.started_at_us) as f64) / 1e6;
    let ranks = json!({
        "processed": counts.ranks_processed,
        "total": meta.world_size,
    });
    dumps(&json!({
        "version": 2,
        "run_id": &*meta.run_id,
        "phase": "completed",
        "workers_active": 0,
        "workers_done": counts.workers_done,
        "shard_count": meta.worker_count,
        "shards": {
            "completed": counts.workers_done,
            "in_flight": 0,
            "pending": 0,
            "failed": meta.worker_count - counts.workers_done,
        },
        "ranks": ranks,
        "elapsed_s": (elapsed_s * 10.0).round() / 10.0,
        "phases": phases,
    }))
}

/// Run phase names in transition order (claim → assign → processing →
/// aggregation) — the key set of status.json's `phases` map, part of its
/// schema.
pub const PHASE_NAMES: [&str; 4] = ["claim", "assign", "processing", "aggregation"];

/// Wall clock in epoch micros — the unit every metadata timestamp uses.
pub fn epoch_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// Real phase-boundary instants recorded by the emitting process (µs epoch,
/// `PHASE_NAMES` order). comms_viz's phases are strictly sequential, so each
/// phase's completed_at is the next phase's started_at; aggregation runs
/// until the status write itself, which resolves its end.
#[derive(Debug, Clone, Copy)]
pub struct PhaseTimes {
    pub starts_us: [i64; 4],
}

/// status.json `phases`: per phase, started_at/completed_at (µs) +
/// duration_s + status, keyed by phase name. Replay-pinned runs
/// (explicit --completed-at-us) must stay byte-deterministic, so the pinned
/// interval is attributed to processing (zero-length claim/assign at its
/// start, aggregation at its end); a live completion time (0 = write time,
/// like RunMeta) uses the recorded boundaries, closed out at write time.
pub fn status_phases(
    meta: &RunMeta,
    times: &PhaseTimes,
    completed_at_us: i64,
) -> serde_json::Map<String, Value> {
    // `completed_at_us` is the caller's resolved end-of-run stamp
    // (emit_metadata resolves once), so aggregation closes at the same
    // instant run.json and status.json's elapsed_s carry — no second
    // clock read.
    let bounds: [(i64, i64); 4] = if meta.completed_at_us != 0 {
        let (s, c) = (meta.started_at_us, meta.completed_at_us);
        [(s, s), (s, s), (s, c), (c, c)]
    } else {
        let [t0, t1, t2, t3] = times.starts_us;
        [(t0, t1), (t1, t2), (t2, t3), (t3, completed_at_us)]
    };
    PHASE_NAMES
        .iter()
        .zip(bounds)
        .map(|(name, (s, c))| {
            (
                name.to_string(),
                json!({
                    "started_at": s,
                    "completed_at": c,
                    "duration_s": (c - s) as f64 / 1e6,
                    "status": "completed",
                }),
            )
        })
        .collect()
}

/// Worker heartbeat document body (the `_meta/heartbeats/worker_{i}.json`
/// schema).
#[allow(clippy::too_many_arguments)]
pub fn worker_heartbeat_json(
    worker_index: i64,
    rank_lo: i64,
    rank_hi: i64,
    processed: i64,
    skipped: i64,
    errors: i64,
    state: &str,
    updated_at_us: i64,
    started_at_us: i64,
    duration_s: f64,
) -> String {
    // duration_s rounds to one decimal.
    let duration = (duration_s * 10.0).round() / 10.0;
    dumps(&json!({
        "worker_index": worker_index,
        "rank_lo": rank_lo,
        "rank_hi": rank_hi,
        "assigned": rank_hi - rank_lo,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "state": state,
        "updated_at": updated_at_us,
        "started_at": started_at_us,
        "duration_s": duration,
    }))
}

/// Context for the per-worker `_meta` docs written at partial-write time
/// (the heartbeat is the last file of a worker's partial emission, so it
/// attests to a complete partial).
///
/// Timestamps are the run-level parameterized values (RunMeta); an
/// `updated_at_us` of 0 resolves to each heartbeat's own write instant
/// (`resolve_write_ts_us`). Workers are emulated synchronously, so they
/// all share the run's wall clock and only the final state="completed"
/// snapshot exists — there is no window in which an in-flight
/// state="processing" heartbeat could be observed.
pub struct WorkerMetaCtx<'a> {
    /// `tree/{run}/_meta`.
    pub meta_dir: PathBuf,
    /// rank → error message for ERRORED ranks. A rank with no RankResult
    /// that is absent from this map was skipped (missing trace).
    pub rank_errors: &'a HashMap<i64, String>,
    pub started_at_us: i64,
    pub updated_at_us: i64,
}

/// Write one emulated worker's completed heartbeat:
/// `tree/{run}/_meta/heartbeats/worker_{i}.json` (state="completed"),
/// after the partial's last file. `updated_at_us` 0 resolves to this
/// write's wall clock; duration_s is derived (updated − started), the
/// run's elapsed time at the write.
#[allow(clippy::too_many_arguments)]
pub fn write_worker_heartbeat(
    meta_dir: &Path,
    worker_index: i64,
    rank_lo: i64,
    rank_hi: i64,
    processed: i64,
    skipped: i64,
    errors: i64,
    updated_at_us: i64,
    started_at_us: i64,
) -> Result<()> {
    let updated_at_us = resolve_write_ts_us(updated_at_us);
    let duration_s = (updated_at_us - started_at_us) as f64 / 1e6;
    write_file(
        &meta_dir
            .join("heartbeats")
            .join(format!("worker_{worker_index}.json")),
        worker_heartbeat_json(
            worker_index,
            rank_lo,
            rank_hi,
            processed,
            skipped,
            errors,
            "completed",
            updated_at_us,
            started_at_us,
            duration_s,
        )
        .as_bytes(),
    )
}

/// Worker failure document body (`_meta/failures/worker_{i}.json`,
/// indent-2 pretty JSON — the one non-compact metadata doc).
///
/// comms_viz has no worker-fatal path — every rank error is caught and the
/// run continues — so rank-level errors are recorded here instead:
/// exception_type "RankProcessingError", traceback = one "rank N: <error>"
/// line per errored rank. failures/ is a diagnostic surface for humans and
/// tooling.
pub fn worker_failure_json(
    worker_index: i64,
    exception_type: &str,
    traceback: &str,
    timestamp_us: i64,
) -> String {
    serde_json::to_string_pretty(&json!({
        "worker_index": worker_index,
        "exception_type": exception_type,
        "traceback": traceback,
        "timestamp": timestamp_us,
    }))
    .expect("string-keyed JSON emission cannot fail")
}

/// Write `tree/{run}/_meta/failures/worker_{i}.json`. `timestamp_us` 0
/// resolves to this write's wall clock (parameterized like the heartbeat
/// stamps).
pub fn write_worker_failure(
    meta_dir: &Path,
    worker_index: i64,
    exception_type: &str,
    traceback: &str,
    timestamp_us: i64,
) -> Result<()> {
    let timestamp_us = resolve_write_ts_us(timestamp_us);
    write_file(
        &meta_dir
            .join("failures")
            .join(format!("worker_{worker_index}.json")),
        worker_failure_json(worker_index, exception_type, traceback, timestamp_us).as_bytes(),
    )
}

/// Failure-doc traceback text for a worker's errored ranks. Capped so a
/// mass failure (every rank of a large worker erroring) stays a readable
/// document, not a multi-MB blob.
pub fn rank_errors_traceback(rank_errors: &[(i64, String)]) -> String {
    const MAX_LINES: usize = 100;
    let mut lines: Vec<String> = rank_errors
        .iter()
        .take(MAX_LINES)
        .map(|(r, msg)| format!("rank {r}: {msg}"))
        .collect();
    if rank_errors.len() > MAX_LINES {
        lines.push(format!(
            "... and {} more errored ranks",
            rank_errors.len() - MAX_LINES
        ));
    }
    lines.join("\n")
}

/// `tree/{run}/metadata.json` — {"config": {...}, "pg_info": ...}.
fn metadata_json(meta: &RunMeta, process_groups: &[ProcessGroupConfig]) -> String {
    dumps(&json!({
        "config": {
            "run_name": &*meta.run_name,
            "world_size": meta.world_size,
            "trace_folder_path": meta.recorded_trace_root(),
        },
        "pg_info": trace_pg_info_json(process_groups),
    }))
}

/// The run environment across ranks: per field, in display order, each
/// distinct value with the ranks that recorded it (`"0-7,16"`).
fn environment_json(environments: &BTreeMap<i64, TraceEnvironment>) -> Value {
    // (key, [(value, ranks)]) in first-seen order.
    type ValueRanks<'a> = Vec<(&'a str, Vec<i64>)>;
    let mut fields: Vec<(&str, ValueRanks)> = Vec::new();
    for (&rank, env) in environments {
        for (key, value) in &env.fields {
            let i = match fields.iter().position(|(k, _)| k == key) {
                Some(i) => i,
                None => {
                    fields.push((key, Vec::new()));
                    fields.len() - 1
                }
            };
            let values = &mut fields[i].1;
            match values.iter_mut().find(|(v, _)| v == value) {
                Some((_, ranks)) => ranks.push(rank),
                None => values.push((value, vec![rank])),
            }
        }
    }
    Value::Array(
        fields
            .into_iter()
            .map(|(key, values)| {
                let values: Vec<Value> = values
                    .into_iter()
                    .map(|(value, ranks)| json!({"value": value, "ranks": rank_ranges(&ranks)}))
                    .collect();
                json!({"key": key, "values": values})
            })
            .collect(),
    )
}

/// Ascending ranks as compact ranges: `[0, 1, 2, 5]` -> `"0-2,5"`.
fn rank_ranges(ranks: &[i64]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < ranks.len() {
        let mut j = i;
        while j + 1 < ranks.len() && ranks[j + 1] == ranks[j] + 1 {
            j += 1;
        }
        out.push(if i == j {
            ranks[i].to_string()
        } else {
            format!("{}-{}", ranks[i], ranks[j])
        });
        i = j + 1;
    }
    out.join(",")
}

/// Write the metadata docs for a completed run.
/// (Per-worker heartbeats/failure docs are NOT written here — they are
/// emitted at partial-write time, one per emulated worker.)
pub fn emit_metadata(
    out_root: &Path,
    meta: &RunMeta,
    trace_process_groups: &[ProcessGroupConfig],
    environments: &BTreeMap<i64, TraceEnvironment>,
    counts: &StatusCounts,
    phases: &PhaseTimes,
) -> Result<()> {
    let run_root = out_root.join("tree").join(&meta.run_id);
    let meta_dir = run_root.join("_meta");

    // config.json: run-config dict.
    let mut config_entries = serde_json::Map::new();
    config_entries.insert("run_id".into(), json!(&*meta.run_id));
    config_entries.insert("trace_root".into(), json!(meta.recorded_trace_root()));
    config_entries.insert("world_size".into(), json!(meta.world_size));
    config_entries.insert("worker_count".into(), json!(meta.worker_count));
    config_entries.insert("result_dir".into(), json!(RESULT_DIR));
    config_entries.extend(meta.config_extra.iter().cloned());
    write_file(
        &meta_dir.join("config.json"),
        dumps(&Value::Object(config_entries)).as_bytes(),
    )?;

    // completed_at: resolved here — emit_metadata runs after aggregation +
    // staging promotion — once for the whole call so run.json and
    // status.json agree.
    let completed_at_us = resolve_write_ts_us(meta.completed_at_us);
    write_file(
        &meta_dir.join("run.json"),
        run_json(meta, completed_at_us).as_bytes(),
    )?;

    write_file(
        &meta_dir.join("status.json"),
        final_status_json(
            meta,
            &status_phases(meta, phases, completed_at_us),
            counts,
            completed_at_us,
        )
        .as_bytes(),
    )?;

    let mut clocks = serde_json::Map::new();
    for w in 0..meta.worker_count.max(1) {
        let path = run_root
            .join("_staging/_combined")
            .join(format!("worker_{w}/rank_clocks.json"));
        if path.exists() {
            let rank_clocks: serde_json::Map<String, Value> =
                serde_json::from_slice(&std::fs::read(path)?)?;
            clocks.extend(rank_clocks);
        }
    }
    let mut metadata: Value = serde_json::from_str(&metadata_json(meta, trace_process_groups))?;
    let step = Path::new(meta.recorded_trace_root())
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(|s| {
            s.strip_prefix("step_")
                .or_else(|| s.strip_prefix("iteration_"))
        })
        .and_then(|s| s.parse::<u64>().ok());
    metadata["capture"] = json!({
        // Include provenance: direct runs can reuse the default "local-run" ID.
        "id": dumps(&json!([meta.run_name, meta.recorded_trace_root(), meta.run_id])),
        "run_id": meta.run_id, "run_name": meta.run_name,
        "step": step, "trace_root": meta.recorded_trace_root(),
        "timestamp_unit": "us", "clock_origin_unit": "ns",
        "rank_clock_origins_ns": clocks,
        "pg_scope": ["capture.id", "replica_id", "pg_id"],
    });
    metadata["environment"] = environment_json(environments);
    write_file(&run_root.join("metadata.json"), dumps(&metadata).as_bytes())?;
    write_file(
        &run_root.join("pg_map.json"),
        pg_map_json(trace_process_groups).as_bytes(),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_partial_trace_process_group_membership() {
        let first = ProcessGroupConfig {
            pg_name: "pg_tp".into(),
            pg_desc: "mesh_tp".into(),
            backend_config: String::new(),
            pg_size: 2,
            ranks: vec![0],
        };
        let second = ProcessGroupConfig {
            ranks: vec![1],
            ..first.clone()
        };

        let groups = merge_process_group_configs([&first, &second].into_iter()).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ranks, vec![0, 1]);
        assert_eq!(groups[0].pg_size, 2);
    }

    #[test]
    fn shared_group_names_split_by_rank_set() {
        let result = |rank: i64, ranks: Vec<i64>| {
            let df = DataFrame::new_infer_height(vec![
                Column::new("pg_name".into(), vec!["abc"; 2]),
                Column::new("rank".into(), vec![rank as i32; 2]),
            ])
            .unwrap();
            Some(RankResult {
                base_ns: None,
                rank,
                breakdown_row: Vec::new(),
                pg_dfs: vec![("pp_abc".to_string(), df)],
                process_groups: vec![ProcessGroupConfig {
                    pg_name: "abc".into(),
                    pg_desc: "pp_p2p_round_0".into(),
                    backend_config: "cuda:nccl".into(),
                    pg_size: 2,
                    ranks,
                }],
                environment: TraceEnvironment::default(),
            })
        };
        let mut results = vec![
            result(0, vec![0, 2]),
            result(1, vec![1, 3]),
            result(2, vec![0, 2]),
            result(3, vec![1, 3]),
        ];
        disambiguate_shared_group_names(&mut results).unwrap();
        let names: Vec<(&str, &str)> = results
            .iter()
            .flatten()
            .map(|r| (r.process_groups[0].pg_name.as_str(), r.pg_dfs[0].0.as_str()))
            .collect();
        assert_eq!(
            names,
            [
                ("abc-r0", "pp_abc-r0"),
                ("abc-r1", "pp_abc-r1"),
                ("abc-r0", "pp_abc-r0"),
                ("abc-r1", "pp_abc-r1"),
            ]
        );
        let column = results[1].as_ref().unwrap().pg_dfs[0]
            .1
            .column("pg_name")
            .unwrap();
        assert_eq!(column.str().unwrap().get(0), Some("abc-r1"));
        // Left alone: one rank set, and partial rank sets within the size (a
        // rank that only knows itself, as when groups come from kernel metadata).
        let mut same = vec![result(0, vec![0, 1]), result(1, vec![0, 1])];
        disambiguate_shared_group_names(&mut same).unwrap();
        assert_eq!(same[0].as_ref().unwrap().process_groups[0].pg_name, "abc");
        let mut partial = vec![result(0, vec![0]), result(1, vec![1])];
        disambiguate_shared_group_names(&mut partial).unwrap();
        assert_eq!(
            partial[1].as_ref().unwrap().process_groups[0].pg_name,
            "abc"
        );
    }

    #[test]
    fn environment_groups_ranks_by_value() {
        let env = |driver: &str| TraceEnvironment {
            host: None,
            fields: vec![
                ("gpu", "8x NVIDIA H100".to_string()),
                ("cuda_driver", driver.to_string()),
            ],
        };
        let envs: BTreeMap<i64, TraceEnvironment> = [
            (0, env("13.4")),
            (1, env("13.4")),
            (2, env("12.8")),
            (3, env("13.4")),
            (5, env("13.4")),
            (4, TraceEnvironment::default()),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            environment_json(&envs),
            json!([
                {"key": "gpu", "values": [{"value": "8x NVIDIA H100", "ranks": "0-3,5"}]},
                {"key": "cuda_driver", "values": [
                    {"value": "13.4", "ranks": "0-1,3,5"},
                    {"value": "12.8", "ranks": "2"},
                ]},
            ])
        );
    }

    #[test]
    fn trace_topology_metadata_preserves_physical_groups() {
        let configs = vec![
            ProcessGroupConfig {
                pg_name: "17".into(),
                pg_desc: "mesh_tp".into(),
                backend_config: "cuda:nccl".into(),
                pg_size: 2,
                ranks: vec![0, 1],
            },
            ProcessGroupConfig {
                pg_name: "29".into(),
                pg_desc: "mesh_tp".into(),
                backend_config: "cuda:nccl".into(),
                pg_size: 2,
                ranks: vec![2, 3],
            },
            ProcessGroupConfig {
                pg_name: "41".into(),
                pg_desc: "mesh_cp".into(),
                backend_config: "cuda:nccl".into(),
                pg_size: 4,
                ranks: vec![0, 1, 2, 3],
            },
        ];
        let map: Value = serde_json::from_str(&pg_map_json(&configs)).unwrap();
        assert_eq!(map["tp"]["pg_names"], json!(["17", "29"]));
        assert_eq!(map["tp"]["pg_ids"], json!(["17", "29"]));
        assert_eq!(map["tp"]["pg_ranks_map"], json!([[0, 1], [2, 3]]));
        assert_eq!(map["cp"]["pg_names"], json!(["41"]));

        let info = trace_pg_info_json(&configs);
        assert_eq!(info["groups"]["tp"]["count"], 2);
        assert_eq!(info["groups"]["cp"]["size"], 4);
        assert!(info.get("fsdp_alias").is_none(), "no FSDP axis is inferred");

        let tmp = tempfile::tempdir().unwrap();
        let counts = StatusCounts {
            ranks_processed: 4,
            workers_done: 1,
        };
        let phases = PhaseTimes {
            starts_us: [meta().started_at_us; 4],
        };
        emit_metadata(
            tmp.path(),
            &meta(),
            &configs,
            &BTreeMap::new(),
            &counts,
            &phases,
        )
        .unwrap();
        let metadata: Value = serde_json::from_slice(
            &std::fs::read(tmp.path().join("tree/r0/metadata.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["pg_info"]["valid_pgs"], json!(["cp", "tp"]));
        assert_eq!(
            metadata["capture"]["run_id"], "r0",
            "trace pg_info must not clobber the capture block"
        );
    }

    #[test]
    fn trace_topology_emits_physical_group_member() {
        let tmp = tempfile::tempdir().unwrap();
        let mut partial = decoded_with_pg("tp_abcd", &[0, 1]);
        partial.process_groups.push(ProcessGroupConfig {
            pg_name: "abcd".into(),
            pg_desc: "mesh_tp".into(),
            backend_config: "cuda:nccl".into(),
            pg_size: 2,
            ranks: vec![0, 1],
        });
        aggregate_decoded(tmp.path(), &meta(), &[partial]).unwrap();

        let per_pg = tmp.path().join("tree/r0/results/pipeline_a/per_pg");
        assert_eq!(per_pg_members(&per_pg), vec!["tp_abcd.dpg"]);
        let payload = std::fs::read(per_pg.join("tp_abcd.dpg")).unwrap();
        assert_eq!(crate::dpg::member_to_text(&payload), "rank,dur\n0,1\n1,1\n");
    }

    /// The parallel row-group encode must reproduce the stock serial
    /// ParquetWriter byte-for-byte — including across MULTIPLE row groups
    /// (>512² rows), the case the small snapshot fixtures never reach.
    /// Mixed dtypes cover both page-encoding families: Plain (f64, with
    /// nulls) and RleDictionary (i64, strings).
    #[test]
    fn parallel_parquet_encode_matches_serial_writer_bytes() {
        const N: usize = 512 * 512 * 2 + 4321; // 2 full row groups + a tail
        let ranks: Vec<i64> = (0..N as i64).map(|i| i % 977).collect();
        let ts: Vec<Option<f64>> = (0..N)
            .map(|i| {
                if i % 251 == 0 {
                    None
                } else {
                    Some(2578675544695.438 + i as f64 * 0.125)
                }
            })
            .collect();
        let names: Vec<String> = (0..N).map(|i| format!("kern_{}", i % 137)).collect();
        let mut df = DataFrame::new_infer_height(vec![
            Column::new("rank".into(), ranks),
            Column::new("ts".into(), ts),
            Column::new("name".into(), names),
        ])
        .unwrap();

        let mut serial = Vec::new();
        ParquetWriter::new(&mut serial)
            .with_compression(ParquetCompression::Zstd(None))
            .with_statistics(StatisticsOptions::default())
            .finish(&mut df.clone())
            .unwrap();

        let parallel = write_parquet_bytes(&mut df).unwrap();
        assert_eq!(serial.len(), parallel.len());
        assert!(
            serial == parallel,
            "parallel parquet bytes diverge from serial writer"
        );
    }

    fn meta() -> RunMeta {
        RunMeta {
            run_id: "r0".into(),
            run_name: "job".into(),
            world_size: 8,
            trace_root: "/tr".into(),
            source_trace_root: None,
            started_at_us: 1,
            completed_at_us: 2,
            worker_count: 1,
            config_extra: vec![],
        }
    }

    /// run.json; exact bytes pinned (sorted key order).
    #[test]
    fn run_doc_shape() {
        let m = meta();
        assert_eq!(
            run_json(&m, m.completed_at_us),
            "{\"completed_at\":2,\"error\":null,\"run_id\":\"r0\",\
             \"run_name\":\"job\",\"started_at\":1,\"status\":4,\
             \"total_ranks\":8}"
        );
    }

    fn provenance_meta(source: Option<&str>) -> RunMeta {
        RunMeta {
            run_id: "test-run".into(),
            run_name: "test-job".into(),
            world_size: 8,
            trace_root: "/stage/step_000000000106".into(),
            source_trace_root: source.map(str::to_string),
            started_at_us: 1_700_000_000_000_000,
            completed_at_us: 1_700_000_001_500_000,
            worker_count: 1,
            config_extra: Vec::new(),
        }
    }

    const REMOTE_URI: &str = "s3://bucket/outputs/j/profiling/traces/step_000000000106";

    /// Some(source) → both provenance sites record the SUBMITTED root,
    /// never the staged local dir.
    #[test]
    fn source_override_lands_in_both_docs() {
        let m = provenance_meta(Some(REMOTE_URI));
        let md = metadata_json(&m, &[]);
        assert!(md.contains(&format!("\"trace_folder_path\":\"{REMOTE_URI}\"")));
        assert!(!md.contains("/stage/"));

        let tmp = tempfile::tempdir().unwrap();
        let counts = StatusCounts {
            ranks_processed: 8,
            workers_done: 1,
        };
        // Replay-pinned meta (completed_at_us != 0): phase instants unused.
        let phases = PhaseTimes {
            starts_us: [m.started_at_us; 4],
        };
        emit_metadata(tmp.path(), &m, &[], &BTreeMap::new(), &counts, &phases).unwrap();
        let config =
            std::fs::read_to_string(tmp.path().join("tree/test-run/_meta/config.json")).unwrap();
        assert!(config.contains(&format!("\"trace_root\":\"{REMOTE_URI}\"")));
        assert!(!config.contains("/stage/"));
    }

    /// None → every site records `trace_root` itself.
    #[test]
    fn absent_override_keeps_trace_root() {
        let m = provenance_meta(None);
        assert!(
            metadata_json(&m, &[]).contains("\"trace_folder_path\":\"/stage/step_000000000106\"")
        );

        let tmp = tempfile::tempdir().unwrap();
        let counts = StatusCounts {
            ranks_processed: 8,
            workers_done: 1,
        };
        // Replay-pinned meta (completed_at_us != 0): phase instants unused.
        let phases = PhaseTimes {
            starts_us: [m.started_at_us; 4],
        };
        emit_metadata(tmp.path(), &m, &[], &BTreeMap::new(), &counts, &phases).unwrap();
        let config =
            std::fs::read_to_string(tmp.path().join("tree/test-run/_meta/config.json")).unwrap();
        assert!(config.contains("\"trace_root\":\"/stage/step_000000000106\""));
    }

    fn decoded_with_pg(key: &str, ranks: &[i64]) -> DecodedPartial {
        let df = DataFrame::new_infer_height(vec![
            Column::new("rank".into(), ranks.to_vec()),
            Column::new("dur".into(), vec![1i64; ranks.len()]),
        ])
        .unwrap();
        DecodedPartial {
            breakdown_df: None,
            pg_keys: vec![key.to_string()],
            pg_dfs: HashMap::from([(key.to_string(), df)]),
            process_groups: Vec::new(),
        }
    }

    fn per_pg_members(per_pg: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(per_pg)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".dpg"))
            .collect();
        names.sort();
        names
    }

    #[test]
    fn diagnostics_member_does_not_require_topology() {
        let tmp = tempfile::tempdir().unwrap();
        let summary = aggregate_decoded(
            tmp.path(),
            &meta(),
            &[decoded_with_pg("diagnostics_0", &[0, 1])],
        )
        .unwrap();

        let per_pg = tmp.path().join("tree/r0/results/pipeline_a/per_pg");
        assert!(per_pg.join("diagnostics_0.dpg").is_file());
        assert!(summary.pg_keys.is_empty());
    }

    /// The breakdown table's pinned schema semantics: sorted column order,
    /// rank i64, totals f64, sentinel Int(-1) as -1.0, absent keys as
    /// null.
    #[test]
    fn rows_to_df_schema_and_values() {
        let rows: Vec<Vec<(&str, BreakdownValue)>> = vec![
            vec![
                ("rank", BreakdownValue::Int(0)),
                ("comp_total", BreakdownValue::Float(12.5)),
                ("mp_all_gather_total", BreakdownValue::Float(3.0)),
            ],
            // Sentinel row: BREAKDOWN_COLUMNS-only, Int(-1) everywhere.
            vec![
                ("rank", BreakdownValue::Int(-1)),
                ("comp_total", BreakdownValue::Int(-1)),
            ],
        ];
        let mut df = rows_to_df(&rows).unwrap();
        assert_eq!(
            df.get_column_names()
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>(),
            vec!["comp_total", "mp_all_gather_total", "rank"]
        );
        assert_eq!(df.column("rank").unwrap().dtype(), &DataType::Int64);
        assert_eq!(df.column("comp_total").unwrap().dtype(), &DataType::Float64);
        assert_eq!(
            df.column("comp_total").unwrap().f64().unwrap().get(1),
            Some(-1.0)
        );
        assert_eq!(
            df.column("mp_all_gather_total")
                .unwrap()
                .f64()
                .unwrap()
                .get(1),
            None
        );
        // The page-facing CSV text form is pinned (the page's breakdown
        // parser reads it).
        let bytes = write_csv_bytes(&mut df).unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "comp_total,mp_all_gather_total,rank\n12.5,3.0,0\n-1.0,,-1\n"
        );
    }
}

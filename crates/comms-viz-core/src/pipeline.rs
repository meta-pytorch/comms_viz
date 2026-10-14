// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Single-host run pipeline.
//!
//! One task per rank on a scoped thread pool; each task runs
//! read → walk → convert → analyze sequentially. Ranks are the parallelism
//! unit — it keeps per-rank memory bounded — and results land in a slot
//! array and are emitted in rank order, the deterministic reference
//! ordering (same inputs + same binary = identical artifact bytes).

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::Context;
use anyhow::Result;

use crate::annotations::AnnotationMode;
use crate::annotations::AnnotationSource;
use crate::emit::RankResult;
use crate::emit::RunMeta;
use crate::legacy::tree;
use crate::trace::Format;
use crate::trace::discover::TraceDiscovery;
use crate::trace::{self};

pub struct RunConfig {
    pub trace_root: PathBuf,
    /// Trace-source URI recorded in the artifacts (metadata.json
    /// trace_folder_path, the flat 1009 doc, _meta/config.json trace_root)
    /// in place of `trace_root`, for when `trace_root` is a temporary copy
    /// of the traces. File access always uses `trace_root`.
    /// None → record `trace_root` itself.
    pub source_trace_root: Option<String>,
    pub out_root: PathBuf,
    pub run_id: String,
    /// Descriptive run name shown in the report; empty when unnamed.
    pub run_name: String,
    pub world_size: Option<i64>,
    /// Where CUDA-graph annotations come from. `Auto` must be resolved by the
    /// caller before `run` (see `annotations::AnnotationMode`).
    pub annotation_mode: AnnotationMode,
    /// Input trace format. `FormatChoice::Auto` must be resolved by the caller
    /// before `run` (see `trace::detect`).
    pub trace_format: Format,
    pub recording_step: Option<i64>,
    pub write_repaired_traces: bool,
    pub threads: usize,
    pub started_at_us: i64,
    /// 0 = unparameterized: every consuming doc is stamped at its own write
    /// time (`emit::resolve_write_ts_us`); nonzero = a pinned timestamp
    /// (the deterministic snapshot suite), wins verbatim everywhere.
    pub completed_at_us: i64,
    pub verbose: bool,
    /// Debug: run the tree-free walk path AND the tree path per rank and
    /// assert identical converted columns.
    pub verify_walks: bool,
}

/// Worker-partition sizing tiers by world size. The partition determines the
/// per-worker heartbeat/status/partial layout, i.e. artifact shape.
pub fn workers_for_world_size(world_size: i64) -> i64 {
    if world_size < 1024 {
        50
    } else if world_size < 10240 {
        100
    } else if world_size < 20480 {
        200
    } else if world_size < 30720 {
        300
    } else if world_size < 40960 {
        400
    } else if world_size < 51200 {
        500
    } else {
        600
    }
}

pub struct RunSummary {
    pub world_size: i64,
    pub processed: usize,
    pub skipped: usize,
    pub errors: usize,
    pub skipped_ranks: Vec<i64>,
    pub errored_ranks: Vec<i64>,
    pub pg_keys: Vec<String>,
}

enum RankOutcome {
    Row(Box<RankResult>),
    Skipped,
    /// Error message (anyhow chain or "panicked") — carried so the
    /// per-worker `_meta/failures` doc can report it.
    Errored(String),
}

/// Rank-loop thread-second accounting (logging only). Sums across workers
/// can exceed wall time.
struct WorkStats {
    t0: std::time::Instant,
    work_ns: std::sync::atomic::AtomicU64,
}

impl WorkStats {
    fn new() -> Self {
        WorkStats {
            t0: std::time::Instant::now(),
            work_ns: Default::default(),
        }
    }

    fn log(&self, threads: usize) {
        let work = self.work_ns.load(Ordering::Relaxed) as f64 / 1e9;
        eprintln!(
            "rank loop: wall={:.1}s threads={} work={:.1}s (thread-seconds)",
            self.t0.elapsed().as_secs_f64(),
            threads,
            work,
        );
        // Per-stage split of the same thread-seconds (crate::timing);
        // whatever the named stages do not cover shows up as the residual.
        let stages = crate::timing::snapshot();
        let sum: f64 = stages.iter().map(|(_, v)| v).sum();
        let mut line = String::new();
        for (n, v) in &stages {
            use std::fmt::Write as _;
            let _ = write!(line, " {n}={v:.1}");
        }
        eprintln!(
            "stages(thread-s):{line} residual={:.1} (stage sum {:.1} of {:.1} accounted)",
            (work - sum).max(0.0),
            sum,
            work,
        );
        {
            use std::sync::atomic::Ordering::Relaxed;
            let noop = crate::legacy::splice::NOOP_RANKS.load(Relaxed);
            let edited = crate::legacy::splice::EDITED_RANKS.load(Relaxed);
            if noop + edited > 0 {
                eprintln!("splice: no-edit ranks={noop} spliced ranks={edited}");
            }
        }
    }
}

/// Byte-bounded off-thread writer for per-rank merged-trace artifacts.
///
/// Off-thread because inline `fs::write` of the full trace copies blocked
/// workers under btrfs dirty-page throttling — measured at up to a quarter
/// of the thread-second budget at full scale. Workers hand `(name, bytes)`
/// to a small writer pool and return to parsing.
///
/// Invariants:
/// - byte-bounded queue: a stalled disk backpressures the workers
///   (visible as the `artifact_q` stage) instead of ballooning RSS;
/// - write errors surface at `finish()` and fail the run — with the
///   inline write an ENOSPC-class failure errored ranks one by one,
///   now it fails the run with the same message, earlier.
struct ArtifactSink {
    dir: PathBuf,
    pool: crate::writer_pool::WriterPool,
}

/// Floor for the in-flight artifact byte cap. A fixed 256 MiB was smaller
/// than ONE admission wave (112 workers × ~3 MB spliced-JSON artifacts on
/// one large run), so every completion burst parked producers —
/// 931–1341 thread-s/shard of `artifact_q` — while the device sat at
/// PSI-io ≈ 0.3%. The cap now scales with the worker count (one generous
/// artifact per worker, ×2 for the next wave) and is clamped against live
/// memory headroom so small hosts keep the old bound.
const ARTIFACT_QUEUE_CAP_FLOOR_BYTES: usize = 256 * 1024 * 1024;
const ARTIFACT_QUEUE_PER_WORKER_BYTES: usize = 64 * 1024 * 1024;

fn artifact_queue_cap_bytes(workers: usize) -> usize {
    let headroom_cap = (crate::memgate::effective_headroom() / 8)
        .min(8 << 30)
        .max(ARTIFACT_QUEUE_CAP_FLOOR_BYTES as u64) as usize;
    workers
        .saturating_mul(ARTIFACT_QUEUE_PER_WORKER_BYTES)
        .clamp(ARTIFACT_QUEUE_CAP_FLOOR_BYTES, headroom_cap)
}

impl ArtifactSink {
    fn new(dir: PathBuf, writers: usize, cap_bytes: usize) -> Result<Self> {
        std::fs::create_dir_all(&dir).with_context(|| format!("create artifact dir {dir:?}"))?;
        Ok(Self {
            dir,
            pool: crate::writer_pool::WriterPool::new(writers, cap_bytes),
        })
    }

    /// Queue one artifact. Blocks only when the in-flight byte cap is
    /// reached (disk slower than the workers); that time is accounted as
    /// the `artifact_q` stage. via_tmp: a concurrent reader of the
    /// output directory must never see a partial artifact.
    fn submit(&self, name: String, artifact: Vec<u8>) {
        let _t = crate::timing::Timer::new(crate::timing::Stage::ArtifactQueue);
        self.pool
            .submit(self.dir.join(name), artifact, /*via_tmp=*/ true);
    }

    /// Gauge of artifact bytes alive in the writer pool, for the memory
    /// gate's ledger.
    fn live_bytes_gauge(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.pool.live_bytes_gauge()
    }

    /// Charge `bytes` to the live-bytes gauge for the guard's lifetime:
    /// keeps the permit-free splice's transients (spans, plan, gz output)
    /// visible to the memory gate's ledger without holding a width slot.
    fn transient_charge(&self, bytes: u64) -> TransientCharge {
        let gauge = self.pool.live_bytes_gauge();
        gauge.fetch_add(bytes, Ordering::Relaxed);
        TransientCharge { gauge, bytes }
    }

    /// Drain, join the writers, and surface any write failures.
    fn finish(self) -> Result<()> {
        self.pool
            .finish()
            .with_context(|| format!("merged-trace artifacts under {:?}", self.dir))
    }
}

enum Merged {
    /// No CUDA-graph handling applied: replay bytes pass through verbatim.
    Passthrough,
    /// A merge pass ran; artifact = serialize_tree(tree).
    Tree(serde_json::Value),
    /// Inline-annotated JSON: artifact bytes already produced by the splice
    /// writer (walk edits substituted into the original document), gzipped.
    Spliced(Vec<u8>),
    /// Fused-path artifact, deferred: everything the splice writer needs,
    /// so the splice+gzip runs AFTER the memory permit is released. The
    /// splice reads only the worker's retained scratch (resident either
    /// way) plus these structures, so holding the permit across it bought
    /// no safety — it only stretched the admission cycle by the tree_ser
    /// stage (~40% of permit-held time on the 09-03 24k-rank profile),
    /// which is width lost exactly where memory-bound runs are width-starved.
    SplicePending(Box<SplicePending>),
    /// Tree-free walk path; no artifact serializable from here. Reached
    /// with an artifact requested only for a non-tree format whose
    /// annotations came from a reassigning side channel — the write site
    /// skips that rank's artifact rather than copying bytes the analysis
    /// did not see.
    RowWalk,
    /// Non-tree (perfetto-native) format, self-annotated: the replay bytes
    /// ARE the merged representation — artifact = verbatim copy.
    Replay,
}

/// RAII for [`ArtifactSink::transient_charge`].
struct TransientCharge {
    gauge: std::sync::Arc<std::sync::atomic::AtomicU64>,
    bytes: u64,
}

impl Drop for TransientCharge {
    fn drop(&mut self) {
        self.gauge.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// A rank's artifact on its way to the writer pool: finished bytes, or a
/// splice that still has to run (permit-free) against the worker's scratch.
enum ArtifactPayload {
    Ready(Vec<u8>),
    Splice(Box<SplicePending>),
}

/// Deferred splice-writer input (see [`Merged::SplicePending`]).
struct SplicePending {
    /// Parse-captured element spans; `None` falls back to the sparse scan.
    spans: Option<crate::legacy::splice::ElemSpans>,
    plan: crate::rows::WalkPlan,
    ann_args: crate::legacy::splice::RenderedAnnArgs,
    /// Kept only for the rare splice-failure tree fallback.
    full_annotations: Option<std::sync::Arc<crate::annotations::Annotations>>,
    noop: bool,
}

/// Converted trace: the lean column view for the single-pass analysis.
enum Converted {
    Columns(crate::rows::ConvertedColumns),
    /// Fused fast path: post-walk rows + sort permutation, no gather.
    View(crate::rows::WalkedView),
}

fn process_groups_from_event_metadata<'a>(
    rank: i64,
    events: impl Iterator<Item = (&'a str, &'a str, i64)>,
) -> Result<Vec<trace::ProcessGroupConfig>> {
    let mut groups = std::collections::BTreeMap::<&str, trace::ProcessGroupConfig>::new();
    for (pg_name, pg_desc, pg_size) in events {
        if pg_name.is_empty() {
            continue;
        }
        let group = groups
            .entry(pg_name)
            .or_insert_with(|| trace::ProcessGroupConfig {
                pg_name: pg_name.to_string(),
                pg_desc: pg_desc.to_string(),
                backend_config: String::new(),
                pg_size: pg_size.max(0),
                ranks: vec![rank],
            });
        if !pg_desc.is_empty() {
            anyhow::ensure!(
                group.pg_desc.is_empty() || group.pg_desc == pg_desc,
                "conflicting process-group descriptions for {pg_name}"
            );
            if group.pg_desc.is_empty() {
                group.pg_desc.push_str(pg_desc);
            }
        }
        if pg_size > 0 {
            anyhow::ensure!(
                group.pg_size == 0 || group.pg_size == pg_size,
                "conflicting process-group sizes for {pg_name}"
            );
            group.pg_size = pg_size;
        }
    }
    Ok(groups.into_values().collect())
}

/// PG inventory from the PG fields on comm kernels, for traces without
/// `distributedInfo.pg_config` (pftrace, or JSON from an older torch). Each
/// rank only knows itself as a member; the aggregator merges memberships.
fn kernel_process_groups(
    converted: &Converted,
    rank: i64,
) -> Result<Vec<trace::ProcessGroupConfig>> {
    match converted {
        Converted::Columns(cols) => process_groups_from_event_metadata(
            rank,
            (0..cols.ts.len()).map(|i| (cols.pg_name(i), cols.pg_desc(i), cols.group_size[i])),
        ),
        Converted::View(view) => process_groups_from_event_metadata(
            rank,
            (0..view.len()).map(|i| (view.pg_name(i), view.pg_desc(i), view.group_size(i))),
        ),
    }
}

/// Splice-writer invocation for the verify path (raw bytes for the oracle).
fn splice_raw(
    doc: &[u8],
    spans: &crate::legacy::splice::DocSpans,
    plan: &crate::rows::WalkPlan,
    ann_args: &crate::legacy::splice::RenderedAnnArgs,
) -> Result<Vec<u8>> {
    crate::legacy::splice::splice(doc, spans, &splice_plan(plan, ann_args))
}

/// Run a deferred fused-path splice (see [`Merged::SplicePending`]): called
/// AFTER the rank's memory permit is dropped, so the tree_ser stage no
/// longer stretches the admission cycle. `doc` is the worker's retained
/// scratch — pristine since the scan parse, resident with or without a
/// permit. The transient structures this stage adds on top of it (spans,
/// plan, gz output) are charged to the writer-pool gauge the ledger already
/// reads, via the caller's `transient_charge` guard.
///
/// A splice failure falls back to the tree path exactly as before, except
/// the serde DOM — the one genuinely heavy allocation here — is built under
/// a FRESH permit at the tree-path estimate, since this thread no longer
/// holds one.
fn finish_splice(
    doc: &[u8],
    p: SplicePending,
    rank: i64,
    replay_path: &std::path::Path,
    mem_gate: &crate::memgate::MemGate,
) -> Result<Vec<u8>> {
    let _t = crate::timing::Timer::new(crate::timing::Stage::TreeSer);
    let gz = if p.noop {
        crate::legacy::splice::minify_gzip(doc)
    } else {
        // Parse-captured spans need no structural pass at all; without
        // them (pftrace, >u32 docs) the sparse scan details the edited
        // rows — dropped rows are skipped whole and everything else is
        // copied whole.
        match p.spans {
            Some(eb) => Ok(eb.into()),
            None => {
                let mut detail: Vec<u32> = p
                    .plan
                    .moved
                    .iter()
                    .copied()
                    .chain(p.plan.fixed.iter().map(|&(r, _)| r))
                    .chain(p.plan.retagged.iter().map(|&(r, _, _)| r))
                    .chain(p.plan.retagged_ac2g.iter().map(|&(r, _)| r))
                    .collect();
                detail.sort_unstable();
                detail.dedup();
                crate::legacy::splice::scan_sparse(doc, &detail)
            }
        }
        .and_then(|spans| {
            crate::legacy::splice::splice_gzip(doc, &spans, &splice_plan(&p.plan, &p.ann_args))
        })
    };
    match gz {
        Ok(gz) => Ok(gz),
        Err(e) => {
            // The scan parse never touched `doc`, so the tree path can
            // serve this rank's artifact.
            eprintln!("rank {rank}: splice failed ({e}) — tree fallback");
            let est = crate::memgate::estimate_rank_cost(replay_path, true);
            let _permit = mem_gate.admit(est);
            let mut tree = tree::parse_tree(doc)?;
            match &p.full_annotations {
                Some(full) => {
                    tree::annotate_tree(&mut tree, full, crate::cudagraph::DEFAULT_STREAM)
                }
                None => tree::inline_tree(&mut tree, crate::cudagraph::DEFAULT_STREAM),
            }
            tree::serialize_tree(&tree, true, doc.len())
        }
    }
}

fn splice_plan<'a>(
    plan: &'a crate::rows::WalkPlan,
    ann_args: &'a crate::legacy::splice::RenderedAnnArgs,
) -> crate::legacy::splice::SplicePlan<'a> {
    crate::legacy::splice::SplicePlan {
        keep: &plan.keep,
        moved: &plan.moved,
        fixed: &plan.fixed,
        appended: &plan.appended,
        overlap_stream: crate::cudagraph::DEFAULT_STREAM + 1,
        retagged: &plan.retagged,
        retagged_ac2g: &plan.retagged_ac2g,
        ann_args,
    }
}

/// Rendered annotation args for the plan's retagged gnids (reassign splice
/// input); empty when the source doesn't reassign.
fn rendered_ann_args(
    plan: &crate::rows::WalkPlan,
    full_annotations: Option<&crate::annotations::Annotations>,
) -> crate::legacy::splice::RenderedAnnArgs {
    match full_annotations {
        Some(full) if !plan.retagged.is_empty() => {
            let wanted: rustc_hash::FxHashSet<i64> =
                plan.retagged.iter().map(|&(_, g, _)| g).collect();
            crate::legacy::pkl::render_ann_args(full, &wanted)
        }
        _ => Default::default(),
    }
}

/// One rank's merge+convert result.
struct MergeOutcome {
    converted: Converted,
    merged: Merged,
}

/// Merge + convert one rank's trace.
///
/// The CUDA-graph dimension enters only through `source`: this function asks
/// the trace whether it annotates itself, asks [`AnnotationSource`] what that
/// resolves to, and from then on deals in "walk or not / reassign or not /
/// these digests". It never learns where the annotations came from — that is
/// what makes the pickle side channel removable without touching the
/// converter, the walks, or the analyzer.
///
/// Three shapes, in descending preference:
///   1. **row walk** (the fast path) — walk passes applied straight to the
///      parsed rows (`cudagraph.rs`), so the `serde_json` tree is never
///      built. Available whenever no recording merge is needed and the
///      annotations don't reassign; an inline-annotated JSON artifact is
///      served from here by the splice writer (`legacy/splice.rs`), which
///      rewrites the original document bytes with the walk's edits.
///   2. **tree** — the `serde_json` tree walks, needed for the recording
///      merge and the pkl-annotated (reassigning) artifact. Values identical
///      to the row walk (that is what `--verify-walks` asserts); see
///      legacy/tree.rs.
///   3. **passthrough** — no CUDA-graph handling applies; convert as-is.
fn merge_and_convert(
    decompressed: &mut [u8],
    recording_bytes: Option<Vec<u8>>,
    rank: i64,
    format: Format,
    source: &AnnotationSource,
    artifact_needed: bool,
    verify_walks: bool,
) -> Result<MergeOutcome> {
    // Does this trace carry its own annotations? Format-specific probe.
    let ann = source.for_trace(trace::probe(format, decompressed).self_annotated);

    // The tree path below (and the passthrough converter) is `serde_json`, so
    // a format with no Chrome-JSON document cannot serve it. Fail with the
    // reason rather than silently degrading. The merged-trace artifact is
    // NOT such a reason: a non-tree trace that annotates itself needs no
    // merge — its replay bytes are already the merged representation, so
    // the artifact is a verbatim copy (`Merged::Replay`).
    if !format.supports_tree() && recording_bytes.is_some() {
        anyhow::bail!(
            "trace_format={} cannot serve this run: the CUDA-graph recording \
             merge requires a serde_json event tree, which only the json \
             format materializes",
            format.as_str(),
        );
    }
    // A reassigning (pkl side-channel) source can't be baked into a
    // non-tree artifact; RowWalk makes the write site skip that rank's
    // artifact instead of copying bytes the analysis did not see.
    let replay_artifact = artifact_needed && !format.supports_tree() && !ann.reassign();

    // Annotated JSON with an artifact request takes the row walk too —
    // inline AND pkl-reassigning: the merged-trace artifact comes from the
    // splice writer (the walk's edits substituted into the original
    // document bytes), so the serde tree is never built. Building the
    // whole document as a `serde_json::Value` for the reassign artifact
    // cost ~2.5 GB live per in-flight rank (TREE_GZ_RANK_FACTOR 25 vs 5)
    // and ~7 s/rank — the width and service-time wall that kept 24k-rank
    // pkl jobs at ~6 ranks/s. The structural scan is
    // DEFERRED until the walk plan exists: a no-edit plan needs no spans
    // at all (the artifact is exactly the minified document), and a scan
    // failure lands in the same per-rank tree fallback as a splice
    // failure. Only --verify-walks scans up front — its artifact oracle
    // wants the spans regardless. The recording merge (arbitrary args
    // rewrites from a second trace) keeps the tree path.
    let want_spliced =
        ann.walks() && artifact_needed && format.supports_tree() && recording_bytes.is_none();
    let splice_spans = if want_spliced && verify_walks {
        match crate::legacy::splice::scan(decompressed) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("rank {rank}: splice scan failed ({e}) — using the tree path");
                None
            }
        }
    } else {
        None
    };
    // Row-walk admission + pristine-buffer requirement for the splice.
    // In verify mode a failed up-front scan falls back to the tree path,
    // exactly as before.
    let fused_artifact = if verify_walks {
        splice_spans.is_some()
    } else {
        want_spliced
    };

    // ---- 1. Tree-free row walk ----
    //
    // The JSON scan parse never mutates its input, so `decompressed` stays
    // pristine for the byte splice with no defensive copy. The live graph
    // ids come from the parsed rows themselves (every event's gnid is
    // extracted anyway), replacing the 38MB byte scan the tree path still
    // needs.
    if ann.walks()
        && recording_bytes.is_none()
        && (!artifact_needed || !format.supports_tree() || fused_artifact)
    {
        let verify_tree = if verify_walks {
            Some(tree::parse_tree(decompressed)?)
        } else {
            None
        };
        let (parsed, bounds) =
            trace::parse_rows_bounds(format, decompressed, fused_artifact && !verify_walks)?;
        // Only a reassigning source consults the live gids; the inline path
        // skips the whole Gids pass.
        let live = if ann.needs_live_gids() {
            let _t = crate::timing::Timer::new(crate::timing::Stage::Gids);
            crate::rows::live_gids(&parsed)
        } else {
            Default::default()
        };
        // The row walk consumes DIGESTED annotations (five fields per
        // annotation). The full map is needed only where the ARTIFACT
        // bakes annotation args in: the reassigning splice (and the tree
        // side of --verify-walks, which derives the digests from that same
        // map so both sides read identical content). Transient per rank —
        // the 432 GB retention hazard was the process-lifetime cache, not
        // this fetch.
        let need_full = ann.reassign() && (verify_walks || fused_artifact);
        let (annotations, full_annotations) = if need_full {
            let full = ann.full(rank, &live)?;
            let d = std::sync::Arc::new(crate::legacy::pkl::digests_from_annotations(&full));
            (d, Some(full))
        } else {
            (ann.digests(rank, &live)?, None)
        };
        let reassign = ann.reassign();
        let verify_side = verify_tree.map(|mut tree| {
            match &full_annotations {
                Some(full) => {
                    tree::annotate_tree(&mut tree, full, crate::cudagraph::DEFAULT_STREAM)
                }
                None => tree::inline_tree(&mut tree, crate::cudagraph::DEFAULT_STREAM),
            }
            let cols = crate::legacy::json::convert_serde_to_columns(&tree);
            (cols, tree)
        });
        if verify_walks {
            // Verification needs both sides materialized for the bitwise
            // column comparison; analysis then runs on the gathered columns.
            let (cols, plan) = crate::rows::convert_parsed_walked_plan(
                parsed,
                &annotations,
                crate::cudagraph::DEFAULT_STREAM,
                reassign,
            )?;
            let mut merged = if replay_artifact {
                Merged::Replay
            } else {
                Merged::RowWalk
            };
            if let Some((tree_cols, tree)) = &verify_side {
                crate::legacy::json::assert_columns_equal(&cols, tree_cols)?;
                // Artifact oracle: the spliced document must parse to the
                // same value as the tree serializer's output.
                if let Some(spans) = &splice_spans {
                    let ann_args = rendered_ann_args(&plan, full_annotations.as_deref());
                    let raw = splice_raw(decompressed, spans, &plan, &ann_args)?;
                    let tree_raw = tree::serialize_tree(tree, false, decompressed.len())?;
                    let a: serde_json::Value = serde_json::from_slice(&tree_raw)
                        .context("verify-walks: tree artifact reparse")?;
                    let b: serde_json::Value = serde_json::from_slice(&raw)
                        .context("verify-walks: spliced artifact reparse")?;
                    anyhow::ensure!(
                        a == b,
                        "verify-walks: spliced artifact diverged from the tree artifact"
                    );
                    merged = Merged::Spliced(tree::gzip_artifact(&raw)?);
                }
            }
            return Ok(MergeOutcome {
                converted: Converted::Columns(cols),
                merged,
            });
        }
        // Fused path: no ConvertedColumns gather — analysis consumes the
        // post-walk rows through the sort permutation directly.
        let (view, plan) = crate::rows::convert_parsed_walked_view_plan(
            parsed,
            &annotations,
            crate::cudagraph::DEFAULT_STREAM,
            reassign,
        )?;
        let merged = if fused_artifact {
            // A plan with no edits splices to exactly the minified document:
            // skip the structural scan and the per-event splice outright.
            // (This arm is never reached in verify mode — verify returned
            // above — so the scan always runs here on demand.)
            let noop = plan.moved.is_empty()
                && plan.fixed.is_empty()
                && plan.appended.is_empty()
                && plan.retagged.is_empty()
                && plan.retagged_ac2g.is_empty()
                && plan.keep.iter().all(|&k| k);
            use std::sync::atomic::Ordering::Relaxed;
            let ann_args = if noop {
                crate::legacy::splice::NOOP_RANKS.fetch_add(1, Relaxed);
                Default::default()
            } else {
                crate::legacy::splice::EDITED_RANKS.fetch_add(1, Relaxed);
                rendered_ann_args(&plan, full_annotations.as_deref())
            };
            // The splice itself is deferred past the permit release —
            // see [`Merged::SplicePending`] and `finish_splice`.
            Merged::SplicePending(Box::new(SplicePending {
                spans: bounds,
                plan,
                ann_args,
                full_annotations,
                noop,
            }))
        } else if replay_artifact {
            Merged::Replay
        } else {
            Merged::RowWalk
        };
        return Ok(MergeOutcome {
            converted: Converted::View(view),
            merged,
        });
    }

    // ---- 2. Tree path ----
    if ann.walks() {
        let mut tree;
        if ann.reassign() {
            // Bytes must stay pristine through the legacy byte scan below
            // (there are no parsed rows on this path), so parse with the
            // copying variant.
            tree = {
                let _t = crate::timing::Timer::new(crate::timing::Stage::TreeParse);
                tree::parse_tree(decompressed)?
            };
            let live = crate::legacy::pkl::extract_live_graph_ids_from_raw(decompressed);
            let annotations = ann.full(rank, &live)?;
            let _t = crate::timing::Timer::new(crate::timing::Stage::TreeWalk);
            match recording_bytes {
                Some(rec) => tree::recording_merge_tree(
                    &mut tree,
                    &rec,
                    &annotations,
                    crate::cudagraph::DEFAULT_STREAM,
                )?,
                None => {
                    tree::annotate_tree(&mut tree, &annotations, crate::cudagraph::DEFAULT_STREAM)
                }
            }
        } else {
            // Inline-annotated: nothing reads the buffer after this parse,
            // so skip the whole-document defensive copy.
            tree = {
                let _t = crate::timing::Timer::new(crate::timing::Stage::TreeParse);
                tree::parse_tree_in_place(decompressed)?
            };
            let _t = crate::timing::Timer::new(crate::timing::Stage::TreeWalk);
            tree::inline_tree(&mut tree, crate::cudagraph::DEFAULT_STREAM);
        }
        let converted = {
            let _t = crate::timing::Timer::new(crate::timing::Stage::Convert);
            Converted::Columns(crate::legacy::json::convert_serde_to_columns(&tree))
        };
        return Ok(MergeOutcome {
            converted,
            merged: Merged::Tree(tree),
        });
    }
    // ---- 3. Passthrough: no CUDA-graph handling applies. ----
    //
    // Reached only for `Format::Json`: a non-tree format either walks (its
    // traces are self-annotated, so `ann.walks()` held above) or was rejected
    // by the `supports_tree` gate.
    //
    // Converted straight off the raw bytes: the borrowed-value DOM this
    // arm used to build (~6-15x the document, per rank, on every worker) was
    // the single dominant cost of un-annotated JSON runs — the churn spent
    // 71% of the run's cycles in kernel memory management. The DOM converter
    // remains the --verify-walks oracle.
    let verify_cols = if verify_walks {
        Some(crate::legacy::json::convert_json_bytes_to_columns(
            decompressed,
        )?)
    } else {
        None
    };
    let cols = crate::legacy::json::convert_json_scan_to_columns(decompressed)?;
    if let Some(dom_cols) = &verify_cols {
        crate::legacy::json::assert_columns_equal(&cols, dom_cols)?;
    }
    Ok(MergeOutcome {
        converted: Converted::Columns(cols),
        merged: Merged::Passthrough,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_rank(
    rank: i64,
    ingress: &TraceDiscovery,
    cfg: &RunConfig,
    ann_source: &AnnotationSource,
    rec_dir: Option<&PathBuf>,
    artifacts: Option<&ArtifactSink>,
    scratch: &mut Vec<u8>,
    stats: &WorkStats,
    mem_gate: &crate::memgate::MemGate,
) -> Result<Option<RankResult>> {
    let t_rank = std::time::Instant::now();
    let format = cfg.trace_format;
    // 1. Resolve + read replay.
    let Some(replay_path) = ingress.find_trace(rank, &cfg.trace_root, format) else {
        return Ok(None); // skip
    };
    // Memory admission: acquired before the first big allocation; the
    // permit spans the whole parse so the estimate covers scratch + rows.
    // On a memory-tight block this thread sheds its retained pools BEFORE
    // parking — parked workers otherwise pin them for the whole episode
    // (160 parked workers held RSS at the invisible
    // allowance with one rank in flight).
    // Tree-path hint for the cost estimate: only the CUDA-graph recording
    // merge still builds the full serde DOM — the reassigning pkl artifact
    // is served by the splice writer now (see merge_and_convert), so a
    // recording-free pkl job estimates at the fused-path factor.
    let dom_tree = cfg.trace_format.supports_tree()
        && matches!(ann_source, AnnotationSource::Pkl(_))
        && rec_dir.is_some();
    let est = crate::memgate::estimate_rank_cost(&replay_path, dom_tree);
    let _mem_permit = mem_gate.admit_with(est, || {
        *scratch = Vec::new();
        crate::rows::pool_clear();
    });
    let replay_bytes = {
        let _t = crate::timing::Timer::new(crate::timing::Stage::Read);
        ingress.read(&replay_path)?
    };

    // 2. Recording-step trace (pkl side-channel world only).
    let mut recording_bytes: Option<Vec<u8>> = None;
    if let Some(dir) = rec_dir {
        if let Some(rec_path) = ingress.find_trace(rank, dir, format) {
            recording_bytes = ingress.read(&rec_path).ok();
        }
    }

    // 3. Decompress ONCE into the worker's reusable scratch; every
    // downstream stage reuses the raw bytes. The scratch is released when it
    // is sized for a much larger rank than this one — same retention policy
    // as the parser pools (rows::RETAIN_FACTOR): held unconditionally it
    // pins one thread's largest-ever rank for the life of the run.
    let t_convert = std::time::Instant::now();
    let n = {
        let _t = crate::timing::Timer::new(crate::timing::Stage::Gunzip);
        crate::rows::maybe_gunzip_into(&replay_bytes, scratch)?
    };
    if scratch.capacity() > crate::rows::RETAIN_FACTOR * n.max(1) {
        scratch.truncate(n);
        scratch.shrink_to_fit();
    }

    let mut process_groups = if cfg.trace_format == trace::Format::Json {
        crate::legacy::json::process_group_configs(&scratch[..n])
    } else {
        Vec::new()
    };

    // 4+6. Merge + convert (fast path). Merge failure falls back to the raw
    // replay — re-decompressing, because the tape parse mutates the scratch
    // in place and may have died mid-buffer.
    let (converted, merged) = match merge_and_convert(
        &mut scratch[..n],
        recording_bytes,
        rank,
        cfg.trace_format,
        ann_source,
        artifacts.is_some(),
        cfg.verify_walks,
    ) {
        Ok(v) => (v.converted, v.merged),
        // In verify mode a mismatch must surface as a rank error, NOT fall
        // back — the fallback silently masks row-walk divergence.
        Err(e) if cfg.verify_walks => return Err(e),
        Err(e) => {
            eprintln!("rank {rank}: merge failed ({e}) — using replay as-is");
            let n = crate::rows::maybe_gunzip_into(&replay_bytes, scratch)?;
            let converted = Converted::Columns(crate::legacy::json::convert_json_bytes_to_columns(
                &scratch[..n],
            )?);
            (converted, Merged::Passthrough)
        }
    };
    if process_groups.is_empty() {
        process_groups = kernel_process_groups(&converted, rank)?;
    }
    anyhow::ensure!(
        !process_groups.is_empty(),
        "trace has no process-group inventory: no distributedInfo.pg_config and no \
         comm kernels with process-group metadata. Start the profiler after \
         torch.distributed is initialized"
    );
    // Footprint sample at the live-set peak: raw bytes + parse structures
    // (+ the serde DOM on the tree path) are all still alive here. The
    // permit's drop-time reading alone misses transients the path frees
    // before completion — see MemPermit::observe_transient.
    _mem_permit.observe_transient();

    // 5. Optional merged-trace artifact. Names must stay rank-scannable
    // (`NNNNNN.` + a known trace suffix) — that is what the publisher's
    // upload set and the page's rank-map paths key on
    // (host_map::scan_trace_paths). The bytes go to the run's writer pool
    // (an inline write here blocked the worker under dirty-page throttling),
    // but only after the analyze stage, permit-free — see below. The
    // verbatim arms MOVE replay_bytes instead of cloning ~3 MB per rank.
    let named: Option<(String, ArtifactPayload)> = if artifacts.is_some() {
        match merged {
            Merged::Passthrough => Some((
                trace::merged_json_artifact_name(rank),
                ArtifactPayload::Ready(replay_bytes),
            )),
            Merged::Tree(tree) => {
                drop(replay_bytes);
                let _t = crate::timing::Timer::new(crate::timing::Stage::TreeSer);
                Some((
                    trace::merged_json_artifact_name(rank),
                    ArtifactPayload::Ready(tree::serialize_tree(&tree, true, n)?),
                ))
            }
            // Already the finished gz bytes (splice+gzip accounted under
            // tree_ser in the worker).
            Merged::Spliced(gz) => {
                drop(replay_bytes);
                Some((
                    trace::merged_json_artifact_name(rank),
                    ArtifactPayload::Ready(gz),
                ))
            }
            // Deferred splice: runs permit-free below, reading the pristine
            // scratch.
            Merged::SplicePending(p) => {
                drop(replay_bytes);
                Some((
                    trace::merged_json_artifact_name(rank),
                    ArtifactPayload::Splice(p),
                ))
            }
            // Verbatim copy of a self-annotated non-tree trace: keep
            // whatever compression the source had.
            Merged::Replay => Some((
                trace::replay_artifact_name(rank, &replay_bytes),
                ArtifactPayload::Ready(replay_bytes),
            )),
            // Reassigning side channel on a non-tree format: nothing honest
            // to write (see merge_and_convert) — skip this rank's artifact.
            Merged::RowWalk => {
                eprintln!(
                    "rank {rank}: merged-trace artifact skipped — a {} trace \
                     with side-channel annotations has no mergeable tree",
                    format.as_str()
                );
                drop(replay_bytes);
                None
            }
        }
    } else {
        drop(merged);
        drop(replay_bytes);
        None
    };
    // Second footprint sample: the tree/splice path allocates its whole
    // serialized artifact AFTER the post-merge sample above — with only
    // that one sample the gate learned ~1.5 GB/rank on ranks that peak
    // several GB later in the pipeline, over-admitted, and runs
    // wedged at the reclaim floor.
    _mem_permit.observe_transient();
    let convert_s = t_convert.elapsed().as_secs_f64();
    let t_analyze = std::time::Instant::now();
    let _t_stage_analyze = crate::timing::Timer::new(crate::timing::Stage::Analyze);
    let base_ns = match &converted {
        Converted::Columns(c) => c.base_ns,
        Converted::View(v) => v.base_ns(),
    };
    let (breakdown, pg_dfs) = match converted {
        Converted::Columns(cols) => {
            // A failure here maps to the all -1 sentinel row + empty pg_dfs:
            // an analysis failure is recorded, never fatal to the run.
            match crate::analysis::process(&cols, rank, &process_groups) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("rank {rank}: fast analysis failed ({e}) — sentinel row");
                    (
                        crate::analysis::BREAKDOWN_COLUMNS
                            .iter()
                            .map(|c| (*c, crate::analysis::BreakdownValue::Int(-1)))
                            .collect(),
                        Vec::new(),
                    )
                }
            }
        }
        Converted::View(view) => {
            let out = match crate::analysis::process_view(&view, rank, &process_groups) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("rank {rank}: fused analysis failed ({e}) — sentinel row");
                    (
                        crate::analysis::BREAKDOWN_COLUMNS
                            .iter()
                            .map(|c| (*c, crate::analysis::BreakdownValue::Int(-1)))
                            .collect(),
                        Vec::new(),
                    )
                }
            };
            // Return the shells (rows/raws/arena) to the thread-local pool.
            view.recycle();
            out
        }
    };
    drop(_t_stage_analyze);
    // Third footprint sample: analysis working set + the finished artifact
    // bytes still held for the writer handoff.
    _mem_permit.observe_transient();
    if cfg.verbose {
        eprintln!(
            "rank {rank}: convert={convert_s:.3}s analyze={:.3}s",
            t_analyze.elapsed().as_secs_f64(),
        );
    }

    // Release the permit BEFORE the writer-queue block: parse and analyze
    // structures are gone by here, so a producer parked on artifact
    // backpressure holds only its finished gz bytes — and those are charged
    // to the gate through the writer pool's live-bytes gauge, not the
    // permit. Holding the permit across `submit` charged the full rank
    // estimate (~3.3 GB uncorrected) per parked producer:
    // one run spent 19.7k thread-s parked in artifact_q while the
    // ledger read full and admissions starved at width 61 of 160.
    drop(_mem_permit);
    if let Some(sink) = artifacts {
        if let Some((name, payload)) = named {
            match payload {
                ArtifactPayload::Ready(artifact) => sink.submit(name, artifact),
                ArtifactPayload::Splice(p) => {
                    // Permit-free: the splice reads the retained scratch
                    // (resident either way); the spans/plan/gz transients it
                    // adds are charged to the ledger through the writer-pool
                    // gauge for the duration.
                    let _charge = sink.transient_charge((n / 4) as u64);
                    match finish_splice(&scratch[..n], *p, rank, &replay_path, mem_gate) {
                        Ok(gz) => sink.submit(name, gz),
                        // Both the splice and its tree fallback failed —
                        // possible only for a document the row parse
                        // accepted but serde rejects, which the fallback's
                        // own error already describes. Analysis stands
                        // (the walked view served it); only the
                        // merged-trace copy is skipped.
                        Err(e) => {
                            eprintln!("rank {rank}: merged-trace artifact skipped ({e})")
                        }
                    }
                }
            }
        }
    }

    let total_ns = t_rank.elapsed().as_nanos() as u64;
    stats.work_ns.fetch_add(total_ns, Ordering::Relaxed);

    Ok(Some(RankResult {
        rank,
        breakdown_row: breakdown,
        base_ns,
        pg_dfs,
        process_groups,
    }))
}

pub fn run(cfg: &RunConfig) -> Result<RunSummary> {
    // status.json phase boundaries (emit::PhaseTimes): claim = this setup
    // span, before the worker/rank partition exists.
    let claim_start_us = crate::emit::epoch_us();
    let ingress = TraceDiscovery::new();
    let index = ingress.index(&cfg.trace_root, cfg.trace_format)?;
    let world_size = cfg.world_size.unwrap_or_else(|| index.world_size());
    let missing = index.missing(world_size);
    if !missing.is_empty() {
        eprintln!(
            "comms_viz: no trace for {} of {world_size} rank(s): {missing:?}",
            missing.len()
        );
    }

    // ---- Annotation source: built once, shared by every rank worker. ----
    let ann_source = build_annotation_source(cfg, &ingress, world_size)?;

    // Recording-step trace, when the annotation source uses one at all.
    let rec_dir = if ann_source.uses_recording_merge() {
        crate::legacy::recording::resolve_dir(
            &ingress,
            &cfg.trace_root,
            world_size,
            cfg.recording_step,
            cfg.trace_format,
        )
    } else {
        None
    };
    // Writer width, overridable for measurement (COMMS_VIZ_ARTIFACT_WRITERS):
    // on a 24k-rank run, workers were blocked in `artifact_q` for 42% of
    // their thread-seconds behind the 4-writer pool.
    let artifact_writers = std::env::var("COMMS_VIZ_ARTIFACT_WRITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(4);
    let artifact_sink = if cfg.write_repaired_traces {
        Some(ArtifactSink::new(
            cfg.out_root.join("tree").join(&cfg.run_id).join("traces"),
            artifact_writers,
            artifact_queue_cap_bytes(cfg.threads.max(1)),
        )?)
    } else {
        None
    };

    // Clamp the worker count to the world so no worker partition is empty.
    let w_count = workers_for_world_size(world_size).min(world_size).max(1);

    // Claim (setup) ends here; assign = the worker/rank-range partition.
    let assign_start_us = crate::emit::epoch_us();

    // ---- Parallel rank loop ----
    let n = world_size as usize;
    let slots: Vec<Mutex<Option<RankOutcome>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let workers = cfg.threads.clamp(1, n.max(1));
    // The worker count is CPU-topology-sized; the MEMORY in flight is
    // bounded separately (see memgate.rs: on one run
    // 160 concurrent JSON parses OOM-killed the whole agent task).
    let mem_gate = crate::memgate::MemGate::new(workers);
    if let Some(sink) = &artifact_sink {
        mem_gate.set_external_bytes(sink.live_bytes_gauge());
    }

    // Systemic-failure probe: when the first SYSTEMIC_PROBE completed ranks
    // ALL error, the rest are the same failure (e.g. a corrupted trace copy
    // where every rank fails parse at character 0). Any errored rank already
    // fails the run, so cutting short never changes the outcome; workers
    // synthesize errors for the remaining ranks instead of burning wall on
    // thousands of doomed parses.
    const SYSTEMIC_PROBE: usize = 64;
    let probe_total = SYSTEMIC_PROBE.min(n);
    let a_probe_errs = AtomicUsize::new(0);
    let systemic = std::sync::atomic::AtomicBool::new(false);
    // Rank outcomes recorded (processed + skipped + errored).
    let a_done = AtomicUsize::new(0);
    let meta_dir = cfg.out_root.join("tree").join(&cfg.run_id).join("_meta");
    let stats = WorkStats::new();
    // Turnstile (bounded worker subset): workers take stable indices and
    // only the first `mem_gate.active_width()` of them claim ranks; the
    // rest park cold. Claiming with all threads and letting the gate park
    // the excess rotated the work across every thread, keeping every
    // thread's retained pools hot: RSS converged to threads-EVER-used ×
    // ~1.3 GB no matter how narrow the gate ran, because healthy churn
    // parks each thread well under SHED_AFTER_PARKED.
    // Parking happens BEFORE the rank claim, so a cold worker can never
    // strand a claimed rank; the gate's FIFO admission is untouched.
    let next_worker_id = AtomicUsize::new(0);
    let turnstile_parked_us = std::sync::atomic::AtomicU64::new(0);
    let turnstile_sheds = AtomicUsize::new(0);
    let processing_start_us = crate::emit::epoch_us();

    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                let w = next_worker_id.fetch_add(1, Ordering::Relaxed);
                let mut scratch: Vec<u8> = Vec::new();
                let mut park: Option<std::time::Instant> = None;
                let mut park_shed = false;
                loop {
                    // Turnstile park — see the comment at next_worker_id.
                    // The systemic drain bypasses it: synthesized outcomes
                    // cost no memory and every idle thread speeds them up.
                    if w >= mem_gate.active_width() && !systemic.load(Ordering::Relaxed) {
                        if next.load(Ordering::Relaxed) >= n {
                            if let Some(t) = park.take() {
                                turnstile_parked_us
                                    .fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                            }
                            break;
                        }
                        let since = *park.get_or_insert_with(std::time::Instant::now);
                        if !park_shed && since.elapsed() >= crate::memgate::SHED_AFTER_PARKED {
                            park_shed = true;
                            scratch = Vec::new();
                            crate::rows::pool_clear();
                            mem_gate.shed_purge();
                            turnstile_sheds.fetch_add(1, Ordering::Relaxed);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        continue;
                    }
                    if let Some(t) = park.take() {
                        turnstile_parked_us
                            .fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                        park_shed = false;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let rank = i as i64;
                    let outcome = if systemic.load(Ordering::Relaxed) {
                        RankOutcome::Errored(
                            "aborted after systemic early-rank failure".to_string(),
                        )
                    } else {
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            process_rank(
                                rank,
                                &ingress,
                                cfg,
                                &ann_source,
                                rec_dir.as_ref(),
                                artifact_sink.as_ref(),
                                &mut scratch,
                                &stats,
                                &mem_gate,
                            )
                        })) {
                            Ok(Ok(Some(r))) => RankOutcome::Row(Box::new(r)),
                            Ok(Ok(None)) => RankOutcome::Skipped,
                            Ok(Err(e)) => {
                                eprintln!("rank {rank}: errored — {e}");
                                RankOutcome::Errored(format!("{e:#}"))
                            }
                            Err(_) => {
                                eprintln!("rank {rank}: panicked");
                                RankOutcome::Errored("panicked".to_string())
                            }
                        }
                    };
                    let errored = matches!(outcome, RankOutcome::Errored(_));
                    *slots[i].lock().unwrap() = Some(outcome);
                    let idx = a_done.fetch_add(1, Ordering::Relaxed);
                    if errored && idx < probe_total {
                        let errs = a_probe_errs.fetch_add(1, Ordering::Relaxed) + 1;
                        if errs == probe_total && !systemic.swap(true, Ordering::Relaxed) {
                            eprintln!(
                                "systemic failure: the first {probe_total} completed ranks \
                                 ALL errored — draining the remaining ranks as errors \
                                 (staged-read corruption class)"
                            );
                        }
                    }
                    if mem_gate.is_tight() {
                        // Memory-tight host: retained per-thread buffers are the
                        // dominant steady-state footprint — hand them back so
                        // gated admissions have something to admit against.
                        // is_tight (ledger/pressure/floor), not cap-blocked: the cap
                        // blocks some workers whenever cap < worker count, and
                        // shedding pools on that benign signal re-created the
                        // fault churn the pools prevent.
                        scratch = Vec::new();
                        crate::rows::pool_clear();
                    }
                }
            });
        }
    });
    // Processing ends when every rank worker has joined; everything after
    // (collect + fold + metadata) is the aggregation span.
    let aggregation_start_us = crate::emit::epoch_us();

    // Drain the artifact writers before anything downstream can look at
    // the traces dir (emit scans it in-process; write failures fail the run
    // here rather than silently missing files).
    if let Some(sink) = artifact_sink {
        sink.finish()?;
    }

    stats.log(workers);
    let turnstile_s = turnstile_parked_us.load(Ordering::Relaxed) as f64 / 1e6;
    if turnstile_s >= 1.0 {
        eprintln!(
            "turnstile: cold-worker park {:.1} thread-s, pool sheds {}",
            turnstile_s,
            turnstile_sheds.load(Ordering::Relaxed),
        );
    }

    // ---- Collect in rank order ----
    let mut results: Vec<Option<RankResult>> = Vec::with_capacity(n);
    let mut skipped: Vec<i64> = Vec::new();
    let mut errors: Vec<(i64, String)> = Vec::new();
    for (i, slot) in slots.into_iter().enumerate() {
        match slot.into_inner().unwrap() {
            Some(RankOutcome::Row(r)) => results.push(Some(*r)),
            Some(RankOutcome::Skipped) => {
                skipped.push(i as i64);
                results.push(None);
            }
            Some(RankOutcome::Errored(msg)) => {
                errors.push((i as i64, msg));
                results.push(None);
            }
            None => {
                errors.push((i as i64, "no outcome recorded".into()));
                results.push(None);
            }
        }
    }
    let processed = results.iter().flatten().count();

    let meta = build_meta(cfg, world_size, w_count);
    let mut results = results;

    // ---- Emit artifacts ----
    let rank_errors: std::collections::HashMap<i64, String> = errors.iter().cloned().collect();
    let meta_ctx = crate::emit::WorkerMetaCtx {
        meta_dir,
        rank_errors: &rank_errors,
        started_at_us: meta.started_at_us,
        updated_at_us: meta.completed_at_us,
    };
    let summary = crate::emit::emit_run(&cfg.out_root, &meta, &mut results, Some(&meta_ctx))?;
    let phase_times = crate::emit::PhaseTimes {
        starts_us: [
            claim_start_us,
            assign_start_us,
            processing_start_us,
            aggregation_start_us,
        ],
    };
    crate::emit::emit_metadata(
        &cfg.out_root,
        &meta,
        &summary.process_groups,
        // emit_run emitted every worker's partial or errored out
        // above — reaching here means all of them completed.
        &crate::emit::StatusCounts {
            ranks_processed: processed as i64,
            workers_done: w_count,
        },
        &phase_times,
    )?;
    crate::host_map::emit_rank_hosts(
        &cfg.out_root.join("tree").join(&cfg.run_id),
        &cfg.trace_root,
        world_size,
    )?;

    Ok(RunSummary {
        world_size,
        processed,
        skipped: skipped.len(),
        errors: errors.len(),
        skipped_ranks: skipped,
        errored_ranks: errors.into_iter().map(|(r, _)| r).collect(),
        pg_keys: summary.pg_keys,
    })
}

/// Build the run's annotation source from its configuration.
///
/// `Auto` is a CLI-level concept: it must be resolved before this point (the
/// CLI probes the job layout), so reaching here with `Auto` is a programming
/// error rather than something to guess at.
fn build_annotation_source(
    cfg: &RunConfig,
    ingress: &TraceDiscovery,
    world_size: i64,
) -> Result<AnnotationSource> {
    let source = match cfg.annotation_mode {
        AnnotationMode::Disabled => AnnotationSource::Disabled,
        AnnotationMode::Inline => AnnotationSource::Inline,
        AnnotationMode::Pkl => AnnotationSource::Pkl(crate::legacy::pkl_fetch::PklSource::new(
            ingress,
            &cfg.trace_root,
            0..world_size,
        )),
        AnnotationMode::Auto => anyhow::bail!(
            "annotation_mode=Auto reached the pipeline unresolved — \
             the caller must resolve it from the job layout first"
        ),
    };
    Ok(source)
}

fn build_meta(cfg: &RunConfig, world_size: i64, w_count: i64) -> RunMeta {
    RunMeta {
        run_id: cfg.run_id.clone(),
        run_name: cfg.run_name.clone(),
        world_size,
        trace_root: cfg.trace_root.to_string_lossy().into_owned(),
        source_trace_root: cfg.source_trace_root.clone(),
        started_at_us: cfg.started_at_us,
        completed_at_us: cfg.completed_at_us,
        worker_count: w_count,
        config_extra: vec![("engine".into(), crate::emit::engine_stamp())],
    }
}

#[cfg(test)]
mod process_group_metadata_tests {
    use super::*;

    #[test]
    fn builds_rank_membership_from_event_metadata() {
        let groups = process_groups_from_event_metadata(
            7,
            [
                ("pg_tp", "mesh_tp", 2),
                ("pg_tp", "mesh_tp", 2),
                ("pg_fsdp", "mesh_fsdp", 4),
            ]
            .into_iter(),
        )
        .unwrap();

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].pg_name, "pg_fsdp");
        assert_eq!(groups[0].ranks, vec![7]);
        assert_eq!(groups[0].pg_size, 4);
        assert_eq!(groups[1].pg_name, "pg_tp");
        assert_eq!(groups[1].pg_desc, "mesh_tp");
        assert_eq!(groups[1].ranks, vec![7]);
    }
}

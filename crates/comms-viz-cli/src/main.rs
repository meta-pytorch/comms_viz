// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Local comms_viz CLI: profiler traces to analysis artifacts and an HTML report.

use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use comms_viz_core::annotations::AnnotationMode;
use comms_viz_core::pipeline::RunConfig;
use comms_viz_core::pipeline::run;
use comms_viz_core::trace::Format;
use comms_viz_core::trace::{self};
use tikv_jemallocator::Jemalloc;

mod convert_pftrace;
mod diagnostics;
mod discovery;
mod publish;
mod static_page;
mod structured_report;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// narenas:224 gives the rank workers enough arenas that they rarely share
// one's lock; oversize_threshold:0 disables the dedicated huge-allocation
// arena, which every thread would otherwise contend on for trace buffers.
#[expect(
    non_upper_case_globals,
    reason = "jemalloc reads this symbol by its C name"
)]
#[export_name = "_rjem_malloc_conf"]
pub static malloc_conf: &[u8; 33] = b"narenas:224,oversize_threshold:0\0";

#[derive(Parser, Debug)]
#[command(
    name = "comms_viz",
    about = "Analyze local PyTorch profiler traces and render an HTML report",
    after_help = "Analyze and render: `comms_viz --trace-root ./traces --out ./analysis --report ./report`\n\
                  Render existing artifacts: `comms_viz --out ./analysis --report ./report`"
)]
struct Args {
    /// Directory holding one step's profiler traces, one file per rank
    /// (e.g. TorchTitan's `profile_traces/iteration_<N>/`).
    #[arg(long)]
    trace_root: Option<PathBuf>,

    /// Output root for analysis artifacts.
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// Directory in which to write the HTML report and its data sidecars
    /// [default: <out>/report].
    #[arg(long, value_name = "DIR")]
    report: Option<PathBuf>,

    /// Directory in which to write a structured report: the HTML
    /// report's data and derived tables as Markdown and CSV
    /// [default: <out>/structured_report].
    #[arg(long, value_name = "DIR")]
    structured_report: Option<PathBuf>,

    /// Source path recorded in artifacts instead of the local trace path.
    #[arg(long, hide = true, requires = "trace_root")]
    source_trace_root: Option<String>,

    /// Run identifier used in the artifact tree. Defaults to `local-run`,
    /// or `local-run-<N>` if that run already exists under `--out`.
    #[arg(long)]
    run_id: Option<String>,

    /// Replace an existing run with the same id under `--out`.
    #[arg(long)]
    overwrite: bool,

    /// Descriptive run name shown in the report.
    #[arg(long)]
    run_name: Option<String>,

    /// World size; inferred from the trace directory when omitted.
    #[arg(long, hide = true)]
    world_size: Option<i64>,

    /// Trace format: auto, json, or pftrace.
    #[arg(long, default_value = "auto")]
    trace_format: String,

    /// CUDA-graph annotation source: auto, pkl, inline, or disabled.
    #[arg(long, hide = true, default_value = "auto")]
    annotations: String,

    /// Also write each rank's trace after the CUDA-graph repair to
    /// `tree/<run>/traces/`, for viewing in Perfetto.
    #[arg(long)]
    write_repaired_traces: bool,

    /// Rank-level parallelism.
    #[arg(long, hide = true)]
    threads: Option<usize>,

    /// started_at timestamp in microseconds; zero uses the current time.
    #[arg(long, hide = true, default_value_t = 0)]
    started_at_us: i64,

    /// completed_at timestamp in microseconds; zero stamps outputs at write time.
    #[arg(long, hide = true, default_value_t = 0)]
    completed_at_us: i64,

    /// Print per-rank analysis time.
    #[arg(short, long)]
    verbose: bool,

    #[arg(long, hide = true)]
    verify_walks: bool,

    /// Convert JSON traces to self-annotated pftrace files instead of analyzing.
    #[arg(long, hide = true, value_name = "DIR")]
    convert_pftrace: Option<PathBuf>,
}

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time is before Unix epoch")
        .as_micros() as i64
}

/// Pick the run id for a new analysis. An explicit id that already exists
/// is an error unless `overwrite`; the default id takes the next free
/// `local-run-<N>` instead.
fn resolve_run_id(out: &Path, requested: Option<&str>, overwrite: bool) -> Result<String> {
    let tree = out.join("tree");
    let base = requested.unwrap_or("local-run");
    if !tree.join(base).exists() {
        return Ok(base.to_string());
    }
    if overwrite {
        eprintln!("comms_viz: replacing run {base} under {}", out.display());
        std::fs::remove_dir_all(tree.join(base))?;
        return Ok(base.to_string());
    }
    if requested.is_some() {
        anyhow::bail!(
            "run {base} already exists under {}; pass --overwrite to replace it",
            out.display()
        );
    }
    let mut n = 2;
    let id = loop {
        let id = format!("{base}-{n}");
        if !tree.join(&id).exists() {
            break id;
        }
        n += 1;
    };
    eprintln!(
        "comms_viz: run {base} exists under {}, writing {id}",
        out.display()
    );
    Ok(id)
}

fn main() -> Result<()> {
    // Larger worker stacks for the rayon and polars pools, which take the
    // std default unless RUST_MIN_STACK is set (read at the first spawn, so
    // this must run before any pool starts). A worker that waits on the
    // other pool runs stolen jobs on its own stack meanwhile, and with ~100k
    // process groups (45k ranks with PP) the default 2 MiB overflowed.
    // Stacks are reserved, not committed, so this costs no memory up front.
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        std::env::set_var("RUST_MIN_STACK", (64usize << 20).to_string());
    }
    let args = Args::parse();
    if args.convert_pftrace.is_some() {
        let trace_root = args
            .trace_root
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--convert-pftrace needs --trace-root"))?;
        return analyze(&args, trace_root, "local-run");
    }
    let out = args
        .out
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--out is required"))?;
    let report = args.report.clone().unwrap_or_else(|| out.join("report"));
    let structured = args
        .structured_report
        .clone()
        .unwrap_or_else(|| out.join("structured_report"));
    let targets = publish::Targets {
        report: Some(&report),
        structured: Some(&structured),
    };

    let Some(trace_root) = args.trace_root.clone() else {
        return publish::render_local(&out, args.run_id.as_deref(), targets);
    };

    let run_id = resolve_run_id(&out, args.run_id.as_deref(), args.overwrite)?;
    analyze(&args, trace_root, &run_id)?;
    publish::render_local(&out, Some(&run_id), targets)
}

/// Analyze one trace directory into `--out` under `run_id`.
fn analyze(args: &Args, trace_root: PathBuf, run_id: &str) -> Result<()> {
    let annotation_mode = match AnnotationMode::parse(&args.annotations)? {
        AnnotationMode::Auto => discovery::detect_annotation_mode(&trace_root),
        explicit => explicit,
    };
    eprintln!("comms_viz: annotations={}", annotation_mode.as_str());

    let threads = args.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
    });

    if let Some(out_dir) = args.convert_pftrace.clone() {
        return convert_pftrace::run(convert_pftrace::ConvertArgs {
            trace_root,
            out_dir,
            threads,
            annotation_mode,
        });
    }

    let out_root = args
        .out
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--out is required"))?;
    let t0 = Instant::now();

    let trace_format = {
        let choice = Format::parse(&args.trace_format)?;
        let names: Vec<String> = std::fs::read_dir(&trace_root)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        let (format, message) = trace::detect(choice, names.iter().map(String::as_str))?;
        eprintln!("comms_viz: {message}");
        format
    };

    if args.verify_walks && !trace_format.supports_tree() {
        anyhow::bail!("--verify-walks is only supported for json traces");
    }

    let cfg = RunConfig {
        trace_root,
        source_trace_root: args.source_trace_root.clone(),
        out_root,
        run_id: run_id.to_string(),
        run_name: args.run_name.clone().unwrap_or_default(),
        world_size: args.world_size,
        trace_format,
        annotation_mode,
        recording_step: None,
        write_repaired_traces: args.write_repaired_traces,
        threads,
        started_at_us: if args.started_at_us == 0 {
            now_us()
        } else {
            args.started_at_us
        },
        completed_at_us: args.completed_at_us,
        verbose: args.verbose,
        verify_walks: args.verify_walks,
    };

    let summary = run(&cfg)?;
    eprintln!(
        "comms_viz: world_size={} processed={} skipped={} errors={} pgs=[{}] wall={:.2}s",
        summary.world_size,
        summary.processed,
        summary.skipped,
        summary.errors,
        summary.pg_keys.join(","),
        t0.elapsed().as_secs_f64(),
    );
    // Capped: jobs with generator/reward ranks legitimately skip most of a large world.
    let show = |label: &str, ranks: &[i64]| {
        const MAX: usize = 64;
        if ranks.is_empty() {
            return;
        }
        if ranks.len() <= MAX {
            eprintln!("  {label} ranks: {ranks:?}");
        } else {
            eprintln!(
                "  {label} ranks ({}): {:?} (+{} more)",
                ranks.len(),
                &ranks[..MAX],
                ranks.len() - MAX
            );
        }
    };
    show("skipped", &summary.skipped_ranks);
    show("errored", &summary.errored_ranks);
    if summary.errors > 0 {
        anyhow::bail!("{} rank(s) failed analysis", summary.errors);
    }
    Ok(())
}

#[cfg(test)]
mod jemalloc_tests {
    /// The memory gate's footprint accounting rides thread.allocated /
    /// thread.deallocated (memgate::thread_live_bytes). In jemalloc 5.3
    /// those per-thread counters are NOT config_stats-gated, so they work
    /// even though this workspace builds jemalloc --disable-stats — but
    /// older jemalloc DID gate them, and a re-gating upgrade would make
    /// the mallctl reads fail SILENTLY (rc != 0 → 0): every footprint
    /// sample becomes "no observation" and the gate quietly degrades to
    /// its static estimates. This test runs inside the bin crate, so the
    /// crate's #[global_allocator] jemalloc is the allocator; it pins both
    /// counters. (The process-wide stats.* namespace IS config-gated and
    /// reads zeros here — don't move the gate onto it without enabling
    /// tikv-jemalloc-sys's `stats` feature.)
    #[test]
    fn jemalloc_thread_stats_are_readable() {
        const N: u64 = 4 << 20;
        let base = comms_viz_core::memgate::thread_live_bytes();
        let buf = vec![7u8; N as usize];
        std::hint::black_box(&buf);
        let with_buf = comms_viz_core::memgate::thread_live_bytes();
        assert!(
            with_buf >= base + N,
            "thread.allocated unreadable (base={base}, with_buf={with_buf}) — \
             did a jemalloc upgrade re-gate the thread stats?"
        );
        drop(buf);
        let after_drop = comms_viz_core::memgate::thread_live_bytes();
        assert!(
            after_drop + N / 2 <= with_buf,
            "thread.deallocated unreadable (with_buf={with_buf}, after_drop={after_drop})"
        );
    }
}

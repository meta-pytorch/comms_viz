# Architecture

## Data flow

Each rank's trace is process in parallel and aggregated for parallelism / breakdown results.

```
<trace-root>/  one trace per rank (Kineto JSON or Cuspy pftrace)
    |
    |  find rank -> file, world size                 trace/discover.rs
    v
per rank, in parallel                                pipeline.rs
    parse events into one row table                  legacy/json.rs, trace/pftrace.rs
    repair CUDA-graph kernels (if any)               annotations.rs, cudagraph.rs
    build PG layout: pg_config, else kernel PG args  pipeline.rs
    map each PG to an axis (tp, fsdp, ...)           analysis.rs
    classify kernels: comm / compute / memory        kernel_types.rs
    sum time per category, collective, axis          analysis.rs -> breakdown row
    keep comm kernels, tag step, split by PG         analysis.rs -> per-group tables
    GPU busy / compute / CPU time, GPU counters      analysis.rs -> diagnostics_0
    |
    |  write in rank order                           emit.rs
    v
<out>/tree/<run-id>/   breakdown, per-group tables, PG map
    |
    |  cross-rank: match each collective across      static_page.html (JS)
    |  its group, then lateness, imbalance,          structured_report.rs
    |  arrival order, utilization
    v
<out>/report/             HTML page + .pgdata
<out>/structured_report/  Markdown + CSV
```

## Input

comms_viz reads one directory holding one trace per rank for a single step so on multi-node jobs traces should be on
shared storage or gathered first.

There are two producers, both part of the PyTorch profiler:

- **Kineto** (the default backend) writes Chrome-trace JSON
  (`export_chrome_trace`). PyTorch adds a top-level `distributedInfo` object to
  the trace: `backend`, `rank`, `world_size`, and `pg_config`, the list of
  every process group with its name, description (`pg_desc`), size and
  ranks. comms_viz takes the rank from it and uses `pg_config` to understand the PG
  layout.
- **Cuspy** (a newer CUPTI-based backend in PyTorch 2.15+) writes Perfetto protobuf
  (`.pftrace`). `distributedInfo` is not written, so PG layout is rebuilt from reading
  kernel metadata.

In both formats, PyTorch's c10d tags each collective kernel with its
process group name and description, collective name, sequence number,
message sizes, dtype and group size, which comms_viz relies on to assign
kernels to PGs and to match collective across ranks.

Groups are sorted into parallelism axes (`tp`, `fsdp`, ...) using the
mesh dimension names from DeviceMesh, or other group descriptions (e.g.
Megatron's `TENSOR_MODEL_PARALLEL_GROUP`). A group created without a
description (`new_group` leaves PyTorch's default `undefined`) is its own axis,
named by its process-group name.

DeviceMesh creates a separate process group per mesh for each axis, so a job that builds separate meshes (e.g. FSDP storage, compute) may show duplicate groups with the same name (e.g. `tp 1/2`, `tp 2/2`). In such cases the per-mesh collectives will be separated (e.g. one group timeline shows TP BWD all-reduces that FSDP runs, the other timeline shows FWD-BWD all-gather, reduce-scatter TP comms).

## Breakdown tab

One row per rank (`breakdown.parquet`, `breakdown.csv`), from one pass over
the rank's GPU events:

- `comm_total`, `comp_total`: summed durations of comm and compute kernels;
  kernels on different streams contribute separately, without accounting for overlap time.
- `<collective>_total`: comm time per collective type.
- `<axis>_<collective>_total`: the same, per axis.
- `total_duration`: first GPU event start to last GPU event end.
- `cudaLaunchKernel_total`, `cudaStreamSynchronize_total`, ...: CPU time in
  those CUDA runtime calls.

## Parallelism tab

The page shows one process group at a time, with each rank's comm kernels
on a shared timeline. Every group's timeline runs from the first to the last
GPU event of all analyzed ranks, so switching groups keeps the axis and
shows which part of the step each group is active in.

A kernel counts as communication if its name is in `kernel_types.rs` (NCCL
plus a few known collective libraries) or if it carries both a
`Collective name` and a `Process Group Name`, as c10d's collectives do. It
appears in a group's timeline only if it carries the process-group metadata.

- **Matching.** A collective's instances across ranks (a cohort) share
  (group, step, collective, sequence number). Calls without a sequence
  number, and p2p send/recv (c10d gives every one sequence 0), join no
  cohort: only their position would pair them across ranks, and one call
  missing from a capture would mispair every later one. They stay on the
  timeline. A cohort is complete when it has exactly one kernel per rank of
  the group; only complete cohorts are used below.
- **Imbalance table.** Per cohort: spread = last rank's start minus first
  rank's start, also as a percentage of the first rank's kernel duration.
  Bandwidth is the payload over the fastest member's duration. Sorted by
  spread.
- **Collective arrival summary.** Per rank and collective: how often the
  rank started last, and its median and p90 arrival position. A rank that
  is persistently last is the straggler; the others' long kernels are
  mostly waiting for it.
- **GPU/CPU utilization.** Per rank, over its GPU window: union of all GPU
  kernel and memcpy time, union of compute kernel time, and CPU time in ops
  and CUDA calls.
- **GPU counters.** When the trace has counters (SM activity, DRAM and NVLink
  bandwidth, ...): a heatmap with one row per rank, each rank's samples cut
  into 64 time bins on the timeline's axis and colored by one chosen
  statistic of the bin (min, p1, p50, p99 or max), and a table of each
  rank's min, p10, p50, mean, p90, p99, max and std over the capture. The
  samples themselves are not kept: at fleet scale they would outweigh every
  other table.

## Outputs

```
<out>/
  tree/<run-id>/                 analysis results; the reports are built only from this
    results/pipeline_a/
      breakdown.parquet          Breakdown tab, one row per rank
      per_pg/<axis>_<pg>.dpg     one group's comm kernels, all ranks
      per_pg/diagnostics_0.dpg   utilization, unannotated comm kernels, GPU counter stats
      per_pg/diagnostics_pm_<n>.dpg   GPU counter time bins of ranks 32n..32n+31
    pg_map.json                  axis -> group -> ranks
    rank_map.json                analyzed rank -> trace file and host
    metadata.json                capture info and run environment
    _meta/                       run config and status
    traces/                      repaired traces, with --write-repaired-traces
  report/<stem>.html             the page; <stem> starts with comms_viz_<run-id>
  report/<stem>.pgdata           the per-group tables, fetched by the page on demand
  structured_report/<stem>/
    README.md                    the summary, for people and LLM agents
    summary.json                 the same content as JSON
```

To re-render the reports for the same run without re-running analysis,
run comms_viz without `--trace-root`, the reports will be built only from `tree/`.

The structured report summarizes the run without growing with world size:
every list is a top-K or a distribution of measured values, with no
thresholds or verdicts. It covers data quality (kernels without
process-group metadata, calls left out of matching), utilization across
ranks, stragglers (ranks sorted by how often they start a matched collective
last, against chance, over all their groups), per metric the median and the
lowest and highest ranks, and per axis the start spread and bandwidth
distributions with the worst collectives and the groups with the lowest
bandwidth.

## CUDA graphs

Kernels replayed from a CUDA graph have no CPU op behind them, so they
come without the c10d tags. PyTorch can annotate them at capture time
(`enable_annotations=True`), and comms_viz copies those annotations onto the
replayed kernels, matching them by `graph node id`.

NCCL kernels that end up without process-group metadata, e.g. because
annotations did not resolve, count as communication time but are on no
group; comms_viz warns with the affected ranks (stderr, page banner,
structured report).

`--write-repaired-traces` also writes each rank's "repaired" trace to
`tree/<run-id>/traces/`, after annotating kernel metadata, dropping stray
annotation events, and moving overlapping graph kernels onto a separate
stream.

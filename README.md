# comms_viz

comms_viz is an analysis tool for visualizing cross-rank timelines for distributed training jobs, and
identifying per-rank inefficiencies like stragglers, lateness, and compute/communication
imbalance. comms_viz reads per-rank PyTorch profiling traces, and aggregates them into a single report that answers:

- which ranks arrive late to which collectives, and how often they do so (persistent stragglers)
- which ranks and collective types consume the most time
- how each rank's time splits between compute, collectives and CPU

The **Breakdown** tab plots each rank's total time per collective
type and process group:

![Breakdown tab](docs/images/breakdown.png)

The **Parallelism** tab shows one process group at a time (e.g. one TP or
FSDP group), with each rank's comm kernels side by side on a shared timeline,
alongside per-rank utilization and GPU counters (SM activity, DRAM and NVLink
bandwidth) when the traces carry them.

It also shows a per-collective breakdown of rank arrival time skew and imbalance, and a per-rank breakdown
of time spent in compute, collectives, and active GPU / CPU time:

![Parallelism tab](docs/images/parallelism.png)

## Usage

```sh
./target/release/comms_viz --trace-root ./traces/step_10 --out ./analysis
python3 -m http.server --directory ./analysis/report 8000
```

comms_viz expects you to gather every rank's trace for a profiling step into a single directory first. The command analyzes the traces into
`./analysis/tree/<run-id>/`, writes the report page and its `.pgdata` data
files to `./analysis/report/` and the structured report to
`./analysis/structured_report/`, for rendering as local HTML page.

| Flag | Description |
| --- | --- |
| `--trace-root DIR` | One step's traces, one file per rank, all in this directory. File names: see `--trace-format`. Omit the flag to rebuild the reports from an existing `--out` without re-analyzing. |
| `--out DIR` | Where analysis results and, by default, both reports are written. Required. |
| `--report DIR` | Where the HTML report is written. Default `<out>/report`. |
| `--structured-report DIR` | Where the analysis is written as a Markdown summary and CSV tables, for scripts or LLMs. Default `<out>/structured_report`. |
| `--run-id ID` | Name of the run's subdirectory under `<out>/tree/`, also used in the report file names. Default `local-run`, or `local-run-2`, `-3`, ... if taken. An explicit id that is taken is an error. |
| `--overwrite` | Replace an existing run with the same id. |
| `--run-name NAME` | Run name shown in the report. |
| `--trace-format auto\|json\|pftrace` | `json` reads `*.json` and `*.json.gz` with any name; the rank comes from the trace. `pftrace` reads `*.pftrace` and `*.pftrace.gz`; the rank comes from the file name (see [Input](#input)). `auto` (default) picks whichever is present; pass the format if the directory has both. |
| `-v`, `--verbose` | Print per-rank analysis time. |

## Installation

Install rustup, then run:

```sh
cargo build --release --locked
```

### PyTorch requirements

- PyTorch's Kineto backend works best with PyTorch 2.6 or later, for DeviceMesh axis names
  to be properly emitted.
- `.pftrace` traces come from PyTorch's Cuspy profiler backend, currently only in nightlies,
  but likely to be released in PyTorch 2.15.

## CUDA graphs

Collectives replayed from a CUDA graph carry process-group metadata only if
the graph is captured with `torch.cuda.graph(..., enable_annotations=True)`.
For Kineto backend, ensure your framework exports the trace with
`export_chrome_trace(path, cuda_graph_annotations=get_kernel_annotations())`.

Frameworks like TorchTitan already do both steps, but without them, collective
kernels aren't properly recognized and attributed to PGs by comms_viz.

## Custom collectives

PyTorch attaches process-group metadata only to c10d collectives. comms_viz also recognizes a few other
collective libraries by kernel name (e.g. HybridEP, DeepEP) and counts their
kernels as communication, but they appear on no group timeline and are not
properly matched across ranks yet. comms_viz doesn't currently provide a clean API
to extend to recognizing custom collective kernels, but we are currently working on this.

## Provenance

comms_viz is an entirely AI-written codebase, with human direction over its
features, metrics and visualizations, and human use on real large-scale
training analysis. Most of the code has had little or no line-by-line human
review.

Contributions and fixes are welcome, to be judged mainly on functionality.

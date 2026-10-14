# Synthetic test traces

comms_viz's `gen_synthetic_traces.py` generates synthetic multi-rank PyTorch
profiler traces (Kineto JSON) for testing comms_viz without running a training
job. The integration tests in `crates/comms-viz-cli/tests/` run on the synthetic traces in
this directory.

```sh
python3 gen_synthetic_traces.py OUT_DIR [--ranks N] [--events N] [--seed N] [--step N] [--mode MODE]
```

| Flag | Description |
| --- | --- |
| `OUT_DIR` | output folder; traces are written to `OUT_DIR/step_<step>/`, one `<rank>.<ts>.pt.trace.json.gz` per rank. |
| `--ranks N` | number of ranks. artificially, all 6 PGs (`dp_shard`, `tp`, `cp`, `ep`, `dp_replicate`, default) each span all ranks. default = 8. |
| `--events N` | number of CPU op and kernel launches per rank; every third launch is an NCCL collective, cycling through the process groups. default = 400. |
| `--seed N` | random seed for kernel names, durations and gaps. default = 7. |
| `--step N` | profiled step number, used for the folder name. default = 106. |
| `--mode MODE` | kind of trace to generate, see below. default = `plain`. |

Modes:

- `plain`: eager traces, with collectives tagged with their process group.
- `inline`: CUDA-graph replay traces, with the graph annotations stored in the
  trace itself.
- `cgp`: CUDA-graph replay traces, with the annotations in separate
  `kernel_annotations/*.pkl` files, plus a recording step
  (`step_000000000002`).

## Traces in this directory

```sh
python3 gen_synthetic_traces.py synth  --ranks 8 --mode plain
python3 gen_synthetic_traces.py inline --ranks 4 --mode inline
python3 gen_synthetic_traces.py cgp    --ranks 4 --mode cgp
```

- `cgp-norec/` is `cgp/` without the recording step.
- `*-pftrace/` hold the same traces converted to Cuspy's `.pftrace` format with
  `comms_viz --trace-root <step dir> --convert-pftrace <out dir>`, to test the
  `.pftrace` parser.

If you regenerate these traces, also regenerate the `.pftrace` copies.

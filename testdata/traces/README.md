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

## DeviceMesh-shaped traces with known faults

`gen_mesh_traces.py` writes traces meant to be analyzed for meaning: every
rank lists only its own process groups, collectives carry sequence numbers
and names that agree across a group, and timing comes from a lockstep
simulation in which a collective ends after its last member arrives. Faults
can be injected and are recorded in `ground_truth.json`.

```sh
python3 gen_mesh_traces.py OUT_DIR --mesh pp=4,dp_replicate=24,dp_shard=64,cp=2,tp=4 --ep 32 \
    [--straggler RANK:FACTOR] [--slow-node NODE:FACTOR] [--slow-pp-link RANK:FACTOR] \
    [--strip-annotations RANK:FRACTION] [--custom-kernels] [--counters] [--workers N]
```

`--kernels-per-compute` sets the trace size: 100 gives about 90k events
(2 MB gzipped) per rank, close to a real step. `mesh/` is the fixture of
`tests/integration/semantics.rs`:

```sh
python3 gen_mesh_traces.py mesh --mesh pp=4,dp_shard=2,tp=2 --ep 2 --layers 2 --microbatches 2 \
    --kernels-per-compute 3 --straggler 5:1.3 --slow-node 3:2.0 --slow-pp-link 1:3.0 \
    --strip-annotations 2:0.5 --custom-kernels --counters --workers 1
```

## Scale

Analysis time and peak memory of `comms_viz --trace-root <dir> --out <out>` (analysis,
HTML report and structured report) on traces from `gen_mesh_traces.py`, ~84k events and
~1 MB gzipped per rank, GPU counters and unannotated custom kernels included. Measured on
one 368-core, 2 TB host with comms_viz using every core; the memory gate sizes its
concurrency to the host's memory, so peak RSS on a smaller host will be lower and runs
slower.

| ranks | pp | dp_replicate | dp_shard | cp | tp | ep | process groups | wall | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| 12,288 | 1 | 12 | 256 | 1 | 4 | 64 | 4,336 | 0:53 | 32 GB |
| 20,480 | 4 | 20 | 32 | 2 | 4 | 32 | 53,504 | 1:57 | 46 GB |
| 45,056 | 8 | 22 | 32 | 2 | 4 | 16 | 124,544 | 5:05 | 82 GB |
| 49,152 | 4 | 24 | 64 | 2 | 4 | 32 | 127,232 | 4:42 | 87 GB |

Commands (generation took 0.5-5 minutes with 300 workers; traces are 13-57 GB):

```sh
python3 gen_mesh_traces.py r12k --mesh dp_replicate=12,dp_shard=256,tp=4 --ep 64 \
    --kernels-per-compute 100 --workers 300 --counters --custom-kernels \
    --straggler 5000:1.2 --straggler 9001:1.1 --slow-node 1500:1.8 --strip-annotations 7000:0.4
python3 gen_mesh_traces.py r20k --mesh pp=4,dp_replicate=20,dp_shard=32,cp=2,tp=4 --ep 32 \
    --kernels-per-compute 100 --workers 300 --counters --custom-kernels \
    --straggler 3333:1.2 --slow-node 2000:1.8 --slow-pp-link 4100:3.0 --strip-annotations 15000:0.5
python3 gen_mesh_traces.py r45k --mesh pp=8,dp_replicate=22,dp_shard=32,cp=2,tp=4 --ep 16 \
    --kernels-per-compute 100 --workers 300 --counters --custom-kernels \
    --straggler 22222:1.15 --straggler 40001:1.25 --slow-node 7000:1.6 --strip-annotations 123:0.3
python3 gen_mesh_traces.py r49k --mesh pp=4,dp_replicate=24,dp_shard=64,cp=2,tp=4 --ep 32 \
    --kernels-per-compute 100 --workers 300 --counters --custom-kernels \
    --straggler 31415:1.2 --slow-node 9999:1.8 --slow-pp-link 27182:3.0 --strip-annotations 44444:0.4

comms_viz --trace-root r49k/step_000000000010 --out r49k_out
```

At 49k ranks parsing the traces takes ~45 s; the rest is writing the per-group tables
(about two files per process group) and moving them into place, which scales with the
number of process groups rather than ranks. The structured report stays ~20 KB (README)
and ~60 KB (JSON) at every size.

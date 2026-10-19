#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate a synthetic Kineto-style trace folder for parity testing.

Produces one `.pt.trace.json.gz` per rank in the flat layout
(`{rank:06d}.{ts}.pt.trace.json.gz`), with event mixes that exercise the
converter coercions, the external-sort tie rules, kernel classification,
PG extraction, and the breakdown queries. Deterministic per (rank, seed).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random

# (pg_name, pg_desc): one process group per description, spanning all ranks.
PROCESS_GROUPS = [
    ("1", "mesh_dp_shard"),
    ("2", "mesh_tp"),
    ("3", "mesh_cp"),
    ("4", "mesh_ep"),
    ("5", "mesh_dp_replicate"),
    ("0", "default_pg"),
]


def pg_config(world: int) -> list[dict]:
    """`distributedInfo.pg_config` as the PyTorch profiler writes it."""
    return [
        {
            "pg_name": name,
            "pg_desc": desc,
            "backend_config": "cuda:nccl",
            "pg_size": world,
            "ranks": list(range(world)),
        }
        for name, desc in PROCESS_GROUPS
    ]


def make_events(rank: int, n_base: int, seed: int, world: int) -> list[dict]:
    rng = random.Random(seed * 1000003 + rank)
    events: list[dict] = []
    t0 = 1_700_000_000_000_000 + rank  # µs epoch-ish

    # Metadata events (ph=M, no cat/ts) — kept by converter, dropped by prepare.
    for tid, name in [(7, "stream 7"), (20, "stream 20")]:
        events.append(
            {
                "ph": "M",
                "pid": 3,
                "tid": tid,
                "name": "thread_name",
                "args": {"name": name},
            }
        )

    nccl_names = [
        "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        "void ncclKernel_SendRecv(ncclDevKernelArgs4K)",
    ]
    compute_names = [
        "triton_red_fused_gemm_123",
        "void cutlass::Kernel2<cutlass_80_tensorop_gemm>(params)",
        "flash_fwd_kernel<Flash_fwd_kernel_traits<128>>",
        "triton_poi_fused_elementwise_88",
        "void at::native::vectorized_elementwise_kernel<4, ...>",
        "std::enable_if<!(false), void>::type internal::gemvx::kernel",
        "grouped_gemm_persistent_kernel",
        "nvjet_tst_128x128",
        "aten_softmax_warp_forward",
        "void at::native::(anonymous namespace)::LayerNormForward",
    ]

    t = float(t0)
    for i in range(n_base):
        # CPU op + runtime launch + kernel triplet.
        cpu_ts = t
        events.append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::mm" if i % 7 else "aten::_fused_adamw_",
                "pid": 3,
                "tid": 1,
                "ts": int(cpu_ts),
                "dur": rng.randint(2, 20),
                "args": {"External id": i},
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": rng.choice(
                    [
                        "cudaLaunchKernel",
                        "cudaEventSynchronize",
                        "cudaStreamSynchronize",
                        "cudaHostAlloc",
                        "cudaMemcpyAsync",
                    ]
                ),
                "pid": 3,
                "tid": 1,
                "ts": int(cpu_ts) + 1,
                "dur": rng.randint(1, 10),
                "args": {"correlation": i},
            }
        )
        is_comm = i % 3 == 0
        if is_comm:
            pg_name, pg_desc = PROCESS_GROUPS[(i // 3) % len(PROCESS_GROUPS)]
            name = nccl_names[(i // 3) % len(nccl_names)]
            ev = {
                "ph": "X",
                "cat": "kernel",
                "name": name,
                "pid": 3,
                "tid": 20 + (i % 3),
                "ts": int(t) + 5,
                "dur": rng.randint(50, 500),
                "args": {
                    "stream": 20 + (i % 3),
                    "correlation": i,
                    "grid": [rng.randint(1, 132), 1, 1],
                    "Process Group Description": pg_desc,
                    "Process Group Name": pg_name,
                    "Group size": world,
                    "In msg nelems": rng.randint(1024, 1 << 20),
                    "Out msg nelems": rng.randint(1024, 1 << 20),
                    "dtype": rng.choice(["Float", "BFloat16"]),
                    "Collective name": rng.choice(
                        ["allreduce", "all_gather", "reduce_scatter"]
                    ),
                },
            }
            events.append(ev)
        else:
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": compute_names[i % len(compute_names)],
                    "pid": 3,
                    "tid": 7,
                    "ts": int(t) + 5,
                    "dur": rng.randint(5, 200),
                    "args": {
                        "stream": 7,
                        "correlation": i,
                        "grid": [rng.randint(1, 8192), 1, 1],
                    },
                }
            )
        # gpu_user_annotation occasionally (all_to_all totals).
        if i % 11 == 0:
            events.append(
                {
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "name": "nccl:all_to_all",
                    "pid": 3,
                    "tid": 20,
                    "ts": int(t) + 6,
                    "dur": rng.randint(10, 100),
                }
            )
        # ac2g flow events: no dur (NULL-dur sort tie exercise).
        if i % 5 == 0:
            events.append(
                {
                    "ph": "s",
                    "cat": "ac2g",
                    "id": i,
                    "pid": 3,
                    "tid": 1,
                    "ts": int(cpu_ts) + 1,
                }
            )
            events.append(
                {
                    "ph": "f",
                    "cat": "ac2g",
                    "id": i,
                    "pid": 3,
                    "tid": 20,
                    "ts": int(t) + 5,
                    "bp": "e",
                }
            )
        # Tie stress: same-ts events with dur 0 / equal durs.
        if i % 13 == 0:
            tie_ts = int(t) + 5
            events.append(
                {
                    "ph": "i",
                    "cat": "user_annotation",
                    "name": "marker_a",
                    "pid": 3,
                    "tid": 1,
                    "ts": tie_ts,
                    "dur": 0,
                }
            )
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "triton_poi_fused_tie_kernel",
                    "pid": 3,
                    "tid": 7,
                    "ts": tie_ts,
                    "dur": 42,
                    "args": {"stream": 7},
                }
            )
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "triton_poi_fused_tie_kernel_b",
                    "pid": 3,
                    "tid": 7,
                    "ts": tie_ts,
                    "dur": 42,
                    "args": {"stream": 7},
                }
            )
        # Coercion edge cases (C++ semantics — both sides must agree).
        # NOTE: kept OFF cat="kernel": V1's deep merge uses raw args.stream
        # values as dict keys (a string stream is a valid bucket there),
        # while comms_viz (and its parity suite) assumes
        # numeric streams — real Kineto traces always emit ints. A string
        # stream on a kernel would diverge only in the merged-trace
        # artifact of CGP runs. user_annotation (ignored by the merge) still
        # exercises the converter's
        # string/float coercion paths.
        if i % 17 == 0:
            events.append(
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "Memcpy DtoD (Device -> Device)",
                    "pid": "3",  # string pid → stoll
                    "tid": 7.9,  # double tid → trunc 7
                    "ts": int(t) + 7,
                    "dur": 3.5,  # fractional dur
                    "args": {"stream": "7abc", "grid": [], "correlation": None},
                }
            )
        t += rng.randint(5, 40)

    return events


def add_graph_metadata(
    events: list[dict], rank: int, seed: int, graph_ids: list[int]
) -> None:
    """Tag GPU kernels with `graph node id` and inject overlap conditions so
    the CUDA-graph walks (annotate / move-overlap / ts-clamp) have work."""
    rng = random.Random(seed * 7919 + rank)
    node = 1
    prev: dict | None = None
    for ev in events:
        if ev.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        gid = graph_ids[node % len(graph_ids)]
        ev.setdefault("args", {})["graph node id"] = (gid << 32) | node
        node += 1
        # Force graphed kernels onto the default stream (tid 7) so the
        # move-overlapping pass has candidates.
        if rng.random() < 0.5:
            ev["tid"] = 7
            ev["args"]["stream"] = 7
        # Small ts overlaps against the previous graphed kernel (clamp pass).
        if prev is not None and rng.random() < 0.3:
            ev["ts"] = prev["ts"] + prev.get("dur", 0) - 1  # 1us overlap
        prev = ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--events", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--step", type=int, default=106, help="step number for the folder name"
    )
    ap.add_argument(
        "--mode",
        choices=["plain", "inline", "cgp"],
        default="plain",
        help="plain: raw kineto; inline: cudaGraphInlineAnnotated envelope; "
        "cgp: replay + recording step + annotation pkls",
    )
    args = ap.parse_args()

    import os
    import pickle

    graph_ids = [3, 9]

    def write_step(step: int, ranks: int, events_fn) -> str:
        step_dir = os.path.join(args.out_dir, f"step_{step:012d}")
        os.makedirs(step_dir, exist_ok=True)
        for rank in range(ranks):
            envelope, events = events_fn(rank)
            trace = {
                "schemaVersion": 1,
                **envelope,
                "distributedInfo": {
                    "rank": rank,
                    "world_size": ranks,
                    "pg_config": pg_config(ranks),
                },
                "traceEvents": events,
                "traceName": f"/tmp/rank{rank}.json",
                "displayTimeUnit": "ms",
            }
            payload = json.dumps(trace).encode()
            fname = f"{rank:06d}.1700000000000.pt.trace.json.gz"
            with open(os.path.join(step_dir, fname), "wb") as f:
                f.write(gzip.compress(payload, 5))
        return step_dir

    if args.mode == "plain":
        step_dir = write_step(
            args.step,
            args.ranks,
            lambda rank: ({}, make_events(rank, args.events, args.seed, args.ranks)),
        )
    elif args.mode == "inline":

        def inline_events(rank: int):
            events = make_events(rank, args.events, args.seed, args.ranks)
            add_graph_metadata(events, rank, args.seed, graph_ids)
            return {"cudaGraphInlineAnnotated": True}, events

        step_dir = write_step(args.step, args.ranks, inline_events)
    else:  # cgp
        # Recording step 2: kernels carry extra metadata args.
        def rec_events(rank: int):
            events = make_events(rank, args.events, args.seed + 1, args.ranks)
            for ev in events:
                if ev.get("cat") == "kernel":
                    ev.setdefault("args", {})["recording_extra"] = "from_recording"
            return {}, events

        write_step(2, args.ranks, rec_events)

        # Replay step: graphed kernels, thinner args.
        def replay_events(rank: int):
            events = make_events(rank, args.events, args.seed, args.ranks)
            add_graph_metadata(events, rank, args.seed, graph_ids)
            return {}, events

        step_dir = write_step(args.step, args.ranks, replay_events)

        # Annotation pkls: kernel_annotations/kernel_annotations_rank{R}_fwd_{gid}.pkl
        ann_dir = os.path.join(args.out_dir, "kernel_annotations")
        os.makedirs(ann_dir, exist_ok=True)
        for rank in range(args.ranks):
            for gid in graph_ids + [17]:  # 17 = stale graph id (filtered out)
                ann: dict[int, list[dict]] = {}
                for node in range(1, args.events):
                    ann[(gid << 32) | node] = [
                        {"stream": 20 + (node % 3), "annotation": f"g{gid}_n{node}"}
                    ]
                with open(
                    os.path.join(
                        ann_dir, f"kernel_annotations_rank{rank}_fwd_{gid}.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(ann, f)

    print(f"wrote {args.ranks} ranks under {step_dir} (mode={args.mode})")


if __name__ == "__main__":
    main()

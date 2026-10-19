#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate Kineto traces for a DeviceMesh-shaped training step.

Unlike gen_synthetic_traces.py (parser coverage), these traces are meant to
be analyzed for meaning: every rank lists only its own process groups in
`distributedInfo.pg_config`, collectives carry c10d sequence numbers and
names that agree across a group, and kernel timing comes from a lockstep
simulation in which a collective ends only after its last member arrives.
Faults with known answers can be injected (stragglers, a slow node, a slow
pipeline link, lost annotations, unannotated custom kernels, GPU counters),
and are written to `ground_truth.json` next to the step directory.

Mesh axes, outermost first: pp, dp_replicate, dp_shard, cp, tp. Expert
parallel groups split the (dp_shard, cp) ranks of one (pp, dp_replicate, tp)
slice into consecutive groups of `--ep` ranks, as torchtitan borrows EP from
dp_shard x cp. Pipeline p2p groups mirror torch.distributed.pipelining's
`split_group` rounds (_p2p.py): adjacent stage pairs are split into disjoint
matchings (even s, then odd s), each matching gets a forward round
(s -> s+1) and a reverse round, and all pairs of a round share one name.

Stdlib only, so it runs anywhere the tests do. Large worlds are written in
parallel with `--workers`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import random
from array import array
from multiprocessing import Pool

AXES = ["pp", "dp_replicate", "dp_shard", "cp", "tp"]
T0 = 1_700_000_000_000.0  # trace epoch, us
COMPUTE_KERNELS = [
    "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn_align8>(Params)",
    "nvjet_tst_192x192_64x3_2x1_v_bz_coopA_NNT",
    "flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>>",
    "triton_poi_fused__to_copy_add_mul_rsqrt_0",
    "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1",
    "void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float>>",
    "void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float>",
    "grouped_gemm_persistent_kernel",
]
NCCL_KERNELS = {
    "all_gather": "ncclDevKernel_AllGather_RING(ncclDevKernelArgsStorage<4096ul>)",
    "allGatherSingleImpl": "ncclDevKernel_AllGather_RING(ncclDevKernelArgsStorage<4096ul>)",
    "reduce_scatter": "ncclDevKernel_ReduceScatter_bf16_RING(ncclDevKernelArgsStorage<4096ul>)",
    "reduceScatterSingleImpl": "ncclDevKernel_ReduceScatter_bf16_RING(ncclDevKernelArgsStorage<4096ul>)",
    "all_reduce": "ncclDevKernel_AllReduce_Sum_f32_RING(ncclDevKernelArgsStorage<4096ul>)",
    "all_to_all": "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)",
    "send": "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)",
    "recv": "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)",
}
CUSTOM_KERNEL = "void hybrid_ep::dispatch_kernel<128, 8>(hybrid_ep::DispatchArgs)"
COUNTERS = ["SM Active (%)", "DRAM BW (%)", "NVLink RX", "NVLink TX", "Power (W)"]


def pg_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class Mesh:
    def __init__(
        self, dims: dict[str, int], ep: int, gpus_per_node: int, p2p_unique_names: bool = False
    ):
        self.dims = {a: dims.get(a, 1) for a in AXES}
        self.shape = [self.dims[a] for a in AXES]
        self.world = math.prod(self.shape)
        self.ep = ep
        self.gpus_per_node = gpus_per_node
        if ep > 1 and (self.dims["dp_shard"] * self.dims["cp"]) % ep:
            raise ValueError("--ep must divide dp_shard * cp")
        strides, s = [], 1
        for n in reversed(self.shape):
            strides.append(s)
            s *= n
        self.strides = list(reversed(strides))
        # name -> (desc, list of member-rank lists); axes with one rank have
        # no NCCL group and are skipped, as are pipeline p2p rounds at pp=1.
        self.groups: dict[str, tuple[str, list[list[int]]]] = {}
        for axis in AXES:
            if self.dims[axis] > 1:
                self.groups[axis] = (f"mesh_{axis}", self._axis_groups([axis]))
        if ep > 1:
            self.groups["ep"] = ("mesh_ep", self._ep_groups())
        for rnd in range(4):
            pairs = self._p2p_pairs(rnd // 2)
            if pairs:
                self.groups[f"pp_p2p_{rnd}"] = (f"pp_p2p_round_{rnd}", pairs)
        # Every group's c10d name. DeviceMesh names each group by a hash;
        # split_group gives all subgroups of one p2p round the same name.
        self.names: dict[str, list[str]] = {}
        for key, (desc, groups) in self.groups.items():
            if key.startswith("pp_p2p") and not p2p_unique_names:
                self.names[key] = [pg_hash(desc)] * len(groups)
            else:
                self.names[key] = [pg_hash(f"{key}:{g[0]}:{len(g)}") for g in groups]
        self.group_of: dict[str, dict[int, int]] = {
            key: {r: i for i, g in enumerate(groups) for r in g}
            for key, (_, groups) in self.groups.items()
        }

    def rank(self, coords: list[int]) -> int:
        return sum(c * s for c, s in zip(coords, self.strides))

    def _axis_groups(self, axes: list[str]) -> list[list[int]]:
        idx = [AXES.index(a) for a in axes]
        others = [range(n) if i not in idx else [0] for i, n in enumerate(self.shape)]
        out = []
        for base in itertools.product(*others):
            group = []
            for inner in itertools.product(*[range(self.shape[i]) for i in idx]):
                coords = list(base)
                for i, c in zip(idx, inner):
                    coords[i] = c
                group.append(self.rank(coords))
            out.append(sorted(group))
        return out

    def _ep_groups(self) -> list[list[int]]:
        out = []
        for flat in self._axis_groups(["dp_shard", "cp"]):
            for i in range(0, len(flat), self.ep):
                out.append(flat[i : i + self.ep])
        return out

    def _p2p_pairs(self, matching: int) -> list[list[int]]:
        """Stage pairs (s, s+1) with s % 2 == matching, as [lower, higher]."""
        pp = self.dims["pp"]
        out = []
        for chain in self._axis_groups(["pp"]) if pp > 1 else []:
            for s in range(matching, pp - 1, 2):
                out.append([chain[s], chain[s + 1]])
        return out

    def p2p_pairs(self, direction: str) -> list[tuple[str, int, int, int]]:
        """(group key, group index, src, dst) per stage pair, pipeline order."""
        out = []
        for matching in range(2):
            key = f"pp_p2p_{2 * matching + (direction == 'bwd')}"
            for gi, (a, b) in enumerate(self.groups.get(key, ("", []))[1]):
                src, dst = (a, b) if direction == "fwd" else (b, a)
                out.append((key, gi, src, dst))
        out.sort(key=lambda e: min(e[2], e[3]), reverse=direction == "bwd")
        return out

    def node(self, rank: int) -> int:
        return rank // self.gpus_per_node


def build_schedule(mesh: Mesh, layers: int, microbatches: int) -> list[tuple]:
    """One training step as ops every rank runs in the same order.

    ("compute", name, base_us) or ("coll", group_key, collective, nelems, base_us)
    or ("p2p", direction, nelems, base_us). Collectives of an axis the mesh
    doesn't have are skipped.
    """
    d = mesh.dims
    ops: list[tuple] = []
    has = mesh.groups.__contains__

    def coll(key: str, name: str, nelems: int, us: float) -> None:
        if has(key):
            ops.append(("coll", key, name, nelems, us))

    for m in range(microbatches):
        for layer in range(layers):
            if m == 0:
                coll("dp_shard", "allGatherSingleImpl", 16 << 20, 900)
            ops.append(("compute", "norm", 60))
            coll("tp", "allGatherSingleImpl", 8 << 20, 120)
            if has("cp"):
                coll("cp", "all_to_all", 4 << 20, 90)
                ops.append(("compute", "attn", 700))
                coll("cp", "all_to_all", 4 << 20, 90)
            else:
                ops.append(("compute", "attn", 700))
            coll("tp", "reduceScatterSingleImpl", 8 << 20, 120)
            if has("ep"):
                coll("ep", "all_to_all", 6 << 20, 260)
                ops.append(("compute", "experts", 900))
                coll("ep", "all_to_all", 6 << 20, 260)
            else:
                ops.append(("compute", "mlp", 900))
        if d["pp"] > 1:
            ops.append(("p2p", "fwd", 16 << 20, 150))
    for m in range(microbatches):
        if d["pp"] > 1:
            ops.append(("p2p", "bwd", 16 << 20, 150))
        for layer in reversed(range(layers)):
            if has("ep"):
                coll("ep", "all_to_all", 6 << 20, 260)
                ops.append(("compute", "experts_bwd", 1800))
                coll("ep", "all_to_all", 6 << 20, 260)
            else:
                ops.append(("compute", "mlp_bwd", 1800))
            coll("tp", "allGatherSingleImpl", 8 << 20, 120)
            ops.append(("compute", "attn_bwd", 1400))
            if has("cp"):
                coll("cp", "all_to_all", 4 << 20, 90)
                coll("cp", "all_to_all", 4 << 20, 90)
            coll("tp", "reduceScatterSingleImpl", 8 << 20, 120)
            if m == microbatches - 1:
                coll("dp_shard", "reduceScatterSingleImpl", 16 << 20, 1100)
    for _ in range(4):
        coll("dp_replicate", "all_reduce", 8 << 20, 1500)
    for _ in range(3):
        coll("pp", "all_reduce", 1, 25)
    ops.append(("compute", "optimizer", 2500))
    return ops


class Sim:
    """Per-op start/end of every rank, from a lockstep run of the schedule."""

    def __init__(self, mesh: Mesh, ops: list[tuple], faults: dict, seed: int):
        world = mesh.world
        rng = random.Random(seed)
        jitter = [rng.gauss(0.0, 0.03) for _ in range(1 << 16)]
        compute_factor = [1.0] * world
        for r, f in faults["stragglers"].items():
            compute_factor[int(r)] = f
        link_factor = [1.0] * world
        for n, f in faults["slow_nodes"].items():
            for r in range(int(n) * mesh.gpus_per_node, (int(n) + 1) * mesh.gpus_per_node):
                link_factor[r] = f
        slow_pairs = {int(r): f for r, f in faults["slow_pp_links"].items()}
        clock = array("d", (abs(jitter[r & 0xFFFF]) * 20 for r in range(world)))
        self.start: list[array] = []
        self.end: list[array] = []
        for i, op in enumerate(ops):
            start = array("d", clock)
            if op[0] == "compute":
                base = op[2]
                for r in range(world):
                    j = jitter[(i * 7919 + r * 104729) & 0xFFFF]
                    clock[r] += base * compute_factor[r] * (1.0 + j)
            elif op[0] == "coll":
                for g in mesh.groups[op[1]][1]:
                    t = max(start[r] for r in g) + 8.0
                    dur = op[4] * max(link_factor[r] for r in g)
                    for r in g:
                        clock[r] = t + dur
            else:  # p2p: stage pairs in pipeline order, so the bubble forms
                for _, _, a, b in mesh.p2p_pairs(op[1]):
                    f = max(slow_pairs.get(a, 1.0), slow_pairs.get(b, 1.0))
                    t = max(clock[a], clock[b]) + 8.0 + op[3] * f
                    clock[a] = clock[b] = t
            self.start.append(start)
            self.end.append(array("d", clock))


def sanitize_faults(args, mesh: Mesh) -> dict:
    def parse(items, key_type=int):
        out = {}
        for item in items or []:
            k, v = item.split(":")
            out[str(key_type(k))] = float(v)
        return out

    faults = {
        "stragglers": parse(args.straggler),
        "slow_nodes": parse(args.slow_node),
        "slow_pp_links": parse(args.slow_pp_link),
        "stripped_annotations": parse(args.strip_annotations),
        "custom_kernels": args.custom_kernels,
        "counters": args.counters,
        "low_sm_nodes": list(parse(args.slow_node).keys()),
    }
    for r in list(faults["stragglers"]) + list(faults["stripped_annotations"]):
        if int(r) >= mesh.world:
            raise ValueError(f"rank {r} outside world {mesh.world}")
    return faults


# Globals for forked writer processes.
G: dict = {}


def rank_pg_config(mesh: Mesh, rank: int) -> list[dict]:
    pgs = [
        {
            "pg_name": "0",
            "pg_desc": "default_pg",
            "backend_config": "cuda:nccl",
            "pg_size": mesh.world,
            "ranks": list(range(mesh.world)),
        }
    ]
    for key, (desc, groups) in mesh.groups.items():
        gi = mesh.group_of[key].get(rank)
        if gi is None:
            continue
        pgs.append(
            {
                "pg_name": mesh.names[key][gi],
                "pg_desc": desc,
                "backend_config": "cuda:nccl",
                "pg_size": len(groups[gi]),
                "ranks": groups[gi],
            }
        )
    return pgs


def write_rank(rank: int) -> int:
    mesh: Mesh = G["mesh"]
    ops = G["ops"]
    sim: Sim = G["sim"]
    faults = G["faults"]
    kpc = G["kernels_per_compute"]
    strip = faults["stripped_annotations"].get(str(rank), 0.0)
    pid, tid_cpu = 1000 + rank, 1000 + rank
    seq: dict[str, int] = {}
    out: list[str] = []
    corr = 0
    ext = 0
    w = out.append

    def launch(ts: float, kernel_ts: float, cpu_name: str, cpu_args: str) -> int:
        nonlocal corr, ext
        corr += 1
        ext += 1
        cpu_ts = max(ts - 40.0, T0)
        w(
            f'{{"ph":"X","cat":"cpu_op","name":"{cpu_name}","pid":{pid},"tid":{tid_cpu},'
            f'"ts":{cpu_ts:.3f},"dur":6.0,"args":{{"External id":{ext}{cpu_args}}}}}'
        )
        w(
            f'{{"ph":"X","cat":"cuda_runtime","name":"cudaLaunchKernel","pid":{pid},"tid":{tid_cpu},'
            f'"ts":{cpu_ts + 1:.3f},"dur":4.0,"args":{{"External id":{ext},"correlation":{corr}}}}}'
        )
        w(f'{{"ph":"s","id":{corr},"pid":{pid},"tid":{tid_cpu},"ts":{cpu_ts + 1:.3f},"cat":"ac2g","name":"ac2g"}}')
        w(f'{{"ph":"f","id":{corr},"pid":0,"tid":7,"ts":{kernel_ts:.3f},"cat":"ac2g","name":"ac2g","bp":"e"}}')
        return corr

    step_start, step_end = T0 + sim.start[0][rank], T0 + sim.end[-1][rank]
    w(
        f'{{"ph":"X","cat":"user_annotation","name":"ProfilerStep#{G["step"]}","pid":{pid},'
        f'"tid":{tid_cpu},"ts":{step_start - 50:.3f},"dur":{step_end - step_start + 50:.3f}}}'
    )
    for i, op in enumerate(ops):
        s = T0 + sim.start[i][rank]
        e = T0 + sim.end[i][rank]
        if op[0] == "compute":
            seg = (e - s) / kpc
            for k in range(kpc):
                kts = s + k * seg
                name = COMPUTE_KERNELS[(i + k) % len(COMPUTE_KERNELS)]
                c = launch(kts, kts, "aten::mm", "")
                w(
                    f'{{"ph":"X","cat":"kernel","name":"{name}","pid":0,"tid":7,"ts":{kts:.3f},'
                    f'"dur":{seg * 0.92:.3f},"args":{{"External id":{ext},"device":0,"stream":7,'
                    f'"correlation":{c},"grid":[132,1,1],"block":[128,1,1]}}}}'
                )
            if faults["custom_kernels"] and op[1] == "experts":
                c = launch(s, s, "hybrid_ep::dispatch", "")
                w(
                    f'{{"ph":"X","cat":"kernel","name":"{CUSTOM_KERNEL}","pid":0,"tid":31,"ts":{s:.3f},'
                    f'"dur":{min(40.0, e - s):.3f},"args":{{"External id":{ext},"device":0,"stream":31,"correlation":{c}}}}}'
                )
            continue
        if op[0] == "coll":
            key, cname, nelems = op[1], op[2], op[3]
            gi = mesh.group_of[key][rank]
            entries = [(key, gi, cname, s, e, nelems, len(mesh.groups[key][1][gi]))]
        else:  # p2p: this rank's send and/or recv for the direction
            entries = [
                (key, gi, "send" if rank == src else "recv", s, e, op[2], 2)
                for key, gi, src, dst in mesh.p2p_pairs(op[1])
                if rank in (src, dst)
            ]
        for key, gi, cname, ks, ke, nelems, gsize in entries:
            name = mesh.names[key][gi]
            stream = 20 + list(mesh.groups).index(key)
            if cname in ("send", "recv"):
                sq = 0  # c10d p2p ops carry seq 0 in Kineto traces
            else:
                sq = seq.get(name, 0) + 1
                seq[name] = sq
            lost = strip > 0 and ((i * 2654435761 + rank) % 1000) < strip * 1000
            meta = (
                f',"Collective name":"{cname}","In msg nelems":{nelems},"Out msg nelems":{nelems},'
                f'"Group size":{gsize},"dtype":"BFloat16","Process Group Name":"{name}",'
                f'"Process Group Ranks":"[]","Seq":{sq}'
            )
            c = launch(ks, ks, "record_param_comms" if not lost else "cudaGraphLaunch", meta if not lost else "")
            kmeta = "" if lost else meta
            w(
                f'{{"ph":"X","cat":"kernel","name":"{NCCL_KERNELS[cname]}","pid":0,"tid":{stream},'
                f'"ts":{ks:.3f},"dur":{ke - ks:.3f},"args":{{"External id":{ext},"device":0,'
                f'"stream":{stream},"correlation":{c},"grid":[32,1,1],"block":[544,1,1]{kmeta}}}}}'
            )

    if faults["counters"]:
        rng = random.Random(rank * 31 + 7)
        low = str(mesh.node(rank)) in faults["low_sm_nodes"]
        t = step_start
        while t < step_end:
            for metric in COUNTERS:
                base = {"SM Active (%)": 35.0 if low else 72.0, "DRAM BW (%)": 41.0,
                        "NVLink RX": 180.0, "NVLink TX": 180.0, "Power (W)": 950.0}[metric]
                v = base * (1.0 + rng.gauss(0.0, 0.08))
                w(f'{{"ph":"C","name":"{metric}","pid":"GPU 0 Counters","ts":{t:.3f},"args":{{"":{v:.3f}}}}}')
            t += G["counter_interval_us"]

    envelope = {
        "schemaVersion": 1,
        "deviceProperties": [{"id": 0, "name": "NVIDIA GB300", "totalGlobalMem": 297020948480,
                              "computeMajor": 10, "computeMinor": 3, "numSms": 152}],
        "distributedInfo": {
            "backend": "nccl",
            "rank": rank,
            "world_size": mesh.world,
            "pg_count": len(rank_pg_config(mesh, rank)),
            "pg_config": rank_pg_config(mesh, rank),
            "nccl_version": "2.32.3",
        },
        "cuda_driver_version": 13040,
        "cuda_runtime_version": 13000,
        "cupti_version": 130401,
        "host_name": f"node{mesh.node(rank):05d}",
        "displayTimeUnit": "ms",
        "baseTimeNanoseconds": 1700000000000000000,
        "traceName": f"rank{rank}_trace.json",
    }
    head = json.dumps(envelope)[:-1]
    body = head + ',"traceEvents":[' + ",".join(out) + "]}"
    path = os.path.join(G["step_dir"], f"rank{rank}_trace.json.gz")
    with open(path, "wb") as f:
        f.write(gzip.compress(body.encode(), 1, mtime=0))
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--mesh", required=True, help="e.g. pp=4,dp_replicate=24,dp_shard=64,cp=2,tp=4")
    ap.add_argument("--ep", type=int, default=1, help="expert-parallel group size")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--microbatches", type=int, default=4)
    ap.add_argument("--kernels-per-compute", type=int, default=40,
                    help="compute kernels per compute op; sets events per rank")
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--straggler", action="append", metavar="RANK:FACTOR",
                    help="scale RANK's compute time by FACTOR")
    ap.add_argument("--slow-node", action="append", metavar="NODE:FACTOR",
                    help="scale transfer time of every group with a rank on NODE, "
                    "and lower its SM Active counter")
    ap.add_argument("--slow-pp-link", action="append", metavar="RANK:FACTOR",
                    help="scale p2p transfer time of the stage pairs RANK is in")
    ap.add_argument("--strip-annotations", action="append", metavar="RANK:FRACTION",
                    help="drop PG metadata from FRACTION of RANK's collective kernels")
    ap.add_argument("--custom-kernels", action="store_true",
                    help="add an unannotated HybridEP-style kernel per expert compute")
    ap.add_argument("--counters", action="store_true", help="emit GPU counter samples")
    ap.add_argument("--p2p-unique-names", action="store_true",
                    help="name each pipeline p2p pair's group uniquely, as PyTorch did before "
                    "split_group, instead of one name per round")
    ap.add_argument("--counter-interval-us", type=float, default=1000.0)
    args = ap.parse_args()

    dims = {k: int(v) for k, v in (kv.split("=") for kv in args.mesh.split(","))}
    unknown = set(dims) - set(AXES)
    if unknown:
        raise ValueError(f"unknown mesh axes {sorted(unknown)}; use {AXES}")
    mesh = Mesh(dims, args.ep, args.gpus_per_node, args.p2p_unique_names)
    faults = sanitize_faults(args, mesh)
    ops = build_schedule(mesh, args.layers, args.microbatches)
    sim = Sim(mesh, ops, faults, args.seed)
    step_dir = os.path.join(args.out_dir, f"step_{args.step:012d}")
    os.makedirs(step_dir, exist_ok=True)
    G.update(mesh=mesh, ops=ops, sim=sim, faults=faults, step_dir=step_dir,
             kernels_per_compute=args.kernels_per_compute, step=args.step,
             counter_interval_us=args.counter_interval_us)
    if args.workers > 1 and mesh.world > 1:
        with Pool(min(args.workers, mesh.world)) as pool:
            counts = pool.map(write_rank, range(mesh.world), chunksize=8)
    else:
        counts = [write_rank(r) for r in range(mesh.world)]
    truth = {
        "mesh": mesh.dims,
        "ep": args.ep,
        "world": mesh.world,
        "gpus_per_node": args.gpus_per_node,
        "faults": faults,
        "groups": {k: len(g) for k, (_, g) in mesh.groups.items()},
        "ops_per_rank": len(ops),
        "events_per_rank": [min(counts), max(counts)],
    }
    with open(os.path.join(args.out_dir, "ground_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print(f"wrote {mesh.world} ranks ({min(counts)}-{max(counts)} events each) under {step_dir}")


if __name__ == "__main__":
    main()

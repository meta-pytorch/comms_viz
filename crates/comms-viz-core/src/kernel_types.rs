// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Kernel-type classification: fixed pattern tables that bucket kernel
//! names into COMMUNICATION / MEMORY / COMPUTATION / OTHER.
//!
//! Pattern tables are compile-time constants. A kernel annotated with c10d
//! collective metadata also counts as communication; see
//! `is_annotated_collective`.

use std::sync::OnceLock;

use regex::Regex;

pub const BASE_COMM_KERNEL_PATTERNS: &[&str] = &[
    "nccl",
    "c10d::symmetric_memory::",
    // Copy-engine peer copies, e.g. async TP's fused all-gather matmul.
    "Memcpy PtoP",
];

pub const SHARED_COMM_KERNEL_PATTERNS: &[&str] = &[
    "_triton_symm_mem_barrier",
    "symmetric_memory::barrier_kernel",
];

pub const DTENSOR_FSDP_COMM_PATTERNS: &[&str] = &["_reduce_scatter", "_all_gather"];

/// Extra comm-kernel patterns per pg type — entry insertion order matters
/// (it drives the dedup order of `all_comm_kernel_patterns`, and pattern
/// order drives classification).
pub fn extra_comm_kernel_patterns() -> &'static [(&'static str, Vec<&'static str>)] {
    static TABLE: OnceLock<Vec<(&'static str, Vec<&'static str>)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let shared = SHARED_COMM_KERNEL_PATTERNS.to_vec();
        let dtensor_plus_shared: Vec<&str> = DTENSOR_FSDP_COMM_PATTERNS
            .iter()
            .chain(SHARED_COMM_KERNEL_PATTERNS.iter())
            .copied()
            .collect();
        vec![
            (
                "tp",
                [
                    "ncclKernelPutNotify",
                    "ncclKernelWaitNotify",
                    "_triton_a2a_v2_kernel",
                    "symm_memall_gather_AllGatherKernel",
                    "symm_memreduce_scatter_ReduceScatterKernel",
                ]
                .into_iter()
                .chain(shared.iter().copied())
                .collect(),
            ),
            ("cp", shared.clone()),
            (
                "ep",
                [
                    "_triton_all_to_all_",
                    "hybrid_ep::dispatch_kernel",
                    "hybrid_ep::combine_kernel",
                    "hybrid_ep::device_sync_kernel",
                    "deep_ep::elastic::dispatch_impl",
                    "deep_ep::elastic::combine_impl",
                    "deep_ep::elastic::hybrid_dispatch_impl",
                    "deep_ep::elastic::hybrid_combine_impl",
                    "deep_ep::elastic::barrier_impl",
                    "deep_ep::legacy::intranode::",
                    "deep_ep::legacy::internode",
                    "deep_ep::intranode::",
                    "deep_ep::internode",
                ]
                .into_iter()
                .chain(shared.iter().copied())
                .collect(),
            ),
            ("fsdp", dtensor_plus_shared.clone()),
            ("hsdp_shard", dtensor_plus_shared.clone()),
            ("hsdp_expert_shard", dtensor_plus_shared.clone()),
        ]
    })
}

pub fn extra_patterns_for(pg: &str) -> &'static [&'static str] {
    extra_comm_kernel_patterns()
        .iter()
        .find(|(k, _)| *k == pg)
        .map(|(_, v)| v.as_slice())
        .unwrap_or(&[])
}

/// All comm-kernel patterns = base + order-preserving dedup flatten of the
/// extra table values, in table entry order.
pub fn all_comm_kernel_patterns() -> &'static [&'static str] {
    static ALL: OnceLock<Vec<&'static str>> = OnceLock::new();
    ALL.get_or_init(|| {
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<&'static str> = BASE_COMM_KERNEL_PATTERNS.to_vec();
        for (_, pats) in extra_comm_kernel_patterns() {
            for p in pats {
                if seen.insert(*p) {
                    out.push(p);
                }
            }
        }
        // Base patterns are deliberately not inserted into `seen`: a pattern
        // appearing in both base and extras is listed twice — harmless for
        // matching, and it keeps the table order fixed.
        out
    })
}

/// Whether a kernel carries c10d-style collective metadata: both a collective
/// name and a process group name. c10d sets them on its own collectives, and
/// `torch.cuda.graph_annotations.mark_kernels` can set them on custom ones.
pub fn is_annotated_collective(collective_name: &str, pg_name: &str) -> bool {
    !collective_name.is_empty() && !pg_name.is_empty()
}

/// Whether `name` matches a built-in comm pattern.
pub fn matches_comm_pattern(name: &str) -> bool {
    static COMM_AC: OnceLock<aho_corasick::AhoCorasick> = OnceLock::new();
    let ac =
        COMM_AC.get_or_init(|| aho_corasick::AhoCorasick::new(all_comm_kernel_patterns()).unwrap());
    ac.is_match(name)
}

pub const KT_COMMUNICATION: &str = "COMMUNICATION";
pub const KT_MEMORY: &str = "MEMORY";
pub const KT_COMPUTATION: &str = "COMPUTATION";
pub const KT_OTHER: &str = "OTHER";

/// Classify a kernel name: `matches_comm_pattern` first, then the regex tiers (comm, memory,
/// not-compute). Regexes are thread_local — a process-global Regex's
/// scratch pool is a mutex bottleneck past ~8 concurrent threads.
pub fn get_kernel_type(name: &str) -> &'static str {
    thread_local! {
        static RE: (Regex, Regex, Regex) = (
            Regex::new(r"(^nccl.*)|(^hip)|(^hccl)|(.*ncclKernel.*)").unwrap(),
            Regex::new(r"(^(hip)?Memcpy)|(^(hip)?Memset)|(^dma)").unwrap(),
            Regex::new(
                r"(^nccl)|(.*Memcpy)|(.*Memset)|(.*dma)|(^.*Sync)|(^Stream Wait Event)|(cuda.*LaunchKernel)|(^runFunction)|(^hip)|(^job_exe)|(^packet)|(^pe_exe)|(^cpum_exe_subgraph)|(^hccl)|(^collective_subgraph_execution)",
            )
            .unwrap(),
        );
    }
    if matches_comm_pattern(name) {
        return KT_COMMUNICATION;
    }
    RE.with(|(comm_re, memory_re, not_compute_re)| {
        if comm_re.is_match(name) {
            return KT_COMMUNICATION;
        }
        if memory_re.is_match(name) {
            return KT_MEMORY;
        }
        if !not_compute_re.is_match(name) {
            return KT_COMPUTATION;
        }
        KT_OTHER
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_basics() {
        assert_eq!(
            get_kernel_type("ncclDevKernel_AllGather_..."),
            KT_COMMUNICATION
        );
        assert_eq!(get_kernel_type("Memcpy DtoH"), KT_MEMORY);
        assert_eq!(get_kernel_type("triton_fused_gemm"), KT_COMPUTATION);
        assert_eq!(get_kernel_type("cudaStreamSynchronize"), KT_OTHER);
        assert_eq!(
            get_kernel_type("_triton_symm_mem_barrier"),
            KT_COMMUNICATION
        );
        assert_eq!(get_kernel_type("_triton_all_to_all_fwd"), KT_COMMUNICATION);
        assert_eq!(
            get_kernel_type(
                "kernel_cutlass_kernel_opskernelssymm_memall_gather_AllGatherKernel_object"
            ),
            KT_COMMUNICATION
        );
        assert_eq!(
            get_kernel_type(
                "kernel_cutlass_kernel_opskernelssymm_memreduce_scatter_ReduceScatterKernel_object"
            ),
            KT_COMMUNICATION
        );
        for name in [
            "c10d::symmetric_memory::multimem_barrier_kernel(unsigned int*, unsigned int*, int, int, int, unsigned long)",
            "Memcpy PtoP (Device -> Device)",
            "void hybrid_ep::dispatch_kernel<unsigned short, hybrid_ep::warp_group<0, 0>>",
            "void hybrid_ep::combine_kernel<hybrid_ep::warp_group<0, 0>>",
            "void hybrid_ep::device_sync_kernel<4>(unsigned int*, unsigned int*, unsigned int*)",
            "void deep_ep::elastic::hybrid_dispatch_impl<1, 2>",
            "void deep_ep::intranode::dispatch<8, 768>(int4*, float*)",
        ] {
            assert_eq!(get_kernel_type(name), KT_COMMUNICATION, "{name}");
        }
        for name in [
            "Memcpy DtoD (Device -> Device)",
            "void hybrid_ep::scan<256, 108>(void const*)",
            "void permute_kernel<unsigned short>(unsigned short const*)",
            "void deep_ep::elastic::dispatch_copy_epilogue_impl<1>",
        ] {
            assert_ne!(get_kernel_type(name), KT_COMMUNICATION, "{name}");
        }
    }

    #[test]
    fn annotated_collective_needs_both_fields() {
        assert!(is_annotated_collective("hybridep_dispatch", "hybridep"));
        assert!(!is_annotated_collective("hybridep_dispatch", ""));
        assert!(!is_annotated_collective("", "hybridep"));
    }

    #[test]
    fn all_patterns_order() {
        let all = all_comm_kernel_patterns();
        assert_eq!(all[0], "nccl");
        assert_eq!(all[1], "c10d::symmetric_memory::");
        assert_eq!(all[2], "Memcpy PtoP");
        assert_eq!(all[3], "ncclKernelPutNotify");
        assert_eq!(all[4], "ncclKernelWaitNotify");
        assert_eq!(all[5], "_triton_a2a_v2_kernel");
        assert_eq!(all[6], "symm_memall_gather_AllGatherKernel");
        assert_eq!(all[7], "symm_memreduce_scatter_ReduceScatterKernel");
        assert_eq!(all[8], "_triton_symm_mem_barrier");
        assert_eq!(all[9], "symmetric_memory::barrier_kernel");
        assert_eq!(all[10], "_triton_all_to_all_");
        assert_eq!(all[11], "hybrid_ep::dispatch_kernel");
        assert_eq!(all[22], "deep_ep::internode");
        assert_eq!(all[23], "_reduce_scatter");
        assert_eq!(all[24], "_all_gather");
        assert_eq!(all.len(), 25);
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Per-stage thread-time accounting for the rank pipeline.
//!
//! Breaks the rank loop's work time down by stage. Each stage of
//! `process_rank` accumulates into a process-global counter, and the run
//! prints the breakdown after the rank loop.
//!
//! Cost: two `Instant::now()` calls and one relaxed atomic add per stage per
//! rank — ~0.5 µs, measured as a no-op in a paired A/B.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

/// Stages of one rank's journey, in pipeline order.
#[derive(Clone, Copy)]
pub enum Stage {
    /// Read the staged trace off tmpfs/disk.
    Read = 0,
    /// gunzip into the worker's reusable scratch (libdeflate).
    Gunzip = 1,
    /// Inline-annotation probe over the decompressed prefix.
    Detect = 2,
    /// simd-json tape parse + row/raw-view build.
    Parse = 3,
    /// Distinct live graph ids off the parsed rows.
    Gids = 4,
    /// Annotation-pkl resolution: handshake, local reads (the blocking part
    /// is reported separately as `pkl_wait`).
    PklFetch = 5,
    /// Pickle decode into the annotation digests.
    PklDecode = 6,
    /// CUDA-graph walk passes + compaction.
    Walk = 7,
    /// Global sort permutation over the post-walk rows.
    Sort = 8,
    /// Breakdown + filtered-kernel accumulation + per-PG split.
    Analyze = 9,
    /// Worker partial encode/write (runs on the thread that finishes a
    /// worker's rank range).
    Partial = 10,
    /// Handing the merged-trace artifact to the writer pool: normally a
    /// queue push; equals backpressure time when the writers can't keep
    /// up with the disk (the time that used to hide in `residual` as
    /// inline fs::write under dirty throttling).
    ArtifactQueue = 11,
    /// Tree path: parse the document into the `serde_json` event tree.
    TreeParse = 12,
    /// Tree path: the merge/annotate/move/fix passes over the tree.
    TreeWalk = 13,
    /// Tree path: serialize + gzip the merged-trace artifact bytes.
    TreeSer = 14,
    /// Tree path: event tree → analysis columns.
    Convert = 15,
}

pub const NAMES: [&str; 16] = [
    "read",
    "gunzip",
    "detect",
    "parse",
    "gids",
    "pkl_fetch",
    "pkl_decode",
    "walk",
    "sort",
    "analyze",
    "partial",
    "artifact_q",
    "tree_parse",
    "tree_walk",
    "tree_ser",
    "convert",
];

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static NS: [AtomicU64; NAMES.len()] = [ZERO; NAMES.len()];

pub fn add(stage: Stage, d: Duration) {
    NS[stage as usize].fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
}

/// Scoped timer: records the elapsed time to `stage` when dropped.
pub struct Timer(Stage, Instant);

impl Timer {
    pub fn new(stage: Stage) -> Self {
        Timer(stage, Instant::now())
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        add(self.0, self.1.elapsed());
    }
}

/// `(name, thread-seconds)` per stage, pipeline order.
pub fn snapshot() -> Vec<(&'static str, f64)> {
    NAMES
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, NS[i].load(Ordering::Relaxed) as f64 / 1e9))
        .collect()
}

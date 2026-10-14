// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Recording-step resolution — the *other* CUDA-graph repair mechanism.
//!
//! Two independent things restore metadata a replay trace lost:
//!   1. annotation pickles keyed by `graph node id` ([`super::pkl_fetch`]);
//!   2. a **recording-step trace** — a second full trace from the step where
//!      the graph was captured, matched kernel-for-kernel against the replay so
//!      its `args` can be copied across (`recording_merge.rs`,
//!      `recording_read.rs`).
//!
//! Both are CUDA-graph-specific and disappear with the pickle side channel —
//! an inline-annotated trace needs neither — so the annotation seam owns them
//! rather than the generic trace discovery.
//!
//! Note the recording merge only exists on the **tree** path — it forces full
//! [`crate::annotations::Annotations`] rather than digests, and the row-walk
//! fast path is disabled for a rank that has a recording.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use regex::Regex;

use crate::trace::Format;
use crate::trace::discover::TraceDiscovery;

/// Derive the sibling recording-step dir: `<parent>/step_{n:012}` from a
/// `.../step_NNN` root.
pub fn derive_recording_dir(trace_root: &Path, recording_step: i64) -> Result<PathBuf> {
    let leaf = trace_root
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let re = Regex::new(r"^step_(\d+)$").unwrap();
    if !re.is_match(&leaf) {
        anyhow::bail!("Cannot derive recording URI: '{leaf}' does not match step_NNNN pattern.");
    }
    let parent = trace_root.parent().unwrap_or(Path::new(""));
    Ok(parent.join(format!("step_{recording_step:012}")))
}

/// An explicit step wins; else probe steps (2, 1) for the first rank's
/// trace.
pub fn resolve_recording_step(
    ingress: &TraceDiscovery,
    trace_root: &Path,
    rank_lo: i64,
    rank_hi: i64,
    explicit_step: Option<i64>,
    format: Format,
) -> Option<i64> {
    if let Some(s) = explicit_step {
        return Some(s);
    }
    let probe_rank = if rank_hi > rank_lo { rank_lo } else { 0 };
    for candidate in [2i64, 1] {
        let rec_dir = match derive_recording_dir(trace_root, candidate) {
            Ok(d) => d,
            Err(_) => return None,
        };
        if ingress.find_trace(probe_rank, &rec_dir, format).is_some() {
            return Some(candidate);
        }
    }
    None
}

/// The recording-trace directory for a run, when one applies.
///
/// `None` covers every "no recording to merge" case:
/// annotations are not pickle-sourced, the trace root is not a `step_NNN` dir,
/// or no earlier step carries the probe rank's trace.
pub fn resolve_dir(
    ingress: &TraceDiscovery,
    trace_root: &Path,
    world_size: i64,
    explicit_step: Option<i64>,
    format: Format,
) -> Option<PathBuf> {
    resolve_recording_step(ingress, trace_root, 0, world_size, explicit_step, format)
        .and_then(|s| derive_recording_dir(trace_root, s).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_sibling_step_dir() {
        let d = derive_recording_dir(Path::new("/j/traces/step_000000328328"), 2).unwrap();
        assert_eq!(d, PathBuf::from("/j/traces/step_000000000002"));
    }

    #[test]
    fn rejects_non_step_leaf() {
        assert!(derive_recording_dir(Path::new("/j/traces"), 2).is_err());
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Annotation-pickle acquisition: turning a rank's live graph ids into pickle
//! bytes.
//!
//! This is the whole side-channel — exactly what self-annotated pftrace
//! output makes unnecessary; it goes away with the rest of the legacy/
//! directory. Pickles are indexed once at startup from the
//! `kernel_annotations/` dir next to the step dir, then filtered per rank to
//! its live graph ids.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use crate::trace::discover::TraceDiscovery;

/// The pickle side-channel, built once per run.
pub struct PklSource {
    /// rank → sorted pkl paths, from the startup listing.
    index: HashMap<i64, Vec<String>>,
}

impl PklSource {
    pub fn new(ingress: &TraceDiscovery, trace_root: &Path, world: std::ops::Range<i64>) -> Self {
        PklSource {
            index: index_annotation_pkls(ingress, trace_root, world),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        PklSource {
            index: HashMap::new(),
        }
    }

    /// Read this rank's annotation pickles, filtered to its live graph ids.
    /// Unreadable files are skipped — a rank with no annotations is a
    /// legitimate steady state.
    pub fn fetch(&self, rank: i64, live: &HashSet<i64>) -> Vec<Vec<u8>> {
        let _t = crate::timing::Timer::new(crate::timing::Stage::PklFetch);
        let Some(uris) = self.index.get(&rank) else {
            return Vec::new();
        };
        let filtered = if live.is_empty() {
            uris.clone()
        } else {
            super::pkl::filter_pkl_uris_by_graph_ids(uris.clone(), live)
        };
        filtered
            .iter()
            .filter_map(|uri| std::fs::read(uri).ok())
            .collect()
    }
}

/// Hash-bucketed rank dir: rank 42 → `0/0/0/0/4/2`.
fn shard_dir_path(rank: i64) -> std::path::PathBuf {
    format!("{rank:06}").chars().map(String::from).collect()
}

/// Build the startup listing index: rank → sorted pkl paths.
///
/// Precedence:
///   1. `<traces>/kernel_annotations/` flat
///   2. `<traces>/kernel_annotations/<hash-bucketed rank dirs>/`
///   3. `<traces>/` legacy flat
///
/// Filename `kernel_annotations_rank{R}_*.pkl`; per-rank list sorted.
pub fn index_annotation_pkls(
    ingress: &TraceDiscovery,
    trace_root: &Path,
    ranks: std::ops::Range<i64>,
) -> HashMap<i64, Vec<String>> {
    let root_str = trace_root.to_string_lossy();
    if !root_str.contains("/step_") && !root_str.starts_with("step_") {
        // A root without a step_ segment has no annotation layout to index:
        // pkls live next to a step_NNN dir.
        // (Local paths: also accept a bare `step_*` root.)
        let leaf_is_step = trace_root
            .file_name()
            .map(|s| s.to_string_lossy().starts_with("step_"))
            .unwrap_or(false);
        if !leaf_is_step {
            return HashMap::new();
        }
    }
    let traces_dir = trace_root.parent().unwrap_or(Path::new("")).to_path_buf();
    let candidates = [traces_dir.join("kernel_annotations"), traces_dir.clone()];

    let pkl_re = regex::Regex::new(r"^kernel_annotations_rank(\d+)_.*\.pkl$").unwrap();
    let index_entries = |names: &[String], dir: &Path| -> HashMap<i64, Vec<String>> {
        let mut index: HashMap<i64, Vec<String>> = HashMap::new();
        for n in names {
            if let Some(c) = pkl_re.captures(n) {
                if let Ok(rank) = c[1].parse::<i64>() {
                    index
                        .entry(rank)
                        .or_default()
                        .push(dir.join(n).to_string_lossy().into_owned());
                }
            }
        }
        index
    };

    for (i, dir) in candidates.iter().enumerate() {
        let Some(names) = ingress.cached_list_pub(dir) else {
            continue;
        };
        let mut index = index_entries(&names, dir);
        if !index.is_empty() {
            for v in index.values_mut() {
                v.sort();
            }
            return index;
        }
        // Hash-bucketed fallback only for the kernel_annotations candidate.
        if i == 0 {
            let mut hash_index: HashMap<i64, Vec<String>> = HashMap::new();
            for rank in ranks.clone() {
                let sub = dir.join(shard_dir_path(rank));
                if let Some(names) = ingress.cached_list_pub(&sub) {
                    for (r, mut paths) in index_entries(&names, &sub) {
                        hash_index.entry(r).or_default().append(&mut paths);
                    }
                }
            }
            if !hash_index.is_empty() {
                for v in hash_index.values_mut() {
                    v.sort();
                }
                return hash_index;
            }
        }
    }
    HashMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unindexed_rank_yields_nothing() {
        let src = PklSource::for_test();
        let live = HashSet::from([5i64]);
        assert!(src.fetch(0, &live).is_empty());
    }
}

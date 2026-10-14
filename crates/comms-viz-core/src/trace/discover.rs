// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Local trace-file discovery: map each rank to its trace file in one flat
//! trace directory.
//!
//! Every file of the run's format directly under the directory is a rank
//! trace. JSON traces name their rank and world size in `distributedInfo`.
//! pftrace carries neither yet, so its rank comes from the file name
//! (`file_name_rank`) and the world size is the highest rank + 1. A file without a rank, or two files
//! with the same rank, is an error.
//!
//! Scope: **generic trace discovery only**. Annotation-pkl indexing and
//! recording-step resolution are CUDA-graph-specific and live in
//! `legacy::pkl_fetch` / `legacy::recording`. The trace-name suffix table is
//! `trace::Format`'s.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Context;
use anyhow::Result;

use crate::trace::Format;

/// Rank from a pftrace file name: the number after the first `rank`
/// (`rank3_trace.pftrace.gz`), else a leading number (`3.pftrace`,
/// `000003.pt.trace.pftrace.gz`).
fn file_name_rank(name: &str) -> Option<i64> {
    let leading = |s: &str| {
        let end = s.bytes().take_while(u8::is_ascii_digit).count();
        s[..end].parse().ok()
    };
    match name.find("rank") {
        Some(i) => leading(&name[i + "rank".len()..]),
        None => leading(name),
    }
}

/// The rank -> trace file map of one directory.
pub(crate) struct RankIndex {
    paths: BTreeMap<i64, PathBuf>,
    /// From the traces; `None` for formats that do not record it.
    world_size: Option<i64>,
}

impl RankIndex {
    fn build(dir: &Path, names: &[String], format: Format) -> Result<Self> {
        let files: Vec<&String> = names.iter().filter(|n| format.matches_name(n)).collect();
        anyhow::ensure!(
            !files.is_empty(),
            "no {} trace files found under {dir:?}",
            format.as_str()
        );
        let ranks: Vec<Result<(i64, Option<i64>)>> = match format {
            Format::Pftrace => files
                .iter()
                .map(|name| {
                    file_name_rank(name).map(|r| (r, None)).with_context(|| {
                        format!(
                            "{name}: pftrace traces carry no rank; put it in the file \
                             name, as rank<N> or a leading number"
                        )
                    })
                })
                .collect(),
            Format::Json => probe_json_ranks(dir, &files),
        };
        let mut paths = BTreeMap::new();
        let mut world_sizes = std::collections::BTreeSet::new();
        for (name, rank) in files.iter().zip(ranks) {
            let (rank, world_size) = rank?;
            world_sizes.extend(world_size);
            if let Some(prev) = paths.insert(rank, dir.join(name)) {
                anyhow::bail!(
                    "rank {rank} has two traces: {} and {}",
                    prev.display(),
                    dir.join(name).display()
                );
            }
        }
        anyhow::ensure!(
            world_sizes.len() <= 1,
            "traces under {dir:?} disagree on world_size: {world_sizes:?}"
        );
        let world_size = world_sizes.into_iter().next();
        if let (Some(ws), Some((&max, _))) = (world_size, paths.last_key_value()) {
            anyhow::ensure!(max < ws, "rank {max} is outside world_size {ws}");
        }
        Ok(RankIndex { paths, world_size })
    }

    /// The traces' world size, else the highest rank + 1.
    pub(crate) fn world_size(&self) -> i64 {
        self.world_size
            .unwrap_or_else(|| self.paths.keys().next_back().map_or(0, |r| r + 1))
    }

    /// Ranks below `world_size` without a trace.
    pub(crate) fn missing(&self, world_size: i64) -> Vec<i64> {
        (0..world_size)
            .filter(|r| !self.paths.contains_key(r))
            .collect()
    }
}

/// `distributedInfo` rank per JSON file, read in parallel: each probe
/// inflates only the file's head, but a large run has thousands of files.
fn probe_json_ranks(dir: &Path, files: &[&String]) -> Vec<Result<(i64, Option<i64>)>> {
    let probe = |name: &String| -> Result<(i64, Option<i64>)> {
        let path = dir.join(name);
        crate::legacy::json::trace_rank(&path)
            .with_context(|| format!("read {path:?}"))?
            .with_context(|| {
                format!(
                    "{name}: no distributedInfo.rank; start the profiler after \
                     torch.distributed is initialized"
                )
            })
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(files.len())
        .max(1);
    let chunk = files.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = files
            .chunks(chunk)
            .map(|part| s.spawn(move || part.iter().map(|n| probe(n)).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| {
                h.join()
                    .unwrap_or_else(|_| vec![Err(anyhow::anyhow!("rank probe panicked"))])
            })
            .collect()
    })
}

/// A built index, or the error building it (kept so callers don't retry).
type CachedIndex = Result<Arc<RankIndex>, Arc<anyhow::Error>>;

// Arc'd listings: find_trace runs once per rank from every worker thread —
// cloning a 14K-entry Vec<String> under the global mutex serialized the
// whole rank loop (165+/224 threads parked in futex_wait).
pub struct TraceDiscovery {
    dir_cache: Mutex<HashMap<PathBuf, Option<Arc<Vec<String>>>>>,
    index_cache: Mutex<HashMap<PathBuf, CachedIndex>>,
}

impl Default for TraceDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceDiscovery {
    pub fn new() -> Self {
        TraceDiscovery {
            dir_cache: Mutex::new(HashMap::new()),
            index_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Sorted file names of `dir` (files only), cached. None if unlistable.
    /// `pub(crate)`: the annotation-pkl indexer lives in `legacy::pkl_fetch`
    /// but must share this run's listing cache.
    pub(crate) fn cached_list_pub(&self, dir: &Path) -> Option<Arc<Vec<String>>> {
        self.cached_list(dir)
    }

    fn cached_list(&self, dir: &Path) -> Option<Arc<Vec<String>>> {
        if let Some(v) = self.dir_cache.lock().unwrap().get(dir) {
            return v.clone();
        }
        let listed: Option<Arc<Vec<String>>> = std::fs::read_dir(dir).ok().map(|rd| {
            let mut names: Vec<String> = rd
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| !t.is_dir()).unwrap_or(false))
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            names.sort_unstable();
            Arc::new(names)
        });
        self.dir_cache
            .lock()
            .unwrap()
            .insert(dir.to_path_buf(), listed.clone());
        listed
    }

    /// The rank index of `dir`, built once per directory. The lock is held
    /// while building so concurrent callers wait instead of re-probing.
    pub(crate) fn index(&self, dir: &Path, format: Format) -> Result<Arc<RankIndex>> {
        let mut cache = self
            .index_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = cache.entry(dir.to_path_buf()).or_insert_with(|| {
            let names = self.cached_list(dir).unwrap_or_default();
            RankIndex::build(dir, &names, format)
                .map(Arc::new)
                .map_err(Arc::new)
        });
        entry.clone().map_err(|e| anyhow::anyhow!("{e:#}"))
    }

    /// Path of `rank`'s trace in `dir`; `None` when the rank has no trace or
    /// the directory has no usable index.
    pub fn find_trace(&self, rank: i64, dir: &Path, format: Format) -> Option<PathBuf> {
        self.index(dir, format).ok()?.paths.get(&rank).cloned()
    }

    pub fn read(&self, path: &Path) -> Result<Vec<u8>> {
        std::fs::read(path).with_context(|| format!("read {path:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_trace(dir: &Path, name: &str, rank: Option<i64>, world_size: i64) {
        let info = match rank {
            Some(rank) => {
                format!(r#""distributedInfo":{{"rank":{rank},"world_size":{world_size}}},"#)
            }
            None => String::new(),
        };
        let body = format!(r#"{{"schemaVersion":1,{info}"traceEvents":[]}}"#);
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut enc, body.as_bytes()).unwrap();
        std::fs::write(dir.join(name), enc.finish().unwrap()).unwrap();
    }

    #[test]
    fn json_rank_comes_from_the_trace() {
        let tmp = tempfile::tempdir().unwrap();
        // File names say nothing about the rank.
        json_trace(tmp.path(), "b.json.gz", Some(0), 4);
        json_trace(tmp.path(), "a.json.gz", Some(2), 4);
        std::fs::write(tmp.path().join("notes.txt"), []).unwrap();
        let disc = TraceDiscovery::new();
        let index = disc.index(tmp.path(), Format::Json).unwrap();
        assert_eq!(index.world_size(), 4);
        assert_eq!(index.missing(4), vec![1, 3]);
        assert_eq!(
            disc.find_trace(2, tmp.path(), Format::Json),
            Some(tmp.path().join("a.json.gz"))
        );
        assert_eq!(disc.find_trace(1, tmp.path(), Format::Json), None);
    }

    #[test]
    fn json_trace_without_rank_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        json_trace(tmp.path(), "0.json.gz", Some(0), 2);
        json_trace(tmp.path(), "1.json.gz", None, 2);
        let err = TraceDiscovery::new()
            .index(tmp.path(), Format::Json)
            .err()
            .expect("no rank");
        assert!(
            err.to_string()
                .contains("1.json.gz: no distributedInfo.rank"),
            "{err}"
        );
    }

    #[test]
    fn duplicate_rank_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        json_trace(tmp.path(), "000000.1.pt.trace.json.gz", Some(0), 1);
        json_trace(tmp.path(), "000000.2.pt.trace.json.gz", Some(0), 1);
        let err = TraceDiscovery::new()
            .index(tmp.path(), Format::Json)
            .err()
            .expect("duplicate");
        assert!(err.to_string().contains("rank 0 has two traces"), "{err}");
    }

    #[test]
    fn pftrace_rank_comes_from_the_file_name() {
        assert_eq!(file_name_rank("000042.123.pt.trace.pftrace.gz"), Some(42));
        assert_eq!(file_name_rank("rank42_trace.pftrace.gz"), Some(42));
        assert_eq!(file_name_rank("rank42_trace.pftrace"), Some(42));
        assert_eq!(file_name_rank("trace_rank7.pftrace"), Some(7));
        assert_eq!(file_name_rank("3.pftrace"), Some(3));
        assert_eq!(file_name_rank("rankx_trace.pftrace"), None);
        assert_eq!(file_name_rank("trace.pftrace"), None);

        let tmp = tempfile::tempdir().unwrap();
        for rank in [0, 7] {
            std::fs::write(tmp.path().join(format!("rank{rank}_trace.pftrace.gz")), []).unwrap();
        }
        let disc = TraceDiscovery::new();
        assert_eq!(
            disc.index(tmp.path(), Format::Pftrace)
                .unwrap()
                .world_size(),
            8
        );
        assert_eq!(
            disc.find_trace(7, tmp.path(), Format::Pftrace),
            Some(tmp.path().join("rank7_trace.pftrace.gz"))
        );

        std::fs::write(tmp.path().join("trace.pftrace"), []).unwrap();
        assert!(
            TraceDiscovery::new()
                .index(tmp.path(), Format::Pftrace)
                .is_err()
        );
    }

    #[test]
    fn empty_dir_names_the_format() {
        let tmp = tempfile::tempdir().unwrap();
        let err = TraceDiscovery::new()
            .index(tmp.path(), Format::Pftrace)
            .err()
            .expect("empty");
        assert!(
            err.to_string()
                .contains("no pftrace trace files found under"),
            "{err}"
        );
    }
}

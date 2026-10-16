// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Per-rank trace analysis: the prepare transforms (cat filter, sentinel
//! fills, ncclDevKernel rename, para_type extraction), the breakdown
//! row, the filtered comm kernels, and the per-PG split.
//!
//! Everything is computed in one pass over the converted columns:
//!   - breakdown sums accumulate in frame row order and reduce with the
//!     deterministic pairwise kernel — exact for integral durations, and
//!     run-to-run bit-stable for fractional ones;
//!   - filtered-kernel rows, their order, and the per-PG frame contents
//!     feed the polars parquet writers unchanged.

use anyhow::Result;
use polars::prelude::*;
use regex::Regex;

use crate::kernel_types::KT_COMMUNICATION;
use crate::kernel_types::KT_COMPUTATION;
use crate::kernel_types::{self};
use crate::rows::ConvertedColumns;
use crate::trace::ProcessGroupConfig;

/// A rank's PG layout: axis -> the para_types (PG descriptions) on it.
type RankInfo = Vec<(String, Vec<String>)>;

/// Per-PG member holding rank activity and PM samples rather than a process group.
pub const DIAGNOSTICS_PG: &str = "diagnostics_0";

/// Prefix of the members holding GPU counter timelines, one per
/// `COUNTER_BIN_RANKS` consecutive ranks so the page fetches only the ranks
/// it shows.
pub const COUNTER_BINS_PREFIX: &str = "diagnostics_pm_";
pub const COUNTER_BIN_RANKS: i64 = 32;
/// Time bins per rank and counter over the rank's counter samples.
pub const COUNTER_BINS: usize = 64;

/// Whether a per-PG member stem holds diagnostics rather than a process group.
pub fn is_diagnostics_member(stem: &str) -> bool {
    stem == DIAGNOSTICS_PG || stem.starts_with(COUNTER_BINS_PREFIX)
}

/// The breakdown row's column set — used for the -1 failure row.
pub const BREAKDOWN_COLUMNS: &[&str] = &[
    "rank",
    "comm_total",
    "comp_total",
    "all_gather_total",
    "reduce_scatter_total",
    "all_reduce_total",
    "all_to_all_total",
    "total_duration",
    "cudaLaunchKernel_total",
    "cudaEventSynchronize_total",
    "cudaStreamSynchronize_total",
    "cudaHostAlloc_total",
];

/// A breakdown-row value: `rank` (and the all -1 failure row) is an int,
/// the totals are floats; the two land as different dtypes in the
/// breakdown parquet (emit::rows_to_df).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BreakdownValue {
    Int(i64),
    Float(f64),
}

/// Extract the para_type from a PG desc/name value. Input is the
/// (never-null, ""-filled) PG column.
///
/// The regexes are thread_local, NOT process-global: this runs per prepared
/// row from every worker thread, and a shared Regex's internal scratch pool
/// degrades to one mutex once >8 threads use it (measured: 165+/224 threads
/// parked in futex_wait, rank loop capped near 35 ranks/s).
fn extract_comm_type(s: &str) -> String {
    thread_local! {
        static RES: (Regex, Regex, Regex, Regex, Regex, Regex) = (
            Regex::new(r"::split::(.+)$").unwrap(),
            Regex::new(r"^\d+_").unwrap(),
            Regex::new(r"_\d+$").unwrap(),
            Regex::new(r"^(default_torchcomm_ncclx|default_pg)(:\d+)?$").unwrap(),
            Regex::new(r"^(default_torchcomm_ncclx|default_pg)").unwrap(),
            Regex::new(r":\d+$").unwrap(),
        );
    }
    RES.with(|res| extract_comm_type_impl(s, res))
}

fn extract_comm_type_impl(
    s: &str,
    (
        re_split_extract,
        re_lead_digits,
        re_trail_digits,
        re_default_full,
        re_default_prefix,
        re_tail_digits,
    ): &(Regex, Regex, Regex, Regex, Regex, Regex),
) -> String {
    if s.is_empty() || s == ":" {
        return String::new();
    }
    if s.contains("::split::") {
        // extract group 1, then strip ^\d+_ and _\d+$ (replace-first each).
        let extracted = match re_split_extract.captures(s) {
            Some(c) => c.get(1).map(|m| m.as_str()).unwrap_or(""),
            None => return String::new(), // extract miss → null → "" (filled)
        };
        let step1 = re_lead_digits.replace(extracted, "");
        let step2 = re_trail_digits.replace(&step1, "");
        return step2.into_owned();
    }
    if re_default_full.is_match(s) {
        return re_default_prefix
            .captures(s)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
    }
    if !s.contains("::") {
        if re_tail_digits.is_match(s) {
            return re_tail_digits.replace(s, "").into_owned();
        }
        if let Some(stripped) = s.strip_suffix(':') {
            return stripped.to_string();
        }
        return s.to_string();
    }
    s.to_string()
}

#[derive(Clone, Copy)]
struct NameInfo {
    kernel_type: &'static str,
    comm_pattern: bool,
    all_gather: bool,
    reduce_scatter: bool,
    all_reduce: bool,
}

/// First occurrence of the `ncclDevKernel` needle. `str::find(&str)` builds a
/// fresh two-way searcher per call (`StrSearcher::new`), which this runs once
/// per kept row; a prebuilt finder reuses the critical factorization.
fn nccl_dev_kernel_pos(hay: &str) -> Option<usize> {
    static F: std::sync::OnceLock<memchr::memmem::Finder<'static>> = std::sync::OnceLock::new();
    F.get_or_init(|| memchr::memmem::Finder::new("ncclDevKernel"))
        .find(hay.as_bytes())
}

/// `NameInfo` is a pure function of the kernel name, so the cache lives for
/// the worker thread instead of being rebuilt per rank — a rebuild cost six
/// aho-corasick automata plus a full reclassification (aho-corasick + three
/// regexes per distinct name) of names already seen on earlier ranks.
fn with_name_cache<R>(f: impl FnOnce(&mut NameCache) -> R) -> R {
    thread_local! {
        static CACHE: std::cell::RefCell<NameCache> =
            std::cell::RefCell::new(NameCache::new());
    }
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        // Bound the footprint on jobs with pathological name cardinality;
        // every entry is recomputable on demand.
        if cache.values.len() > (1 << 18) {
            cache.values.clear();
        }
        f(&mut cache)
    })
}

struct NameCache {
    values: rustc_hash::FxHashMap<String, NameInfo>,
}

impl NameCache {
    fn new() -> Self {
        Self {
            values: rustc_hash::FxHashMap::default(),
        }
    }

    fn get(&mut self, name: &str) -> NameInfo {
        if let Some(info) = self.values.get(name) {
            return *info;
        }
        let info = NameInfo {
            kernel_type: kernel_types::get_kernel_type(name),
            comm_pattern: kernel_types::matches_comm_pattern(name),
            all_gather: name.contains("AllGather"),
            reduce_scatter: name.contains("ReduceScatter"),
            all_reduce: name.contains("AllReduce"),
        };
        self.values.insert(name.to_string(), info);
        info
    }
}

/// Prepared per-rank row view (after the prepare transforms).
struct Prepared {
    /// indices into the converted columns of kept rows (cat non-null, != "").
    keep: Vec<u32>,
    /// transformed kernel names (ncclDevKernel→ncclKernel, first occurrence):
    /// Some only for the rows actually containing the needle; every other
    /// row reads the converted column directly (null names propagate — polars
    /// string ops keep them out of classification and contains-filters).
    renamed: Vec<Option<String>>,
    para_type: Vec<String>,
    pg_name: Vec<String>,
}

/// Group name -> what its axis is derived from (`axis_source`).
fn process_group_descriptions(
    process_groups: &[ProcessGroupConfig],
) -> rustc_hash::FxHashMap<&str, String> {
    process_groups
        .iter()
        .map(|group| (group.pg_name.as_str(), axis_source(group).into_owned()))
        .collect()
}

fn sanitize_axis(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '+') {
                character
            } else {
                '-'
            }
        })
        .collect()
}

/// Group id of a physical process group in `{axis}_{id}` member stems and
/// `pg_map.json`'s `pg_ids`. Stems split at the last `_`, so ids never
/// contain one; unannotated rows get a non-numeric id that cannot collide
/// with a real (numeric) `pg_name`.
pub(crate) fn pg_group_id(pg_name: &str) -> String {
    if pg_name.is_empty() {
        return "unnamed".to_string();
    }
    pg_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' {
                character
            } else {
                '-'
            }
        })
        .collect()
}

/// Group physical process groups by their trace-provided semantic axis.
/// What a group's axis is derived from: its description, else its name.
/// Groups created without a description (plain `new_group` leaves PyTorch's
/// default "undefined") are split by size and rank stride, so e.g. TP pairs
/// and DP pairs don't land in one axis and get compared with each other.
fn axis_source(config: &ProcessGroupConfig) -> std::borrow::Cow<'_, str> {
    if config.pg_desc.is_empty() {
        return config.pg_name.as_str().into();
    }
    if config.pg_desc != "undefined" {
        return config.pg_desc.as_str().into();
    }
    let mut ranks = config.ranks.clone();
    ranks.sort_unstable();
    let stride = match ranks.as_slice() {
        [a, b, ..] if ranks.windows(2).all(|w| w[1] - w[0] == b - a) => format!("_stride{}", b - a),
        [_, _, ..] => "_mixed".to_string(),
        _ => String::new(),
    };
    format!("undefined_size{}{stride}", config.pg_size).into()
}

pub(crate) fn process_group_axes(
    configs: &[ProcessGroupConfig],
) -> std::collections::BTreeMap<String, Vec<&ProcessGroupConfig>> {
    let mut axes = std::collections::BTreeMap::new();
    for config in configs {
        let para_type = extract_comm_type(&axis_source(config));
        let axis = sanitize_axis(para_type.strip_prefix("mesh_").unwrap_or(&para_type));
        if !axis.is_empty() {
            axes.entry(axis).or_insert_with(Vec::new).push(config);
        }
    }
    for configs in axes.values_mut() {
        configs.sort_by(|left, right| {
            match (left.pg_name.parse::<u64>(), right.pg_name.parse::<u64>()) {
                (Ok(left), Ok(right)) => left.cmp(&right),
                _ => left.pg_name.cmp(&right.pg_name),
            }
        });
    }
    axes
}

/// The rank's PG layout from its trace's PG inventory.
fn trace_rank_info(process_groups: &[ProcessGroupConfig]) -> RankInfo {
    process_group_axes(process_groups)
        .into_iter()
        .map(|(axis, configs)| {
            let mut para_types = configs
                .into_iter()
                .map(|config| extract_comm_type(&axis_source(config)))
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>();
            para_types.sort();
            para_types.dedup();
            (axis, para_types)
        })
        .collect()
}

fn event_comm_type(
    pg_desc: &str,
    pg_name: &str,
    process_group_descriptions: &rustc_hash::FxHashMap<&str, String>,
) -> String {
    // An "undefined" description is split by the group's layout, which only
    // the PG inventory knows.
    if pg_desc != "undefined" {
        let desc = extract_comm_type(pg_desc);
        if !desc.is_empty() {
            return desc;
        }
    }
    extract_comm_type(
        process_group_descriptions
            .get(pg_name)
            .map_or(pg_name, String::as_str),
    )
}

/// Per row: a GPU event (kernel, copy, set) whose launching call isn't in the
/// trace. Every GPU activity carries its launch's correlation id; one with no
/// matching CUDA runtime/driver call was launched before the profiling window
/// opened -- typically the previous step still running on the GPU -- so it
/// belongs to a step the trace doesn't cover. A trace with no launch records
/// can't tell and keeps everything.
fn launched_before_window<'a>(rows: impl Iterator<Item = (Option<&'a str>, i64)>) -> Vec<bool> {
    // 1: GPU event, 2: launch call.
    let rows: Vec<(u8, i64)> = rows
        .map(|(cat, corr)| match cat {
            Some("kernel" | "gpu_memcpy" | "gpu_memset") => (1, corr),
            Some("cuda_runtime" | "cuda_driver") => (2, corr),
            _ => (0, corr),
        })
        .collect();
    let launches: rustc_hash::FxHashSet<i64> = rows
        .iter()
        .filter(|&&(kind, corr)| kind == 2 && corr > 0)
        .map(|&(_, corr)| corr)
        .collect();
    if launches.is_empty() {
        return vec![false; rows.len()];
    }
    rows.iter()
        .map(|&(kind, corr)| kind == 1 && corr > 0 && !launches.contains(&corr))
        .collect()
}

fn prepare(
    cols: &ConvertedColumns,
    process_groups: &[ProcessGroupConfig],
    excluded: &[bool],
) -> Prepared {
    let n = cols.ts.len();
    let mut keep = Vec::with_capacity(n);
    let mut name = Vec::with_capacity(n);
    let mut para = Vec::with_capacity(n);
    let mut pg_name = Vec::with_capacity(n);
    let process_group_descriptions = process_group_descriptions(process_groups);
    for (i, &skip) in excluded.iter().enumerate().take(n) {
        let Some(cat) = cols.cat(i) else { continue };
        if cat.is_empty() || skip {
            continue;
        }
        keep.push(i as u32);
        // polars str.replace(pat, val, literal=false, n=1): the pattern has
        // no metacharacters, so this is a first-occurrence literal replace.
        name.push(cols.name(i).and_then(|nm| {
            nm.find("ncclDevKernel").map(|pos| {
                let mut s = String::with_capacity(nm.len());
                s.push_str(&nm[..pos]);
                s.push_str("ncclKernel");
                s.push_str(&nm[pos + "ncclDevKernel".len()..]);
                s
            })
        }));
        para.push(event_comm_type(
            cols.pg_desc(i),
            cols.pg_name(i),
            &process_group_descriptions,
        ));
        pg_name.push(cols.pg_name(i).to_string());
    }
    Prepared {
        keep,
        renamed: name,
        para_type: para,
        pg_name,
    }
}

/// Deterministic pairwise f64 summation: O(log n) error growth (vs O(n)
/// for a naive running sum) and a fixed addition order, so equal inputs
/// give bit-equal sums on every run and host.
pub fn f64_sum(f: &[f64]) -> f64 {
    fn pairwise(f: &[f64]) -> f64 {
        if f.len() <= 32 {
            return f.iter().sum();
        }
        let mid = f.len() / 2;
        pairwise(&f[..mid]) + pairwise(&f[mid..])
    }
    // std's f64 Sum identity is -0.0 (empty input sums to it), and absent
    // metrics would then serialize as "-0.0" in artifact text; + 0.0
    // canonicalizes.
    pairwise(f) + 0.0
}

/// Collective types with their own breakdown columns, in column-suffix form.
const COLLECTIVES: [&str; 4] = ["all_gather", "reduce_scatter", "all_reduce", "all_to_all"];

/// Fixed breakdown columns besides `rank`; per-axis columns are added per run.
/// Entries 2..6 are the per-collective totals in `COLLECTIVES` order.
const FIXED_KEYS: [&str; 11] = [
    "comm_total",
    "comp_total",
    "all_gather_total",
    "reduce_scatter_total",
    "all_reduce_total",
    "all_to_all_total",
    "total_duration",
    "cudaLaunchKernel_total",
    "cudaEventSynchronize_total",
    "cudaStreamSynchronize_total",
    "cudaHostAlloc_total",
];

/// Index into `COLLECTIVES`. The c10d collective name wins, so annotated
/// custom kernels are classified too; NCCL kernel names are the fallback for
/// kernels without PG metadata.
fn collective_index(collective_name: &str, info: &NameInfo) -> Option<usize> {
    if collective_name.is_empty() {
        return if info.all_gather {
            Some(0)
        } else if info.reduce_scatter {
            Some(1)
        } else if info.all_reduce {
            Some(2)
        } else {
            None
        };
    }
    let norm: String = collective_name
        .chars()
        .filter(|c| *c != '_')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    ["allgather", "reducescatter", "allreduce", "alltoall"]
        .iter()
        .position(|pat| norm.contains(pat))
}

/// Breakdown column names must be `&'static str`; per-axis names are only
/// known at run time, so each distinct one is leaked once per process.
fn intern(s: String) -> &'static str {
    static SET: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<&'static str>>> =
        std::sync::OnceLock::new();
    let mut set = SET
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(k) = set.get(s.as_str()) {
        return k;
    }
    let k: &'static str = Box::leak(s.into_boxed_str());
    set.insert(k);
    k
}

/// Streaming breakdown accumulator: per-row `visit` in frame order, then
/// `finish` for the deterministic sums. The materialized and fused paths
/// share this single implementation, so their values agree by construction.
struct BreakdownAcc {
    /// para_type -> per-collective `<axis>_<collective>_total` keys of every
    /// axis listing it.
    axis_keys: rustc_hash::FxHashMap<String, Vec<[&'static str; 4]>>,
    /// Per-axis keys in `rank_info` order.
    axis_order: Vec<&'static str>,
    m: rustc_hash::FxHashMap<&'static str, Vec<f64>>,
    ts_min: f64,
    end_max: f64,
    any_gpu: bool,
}

impl BreakdownAcc {
    fn new(rank_info: &RankInfo) -> Self {
        // Per-metric value collectors in frame row order; summed at the end
        // with the deterministic pairwise kernel.
        let mut m = rustc_hash::FxHashMap::<&'static str, Vec<f64>>::default();
        for k in FIXED_KEYS {
            m.insert(k, Vec::new());
        }
        let mut axis_keys = rustc_hash::FxHashMap::<String, Vec<[&'static str; 4]>>::default();
        let mut axis_order = Vec::new();
        for (axis, para_types) in rank_info {
            let keys = COLLECTIVES.map(|c| intern(format!("{axis}_{c}_total")));
            for k in keys {
                if m.insert(k, Vec::new()).is_none() {
                    axis_order.push(k);
                }
            }
            for para in para_types {
                axis_keys.entry(para.clone()).or_default().push(keys);
            }
        }
        BreakdownAcc {
            axis_keys,
            axis_order,
            m,
            ts_min: f64::INFINITY,
            end_max: f64::NEG_INFINITY,
            any_gpu: false,
        }
    }

    /// One kept row, frame order. `cat` is non-null non-empty (keep filter).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &mut self,
        cat: &str,
        name: Option<&str>,
        para: &str,
        collective_name: &str,
        pg_name: &str,
        ts: f64,
        dur: f64,
        stream: i64,
        names: &mut NameCache,
    ) {
        let m = &mut self.m;
        let mut add = |k: &'static str, v: f64| {
            if let Some(vals) = m.get_mut(k) {
                vals.push(v);
            }
        };

        if cat == "cuda_runtime" {
            match name {
                Some("cudaLaunchKernel") => add("cudaLaunchKernel_total", dur),
                Some("cudaEventSynchronize") => add("cudaEventSynchronize_total", dur),
                Some("cudaStreamSynchronize") => add("cudaStreamSynchronize_total", dur),
                Some("cudaHostAlloc") => add("cudaHostAlloc_total", dur),
                _ => {}
            }
        }

        // GPU-row subset: stream != -1 && cat != gpu_user_annotation.
        if stream == -1 || cat == "gpu_user_annotation" {
            return;
        }
        self.any_gpu = true;
        if ts < self.ts_min {
            self.ts_min = ts;
        }
        if ts + dur > self.end_max {
            self.end_max = ts + dur;
        }

        // Null name -> no kernel type -> no bucket.
        let Some(info) = name.map(|n| names.get(n)) else {
            return;
        };
        let annotated =
            cat == "kernel" && kernel_types::is_annotated_collective(collective_name, pg_name);
        if info.kernel_type == KT_COMMUNICATION || annotated {
            add("comm_total", dur);
            let Some(c) = collective_index(collective_name, &info) else {
                return;
            };
            add(FIXED_KEYS[2 + c], dur);
            for keys in self.axis_keys.get(para).into_iter().flatten() {
                add(keys[c], dur);
            }
        } else if info.kernel_type == KT_COMPUTATION {
            add("comp_total", dur);
        }
    }

    /// Deterministic sums + row assembly.
    fn finish(self, rank: i64) -> Vec<(&'static str, BreakdownValue)> {
        // total_duration = max(end) - min(ts) over gpu rows; empty -> 0.0.
        let total_duration = if self.any_gpu {
            self.end_max - self.ts_min
        } else {
            0.0
        };
        let mut row: Vec<(&'static str, BreakdownValue)> =
            Vec::with_capacity(1 + FIXED_KEYS.len() + self.axis_order.len());
        row.push(("rank", BreakdownValue::Int(rank)));
        for key in FIXED_KEYS
            .into_iter()
            .chain(self.axis_order.iter().copied())
        {
            let v = if key == "total_duration" {
                total_duration
            } else {
                self.m.get(key).map_or(0.0, |vals| f64_sum(vals))
            };
            row.push((key, BreakdownValue::Float(v)));
        }
        row
    }
}

/// Breakdown over the materialized columns (wrapper around the shared
/// accumulator; the fused path calls the accumulator directly).
fn breakdown(
    cols: &ConvertedColumns,
    prep: &Prepared,
    rank: i64,
    rank_info: &RankInfo,
    names: &mut NameCache,
) -> Vec<(&'static str, BreakdownValue)> {
    let mut acc = BreakdownAcc::new(rank_info);
    for (j, &i) in prep.keep.iter().enumerate() {
        let i = i as usize;
        let cat = cols.cat(i).unwrap_or("");
        let name: Option<&str> = prep.renamed[j].as_deref().or(cols.name(i));
        acc.visit(
            cat,
            name,
            &prep.para_type[j],
            cols.collective_name(i),
            cols.pg_name(i),
            cols.ts[i],
            cols.dur[i],
            cols.stream[i],
            names,
        );
    }
    acc.finish(rank)
}

/// Filtered comm kernels with compact transfer metadata.
struct Filtered {
    original_name: Vec<String>,
    /// Kernel (true) or copy (false); copies join only as tagged collective work.
    kernel: Vec<bool>,
    comm: Vec<crate::comm::CommMetadata>,
    name: Vec<String>,
    ts: Vec<f64>,
    dur: Vec<f64>,
    stream: Vec<i64>,
    grid: Vec<i64>,
    para_type: Vec<String>,
    pg_name: Vec<String>,
    collective_name: Vec<String>,
    in_nelems: Vec<i64>,
    out_nelems: Vec<i64>,
    group_size: Vec<i64>,
    step: Vec<i64>,
}

impl Filtered {
    fn new() -> Self {
        Filtered {
            original_name: Vec::new(),
            kernel: Vec::new(),
            comm: Vec::new(),
            name: Vec::new(),
            ts: Vec::new(),
            dur: Vec::new(),
            stream: Vec::new(),
            grid: Vec::new(),
            para_type: Vec::new(),
            pg_name: Vec::new(),
            collective_name: Vec::new(),
            in_nelems: Vec::new(),
            out_nelems: Vec::new(),
            group_size: Vec::new(),
            step: Vec::new(),
        }
    }

    /// One kept row, frame order (shared by materialized + fused paths).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &mut self,
        cat: &str,
        name: Option<&str>,
        para: &str,
        pg_name: &str,
        ts: f64,
        dur: f64,
        stream: i64,
        grid: i64,
        collective_name: &str,
        in_nelems: i64,
        out_nelems: i64,
        group_size: i64,
        names: &mut NameCache,
        original_name: Option<&str>,
        comm: Option<&crate::comm::CommMetadata>,
    ) {
        if stream == -1 {
            return;
        }
        let annotated = kernel_types::is_annotated_collective(collective_name, pg_name);
        // Kernels, plus copies a collective call tags with its metadata, e.g. a
        // copy-engine all-gather's copies, which have no kernel to stand for them.
        match cat {
            "kernel" => {}
            "gpu_memcpy" if annotated => {}
            _ => return,
        }
        let Some(name) = name else {
            return;
        };
        if !names.get(name).comm_pattern && !annotated {
            return;
        }
        // strip_prefix("void ") then regex-replace-first `[<(].*` → "" ==
        // truncate at the first '<' or '('.
        let stripped = name.strip_prefix("void ").unwrap_or(name);
        let cut = stripped
            .find(['<', '('])
            .map(|p| &stripped[..p])
            .unwrap_or(stripped);
        self.original_name
            .push(original_name.unwrap_or(name).to_string());
        self.kernel.push(cat == "kernel");
        self.comm.push(comm.cloned().unwrap_or_default());
        self.name.push(cut.to_string());
        self.ts.push(ts);
        self.dur.push(dur);
        self.stream.push(stream);
        self.grid.push(grid);
        self.para_type.push(para.to_string());
        self.pg_name.push(pg_name.to_string());
        self.collective_name.push(collective_name.to_string());
        self.in_nelems.push(in_nelems);
        self.out_nelems.push(out_nelems);
        self.group_size.push(group_size);
        self.step.push(-1);
    }

    fn assign_steps(&mut self, steps: &[(i64, f64, f64)]) {
        for (i, ts) in self.ts.iter().enumerate() {
            if let Some((step, _, _)) = steps.iter().find(|(_, start, end)| ts >= start && ts < end)
            {
                self.step[i] = *step;
            }
        }
    }
}

fn profiler_step(name: Option<&str>) -> Option<i64> {
    name?.strip_prefix("ProfilerStep#")?.parse().ok()
}

/// `(step, start, end)` windows for tagging GPU kernels with their step.
/// GPU work trails the CPU `ProfilerStep#N` span, so the GPU-side copies
/// (`gpu_user_annotation`) are used when present: step N runs from its
/// earliest GPU span to the next step's, and the last step is open-ended.
fn step_spans<'a>(
    events: impl Iterator<Item = (Option<&'a str>, Option<&'a str>, f64, f64)>,
) -> Vec<(i64, f64, f64)> {
    let mut cpu = Vec::new();
    let mut gpu: rustc_hash::FxHashMap<i64, f64> = Default::default();
    for (cat, name, ts, dur) in events {
        let Some(step) = profiler_step(name) else {
            continue;
        };
        match cat {
            Some("user_annotation") => cpu.push((step, ts, ts + dur)),
            Some("gpu_user_annotation") => {
                let start = gpu.entry(step).or_insert(ts);
                *start = start.min(ts);
            }
            _ => {}
        }
    }
    if gpu.is_empty() {
        return cpu;
    }
    let mut starts: Vec<(i64, f64)> = gpu.into_iter().collect();
    starts.sort_by(|a, b| a.1.total_cmp(&b.1));
    starts
        .iter()
        .enumerate()
        .map(|(i, &(step, start))| {
            (
                step,
                start,
                starts.get(i + 1).map_or(f64::INFINITY, |next| next.1),
            )
        })
        .collect()
}

fn union_duration(mut spans: Vec<(f64, f64)>) -> f64 {
    spans.retain(|(start, end)| start.is_finite() && end.is_finite() && end > start);
    spans.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
    let Some((mut start, mut end)) = spans.first().copied() else {
        return 0.0;
    };
    let mut total = 0.0;
    for &(next_start, next_end) in &spans[1..] {
        if next_start <= end {
            end = end.max(next_end);
        } else {
            total += end - start;
            (start, end) = (next_start, next_end);
        }
    }
    total + end - start
}

struct DiagnosticEvent<'a> {
    cat: &'a str,
    name: &'a str,
    ph: &'a str,
    ts: f64,
    dur: f64,
    counter_value: Option<f64>,
    annotated_collective: bool,
}

/// GPU counter that Cuspy samples once per kernel rather than over time, so
/// its distribution describes the kernel mix, not the GPU's state.
const PER_KERNEL_COUNTER: &str = "Cycles (Current Kernel)";

/// Summary statistics of one GPU counter on one rank, in `PM_STATS` order.
fn counter_stats(values: &mut [f64]) -> [f64; 9] {
    values.sort_by(f64::total_cmp);
    let n = values.len() as f64;
    let at = |p: f64| values[((p * n).ceil() as usize).clamp(1, values.len()) - 1];
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    [
        n,
        values[0],
        at(0.10),
        at(0.50),
        mean,
        at(0.90),
        at(0.99),
        values[values.len() - 1],
        var.sqrt(),
    ]
}

/// Stat columns of the `pm` rows of the diagnostics member.
pub const PM_STATS: [&str; 9] = ["n", "min", "p10", "p50", "mean", "p90", "p99", "max", "std"];

/// One rank's diagnostics: an `activity` row (GPU busy / compute / CPU time
/// over the GPU window, and counts of communication kernels without c10d
/// process-group metadata) and one `pm` row of summary stats per GPU
/// counter. Counter samples themselves are not kept: at fleet scale they
/// outweigh every other table combined.
fn diagnostics_df<'a>(
    rank: i64,
    events: impl Iterator<Item = DiagnosticEvent<'a>>,
) -> Result<(DataFrame, Option<DataFrame>)> {
    let mut gpu = Vec::new();
    let mut compute = Vec::new();
    let mut cpu = Vec::new();
    // Per counter: (ts, value) samples.
    let mut counters: std::collections::BTreeMap<&str, Vec<(f64, f64)>> = Default::default();
    // NCCL kernels should all carry c10d metadata; those without it lost
    // their annotation (e.g. CUDA-graph replays without capture
    // annotations). Other communication kernels (HybridEP, DeepEP, ...)
    // never carry it.
    let (mut nccl, mut nccl_bare, mut other_bare) = (0u64, 0u64, 0u64);
    let (mut nccl_us, mut nccl_bare_us) = (0.0, 0.0);
    for event in events {
        let span = (event.ts, event.ts + event.dur);
        if matches!(event.cat, "kernel" | "gpu_memcpy" | "gpu_memset") {
            gpu.push(span);
        }
        if event.cat == "kernel" {
            let kind = kernel_types::get_kernel_type(event.name);
            if !event.annotated_collective && kind == KT_COMPUTATION {
                compute.push(span);
            }
            let is_nccl = event.name.starts_with("nccl") || event.name.contains(" nccl");
            if is_nccl {
                nccl += 1;
                nccl_us += event.dur;
                if !event.annotated_collective {
                    nccl_bare += 1;
                    nccl_bare_us += event.dur;
                }
            } else if kind == KT_COMMUNICATION && !event.annotated_collective {
                other_bare += 1;
            }
        }
        if matches!(event.cat, "cpu_op" | "cuda_runtime" | "cuda_driver") {
            cpu.push(span);
        }
        if event.ph == "C" && event.name != PER_KERNEL_COUNTER {
            if let Some(value) = event.counter_value {
                counters
                    .entry(event.name)
                    .or_default()
                    .push((event.ts, value));
            }
        }
    }
    let start = gpu.iter().map(|x| x.0).reduce(f64::min).unwrap_or(0.0);
    let end = gpu.iter().map(|x| x.1).reduce(f64::max).unwrap_or(start);
    let window = end - start;
    let gpu_busy = union_duration(gpu);
    let compute_busy = union_duration(compute);
    let cpu_active = union_duration(
        cpu.into_iter()
            .filter_map(|(a, b)| {
                let clipped = (a.max(start), b.min(end));
                (clipped.1 > clipped.0).then_some(clipped)
            })
            .collect(),
    );
    let n = counters.len() + 1;
    let mut kind = Vec::with_capacity(n);
    let mut metric = Vec::with_capacity(n);
    let mut stats: Vec<Vec<f64>> = vec![Vec::with_capacity(n); PM_STATS.len()];
    kind.push("activity".to_string());
    metric.push(String::new());
    for col in &mut stats {
        col.push(0.0);
    }
    let bins = counter_bins_df(rank, &counters)?;
    for (name, samples) in counters {
        kind.push("pm".to_string());
        metric.push(name.to_string());
        let mut values: Vec<f64> = samples.into_iter().map(|(_, v)| v).collect();
        for (col, v) in stats.iter_mut().zip(counter_stats(&mut values)) {
            col.push(v);
        }
    }
    let first_only = |v: f64| {
        let mut col = vec![0.0; n];
        col[0] = v;
        col
    };
    let first_only_u = |v: u64| {
        let mut col = vec![0i64; n];
        col[0] = v as i64;
        col
    };
    let mut columns = vec![
        Column::new("kind".into(), kind),
        Column::new("rank".into(), vec![rank as i32; n]),
        Column::new("metric".into(), metric),
        Column::new("gpu_busy".into(), first_only(gpu_busy)),
        Column::new("compute".into(), first_only(compute_busy)),
        Column::new("cpu_active".into(), first_only(cpu_active)),
        Column::new("window".into(), first_only(window)),
        Column::new("nccl_kernels".into(), first_only_u(nccl)),
        Column::new("nccl_unannotated".into(), first_only_u(nccl_bare)),
        Column::new("other_comm_unannotated".into(), first_only_u(other_bare)),
        Column::new("nccl_us".into(), first_only(nccl_us)),
        Column::new("nccl_unannotated_us".into(), first_only(nccl_bare_us)),
    ];
    for (name, col) in PM_STATS.iter().zip(stats) {
        columns.push(Column::new((*name).into(), col));
    }
    Ok((DataFrame::new_infer_height(columns)?, bins))
}

/// Stat columns of the counter timeline bins.
pub const COUNTER_BIN_STATS: [&str; 5] = ["min", "p1", "p50", "p99", "max"];

/// One rank's counter timelines: its counter samples' time span cut into
/// `COUNTER_BINS` equal bins, with min, p1, p50, p99 and max of each
/// counter's samples per bin (empty bins have no row). `t0` is the bin's
/// start in trace time and `width` its length, both us.
fn counter_bins_df(
    rank: i64,
    counters: &std::collections::BTreeMap<&str, Vec<(f64, f64)>>,
) -> Result<Option<DataFrame>> {
    let all = counters.values().flatten().map(|(ts, _)| *ts);
    let (lo, hi) = all.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| {
        (lo.min(t), hi.max(t))
    });
    if !lo.is_finite() {
        return Ok(None);
    }
    let width = ((hi - lo) / COUNTER_BINS as f64).max(1.0);
    let (mut metric, mut bin, mut t0) = (Vec::new(), Vec::new(), Vec::new());
    let mut stats: Vec<Vec<f64>> = vec![Vec::new(); COUNTER_BIN_STATS.len()];
    for (name, samples) in counters {
        let mut by_bin: Vec<Vec<f64>> = vec![Vec::new(); COUNTER_BINS];
        for (ts, v) in samples {
            let b = (((ts - lo) / width) as usize).min(COUNTER_BINS - 1);
            by_bin[b].push(*v);
        }
        for (b, mut values) in by_bin.into_iter().enumerate() {
            if values.is_empty() {
                continue;
            }
            values.sort_by(f64::total_cmp);
            let n = values.len() as f64;
            let at = |p: f64| values[((p * n).ceil() as usize).clamp(1, values.len()) - 1];
            metric.push(name.to_string());
            bin.push(b as i32);
            t0.push(lo + b as f64 * width);
            for (col, v) in stats.iter_mut().zip([
                values[0],
                at(0.01),
                at(0.5),
                at(0.99),
                values[values.len() - 1],
            ]) {
                col.push(v);
            }
        }
    }
    let n = metric.len();
    let mut columns = vec![
        Column::new("rank".into(), vec![rank as i32; n]),
        Column::new("metric".into(), metric),
        Column::new("bin".into(), bin),
        Column::new("t0".into(), t0),
        Column::new("width".into(), vec![width; n]),
    ];
    for (name, col) in COUNTER_BIN_STATS.iter().zip(stats) {
        columns.push(Column::new((*name).into(), col));
    }
    Ok(Some(DataFrame::new_infer_height(columns)?))
}

fn filtered_kernels(cols: &ConvertedColumns, prep: &Prepared, names: &mut NameCache) -> Filtered {
    let mut out = Filtered::new();
    for (j, &i) in prep.keep.iter().enumerate() {
        let i = i as usize;
        let cat = cols.cat(i).unwrap_or("");
        let name = prep.renamed[j].as_deref().or(cols.name(i));
        out.visit(
            cat,
            name,
            &prep.para_type[j],
            &prep.pg_name[j],
            cols.ts[i],
            cols.dur[i],
            cols.stream[i],
            cols.grid[i],
            cols.collective_name(i),
            cols.in_nelems[i],
            cols.out_nelems[i],
            cols.group_size[i],
            names,
            cols.name(i),
            cols.comm(i),
        );
    }
    out
}

fn filtered_to_df(f: &Filtered, sel: &[u32], rank: i64) -> Result<DataFrame> {
    let take = |v: &Vec<f64>| -> Vec<f64> { sel.iter().map(|&i| v[i as usize]).collect() };
    let take_i = |v: &Vec<i64>| -> Vec<i64> { sel.iter().map(|&i| v[i as usize]).collect() };
    let take_s =
        |v: &Vec<String>| -> Vec<String> { sel.iter().map(|&i| v[i as usize].clone()).collect() };
    let field = |idx: usize| -> Vec<Option<&str>> {
        sel.iter().map(|&i| f.comm[i as usize].get(idx)).collect()
    };
    let operations: Vec<_> = sel
        .iter()
        .map(|&i| f.comm[i as usize].operation())
        .collect();
    let peers: Vec<_> = sel
        .iter()
        .zip(&operations)
        .map(|(&i, op)| op.and_then(|_| crate::comm::peer_global(&f.comm[i as usize], rank)))
        .collect();
    let src: Vec<_> = operations
        .iter()
        .zip(&peers)
        .map(|(op, peer)| match op {
            Some("send") => Some(rank),
            Some("recv") => *peer,
            _ => None,
        })
        .collect();
    let dst: Vec<_> = operations
        .iter()
        .zip(&peers)
        .map(|(op, peer)| match op {
            Some("recv") => Some(rank),
            Some("send") => *peer,
            _ => None,
        })
        .collect();
    let (graph, node): (Vec<_>, Vec<_>) =
        sel.iter().map(|&i| f.comm[i as usize].graph_ids()).unzip();
    // A graph-replayed event's correlation is its cudaGraphLaunch's, so it names
    // the replay; on an eager event it names only that one launch.
    let replay: Vec<Option<String>> = sel
        .iter()
        .zip(&graph)
        .map(|(&i, g)| g.as_ref().and(f.comm[i as usize].get(9).map(str::to_owned)))
        .collect();
    // A collective call that launches a communication kernel is timed by it,
    // not by local work the call also tags (copy-in/out); a call made only of
    // copies (copy-engine collectives) is timed by all of them.
    let comm_kernel: Vec<i64> = sel
        .iter()
        .map(|&i| i64::from(f.kernel[i as usize]))
        .collect();
    let columns = vec![
        Column::new("name".into(), take_s(&f.name)),
        Column::new("ts".into(), take(&f.ts)),
        Column::new("dur".into(), take(&f.dur)),
        Column::new("stream".into(), take_i(&f.stream)),
        Column::new("grid".into(), take_i(&f.grid)),
        Column::new("para_type".into(), take_s(&f.para_type)),
        Column::new("pg_name".into(), take_s(&f.pg_name)),
        Column::new("collective_name".into(), take_s(&f.collective_name)),
        Column::new("in_nelems".into(), take_i(&f.in_nelems)),
        Column::new("out_nelems".into(), take_i(&f.out_nelems)),
        Column::new("group_size".into(), take_i(&f.group_size)),
        Column::new("step".into(), take_i(&f.step)),
        Column::new("rank".into(), vec![rank as i32; sel.len()]),
        Column::new("original_name".into(), take_s(&f.original_name)),
        Column::new("operation".into(), operations),
        Column::new("src_rank".into(), src),
        Column::new("dst_rank".into(), dst),
        Column::new("pg_id".into(), field(8)),
        Column::new("replica_id".into(), vec![0i64; sel.len()]),
        Column::new("msg_nelems".into(), field(3)),
        Column::new("dtype".into(), field(4)),
        Column::new("sequence_id".into(), field(5)),
        Column::new("graph_id".into(), graph),
        Column::new("node_id".into(), node),
        Column::new("graph_node_id".into(), field(7)),
        Column::new("replay_id".into(), replay),
        Column::new("comm_kernel".into(), comm_kernel),
    ];
    Ok(DataFrame::new_infer_height(columns)?)
}

/// One rank's breakdown row: (column name, value) pairs.
pub type BreakdownRow = Vec<(&'static str, BreakdownValue)>;
/// Per-PG split output: (group key, dataframe) pairs.
pub type PgDataFrames = Vec<(String, DataFrame)>;

/// Full per-rank analysis over materialized columns.
/// Returns (breakdown_row, pg_dfs).
pub fn process(
    cols: &ConvertedColumns,
    rank: i64,
    process_groups: &[ProcessGroupConfig],
) -> Result<(BreakdownRow, PgDataFrames)> {
    let rank_info = trace_rank_info(process_groups);
    let excluded =
        launched_before_window((0..cols.ts.len()).map(|i| (cols.cat(i), cols.correlation[i])));
    let prep = prepare(cols, process_groups, &excluded);
    let (row, mut filt) = with_name_cache(|names| {
        let row = breakdown(cols, &prep, rank, &rank_info, names);
        let filt = filtered_kernels(cols, &prep, names);
        (row, filt)
    });

    let steps = step_spans(
        (0..cols.ts.len()).map(|i| (cols.cat(i), cols.name(i), cols.ts[i], cols.dur[i])),
    );
    filt.assign_steps(&steps);
    let (diagnostics, bins) = diagnostics_df(
        rank,
        (0..cols.ts.len())
            .filter(|&i| !excluded[i])
            .map(|i| DiagnosticEvent {
                cat: cols.cat(i).unwrap_or(""),
                name: cols.name(i).unwrap_or(""),
                ph: cols.ph(i),
                ts: cols.ts[i],
                dur: cols.dur[i],
                counter_value: cols.counter_value[i],
                annotated_collective: kernel_types::is_annotated_collective(
                    cols.collective_name(i),
                    cols.pg_name(i),
                ),
            }),
    )?;

    let mut pg_dfs = pg_split(&filt, rank, &rank_info)?;
    pg_dfs.push((DIAGNOSTICS_PG.to_string(), diagnostics));
    if let Some(bins) = bins {
        pg_dfs.push((
            format!("{COUNTER_BINS_PREFIX}{}", rank / COUNTER_BIN_RANKS),
            bins,
        ));
    }
    Ok((row, pg_dfs))
}

/// Fused single-pass analysis over the lean post-walk view: the prepare/
/// breakdown/filtered passes and the `ConvertedColumns` gather collapse into
/// ONE traversal of the sort permutation. Row visit order, accumulation
/// order, and every per-row value match the materialized path exactly, so
/// artifacts are byte-identical (shared visit/finish implementations).
pub fn process_view(
    view: &crate::rows::WalkedView,
    rank: i64,
    process_groups: &[ProcessGroupConfig],
) -> Result<(BreakdownRow, PgDataFrames)> {
    let rank_info = trace_rank_info(process_groups);
    let excluded =
        launched_before_window((0..view.len()).map(|j| (view.cat(j), view.correlation(j))));
    let process_group_descriptions = process_group_descriptions(process_groups);
    let (row, mut filt) = with_name_cache(|names| {
        let mut acc = BreakdownAcc::new(&rank_info);
        let mut filt = Filtered::new();
        // `extract_comm_type` is a pure function of the (pg_desc, pg_name)
        // pair, and consecutive rows overwhelmingly repeat the same pair
        // (one PG per kernel stream), so a last-value memo removes the
        // regex pass and the result allocation from the per-row path.
        //
        // The memo is keyed on the pair's ARENA SPANS first: the walk gives
        // every row of a graph id the same spans, so the usual answer is two
        // integer compares. Comparing the strings themselves cost a memcmp
        // over both values on every row — 6.4% of analyzer cycles on a 24k-rank run,
        // where the PG name runs ~35 bytes and the memo essentially always
        // hits. Rows whose spans differ but whose bytes match (a value that
        // came from the trace's own args rather than an annotation) still
        // hit through the byte comparison, and adopt the new spans so the
        // following rows take the fast path again.
        // (pg_desc span, pg_name span, pg_desc, pg_name, comm_type).
        type CommMemo = (
            Option<crate::rows::Span>,
            Option<crate::rows::Span>,
            String,
            String,
            String,
        );
        let mut memo: Option<CommMemo> = None;
        for (j, &skip) in excluded.iter().enumerate() {
            // prepare()'s keep filter: cat non-null, non-empty, launched in the window.
            let Some(cat) = view.cat(j) else { continue };
            if cat.is_empty() || skip {
                continue;
            }
            // prepare()'s first-occurrence literal rename, computed on demand.
            let renamed: Option<String> = view.name(j).and_then(|nm| {
                nccl_dev_kernel_pos(nm).map(|pos| {
                    let mut s = String::with_capacity(nm.len());
                    s.push_str(&nm[..pos]);
                    s.push_str("ncclKernel");
                    s.push_str(&nm[pos + "ncclDevKernel".len()..]);
                    s
                })
            });
            let name: Option<&str> = renamed.as_deref().or_else(|| view.name(j));
            let (ds, ns) = view.pg_spans(j);
            let mut hit = false;
            if let Some((mds, mns, d, n, _)) = &mut memo {
                if *mds == ds && *mns == ns {
                    hit = true;
                } else if d.as_str() == view.pg_desc(j) && n.as_str() == view.pg_name(j) {
                    *mds = ds;
                    *mns = ns;
                    hit = true;
                }
            }
            if !hit {
                let (d_in, n_in) = (view.pg_desc(j), view.pg_name(j));
                let para = event_comm_type(d_in, n_in, &process_group_descriptions);
                memo = Some((ds, ns, d_in.to_string(), n_in.to_string(), para));
            }
            let para: &str = &memo.as_ref().unwrap().4;
            let (ts, dur, stream) = (view.ts(j), view.dur(j), view.stream(j));
            acc.visit(
                cat,
                name,
                para,
                view.collective_name(j),
                view.pg_name(j),
                ts,
                dur,
                stream,
                names,
            );
            filt.visit(
                cat,
                name,
                para,
                view.pg_name(j),
                ts,
                dur,
                stream,
                view.grid(j),
                view.collective_name(j),
                view.in_nelems(j),
                view.out_nelems(j),
                view.group_size(j),
                names,
                view.name(j),
                view.comm(j),
            );
        }
        (acc.finish(rank), filt)
    });
    let steps =
        step_spans((0..view.len()).map(|i| (view.cat(i), view.name(i), view.ts(i), view.dur(i))));
    filt.assign_steps(&steps);
    let (diagnostics, bins) = diagnostics_df(
        rank,
        (0..view.len())
            .filter(|&i| !excluded[i])
            .map(|i| DiagnosticEvent {
                cat: view.cat(i).unwrap_or(""),
                name: view.name(i).unwrap_or(""),
                ph: view.ph(i),
                ts: view.ts(i),
                dur: view.dur(i),
                counter_value: view.counter_value(i),
                annotated_collective: kernel_types::is_annotated_collective(
                    view.collective_name(i),
                    view.pg_name(i),
                ),
            }),
    )?;
    let mut pg_dfs = pg_split(&filt, rank, &rank_info)?;
    pg_dfs.push((DIAGNOSTICS_PG.to_string(), diagnostics));
    if let Some(bins) = bins {
        pg_dfs.push((
            format!("{COUNTER_BINS_PREFIX}{}", rank / COUNTER_BIN_RANKS),
            bins,
        ));
    }
    Ok((row, pg_dfs))
}

/// Case-insensitive rank_info set membership for one para value: exact
/// match is the fast path, the lowercased lookup runs only on miss,
/// memoized per unique para value (paras repeat per kernel stream).
fn para_in_set<'a>(
    para: &'a str,
    set: &std::collections::HashSet<&str>,
    lowered: &std::collections::HashSet<String>,
    memo: &mut rustc_hash::FxHashMap<&'a str, bool>,
) -> bool {
    if set.contains(para) {
        return true;
    }
    *memo
        .entry(para)
        .or_insert_with(|| lowered.contains(&para.to_ascii_lowercase()))
}

/// Per-PG split of the filtered kernels (shared tail): one frame per
/// `{axis}_{pg group id}`. Only kernels carrying process-group metadata are
/// placed: a kernel recognized as communication by name alone (e.g. HybridEP,
/// DeepEP, symmetric-memory barriers) has no group or sequence number, so it
/// counts toward comm time but appears on no group timeline.
fn pg_split(filt: &Filtered, rank: i64, ri: &RankInfo) -> Result<Vec<(String, DataFrame)>> {
    let n = filt.name.len();
    let mut pg_dfs: Vec<(String, DataFrame)> = Vec::new();
    if n == 0 {
        return Ok(pg_dfs);
    }

    for (pg_type, para_types) in ri {
        if para_types.is_empty() {
            continue;
        }
        let set: std::collections::HashSet<&str> = para_types.iter().map(|s| s.as_str()).collect();
        // Case-insensitive fallback: trace annotations carry the torch
        // group_name verbatim (e.g. "ep"), while the rank_info tables use
        // the canonical description ("EP"). Exact match stays the fast
        // path; the lowercased lookup runs only on miss, memoized per
        // unique para value (paras repeat per kernel stream).
        let lowered: std::collections::HashSet<String> =
            para_types.iter().map(|s| s.to_lowercase()).collect();
        let mut memo: rustc_hash::FxHashMap<&str, bool> = rustc_hash::FxHashMap::default();
        let sel1: Vec<u32> = (0..n as u32)
            .filter(|&i| {
                para_in_set(
                    filt.para_type[i as usize].as_str(),
                    &set,
                    &lowered,
                    &mut memo,
                )
            })
            .collect();
        let selected = sel1;
        if selected.is_empty() {
            continue;
        }

        let mut by_group: std::collections::BTreeMap<String, Vec<u32>> =
            std::collections::BTreeMap::new();
        for row in selected {
            by_group
                .entry(pg_group_id(&filt.pg_name[row as usize]))
                .or_default()
                .push(row);
        }
        for (group, rows) in by_group {
            pg_dfs.push((
                format!("{pg_type}_{group}"),
                filtered_to_df(filt, &rows, rank)?,
            ));
        }
    }
    Ok(pg_dfs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collective_columns_follow_collective_name_metadata() {
        // (kernel name, collective name, dur): the custom kernel and the
        // SendRecv all-to-all count via metadata, the unannotated NCCL kernel
        // via its name, the compute kernel only in comp_total, and the
        // unrecognized kernel with collective + PG metadata as comm.
        let events: Vec<_> = [
            ("ncclDevKernel_AllGather_RING(x)", "", 3.0),
            ("ncclDevKernel_SendRecv(x)", "all_to_all", 5.0),
            ("ncclKernel_custom(x)", "_reduce_scatter_base", 7.0),
            ("gemm_kernel", "", 11.0),
            ("my_lib::a2a_dispatch(x)", "my_dispatch", 13.0),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (name, coll, dur))| {
            serde_json::json!({
                "ph": "X", "cat": "kernel", "name": name,
                "ts": i * 100, "dur": dur, "pid": 0, "tid": 1,
                "args": {"stream": 9, "grid": [1, 1, 1],
                         "Process Group Description": "mesh_dp_shard",
                         "Process Group Name": "5",
                         "Collective name": coll}
            })
        })
        .collect();
        let bytes = serde_json::to_vec(&serde_json::json!({"traceEvents": events})).unwrap();
        let cols = crate::legacy::json::convert_json_scan_to_columns(&bytes).unwrap();
        let parsed = crate::legacy::json::parse_bytes_walked(&bytes).unwrap();
        let view =
            crate::rows::convert_parsed_walked_view(parsed, &Default::default(), 0, false).unwrap();
        let pgs = [ProcessGroupConfig {
            pg_name: "5".into(),
            pg_desc: "mesh_dp_shard".into(),
            backend_config: String::new(),
            pg_size: 2,
            ranks: vec![0, 1],
        }];
        for (breakdown, groups) in [
            process(&cols, 0, &pgs).unwrap(),
            process_view(&view, 0, &pgs).unwrap(),
        ] {
            let get = |key: &str| breakdown.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
            for (key, expected) in [
                ("comm_total", 28.0),
                ("comp_total", 11.0),
                ("all_gather_total", 3.0),
                ("all_to_all_total", 5.0),
                ("reduce_scatter_total", 7.0),
                ("all_reduce_total", 0.0),
                ("dp_shard_all_gather_total", 3.0),
                ("dp_shard_all_to_all_total", 5.0),
                ("dp_shard_reduce_scatter_total", 7.0),
                ("dp_shard_all_reduce_total", 0.0),
            ] {
                assert_eq!(get(key), Some(BreakdownValue::Float(expected)), "{key}");
            }
            assert_eq!(get("gemm_total"), None, "compute sub-columns are gone");
            let keys: Vec<_> = groups
                .iter()
                .map(|(k, df)| (k.as_str(), df.height()))
                .collect();
            assert_eq!(keys, [("dp_shard_5", 4), (DIAGNOSTICS_PG, keys[1].1)]);
        }
    }

    fn one_kernel(name: &str, para_type: &str) -> Filtered {
        Filtered {
            original_name: vec![name.to_string()],
            kernel: vec![true],
            comm: vec![Default::default()],
            name: vec![name.to_string()],
            ts: vec![1.0],
            dur: vec![2.0],
            stream: vec![3],
            grid: vec![4],
            para_type: vec![para_type.to_string()],
            pg_name: vec![String::new()],
            collective_name: vec![String::new()],
            in_nelems: vec![-1],
            out_nelems: vec![-1],
            group_size: vec![-1],
            step: vec![-1],
        }
    }

    fn tp_ep_rank_info() -> RankInfo {
        vec![
            ("tp".to_string(), vec!["TP".to_string()]),
            ("ep".to_string(), vec!["EP".to_string()]),
        ]
    }

    #[test]
    fn explicit_pg_annotation_prevents_shared_pattern_cross_attribution() {
        let frames = pg_split(
            &one_kernel("_triton_symm_mem_barrier_multimem", "EP"),
            0,
            &tp_ep_rank_info(),
        )
        .unwrap();

        assert_eq!(
            frames
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            vec!["ep_unnamed"]
        );
    }

    #[test]
    fn unannotated_kernel_is_on_no_group() {
        // Recognized as communication by name, but with no process group to
        // place it in or sequence number to match it by.
        let frames = pg_split(
            &one_kernel("_triton_a2a_v2_kernel", ""),
            0,
            &tp_ep_rank_info(),
        )
        .unwrap();
        assert!(frames.is_empty());
    }
}

#[cfg(test)]
mod process_group_tests {
    use super::*;

    fn group(name: &str, description: &str, ranks: &[i64]) -> ProcessGroupConfig {
        ProcessGroupConfig {
            pg_name: name.to_string(),
            pg_desc: description.to_string(),
            backend_config: "cuda:nccl".to_string(),
            pg_size: ranks.len() as i64,
            ranks: ranks.to_vec(),
        }
    }

    #[test]
    fn counter_bins_cover_the_samples_span() {
        // 640 samples over [0, 640) us: 10 per bin, values 0..639.
        let samples: Vec<(f64, f64)> = (0..640).map(|i| (i as f64, i as f64)).collect();
        let counters = std::collections::BTreeMap::from([("SM Active (%)", samples)]);
        let df = counter_bins_df(3, &counters).unwrap().expect("bins");
        assert_eq!(df.height(), COUNTER_BINS);
        let col = |name: &str| df.column(name).unwrap().f64().unwrap().clone();
        assert_eq!(col("min").get(0), Some(0.0));
        assert_eq!(col("max").get(0), Some(9.0));
        assert_eq!(col("p50").get(1), Some(14.0));
        assert_eq!(col("max").get(COUNTER_BINS - 1), Some(639.0));
        assert!(counter_bins_df(3, &Default::default()).unwrap().is_none());
    }

    #[test]
    fn undescribed_groups_split_by_size_and_stride() {
        let group = |name: &str, ranks: Vec<i64>| ProcessGroupConfig {
            pg_name: name.into(),
            pg_desc: "undefined".into(),
            backend_config: "cuda:nccl".into(),
            pg_size: ranks.len() as i64,
            ranks,
        };
        let configs = vec![
            group("1", vec![0, 1]),
            group("2", vec![2, 3]),
            group("21", vec![0, 4]),
            group("30", vec![0, 1, 2, 3]),
        ];
        let axes: Vec<String> = process_group_axes(&configs).into_keys().collect();
        assert_eq!(
            axes,
            [
                "undefined_size2_stride1",
                "undefined_size2_stride4",
                "undefined_size4_stride1"
            ]
        );
    }

    #[test]
    fn axes_preserve_one_and_multi_axis_topologies() {
        let configs = vec![
            group("0", "default_pg", &[0, 1, 2, 3]),
            group("20", "mesh_tp", &[2, 3]),
            group("3", "mesh_tp", &[0, 1]),
            group("31", "mesh_cp", &[0, 2]),
            group("45", "mesh_dp_replicate", &[0, 3]),
            group("51", "mesh_custom_axis", &[1, 2]),
            group("60", "mesh_dp_shard", &[0, 1]),
        ];
        let axes = process_group_axes(&configs);

        assert_eq!(
            axes.keys().map(String::as_str).collect::<Vec<_>>(),
            vec![
                "cp",
                "custom_axis",
                "default_pg",
                "dp_replicate",
                "dp_shard",
                "tp"
            ],
            "descriptions are used as is, without mapping onto built-in axes"
        );
        assert_eq!(
            axes["tp"]
                .iter()
                .map(|group| group.pg_name.as_str())
                .collect::<Vec<_>>(),
            vec!["3", "20"]
        );
    }

    #[test]
    fn gpu_events_launched_before_the_window_are_excluded() {
        let rows = [
            (Some("cuda_runtime"), 10), // launch of the kernel below
            (Some("kernel"), 10),       // launched in the window
            (Some("kernel"), 3),        // its launch predates the window
            (Some("gpu_memcpy"), 4),    // likewise
            (Some("kernel"), -1),       // no correlation: can't tell, kept
            (Some("cpu_op"), 3),        // CPU rows are never excluded
        ];
        assert_eq!(
            launched_before_window(rows.into_iter()),
            [false, false, true, true, false, false]
        );
        // Without any launch records nothing can be attributed: keep all.
        let gpu_only = [(Some("kernel"), 3), (Some("gpu_memset"), 4)];
        assert_eq!(launched_before_window(gpu_only.into_iter()), [false, false]);
    }

    #[test]
    fn trace_topology_splits_by_physical_group() {
        let filt = Filtered {
            original_name: vec!["ncclKernel_AllReduce_RING_LL".into()],
            kernel: vec![true],
            comm: vec![Default::default()],
            name: vec!["ncclKernel_AllReduce_RING_LL".into()],
            ts: vec![1.0],
            dur: vec![2.0],
            stream: vec![7],
            grid: vec![1],
            para_type: vec!["mesh_tp".into()],
            pg_name: vec!["17".into()],
            collective_name: vec!["all_reduce".into()],
            in_nelems: vec![1],
            out_nelems: vec![1],
            group_size: vec![2],
            step: vec![0],
        };
        let trace_topology = trace_rank_info(&[group("17", "mesh_tp", &[0, 1])]);

        let groups = pg_split(&filt, 0, &trace_topology).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "tp_17");
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Structured report: a summary of the run for scripts and LLM agents, as
//! `README.md` plus the same content in `summary.json`.
//!
//! Its size does not grow with world size: every list is a top-K or a
//! distribution, so a 50k-rank run reads like an 8-rank one. Cross-rank
//! matching follows the page's JavaScript (`pl.cohorts` in
//! static_page.html): calls pair across ranks by (process group, step,
//! collective, c10d sequence, occurrence), and calls without a sequence or
//! p2p calls join no cohort.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use rayon::prelude::*;
use serde_json::json;

use crate::diagnostics::Activity;
use crate::diagnostics::Diagnostics;
use crate::diagnostics::num;
use crate::diagnostics::ranks_short;
use crate::diagnostics::table_from_csv;
use crate::diagnostics::table_from_member;
use crate::publish::RunData;

/// Rows in each top-K table.
const TOP: usize = 10;
/// Axes listed, busiest first: groups without a description are each their
/// own axis, which at scale could be thousands.
const MAX_AXES: usize = 30;
/// Ranks in the straggler table.
const TOP_RANKS: usize = 20;
/// Worst cohorts kept per group before the axis-wide top-K.
const WORST_PER_GROUP: usize = 3;
/// Lowest and highest ranks listed per per-rank metric.
const EXTREMES: usize = 5;

#[derive(Clone, Debug)]
struct Event {
    rank: i64,
    ts: f64,
    dur: f64,
    original_name: String,
    sequence: Option<String>,
    pg: String,
    collective: String,
    dtype: String,
    in_n: f64,
    out_n: f64,
    group_size: i64,
    step: Option<i64>,
    /// c10d send/recv, whose sequence is 0 on every call.
    p2p: bool,
    /// Grouping key parts of the page's `instancesOf`: p2p direction and
    /// peer, CUDA-graph replay, and whether this is a communication kernel
    /// (else a copy or local work the call tagged).
    operation: String,
    peer: String,
    replay: String,
    comm_kernel: bool,
}

fn events_from_table(table: &crate::diagnostics::Table) -> Vec<Event> {
    let get = |name: &str, row: usize| table.col(name).map(|col| col[row].as_str());
    let optional = |name: &str, row: usize| get(name, row).filter(|v| !v.is_empty());
    (0..table.nrows())
        .map(|row| {
            let raw = get("name", row).unwrap_or("");
            let collective = get("collective_name", row).unwrap_or("").to_string();
            let operation = optional("operation", row);
            let peer = match operation {
                Some("send") => optional("dst_rank", row),
                Some("recv") => optional("src_rank", row),
                _ => None,
            };
            Event {
                rank: num(get("rank", row).unwrap_or("0")) as i64,
                ts: num(get("ts", row).unwrap_or("0")),
                dur: num(get("dur", row).unwrap_or("0")),
                original_name: optional("original_name", row).unwrap_or(raw).to_string(),
                sequence: optional("sequence_id", row).map(str::to_string),
                pg: get("pg_name", row).unwrap_or("").to_string(),
                p2p: matches!(operation, Some("send" | "recv"))
                    || matches!(collective.as_str(), "send" | "recv"),
                collective,
                dtype: get("dtype", row).unwrap_or("").to_string(),
                in_n: get("in_nelems", row).map_or(-1.0, num),
                out_n: get("out_nelems", row).map_or(-1.0, num),
                group_size: get("group_size", row).map_or(-1, |v| num(v) as i64),
                step: get("step", row)
                    .map(num)
                    .filter(|step| *step >= 0.0)
                    .map(|step| step as i64),
                operation: operation.unwrap_or_default().to_string(),
                peer: peer.unwrap_or_default().to_string(),
                replay: optional("replay_id", row).unwrap_or_default().to_string(),
                comm_kernel: get("comm_kernel", row).is_none_or(|v| num(v) == 1.0),
            }
        })
        .collect()
}

/// One instance per collective call on each rank (the page's `instancesOf`):
/// consecutive events of one PG with the same step, collective, sequence,
/// p2p direction and peer, and graph replay, e.g. a copy-engine
/// collective's copies, merge into one call. A call with a communication
/// kernel spans its kernels; one made only of copies spans all of them.
/// Events without a sequence, and p2p send/recv, stay single: c10d gives
/// every send/recv sequence 0, so back-to-back sends to one peer would merge.
fn instances(events: &[Event]) -> Vec<Event> {
    let mut order: Vec<usize> = (0..events.len()).collect();
    order.sort_by(|&a, &b| {
        events[a]
            .rank
            .cmp(&events[b].rank)
            .then(events[a].ts.total_cmp(&events[b].ts))
    });
    struct Call {
        first: Event,
        kernels: Option<(f64, f64)>,
        all: (f64, f64),
    }
    let mut out: Vec<Call> = Vec::new();
    let mut open: HashMap<(i64, &str), (String, usize)> = HashMap::new();
    for i in order {
        let ev = &events[i];
        let key = ev.sequence.as_ref().filter(|_| !ev.p2p).map(|seq| {
            format!(
                "{:?}\0{}\0{seq}\0{}\0{}\0{}",
                ev.step, ev.collective, ev.operation, ev.peer, ev.replay
            )
        });
        let span = (ev.ts, ev.ts + ev.dur);
        if let Some(key) = &key {
            if let Some((last_key, idx)) = open.get(&(ev.rank, ev.pg.as_str())) {
                if last_key == key {
                    let call = &mut out[*idx];
                    if ev.comm_kernel {
                        let k = call.kernels.get_or_insert(span);
                        *k = (k.0.min(span.0), k.1.max(span.1));
                    }
                    call.all = (call.all.0.min(span.0), call.all.1.max(span.1));
                    continue;
                }
            }
        }
        open.insert(
            (ev.rank, ev.pg.as_str()),
            (key.unwrap_or_default(), out.len()),
        );
        out.push(Call {
            first: ev.clone(),
            kernels: ev.comm_kernel.then_some(span),
            all: span,
        });
    }
    out.into_iter()
        .map(|call| {
            let (a, b) = call.kernels.unwrap_or(call.all);
            Event {
                ts: a,
                dur: b - a,
                ..call.first
            }
        })
        .collect()
}

struct Cohort {
    seq: String,
    step: Option<i64>,
    matched: bool,
    members: Vec<usize>,
    ear: usize,
    lat: usize,
    quick: usize,
}

/// Calls that pair across ranks. Calls without a sequence and p2p calls
/// join none: only their position would pair them, and one call missing
/// from a rank's capture would silently mispair every later one.
fn cohorts(events: &[Event]) -> Vec<Cohort> {
    let mut rank_order = Vec::new();
    let mut per_rank: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, e) in events.iter().enumerate() {
        if e.sequence.is_none() || e.p2p {
            continue;
        }
        per_rank
            .entry(e.rank)
            .or_insert_with(|| {
                rank_order.push(e.rank);
                Vec::new()
            })
            .push(i);
    }
    let mut by_key: HashMap<String, usize> = HashMap::new();
    let mut list: Vec<Cohort> = Vec::new();
    for rank in rank_order {
        let evs = per_rank
            .get_mut(&rank)
            .expect("should have an entry for every rank in rank_order");
        evs.sort_by(|a, b| events[*a].ts.total_cmp(&events[*b].ts));
        // Occurrence within the key: CUDA-graph replays repeat a sequence.
        let mut occurrences: HashMap<String, usize> = HashMap::new();
        for &i in evs.iter() {
            let ev = &events[i];
            let seq = ev.sequence.as_deref().unwrap_or_default();
            let base = format!(
                "{}\0{}\0{}\0{seq}",
                ev.pg,
                ev.step.map_or("?".to_string(), |s| s.to_string()),
                ev.collective
            );
            let occurrence = occurrences.entry(base.clone()).or_default();
            let key = format!("{base}\0{occurrence}");
            *occurrence += 1;
            let index = *by_key.entry(key).or_insert_with(|| {
                list.push(Cohort {
                    seq: seq
                        .parse::<i64>()
                        .map_or(seq.to_string(), |n| n.to_string()),
                    step: ev.step,
                    matched: true,
                    members: Vec::new(),
                    ear: i,
                    lat: i,
                    quick: i,
                });
                list.len() - 1
            });
            let c = &mut list[index];
            if match_name(&events[c.ear]) != match_name(ev) {
                c.matched = false;
            }
            if c.step != ev.step {
                c.step = None;
            }
            c.members.push(i);
            if ev.ts < events[c.ear].ts {
                c.ear = i;
            }
            if ev.ts > events[c.lat].ts {
                c.lat = i;
            }
            if ev.dur < events[c.quick].dur {
                c.quick = i;
            }
        }
    }
    list
}

fn match_name(e: &Event) -> &str {
    if e.collective.is_empty() {
        &e.original_name
    } else {
        &e.collective
    }
}

fn cohort_complete(c: &Cohort, events: &[Event]) -> bool {
    if c.members.len() < 2 || !c.matched {
        return false;
    }
    let Some(expected) = c
        .members
        .iter()
        .map(|&i| events[i].group_size)
        .find(|size| *size > 0)
    else {
        return true;
    };
    let ranks: HashSet<i64> = c.members.iter().map(|&i| events[i].rank).collect();
    c.members.len() as i64 == expected
        && ranks.len() as i64 == expected
        && c.members.iter().all(|&i| {
            let size = events[i].group_size;
            size < 0 || size == expected
        })
}

fn payload_bytes(members: &[usize], events: &[Event]) -> f64 {
    let sample = members
        .iter()
        .map(|&i| &events[i])
        .find(|e| e.in_n >= 0.0 || e.out_n >= 0.0)
        .unwrap_or(&events[members[0]]);
    let width = match sample.dtype.to_ascii_lowercase().as_str() {
        "bool" | "byte" | "char" | "uint8" | "int8" => 1.0,
        "half" | "float16" | "bfloat16" | "short" | "int16" => 2.0,
        "float" | "float32" | "int" | "int32" => 4.0,
        "long" | "int64" | "double" | "float64" => 8.0,
        _ => 0.0,
    };
    sample.in_n.max(sample.out_n).max(0.0) * width
}

/// How a cohort's bandwidth is reported, by collective family (the page's
/// `bwKind`): whether algbw (full-buffer payload / time, nccl-tests' convention)
/// means anything, and the nccl-tests factor turning algbw into busbw (data over
/// the links per rank). Neither holds where ranks move different amounts: an
/// all-to-all with uneven splits, or rooted gather/scatter.
fn bw_kind(members: &[usize], events: &[Event]) -> (bool, Option<f64>) {
    let first = &events[members[0]];
    let name: String = first
        .collective
        .chars()
        .filter(|c| *c != '_')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let n = members
        .iter()
        .map(|&i| events[i].group_size)
        .find(|&size| size > 0)
        .unwrap_or(members.len() as i64) as f64;
    let ring = (n - 1.0) / n;
    if name.contains("allreduce") {
        return (true, Some(2.0 * ring));
    }
    if name.contains("allgather") || name.contains("reducescatter") {
        return (true, Some(ring));
    }
    if name.contains("alltoall") {
        let even = members.iter().all(|&i| {
            let e = &events[i];
            e.in_n >= 0.0 && e.in_n == e.out_n && e.in_n == first.in_n
        });
        return if even {
            (true, Some(ring))
        } else {
            (false, None)
        };
    }
    if name.contains("gather") || name.contains("scatter") {
        return (false, None);
    }
    if name.contains("broadcast") || name.contains("reduce") || name == "send" || name == "recv" {
        return (true, Some(1.0));
    }
    (true, None)
}

fn display_name(e: &Event) -> &str {
    let c = e.collective.as_str();
    if c.is_empty() {
        return &e.original_name;
    }
    c.strip_suffix("_into_tensor")
        .or_else(|| c.strip_suffix("_tensor"))
        .unwrap_or(c)
}

/// An axis as the page titles it (`prettyAxis` in static_page.html): `_`
/// becomes a space and `+` a spaced plus; parallelism abbreviations are
/// upper-cased and other words capitalized.
fn pretty_axis(axis: &str) -> String {
    const UPPER: [&str; 7] = ["cp", "dp", "ep", "fsdp", "hsdp", "pp", "tp"];
    let mut out = String::new();
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if UPPER.contains(&word.to_lowercase().as_str()) {
            out.push_str(&word.to_uppercase());
        } else {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                out.extend(first.to_uppercase());
                out.push_str(chars.as_str());
            }
        }
        word.clear();
    };
    for c in axis.chars() {
        match c {
            '_' => {
                flush(&mut word, &mut out);
                out.push(' ');
            }
            '+' => {
                flush(&mut word, &mut out);
                out.push_str(" + ");
            }
            c => word.push(c),
        }
    }
    flush(&mut word, &mut out);
    out
}

/// An integer with en-US digit grouping (the page's `loc`).
fn en_us(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Each group's label as the page shows it in the group picker and the PG
/// calculator (`installGroupLabels` in static_page.html), by per-PG stem:
/// the axis title, the first eight ranks, and ` · i/n` when several groups
/// read the same. The report names groups by these labels, so a name copied
/// from it finds the group on the page.
fn page_group_labels(data: &RunData, stems: &HashSet<&str>) -> HashMap<String, String> {
    let Some(map) = data
        .pg_map_json
        .as_ref()
        .and_then(serde_json::Value::as_object)
    else {
        return HashMap::new();
    };
    let info = data
        .metadata_json
        .as_ref()
        .and_then(|m| m.get("pg_info"))
        .and_then(|p| p.get("groups"));
    // The page walks pg_map in JavaScript object order: integer-like keys
    // ascending, then the others in the file's (sorted) order.
    let array_index = |k: &str| k.parse::<u32>().ok().filter(|n| n.to_string() == k);
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort_by(|a, b| match (array_index(a), array_index(b)) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.cmp(b),
    });
    let mut entries: Vec<(String, String)> = Vec::new();
    for key in keys {
        let v = &map[key.as_str()];
        let alias = info
            .and_then(|g| g.get(key.as_str()))
            .and_then(|g| g.get("alias"))
            .and_then(serde_json::Value::as_str)
            .filter(|a| !a.is_empty())
            .unwrap_or(key);
        let ids = v.get("pg_ids").and_then(serde_json::Value::as_array);
        let groups = v.get("pg_ranks_map").and_then(serde_json::Value::as_array);
        for (i, ranks) in groups.into_iter().flatten().enumerate() {
            let gid = ids
                .and_then(|ids| ids.get(i))
                .map(|id| id.as_str().map_or_else(|| id.to_string(), str::to_string))
                .unwrap_or_else(|| i.to_string());
            let Some(stem) = [format!("{alias}_{gid}"), format!("{key}_{gid}")]
                .into_iter()
                .find(|stem| stems.contains(stem.as_str()))
            else {
                continue;
            };
            let ranks: Vec<String> = ranks
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| r.to_string())
                .collect();
            let more = if ranks.len() > 8 {
                format!(", … +{}", en_us(ranks.len() - 8))
            } else {
                String::new()
            };
            let shown = ranks.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
            entries.push((stem, format!("{} (r {shown}{more})", pretty_axis(alias))));
        }
    }
    let mut totals: HashMap<&str, usize> = HashMap::new();
    for (_, base) in &entries {
        *totals.entry(base).or_default() += 1;
    }
    let mut seen: HashMap<&str, usize> = HashMap::new();
    let mut labels = HashMap::new();
    for (stem, base) in &entries {
        let ordinal = seen.entry(base).or_default();
        *ordinal += 1;
        let total = totals[base.as_str()];
        let label = if total > 1 {
            format!("{base} · {ordinal}/{total}")
        } else {
            base.clone()
        };
        labels.insert(stem.clone(), label);
    }
    labels
}

/// The parallelism axis of a per-PG stem (`tp_<hash>`, `pp_p2p_round_0_<hash>-r4`):
/// the longest axis name in the PG map that prefixes it.
fn axis_of<'a>(stem: &str, axes: &'a [String]) -> &'a str {
    axes.iter()
        .filter(|a| {
            stem.strip_prefix(a.as_str())
                .is_some_and(|rest| rest.starts_with('_'))
        })
        .max_by_key(|a| a.len())
        .map_or("other", String::as_str)
}

/// One decimal, without a trailing ".0".
fn d1(value: f64) -> String {
    let s = format!("{value:.1}");
    s.strip_suffix(".0").map_or(s.clone(), str::to_string)
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.retain(|v| v.is_finite());
    values.sort_by(f64::total_cmp);
    values
}

/// Nearest-rank percentile of sorted values.
fn pctl(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[i]
}

/// Bandwidth distribution: the slow tail is the low end.
fn bw_dist(values: Vec<f64>) -> serde_json::Value {
    let s = sorted(values);
    if s.is_empty() {
        return serde_json::Value::Null;
    }
    json!({
        "n": s.len(),
        "min": s[0],
        "p10": pctl(&s, 0.1),
        "p50": pctl(&s, 0.5),
        "max": s[s.len() - 1],
    })
}

fn dist(values: Vec<f64>) -> serde_json::Value {
    let s = sorted(values);
    if s.is_empty() {
        return serde_json::Value::Null;
    }
    json!({
        "n": s.len(),
        "p50": pctl(&s, 0.5),
        "p90": pctl(&s, 0.9),
        "p99": pctl(&s, 0.99),
        "max": s[s.len() - 1],
    })
}

#[derive(Clone)]
struct WorstCohort {
    group: usize,
    step: Option<i64>,
    seq: String,
    op: String,
    min_dur: f64,
    spread: f64,
    first: i64,
    last: i64,
}

#[derive(Default, Clone, Copy)]
struct Arrivals {
    cohorts: u32,
    last: u32,
    /// Expected last-arrival count by chance: sum of 1/n over cohorts.
    expected: f64,
    /// Variance of the last-arrival count by chance: sum of p(1-p).
    variance: f64,
    /// Sum over cohorts of (this rank's start - earliest start), us.
    lag: f64,
}

impl Arrivals {
    fn add(&mut self, other: &Arrivals) {
        self.cohorts += other.cohorts;
        self.last += other.last;
        self.expected += other.expected;
        self.variance += other.variance;
        self.lag += other.lag;
    }

    fn z(&self) -> f64 {
        if self.variance > 0.0 {
            (self.last as f64 - self.expected) / self.variance.sqrt()
        } else {
            0.0
        }
    }
}

/// What one process group contributes to the report.
struct GroupSummary {
    stem: String,
    axis: String,
    ranks: Vec<i64>,
    kernels: usize,
    no_seq: usize,
    p2p: usize,
    /// Per rank, kernel time of the calls left out of matching (no sequence
    /// or p2p), including time waiting for the peer.
    unmatched_us: Vec<(i64, f64)>,
    complete: usize,
    incomplete: usize,
    spreads: Vec<f64>,
    /// (bus bandwidth GB/s, payload bytes) of complete cohorts, by collective.
    busbw: BTreeMap<String, Vec<(f64, f64)>>,
    /// Complete cohorts without a bandwidth because ranks move different
    /// amounts (uneven all-to-all, rooted gather/scatter), by collective.
    no_bw: BTreeMap<String, usize>,
    worst: Vec<WorstCohort>,
    arrivals: Vec<(i64, Arrivals)>,
}

fn summarize_group(index: usize, stem: &str, axis: &str, raw: &[Event]) -> GroupSummary {
    let calls = instances(raw);
    let events = &calls[..];
    let ranks: Vec<i64> = events
        .iter()
        .map(|e| e.rank)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let list = cohorts(events);
    let mut summary = GroupSummary {
        stem: stem.to_string(),
        axis: axis.to_string(),
        ranks,
        kernels: raw.len(),
        no_seq: events.iter().filter(|e| e.sequence.is_none()).count(),
        p2p: events
            .iter()
            .filter(|e| e.p2p && e.sequence.is_some())
            .count(),
        unmatched_us: {
            let mut by_rank: BTreeMap<i64, f64> = BTreeMap::new();
            for e in raw.iter().filter(|e| e.sequence.is_none() || e.p2p) {
                *by_rank.entry(e.rank).or_default() += e.dur;
            }
            by_rank.into_iter().collect()
        },
        complete: 0,
        incomplete: 0,
        spreads: Vec::new(),
        busbw: BTreeMap::new(),
        no_bw: BTreeMap::new(),
        worst: Vec::new(),
        arrivals: Vec::new(),
    };
    let mut arrivals: BTreeMap<i64, Arrivals> = BTreeMap::new();
    for c in &list {
        if !cohort_complete(c, events) {
            summary.incomplete += 1;
            continue;
        }
        summary.complete += 1;
        let (ear, lat, quick) = (&events[c.ear], &events[c.lat], &events[c.quick]);
        let spread = lat.ts - ear.ts;
        summary.spreads.push(spread);
        let payload = payload_bytes(&c.members, events);
        let kind = bw_kind(&c.members, events);
        if !kind.0 {
            *summary
                .no_bw
                .entry(display_name(ear).to_string())
                .or_default() += 1;
        }
        if let (true, Some(factor)) = kind {
            if quick.dur > 0.0 && payload > 0.0 {
                summary
                    .busbw
                    .entry(display_name(ear).to_string())
                    .or_default()
                    .push((payload / quick.dur / 1000.0 * factor, payload));
            }
        }
        summary.worst.push(WorstCohort {
            group: index,
            step: c.step,
            seq: c.seq.clone(),
            op: display_name(ear).to_string(),
            min_dur: quick.dur,
            spread,
            first: ear.rank,
            last: lat.rank,
        });
        let n = c.members.len() as f64;
        let p = 1.0 / n;
        for &i in &c.members {
            let a = arrivals.entry(events[i].rank).or_default();
            a.cohorts += 1;
            a.last += u32::from(i == c.lat);
            a.expected += p;
            a.variance += p * (1.0 - p);
            a.lag += events[i].ts - ear.ts;
        }
    }
    summary.worst.sort_by(|a, b| b.spread.total_cmp(&a.spread));
    summary.worst.truncate(WORST_PER_GROUP);
    summary.arrivals = arrivals.into_iter().collect();
    summary
}

/// One per-rank metric: its median over ranks and the `EXTREMES` lowest and
/// highest ranks.
type RankValues = Vec<(i64, f64)>;

fn extremes(values: &[(i64, f64)]) -> Option<(f64, RankValues, RankValues)> {
    let mut v: Vec<(i64, f64)> = values
        .iter()
        .copied()
        .filter(|(_, x)| x.is_finite())
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let median = v[(v.len() - 1) / 2].1;
    let low = v.iter().take(EXTREMES).copied().collect();
    let high = v.iter().rev().take(EXTREMES).copied().collect();
    Some((median, low, high))
}

/// Rank -> host from `rank_map.json` (`runs` of [start, len, host index, _]).
fn rank_hosts(rank_map_json: Option<&str>) -> HashMap<i64, String> {
    let Some(doc) = rank_map_json.and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok())
    else {
        return HashMap::new();
    };
    let hosts: Vec<&str> = doc["hosts"]
        .as_array()
        .map(|a| a.iter().filter_map(|h| h.as_str()).collect())
        .unwrap_or_default();
    let mut out = HashMap::new();
    for run in doc["runs"].as_array().into_iter().flatten() {
        let (Some(start), Some(len), Some(h)) = (run[0].as_i64(), run[1].as_i64(), run[2].as_i64())
        else {
            continue;
        };
        if let Some(host) = usize::try_from(h).ok().and_then(|h| hosts.get(h)) {
            for r in start..start + len {
                out.insert(r, host.to_string());
            }
        }
    }
    out
}

fn write(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

/// Write one run's structured report into `dest/<stem>/`.
pub(crate) fn write_structured_report(data: &RunData, stem: &str, dest: &Path) -> Result<()> {
    let dir = dest.join(stem);
    // Start empty: files of an earlier report version would otherwise remain.
    if dir.exists() {
        std::fs::remove_dir_all(&dir).with_context(|| format!("clear {}", dir.display()))?;
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

    let axes: Vec<String> = data
        .pg_map_json
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    let members: Vec<&(String, u64, std::path::PathBuf)> = data
        .per_pg_all
        .iter()
        .filter(|(s, _, _)| !comms_viz_core::analysis::is_diagnostics_member(s))
        .collect();
    let groups: Vec<GroupSummary> = members
        .par_iter()
        .enumerate()
        .map(|(i, (stem, _, path))| -> Result<GroupSummary> {
            let payload =
                std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
            let events = events_from_table(&table_from_member(&payload));
            Ok(summarize_group(i, stem, axis_of(stem, &axes), &events))
        })
        .collect::<Result<_>>()?;
    let stems: HashSet<&str> = data.per_pg_all.iter().map(|(s, _, _)| s.as_str()).collect();
    let labels = page_group_labels(data, &stems);
    let aliases: Vec<String> = groups
        .iter()
        .map(|g| {
            labels
                .get(&g.stem)
                .cloned()
                .unwrap_or_else(|| g.stem.clone())
        })
        .collect();
    let diag = crate::diagnostics::read(&data.per_pg_all)?;
    let hosts = rank_hosts(data.rank_map_json.as_deref());
    let summary = build_summary(data, &groups, &aliases, &diag, &hosts);
    write(
        &dir.join("summary.json"),
        &serde_json::to_string_pretty(&summary)?,
    )?;
    write(&dir.join("README.md"), &readme(&summary))?;
    eprintln!("comms_viz: wrote structured report {}", dir.display());
    Ok(())
}

fn build_summary(
    data: &RunData,
    groups: &[GroupSummary],
    aliases: &[String],
    diag: &Diagnostics,
    hosts: &HashMap<i64, String>,
) -> serde_json::Value {
    let host_of = |rank: i64| hosts.get(&rank).cloned();

    // Axes: spread and bandwidth distributions, worst and slowest groups.
    let mut by_axis: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        by_axis.entry(&g.axis).or_default().push(i);
    }
    let mut axis_docs = Vec::new();
    for (axis, idx) in &by_axis {
        let gs: Vec<&GroupSummary> = idx.iter().map(|&i| &groups[i]).collect();
        let sizes = sorted(gs.iter().map(|g| g.ranks.len() as f64).collect());
        let mut busbw_all: BTreeMap<&str, Vec<(f64, f64)>> = BTreeMap::new();
        for g in &gs {
            for (op, v) in &g.busbw {
                busbw_all.entry(op).or_default().extend(v);
            }
        }
        let axis_median: BTreeMap<&str, f64> = busbw_all
            .iter()
            .map(|(op, v)| (*op, pctl(&sorted(v.iter().map(|x| x.0).collect()), 0.5)))
            .collect();
        // Each group's median bandwidth for its most frequent collective,
        // lowest first, against the axis median for that collective.
        let mut lowest: Vec<(usize, String, f64, f64)> = idx
            .iter()
            .filter_map(|&i| {
                let (op, v) = groups[i].busbw.iter().max_by_key(|(_, v)| v.len())?;
                let med = pctl(&sorted(v.iter().map(|x| x.0).collect()), 0.5);
                Some((i, op.clone(), med, med / axis_median[op.as_str()]))
            })
            .collect();
        lowest.sort_by(|a, b| a.3.total_cmp(&b.3).then(a.0.cmp(&b.0)));
        // The worst collective of each group, so one bad group doesn't fill
        // the table with its consecutive sequence numbers.
        let mut worst: Vec<WorstCohort> =
            gs.iter().filter_map(|g| g.worst.first().cloned()).collect();
        worst.sort_by(|a, b| b.spread.total_cmp(&a.spread));
        let mut median_spread: Vec<(usize, f64)> = idx
            .iter()
            .filter(|&&i| !groups[i].spreads.is_empty())
            .map(|&i| (i, pctl(&sorted(groups[i].spreads.clone()), 0.5)))
            .filter(|&(_, m)| m > 0.0)
            .collect();
        let mut unmatched_by_rank: HashMap<i64, f64> = HashMap::new();
        for g in &gs {
            for (r, us) in &g.unmatched_us {
                *unmatched_by_rank.entry(*r).or_default() += us;
            }
        }
        // Rank stride within a group, when the first group's ranks are evenly
        // spaced: with the group size it gives the axis' place in the layout.
        let stride = gs.first().and_then(|g| match g.ranks.as_slice() {
            [a, b, ..] if g.ranks.windows(2).all(|w| w[1] - w[0] == b - a) => Some(b - a),
            _ => None,
        });
        median_spread.sort_by(|a, b| b.1.total_cmp(&a.1));
        let group_ref = |i: usize| {
            json!({
                "group": aliases[i],
                "ranks": ranks_short(&groups[i].ranks, 8),
                "pg": groups[i].stem,
            })
        };
        let mut no_bw: BTreeMap<&str, usize> = BTreeMap::new();
        for g in &gs {
            for (op, n) in &g.no_bw {
                *no_bw.entry(op).or_default() += n;
            }
        }
        axis_docs.push(json!({
            "axis": axis,
            "groups": gs.len(),
            "group_size": {"min": sizes.first(), "max": sizes.last()},
            "rank_stride": stride,
            "kernels": gs.iter().map(|g| g.kernels).sum::<usize>(),
            "matched_collectives": gs.iter().map(|g| g.complete).sum::<usize>(),
            "incomplete_collectives": gs.iter().map(|g| g.incomplete).sum::<usize>(),
            "unmatchable_kernels": {
                "no_sequence": gs.iter().map(|g| g.no_seq).sum::<usize>(),
                "p2p": gs.iter().map(|g| g.p2p).sum::<usize>(),
                "kernel_time_per_rank_us": dist(unmatched_by_rank.values().copied().collect()),
            },
            "start_spread_us": dist(gs.iter().flat_map(|g| g.spreads.iter().copied()).collect()),
            "busbw_gbps": busbw_all.into_iter().map(|(op, v)| {
                let mut doc = bw_dist(v.iter().map(|x| x.0).collect());
                doc["payload_bytes_p50"] = json!(pctl(&sorted(v.iter().map(|x| x.1).collect()), 0.5));
                (op.to_string(), doc)
            }).collect::<serde_json::Map<_, _>>(),
            "bandwidth_not_reported": no_bw,
            "worst_collectives": worst.iter().take(TOP).map(|w| json!({
                "group": aliases[w.group],
                "step": w.step,
                "seq": w.seq,
                "op": w.op,
                "spread_us": w.spread,
                "min_dur_us": w.min_dur,
                "first_rank": w.first,
                "last_rank": w.last,
            })).collect::<Vec<_>>(),
            "highest_median_spread_groups": median_spread.iter().take(TOP).map(|&(i, m)| {
                let mut doc = group_ref(i);
                doc["median_spread_us"] = json!(m);
                doc
            }).collect::<Vec<_>>(),
            "lowest_busbw_groups": lowest.iter().take(TOP).map(|(i, op, med, share)| {
                let mut doc = group_ref(*i);
                doc["op"] = json!(op);
                doc["median_busbw_gbps"] = json!(med);
                doc["share_of_axis_median"] = json!(share);
                doc
            }).collect::<Vec<_>>(),
        }));
    }

    let axes_total = axis_docs.len();
    axis_docs.sort_by_key(|a| std::cmp::Reverse(a["kernels"].as_u64().unwrap_or(0)));
    axis_docs.truncate(MAX_AXES);

    // Stragglers: last-arrival counts against chance, over every group.
    let mut arrivals: HashMap<i64, Arrivals> = HashMap::new();
    let mut arrivals_by_axis: HashMap<(i64, &str), Arrivals> = HashMap::new();
    for g in groups {
        for (rank, a) in &g.arrivals {
            arrivals.entry(*rank).or_default().add(a);
            arrivals_by_axis.entry((*rank, &g.axis)).or_default().add(a);
        }
    }
    let mut stragglers: Vec<(i64, Arrivals)> = arrivals.iter().map(|(r, a)| (*r, *a)).collect();
    stragglers.sort_by(|a, b| b.1.z().total_cmp(&a.1.z()).then(a.0.cmp(&b.0)));
    // Per-rank metrics: utilization, kernel time, GPU counters.
    let window_of: HashMap<i64, f64> = diag.activity.iter().map(|a| (a.rank, a.window)).collect();
    let share = |a: &Activity, v: f64| {
        if a.window > 0.0 {
            100.0 * v / a.window
        } else {
            f64::NAN
        }
    };
    let mut metrics: Vec<(String, Vec<(i64, f64)>)> = vec![
        (
            "GPU busy %".into(),
            diag.activity
                .iter()
                .map(|a| (a.rank, share(a, a.gpu)))
                .collect(),
        ),
        (
            "compute kernels %".into(),
            diag.activity
                .iter()
                .map(|a| (a.rank, share(a, a.compute)))
                .collect(),
        ),
        (
            "CPU active %".into(),
            diag.activity
                .iter()
                .map(|a| (a.rank, share(a, a.cpu)))
                .collect(),
        ),
    ];
    let breakdown = table_from_csv(&data.breakdown_csv);
    if let (Some(ranks), Some(comm), Some(comp)) = (
        breakdown.col("rank"),
        breakdown.col("comm_total"),
        breakdown.col("comp_total"),
    ) {
        let rows: Vec<(i64, f64, f64)> = ranks
            .iter()
            .zip(comm)
            .zip(comp)
            .map(|((r, c), p)| (num(r) as i64, num(c), num(p)))
            .collect();
        metrics.push((
            "communication kernel time (us)".into(),
            rows.iter().map(|(r, c, _)| (*r, *c)).collect(),
        ));
        metrics.push((
            "compute kernel time (us)".into(),
            rows.iter().map(|(r, _, p)| (*r, *p)).collect(),
        ));
    }
    let mut counter_metrics: BTreeMap<String, Vec<(i64, f64)>> = BTreeMap::new();
    for c in &diag.counters {
        for stat in ["mean", "max"] {
            counter_metrics
                .entry(format!("{} {stat}", c.metric))
                .or_default()
                .push((c.rank, c.get(stat)));
        }
    }
    metrics.extend(counter_metrics);
    let rank_doc =
        |(rank, v): &(i64, f64)| json!({"rank": rank, "host": host_of(*rank), "value": v});
    let extreme_docs: Vec<serde_json::Value> = metrics
        .iter()
        .filter_map(|(name, values)| {
            let (median, low, high) = extremes(values)?;
            Some(json!({
                "metric": name,
                "median": median,
                "lowest": low.iter().map(rank_doc).collect::<Vec<_>>(),
                "highest": high.iter().map(rank_doc).collect::<Vec<_>>(),
            }))
        })
        .collect();
    let ranks_per_host: HashMap<&str, usize> = hosts.values().fold(HashMap::new(), |mut m, h| {
        *m.entry(h.as_str()).or_default() += 1;
        m
    });

    let straggler_docs: Vec<serde_json::Value> = stragglers
        .iter()
        .take(TOP_RANKS)
        .map(|(rank, a)| {
            let per_axis: serde_json::Map<String, serde_json::Value> = by_axis
                .keys()
                .filter_map(|axis| {
                    arrivals_by_axis
                        .get(&(*rank, *axis))
                        .map(|x| (axis.to_string(), json!(format!("{}/{}", x.last, x.cohorts))))
                })
                .collect();
            json!({
                "rank": rank,
                "host": host_of(*rank),
                "last": a.last,
                "cohorts": a.cohorts,
                "expected_last": a.expected,
                "z": a.z(),
                "mean_lag_us": if a.cohorts > 0 { a.lag / a.cohorts as f64 } else { 0.0 },
                "last_by_axis": per_axis,
            })
        })
        .collect();

    // Data quality.
    let nccl: u64 = diag.activity.iter().map(|a| a.nccl_kernels).sum();
    let nccl_bare: u64 = diag.activity.iter().map(|a| a.nccl_unannotated).sum();
    let other_bare: u64 = diag.activity.iter().map(|a| a.other_comm_unannotated).sum();
    let nccl_us: f64 = diag.activity.iter().map(|a| a.nccl_us).sum();
    let nccl_bare_us: f64 = diag.activity.iter().map(|a| a.nccl_unannotated_us).sum();
    let mut lost: Vec<&Activity> = diag
        .activity
        .iter()
        .filter(|a| a.loses_pg_metadata())
        .collect();
    let lost_share = |a: &Activity| a.nccl_unannotated_us / a.nccl_us.max(1e-9);
    lost.sort_by(|a, b| {
        lost_share(b)
            .total_cmp(&lost_share(a))
            .then(a.rank.cmp(&b.rank))
    });
    // The ProfilerSteps the capture covers, from the ranks' step markers.
    let step_ranges: BTreeSet<(i64, i64)> = diag.activity.iter().filter_map(|a| a.steps).collect();
    let steps_text = match step_ranges.len() {
        0 => "step unknown".to_string(),
        1 => format!(
            "Profiler {}",
            crate::diagnostics::step_label(*step_ranges.iter().next().expect("one range"))
        ),
        _ => "different steps per rank (see warnings)".to_string(),
    };
    let unmatched_dist = {
        let mut by_rank: HashMap<i64, f64> = HashMap::new();
        for g in groups {
            for (r, us) in &g.unmatched_us {
                *by_rank.entry(*r).or_default() += us;
            }
        }
        dist(by_rank.into_values().collect())
    };
    let utilization = |f: &dyn Fn(&Activity) -> f64| dist(diag.activity.iter().map(f).collect());
    let run_name = data
        .run_json
        .get("run_name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    json!({
        "run": {
            "name": run_name,
            "run_id": data.run_id,
            "steps": steps_text,
            "ranks": data.status_json["ranks"]["processed"],
            "world_size": data.status_json["ranks"]["total"],
            "hosts": if ranks_per_host.is_empty() { json!(null) } else { json!(ranks_per_host.len()) },
            "process_groups": groups.len(),
        },
        "warnings": data.warnings,
        "data_quality": {
            "nccl_kernels": nccl,
            "nccl_kernels_without_pg_metadata": nccl_bare,
            "nccl_time_without_pg_metadata_pct": if nccl_us > 0.0 { 100.0 * nccl_bare_us / nccl_us } else { 0.0 },
            "ranks_losing_pg_metadata": lost.len(),
            "worst_ranks_losing_pg_metadata": lost.iter().take(TOP).map(|a| json!({
                "rank": a.rank, "host": host_of(a.rank), "unannotated": a.nccl_unannotated, "nccl_kernels": a.nccl_kernels,
                "unannotated_time_pct": 100.0 * lost_share(a),
            })).collect::<Vec<_>>(),
            "other_comm_kernels_without_pg_metadata": other_bare,
            "unmatchable_kernels": {
                "no_sequence": groups.iter().map(|g| g.no_seq).sum::<usize>(),
                "p2p": groups.iter().map(|g| g.p2p).sum::<usize>(),
                "kernel_time_per_rank_us": unmatched_dist,
            },
            "incomplete_collectives": groups.iter().map(|g| g.incomplete).sum::<usize>(),
        },
        "utilization_pct": {
            "gpu_busy": utilization(&|a| share(a, a.gpu)),
            "compute": utilization(&|a| share(a, a.compute)),
            "cpu_active": utilization(&|a| share(a, a.cpu)),
            "window_us": dist(window_of.values().copied().collect()),
        },
        "stragglers": straggler_docs,
        "per_rank_extremes": extreme_docs,
        "axes": axis_docs,
        "axes_total": axes_total,
    })
}

/// A percentage, with small nonzero values shown as "<0.1" instead of "0".
fn small_pct(value: &serde_json::Value) -> String {
    match value.as_f64() {
        Some(v) if v > 0.0 && v < 0.1 => "<0.1".to_string(),
        _ => f(value),
    }
}

fn f(value: &serde_json::Value) -> String {
    if value.is_null() {
        return "-".to_string();
    }
    match value.as_f64().map(|v| if v == 0.0 { 0.0 } else { v }) {
        Some(v) if v.abs() >= 100.0 => format!("{v:.0}"),
        Some(v) => d1(v),
        None => value
            .as_str()
            .map_or_else(|| value.to_string(), str::to_string),
    }
}

fn readme(s: &serde_json::Value) -> String {
    let mut out = String::new();
    let run = &s["run"];
    let _ = writeln!(out, "# comms_viz summary: {}\n", f(&run["name"]));
    let _ = writeln!(
        out,
        "run id `{}`, {}, {} of {} ranks on {} hosts, {} process groups. Times are us, \
         bandwidth GB/s (nccl-tests busbw). Every table is a top-{TOP} or a distribution; \
         `summary.json` has the same content. Rank lists use `a-b` for ranges and `a-b:s` \
         for every s-th rank.\n",
        f(&run["run_id"]),
        f(&run["steps"]),
        f(&run["ranks"]),
        f(&run["world_size"]),
        f(&run["hosts"]),
        f(&run["process_groups"])
    );
    for w in s["warnings"].as_array().into_iter().flatten() {
        let _ = writeln!(out, "- **warning:** {}", f(w));
    }

    let q = &s["data_quality"];
    let _ = writeln!(out, "\n## Data quality\n");
    let _ = writeln!(
        out,
        "- NCCL kernels without process-group metadata: {} of {}, {}% of NCCL kernel time \
         ({} ranks). They count as communication time but are in no group or \
         statistic below.",
        f(&q["nccl_kernels_without_pg_metadata"]),
        f(&q["nccl_kernels"]),
        small_pct(&q["nccl_time_without_pg_metadata_pct"]),
        f(&q["ranks_losing_pg_metadata"])
    );
    for r in q["worst_ranks_losing_pg_metadata"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let _ = writeln!(
            out,
            "  - rank {} ({}): {} of {} kernels, {}% of NCCL time",
            f(&r["rank"]),
            f(&r["host"]),
            f(&r["unannotated"]),
            f(&r["nccl_kernels"]),
            f(&r["unannotated_time_pct"])
        );
    }
    let _ = writeln!(
        out,
        "- Communication kernels outside c10d (HybridEP, DeepEP, symm_mem barriers, \
         ...), which never carry its metadata: {}",
        f(&q["other_comm_kernels_without_pg_metadata"])
    );
    let _ = writeln!(
        out,
        "- Kernels left out of cross-rank matching (only their order would pair them): {} \
         without a c10d sequence, {} p2p send/recv; per rank {} us p50, {} us max of kernel \
         time, including time waiting for the peer. Collectives missing a member: {}.",
        f(&q["unmatchable_kernels"]["no_sequence"]),
        f(&q["unmatchable_kernels"]["p2p"]),
        f(&q["unmatchable_kernels"]["kernel_time_per_rank_us"]["p50"]),
        f(&q["unmatchable_kernels"]["kernel_time_per_rank_us"]["max"]),
        f(&q["incomplete_collectives"])
    );

    let u = &s["utilization_pct"];
    let _ = writeln!(
        out,
        "\n## Utilization across ranks (% of each rank's GPU window)\n"
    );
    let _ = writeln!(
        out,
        "| metric | p50 | p90 | p99 | max |\n|---|---|---|---|---|"
    );
    for (name, key) in [
        ("GPU busy", "gpu_busy"),
        ("compute kernels", "compute"),
        ("CPU active", "cpu_active"),
        ("window (us)", "window_us"),
    ] {
        let d = &u[key];
        if d.is_null() {
            continue;
        }
        let _ = writeln!(
            out,
            "| {name} | {} | {} | {} | {} |",
            f(&d["p50"]),
            f(&d["p90"]),
            f(&d["p99"]),
            f(&d["max"])
        );
    }

    let _ = writeln!(out, "\n## Stragglers\n");
    let _ = writeln!(
        out,
        "Ranks by how often they start a matched collective last, over every group, highest \
         z first: `last / collectives` counts its matched collectives and those it started \
         last; `expected` is the count by chance (1/n per collective of n ranks), `z` the \
         excess over chance in standard deviations, `mean lag` its average start after the \
         first rank (us), `last by axis` the same counts per axis. Covers collectives with \
         process-group metadata: {}% of NCCL kernel time has none.\n",
        small_pct(&q["nccl_time_without_pg_metadata_pct"])
    );
    let _ = writeln!(
        out,
        "| rank | host | last / collectives | expected | z | mean lag | last by axis |\n|---|---|---|---|---|---|---|"
    );
    for r in s["stragglers"].as_array().into_iter().flatten() {
        let by_axis = r["last_by_axis"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| format!("{k} {}", f(v)))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "| {} | {} | {}/{} | {} | {} | {} | {by_axis} |",
            f(&r["rank"]),
            f(&r["host"]),
            f(&r["last"]),
            f(&r["cohorts"]),
            f(&r["expected_last"]),
            f(&r["z"]),
            f(&r["mean_lag_us"])
        );
    }

    let _ = writeln!(out, "\n## Per-rank extremes\n");
    let _ = writeln!(
        out,
        "Per metric, the median over ranks and the {EXTREMES} lowest and highest ranks \
         (rank: value).\n"
    );
    let ranks_text = |list: &serde_json::Value| {
        list.as_array()
            .into_iter()
            .flatten()
            .map(|r| format!("{}: {}", f(&r["rank"]), f(&r["value"])))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(
        out,
        "| metric | median | lowest | highest |\n|---|---|---|---|"
    );
    for m in s["per_rank_extremes"].as_array().into_iter().flatten() {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} |",
            f(&m["metric"]),
            f(&m["median"]),
            ranks_text(&m["lowest"]),
            ranks_text(&m["highest"])
        );
    }
    let _ = writeln!(out, "\n## Axes\n");
    let listed = s["axes"].as_array().map_or(0, Vec::len);
    let total = s["axes_total"].as_u64().unwrap_or(0) as usize;
    if total > listed {
        let _ = writeln!(
            out,
            "The {listed} axes with the most kernels of {total}; groups without a description \
             (PyTorch's default `undefined`) are each their own axis.\n"
        );
    }
    for a in s["axes"].as_array().into_iter().flatten() {
        let gs = &a["group_size"];
        let _ = writeln!(
            out,
            "### {}\n\n{} groups of {} ranks{}; {} kernels, {} matched collectives, {} missing a \
             member, {} kernels unmatched (no sequence {}, p2p {}; per rank {} us p50, {} us \
             max).\n",
            f(&a["axis"]),
            f(&a["groups"]),
            if gs["min"] == gs["max"] {
                f(&gs["min"])
            } else {
                format!("{}-{}", f(&gs["min"]), f(&gs["max"]))
            },
            if a["rank_stride"].is_null() {
                String::new()
            } else {
                format!(" at rank stride {}", f(&a["rank_stride"]))
            },
            f(&a["kernels"]),
            f(&a["matched_collectives"]),
            f(&a["incomplete_collectives"]),
            f(&json!(
                a["unmatchable_kernels"]["no_sequence"]
                    .as_u64()
                    .unwrap_or(0)
                    + a["unmatchable_kernels"]["p2p"].as_u64().unwrap_or(0)
            )),
            f(&a["unmatchable_kernels"]["no_sequence"]),
            f(&a["unmatchable_kernels"]["p2p"]),
            f(&a["unmatchable_kernels"]["kernel_time_per_rank_us"]["p50"]),
            f(&a["unmatchable_kernels"]["kernel_time_per_rank_us"]["max"])
        );
        let sp = &a["start_spread_us"];
        if !sp.is_null() {
            let _ = writeln!(
                out,
                "Start spread (last start - first start) over matched collectives: p50 {}, \
                 p90 {}, p99 {}, max {}.\n",
                f(&sp["p50"]),
                f(&sp["p90"]),
                f(&sp["p99"]),
                f(&sp["max"])
            );
        }
        if let Some(bw) = a["busbw_gbps"].as_object().filter(|m| !m.is_empty()) {
            let _ = writeln!(
                out,
                "Bus bandwidth of matched collectives:\n\n| collective | payload p50 (bytes) | busbw min | p10 | p50 | max | n |\n|---|---|---|---|---|---|---|"
            );
            for (op, d) in bw {
                let _ = writeln!(
                    out,
                    "| {op} | {} | {} | {} | {} | {} | {} |",
                    f(&d["payload_bytes_p50"]),
                    f(&d["min"]),
                    f(&d["p10"]),
                    f(&d["p50"]),
                    f(&d["max"]),
                    f(&d["n"])
                );
            }
            out.push('\n');
        }
        if let Some(no_bw) = a["bandwidth_not_reported"]
            .as_object()
            .filter(|m| !m.is_empty())
        {
            let ops = no_bw
                .iter()
                .map(|(op, n)| format!("{op} ({} collectives)", f(n)))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "No bandwidth for {ops}: ranks move different amounts (uneven all-to-all, or \
                 rooted gather/scatter), so no single payload over time describes them.\n"
            );
        }
        let lowest = a["lowest_busbw_groups"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !lowest.is_empty() {
            let _ = writeln!(
                out,
                "Groups with the lowest median busbw (for each group's most frequent \
                 collective), against the axis median for that collective:\n"
            );
            for g in &lowest {
                let _ = writeln!(
                    out,
                    "- {} (ranks {}): {} {} GB/s, {}% of axis median",
                    f(&g["group"]),
                    f(&g["ranks"]),
                    f(&g["op"]),
                    f(&g["median_busbw_gbps"]),
                    d1(g["share_of_axis_median"].as_f64().unwrap_or(0.0) * 100.0)
                );
            }
            out.push('\n');
        }
        let worst = a["worst_collectives"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !worst.is_empty() {
            let _ = writeln!(
                out,
                "Worst start spread:\n\n| group | step | seq | op | spread | min dur | first | last |\n|---|---|---|---|---|---|---|---|"
            );
            for w in &worst {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} | {} | {} |",
                    f(&w["group"]),
                    f(&w["step"]),
                    f(&w["seq"]),
                    f(&w["op"]),
                    f(&w["spread_us"]),
                    f(&w["min_dur_us"]),
                    f(&w["first_rank"]),
                    f(&w["last_rank"])
                );
            }
            out.push('\n');
        }
        let high = a["highest_median_spread_groups"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !high.is_empty() {
            let line = high
                .iter()
                .map(|g| format!("{} ({})", f(&g["group"]), f(&g["median_spread_us"])))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "Highest median start spread: {line}\n");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(rank: i64, ts: f64, seq: &str) -> Event {
        Event {
            rank,
            ts,
            dur: 10.0,
            original_name: "ncclAllReduce".into(),
            sequence: Some(seq.into()),
            pg: "0".into(),
            collective: "allreduce".into(),
            dtype: "float".into(),
            in_n: 4.0,
            out_n: 4.0,
            group_size: 2,
            step: Some(1),
            p2p: false,
            operation: String::new(),
            peer: String::new(),
            replay: String::new(),
            comm_kernel: true,
        }
    }

    #[test]
    fn bus_bandwidth_factor_by_collective() {
        let of = |collective: &str, sizes: &[(f64, f64)]| {
            let events: Vec<Event> = sizes
                .iter()
                .enumerate()
                .map(|(rank, &(in_n, out_n))| Event {
                    collective: collective.into(),
                    in_n,
                    out_n,
                    group_size: 4,
                    ..event(rank as i64, 0.0, "1")
                })
                .collect();
            bw_kind(&(0..events.len()).collect::<Vec<_>>(), &events)
        };
        let even = [(8.0, 8.0); 4];
        assert_eq!(of("allreduce", &even), (true, Some(1.5)));
        assert_eq!(of("_allgather_base", &[(2.0, 8.0); 4]), (true, Some(0.75)));
        assert_eq!(
            of("allGatherSingleImpl", &[(2.0, 8.0); 4]),
            (true, Some(0.75))
        );
        assert_eq!(of("reduce_scatter", &[(8.0, 2.0); 4]), (true, Some(0.75)));
        assert_eq!(of("all_to_all", &even), (true, Some(0.75)));
        // Uneven splits: ranks move different amounts, so no bandwidth at all.
        let uneven = [(8.0, 6.0), (8.0, 10.0), (8.0, 8.0), (8.0, 8.0)];
        assert_eq!(of("all_to_all", &uneven), (false, None));
        assert_eq!(of("gather", &even), (false, None));
        assert_eq!(of("broadcast", &even), (true, Some(1.0)));
        assert_eq!(of("symm_mem::barrier", &even), (true, None));
    }

    #[test]
    fn cohorts_match_on_sequence_not_ordinal() {
        // Rank 1 is missing seq 1, so ordinal matching would pair 1 with 2.
        let events = vec![event(0, 0.0, "1"), event(0, 20.0, "2"), event(1, 25.0, "2")];
        let list = cohorts(&events);
        let complete: Vec<_> = list
            .iter()
            .filter(|c| cohort_complete(c, &events))
            .collect();
        assert_eq!(complete.len(), 1);
        assert_eq!(complete[0].seq, "2");
        assert_eq!(events[complete[0].lat].rank, 1);
        assert_eq!(payload_bytes(&complete[0].members, &events), 16.0);
    }

    #[test]
    fn copies_of_one_call_merge_into_one_instance() {
        // A copy-engine all-gather: two copies of one call, then the next call.
        let copy = |ts| Event {
            comm_kernel: false,
            ..event(0, ts, "7")
        };
        let calls = instances(&[copy(0.0), copy(5.0), event(0, 30.0, "8")]);
        assert_eq!(calls.len(), 2);
        assert_eq!((calls[0].ts, calls[0].dur), (0.0, 15.0));
        assert_eq!(calls[1].sequence.as_deref(), Some("8"));
    }

    #[test]
    fn unsequenced_and_p2p_calls_join_no_cohort() {
        let p2p = |rank, ts| Event {
            p2p: true,
            collective: "send".into(),
            ..event(rank, ts, "0")
        };
        let bare = |rank, ts| Event {
            sequence: None,
            ..event(rank, ts, "")
        };
        let events = vec![p2p(0, 0.0), p2p(1, 5.0), bare(0, 10.0), bare(1, 12.0)];
        assert!(cohorts(&events).is_empty());
        let g = summarize_group(0, "pp_p2p_x", "pp_p2p", &events);
        assert_eq!((g.complete, g.p2p, g.no_seq), (0, 2, 2));
        assert!(g.arrivals.is_empty());
    }

    #[test]
    fn straggler_score_counts_last_arrivals_against_chance() {
        // Rank 1 starts last in all 10 two-rank collectives.
        let events: Vec<Event> = (0..10)
            .flat_map(|s| {
                let seq = s.to_string();
                [
                    event(0, s as f64 * 100.0, &seq),
                    event(1, s as f64 * 100.0 + 30.0, &seq),
                ]
            })
            .collect();
        let g = summarize_group(0, "tp_x", "tp", &events);
        let a: HashMap<i64, Arrivals> = g.arrivals.into_iter().collect();
        assert_eq!((a[&1].last, a[&1].cohorts), (10, 10));
        assert_eq!(a[&1].expected, 5.0);
        assert!(a[&1].z() > 3.0 && a[&0].z() < -3.0);
        assert_eq!(g.spreads, vec![30.0; 10]);
    }

    #[test]
    fn extremes_list_lowest_and_highest_ranks() {
        let values: Vec<(i64, f64)> = (0..20).map(|r| (r, r as f64)).collect();
        let (median, low, high) = extremes(&values).unwrap();
        assert_eq!(median, 9.0);
        assert_eq!(low.iter().map(|x| x.0).collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        assert_eq!(
            high.iter().map(|x| x.0).collect::<Vec<_>>(),
            [19, 18, 17, 16, 15]
        );
        assert!(extremes(&[]).is_none());
    }

    #[test]
    fn axis_is_the_longest_matching_prefix() {
        let axes = vec![
            "pp".to_string(),
            "pp_p2p_round_0".to_string(),
            "tp".to_string(),
        ];
        assert_eq!(axis_of(&format!("pp_{}", "a".repeat(40)), &axes), "pp");
        assert_eq!(axis_of("pp_p2p_round_0_abc-r4", &axes), "pp_p2p_round_0");
        assert_eq!(axis_of("ep_abc", &axes), "other");
    }

    #[test]
    fn axis_titles_and_rank_counts_match_the_page() {
        assert_eq!(pretty_axis("dp_shard"), "DP Shard");
        assert_eq!(
            pretty_axis("TENSOR_MODEL_PARALLEL_GROUP"),
            "TENSOR MODEL PARALLEL GROUP"
        );
        assert_eq!(pretty_axis("dp+cp"), "DP + CP");
        assert_eq!(pretty_axis("pp_p2p_round_0"), "PP P2p Round 0");
        assert_eq!(en_us(248), "248");
        assert_eq!(en_us(49144), "49,144");
        assert_eq!(en_us(1234567), "1,234,567");
    }

    #[test]
    fn hosts_come_from_rank_map_runs() {
        let doc = r#"{"hosts":["a","b"],"runs":[[0,2,0,-1],[2,2,1,-1]],"v":1}"#;
        let hosts = rank_hosts(Some(doc));
        assert_eq!(hosts[&1], "a");
        assert_eq!(hosts[&3], "b");
    }
}

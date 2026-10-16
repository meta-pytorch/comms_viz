// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Structured report: the HTML report's data and derived tables as
//! a Markdown index plus compact CSVs instead of an interactive page.
//!
//! The cohort, imbalance, arrival and utilization rules mirror the page's
//! JavaScript (`pl.cohorts` and friends in static_page.html), so both
//! reports agree on who the straggler is.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;

use crate::publish::RunData;

const TOP: usize = 10;
const TOP_PER_GROUP: usize = 5;
const IMBALANCE_ROWS: usize = 200;

#[derive(Clone, Debug)]
struct Event {
    rank: i64,
    ts: f64,
    dur: f64,
    name: String,
    original_name: String,
    sequence: Option<String>,
    pg: String,
    collective: String,
    dtype: String,
    in_n: f64,
    out_n: f64,
    group_size: i64,
    step: Option<i64>,
}

struct Table {
    names: Vec<String>,
    cols: Vec<Vec<String>>,
}

impl Table {
    fn nrows(&self) -> usize {
        self.cols.first().map_or(0, Vec::len)
    }

    fn col(&self, name: &str) -> Option<&[String]> {
        let index = self.names.iter().position(|n| n == name)?;
        Some(&self.cols[index])
    }
}

/// Minimal RFC 4180 reader for the raw-CSV member fallback.
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (true, '"') if chars.peek() == Some(&'"') => {
                chars.next();
                field.push('"');
            }
            (true, '"') => quoted = false,
            (true, _) => field.push(c),
            (false, '"') => quoted = true,
            (false, ',') => row.push(std::mem::take(&mut field)),
            (false, '\n') => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            (false, '\r') => {}
            (false, _) => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

fn table_from_csv(text: &str) -> Table {
    let mut rows = parse_csv(text).into_iter();
    let names = rows.next().unwrap_or_default();
    let mut cols = vec![Vec::new(); names.len()];
    for row in rows.filter(|row| row.len() == names.len()) {
        for (col, value) in cols.iter_mut().zip(row) {
            col.push(value);
        }
    }
    Table { names, cols }
}

fn table_from_member(payload: &[u8]) -> Table {
    if payload.starts_with(b"DPG1") {
        let (names, cols) = comms_viz_core::dpg::decode_member(payload);
        Table { names, cols }
    } else {
        table_from_csv(&String::from_utf8_lossy(payload))
    }
}

fn num(value: &str) -> f64 {
    value.parse().unwrap_or(0.0)
}

fn events_from_table(table: &Table) -> Vec<Event> {
    let get = |name: &str, row: usize| table.col(name).map(|col| col[row].as_str());
    let optional = |name: &str, row: usize| get(name, row).filter(|v| !v.is_empty());
    (0..table.nrows())
        .filter(|&row| get("para_type", row) != Some("flash"))
        .map(|row| {
            let raw = get("name", row).unwrap_or("").to_string();
            let operation = optional("operation", row);
            let name = match operation {
                Some("send") => format!(
                    "Send -> rank {}",
                    optional("dst_rank", row).unwrap_or("unknown")
                ),
                Some("recv") => format!(
                    "Recv <- rank {}",
                    optional("src_rank", row).unwrap_or("unknown")
                ),
                _ if raw.to_ascii_lowercase().contains("sendrecv") => {
                    "SendRecv (direction unknown)".to_string()
                }
                _ => raw.clone(),
            };
            Event {
                rank: num(get("rank", row).unwrap_or("0")) as i64,
                ts: num(get("ts", row).unwrap_or("0")),
                dur: num(get("dur", row).unwrap_or("0")),
                original_name: optional("original_name", row).unwrap_or(&raw).to_string(),
                name,
                sequence: optional("sequence_id", row).map(str::to_string),
                pg: get("pg_name", row).unwrap_or("").to_string(),
                collective: get("collective_name", row).unwrap_or("").to_string(),
                dtype: get("dtype", row).unwrap_or("").to_string(),
                in_n: get("in_nelems", row).map_or(-1.0, num),
                out_n: get("out_nelems", row).map_or(-1.0, num),
                group_size: get("group_size", row).map_or(-1, |v| num(v) as i64),
                step: get("step", row)
                    .map(num)
                    .filter(|step| *step >= 0.0)
                    .map(|step| step as i64),
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

fn cohorts(events: &[Event]) -> (Vec<Cohort>, bool) {
    let by_seq = events.iter().any(|e| e.sequence.is_some());
    let mut rank_order = Vec::new();
    let mut per_rank: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, e) in events.iter().enumerate() {
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
        let mut occurrences: HashMap<String, usize> = HashMap::new();
        for (k, &i) in evs.iter().enumerate() {
            let ev = &events[i];
            let base = match (&ev.sequence, by_seq) {
                (None, true) => continue,
                (Some(seq), true) => format!(
                    "{}\0{}\0{seq}",
                    ev.pg,
                    ev.step.map_or("?".to_string(), |s| s.to_string())
                ),
                (_, false) => format!("ordinal\0{k}"),
            };
            let occurrence = occurrences.entry(base.clone()).or_default();
            let key = format!("{base}\0{occurrence}");
            *occurrence += 1;
            let index = *by_key.entry(key).or_insert_with(|| {
                list.push(Cohort {
                    seq: if let Some(seq) = &ev.sequence {
                        seq.parse::<i64>().map_or(seq.clone(), |n| n.to_string())
                    } else {
                        k.to_string()
                    },
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
            if events[c.ear].original_name != ev.original_name {
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
    (list, by_seq)
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
    if e.collective.is_empty() {
        &e.name
    } else {
        let c = e.collective.as_str();
        c.strip_suffix("_into_tensor")
            .or_else(|| c.strip_suffix("_tensor"))
            .unwrap_or(c)
    }
}

fn percentile(sorted: &[usize], p: f64) -> usize {
    let index = ((p * sorted.len() as f64).ceil() as usize).saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

fn union_duration(mut spans: Vec<(f64, f64)>) -> f64 {
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut total, mut end) = (0.0, f64::NEG_INFINITY);
    for (a, b) in spans {
        if a > end {
            total += b - a;
        } else if b > end {
            total += b - end;
        }
        end = end.max(b);
    }
    total
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

struct Csv(String);

impl Csv {
    fn new(header: &[&str]) -> Self {
        Csv(header.join(",") + "\n")
    }

    fn row(&mut self, fields: &[String]) {
        let fields: Vec<String> = fields.iter().map(|f| csv_field(f)).collect();
        self.0.push_str(&fields.join(","));
        self.0.push('\n');
    }
}

/// One decimal, without a trailing ".0".
fn d1(value: f64) -> String {
    let s = format!("{value:.1}");
    s.strip_suffix(".0").map_or(s.clone(), str::to_string)
}

/// Whole numbers from 100 up, one decimal below.
fn short(value: f64) -> String {
    if value.abs() >= 100.0 {
        format!("{value:.0}")
    } else {
        d1(value)
    }
}

fn pct(value: f64, window: f64) -> String {
    if window > 0.0 {
        d1(100.0 * value / window)
    } else {
        String::new()
    }
}

/// "0-3,8,10-11"
fn rank_ranges(ranks: &[i64]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < ranks.len() {
        let mut j = i;
        while j + 1 < ranks.len() && ranks[j + 1] == ranks[j] + 1 {
            j += 1;
        }
        out.push(if j > i + 1 {
            format!("{}-{}", ranks[i], ranks[j])
        } else if j == i + 1 {
            format!("{},{}", ranks[i], ranks[j])
        } else {
            ranks[i].to_string()
        });
        i = j + 1;
    }
    out.join(",")
}

/// `tp_<40 hex>` groups become `tp0`, `tp1`, ... ordered by lowest rank;
/// a prefix with a single group keeps the bare prefix.
fn group_aliases(groups: &[(String, i64)]) -> Vec<String> {
    let prefix = |stem: &str| {
        stem.rsplit_once('_')
            .filter(|(_, hash)| hash.len() == 40 && hash.bytes().all(|b| b.is_ascii_hexdigit()))
            .map(|(prefix, _)| prefix.to_string())
    };
    let mut by_prefix: BTreeMap<String, Vec<(i64, usize)>> = BTreeMap::new();
    for (i, (stem, min_rank)) in groups.iter().enumerate() {
        if let Some(p) = prefix(stem) {
            by_prefix.entry(p).or_default().push((*min_rank, i));
        }
    }
    let mut aliases: Vec<String> = groups.iter().map(|(stem, _)| stem.clone()).collect();
    for (p, mut members) in by_prefix {
        if members.len() == 1 {
            aliases[members[0].1] = p;
            continue;
        }
        members.sort_unstable();
        for (n, (_, i)) in members.into_iter().enumerate() {
            aliases[i] = format!("{p}{n}");
        }
    }
    aliases
}

/// Sort key putting `tp2` before `tp10`.
fn alias_order(alias: &str) -> (String, u64) {
    let prefix = alias.trim_end_matches(|c: char| c.is_ascii_digit());
    (
        prefix.to_string(),
        alias[prefix.len()..].parse().unwrap_or(0),
    )
}

/// Smallest 1/2/5 x 10^k at or above `value`.
fn nice_ceil(value: f64) -> f64 {
    let base = 10f64.powf(value.log10().floor());
    [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * base)
        .find(|v| *v >= value)
        .unwrap_or(10.0 * base)
}

struct Activity {
    rank: i64,
    gpu: f64,
    compute: f64,
    cpu: f64,
    window: f64,
}

struct PmSample {
    rank: i64,
    metric: String,
    ts: f64,
    value: f64,
}

struct Group {
    stem: String,
    alias: String,
    events: Vec<Event>,
    ranks: Vec<i64>,
}

struct Worst {
    step: String,
    seq: String,
    collective: String,
    min_dur: f64,
    first: i64,
    last: i64,
    spread_pct: f64,
}

struct GroupStats {
    alias: String,
    stem: String,
    ranks: Vec<i64>,
    kernels: usize,
    complete: usize,
    by_seq: bool,
    worst: Vec<Worst>,
    last_arrivals: Vec<(i64, usize, usize)>,
}

fn write(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

/// Write one run's structured report into `dest/<stem>/`.
pub(crate) fn write_structured_report(data: &RunData, stem: &str, dest: &Path) -> Result<()> {
    let dir = dest.join(stem);
    // Start empty: a rerun can drop groups or optional tables, whose files
    // from the previous run would otherwise remain.
    if dir.exists() {
        std::fs::remove_dir_all(&dir).with_context(|| format!("clear {}", dir.display()))?;
    }
    let groups_dir = dir.join("groups");
    std::fs::create_dir_all(&groups_dir)
        .with_context(|| format!("create {}", groups_dir.display()))?;

    let mut pm_samples: Vec<PmSample> = Vec::new();
    let mut activity: Vec<Activity> = Vec::new();
    let mut groups: Vec<Group> = Vec::new();
    for (group, _, path) in &data.per_pg_all {
        let payload = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let table = table_from_member(&payload);
        if group != comms_viz_core::analysis::DIAGNOSTICS_PG {
            let events = events_from_table(&table);
            let ranks: Vec<i64> = events
                .iter()
                .map(|e| e.rank)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            groups.push(Group {
                stem: group.clone(),
                alias: String::new(),
                events,
                ranks,
            });
            continue;
        }
        let get = |name: &str, row: usize| table.col(name).map_or("", |c| c[row].as_str());
        for row in 0..table.nrows() {
            let rank = num(get("rank", row)) as i64;
            if get("kind", row) == "pm" {
                pm_samples.push(PmSample {
                    rank,
                    metric: get("metric", row).to_string(),
                    ts: num(get("ts", row)),
                    value: num(get("value", row)),
                });
            } else {
                activity.push(Activity {
                    rank,
                    gpu: num(get("gpu_busy", row)),
                    compute: num(get("compute", row)),
                    cpu: num(get("cpu_active", row)),
                    window: num(get("window", row)),
                });
            }
        }
    }
    let keys: Vec<(String, i64)> = groups
        .iter()
        .map(|g| (g.stem.clone(), g.ranks.first().copied().unwrap_or(0)))
        .collect();
    for (g, alias) in groups.iter_mut().zip(group_aliases(&keys)) {
        g.alias = alias;
    }
    groups.sort_by_key(|g| alias_order(&g.alias));
    let t0 = groups
        .iter()
        .flat_map(|g| g.events.iter().map(|e| e.ts))
        .chain(pm_samples.iter().map(|s| s.ts))
        .fold(f64::INFINITY, f64::min);
    let t0 = if t0.is_finite() { t0 } else { 0.0 };

    let mut imbalance = Csv::new(&[
        "group",
        "step",
        "seq",
        "op",
        "bytes",
        "dur",
        "algbw_gbps",
        "busbw_gbps",
        "first",
        "last",
        "t",
        "spread",
        "spread_pct",
    ]);
    let mut arrivals = Csv::new(&["group", "rank", "op", "last", "runs", "pos_p50", "pos_p90"]);
    let mut comm_csv = Csv::new(&["group", "rank", "comm", "comm_pct"]);
    let window_of: HashMap<i64, f64> = activity.iter().map(|a| (a.rank, a.window)).collect();
    let mut imbalance_rows: Vec<(f64, Vec<String>)> = Vec::new();
    let mut group_stats = Vec::new();

    for g in &groups {
        let events = &g.events;
        let (list, by_seq) = cohorts(events);
        let complete: Vec<&Cohort> = list.iter().filter(|c| cohort_complete(c, events)).collect();
        write(
            &groups_dir.join(format!("{}.csv", g.alias)),
            &timeline(events, t0),
        )?;

        let mut worst = Vec::new();
        for c in &complete {
            let (ear, lat, quick) = (&events[c.ear], &events[c.lat], &events[c.quick]);
            let payload = payload_bytes(&c.members, events);
            let bandwidth = if quick.dur > 0.0 && payload > 0.0 {
                payload / quick.dur / 1000.0
            } else {
                0.0
            };
            let (alg, bus) = bw_kind(&c.members, events);
            let spread = lat.ts - ear.ts;
            let spread_pct = if ear.dur > 0.0 {
                spread / ear.dur * 100.0
            } else {
                0.0
            };
            let step = c.step.map_or(String::new(), |s| s.to_string());
            imbalance_rows.push((
                spread_pct,
                vec![
                    g.alias.clone(),
                    step.clone(),
                    c.seq.clone(),
                    display_name(ear).to_string(),
                    format!("{payload}"),
                    d1(quick.dur),
                    if alg { d1(bandwidth) } else { String::new() },
                    match bus {
                        Some(factor) if alg => d1(bandwidth * factor),
                        _ => String::new(),
                    },
                    ear.rank.to_string(),
                    lat.rank.to_string(),
                    d1(ear.ts - t0),
                    d1(spread),
                    format!("{spread_pct:.0}"),
                ],
            ));
            worst.push(Worst {
                step,
                seq: c.seq.clone(),
                collective: display_name(ear).to_string(),
                min_dur: quick.dur,
                first: ear.rank,
                last: lat.rank,
                spread_pct,
            });
        }
        worst.sort_by(|a, b| b.spread_pct.total_cmp(&a.spread_pct));
        worst.truncate(TOP_PER_GROUP);

        let mut stats: BTreeMap<(i64, String), (usize, Vec<usize>)> = BTreeMap::new();
        for c in &complete {
            let mut ordered = c.members.clone();
            ordered.sort_by(|a, b| events[*a].ts.total_cmp(&events[*b].ts));
            let name = display_name(&events[c.ear]).to_string();
            for (position, &i) in ordered.iter().enumerate() {
                let entry = stats.entry((events[i].rank, name.clone())).or_default();
                entry.0 += usize::from(position == ordered.len() - 1);
                entry.1.push(position + 1);
            }
        }
        let mut last_by_rank: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
        for ((rank, name), (latest, positions)) in &mut stats {
            positions.sort_unstable();
            arrivals.row(&[
                g.alias.clone(),
                rank.to_string(),
                name.clone(),
                latest.to_string(),
                positions.len().to_string(),
                percentile(positions, 0.5).to_string(),
                percentile(positions, 0.9).to_string(),
            ]);
            let total = last_by_rank.entry(*rank).or_default();
            total.0 += *latest;
            total.1 += positions.len();
        }
        let mut last_arrivals: Vec<(i64, usize, usize)> = last_by_rank
            .into_iter()
            .filter(|(_, (latest, _))| *latest > 0)
            .map(|(rank, (latest, runs))| (rank, latest, runs))
            .collect();
        last_arrivals.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let mut spans: BTreeMap<i64, Vec<(f64, f64)>> = BTreeMap::new();
        for e in events {
            spans.entry(e.rank).or_default().push((e.ts, e.ts + e.dur));
        }
        for (rank, spans) in spans {
            let us = union_duration(spans);
            comm_csv.row(&[
                g.alias.clone(),
                rank.to_string(),
                format!("{us:.0}"),
                pct(us, window_of.get(&rank).copied().unwrap_or(0.0)),
            ]);
        }

        group_stats.push(GroupStats {
            alias: g.alias.clone(),
            stem: g.stem.clone(),
            ranks: g.ranks.clone(),
            kernels: events.len(),
            complete: complete.len(),
            by_seq,
            worst,
            last_arrivals,
        });
    }
    imbalance_rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, row) in imbalance_rows.iter().take(IMBALANCE_ROWS) {
        imbalance.row(row);
    }

    activity.sort_by_key(|a| a.rank);
    let mut utilization = Csv::new(&["rank", "gpu_pct", "compute_pct", "cpu_pct", "window"]);
    for a in &activity {
        utilization.row(&[
            a.rank.to_string(),
            pct(a.gpu, a.window),
            pct(a.compute, a.window),
            pct(a.cpu, a.window),
            format!("{:.0}", a.window),
        ]);
    }

    write(&dir.join("breakdown.csv"), &breakdown(&data.breakdown_csv))?;
    write(&dir.join("imbalance.csv"), &imbalance.0)?;
    write(&dir.join("arrivals.csv"), &arrivals.0)?;
    write(&dir.join("utilization.csv"), &utilization.0)?;
    write(&dir.join("comm_by_group.csv"), &comm_csv.0)?;
    let pm_bin = if pm_samples.is_empty() {
        None
    } else {
        write(&dir.join("pm_summary.csv"), &pm_summary(&pm_samples))?;
        let (text, bin) = pm_timeline(&pm_samples, t0);
        write(&dir.join("pm_timeline.csv"), &text)?;
        Some(bin)
    };
    write(
        &dir.join("README.md"),
        &readme(&ReadmeInput {
            data,
            t0,
            groups: &group_stats,
            activity: &activity,
            pm_bin,
        }),
    )?;
    eprintln!("comms_viz: wrote structured report {}", dir.display());
    Ok(())
}

/// One row per kernel, cohorts adjacent: sorted by step, seq, rank.
fn timeline(events: &[Event], t0: f64) -> String {
    let seq_of = |e: &Event| e.sequence.as_deref().and_then(|s| s.parse::<i64>().ok());
    let mut order: Vec<&Event> = events.iter().collect();
    order.sort_by(|a, b| {
        (a.step, seq_of(a).is_none(), seq_of(a), a.rank)
            .cmp(&(b.step, seq_of(b).is_none(), seq_of(b), b.rank))
            .then(a.ts.total_cmp(&b.ts))
    });
    let mut csv = Csv::new(&["step", "seq", "rank", "op", "t", "dur", "bytes"]);
    for e in order {
        let bytes = payload_bytes(&[0], std::slice::from_ref(e));
        csv.row(&[
            e.step.map_or(String::new(), |s| s.to_string()),
            e.sequence.clone().unwrap_or_default(),
            e.rank.to_string(),
            display_name(e).to_string(),
            d1(e.ts - t0),
            d1(e.dur),
            if bytes > 0.0 {
                format!("{bytes}")
            } else {
                String::new()
            },
        ]);
    }
    csv.0
}

/// Per-rank breakdown with all-zero columns dropped, rank first, whole us.
fn breakdown(text: &str) -> String {
    let table = table_from_csv(text);
    let Some(rank_index) = table.names.iter().position(|n| n == "rank") else {
        return text.to_string();
    };
    let keep: Vec<usize> = std::iter::once(rank_index)
        .chain(
            (0..table.names.len())
                .filter(|&i| i != rank_index && table.cols[i].iter().any(|v| num(v) != 0.0)),
        )
        .collect();
    let header: Vec<&str> = keep.iter().map(|&i| table.names[i].as_str()).collect();
    let mut csv = Csv::new(&header);
    let mut rows: Vec<usize> = (0..table.nrows()).collect();
    rows.sort_by_key(|&r| num(&table.cols[rank_index][r]) as i64);
    for r in rows {
        let fields: Vec<String> = keep
            .iter()
            .map(|&i| {
                if i == rank_index {
                    table.cols[i][r].clone()
                } else {
                    format!("{:.0}", num(&table.cols[i][r]))
                }
            })
            .collect();
        csv.row(&fields);
    }
    csv.0
}

fn metric_order(samples: &[PmSample]) -> Vec<&str> {
    let mut seen = HashSet::new();
    samples
        .iter()
        .map(|s| s.metric.as_str())
        .filter(|m| seen.insert(*m))
        .collect()
}

fn pm_summary(samples: &[PmSample]) -> String {
    let mut by_key: BTreeMap<(&str, i64), Vec<f64>> = BTreeMap::new();
    for s in samples {
        by_key.entry((&s.metric, s.rank)).or_default().push(s.value);
    }
    let mut csv = Csv::new(&["metric", "rank", "n", "mean", "min", "max"]);
    for ((metric, rank), values) in by_key {
        let (lo, hi) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(*v), hi.max(*v))
            });
        csv.row(&[
            metric.to_string(),
            rank.to_string(),
            values.len().to_string(),
            short(values.iter().sum::<f64>() / values.len() as f64),
            short(lo),
            short(hi),
        ]);
    }
    csv.0
}

/// Samples averaged into fixed bins (about 100 over the capture), one
/// column per metric. Returns the CSV and the bin width in us.
fn pm_timeline(samples: &[PmSample], t0: f64) -> (String, f64) {
    let (lo, hi) = samples
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s.ts), hi.max(s.ts))
        });
    let bin = nice_ceil(((hi - lo) / 100.0).max(1000.0));
    let metrics = metric_order(samples);
    let column: HashMap<&str, usize> = metrics.iter().enumerate().map(|(i, m)| (*m, i)).collect();
    let mut cells: BTreeMap<(i64, i64), Vec<(f64, usize)>> = BTreeMap::new();
    for s in samples {
        let row = cells
            .entry((s.rank, ((s.ts - t0) / bin).floor() as i64))
            .or_insert_with(|| vec![(0.0, 0); metrics.len()]);
        let cell = &mut row[column[s.metric.as_str()]];
        cell.0 += s.value;
        cell.1 += 1;
    }
    let header: Vec<&str> = ["rank", "t_ms"].into_iter().chain(metrics).collect();
    let mut csv = Csv::new(&header);
    for ((rank, index), row) in cells {
        let fields: Vec<String> = [rank.to_string(), d1(index as f64 * bin / 1000.0)]
            .into_iter()
            .chain(row.into_iter().map(|(sum, n)| {
                if n > 0 {
                    short(sum / n as f64)
                } else {
                    String::new()
                }
            }))
            .collect();
        csv.row(&fields);
    }
    (csv.0, bin)
}

struct ReadmeInput<'a> {
    data: &'a RunData,
    t0: f64,
    groups: &'a [GroupStats],
    activity: &'a [Activity],
    pm_bin: Option<f64>,
}

fn readme(input: &ReadmeInput<'_>) -> String {
    let data = input.data;
    let mut s = String::new();
    let run_name = data
        .run_json
        .get("run_name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let _ = writeln!(s, "# comms_viz report: {run_name}\n");
    let _ = writeln!(s, "- run id: `{}`", data.run_id);
    if let Some(step) = data
        .metadata_json
        .as_ref()
        .and_then(|m| m["capture"]["step"].as_i64())
    {
        let _ = writeln!(s, "- step: {step}");
    }
    let _ = writeln!(s, "- ranks: {}", data.status_json["ranks"]["processed"]);
    for warning in &data.warnings {
        let _ = writeln!(s, "- warning: {warning}");
    }
    let _ = writeln!(
        s,
        "\nTimes and durations are us. `t` columns are us after the first kernel or \
         sample (trace ts {:.3}). `bytes` is the collective payload.",
        input.t0
    );
    let _ = write!(
        s,
        "\n## Files\n\n\
         - `breakdown.csv`: per-rank time by category, non-zero categories only\n\
         - `groups/<group>.csv`: every kernel in the group, one row per rank per \
         collective, rows of one collective adjacent\n\
         - `imbalance.csv`: the {IMBALANCE_ROWS} collectives with the worst start spread across \
         their group, worst first; `first`/`last` are the first and last ranks to start, \
         `spread_pct` is the spread over the first rank's duration\n\
         - `arrivals.csv`: per rank and op, how often the rank started last and its \
         arrival position\n\
         - `utilization.csv`: per-rank GPU occupied, compute kernel and CPU active %\n\
         - `comm_by_group.csv`: per-rank time in each group's collectives, including \
         waiting\n"
    );
    if let Some(bin) = input.pm_bin {
        let _ = writeln!(
            s,
            "- `pm_summary.csv`: GPU counter stats per metric and rank\n\
             - `pm_timeline.csv`: GPU counters averaged over {} ms bins",
            d1(bin / 1000.0)
        );
    }

    let _ = writeln!(s, "\n## Groups\n");
    let _ = writeln!(
        s,
        "| group | ranks | kernels | matched | most often last | name |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|");
    for g in input.groups {
        let last = g
            .last_arrivals
            .iter()
            .take(3)
            .map(|(rank, latest, runs)| format!("{rank} ({latest}/{runs})"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {last} | `{}` |",
            g.alias,
            rank_ranges(&g.ranks),
            g.kernels,
            g.complete,
            g.stem
        );
    }
    if input.groups.iter().any(|g| !g.by_seq) {
        s.push_str(
            "\nGroups without c10d sequence numbers are matched by position on each rank.\n",
        );
    }

    if !input.activity.is_empty() {
        let _ = writeln!(s, "\n## Utilization (lowest GPU occupied first)\n");
        let _ = writeln!(s, "| rank | GPU % | compute % | CPU % |");
        let _ = writeln!(s, "|---|---|---|---|");
        let mut sorted: Vec<&Activity> = input.activity.iter().collect();
        sorted.sort_by(|a, b| (a.gpu / a.window).total_cmp(&(b.gpu / b.window)));
        for a in sorted.iter().take(TOP) {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} |",
                a.rank,
                pct(a.gpu, a.window),
                pct(a.compute, a.window),
                pct(a.cpu, a.window)
            );
        }
    }

    let _ = writeln!(s, "\n## Worst start spread per group\n");
    let _ = writeln!(
        s,
        "| group | step | seq | op | dur | first | last | spread % |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|---|---|");
    for g in input.groups {
        for w in &g.worst {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} | {} | {} | {:.0} |",
                g.alias,
                w.step,
                w.seq,
                w.collective,
                d1(w.min_dur),
                w.first,
                w.last,
                w.spread_pct
            );
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(rank: i64, ts: f64, seq: &str) -> Event {
        Event {
            rank,
            ts,
            dur: 10.0,
            name: "ncclAllReduce".into(),
            original_name: "ncclAllReduce".into(),
            sequence: Some(seq.into()),
            pg: "0".into(),
            collective: "allreduce".into(),
            dtype: "float".into(),
            in_n: 4.0,
            out_n: 4.0,
            group_size: 2,
            step: Some(1),
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
        let (list, by_seq) = cohorts(&events);
        assert!(by_seq);
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
    fn parses_quoted_csv() {
        let table = table_from_csv("name,rank\n\"void k<a, b>\",3\n");
        assert_eq!(table.col("name").unwrap(), ["void k<a, b>"]);
        assert_eq!(table.col("rank").unwrap(), ["3"]);
    }

    #[test]
    fn hashed_groups_get_short_aliases() {
        let hash = |c: char| c.to_string().repeat(40);
        let groups = vec![
            (format!("tp_{}", hash('b')), 2),
            (format!("tp_{}", hash('a')), 0),
            (format!("fsdp_{}", hash('c')), 0),
            ("default_0".to_string(), 0),
            ("tp_short".to_string(), 0),
        ];
        assert_eq!(
            group_aliases(&groups),
            ["tp1", "tp0", "fsdp", "default_0", "tp_short"],
            "tp groups are numbered by lowest rank; a lone prefix and non-hash stems keep their names"
        );
    }

    #[test]
    fn formats_rank_ranges() {
        assert_eq!(rank_ranges(&[0, 1, 2, 3, 5, 7, 8]), "0-3,5,7,8");
        assert_eq!(rank_ranges(&[]), "");
    }

    #[test]
    fn rounds_bins_up_to_nice_widths() {
        assert_eq!(nice_ceil(1000.0), 1000.0);
        assert_eq!(nice_ceil(1300.0), 2000.0);
        assert_eq!(nice_ceil(6000.0), 10000.0);
    }

    #[test]
    fn union_merges_overlaps() {
        assert_eq!(
            union_duration(vec![(0.0, 10.0), (5.0, 15.0), (20.0, 25.0)]),
            20.0
        );
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! The per-rank diagnostics member (`diagnostics_0`): utilization, counts of
//! communication kernels without c10d metadata, and GPU counter stats, plus
//! the data-quality warnings derived from it.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use comms_viz_core::analysis::PM_STATS;

/// A decoded per-PG member: column names and string cells.
pub(crate) struct Table {
    pub names: Vec<String>,
    pub cols: Vec<Vec<String>>,
}

impl Table {
    pub fn nrows(&self) -> usize {
        self.cols.first().map_or(0, Vec::len)
    }

    pub fn col(&self, name: &str) -> Option<&[String]> {
        let index = self.names.iter().position(|n| n == name)?;
        Some(&self.cols[index])
    }
}

/// Minimal RFC 4180 reader for the raw-CSV member fallback.
pub(crate) fn parse_csv(text: &str) -> Vec<Vec<String>> {
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

pub(crate) fn table_from_csv(text: &str) -> Table {
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

pub(crate) fn table_from_member(payload: &[u8]) -> Table {
    if payload.starts_with(b"DPG1") {
        let (names, cols) = comms_viz_core::dpg::decode_member(payload);
        Table { names, cols }
    } else {
        table_from_csv(&String::from_utf8_lossy(payload))
    }
}

pub(crate) fn num(value: &str) -> f64 {
    value.parse().unwrap_or(0.0)
}

pub(crate) struct Activity {
    pub rank: i64,
    pub gpu: f64,
    pub compute: f64,
    pub cpu: f64,
    pub window: f64,
    pub nccl_kernels: u64,
    pub nccl_unannotated: u64,
    pub other_comm_unannotated: u64,
}

pub(crate) struct CounterStats {
    pub rank: i64,
    pub metric: String,
    /// In `PM_STATS` order: n, min, p10, p50, mean, p90, p99, max, std.
    pub stats: [f64; 9],
}

impl CounterStats {
    pub fn get(&self, stat: &str) -> f64 {
        PM_STATS
            .iter()
            .position(|s| *s == stat)
            .map_or(f64::NAN, |i| self.stats[i])
    }
}

#[derive(Default)]
pub(crate) struct Diagnostics {
    pub activity: Vec<Activity>,
    pub counters: Vec<CounterStats>,
}

/// Read the diagnostics member, if the run has one.
pub(crate) fn read(per_pg_all: &[(String, u64, std::path::PathBuf)]) -> Result<Diagnostics> {
    let Some((_, _, path)) = per_pg_all
        .iter()
        .find(|(stem, _, _)| stem == comms_viz_core::analysis::DIAGNOSTICS_PG)
    else {
        return Ok(Diagnostics::default());
    };
    from_path(path)
}

fn from_path(path: &Path) -> Result<Diagnostics> {
    let payload = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let table = table_from_member(&payload);
    let get = |name: &str, row: usize| table.col(name).map_or("", |c| c[row].as_str());
    let mut out = Diagnostics::default();
    for row in 0..table.nrows() {
        let rank = num(get("rank", row)) as i64;
        if get("kind", row) == "pm" {
            let mut stats = [0.0; 9];
            for (slot, name) in stats.iter_mut().zip(PM_STATS) {
                *slot = num(get(name, row));
            }
            out.counters.push(CounterStats {
                rank,
                metric: get("metric", row).to_string(),
                stats,
            });
        } else {
            out.activity.push(Activity {
                rank,
                gpu: num(get("gpu_busy", row)),
                compute: num(get("compute", row)),
                cpu: num(get("cpu_active", row)),
                window: num(get("window", row)),
                nccl_kernels: num(get("nccl_kernels", row)) as u64,
                nccl_unannotated: num(get("nccl_unannotated", row)) as u64,
                other_comm_unannotated: num(get("other_comm_unannotated", row)) as u64,
            });
        }
    }
    out.activity.sort_by_key(|a| a.rank);
    Ok(out)
}

/// Share of a rank's NCCL kernels without process-group metadata above which
/// the run is flagged: such kernels are counted as communication but appear
/// on no group timeline and in no cross-rank statistic.
pub(crate) const ANNOTATION_LOSS_SHARE: f64 = 0.01;

/// "0-3,8,10-11"
pub(crate) fn rank_ranges(ranks: &[i64]) -> String {
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

/// Compact rank list: ranges, cut after `max` entries.
pub(crate) fn ranks_short(ranks: &[i64], max: usize) -> String {
    let text = rank_ranges(ranks);
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() <= max {
        text
    } else {
        format!("{},... ({} ranks)", parts[..max].join(","), ranks.len())
    }
}

/// Ranks whose NCCL kernels lost their process-group metadata, e.g.
/// CUDA-graph replays whose capture annotations did not resolve.
pub(crate) fn annotation_warnings(diag: &Diagnostics) -> Vec<String> {
    let flagged: Vec<&Activity> = diag
        .activity
        .iter()
        .filter(|a| {
            a.nccl_kernels > 0
                && a.nccl_unannotated as f64 > ANNOTATION_LOSS_SHARE * a.nccl_kernels as f64
        })
        .collect();
    if flagged.is_empty() {
        return Vec::new();
    }
    let ranks: Vec<i64> = flagged.iter().map(|a| a.rank).collect();
    let (bare, total) = flagged.iter().fold((0, 0), |(b, t), a| {
        (b + a.nccl_unannotated, t + a.nccl_kernels)
    });
    vec![format!(
        "{} of {} ranks have NCCL kernels without process-group metadata ({bare} of their \
         {total} NCCL kernels; ranks {}). Those kernels count as communication time but are \
         on no group timeline and in no cross-rank statistic. For CUDA graphs, check that \
         capture annotations resolve (see README, CUDA graphs).",
        flagged.len(),
        diag.activity.len(),
        ranks_short(&ranks, 20)
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_lists_are_ranges_cut_at_a_limit() {
        assert_eq!(rank_ranges(&[0, 1, 2, 3, 5, 7, 8]), "0-3,5,7,8");
        assert_eq!(ranks_short(&[0, 2, 4, 6], 2), "0,2,... (4 ranks)");
    }

    #[test]
    fn annotation_loss_is_flagged_per_rank() {
        let activity = |rank, nccl, bare| Activity {
            rank,
            gpu: 0.0,
            compute: 0.0,
            cpu: 0.0,
            window: 0.0,
            nccl_kernels: nccl,
            nccl_unannotated: bare,
            other_comm_unannotated: 0,
        };
        let diag = Diagnostics {
            activity: vec![
                activity(0, 100, 0),
                activity(1, 100, 1),
                activity(2, 100, 60),
            ],
            counters: Vec::new(),
        };
        let warnings = annotation_warnings(&diag);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].starts_with("1 of 3 ranks"), "{}", warnings[0]);
        assert!(warnings[0].contains("ranks 2)"), "{}", warnings[0]);
    }
}

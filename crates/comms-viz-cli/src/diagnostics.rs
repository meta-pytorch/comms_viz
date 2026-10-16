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
    /// NCCL kernel time, and the part without process-group metadata (us).
    pub nccl_us: f64,
    pub nccl_unannotated_us: f64,
    /// ProfilerStep range the rank's capture covers; None without markers.
    pub steps: Option<(i64, i64)>,
}

impl Activity {
    /// Whether this rank lost process-group metadata on more than
    /// `ANNOTATION_LOSS_SHARE` of its NCCL kernels, by count or by time.
    pub fn loses_pg_metadata(&self) -> bool {
        (self.nccl_kernels > 0
            && self.nccl_unannotated as f64 > ANNOTATION_LOSS_SHARE * self.nccl_kernels as f64)
            || (self.nccl_us > 0.0
                && self.nccl_unannotated_us > ANNOTATION_LOSS_SHARE * self.nccl_us)
    }
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
                nccl_us: num(get("nccl_us", row)),
                nccl_unannotated_us: num(get("nccl_unannotated_us", row)),
                steps: match (get("step_min", row), get("step_max", row)) {
                    ("", _) | (_, "") => None,
                    (lo, hi) => Some((num(lo) as i64, num(hi) as i64)).filter(|s| s.0 >= 0),
                },
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

/// Ascending ranks as ranges and strides: `0-3,8,12-60:4` (`a-b:s` is every
/// s-th rank from a to b), so strided groups like a 256-rank FSDP group stay
/// one token.
pub(crate) fn rank_ranges(ranks: &[i64]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < ranks.len() {
        let mut j = i;
        if i + 1 < ranks.len() {
            let stride = ranks[i + 1] - ranks[i];
            while j + 1 < ranks.len() && ranks[j + 1] - ranks[j] == stride {
                j += 1;
            }
            if j - i < 2 && stride != 1 {
                j = i;
            }
        }
        out.push(match j - i {
            0 => ranks[i].to_string(),
            1 => format!("{},{}", ranks[i], ranks[j]),
            _ if ranks[i + 1] - ranks[i] == 1 => format!("{}-{}", ranks[i], ranks[j]),
            _ => format!("{}-{}:{}", ranks[i], ranks[j], ranks[i + 1] - ranks[i]),
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

/// Ranks whose NCCL kernels lack process-group metadata: p2p from
/// `batch_isend_irecv`, which PyTorch does not annotate, or CUDA-graph
/// replays whose capture annotations did not resolve.
pub(crate) fn annotation_warnings(diag: &Diagnostics) -> Vec<String> {
    let flagged: Vec<&Activity> = diag
        .activity
        .iter()
        .filter(|a| a.loses_pg_metadata())
        .collect();
    if flagged.is_empty() {
        return Vec::new();
    }
    let ranks: Vec<i64> = flagged.iter().map(|a| a.rank).collect();
    let (bare, total, bare_us, total_us) = flagged.iter().fold((0, 0, 0.0, 0.0), |acc, a| {
        (
            acc.0 + a.nccl_unannotated,
            acc.1 + a.nccl_kernels,
            acc.2 + a.nccl_unannotated_us,
            acc.3 + a.nccl_us,
        )
    });
    let share = if total_us > 0.0 {
        100.0 * bare_us / total_us
    } else {
        0.0
    };
    vec![format!(
        "{} of {} ranks have NCCL kernels without process-group metadata ({share:.0}% of \
         their NCCL kernel time, {bare} of {total} kernels; ranks {}): counted as \
         communication, but on no group timeline and in no cross-rank statistic. See README, \
         Limitations.",
        flagged.len(),
        diag.activity.len(),
        ranks_short(&ranks, 20)
    )]
}

/// The profiler steps ranks captured, as text per distinct range.
pub(crate) fn step_label(steps: (i64, i64)) -> String {
    if steps.0 == steps.1 {
        format!("step {}", steps.0)
    } else {
        format!("steps {}-{}", steps.0, steps.1)
    }
}

/// Ranks whose captures cover different ProfilerSteps: their collectives
/// never pair up, so cross-rank statistics silently come out empty.
pub(crate) fn step_warnings(diag: &Diagnostics) -> Vec<String> {
    let mut by_steps: std::collections::BTreeMap<(i64, i64), Vec<i64>> = Default::default();
    for a in &diag.activity {
        if let Some(steps) = a.steps {
            by_steps.entry(steps).or_default().push(a.rank);
        }
    }
    if by_steps.len() < 2 {
        return Vec::new();
    }
    let parts: Vec<String> = by_steps
        .iter()
        .map(|(steps, ranks)| format!("{} on ranks {}", step_label(*steps), ranks_short(ranks, 8)))
        .collect();
    vec![format!(
        "Ranks captured different profiler steps ({}). Collectives of different steps don't \
         match across ranks, so imbalance, straggler and bandwidth statistics miss them.",
        parts.join("; ")
    )]
}

/// Whether any of the first few process groups' kernels carry a c10d
/// sequence number. Sequence numbers come from the PyTorch version, so a
/// sample of groups decides for the run without decoding every member.
pub(crate) fn any_sequence(per_pg_all: &[(String, u64, std::path::PathBuf)]) -> Result<bool> {
    let groups = per_pg_all
        .iter()
        .filter(|(stem, _, _)| !comms_viz_core::analysis::is_diagnostics_member(stem))
        .take(16);
    let mut seen = false;
    for (_, _, path) in groups {
        seen = true;
        let payload = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let table = table_from_member(&payload);
        if table
            .col("sequence_id")
            .is_some_and(|col| col.iter().any(|v| !v.is_empty()))
        {
            return Ok(true);
        }
    }
    Ok(!seen)
}

/// Warnings about what the analysis could not cover: ranks missing from the
/// trace directory, or traces without c10d sequence numbers (PyTorch before
/// 2.12 does not record them on kernels), which leave nothing to match
/// across ranks.
pub(crate) fn coverage_warnings(status: &serde_json::Value, any_sequence: bool) -> Vec<String> {
    let mut out = Vec::new();
    let processed = status["ranks"]["processed"].as_i64().unwrap_or(0);
    let total = status["ranks"]["total"].as_i64().unwrap_or(0);
    if total > processed {
        out.push(format!(
            "Only {processed} of {total} ranks were analyzed (the others' traces are missing \
             or failed to parse). Group sizes, stragglers and outliers cover only those ranks."
        ));
    }
    if !any_sequence {
        out.push(
            "No collective kernel carries a c10d sequence number, so no collective is matched \
             across ranks: no imbalance, straggler or bandwidth statistics. PyTorch before 2.12 \
             records none; some collectives (e.g. symm_mem ops) never have one."
                .to_string(),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_lists_are_ranges_cut_at_a_limit() {
        assert_eq!(rank_ranges(&[0, 1, 2, 3, 5, 7, 8]), "0-3,5,7,8");
        assert_eq!(rank_ranges(&[0, 1, 4, 8, 12, 16]), "0,1,4-16:4");
        assert_eq!(rank_ranges(&[3, 7]), "3,7");
        assert_eq!(rank_ranges(&[]), "");
        assert_eq!(
            ranks_short(&[0, 2, 4, 6, 9, 11], 2),
            "0-6:2,9,... (6 ranks)"
        );
    }

    #[test]
    fn mismatched_steps_are_flagged() {
        let at = |rank, steps| Activity {
            rank,
            gpu: 0.0,
            compute: 0.0,
            cpu: 0.0,
            window: 0.0,
            nccl_kernels: 0,
            nccl_unannotated: 0,
            other_comm_unannotated: 0,
            nccl_us: 0.0,
            nccl_unannotated_us: 0.0,
            steps,
        };
        let diag = |activity| Diagnostics {
            activity,
            counters: Vec::new(),
        };
        let same = diag(vec![at(0, Some((5, 5))), at(1, Some((5, 5))), at(2, None)]);
        assert!(step_warnings(&same).is_empty());
        let skewed = diag(vec![
            at(0, Some((5, 5))),
            at(1, Some((5, 5))),
            at(2, Some((6, 6))),
        ]);
        let w = step_warnings(&skewed);
        assert_eq!(w.len(), 1);
        assert!(
            w[0].contains("step 5 on ranks 0,1; step 6 on ranks 2"),
            "{}",
            w[0]
        );
        assert_eq!(step_label((4, 6)), "steps 4-6");
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
            nccl_us: nccl as f64,
            nccl_unannotated_us: bare as f64,
            steps: None,
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
        assert!(
            warnings[0].contains("60% of their NCCL kernel time"),
            "{}",
            warnings[0]
        );
        assert!(warnings[0].contains("ranks 2)"), "{}", warnings[0]);
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! `comms_viz <trace_root> --convert-pftrace <dir>` — synthesize a pftrace twin
//! of a chrome-JSON step: one `.pt.trace.pftrace.gz` per rank, annotations
//! baked inline (the pftrace representation is self-annotated), named after
//! the source file with the format suffix swapped.
//!
//! The point is scale-testing the pftrace ingestion path with a real job's
//! shape and content; see `comms_viz_core::legacy::pftrace_twin` for the
//! faithfulness contract.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use comms_viz_core::annotations::AnnotationMode;
use comms_viz_core::annotations::AnnotationSource;
use comms_viz_core::legacy::pftrace_twin::WriteStats;
use comms_viz_core::legacy::pftrace_twin::tree_to_pftrace;
use comms_viz_core::legacy::pkl_fetch::PklSource;
use comms_viz_core::trace::Format;

pub struct ConvertArgs {
    pub trace_root: PathBuf,
    pub out_dir: PathBuf,
    pub threads: usize,
    pub annotation_mode: AnnotationMode,
}

#[derive(Default)]
struct Aggregate {
    done: usize,
    skipped_existing: usize,
    in_bytes: u64,
    out_bytes: u64,
    raw_bytes: u64,
    st: WriteStats,
    errors: Vec<(i64, String)>,
}

impl Aggregate {
    fn fold(&mut self, in_len: usize, out_len: usize, st: &WriteStats) {
        self.done += 1;
        self.in_bytes += in_len as u64;
        self.out_bytes += out_len as u64;
        self.raw_bytes += st.raw_bytes as u64;
        self.st.cpu_slices += st.cpu_slices;
        self.st.gpu_stages += st.gpu_stages;
        self.st.annotated_rows += st.annotated_rows;
        self.st.dropped_flows += st.dropped_flows;
        self.st.dropped_meta += st.dropped_meta;
        self.st.dropped_other += st.dropped_other;
        self.st.dropped_string_track += st.dropped_string_track;
        self.st.nest_violations += st.nest_violations;
        self.st.int_ts_literals += st.int_ts_literals;
    }
}

pub fn run(a: ConvertArgs) -> Result<()> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&a.out_dir)?;

    // Rank traces: every JSON-format trace name in the folder, rank parsed
    // from the leading digit run (the shared production naming).
    let mut files: Vec<(i64, String)> = std::fs::read_dir(&a.trace_root)
        .with_context(|| format!("list {}", a.trace_root.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !Format::Json.matches_name(&name) {
                return None;
            }
            let digits: String = name.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse::<i64>().ok().map(|r| (r, name))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!(
            "no *.json[.gz] rank traces under {}",
            a.trace_root.display()
        );
    }
    let world = files.iter().map(|(r, _)| *r).max().unwrap() + 1;
    eprintln!(
        "comms_viz-convert: {} rank traces (ranks 0..{world}) -> {}",
        files.len(),
        a.out_dir.display()
    );

    // The run's annotation source, exactly as the analysis pipeline builds
    // it (same index, same live-gid filter, same pkl decode + cache).
    let ingress = comms_viz_core::trace::discover::TraceDiscovery::new();
    let source = match a.annotation_mode {
        AnnotationMode::Disabled => AnnotationSource::Disabled,
        AnnotationMode::Inline => AnnotationSource::Inline,
        AnnotationMode::Pkl => {
            AnnotationSource::Pkl(PklSource::new(&ingress, &a.trace_root, 0..world))
        }
        AnnotationMode::Auto => bail!("annotation mode Auto must be resolved by the caller"),
    };
    if matches!(source, AnnotationSource::Disabled) {
        eprintln!(
            "comms_viz-convert: WARNING: no annotation source — graphed rows keep \
             their capture lanes and carry no PG metadata (fine only for jobs \
             without CUDA-graph annotation pickles)"
        );
    }

    let agg = Mutex::new(Aggregate::default());
    let cursor = AtomicUsize::new(0);
    let n_total = files.len();

    std::thread::scope(|s| {
        for _ in 0..a.threads.max(1) {
            s.spawn(|| {
                let mut compressor =
                    libdeflater::Compressor::new(libdeflater::CompressionLvl::new(6).unwrap());
                let mut raw: Vec<u8> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= n_total {
                        break;
                    }
                    let (rank, name) = &files[i];
                    let out_name = match Format::Json
                        .suffixes()
                        .iter()
                        .find_map(|sfx| name.strip_suffix(sfx))
                    {
                        Some(stem) => format!("{stem}.pftrace.gz"),
                        None => continue,
                    };
                    let out_path = a.out_dir.join(&out_name);
                    if out_path.exists() {
                        let mut g = agg.lock().unwrap();
                        g.skipped_existing += 1;
                        continue;
                    }
                    let one = (|| -> Result<(usize, usize, WriteStats)> {
                        let gz = std::fs::read(a.trace_root.join(name))?;
                        let n = comms_viz_core::rows::maybe_gunzip_into(&gz, &mut raw)?;
                        let live =
                            comms_viz_core::legacy::pkl::extract_live_graph_ids_from_raw(&raw[..n]);
                        let tree = comms_viz_core::legacy::tree::parse_tree(&raw[..n])?;
                        // Full maps parsed per rank and DROPPED after use —
                        // the pipeline's content-hash cache retains every
                        // distinct payload for the process lifetime, which
                        // at 50k-rank scale is hundreds of GB of Value maps.
                        let anns = match &source {
                            AnnotationSource::Pkl(p) => {
                                comms_viz_core::legacy::pkl::load_annotations_from_pkl_bytes(
                                    &p.fetch(*rank, &live),
                                )?
                            }
                            _ => Default::default(),
                        };
                        let (bytes, st) = tree_to_pftrace(
                            &tree,
                            &anns,
                            comms_viz_core::cudagraph::DEFAULT_STREAM,
                        )?;
                        let bound = compressor.gzip_compress_bound(bytes.len());
                        let mut out = vec![0u8; bound];
                        let m = compressor
                            .gzip_compress(&bytes, &mut out)
                            .map_err(|e| anyhow::anyhow!("gzip: {e:?}"))?;
                        out.truncate(m);
                        let tmp = out_path.with_extension("tmp");
                        std::fs::write(&tmp, &out)?;
                        std::fs::rename(&tmp, &out_path)?;
                        Ok((gz.len(), out.len(), st))
                    })();
                    let mut g = agg.lock().unwrap();
                    match one {
                        Ok((in_len, out_len, st)) => {
                            g.fold(in_len, out_len, &st);
                            let done = g.done;
                            if done % 2048 == 0 {
                                let dt = t0.elapsed().as_secs_f64();
                                eprintln!(
                                    "comms_viz-convert: {done}/{n_total} ({:.1}/s) \
                                     in={:.1}GB out={:.1}GB",
                                    done as f64 / dt,
                                    g.in_bytes as f64 / 1e9,
                                    g.out_bytes as f64 / 1e9,
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("comms_viz-convert: rank {rank} FAILED: {e:#}");
                            g.errors.push((*rank, format!("{e:#}")));
                        }
                    }
                }
            });
        }
    });

    let g = agg.into_inner().unwrap();
    eprintln!(
        "comms_viz-convert: DONE {}/{} (skipped-existing {}) in {:.1}s",
        g.done,
        n_total,
        g.skipped_existing,
        t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "comms_viz-convert: bytes in={:.2}GB out={:.2}GB (ratio {:.3}) raw-pftrace={:.2}GB",
        g.in_bytes as f64 / 1e9,
        g.out_bytes as f64 / 1e9,
        g.out_bytes as f64 / g.in_bytes.max(1) as f64,
        g.raw_bytes as f64 / 1e9,
    );
    eprintln!(
        "comms_viz-convert: rows cpu={} gpu={} annotated={} | dropped flows={} \
         meta={} other={} string-track={} | nest-violations={} int-ts={}",
        g.st.cpu_slices,
        g.st.gpu_stages,
        g.st.annotated_rows,
        g.st.dropped_flows,
        g.st.dropped_meta,
        g.st.dropped_other,
        g.st.dropped_string_track,
        g.st.nest_violations,
        g.st.int_ts_literals,
    );
    if !g.errors.is_empty() {
        bail!(
            "{} ranks failed to convert (first: rank {} — {})",
            g.errors.len(),
            g.errors[0].0,
            g.errors[0].1
        );
    }
    Ok(())
}

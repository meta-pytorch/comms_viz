// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! The JSON-era ingestion world, quarantined for deletion.
//!
//! Everything in this directory exists to serve trace producers that have
//! not yet moved to self-annotated `.pftrace` output:
//!
//! - **Chrome-JSON rank traces** (`.pt.trace.json.gz`): the envelope probe
//!   and the row parsers ([`json`]).
//! - **Out-of-line annotation pickles**: a CUDA-graph replay trace that does
//!   not annotate itself needs a side channel of per-rank `.pkl` files to
//!   restore process-group metadata — acquisition ([`pkl_fetch`]), decode
//!   ([`pkl`]).
//! - **The serde-tree pipeline** ([`tree`], [`tree_walks`]): the
//!   recording-step merge and the pkl-annotated merged-trace artifact
//!   require materializing the trace as a `serde_json::Value`, which only
//!   the JSON format supports.
//! - **The splice writer** ([`splice`]): the inline-annotated merged-trace
//!   artifact WITHOUT the tree — the walk's edits spliced into the
//!   original document bytes (the tree remains the `--verify-walks`
//!   oracle for it).
//! - **Recording-step merges** ([`recording`], [`recording_read`],
//!   [`recording_merge`]): the second repair mechanism for replay traces,
//!   copying kernel args from the step where the graph was recorded.
//! - **The pftrace twin writer** ([`pftrace_twin`]): synthesizes the pftrace
//!   a self-annotating exporter would have written from a JSON trace + its
//!   pickles — the migration/scale-testing bridge between the two worlds.
//!
//! Deleting this directory would drop JSON trace support. The compiler then
//! points at every seam to unwind: the `Format::Json` arms in `trace`, the
//! `Pkl` arms in `annotations`, the tree path in `pipeline`, the pkl stages
//! in `timing`, and the `--convert-pftrace`/pkl flags in the CLI. Nothing in
//! `analysis`, `emit`, or `cudagraph` depends on this directory.

pub mod json;
pub mod minify;
pub mod pftrace_twin;
pub mod pkl;
pub mod pkl_fetch;
pub mod recording;
pub(crate) mod recording_merge;
pub(crate) mod recording_read;
pub mod splice;
pub mod tree;
pub(crate) mod tree_walks;

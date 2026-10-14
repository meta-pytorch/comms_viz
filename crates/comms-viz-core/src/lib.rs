// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! comms-viz-core — the single-host trace-analysis pipeline.
//!
//! Data flows `trace` (format detection, discovery, per-rank decode) →
//! `rows` (row model + sort) → `cudagraph` (CUDA-graph repair) → `analysis`
//! (per-rank breakdown) → `emit` (artifact tree). `pipeline` orchestrates
//! the rank loop and aggregation; `annotations` decides where CUDA-graph
//! annotations come from. `legacy` quarantines everything that only exists
//! for the JSON-trace + annotation-pickle world.
//!
//! Output contract: artifact *shape* (file layout, JSON/parquet/sidecar
//! schemas the report page and tooling read) plus comms_viz-vs-comms_viz
//! determinism — the same inputs on the same binary produce identical bytes
//! (enforced by the integration snapshot suite).

/// The git revision baked in by build.rs (the same stamp emit.rs writes
/// into `engine.rev`); "unknown" when built outside a git checkout.
pub const GIT_REV: &str = env!("COMMS_VIZ_GIT_REV");

pub mod analysis;
pub mod annotations;
pub mod comm;
pub mod cudagraph;
pub mod dpg;
pub mod emit;
pub mod host_map;
pub mod kernel_types;
pub mod legacy;
pub mod memgate;
pub mod pipeline;
pub mod rows;
pub mod timing;
pub mod trace;
pub mod writer_pool;

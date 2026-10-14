// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! End-to-end integration suite over the vendored raw trace fixtures
//! (testdata/traces at the workspace root — real on-disk step dirs in both
//! the Kineto Chrome-JSON and perfetto-native pftrace formats).
//!
//! Every test drives the compiled `comms_viz` binary (CARGO_BIN_EXE_comms_viz)
//! exactly the way the harness scripts do: direct mode
//! (`--trace-root … --out …`) for analysis and `--out … --report …` for
//! static page rendering. The suite touches no network.
//!
//! Validation is insta snapshots (see harness::settings for the volatile
//! redactions: engine git rev, page generated_at_s) plus structural
//! assertions — cross-run/thread-count byte determinism, format-twin PG
//! parity, converter/fixture sync. Snapshots live in
//! tests/integration/snapshots; review changes with `cargo insta review`
//! (or `INSTA_UPDATE=always cargo test -p comms-viz-cli --test integration`).

mod analysis;
mod convert_pftrace;
mod harness;
mod page;
mod robustness;

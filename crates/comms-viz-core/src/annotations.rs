// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! CUDA-graph annotation sourcing — where a rank's graph annotations come
//! from, kept out of the analysis path.
//!
//! A CUDA-graph *replay* trace has lost the metadata that existed when the
//! graph was *recorded*: the kernels carry a `graph node id` but not the
//! process group, dtype, collective name, or the stream the launch really
//! belongs to. Something has to put that back before the analysis can bucket
//! the work per PG. Historically that "something" was a side-channel of
//! per-rank annotation pickles (`kernel_annotations_rank{R}_{scope}_{gid}.pkl`)
//! written next to the traces; the profiler write pipeline is moving to
//! emitting the annotations *inside* the trace instead.
//!
//! This module is the seam between those worlds. [`AnnotationSource`] is what
//! a run is configured with; [`Effective`] is what one rank's trace resolves
//! to once the trace itself has been probed. The analysis path downstream
//! only ever sees [`AnnotationDigests`] plus a `reassign` flag — it cannot
//! tell which source produced them, which is the point: retiring the pickle
//! side-channel means deleting `legacy/pkl_fetch.rs` (with the rest of the
//! `legacy/` directory) and one enum variant, not touching the converter,
//! the walks, or the analyzer.
//!
//! Dispatch granularity is deliberately **per rank**, never per row. One enum
//! match per rank against a ~1.2 s rank is unmeasurable; a match in a
//! row loop would not be. Nothing in this module is called from inside a row
//! loop — [`crate::cudagraph::apply_walks`] receives the resolved digests up
//! front and reads them through a plain `FxHashMap` lookup.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use crate::legacy::pkl;
use crate::legacy::pkl_fetch;

// ---------------------------------------------------------------------------
// Annotation data shapes.
//
// These live here rather than in `cudagraph.rs` because they describe the
// annotation *payload*, not the walk that consumes it. The digest exists so
// the row walk never materialises the full per-key maps; only the tree path
// in `legacy/tree.rs` needs the full [`Annotations`]. Keeping them here is
// what lets an inline-only world drop `legacy/{pkl_fetch,pkl}.rs` without
// touching the walk code.
// ---------------------------------------------------------------------------

/// The converted annotations table, as the tree walks consume it.
/// Keys are `graph_node_id`; each value is a list of dicts (mirrors the
/// pickle layout in `load_annotations_from_pkl_bytes`).
pub type Annotations = HashMap<i64, Vec<HashMap<String, Value>>>;

/// Fields consumed by the row walk and compact communication export. Values
/// follow the tree path's `python_str` conversion, without retaining the full
/// per-key maps or unrelated annotation payloads.
/// The tree path (`legacy::tree::annotate_tree`) still needs [`Annotations`].
#[derive(Default, Clone, Debug)]
pub struct AnnDigest {
    pub comm: crate::comm::CommMetadata,
    pub pg_desc: Option<Box<str>>,
    pub pg_name: Option<Box<str>>,
    /// `ann.get("stream")` through `coerce_to_i64` (None when absent or
    /// non-numeric — the walk leaves the running stream id untouched).
    pub stream: Option<i64>,
}

/// Digested annotations table, keyed like [`Annotations`]. FxHash: the keys
/// are graph node ids looked up once per graphed row.
pub type AnnotationDigests = rustc_hash::FxHashMap<i64, Vec<AnnDigest>>;

// ---------------------------------------------------------------------------
// Configuration.
// ---------------------------------------------------------------------------

/// How a run is configured to obtain CUDA-graph annotations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnnotationMode {
    /// No CUDA-graph handling at all. A replay trace passes through the
    /// converter untouched.
    /// An inline-annotated trace still gets walked — see [`AnnotationSource::for_trace`].
    Disabled,
    /// The trace carries its own annotations; there is no side channel.
    /// This is the direction the profiler write pipeline is moving in.
    Inline,
    /// Side-channel annotation pickles, fetched/staged per rank.
    Pkl,
    /// Decide from the job layout (a `kernel_annotations/` sibling of the step
    /// dir means pickles exist). Resolved before the run starts; a run never
    /// carries `Auto`.
    Auto,
}

impl AnnotationMode {
    /// Parse a CLI spelling.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "disabled" | "off" | "none" => Ok(AnnotationMode::Disabled),
            "inline" => Ok(AnnotationMode::Inline),
            "pkl" => Ok(AnnotationMode::Pkl),
            "auto" => Ok(AnnotationMode::Auto),
            other => {
                anyhow::bail!("unknown --annotations {other:?} (expected disabled|inline|pkl|auto)")
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AnnotationMode::Disabled => "disabled",
            AnnotationMode::Inline => "inline",
            AnnotationMode::Pkl => "pkl",
            AnnotationMode::Auto => "auto",
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime source.
// ---------------------------------------------------------------------------

/// The configured annotation source for a run, built once in
/// [`crate::pipeline::run`] and shared by every rank worker.
pub enum AnnotationSource {
    /// No CUDA-graph handling configured.
    Disabled,
    /// Annotations arrive inside the trace.
    Inline,
    /// Annotations arrive as per-rank pickles.
    Pkl(pkl_fetch::PklSource),
}

/// What one rank's trace actually resolves to, after the trace has been
/// probed. Borrowed from the run's [`AnnotationSource`]; free to construct.
pub enum Effective<'a> {
    /// Convert the trace as-is — no walks at all.
    Passthrough,
    /// Walk without reassignment: the trace already carries its annotations
    /// and its producer already assigned tids/streams.
    Inline,
    /// Walk with reassignment, applying the pickled annotations.
    Pkl(&'a pkl_fetch::PklSource),
}

impl AnnotationSource {
    /// Resolve the configured source against what a rank's trace turned out
    /// to contain.
    ///
    /// `trace_carries_annotations` wins over the configured side channel,
    /// and that is load-bearing in both directions: an inline-annotated
    /// trace is walked even under `--annotations disabled`, and applying
    /// pickles on top of one would clobber the producer's own tid/stream
    /// assignment back to the default stream.
    pub fn for_trace(&self, trace_carries_annotations: bool) -> Effective<'_> {
        if trace_carries_annotations {
            return Effective::Inline;
        }
        match self {
            AnnotationSource::Pkl(p) => Effective::Pkl(p),
            // Inline was configured but this trace has no inline envelope:
            // there is no side channel to fall back to, so there is nothing
            // to apply. Walking with empty annotations would still reassign
            // every graphed row's stream to the default — strictly worse than
            // passing through. (A genuinely inline job's traces all carry the
            // envelope; a miss here means the trace predates the switch.)
            AnnotationSource::Inline | AnnotationSource::Disabled => Effective::Passthrough,
        }
    }

    /// Does any rank need a recording-step merge? (Pickle world only.)
    pub fn uses_recording_merge(&self) -> bool {
        matches!(self, AnnotationSource::Pkl(_))
    }
}

impl Effective<'_> {
    /// Whether the CUDA-graph walk passes run for this rank.
    #[inline]
    pub fn walks(&self) -> bool {
        !matches!(self, Effective::Passthrough)
    }

    /// Whether the walk reassigns tid/`args.stream` from annotations
    /// (`walk_annotate`) — false for a trace that annotated itself.
    #[inline]
    pub fn reassign(&self) -> bool {
        matches!(self, Effective::Pkl(_))
    }

    /// Whether resolving needs the rank's live graph ids. False for the
    /// inline path, which is how it skips the `Gids` pass entirely.
    #[inline]
    pub fn needs_live_gids(&self) -> bool {
        matches!(self, Effective::Pkl(_))
    }
}

/// Empty digests, shared by every inline rank — the inline path allocates no
/// annotation state at all.
fn empty_digests() -> Arc<AnnotationDigests> {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<Arc<AnnotationDigests>> = OnceLock::new();
    EMPTY
        .get_or_init(|| Arc::new(AnnotationDigests::default()))
        .clone()
}

fn empty_annotations() -> Arc<Annotations> {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<Arc<Annotations>> = OnceLock::new();
    EMPTY
        .get_or_init(|| Arc::new(Annotations::default()))
        .clone()
}

impl Effective<'_> {
    /// This rank's digested annotations (the row-walk path).
    ///
    /// `live` is only consulted when [`Self::needs_live_gids`] is true, so the
    /// caller can pass an empty set on the inline path without computing it.
    pub fn digests(
        &self,
        rank: i64,
        live: &std::collections::HashSet<i64>,
    ) -> Result<Arc<AnnotationDigests>> {
        match self {
            Effective::Passthrough | Effective::Inline => Ok(empty_digests()),
            Effective::Pkl(p) => {
                let bytes = p.fetch(rank, live);
                let _t = crate::timing::Timer::new(crate::timing::Stage::PklDecode);
                pkl::load_digests_cached(&bytes)
            }
        }
    }

    /// This rank's FULL annotations map — the tree path (merged-trace
    /// artifact, recording merge) and `--verify-walks`.
    pub fn full(
        &self,
        rank: i64,
        live: &std::collections::HashSet<i64>,
    ) -> Result<Arc<Annotations>> {
        match self {
            Effective::Passthrough | Effective::Inline => Ok(empty_annotations()),
            Effective::Pkl(p) => {
                let bytes = p.fetch(rank, live);
                let _t = crate::timing::Timer::new(crate::timing::Stage::PklDecode);
                pkl::load_annotations_cached(&bytes)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_trace_wins_over_every_configured_source() {
        // The truth table the pre-refactor pipeline implemented: a trace that
        // carries its own annotations is walked without reassignment no
        // matter what the run was configured with.
        for src in [
            AnnotationSource::Disabled,
            AnnotationSource::Inline,
            AnnotationSource::Pkl(pkl_fetch::PklSource::for_test()),
        ] {
            let eff = src.for_trace(true);
            assert!(eff.walks());
            assert!(!eff.reassign());
            assert!(!eff.needs_live_gids());
        }
    }

    #[test]
    fn non_inline_trace_follows_configuration() {
        let d = AnnotationSource::Disabled.for_trace(false);
        assert!(!d.walks());

        // Inline configured but the trace has no envelope: nothing to apply.
        let i = AnnotationSource::Inline.for_trace(false);
        assert!(!i.walks());

        let p = AnnotationSource::Pkl(pkl_fetch::PklSource::for_test());
        let p = p.for_trace(false);
        assert!(p.walks());
        assert!(p.reassign());
        assert!(p.needs_live_gids());
    }

    #[test]
    fn inline_resolution_allocates_nothing() {
        let eff = AnnotationSource::Inline.for_trace(true);
        let live = std::collections::HashSet::new();
        let r = eff.digests(0, &live).unwrap();
        assert!(r.is_empty());
        // Same shared allocation every time.
        let r2 = eff.digests(1, &live).unwrap();
        assert!(Arc::ptr_eq(&r, &r2));
    }

    #[test]
    fn mode_parse_roundtrip() {
        for m in [
            AnnotationMode::Disabled,
            AnnotationMode::Inline,
            AnnotationMode::Pkl,
            AnnotationMode::Auto,
        ] {
            assert_eq!(AnnotationMode::parse(m.as_str()).unwrap(), m);
        }
        assert_eq!(
            AnnotationMode::parse("off").unwrap(),
            AnnotationMode::Disabled
        );
        assert_eq!(
            AnnotationMode::parse("none").unwrap(),
            AnnotationMode::Disabled
        );
        assert!(AnnotationMode::parse("pickle").is_err());
    }
}

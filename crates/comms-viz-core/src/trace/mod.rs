// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Trace input formats — reading a rank's trace, kept out of the analysis
//! path.
//!
//! The profiler write pipeline is moving to perfetto `.pftrace`, so the
//! JSON chain (gunzip, envelope scan for the inline-annotation flag,
//! simd-json tape parse) sits behind [`Format`] next to it, and everything
//! downstream of [`parse_rows`] — the cudagraph walk, the sort,
//! `WalkedView`, the analyzer — is format-blind.
//!
//! What actually varies by format:
//!
//! | concern | JSON | pftrace |
//! |---|---|---|
//! | container | gzip | gzip (same `maybe_gunzip_into`) |
//! | rank-file suffix | `.json{,.gz}` | `.pftrace{,.gz}` |
//! | self-annotation probe | `cudaGraphInlineAnnotated` envelope key | format property |
//! | rows | simd-json tape walk | protobuf wire walk |
//! | `serde_json` tree | yes | **no** — see [`Format::supports_tree`] |
//!
//! Dispatch is per rank, never per row: [`parse_rows`] is called once and
//! returns the same `ParsedRows` either way, so no row loop gains a branch.

use anyhow::Result;

use crate::rows::ParsedRows;

pub mod discover;
pub mod pftrace;

/// A trace input format.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    /// Kineto Chrome-JSON (`.pt.trace.json.gz`) — what most jobs write today.
    Json,
    /// Perfetto native (`.pftrace`) — where the write pipeline is going.
    Pftrace,
}

/// What a format is asked to decide before the run starts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FormatChoice {
    Fixed(Format),
    /// Discover from the run's own inventory (see [`detect`]).
    Auto,
}

impl Format {
    pub fn parse(s: &str) -> Result<FormatChoice> {
        match s {
            "json" => Ok(FormatChoice::Fixed(Format::Json)),
            "pftrace" => Ok(FormatChoice::Fixed(Format::Pftrace)),
            "auto" => Ok(FormatChoice::Auto),
            other => {
                anyhow::bail!("unknown --trace-format {other:?} (expected json|pftrace|auto)")
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Format::Json => "json",
            Format::Pftrace => "pftrace",
        }
    }

    /// Candidate rank-trace filename suffixes, most specific first.
    ///
    /// A table rather than one constant because the suffix is a property of
    /// the writer, not of comms_viz: discovery matches these against what the
    /// run's OWN listing contains ([`detect`]) instead of assuming a spelling.
    /// This table is the single source of suffix truth.
    pub fn suffixes(self) -> &'static [&'static str] {
        match self {
            Format::Json => &[".json.gz", ".json"],
            // The perfetto route may or may not gzip; both spellings observed.
            Format::Pftrace => &[".pftrace.gz", ".pftrace"],
        }
    }

    /// Whether this format can be materialized as a `serde_json` event tree.
    ///
    /// The tree is what the tree-based CUDA-graph walks
    /// (`legacy::tree_walks`) mutate and what the merged-trace artifact is
    /// serialized from. Only JSON has one: a perfetto trace has no
    /// Chrome-JSON document to rebuild, and inventing one would be a lossy
    /// re-encode rather than a value-preserving path.
    pub fn supports_tree(self) -> bool {
        matches!(self, Format::Json)
    }

    /// Whether a filename looks like a rank trace in this format.
    pub fn matches_name(self, name: &str) -> bool {
        self.suffixes().iter().any(|s| name.ends_with(s))
    }
}

/// One physical process group reported by PyTorch profiler metadata.
#[derive(Clone, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
pub struct ProcessGroupConfig {
    pub pg_name: String,
    pub pg_desc: String,
    pub backend_config: String,
    pub pg_size: i64,
    pub ranks: Vec<i64>,
}

/// Every format, for callers that need "is this a rank trace file at all?"
/// rather than "can I parse it?".
pub const ALL: [Format; 2] = [Format::Json, Format::Pftrace];

/// Whether `name` is a rank trace in ANY known format.
///
/// For consumers that only need to *identify* trace files rather than decode
/// them — rank→host resolution, inventory ranksets. Format-blind on purpose:
/// those consumers keep working unchanged when a job switches writers, which
/// is why they must not carry their own suffix constant.
pub fn is_trace_name(name: &str) -> bool {
    ALL.iter().any(|f| f.matches_name(name))
}

/// The merged-trace artifact name for a rank whose artifact is a VERBATIM
/// copy of its replay bytes (`pipeline::Merged::Replay` — non-tree formats
/// with non-reassigning annotations). The suffix keeps whatever compression
/// the source had, decided by the bytes themselves, never the source's
/// filename.
///
/// One definition on purpose: the analyzer's write site and the page's
/// rank-map paths must agree on the object name for a rank.
pub fn replay_artifact_name(rank: i64, bytes: &[u8]) -> String {
    format!(
        "{rank:06}.merged.pftrace{}",
        if bytes.starts_with(&[0x1f, 0x8b]) {
            ".gz"
        } else {
            ""
        }
    )
}

/// The merged-trace artifact name for a rank whose artifact is Chrome-JSON
/// (`pipeline::Merged::Passthrough` and `Merged::Tree`). One definition for
/// the same reason as [`replay_artifact_name`]: the analyzer's write site
/// and the page's rank-map paths must agree on the object name for a rank.
pub fn merged_json_artifact_name(rank: i64) -> String {
    format!("{rank:06}.merged.pt.trace.json.gz")
}

/// Everything a reader discovered about one rank's trace up front.
#[derive(Clone, Copy, Default, Debug)]
pub struct Probe {
    /// The trace carries its own CUDA-graph annotations, so no side channel
    /// applies and the walk must not reassign tids/streams.
    pub self_annotated: bool,
}

/// Probe a rank's decompressed trace bytes.
///
/// Cheap by contract — this runs before the parse on every rank, so it reads a
/// bounded prefix, never the whole buffer.
pub fn probe(format: Format, decompressed: &[u8]) -> Probe {
    let _t = crate::timing::Timer::new(crate::timing::Stage::Detect);
    match format {
        Format::Json => Probe {
            self_annotated: crate::legacy::json::is_inline_annotated_bytes(decompressed),
        },
        Format::Pftrace => Probe {
            // The perfetto route exists because the consumer runs the
            // annotation resolver itself; annotations ride in the trace.
            self_annotated: pftrace::SELF_ANNOTATED,
        },
    }
}

/// Parse a rank's decompressed trace into the converter's row representation.
///
/// The JSON scan never mutates `decompressed` (the splice writer reads the
/// same buffer afterwards); only the pftrace wire walk takes the `&mut`.
pub fn parse_rows(format: Format, decompressed: &mut [u8]) -> Result<ParsedRows> {
    parse_rows_bounds(format, decompressed, false).map(|(p, _)| p)
}

/// [`parse_rows`] + splice-writer element spans when `want_bounds` and the
/// format has them (JSON only; `None` otherwise — the caller falls back to
/// the sparse structural scan).
pub fn parse_rows_bounds(
    format: Format,
    decompressed: &mut [u8],
    want_bounds: bool,
) -> Result<(ParsedRows, Option<crate::legacy::splice::ElemSpans>)> {
    let _t = crate::timing::Timer::new(crate::timing::Stage::Parse);
    match format {
        Format::Json if want_bounds => crate::legacy::json::parse_bytes_walked_bounds(decompressed),
        Format::Json => Ok((crate::legacy::json::parse_bytes_walked(decompressed)?, None)),
        Format::Pftrace => Ok((pftrace::parse_rows(decompressed)?, None)),
    }
}

/// Resolve a [`FormatChoice`] against a directory listing.
///
/// `Auto` takes the one format present. A directory with both is an error
/// (which one the user meant is not ours to guess); an empty or unreadable
/// listing resolves to [`Format::Json`] and fails at discovery.
pub fn detect<'a>(
    choice: FormatChoice,
    names: impl Iterator<Item = &'a str>,
) -> Result<(Format, DetectReport)> {
    match choice {
        FormatChoice::Fixed(f) => {
            return Ok((
                f,
                DetectReport {
                    chosen: f,
                    explicit: true,
                    json_files: 0,
                    pftrace_files: 0,
                },
            ));
        }
        FormatChoice::Auto => (),
    };
    let (mut n_json, mut n_pf) = (0usize, 0usize);
    for n in names {
        if Format::Json.matches_name(n) {
            n_json += 1;
        } else if Format::Pftrace.matches_name(n) {
            n_pf += 1;
        }
    }
    anyhow::ensure!(
        n_json == 0 || n_pf == 0,
        "trace directory has {n_json} json and {n_pf} pftrace trace files; \
         pass --trace-format json|pftrace"
    );
    let chosen = if n_pf > 0 {
        Format::Pftrace
    } else {
        Format::Json
    };
    Ok((
        chosen,
        DetectReport {
            chosen,
            explicit: false,
            json_files: n_json,
            pftrace_files: n_pf,
        },
    ))
}

/// Why [`detect`] chose what it chose — logged, so an auto-detected run says
/// on stderr what it saw (the same discipline as the annotation auto-detect).
pub struct DetectReport {
    pub chosen: Format,
    pub explicit: bool,
    pub json_files: usize,
    pub pftrace_files: usize,
}

impl std::fmt::Display for DetectReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.explicit {
            write!(f, "trace_format={} (explicit)", self.chosen.as_str())
        } else {
            write!(
                f,
                "trace_format={} (auto: {} json / {} pftrace entries)",
                self.chosen.as_str(),
                self.json_files,
                self.pftrace_files
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_matching() {
        assert!(Format::Json.matches_name("000042.1784846093727597011.pt.trace.json.gz"));
        assert!(Format::Json.matches_name("rank42_trace.json.gz"));
        assert!(Format::Json.matches_name("trace.json"));
        assert!(!Format::Json.matches_name("rank42_trace.pftrace.gz"));
        assert!(!Format::Json.matches_name("000042.pftrace.gz"));
        assert!(Format::Pftrace.matches_name("000042.pftrace.gz"));
        assert!(Format::Pftrace.matches_name("rank42_trace.pftrace.gz"));
        assert!(Format::Pftrace.matches_name("rank42_trace.pftrace"));
        assert!(Format::Pftrace.matches_name("trace_6e535d47_000000.pftrace"));
        assert!(!Format::Pftrace.matches_name("000042.pt.trace.json.gz"));
    }

    #[test]
    fn explicit_choice_ignores_the_listing() {
        let names = ["a.pftrace", "b.pftrace"];
        let (f, r) = detect(FormatChoice::Fixed(Format::Json), names.iter().copied()).unwrap();
        assert_eq!(f, Format::Json);
        assert!(r.explicit);
    }

    #[test]
    fn auto_takes_the_only_format() {
        let json = ["0.pt.trace.json.gz", "1.pt.trace.json.gz", "notes.txt"];
        assert_eq!(
            detect(FormatChoice::Auto, json.iter().copied()).unwrap().0,
            Format::Json
        );
        let pf = ["0.pftrace.gz", "1.pftrace.gz"];
        assert_eq!(
            detect(FormatChoice::Auto, pf.iter().copied()).unwrap().0,
            Format::Pftrace
        );
    }

    #[test]
    fn auto_rejects_mixed_formats() {
        let mixed = ["0.pftrace.gz", "1.pftrace.gz", "0.pt.trace.json.gz"];
        let err = detect(FormatChoice::Auto, mixed.iter().copied())
            .err()
            .expect("mixed formats");
        assert!(err.to_string().contains("1 json and 2 pftrace"), "{err}");
    }

    #[test]
    fn auto_on_an_empty_listing_falls_back_to_json() {
        let (f, r) = detect(FormatChoice::Auto, std::iter::empty()).unwrap();
        assert_eq!(f, Format::Json);
        assert_eq!((r.json_files, r.pftrace_files), (0, 0));
    }

    #[test]
    fn is_trace_name_spans_formats() {
        assert!(is_trace_name("000042.1784846093727597011.pt.trace.json.gz"));
        assert!(is_trace_name("000042.pftrace.gz"));
        assert!(!is_trace_name("kernel_annotations_rank7_fwd_bwd_248.pkl"));
    }

    #[test]
    fn only_json_materializes_a_tree() {
        assert!(Format::Json.supports_tree());
        assert!(!Format::Pftrace.supports_tree());
    }

    #[test]
    fn replay_artifact_names_follow_the_bytes() {
        // Suffix from the gzip magic, never from any filename.
        assert_eq!(
            replay_artifact_name(7, &[0x1f, 0x8b, 0x08]),
            "000007.merged.pftrace.gz"
        );
        assert_eq!(
            replay_artifact_name(53_247, &[0x0a, 0x00]),
            "053247.merged.pftrace"
        );
        // Both spellings are rank-scannable (the publisher's and the
        // page's shared key).
        assert!(is_trace_name("000007.merged.pftrace.gz"));
        assert!(is_trace_name("053247.merged.pftrace"));
    }

    #[test]
    fn format_parse_roundtrip() {
        assert_eq!(
            Format::parse("json").unwrap(),
            FormatChoice::Fixed(Format::Json)
        );
        assert_eq!(
            Format::parse("pftrace").unwrap(),
            FormatChoice::Fixed(Format::Pftrace)
        );
        assert_eq!(Format::parse("auto").unwrap(), FormatChoice::Auto);
        assert!(Format::parse("perfetto").is_err());
    }
}

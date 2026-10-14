// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! The serde-tree pipeline: parse a JSON trace into a `serde_json::Value`,
//! run the CUDA-graph passes on it, serialize it back to the merged-trace
//! artifact bytes.
//!
//! The tree exists for two consumers the row walk cannot serve: the
//! serialized `.merged.pt.trace.json.gz` artifact, and the recording-step
//! merge (which rewrites arbitrary `args` keys). Pass sequence per rank:
//!   - inline-annotated trace → move/fix only ([`inline_tree`])
//!   - pickles, no recording → annotate → move → fix ([`annotate_tree`])
//!   - pickles + recording   → merge → annotate → move → fix
//!     ([`recording_merge_tree`])
//!
//! Values are identical to the row-walk path (`--verify-walks` asserts the
//! converted columns match bitwise); the tree additionally carries every
//! event field, which is what the artifact serialization needs.
//!
//! `Annotations` is `HashMap<i64, Vec<HashMap<String, serde_json::Value>>>`,
//! the unpickled `dict[int, list[dict]]` merged across pkl payloads.

use std::io::Read;

use anyhow::Context;
use anyhow::Result;
use flate2::read::GzDecoder;

use super::recording_merge;
use super::tree_walks;
use crate::annotations::Annotations;

/// Parse (possibly gzipped) replay bytes into the event tree.
pub fn parse_tree(replay_bytes: &[u8]) -> Result<serde_json::Value> {
    gzip_then_parse(replay_bytes)
}

/// Parse decompressed JSON bytes into the event tree, mutating the buffer
/// in place (simd-json unescapes strings into it). For callers whose buffer
/// is dead after the parse — skips the defensive whole-document copy
/// `parse_tree` makes. Gzipped input falls back to the copying path.
pub fn parse_tree_in_place(json_bytes: &mut [u8]) -> Result<serde_json::Value> {
    if json_bytes.starts_with(GZIP_MAGIC) {
        return gzip_then_parse(json_bytes);
    }
    simd_json::serde::from_slice(json_bytes).context("simd-json parse failed")
}

/// Annotate passes on a tree: annotate(annotations) → move → fix.
pub fn annotate_tree(root: &mut serde_json::Value, annotations: &Annotations, default_stream: i64) {
    tree_walks::walk_passes(root, annotations, default_stream);
}

/// Recording passes on a tree: recording merge → annotate → move → fix.
pub fn recording_merge_tree(
    root: &mut serde_json::Value,
    recording_bytes: &[u8],
    annotations: &Annotations,
    default_stream: i64,
) -> Result<()> {
    let rec_sparse = super::recording_read::read_recording_sparse(recording_bytes)?;
    recording_merge::merge_cudagraph_metadata_from_rec_events(rec_sparse, root);
    tree_walks::walk_passes(root, annotations, default_stream);
    Ok(())
}

/// Inline-annotated passes on a tree (no annotate/reassign): move → fix.
pub fn inline_tree(root: &mut serde_json::Value, default_stream: i64) {
    tree_walks::walk_passes_no_reassign(root, default_stream);
}

/// Serialize a merged tree to the gzip artifact bytes.
///
/// `size_hint` is the expected serialized size (callers know the source
/// document's decompressed length); it pre-sizes the JSON buffer so the
/// serializer never reallocates mid-document.
pub fn serialize_tree(
    root: &serde_json::Value,
    gzipped: bool,
    size_hint: usize,
) -> Result<Vec<u8>> {
    to_gzipped_json(root, gzipped, size_hint)
}

// ---------------------------------------------------------------------------
// Tree parse (bytes → serde_json::Value).
// ---------------------------------------------------------------------------

const GZIP_MAGIC: &[u8; 2] = b"\x1f\x8b";

/// Gzip-detect + decompress, then parse with simd-json but materialise into
/// a `serde_json::Value` (the type the tree walks and the merged-trace
/// serialization operate on).
fn gzip_then_parse(input: &[u8]) -> Result<serde_json::Value> {
    let mut buf: Vec<u8>;
    let json_bytes: &[u8] = if input.starts_with(GZIP_MAGIC) {
        buf = Vec::with_capacity(input.len() * 4); // heuristic: ~4x expansion
        let mut decoder = GzDecoder::new(input);
        decoder
            .read_to_end(&mut buf)
            .context("gzip decompress failed")?;
        &buf
    } else {
        input
    };

    // simd-json mutates its input in place during parsing, so clone
    // into an owned mutable buffer. The clone is cheap relative to the
    // SIMD parse it enables.
    let mut owned = json_bytes.to_vec();
    let value: serde_json::Value =
        simd_json::serde::from_slice(&mut owned).context("simd-json parse failed")?;
    Ok(value)
}

// ---------------------------------------------------------------------------
// Tree serialize (serde_json::Value → gzip bytes).
//
// Two facts here are pinned by the comms_viz-vs-comms_viz determinism suites and
// must not drift: serde_json's Map is a BTreeMap (no `preserve_order`
// feature), so keys serialize in sorted order; and the gzip level is 1.
// ---------------------------------------------------------------------------

/// Buffered serializer: serialize into one pre-sized Vec, then one-shot
/// libdeflate gzip (level 1).
///
/// This replaced streaming `serde_json::to_writer(GzEncoder)`: the
/// serializer emits token-sized writes, and pushing those through the
/// deflate state machine one at a time ran the whole encode at ~20 MB/s
/// per thread — gzip of the merged-trace artifacts was 40-45% of shard
/// CPU on a 10240-rank JSON job. One deflate call over the full document
/// (plus libdeflate's vectorized crc32) removes almost all of it, for one
/// transient uncompressed-document buffer per in-flight rank — small next
/// to the parsed tree the caller already holds.
fn to_gzipped_json(root: &serde_json::Value, gzipped: bool, size_hint: usize) -> Result<Vec<u8>> {
    let mut json = Vec::with_capacity(size_hint.max(64 * 1024));
    serde_json::to_writer(&mut json, root).context("serde_json::to_writer failed")?;
    if !gzipped {
        return Ok(json);
    }
    gzip_artifact(&json)
}

/// One-shot libdeflate gzip (level 1) — the artifact compressor, shared
/// with the splice writer. Returns an exact-size Vec: the compress-bound
/// scratch (~1x the document, ~45 MB for a ~3 MB artifact) is dropped here
/// instead of riding the writer queue and upload as spare capacity.
pub fn gzip_artifact(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut c = libdeflater::Compressor::new(
        libdeflater::CompressionLvl::new(1).expect("gzip level 1 is always valid"),
    );
    let mut scratch = vec![0u8; c.gzip_compress_bound(bytes.len())];
    let n = c
        .gzip_compress(bytes, &mut scratch)
        .context("libdeflate gzip failed")?;
    Ok(scratch[..n].to_vec())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;

    use super::*;

    #[test]
    fn parses_raw_json() {
        let raw = br#"{"a":1,"b":"x"}"#;
        let v = gzip_then_parse(raw).expect("parse should succeed");
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"], "x");
    }

    #[test]
    fn detects_and_decompresses_gzip() {
        let raw = br#"{"traceEvents":[{"ph":"X","ts":100}]}"#;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(1));
        encoder
            .write_all(raw)
            .expect("gzip write_all should succeed in test fixture");
        let gz = encoder.finish().expect("gzip finish should succeed");

        let v = gzip_then_parse(&gz).expect("parse should succeed");
        assert_eq!(v["traceEvents"][0]["ph"], "X");
        assert_eq!(v["traceEvents"][0]["ts"], 100);
    }

    #[test]
    fn rejects_malformed_json() {
        let bad = b"{ this is not json";
        assert!(gzip_then_parse(bad).is_err());
    }

    #[test]
    fn in_place_matches_copying_parse() {
        let raw = br#"{"traceEvents":[{"ph":"X","ts":100,"args":{"k":"a\nb"}}]}"#;
        let copied = parse_tree(raw).unwrap();
        let mut buf = raw.to_vec();
        let in_place = parse_tree_in_place(&mut buf).unwrap();
        assert_eq!(copied, in_place);
    }

    #[test]
    fn in_place_falls_back_on_gzip() {
        let raw = br#"{"a":1}"#;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(1));
        encoder.write_all(raw).unwrap();
        let mut gz = encoder.finish().unwrap();
        let v = parse_tree_in_place(&mut gz).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn serialize_is_deterministic() {
        let v = serde_json::json!({"traceEvents": [{"ts": 1, "dur": 2, "args": {"k": "v"}}]});
        let a = serialize_tree(&v, true, 0).unwrap();
        let b = serialize_tree(&v, true, 4096).unwrap();
        assert_eq!(a, b, "gz bytes must not depend on the size hint");
        let raw_a = serialize_tree(&v, false, 0).unwrap();
        assert_eq!(raw_a, serde_json::to_vec(&v).unwrap());
    }

    #[test]
    fn gzipped_round_trips() {
        let v = serde_json::json!({"traceEvents": [{"ts": 100, "dur": 50}]});
        let bytes = serialize_tree(&v, true, 0).unwrap();
        assert_eq!(&bytes[..2], b"\x1f\x8b");

        let mut decoder = GzDecoder::new(&bytes[..]);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).unwrap();
        let back: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(back, v);
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Streaming (never-fully-resident) readers for the recording trace — a
//! hand-rolled streaming array reader so the recording never fully
//! materializes:
//!
//!   - `decompress_to_tmp`: decompress a `.gz` ONCE to an anonymous
//!     tempfile so the decompressed bytes live on disk, not RAM.
//!   - `ArrayElementReader`: pull reader yielding each top-level
//!     `traceEvents` array element's raw bytes (a tiny depth+string-state
//!     scanner — `serde_json` exposes no incremental array-element
//!     reader).
//!   - `read_recording_sparse`: builds the SPARSE events array the
//!     recording-step merge consumes (recording_merge.rs).

use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use flate2::read::GzDecoder;
use serde_json::Value;
use tempfile::tempfile;

const GZIP_MAGIC: &[u8; 2] = b"\x1f\x8b";

/// 1 MiB read buffer for the byte scanner.
const SCAN_BUF: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Decompress-to-tempfile.
// ---------------------------------------------------------------------------

/// Decompress `input` (gzip or raw JSON) ONCE into an anonymous tempfile,
/// rewound to byte 0. The decompressed bytes live on disk, not in RAM, so
/// the streaming reader scans them from disk without ever holding the
/// whole decompressed document resident.
pub(crate) fn decompress_to_tmp(input: &[u8]) -> Result<File> {
    let mut tmp = tempfile().context("create tempfile for decompressed trace")?;
    if input.starts_with(GZIP_MAGIC) {
        let mut decoder = GzDecoder::new(input);
        let mut buf = vec![0u8; SCAN_BUF];
        loop {
            let n = decoder.read(&mut buf).context("gzip decompress chunk")?;
            if n == 0 {
                break;
            }
            tmp.write_all(&buf[..n])
                .context("write decompressed chunk to tempfile")?;
        }
    } else {
        tmp.write_all(input).context("write raw JSON to tempfile")?;
    }
    tmp.seek(SeekFrom::Start(0))
        .context("rewind decompressed tempfile")?;
    Ok(tmp)
}

// ---------------------------------------------------------------------------
// Streaming array-element reader.
// ---------------------------------------------------------------------------

/// Pull reader over a JSON document of the shape
/// `{ <prefix-keys>, "traceEvents": [ <elem>, <elem>, ... ], <suffix-keys> }`.
///
/// Yields the raw bytes of each top-level element of the `traceEvents`
/// array one at a time. Captures the bytes BEFORE the `[` (`prefix`,
/// excluding the bracket) and AFTER the matching `]` (`suffix`, excluding
/// the bracket) so the writer can reproduce the envelope verbatim.
pub(crate) struct ArrayElementReader<R: Read> {
    inner: BufReader<R>,
    buf: Vec<u8>,
    pos: usize,
    filled: usize,
    eof: bool,
    /// Bytes consumed before the array's opening `[` (inclusive of nothing
    /// after it). Set once `open()` has run.
    pub(crate) prefix: Vec<u8>,
    /// Bytes after the array's closing `]`. Set once the array is drained.
    pub(crate) suffix: Vec<u8>,
    done: bool,
}

impl<R: Read> ArrayElementReader<R> {
    pub(crate) fn new(inner: R) -> Self {
        Self {
            inner: BufReader::with_capacity(SCAN_BUF, inner),
            buf: vec![0u8; SCAN_BUF],
            pos: 0,
            filled: 0,
            eof: false,
            prefix: Vec::new(),
            suffix: Vec::new(),
            done: false,
        }
    }

    /// Read one byte, refilling the buffer as needed. Returns None at EOF.
    fn next_byte(&mut self) -> Result<Option<u8>> {
        if self.pos >= self.filled {
            if self.eof {
                return Ok(None);
            }
            self.filled = self.inner.read(&mut self.buf).context("scan read")?;
            self.pos = 0;
            if self.filled == 0 {
                self.eof = true;
                return Ok(None);
            }
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(Some(b))
    }

    /// Consume bytes up to and including the array's opening `[`,
    /// recording everything before the `[` into `prefix`. We scan for the
    /// `traceEvents` key, then its `[`. To stay robust against `[` inside
    /// other string values we track string/escape state and only accept
    /// the `[` that immediately follows the `traceEvents` colon.
    pub(crate) fn open(&mut self) -> Result<()> {
        let needle = b"\"traceEvents\"";
        let mut match_len = 0usize;
        let mut in_string = false;
        let mut escape = false;
        // Scan to the end of the `traceEvents` key token.
        loop {
            let Some(b) = self.next_byte()? else {
                bail!("traceEvents key not found");
            };
            self.prefix.push(b);
            if escape {
                escape = false;
                match_len = 0;
                continue;
            }
            if in_string {
                if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    in_string = false;
                    // The closing quote may complete the needle.
                    if match_len + 1 == needle.len() && needle[match_len] == b'"' {
                        match_len = needle.len();
                    } else {
                        match_len = 0;
                    }
                    if match_len == needle.len() {
                        break;
                    }
                } else if match_len < needle.len() && needle[match_len] == b {
                    match_len += 1;
                } else {
                    match_len = 0;
                }
                continue;
            }
            // Not in string.
            if b == b'"' {
                in_string = true;
                if needle[0] == b'"' {
                    match_len = 1;
                } else {
                    match_len = 0;
                }
            }
        }
        // Now consume up to and including the `[` (skip whitespace + the
        // colon). Everything except the `[` goes into prefix.
        loop {
            let Some(b) = self.next_byte()? else {
                bail!("traceEvents array open bracket not found");
            };
            if b == b'[' {
                break;
            }
            self.prefix.push(b);
        }
        Ok(())
    }

    /// Yield the next element's raw bytes, or None when the array is
    /// exhausted (its `]` consumed). Captures the suffix on completion.
    pub(crate) fn next_element(&mut self) -> Result<Option<Vec<u8>>> {
        if self.done {
            return Ok(None);
        }
        // Skip leading whitespace and a separating comma.
        loop {
            let Some(b) = self.next_byte()? else {
                bail!("unexpected EOF inside traceEvents array");
            };
            match b {
                b' ' | b'\n' | b'\r' | b'\t' | b',' => continue,
                b']' => {
                    // End of array — capture suffix (rest of file).
                    self.capture_suffix()?;
                    self.done = true;
                    return Ok(None);
                }
                _ => {
                    // Start of an element. Scan it preserving structure.
                    let mut elem = Vec::with_capacity(256);
                    elem.push(b);
                    let mut depth: i32 = match b {
                        b'{' | b'[' => 1,
                        _ => 0,
                    };
                    let mut in_string = b == b'"';
                    let mut escape = false;
                    // For a scalar element (number/true/false/null with no
                    // braces) we must stop at the element boundary (comma /
                    // `]` / whitespace).
                    let is_container = depth == 1;
                    loop {
                        let Some(c) = self.next_byte()? else {
                            bail!("unexpected EOF mid-element");
                        };
                        if escape {
                            escape = false;
                            elem.push(c);
                            continue;
                        }
                        if in_string {
                            elem.push(c);
                            if c == b'\\' {
                                escape = true;
                            } else if c == b'"' {
                                in_string = false;
                                if !is_container && depth == 0 {
                                    // scalar string element complete
                                    return Ok(Some(elem));
                                }
                            }
                            continue;
                        }
                        match c {
                            b'"' => {
                                in_string = true;
                                elem.push(c);
                            }
                            b'{' | b'[' => {
                                depth += 1;
                                elem.push(c);
                            }
                            b'}' | b']' => {
                                if !is_container {
                                    // Scalar element terminated by the enclosing
                                    // array's `]` (e.g. a trailing number
                                    // `...,4]`). Push the bracket back so the
                                    // next next_element() sees the array-end, and
                                    // return the scalar without it — mirrors the
                                    // comma/whitespace boundary handling below.
                                    self.pos -= 1;
                                    return Ok(Some(elem));
                                }
                                depth -= 1;
                                elem.push(c);
                                if depth == 0 {
                                    return Ok(Some(elem));
                                }
                            }
                            b',' | b' ' | b'\n' | b'\r' | b'\t' => {
                                if !is_container {
                                    // scalar element boundary; push the
                                    // terminator back so next_element sees it.
                                    self.pos -= 1;
                                    return Ok(Some(elem));
                                }
                                elem.push(c);
                            }
                            _ => elem.push(c),
                        }
                    }
                }
            }
        }
    }

    fn capture_suffix(&mut self) -> Result<()> {
        // Drain any buffered bytes first.
        if self.pos < self.filled {
            self.suffix
                .extend_from_slice(&self.buf[self.pos..self.filled]);
            self.pos = self.filled;
        }
        // Then the rest of the underlying reader.
        let mut rest = Vec::new();
        self.inner
            .read_to_end(&mut rest)
            .context("read trace suffix")?;
        self.suffix.extend_from_slice(&rest);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Streaming recording reader — build the SPARSE recording array.
// ---------------------------------------------------------------------------

/// Stream the recording trace ONCE and build a SPARSE `traceEvents` array in
/// which only match-relevant events (kernel/memcpy/memset — see
/// `recording_merge::recording_event_is_relevant`) retain their parsed
/// `Value`; every other event is replaced by `Value::Null`. Original event
/// indices are preserved (the array has the same length as the full
/// recording), so the `event_idx` carried by matched recording kernels still
/// indexes correctly.
///
/// This is the merge path's dominant memory saving: the recording is ~3x the size
/// of the replay and is almost entirely cpu_op / ac2g / annotation / metadata
/// events, none of which the merge ever reads. By dropping them at parse time
/// the recording never fully materialises as a `serde_json::Value` tree — only
/// its (far smaller) kernel events stay resident.
///
/// `input` is the gzipped (or raw) recording bytes. Decompresses to a tempfile
/// first (bytes on disk, not RAM) then streams the array element-by-element,
/// holding one event `Value` at a time.
pub(crate) fn read_recording_sparse(input: &[u8]) -> Result<Vec<Value>> {
    let tmp = decompress_to_tmp(input)?;
    let mut reader = ArrayElementReader::new(&tmp);
    reader.open()?;
    let mut out: Vec<Value> = Vec::new();
    let mut idx = 0usize;
    while let Some(raw) = reader.next_element()? {
        let ev: Value = serde_json::from_slice(&raw)
            .with_context(|| format!("parse recording traceEvents element {idx}"))?;
        if super::recording_merge::recording_event_is_relevant(&ev) {
            out.push(ev);
        } else {
            out.push(Value::Null);
        }
        idx += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gz(raw: &[u8]) -> Vec<u8> {
        use flate2::write::GzEncoder;
        let mut e = GzEncoder::new(Vec::new(), flate2::Compression::new(1));
        e.write_all(raw).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn array_reader_splits_elements() {
        let doc = br#"{"x":1,"traceEvents":[{"a":1},{"b":[2,3]},"s",4],"y":2}"#;
        let mut r = ArrayElementReader::new(&doc[..]);
        r.open().unwrap();
        let mut got = Vec::new();
        while let Some(e) = r.next_element().unwrap() {
            got.push(String::from_utf8(e).unwrap());
        }
        assert_eq!(got, vec!["{\"a\":1}", "{\"b\":[2,3]}", "\"s\"", "4"]);
        assert!(r.suffix.starts_with(b","));
    }

    #[test]
    fn recording_sparse_keeps_only_relevant_events() {
        // Gzip input exercises decompress_to_tmp's decode branch; the sparse
        // array must preserve indices (Null for non-kernel events).
        let doc = br#"{"traceEvents":[{"ph":"X","cat":"kernel","tid":7},{"ph":"X","cat":"cpu_op"},{"ph":"X","cat":"gpu_memcpy"}]}"#;
        let out = read_recording_sparse(&gz(doc)).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out[0].is_object());
        assert!(out[1].is_null());
        assert!(out[2].is_object());
    }
}

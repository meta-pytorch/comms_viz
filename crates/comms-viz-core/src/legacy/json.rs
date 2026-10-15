// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! The Kineto Chrome-JSON trace format: the self-annotation envelope probe
//! and the row parsers.
//!
//! Three parser flavors feed the shared row model (`rows::Rows`), all
//! applying the same scalar-coercion table:
//! - **scan** ([`parse_bytes_walked`]): one raw-byte pass straight off the
//!   decompressed bytes, building rows + the walk-facing `RawEvent` view
//!   without materializing a DOM or a tape — the per-rank production path.
//! - **serde** ([`convert_serde_to_columns`]): converts an already-parsed
//!   `serde_json::Value` — the tree path's converter (the tree was needed
//!   anyway for the merged-trace artifact or a recording merge).
//! - **DOM** ([`convert_json_bytes_to_columns`]): simd-json borrowed-value
//!   parse — the no-walk passthrough converter.
//!
//! The probe ([`is_inline_annotated_bytes`]) streams the (possibly gzipped)
//! JSON prefix and scans TOP-LEVEL keys only: returns the truthiness of
//! `cudaGraphInlineAnnotated` if it appears before `traceEvents`; stops
//! (false) at `traceEvents`; false on malformed input. This is what tells
//! the pipeline that a trace annotated itself and needs no side channel —
//! see `crate::annotations::AnnotationSource::for_trace`.

use std::io::Read;

use anyhow::Result;
use anyhow::anyhow;
use serde_json::Value as SValue;
use simd_json::borrowed::Value as BValue;
use simd_json::prelude::*;

use super::splice::ElemSpans;
use super::splice::EvSpan;
use super::splice::read_key;
use super::splice::skip_string;
use super::splice::skip_value;
use super::splice::skip_ws;
use crate::rows::ConvertedColumns;
use crate::rows::ParsedRows;
use crate::rows::Rows;
use crate::rows::Span;
use crate::rows::arena_span;
use crate::rows::finish_columns;
use crate::rows::maybe_gunzip;
use crate::rows::pool_shells;
use crate::rows::pool_take_shells;
use crate::rows::stoll_prefix_bytes;

pub const INLINE_ANNOTATED_KEY: &str = "cudaGraphInlineAnnotated";

#[derive(serde::Deserialize)]
struct DistributedInfo {
    #[serde(default)]
    pg_config: Vec<crate::trace::ProcessGroupConfig>,
}

#[derive(serde::Deserialize)]
struct DistributedRank {
    rank: Option<i64>,
    world_size: Option<i64>,
}

/// Read up to `limit` decompressed bytes (lazy gzip decode like GzipFile).
/// Borrowed for already-decompressed input: the fused path calls this once
/// per rank on the decompressed buffer, where copying the 256KiB prefix was
/// pure waste (the scanner only reads).
fn head_bytes(replay_bytes: &[u8], limit: usize) -> std::borrow::Cow<'_, [u8]> {
    let is_gz = replay_bytes.len() >= 2 && replay_bytes[0] == 0x1f && replay_bytes[1] == 0x8b;
    if !is_gz {
        return std::borrow::Cow::Borrowed(&replay_bytes[..replay_bytes.len().min(limit)]);
    }
    let mut out = vec![0u8; limit];
    let mut n = 0;
    let mut dec = flate2::read::MultiGzDecoder::new(replay_bytes);
    while n < limit {
        match dec.read(&mut out[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(_) => break,
        }
    }
    out.truncate(n);
    std::borrow::Cow::Owned(out)
}

/// Read PyTorch's process-group inventory from the profiler envelope.
pub fn process_group_configs(replay_bytes: &[u8]) -> Vec<crate::trace::ProcessGroupConfig> {
    let mut scanner = Scanner {
        b: replay_bytes,
        i: 0,
    };
    if scanner.bump() != Some(b'{') {
        return Vec::new();
    }

    loop {
        match scanner.peek() {
            Some(b'}') | None => return Vec::new(),
            Some(b',') => {
                scanner.bump();
            }
            _ => {}
        }
        let Some(key) = scanner.parse_string() else {
            return Vec::new();
        };
        if scanner.bump() != Some(b':') {
            return Vec::new();
        }
        scanner.ws();
        let value_start = scanner.i;
        if scanner.skip_value().is_none() {
            return Vec::new();
        }
        if key == "distributedInfo" {
            return match serde_json::from_slice::<DistributedInfo>(
                &replay_bytes[value_start..scanner.i],
            ) {
                Ok(info) => info.pg_config,
                Err(e) => {
                    // Once per process: every rank of a run shares the schema.
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        eprintln!(
                            "comms_viz: ignoring unparseable distributedInfo.pg_config \
                             (falling back to comm-kernel metadata): {e}"
                        );
                    });
                    Vec::new()
                }
            };
        }
    }
}

struct Scanner<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Scanner<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.b.get(self.i).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.i += 1;
        Some(c)
    }
    fn parse_string(&mut self) -> Option<String> {
        if self.bump()? != b'"' {
            return None;
        }
        let mut s = Vec::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(s).ok(),
                b'\\' => {
                    let e = *self.b.get(self.i)?;
                    self.i += 1;
                    match e {
                        b'u' => {
                            // keep raw; keys we care about are ASCII
                            self.i += 4;
                            s.push(b'?');
                        }
                        b'n' => s.push(b'\n'),
                        b't' => s.push(b'\t'),
                        b'r' => s.push(b'\r'),
                        other => s.push(other),
                    }
                }
                other => s.push(other),
            }
        }
        None
    }
    /// Skip one JSON value; returns Some(truthy) for scalar values,
    /// Some(true) for non-empty containers... callers only use truthiness of
    /// scalars — containers report their python truthiness approximation.
    fn skip_value(&mut self) -> Option<bool> {
        match self.peek()? {
            b'{' | b'[' => {
                let open = self.bump()?;
                let close = if open == b'{' { b'}' } else { b']' };
                let mut depth = 1usize;
                let mut nonempty = false;
                while depth > 0 {
                    let c = self.bump()?;
                    match c {
                        b'"' => {
                            self.i -= 1;
                            self.parse_string()?;
                            nonempty = true;
                        }
                        c if c == open => depth += 1,
                        c if c == close => depth -= 1,
                        c if !(c as char).is_ascii_whitespace() && c != b',' && c != b':' => {
                            nonempty = true
                        }
                        _ => {}
                    }
                }
                Some(nonempty)
            }
            b'"' => {
                let s = self.parse_string()?;
                Some(!s.is_empty())
            }
            _ => {
                // literal: number / true / false / null
                let start = self.i;
                while self.i < self.b.len()
                    && !matches!(self.b[self.i], b',' | b'}' | b']')
                    && !(self.b[self.i] as char).is_ascii_whitespace()
                {
                    self.i += 1;
                }
                let tok = std::str::from_utf8(&self.b[start..self.i]).ok()?;
                Some(match tok {
                    "true" => true,
                    "false" | "null" => false,
                    n => n.parse::<f64>().map(|f| f != 0.0).unwrap_or(true),
                })
            }
        }
    }
}

/// `distributedInfo.rank` and `world_size` of a (possibly gzipped) trace
/// file. Reads only up to `distributedInfo`, which Kineto writes before
/// `traceEvents`; a writer that puts it later costs a full read.
pub fn trace_rank(path: &std::path::Path) -> Result<Option<(i64, Option<i64>)>> {
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0u8; 2];
    let n = file.read(&mut magic)?;
    let gz = magic[..n] == [0x1f, 0x8b];
    let file = std::io::Read::chain(&magic[..n], file);
    let mut reader: Box<dyn Read + '_> = if gz {
        Box::new(flate2::read::MultiGzDecoder::new(std::io::BufReader::new(
            file,
        )))
    } else {
        Box::new(file)
    };
    let mut buf = Vec::new();
    let mut want = 256 * 1024;
    loop {
        let n = reader.by_ref().take(want as u64).read_to_end(&mut buf)?;
        match head_rank(&buf) {
            Some(found) => return Ok(found),
            None if n == 0 => return Ok(None),
            None => want = buf.len(),
        }
    }
}

/// `Some(rank info)` once the top-level `distributedInfo` (or the end of the
/// envelope) is in `head`; `None` while it may still be ahead.
fn head_rank(head: &[u8]) -> Option<Option<(i64, Option<i64>)>> {
    let mut sc = Scanner { b: head, i: 0 };
    if sc.bump()? != b'{' {
        return Some(None);
    }
    loop {
        match sc.peek()? {
            b'}' => return Some(None),
            b',' => {
                sc.bump();
            }
            _ => {}
        }
        let key = sc.parse_string()?;
        if sc.bump()? != b':' {
            return Some(None);
        }
        sc.ws();
        let start = sc.i;
        sc.skip_value()?;
        if key == "distributedInfo" {
            return Some(
                serde_json::from_slice::<DistributedRank>(&head[start..sc.i])
                    .ok()
                    .and_then(|d| Some((d.rank?, d.world_size))),
            );
        }
    }
}

/// Probe the (possibly gzipped) envelope prefix for the self-annotation
/// flag; module docs give the exact key/truthiness rules.
pub fn is_inline_annotated_bytes(replay_bytes: &[u8]) -> bool {
    // The flag is written before traceEvents in a small envelope prefix;
    // 256 KiB is far beyond any observed envelope.
    let head = head_bytes(replay_bytes, 256 * 1024);
    let mut sc = Scanner { b: &head, i: 0 };
    if sc.bump() != Some(b'{') {
        return false;
    }
    loop {
        match sc.peek() {
            Some(b'}') | None => return false,
            Some(b',') => {
                sc.bump();
            }
            _ => {}
        }
        let Some(key) = sc.parse_string() else {
            return false;
        };
        if sc.bump() != Some(b':') {
            return false;
        }
        if key == "traceEvents" {
            return false;
        }
        let Some(truthy) = sc.skip_value() else {
            return false;
        };
        if key == INLINE_ANNOTATED_KEY {
            return truthy;
        }
    }
}

/// C++ `std::stoll` prefix parse: optional whitespace, sign, >=1 digit;
/// overflow → NULL (exception in C++).
fn stoll_prefix(s: &str) -> Option<i64> {
    stoll_prefix_bytes(s.as_bytes())
}

/// C++ `std::stod` prefix parse (strtod semantics): optional whitespace,
/// sign, then decimal/exponent form, or inf/infinity/nan.
fn stod_prefix(s: &str) -> Option<f64> {
    let t = s.trim_start();
    let bytes = t.as_bytes();
    let mut i = 0;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let rest = &t[i..];
    let lower = rest.to_ascii_lowercase();
    if lower.starts_with("infinity") {
        i += 8;
    } else if lower.starts_with("inf") || lower.starts_with("nan") {
        i += 3;
    } else {
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i == start || (i == start + 1 && bytes[start] == b'.') {
            return None; // no digits at all
        }
        // exponent
        if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
            let mut j = i + 1;
            if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                j += 1;
            }
            let dstart = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > dstart {
                i = j;
            }
        }
    }
    t[..i].parse::<f64>().ok()
}

/// C++ `std::to_string(double)` = sprintf "%f" (6 decimals).
fn cpp_to_string_double(v: f64) -> String {
    format!("{v:.6}")
}

/// Scalar → INT column value.
fn coerce_int(v: &BValue) -> Option<i64> {
    match v {
        BValue::Static(simd_json::StaticNode::I64(i)) => Some(*i),
        BValue::Static(simd_json::StaticNode::U64(u)) => Some(*u as i64), // wrap
        #[allow(clippy::cast_possible_truncation)]
        BValue::Static(simd_json::StaticNode::F64(f)) => Some(*f as i64), // trunc
        BValue::String(s) => stoll_prefix(s),
        _ => None, // bool / null / object / array → NULL
    }
}

/// Scalar → FLOAT column value.
fn coerce_float(v: &BValue) -> Option<f64> {
    match v {
        BValue::Static(simd_json::StaticNode::I64(i)) => Some(*i as f64),
        BValue::Static(simd_json::StaticNode::U64(u)) => Some(*u as f64),
        BValue::Static(simd_json::StaticNode::F64(f)) => Some(*f),
        BValue::String(s) => stod_prefix(s),
        _ => None,
    }
}

/// Scalar → STRING column value, appended to `arena`. Number formatting is
/// unchanged: `{i}`/`{u}` are the `to_string` Display paths, floats go
/// through `cpp_to_string_double`.
fn coerce_string(v: &BValue, arena: &mut String) -> Option<Span> {
    use std::fmt::Write as _;
    let off = arena.len();
    match v {
        BValue::String(s) => arena.push_str(s),
        BValue::Static(simd_json::StaticNode::I64(i)) => write!(arena, "{i}").unwrap(),
        BValue::Static(simd_json::StaticNode::U64(u)) => write!(arena, "{u}").unwrap(),
        BValue::Static(simd_json::StaticNode::F64(f)) => arena.push_str(&cpp_to_string_double(*f)),
        _ => return None, // bool → dropped → NULL
    }
    Some(arena_span(arena, off))
}

/// `grid` (ArrayExtractMode::FIRST): first array element captured raw then
/// coerced at buffer-append (double → trunc), string via the normal string
/// handler; empty array → NULL; scalar → normal coercion.
fn coerce_grid(v: &BValue) -> Option<i64> {
    match v {
        BValue::Array(arr) => arr.first().and_then(coerce_int),
        other => coerce_int(other),
    }
}

// --- serde_json tree coercions: the tree path converts its already-parsed
// merged tree directly. Semantically identical to the simd-json path: both
// parse correctly rounded. ---

fn s_coerce_int(v: &SValue) -> Option<i64> {
    match v {
        SValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else if let Some(u) = n.as_u64() {
                Some(u as i64) // wrap
            } else {
                #[allow(clippy::cast_possible_truncation)]
                n.as_f64().map(|f| f as i64) // trunc
            }
        }
        SValue::String(s) => stoll_prefix(s),
        _ => None,
    }
}

fn s_coerce_float(v: &SValue) -> Option<f64> {
    match v {
        SValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i as f64)
            } else if let Some(u) = n.as_u64() {
                Some(u as f64)
            } else {
                n.as_f64()
            }
        }
        SValue::String(s) => stod_prefix(s),
        _ => None,
    }
}

fn s_coerce_string(v: &SValue, arena: &mut String) -> Option<Span> {
    use std::fmt::Write as _;
    let off = arena.len();
    match v {
        SValue::String(s) => arena.push_str(s),
        SValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                write!(arena, "{i}").unwrap();
            } else if let Some(u) = n.as_u64() {
                write!(arena, "{u}").unwrap();
            } else {
                let f = n.as_f64()?;
                arena.push_str(&cpp_to_string_double(f));
            }
        }
        _ => return None,
    }
    Some(arena_span(arena, off))
}

fn s_coerce_grid(v: &SValue) -> Option<i64> {
    match v {
        SValue::Array(arr) => arr.first().and_then(s_coerce_int),
        other => s_coerce_int(other),
    }
}

fn parse_rows_serde(root: &SValue) -> Rows {
    let events: &[SValue] = match root.get("traceEvents") {
        Some(SValue::Array(a)) => a.as_slice(),
        _ => &[],
    };

    let mut rows = Rows {
        base_ns: root.get("baseTimeNanoseconds").and_then(|v| v.as_u64()),
        ..Default::default()
    };
    // Modest generic arena pre-size; reserve-doubling growth covers the rest.
    rows.arena.reserve(events.len().saturating_mul(32));

    for ev in events {
        let SValue::Object(obj) = ev else { continue };

        let cat = obj
            .get("cat")
            .and_then(|v| s_coerce_string(v, &mut rows.arena));
        let name = obj
            .get("name")
            .and_then(|v| s_coerce_string(v, &mut rows.arena));
        let ph = obj
            .get("ph")
            .and_then(|v| s_coerce_string(v, &mut rows.arena));
        let ts = obj.get("ts").and_then(s_coerce_float);
        let dur = obj.get("dur").and_then(s_coerce_float);

        let args_obj = match obj.get("args") {
            Some(SValue::Object(o)) => Some(o),
            _ => None,
        };

        let (mut stream, mut grid, mut pg_desc, mut pg_name) = (None, None, None, None);
        let mut collective_name = None;
        let (mut in_nelems, mut out_nelems, mut group_size, mut counter_value) =
            (None, None, None, None);

        if let Some(a) = args_obj {
            stream = a.get("stream").and_then(s_coerce_int);
            grid = a.get("grid").and_then(s_coerce_grid);
            pg_desc = a
                .get("Process Group Description")
                .and_then(|v| s_coerce_string(v, &mut rows.arena));
            pg_name = a
                .get("Process Group Name")
                .and_then(|v| s_coerce_string(v, &mut rows.arena));
            collective_name = a
                .get("Collective name")
                .and_then(|v| s_coerce_string(v, &mut rows.arena));
            in_nelems = a.get("In msg nelems").and_then(s_coerce_int);
            out_nelems = a.get("Out msg nelems").and_then(s_coerce_int);
            group_size = a.get("Group size").and_then(s_coerce_int);
            counter_value = a.get("").and_then(s_coerce_float);
        }

        let keep_comm = name.is_some_and(|s| crate::comm::is_comm(rows.get(s)))
            || matches!((collective_name, pg_name), (Some(c), Some(p))
                if crate::kernel_types::is_annotated_collective(rows.get(c), rows.get(p)));
        if keep_comm {
            let mut m = crate::comm::CommMetadata::default();
            if let Some(a) = args_obj {
                for (key, value) in a.iter() {
                    if crate::comm::CommMetadata::key_index(key).is_some() {
                        let value = serde_json::to_value(value).expect("JSON value");
                        if let Some(value) = crate::comm::json_text(&value) {
                            m.set(key, value);
                        }
                    }
                }
            }
            rows.comm.insert(rows.ts.len(), m);
        }
        rows.cat.push(cat);
        rows.name.push(name);
        rows.ts.push(ts);
        rows.dur.push(dur);
        rows.stream.push(stream);
        rows.grid.push(grid);
        rows.pg_desc.push(pg_desc);
        rows.pg_name.push(pg_name);
        rows.ph.push(ph);
        rows.collective_name.push(collective_name);
        rows.in_nelems.push(in_nelems);
        rows.out_nelems.push(out_nelems);
        rows.group_size.push(group_size);
        rows.counter_value.push(counter_value);
    }

    rows
}

fn parse_rows_simd(root: &BValue) -> Rows {
    let events = match root.get("traceEvents") {
        Some(BValue::Array(a)) => a.as_slice(),
        _ => &[],
    };

    let mut rows = Rows {
        base_ns: root.get("baseTimeNanoseconds").and_then(|v| v.as_u64()),
        ..Default::default()
    };
    // Modest generic arena pre-size; reserve-doubling growth covers the rest.
    rows.arena.reserve(events.len().saturating_mul(32));

    for ev in events {
        let BValue::Object(obj) = ev else { continue }; // non-objects → no row

        let cat = obj
            .get("cat")
            .and_then(|v| coerce_string(v, &mut rows.arena));
        let name = obj
            .get("name")
            .and_then(|v| coerce_string(v, &mut rows.arena));
        let ph = obj
            .get("ph")
            .and_then(|v| coerce_string(v, &mut rows.arena));
        let ts = obj.get("ts").and_then(coerce_float);
        let dur = obj.get("dur").and_then(coerce_float);

        let args = obj.get("args");
        let args_obj = match args {
            Some(BValue::Object(o)) => Some(o),
            _ => None,
        };

        let (mut stream, mut grid, mut pg_desc, mut pg_name) = (None, None, None, None);
        let mut collective_name = None;
        let (mut in_nelems, mut out_nelems, mut group_size, mut counter_value) =
            (None, None, None, None);

        if let Some(a) = args_obj {
            stream = a.get("stream").and_then(coerce_int);
            grid = a.get("grid").and_then(coerce_grid);
            pg_desc = a
                .get("Process Group Description")
                .and_then(|v| coerce_string(v, &mut rows.arena));
            pg_name = a
                .get("Process Group Name")
                .and_then(|v| coerce_string(v, &mut rows.arena));
            collective_name = a
                .get("Collective name")
                .and_then(|v| coerce_string(v, &mut rows.arena));
            in_nelems = a.get("In msg nelems").and_then(coerce_int);
            out_nelems = a.get("Out msg nelems").and_then(coerce_int);
            group_size = a.get("Group size").and_then(coerce_int);
            counter_value = a.get("").and_then(coerce_float);
        }

        let keep_comm = name.is_some_and(|s| crate::comm::is_comm(rows.get(s)))
            || matches!((collective_name, pg_name), (Some(c), Some(p))
                if crate::kernel_types::is_annotated_collective(rows.get(c), rows.get(p)));
        if keep_comm {
            let mut m = crate::comm::CommMetadata::default();
            if let Some(a) = args_obj {
                for (key, value) in a.iter() {
                    if crate::comm::CommMetadata::key_index(key).is_some() {
                        let value = serde_json::to_value(value).expect("JSON value");
                        if let Some(value) = crate::comm::json_text(&value) {
                            m.set(key, value);
                        }
                    }
                }
            }
            rows.comm.insert(rows.ts.len(), m);
        }
        rows.cat.push(cat);
        rows.name.push(name);
        rows.ts.push(ts);
        rows.dur.push(dur);
        rows.stream.push(stream);
        rows.grid.push(grid);
        rows.pg_desc.push(pg_desc);
        rows.pg_name.push(pg_name);
        rows.ph.push(ph);
        rows.collective_name.push(collective_name);
        rows.in_nelems.push(in_nelems);
        rows.out_nelems.push(out_nelems);
        rows.group_size.push(group_size);
        rows.counter_value.push(counter_value);
    }

    rows
}

/// Fast-path converters (columns only).
pub fn convert_serde_to_columns(root: &SValue) -> ConvertedColumns {
    let mut rows = parse_rows_serde(root);
    finish_columns(&mut rows)
}

pub fn convert_json_bytes_to_columns(bytes: &[u8]) -> Result<ConvertedColumns> {
    let mut raw = maybe_gunzip(bytes)?;
    let root = simd_json::to_borrowed_value(&mut raw)
        .map_err(|e| anyhow!("trace JSON parse failed: {e}"))?;
    let mut rows = parse_rows_simd(&root);
    Ok(finish_columns(&mut rows))
}

// ---------------------------------------------------------------------------
// Scan parser: one raw-byte pass fills the converter columns and the walk
// view directly — no tape, no structural index, no unescape buffer, and no
// in-place mutation, so the splice writer reads the same pristine input.
// The simd-json tape machinery held ~7x the document per in-flight rank
// (tape nodes + stage-1 index + string buffer + the defensive copy the
// in-place unescape forced); the memory gate divides the host budget by
// that live-set, so parser footprint — not instruction count — set the
// admitted width on memory-tight agents.
// ---------------------------------------------------------------------------

/// Scanned scalar, typed like the tape/DOM number nodes (the coercion
/// table keys off the type: I64/U64/F64 behave differently per column).
#[derive(Clone, Copy)]
enum Sc<'a> {
    /// Inner string bytes, escapes intact (unescaped on demand).
    Str(&'a [u8]),
    I64(i64),
    U64(u64),
    F64(f64),
    /// bool / null / object / array — NULL in every coercion.
    Other,
}

/// JSON-number grammar check. The scalar-token run is delimiter-based, so
/// junk like `nan` or `1.5x` would otherwise reach Rust's float parser and
/// diverge from the tape/DOM parsers, which reject it.
fn valid_json_number(t: &[u8]) -> bool {
    let mut i = 0;
    if t.first() == Some(&b'-') {
        i += 1;
    }
    let ds = i;
    while i < t.len() && t[i].is_ascii_digit() {
        i += 1;
    }
    if i == ds {
        return false;
    }
    if i < t.len() && t[i] == b'.' {
        i += 1;
        let fs = i;
        while i < t.len() && t[i].is_ascii_digit() {
            i += 1;
        }
        if i == fs {
            return false;
        }
    }
    if i < t.len() && (t[i] == b'e' || t[i] == b'E') {
        i += 1;
        if i < t.len() && (t[i] == b'+' || t[i] == b'-') {
            i += 1;
        }
        let es = i;
        while i < t.len() && t[i].is_ascii_digit() {
            i += 1;
        }
        if i == es {
            return false;
        }
    }
    i == t.len()
}

/// One scalar token (number/true/false/null) → typed scalar, with the tape
/// parser's number typing: integral fitting i64 → I64, else u64 → U64,
/// else (or fractional/exponent) → F64.
fn parse_token(t: &[u8]) -> Result<Sc<'static>> {
    match t {
        b"true" | b"false" | b"null" => return Ok(Sc::Other),
        _ => {}
    }
    if !valid_json_number(t) {
        anyhow::bail!("invalid scalar token");
    }
    let s = std::str::from_utf8(t).expect("digits are ASCII");
    if !t.iter().any(|&b| matches!(b, b'.' | b'e' | b'E')) {
        if let Ok(i) = s.parse::<i64>() {
            return Ok(Sc::I64(i));
        }
        if t[0] != b'-' {
            if let Ok(u) = s.parse::<u64>() {
                return Ok(Sc::U64(u));
            }
        }
    }
    Ok(Sc::F64(s.parse::<f64>().map_err(|e| anyhow!("{e}"))?))
}

/// JSON string unescape (slow path; escapes are vanishingly rare here).
#[cold]
fn unescape_into(raw: &[u8], out: &mut String) -> Result<()> {
    let mut i = 0;
    while i < raw.len() {
        if raw[i] != b'\\' {
            let end = memchr::memchr(b'\\', &raw[i..]).map_or(raw.len(), |k| i + k);
            out.push_str(std::str::from_utf8(&raw[i..end]).map_err(|_| anyhow!("invalid UTF-8"))?);
            i = end;
            continue;
        }
        let e = *raw.get(i + 1).ok_or_else(|| anyhow!("dangling escape"))?;
        i += 2;
        match e {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{0008}'),
            b'f' => out.push('\u{000C}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let hex4 = |b: &[u8]| -> Result<u16> {
                    let h = std::str::from_utf8(b).map_err(|_| anyhow!("bad \\u"))?;
                    u16::from_str_radix(h, 16).map_err(|_| anyhow!("bad \\u"))
                };
                let cp = hex4(raw.get(i..i + 4).ok_or_else(|| anyhow!("short \\u"))?)?;
                i += 4;
                if (0xD800..0xDC00).contains(&cp) {
                    // High surrogate: a \uXXXX low surrogate must follow.
                    anyhow::ensure!(
                        raw.get(i) == Some(&b'\\') && raw.get(i + 1) == Some(&b'u'),
                        "unpaired surrogate"
                    );
                    let lo = hex4(raw.get(i + 2..i + 6).ok_or_else(|| anyhow!("short \\u"))?)?;
                    anyhow::ensure!((0xDC00..0xE000).contains(&lo), "unpaired surrogate");
                    i += 6;
                    let c = 0x10000 + ((cp as u32 - 0xD800) << 10) + (lo as u32 - 0xDC00);
                    out.push(char::from_u32(c).ok_or_else(|| anyhow!("bad codepoint"))?);
                } else {
                    anyhow::ensure!(!(0xDC00..0xE000).contains(&cp), "unpaired surrogate");
                    out.push(char::from_u32(cp as u32).ok_or_else(|| anyhow!("bad codepoint"))?);
                }
            }
            _ => anyhow::bail!("bad escape"),
        }
    }
    Ok(())
}

/// Unescaped equality between raw string bytes and an ASCII literal.
#[inline]
fn sc_str_eq(raw: &[u8], want: &str) -> bool {
    if memchr::memchr(b'\\', raw).is_none() {
        raw == want.as_bytes()
    } else {
        sc_str_unescaped_pred(raw, |s| s == want)
    }
}

#[inline]
fn sc_str_starts_with(raw: &[u8], want: &str) -> bool {
    if memchr::memchr(b'\\', raw).is_none() {
        raw.starts_with(want.as_bytes())
    } else {
        sc_str_unescaped_pred(raw, |s| s.starts_with(want))
    }
}

#[cold]
fn sc_str_unescaped_pred(raw: &[u8], pred: impl FnOnce(&str) -> bool) -> bool {
    let mut s = String::new();
    unescape_into(raw, &mut s).is_ok() && pred(&s)
}

fn sc_coerce_int(v: Sc) -> Option<i64> {
    match v {
        Sc::I64(i) => Some(i),
        Sc::U64(u) => Some(u as i64), // wrap
        #[allow(clippy::cast_possible_truncation)]
        Sc::F64(f) => Some(f as i64), // trunc
        Sc::Str(raw) => {
            if memchr::memchr(b'\\', raw).is_none() {
                stoll_prefix_bytes(raw)
            } else {
                let mut s = String::new();
                unescape_into(raw, &mut s)
                    .ok()
                    .and_then(|()| stoll_prefix(&s))
            }
        }
        Sc::Other => None,
    }
}

fn sc_coerce_float(v: Sc) -> Option<f64> {
    match v {
        Sc::I64(i) => Some(i as f64),
        Sc::U64(u) => Some(u as f64),
        Sc::F64(f) => Some(f),
        Sc::Str(raw) => {
            if memchr::memchr(b'\\', raw).is_none() {
                std::str::from_utf8(raw).ok().and_then(stod_prefix)
            } else {
                let mut s = String::new();
                unescape_into(raw, &mut s)
                    .ok()
                    .and_then(|()| stod_prefix(&s))
            }
        }
        Sc::Other => None,
    }
}

fn sc_coerce_string(v: Sc, arena: &mut String) -> Option<Span> {
    use std::fmt::Write as _;
    let off = arena.len();
    match v {
        Sc::Str(raw) => {
            let ok = if memchr::memchr(b'\\', raw).is_none() {
                match std::str::from_utf8(raw) {
                    Ok(s) => {
                        arena.push_str(s);
                        true
                    }
                    Err(_) => false,
                }
            } else {
                unescape_into(raw, arena).is_ok()
            };
            if !ok {
                arena.truncate(off);
                return None;
            }
        }
        Sc::I64(i) => write!(arena, "{i}").unwrap(),
        Sc::U64(u) => write!(arena, "{u}").unwrap(),
        Sc::F64(f) => arena.push_str(&cpp_to_string_double(f)),
        Sc::Other => return None, // bool → dropped → NULL
    }
    Some(arena_span(arena, off))
}

/// `n_num_i64` twin: integer literals only (F64/string → None).
fn sc_num_i64(v: Option<Sc>) -> Option<i64> {
    match v {
        Some(Sc::I64(i)) => Some(i),
        Some(Sc::U64(u)) => i64::try_from(u).ok(),
        _ => None,
    }
}

/// `n_num_f64` twin: number literals only (string → None).
fn sc_num_f64(v: Option<Sc>) -> Option<f64> {
    match v {
        Some(Sc::I64(i)) => Some(i as f64),
        Some(Sc::U64(u)) => Some(u as f64),
        Some(Sc::F64(f)) => Some(f),
        _ => None,
    }
}

fn canon_bits(f: f64) -> u64 {
    if f.is_nan() {
        f64::NAN.to_bits()
    } else {
        f.to_bits()
    }
}

/// Scan the value at `i`: scalars parsed and typed, containers skipped
/// whole. Returns (scalar, index past the value).
fn scan_value<'a>(d: &'a [u8], i: usize) -> Result<(Sc<'a>, usize)> {
    match d.get(i) {
        Some(b'"') => {
            let end = skip_string(d, i)?;
            Ok((Sc::Str(&d[i + 1..end - 1]), end))
        }
        Some(b'{') | Some(b'[') => Ok((Sc::Other, skip_value(d, i)?)),
        Some(_) => {
            let end = skip_value(d, i)?;
            Ok((parse_token(&d[i..end])?, end))
        }
        None => anyhow::bail!("eof at value start"),
    }
}

/// `grid`: array → typed first element (empty → NULL), scalar → itself.
fn scan_grid<'a>(d: &'a [u8], i: usize) -> Result<(Sc<'a>, usize)> {
    if d.get(i) == Some(&b'[') {
        let end = skip_value(d, i)?;
        let j = skip_ws(d, i + 1);
        if d.get(j) == Some(&b']') {
            return Ok((Sc::Other, end));
        }
        let (first, _) = scan_value(d, j)?;
        return Ok((first, end));
    }
    scan_value(d, i)
}

/// The event fields the converter and the walk consume, captured per event.
#[derive(Default, Clone, Copy)]
struct EvCap<'a> {
    comm: [Option<&'a [u8]>; crate::comm::KEYS.len()],
    cat: Option<Sc<'a>>,
    name: Option<Sc<'a>>,
    ph: Option<Sc<'a>>,
    ts: Option<Sc<'a>>,
    dur: Option<Sc<'a>>,
    pid: Option<Sc<'a>>,
    tid: Option<Sc<'a>>,
    args_seen: bool,
    stream: Option<Sc<'a>>,
    grid: Option<Sc<'a>>,
    pg_desc: Option<Sc<'a>>,
    pg_name: Option<Sc<'a>>,
    collective_name: Option<Sc<'a>>,
    in_nelems: Option<Sc<'a>>,
    out_nelems: Option<Sc<'a>>,
    group_size: Option<Sc<'a>>,
    counter_value: Option<Sc<'a>>,
    aname: Option<Sc<'a>>,
    gnid: Option<Sc<'a>>,
}

impl EvCap<'_> {
    /// A replacing (last-wins) `args` discards the previous object's fields.
    fn clear_args(&mut self) {
        self.comm = [None; crate::comm::KEYS.len()];
        self.stream = None;
        self.grid = None;
        self.pg_desc = None;
        self.pg_name = None;
        self.collective_name = None;
        self.in_nelems = None;
        self.out_nelems = None;
        self.group_size = None;
        self.counter_value = None;
        self.aname = None;
        self.gnid = None;
    }
}

/// Keys may (pathologically) carry escapes; the tape compared unescaped.
#[inline]
fn key_norm<'a>(raw: &'a [u8], buf: &'a mut String) -> Result<&'a [u8]> {
    if memchr::memchr(b'\\', raw).is_none() {
        Ok(raw)
    } else {
        unescape_into(raw, buf)?;
        Ok(buf.as_bytes())
    }
}

/// Scan one `args` object, capturing the walked fields. `span`, when the
/// splice-writer capture is on, receives `args_has_keys` and the
/// `args.stream` value span — the fields `splice::scan_event` used to
/// re-derive with a second byte walk.
fn scan_args<'a, const FIRST_WINS: bool>(
    d: &'a [u8],
    i: usize,
    cap: &mut EvCap<'a>,
    mut span: Option<&mut EvSpan>,
) -> Result<usize> {
    debug_assert_eq!(d.get(i), Some(&b'{'));
    let mut j = skip_ws(d, i + 1);
    if d.get(j) == Some(&b'}') {
        return Ok(j + 1);
    }
    if let Some(sp) = span.as_deref_mut() {
        sp.args_has_keys = true;
    }
    macro_rules! setk {
        ($slot:expr) => {{
            let (v, e) = scan_value(d, j)?;
            if !(FIRST_WINS && $slot.is_some()) {
                $slot = Some(v);
            }
            e
        }};
    }
    loop {
        anyhow::ensure!(d.get(j) == Some(&b'"'), "expected key at {j}");
        let (rkey, vi) = read_key(d, j)?;
        let mut kbuf = String::new();
        let key = key_norm(rkey, &mut kbuf)?;
        j = vi;
        if let Some(idx) = std::str::from_utf8(key)
            .ok()
            .and_then(crate::comm::CommMetadata::key_index)
        {
            let end = skip_value(d, j)?;
            if !(FIRST_WINS && cap.comm[idx].is_some()) {
                cap.comm[idx] = Some(&d[j..end]);
            }
        }
        j = match key {
            b"stream" => {
                let (v, e) = scan_value(d, j)?;
                if !(FIRST_WINS && cap.stream.is_some()) {
                    if let Some(sp) = span.as_deref_mut() {
                        sp.stream = Some((j as u32, e as u32));
                    }
                    cap.stream = Some(v);
                }
                e
            }
            b"grid" => {
                let (v, e) = scan_grid(d, j)?;
                if !(FIRST_WINS && cap.grid.is_some()) {
                    cap.grid = Some(v);
                }
                e
            }
            b"Process Group Description" => setk!(cap.pg_desc),
            b"Process Group Name" => setk!(cap.pg_name),
            b"Collective name" => setk!(cap.collective_name),
            b"In msg nelems" => setk!(cap.in_nelems),
            b"Out msg nelems" => setk!(cap.out_nelems),
            b"Group size" => setk!(cap.group_size),
            b"" => setk!(cap.counter_value),
            b"name" => setk!(cap.aname),
            b"graph node id" => setk!(cap.gnid),
            _ => skip_value(d, j)?,
        };
        j = skip_ws(d, j);
        match d.get(j) {
            Some(b',') => j = skip_ws(d, j + 1),
            Some(b'}') => return Ok(j + 1),
            _ => anyhow::bail!("expected ',' or '}}' at {j}"),
        }
    }
}

/// Scan one event object starting at `i` (its `{`). `span`, when the
/// splice-writer capture is on, receives the inner value spans
/// (`ts`/`tid`/`args`/`args.stream`) as they are matched — the same fields,
/// with the same last-wins overwrites, that `splice::scan_event` derived
/// with a second walk over these bytes.
fn scan_event<'a, const FIRST_WINS: bool>(
    d: &'a [u8],
    i: usize,
    mut span: Option<&mut EvSpan>,
) -> Result<(EvCap<'a>, usize)> {
    let mut cap = EvCap::default();
    let mut j = skip_ws(d, i + 1);
    if d.get(j) == Some(&b'}') {
        return Ok((cap, j + 1));
    }
    if let Some(sp) = span.as_deref_mut() {
        sp.has_keys = true;
    }
    macro_rules! setk {
        ($slot:expr) => {{
            let (v, e) = scan_value(d, j)?;
            if !(FIRST_WINS && $slot.is_some()) {
                $slot = Some(v);
            }
            e
        }};
    }
    loop {
        anyhow::ensure!(d.get(j) == Some(&b'"'), "expected key at {j}");
        let (rkey, vi) = read_key(d, j)?;
        let mut kbuf = String::new();
        let key = key_norm(rkey, &mut kbuf)?;
        j = vi;
        j = match key {
            b"cat" => setk!(cap.cat),
            b"name" => setk!(cap.name),
            b"ph" => setk!(cap.ph),
            b"ts" => {
                let (v, e) = scan_value(d, j)?;
                if !(FIRST_WINS && cap.ts.is_some()) {
                    if let Some(sp) = span.as_deref_mut() {
                        // "integer literal" = parses as i64/u64, matching the
                        // tree twin's `v.is_i64() || v.is_u64()`.
                        sp.ts = Some((j as u32, e as u32, matches!(v, Sc::I64(_) | Sc::U64(_))));
                    }
                    cap.ts = Some(v);
                }
                e
            }
            b"dur" => setk!(cap.dur),
            b"pid" => setk!(cap.pid),
            b"tid" => {
                let (v, e) = scan_value(d, j)?;
                if !(FIRST_WINS && cap.tid.is_some()) {
                    if let Some(sp) = span.as_deref_mut() {
                        sp.tid = Some((j as u32, e as u32));
                    }
                    cap.tid = Some(v);
                }
                e
            }
            b"args" => {
                if FIRST_WINS && cap.args_seen {
                    skip_value(d, j)?
                } else {
                    cap.args_seen = true;
                    cap.clear_args(); // last-wins: replace wholesale
                    if d.get(j) == Some(&b'{') {
                        if let Some(sp) = span.as_deref_mut() {
                            sp.args_open = Some(j as u32);
                            sp.args_has_keys = false;
                            sp.stream = None;
                        }
                        scan_args::<FIRST_WINS>(d, j, &mut cap, span.as_deref_mut())?
                    } else {
                        // Non-object args → absent (field-for-field what
                        // splice::scan_event records, incl. stale has_keys).
                        if let Some(sp) = span.as_deref_mut() {
                            sp.args_open = None;
                            sp.stream = None;
                        }
                        skip_value(d, j)?
                    }
                }
            }
            _ => skip_value(d, j)?,
        };
        j = skip_ws(d, j);
        match d.get(j) {
            Some(b',') => j = skip_ws(d, j + 1),
            Some(b'}') => return Ok((cap, j + 1)),
            _ => anyhow::bail!("expected ',' or '}}' at {j}"),
        }
    }
}

/// Converter columns + (walk callers) the raw walk view — identical
/// coercions to the DOM paths, straight off the captured scalars.
fn push_row(cap: &EvCap, rows: &mut Rows, raws: Option<&mut Vec<crate::cudagraph::RawEvent>>) {
    let cat = cap.cat.and_then(|v| sc_coerce_string(v, &mut rows.arena));
    let name = cap.name.and_then(|v| sc_coerce_string(v, &mut rows.arena));
    let pg_desc = cap
        .pg_desc
        .and_then(|v| sc_coerce_string(v, &mut rows.arena));
    let pg_name = cap
        .pg_name
        .and_then(|v| sc_coerce_string(v, &mut rows.arena));
    let collective_name = cap
        .collective_name
        .and_then(|v| sc_coerce_string(v, &mut rows.arena));
    let keep_comm = name.is_some_and(|s| crate::comm::is_comm(rows.get(s)))
        || matches!((collective_name, pg_name), (Some(c), Some(p))
            if crate::kernel_types::is_annotated_collective(rows.get(c), rows.get(p)));
    if keep_comm {
        let mut m = crate::comm::CommMetadata::default();
        for (i, raw) in cap.comm.iter().enumerate() {
            m.0[i] = raw
                .and_then(|r| serde_json::from_slice(r).ok())
                .as_ref()
                .and_then(crate::comm::json_text);
        }
        rows.comm.insert(rows.ts.len(), m);
    }
    let ph = cap.ph.and_then(|v| sc_coerce_string(v, &mut rows.arena));
    rows.cat.push(cat);
    rows.name.push(name);
    rows.ts.push(cap.ts.and_then(sc_coerce_float));
    rows.dur.push(cap.dur.and_then(sc_coerce_float));
    rows.stream.push(cap.stream.and_then(sc_coerce_int));
    rows.grid.push(cap.grid.and_then(sc_coerce_int));
    rows.pg_desc.push(pg_desc);
    rows.pg_name.push(pg_name);
    rows.ph.push(ph);
    rows.collective_name.push(collective_name);
    rows.in_nelems.push(cap.in_nelems.and_then(sc_coerce_int));
    rows.out_nelems.push(cap.out_nelems.and_then(sc_coerce_int));
    rows.group_size.push(cap.group_size.and_then(sc_coerce_int));
    rows.counter_value
        .push(cap.counter_value.and_then(sc_coerce_float));

    // Raw walk view (`tree_walks` field semantics) — walk callers only.
    let Some(raws) = raws else { return };
    let cat_s = match cap.cat {
        Some(Sc::Str(s)) => Some(s),
        _ => None,
    };
    let ph_s = match cap.ph {
        Some(Sc::Str(s)) => Some(s),
        _ => None,
    };
    let name_s = match cap.name {
        Some(Sc::Str(s)) => Some(s),
        _ => None,
    };
    let ph_is_m = ph_s.is_some_and(|s| sc_str_eq(s, "M"));
    let is_thread_name_meta = ph_is_m && name_s.is_some_and(|s| sc_str_eq(s, "thread_name"));
    let meta_stream_name = is_thread_name_meta
        && match cap.aname {
            Some(Sc::Str(s)) => sc_str_starts_with(s, "stream "),
            _ => false,
        };
    raws.push(crate::cudagraph::RawEvent {
        tid_i: sc_num_i64(cap.tid).unwrap_or(0),
        tid_bits: sc_num_f64(cap.tid).map(canon_bits),
        ts_bits: sc_num_f64(cap.ts).map(canon_bits),
        ts_f: sc_num_f64(cap.ts).unwrap_or(0.0),
        dur_f: sc_num_f64(cap.dur).unwrap_or(0.0),
        pid_i: sc_num_i64(cap.pid).unwrap_or(0),
        ph_is_m,
        ph_is_f: ph_s.is_some_and(|s| sc_str_eq(s, "f")),
        cat_is_ac2g: cat_s.is_some_and(|s| sc_str_eq(s, "ac2g")),
        cat_is_work: cat_s.is_some_and(|s| WALK_WORK.iter().any(|w| sc_str_eq(s, w))),
        cat_is_gua: cat_s.is_some_and(|s| sc_str_eq(s, "gpu_user_annotation")),
        cat_is_kernel: cat_s.is_some_and(|s| sc_str_eq(s, "kernel")),
        is_thread_name_meta,
        meta_stream_name,
        gnid: sc_num_i64(cap.gnid).unwrap_or(0),
    });
}

/// Scan the winning `traceEvents` array: one row per OBJECT element, in
/// document order (non-objects → no row, like the DOM paths) — the row/
/// event alignment the splice writer cross-checks. `bcap` records every
/// element's byte span for the splice writer (bounds capture).
fn scan_events<const FIRST_WINS: bool>(
    d: &[u8],
    i: usize,
    rows: &mut Rows,
    mut raws: Option<&mut Vec<crate::cudagraph::RawEvent>>,
    mut bcap: Option<&mut BoundsCap>,
) -> Result<usize> {
    debug_assert_eq!(d.get(i), Some(&b'['));
    let mut j = skip_ws(d, i + 1);
    if d.get(j) == Some(&b']') {
        return Ok(j + 1);
    }
    loop {
        let s = j;
        let mut span = bcap.as_deref_mut().map(|_| EvSpan {
            start: s as u32,
            ..Default::default()
        });
        if d.get(j) == Some(&b'{') {
            if let Some(sp) = span.as_mut() {
                sp.is_object = true;
            }
            let (cap, e) = scan_event::<FIRST_WINS>(d, j, span.as_mut())?;
            push_row(&cap, rows, raws.as_deref_mut());
            j = e;
        } else {
            j = skip_value(d, j)?; // non-object event: no row
        }
        if let Some(c) = bcap.as_deref_mut() {
            let mut sp = span.expect("span exists iff bcap does");
            sp.end = j as u32;
            c.n_objects += sp.is_object as u32;
            c.events.push(sp);
        }
        j = skip_ws(d, j);
        match d.get(j) {
            Some(b',') => j = skip_ws(d, j + 1),
            Some(b']') => return Ok(j + 1),
            _ => anyhow::bail!("expected ',' or ']' at {j}"),
        }
    }
}

/// Root walk. Duplicate-key semantics follow the converter each flavor
/// replaces — the serde tree (walk oracle) is last-insert-wins, the
/// simd-json borrowed DOM (passthrough oracle) is FIRST-wins at every
/// level (root, event, args); real traces carry no duplicate keys, so the
/// flavors only diverge on pathological input. Structure is validated on
/// the walked path (unterminated containers/strings and malformed scalar
/// tokens fail the rank, like the parsers this replaces); a non-object
/// root yields zero rows, like `root.get("traceEvents")` failing.
fn scan_doc<const FIRST_WINS: bool>(
    d: &[u8],
    rows: &mut Rows,
    mut raws: Option<&mut Vec<crate::cudagraph::RawEvent>>,
    arena_hint: usize,
    mut bcap: Option<&mut BoundsCap>,
) -> Result<()> {
    // Event-count estimate for the column reserve (Kineto events run a few
    // hundred bytes); the pooled shells make this a first-rank concern only.
    let n_est = d.len() / 256;
    rows.recycle(n_est, arena_hint);
    if let Some(raws) = raws.as_deref_mut() {
        raws.clear();
        raws.reserve(n_est);
    }
    let mut i = skip_ws(d, 0);
    anyhow::ensure!(i < d.len(), "empty document");
    if d[i] != b'{' {
        // Non-object root → no rows, but the document must still be ONE
        // well-formed value (the parsers this replaces validated the whole
        // document; corrupt bytes must keep failing the rank).
        let (_, e) = scan_value(d, i)?;
        anyhow::ensure!(skip_ws(d, e) == d.len(), "trailing bytes at {e}");
        return Ok(());
    }
    i = skip_ws(d, i + 1);
    if d.get(i) == Some(&b'}') {
        anyhow::ensure!(skip_ws(d, i + 1) == d.len(), "trailing bytes");
        return Ok(());
    }
    let mut te_seen = false;
    let mut base_seen = false;
    let mut base_ns = None;
    loop {
        anyhow::ensure!(d.get(i) == Some(&b'"'), "expected key at {i}");
        let (rkey, vi) = read_key(d, i)?;
        let mut kbuf = String::new();
        let key = key_norm(rkey, &mut kbuf)?;
        i = vi;
        if key == b"baseTimeNanoseconds" && !(FIRST_WINS && base_seen) {
            base_seen = true;
            let (v, end) = scan_value(d, i)?;
            base_ns = match v {
                Sc::U64(v) => Some(v),
                Sc::I64(v) => u64::try_from(v).ok(),
                _ => None,
            };
            i = end;
        } else if key == b"traceEvents" && !(FIRST_WINS && te_seen) {
            te_seen = true;
            // Last-wins: a later traceEvents replaces rows parsed so far.
            rows.recycle(n_est, arena_hint);
            if let Some(raws) = raws.as_deref_mut() {
                raws.clear();
            }
            if d.get(i) == Some(&b'[') {
                if let Some(c) = bcap.as_deref_mut() {
                    c.events.clear();
                    c.n_objects = 0;
                    c.arr_open = i as u32;
                    c.valid = true;
                }
                i = scan_events::<FIRST_WINS>(
                    d,
                    i,
                    rows,
                    raws.as_deref_mut(),
                    bcap.as_deref_mut(),
                )?;
                if let Some(c) = bcap.as_deref_mut() {
                    // scan_events returned just past the `]`.
                    c.arr_close = (i - 1) as u32;
                }
            } else {
                if let Some(c) = bcap.as_deref_mut() {
                    c.events.clear();
                    c.n_objects = 0;
                    c.valid = false;
                }
                i = skip_value(d, i)?; // non-array traceEvents → no rows
            }
        } else {
            i = skip_value(d, i)?;
        }
        i = skip_ws(d, i);
        match d.get(i) {
            Some(b',') => i = skip_ws(d, i + 1),
            Some(b'}') => {
                anyhow::ensure!(skip_ws(d, i + 1) == d.len(), "trailing bytes at {i}");
                rows.base_ns = base_ns;
                return Ok(());
            }
            _ => anyhow::bail!("expected ',' or '}}' at {i}"),
        }
    }
}

const WALK_WORK: &[&str] = &["kernel", "gpu_memcpy", "gpu_memset"];

/// Phase 1 of the tree-free conversion: one raw-byte scan fills the rows
/// and the raw walk view — no DOM, no tape, and the input is NOT mutated,
/// so the splice writer reads the same buffer (the pristine-copy pool died
/// with the tape parse). The live graph ids come straight off the parsed
/// rows (`live_gids`); the parse extracts every event's gnid anyway.
pub fn parse_bytes_walked(decompressed: &[u8]) -> Result<ParsedRows> {
    // Arena capacity heuristic: string-column bytes run well under
    // 1/8 of the decompressed JSON (keys, numbers, punctuation and
    // non-column values dominate); shortfall grows by doubling.
    let arena_hint = decompressed.len() / 8;
    let (mut rows, mut raws) = pool_take_shells();
    scan_doc::<false>(decompressed, &mut rows, Some(&mut raws), arena_hint, None)
        .map_err(|e| anyhow!("trace JSON parse failed: {e}"))?;
    Ok(ParsedRows { rows, raws })
}

/// Splice-writer span capture (see [`ElemSpans`]).
#[derive(Default)]
struct BoundsCap {
    events: Vec<EvSpan>,
    arr_open: u32,
    arr_close: u32,
    n_objects: u32,
    /// The winning traceEvents value was an array and was captured whole.
    valid: bool,
}

/// [`parse_bytes_walked`] + full element spans for the splice writer. The
/// scan visits every element's bytes anyway; recording the spans here
/// removes the splice's second structural pass over the document outright.
/// `None` when there is no splice-able traceEvents array or the document
/// exceeds u32 span indices — those ranks keep the old sparse-scan (and its
/// tree-fallback) behavior.
pub fn parse_bytes_walked_bounds(decompressed: &[u8]) -> Result<(ParsedRows, Option<ElemSpans>)> {
    let arena_hint = decompressed.len() / 8;
    let (mut rows, mut raws) = pool_take_shells();
    let mut bcap = BoundsCap::default();
    let cap = (decompressed.len() <= u32::MAX as usize).then_some(&mut bcap);
    scan_doc::<false>(decompressed, &mut rows, Some(&mut raws), arena_hint, cap)
        .map_err(|e| anyhow!("trace JSON parse failed: {e}"))?;
    let bounds = bcap.valid.then_some(ElemSpans {
        events: bcap.events,
        arr_open: bcap.arr_open,
        arr_close: bcap.arr_close,
        n_objects: bcap.n_objects,
    });
    Ok((ParsedRows { rows, raws }, bounds))
}

/// Columns-only scan for the passthrough (no-walk) path: same output as
/// `convert_json_bytes_to_columns` — the coercion table is shared and
/// `--verify-walks` asserts bitwise equality — with the borrowed DOM's
/// FIRST-wins duplicate-key semantics and none of its allocation churn.
pub fn convert_json_scan_to_columns(decompressed: &[u8]) -> Result<ConvertedColumns> {
    let arena_hint = decompressed.len() / 8;
    let (mut rows, raws) = pool_take_shells();
    scan_doc::<true>(decompressed, &mut rows, None, arena_hint, None)
        .map_err(|e| anyhow!("trace JSON parse failed: {e}"))?;
    let cols = finish_columns(&mut rows);
    pool_shells(rows, raws);
    Ok(cols)
}

/// Bitwise column comparison for --verify-walks.
pub fn assert_columns_equal(a: &ConvertedColumns, b: &ConvertedColumns) -> Result<()> {
    anyhow::ensure!(a.base_ns == b.base_ns, "clock origin differs");
    for i in 0..a.ts.len() {
        anyhow::ensure!(
            a.comm(i) == b.comm(i),
            "communication metadata differs at row {i}: {:?} != {:?}",
            a.comm(i),
            b.comm(i)
        );
    }
    anyhow::ensure!(
        a.ts.len() == b.ts.len(),
        "row count {} != {}",
        a.ts.len(),
        b.ts.len()
    );
    for i in 0..a.ts.len() {
        anyhow::ensure!(
            a.cat(i) == b.cat(i),
            "cat[{i}]: {:?} != {:?}",
            a.cat(i),
            b.cat(i)
        );
        anyhow::ensure!(
            a.name(i) == b.name(i),
            "name[{i}]: {:?} != {:?}",
            a.name(i),
            b.name(i)
        );
        anyhow::ensure!(
            a.ts[i].to_bits() == b.ts[i].to_bits(),
            "ts[{i}]: {:?} != {:?}",
            a.ts[i],
            b.ts[i]
        );
        anyhow::ensure!(
            a.dur[i].to_bits() == b.dur[i].to_bits(),
            "dur[{i}]: {:?} != {:?}",
            a.dur[i],
            b.dur[i]
        );
        anyhow::ensure!(
            a.stream[i] == b.stream[i],
            "stream[{i}]: {} != {}",
            a.stream[i],
            b.stream[i]
        );
        anyhow::ensure!(a.grid[i] == b.grid[i], "grid[{i}]");
        anyhow::ensure!(
            a.pg_desc(i) == b.pg_desc(i),
            "pg_desc[{i}]: {:?} != {:?}",
            a.pg_desc(i),
            b.pg_desc(i)
        );
        anyhow::ensure!(
            a.pg_name(i) == b.pg_name(i),
            "pg_name[{i}]: {:?} != {:?}",
            a.pg_name(i),
            b.pg_name(i)
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> ConvertedColumns {
        convert_json_bytes_to_columns(json.as_bytes()).unwrap()
    }

    #[test]
    fn basic_event() {
        let cols = parse(
            r#"{"traceEvents": [
                {"cat": "kernel", "name": "k1", "ts": 100, "dur": 5, "pid": 1, "tid": 2,
                 "args": {"stream": 7, "grid": [128, 1, 1], "Group size": 8}}
            ]}"#,
        );
        assert_eq!(cols.ts.len(), 1);
        assert_eq!(cols.grid[0], 128);
        assert_eq!(cols.stream[0], 7);
        assert_eq!(cols.ts[0], 100.0);
        assert_eq!(cols.cat(0), Some("kernel"));
        assert_eq!(cols.pg_desc(0), "");
    }

    #[test]
    fn coercions() {
        assert_eq!(stoll_prefix("123abc"), Some(123));
        assert_eq!(stoll_prefix("  -42"), Some(-42));
        assert_eq!(stoll_prefix("123.5"), Some(123));
        assert_eq!(stoll_prefix("abc"), None);
        assert_eq!(stoll_prefix(""), None);
        assert_eq!(stod_prefix("1.5x"), Some(1.5));
        assert_eq!(stod_prefix("x"), None);
        assert_eq!(cpp_to_string_double(1.5), "1.500000");
    }

    #[test]
    fn sort_order_zero_dur_first() {
        // Same ts: null dur first, then dur==0, then dur desc; stable ties.
        let cols = parse(
            r#"{"traceEvents": [
                {"cat": "a", "name": "wide", "ts": 10, "dur": 5},
                {"cat": "b", "name": "zero", "ts": 10, "dur": 0},
                {"cat": "c", "name": "nodur", "ts": 10},
                {"cat": "d", "name": "narrow", "ts": 10, "dur": 2},
                {"cat": "e", "name": "early", "ts": 5, "dur": 1}
            ]}"#,
        );
        let names: Vec<&str> = (0..cols.ts.len()).map(|i| cols.name(i).unwrap()).collect();
        assert_eq!(names, vec!["early", "nodur", "zero", "wide", "narrow"]);
    }

    #[test]
    fn metadata_events_kept_with_null_cat() {
        let cols = parse(
            r#"{"traceEvents": [
                {"ph": "M", "name": "thread_name", "pid": 3, "tid": 9, "args": {"name": "stream 7"}},
                {"cat": "kernel", "name": "k", "ts": 1, "dur": 1}
            ]}"#,
        );
        assert_eq!(cols.ts.len(), 2);
        // M event: no ts → sorts first (null AtStart); cat is NULL
        assert_eq!(cols.cat(0), None);
        assert_eq!(cols.name(0), Some("thread_name"));
    }

    /// The passthrough scan converter must be indistinguishable from the DOM
    /// converter (same coercions, same duplicate-key and non-object-event
    /// semantics, same sort) — this is the offline half of the
    /// `--verify-walks` oracle in pipeline.rs.
    #[test]
    fn scan_columns_match_dom_columns() {
        let cases: &[&str] = &[
            // Plain event incl. args scalars, grid array, unknown keys.
            r#"{"traceEvents": [
                {"cat": "kernel", "name": "k1", "ts": 100, "dur": 5, "pid": 1, "tid": 2,
                 "args": {"stream": 7, "grid": [128, 1, 1], "Group size": 8}}
            ]}"#,
            // Non-object events; missing fields; null/bool values; empty grid.
            r#"{"traceEvents": [
                42, "str", [1,2], null,
                {"name": "a"},
                {"cat": null, "name": true, "ts": "123.5abc", "dur": "x"},
                {"args": {"grid": [], "stream": "9z", "Process Group Name": 3.5}}
            ]}"#,
            // Duplicate keys: event-level, args-level, and traceEvents itself
            // (last wins everywhere, like the DOM's map insert).
            r#"{"traceEvents": [{"ts": 1, "ts": 2, "args": {"stream": 1}, "args": {"stream": 5}}],
                "traceEvents": [{"ts": 9, "name": "last", "args": 7, "args": {"grid": 3}}]}"#,
            // args object then non-object (absent), nested containers to skip,
            // numeric strings, u64 wrap, unicode escapes.
            r#"{"traceEvents": [
                {"name": "né", "ts": 18446744073709551615, "dur": 1.25,
                 "args": {"deep": {"a": [1, {"b": 2}]}, "grid": "12abc",
                          "Process Group Description": "tp\t1"}},
                {"ts": 3, "args": {"stream": 2}, "args": null}
            ]}"#,
            // Same-ts sort ties (null dur first, dur 0, then desc).
            r#"{"traceEvents": [
                {"name": "wide", "ts": 10, "dur": 5}, {"name": "zero", "ts": 10, "dur": 0},
                {"name": "nodur", "ts": 10}, {"name": "narrow", "ts": 10, "dur": 2}
            ]}"#,
            // Degenerate roots.
            r#"{}"#,
            r#"{"traceEvents": {}}"#,
            r#"{"traceEvents": []}"#,
            r#"[1, 2, 3]"#,
            r#""scalar""#,
        ];
        for json in cases {
            let dom = convert_json_bytes_to_columns(json.as_bytes()).unwrap();
            let scan = convert_json_scan_to_columns(json.as_bytes()).unwrap();
            assert_columns_equal(&scan, &dom)
                .unwrap_or_else(|e| panic!("scan != DOM for {json}: {e}"));
        }
    }

    /// Walk-flavor scan: last-wins duplicate keys (the serde-tree oracle's
    /// semantics) and the RawEvent field/flag extraction.
    #[test]
    fn scan_walk_flavor_semantics() {
        let parsed = parse_bytes_walked(
            r#"{"traceEvents": [{"ts": 1}],
                "traceEvents": [
                 {"cat": "kernel", "name": "k", "ph": "X", "ts": 10.5, "dur": 2,
                  "pid": 7, "tid": 3.5,
                  "args": {"stream": 20, "graph node id": 42}},
                 {"ph": "M", "name": "thread_name", "args": {"name": "stream 20"}},
                 {"cat": "ac2g", "ph": "f", "ts": 1, "ts": 2},
                 {"args": {"stream": 1}, "args": {"stream": 5}}
            ]}"#
            .as_bytes(),
        )
        .unwrap();
        // Last traceEvents wins; 4 rows.
        assert_eq!(parsed.rows.ts.len(), 4);
        assert_eq!(parsed.raws.len(), 4);
        let r0 = &parsed.raws[0];
        assert!(r0.cat_is_kernel && r0.cat_is_work && !r0.ph_is_m);
        assert_eq!(r0.gnid, 42);
        assert_eq!(r0.pid_i, 7);
        // tid 3.5 is not an integer literal → tid_i falls to 0, bits kept.
        assert_eq!(r0.tid_i, 0);
        assert_eq!(r0.tid_bits, Some(3.5f64.to_bits()));
        assert_eq!(r0.ts_f, 10.5);
        let r1 = &parsed.raws[1];
        assert!(r1.ph_is_m && r1.is_thread_name_meta && r1.meta_stream_name);
        let r2 = &parsed.raws[2];
        assert!(r2.cat_is_ac2g && r2.ph_is_f);
        assert_eq!(r2.ts_f, 2.0); // duplicate ts: last wins
        assert_eq!(parsed.rows.stream[3], Some(5)); // duplicate args: last wins
    }

    /// Corrupt bytes must fail the parse (rank error), not yield 0 rows.
    #[test]
    fn scan_rejects_garbage() {
        assert!(parse_bytes_walked(b"\x1f\x8bnot really gzip").is_err());
        assert!(parse_bytes_walked(b"").is_err());
        assert!(parse_bytes_walked(br#"{"traceEvents": [}"#).is_err());
        assert!(parse_bytes_walked(br#"{"traceEvents": [{"ts": nan}]}"#).is_err());
        assert!(parse_bytes_walked(br#"{"a": 1} trailing"#).is_err());
        assert!(convert_json_scan_to_columns(b"\x1f\x8bgarbage").is_err());
    }

    #[test]
    fn detects_flag() {
        assert!(is_inline_annotated_bytes(
            br#"{"schemaVersion": 1, "cudaGraphInlineAnnotated": true, "traceEvents": []}"#
        ));
        assert!(!is_inline_annotated_bytes(
            br#"{"schemaVersion": 1, "cudaGraphInlineAnnotated": false, "traceEvents": []}"#
        ));
        assert!(!is_inline_annotated_bytes(
            br#"{"schemaVersion": 1, "traceEvents": [{"cudaGraphInlineAnnotated": true}]}"#
        ));
        assert!(!is_inline_annotated_bytes(br#"{"traceEvents": []}"#));
        assert!(!is_inline_annotated_bytes(b"not json"));
    }

    #[test]
    fn extracts_process_groups_from_profiler_envelope() {
        let groups = process_group_configs(
            br#"{
                "distributedInfo": {
                    "pg_config": [{
                        "pg_name": "17",
                        "pg_desc": "mesh_tp",
                        "backend_config": "cuda:nccl",
                        "pg_size": 2,
                        "ranks": [0, 1]
                    }]
                },
                "traceEvents": []
            }"#,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].pg_name, "17");
        assert_eq!(groups[0].pg_desc, "mesh_tp");
        assert_eq!(groups[0].ranks, vec![0, 1]);
    }

    #[test]
    fn extracts_process_groups_after_trace_events() {
        let groups = process_group_configs(
            br#"{
                "traceEvents": [],
                "distributedInfo": {
                    "pg_config": [{
                        "pg_name": "17",
                        "pg_desc": "mesh_tp",
                        "backend_config": "cuda:nccl",
                        "pg_size": 2,
                        "ranks": [0, 1]
                    }]
                }
            }"#,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].pg_name, "17");
    }

    #[test]
    fn gzipped_envelope() {
        use std::io::Write;

        use flate2::write::GzEncoder;
        let mut enc = GzEncoder::new(Vec::new(), flate2::Compression::new(1));
        enc.write_all(br#"{"cudaGraphInlineAnnotated": 1, "traceEvents": []}"#)
            .unwrap();
        let gz = enc.finish().unwrap();
        assert!(is_inline_annotated_bytes(&gz));
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Compact columnar encoding for per-PG kernel-timeline CSVs ("DPG1").
//!
//! Sidecar members were plain gzipped CSV; the CSVs are extremely
//! redundant column-wise (repeated kernel names, a constant para_type
//! hash per file, near-monotonic timestamps stored as decimal floats), so
//! a columnar transform ahead of gzip roughly halves them again. Format,
//! all varints LEB128 u64:
//!
//! ```text
//! "DPG1" | nrows | ncols
//! per column: name_len, name utf8, type u8 (0 = dict, 1 = num), num: scale
//! per column payload:
//!   dict: nuniq, nuniq × (len, utf8), nrows × id
//!   num:  nrows × zigzag(delta) of value*scale (first delta from 0)
//! ```
//!
//! The decoder is JavaScript in static_page.html, so exactness is defined
//! against `+field` there: a column is `num` only when every field is a
//! plain decimal with ≤3 fraction digits and |value·scale| < 2^52 — then
//! `n/scale` in f64 equals the double `+field` parses (both are the
//! nearest double to the same decimal), digits parsed as integers here
//! (never through f64). Anything else — text, exotic numbers, quoted
//! fields — falls back to `dict`, which reproduces the exact strings.
//! `encode_member` returns None for CSVs that don't fit the model at all
//! (quoting, ragged rows); the caller stores the raw CSV bytes instead and
//! the page sniffs the "DPG1" magic after inflating.
//!
//! Since the per-group CSVs were retired, the emitted artifact itself is
//! the member payload: `per_pg/{pg}_{gid}.dpg` holds either the DPG1
//! bytes or the raw-CSV fallback, pre-gzip (`emit::emit_group_members`);
//! the publisher gzips each file verbatim into the `.pgdata` sidecar.

/// |value·scale| bound for `num` columns: doubles are exact below 2^53;
/// one bit of margin keeps delta arithmetic comfortably exact too.
const MAX_ABS_SCALED: i128 = 1 << 52;

enum Column {
    /// (scale, per-row scaled values)
    Num(u32, Vec<i64>),
    /// (unique values in first-seen order, per-row indexes)
    Dict(Vec<String>, Vec<u32>),
}

/// Parse a strict fixed-point decimal: optional '-', digits, optionally
/// '.' plus 1..=3 digits. Returns (digits as i128 ignoring the point,
/// fraction-digit count). Exact by construction — no float in the path.
fn parse_fixed(field: &str) -> Option<(i128, u32)> {
    let b = field.as_bytes();
    let (neg, b) = match b.split_first() {
        Some((b'-', rest)) => (true, rest),
        _ => (false, b),
    };
    if b.is_empty() || !b[0].is_ascii_digit() {
        return None;
    }
    let mut v: i128 = 0;
    let mut frac: Option<u32> = None;
    for &c in b {
        match c {
            b'0'..=b'9' => {
                v = v * 10 + i128::from(c - b'0');
                if let Some(f) = frac.as_mut() {
                    *f += 1;
                    if *f > 3 {
                        return None;
                    }
                }
                if v > MAX_ABS_SCALED {
                    return None;
                }
            }
            b'.' if frac.is_none() => frac = Some(0),
            _ => return None,
        }
    }
    match frac {
        Some(0) => None, // "1." is not a number we accept
        f => Some((if neg { -v } else { v }, f.unwrap_or(0))),
    }
}

fn encode_column(fields: &[&str], preserve_text: bool) -> Column {
    // Numeric only if every field parses and fits at the column's common
    // scale (max fraction digits seen).
    let mut parsed = Vec::with_capacity(fields.len());
    let mut max_frac = 0u32;
    let mut numeric = true;
    for f in fields {
        match parse_fixed(f) {
            Some((v, frac)) => {
                max_frac = max_frac.max(frac);
                parsed.push((v, frac));
            }
            None => {
                numeric = false;
                break;
            }
        }
    }
    if numeric && !preserve_text {
        let mut vals = Vec::with_capacity(parsed.len());
        for (v, frac) in &parsed {
            let scaled = v * i128::from(10u32.pow(max_frac - frac));
            if scaled.abs() >= MAX_ABS_SCALED {
                numeric = false;
                break;
            }
            vals.push(scaled as i64);
        }
        if numeric {
            return Column::Num(10u32.pow(max_frac), vals);
        }
    }
    let mut uniq: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut ids = Vec::with_capacity(fields.len());
    for f in fields {
        let id = *uniq.entry(f).or_insert_with(|| {
            order.push((*f).to_string());
            (order.len() - 1) as u32
        });
        ids.push(id);
    }
    Column::Dict(order, ids)
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Encode one CSV into the DPG1 columnar form, or None when the CSV
/// doesn't fit the model (quoted fields, ragged rows, no header) — the
/// caller falls back to the raw CSV bytes. The bool is a "suspect" flag:
/// some dict column is high-cardinality (mostly-unique strings — e.g. a
/// numeric column that missed the fixed-point window), where columnar can
/// gzip WORSE than the raw CSV; the caller should compress both and keep
/// the smaller.
pub fn encode_member(csv: &str) -> Option<(Vec<u8>, bool)> {
    let mut lines = csv.split('\n');
    let header = lines.next()?.trim_end_matches('\r');
    if header.is_empty() {
        return None;
    }
    let names: Vec<&str> = header.split(',').collect();
    let mut rows: Vec<&str> = Vec::new(); // flattened row-major fields
    for line in lines {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue; // trailing newline; a truly empty row is ragged anyway
        }
        let start = rows.len();
        rows.extend(line.split(','));
        if rows.len() - start != names.len() {
            return None;
        }
    }
    if csv.contains('"') {
        return None; // quoted CSV — field splitting above would be wrong
    }
    let nrows = rows.len() / names.len();

    let mut out = Vec::new();
    out.extend_from_slice(b"DPG1");
    put_varint(&mut out, nrows as u64);
    put_varint(&mut out, names.len() as u64);
    let cols: Vec<Column> = (0..names.len())
        .map(|c| {
            let fields: Vec<&str> = (0..nrows).map(|r| rows[r * names.len() + c]).collect();
            encode_column(
                &fields,
                matches!(
                    names[c],
                    "pg_id" | "sequence_id" | "graph_id" | "node_id" | "graph_node_id"
                ),
            )
        })
        .collect();
    for (name, col) in names.iter().zip(&cols) {
        put_varint(&mut out, name.len() as u64);
        out.extend_from_slice(name.as_bytes());
        match col {
            Column::Num(scale, _) => {
                out.push(1);
                put_varint(&mut out, u64::from(*scale));
            }
            Column::Dict(..) => out.push(0),
        }
    }
    let mut suspect = false;
    for col in &cols {
        match col {
            Column::Num(_, vals) => {
                let mut prev = 0i64;
                for v in vals {
                    put_varint(&mut out, zigzag(v - prev));
                    prev = *v;
                }
            }
            Column::Dict(order, ids) => {
                suspect |= nrows > 64 && order.len() * 2 > nrows;
                put_varint(&mut out, order.len() as u64);
                for s in order {
                    put_varint(&mut out, s.len() as u64);
                    out.extend_from_slice(s.as_bytes());
                }
                for id in ids {
                    put_varint(&mut out, u64::from(*id));
                }
            }
        }
    }
    Some((out, suspect))
}

/// Reference decoder mirroring the page's decodeDpg — the Rust twin that
/// pins both implementations to the same format. Not a production read
/// path: the snapshot suite and the unit tests below use it to render a
/// member payload back into reviewable per-column strings. Numeric values
/// render as the page's `n/scale` double would display.
pub fn decode_member(b: &[u8]) -> (Vec<String>, Vec<Vec<String>>) {
    fn vu(b: &[u8], p: &mut usize) -> u64 {
        let mut v = 0u64;
        let mut s = 0;
        loop {
            let x = b[*p];
            *p += 1;
            v |= u64::from(x & 0x7f) << s;
            if x < 0x80 {
                return v;
            }
            s += 7;
        }
    }
    fn vstr(b: &[u8], p: &mut usize) -> String {
        let n = vu(b, p) as usize;
        let s = String::from_utf8(b[*p..*p + n].to_vec()).unwrap();
        *p += n;
        s
    }
    assert_eq!(&b[..4], b"DPG1");
    let p = &mut 4usize;
    let nrows = vu(b, p) as usize;
    let ncols = vu(b, p) as usize;
    let mut names = Vec::new();
    let mut kinds = Vec::new();
    for _ in 0..ncols {
        names.push(vstr(b, p));
        let ty = b[*p];
        *p += 1;
        let scale = if ty == 1 { vu(b, p) } else { 1 };
        kinds.push((ty, scale));
    }
    let mut cols = Vec::new();
    for &(ty, scale) in &kinds {
        let mut vals = Vec::with_capacity(nrows);
        if ty == 1 {
            let mut acc = 0i64;
            for _ in 0..nrows {
                let z = vu(b, p);
                acc += ((z >> 1) as i64) ^ -((z & 1) as i64);
                // render as the page's `n/scale` double would display
                vals.push(format!("{}", acc as f64 / scale as f64));
            }
        } else {
            let nu = vu(b, p) as usize;
            let mut uniq = Vec::with_capacity(nu);
            for _ in 0..nu {
                uniq.push(vstr(b, p));
            }
            for _ in 0..nrows {
                vals.push(uniq[vu(b, p) as usize].clone());
            }
        }
        cols.push(vals);
    }
    assert_eq!(*p, b.len());
    (names, cols)
}

/// Render a member payload (DPG1 or the raw-CSV fallback) as CSV-shaped
/// text, for content snapshots and debugging. DPG1 numeric cells render as
/// the doubles the page's decoder yields — this is the page-visible view
/// of the data, not the retired on-disk CSV bytes.
pub fn member_to_text(payload: &[u8]) -> String {
    if payload.len() < 4 || &payload[..4] != b"DPG1" {
        return String::from_utf8_lossy(payload).into_owned();
    }
    let (names, cols) = decode_member(payload);
    let nrows = cols.first().map_or(0, Vec::len);
    let mut out = names.join(",");
    out.push('\n');
    for r in 0..nrows {
        let row: Vec<&str> = cols.iter().map(|c| c[r].as_str()).collect();
        out.push_str(&row.join(","));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_timeline_csv() {
        let csv = "name,ts,dur,stream,grid,para_type,rank\n\
                   kern_a,2578675544695.438,213.026,7,1,hashhash,0\n\
                   kern_b,2578675544908.5,0.75,7,1,hashhash,0\n\
                   kern_a,2578675544910.0,12.0,20,1,hashhash,1\n";
        let (enc, suspect) = encode_member(csv).unwrap();
        assert!(!suspect);
        let (names, cols) = decode_member(&enc);
        assert_eq!(
            names,
            ["name", "ts", "dur", "stream", "grid", "para_type", "rank"]
        );
        assert_eq!(cols[0], ["kern_a", "kern_b", "kern_a"]);
        // Numeric columns reproduce the exact doubles `+field` yields.
        assert_eq!(
            cols[1],
            ["2578675544695.438", "2578675544908.5", "2578675544910"]
        );
        assert_eq!(cols[2], ["213.026", "0.75", "12"]);
        assert_eq!(cols[3], ["7", "7", "20"]);
        assert_eq!(cols[6], ["0", "0", "1"]);
        assert_eq!(cols[5], ["hashhash", "hashhash", "hashhash"]);
    }

    #[test]
    fn negative_and_scale_mixing() {
        let csv = "a,b\n-1.5,10\n2,-3.25\n0.001,0\n";
        let (_, cols) = decode_member(&encode_member(csv).unwrap().0);
        assert_eq!(cols[0], ["-1.5", "2", "0.001"]);
        assert_eq!(cols[1], ["10", "-3.25", "0"]);
    }

    #[test]
    fn oversized_and_exotic_numbers_fall_back_to_dict_exactly() {
        // 1.7e16 > 2^52: must come back as the verbatim string, and ".5"/
        // "1e3"/"1." are not fixed-point — dict keeps them verbatim too.
        let csv = "ts,x\n17000000000000141.0,.5\n17000000000000142.0,1e3\n17000000000000143.0,1.\n";
        let (_, cols) = decode_member(&encode_member(csv).unwrap().0);
        assert_eq!(
            cols[0],
            [
                "17000000000000141.0",
                "17000000000000142.0",
                "17000000000000143.0"
            ]
        );
        assert_eq!(cols[1], [".5", "1e3", "1."]);
    }

    #[test]
    fn rejects_quoted_or_ragged_csv() {
        assert!(encode_member("a,b\n\"x,y\",2\n").is_none());
        assert!(encode_member("a,b\n1,2,3\n").is_none());
        assert!(encode_member("").is_none());
    }

    #[test]
    fn empty_and_header_only_csvs_encode() {
        let (names, cols) = decode_member(&encode_member("rank\n").unwrap().0);
        assert_eq!(names, ["rank"]);
        assert_eq!(cols, [Vec::<String>::new()]);
    }
}

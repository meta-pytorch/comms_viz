// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Rank→host resolution + the compact rank-map encoding behind
//! `rank_hosts.json` and the published page's `rank_map` payload.
//!
//! Rank→host facts come from profile_logs: per-rank JSON files next to the
//! trace folder (`/{traces|chrome_traces|compute}/` → `/profile_logs/`),
//! flat `<rank:06d>.txt` leaves or hash-bucketed digit dirs.
//!
//! What lands on disk is `tree/{run}/rank_hosts.json`, the hosts-only form
//! of the compact encoding below (a per-rank `Rank,Host,Path` rank_map.csv
//! predecessor is retired: at large world sizes it was megabytes of text
//! whose Host column repeated each hostname once per local rank and whose
//! Path column is recomputable from the traces/ listing). Hosts are the
//! only analysis-time data the publisher cannot re-derive later, so they
//! are all it persists; the page payload's rank map is rebuilt at publish
//! time by `rank_map_payload` from this doc plus a fresh
//! `scan_trace_paths`.

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;

/// Trace-folder path tokens whose `/{token}/` segment maps onto
/// `/profile_logs/`.
const TRACE_FOLDER_TOKENS: [&str; 3] = ["traces", "chrome_traces", "compute"];

/// CPython `repr(float)` / `str(float)` — shortest round-trip digits with
/// Python's fixed/exponential switchover. The Python spelling is the data
/// contract wherever a float becomes artifact text: host/rank field coercion
/// here (`py_str_of`), and annotation-value stringification in the pickle
/// decoder (`legacy::pkl`).
pub(crate) fn py_float_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let neg = v.is_sign_negative();
    let v_abs = v.abs();

    // Shortest round-trip digits via Rust's {:e} (Grisu/Dragon4 shortest).
    // Format: "d.ddddde<exp>" or "de<exp>".
    let sci = format!("{:e}", v_abs);
    let (mantissa, exp_str) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp_str.parse().expect("exponent parses");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    // decpt: position of the decimal point relative to the digit string.
    let decpt = exp + 1;

    let body = if v_abs == 0.0 {
        "0.0".to_string()
    } else if decpt > -4 && decpt <= 16 {
        // Fixed notation.
        let nd = digits.len() as i32;
        if decpt <= 0 {
            let zeros = "0".repeat((-decpt) as usize);
            format!("0.{zeros}{digits}")
        } else if decpt >= nd {
            let zeros = "0".repeat((decpt - nd) as usize);
            format!("{digits}{zeros}.0")
        } else {
            let (int_part, frac_part) = digits.split_at(decpt as usize);
            format!("{int_part}.{frac_part}")
        }
    } else {
        // Exponential notation: d[.ddd]e±XX (>=2 exponent digits).
        let e = decpt - 1;
        let mant = if digits.len() == 1 {
            digits.to_string()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let sign = if e < 0 { '-' } else { '+' };
        format!("{mant}e{sign}{:02}", e.abs())
    };

    if neg { format!("-{body}") } else { body }
}

/// CPython `str()` of a scalar JSON value (host fields are strings in
/// practice; the fallbacks match `str(data["host"])` on degenerate input).
fn py_str_of(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(true) => "True".into(),
        serde_json::Value::Bool(false) => "False".into(),
        serde_json::Value::Null => "None".into(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else {
                py_float_repr(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        other => other.to_string(),
    }
}

/// `int(x)` for the JSON `rank` field: ints pass through, floats truncate
/// toward zero, numeric strings parse, bools coerce (Python bool ⊂ int).
fn py_int_of(v: &serde_json::Value) -> Option<i64> {
    match v {
        serde_json::Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64()
                .filter(|f| f.is_finite())
                .map(|f| f.trunc() as i64)
        }),
        serde_json::Value::String(s) => s.trim().parse::<i64>().ok(),
        serde_json::Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// Profile-log JSON line → (rank, normalized hostname); `(-1, "")` on
/// parse failure. Rank filtering (`r >= 0 && host non-empty`) is the
/// call site's job.
pub fn parse_profile_log_line(line: &str) -> (i64, String) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        return (-1, String::new());
    };
    let (Some(rank_v), Some(host_v)) = (v.get("rank"), v.get("host")) else {
        return (-1, String::new());
    };
    let Some(rank) = py_int_of(rank_v) else {
        return (-1, String::new());
    };
    (rank, py_str_of(host_v))
}

/// Trace root → profile_logs base: replace every `/{token}/` occurrence of
/// the first matching trace-folder token with `/profile_logs/`, rstrip `/`.
pub fn profile_log_base(trace_root: &str) -> Option<String> {
    for token in TRACE_FOLDER_TOKENS {
        let needle = format!("/{token}/");
        if trace_root.contains(&needle) {
            return Some(
                trace_root
                    .replace(&needle, "/profile_logs/")
                    .trim_end_matches('/')
                    .to_string(),
            );
        }
    }
    None
}

/// Hash-bucketed leaf path: the six digits of `rank:06d` as nested dirs.
fn bucketed_uri(base: &Path, rank: i64) -> PathBuf {
    let digits = format!("{rank:06}");
    let mut p = base.to_path_buf();
    for d in digits.chars() {
        p.push(d.to_string());
    }
    p.join(format!("{digits}.txt"))
}

/// First 4096 bytes of the leaf, lossy-decoded, cut at the first newline.
fn read_first_line(path: &Path) -> Option<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; 4096];
    let mut n = 0usize;
    loop {
        match f.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => {
                n += k;
                if n == buf.len() {
                    break;
                }
            }
            Err(_) => return None,
        }
    }
    let text = String::from_utf8_lossy(&buf[..n]);
    Some(text.split('\n').next().unwrap_or("").to_string())
}

/// The profile_logs tier, on a local/FUSE directory: derive the
/// profile_logs base from the trace root, prefer flat `.txt` leaves, else
/// reconstruct hash-bucketed per-rank URIs from `world_size`. Rank is always
/// taken from each file's JSON content, never from the path. Best-effort:
/// every failure degrades to an empty map.
pub fn from_profile_logs(trace_root: &Path, world_size: i64) -> BTreeMap<i64, String> {
    let mut out = BTreeMap::new();
    let Some(base) = profile_log_base(&trace_root.to_string_lossy()) else {
        return out;
    };
    if world_size <= 0 {
        return out;
    }
    let base = PathBuf::from(base);
    // Flat `.txt` file leaves directly under base (`_flat_leaf_uris`).
    let mut uris: Vec<PathBuf> = std::fs::read_dir(&base)
        .map(|rd| {
            let mut v: Vec<PathBuf> = rd
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| !t.is_dir()).unwrap_or(false))
                .map(|e| e.path())
                .filter(|p| p.to_string_lossy().ends_with(".txt"))
                .collect();
            v.sort(); // deterministic listing order
            v
        })
        .unwrap_or_default();
    if uris.is_empty() {
        uris = (0..world_size).map(|r| bucketed_uri(&base, r)).collect();
    }
    for uri in uris {
        if let Some(line) = read_first_line(&uri) {
            let (r, host) = parse_profile_log_line(&line);
            if r >= 0 && !host.is_empty() {
                out.insert(r, host);
            }
        }
    }
    out
}

/// Compact rank-map JSON, the encoding both `rank_hosts.json` and the
/// published page's `rank_map` payload use:
///
/// ```json
/// {"v":1,
///  "hosts":["host-a", …],                 // unique hosts, first-use order
///  "sfx":[".merged.pt.trace.json.gz", …], // unique trace-name suffix entries
///  "runs":[[start,len,h,s], …],           // maximal contiguous rank runs
///  "tok":[0, 77043154, …],                // per-rank tokens of "#" entries, run order
///  "bkt":[1,2,3,4,5,6],                   // hash-bucket dirs cut from the rank digits
///  "x":{"7":"odd-name"}}                  // only when a name defies every form
/// ```
///
/// A run covers ranks `start..start+len`, all sharing host index `h` and
/// suffix index `s` (−1 = absent). Three suffix-entry forms reconstruct a
/// rank's trace name:
/// - a string `sfx[s]`: `bkt(r)` + the six-zero-padded rank + the entry
///   (the on-disk shape `scan_trace_paths` guarantees). Restart-tag
///   names (`<rank:06>.v<version>_a<attempt>_t<task-group attempt>.pt.trace.<suffix>`)
///   share one such entry per restart and format;
/// - `["rank_", suffix]`: `rank_` + the unpadded rank + suffix;
/// - `["#", mid, tail, base]`: `bkt(r)` + padded rank + mid + (base +
///   the rank's `tok` entry) + tail — legacy timestamp capture names
///   (`<rank:06>.<time_ns>.pt.trace.<suffix>`),
///   whose per-rank digits would otherwise cost a suffix entry AND a run
///   per rank. `base` is a decimal string (epoch ns exceed 2^53), the
///   deltas are plain numbers; the page consumes `tok` in run order, one
///   per rank of a `#` run. Digits with a leading zero are not numeric
///   tokens and keep the string form (lossless either way).
///
/// `bkt`, when present, is the hash-bucketed dump layout as cumulative
/// digit cuts: the padded rank's digit groups joined by `/` prefix the
/// name (`[1,2,3,4,5,6]` → `0/5/3/2/4/7/053247.…`). Set only when every
/// padded-form name carries exactly that prefix; any other prefix sends
/// the name to `x`. Any other name — possible only for hand-built maps —
/// goes into the `x` escape verbatim, so the encoding is lossless for
/// arbitrary input. Ranks in neither map appear in no run. Returns None
/// when there is nothing to encode. Output is deterministic: ranks
/// ascend, dictionaries are in first-use order.
pub fn rank_map_payload(
    rank_host: &BTreeMap<i64, String>,
    rank_path: &BTreeMap<i64, String>,
) -> Option<String> {
    if rank_host.is_empty() && rank_path.is_empty() {
        return None;
    }
    let ranks: Vec<i64> = rank_host
        .keys()
        .chain(rank_path.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut shapes: Vec<Option<NameShape<'_>>> = ranks
        .iter()
        .map(|r| rank_path.get(r).map(|n| NameShape::of(*r, n)))
        .collect();

    // Bucket cuts: all-or-nothing across the padded-form names, so a
    // stray prefixed name never demotes a whole flat layout to escapes.
    let cuts: Option<Vec<usize>> = {
        let padded: Vec<(i64, &str)> = ranks
            .iter()
            .zip(&shapes)
            .filter_map(|(r, sh)| match sh {
                Some(NameShape::Padded { pre, .. }) => Some((*r, *pre)),
                _ => None,
            })
            .collect();
        match padded.first() {
            Some((r, pre)) if !pre.is_empty() => bucket_cuts(*r, pre)
                .filter(|c| padded.iter().all(|(r, pre)| bucket_prefix(*r, c) == *pre)),
            _ => None,
        }
    };
    for (r, sh) in ranks.iter().zip(shapes.iter_mut()) {
        if let Some(NameShape::Padded { pre, name, .. }) = sh {
            let expect = cuts
                .as_deref()
                .map(|c| bucket_prefix(*r, c))
                .unwrap_or_default();
            if *pre != expect {
                *sh = Some(NameShape::Extra(name));
            }
        }
    }

    // Token groups (mid, tail) → base = min token; a spread past 2^53
    // would not survive the page's Number deltas, so such a group keeps
    // the string form.
    let mut groups: BTreeMap<(&str, &str), (u64, u64)> = BTreeMap::new(); // (min, max)
    for sh in shapes.iter().flatten() {
        if let NameShape::Padded {
            tok: Some((mid, digits, tail)),
            ..
        } = sh
        {
            let e = groups.entry((mid, tail)).or_insert((*digits, *digits));
            e.0 = e.0.min(*digits);
            e.1 = e.1.max(*digits);
        }
    }
    groups.retain(|_, (lo, hi)| *hi - *lo < (1u64 << 53));

    let mut hosts: Vec<&str> = Vec::new();
    let mut host_idx: BTreeMap<&str, i64> = BTreeMap::new();
    let mut sfx: Vec<SfxEntry<'_>> = Vec::new();
    let mut sfx_idx: BTreeMap<SfxEntry<'_>, i64> = BTreeMap::new();
    let mut extras: Vec<(i64, &str)> = Vec::new();
    let mut runs: Vec<[i64; 4]> = Vec::new();
    let mut tok: Vec<u64> = Vec::new();
    for (r, sh) in ranks.iter().zip(&shapes) {
        let r = *r;
        let h = match rank_host.get(&r) {
            Some(host) => *host_idx.entry(host).or_insert_with(|| {
                hosts.push(host);
                hosts.len() as i64 - 1
            }),
            None => -1,
        };
        let entry: Option<SfxEntry<'_>> = match sh {
            None => None,
            Some(NameShape::RankForm(rest)) => Some(SfxEntry::RankForm(rest)),
            Some(NameShape::Padded {
                tok: Some((mid, digits, tail)),
                ..
            }) if groups.contains_key(&(*mid, *tail)) => {
                tok.push(digits - groups[&(*mid, *tail)].0);
                Some(SfxEntry::Token {
                    mid,
                    tail,
                    base: groups[&(*mid, *tail)].0,
                })
            }
            Some(NameShape::Padded { rest, .. }) => Some(SfxEntry::Padded(rest)),
            Some(NameShape::Extra(name)) => {
                extras.push((r, name));
                None
            }
        };
        let s = match entry {
            Some(e) => *sfx_idx.entry(e.clone()).or_insert_with(|| {
                sfx.push(e);
                sfx.len() as i64 - 1
            }),
            None => -1,
        };
        match runs.last_mut() {
            Some([start, len, rh, rs]) if *start + *len == r && *rh == h && *rs == s => {
                *len += 1;
            }
            _ => runs.push([r, 1, h, s]),
        }
    }
    let sfx: Vec<serde_json::Value> = sfx
        .into_iter()
        .map(|e| match e {
            SfxEntry::Padded(rest) => serde_json::json!(rest),
            SfxEntry::RankForm(rest) => serde_json::json!(["rank_", rest]),
            SfxEntry::Token { mid, tail, base } => {
                serde_json::json!(["#", mid, tail, base.to_string()])
            }
        })
        .collect();
    let mut doc = serde_json::json!({
        "v": 1,
        "hosts": hosts,
        "sfx": sfx,
        "runs": runs,
    });
    if !tok.is_empty() {
        doc["tok"] = serde_json::json!(tok);
    }
    if let Some(c) = &cuts {
        doc["bkt"] = serde_json::json!(c);
    }
    if !extras.is_empty() {
        doc["x"] = extras
            .into_iter()
            .map(|(r, name)| (r.to_string(), serde_json::json!(name)))
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into();
    }
    Some(doc.to_string())
}

/// How one rank's trace name decomposes for `rank_map_payload`.
enum NameShape<'a> {
    /// `rank_<r>` + rest (published leaf names).
    RankForm(&'a str),
    /// `pre` (`""` or a `/`-terminated dir prefix) + `<r:06>` + rest;
    /// `tok` = rest split as mid + numeric token + tail when it has one.
    Padded {
        pre: &'a str,
        rest: &'a str,
        tok: Option<(&'a str, u64, &'a str)>,
        name: &'a str,
    },
    Extra(&'a str),
}

impl<'a> NameShape<'a> {
    fn of(rank: i64, name: &'a str) -> NameShape<'a> {
        if let Some(rest) = name.strip_prefix(&format!("rank_{rank}")) {
            if !rest.is_empty() {
                return NameShape::RankForm(rest);
            }
        }
        let (pre, leaf) = match name.rfind('/') {
            Some(i) => name.split_at(i + 1),
            None => ("", name),
        };
        match leaf.strip_prefix(&format!("{rank:06}")) {
            Some(rest) if !rest.is_empty() => NameShape::Padded {
                pre,
                rest,
                tok: split_token(rest),
                name,
            },
            _ => NameShape::Extra(name),
        }
    }
}

/// Tokenize a legacy `.<time_ns>.<tail>` capture field: a canonical u64
/// (no leading zero) between the first two dots. Other capture IDs stay
/// opaque shared suffixes. In particular, searching for digits anywhere
/// would tokenize the version in `v100_a0000_t0000` and add redundant
/// per-rank deltas for a tag the whole restart shares.
fn split_token(rest: &str) -> Option<(&str, u64, &str)> {
    let field = rest.strip_prefix('.')?;
    let end = field.find('.')?;
    let digits = &field[..end];
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if digits.len() > 1 && digits.starts_with('0') {
        return None;
    }
    let value: u64 = digits.parse().ok()?;
    Some((".", value, &field[end..]))
}

/// Cumulative digit cuts that reproduce `pre` from this rank's padded
/// digits (`0/5/3/` for rank 53247 → `[1,2,3]`); None when `pre` is not
/// made of the rank's leading digits.
fn bucket_cuts(rank: i64, pre: &str) -> Option<Vec<usize>> {
    let digits = format!("{rank:06}");
    let body = pre.strip_suffix('/')?;
    let mut cuts = Vec::new();
    let mut at = 0usize;
    for seg in body.split('/') {
        if seg.is_empty() || !digits[at..].starts_with(seg) {
            return None;
        }
        at += seg.len();
        cuts.push(at);
    }
    (!cuts.is_empty()).then_some(cuts)
}

/// The bucket prefix `cuts` produce for `rank` (`""` when they overrun
/// the padded digits — such a rank then fails the exact-prefix check and
/// is escaped verbatim).
fn bucket_prefix(rank: i64, cuts: &[usize]) -> String {
    let digits = format!("{rank:06}");
    let mut out = String::new();
    let mut at = 0usize;
    for &c in cuts {
        if c <= at || c > digits.len() {
            return String::new();
        }
        out.push_str(&digits[at..c]);
        out.push('/');
        at = c;
    }
    out
}

/// A suffix dictionary entry, keyed for first-use dedup.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum SfxEntry<'a> {
    Padded(&'a str),
    RankForm(&'a str),
    Token {
        mid: &'a str,
        tail: &'a str,
        base: u64,
    },
}

/// Decode every rank's trace path from a `rank_map_payload` doc — the
/// Rust twin of the page's decoder (static_page.html boot), kept so the
/// encoding contract is pinned by round-trip tests rather than by
/// reading two implementations side by side.
pub fn decode_rank_paths(text: &str) -> Result<BTreeMap<i64, String>> {
    let doc: serde_json::Value = serde_json::from_str(text).context("parse rank-map doc")?;
    let arr = |k: &str| doc.get(k).and_then(serde_json::Value::as_array).cloned();
    let sfx = arr("sfx").unwrap_or_default();
    let tok = arr("tok").unwrap_or_default();
    let cuts: Vec<usize> = arr("bkt")
        .unwrap_or_default()
        .iter()
        .filter_map(|v| v.as_u64().map(|c| c as usize))
        .collect();
    let extras = doc.get("x").and_then(serde_json::Value::as_object);
    let mut ti = 0usize;
    let mut out = BTreeMap::new();
    for run in arr("runs").unwrap_or_default() {
        let f = |i: usize| run.get(i).and_then(serde_json::Value::as_i64);
        let (Some(start), Some(len), Some(s)) = (f(0), f(1), f(3)) else {
            anyhow::bail!("rank-map doc: malformed run {run}");
        };
        for r in start..start + len {
            if let Some(x) = extras
                .and_then(|m| m.get(&r.to_string()))
                .and_then(|v| v.as_str())
            {
                out.insert(r, x.to_string());
                continue;
            }
            if s < 0 {
                continue;
            }
            let e = sfx
                .get(s as usize)
                .with_context(|| format!("rank-map doc: sfx index {s} out of range"))?;
            let padded = format!("{}{r:06}", bucket_prefix(r, &cuts));
            let name = match e {
                serde_json::Value::String(suffix) => format!("{padded}{suffix}"),
                serde_json::Value::Array(a) if a.len() == 2 => {
                    format!(
                        "{}{r}{}",
                        a[0].as_str().unwrap_or(""),
                        a[1].as_str().unwrap_or("")
                    )
                }
                serde_json::Value::Array(a) if a.len() == 4 && a[0] == "#" => {
                    let base: u64 = a[3].as_str().unwrap_or("").parse().context("token base")?;
                    let d = tok
                        .get(ti)
                        .and_then(serde_json::Value::as_u64)
                        .with_context(|| format!("rank-map doc: missing token {ti}"))?;
                    ti += 1;
                    format!(
                        "{padded}{}{}{}",
                        a[1].as_str().unwrap_or(""),
                        base + d,
                        a[2].as_str().unwrap_or("")
                    )
                }
                other => anyhow::bail!("rank-map doc: bad sfx entry {other}"),
            };
            out.insert(r, name);
        }
    }
    Ok(out)
}

/// Decode the rank→host side of a `rank_map_payload` doc (the publisher
/// reads `rank_hosts.json` this way; suffixes/extras are path data it
/// re-derives from the traces listing instead). Malformed input is a hard
/// error so a corrupt doc degrades to "hosts unresolved" at the call site
/// rather than silently mis-mapping ranks.
pub fn parse_rank_hosts(text: &str) -> Result<BTreeMap<i64, String>> {
    let doc: serde_json::Value = serde_json::from_str(text).context("parse rank-hosts doc")?;
    anyhow::ensure!(
        doc.get("v").and_then(serde_json::Value::as_i64) == Some(1),
        "rank-hosts doc: unsupported version {:?}",
        doc.get("v")
    );
    let hosts: Vec<&str> = doc
        .get("hosts")
        .and_then(serde_json::Value::as_array)
        .map(|a| a.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default();
    let mut out = BTreeMap::new();
    for run in doc
        .get("runs")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        let f = |i: usize| run.get(i).and_then(serde_json::Value::as_i64);
        let (Some(start), Some(len), Some(h)) = (f(0), f(1), f(2)) else {
            anyhow::bail!("rank-hosts doc: malformed run {run}");
        };
        if h < 0 {
            continue;
        }
        let host = *hosts
            .get(h as usize)
            .with_context(|| format!("rank-hosts doc: host index {h} out of range"))?;
        anyhow::ensure!(
            (0..=MAX_DECODED_RANKS).contains(&len),
            "rank-hosts doc: implausible run length {len}"
        );
        for r in start..start.saturating_add(len) {
            out.insert(r, host.to_string());
        }
    }
    Ok(out)
}

/// Decode bound for a single run (matches any real-world world size with
/// room to spare) so a corrupt length can't balloon the map.
const MAX_DECODED_RANKS: i64 = 100_000_000;

/// Trace scan for the rank map: basenames under `tree/{run}/traces/`
/// ending `.pt.trace.json.gz` whose name starts `NNNNNN.` (exactly six
/// digits); per rank the lexicographically greatest basename wins.
pub fn scan_trace_paths(traces_dir: &Path) -> BTreeMap<i64, String> {
    let mut out: BTreeMap<i64, String> = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(traces_dir) else {
        return out;
    };
    for e in rd.filter_map(|e| e.ok()) {
        let name = e.file_name().to_string_lossy().into_owned();
        // Any known trace format: this scan only needs to identify a rank's
        // trace file, not decode it (`crate::trace::is_trace_name`).
        if !crate::trace::is_trace_name(&name) {
            continue;
        }
        let b = name.as_bytes();
        if b.len() < 7 || b[6] != b'.' || !b[..6].iter().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let r: i64 = name[..6].parse().unwrap();
        match out.get(&r) {
            Some(cur) if name <= *cur => {}
            _ => {
                out.insert(r, name);
            }
        }
    }
    out
}

/// Emit `tree/{run}/rank_hosts.json` into `run_root`:
///   - hosts: from the profile_logs tier;
///   - no hosts → no rank_hosts.json (the doc carries no trace paths —
///     the publisher re-derives those from the traces/ listing, so a
///     hostless run has nothing to persist).
pub fn emit_rank_hosts(run_root: &Path, trace_root: &Path, world_size: i64) -> Result<()> {
    let rank_host = from_profile_logs(trace_root, world_size);
    let Some(doc) = rank_map_payload(&rank_host, &BTreeMap::new()) else {
        return Ok(());
    };
    let p = run_root.join("rank_hosts.json");
    std::fs::create_dir_all(run_root).ok();
    std::fs::write(&p, doc).with_context(|| format!("write {p:?}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_cpython() {
        assert_eq!(py_float_repr(0.0), "0.0");
        assert_eq!(py_float_repr(-0.0), "-0.0");
        assert_eq!(py_float_repr(1.0), "1.0");
        assert_eq!(py_float_repr(0.1), "0.1");
        assert_eq!(py_float_repr(1.5), "1.5");
        assert_eq!(py_float_repr(123456789.0), "123456789.0");
        assert_eq!(py_float_repr(1e15), "1000000000000000.0");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(1.5e16), "1.5e+16");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_float_repr(1e-5), "1e-05");
        assert_eq!(py_float_repr(-2.5e-7), "-2.5e-07");
        assert_eq!(py_float_repr(1234567.125), "1234567.125");
        assert_eq!(py_float_repr(9007199254740993.0), "9007199254740992.0");
        assert_eq!(py_float_repr(1e100), "1e+100");
        assert_eq!(
            py_float_repr(2.2250738585072014e-308),
            "2.2250738585072014e-308"
        );
    }

    #[test]
    fn parse_profile_log_line_cases() {
        assert_eq!(
            parse_profile_log_line(r#"{"rank": 7, "host": "h7.example.com"}"#),
            (7, "h7.example.com".to_string())
        );
        // Rank as numeric string (Python int("7")).
        assert_eq!(
            parse_profile_log_line(r#"{"rank": "7", "host": "h7"}"#),
            (7, "h7".to_string())
        );
        // Missing keys / bad JSON → (-1, "").
        assert_eq!(
            parse_profile_log_line(r#"{"host": "h"}"#),
            (-1, String::new())
        );
        assert_eq!(parse_profile_log_line("not json"), (-1, String::new()));
        // Negative rank parses; the r>=0 filter is the caller's.
        assert_eq!(
            parse_profile_log_line(r#"{"rank": -3, "host": "h"}"#),
            (-3, "h".to_string())
        );
        // Non-coercible rank (Python int(None) → TypeError) → (-1, "").
        assert_eq!(
            parse_profile_log_line(r#"{"rank": null, "host": "h"}"#),
            (-1, String::new())
        );
    }

    #[test]
    fn profile_log_base_cases() {
        assert_eq!(
            profile_log_base("/a/traces/step_0001"),
            Some("/a/profile_logs/step_0001".to_string())
        );
        assert_eq!(
            profile_log_base("/a/chrome_traces/step_0001/"),
            Some("/a/profile_logs/step_0001".to_string())
        );
        // RL layout: .../profiling/compute/<role>/<ts>.
        assert_eq!(
            profile_log_base("/j/profiling/compute/trainer_0/171"),
            Some("/j/profiling/profile_logs/trainer_0/171".to_string())
        );
        // No token → None (e.g. the local fixture layout /x/step_NNN).
        assert_eq!(
            profile_log_base("/tmp/comms_viz-synth/step_000000000106"),
            None
        );
        // ALL occurrences of the chosen token are replaced.
        assert_eq!(
            profile_log_base("/a/traces/b/traces/c"),
            Some("/a/profile_logs/b/profile_logs/c".to_string())
        );
    }

    #[test]
    fn profile_logs_flat_layout_rank_from_content() {
        let td = tempfile::tempdir().unwrap();
        let base = td.path().join("profiling/profile_logs/step_0001");
        std::fs::create_dir_all(&base).unwrap();
        // Rank comes from file CONTENT, never from the filename.
        std::fs::write(
            base.join("000000.txt"),
            "{\"rank\": 1, \"host\": \"hostB\"}\ntrailing\n",
        )
        .unwrap();
        std::fs::write(
            base.join("000001.txt"),
            "{\"rank\": 0, \"host\": \"hostA\"}",
        )
        .unwrap();
        std::fs::write(base.join("ignored.log"), "{\"rank\": 5, \"host\": \"x\"}").unwrap();
        let root = td.path().join("profiling/traces/step_0001");
        let m = from_profile_logs(&root, 2);
        assert_eq!(m.get(&0).map(String::as_str), Some("hostA"));
        assert_eq!(m.get(&1).map(String::as_str), Some("hostB"));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn profile_logs_bucketed_layout() {
        let td = tempfile::tempdir().unwrap();
        // trace root .../traces/step_x → bucketed logs .../profile_logs/step_x
        let logs = td.path().join("profile_logs/step_x");
        // rank 1 → 0/0/0/0/0/1/000001.txt
        let d = logs.join("0/0/0/0/0/1");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("000001.txt"), "{\"rank\": 1, \"host\": \"bk1\"}").unwrap();
        let root = td.path().join("traces/step_x");
        // world_size bounds the bucketed reconstruction.
        let m = from_profile_logs(&root, 2);
        assert_eq!(m.get(&1).map(String::as_str), Some("bk1"));
        assert_eq!(m.len(), 1);
        assert!(from_profile_logs(&root, 0).is_empty());
        // world_size 1 never probes rank 1's bucket.
        assert!(from_profile_logs(&root, 1).is_empty());
    }

    #[test]
    fn rank_map_payload_runs_dicts_and_escapes() {
        // 8 ranks per host, uniform suffixes → one run per host block.
        let mut host = BTreeMap::new();
        let mut path = BTreeMap::new();
        for r in 0..16i64 {
            host.insert(r, format!("host{}", r / 8));
            path.insert(r, format!("{r:06}.merged.pt.trace.json.gz"));
        }
        assert_eq!(
            rank_map_payload(&host, &path).unwrap(),
            r#"{"hosts":["host0","host1"],"runs":[[0,8,0,0],[8,8,1,0]],"sfx":[".merged.pt.trace.json.gz"],"v":1}"#
        );

        // Absences split runs: a hostless rank and a pathless rank each cut
        // a boundary; a rank in neither map appears in no run.
        let mut host = BTreeMap::new();
        host.insert(0, "h".to_string());
        host.insert(1, "h".to_string());
        host.insert(3, "h".to_string());
        let mut path = BTreeMap::new();
        path.insert(1, "000001.a.gz".to_string());
        path.insert(2, "000002.a.gz".to_string());
        assert_eq!(
            rank_map_payload(&host, &path).unwrap(),
            r#"{"hosts":["h"],"runs":[[0,1,0,-1],[1,1,0,0],[2,1,-1,0],[3,1,0,-1]],"sfx":[".a.gz"],"v":1}"#
        );

        // A name that is not <rank:06> + suffix survives via the escape map.
        let mut odd = BTreeMap::new();
        odd.insert(7i64, "weird-name.json.gz".to_string());
        assert_eq!(
            rank_map_payload(&BTreeMap::new(), &odd).unwrap(),
            r#"{"hosts":[],"runs":[[7,1,-1,-1]],"sfx":[],"v":1,"x":{"7":"weird-name.json.gz"}}"#
        );

        // rank_<r> names share one
        // [prefix, suffix] dict entry per run — never the escape map.
        let mut path = BTreeMap::new();
        for r in 0..4i64 {
            path.insert(r, format!("rank_{r}.merged.pftrace.gz"));
        }
        assert_eq!(
            rank_map_payload(&BTreeMap::new(), &path).unwrap(),
            r#"{"hosts":[],"runs":[[0,4,-1,0]],"sfx":[["rank_",".merged.pftrace.gz"]],"v":1}"#
        );

        // Nothing to encode → None.
        assert_eq!(rank_map_payload(&BTreeMap::new(), &BTreeMap::new()), None);
    }

    #[test]
    fn rank_map_payload_shares_restart_tags() {
        // The capture ID is shared by ranks of a restart.
        // Version padding is a minimum width: v100 must remain a shared
        // suffix too, not become a numeric token with one delta per rank.
        for bucketed in [false, true] {
            let tags = [
                "v000_a0000_t0000",
                "v002_a0013_t0001",
                "v100_a0000_t0000",
                "v101_a0000_t0000",
            ];
            let mut paths = BTreeMap::new();
            let mut suffixes = Vec::new();
            for (i, tag) in tags.iter().enumerate() {
                for (f, ext) in ["json.gz", "pftrace"].iter().enumerate() {
                    let suffix = format!(".{tag}.pt.trace.{ext}");
                    let start = i * 4 + f * 2;
                    for r in start..start + 2 {
                        let pre = if bucketed {
                            bucket_prefix(r as i64, &[1, 2, 3, 4, 5, 6])
                        } else {
                            String::new()
                        };
                        paths.insert(r as i64, format!("{pre}{r:06}{suffix}"));
                    }
                    suffixes.push(suffix);
                }
            }
            let doc = rank_map_payload(&BTreeMap::new(), &paths).unwrap();
            let v: serde_json::Value = serde_json::from_str(&doc).unwrap();
            assert_eq!(v["sfx"], serde_json::json!(suffixes));
            assert!(v.get("tok").is_none() && v.get("x").is_none());
            assert_eq!(v["runs"].as_array().unwrap().len(), tags.len() * 2);
            assert_eq!(decode_rank_paths(&doc).unwrap(), paths);
            assert_eq!(v.get("bkt").is_some(), bucketed);
        }
    }

    #[test]
    fn rank_map_payload_mixes_legacy_and_restart_names() {
        let paths = [
            (0, "000000.1787899943589106037.pt.trace.json.gz".into()),
            (1, "000001.v002_a0013_t0001.pt.trace.json.gz".into()),
            (2, "000002.1787899943512062883.pt.trace.json.gz".into()),
            (3, "000003.v002_a0013_t0001.pt.trace.json.gz".into()),
        ]
        .into();
        let doc = rank_map_payload(&BTreeMap::new(), &paths).unwrap();
        let v: serde_json::Value = serde_json::from_str(&doc).unwrap();
        assert_eq!(v["tok"], serde_json::json!([77043154, 0]));
        assert_eq!(v["sfx"].as_array().unwrap().len(), 2);
        assert_eq!(decode_rank_paths(&doc).unwrap(), paths);
    }

    #[test]
    fn rank_map_restart_paths_have_constant_fleet_scale_cost() {
        let mut hosts = BTreeMap::new();
        let mut paths = BTreeMap::new();
        for r in 0..100_000 {
            hosts.insert(r, format!("host{}", r / 8));
            let pre = bucket_prefix(r, &[1, 2, 3, 4, 5, 6]);
            paths.insert(r, format!("{pre}{r:06}.v100_a0013_t0001.pt.trace.pftrace"));
        }
        let doc = rank_map_payload(&hosts, &paths).unwrap();
        let hosts_only = rank_map_payload(&hosts, &BTreeMap::new()).unwrap();
        let paths_only = rank_map_payload(&BTreeMap::new(), &paths).unwrap();
        // No per-rank name data: one suffix, one bucket layout, and the
        // same host runs (or just one run without hosts).
        assert!(doc.len() <= hosts_only.len() + 100);
        assert!(paths_only.len() < 200, "{}", paths_only.len());
        assert_eq!(decode_rank_paths(&doc).unwrap(), paths);
    }

    #[test]
    fn rank_map_payload_tokenizes_exporter_names() {
        // Legacy exporter names: <rank:06>.<time_ns>.pt.trace.json.gz — one
        // "#" entry for the whole map, deltas from the smallest ns, and
        // runs that still coalesce per host block.
        let ns = [
            1787899943589106037u64,
            1787899943512062883,
            1787899943763985411,
            1787899943889014272,
            1787899943571888770,
            1787899943552851251,
            1787899943550822251,
            1787899943550816107,
        ];
        let mut host = BTreeMap::new();
        let mut path = BTreeMap::new();
        for (r, t) in ns.iter().enumerate() {
            host.insert(r as i64, format!("host{}", r / 4));
            path.insert(r as i64, format!("{r:06}.{t}.pt.trace.json.gz"));
        }
        let doc = rank_map_payload(&host, &path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&doc).unwrap();
        assert_eq!(
            v["sfx"],
            serde_json::json!([["#", ".", ".pt.trace.json.gz", "1787899943512062883"]])
        );
        assert_eq!(v["runs"], serde_json::json!([[0, 4, 0, 0], [4, 4, 1, 0]]));
        assert_eq!(v["tok"][1], 0);
        assert_eq!(v["tok"][0], 77043154);
        assert!(v.get("bkt").is_none() && v.get("x").is_none());
        assert_eq!(decode_rank_paths(&doc).unwrap(), path);

        // Hash-bucketed layout: the dirs are the rank's digits, carried as
        // one cut list instead of per-rank escapes.
        let mut bucketed = BTreeMap::new();
        for (r, t) in ns.iter().enumerate() {
            let d = format!("{r:06}");
            let pre: String = d.chars().map(|c| format!("{c}/")).collect();
            bucketed.insert(r as i64, format!("{pre}{r:06}.{t}.pt.trace.json.gz"));
        }
        let doc = rank_map_payload(&host, &bucketed).unwrap();
        let v: serde_json::Value = serde_json::from_str(&doc).unwrap();
        assert_eq!(v["bkt"], serde_json::json!([1, 2, 3, 4, 5, 6]));
        assert!(v.get("x").is_none());
        assert_eq!(decode_rank_paths(&doc).unwrap(), bucketed);
        // Two-digit groups and a partial depth work the same way.
        let mut two = BTreeMap::new();
        two.insert(53247i64, "05/32/053247.9.pt.trace.json.gz".to_string());
        two.insert(53248i64, "05/32/053248.10.pt.trace.json.gz".to_string());
        let doc = rank_map_payload(&BTreeMap::new(), &two).unwrap();
        let v: serde_json::Value = serde_json::from_str(&doc).unwrap();
        assert_eq!(v["bkt"], serde_json::json!([2, 4]));
        assert_eq!(decode_rank_paths(&doc).unwrap(), two);

        // Lossless fallbacks: a leading-zero token is not numeric (string
        // form), a lone prefixed name in a flat map is escaped, and a
        // token spread past 2^53 keeps the string form for its group.
        let mut odd = BTreeMap::new();
        odd.insert(0i64, "000000.0123.pt.trace.json.gz".to_string());
        odd.insert(1i64, "0/000001.5.pt.trace.json.gz".to_string());
        odd.insert(2i64, "000002.1.pt.trace.json.gz".to_string());
        odd.insert(3i64, "000003.9007199254740993.pt.trace.json.gz".to_string());
        odd.insert(4i64, "rank_4.merged.pftrace.gz".to_string());
        odd.insert(5i64, "000005.merged.pt.trace.json.gz".to_string());
        let doc = rank_map_payload(&BTreeMap::new(), &odd).unwrap();
        let v: serde_json::Value = serde_json::from_str(&doc).unwrap();
        assert_eq!(
            v["x"],
            serde_json::json!({"1": "0/000001.5.pt.trace.json.gz"})
        );
        assert!(v.get("bkt").is_none() && v.get("tok").is_none());
        assert!(
            v["sfx"]
                .as_array()
                .unwrap()
                .iter()
                .all(|e| e.get(0) != Some(&serde_json::json!("#")))
        );
        assert_eq!(decode_rank_paths(&doc).unwrap(), odd);
    }

    #[test]
    fn rank_map_payload_stays_small_at_fleet_scale() {
        // 100k exporter names, 8 ranks per host, exports spread over ~30s:
        // the tokens are the only per-rank cost left (~12 bytes raw each).
        let n = 100_000i64;
        let base = 1_787_899_943_000_000_000u64;
        let mut host = BTreeMap::new();
        let mut path = BTreeMap::new();
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for r in 0..n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let t = base + x % 30_000_000_000;
            host.insert(r, format!("hostname{:05}.{:02}", r / 8, (r / 8) % 97));
            path.insert(r, format!("{r:06}.{t}.pt.trace.json.gz"));
        }
        let doc = rank_map_payload(&host, &path).unwrap();
        assert_eq!(decode_rank_paths(&doc).unwrap(), path);
        let hosts_only = rank_map_payload(&host, &BTreeMap::new()).unwrap();
        let gz = |t: &str| {
            use std::io::Write;
            let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            e.write_all(t.as_bytes()).unwrap();
            e.finish().unwrap().len()
        };
        eprintln!(
            "rank_map 100k: hosts-only {} B raw / {} B gz; with exporter paths {} B raw / {} B gz",
            hosts_only.len(),
            gz(&hosts_only),
            doc.len(),
            gz(&doc)
        );
        // The paths add at most ~14 bytes raw per rank over the hosts-only
        // map (a delta + comma, plus the shared dictionary entry).
        assert!(
            doc.len() - hosts_only.len() < 14 * n as usize,
            "{}",
            doc.len()
        );
    }

    #[test]
    fn parse_rank_hosts_round_trip_and_validation() {
        let mut host = BTreeMap::new();
        for r in 0..16i64 {
            host.insert(r, format!("host{}", r / 8));
        }
        host.insert(20, "host9".to_string());
        let doc = rank_map_payload(&host, &BTreeMap::new()).unwrap();
        assert_eq!(parse_rank_hosts(&doc).unwrap(), host);

        // Paths don't perturb the host decode.
        let mut path = BTreeMap::new();
        path.insert(3i64, "000003.t.gz".to_string());
        path.insert(30i64, "000030.t.gz".to_string()); // paths-only rank
        let doc = rank_map_payload(&host, &path).unwrap();
        assert_eq!(parse_rank_hosts(&doc).unwrap(), host);

        // Malformed docs are hard errors, not silent empties.
        assert!(parse_rank_hosts("not json").is_err());
        assert!(parse_rank_hosts(r#"{"v":2,"hosts":[],"runs":[]}"#).is_err());
        assert!(
            parse_rank_hosts(r#"{"v":1,"hosts":[],"runs":[[0,1,0,-1]]}"#).is_err(),
            "host index out of range must error"
        );
        assert!(
            parse_rank_hosts(r#"{"v":1,"hosts":["h"],"runs":[[0,9999999999,0,-1]]}"#).is_err(),
            "implausible run length must error"
        );
    }

    #[test]
    fn scan_trace_paths_filters_and_lexmax() {
        let td = tempfile::tempdir().unwrap();
        let d = td.path();
        for n in [
            "000000.1111.pt.trace.json.gz",
            "000000.2222.pt.trace.json.gz", // lex-max wins for rank 0
            "000001.merged.pt.trace.json.gz",
            "00002.x.pt.trace.json.gz",   // 5 digits → no match
            "0000021.x.pt.trace.json.gz", // 7 digits → no match
            "000003.something.json.txt",  // wrong suffix
            "000004.merged.pftrace.gz",   // perfetto-native Replay artifact
            "000005.merged.pftrace",      // ... uncompressed spelling
            "readme.txt",
        ] {
            std::fs::write(d.join(n), b"").unwrap();
        }
        let m = scan_trace_paths(d);
        assert_eq!(
            m.get(&0).map(String::as_str),
            Some("000000.2222.pt.trace.json.gz")
        );
        assert_eq!(
            m.get(&1).map(String::as_str),
            Some("000001.merged.pt.trace.json.gz")
        );
        assert_eq!(
            m.get(&4).map(String::as_str),
            Some("000004.merged.pftrace.gz")
        );
        assert_eq!(m.get(&5).map(String::as_str), Some("000005.merged.pftrace"));
        assert_eq!(m.len(), 4);
        // Missing dir → empty (a listing failure is non-fatal).
        assert!(scan_trace_paths(&d.join("nope")).is_empty());
    }

    #[test]
    fn emit_rank_hosts_from_profile_logs_or_nothing() {
        let td = tempfile::tempdir().unwrap();
        let run_root = td.path().join("tree/run1");
        std::fs::create_dir_all(&run_root).unwrap();

        // profile_logs fixture next to the trace root.
        let trace_root = td.path().join("job/traces/step_1");
        let logs = td.path().join("job/profile_logs/step_1");
        std::fs::create_dir_all(&logs).unwrap();
        std::fs::write(logs.join("000000.txt"), "{\"rank\": 0, \"host\": \"plh0\"}").unwrap();

        emit_rank_hosts(&run_root, &trace_root, 1).unwrap();
        assert_eq!(
            std::fs::read_to_string(run_root.join("rank_hosts.json")).unwrap(),
            r#"{"hosts":["plh0"],"runs":[[0,1,0,-1]],"sfx":[],"v":1}"#
        );

        // No hosts (no tier data) → no doc at all: trace paths are the
        // publisher's to re-derive, so a hostless run persists nothing.
        let bare_root = td.path().join("nolayout/step_1");
        let run2 = td.path().join("tree/run2");
        std::fs::create_dir_all(&run2).unwrap();
        emit_rank_hosts(&run2, &bare_root, 1).unwrap();
        assert!(!run2.join("rank_hosts.json").exists());
    }
}

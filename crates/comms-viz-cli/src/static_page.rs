// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Standalone HTML rendering from local comms_viz artifacts.
//!
//! The page is the template `static_page.html` with three placeholders
//! spliced in: the title, the critical JSON payload (a
//! `<script type="application/json">` island holding everything the first
//! paint needs — run/status/config docs, the PG summary, the sidecar
//! index, and small artifacts), and a second island carrying pg_map
//! verbatim, which the page parses lazily after first paint so paint cost
//! never scales with world size. Text artifacts embed plain below
//! GZ_THRESHOLD, gzip+base64 (`{"t":"gz64"}`, inflated in the browser via
//! `DecompressionStream("gzip")`) up to INLINE_BUDGET, and above that move
//! to the data sidecar (`{"t":"ref"}`) — the page range-fetches them after
//! first paint. Every `<` in island JSON is emitted as `<` so
//! artifact contents can never terminate the script island.

use std::io::Write as _;

use anyhow::Context;
use anyhow::Result;
use serde_json::json;

use crate::publish::RunData;
use crate::publish::Sidecar;

const TEMPLATE: &str = include_str!("static_page.html");

/// Below this size a text artifact embeds as plain JSON (readable in the
/// page source); at or above, gzip+base64 (~10x smaller for CSVs).
/// Larger payloads are cheaper compressed despite the base64 tax.
const GZ_THRESHOLD: usize = 16 * 1024;

/// An artifact whose encoded island value would exceed this moves to the
/// data sidecar instead of inlining. Small runs stay fully self-contained
/// (the page works from file://); on production-scale runs the shell stays
/// small enough to paint before any data arrives.
const INLINE_BUDGET: usize = 64 * 1024;

/// The breakdown preview embeds at most this many rows; longer CSVs are
/// bucket-averaged down to it. The chart itself plots one averaged dot
/// per x-pixel, so this is at or beyond screen resolution.
const PREVIEW_MAX_ROWS: usize = 2048;

fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn gzip(data: &[u8]) -> Result<Vec<u8>> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(data)?;
    Ok(enc.finish()?)
}

/// Encode a text artifact for embedding: `{"t":"plain","v":…}` or
/// `{"t":"gz64","v":…}` (the page's `decodeEnc` understands both).
pub(crate) fn enc(text: &str) -> Result<serde_json::Value> {
    if text.len() < GZ_THRESHOLD {
        Ok(json!({"t": "plain", "v": text}))
    } else {
        Ok(json!({"t": "gz64", "v": b64(&gzip(text.as_bytes())?)}))
    }
}

/// How one page artifact embeds: as an inline island value, or as a
/// reference to a sidecar member (its text then goes to the sidecar
/// builder and the page range-fetches it after first paint).
pub enum Emb {
    Inline(serde_json::Value),
    Ref,
}

fn embed_or_ref(text: &str) -> Result<Emb> {
    let v = enc(text)?;
    let inline_len = v["v"].as_str().map_or(0, str::len);
    if inline_len > INLINE_BUDGET {
        Ok(Emb::Ref)
    } else {
        Ok(Emb::Inline(v))
    }
}

/// The page's embed plan: which artifacts inline and which live in the
/// sidecar. Computed before the sidecar is written (the sidecar carries
/// the referenced texts) and consumed by `render`.
pub struct EmbedPlan {
    pub breakdown: Emb,
    pub rank_map: Option<Emb>,
    /// The default group's inline embed — base64 of the gzipped member
    /// payload, exactly what a sidecar range-fetch would yield (the page
    /// gunzips and magic-sniffs both the same way); None when it is
    /// sidecar-only (the group is a sidecar member either way).
    pub warm: Option<(String, String)>,
}

pub fn plan_embeds(data: &RunData) -> Result<EmbedPlan> {
    let warm = match data.per_pg_warm.first() {
        Some((stem, payload)) => {
            let b = b64(&gzip(payload)?);
            (b.len() <= INLINE_BUDGET).then(|| (stem.clone(), b))
        }
        None => None,
    };
    Ok(EmbedPlan {
        breakdown: embed_or_ref(&data.breakdown_csv)?,
        rank_map: data
            .rank_map_json
            .as_deref()
            .map(embed_or_ref)
            .transpose()?,
        warm,
    })
}

impl EmbedPlan {
    /// The (payload key, artifact text) list for `write_sidecar`, in the
    /// fixed artifact layout order the page's boot fetch assumes (adjacent,
    /// at the start of the sidecar).
    pub fn sidecar_artifacts<'a>(&self, data: &'a RunData) -> Vec<(&'static str, &'a str)> {
        let mut v: Vec<(&'static str, &'a str)> = Vec::new();
        if matches!(self.breakdown, Emb::Ref) {
            v.push(("breakdown", data.breakdown_csv.as_str()));
        }
        if matches!(self.rank_map, Some(Emb::Ref)) {
            if let Some(t) = data.rank_map_json.as_deref() {
                v.push(("rank_map", t));
            }
        }
        v
    }
}

/// Payload value for an optional artifact: absent → null, inline → the
/// enc() value, sidecar → `{"t":"ref"}` (the page resolves it through
/// `artifact_index`).
fn emb_value(emb: &Option<Emb>) -> serde_json::Value {
    match emb {
        None => serde_json::Value::Null,
        Some(Emb::Inline(v)) => v.clone(),
        Some(Emb::Ref) => json!({"t": "ref"}),
    }
}

pub(crate) fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Format a mean without noise: integers plainly, fractions to 2 places.
fn fmt_mean(x: f64) -> String {
    if (x - x.round()).abs() < 1e-9 {
        format!("{}", x.round() as i64)
    } else {
        format!("{x:.2}")
    }
}

/// A paint-ready stand-in for a sidecar-referenced breakdown CSV: only the
/// default-enabled series (the page's rule: comp_total/comm_total when
/// present, else the first two categories), rows sorted by rank and
/// bucket-averaged down to PREVIEW_MAX_ROWS, values rounded to integer
/// microseconds (every value the UI shows is integer-rounded). The exact
/// CSV replaces the preview when the sidecar fetch lands. Returns the
/// full category list (for the checkbox rail) and the mini CSV.
fn breakdown_preview(csv: &str) -> Option<(Vec<String>, String)> {
    let mut lines = csv.lines();
    let head: Vec<&str> = lines.next()?.split(',').collect();
    let ri = head.iter().position(|h| *h == "rank")?;
    let cats: Vec<String> = head
        .iter()
        .filter(|h| **h != "rank")
        .map(|s| s.to_string())
        .collect();
    if cats.is_empty() {
        return None;
    }
    let picked: Vec<&str> = {
        let defaults: Vec<&str> = ["comp_total", "comm_total"]
            .into_iter()
            .filter(|c| cats.iter().any(|x| x == c))
            .collect();
        if defaults.is_empty() {
            cats.iter().take(2).map(String::as_str).collect()
        } else {
            defaults
        }
    };
    let pidx: Vec<usize> = picked
        .iter()
        .map(|c| head.iter().position(|h| h == c).expect("picked from head"))
        .collect();

    // `+field || 0` semantics, like the page's parser.
    let num = |s: &str| {
        s.parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    };
    let mut rows: Vec<(f64, Vec<f64>)> = lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let fields: Vec<&str> = l.split(',').collect();
            let vals = pidx
                .iter()
                .map(|&i| num(fields.get(i).copied().unwrap_or("")))
                .collect();
            (num(fields.get(ri).copied().unwrap_or("")), vals)
        })
        .collect();
    if rows.is_empty() {
        return None;
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));

    let k = rows.len().div_ceil(PREVIEW_MAX_ROWS).max(1);
    let n_chunks = rows.len().div_ceil(k);
    let mut out = format!("rank,{}\n", picked.join(","));
    for (ci, chunk) in rows.chunks(k).enumerate() {
        let n = chunk.len() as f64;
        // The chart's x-domain is [first rank, last rank]: the end
        // buckets keep their exact endpoint rank (not the mean) so the
        // preview's axis is identical to the full data's — otherwise
        // the whole plot re-scales when the exact CSV swaps in.
        let rank = if ci == 0 {
            chunk[0].0
        } else if ci == n_chunks - 1 {
            chunk[chunk.len() - 1].0
        } else {
            chunk.iter().map(|r| r.0).sum::<f64>() / n
        };
        out.push_str(&fmt_mean(rank));
        for c in 0..picked.len() {
            let mean = chunk.iter().map(|r| r.1[c]).sum::<f64>() / n;
            out.push(',');
            out.push_str(&format!("{mean:.0}"));
        }
        out.push('\n');
    }
    Some((cats, out))
}

/// Per-PG display summary for the page's PG table (alias/count/size from
/// pg_info, pg_map rank-list dimensions as the fallback) minus the rank
/// lists — those stay in the lazily parsed pg_map island so first paint
/// never touches them.
fn pg_summary(data: &RunData) -> Vec<serde_json::Value> {
    let Some(map) = data.pg_map_json.as_ref().and_then(|v| v.as_object()) else {
        return Vec::new();
    };
    let groups_info = data
        .metadata_json
        .as_ref()
        .and_then(|m| m.get("pg_info"))
        .and_then(|p| p.get("groups"));
    map.iter()
        .map(|(key, v)| {
            let g = groups_info.and_then(|gi| gi.get(key));
            let groups = v.get("pg_ranks_map").and_then(serde_json::Value::as_array);
            let count = g
                .and_then(|g| g.get("count"))
                .and_then(serde_json::Value::as_i64)
                .unwrap_or_else(|| groups.map_or(0, |g| g.len() as i64));
            let size = g
                .and_then(|g| g.get("size"))
                .and_then(serde_json::Value::as_i64)
                .unwrap_or_else(|| {
                    groups
                        .and_then(|g| g.first())
                        .and_then(serde_json::Value::as_array)
                        .map_or(0, |a| a.len() as i64)
                });
            let alias = g
                .and_then(|g| g.get("alias"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or(key);
            json!({"key": key, "alias": alias, "count": count, "size": size})
        })
        .collect()
}

/// Compact sidecar group index: `[[prefix, [gid, len, gid, len, …]], …]`
/// in layout order (publish.rs's `layout_key` keeps every prefix's
/// groups consecutive), gid `null` for stems without a numeric suffix.
/// Offsets are derived client-side: groups start at `pg_base` and are
/// laid out back to back in index order.
fn compact_pg_index(sidecar: &Sidecar) -> Vec<serde_json::Value> {
    let mut out: Vec<(String, Vec<serde_json::Value>)> = Vec::new();
    for e in &sidecar.groups {
        let (prefix, gid) = match e.stem.rsplit_once('_') {
            Some((p, g)) if !g.is_empty() && g.bytes().all(|b| b.is_ascii_digit()) => {
                (p, g.parse::<u64>().ok().map(serde_json::Value::from))
            }
            _ => (e.stem.as_str(), None),
        };
        if out.last().map(|(p, _)| p.as_str()) != Some(prefix) {
            out.push((prefix.to_string(), Vec::new()));
        }
        let pairs = &mut out.last_mut().expect("just pushed").1;
        pairs.push(gid.unwrap_or(serde_json::Value::Null));
        pairs.push(json!(e.len));
    }
    out.into_iter()
        .map(|(p, pairs)| json!([p, pairs]))
        .collect()
}

/// Render the shell page. `emb` says which artifacts inline; `sidecar`
/// provides the byte-offset index for everything else. The critical
/// island stays small and world-size-independent — pg_map goes into its
/// own island at the end of the document, parsed lazily after paint.
pub fn render(
    data: &RunData,
    emb: &EmbedPlan,
    sidecar: &Sidecar,
    data_name: &str,
) -> Result<String> {
    let per_pg = match &emb.warm {
        // The value is base64 of the gzipped member payload (the page
        // gunzips + magic-sniffs it exactly like a sidecar range-fetch).
        Some((stem, b)) => vec![json!([stem, b])],
        None => Vec::new(),
    };

    let generated_at_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let (bd_value, bd_preview) = match &emb.breakdown {
        Emb::Inline(v) => (v.clone(), serde_json::Value::Null),
        Emb::Ref => (
            json!({"t": "ref"}),
            match breakdown_preview(&data.breakdown_csv) {
                Some((cats, csv)) => json!({"cats": cats, "csv": enc(&csv)?}),
                None => serde_json::Value::Null,
            },
        ),
    };

    let artifact_index: serde_json::Map<String, serde_json::Value> = sidecar
        .artifacts
        .iter()
        .map(|(key, off, len)| (key.to_string(), json!([off, len])))
        .collect();
    let pg_base = sidecar.groups.first().map_or(0, |e| e.offset);

    let payload = json!({
        "run_id": data.run_id,
        "result_dir": data.result_dir,
        "run": data.run_json,
        "config": data.config_json,
        "status": data.status_json,
        "metadata": data.metadata_json,
        "pg_summary": pg_summary(data),
        "breakdown": bd_value,
        "bd_preview": bd_preview,
        "per_pg": per_pg,
        "default_group": data.per_pg_warm.first().map(|(s, _)| s.clone()),
        "pg_index": compact_pg_index(sidecar),
        "pg_base": pg_base,
        "artifact_index": artifact_index,
        "data_name": if sidecar.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(data_name.to_string())
        },
        // Shard byte lengths when the sidecar spans several objects
        // (`<data_name>`, `<data_name>.1`, …); null for the single-object
        // layout, which the page treats exactly as before.
        "data_shards": if sidecar.shard_lens.len() >= 2 {
            json!(sidecar.shard_lens)
        } else {
            serde_json::Value::Null
        },
        "rank_map": emb_value(&emb.rank_map),
        "repaired_traces_dir": data.repaired_traces_dir,
        // Decimal STRINGS: real bases are epoch-scale ns (> 2^53), and the
        // page needs them BigInt-exact for Perfetto's slice select.
        "generated_at_s": generated_at_s,
    });
    // `<` never appears in JSON syntax outside strings, so a global escape
    // is safe and keeps `</script>`-bearing artifact content inert.
    let payload_text = serde_json::to_string(&payload)
        .context("serialize page payload")?
        .replace('<', "\\u003c");
    // The pg_map island is raw pg_map JSON when small (readable, parsed
    // directly) and an enc() wrapper when large — the page tells them
    // apart by the {"t":"gz64","v":…} shape, which a real pg_map (an
    // object of objects) can never take.
    let pgmap_text = {
        let raw = serde_json::to_string(&data.pg_map_json).context("serialize pg_map island")?;
        match enc(&raw)? {
            v if v["t"] == "gz64" => serde_json::to_string(&v)?,
            _ => raw,
        }
        .replace('<', "\\u003c")
    };

    let run_name = data
        .run_json
        .get("run_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let title = if run_name.is_empty() {
        let short: String = data.run_id.chars().take(8).collect();
        format!("comms_viz report {short}")
    } else {
        format!("{run_name} — comms_viz report")
    };

    let page = TEMPLATE
        .replacen("__COMMS_VIZ_TITLE__", &html_escape(&title), 1)
        .replacen("__COMMS_VIZ_DATA__", &payload_text, 1)
        .replacen("__COMMS_VIZ_PGMAP__", &pgmap_text, 1);
    Ok(page)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64_known_vectors() {
        assert_eq!(b64(b""), "");
        assert_eq!(b64(b"f"), "Zg==");
        assert_eq!(b64(b"fo"), "Zm8=");
        assert_eq!(b64(b"foo"), "Zm9v");
        assert_eq!(b64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn embed_or_ref_is_governed_by_encoded_size() {
        // Compresses to almost nothing → inlines as gz64 despite 200KB raw.
        match embed_or_ref(&"x".repeat(200 * 1024)).unwrap() {
            Emb::Inline(v) => assert_eq!(v["t"], "gz64"),
            Emb::Ref => panic!("compressible text must inline"),
        }
        // Small stays plain.
        match embed_or_ref("a,b\n1,2\n").unwrap() {
            Emb::Inline(v) => assert_eq!(v["t"], "plain"),
            Emb::Ref => panic!("small text must inline"),
        }
        // Over the budget once encoded (plain, 100KB of digits that gzip
        // can't crush below 64KB is hard to fake — use >GZ_THRESHOLD of
        // pseudo-random digits so the gz64 form still exceeds the budget).
        let mut x: u64 = 0x9e3779b97f4a7c15;
        let noise: String = (0..300 * 1024)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                char::from(b'0' + (x % 96) as u8)
            })
            .collect();
        assert!(matches!(embed_or_ref(&noise).unwrap(), Emb::Ref));
    }

    #[test]
    fn preview_picks_defaults_sorts_and_buckets() {
        let mut csv = String::from("other,comp_total,rank,comm_total\n");
        for r in 0..5000 {
            csv.push_str(&format!("1.5,{}.123,{},{}\n", r, 4999 - r, r * 2));
        }
        let (cats, mini) = breakdown_preview(&csv).unwrap();
        assert_eq!(cats, vec!["other", "comp_total", "comm_total"]);
        let lines: Vec<&str> = mini.lines().collect();
        assert_eq!(lines[0], "rank,comp_total,comm_total");
        assert!(lines.len() - 1 <= PREVIEW_MAX_ROWS);
        // Sorted by rank then bucket-averaged (k = ceil(5000/2048) = 3):
        // ranks {0,1,2} hold comp {4999.123, 4998.123, 4997.123} and comm
        // {9998, 9996, 9994} → value means 4998, 9996 — and the first
        // bucket keeps its exact endpoint rank (0, not the mean 1) so
        // the preview's x-domain matches the full data's.
        assert_eq!(lines[1], "0,4998,9996");
        assert!(
            lines[lines.len() - 1].starts_with("4999,"),
            "last bucket keeps the exact end rank: {}",
            lines[lines.len() - 1]
        );
    }

    #[test]
    fn preview_falls_back_to_first_two_categories() {
        let csv = "rank,alpha,beta,gamma\n0,1,2,3\n1,4,5,6\n";
        let (cats, mini) = breakdown_preview(csv).unwrap();
        assert_eq!(cats, vec!["alpha", "beta", "gamma"]);
        assert_eq!(mini, "rank,alpha,beta\n0,1,2\n1,4,5\n");
        // No rank column → no preview.
        assert!(breakdown_preview("a,b\n1,2\n").is_none());
    }

    #[test]
    fn compact_index_groups_by_prefix_with_derivable_offsets() {
        let sidecar = Sidecar {
            artifacts: vec![("breakdown", 0, 10)],
            groups: vec![
                crate::publish::PgIndexEntry {
                    stem: "pp_0".into(),
                    offset: 10,
                    len: 5,
                },
                crate::publish::PgIndexEntry {
                    stem: "pp_1".into(),
                    offset: 15,
                    len: 7,
                },
                crate::publish::PgIndexEntry {
                    stem: "odd".into(),
                    offset: 22,
                    len: 3,
                },
            ],
            shard_lens: vec![25],
        };
        let idx = compact_pg_index(&sidecar);
        assert_eq!(
            serde_json::Value::Array(idx),
            serde_json::json!([["pp", [0, 5, 1, 7]], ["odd", [null, 3]]])
        );
    }

    #[test]
    fn enc_switches_to_gz64_and_round_trips() {
        let small = enc("a,b\r\n1,2\r\n").unwrap();
        assert_eq!(small["t"], "plain");

        let big_text = "x".repeat(GZ_THRESHOLD);
        let big = enc(&big_text).unwrap();
        assert_eq!(big["t"], "gz64");
        // Round-trip: base64 → gunzip → original bytes.
        let raw = big["v"].as_str().unwrap();
        let bytes: Vec<u8> = {
            const T: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let vals: Vec<u8> = raw
                .bytes()
                .filter(|b| *b != b'=')
                .map(|b| T.find(b as char).unwrap() as u8)
                .collect();
            let mut out = Vec::new();
            for c in vals.chunks(4) {
                let n = c
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, v)| acc | (u32::from(*v) << (18 - 6 * i)));
                out.push((n >> 16) as u8);
                if c.len() > 2 {
                    out.push((n >> 8) as u8);
                }
                if c.len() > 3 {
                    out.push(n as u8);
                }
            }
            out
        };
        let mut dec = flate2::read::GzDecoder::new(&bytes[..]);
        let mut inflated = String::new();
        std::io::Read::read_to_string(&mut dec, &mut inflated).unwrap();
        assert_eq!(inflated, big_text);
    }
}

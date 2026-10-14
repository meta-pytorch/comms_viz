// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Annotation-pickle handling: decoding the side-channel pkl payloads and
//! live CUDA graph-id filtering of pkl names.
//!
//! The pkls are CPython pickle protocol 4/5 with heavy memo sharing of
//! *mutable* objects (lists referenced before their final APPENDs) —
//! serde-pickle rejects that shape ("recursive structure found"), so a
//! minimal arena-based pickle VM is implemented here for the observed
//! opcode set. Shared references mutate through the arena exactly like
//! CPython's object graph.
//!
//! A payload must unpickle to `dict[int, list[dict[str, value]]]`; any
//! other shape is an error. Value coercion: None/bool/int/float/str map to
//! typed JSON values; anything else is stringified with Python
//! `str()`/`repr()` semantics (e.g. a `Process Group Ranks` list becomes
//! `"[0, 8, 16]"`) — CPython formatting is the format the analysis
//! buckets on.

use std::collections::HashMap;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;

use crate::annotations::AnnDigest;
use crate::annotations::AnnotationDigests;
use crate::annotations::Annotations;

// ---------------------------------------------------------------------------
// Minimal pickle VM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum PObj {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<usize>),
    Dict(Vec<(usize, usize)>),
}

struct Vm<'a> {
    data: &'a [u8],
    pos: usize,
    arena: Vec<PObj>,
    stack: Vec<usize>,
    memo: Vec<usize>,
    marks: Vec<usize>,
}

impl<'a> Vm<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            bail!("pickle truncated");
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn push_obj(&mut self, o: PObj) {
        self.arena.push(o);
        self.stack.push(self.arena.len() - 1);
    }

    fn pop(&mut self) -> Result<usize> {
        self.stack
            .pop()
            .ok_or_else(|| anyhow!("pickle stack underflow"))
    }

    fn run(mut self) -> Result<(Vec<PObj>, usize)> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    self.u8()?; // PROTO version
                }
                0x95 => {
                    self.take(8)?; // FRAME length
                }
                b'}' => self.push_obj(PObj::Dict(Vec::new())),
                b']' => self.push_obj(PObj::List(Vec::new())),
                0x94 => {
                    // MEMOIZE
                    let top = *self.stack.last().ok_or_else(|| anyhow!("MEMOIZE empty"))?;
                    self.memo.push(top);
                }
                b'h' => {
                    // BINGET
                    let i = self.u8()? as usize;
                    let id = *self.memo.get(i).ok_or_else(|| anyhow!("BINGET {i}"))?;
                    self.stack.push(id);
                }
                b'j' => {
                    // LONG_BINGET
                    let b = self.take(4)?;
                    let i = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
                    let id = *self.memo.get(i).ok_or_else(|| anyhow!("LONG_BINGET {i}"))?;
                    self.stack.push(id);
                }
                0x8a => {
                    // LONG1: little-endian two's-complement
                    let n = self.u8()? as usize;
                    if n > 8 {
                        bail!("LONG1 wider than i64 ({n} bytes)");
                    }
                    let b = self.take(n)?;
                    let mut buf = [0u8; 8];
                    buf[..n].copy_from_slice(b);
                    // sign-extend
                    if n > 0 && b[n - 1] & 0x80 != 0 {
                        for x in buf.iter_mut().skip(n) {
                            *x = 0xff;
                        }
                    }
                    self.push_obj(PObj::Int(i64::from_le_bytes(buf)));
                }
                b'J' => {
                    let b = self.take(4)?;
                    self.push_obj(PObj::Int(
                        i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64
                    ));
                }
                b'K' => {
                    let v = self.u8()? as i64;
                    self.push_obj(PObj::Int(v));
                }
                b'M' => {
                    let b = self.take(2)?;
                    self.push_obj(PObj::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
                }
                0x8c => {
                    // SHORT_BINUNICODE
                    let n = self.u8()? as usize;
                    let s = std::str::from_utf8(self.take(n)?)?.to_string();
                    self.push_obj(PObj::Str(s));
                }
                b'X' => {
                    // BINUNICODE
                    let b = self.take(4)?;
                    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
                    let s = std::str::from_utf8(self.take(n)?)?.to_string();
                    self.push_obj(PObj::Str(s));
                }
                0x8d => {
                    // BINUNICODE8
                    let b = self.take(8)?;
                    let n = u64::from_le_bytes(b.try_into().unwrap()) as usize;
                    let s = std::str::from_utf8(self.take(n)?)?.to_string();
                    self.push_obj(PObj::Str(s));
                }
                b'N' => self.push_obj(PObj::None),
                0x88 => self.push_obj(PObj::Bool(true)),
                0x89 => self.push_obj(PObj::Bool(false)),
                b'G' => {
                    // BINFLOAT: big-endian f64
                    let b = self.take(8)?;
                    self.push_obj(PObj::Float(f64::from_be_bytes(b.try_into().unwrap())));
                }
                b'(' => self.marks.push(self.stack.len()),
                b'u' => {
                    // SETITEMS
                    let mark = self
                        .marks
                        .pop()
                        .ok_or_else(|| anyhow!("SETITEMS no mark"))?;
                    let items: Vec<usize> = self.stack.split_off(mark);
                    let dict_id = *self.stack.last().ok_or_else(|| anyhow!("SETITEMS"))?;
                    let PObj::Dict(d) = &mut self.arena[dict_id] else {
                        bail!("SETITEMS target not a dict");
                    };
                    for kv in items.chunks(2) {
                        d.push((kv[0], kv[1]));
                    }
                }
                b's' => {
                    // SETITEM
                    let v = self.pop()?;
                    let k = self.pop()?;
                    let dict_id = *self.stack.last().ok_or_else(|| anyhow!("SETITEM"))?;
                    let PObj::Dict(d) = &mut self.arena[dict_id] else {
                        bail!("SETITEM target not a dict");
                    };
                    d.push((k, v));
                }
                b'a' => {
                    // APPEND
                    let v = self.pop()?;
                    let list_id = *self.stack.last().ok_or_else(|| anyhow!("APPEND"))?;
                    let PObj::List(l) = &mut self.arena[list_id] else {
                        bail!("APPEND target not a list");
                    };
                    l.push(v);
                }
                b'e' => {
                    // APPENDS
                    let mark = self.marks.pop().ok_or_else(|| anyhow!("APPENDS no mark"))?;
                    let items: Vec<usize> = self.stack.split_off(mark);
                    let list_id = *self.stack.last().ok_or_else(|| anyhow!("APPENDS"))?;
                    let PObj::List(l) = &mut self.arena[list_id] else {
                        bail!("APPENDS target not a list");
                    };
                    l.extend(items);
                }
                b'.' => {
                    let root = self.pop()?;
                    return Ok((self.arena, root));
                }
                other => bail!(
                    "unsupported pickle opcode 0x{other:02x} at {}",
                    self.pos - 1
                ),
            }
        }
    }
}

fn parse_pickle(data: &[u8]) -> Result<(Vec<PObj>, usize)> {
    Vm {
        data,
        pos: 0,
        arena: Vec::with_capacity(1024),
        stack: Vec::with_capacity(64),
        memo: Vec::with_capacity(1024),
        marks: Vec::with_capacity(8),
    }
    .run()
}

use crate::host_map::py_float_repr;

/// Python `repr()` for pickled objects (used via the `str()` fallback for
/// non-primitive annotation values; containers repr their elements).
fn py_repr(arena: &[PObj], id: usize, out: &mut String) {
    match &arena[id] {
        PObj::None => out.push_str("None"),
        PObj::Bool(true) => out.push_str("True"),
        PObj::Bool(false) => out.push_str("False"),
        PObj::Int(i) => out.push_str(&i.to_string()),
        PObj::Float(f) => out.push_str(&py_float_repr(*f)),
        PObj::Str(s) => {
            // Python repr: single quotes, switching to double when the
            // string contains ' but no ".
            let has_sq = s.contains('\'');
            let has_dq = s.contains('"');
            let (q, esc_q) = if has_sq && !has_dq {
                ('"', '"')
            } else {
                ('\'', '\'')
            };
            out.push(q);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == esc_q => {
                        out.push('\\');
                        out.push(c);
                    }
                    c => out.push(c),
                }
            }
            out.push(q);
        }
        PObj::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_repr(arena, *item, out);
            }
            out.push(']');
        }
        PObj::Dict(entries) => {
            out.push('{');
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_repr(arena, *k, out);
                out.push_str(": ");
                py_repr(arena, *v, out);
            }
            out.push('}');
        }
    }
}

/// Primitives map to typed JSON values; anything else becomes `str(v)`.
/// (`str` of a container == `repr` of it.)
fn pobj_to_json(arena: &[PObj], id: usize) -> serde_json::Value {
    match &arena[id] {
        PObj::None => serde_json::Value::Null,
        PObj::Bool(b) => serde_json::Value::Bool(*b),
        PObj::Int(i) => serde_json::Value::from(*i),
        PObj::Float(f) => serde_json::Value::from(*f),
        PObj::Str(s) => serde_json::Value::String(s.clone()),
        _ => {
            let mut s = String::new();
            py_repr(arena, id, &mut s);
            serde_json::Value::String(s)
        }
    }
}

/// [`load_annotations_from_pkl_bytes`], digested for the row-walk path: same
/// unpickling and same error semantics, but only the fields the row walk
/// reads are materialised (see [`AnnDigest`]).
/// Values go through the identical `pobj_to_json` → `python_str` /
/// `coerce_to_i64` conversions, so the columns the walk writes are
/// byte-identical to the full-map path.
pub fn load_digests_from_pkl_bytes(pkls: &[Vec<u8>]) -> Result<AnnotationDigests> {
    let mut out: AnnotationDigests = AnnotationDigests::default();
    for pkl in pkls {
        let (arena, root) = parse_pickle(pkl)?;
        let PObj::Dict(entries) = &arena[root] else {
            bail!("annotation pkl root is not a dict");
        };
        for (k_id, v_id) in entries {
            let key = match &arena[*k_id] {
                PObj::Int(i) => *i,
                other => bail!("annotation key must be int, got {other:?}"),
            };
            let PObj::List(anns) = &arena[*v_id] else {
                bail!("annotation pkl value is not a list");
            };
            let slot = out.entry(key).or_default();
            for ann_id in anns {
                let PObj::Dict(fields) = &arena[*ann_id] else {
                    // A non-dict annotation entry is a hard error, not a skip.
                    bail!("annotation entry is not a dict");
                };
                let mut d = AnnDigest::default();
                for (fk, fv) in fields {
                    let name = match &arena[*fk] {
                        PObj::Str(s) => s.as_str(),
                        other => bail!("annotation field key must be str, got {other:?}"),
                    };
                    if crate::comm::CommMetadata::key_index(name).is_some() {
                        d.comm.set(name, digest_str(&arena, *fv).into());
                    }
                    // Duplicate keys cannot survive the dict this came from,
                    // but assigning in order keeps last-wins either way.
                    match name {
                        "Process Group Description" => d.pg_desc = Some(digest_str(&arena, *fv)),
                        "Process Group Name" => d.pg_name = Some(digest_str(&arena, *fv)),
                        "stream" => {
                            d.stream = crate::cudagraph::coerce_to_i64(&pobj_to_json(&arena, *fv))
                        }
                        // Compact communication fields were captured above.
                        _ => {}
                    }
                }
                slot.push(d);
            }
        }
    }
    Ok(out)
}

/// `python_str(pobj_to_json(v))` — the walk's own value stringification.
fn digest_str(arena: &[PObj], id: usize) -> Box<str> {
    crate::cudagraph::python_str(&pobj_to_json(arena, id)).into()
}

/// Digest an already-materialised full annotations table (verify/tree paths
/// need both representations of the same content).
/// Rendered `(key, python_str value)` upserts per gnid for the reassign
/// splice, restricted to `wanted` (the plan's retagged gnids). Folded
/// across each gnid's annotation list in order — later dicts overwrite,
/// matching the tree twin's sequential `args.insert` — with `stream`
/// excluded: the splice writes it numerically, mirroring `walk_annotate`'s
/// final `args["stream"]` overwrite. BTreeMap fold keeps the emit order
/// deterministic (the artifact is a pure function of the inputs).
pub fn render_ann_args(
    a: &Annotations,
    wanted: &rustc_hash::FxHashSet<i64>,
) -> crate::legacy::splice::RenderedAnnArgs {
    let mut out = crate::legacy::splice::RenderedAnnArgs::default();
    for (gid, anns) in a {
        if !wanted.contains(gid) {
            continue;
        }
        let mut fold: std::collections::BTreeMap<&str, String> = std::collections::BTreeMap::new();
        for ann in anns {
            for (k, v) in ann {
                if k != "stream" {
                    fold.insert(k.as_str(), crate::cudagraph::python_str(v));
                }
            }
        }
        if !fold.is_empty() {
            out.insert(
                *gid,
                fold.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            );
        }
    }
    out
}

pub fn digests_from_annotations(a: &Annotations) -> AnnotationDigests {
    let mut out: AnnotationDigests = AnnotationDigests::default();
    for (gid, anns) in a {
        let slot = out
            .entry(*gid)
            .or_insert_with(|| Vec::with_capacity(anns.len()));
        for ann in anns {
            let get = |k: &str| ann.get(k).map(|v| crate::cudagraph::python_str(v).into());
            let mut comm = crate::comm::CommMetadata::default();
            for (key, value) in ann {
                if crate::comm::CommMetadata::key_index(key).is_some() {
                    comm.set(key, crate::cudagraph::python_str(value));
                }
            }
            slot.push(AnnDigest {
                comm,
                pg_desc: get("Process Group Description"),
                pg_name: get("Process Group Name"),
                stream: ann.get("stream").and_then(crate::cudagraph::coerce_to_i64),
            });
        }
    }
    out
}

/// Unpickle each payload (`dict[int, list[dict]]`) and merge lists per key
/// across payloads.
pub fn load_annotations_from_pkl_bytes(pkls: &[Vec<u8>]) -> Result<Annotations> {
    let mut annotations: Annotations = HashMap::new();
    for pkl in pkls {
        let (arena, root) = parse_pickle(pkl)?;
        let PObj::Dict(entries) = &arena[root] else {
            bail!("annotation pkl root is not a dict");
        };
        for (k_id, v_id) in entries {
            let key = match &arena[*k_id] {
                PObj::Int(i) => *i,
                other => bail!("annotation key must be int, got {other:?}"),
            };
            let PObj::List(anns) = &arena[*v_id] else {
                bail!("annotation pkl value is not a list");
            };
            let out = annotations.entry(key).or_default();
            for ann_id in anns {
                let PObj::Dict(fields) = &arena[*ann_id] else {
                    // A non-dict annotation entry is a hard error, not a skip.
                    bail!("annotation entry is not a dict");
                };
                let mut h: HashMap<String, serde_json::Value> =
                    HashMap::with_capacity(fields.len());
                for (fk, fv) in fields {
                    let name = match &arena[*fk] {
                        PObj::Str(s) => s.clone(),
                        other => bail!("annotation field key must be str, got {other:?}"),
                    };
                    h.insert(name, pobj_to_json(&arena, *fv));
                }
                out.push(h);
            }
        }
    }
    Ok(annotations)
}

/// Scan (decompressed) replay bytes for `"graph node id": N` and return
/// the set of `N >> 32`.
pub fn extract_live_graph_ids(replay_bytes: &[u8]) -> std::collections::HashSet<i64> {
    let data = match crate::rows::maybe_gunzip(replay_bytes) {
        Ok(d) => d,
        Err(_) => return Default::default(),
    };
    extract_live_graph_ids_from_raw(&data)
}

/// Same scan over ALREADY-decompressed bytes (fast path decompresses once).
/// memmem + manual tail parse of `"graph node id"\s*:\s*(\d+)`: the regex
/// version ran the full engine over the 38MB buffer (~43% of all rank-loop
/// cycles); the SIMD literal search only visits actual key sites.
pub fn extract_live_graph_ids_from_raw(data: &[u8]) -> std::collections::HashSet<i64> {
    const NEEDLE: &[u8] = b"\"graph node id\"";
    let is_ws = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c);
    let mut out = std::collections::HashSet::new();
    let finder = memchr::memmem::Finder::new(NEEDLE);
    for pos in finder.find_iter(data) {
        let mut i = pos + NEEDLE.len();
        while i < data.len() && is_ws(data[i]) {
            i += 1;
        }
        if i >= data.len() || data[i] != b':' {
            continue;
        }
        i += 1;
        while i < data.len() && is_ws(data[i]) {
            i += 1;
        }
        let start = i;
        while i < data.len() && data[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            continue;
        }
        // Digit runs too wide for u64 are skipped, not clamped.
        if let Ok(v) = unsafe { std::str::from_utf8_unchecked(&data[start..i]) }.parse::<u64>() {
            out.insert((v >> 32) as i64);
        }
    }
    out
}

/// Distinct byte contents the pkl caches will retain before refusing new
/// inserts. The caches exist for SPMD jobs, where every rank fetches an
/// IDENTICAL payload — a handful of entries covers those completely. On
/// per-rank-DISTINCT payloads an unbounded cache accreted ~10 MB of
/// live digests per completed rank for the run's lifetime (~250 GB at 24k
/// ranks; the memory gate squeezed analysis width to ~10 defending against
/// it — and the pre-digest form grew one analysis process to 432 GB).
/// Past the cap the workload is per-rank-distinct, hits were never coming,
/// and callers simply parse.
const PKL_CACHE_CAP: usize = 64;

/// Content-hash-cached annotation loading: the per-suffix pkl payloads are
/// identical across ranks in SPMD jobs, so parse each distinct byte content
/// once per process (bounded — see [`PKL_CACHE_CAP`]).
pub fn load_annotations_cached(pkls: &[Vec<u8>]) -> Result<std::sync::Arc<Annotations>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    static CACHE: OnceLock<Mutex<HashMap<u64, Arc<Annotations>>>> = OnceLock::new();
    let mut h = DefaultHasher::new();
    for p in pkls {
        p.hash(&mut h);
    }
    let key = h.finish();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(hit) = cache.lock().unwrap().get(&key) {
        return Ok(hit.clone());
    }
    let parsed = Arc::new(load_annotations_from_pkl_bytes(pkls)?);
    let mut c = cache.lock().unwrap();
    if c.len() < PKL_CACHE_CAP {
        c.insert(key, parsed.clone());
    }
    Ok(parsed)
}

/// [`load_annotations_cached`] for the row-walk path: the cache retains the
/// digested form, which is what dominates the process's steady-state RSS at world
/// scale (the full maps grew one analysis process to 432GB — every distinct payload's
/// `HashMap<String, Value>` retained for the whole run).
pub fn load_digests_cached(pkls: &[Vec<u8>]) -> Result<std::sync::Arc<AnnotationDigests>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    static CACHE: OnceLock<Mutex<HashMap<u64, Arc<AnnotationDigests>>>> = OnceLock::new();
    let mut h = DefaultHasher::new();
    for p in pkls {
        p.hash(&mut h);
    }
    let key = h.finish();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(hit) = cache.lock().unwrap().get(&key) {
        return Ok(hit.clone());
    }
    let parsed = Arc::new(load_digests_from_pkl_bytes(pkls)?);
    let mut c = cache.lock().unwrap();
    if c.len() < PKL_CACHE_CAP {
        c.insert(key, parsed.clone());
    }
    Ok(parsed)
}

/// Keep pkl paths whose trailing `_{graph_id}.pkl` matches a live id;
/// all when empty or none match.
pub fn filter_pkl_uris_by_graph_ids(
    uris: Vec<String>,
    live_graph_ids: &std::collections::HashSet<i64>,
) -> Vec<String> {
    if live_graph_ids.is_empty() {
        return uris;
    }
    let re = regex::Regex::new(r"_(\d+)\.pkl$").unwrap();
    let filtered: Vec<String> = uris
        .iter()
        .filter(|uri| {
            re.captures(uri)
                .and_then(|c| c[1].parse::<i64>().ok())
                .is_some_and(|id| live_graph_ids.contains(&id))
        })
        .cloned()
        .collect();
    if filtered.is_empty() { uris } else { filtered }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_real_shape() {
        // Build a pkl-like structure with python? Not available in unit
        // tests — construct bytes via a tiny hand-rolled protocol-4 stream:
        // {433791696896: [{'name': 'ag_copy_in'}]}
        let mut b: Vec<u8> = vec![0x80, 4];
        b.push(b'}'); // dict
        b.push(0x94); // memoize
        b.push(0x8a); // LONG1
        let key: i64 = 433791696896;
        let kb = key.to_le_bytes();
        b.push(6);
        b.extend_from_slice(&kb[..6]);
        b.push(b']'); // list
        b.push(0x94);
        b.push(b'}'); // inner dict
        b.push(0x94);
        b.push(0x8c);
        b.push(4);
        b.extend_from_slice(b"name");
        b.push(0x8c);
        b.push(10);
        b.extend_from_slice(b"ag_copy_in");
        b.push(b's'); // SETITEM
        b.push(b'a'); // APPEND into list
        b.push(b's'); // SETITEM key->list into root dict
        b.push(b'.');
        let ann = load_annotations_from_pkl_bytes(&[b]).unwrap();
        assert_eq!(ann.len(), 1);
        let v = &ann[&433791696896];
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["name"], serde_json::Value::String("ag_copy_in".into()));
    }

    #[test]
    fn shared_mutable_memo() {
        // lst = [1]; d = {1: lst, 2: lst}; lst referenced before final
        // append — the serde-pickle failure shape.
        let mut b: Vec<u8> = vec![0x80, 4];
        b.push(b'}');
        b.push(0x94); // memo 0 = dict
        b.push(b'K');
        b.push(1); // key 1
        b.push(b']');
        b.push(0x94); // memo 1 = list
        b.push(b'K');
        b.push(7);
        b.push(b'a'); // list.append(7)
        b.push(b's'); // d[1] = list
        b.push(b'K');
        b.push(2);
        b.push(b'h');
        b.push(1); // BINGET memo 1 (shared list)
        b.push(b'K');
        b.push(9);
        b.push(b'a'); // append AFTER sharing
        b.push(b's'); // d[2] = list
        b.push(b'.');
        let (arena, root) = parse_pickle(&b).unwrap();
        let PObj::Dict(entries) = &arena[root] else {
            panic!()
        };
        assert_eq!(entries.len(), 2);
        // Both values are the same arena id and contain [7, 9].
        assert_eq!(entries[0].1, entries[1].1);
        let PObj::List(l) = &arena[entries[0].1] else {
            panic!()
        };
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn repr_of_list() {
        let (arena, root) = {
            // [0, 8, 'x']
            let mut b: Vec<u8> = vec![0x80, 4];
            b.push(b']');
            b.push(0x94);
            b.push(b'(');
            b.push(b'K');
            b.push(0);
            b.push(b'K');
            b.push(8);
            b.push(0x8c);
            b.push(1);
            b.push(b'x');
            b.push(b'e');
            b.push(b'.');
            parse_pickle(&b).unwrap()
        };
        let mut s = String::new();
        py_repr(&arena, root, &mut s);
        assert_eq!(s, "[0, 8, 'x']");
    }
}

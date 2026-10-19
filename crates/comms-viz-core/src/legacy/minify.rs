// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! JSON whitespace stripper: copy `src` dropping whitespace outside strings
//! — the splice writer's copy primitive (producers pretty-print; verbatim
//! copies made the artifact ~13% larger, paid on every upload and in
//! storage forever).
//!
//! Two implementations with identical output for well-formed JSON (which is
//! all the splice path can see: every spliced document already survived the
//! tape parse, and scan/splice reject anything structurally surprising):
//!
//!   - [`copy_min_scalar`]: byte loop, strings jumped via memchr — the
//!     reference. ~0.4 GB/s on pretty-printed Kineto traces (whitespace
//!     every few bytes defeats whitespace-needle memchr).
//!   - AVX-512 (`VBMI2` + `PCLMULQDQ`): simdjson-style string masking
//!     (carryless-multiply prefix XOR over unescaped quotes) and
//!     `vpcompressb` left-packing, 64 bytes per iteration with no
//!     byte-level branches. The splice writer spends ~35% of the analyze
//!     window's CPU walking these bytes at scalar speed (08-19 profile,
//!     10240-rank job).
//!
//! Divergence is possible only on malformed input (e.g. a backslash outside
//! any string, which valid JSON cannot contain): the scalar treats it as an
//! ordinary byte, the mask pipeline lets it escape a following quote. The
//! dispatcher is still safe there — both variants strip no byte inside what
//! they believe is a string, and such documents never reach the splice
//! writer (the tape parse rejected them first).
//!
//! `COMMS_VIZ_NO_SIMD_MINIFY=1` pins the scalar path (debug escape hatch).

/// Append `src` to `out` with JSON whitespace outside strings removed.
pub fn copy_min(out: &mut Vec<u8>, src: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static USABLE: OnceLock<bool> = OnceLock::new();
        let usable = *USABLE.get_or_init(|| {
            std::env::var_os("COMMS_VIZ_NO_SIMD_MINIFY").is_none()
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vbmi2")
                && std::arch::is_x86_feature_detected!("pclmulqdq")
        });
        if usable {
            // SAFETY: required CPU features verified above.
            unsafe { avx512::copy_min(out, src) };
            return;
        }
    }
    copy_min_scalar(out, src)
}

/// Scalar reference: whitespace scan with batched copies; string content
/// jumps via the memchr-backed skip. Escape handling matches
/// `splice::skip_string` (an escape pair is consumed as a unit; an
/// unterminated string runs to the end of the input, copied verbatim).
pub fn copy_min_scalar(out: &mut Vec<u8>, src: &[u8]) {
    let mut i = 0;
    let mut run = 0; // start of the pending verbatim segment
    while i < src.len() {
        match src[i] {
            b' ' | b'\t' | b'\n' | b'\r' => {
                out.extend_from_slice(&src[run..i]);
                i += 1;
                run = i;
            }
            b'"' => {
                // The whole string stays inside the running segment.
                i = skip_string_clamped(src, i);
            }
            _ => i += 1,
        }
    }
    out.extend_from_slice(&src[run..]);
}

/// `i` at the opening quote; index just past the closing quote, clamped to
/// `src.len()` when the string never closes.
fn skip_string_clamped(d: &[u8], i: usize) -> usize {
    let mut j = i + 1;
    while j < d.len() {
        match memchr::memchr2(b'"', b'\\', &d[j..]) {
            Some(k) if d[j + k] == b'"' => return j + k + 1,
            Some(k) => j += k + 2, // step past the escape pair
            None => break,
        }
    }
    d.len()
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Bit i of the result = XOR of bits 0..=i of `q` (carryless multiply
    /// by all-ones): after masking escaped quotes out of `q`, this is the
    /// "inside a string" mask, simdjson-style.
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    unsafe fn prefix_xor(q: u64) -> u64 {
        let v = _mm_set_epi64x(0, q as i64);
        let ones = _mm_set1_epi8(-1);
        _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(v, ones)) as u64
    }

    /// Mask of bytes escaped by a preceding unescaped backslash (simdjson's
    /// odd-length-backslash-run scanner); `prev` carries "the next block's
    /// first byte is escaped" across the 64-byte seam.
    #[inline]
    fn find_escaped(backslash: u64, prev: &mut u64) -> u64 {
        if backslash == 0 {
            let e = *prev;
            *prev = 0;
            return e;
        }
        const EVEN_BITS: u64 = 0x5555_5555_5555_5555;
        let backslash = backslash & !*prev;
        let follows_escape = backslash << 1 | *prev;
        let odd_sequence_starts = backslash & !EVEN_BITS & !follows_escape;
        let (sequences_starting_on_even, overflow) = odd_sequence_starts.overflowing_add(backslash);
        *prev = overflow as u64;
        (EVEN_BITS ^ (sequences_starting_on_even << 1)) & follows_escape
    }

    /// 64 bytes per iteration: classify quotes/escapes/whitespace as bit
    /// masks, derive the in-string mask, left-pack the kept bytes with
    /// `vpcompressb`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi2,pclmulqdq")]
    pub unsafe fn copy_min(out: &mut Vec<u8>, src: &[u8]) {
        // +64: the compressed store always writes a full vector; the tail
        // block may overhang the packed length by up to 63 bytes.
        out.reserve(src.len() + 64);
        let mut written = out.len();
        let mut prev_in_string: u64 = 0; // all-ones while inside a string
        let mut prev_escaped: u64 = 0;
        let mut i = 0;
        while i < src.len() {
            let rem = src.len() - i;
            let tail: u64 = if rem >= 64 { !0 } else { (1u64 << rem) - 1 };
            let v = _mm512_maskz_loadu_epi8(tail, src.as_ptr().add(i) as *const i8);
            let quotes = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'"' as i8)) & tail;
            let bs = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\\' as i8)) & tail;
            let escaped = find_escaped(bs, &mut prev_escaped);
            let in_string = prefix_xor(quotes & !escaped) ^ prev_in_string;
            prev_in_string = (in_string as i64 >> 63) as u64;
            let ws = (_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b' ' as i8))
                | _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\t' as i8))
                | _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\n' as i8))
                | _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\r' as i8)))
                & tail;
            let keep = tail & !(ws & !in_string);
            let packed = _mm512_maskz_compress_epi8(keep, v);
            _mm512_storeu_si512(out.as_mut_ptr().add(written) as *mut __m512i, packed);
            written += keep.count_ones() as usize;
            i += 64;
        }
        out.set_len(written);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(src: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        copy_min_scalar(&mut out, src);
        out
    }

    fn dispatched(src: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        copy_min(&mut out, src);
        out
    }

    #[track_caller]
    fn assert_same(src: &[u8]) {
        assert_eq!(
            scalar(src),
            dispatched(src),
            "divergence on {:?}",
            String::from_utf8_lossy(src)
        );
    }

    #[test]
    fn strips_ws_outside_strings_only() {
        let src = b"{ \"a b\": \"c \\\" d\" ,\n\t\"n\": 1.5 }";
        assert_eq!(scalar(src), b"{\"a b\":\"c \\\" d\",\"n\":1.5}");
        assert_same(src);
    }

    #[test]
    fn edge_shapes() {
        assert_same(b"");
        assert_same(b" \n\t\r ");
        assert_same(b"\"unterminated with  spaces");
        assert_same(b"\"ends in backslash \\");
        assert_same(br#"{"a":"\\","b":"\\\" x ","c":"\n \t"}"#);
        // Escape runs and quotes straddling the 64-byte block seam.
        for pad in 48..80 {
            let mut s = vec![b'{'; 1];
            s.extend(std::iter::repeat_n(b' ', pad));
            s.extend_from_slice(br#""k \\\\ \" v ":  "x  y""#);
            s.extend(std::iter::repeat_n(b'\\', pad % 7));
            s.push(b'"');
            assert_same(&s);
        }
    }

    #[test]
    fn appends_after_existing_content() {
        let mut out = b"prefix".to_vec();
        copy_min(&mut out, b" {\"a\" : 1} ");
        assert_eq!(out, b"prefix{\"a\":1}");
    }

    #[test]
    fn differential_random_json() {
        // Deterministic LCG over JSON-ish fragments: valid documents with
        // heavy whitespace, escapes, and long strings, at lengths chosen to
        // cross block seams.
        let mut state = 0x243F6A8885A308D3u64;
        let mut rng = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let frags: &[&[u8]] = &[
            b"{\"ph\": \"X\", \"name\": \"kernel a b\", \"ts\": 123.5}",
            b"  ",
            b"\n\t",
            b"[1, 2,  3]",
            b"\"s p a c e s\"",
            br#""esc \" and \\ and \\\" mix ""#,
            b"true",
            b"null,  ",
            br#"{"args": {"stream": 7, "grid": [1, 2, 3]}}"#,
            "\"unicode \u{e9} \\t tab\"".as_bytes(),
        ];
        for _ in 0..2000 {
            let mut s = Vec::new();
            for _ in 0..(rng() % 12) {
                s.extend_from_slice(frags[rng() % frags.len()]);
            }
            assert_same(&s);
        }
    }

    #[test]
    fn differential_real_trace_shapes() {
        // A pretty-printed Kineto-style document, one event per line.
        let mut doc = Vec::new();
        doc.extend_from_slice(b"{\n  \"schemaVersion\": 1,\n  \"traceEvents\": [\n");
        for i in 0..500 {
            doc.extend_from_slice(
                format!(
                    "  {{\"ph\": \"X\", \"cat\": \"kernel\", \"name\": \"void foo<int, {i}>(bar \\\"q\\\")\", \
                     \"pid\": {i}, \"tid\": {}, \"ts\": {}.{}, \"dur\": 4, \
                     \"args\": {{\"stream\": {}, \"correlation\": {i}}}}},\n",
                    i % 7,
                    1000 + i,
                    i % 10,
                    i % 3
                )
                .as_bytes(),
            );
        }
        doc.extend_from_slice(b"  {\"ph\": \"M\"}\n]\n}\n");
        assert_same(&doc);
    }
}

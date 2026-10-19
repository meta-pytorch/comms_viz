// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Memory-gated admission for the parallel rank loop.
//!
//! The rank loop sizes its worker pool from CPU topology alone, but each
//! in-flight rank holds its decompressed trace plus parse structures —
//! several GB for large gzipped JSON ranks. On big-RAM tuning hosts that
//! product never mattered; on a shared CI machine 160 workers × ~2 GB
//! drove the box into reclaim livelock and the OOM killer took the whole
//! agent down with the run.
//!
//! `MemGate` bounds the *memory* in flight instead of the worker count:
//! every rank must acquire an estimated-cost permit before its first big
//! allocation. Admission is FIFO (tickets) so one huge rank cannot starve
//! behind a stream of small ones, and an empty gate always admits — a
//! single rank larger than the whole budget still gets attempted alone.
//! Two conditions gate each admission:
//!   1. ledger: sum of admitted estimates stays under a budget fixed at
//!      construction (a fraction of the effective headroom, split across
//!      shards) — bounds allocation *velocity* the live signal can't see;
//!   2. live floor: current effective headroom (host MemAvailable min
//!      cgroup limit-minus-current) keeps a reserve — absorbs estimate
//!      error, co-tenants, and tmpfs staging growth.
//!
//! Estimates come from the gzip ISIZE trailer (exact decompressed size for
//! single-member <4 GiB streams) times a parse-structure factor. Raw
//! (non-gzip) formats estimate from file size, so small pftrace ranks pass
//! through effectively ungated. The gate changes scheduling only — artifact
//! bytes are shard/thread-shape invariant.
//!
//! The ledger and floor alone are NOT sufficient on managed hosts: the
//! binding limit there is an ancestor cgroup (the container's task) that a
//! namespaced container cannot read, so both signals overestimate the
//! allowance (RSS plateaued at 170 GB with 48 GB host
//! MemAvailable — every thread stalled in ancestor-limit reclaim, zero
//! ranks completed, and the scheduler's sustained-pressure policy killed the
//! task at exactly 8 minutes, twice). The *pressure* that regime creates
//! IS observable from inside (/proc/pressure/memory, and the container
//! root's memory.pressure — a stalled descendant accounts to every
//! ancestor), so admission runs TCP-style AIMD on it:
//!   3. slow start: the in-flight cap begins at a fraction of the worker
//!      pool and grows additively per completed rank while pressure is
//!      quiet — the full ledger is never admitted in one burst;
//!   4. backoff: CONFIRMED stall pressure — a hot smoothed signal, or
//!      two consecutive hot fast-attack polls — halves the cap
//!      (hysteresis latched until it subsides). Admissions keep flowing
//!      under the halved cap whenever the fast signal is quiet; only cap
//!      growth waits for the episode to end (TCP shape: halve the
//!      window, keep sending). Confirmation matters: 1–2 s kernel-MM
//!      micro-bursts read hot on a single 500 ms poll, and ratcheting on
//!      each one costs a quarter of the concurrency.
//!
//! Two measured corrections close the static model's error (the
//! model was wrong in BOTH directions at once: giant fused
//! ranks estimate ~1.3-3× their footprint and pinned width at ~20
//! of 160 for an hour, while tree-path ranks under-estimated and OOM'd):
//!   5. cost correction: the footprint EWMA is paired with an EWMA of the
//!      same ranks' static estimates, and future admissions are charged
//!      cost × foot/est (bounded) — the ledger converges on the real
//!      per-rank footprint in either direction;
//!   6. parked-worker shed: a worker parked at the gate ≥5 s sheds its
//!      retained pools (+ rate-limited purge) even on a benign cap block —
//!      at giant-step width the parked majority's pools were most of RSS;
//!   7. own-RSS ceiling: admissions keep self RSS + cost under 80% of the
//!      construction-time allowance — the managed-host ancestor limit
//!      kills ~15-20% below the advertised headroom with PSI quiet, so no
//! reactive signal can fire in time (dead at RSS
//!      220-225 GB with 45-115 GB "available").
//!
//! Env overrides: COMMS_VIZ_MEM_BUDGET_GB (per-process ledger budget; 0
//! disables the gate entirely), COMMS_VIZ_MEM_RANK_FACTOR (per-rank
//! multiplier over the decompressed size).

use std::path::Path;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

/// Fraction of the effective headroom the ledger budget may claim.
const BUDGET_FRACTION_PCT: u64 = 70;
/// Estimated peak bytes per decompressed trace byte (scratch + rows/raws
/// pools + converted columns; thread pools retain to RETAIN_FACTOR).
const DEFAULT_RANK_FACTOR: u64 = 3;
/// Gzipped (Chrome-JSON) ranks: measured live-set per in-flight rank, not a
/// modeled parse expansion. The old ×10 was calibrated against the simd-json
/// tape parser; the raw-byte scan parser
/// cut the fused-path footprint ~2.5× — a ~355 MB-raw rank now
/// peaks ~1.3 GB resident (3.7×: doc + rows/arena + minify + gzip-bound),
/// and riding the stale seed pinned width at 61 of 160 with PSI 0%
/// (199 GB ledger / ~3.3 GB uncorrected charge).
/// The EWMA footprint (`observe_live`) corrects residual drift per workload
/// in either direction.
const GZ_RANK_FACTOR: u64 = 5;
/// Gzipped ranks headed for the serde-DOM tree path (a reassigning pkl side
/// channel with a merged-trace artifact — see pipeline::merge_and_convert):
/// the whole document becomes a `serde_json::Value` on top of the raw bytes,
/// so the live set runs ~2-3× the fused-path factor. Under the 10× estimate
/// a 24576-rank pkl job admitted width 82 and OOM-killed its agent at
/// 224 GB RSS with PSI at 2% — the binding limit was an ancestor cgroup no
/// signal here can see, so the first admission wave must not overshoot
/// (~85% of such runs died).
const TREE_GZ_RANK_FACTOR: u64 = 25;
/// Raw (non-gzip) tree-path ranks: DEFAULT factor plus the DOM.
const TREE_RAW_RANK_FACTOR: u64 = 10;
/// Fraction (num/den = ½) of the ledger the one-shot seed may fill. The
/// estimate is trustworthy but whether a full-ledger burst trips PSI is a
/// property of the host's page-cache state at that moment (two runs on the
/// same host type with the same seed: one settled clean, one tripped and
/// ratcheted). Seed at half, and let the measured footprint
/// raise the cap to the real width within seconds.
const SEED_UTIL_NUM: u64 = 1;
const SEED_UTIL_DEN: u64 = 2;
/// Observations below this are a cold pool or a non-jemalloc build, not a
/// footprint — folding them in would inflate the model cap.
const FOOT_MIN_BYTES: u64 = 64_000_000;
/// Stall pressure (%) above which admissions pause and the cap halves.
const PSI_HIGH: f64 = 15.0;
/// ... and below which a backoff episode ends (hysteresis).
const PSI_LOW: f64 = 5.0;
/// Additive cap growth per quiet growth window.
const CAP_GROW: usize = 2;
/// Minimum spacing between cap growths. Growth per completed rank with no
/// time bound exploded the cap: PSI avg10 lags ~10 s, and one lag window
/// of completions at +4 each took the cap from 40 to 120+.
const CAP_GROW_EVERY: std::time::Duration = std::time::Duration::from_secs(1);
/// The cap never drops below this (progress guarantee works above it).
const CAP_MIN: usize = 2;
/// A worker parked at the gate longer than this sheds its retained pools
/// even when the block is the benign cap, not memory-tight. At width ~20 of
/// 160 (giant gz steps: the ledger admits few 5 GB estimates) a parked
/// thread will not run again for minutes, and 140 parked threads × ~1.3 GB
/// of retained tape/scratch/row pools pinned ~180 GB — the RSS that held
/// giant-step runs at 50 GB avail / PSI 60-100% for a whole hour.
/// Slot churn at healthy width parks
/// for well under a second and never reaches this.
pub(crate) const SHED_AFTER_PARKED: std::time::Duration = std::time::Duration::from_secs(5);
/// Bounds on the measured cost correction (foot_ewma / est_ewma) applied to
/// admission estimates: never trust the measurement to shrink a static
/// estimate more than 8× or grow it more than 4×.
const COST_SCALE_MIN_DIV: u64 = 8;
const COST_SCALE_MAX_MUL: u64 = 4;
/// Minimum spacing between ceiling re-probes (see MemPermit::drop). Slow on
/// purpose: each step risks a bounded pressure excursion on a genuinely
/// limited host, and the fast-attack + shed hook keep that excursion cheap.
const CEILING_GROW_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

struct GateState {
    /// Sum of admitted, unreleased cost estimates.
    est_sum: u64,
    /// Admitted, unreleased permits.
    count: usize,
    next_ticket: u64,
    served: u64,
    /// Slow-start/AIMD in-flight cap (≤ worker count).
    cap: usize,
    /// One-shot ledger seed applied at the first admission (see `admit`).
    seeded: bool,
    /// Concurrency ceiling: ¾ of the in-flight count that last triggered
    /// backoff. Slammed down multiplicatively on every backoff; under
    /// SUSTAINED quiet it re-probes upward slowly (AIMD congestion
    /// avoidance, `MemPermit::drop`). A pure never-raised ratchet pinned a
    /// run at 2-in-flight for its whole window after one transient
    /// startup spike (psi 0%, 130-240 GB free, 1019 s e2e) — the
    /// pressure point is a property of the moment, not only of the host.
    /// On a genuinely limited host pressure never goes quiet at the
    /// ceiling, so recovery never fires there.
    ceiling: Option<usize>,
    /// Last ceiling re-probe (rate limit).
    ceiling_grew_at: Option<std::time::Instant>,
    /// Inside a pressure-backoff episode (latched until PSI_LOW).
    backoff: bool,
    /// Cached pressure read — PSI files are re-read at most every 500 ms.
    psi_at: Option<std::time::Instant>,
    psi: f64,
    /// Previous (instant, max stall-total µs) for the fast-attack rate.
    psi_total_prev: Option<(std::time::Instant, u64)>,
    /// Last cap growth (rate limit).
    grew_at: Option<std::time::Instant>,
    /// Last allocator purge (rate-limited: a purge walks every arena).
    purged_at: Option<std::time::Instant>,
    /// EWMA of measured per-thread live bytes at rank completion (0 = no
    /// observation yet). The ledger models per-RANK transients, but what
    /// actually pins memory is each worker thread's retained live set —
    /// pools keep their peak across ranks — so sustainable width is
    /// budget / THIS, not budget / estimate.
    foot_ewma: u64,
    /// EWMA of the STATIC admission estimates of the same completed ranks
    /// `foot_ewma` sampled. The pair yields a measured correction for the
    /// per-rank cost model (foot/est) that the ledger applies to future
    /// admissions — in BOTH directions: giant fused-path ranks estimate ~3×
    /// their real footprint (ledger pinned width at ~20 of 160 for an hour),
    /// and tree-path ranks under-estimate (OOM at width 82).
    est_ewma: u64,
    /// budget / foot_ewma: the measured-width cap. Admission takes
    /// min(cap, model_cap) — AIMD still owns pressure events, this owns
    /// the steady state the estimates can't see.
    model_cap: Option<usize>,
    /// Consecutive ≥500 ms polls with a hot attack signal; backoff needs
    /// two in a row (or a hot avg10), so a one-poll blip never ratchets.
    hot_polls: u8,
}

pub struct MemGate {
    /// Ledger budget in bytes; u64::MAX = gate disabled.
    budget: u64,
    /// Live-floor reserve in bytes.
    reserve: u64,
    /// Absolute bound on this process's OWN resident set (u64::MAX = none):
    /// admissions require self_rss + cost to stay under it. See `new` — the
    /// managed-host ancestor limit kills below the advertised headroom
    /// with no readable or reactive signal.
    rss_cap: u64,
    /// Rank-loop thread count — the cap's ceiling.
    workers: usize,
    state: Mutex<GateState>,
    cv: Condvar,
    /// True while any admission is blocked — surfaced in the progress file
    /// so the supervising driver can label the run "memory-gated".
    /// True while an admission is blocked by MEMORY (ledger, pressure,
    /// backoff, or the live floor) rather than merely by the in-flight cap.
    /// Cap-blocking is the intended steady state whenever cap < workers;
    /// clearing thread pools on that benign signal re-created the very
    /// fault churn the pools exist to prevent (the clear→fault→PSI→backoff
    /// feedback loop measured on the first P1 candidate run).
    tight: AtomicBool,
    /// One-shot log latch: say once (to stderr → comms_viz.log) when the gate
    /// first engages.
    engaged_logged: AtomicBool,
    /// Bytes resident outside the permit ledger but inside this process —
    /// today the artifact writer queue (finished gz artifacts parked for
    /// the disk crew). Permits release at rank completion while those bytes
    /// can sit queued for seconds, so without this term the ledger admits
    /// as if they were gone. Charged against the budget at admission.
    external_bytes: std::sync::OnceLock<std::sync::Arc<std::sync::atomic::AtomicU64>>,
    /// Coalescing request line to the dedicated purger thread (lazily
    /// spawned on first use). Depth-1 channel: a purge already pending
    /// absorbs any number of new requests; call sites keep their own
    /// `purged_at` cadence, this only moves execution off the state lock
    /// and off the worker.
    purge_tx: std::sync::OnceLock<std::sync::mpsc::SyncSender<()>>,
}

/// RAII admission permit; releases its cost estimate on drop.
pub struct MemPermit<'a> {
    gate: &'a MemGate,
    /// What the ledger is currently charged for this rank: the
    /// measured-corrected estimate at admission, ratcheted UP by mid-rank
    /// footprint samples that exceed it (`observe_transient`). Without the
    /// ratchet, in-flight exposure was budget-blind: the correction ratio
    /// is learned mostly from ordinary ranks, so a burst of giants admitted
    /// on discounted charges could hold multiples of the ledger while the
    /// gate kept admitting — one run re-admitted to
    /// width 96 at RSS 227 GB / PSI 86 % right after recovering from the
    /// same cliff. Released at its final (possibly ratcheted) value.
    cost: std::cell::Cell<u64>,
    /// The raw static estimate — folded into `est_ewma` so the correction
    /// ratio is learned against the uncorrected model (folding the corrected
    /// value would compound the ratio toward 1 across ranks).
    raw_est: u64,
    /// Largest mid-rank live sample (`observe_transient`). The drop-time
    /// reading alone measures what a rank RETAINS, which for pool-recycling
    /// paths equals the pinned footprint — but the serde-DOM path frees its
    /// tree before completion, so its drop reading is a few hundred MB while
    /// 2-4 GB were alive mid-parse. Folding only the retained value scaled
    /// admissions on a 24k-rank run down 8× and drove width to 128 of 176 (WORSE than the
    /// static model's OOM at 82). The footprint that matters is
    /// max(transient peak, retained).
    transient: std::cell::Cell<u64>,
    /// `thread_live_bytes()` at admission. Footprint samples are deltas from
    /// here: the absolute reading is cumulative allocated − deallocated for
    /// THIS thread, so every block another thread frees on our behalf (the
    /// writer pool takes each rank's artifact bytes) inflates it forever.
    /// Rotating workers diluted the drift across all threads; a bounded
    /// worker subset concentrates it — foot_ewma crept 1.1 → 4 GB over one
    /// large run and the measured width collapsed 92 → 24.
    base_live: u64,
    /// This thread's retained pool bytes at admission (`rows::pool_bytes`).
    /// The delta above can't see capacity an earlier rank allocated, but the
    /// pool is real standing footprint per active worker, so it is added to
    /// every sample. Growth this rank lands in the delta; no double count.
    pool_at_admit: u64,
}

impl MemPermit<'_> {
    /// Record a mid-rank footprint sample for this permit's thread — call at
    /// the live-set peak (right after parse/merge, while the document
    /// structures are all still alive). Cheap: two jemalloc mallctl reads.
    pub fn observe_transient(&self) {
        let live = self.sample();
        if live > self.transient.get() {
            self.transient.set(live);
        }
        // Ledger ratchet: this rank is measurably bigger than what it was
        // admitted at — charge the difference so admissions see the real
        // in-flight exposure (see the `cost` field).
        let cost = self.cost.get();
        if live > cost {
            let mut st = self.gate.lock_state();
            st.est_sum = st.est_sum.saturating_add(live - cost);
            self.cost.set(live);
        }
    }

    /// One footprint sample: this rank's same-thread allocation delta plus
    /// the pool the thread already held — see `base_live` / `pool_at_admit`.
    /// A thread that mostly freed other threads' blocks reads a negative
    /// delta; saturation turns that into "pool only", and sub-FOOT_MIN
    /// samples are discarded by `observe_live` as no observation.
    fn sample(&self) -> u64 {
        thread_live_bytes()
            .saturating_sub(self.base_live)
            .saturating_add(self.pool_at_admit)
    }
}

impl Drop for MemPermit<'_> {
    fn drop(&mut self) {
        // Footprint sample: max of the mid-rank transient peak (see
        // `observe_transient`) and the completion reading — this rank's
        // same-thread delta plus the thread's standing pool (see `sample`;
        // the absolute jemalloc reading drifts up under cross-thread frees
        // and collapsed the measured width once workers stopped rotating).
        // Process RSS / in-flight width was tried instead and
        // death-spirals: after any backoff, RSS still holds every thread's
        // retained pools, so RSS/2 reads as a 75 GB rank, the EWMA
        // explodes, and the model cap pins at CAP_MIN for the rest of the
        // run.
        let transient = self.transient.get();
        let live = self.sample().max(transient);
        let mut st = self.gate.lock_state();
        st.est_sum = st.est_sum.saturating_sub(self.cost.get());
        st.count -= 1;
        self.gate
            .observe_live(&mut st, live, transient, self.raw_est);
        // Additive growth: only while pressure is QUIET (below the
        // recovery line, not merely below backoff), at most once per
        // growth window, and never past the ratcheted ceiling.
        let limit = st
            .ceiling
            .unwrap_or(self.gate.workers)
            .min(self.gate.workers);
        let due = st.grew_at.is_none_or(|t| t.elapsed() >= CAP_GROW_EVERY);
        if !st.backoff && st.psi < PSI_LOW && due && st.cap < limit {
            st.cap = (st.cap + CAP_GROW).min(limit);
            st.grew_at = Some(std::time::Instant::now());
        }
        // Ceiling re-probe (AIMD additive-increase leg): only under quiet
        // pressure, rate-limited, and only with generous visible headroom.
        if !st.backoff && st.psi < PSI_LOW {
            if let Some(c) = st.ceiling {
                let due = st
                    .ceiling_grew_at
                    .is_none_or(|t| t.elapsed() >= CEILING_GROW_EVERY);
                if due && c < self.gate.workers {
                    let floor = self.gate.reserve.saturating_mul(4);
                    let head = effective_headroom();
                    // Self-anchored allowance: visible headroom can sit
                    // pinned under the floor by bytes that are not OURS to
                    // wait on — the overlapped trace download's dirty pages
                    // and read-once file cache count against the cgroup
                    // while this process idles far below its own rss cap
                    // (width 3-15 of 160 for ~150 s
                    // at PSI ≤ 13%, self RSS 130-146 GB vs a 185 GB cap,
                    // avail oscillating just under the 51 GB floor). If two
                    // more measured ranks fit under the proactive rss cap,
                    // probe anyway — PSI fast-attack and the rss-overshoot
                    // trigger still bound the excursion.
                    let per_rank = st.foot_ewma.max(FOOT_MIN_BYTES);
                    let rss_room = self.gate.rss_cap != u64::MAX
                        && self_rss_bytes().saturating_add(per_rank.saturating_mul(2))
                            <= self.gate.rss_cap;
                    if head < floor {
                        // Allocator retention can pin visible headroom low
                        // on a quiet box: purge only ran while pressure was
                        // hot, so freed-but-retained pages held headroom
                        // under this guard and froze the ceiling for the
                        // rest of the run (62/176 wide at psi
                        // 0% for 90 s, 2026-08-20). Async request — growth
                        // waits for a later drop to observe the freed
                        // headroom instead of stalling this one behind the
                        // purge.
                        let purge_due = st
                            .purged_at
                            .is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(5));
                        if purge_due {
                            self.gate.purge_request();
                            st.purged_at = Some(std::time::Instant::now());
                        }
                    }
                    if head >= floor || rss_room {
                        // Recover geometrically toward the measured width:
                        // +25% steps lost the race against the shed→purge→
                        // refill re-trip cycle (pinned at 9-28 of
                        // a measured ~95 for a whole run). The doubling is
                        // still gated on quiet PSI + generous headroom, and
                        // an ancestor-limited host never goes quiet at its
                        // wall, so the slow-recovery case it was built for
                        // keeps its guard.
                        let bound = st
                            .model_cap
                            .unwrap_or(self.gate.workers)
                            .min(self.gate.workers);
                        st.ceiling = Some((c * 2).min(bound.max(c + 1)));
                        st.ceiling_grew_at = Some(std::time::Instant::now());
                    }
                }
            }
        }
        drop(st);
        self.gate.cv.notify_all();
    }
}

/// Hand jemalloc's retained-but-freed pages back to the kernel. Parked
/// worker threads run no allocator activity, so decay never ticks their
/// arenas and the cgroup stays pinned at its ceiling even once the gate
/// has drained (RSS flat at 195 GB, PSI ~69%, 1 rank in
/// flight). Purging drops memory.current, which is what ends an
/// ancestor-limit reclaim regime. Best-effort: a no-op when jemalloc is
/// not this binary's allocator.
///
/// NEVER call this holding the gate's state lock, and never from a worker:
/// the purge walks every arena issuing one madvise per extent — seconds to
/// MINUTES against hundreds of retained GB under reclaim. Run under the
/// lock (as it was until 2026-08-25), every drop/admit/snapshot convoys
/// behind it: 74 of 112 workers were observed blocked in MemPermit::drop
/// on the mutex and the progress thread froze mid-purge. Workers request;
/// the purger thread executes.
fn purge_retained() {
    // MALLCTL_ARENAS_ALL = 4096 (jemalloc ABI constant).
    unsafe {
        tikv_jemalloc_sys::mallctl(
            c"arena.4096.purge".as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
    }
}

impl MemGate {
    /// Gate for a process running `workers` rank-loop threads: the ledger
    /// budget is a fraction of the host headroom.
    pub fn new(workers: usize) -> Self {
        let (budget, reserve) = match env_gb("COMMS_VIZ_MEM_BUDGET_GB") {
            Some(0) => (u64::MAX, 0),
            Some(gb) => (gb * 1_000_000_000, 2_000_000_000),
            None => {
                let head = effective_headroom();
                if head == u64::MAX {
                    (u64::MAX, 0) // nothing measurable: gate inert
                } else {
                    let total = meminfo_bytes("MemTotal").unwrap_or(head);
                    (
                        head / 100 * BUDGET_FRACTION_PCT,
                        (total / 20).clamp(2_000_000_000, 16_000_000_000),
                    )
                }
            }
        };
        // Own-RSS ceiling: 80% of everything this process could believe it
        // may reach (current RSS + advertised headroom). The
        // ledger bounds ESTIMATES; nothing bounded the process's actual
        // resident set, and the binding limit on managed hosts is an
        // ancestor cgroup that under-runs the advertised headroom by
        // ~15-20%: every 24k-rank pkl run on such a host died at RSS
        // 220-225 GB with 45-115 GB "available" and PSI ≤ 37% — no reactive signal fires before the kill, so the
        // only defense is a proactive absolute bound. 80% clears the
        // observed kill line; on healthy boxes the ledger binds first and
        // this never engages.
        let rss_cap = if budget == u64::MAX {
            u64::MAX
        } else {
            self_rss_bytes().saturating_add(effective_headroom()) / 10 * 8
        };
        let workers = workers.max(1);
        // Slow start at a quarter of the pool (whole pool when the gate is
        // disabled): the ceiling is unknowable, so never bet the full
        // ledger on the first burst.
        let cap = if budget == u64::MAX {
            workers
        } else {
            (workers / 4).max(CAP_MIN).min(workers)
        };
        MemGate {
            budget,
            reserve,
            rss_cap,
            workers,
            state: Mutex::new(GateState {
                est_sum: 0,
                count: 0,
                next_ticket: 0,
                served: 0,
                cap,
                seeded: false,
                ceiling: None,
                ceiling_grew_at: None,
                backoff: false,
                psi_at: None,
                psi: 0.0,
                psi_total_prev: None,
                grew_at: None,
                purged_at: None,
                foot_ewma: 0,
                est_ewma: 0,
                model_cap: None,
                hot_polls: 0,
            }),
            cv: Condvar::new(),
            tight: AtomicBool::new(false),
            engaged_logged: AtomicBool::new(false),
            external_bytes: std::sync::OnceLock::new(),
            purge_tx: std::sync::OnceLock::new(),
        }
    }

    /// Register a gauge of bytes held outside the permit ledger (the
    /// artifact writer queue). First registration wins; the gauge is read
    /// per admission check.
    pub fn set_external_bytes(&self, gauge: std::sync::Arc<std::sync::atomic::AtomicU64>) {
        let _ = self.external_bytes.set(gauge);
    }

    fn external(&self) -> u64 {
        self.external_bytes
            .get()
            .map_or(0, |g| g.load(Ordering::Relaxed))
    }

    /// Ask the purger thread for a purge. Coalescing: a request already
    /// pending absorbs this one (depth-1 channel, `try_send`). Callers keep
    /// their own `purged_at` cadence; this only decides WHO runs the purge —
    /// never a worker, never under the state lock (see [`purge_retained`]).
    /// Lock the gate state, recovering a poisoned lock. Every mutation
    /// keeps the state internally valid at each step (saturating counters
    /// and EWMAs), so recovery is safe — and mandatory: a panicking worker
    /// must cost ONE rank (its own catch_unwind), never the run. With a
    /// plain unwrap, one mid-rank panic poisoned the lock, every other
    /// worker's PoisonError panic unwound into MemPermit::drop whose own
    /// unwrap panicked again — double panic, SIGABRT, whole shard gone
    /// (at rank 15,856 of 24,576).
    fn lock_state(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn purge_request(&self) {
        let tx = self.purge_tx.get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
            std::thread::Builder::new()
                .name("memgate-purge".into())
                .spawn(move || {
                    while rx.recv().is_ok() {
                        purge_retained();
                    }
                })
                .expect("spawn memgate-purge thread");
            tx
        });
        let _ = tx.try_send(());
    }

    /// Fold one measured per-rank live sample into the footprint EWMA and
    /// re-derive the measured-width cap. During a backoff episode the
    /// completion reading is shed noise (parked threads dropped their pools,
    /// so it reads low and would RAISE the cap exactly when the box says
    /// shrink) — but the mid-rank TRANSIENT peak was sampled while the rank
    /// ran and shedding cannot deflate it, so it stays a valid footprint.
    /// Folding it keeps the estimator learning through stormy runs, where
    /// freezing left the ledger riding the static seed for the whole window
    /// (width pinned at 61 of 160, PSI 0%).
    /// Cap ACTUATION still waits for the episode to end — during backoff
    /// only the EWMAs and the derived model cap move, never `cap` itself.
    /// `raw_est` is the same rank's static admission estimate, folded in
    /// lockstep so foot/est stays a paired ratio over the same population.
    fn observe_live(&self, st: &mut GateState, live: u64, transient: u64, raw_est: u64) {
        let sample = if st.backoff { transient } else { live };
        if sample < FOOT_MIN_BYTES || self.budget == u64::MAX {
            return;
        }
        st.foot_ewma = if st.foot_ewma == 0 {
            sample
        } else {
            st.foot_ewma - st.foot_ewma / 8 + sample / 8
        };
        if raw_est > 0 {
            st.est_ewma = if st.est_ewma == 0 {
                raw_est
            } else {
                st.est_ewma - st.est_ewma / 8 + raw_est / 8
            };
        }
        let cap = usize::try_from(self.budget / st.foot_ewma.max(1)).unwrap_or(usize::MAX);
        // A one-worker pool (one rank, or --threads 1) can't reach CAP_MIN;
        // clamp panics when min > max.
        let model = cap.clamp(CAP_MIN.min(self.workers), self.workers);
        st.model_cap = Some(model);
        if st.backoff {
            return;
        }
        // Measurements say `model` threads fit: lift the cap straight there
        // (the ½-ledger seed leaves room on purpose), under the ratcheted
        // ceiling — pressure events still win, and admissions re-check PSI
        // and the live floor per rank either way.
        let limit = st.ceiling.unwrap_or(self.workers).min(self.workers);
        st.cap = st.cap.max(model.min(limit));
    }

    /// Measured correction of one static estimate: cost × foot_ewma/est_ewma,
    /// bounded, and identity until both EWMAs are warm. This is what lets the
    /// ledger converge on the real per-rank footprint instead of riding the
    /// static factor for the whole run — the static factors are sized for
    /// safety at first admission, not for steady-state width.
    fn scaled_cost(&self, st: &GateState, cost: u64) -> u64 {
        if st.foot_ewma == 0 || st.est_ewma == 0 || cost == 0 {
            return cost;
        }
        let scaled = (cost as u128 * st.foot_ewma as u128 / st.est_ewma as u128) as u64;
        scaled.clamp(
            cost / COST_SCALE_MIN_DIV,
            cost.saturating_mul(COST_SCALE_MAX_MUL),
        )
    }

    /// Memory-tight: an admission is blocked by ledger/pressure/floor (not
    /// just the cap). This — not the cap — is the signal for shedding
    /// retained thread pools.
    pub fn is_tight(&self) -> bool {
        self.tight.load(Ordering::Relaxed)
    }

    /// Refresh the cached stall-pressure reading (≥500 ms apart) and apply
    /// backoff/recovery transitions. Caller holds the state lock.
    ///
    /// Attack uses max(avg10, instantaneous stall rate from the PSI total
    /// counters) — the rate reacts within one poll where avg10 lags ~10 s.
    /// Recovery uses avg10 alone, so an episode ends only once the smoothed
    /// signal agrees the regime is over.
    fn poll_pressure(&self, st: &mut GateState) -> f64 {
        let now = std::time::Instant::now();
        let stale = st
            .psi_at
            .is_none_or(|t| t.elapsed() >= std::time::Duration::from_millis(500));
        if stale {
            let (avg10, total_us) = stall_pressure();
            self.apply_pressure(st, avg10, total_us, now);
        }
        st.psi
    }

    /// Is stall pressure something THIS process can plausibly relieve by
    /// throttling ranks? Only when it holds a meaningful share of memory
    /// (RSS ≥ budget/4). A giant run streams ~2× its input through the page
    /// cache of a ~250 GB task limit, so 15-30% memory-stall pressure is
    /// ENDEMIC cache churn — treating it as a stop signal starved runs to
    /// width 1 with RSS 16-28 GB and 230+ GB available for 30+ minutes,
    /// until the scheduler killed the crawling task.
    /// Every genuine reclaim-livelock this gate exists for
    /// (the 220 GB kill lines) had RSS far above this bar.
    /// Budgets ≥ u64::MAX/2 mean "nothing measurable": stay conservative.
    fn psi_relevant(&self) -> bool {
        if self.budget > u64::MAX / 2 {
            return true;
        }
        self_rss_bytes().saturating_mul(4) >= self.budget
    }

    /// One pressure reading, applied to the state machine (split from
    /// [`Self::poll_pressure`] so tests can drive it without PSI files).
    ///
    /// Attack uses max(avg10, instantaneous stall rate from the PSI total
    /// counters) — the rate reacts within one poll where avg10 lags ~10 s
    /// — but backoff (and the ceiling ratchet) needs CONFIRMATION: a hot
    /// avg10, or two consecutive hot polls. A single hot poll is routinely
    /// a 1–2 s kernel-MM micro-burst (anon refaults with zero reclaim and
    /// >100 GB free), and ratcheting on those pinned a 10,240-rank run at
    /// > 33–54 of 160 workers with CPU ~60% idle (the
    /// > confirmed variant ran 41% faster with the ramp's real bursts
    /// > still backing off in ~1 s). Recovery uses avg10 alone, so an
    /// > episode ends only once the smoothed signal agrees it is over.
    fn apply_pressure(
        &self,
        st: &mut GateState,
        avg10: f64,
        total_us: u64,
        now: std::time::Instant,
    ) {
        let rate = match st.psi_total_prev {
            Some((t0, prev)) if now > t0 => {
                let wall_us = now.duration_since(t0).as_micros() as u64;
                ((total_us.saturating_sub(prev)) as f64 / wall_us.max(1) as f64 * 100.0).min(100.0)
            }
            _ => 0.0,
        };
        st.psi_total_prev = Some((now, total_us));
        st.psi = avg10.max(rate);
        st.psi_at = Some(now);
        st.hot_polls = if st.psi >= PSI_HIGH {
            st.hot_polls.saturating_add(1)
        } else {
            0
        };
        // Own-RSS overshoot is a backoff trigger in its own right: the cap
        // bounds ADMISSION (self_rss + cost at admit time), but admitted
        // ranks keep allocating past their estimates —
        // ran 17 GB OVER the cap (227 vs ~210) with PSI reading quiet
        // between purges, and the invisible ancestor limit killed the task.
        // Halving in-flight + the ceiling ratchet is the only lever that
        // actually sheds live memory here.
        let rss_hot = self.rss_cap != u64::MAX && self_rss_bytes() > self.rss_cap;
        if ((avg10 >= PSI_HIGH || st.hot_polls >= 2) && self.psi_relevant()) || rss_hot {
            if !st.backoff {
                // Multiplicative decrease, latched until pressure
                // subsides; the pressure point ratchets the ceiling.
                // Halve from the operating width, not the raw in-flight
                // count: a completion burst right before the blip can
                // drain count to near zero, and halving from that pinned
                // recovery at CAP_MIN + additive growth for ~30 s.
                let width = st.count.max(st.cap / 2);
                st.backoff = true;
                st.cap = (width / 2).max(CAP_MIN);
                let hit = (width * 3 / 4).max(CAP_MIN);
                st.ceiling = Some(st.ceiling.map_or(hit, |c| c.min(hit)));
                let cause = if rss_hot {
                    format!("self rss over the {} GB cap", self.rss_cap / 1_000_000_000)
                } else {
                    format!("stall pressure {:.0}% ≥ {PSI_HIGH}%", st.psi)
                };
                eprintln!(
                    "memory gate backoff: {cause} — in-flight cap halved to {} (ceiling {})",
                    st.cap,
                    st.ceiling.unwrap_or(0),
                );
            }
            // Return freed pages to the kernel while pressure holds —
            // backing off admissions alone cannot lower memory.current
            // when the allocator retains the pages. Rate-limited, and
            // handed to the purger thread: run here it holds the state
            // lock for the whole arena walk.
            let due = st
                .purged_at
                .is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(5));
            if due {
                self.purge_request();
                st.purged_at = Some(now);
            }
        } else if st.backoff && st.psi < PSI_HIGH && avg10 <= PSI_LOW {
            st.backoff = false;
        }
    }

    /// Block until `cost` estimated bytes may enter the rank loop. FIFO:
    /// admissions happen in call order. An empty gate always admits.
    pub fn admit(&self, cost: u64) -> MemPermit<'_> {
        self.admit_with(cost, || {})
    }

    /// `admit`, plus a shed hook: called (once) the first time this
    /// admission observes a MEMORY-tight block — the caller drops its
    /// retained thread pools before parking. Parked workers otherwise pin
    /// their pools for the whole episode: on an ancestor-limited agent,
    /// 160 parked workers × ~1.5 GB of retained tape/scratch/row buffers
    /// held RSS at the invisible allowance with 1 rank in flight and no
    /// way out.
    /// Cap-blocked parking does NOT shed: that is the benign steady state.
    pub fn admit_with(&self, cost: u64, mut shed: impl FnMut()) -> MemPermit<'_> {
        let mut st = self.lock_state();
        // Ledger seed, once, at the first admission: the workers/4 slow
        // start exists because the sustainable concurrency is unknowable at
        // construction — but the first rank's cost estimate makes the
        // ledger-implied concurrency computable, so start there instead of
        // ramping to it at +CAP_GROW/s (the ramp alone cost ~25% of e2e on
        // big-RAM hosts). Seeding at ¾ ledger utilization is intrinsically
        // host-proportional: a memory-tight host seeds LOW, and the PSI
        // fast-attack + ratcheting ceiling still own the invisible-limit
        // case (they are untouched). The remaining ¼ is burst margin until
        // the first footprint observations land (seconds).
        if !st.seeded {
            st.seeded = true;
            if self.budget != u64::MAX && cost > 0 {
                let ledger_cap =
                    usize::try_from(self.budget / SEED_UTIL_DEN * SEED_UTIL_NUM / cost.max(1))
                        .unwrap_or(usize::MAX);
                let seeded = ledger_cap.min(self.workers);
                if seeded > st.cap {
                    eprintln!(
                        "memory gate: ledger seed — in-flight cap {} → {} \
                         (budget {} GB, first rank estimate {} MB)",
                        st.cap,
                        seeded,
                        self.budget / 1_000_000_000,
                        cost / 1_000_000,
                    );
                    st.cap = seeded;
                }
            }
        }
        let ticket = st.next_ticket;
        st.next_ticket += 1;
        let mut blocked = false;
        let mut blocked_at: Option<std::time::Instant> = None;
        let mut shed_done = false;
        loop {
            // Re-derived each iteration: the correction ratio moves as
            // completions land while this thread is parked.
            let eff = self.scaled_cost(&st, cost);
            let ledger_ok = st
                .est_sum
                .saturating_add(self.external())
                .saturating_add(eff)
                <= self.budget;
            let cap_ok = st.count < st.cap.min(st.model_cap.unwrap_or(usize::MAX));
            // Pressure + live floor are only consulted once the cheap
            // checks pass — they cost a few /proc//sys reads.
            let mut mem_blocked = false;
            // A latched backoff episode blocks cap GROWTH (MemPermit::drop),
            // not admission: with the cap already halved, admitting under a
            // quiet fast signal is the TCP shape — halve the window, keep
            // sending. Pausing until avg10 decayed below PSI_LOW cost ~16 s
            // of a 97 s analysis window on one confirmed blip; a genuinely limited host keeps the fast
            // signal hot, which still parks admissions here.
            let admit = st.served == ticket
                && (st.count == 0 || {
                    let quiet = ledger_ok
                        && cap_ok
                        && (self.poll_pressure(&mut st) < PSI_HIGH || !self.psi_relevant())
                        && effective_headroom() >= self.reserve.saturating_add(eff)
                        && self_rss_bytes().saturating_add(eff) <= self.rss_cap;
                    // Anything but pure cap/FIFO blocking is memory-tight.
                    mem_blocked = !quiet && (!ledger_ok || cap_ok);
                    quiet
                });
            if admit {
                st.served += 1;
                st.est_sum = st.est_sum.saturating_add(eff);
                st.count += 1;
                self.tight.store(false, Ordering::Relaxed);
                drop(st);
                self.cv.notify_all();
                return MemPermit {
                    gate: self,
                    cost: std::cell::Cell::new(eff),
                    raw_est: cost,
                    transient: std::cell::Cell::new(0),
                    base_live: thread_live_bytes(),
                    pool_at_admit: crate::rows::pool_bytes(),
                };
            }
            // Long cap-parks shed too: at giant-step width the queue drains
            // permits over minutes, and a parked thread's retained pools are
            // pure dead weight for that whole span (see SHED_AFTER_PARKED).
            // The rate-limited purge hands the freed pages back to the
            // kernel — shedding alone leaves them jemalloc-retained, still
            // resident, still feeding reclaim pressure.
            let parked_long = blocked_at.is_some_and(|t| t.elapsed() >= SHED_AFTER_PARKED);
            if mem_blocked || st.backoff || parked_long {
                if mem_blocked || st.backoff {
                    self.tight.store(true, Ordering::Relaxed);
                }
                if !shed_done {
                    shed_done = true;
                    shed();
                    if parked_long {
                        let due = st
                            .purged_at
                            .is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(5));
                        if due {
                            self.purge_request();
                            st.purged_at = Some(std::time::Instant::now());
                        }
                    }
                }
            }
            // Keep backoff/recovery ticking even while parked, or a
            // latched episode would need a release to ever re-poll.
            self.poll_pressure(&mut st);
            if !blocked {
                blocked = true;
                blocked_at = Some(std::time::Instant::now());
                if !self.engaged_logged.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "memory gate engaged: {} GB in flight of a {} GB budget \
                         (cap {}, headroom {} GB, reserve {} GB)",
                        st.est_sum / 1_000_000_000,
                        self.budget / 1_000_000_000,
                        st.cap,
                        effective_headroom() / 1_000_000_000,
                        self.reserve / 1_000_000_000,
                    );
                }
            }
            // Timed wait: the live floor and PSI move without any permit
            // release (staging growth, co-tenants), so re-poll periodically.
            let (guard, _) = self
                .cv
                .wait_timeout(st, std::time::Duration::from_millis(100))
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            st = guard;
        }
    }

    /// Width of the worker subset that should actively claim ranks: the
    /// same cap bound admissions enforce (`cap_ok`). Workers with index ≥
    /// this park at the turnstile in the rank loop instead of rotating
    /// through the gate. Rotation kept every thread's retained pools hot —
    /// RSS converged to threads-EVER-used × pool regardless of how narrow
    /// the gate ran, because healthy churn parks each thread well under
    /// SHED_AFTER_PARKED.
    ///
    /// Additionally bounded by what the process's OWN residency affords:
    /// (rss_cap − rss) / foot_ewma more ranks beyond the ones in flight.
    /// The ledger and the admission checks bound new admissions, but
    /// nothing pulls standing memory back DOWN once RSS has climbed —
    /// two runs wedged in a reclaim livelock at RSS
    /// ~218 GB (avail pinned at the floor, PSI 76-97%, no forward progress
    /// for 20 min, agent killed) with admissions already blocked. Parking
    /// the unaffordable tail of workers sheds their pools (~1.3 GB each)
    /// and purges, so RSS falls back under the cap instead of livelocking.
    /// On healthy hosts rss stays far below rss_cap and this bound never
    /// binds.
    pub fn active_width(&self) -> usize {
        let st = self.lock_state();
        let cap = st.cap.min(st.model_cap.unwrap_or(usize::MAX)).max(CAP_MIN);
        if self.rss_cap == u64::MAX {
            return cap;
        }
        let foot = st.foot_ewma.max(FOOT_MIN_BYTES);
        let afford = usize::try_from(self.rss_cap.saturating_sub(self_rss_bytes()) / foot)
            .unwrap_or(usize::MAX)
            .saturating_add(st.count);
        cap.min(afford).max(CAP_MIN)
    }

    /// Rate-limited allocator purge for a worker that just shed its pools
    /// outside the gate (turnstile park). Shedding alone leaves the pages
    /// jemalloc-retained — still resident, still feeding reclaim pressure.
    pub fn shed_purge(&self) {
        let mut st = self.lock_state();
        let due = st
            .purged_at
            .is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(5));
        if due {
            self.purge_request();
            st.purged_at = Some(std::time::Instant::now());
        }
    }
}

fn env_gb(key: &str) -> Option<u64> {
    std::env::var(key).ok()?.trim().parse().ok()
}

/// Estimated peak rank-loop bytes for one trace file: decompressed size
/// (gzip ISIZE trailer, exact for our single-member <4 GiB traces; the
/// compressed size floors multi-member/wrapped oddities) times the
/// parse-structure factor. `dom_tree` = this rank will (likely) take the
/// serde-DOM tree path, whose live set runs ~2-3× the fused path's.
/// Unreadable/odd files estimate 0 — admission then rides the live floor
/// alone rather than inventing a number.
pub fn estimate_rank_cost(path: &Path, dom_tree: bool) -> u64 {
    let Ok(len) = std::fs::metadata(path).map(|m| m.len()) else {
        return 0;
    };
    let (raw, default_factor) = if is_gzip(path, len) {
        let f = if dom_tree {
            TREE_GZ_RANK_FACTOR
        } else {
            GZ_RANK_FACTOR
        };
        (gzip_isize(path, len).max(len), f)
    } else {
        let f = if dom_tree {
            TREE_RAW_RANK_FACTOR
        } else {
            DEFAULT_RANK_FACTOR
        };
        (len, f)
    };
    let factor = env_gb("COMMS_VIZ_MEM_RANK_FACTOR").unwrap_or(default_factor);
    raw.saturating_mul(factor).saturating_add(len)
}

/// This thread's live jemalloc bytes (allocated − deallocated since thread
/// start). ~0 when jemalloc is not the global allocator — callers treat
/// small values as "no observation".
pub fn thread_live_bytes() -> u64 {
    fn read(name: &std::ffi::CStr) -> u64 {
        let mut v: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let rc = unsafe {
            tikv_jemalloc_sys::mallctl(
                name.as_ptr(),
                &mut v as *mut u64 as *mut std::ffi::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 { v } else { 0 }
    }
    read(c"thread.allocated").saturating_sub(read(c"thread.deallocated"))
}

fn is_gzip(path: &Path, len: u64) -> bool {
    if len < 18 {
        return false;
    }
    let mut magic = [0u8; 2];
    read_at(path, 0, &mut magic).is_some() && magic == [0x1f, 0x8b]
}

fn gzip_isize(path: &Path, len: u64) -> u64 {
    let mut trailer = [0u8; 4];
    match read_at(path, len - 4, &mut trailer) {
        Some(()) => u32::from_le_bytes(trailer) as u64,
        None => 0,
    }
}

fn read_at(path: &Path, off: u64, buf: &mut [u8]) -> Option<()> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path).ok()?;
    f.read_exact_at(buf, off).ok()
}

/// Memory-stall pressure, preferring this cgroup subtree's view (`some`
/// line — a task stalled on memory). PSI is task-based: OUR stalled tasks
/// account to our own cgroup root, INCLUDING stalls caused by an ancestor
/// limit's reclaim, so the container view sees limits the namespace hides.
/// The host's global /proc/pressure/memory is only a fallback when the
/// cgroup file is unreadable: on multi-tenant hosts it folds in the
/// NEIGHBORS' reclaim — taking max(host, cgroup) starved a healthy run to
/// width 1 for 30 minutes (RSS 16 GB, 243 GB available, host avg10 pinned
/// 17-21% by a co-tenant) until the scheduler killed the crawling task.
/// (0.0, 0) when PSI is unavailable (non-Linux / CONFIG_PSI off) — the
/// gate then degrades to ledger + floor.
fn stall_pressure() -> (f64, u64) {
    let read = |p: &str| -> Option<(f64, u64)> { psi_some(&std::fs::read_to_string(p).ok()?) };
    read("/sys/fs/cgroup/memory.pressure")
        .or_else(|| read("/proc/pressure/memory"))
        .unwrap_or((0.0, 0))
}

fn psi_some(text: &str) -> Option<(f64, u64)> {
    let line = text.lines().find(|l| l.starts_with("some"))?;
    let field = |key: &str| line.split_whitespace().find_map(|f| f.strip_prefix(key));
    let avg10: f64 = field("avg10=")?.parse().ok()?;
    let total: u64 = field("total=").and_then(|v| v.parse().ok()).unwrap_or(0);
    Some((avg10, total))
}

/// Bytes this process may still allocate before something breaks: host
/// MemAvailable min every readable cgroup allowance (v2 then v1,
/// limit − current). u64::MAX when nothing is measurable (non-Linux).
/// Ancestor cgroups outside our namespace stay invisible — the reserve
/// plus the ledger budget absorb what we cannot see.
pub fn effective_headroom() -> u64 {
    let mut head = meminfo_bytes("MemAvailable").unwrap_or(u64::MAX);
    if let Some(h) = cgroup_headroom(Path::new("/sys/fs/cgroup")) {
        head = head.min(h);
    }
    head
}

fn cgroup_headroom(root: &Path) -> Option<u64> {
    // v2: memory.max is "max" (no limit) or bytes.
    if let Ok(max) = std::fs::read_to_string(root.join("memory.max")) {
        let limit: u64 = max.trim().parse().ok()?; // "max" → None
        let current: u64 = std::fs::read_to_string(root.join("memory.current"))
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let reclaimable = std::fs::read_to_string(root.join("memory.stat"))
            .ok()
            .map_or(0, |s| reclaimable_file_bytes(&s, false));
        return Some(limit.saturating_sub(current).saturating_add(reclaimable));
    }
    // v1: an unset limit reads back as a huge page-rounded sentinel.
    let limit: u64 = std::fs::read_to_string(root.join("memory/memory.limit_in_bytes"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    if limit >= 1 << 60 {
        return None;
    }
    let usage: u64 = std::fs::read_to_string(root.join("memory/memory.usage_in_bytes"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let reclaimable = std::fs::read_to_string(root.join("memory/memory.stat"))
        .ok()
        .map_or(0, |s| reclaimable_file_bytes(&s, true));
    Some(limit.saturating_sub(usage).saturating_add(reclaimable))
}

/// Clean, unmapped, non-tmpfs page cache charged to the cgroup — memory the
/// kernel reclaims before it OOMs anything, which `limit − current` counts
/// as gone. A run that streams a large input from disk (252 GB in one
/// measured case) pushes it all through this bucket: with the raw subtraction the gate's
/// headroom collapsed to 35–61 GB for the entire download window and
/// admissions parked on the reserve floor at width 1–34 while the CPUs sat
/// idle. Host MemAvailable already treats clean cache as
/// available; this makes the cgroup term agree.
///
/// Deliberately conservative: tmpfs/shmem pages are unreclaimable-in-place
/// (swap only), dirty/writeback pages are pinned until flushed, and mapped
/// file pages fault straight back — all subtracted. Missing file or fields
/// read as 0, i.e. the old behavior.
fn reclaimable_file_bytes(stat: &str, v1: bool) -> u64 {
    let field = |key: &str| -> u64 {
        stat.lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    };
    let (file, shmem, mapped, dirty, writeback) = if v1 {
        // v1 buckets tmpfs under "cache" too; total_* covers the subtree.
        (
            field("total_cache "),
            field("total_shmem "),
            field("total_mapped_file "),
            field("total_dirty "),
            field("total_writeback "),
        )
    } else {
        (
            field("file "),
            field("shmem "),
            field("file_mapped "),
            field("file_dirty "),
            field("file_writeback "),
        )
    };
    file.saturating_sub(shmem)
        .saturating_sub(mapped)
        .saturating_sub(dirty)
        .saturating_sub(writeback)
}

pub fn meminfo_bytes(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with(key))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// This process's resident set in bytes (0 when unreadable) — progress-file
/// telemetry.
pub fn self_rss_bytes() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    text.split_whitespace()
        .nth(1)
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(0, |pages| pages * 4096)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use super::*;

    fn test_gate(budget: u64, cap: usize) -> MemGate {
        MemGate {
            budget,
            reserve: 0, // live floor inert: headroom >= 0 always
            rss_cap: u64::MAX,
            workers: 64,
            state: Mutex::new(GateState {
                est_sum: 0,
                count: 0,
                next_ticket: 0,
                served: 0,
                cap,
                // Tests exercise post-seed behavior; the seed has its own test.
                seeded: true,
                ceiling: None,
                ceiling_grew_at: None,
                backoff: false,
                // A fresh reading pinned far in the future: tests must not
                // depend on this host's live PSI.
                psi_at: Some(std::time::Instant::now() + std::time::Duration::from_secs(3600)),
                psi: 0.0,
                psi_total_prev: None,
                grew_at: None,
                purged_at: None,
                foot_ewma: 0,
                est_ewma: 0,
                model_cap: None,
                hot_polls: 0,
            }),
            cv: Condvar::new(),
            tight: AtomicBool::new(false),
            engaged_logged: AtomicBool::new(true),
            external_bytes: std::sync::OnceLock::new(),
            purge_tx: std::sync::OnceLock::new(),
        }
    }

    #[test]
    fn empty_gate_admits_oversized_rank() {
        let g = test_gate(100, 64);
        let p = g.admit(10_000); // way over budget, but nothing in flight
        drop(p);
        let p2 = g.admit(50);
        drop(p2);
    }

    #[test]
    fn external_bytes_charge_ledger_and_drain_unblocks() {
        let g = std::sync::Arc::new(test_gate(100, 64));
        let gauge = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(90));
        g.set_external_bytes(gauge.clone());
        // Empty-gate progress guarantee is untouched by the external charge.
        let first = g.admit(50);
        let admitted = std::sync::Arc::new(AtomicUsize::new(0));
        let j = {
            let (g, admitted) = (g.clone(), admitted.clone());
            std::thread::spawn(move || {
                let p = g.admit(20); // 50 + 90 + 20 > 100: blocked on external
                admitted.fetch_add(1, Ordering::SeqCst);
                drop(p);
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(120));
        assert_eq!(
            admitted.load(Ordering::SeqCst),
            0,
            "external bytes must charge"
        );
        gauge.store(0, Ordering::Relaxed); // queue drained
        j.join().unwrap();
        drop(first);
        assert_eq!(g.state.lock().unwrap().est_sum, 0);
    }

    #[test]
    fn release_unblocks_and_fifo_holds() {
        let g = std::sync::Arc::new(test_gate(100, 64));
        let first = g.admit(80);
        let order = std::sync::Arc::new(AtomicUsize::new(0));
        let mut joins = Vec::new();
        // Two waiters that both fit only after `first` releases; tickets
        // force big-then-small admission order.
        for (i, cost) in [(1usize, 90u64), (2usize, 5u64)] {
            let (g, order) = (g.clone(), order.clone());
            joins.push(std::thread::spawn(move || {
                // Stagger ticket acquisition deterministically.
                std::thread::sleep(std::time::Duration::from_millis(20 * i as u64));
                let p = g.admit(cost);
                let seen = order.fetch_add(1, Ordering::SeqCst);
                if i == 1 {
                    assert_eq!(seen, 0, "big rank admitted first (FIFO)");
                }
                drop(p);
            }));
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        drop(first);
        for j in joins {
            j.join().unwrap();
        }
        assert_eq!(g.state.lock().unwrap().est_sum, 0);
    }

    #[test]
    fn cap_bounds_inflight_and_grows_on_release() {
        let g = test_gate(u64::MAX, 1); // gate "disabled" budget: no seed, no
        // footprint fold — cap stays binding
        let p1 = g.admit(1);
        let waited = std::sync::Arc::new(AtomicUsize::new(0));
        let j = {
            let waited = waited.clone();
            let g: &'static MemGate = unsafe { std::mem::transmute(&g) };
            std::thread::spawn(move || {
                let p = g.admit(1); // blocked: count 1 == cap 1
                waited.fetch_add(1, Ordering::SeqCst);
                drop(p);
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(80));
        assert_eq!(waited.load(Ordering::SeqCst), 0, "cap must hold");
        drop(p1); // release grows cap by CAP_GROW and wakes the waiter
        j.join().unwrap();
        // Only the first drop grows the cap — the second falls inside the
        // growth-rate window.
        assert_eq!(g.state.lock().unwrap().cap, 1 + CAP_GROW);
    }

    #[test]
    fn backoff_halves_cap_and_recovers() {
        let g = test_gate(u64::MAX - 1, 64);
        {
            let mut st = g.state.lock().unwrap();
            st.count = 40;
            st.psi = PSI_HIGH + 1.0;
            st.psi_at = None; // force a poll transition using the fake read
            // poll_pressure re-reads the real PSI files; emulate the
            // transition directly instead (the parser has its own test).
            st.backoff = true;
            st.cap = (st.count / 2).max(CAP_MIN);
            assert_eq!(st.cap, 20);
            // Recovery path: pressure subsides, growth resumes on release
            // but never past the ratcheted ceiling.
            st.backoff = false;
            st.psi = 0.0;
            st.count = 1;
            st.ceiling = Some(21);
        }
        g.state.lock().unwrap().est_sum = 1;
        drop(MemPermit {
            gate: &g,
            cost: std::cell::Cell::new(1),
            raw_est: 1,
            transient: std::cell::Cell::new(0),
            base_live: thread_live_bytes(),
            pool_at_admit: 0,
        });
        assert_eq!(g.state.lock().unwrap().cap, 21, "growth clamps at ceiling");
    }

    #[test]
    fn ceiling_reprobes_under_quiet_but_not_in_backoff() {
        let g = test_gate(u64::MAX - 1, 8);
        {
            let mut st = g.state.lock().unwrap();
            st.ceiling = Some(4);
            st.count = 1;
            st.est_sum = 1;
        }
        drop(MemPermit {
            gate: &g,
            cost: std::cell::Cell::new(1),
            raw_est: 1,
            transient: std::cell::Cell::new(0),
            base_live: thread_live_bytes(),
            pool_at_admit: 0,
        }); // quiet: ceiling doubles, 4 -> 8
        assert_eq!(g.state.lock().unwrap().ceiling, Some(8));
        {
            let mut st = g.state.lock().unwrap();
            st.backoff = true;
            st.count = 1;
            st.est_sum = 1;
            st.ceiling_grew_at = None;
        }
        drop(MemPermit {
            gate: &g,
            cost: std::cell::Cell::new(1),
            raw_est: 1,
            transient: std::cell::Cell::new(0),
            base_live: thread_live_bytes(),
            pool_at_admit: 0,
        }); // backoff: no re-probe
        assert_eq!(g.state.lock().unwrap().ceiling, Some(8));
    }

    #[test]
    fn transient_ratchets_ledger_charge() {
        let g = test_gate(64_000_000_000, 8);
        {
            let mut st = g.state.lock().unwrap();
            st.est_sum = 10;
            st.count = 1;
        }
        let p = MemPermit {
            gate: &g,
            cost: std::cell::Cell::new(10),
            raw_est: 10,
            transient: std::cell::Cell::new(0),
            // The std-allocator test binary reads thread_live_bytes() as 0,
            // so sample() == pool_at_admit exactly — a deterministic probe.
            base_live: 0,
            pool_at_admit: FOOT_MIN_BYTES * 2,
        };
        p.observe_transient();
        assert_eq!(
            p.cost.get(),
            FOOT_MIN_BYTES * 2,
            "charge ratchets to sample"
        );
        assert_eq!(
            g.state.lock().unwrap().est_sum,
            FOOT_MIN_BYTES * 2,
            "ledger charged the growth"
        );
        drop(p);
        assert_eq!(
            g.state.lock().unwrap().est_sum,
            0,
            "release covers the ratcheted charge"
        );
    }

    #[test]
    fn turnstile_width_rss_bound() {
        let mut g = test_gate(64_000_000_000, 8);
        g.state.lock().unwrap().foot_ewma = FOOT_MIN_BYTES;
        // rss_cap unset: cap alone rules.
        assert_eq!(g.active_width(), 8);
        // rss_cap below any real process rss: only in-flight ranks afford,
        // floored at CAP_MIN.
        g.rss_cap = 1;
        assert_eq!(g.active_width(), CAP_MIN);
        g.state.lock().unwrap().count = 5;
        assert_eq!(g.active_width(), 5);
    }

    #[test]
    fn ledger_seed_raises_initial_cap_once() {
        let g = test_gate(64_000, 4);
        g.state.lock().unwrap().seeded = false;
        let p = g.admit(1_000); // ledger seed: ½ × 64000 / 1000 = 32
        assert_eq!(g.state.lock().unwrap().cap, 32);
        drop(p);
        // A later, smaller estimate must not re-seed (growth from here is
        // the normal additive path).
        let p2 = g.admit(10);
        assert!(g.state.lock().unwrap().cap <= 32 + CAP_GROW);
        drop(p2);
        // A seed below the current cap never lowers it.
        let g = test_gate(64_000, 32);
        g.state.lock().unwrap().seeded = false;
        let p = g.admit(16_000); // ledger cap = 1 < 32
        assert_eq!(g.state.lock().unwrap().cap, 32);
        drop(p);
    }

    /// Drive one apply_pressure step with a synthetic reading. The stall
    /// rate is fed via total-µs deltas: `pct` % of the wall since the
    /// previous step.
    fn feed(g: &MemGate, avg10: f64, rate_pct: f64, at: std::time::Instant) {
        let mut st = g.state.lock().unwrap();
        let total = match st.psi_total_prev {
            Some((t0, tot)) => {
                tot + (at.duration_since(t0).as_micros() as f64 * rate_pct / 100.0) as u64
            }
            None => 0,
        };
        g.apply_pressure(&mut st, avg10, total, at);
    }

    #[test]
    fn single_hot_poll_does_not_ratchet_two_do() {
        let g = test_gate(u64::MAX - 1, 64);
        let t0 = std::time::Instant::now();
        g.state.lock().unwrap().count = 40;
        feed(&g, 0.0, 0.0, t0); // establish psi_total_prev
        let s500 = std::time::Duration::from_millis(500);
        // One hot fast-attack poll: a micro-burst. No backoff, no ceiling.
        feed(&g, 0.0, 40.0, t0 + s500);
        {
            let st = g.state.lock().unwrap();
            assert!(!st.backoff, "single hot poll must not back off");
            assert_eq!(st.ceiling, None);
            assert_eq!(st.hot_polls, 1);
        }
        // Quiet again: the streak resets.
        feed(&g, 0.0, 0.0, t0 + 2 * s500);
        assert_eq!(g.state.lock().unwrap().hot_polls, 0);
        // Two consecutive hot polls: a confirmed regime — backoff + ratchet.
        feed(&g, 0.0, 40.0, t0 + 3 * s500);
        feed(&g, 0.0, 40.0, t0 + 4 * s500);
        let st = g.state.lock().unwrap();
        assert!(st.backoff);
        assert_eq!(st.cap, 20, "cap halves at count 40");
        assert_eq!(st.ceiling, Some(30), "ceiling ratchets to 3/4 of count");
    }

    #[test]
    fn backoff_admits_under_halved_cap_when_fast_signal_quiet() {
        let g = test_gate(u64::MAX - 1, 8);
        let p1 = g.admit(1);
        {
            // Latched episode, but the fast signal has gone quiet: the
            // halved cap governs, admission must not wait for avg10 decay.
            let mut st = g.state.lock().unwrap();
            st.backoff = true;
            st.psi = PSI_HIGH - 1.0;
        }
        let p2 = g.admit(1); // returns immediately: count 1 < cap 8
        drop(p2);
        drop(p1);
    }

    #[test]
    fn halving_uses_operating_width_not_drained_count() {
        let g = test_gate(u64::MAX - 1, 64);
        let t0 = std::time::Instant::now();
        // A completion burst just drained in-flight to 3 of a 64-wide cap.
        g.state.lock().unwrap().count = 3;
        feed(&g, 0.0, 0.0, t0);
        let s500 = std::time::Duration::from_millis(500);
        feed(&g, 0.0, 40.0, t0 + s500);
        feed(&g, 0.0, 40.0, t0 + 2 * s500);
        let st = g.state.lock().unwrap();
        assert!(st.backoff);
        assert_eq!(st.cap, 16, "halves from width max(count 3, cap/2 32)");
        assert_eq!(st.ceiling, Some(24), "ceiling from width, not count");
    }

    #[test]
    fn hot_avg10_backs_off_on_first_poll() {
        let g = test_gate(u64::MAX - 1, 64);
        let t0 = std::time::Instant::now();
        g.state.lock().unwrap().count = 40;
        feed(&g, PSI_HIGH + 1.0, 0.0, t0);
        assert!(
            g.state.lock().unwrap().backoff,
            "hot avg10 needs no second poll"
        );
    }

    #[test]
    fn rss_over_cap_forces_backoff_and_blocks_recovery() {
        let mut g = test_gate(u64::MAX - 1, 64);
        g.rss_cap = 1; // any real process rss exceeds this
        let t0 = std::time::Instant::now();
        g.state.lock().unwrap().count = 40;
        feed(&g, 0.0, 0.0, t0);
        {
            let st = g.state.lock().unwrap();
            assert!(st.backoff, "rss overshoot must back off without hot PSI");
            assert_eq!(st.cap, 20);
        }
        // Quiet PSI does not clear the episode while rss stays hot.
        feed(&g, 0.0, 0.0, t0 + std::time::Duration::from_millis(500));
        assert!(g.state.lock().unwrap().backoff);
    }

    #[test]
    fn psi_parser_reads_some_avg10_and_total() {
        let text = "some avg10=12.34 avg60=3.21 avg300=0.50 total=123456\n\
                    full avg10=8.00 avg60=1.00 avg300=0.10 total=6543\n";
        assert_eq!(psi_some(text), Some((12.34, 123456)));
        assert_eq!(psi_some(""), None);
    }

    #[test]
    fn estimate_reads_isize_trailer() {
        use std::io::Write;

        use flate2::Compression;
        use flate2::write::GzEncoder;
        let dir = tempfile::tempdir().unwrap();
        let gz = dir.path().join("r.json.gz");
        let payload = vec![b'x'; 100_000];
        let mut enc = GzEncoder::new(std::fs::File::create(&gz).unwrap(), Compression::fast());
        enc.write_all(&payload).unwrap();
        enc.finish().unwrap();
        let clen = std::fs::metadata(&gz).unwrap().len();
        assert_eq!(
            estimate_rank_cost(&gz, false),
            100_000 * GZ_RANK_FACTOR + clen
        );
        assert_eq!(
            estimate_rank_cost(&gz, true),
            100_000 * TREE_GZ_RANK_FACTOR + clen
        );

        let raw = dir.path().join("r.json");
        std::fs::write(&raw, &payload).unwrap();
        assert_eq!(
            estimate_rank_cost(&raw, false),
            100_000 * DEFAULT_RANK_FACTOR + 100_000
        );
        assert_eq!(
            estimate_rank_cost(&raw, true),
            100_000 * TREE_RAW_RANK_FACTOR + 100_000
        );

        assert_eq!(estimate_rank_cost(&dir.path().join("absent"), false), 0);
    }

    #[test]
    fn footprint_observation_caps_width_below_estimates() {
        // Budget fits 5 measured threads; estimates would admit 64.
        let g = test_gate(FOOT_MIN_BYTES * 10, 64);
        {
            let mut st = g.state.lock().unwrap();
            g.observe_live(&mut st, FOOT_MIN_BYTES * 2, 0, 0);
            assert_eq!(st.model_cap, Some(5));
        }
        // Admission honors min(cap, model_cap): the 6th admit blocks.
        let permits: Vec<_> = (0..5).map(|_| g.admit(1)).collect();
        assert_eq!(g.state.lock().unwrap().count, 5);
        let g2: &'static MemGate = unsafe { std::mem::transmute(&g) };
        let unblocked = std::sync::Arc::new(AtomicUsize::new(0));
        let u2 = unblocked.clone();
        let j = std::thread::spawn(move || {
            let p = g2.admit(1);
            u2.fetch_add(1, Ordering::SeqCst);
            drop(p);
        });
        std::thread::sleep(std::time::Duration::from_millis(80));
        assert_eq!(unblocked.load(Ordering::SeqCst), 0, "model cap must hold");
        drop(permits);
        j.join().unwrap();
    }

    #[test]
    fn observe_live_with_one_worker() {
        let mut g = test_gate(1 << 40, 64);
        g.workers = 1;
        let mut st = g.state.lock().unwrap();
        g.observe_live(&mut st, FOOT_MIN_BYTES * 2, 0, 0);
        assert_eq!(st.model_cap, Some(1));
    }

    #[test]
    fn observe_live_skips_backoff_and_small_readings() {
        let g = test_gate(u64::MAX, 64);
        let mut st = g.state.lock().unwrap();
        g.observe_live(&mut st, FOOT_MIN_BYTES * 2, 0, 0);
        assert_eq!(st.foot_ewma, 0, "disabled ledger folds nothing");
        drop(st);
        let g = test_gate(1 << 40, 64);
        let mut st = g.state.lock().unwrap();
        g.observe_live(&mut st, FOOT_MIN_BYTES - 1, 0, 0);
        assert_eq!(st.foot_ewma, 0, "sub-floor reading is not a footprint");
        st.backoff = true;
        g.observe_live(&mut st, FOOT_MIN_BYTES * 2, 0, 0);
        assert_eq!(
            st.foot_ewma, 0,
            "backoff completion readings are shed noise"
        );
        st.backoff = false;
        g.observe_live(&mut st, FOOT_MIN_BYTES * 2, 0, 0);
        assert_eq!(st.foot_ewma, FOOT_MIN_BYTES * 2);
        // Budget/footprint far exceeds the pool: clamps at workers.
        assert_eq!(st.model_cap, Some(64));
        // EWMA folds at 1/8 weight.
        g.observe_live(&mut st, FOOT_MIN_BYTES * 10, 0, 0);
        assert_eq!(
            st.foot_ewma,
            FOOT_MIN_BYTES * 2 - FOOT_MIN_BYTES * 2 / 8 + FOOT_MIN_BYTES * 10 / 8
        );
    }

    #[test]
    fn observe_live_folds_transient_during_backoff_without_raising_cap() {
        let g = test_gate(FOOT_MIN_BYTES * 10, 64);
        let mut st = g.state.lock().unwrap();
        st.backoff = true;
        st.cap = 4;
        // Transient peak folds even mid-episode; the completion reading is
        // ignored, and the cap must not move.
        g.observe_live(
            &mut st,
            FOOT_MIN_BYTES * 9,
            FOOT_MIN_BYTES * 2,
            FOOT_MIN_BYTES * 5,
        );
        assert_eq!(st.foot_ewma, FOOT_MIN_BYTES * 2, "transient sample folds");
        assert_eq!(st.est_ewma, FOOT_MIN_BYTES * 5);
        assert_eq!(st.model_cap, Some(5));
        assert_eq!(st.cap, 4, "no cap actuation during backoff");
        // Once the episode ends, the max(completion, transient) reading
        // folds again and the cap may lift to the model.
        st.backoff = false;
        g.observe_live(&mut st, FOOT_MIN_BYTES * 2, FOOT_MIN_BYTES * 2, 0);
        assert_eq!(st.cap, 5, "cap lifts to the measured width after recovery");
    }

    #[test]
    fn measured_cost_correction_scales_both_directions_and_clamps() {
        let g = test_gate(1 << 40, 64);
        {
            // No observations: identity.
            let st = g.state.lock().unwrap();
            assert_eq!(g.scaled_cost(&st, 1_000), 1_000);
        }
        {
            // Over-estimating model (giant fused ranks): est 5 GB vs
            // measured ~1.25 GB → future admissions charged a quarter.
            let mut st = g.state.lock().unwrap();
            st.foot_ewma = FOOT_MIN_BYTES * 2;
            st.est_ewma = FOOT_MIN_BYTES * 8;
            assert_eq!(g.scaled_cost(&st, 1_000), 250);
            // Under-estimating model (tree-path ranks): measured 3× the
            // estimate → admissions charged 3×.
            st.foot_ewma = FOOT_MIN_BYTES * 24;
            assert_eq!(g.scaled_cost(&st, 1_000), 3_000);
            // Clamps: never below cost/8 nor above cost×4.
            st.foot_ewma = FOOT_MIN_BYTES;
            st.est_ewma = FOOT_MIN_BYTES * 100;
            assert_eq!(g.scaled_cost(&st, 1_000), 125);
            st.foot_ewma = FOOT_MIN_BYTES * 100;
            st.est_ewma = FOOT_MIN_BYTES;
            assert_eq!(g.scaled_cost(&st, 1_000), 4_000);
        }
    }

    #[test]
    fn permit_drop_folds_raw_estimate_not_ledger_cost() {
        let g = test_gate(1 << 40, 64);
        {
            let mut st = g.state.lock().unwrap();
            st.count = 1;
            st.est_sum = 500;
        }
        // A permit whose ledger cost was corrected (500) still folds the
        // RAW estimate (2 × FOOT_MIN) into est_ewma at first observation.
        // (observe_live needs a valid live sample to fold anything.)
        {
            let mut st = g.state.lock().unwrap();
            g.observe_live(&mut st, FOOT_MIN_BYTES * 2, 0, FOOT_MIN_BYTES * 4);
            assert_eq!(st.est_ewma, FOOT_MIN_BYTES * 4);
            assert_eq!(st.foot_ewma, FOOT_MIN_BYTES * 2);
        }
    }

    #[test]
    fn cgroup_headroom_parses_v2_and_v1() {
        let dir = tempfile::tempdir().unwrap();
        // v2, limited
        std::fs::write(dir.path().join("memory.max"), "1000000\n").unwrap();
        std::fs::write(dir.path().join("memory.current"), "400000\n").unwrap();
        assert_eq!(cgroup_headroom(dir.path()), Some(600_000));
        // v2, unlimited
        std::fs::write(dir.path().join("memory.max"), "max\n").unwrap();
        assert_eq!(cgroup_headroom(dir.path()), None);

        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("memory")).unwrap();
        std::fs::write(dir.path().join("memory/memory.limit_in_bytes"), "2000000\n").unwrap();
        std::fs::write(dir.path().join("memory/memory.usage_in_bytes"), "500000\n").unwrap();
        assert_eq!(cgroup_headroom(dir.path()), Some(1_500_000));
        // v1 no-limit sentinel
        std::fs::write(
            dir.path().join("memory/memory.limit_in_bytes"),
            format!("{}\n", u64::MAX / 4096 * 4096),
        )
        .unwrap();
        assert_eq!(cgroup_headroom(dir.path()), None);
    }

    #[test]
    fn cgroup_headroom_counts_reclaimable_file_cache() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("memory.max"), "1000000\n").unwrap();
        std::fs::write(dir.path().join("memory.current"), "900000\n").unwrap();
        // 500k file cache, of which 100k tmpfs + 50k mapped + 40k dirty +
        // 10k writeback are pinned → 300k reclaimable on top of the 100k raw.
        std::fs::write(
            dir.path().join("memory.stat"),
            "anon 400000\nfile 500000\nfile_mapped 50000\nfile_dirty 40000\n\
             file_writeback 10000\nshmem 100000\n",
        )
        .unwrap();
        assert_eq!(cgroup_headroom(dir.path()), Some(400_000));

        // v1 spelling.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("memory")).unwrap();
        std::fs::write(dir.path().join("memory/memory.limit_in_bytes"), "1000000\n").unwrap();
        std::fs::write(dir.path().join("memory/memory.usage_in_bytes"), "900000\n").unwrap();
        std::fs::write(
            dir.path().join("memory/memory.stat"),
            "total_cache 500000\ntotal_shmem 100000\ntotal_mapped_file 50000\n\
             total_dirty 40000\ntotal_writeback 10000\n",
        )
        .unwrap();
        assert_eq!(cgroup_headroom(dir.path()), Some(400_000));
    }

    #[test]
    fn reclaimable_file_bytes_missing_fields_read_zero() {
        // No memory.stat fields at all → 0 added, the pre-change behavior.
        assert_eq!(reclaimable_file_bytes("", false), 0);
        // Pinned buckets exceeding file must not underflow.
        assert_eq!(reclaimable_file_bytes("file 100\nshmem 300\n", false), 0);
    }
}

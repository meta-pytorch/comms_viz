// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Byte-bounded off-thread file writer.
//!
//! btrfs serializes concurrent buffered writers on the per-filesystem
//! delalloc extent-reservation spin lock: profiling measured the
//! aggregator spending 51.4% of its on-CPU time in
//! `native_queued_spin_lock_slowpath` via `btrfs_buffered_write` while the
//! rayon pool fanned 11,668 small group-CSV writes, and the analysis
//! workers' inline artifact writes surfaced as up to 11.2k thread-seconds
//! of unattributed stall. Sixty-four threads queueing on one spin lock
//! write no faster than four — they only burn CPU spinning.
//!
//! This pool inverts the shape: producers serialize bytes in parallel and
//! hand `(path, bytes)` over; a small fixed crew does the `write()`s. The
//! queue is byte-bounded so a stalled disk backpressures producers instead
//! of ballooning RSS, and `flush()`/`finish()` give the completeness
//! points the callers' ordering contracts need (worker `.done` markers,
//! staging promotion, MANIFEST-last).

use std::collections::HashSet;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;

struct Job {
    path: PathBuf,
    bytes: Vec<u8>,
    /// Write to `<name>.partial` then rename: for dirs a concurrent
    /// scanner walks; plain write otherwise.
    via_tmp: bool,
}

#[derive(Default)]
struct Queue {
    q: VecDeque<Job>,
    bytes: usize,
    closed: bool,
}

pub struct WriterPool {
    state: Arc<(Mutex<Queue>, Condvar)>,
    cap_bytes: usize,
    submitted: Arc<AtomicUsize>,
    written: Arc<AtomicUsize>,
    /// Bytes alive in this pool: queued plus in-write (a popped job's buffer
    /// stays resident until its write returns). Shared with the memory gate,
    /// which charges them against its ledger — a permit releases at rank
    /// completion while the artifact can sit here for seconds.
    live_bytes: Arc<AtomicU64>,
    errors: Arc<Mutex<Vec<String>>>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl WriterPool {
    pub fn new(writers: usize, cap_bytes: usize) -> Self {
        let state = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let submitted = Arc::new(AtomicUsize::new(0));
        let written = Arc::new(AtomicUsize::new(0));
        let live_bytes = Arc::new(AtomicU64::new(0));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let handles = (0..writers.max(1))
            .map(|_| {
                let state = state.clone();
                let written = written.clone();
                let live_bytes = live_bytes.clone();
                let errors = errors.clone();
                std::thread::spawn(move || {
                    // Parent dirs already ensured by THIS thread; the
                    // create_dir_all per file was its own metadata storm.
                    let mut made_dirs: HashSet<PathBuf> = HashSet::new();
                    loop {
                        let job = {
                            let (m, cv) = &*state;
                            let mut st = m.lock().unwrap();
                            loop {
                                if let Some(j) = st.q.pop_front() {
                                    st.bytes -= j.bytes.len();
                                    cv.notify_all();
                                    break Some(j);
                                }
                                if st.closed {
                                    break None;
                                }
                                st = cv.wait(st).unwrap();
                            }
                        };
                        let Some(job) = job else { return };
                        if let Err(e) = write_job(&job, &mut made_dirs) {
                            errors.lock().unwrap().push(format!("{e:#}"));
                        }
                        live_bytes.fetch_sub(job.bytes.len() as u64, Ordering::Relaxed);
                        written.fetch_add(1, Ordering::Release);
                    }
                })
            })
            .collect();
        Self {
            state,
            cap_bytes,
            submitted,
            written,
            live_bytes,
            errors,
            handles,
        }
    }

    /// Gauge of bytes alive in the pool (queued + in-write), for the memory
    /// gate's ledger.
    pub fn live_bytes_gauge(&self) -> Arc<AtomicU64> {
        self.live_bytes.clone()
    }

    /// Queue one file. Blocks only when the in-flight byte cap is reached
    /// (the disk is slower than the producers).
    pub fn submit(&self, path: PathBuf, bytes: Vec<u8>, via_tmp: bool) {
        let (m, cv) = &*self.state;
        let mut st = m.lock().unwrap();
        while st.bytes > self.cap_bytes && !st.closed {
            st = cv.wait(st).unwrap();
        }
        st.bytes += bytes.len();
        self.live_bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        st.q.push_back(Job {
            path,
            bytes,
            via_tmp,
        });
        self.submitted.fetch_add(1, Ordering::Release);
        cv.notify_all();
    }

    /// Wait until everything submitted so far is durable (written or
    /// recorded as failed).
    pub fn flush(&self) {
        let target = self.submitted.load(Ordering::Acquire);
        while self.written.load(Ordering::Acquire) < target {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Drain, join the writers, and surface any write failures.
    pub fn finish(self) -> Result<()> {
        {
            let (m, cv) = &*self.state;
            m.lock().unwrap().closed = true;
            cv.notify_all();
        }
        for h in self.handles {
            let _ = h.join();
        }
        let errs = self.errors.lock().unwrap();
        if !errs.is_empty() {
            bail!(
                "{} file write(s) failed: {}",
                errs.len(),
                errs.iter().take(3).cloned().collect::<Vec<_>>().join("; ")
            );
        }
        Ok(())
    }
}

fn write_job(job: &Job, made_dirs: &mut HashSet<PathBuf>) -> Result<()> {
    if let Some(parent) = job.path.parent() {
        if !made_dirs.contains(parent) {
            std::fs::create_dir_all(parent).with_context(|| format!("create dir {parent:?}"))?;
            made_dirs.insert(parent.to_path_buf());
        }
    }
    if job.via_tmp {
        let mut name = job
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        name.push_str(".partial");
        let tmp = job.path.with_file_name(name);
        write_and_nudge(&tmp, &job.bytes).with_context(|| format!("write {tmp:?}"))?;
        std::fs::rename(&tmp, &job.path).with_context(|| format!("rename to {:?}", job.path))
    } else {
        write_and_nudge(&job.path, &job.bytes).with_context(|| format!("write {:?}", job.path))
    }
}

/// `fs::write`, then start (not wait for) writeback. Dirty pages can only
/// be reclaimed after writeback, and the kernel's own cadence leaves tens
/// of seconds of artifacts dirty inside the run's cgroup — reclaim then has
/// to flush them synchronously exactly when memory is tight, which is the
/// stall pressure the memory gate halves the rank loop on. Same total disk
/// IO, just started while the pages are still cheap to clean. Best-effort:
/// tmpfs has no writeback and foreign kernels just ignore it.
fn write_and_nudge(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(bytes)?;
    unsafe {
        use std::os::unix::io::AsRawFd;
        libc::sync_file_range(f.as_raw_fd(), 0, 0, libc::SYNC_FILE_RANGE_WRITE);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_land_and_finish_surfaces_errors() {
        let d = tempfile::tempdir().unwrap();
        let pool = WriterPool::new(2, 1024);
        pool.submit(d.path().join("a/b.txt"), b"hello".to_vec(), false);
        pool.submit(d.path().join("c.bin"), vec![7u8; 4096], true);
        pool.flush();
        assert_eq!(std::fs::read(d.path().join("a/b.txt")).unwrap(), b"hello");
        assert_eq!(
            std::fs::read(d.path().join("c.bin")).unwrap(),
            vec![7u8; 4096]
        );
        // No stray .partial left behind.
        assert!(!d.path().join("c.bin.partial").exists());
        pool.finish().unwrap();

        // A write into an uncreatable dir (file where a dir must go)
        // surfaces at finish.
        let pool = WriterPool::new(1, 1024);
        pool.submit(d.path().join("a/b.txt/child.txt"), b"x".to_vec(), false);
        assert!(pool.finish().is_err());
    }

    #[test]
    fn byte_cap_backpressures_then_drains() {
        let d = tempfile::tempdir().unwrap();
        let pool = WriterPool::new(1, 8); // tiny cap: every submit drains
        for i in 0..64 {
            pool.submit(d.path().join(format!("f{i}")), vec![0u8; 64], false);
        }
        pool.finish().unwrap();
        assert_eq!(std::fs::read_dir(d.path()).unwrap().count(), 64);
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Captures the git revision at build time for the engine provenance stamp
//! (`_meta/config.json` `engine.rev` — see `emit::engine_stamp`). Best-effort:
//! a build outside a git checkout stamps `unknown` rather than failing the
//! build.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn main() {
    let rev = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=COMMS_VIZ_GIT_REV={rev}");
    // Re-stamp when HEAD moves. The watched paths must be resolved through
    // git: in a linked worktree `.git` is a file and the real HEAD lives
    // under the main repo's .git/worktrees/<name>/. HEAD catches
    // checkout/detach; the logs/HEAD reflog is appended on every HEAD
    // movement (commits included), so together they cover branch-pointer
    // moves without watching a refs directory. Only existing paths may be
    // emitted — cargo treats a missing rerun-if-changed path as ALWAYS
    // dirty, which silently forced a full workspace rebuild (~4 min) on
    // every invocation in linked worktrees.
    let mut watched = false;
    for name in ["HEAD", "logs/HEAD"] {
        if let Some(p) = git(&["rev-parse", "--path-format=absolute", "--git-path", name]) {
            if std::path::Path::new(&p).exists() {
                println!("cargo:rerun-if-changed={p}");
                watched = true;
            }
        }
    }
    // Outside a git checkout (or with a git too old to resolve the paths)
    // fall back to explicit self-tracking so cargo doesn't rescan the whole
    // package for this script.
    if !watched {
        println!("cargo:rerun-if-changed=build.rs");
    }
}

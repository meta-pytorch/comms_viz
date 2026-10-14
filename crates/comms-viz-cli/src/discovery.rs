// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Run-setup discovery: the annotation mode, resolved from the trace
//! root's own layout before the pipeline starts.
//! `detect_annotation_mode` probes for a `kernel_annotations/` sibling of
//! the step dir, the layout signal that annotation pickles exist for this
//! job.

use std::path::PathBuf;

use comms_viz_core::annotations::AnnotationMode;

/// Absolutize a local trace root so its parent is a real directory.
fn absolutize(path: &std::path::Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

/// Auto-detect the annotation source from the job layout.
///
/// Today the only discoverable side channel is the pickle one:
/// `trace_root.rsplit("/", 1)[0] + "/kernel_annotations"` exists — i.e. a
/// kernel_annotations entry adjacent to the step dirs.
///
/// This is the single function to extend when the write pipeline starts
/// publishing inline-annotated traces with a discoverable marker of their own;
/// note that an individual inline-annotated trace is already handled without
/// any layout signal, because the trace declares it in its own envelope.
pub(crate) fn detect_annotation_mode(trace_root: &std::path::Path) -> AnnotationMode {
    if detect_cuda_graph_preprocessing(trace_root) {
        AnnotationMode::Pkl
    } else {
        AnnotationMode::Disabled
    }
}

pub(crate) fn detect_cuda_graph_preprocessing(trace_root: &std::path::Path) -> bool {
    let root = absolutize(trace_root);
    let path = root.to_string_lossy().trim_end_matches('/').to_string();
    let Some((parent, _leaf)) = path.rsplit_once('/') else {
        return false;
    };
    let candidate = format!("{parent}/kernel_annotations");
    let found = std::path::Path::new(&candidate).exists();
    eprintln!(
        "comms_viz: cuda_graph_preprocessing={found} (kernel_annotations sibling {} at {candidate})",
        if found { "present" } else { "absent" }
    );
    found
}

#[cfg(test)]
mod discovery_tests {
    use std::fs;

    use super::*;

    // ---- CUDA-graph auto-detect --------------------------------------------

    #[test]
    fn cgp_detected_from_kernel_annotations_sibling() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("traces/step_000000000106");
        fs::create_dir_all(&root).unwrap();
        assert!(!detect_cuda_graph_preprocessing(&root));
        fs::create_dir_all(tmp.path().join("traces/kernel_annotations")).unwrap();
        assert!(detect_cuda_graph_preprocessing(&root));
    }

    #[test]
    fn cgp_not_detected_from_dir_inside_trace_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("traces/step_000000000106");
        // kernel_annotations INSIDE the step dir is not the layout
        // signal (it must sit alongside the step dirs).
        fs::create_dir_all(root.join("kernel_annotations")).unwrap();
        assert!(!detect_cuda_graph_preprocessing(&root));
    }
}

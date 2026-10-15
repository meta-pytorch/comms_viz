// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Communication annotations retained by the compact timeline. Values remain
//! strings so optional and large identifiers survive CSV/DPG1 without rounding.

pub const KEYS: [&str; 10] = [
    "Collective name",
    "Process Group Ranks",
    "peer",
    "msg nelems",
    "dtype",
    "Sequence number",
    "graph id",
    "graph node id",
    "Process Group Name",
    // The launching call's id. A graph-replayed event carries its
    // cudaGraphLaunch's, so it identifies the replay.
    "correlation",
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommMetadata(pub [Option<String>; KEYS.len()]);

pub fn is_comm(name: &str) -> bool {
    crate::kernel_types::matches_comm_pattern(name)
}

pub fn json_text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(_) | serde_json::Value::Array(_) => Some(v.to_string()),
        _ => None,
    }
}

impl CommMetadata {
    pub fn key_index(key: &str) -> Option<usize> {
        // c10d stamps collective kernels with Seq; some producers use Sequence ID.
        if matches!(key, "Seq" | "Sequence ID" | "sequence_id") {
            return Some(5);
        }
        KEYS.iter().position(|k| *k == key)
    }
    pub fn set(&mut self, key: &str, value: String) {
        if let Some(i) = Self::key_index(key) {
            self.0[i] = Some(value);
        }
    }
    pub fn get(&self, i: usize) -> Option<&str> {
        self.0[i].as_deref()
    }
    pub fn overlay(&mut self, other: &Self) {
        for (dst, src) in self.0.iter_mut().zip(&other.0) {
            if src.is_some() {
                dst.clone_from(src);
            }
        }
    }
    pub fn operation(&self) -> Option<&str> {
        match self.get(0)?.trim().to_ascii_lowercase().as_str() {
            "send" => Some("send"),
            "recv" => Some("recv"),
            _ => None,
        }
    }
    pub fn graph_ids(&self) -> (Option<String>, Option<String>) {
        let node = self
            .get(7)
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n != 0);
        let graph = self.get(6).map(str::to_owned).or_else(|| {
            node.filter(|n| *n >> 32 != 0)
                .map(|n| (n >> 32).to_string())
        });
        (graph, node.map(|n| (n & 0xffff_ffff).to_string()))
    }
    pub fn ranks(&self) -> Option<Vec<i64>> {
        let ranks: Vec<i64> = serde_json::from_str(self.get(1)?).ok()?;
        if ranks.iter().any(|r| *r < 0) {
            return None;
        }
        Some(ranks)
    }
}

/// Global rank of a p2p peer: `peer` indexes the PG's member list.
pub fn peer_global(m: &CommMetadata, rank: i64) -> Option<i64> {
    let peer = m.get(2)?.parse::<usize>().ok()?;
    let members = m.ranks()?;
    // Never interpret a PG-relative peer as a global rank when annotations
    // are missing or inconsistent.
    if !members.contains(&rank) {
        return None;
    }
    members.get(peer).copied()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::analysis;
    use crate::dpg;
    use crate::emit;
    use crate::legacy;
    use crate::rows;
    use crate::trace;

    fn transfer(operation: &str, node: u64, peer: i64) -> serde_json::Value {
        json!({"name":"ncclDevKernel_SendRecv", "cat":"kernel", "ph":"X", "pid":0,"tid":7,
            "ts":100.125,"dur":210232.0,"args": {"stream":7,"grid":[1,1,1],
                "Collective name":operation,"Process Group Description":"PP", "Process Group Name":"0007",
                "Process Group Ranks":[959,1023], "peer":peer,"msg nelems":8192,"dtype":"BFloat16",
                "Seq":"00042","graph node id":(5_u64<<32)|node}})
    }

    fn pp_group() -> [trace::ProcessGroupConfig; 1] {
        [trace::ProcessGroupConfig {
            pg_name: "0007".into(),
            pg_desc: "PP".into(),
            backend_config: String::new(),
            pg_size: 2,
            ranks: vec![959, 1023],
        }]
    }

    fn frame(cols: &rows::ConvertedColumns, rank: i64) -> polars::prelude::DataFrame {
        analysis::process(cols, rank, &pp_group())
            .unwrap()
            .1
            .remove(0)
            .1
    }

    fn cell(df: &polars::prelude::DataFrame, column: &str) -> Option<String> {
        let v = df.column(column).unwrap().get(0).unwrap();
        if v.is_null() {
            None
        } else {
            Some(v.str_value().into_owned())
        }
    }

    #[test]
    fn endpoints_resolve_through_pg_membership() {
        for (op, rank, peer, node, dur) in [
            ("send", 959, 1, 6080, 210232.0),
            ("recv", 1023, 0, 4742, 193538.0),
        ] {
            let mut event = transfer(op, node, peer);
            event["dur"] = json!(dur);
            let root = json!({"baseTimeNanoseconds":1753000000000000123_u64,"traceEvents":[event]});
            let bytes = serde_json::to_vec(&root).unwrap();
            let scan = legacy::json::convert_json_scan_to_columns(&bytes).unwrap();
            let dom = legacy::json::convert_serde_to_columns(&root);
            let simd = legacy::json::convert_json_bytes_to_columns(&bytes).unwrap();
            legacy::json::assert_columns_equal(&scan, &dom).unwrap();
            legacy::json::assert_columns_equal(&scan, &simd).unwrap();
            assert_eq!(scan.base_ns, Some(1753000000000000123));
            let df = frame(&scan, rank);
            for (key, value) in [
                ("operation", op),
                ("src_rank", "959"),
                ("dst_rank", "1023"),
                ("pg_id", "0007"),
                ("replica_id", "0"),
                ("msg_nelems", "8192"),
                ("dtype", "BFloat16"),
                ("sequence_id", "00042"),
                ("graph_id", "5"),
                ("original_name", "ncclDevKernel_SendRecv"),
            ] {
                assert_eq!(cell(&df, key).as_deref(), Some(value), "{key}");
            }
            assert_eq!(cell(&df, "node_id"), Some(node.to_string()));
            assert_eq!(df.column("dur").unwrap().f64().unwrap().get(0), Some(dur));
            let csv = String::from_utf8(emit::write_csv_bytes(&mut df.clone()).unwrap()).unwrap();
            let (encoded, _) = dpg::encode_member(&csv).unwrap();
            let (names, columns) = dpg::decode_member(&encoded);
            for key in [
                "operation",
                "src_rank",
                "dst_rank",
                "pg_id",
                "sequence_id",
                "graph_node_id",
                "node_id",
                "original_name",
            ] {
                assert_eq!(
                    columns[names.iter().position(|n| n == key).unwrap()][0],
                    cell(&df, key).unwrap()
                );
            }
            // Native pftrace annotations take the same path after decoding.
            let (pf, _) =
                legacy::pftrace_twin::tree_to_pftrace(&root, &Default::default(), 7).unwrap();
            let parsed = trace::pftrace::parse_rows(&mut pf.clone()).unwrap();
            let converted =
                rows::convert_parsed_walked(parsed, &Default::default(), 7, false).unwrap();
            let native = frame(&converted, rank);
            for key in names.iter().filter(|n| !matches!(n.as_str(), "ts" | "dur")) {
                assert_eq!(cell(&native, key), cell(&df, key), "native {key}");
            }
        }
    }

    #[test]
    fn c10d_and_kineto_sequence_keys_share_a_slot() {
        for key in ["Seq", "Sequence number"] {
            let mut m = CommMetadata::default();
            m.set(key, "00042".into());
            assert_eq!(m.get(5), Some("00042"), "{key}");
        }
    }

    #[test]
    fn missing_transfer_fields_stay_null_and_top_level_fields_are_ignored() {
        let mut event = transfer("send", 6080, 1);
        // Only args carry communication annotations. A similarly named
        // top-level field must not overwrite or invent metadata.
        event["peer"] = json!(0);
        event["Collective name"] = json!("recv");
        event["args"].as_object_mut().unwrap().remove("Seq");
        event["args"].as_object_mut().unwrap().remove("peer");
        let root = json!({"traceEvents":[event]});
        let bytes = serde_json::to_vec(&root).unwrap();
        let cols = legacy::json::convert_json_scan_to_columns(&bytes).unwrap();
        legacy::json::assert_columns_equal(&cols, &legacy::json::convert_serde_to_columns(&root))
            .unwrap();
        let df = frame(&cols, 959);
        assert_eq!(cell(&df, "operation").as_deref(), Some("send"));
        assert_eq!(cell(&df, "src_rank").as_deref(), Some("959"));
        assert_eq!(cell(&df, "dst_rank"), None);
        assert_eq!(cell(&df, "sequence_id"), None);
        let csv = String::from_utf8(emit::write_csv_bytes(&mut df.clone()).unwrap()).unwrap();
        let (names, columns) = dpg::decode_member(&dpg::encode_member(&csv).unwrap().0);
        for key in ["dst_rank", "sequence_id"] {
            assert_eq!(columns[names.iter().position(|n| n == key).unwrap()][0], "");
        }
    }

    #[test]
    fn absent_or_invalid_annotations_do_not_invent_transfers() {
        let mut m = CommMetadata::default();
        assert_eq!(m.operation(), None);
        m.set("Collective name", "sendrecv".into());
        assert_eq!(m.operation(), None);
        m.set("Process Group Ranks", "[959, 1023]".into());
        m.set("peer", "1".into());
        assert_eq!(peer_global(&m, 959), Some(1023));
        assert_eq!(peer_global(&m, 958), None, "rank outside the PG");
        for peer in ["-1", "2", "1.0", "bad"] {
            m.set("peer", peer.into());
            assert_eq!(peer_global(&m, 959), None);
        }
        m.set("peer", "1".into());
        for ranks in ["[959, -1]", "[959, ...]", "not ranks"] {
            m.set("Process Group Ranks", ranks.into());
            assert_eq!(peer_global(&m, 959), None);
        }
        // Unsorted PG membership: the peer index is into the listed order.
        m.set("Process Group Ranks", "[6, 2, 4]".into());
        m.set("peer", "0".into());
        assert_eq!(peer_global(&m, 2), Some(6));
    }

    #[test]
    fn pickle_annotations_survive_walk_sort_and_compaction() {
        let event = transfer("recv", 4742, 0);
        let gnid = event["args"]["graph node id"].as_i64().unwrap();
        let anns = std::collections::HashMap::from([(
            gnid,
            vec![
                event["args"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ],
        )]);
        let root = json!({"traceEvents":[
            {"cat":"kernel","name":"ncclDevKernel_SendRecv","ts":2,"dur":3,"pid":0,"tid":7,"args":{"stream":7,"graph node id":gnid}},
            {"cat":"kernel","name":"compute","ts":1,"dur":1,"pid":0,"tid":8,"args":{"stream":8}}
        ]});
        let bytes = serde_json::to_vec(&root).unwrap();
        let digests = legacy::pkl::digests_from_annotations(&anns);
        let parsed = legacy::json::parse_bytes_walked(&bytes).unwrap();
        let view = rows::convert_parsed_walked_view(parsed, &digests, 7, true).unwrap();
        let df = analysis::process_view(&view, 1023, &pp_group())
            .unwrap()
            .1
            .remove(0)
            .1;
        assert_eq!(cell(&df, "src_rank").as_deref(), Some("959"));
        assert_eq!(cell(&df, "sequence_id").as_deref(), Some("00042"));
        let mut tree = root;
        legacy::tree::annotate_tree(&mut tree, &anns, 7);
        let columns = legacy::json::convert_serde_to_columns(&tree);
        assert_eq!(
            cell(&frame(&columns, 1023), "src_rank"),
            cell(&df, "src_rank")
        );
    }
}

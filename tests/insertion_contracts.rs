use sporse::{RawImpactQuantizer, SparseVec, SporseIndex};

#[test]
fn upsert_reports_identity_and_insert_remains_a_unit_adapter() {
    use sporse::{InsertOutcome, WandStats};
    let mut index = SporseIndex::new();
    let vector = SparseVec::new(vec![(2, 4.0)]);
    assert_eq!(
        index.upsert(7, &SparseVec::default()),
        InsertOutcome::Inserted
    );
    assert_eq!(
        index.upsert(7, &SparseVec::default()),
        InsertOutcome::Replaced
    );
    assert!(index.upsert(7, &vector).was_replaced());
    assert!(index.upsert(7, &vector).was_replaced());
    let insert: fn(&mut SporseIndex, u32, &SparseVec) = SporseIndex::insert;
    [8, 9]
        .into_iter()
        .for_each(|id| insert(&mut index, id, &vector));
    assert_eq!(index.len(), 3);
    index.build();
    let (results, stats): (Vec<(u32, f32)>, WandStats) = index.search_with_stats(&vector, 3);
    assert_eq!(results, vec![(7, 16.0), (8, 16.0), (9, 16.0)]);
    assert!(stats.docs_scored > 0);
}

#[test]
fn repeated_id_replaces_all_old_dimensions_and_counts_once() {
    let mut index = SporseIndex::new();
    index.insert(7, &SparseVec::new(vec![(0, 1.0), (1, 8.0)]));
    index.insert(7, &SparseVec::new(vec![(0, 2.0), (2, 3.0)]));
    assert_eq!(index.len(), 1);
    assert_eq!(index.num_dimensions(), 2);
    index.build();
    assert!(index.search(&SparseVec::new(vec![(1, 1.0)]), 2).is_empty());
    for query in [
        SparseVec::new(vec![(0, 1.0), (2, 1.0)]),
        SparseVec::new(vec![(0, 1.0), (2, 1.0), (99, -1.0)]),
    ] {
        assert_eq!(index.search(&query, 2), vec![(7, 5.0)]);
        assert_eq!(index.search_with_stats(&query, 2).0, vec![(7, 5.0)]);
    }
}

#[test]
fn replacement_to_empty_removes_scores_but_preserves_identity() {
    let mut index = SporseIndex::new();
    index.insert(7, &SparseVec::new(vec![(0, 9.0)]));
    index.insert(7, &SparseVec::default());
    index.insert(8, &SparseVec::default());
    index.insert(8, &SparseVec::new(vec![(1, 3.0)]));
    let mut cloned = index.clone();
    cloned.insert(7, &SparseVec::new(vec![(0, 4.0)]));
    assert_eq!(index.len(), 2);
    assert_eq!(cloned.len(), 2);
    index.build();
    cloned.build();
    let query = SparseVec::new(vec![(0, 1.0), (1, 1.0)]);
    assert_eq!(index.search(&query, 3), vec![(8, 3.0)]);
    assert_eq!(cloned.search(&query, 3), vec![(7, 4.0), (8, 3.0)]);
}

#[test]
fn raw_query_rejects_rescaling_overflow_and_underflow() {
    let large = SparseVec::new(vec![(3, f32::MAX)]);
    assert!(RawImpactQuantizer::new(0.5).unwrap().query(&large).is_err());
    let tiny = SparseVec::new(vec![(3, f32::from_bits(1))]);
    assert!(RawImpactQuantizer::new(2.0).unwrap().query(&tiny).is_err());
    assert_eq!(
        RawImpactQuantizer::new(2.0)
            .unwrap()
            .query(&SparseVec::new(vec![(3, 4.0)]))
            .unwrap(),
        vec![(3, 2.0)]
    );
}

#[test]
fn rounding_bound_remains_conservative_for_subnormal_query_weights() {
    let quantizer = RawImpactQuantizer::new(1e-30).unwrap();
    let doc = SparseVec::new(vec![(0, 1.5e30)]);
    let query = SparseVec::new(vec![(0, f32::from_bits(1))]);
    let encoded = quantizer.document(&doc).unwrap()[0].1;
    let weight = f64::from(query.pairs()[0].1);
    let original_score = weight * f64::from(doc.pairs()[0].1);
    let rounded_score = weight * f64::from(encoded) / f64::from(quantizer.scale());
    let error = (original_score - rounded_score).abs();
    assert!(error > 0.0);
    assert!(f64::from(quantizer.score_error_bound(&query).unwrap()) >= error);
}

#[cfg(feature = "serde")]
#[test]
fn deserialized_sparse_vectors_obey_constructor_invariants() {
    let decoded: SparseVec =
        serde_json::from_str(r#"{"pairs":[[1,2.0],[0,1.0],[1,3.0],[2,0.0],[3,1.0],[3,-1.0]]}"#)
            .unwrap();
    assert_eq!(decoded.pairs(), &[(0, 1.0), (1, 5.0)]);
    assert_eq!(decoded.nnz(), 2);
    assert_eq!(decoded.dot(&SparseVec::new(vec![(0, 1.0)])), 1.0);
}

#[cfg(feature = "serde")]
#[test]
fn unbuilt_index_serialization_and_deserialization_are_rejected() {
    assert!(serde_json::to_string(&SporseIndex::new()).is_err());
    assert!(serde_json::from_str::<SporseIndex>(
        r#"{"postings":{},"num_docs":0,"wand_safe":true,"built":false}"#,
    )
    .is_err());
}

#[cfg(feature = "cli")]
#[test]
fn cli_build_reports_unique_ids_and_queries_the_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("documents.jsonl");
    let index = dir.path().join("index.json");
    std::fs::write(
        &docs,
        "{\"id\":7,\"vec\":[[0,9.0]]}\n{\"id\":7,\"vec\":[[1,2.0]]}\n",
    )
    .unwrap();
    let build = std::process::Command::new(env!("CARGO_BIN_EXE_sporse"))
        .arg("build")
        .arg(&docs)
        .arg("-o")
        .arg(&index)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(String::from_utf8_lossy(&build.stderr).starts_with("indexed 1 documents"));
    let query = std::process::Command::new(env!("CARGO_BIN_EXE_sporse"))
        .arg("search")
        .arg(&index)
        .args(["--query", "[[0,1.0],[1,1.0]]", "-k", "2"])
        .output()
        .unwrap();
    assert!(
        query.status.success(),
        "{}",
        String::from_utf8_lossy(&query.stderr)
    );
    assert_eq!(String::from_utf8(query.stdout).unwrap(), "7\t2.000000\n");
}

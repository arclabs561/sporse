//! Convert weighted `postings` entries into the `SparseVec` inputs `sporse`
//! indexes for built top-k retrieval.
//!
//! `postings` owns mutable term postings and update semantics. `sporse` owns a
//! finalized sparse-vector index with Block-Max WAND metadata. The bridge is a
//! small adapter: assign stable term dimensions, reconstruct per-document sparse
//! vectors from live postings, then compare scores against direct weighted
//! postings accumulation.
//!
//! Run with:
//! `cargo run --example postings_bridge`

use std::collections::BTreeMap;

use postings::PostingsIndex;
use sporse::{SparseVec, SporseIndex};

fn main() {
    let docs: &[(u32, &[(&str, f32)])] = &[
        (
            0,
            &[
                ("neural", 1.8),
                ("network", 2.1),
                ("deep", 0.9),
                ("learning", 1.2),
            ],
        ),
        (1, &[("graph", 2.4), ("network", 1.1), ("node", 1.7)]),
        (
            2,
            &[
                ("neural", 0.7),
                ("search", 2.2),
                ("retrieval", 2.6),
                ("learning", 0.5),
            ],
        ),
        (
            3,
            &[
                ("retrieval", 1.9),
                ("sparse", 2.8),
                ("index", 1.3),
                ("search", 1.0),
            ],
        ),
    ];

    let mut postings: PostingsIndex<String, f32> = PostingsIndex::new();
    for &(doc_id, terms) in docs {
        let weighted: Vec<(String, f32)> = terms.iter().map(|&(t, w)| (t.to_string(), w)).collect();
        postings
            .add_weighted_document(doc_id, &weighted)
            .expect("doc ids are unique");
    }

    let vocab = vocabulary(&postings);
    let doc_vectors = sparse_vectors_from_postings(&postings, &vocab);

    let mut sporse = SporseIndex::new();
    for (&doc_id, vector) in &doc_vectors {
        sporse.insert(doc_id, vector);
    }
    sporse.build();

    let query_terms: &[(&str, f32)] = &[("neural", 1.5), ("retrieval", 2.0), ("search", 1.0)];
    let postings_ranked = score_weighted_postings(&postings, query_terms, 3);
    let query = sparse_query(query_terms, &vocab);
    let sporse_ranked = sporse.search(&query, 3);

    assert_rankings_close(&postings_ranked, &sporse_ranked);

    println!("vocabulary:");
    for (term, dim) in &vocab {
        println!("  {dim:02} {term}");
    }

    println!("\npostings top-k: {postings_ranked:?}");
    println!("sporse top-k:   {sporse_ranked:?}");
}

fn vocabulary(index: &PostingsIndex<String, f32>) -> BTreeMap<String, u32> {
    let mut terms: Vec<String> = index.terms().cloned().collect();
    terms.sort();
    terms
        .into_iter()
        .enumerate()
        .map(|(dim, term)| (term, dim as u32))
        .collect()
}

fn sparse_vectors_from_postings(
    index: &PostingsIndex<String, f32>,
    vocab: &BTreeMap<String, u32>,
) -> BTreeMap<u32, SparseVec> {
    let mut per_doc: BTreeMap<u32, Vec<(u32, f32)>> = BTreeMap::new();

    for (term, &dim) in vocab {
        for (doc_id, weight) in index.postings_iter(term.as_str()) {
            per_doc.entry(doc_id).or_default().push((dim, weight));
        }
    }

    per_doc
        .into_iter()
        .map(|(doc_id, pairs)| (doc_id, SparseVec::new(pairs)))
        .collect()
}

fn sparse_query(terms: &[(&str, f32)], vocab: &BTreeMap<String, u32>) -> SparseVec {
    SparseVec::new(
        terms
            .iter()
            .filter_map(|&(term, weight)| vocab.get(term).map(|&dim| (dim, weight)))
            .collect(),
    )
}

fn score_weighted_postings(
    index: &PostingsIndex<String, f32>,
    terms: &[(&str, f32)],
    k: usize,
) -> Vec<(u32, f32)> {
    let mut scores = BTreeMap::new();
    for &(term, query_weight) in terms {
        for (doc_id, doc_weight) in index.postings_iter(term) {
            *scores.entry(doc_id).or_insert(0.0) += query_weight * doc_weight;
        }
    }

    let mut ranked: Vec<(u32, f32)> = scores
        .into_iter()
        .filter(|&(_, score)| score != 0.0)
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k);
    ranked
}

fn assert_rankings_close(expected: &[(u32, f32)], got: &[(u32, f32)]) {
    assert_eq!(expected.len(), got.len());
    for (&(expected_doc, expected_score), &(got_doc, got_score)) in expected.iter().zip(got) {
        assert_eq!(expected_doc, got_doc);
        assert!(
            (expected_score - got_score).abs() < 1e-5,
            "doc {expected_doc}: expected {expected_score}, got {got_score}"
        );
    }
}

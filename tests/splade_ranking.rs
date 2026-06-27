//! End-to-end integration test against the public crate API.
//!
//! Invariant under test: top-k inner-product ranking correctness on a
//! SPLADE-style lexical collection. Dimensions stand for vocabulary terms and
//! weights for learned term importance. The document sharing the most
//! high-weight query terms must rank first, scores must be strictly
//! decreasing down the ranking, and a document with zero query-term overlap
//! must be excluded from the results entirely (the inverted index never
//! touches its postings).
//!
//! This exercises the full pipeline through the public surface only:
//! `SparseVec::new` -> `SporseIndex::insert` -> `SporseIndex::build` ->
//! `SporseIndex::search`. The expected order is hand-constructed so that the
//! ground-truth ranking is unambiguous (no ties).

use sporse::{SparseVec, SporseIndex};

// Vocabulary-term dimension ids ("query terms" carrying high weight).
const MACHINE: u32 = 10;
const LEARNING: u32 = 11;
const MODEL: u32 = 12;
// Non-query "noise" terms each doc may also carry.
const NOISE_A: u32 = 40;
const NOISE_B: u32 = 50;
const NOISE_C: u32 = 60;
const NOISE_D: u32 = 61;
const NOISE_E: u32 = 99;

#[test]
fn splade_style_ranking_is_correct_and_excludes_zero_overlap() {
    let mut index = SporseIndex::new();

    // doc 0: shares all three query terms (most overlap) + one noise term.
    index.insert(
        0,
        &SparseVec::new(vec![
            (MACHINE, 2.0),
            (LEARNING, 2.0),
            (MODEL, 1.0),
            (NOISE_B, 5.0),
        ]),
    );
    // doc 1: shares two query terms + one noise term.
    index.insert(
        1,
        &SparseVec::new(vec![(MACHINE, 1.0), (LEARNING, 1.0), (NOISE_E, 9.0)]),
    );
    // doc 2: shares one query term + one noise term.
    index.insert(2, &SparseVec::new(vec![(MODEL, 1.0), (NOISE_A, 8.0)]));
    // doc 3: zero query-term overlap; should never appear in results.
    index.insert(3, &SparseVec::new(vec![(NOISE_C, 4.0), (NOISE_D, 4.0)]));

    index.build();

    // Query weights: machine/learning are the dominant terms, model secondary.
    let query = SparseVec::new(vec![(MACHINE, 3.0), (LEARNING, 3.0), (MODEL, 2.0)]);

    // Ground-truth inner products against the query:
    //   doc 0: 3*2 + 3*2 + 2*1 = 14.0
    //   doc 1: 3*1 + 3*1       =  6.0
    //   doc 2: 2*1             =  2.0
    //   doc 3: 0               (no overlap -> excluded)
    let results = index.search(&query, 4);

    // Zero-overlap doc is excluded: only 3 of the 4 documents score > 0.
    assert_eq!(
        results.len(),
        3,
        "zero-overlap doc 3 must be excluded, got {results:?}"
    );
    assert!(
        results.iter().all(|&(doc_id, _)| doc_id != 3),
        "doc 3 has no query-term overlap and must not appear: {results:?}"
    );

    // Ranking: most-shared-high-weight-terms doc ranks first, then by score.
    assert_eq!(
        results[0].0, 0,
        "doc 0 shares all query terms and must rank first"
    );
    assert_eq!(results[1].0, 1);
    assert_eq!(results[2].0, 2);

    // Exact inner-product scores.
    assert!(
        (results[0].1 - 14.0).abs() < 1e-5,
        "doc 0 score: {}",
        results[0].1
    );
    assert!(
        (results[1].1 - 6.0).abs() < 1e-5,
        "doc 1 score: {}",
        results[1].1
    );
    assert!(
        (results[2].1 - 2.0).abs() < 1e-5,
        "doc 2 score: {}",
        results[2].1
    );

    // Scores are strictly decreasing down the ranking (no ties in this case).
    assert!(
        results[0].1 > results[1].1 && results[1].1 > results[2].1,
        "scores must be strictly decreasing: {results:?}"
    );
}

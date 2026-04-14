//! Sparse vector index for learned sparse retrieval.
//!
//! Indexes sparse vectors (SPLADE, LADE, learned sparse representations)
//! using an inverted index with Block-Max WAND traversal for exact top-k
//! inner product search.
//!
//! # Quick start
//!
//! ```
//! use sporse::{SparseVec, SporseIndex};
//!
//! let mut index = SporseIndex::new();
//!
//! // Insert documents as sparse vectors
//! index.insert(0, &SparseVec::new(vec![(0, 1.0), (3, 2.5), (7, 0.8)]));
//! index.insert(1, &SparseVec::new(vec![(1, 3.0), (3, 1.0)]));
//! index.insert(2, &SparseVec::new(vec![(0, 0.5), (7, 2.0)]));
//!
//! // Build the index (computes block-max metadata)
//! index.build();
//!
//! // Search: returns (doc_id, score) pairs, highest score first
//! let query = SparseVec::new(vec![(0, 1.0), (3, 1.0)]);
//! let results = index.search(&query, 2);
//! assert_eq!(results[0].0, 0); // doc 0 scores 1.0*1.0 + 2.5*1.0 = 3.5
//! ```

mod posting;
mod wand;

use std::collections::HashMap;

// ── SparseVec ────────────────────────────────────────────────────────────────

/// A sparse vector: sorted list of (dimension, weight) pairs.
///
/// Dimensions are sorted ascending. Zero-weight entries are removed
/// on construction. Weights should be non-negative for correct WAND search.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseVec {
    pairs: Vec<(u32, f32)>,
}

impl SparseVec {
    /// Create from unsorted pairs. Sorts by dimension, deduplicates,
    /// and removes zero-weight entries.
    pub fn new(mut pairs: Vec<(u32, f32)>) -> Self {
        pairs.sort_unstable_by_key(|&(d, _)| d);
        pairs.dedup_by_key(|p| p.0);
        pairs.retain(|&(_, w)| w != 0.0);
        Self { pairs }
    }

    /// Create from pre-sorted, deduplicated pairs without validation.
    pub fn from_sorted(pairs: Vec<(u32, f32)>) -> Self {
        Self { pairs }
    }

    /// The (dimension, weight) pairs, sorted by dimension.
    #[inline]
    pub fn pairs(&self) -> &[(u32, f32)] {
        &self.pairs
    }

    /// Number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.pairs.len()
    }

    /// Whether the vector has no non-zero entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

impl From<Vec<(u32, f32)>> for SparseVec {
    fn from(pairs: Vec<(u32, f32)>) -> Self {
        Self::new(pairs)
    }
}

// ── SporseIndex ──────────────────────────────────────────────────────────────

/// Inverted index for sparse vector retrieval using Block-Max WAND.
///
/// Insert documents with [`insert`](SporseIndex::insert), call
/// [`build`](SporseIndex::build) to finalize, then
/// [`search`](SporseIndex::search) for top-k results by inner product.
pub struct SporseIndex {
    postings: HashMap<u32, posting::PostingList>,
    num_docs: u32,
    built: bool,
}

impl SporseIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            num_docs: 0,
            built: false,
        }
    }

    /// Insert a document. Each non-zero dimension of `vec` adds an entry
    /// to the corresponding posting list.
    ///
    /// # Panics
    ///
    /// Panics if called after [`build`](SporseIndex::build).
    pub fn insert(&mut self, doc_id: u32, vec: &SparseVec) {
        assert!(!self.built, "cannot insert after build");
        for &(dim, weight) in vec.pairs() {
            self.postings
                .entry(dim)
                .or_insert_with(posting::PostingList::new)
                .push(doc_id, weight);
        }
        self.num_docs += 1;
    }

    /// Finalize the index: sort posting lists and compute block-max metadata.
    /// Must be called before [`search`](SporseIndex::search).
    pub fn build(&mut self) {
        for list in self.postings.values_mut() {
            list.finalize();
        }
        self.built = true;
    }

    /// Search for the top-k documents by inner product with `query`.
    ///
    /// Returns `(doc_id, score)` pairs in descending score order.
    ///
    /// # Panics
    ///
    /// Panics if [`build`](SporseIndex::build) has not been called.
    pub fn search(&self, query: &SparseVec, k: usize) -> Vec<(u32, f32)> {
        assert!(self.built, "must call build() before search()");
        if k == 0 || query.is_empty() {
            return Vec::new();
        }

        let mut cursors: Vec<wand::Cursor> = Vec::new();
        for &(dim, query_weight) in query.pairs() {
            if let Some(list) = self.postings.get(&dim) {
                cursors.push(wand::Cursor::new(list, query_weight));
            }
        }

        if cursors.is_empty() {
            return Vec::new();
        }

        wand::search_bmw(&mut cursors, k)
    }

    /// Number of documents inserted.
    pub fn len(&self) -> u32 {
        self.num_docs
    }

    /// Whether no documents have been inserted.
    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
    }

    /// Number of distinct dimensions across all documents.
    pub fn num_dimensions(&self) -> usize {
        self.postings.len()
    }
}

impl Default for SporseIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_search() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 1.0), (3, 2.5), (7, 0.8)]));
        index.insert(1, &SparseVec::new(vec![(1, 3.0), (3, 1.0)]));
        index.insert(2, &SparseVec::new(vec![(0, 0.5), (7, 2.0)]));
        index.build();

        let query = SparseVec::new(vec![(0, 1.0), (3, 1.0)]);
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        // doc 0: 1.0*1.0 + 2.5*1.0 = 3.5
        // doc 1: 1.0*1.0 = 1.0
        // doc 2: 0.5*1.0 = 0.5
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 3.5).abs() < 1e-5);
        assert_eq!(results[1].0, 1);
        assert!((results[1].1 - 1.0).abs() < 1e-5);
        assert_eq!(results[2].0, 2);
        assert!((results[2].1 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn top_k_limits_results() {
        let mut index = SporseIndex::new();
        for i in 0..10u32 {
            index.insert(i, &SparseVec::new(vec![(0, i as f32 + 1.0)]));
        }
        index.build();

        let query = SparseVec::new(vec![(0, 1.0)]);
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 9);
        assert_eq!(results[1].0, 8);
        assert_eq!(results[2].0, 7);
    }

    #[test]
    fn disjoint_query_returns_empty() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 1.0), (1, 2.0)]));
        index.build();

        let query = SparseVec::new(vec![(99, 1.0)]);
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn empty_query_returns_empty() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 1.0)]));
        index.build();

        let results = index.search(&SparseVec::default(), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn single_document() {
        let mut index = SporseIndex::new();
        index.insert(42, &SparseVec::new(vec![(5, 3.0), (10, 2.0)]));
        index.build();

        let query = SparseVec::new(vec![(5, 1.0), (10, 1.0)]);
        let results = index.search(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
        assert!((results[0].1 - 5.0).abs() < 1e-5);
    }

    #[test]
    fn score_correctness_multi_term() {
        let mut index = SporseIndex::new();
        // doc 0: dims 1=2.0, 2=3.0, 5=1.0
        index.insert(0, &SparseVec::new(vec![(1, 2.0), (2, 3.0), (5, 1.0)]));
        // doc 1: dims 2=1.0, 3=4.0
        index.insert(1, &SparseVec::new(vec![(2, 1.0), (3, 4.0)]));
        index.build();

        // query: dims 1=0.5, 2=2.0, 3=1.0
        let query = SparseVec::new(vec![(1, 0.5), (2, 2.0), (3, 1.0)]);
        let results = index.search(&query, 2);

        // doc 0: 2.0*0.5 + 3.0*2.0 = 7.0
        // doc 1: 1.0*2.0 + 4.0*1.0 = 6.0
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 7.0).abs() < 1e-5);
        assert_eq!(results[1].0, 1);
        assert!((results[1].1 - 6.0).abs() < 1e-5);
    }

    #[test]
    fn sparse_vec_sorts_and_deduplicates() {
        let sv = SparseVec::new(vec![(5, 1.0), (2, 0.0), (3, 2.0), (5, 3.0), (1, 1.0)]);
        // Sorted, deduplicated, zeros removed: dims 1, 3, 5
        assert_eq!(sv.nnz(), 3);
        assert_eq!(sv.pairs()[0].0, 1);
        assert_eq!(sv.pairs()[1].0, 3);
        assert_eq!(sv.pairs()[2].0, 5);
    }

    #[test]
    fn many_documents_block_boundary() {
        // Test with enough docs to span multiple blocks (BLOCK_SIZE = 128).
        let mut index = SporseIndex::new();
        for i in 0..500u32 {
            index.insert(i, &SparseVec::new(vec![(0, i as f32 + 1.0)]));
        }
        index.build();

        let query = SparseVec::new(vec![(0, 1.0)]);
        let results = index.search(&query, 5);

        assert_eq!(results.len(), 5);
        for (rank, &(doc_id, _)) in results.iter().enumerate() {
            assert_eq!(doc_id, 499 - rank as u32);
        }
    }

    #[test]
    fn partial_term_overlap() {
        // Query shares some but not all terms with each doc.
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 1.0), (1, 1.0), (2, 1.0)]));
        index.insert(1, &SparseVec::new(vec![(3, 1.0), (4, 1.0), (5, 1.0)]));
        index.insert(2, &SparseVec::new(vec![(0, 1.0), (3, 1.0)]));
        index.build();

        // Query touches dims 0 and 3
        let query = SparseVec::new(vec![(0, 2.0), (3, 2.0)]);
        let results = index.search(&query, 3);

        // doc 0: 1.0*2.0 = 2.0
        // doc 1: 1.0*2.0 = 2.0
        // doc 2: 1.0*2.0 + 1.0*2.0 = 4.0
        assert_eq!(results[0].0, 2);
        assert!((results[0].1 - 4.0).abs() < 1e-5);
    }

    #[test]
    fn k_larger_than_collection() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 1.0)]));
        index.insert(1, &SparseVec::new(vec![(0, 2.0)]));
        index.build();

        let query = SparseVec::new(vec![(0, 1.0)]);
        let results = index.search(&query, 100);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 0);
    }

    #[test]
    #[should_panic(expected = "cannot insert after build")]
    fn insert_after_build_panics() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 1.0)]));
        index.build();
        index.insert(1, &SparseVec::new(vec![(0, 1.0)]));
    }

    #[test]
    #[should_panic(expected = "must call build")]
    fn search_before_build_panics() {
        let index = SporseIndex::new();
        index.search(&SparseVec::new(vec![(0, 1.0)]), 1);
    }

    #[test]
    fn brute_force_parity() {
        // Verify WAND results match brute-force inner product on a small collection.
        let docs: Vec<SparseVec> = vec![
            SparseVec::new(vec![(0, 1.0), (2, 3.0), (5, 0.5)]),
            SparseVec::new(vec![(1, 2.0), (2, 1.0), (4, 4.0)]),
            SparseVec::new(vec![(0, 0.5), (1, 0.5), (3, 2.0)]),
            SparseVec::new(vec![(2, 2.0), (5, 3.0)]),
            SparseVec::new(vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]),
        ];

        let mut index = SporseIndex::new();
        for (i, doc) in docs.iter().enumerate() {
            index.insert(i as u32, doc);
        }
        index.build();

        let query = SparseVec::new(vec![(0, 1.0), (2, 2.0), (5, 1.0)]);
        let results = index.search(&query, 5);

        // Brute-force scores:
        // doc 0: 1.0*1.0 + 3.0*2.0 + 0.5*1.0 = 7.5
        // doc 1: 0 + 1.0*2.0 + 0 = 2.0
        // doc 2: 0.5*1.0 + 0 + 0 = 0.5
        // doc 3: 0 + 2.0*2.0 + 3.0*1.0 = 7.0
        // doc 4: 1.0*1.0 + 1.0*2.0 + 0 = 3.0
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 7.5).abs() < 1e-5);
        assert_eq!(results[1].0, 3);
        assert!((results[1].1 - 7.0).abs() < 1e-5);
        assert_eq!(results[2].0, 4);
        assert!((results[2].1 - 3.0).abs() < 1e-5);
        assert_eq!(results[3].0, 1);
        assert!((results[3].1 - 2.0).abs() < 1e-5);
        assert_eq!(results[4].0, 2);
        assert!((results[4].1 - 0.5).abs() < 1e-5);
    }
}

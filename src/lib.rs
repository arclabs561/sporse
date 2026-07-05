//! Sparse vector index for learned sparse retrieval.
//!
//! Indexes sparse vectors using an inverted index with Block-Max WAND
//! traversal for exact top-k inner product search. The vectors can be lexical
//! (SPLADE-style vocabulary-term weights) or latent (the composite codes the
//! `sae` module learns); the index treats both the same way.
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

/// Composite-code sparse-autoencoder codes (CCSA) and their index integration.
pub mod sae;

/// Updatable, durable index backed by segstore (the optional `store` feature).
#[cfg(feature = "store")]
pub mod store;

use std::{collections::HashMap, fmt};

/// Errors returned when converting sparse vectors to quantized raw impacts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RawImpactError {
    /// Quantization scale must be finite and positive.
    InvalidScale { scale: f32 },
    /// Raw impact documents and queries require finite non-negative weights.
    InvalidWeight { dim: u32, weight: f32 },
    /// A positive weight rounded to zero at the requested scale.
    RoundedToZero { dim: u32, weight: f32, scale: f32 },
    /// A scaled document impact cannot fit in `u32`.
    WeightOverflow { dim: u32, weight: f32, scale: f32 },
}

impl fmt::Display for RawImpactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidScale { scale } => {
                write!(f, "raw impact scale must be finite and positive: {scale}")
            }
            Self::InvalidWeight { dim, weight } => {
                write!(
                    f,
                    "raw impact weight must be finite and non-negative for dim {dim}: {weight}"
                )
            }
            Self::RoundedToZero { dim, weight, scale } => {
                write!(
                    f,
                    "raw impact weight for dim {dim} rounds to zero: weight={weight}, scale={scale}"
                )
            }
            Self::WeightOverflow { dim, weight, scale } => {
                write!(
                    f,
                    "raw impact weight for dim {dim} exceeds u32: weight={weight}, scale={scale}"
                )
            }
        }
    }
}

impl std::error::Error for RawImpactError {}

/// Quantization policy for writing `SparseVec` weights as raw impacts.
///
/// Persist the scale value beside a raw impact generation and reload it with
/// [`RawImpactQuantizer::new`] before encoding queries for that generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawImpactQuantizer {
    scale: f32,
}

impl RawImpactQuantizer {
    /// Create a quantizer from a finite positive scale.
    pub fn new(scale: f32) -> Result<Self, RawImpactError> {
        validate_raw_impact_scale(scale)?;
        Ok(Self { scale })
    }

    /// The scale used to convert `f32` weights to `u32` impacts.
    pub fn scale(self) -> f32 {
        self.scale
    }

    /// Quantize a document vector for `postings::raw`.
    pub fn document(self, vector: &SparseVec) -> Result<Vec<(u64, u32)>, RawImpactError> {
        vector.to_raw_impact_document(self.scale)
    }

    /// Rescale a query vector for documents written by this quantizer.
    pub fn query(self, vector: &SparseVec) -> Result<Vec<(u64, f32)>, RawImpactError> {
        vector.to_raw_impact_query(self.scale)
    }

    /// Bound raw-impact score error from document-weight rounding for a query.
    ///
    /// Document quantization rounds each stored weight to
    /// `round(weight * scale)`. For non-negative query weights, each matching
    /// dimension can therefore change a document score by at most
    /// `query_weight * 0.5 / scale`. The returned value is a conservative
    /// absolute per-document score bound for this quantizer and query.
    pub fn score_error_bound(self, query: &SparseVec) -> Result<f32, RawImpactError> {
        let mut bound = 0.0;
        for &(dim, weight) in query.pairs() {
            validate_raw_impact_weight(dim, weight)?;
            bound += weight * 0.5 / self.scale;
        }
        Ok(bound)
    }
}

// ── SparseVec ────────────────────────────────────────────────────────────────

/// A sparse vector: sorted list of (dimension, weight) pairs.
///
/// Dimensions are sorted ascending. Zero-weight entries are removed
/// on construction. Finite non-negative weights use Block-Max WAND search;
/// negative or non-finite weights fall back to exact sparse accumulation.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseVec {
    pairs: Vec<(u32, f32)>,
}

impl SparseVec {
    /// Create from unsorted pairs. Sorts by dimension, sums duplicate
    /// dimensions, and removes zero-weight entries.
    pub fn new(mut pairs: Vec<(u32, f32)>) -> Self {
        pairs.sort_unstable_by_key(|&(d, _)| d);
        let mut folded = Vec::with_capacity(pairs.len());
        for (dim, weight) in pairs {
            if weight == 0.0 {
                continue;
            }
            if let Some((last_dim, last_weight)) = folded.last_mut() {
                if *last_dim == dim {
                    *last_weight += weight;
                    continue;
                }
            }
            folded.push((dim, weight));
        }
        folded.retain(|&(_, weight)| weight != 0.0);
        Self { pairs: folded }
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

    #[inline]
    fn wand_safe(&self) -> bool {
        self.pairs
            .iter()
            .all(|&(_, weight)| weight.is_finite() && weight >= 0.0)
    }

    /// Inner product with another sparse vector.
    ///
    /// Uses a merge-join on sorted dimensions: O(nnz(self) + nnz(other)).
    pub fn dot(&self, other: &SparseVec) -> f32 {
        let (mut ai, mut bi) = (0, 0);
        let (a, b) = (&self.pairs, &other.pairs);
        let mut sum = 0.0f32;
        while ai < a.len() && bi < b.len() {
            match a[ai].0.cmp(&b[bi].0) {
                std::cmp::Ordering::Equal => {
                    sum += a[ai].1 * b[bi].1;
                    ai += 1;
                    bi += 1;
                }
                std::cmp::Ordering::Less => ai += 1,
                std::cmp::Ordering::Greater => bi += 1,
            }
        }
        sum
    }

    /// Quantize this vector as raw document impacts.
    ///
    /// The returned `(dimension, impact)` pairs are sorted by dimension and use
    /// `u64` dimensions to match `postings::raw::RawTermId` without depending on
    /// `postings`. This is for non-negative learned-sparse weights; negative or
    /// non-finite weights return an error.
    pub fn to_raw_impact_document(&self, scale: f32) -> Result<Vec<(u64, u32)>, RawImpactError> {
        validate_raw_impact_scale(scale)?;
        self.pairs
            .iter()
            .map(|&(dim, weight)| {
                validate_raw_impact_weight(dim, weight)?;
                let scaled = weight as f64 * scale as f64;
                if scaled > u32::MAX as f64 {
                    return Err(RawImpactError::WeightOverflow { dim, weight, scale });
                }
                let impact = scaled.round();
                if impact < 1.0 {
                    return Err(RawImpactError::RoundedToZero { dim, weight, scale });
                }
                Ok((dim as u64, impact as u32))
            })
            .collect()
    }

    /// Rescale this vector as a raw impact query.
    ///
    /// Pair this with documents from [`Self::to_raw_impact_document`] at the
    /// same scale. The approximate raw score is
    /// `round(doc_weight * scale) * query_weight / scale`.
    pub fn to_raw_impact_query(&self, scale: f32) -> Result<Vec<(u64, f32)>, RawImpactError> {
        validate_raw_impact_scale(scale)?;
        self.pairs
            .iter()
            .map(|&(dim, weight)| {
                validate_raw_impact_weight(dim, weight)?;
                Ok((dim as u64, weight / scale))
            })
            .collect()
    }
}

fn validate_raw_impact_scale(scale: f32) -> Result<(), RawImpactError> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err(RawImpactError::InvalidScale { scale })
    }
}

fn validate_raw_impact_weight(dim: u32, weight: f32) -> Result<(), RawImpactError> {
    if weight.is_finite() && weight >= 0.0 {
        Ok(())
    } else {
        Err(RawImpactError::InvalidWeight { dim, weight })
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
///
/// With the `serde` feature, the entire index can be serialized after
/// [`build`](SporseIndex::build) and deserialized without rebuilding.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SporseIndex {
    postings: HashMap<u32, posting::PostingList>,
    num_docs: u32,
    #[cfg_attr(feature = "serde", serde(default = "default_wand_safe"))]
    wand_safe: bool,
    built: bool,
}

#[cfg(feature = "serde")]
fn default_wand_safe() -> bool {
    false
}

impl SporseIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            num_docs: 0,
            wand_safe: true,
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
        self.wand_safe &= vec.wand_safe();
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
        if !self.can_use_wand(query) {
            return self.search_exact_above(query, k, 0.0);
        }

        let mut cursors = self.cursors_for(query);
        if cursors.is_empty() {
            return Vec::new();
        }

        wand::search_bmw(&mut cursors, k)
    }

    #[cfg(any(feature = "store", test))]
    pub(crate) fn search_above(
        &self,
        query: &SparseVec,
        k: usize,
        min_score: f32,
    ) -> Vec<(u32, f32)> {
        assert!(self.built, "must call build() before search()");
        if k == 0 || query.is_empty() {
            return Vec::new();
        }
        if !self.can_use_wand(query) {
            return self.search_exact_above(query, k, min_score);
        }

        let mut cursors = self.cursors_for(query);
        if cursors.is_empty() {
            return Vec::new();
        }

        wand::search_bmw_above(&mut cursors, k, min_score)
    }

    #[cfg(any(feature = "store", test))]
    pub(crate) fn query_upper_bound(&self, query: &SparseVec) -> f32 {
        assert!(self.built, "must call build() before search()");
        if !self.can_use_wand(query) {
            return f32::INFINITY;
        }
        query
            .pairs()
            .iter()
            .filter_map(|(dim, query_weight)| {
                self.postings
                    .get(dim)
                    .map(|list| list.max_weight * *query_weight)
            })
            .sum()
    }

    /// Search with per-query WAND statistics. For profiling and diagnostics.
    ///
    /// Returns `(results, stats)` where results are `(doc_id, score)` pairs
    /// in descending score order, and stats describe the WAND traversal.
    #[doc(hidden)]
    pub fn search_with_stats(
        &self,
        query: &SparseVec,
        k: usize,
    ) -> (Vec<(u32, f32)>, wand::WandStats) {
        assert!(self.built, "must call build() before search()");
        if k == 0 || query.is_empty() {
            return (Vec::new(), wand::WandStats::default());
        }
        if !self.can_use_wand(query) {
            return (
                self.search_exact_above(query, k, 0.0),
                wand::WandStats::default(),
            );
        }
        let mut cursors = self.cursors_for(query);
        if cursors.is_empty() {
            return (Vec::new(), wand::WandStats::default());
        }
        wand::search_bmw_with_stats(&mut cursors, k)
    }

    fn cursors_for(&self, query: &SparseVec) -> Vec<wand::Cursor<'_>> {
        let mut cursors = Vec::new();
        for &(dim, query_weight) in query.pairs() {
            if let Some(list) = self.postings.get(&dim) {
                cursors.push(wand::Cursor::new(list, query_weight));
            }
        }
        cursors
    }

    #[inline]
    fn can_use_wand(&self, query: &SparseVec) -> bool {
        self.wand_safe && query.wand_safe()
    }

    fn search_exact_above(&self, query: &SparseVec, k: usize, min_score: f32) -> Vec<(u32, f32)> {
        if k == 0 || query.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<u32, f32> = HashMap::new();
        for &(dim, query_weight) in query.pairs() {
            let Some(list) = self.postings.get(&dim) else {
                continue;
            };
            for entry in list.entries() {
                let contribution = entry.weight * query_weight;
                if contribution != 0.0 {
                    *scores.entry(entry.doc_id).or_insert(0.0) += contribution;
                }
            }
        }

        let mut ranked: Vec<_> = scores
            .into_iter()
            .filter(|&(_, score)| score.is_finite() && score > min_score)
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(k);
        ranked
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
    fn search_falls_back_to_exact_for_negative_query_weights() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 10.0), (1, 1.0)]));
        index.insert(1, &SparseVec::new(vec![(1, 1.0)]));
        index.build();

        let query = SparseVec::new(vec![(0, -1.0), (1, 20.0)]);
        let results = index.search(&query, 10);

        assert_eq!(results, vec![(1, 20.0), (0, 10.0)]);
    }

    #[test]
    fn search_falls_back_to_exact_for_negative_document_weights() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, -10.0), (1, 10.0)]));
        index.insert(1, &SparseVec::new(vec![(1, 1.0)]));
        index.build();

        let query = SparseVec::new(vec![(0, 1.0), (1, 1.0)]);
        let results = index.search(&query, 10);

        assert_eq!(results, vec![(1, 1.0)]);
    }

    #[test]
    fn thresholded_search_falls_back_to_exact_for_negative_weights() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 10.0), (1, 1.0)]));
        index.insert(1, &SparseVec::new(vec![(1, 1.0)]));
        index.build();

        let query = SparseVec::new(vec![(0, -1.0), (1, 20.0)]);
        let results = index.search_above(&query, 10, 15.0);

        assert_eq!(results, vec![(1, 20.0)]);
    }

    #[test]
    fn sparse_vec_sorts_folds_duplicates_and_removes_zeros() {
        let sv = SparseVec::new(vec![
            (5, 1.0),
            (2, 0.0),
            (3, 2.0),
            (5, 3.0),
            (1, 1.0),
            (3, -2.0),
        ]);

        assert_eq!(sv.pairs(), &[(1, 1.0), (5, 4.0)]);
    }

    #[test]
    fn sparse_vec_quantizes_raw_impact_document_and_query() {
        let sv = SparseVec::new(vec![(3, 1.25), (1, 0.25), (3, 0.25)]);

        assert_eq!(
            sv.to_raw_impact_document(100.0).unwrap(),
            vec![(1, 25), (3, 150)]
        );
        let query = sv.to_raw_impact_query(100.0).unwrap();
        assert_eq!(query[0].0, 1);
        assert!((query[0].1 - 0.0025).abs() < 1e-8);
        assert_eq!(query[1].0, 3);
        assert!((query[1].1 - 0.015).abs() < 1e-8);
    }

    #[test]
    fn raw_impact_quantizer_carries_scale_for_documents_and_queries() {
        let quantizer = RawImpactQuantizer::new(100.0).unwrap();
        let sv = SparseVec::new(vec![(3, 1.25), (1, 0.25), (3, 0.25)]);

        assert_eq!(quantizer.scale(), 100.0);
        assert_eq!(
            quantizer.document(&sv).unwrap(),
            sv.to_raw_impact_document(100.0).unwrap()
        );
        assert_eq!(
            quantizer.query(&sv).unwrap(),
            sv.to_raw_impact_query(100.0).unwrap()
        );
        assert_eq!(
            RawImpactQuantizer::new(0.0).unwrap_err(),
            RawImpactError::InvalidScale { scale: 0.0 }
        );
    }

    #[test]
    fn raw_impact_quantizer_bounds_query_score_error() {
        let quantizer = RawImpactQuantizer::new(100.0).unwrap();
        let query = SparseVec::new(vec![(1, 1.70), (2, 0.80), (3, 0.40)]);

        let bound = quantizer.score_error_bound(&query).unwrap();
        assert!((bound - 0.0145).abs() < 1e-7);

        assert_eq!(
            quantizer
                .score_error_bound(&SparseVec::new(vec![(1, -1.0)]))
                .unwrap_err(),
            RawImpactError::InvalidWeight {
                dim: 1,
                weight: -1.0,
            }
        );
    }

    #[test]
    fn sparse_vec_raw_impact_quantization_rejects_invalid_inputs() {
        assert_eq!(
            SparseVec::new(vec![(1, 1.0)])
                .to_raw_impact_document(0.0)
                .unwrap_err(),
            RawImpactError::InvalidScale { scale: 0.0 }
        );
        assert_eq!(
            SparseVec::new(vec![(1, -1.0)])
                .to_raw_impact_document(100.0)
                .unwrap_err(),
            RawImpactError::InvalidWeight {
                dim: 1,
                weight: -1.0,
            }
        );
        assert_eq!(
            SparseVec::new(vec![(1, 0.001)])
                .to_raw_impact_document(100.0)
                .unwrap_err(),
            RawImpactError::RoundedToZero {
                dim: 1,
                weight: 0.001,
                scale: 100.0,
            }
        );
        assert_eq!(
            SparseVec::new(vec![(1, u32::MAX as f32)])
                .to_raw_impact_document(2.0)
                .unwrap_err(),
            RawImpactError::WeightOverflow {
                dim: 1,
                weight: u32::MAX as f32,
                scale: 2.0,
            }
        );
        assert!(matches!(
            SparseVec::new(vec![(1, f32::NAN)])
                .to_raw_impact_query(100.0)
                .unwrap_err(),
            RawImpactError::InvalidWeight { dim: 1, weight } if weight.is_nan()
        ));
    }

    #[test]
    fn many_documents_block_boundary() {
        // Test with enough docs to span multiple block-max blocks.
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
    fn thresholded_search_does_not_fill_below_threshold() {
        let mut index = SporseIndex::new();
        for i in 0..6u32 {
            index.insert(i, &SparseVec::new(vec![(0, i as f32 + 1.0)]));
        }
        index.build();

        let query = SparseVec::new(vec![(0, 1.0)]);
        let results = index.search_above(&query, 10, 4.0);

        assert_eq!(results.len(), 2);
        assert_eq!(
            results.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![5, 4]
        );
        assert!(results.iter().all(|(_, score)| *score > 4.0));
    }

    #[test]
    fn query_upper_bound_sums_matching_term_maxima() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 2.0), (1, 3.0)]));
        index.insert(1, &SparseVec::new(vec![(0, 5.0), (2, 7.0)]));
        index.build();

        let query = SparseVec::new(vec![(0, 2.0), (1, 4.0), (99, 100.0)]);

        assert!((index.query_upper_bound(&query) - 22.0).abs() < 1e-5);
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

    #[test]
    fn dot_product() {
        let a = SparseVec::new(vec![(0, 1.0), (2, 3.0), (5, 2.0)]);
        let b = SparseVec::new(vec![(1, 4.0), (2, 2.0), (5, 1.0)]);
        // overlap: dim 2 (3*2=6) + dim 5 (2*1=2) = 8.0
        assert!((a.dot(&b) - 8.0).abs() < 1e-5);

        let c = SparseVec::new(vec![(99, 1.0)]);
        assert!((a.dot(&c)).abs() < 1e-5); // disjoint = 0
    }

    #[test]
    fn randomized_brute_force_parity() {
        // Generate random sparse docs, verify WAND matches brute-force for all queries.
        let mut rng: u64 = 12345;
        let lcg = |state: &mut u64| -> u64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        };

        let n = 200;
        let nnz = 15;
        let max_dim = 500u32;

        // Generate docs.
        let docs: Vec<SparseVec> = (0..n)
            .map(|_| {
                let pairs: Vec<(u32, f32)> = (0..nnz)
                    .map(|_| {
                        let dim = (lcg(&mut rng) >> 33) as u32 % max_dim;
                        let weight = ((lcg(&mut rng) >> 33) as f32 / (1u64 << 31) as f32) + 0.01;
                        (dim, weight)
                    })
                    .collect();
                SparseVec::new(pairs)
            })
            .collect();

        let mut index = SporseIndex::new();
        for (i, doc) in docs.iter().enumerate() {
            index.insert(i as u32, doc);
        }
        index.build();

        // Test with several random queries.
        for _ in 0..10 {
            let query_pairs: Vec<(u32, f32)> = (0..8)
                .map(|_| {
                    let dim = (lcg(&mut rng) >> 33) as u32 % max_dim;
                    let weight = ((lcg(&mut rng) >> 33) as f32 / (1u64 << 31) as f32) + 0.01;
                    (dim, weight)
                })
                .collect();
            let query = SparseVec::new(query_pairs);

            let k = 10;
            let wand_results = index.search(&query, k);

            // Brute-force: compute all dot products and sort.
            let mut bf_scores: Vec<(u32, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, doc)| (i as u32, query.dot(doc)))
                .filter(|&(_, s)| s > 0.0)
                .collect();
            bf_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            bf_scores.truncate(k);

            // WAND must return the same top-k as brute force.
            assert_eq!(
                wand_results.len(),
                bf_scores.len(),
                "result count mismatch: wand={} bf={}",
                wand_results.len(),
                bf_scores.len()
            );
            for (wand, bf) in wand_results.iter().zip(bf_scores.iter()) {
                assert_eq!(wand.0, bf.0, "doc_id mismatch: wand={} bf={}", wand.0, bf.0);
                assert!(
                    (wand.1 - bf.1).abs() < 1e-4,
                    "score mismatch for doc {}: wand={} bf={}",
                    wand.0,
                    wand.1,
                    bf.1
                );
            }
        }
    }

    #[test]
    fn bmw_scores_high_weight_entry_in_a_later_block() {
        // Regression: pivot selection must use GLOBAL term maxima, not the
        // current block's max. With current-block bounds, dim 0's block 0
        // (maxes at 1.0) hides the 100.0 entry in block 1: the pivot lands on
        // the dim-1 cursor at doc 500 and advance_to(500) skips doc 40
        // permanently, returning [(500, 2.0)] instead of [(40, 100.0)].
        let mut index = SporseIndex::new();
        // Block 0 of dim 0: docs 0..32, weight 1.0.
        for i in 0..32u32 {
            index.insert(i, &SparseVec::new(vec![(0, 1.0)]));
        }
        // Block 1 of dim 0 holds the true winner.
        index.insert(40, &SparseVec::new(vec![(0, 100.0)]));
        // Shared tail doc so the dim-1 cursor offers a plausible pivot.
        index.insert(500, &SparseVec::new(vec![(0, 1.0), (1, 1.0)]));
        index.build();

        let query = SparseVec::new(vec![(0, 1.0), (1, 1.0)]);
        let results = index.search(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 40, "true top-1 lives in a later block");
        assert!((results[0].1 - 100.0).abs() < 1e-5);
    }

    #[test]
    fn randomized_brute_force_parity_multiblock() {
        // Same parity property as randomized_brute_force_parity, but with
        // dense posting lists spanning many BLOCK_SIZE blocks and occasional
        // large weight spikes, so block-max pruning and cross-block skips are
        // actually exercised. Guards the WAND exactness claim in the regime
        // real learned-sparse corpora occupy (long, skewed lists).
        let mut rng: u64 = 987654321;
        let lcg = |state: &mut u64| -> u64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        };
        let unit = |state: &mut u64| -> f32 { (lcg(state) >> 33) as f32 / (1u64 << 31) as f32 };

        let n = 400u32;
        let max_dim = 6u32;

        let docs: Vec<SparseVec> = (0..n)
            .map(|i| {
                let mut pairs: Vec<(u32, f32)> = Vec::new();
                for dim in 0..max_dim {
                    // Dim 5 is rare: a sparse cursor whose next doc sits far
                    // ahead, pulling the pivot past dense lists' early blocks.
                    if dim == 5 && i % 89 != 0 {
                        continue;
                    }
                    if dim < 5 && lcg(&mut rng) % 4 == 0 {
                        continue; // drop ~25% of dims
                    }
                    let mut weight = unit(&mut rng) + 0.01;
                    // Spikes cluster in LATE blocks so early block maxima
                    // understate the tail — the regime where a current-block
                    // pivot bound (instead of the global max) loses winners.
                    if i >= 300 && lcg(&mut rng) % 8 == 0 {
                        weight *= 50.0;
                    }
                    pairs.push((dim, weight));
                }
                SparseVec::new(pairs)
            })
            .collect();

        let mut index = SporseIndex::new();
        for (i, doc) in docs.iter().enumerate() {
            index.insert(i as u32, doc);
        }
        index.build();

        for _ in 0..20 {
            let mut query_pairs: Vec<(u32, f32)> = Vec::new();
            for dim in 0..max_dim {
                if lcg(&mut rng) % 2 == 0 {
                    query_pairs.push((dim, unit(&mut rng) + 0.01));
                }
            }
            if query_pairs.is_empty() {
                continue;
            }
            let query = SparseVec::new(query_pairs);

            for k in [1usize, 5, 20] {
                let wand_results = index.search(&query, k);

                let mut bf_scores: Vec<(u32, f32)> = docs
                    .iter()
                    .enumerate()
                    .map(|(i, doc)| (i as u32, query.dot(doc)))
                    .filter(|&(_, s)| s > 0.0)
                    .collect();
                bf_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
                bf_scores.truncate(k);

                assert_eq!(
                    wand_results.len(),
                    bf_scores.len(),
                    "result count mismatch at k={k}: wand={} bf={}",
                    wand_results.len(),
                    bf_scores.len()
                );
                for (wand, bf) in wand_results.iter().zip(bf_scores.iter()) {
                    assert_eq!(
                        wand.0, bf.0,
                        "doc_id mismatch at k={k}: wand={} bf={}",
                        wand.0, bf.0
                    );
                    assert!(
                        (wand.1 - bf.1).abs() < 1e-3,
                        "score mismatch at k={k} for doc {}: wand={} bf={}",
                        wand.0,
                        wand.1,
                        bf.1
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde_index_round_trip() {
        let mut index = SporseIndex::new();
        index.insert(0, &SparseVec::new(vec![(0, 1.0), (3, 2.5)]));
        index.insert(1, &SparseVec::new(vec![(1, 3.0), (3, 1.0)]));
        index.insert(2, &SparseVec::new(vec![(0, 0.5), (3, 0.8)]));
        index.build();

        let query = SparseVec::new(vec![(0, 1.0), (3, 1.0)]);
        let original_results = index.search(&query, 3);

        // Serialize -> deserialize.
        let json = serde_json::to_string(&index).unwrap();
        let loaded: SporseIndex = serde_json::from_str(&json).unwrap();

        // Search results must be identical.
        let loaded_results = loaded.search(&query, 3);
        assert_eq!(original_results.len(), loaded_results.len());
        for (a, b) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde_sparse_vec_round_trip() {
        let sv = SparseVec::new(vec![(10, 2.5), (0, 1.0), (5, 0.3)]);
        let json = serde_json::to_string(&sv).unwrap();
        let loaded: SparseVec = serde_json::from_str(&json).unwrap();
        assert_eq!(sv.pairs(), loaded.pairs());
    }
}

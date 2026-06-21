//! Composite-code sparse-autoencoder codes (CCSA; Lassance, Formal, and
//! Clinchant 2022, arXiv:2204.07023): the encoding and index-integration half.
//!
//! A CCSA encodes a dense vector into `C` chunks of size `L`, with exactly one
//! dimension active per chunk: a C-hot "composite code" over `D = C * L`
//! dimensions. Within each chunk the active dimension is the argmax of that
//! chunk's encoder logits (CCSA trains this with a Gumbel-Softmax
//! straight-through estimator; at inference the hard argmax is what the index
//! sees). The codes drive an inverted index, one posting list per dimension;
//! [`CompositeCode::to_sparse_vec`] adapts a code into the [`SparseVec`] the
//! existing [`SporseIndex`](crate::SporseIndex) already serves.
//!
//! Scope: this is the serving path (given encoder logits from a trained CCSA,
//! produce composite codes and index them). Training the encoder weights
//! (reconstruction + uniformity loss, Gumbel-Softmax straight-through, autodiff)
//! is a separate, heavier concern and is not implemented here.

use crate::SparseVec;

/// Composite-code geometry: `chunks` (C) chunks of `chunk_size` (L) each, so the
/// code lives over `D = C * L` dimensions and is C-hot (one active dim per chunk).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositeCodeConfig {
    /// Number of chunks C.
    pub chunks: usize,
    /// Dimensions per chunk L.
    pub chunk_size: usize,
}

impl CompositeCodeConfig {
    /// Create a config with `chunks` chunks of `chunk_size` dimensions each.
    pub fn new(chunks: usize, chunk_size: usize) -> Self {
        Self { chunks, chunk_size }
    }

    /// Total dimensionality `D = C * L`.
    pub fn dim(&self) -> usize {
        self.chunks * self.chunk_size
    }
}

/// A C-hot composite code: the active index (in `0..L`) for each of the C chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeCode {
    config: CompositeCodeConfig,
    active: Vec<usize>,
}

impl CompositeCode {
    /// Encode per-dimension encoder logits (length `D = C * L`) into a composite
    /// code by taking the argmax within each chunk. Returns `None` if `logits`
    /// has the wrong length or any chunk is empty.
    pub fn from_logits(logits: &[f32], config: CompositeCodeConfig) -> Option<Self> {
        if config.chunk_size == 0 || logits.len() != config.dim() {
            return None;
        }
        let l = config.chunk_size;
        let active = (0..config.chunks)
            .map(|c| argmax(&logits[c * l..(c + 1) * l]))
            .collect();
        Some(Self { config, active })
    }

    /// The active dimension index (`0..L`) for each chunk; length C.
    pub fn active(&self) -> &[usize] {
        &self.active
    }

    /// The code's geometry.
    pub fn config(&self) -> CompositeCodeConfig {
        self.config
    }

    /// The C-hot code as a [`SparseVec`] over the `D`-dimensional code space:
    /// active dim `= chunk * L + active_index`, weight `1.0`. Ready to index.
    pub fn to_sparse_vec(&self) -> SparseVec {
        let l = self.config.chunk_size;
        let pairs = self
            .active
            .iter()
            .enumerate()
            .map(|(c, &a)| ((c * l + a) as u32, 1.0))
            .collect();
        SparseVec::new(pairs)
    }
}

/// Index of the largest element; first on ties. Empty input returns 0 (callers
/// guarantee non-empty chunks via [`CompositeCode::from_logits`]).
fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SporseIndex;

    #[test]
    fn composite_code_is_c_hot() {
        // C = 3 chunks of L = 4; argmax per chunk picks one dim each.
        let cfg = CompositeCodeConfig::new(3, 4);
        // chunk0 argmax @1, chunk1 @3, chunk2 @0.
        let logits = [
            0.0, 9.0, 1.0, 2.0, // chunk 0 -> 1
            1.0, 1.0, 2.0, 5.0, // chunk 1 -> 3
            8.0, 0.0, 0.0, 7.0, // chunk 2 -> 0
        ];
        let code = CompositeCode::from_logits(&logits, cfg).unwrap();
        assert_eq!(code.active(), &[1, 3, 0]);

        // Exactly C active dimensions at the right global positions.
        let sv = code.to_sparse_vec();
        // global dims: 0*4+1=1, 1*4+3=7, 2*4+0=8.
        let mut idx = SporseIndex::new();
        idx.insert(0, &sv);
        idx.build();
        // A query hitting one of the active dims retrieves the doc.
        let q = SparseVec::new(vec![(7, 1.0)]);
        let hits = idx.search(&q, 1);
        assert_eq!(hits.first().map(|(d, _)| *d), Some(0));
    }

    #[test]
    fn rejects_wrong_length() {
        let cfg = CompositeCodeConfig::new(2, 3); // dim = 6
        assert!(CompositeCode::from_logits(&[0.0; 5], cfg).is_none());
        assert!(CompositeCode::from_logits(&[0.0; 6], cfg).is_some());
        assert_eq!(cfg.dim(), 6);
    }

    #[test]
    fn round_trip_through_index_ranks_matching_doc_first() {
        let cfg = CompositeCodeConfig::new(2, 4);
        // doc A: chunks -> [0, 0]; doc B: chunks -> [3, 3].
        let a = CompositeCode::from_logits(&[5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0], cfg).unwrap();
        let b = CompositeCode::from_logits(&[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0], cfg).unwrap();
        let mut idx = SporseIndex::new();
        idx.insert(0, &a.to_sparse_vec());
        idx.insert(1, &b.to_sparse_vec());
        idx.build();
        // Query shaped like A retrieves A above B.
        let hits = idx.search(&a.to_sparse_vec(), 2);
        assert_eq!(hits.first().map(|(d, _)| *d), Some(0));
    }
}

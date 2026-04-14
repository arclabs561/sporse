//! Sparse vector index for learned sparse retrieval.
//!
//! Indexes sparse float vectors (SPLADE, LADE, learned sparse representations)
//! for approximate nearest-neighbor search via inverted index + WAND/MaxScore.

/// A sparse vector: sorted list of (dimension, weight) pairs.
pub type SparseVec = Vec<(u32, f32)>;

//! segstore's shared conformance suite against sporse's updatable index.
#![cfg(feature = "store")]

use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::conformance::{run_all, ConformanceStore};
use sporse::store::UpdatableIndex;
use sporse::SparseVec;

struct Adapter(UpdatableIndex);

/// Eight pseudo-random dimensions out of a million with weights in [1, 2):
/// unrelated vectors almost never share a dimension, so a document's dot
/// product with itself is far above any other document's.
fn vector(seed: u64) -> SparseVec {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let pairs = (0..8)
        .map(|_| {
            (
                (next() % 1_000_000) as u32,
                1.0 + (next() % 1000) as f32 / 1000.0,
            )
        })
        .collect();
    SparseVec::new(pairs)
}

fn self_score(v: &SparseVec) -> f32 {
    v.pairs().iter().map(|&(_, w)| w * w).sum()
}

impl ConformanceStore for Adapter {
    type Doc = SparseVec;

    fn open(dir: Arc<dyn Directory>) -> PersistenceResult<Self> {
        UpdatableIndex::open(dir, 3).map(Self)
    }
    fn add(&mut self, id: u32, doc: SparseVec) -> PersistenceResult<()> {
        self.0.add(id, doc)
    }
    fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.0.delete(id)
    }
    fn compact(&mut self) -> PersistenceResult<()> {
        self.0.compact()
    }
    fn compact_tiers(&mut self) -> PersistenceResult<()> {
        self.0.compact_tiers()
    }
    fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        self.0.reclaim(min_live_ratio)
    }
    fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.0.checkpoint()
    }
    fn doc(seed: u64) -> SparseVec {
        vector(seed)
    }
    /// Ids scoring at least 99% of the document's dot product with itself.
    fn matching(&self, doc: &SparseVec) -> Vec<u32> {
        let threshold = 0.99 * self_score(doc);
        self.0
            .search(doc, 64)
            .into_iter()
            .filter(|&(_, score)| score >= threshold)
            .map(|(id, _)| id)
            .collect()
    }
}

#[test]
fn updatable_index_meets_segstore_conformance() {
    run_all::<Adapter>();
}

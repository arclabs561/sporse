//! Updatable, durable learned-sparse index, backed by `segstore`.
//!
//! Enabled by the optional `store` feature. The base [`SporseIndex`] is
//! build-once; this wraps it in a segstore `SegmentedStore` so documents can be
//! added and deleted incrementally and the index survives a restart (write-ahead
//! log + checkpoint + compaction).
//!
//! Because `SporseIndex` exposes no way to read its source vectors back out, the
//! durable segment payload is the source `(id, SparseVec)` batch; a real
//! `SporseIndex` is built per segment at query time and the per-segment top-k are
//! merged. The merge is exact: every global top-k document is, within its own
//! segment, ranked at or above its global rank, so it appears in that segment's
//! top-k.

use std::cmp::Ordering;
use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::{SparseVec, SporseIndex};

/// The segstore payload model: items are sparse document vectors, a segment is a
/// batch of source vectors (the sporse index is rebuilt from them per query).
struct SparseBacking;

impl Store for SparseBacking {
    type Id = u32;
    type Item = SparseVec;
    type Segment = Vec<(u32, SparseVec)>;

    fn build_segment(&self, batch: &[(u32, SparseVec)]) -> Vec<(u32, SparseVec)> {
        batch.to_vec()
    }

    fn merge_segments(
        &self,
        segs: &[Vec<(u32, SparseVec)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, SparseVec)> {
        segs.iter()
            .flatten()
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }
}

/// An updatable, durable learned-sparse index.
pub struct UpdatableIndex {
    inner: SegmentedStore<SparseBacking>,
}

impl UpdatableIndex {
    /// Open (or recover) an index under `dir`. Up to `flush_threshold` documents
    /// are buffered before a new immutable segment is sealed.
    pub fn open(dir: Arc<dyn Directory>, flush_threshold: usize) -> PersistenceResult<Self> {
        Ok(Self {
            inner: SegmentedStore::open(dir, SparseBacking, flush_threshold)?,
        })
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, vec: SparseVec) -> PersistenceResult<()> {
        self.inner.add(id, vec)
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Top-k documents by Block-Max WAND inner product over the live corpus.
    pub fn search(&self, query: &SparseVec, k: usize) -> Vec<(u32, f32)> {
        let mut cand: Vec<(u32, f32)> = Vec::new();
        for seg in self.inner.segments() {
            cand.extend(self.search_batch(seg, query, k));
        }
        let buffered = self.inner.buffer().to_vec();
        cand.extend(self.search_batch(&buffered, query, k));
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        cand.truncate(k);
        cand
    }

    /// Build a real `SporseIndex` over the live docs of one batch and run WAND.
    fn search_batch(
        &self,
        batch: &[(u32, SparseVec)],
        query: &SparseVec,
        k: usize,
    ) -> Vec<(u32, f32)> {
        let mut idx = SporseIndex::new();
        let mut any = false;
        for (id, v) in batch {
            if self.inner.is_live(id) {
                idx.insert(*id, v);
                any = true;
            }
        }
        if !any {
            return Vec::new();
        }
        idx.build();
        idx.search(query, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;

    #[test]
    fn add_delete_compact_recover_through_real_wand() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2).unwrap();
            store
                .add(0, SparseVec::new(vec![(0, 1.0), (3, 2.5), (7, 0.8)]))
                .unwrap();
            store
                .add(1, SparseVec::new(vec![(1, 3.0), (3, 1.0)]))
                .unwrap();
            store
                .add(2, SparseVec::new(vec![(0, 0.5), (7, 2.0)]))
                .unwrap();

            let q = SparseVec::new(vec![(0, 1.0), (3, 1.0)]);
            let top: Vec<u32> = store.search(&q, 3).into_iter().map(|(id, _)| id).collect();
            assert_eq!(
                top,
                vec![0, 1, 2],
                "WAND ranks by inner product across segments"
            );

            store.delete(0).unwrap();
            let top: Vec<u32> = store.search(&q, 3).into_iter().map(|(id, _)| id).collect();
            assert_eq!(top, vec![1, 2], "delete removes doc 0");

            store.compact().unwrap();
            let top: Vec<u32> = store.search(&q, 3).into_iter().map(|(id, _)| id).collect();
            assert_eq!(top, vec![1, 2], "compaction preserves results");
        }
        // Reopen and re-query through recovery.
        let store = UpdatableIndex::open(dir, 2).unwrap();
        let q = SparseVec::new(vec![(0, 1.0), (3, 1.0)]);
        let top: Vec<u32> = store.search(&q, 3).into_iter().map(|(id, _)| id).collect();
        assert_eq!(top, vec![1, 2], "recovery preserves results");
    }
}

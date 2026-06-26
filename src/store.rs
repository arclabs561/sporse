//! Updatable, durable learned-sparse index, backed by `segstore`.
//!
//! Enabled by the optional `store` feature. The base [`SporseIndex`] is
//! build-once; this wraps it in a segstore `SegmentedStore` so documents can be
//! added and deleted incrementally and the index survives a restart (write-ahead
//! log + checkpoint + compaction).
//!
//! `SporseIndex` exposes no way to read its source vectors back out, so the
//! durable segment payload is the source `(id, SparseVec)` batch. A real
//! `SporseIndex` is built per segment and **cached**; it is rebuilt only when the
//! index is mutated (an add that seals a segment, a delete, or a compaction),
//! not on every query. The small unflushed buffer is built per query. Each
//! segment index is built over its *live* documents, so deletes drop out on the
//! next rebuild without any over-fetch.
//!
//! Per-segment top-k are merged; the merge is exact given exact per-segment
//! results (Block-Max WAND is exact), since every global top-k document ranks at
//! or above its global rank within its own segment.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::{SparseVec, SporseIndex};

/// The segstore payload model: items are sparse document vectors, a segment is a
/// batch of source vectors (a `SporseIndex` is built + cached from them).
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

/// Cached per-segment indexes, valid for a given mutation generation.
struct Cache {
    generation: u64,
    segments: Vec<Option<SporseIndex>>,
}

/// An updatable, durable learned-sparse index.
pub struct UpdatableIndex {
    inner: SegmentedStore<SparseBacking>,
    /// Bumped on every mutation; invalidates the cache.
    generation: u64,
    cache: RefCell<Cache>,
}

impl UpdatableIndex {
    /// Open (or recover) an index under `dir`. Up to `flush_threshold` documents
    /// are buffered before a new immutable segment is sealed.
    pub fn open(dir: Arc<dyn Directory>, flush_threshold: usize) -> PersistenceResult<Self> {
        Ok(Self {
            inner: SegmentedStore::open(dir, SparseBacking, flush_threshold)?,
            generation: 0,
            // A generation the first query cannot match forces an initial build.
            cache: RefCell::new(Cache {
                generation: u64::MAX,
                segments: Vec::new(),
            }),
        })
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, vec: SparseVec) -> PersistenceResult<()> {
        self.inner.add(id, vec)?;
        self.generation += 1;
        Ok(())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        self.generation += 1;
        Ok(())
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        self.generation += 1;
        Ok(())
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Top-k documents by Block-Max WAND inner product over the live corpus.
    pub fn search(&self, query: &SparseVec, k: usize) -> Vec<(u32, f32)> {
        self.refresh_cache();
        let mut cand: Vec<(u32, f32)> = Vec::new();
        {
            let cache = self.cache.borrow();
            for idx in cache.segments.iter().flatten() {
                cand.extend(idx.search(query, k));
            }
        }
        // The unflushed buffer is bounded by the flush threshold; build it fresh.
        let buffered: Vec<(u32, SparseVec)> = self.inner.buffer().to_vec();
        if let Some(idx) = self.build_live_index(&buffered) {
            cand.extend(idx.search(query, k));
        }
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        cand.truncate(k);
        cand
    }

    /// Rebuild the per-segment cache iff a mutation has occurred since it was built.
    fn refresh_cache(&self) {
        let mut cache = self.cache.borrow_mut();
        if cache.generation == self.generation {
            return;
        }
        cache.segments.clear();
        for seg in self.inner.segments() {
            cache.segments.push(self.build_live_index(seg));
        }
        cache.generation = self.generation;
    }

    /// Build a `SporseIndex` over the live documents of `items` (None if empty).
    fn build_live_index(&self, items: &[(u32, SparseVec)]) -> Option<SporseIndex> {
        let mut idx = SporseIndex::new();
        let mut any = false;
        for (id, v) in items {
            if self.inner.is_live(id) {
                idx.insert(*id, v);
                any = true;
            }
        }
        if !any {
            return None;
        }
        idx.build();
        Some(idx)
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
            // Second query (no mutation) must use the cache and stay correct.
            let again: Vec<u32> = store.search(&q, 3).into_iter().map(|(id, _)| id).collect();
            assert_eq!(again, vec![0, 1, 2], "cached query is stable");

            store.delete(0).unwrap();
            let top: Vec<u32> = store.search(&q, 3).into_iter().map(|(id, _)| id).collect();
            assert_eq!(
                top,
                vec![1, 2],
                "delete invalidates the cache; doc 0 is gone"
            );

            store.compact().unwrap();
            let top: Vec<u32> = store.search(&q, 3).into_iter().map(|(id, _)| id).collect();
            assert_eq!(top, vec![1, 2], "compaction preserves results");
        }
        let store = UpdatableIndex::open(dir, 2).unwrap();
        let q = SparseVec::new(vec![(0, 1.0), (3, 1.0)]);
        let top: Vec<u32> = store.search(&q, 3).into_iter().map(|(id, _)| id).collect();
        assert_eq!(top, vec![1, 2], "recovery preserves results");
    }
}

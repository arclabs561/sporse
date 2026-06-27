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
use std::collections::HashMap;
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

    fn segment_len(&self, seg: &Vec<(u32, SparseVec)>) -> usize {
        seg.len()
    }

    fn live_len(&self, seg: &Vec<(u32, SparseVec)>, live: &dyn Fn(&u32) -> bool) -> Option<usize> {
        Some(seg.iter().filter(|(id, _)| live(id)).count())
    }
}

/// Per-segment indexes keyed by the segment's stable `Arc` identity. Because
/// segstore keeps an unchanged segment's `Arc` across mutations, a sealed add
/// only builds the one new segment's index (the rest are reused) instead of
/// rebuilding the whole corpus -- the dominant cost in an add-then-search loop.
struct Cache {
    by_ptr: HashMap<usize, Option<SporseIndex>>,
}

/// An updatable, durable learned-sparse index.
pub struct UpdatableIndex {
    inner: SegmentedStore<SparseBacking>,
    cache: RefCell<Cache>,
}

impl UpdatableIndex {
    /// Open (or recover) an index under `dir`. Up to `flush_threshold` documents
    /// are buffered before a new immutable segment is sealed.
    pub fn open(dir: Arc<dyn Directory>, flush_threshold: usize) -> PersistenceResult<Self> {
        Ok(Self {
            inner: SegmentedStore::open(dir, SparseBacking, flush_threshold)?,
            cache: RefCell::new(Cache {
                by_ptr: HashMap::new(),
            }),
        })
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, vec: SparseVec) -> PersistenceResult<()> {
        // A sealed add introduces a new segment (a new Arc identity); existing
        // segments keep theirs, so the cache reuses them and builds only the new one.
        self.inner.add(id, vec)?;
        Ok(())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        // A tombstone only changes the live-set of the segment that holds `id`, so
        // invalidate just that segment's cached index -- not the whole cache. The
        // other segments' indexes stay valid and are reused on the next query.
        let mut cache = self.cache.borrow_mut();
        for seg in self.inner.segments() {
            if seg.iter().any(|(sid, _)| *sid == id) {
                cache.by_ptr.remove(&(Arc::as_ptr(seg) as usize));
            }
        }
        Ok(())
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        Ok(())
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Run one round of size-tiered compaction, merging similarly-sized segments
    /// so the segment count stays bounded without a full [`compact`](Self::compact).
    pub fn compact_tiers(&mut self) -> PersistenceResult<()> {
        self.inner.compact_tiers()?;
        Ok(())
    }

    /// Merge only the segments whose live ratio is below `min_live_ratio`,
    /// reclaiming tombstoned documents -- the cheap alternative to a full
    /// [`compact`](Self::compact) when a few segments are delete-heavy.
    pub fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        self.inner.reclaim_tombstones(min_live_ratio)?;
        Ok(())
    }

    /// Storage amplification: stored documents divided by live documents (`1.0`
    /// when there are no tombstones, higher as deletes accumulate).
    pub fn space_amplification(&self) -> Option<f64> {
        self.inner.space_amplification()
    }

    /// Top-k documents by Block-Max WAND inner product over the live corpus.
    pub fn search(&self, query: &SparseVec, k: usize) -> Vec<(u32, f32)> {
        let mut cand: Vec<(u32, f32)> = Vec::new();
        {
            let segs = self.inner.segments();
            let mut cache = self.cache.borrow_mut();
            // Drop cached indexes for segments no longer present (post-compaction).
            let current: std::collections::HashSet<usize> =
                segs.iter().map(|a| Arc::as_ptr(a) as usize).collect();
            cache.by_ptr.retain(|key, _| current.contains(key));
            // Build only segments not already cached (i.e. new ones).
            for seg in segs {
                let key = Arc::as_ptr(seg) as usize;
                cache
                    .by_ptr
                    .entry(key)
                    .or_insert_with(|| self.build_live_index(&seg[..]));
            }
            for idx in cache.by_ptr.values().flatten() {
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

    #[test]
    fn reclaim_drops_tombstone_heavy_segments() {
        let dir = MemoryDirectory::arc();
        let mut store = UpdatableIndex::open(dir, 2).unwrap();
        for i in 0..4u32 {
            store
                .add(i, SparseVec::new(vec![(0, 1.0 + i as f32)]))
                .unwrap();
        }
        store.checkpoint().unwrap();
        store.delete(0).unwrap();
        store.delete(1).unwrap();
        store.delete(2).unwrap();

        let before = store.space_amplification().unwrap();
        assert!(before > 1.0, "tombstones inflate stored/live: {before}");
        store.reclaim(0.9).unwrap();
        let after = store.space_amplification().unwrap();
        assert!(
            after <= before,
            "reclaim must not grow space amp: {before} -> {after}"
        );

        // The one surviving doc is still searchable.
        let q = SparseVec::new(vec![(0, 1.0)]);
        let top: Vec<u32> = store.search(&q, 5).into_iter().map(|(id, _)| id).collect();
        assert_eq!(top, vec![3], "only the live doc remains after reclaim");
    }

    #[test]
    fn delete_invalidates_only_the_holding_segment() {
        let dir = MemoryDirectory::arc();
        let mut store = UpdatableIndex::open(dir, 2).unwrap();
        // 4 docs at flush_threshold 2 -> two sealed segments {0,1} and {2,3}.
        for i in 0..4u32 {
            store
                .add(i, SparseVec::new(vec![(0, 1.0 + i as f32)]))
                .unwrap();
        }
        store.checkpoint().unwrap();

        // Populate the cache.
        let q = SparseVec::new(vec![(0, 1.0)]);
        let _ = store.search(&q, 4);
        assert_eq!(store.cache.borrow().by_ptr.len(), 2, "both segments cached");

        // Pointer of the segment that holds id 0.
        let holder = store
            .inner
            .segments()
            .iter()
            .find(|s| s.iter().any(|(id, _)| *id == 0))
            .map(|s| Arc::as_ptr(s) as usize)
            .unwrap();

        store.delete(0).unwrap();

        let keys: std::collections::HashSet<usize> =
            store.cache.borrow().by_ptr.keys().copied().collect();
        assert!(!keys.contains(&holder), "holding segment was invalidated");
        assert_eq!(keys.len(), 1, "the other segment's cache was preserved");

        // Search remains correct: doc 0 is gone, 1/2/3 remain.
        let top: Vec<u32> = store.search(&q, 4).into_iter().map(|(id, _)| id).collect();
        assert!(
            !top.contains(&0) && top.contains(&1),
            "delete correct: {top:?}"
        );
    }
}

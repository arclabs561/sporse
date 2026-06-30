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
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::{posting::BLOCK_SIZE, SparseVec, SporseIndex};

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
        segs: &[&Vec<(u32, SparseVec)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, SparseVec)> {
        segs.iter()
            .flat_map(|s| s.iter())
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

/// The `kind` tag for a persisted per-segment WAND sidecar.
const INDEX_KIND: &str = "wand";
const SIDECAR_MAGIC: &[u8; 8] = b"SPORIDX1";
const SIDECAR_VERSION: u32 = 1;

#[derive(serde::Serialize, serde::Deserialize)]
struct SporseSidecar {
    index: SporseIndex,
    ids: Vec<u32>,
}

/// An updatable, durable learned-sparse index.
pub struct UpdatableIndex {
    inner: SegmentedStore<SparseBacking>,
    sidecar_recipe: String,
    cache: RefCell<Cache>,
    /// Segment ids whose on-disk WAND sidecar was validated or written in this
    /// process, so checkpoint persistence stays O(new segments).
    persisted: RefCell<HashSet<u64>>,
}

impl UpdatableIndex {
    /// Open (or recover) an index under `dir`. Up to `flush_threshold` documents
    /// are buffered before a new immutable segment is sealed.
    pub fn open(dir: Arc<dyn Directory>, flush_threshold: usize) -> PersistenceResult<Self> {
        Ok(Self {
            inner: SegmentedStore::open(dir, SparseBacking, flush_threshold)?,
            sidecar_recipe: Self::make_sidecar_recipe(),
            cache: RefCell::new(Cache {
                by_ptr: HashMap::new(),
            }),
            persisted: RefCell::new(HashSet::new()),
        })
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, vec: SparseVec) -> PersistenceResult<()> {
        // A sealed add introduces a new segment (a new Arc identity); existing
        // segments keep theirs, so the cache reuses them and builds only the new one.
        self.inner.add(id, vec)?;
        Ok(())
    }

    /// Add (or re-add) many documents, syncing the write-ahead log once for the
    /// whole batch instead of once per document. This is the bulk-ingest path (the
    /// corpus-load phase): per-item WAL sync is the dominant cost on a real disk, so
    /// one sync per batch is several times faster than a loop of [`Self::add`]. A
    /// crash mid-batch recovers a consistent prefix (each document is an
    /// independently CRC-checked WAL record).
    pub fn extend(
        &mut self,
        docs: impl IntoIterator<Item = (u32, SparseVec)>,
    ) -> PersistenceResult<()> {
        self.inner.extend(docs)?;
        Ok(())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        // A tombstone only changes the live-set of the segment that holds `id`, so
        // invalidate just that segment's cached index -- not the whole cache --
        // and remove its now-stale sidecar. The live-id guard would reject it
        // anyway; deleting avoids a wasted load on the next query.
        let mut cache = self.cache.borrow_mut();
        let ids = self.inner.segment_ids();
        for (seg_idx, seg) in self.inner.segments().iter().enumerate() {
            if seg.iter().any(|(sid, _)| *sid == id) {
                cache.by_ptr.remove(&(Arc::as_ptr(seg) as usize));
                let seg_id = ids[seg_idx];
                self.persisted.borrow_mut().remove(&seg_id);
                let _ = self
                    .inner
                    .dir()
                    .delete(&self.inner.index_name(seg_id, INDEX_KIND));
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
        self.inner.checkpoint()?;
        self.persist_new_segments();
        Ok(())
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
            // Build only segments not already cached, loading a persisted sidecar
            // first when one matches the current recipe and live id set.
            let ids = self.inner.segment_ids();
            for (i, seg) in segs.iter().enumerate() {
                let key = Arc::as_ptr(seg) as usize;
                let seg_id = ids[i];
                cache
                    .by_ptr
                    .entry(key)
                    .or_insert_with(|| self.build_or_load(&seg[..], seg_id));
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

    /// Load segment `seg_id`'s persisted WAND index sidecar, or build it over the
    /// segment's live vectors and persist it for the next restart.
    fn build_or_load(&self, seg: &[(u32, SparseVec)], seg_id: u64) -> Option<SporseIndex> {
        if let Some(idx) = self.load_sidecar(seg, seg_id) {
            self.persisted.borrow_mut().insert(seg_id);
            return Some(idx);
        }
        let idx = self.build_live_index(seg)?;
        self.persist_sidecar(&idx, seg, seg_id);
        Some(idx)
    }

    /// Load a sidecar only if its recipe matches and its ids match the segment's
    /// current live ids. A stale sidecar can never serve a tombstoned document.
    fn load_sidecar(&self, seg: &[(u32, SparseVec)], seg_id: u64) -> Option<SporseIndex> {
        let name = self.inner.index_name(seg_id, INDEX_KIND);
        if !self.inner.dir().exists(&name) {
            return None;
        }
        let mut bytes = Vec::new();
        self.inner
            .dir()
            .open_file(&name)
            .ok()?
            .read_to_end(&mut bytes)
            .ok()?;
        let index_bytes = self.decode_sidecar(&bytes)?;
        let sidecar: SporseSidecar = postcard::from_bytes(index_bytes).ok()?;
        if !sidecar.index.built {
            return None;
        }
        let live = self.live_ids(seg);
        if sidecar.ids.len() == live.len() && sidecar.ids.iter().all(|id| live.contains(id)) {
            Some(sidecar.index)
        } else {
            None
        }
    }

    /// Persist a built per-segment WAND index as its sidecar. Best-effort: a
    /// failed write leaves the in-memory index usable and simply rebuilds later.
    fn persist_sidecar(&self, idx: &SporseIndex, seg: &[(u32, SparseVec)], seg_id: u64) {
        let sidecar = SporseSidecar {
            index: idx.clone(),
            ids: self.live_ids_vec(seg),
        };
        if let Ok(index) = postcard::to_allocvec(&sidecar) {
            let Some(bytes) = self.encode_sidecar(&index) else {
                return;
            };
            if self
                .inner
                .dir()
                .atomic_write(&self.inner.index_name(seg_id, INDEX_KIND), &bytes)
                .is_ok()
            {
                self.persisted.borrow_mut().insert(seg_id);
            }
        }
    }

    fn live_ids(&self, seg: &[(u32, SparseVec)]) -> HashSet<u32> {
        seg.iter()
            .filter_map(|(id, _)| self.inner.is_live(id).then_some(*id))
            .collect()
    }

    fn live_ids_vec(&self, seg: &[(u32, SparseVec)]) -> Vec<u32> {
        let mut ids: Vec<u32> = seg
            .iter()
            .filter_map(|(id, _)| self.inner.is_live(id).then_some(*id))
            .collect();
        ids.sort_unstable();
        ids
    }

    fn make_sidecar_recipe() -> String {
        format!(
            "sporse-store-wand-v1;\
             codec=postcard-sporse-index-v1;\
             block_size={};score=inner-product-f32;weights=nonnegative",
            BLOCK_SIZE
        )
    }

    fn encode_sidecar(&self, index: &[u8]) -> Option<Vec<u8>> {
        let recipe = self.sidecar_recipe.as_bytes();
        let recipe_len = u32::try_from(recipe.len()).ok()?;
        let mut bytes = Vec::with_capacity(16 + recipe.len() + index.len());
        bytes.extend_from_slice(SIDECAR_MAGIC);
        bytes.extend_from_slice(&SIDECAR_VERSION.to_le_bytes());
        bytes.extend_from_slice(&recipe_len.to_le_bytes());
        bytes.extend_from_slice(recipe);
        bytes.extend_from_slice(index);
        Some(bytes)
    }

    fn decode_sidecar<'a>(&self, bytes: &'a [u8]) -> Option<&'a [u8]> {
        if bytes.len() < 16 {
            return None;
        }
        if &bytes[..8] != SIDECAR_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        if version != SIDECAR_VERSION {
            return None;
        }
        let recipe_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
        let recipe_start = 16usize;
        let recipe_end = recipe_start.checked_add(recipe_len)?;
        if bytes.len() < recipe_end {
            return None;
        }
        if &bytes[recipe_start..recipe_end] != self.sidecar_recipe.as_bytes() {
            return None;
        }
        Some(&bytes[recipe_end..])
    }

    /// Persist sidecars for sealed segments that lack a current one. This is
    /// incremental: already validated/written segment ids are skipped.
    fn persist_new_segments(&self) {
        let ids = self.inner.segment_ids();
        let id_set: HashSet<u64> = ids.iter().copied().collect();
        self.persisted.borrow_mut().retain(|id| id_set.contains(id));
        for (i, seg) in self.inner.segments().iter().enumerate() {
            let seg_id = ids[i];
            if self.persisted.borrow().contains(&seg_id) {
                continue;
            }
            if self.load_sidecar(&seg[..], seg_id).is_some() {
                self.persisted.borrow_mut().insert(seg_id);
                continue;
            }
            if let Some(idx) = self.build_live_index(&seg[..]) {
                self.persist_sidecar(&idx, &seg[..], seg_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;

    fn sv(pairs: &[(u32, f32)]) -> SparseVec {
        SparseVec::new(pairs.to_vec())
    }

    fn read_file(dir: &Arc<dyn Directory>, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        dir.open_file(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    fn checkpointed_store(dir: Arc<dyn Directory>) -> (String, Vec<u8>) {
        let mut store = UpdatableIndex::open(dir, 2).unwrap();
        store.add(0, sv(&[(0, 3.0), (3, 1.0)])).unwrap();
        store.add(1, sv(&[(0, 2.0), (4, 1.0)])).unwrap();
        store.add(2, sv(&[(1, 4.0)])).unwrap();
        store.add(3, sv(&[(0, 1.0), (1, 1.0)])).unwrap();
        store.checkpoint().unwrap();
        let seg_id = store.inner.segment_ids()[0];
        let name = store.inner.index_name(seg_id, INDEX_KIND);
        let bytes = read_file(store.inner.dir(), &name);
        (name, bytes)
    }

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

    #[test]
    fn checkpoint_persists_sidecars_and_reopen_loads_them() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2).unwrap();
            store.add(0, sv(&[(0, 3.0), (3, 1.0)])).unwrap();
            store.add(1, sv(&[(0, 2.0), (4, 1.0)])).unwrap();
            store.add(2, sv(&[(1, 4.0)])).unwrap();
            store.add(3, sv(&[(0, 1.0), (1, 1.0)])).unwrap();
            store.checkpoint().unwrap();

            let ids: Vec<u64> = store.inner.segment_ids().to_vec();
            assert!(
                !ids.is_empty(),
                "4 docs at flush 2 seal at least one segment"
            );
            for id in &ids {
                assert!(
                    store
                        .inner
                        .dir()
                        .exists(&store.inner.index_name(*id, INDEX_KIND)),
                    "segment {id} must have a persisted sidecar after checkpoint"
                );
            }
        }

        let store = UpdatableIndex::open(dir, 2).unwrap();
        let q = sv(&[(0, 1.0)]);
        assert_eq!(
            store
                .search(&q, 2)
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>(),
            vec![0, 1],
            "search over loaded sidecars returns the WAND ranking"
        );
    }

    #[test]
    fn wand_sidecar_envelope_rejects_corrupt_headers() {
        let store = UpdatableIndex::open(MemoryDirectory::arc(), 2).unwrap();
        let index = b"index-bytes";
        let bytes = store.encode_sidecar(index).unwrap();
        assert_eq!(store.decode_sidecar(&bytes), Some(index.as_slice()));

        assert!(store.decode_sidecar(&bytes[..8]).is_none());

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        assert!(store.decode_sidecar(&bad_magic).is_none());

        let mut bad_version = bytes.clone();
        bad_version[8..12].copy_from_slice(&(SIDECAR_VERSION + 1).to_le_bytes());
        assert!(store.decode_sidecar(&bad_version).is_none());

        let mut bad_recipe_len = bytes.clone();
        bad_recipe_len[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(store.decode_sidecar(&bad_recipe_len).is_none());

        let mut bad_recipe = bytes.clone();
        bad_recipe[16] ^= 0x01;
        assert!(store.decode_sidecar(&bad_recipe).is_none());
    }

    #[test]
    fn wand_sidecar_invalid_payload_rebuilds() {
        let dir = MemoryDirectory::arc();
        let (name, _) = checkpointed_store(dir.clone());
        {
            let store = UpdatableIndex::open(dir.clone(), 2).unwrap();
            let corrupt = store
                .encode_sidecar(b"not-a-postcard-sporse-index")
                .unwrap();
            store.inner.dir().atomic_write(&name, &corrupt).unwrap();
        }

        let store = UpdatableIndex::open(dir.clone(), 2).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "valid envelope with invalid WAND payload is rejected"
        );
        let q = sv(&[(0, 1.0)]);
        assert!(
            !store.search(&q, 3).is_empty(),
            "invalid payload falls back to rebuild"
        );
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "rebuilt sidecar loads after the fallback"
        );
    }

    #[test]
    fn deleted_id_does_not_resurface_through_a_sidecar() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2).unwrap();
            store.add(0, sv(&[(0, 3.0), (3, 1.0)])).unwrap();
            store.add(1, sv(&[(0, 2.0), (4, 1.0)])).unwrap();
            store.add(2, sv(&[(1, 4.0)])).unwrap();
            store.checkpoint().unwrap();
            store.delete(0).unwrap();
            store.checkpoint().unwrap();
        }

        let store = UpdatableIndex::open(dir, 2).unwrap();
        let q = sv(&[(0, 1.0)]);
        let top: Vec<u32> = store.search(&q, 3).into_iter().map(|(id, _)| id).collect();
        assert!(
            !top.contains(&0),
            "deleted id 0 must not resurface from a persisted sidecar"
        );
        assert!(
            top.contains(&1),
            "nearest live doc should remain searchable"
        );
    }

    #[test]
    fn checkpoint_after_replayed_delete_rewrites_stale_sidecar() {
        let dir = MemoryDirectory::arc();
        let (name, stale_bytes) = {
            let mut store = UpdatableIndex::open(dir.clone(), 2).unwrap();
            store.add(0, sv(&[(0, 3.0), (3, 1.0)])).unwrap();
            store.add(1, sv(&[(0, 2.0), (4, 1.0)])).unwrap();
            store.add(2, sv(&[(1, 4.0)])).unwrap();
            store.checkpoint().unwrap();

            let seg_id = store.inner.segment_ids()[0];
            let name = store.inner.index_name(seg_id, INDEX_KIND);
            let bytes = read_file(store.inner.dir(), &name);

            // Simulate a crash after the delete is durably logged but before
            // `UpdatableIndex::delete` removes the now-stale sidecar.
            store.inner.delete(0).unwrap();
            (name, bytes)
        };

        let mut store = UpdatableIndex::open(dir.clone(), 2).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "replayed tombstone must make the old sidecar stale"
        );

        store.checkpoint().unwrap();

        let rewritten = read_file(&dir, &name);
        assert_ne!(
            rewritten, stale_bytes,
            "checkpoint should rewrite stale sidecars even before search"
        );
        let idx = store
            .load_sidecar(&store.inner.segments()[0][..], seg_id)
            .expect("rewritten sidecar should be valid");
        let q = sv(&[(0, 1.0)]);
        let top: Vec<u32> = idx.search(&q, 3).into_iter().map(|(id, _)| id).collect();
        assert!(
            !top.contains(&0),
            "rewritten sidecar must exclude the replayed delete"
        );
        assert!(
            top.contains(&1),
            "rewritten sidecar should keep live ids from the segment"
        );
    }
}

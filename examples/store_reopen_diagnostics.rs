//! Measure sidecar-first snapshot reopen vs rebuild for the segstore-backed store.
//!
//! Run:
//! `cargo run --release --features store --example store_reopen_diagnostics`

use std::sync::Arc;
use std::time::{Duration, Instant};

use durability::{Directory, MemoryDirectory};
use sporse::{
    store::{SnapshotIndex, StoreSearchStats, UpdatableIndex},
    SparseVec,
};

const SEGMENTS: usize = 5;
const DOCS_PER_SEGMENT: usize = 200;
const SEGMENT_VOCAB: u32 = 512;
const DOC_NNZ: usize = 24;
const QUERY_NNZ: usize = 16;
const TARGET_SEGMENT: usize = 0;
const TARGET_ORDINAL: usize = 37;
const TARGET_ID: u32 = (TARGET_SEGMENT * DOCS_PER_SEGMENT + TARGET_ORDINAL) as u32;
const TOP_K: usize = 10;

type DynError = Box<dyn std::error::Error>;
type StoreDir = Arc<dyn Directory>;
type SearchHits = Vec<(u32, f32)>;
type SnapshotSearch = (Duration, SearchHits, StoreSearchStats);

fn main() -> Result<(), DynError> {
    let (load_dir, query) = build_checkpointed_dir()?;
    let load_sidecars = sidecar_count(&load_dir)?;
    let (load_elapsed, load_hits, load_stats) = first_snapshot_query(load_dir.clone(), &query)?;

    let (rebuild_dir, rebuild_query) = build_checkpointed_dir()?;
    let sidecars_before_delete = sidecar_count(&rebuild_dir)?;
    delete_sidecars(&rebuild_dir)?;
    let sidecars_after_delete = sidecar_count(&rebuild_dir)?;
    let (rebuild_elapsed, rebuild_hits, rebuild_stats) =
        first_snapshot_query(rebuild_dir, &rebuild_query)?;

    assert_eq!(load_hits, rebuild_hits);
    assert_eq!(load_hits.first().map(|(id, _)| *id), Some(TARGET_ID));

    println!(
        "documents: {}, segments: {SEGMENTS}, flush threshold: {DOCS_PER_SEGMENT}",
        SEGMENTS * DOCS_PER_SEGMENT
    );
    println!("sidecars loaded path: {load_sidecars}");
    println!(
        "sidecars rebuild path before/after delete: {sidecars_before_delete}/{sidecars_after_delete}"
    );
    println!(
        "first snapshot query with sidecars: {}",
        micros(load_elapsed)
    );
    println!(
        "first snapshot query after deleting sidecars: {}",
        micros(rebuild_elapsed)
    );
    println!(
        "with sidecars searched/pruned segments: {}/{}",
        load_stats.searched_segments, load_stats.pruned_segments
    );
    println!(
        "after rebuild searched/pruned segments: {}/{}",
        rebuild_stats.searched_segments, rebuild_stats.pruned_segments
    );
    println!("top hit: {:?}", load_hits.first());

    Ok(())
}

fn build_checkpointed_dir() -> Result<(StoreDir, SparseVec), DynError> {
    let dir: StoreDir = MemoryDirectory::arc();
    let mut index = UpdatableIndex::open(dir.clone(), DOCS_PER_SEGMENT)?;
    let mut state = 0x1234_5678_9abc_def0u64;

    for segment in 0..SEGMENTS {
        for ordinal in 0..DOCS_PER_SEGMENT {
            let id = (segment * DOCS_PER_SEGMENT + ordinal) as u32;
            index.add(id, document(&mut state, segment, ordinal))?;
        }
    }
    index.checkpoint()?;
    Ok((dir, query()))
}

fn first_snapshot_query(dir: StoreDir, query: &SparseVec) -> Result<SnapshotSearch, DynError> {
    let snapshot = SnapshotIndex::open(dir)?;
    let start = Instant::now();
    let (hits, stats) = snapshot.search_with_stats(query, TOP_K)?;
    Ok((start.elapsed(), hits, stats))
}

fn delete_sidecars(dir: &StoreDir) -> Result<(), DynError> {
    for name in dir.list_dir("")? {
        if name.starts_with("segstore.idx.") {
            dir.delete(&name)?;
        }
    }
    Ok(())
}

fn sidecar_count(dir: &StoreDir) -> Result<usize, DynError> {
    Ok(dir
        .list_dir("")?
        .into_iter()
        .filter(|name| name.starts_with("segstore.idx."))
        .count())
}

fn document(state: &mut u64, segment: usize, ordinal: usize) -> SparseVec {
    let base = segment as u32 * SEGMENT_VOCAB;
    let mut pairs: Vec<(u32, f32)> = (0..DOC_NNZ)
        .map(|_| {
            (
                base + (xorshift(state) % SEGMENT_VOCAB as u64) as u32,
                1.0 + (xorshift(state) % 8) as f32,
            )
        })
        .collect();

    if segment == TARGET_SEGMENT && ordinal == TARGET_ORDINAL {
        pairs.extend((0..QUERY_NNZ).map(|dim| (dim as u32, 20.0)));
    }

    SparseVec::new(pairs)
}

fn query() -> SparseVec {
    SparseVec::new((0..QUERY_NNZ).map(|dim| (dim as u32, 1.0)).collect())
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn micros(duration: Duration) -> String {
    format!("{} us", duration.as_micros())
}

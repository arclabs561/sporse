//! Benchmarks for the `store` feature (segstore-backed updatable index).
//!
//! Run: `cargo bench --features store --bench store`. Without the feature the
//! harness is an empty no-op so the bench target still compiles.
//!
//! Measures the costs that matter for an updatable WAND index: build throughput,
//! warm query latency (per-segment indexes cached), checkpoint-visible reader
//! query latency, cold restart latency with persisted sidecars, and the cold
//! rebuild cost when sidecars are missing or stale.

#[cfg(not(feature = "store"))]
fn main() {}

#[cfg(feature = "store")]
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};

#[cfg(feature = "store")]
const N_DOCS: usize = 20_000;
#[cfg(feature = "store")]
const VOCAB: u32 = 30_000;
#[cfg(feature = "store")]
const DOC_NNZ: usize = 80;
#[cfg(feature = "store")]
const FLUSH: usize = 2_000; // ~10 segments
#[cfg(feature = "store")]
const DIAG_SEGMENTS: usize = 10;
#[cfg(feature = "store")]
const DIAG_DOCS_PER_SEGMENT: usize = 200;
#[cfg(feature = "store")]
const DIAG_SEGMENT_VOCAB: u32 = 512;
#[cfg(feature = "store")]
const DIAG_DOC_NNZ: usize = 24;
#[cfg(feature = "store")]
const DIAG_QUERY_NNZ: usize = 16;

#[cfg(feature = "store")]
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(feature = "store")]
fn doc(state: &mut u64) -> sporse::SparseVec {
    let mut v: Vec<(u32, f32)> = (0..DOC_NNZ)
        .map(|_| {
            (
                (xorshift(state) % VOCAB as u64) as u32,
                1.0 + (xorshift(state) % 8) as f32,
            )
        })
        .collect();
    v.sort_by_key(|(t, _)| *t);
    v.dedup_by_key(|(t, _)| *t);
    sporse::SparseVec::new(v)
}

#[cfg(feature = "store")]
fn partitioned_doc(state: &mut u64, segment: usize) -> sporse::SparseVec {
    let base = segment as u32 * DIAG_SEGMENT_VOCAB;
    let mut v: Vec<(u32, f32)> = (0..DIAG_DOC_NNZ)
        .map(|_| {
            (
                base + (xorshift(state) % DIAG_SEGMENT_VOCAB as u64) as u32,
                1.0 + (xorshift(state) % 8) as f32,
            )
        })
        .collect();
    v.sort_by_key(|(t, _)| *t);
    v.dedup_by_key(|(t, _)| *t);
    sporse::SparseVec::new(v)
}

#[cfg(feature = "store")]
fn partitioned_query(segment: usize) -> sporse::SparseVec {
    let base = segment as u32 * DIAG_SEGMENT_VOCAB;
    sporse::SparseVec::new(
        (0..DIAG_QUERY_NNZ)
            .map(|i| (base + i as u32, 1.0 + (i % 4) as f32))
            .collect(),
    )
}

#[cfg(feature = "store")]
fn fresh_store(
    warm: bool,
    checkpoint: bool,
) -> (
    std::sync::Arc<dyn durability::Directory>,
    sporse::store::UpdatableIndex,
    sporse::SparseVec,
) {
    use durability::MemoryDirectory;
    let mut s = 0x1234_5678_9abc_def0u64;
    let dir = MemoryDirectory::arc();
    let mut store = sporse::store::UpdatableIndex::open(dir.clone(), FLUSH).unwrap();
    for i in 0..N_DOCS {
        store.add(i as u32, doc(&mut s)).unwrap();
    }
    if checkpoint {
        store.checkpoint().unwrap();
    }
    let q = doc(&mut s);
    if warm {
        let _ = store.search(&q, 10); // populate the per-segment cache
    }
    (dir, store, q)
}

#[cfg(feature = "store")]
fn partitioned_store() -> (sporse::store::UpdatableIndex, sporse::SparseVec) {
    use durability::MemoryDirectory;
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let dir = MemoryDirectory::arc();
    let mut store = sporse::store::UpdatableIndex::open(dir, DIAG_DOCS_PER_SEGMENT).unwrap();
    for segment in 0..DIAG_SEGMENTS {
        for doc in 0..DIAG_DOCS_PER_SEGMENT {
            let id = (segment * DIAG_DOCS_PER_SEGMENT + doc) as u32;
            store.add(id, partitioned_doc(&mut s, segment)).unwrap();
        }
    }
    store.checkpoint().unwrap();
    let query = partitioned_query(0);
    let _ = store.search(&query, 10);
    (store, query)
}

#[cfg(feature = "store")]
fn benches(c: &mut Criterion) {
    let mut g = c.benchmark_group("store");

    // Build throughput: add + seal N docs.
    g.throughput(Throughput::Elements(N_DOCS as u64));
    g.bench_function("build", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = fresh_store(false, true);
            },
            BatchSize::SmallInput,
        )
    });

    // Warm query: every segment's index already cached.
    let (_, warm, q) = fresh_store(true, true);
    g.bench_function("search_warm", |b| {
        b.iter(|| warm.search(&q, 10));
    });

    let reader = warm.reader();
    let held_view = reader.view();
    let _ = held_view.search(&q, 10); // populate the reader's per-segment cache
    g.bench_function("reader_search_warm", |b| {
        b.iter(|| reader.search(&q, 10));
    });
    g.bench_function("view_search_warm", |b| {
        b.iter(|| held_view.search(&q, 10));
    });

    g.bench_function("search_cold_load_sidecars", |b| {
        b.iter_batched(
            || {
                let (dir, _, q) = fresh_store(false, true);
                let store = sporse::store::UpdatableIndex::open(dir, FLUSH).unwrap();
                (store, q)
            },
            |(store, q)| store.search(&q, 10),
            BatchSize::SmallInput,
        )
    });

    g.bench_function("search_cold_rebuild_missing_sidecars", |b| {
        b.iter_batched(
            || {
                let (_, store, q) = fresh_store(false, false);
                (store, q)
            },
            |(store, q)| store.search(&q, 10),
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

#[cfg(feature = "store")]
fn store_pruning_diagnostics(c: &mut Criterion) {
    let (store, query) = partitioned_store();
    let (writer_hits, writer_stats) = store.search_with_stats(&query, 10);
    let reader = store.reader();
    let (reader_hits, reader_stats) = reader.search_with_stats(&query, 10);

    assert_eq!(
        writer_hits, reader_hits,
        "writer and checkpoint reader should agree after checkpoint"
    );
    assert!(
        writer_stats.pruned_segments > 0 && reader_stats.pruned_segments > 0,
        "partitioned diagnostic should exercise segment pruning"
    );

    eprintln!(
        "\n[sporse store diagnostics] partitioned corpus: {DIAG_SEGMENTS} segments, \
         {DIAG_DOCS_PER_SEGMENT} docs/segment"
    );
    eprintln!(
        "[sporse store diagnostics] writer searched/pruned segments: {}/{} \
         (sealed={})",
        writer_stats.searched_segments, writer_stats.pruned_segments, writer_stats.sealed_segments
    );
    eprintln!(
        "[sporse store diagnostics] reader searched/pruned segments: {}/{} \
         (sealed={})",
        reader_stats.searched_segments, reader_stats.pruned_segments, reader_stats.sealed_segments
    );

    c.bench_function("store_pruning_diagnostics/noop", |b| {
        b.iter(|| writer_stats.pruned_segments)
    });
}

#[cfg(feature = "store")]
fn ingest_fs(c: &mut Criterion) {
    // The extend() win is invisible on MemoryDirectory (flush is free); on a real
    // filesystem the per-item WAL flush is the cost extend amortizes into one batch
    // sync. add-per-item vs extend over the same documents.
    use durability::FsDirectory;
    let mut g = c.benchmark_group("ingest_fs");
    let n = 4_000usize;
    g.throughput(Throughput::Elements(n as u64));
    let mk = |tag: &str| {
        let mut p = std::env::temp_dir();
        p.push(format!("sporse-bench-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    };
    g.bench_function("add", |b| {
        b.iter_batched(
            || mk("add"),
            |p| {
                let mut s = 0x1234_5678_9abc_def0u64;
                let mut store =
                    sporse::store::UpdatableIndex::open(FsDirectory::arc(&p).unwrap(), FLUSH)
                        .unwrap();
                for i in 0..n {
                    store.add(i as u32, doc(&mut s)).unwrap();
                }
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("extend", |b| {
        b.iter_batched(
            || mk("extend"),
            |p| {
                let mut s = 0x1234_5678_9abc_def0u64;
                let mut store =
                    sporse::store::UpdatableIndex::open(FsDirectory::arc(&p).unwrap(), FLUSH)
                        .unwrap();
                store
                    .extend((0..n).map(|i| (i as u32, doc(&mut s))))
                    .unwrap();
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

#[cfg(feature = "store")]
criterion_group!(g, benches, store_pruning_diagnostics, ingest_fs);
#[cfg(feature = "store")]
criterion_main!(g);

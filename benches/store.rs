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
criterion_group!(g, benches, ingest_fs);
#[cfg(feature = "store")]
criterion_main!(g);

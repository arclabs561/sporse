//! Benchmarks for the `store` feature (segstore-backed updatable index).
//!
//! Run: `cargo bench --features store --bench store`. Without the feature the
//! harness is an empty no-op so the bench target still compiles.
//!
//! Measures the three costs that matter for an updatable WAND index: build
//! throughput, warm query latency (per-segment indexes cached), and the cold
//! "rebuild every segment" cost -- the latter is what a delete that clears the
//! whole cache incurs, and what the targeted-invalidation delete avoids
//! (a delete rebuilds one segment, i.e. ~1/segments of the cold cost).

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
fn fresh_store(warm: bool) -> (sporse::store::UpdatableIndex, sporse::SparseVec) {
    use durability::MemoryDirectory;
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut store = sporse::store::UpdatableIndex::open(MemoryDirectory::arc(), FLUSH).unwrap();
    for i in 0..N_DOCS {
        store.add(i as u32, doc(&mut s)).unwrap();
    }
    store.checkpoint().unwrap();
    let q = doc(&mut s);
    if warm {
        let _ = store.search(&q, 10); // populate the per-segment cache
    }
    (store, q)
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
                let _ = fresh_store(false);
            },
            BatchSize::SmallInput,
        )
    });

    // Warm query: every segment's index already cached.
    let (warm, q) = fresh_store(true);
    g.bench_function("search_warm", |b| {
        b.iter(|| warm.search(&q, 10));
    });

    // Cold query: no segment cached, so the first query rebuilds every segment.
    // This is the cost a delete-clears-the-whole-cache pays; targeted delete
    // invalidation rebuilds one segment instead.
    g.bench_function("search_cold_rebuild_all", |b| {
        b.iter_batched(
            || fresh_store(false),
            |(store, q)| store.search(&q, 10),
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

#[cfg(feature = "store")]
criterion_group!(g, benches);
#[cfg(feature = "store")]
criterion_main!(g);

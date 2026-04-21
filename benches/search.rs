/// Benchmark suite for sporse Block-Max WAND search.
///
/// Synthetic corpus mimics SPLADE-style sparse representations:
/// - Log-normal weight distribution (heavy-tailed impact scores)
/// - 10K documents, 30K vocabulary, ~120 nonzero dims per doc
/// - Queries: ~50 nonzero dims (also log-normal weights)
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sporse::{SparseVec, SporseIndex};

// ── Corpus parameters ────────────────────────────────────────────────────────

const N_DOCS: u32 = 10_000;
const VOCAB: u32 = 30_000;
const DOC_NNZ: usize = 120; // nonzero dims per document
const QUERY_NNZ: usize = 50; // nonzero dims per query
const N_QUERIES: usize = 32; // query batch size for latency benchmarks
const BENCH_SEED: u64 = 0xDEAD_BEEF_CAFE_1337;

// ── Minimal deterministic RNG (xorshift64) ───────────────────────────────────

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Uniform [0, 1) float from RNG state.
fn rand_f32(state: &mut u64) -> f32 {
    (xorshift(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Sample from a log-normal distribution approximated via Box-Muller.
/// mu=0, sigma=1 gives SPLADE-like heavy-tailed impact scores in [0, ~10].
fn lognormal(state: &mut u64) -> f32 {
    let u1 = rand_f32(state).max(1e-9);
    let u2 = rand_f32(state);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    // mu=0.3, sigma=0.8: keeps most weights in [0.05, 5.0]
    let w = (0.3 + 0.8 * z).exp();
    w.clamp(0.01, 20.0)
}

/// Generate a sparse vector with `nnz` nonzero dimensions from [0, vocab).
fn gen_sparse(state: &mut u64, vocab: u32, nnz: usize) -> SparseVec {
    let pairs: Vec<(u32, f32)> = (0..nnz)
        .map(|_| {
            let dim = (xorshift(state) % vocab as u64) as u32;
            let w = lognormal(state);
            (dim, w)
        })
        .collect();
    // SparseVec::new deduplicates — actual nnz may be slightly < requested
    SparseVec::new(pairs)
}

// ── Corpus/query fixtures ────────────────────────────────────────────────────

struct Fixture {
    index: SporseIndex,
    docs: Vec<SparseVec>,
    queries: Vec<SparseVec>,
}

impl Fixture {
    fn build() -> Self {
        let mut rng = BENCH_SEED;

        let docs: Vec<SparseVec> = (0..N_DOCS)
            .map(|_| gen_sparse(&mut rng, VOCAB, DOC_NNZ))
            .collect();

        let mut index = SporseIndex::new();
        for (id, doc) in docs.iter().enumerate() {
            index.insert(id as u32, doc);
        }
        index.build();

        // Use a different seed for queries so they're independent of corpus.
        let mut qrng = rng ^ 0x1234_5678_9ABC_DEF0;
        let queries: Vec<SparseVec> = (0..N_QUERIES)
            .map(|_| gen_sparse(&mut qrng, VOCAB, QUERY_NNZ))
            .collect();

        Fixture {
            index,
            docs,
            queries,
        }
    }
}

// ── Brute-force baseline ─────────────────────────────────────────────────────

/// Exhaustive inner product scan. O(N_DOCS * QUERY_NNZ).
fn brute_force(docs: &[SparseVec], query: &SparseVec, k: usize) -> Vec<(u32, f32)> {
    let mut scores: Vec<(u32, f32)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| (i as u32, query.dot(doc)))
        .filter(|&(_, s)| s > 0.0)
        .collect();
    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scores.truncate(k);
    scores
}

// ── Benchmarks ───────────────────────────────────────────────────────────────

fn bench_build(c: &mut Criterion) {
    let mut rng = BENCH_SEED;
    let docs: Vec<SparseVec> = (0..N_DOCS)
        .map(|_| gen_sparse(&mut rng, VOCAB, DOC_NNZ))
        .collect();

    c.bench_function("insert_10k", |b| {
        b.iter(|| {
            let mut index = SporseIndex::new();
            for (id, doc) in docs.iter().enumerate() {
                index.insert(id as u32, doc);
            }
            index.build();
            index
        });
    });
}

fn bench_search(c: &mut Criterion) {
    let fixture = Fixture::build();
    let mut group = c.benchmark_group("search");

    for k in [10usize, 100] {
        group.bench_with_input(BenchmarkId::new("wand_top", k), &k, |b, &k| {
            b.iter(|| {
                let mut total_score = 0.0f32;
                for q in &fixture.queries {
                    let results = fixture.index.search(q, k);
                    if let Some(&(_, s)) = results.first() {
                        total_score += s;
                    }
                }
                total_score
            });
        });

        group.bench_with_input(BenchmarkId::new("brute_force_top", k), &k, |b, &k| {
            b.iter(|| {
                let mut total_score = 0.0f32;
                for q in &fixture.queries {
                    let results = brute_force(&fixture.docs, q, k);
                    if let Some(&(_, s)) = results.first() {
                        total_score += s;
                    }
                }
                total_score
            });
        });
    }

    group.finish();
}

// ── Diagnostics (printed, not benched) ───────────────────────────────────────

fn print_diagnostics(c: &mut Criterion) {
    let fixture = Fixture::build();
    let k = 10;
    let n_check = N_QUERIES;

    let mut wand_agrees = 0;
    let mut total_iterations = 0u64;
    let mut total_scored = 0u64;
    let mut total_skips = 0u64;

    for q in fixture.queries.iter().take(n_check) {
        let (wand, stats) = fixture.index.search_with_stats(q, k);
        let bf = brute_force(&fixture.docs, q, k);

        let wand_ids: std::collections::HashSet<u32> = wand.iter().map(|r| r.0).collect();
        let bf_ids: std::collections::HashSet<u32> = bf.iter().map(|r| r.0).collect();
        if wand_ids == bf_ids {
            wand_agrees += 1;
        }
        total_iterations += stats.iterations;
        total_scored += stats.docs_scored;
        total_skips += stats.cursor_skips;
    }

    let n = n_check as u64;
    let avg_iter = total_iterations / n;
    let avg_scored = total_scored / n;
    let avg_skips = total_skips / n;
    let skip_rate = if total_iterations > 0 {
        100.0 * total_skips as f64 / total_iterations as f64
    } else {
        0.0
    };
    // WAND efficiency: fraction of the 10K collection actually scored
    let scored_frac = 100.0 * avg_scored as f64 / N_DOCS as f64;

    eprintln!(
        "\n[sporse diagnostics] WAND top-{k} agrees with brute force: {}/{} queries",
        wand_agrees, n_check
    );
    eprintln!(
        "[sporse diagnostics] Index: {} docs, {} vocab dims, ~{DOC_NNZ} nnz/doc, ~{QUERY_NNZ} nnz/query",
        fixture.index.len(),
        fixture.index.num_dimensions()
    );
    eprintln!("[sporse diagnostics] Per-query averages over {n_check} queries:");
    eprintln!("  iterations:  {avg_iter}");
    eprintln!("  docs scored: {avg_scored} ({scored_frac:.1}% of collection)");
    eprintln!("  cursor skips (advance_to calls): {avg_skips}");
    eprintln!("  skip-to-score ratio: {skip_rate:.1}% of iterations were skips");

    // Dummy bench so criterion doesn't complain about an unused group
    c.bench_function("diagnostics_noop", |b| b.iter(|| 0u64));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_build, bench_search, print_diagnostics
);
criterion_main!(benches);

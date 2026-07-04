/// Benchmark suite for sporse Block-Max WAND search.
///
/// Synthetic corpus mimics SPLADE-style sparse representations:
/// - Log-normal weight distribution (heavy-tailed impact scores)
/// - 10K documents, 30K vocabulary, ~120 nonzero dims per doc
/// - Queries: ~50 nonzero dims (also log-normal weights)
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use postings::raw::{
    top_k_weighted_u32_files, top_k_weighted_u32_files_with_stats,
    write_u64_u32_segment_sorted_from_iter_to, RawDocument, RawSegmentFile, RawTermId,
};
use sporse::{SparseVec, SporseIndex};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Corpus parameters ────────────────────────────────────────────────────────

const N_DOCS: u32 = 10_000;
const VOCAB: u32 = 30_000;
const DOC_NNZ: usize = 120; // nonzero dims per document
const QUERY_NNZ: usize = 50; // nonzero dims per query
const N_QUERIES: usize = 32; // query batch size for latency benchmarks
const BENCH_SEED: u64 = 0xDEAD_BEEF_CAFE_1337;
const IMPACT_SCALE: f32 = 100.0;

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
    // SparseVec::new folds duplicate dims, so actual nnz may be slightly < requested.
    SparseVec::new(pairs)
}

// ── Corpus/query fixtures ────────────────────────────────────────────────────

struct Fixture {
    index: SporseIndex,
    docs: Vec<SparseVec>,
    queries: Vec<SparseVec>,
    raw_path: TempRawPath,
    raw_paths: Vec<TempRawPath>,
    raw_queries: Vec<Vec<(RawTermId, f32)>>,
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
        let raw_path = write_raw_impact_file(&docs);
        let raw_paths = write_raw_impact_files(&docs, 4);
        let raw_queries = queries
            .iter()
            .map(|query| raw_query(query, IMPACT_SCALE))
            .collect();

        Fixture {
            index,
            docs,
            queries,
            raw_path,
            raw_paths,
            raw_queries,
        }
    }
}

fn write_raw_impact_file(docs: &[SparseVec]) -> TempRawPath {
    write_raw_impact_file_with_base(docs, 0)
}

fn write_raw_impact_file_with_base(docs: &[SparseVec], base_doc_id: u32) -> TempRawPath {
    let raw_terms: Vec<Vec<_>> = docs
        .iter()
        .map(|doc| doc.to_raw_impact_document(IMPACT_SCALE).unwrap())
        .collect();
    let path = TempRawPath::new();
    let docs = raw_terms
        .iter()
        .enumerate()
        .map(|(doc_id, terms)| RawDocument::new(base_doc_id + doc_id as u32, terms));
    let mut file = std::fs::File::create(path.as_path()).unwrap();
    write_u64_u32_segment_sorted_from_iter_to(docs, &mut file).unwrap();
    path
}

fn write_raw_impact_files(docs: &[SparseVec], n_files: usize) -> Vec<TempRawPath> {
    assert!(n_files > 0);
    let chunk_size = docs.len().div_ceil(n_files);
    docs.chunks(chunk_size)
        .enumerate()
        .map(|(chunk_id, chunk)| {
            write_raw_impact_file_with_base(chunk, (chunk_id * chunk_size) as u32)
        })
        .collect()
}

fn raw_query(query: &SparseVec, scale: f32) -> Vec<(RawTermId, f32)> {
    query.to_raw_impact_query(scale).unwrap()
}

struct TempRawPath {
    path: PathBuf,
}

impl TempRawPath {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self {
            path: std::env::temp_dir().join(format!(
                "sporse-raw-impact-bench-{}-{nanos}.raw",
                std::process::id()
            )),
        }
    }

    fn as_path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempRawPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
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
    let mut raw_segment = RawSegmentFile::open(fixture.raw_path.as_path()).unwrap();
    let mut raw_segments: Vec<RawSegmentFile> = fixture
        .raw_paths
        .iter()
        .map(|path| RawSegmentFile::open(path.as_path()).unwrap())
        .collect();
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

        group.bench_with_input(BenchmarkId::new("raw_u32_file_top", k), &k, |b, &k| {
            b.iter(|| {
                let mut total_score = 0.0f32;
                for q in &fixture.raw_queries {
                    let results = raw_segment.top_k_weighted_u32(q, k).unwrap();
                    if let Some(&(_, s)) = results.first() {
                        total_score += s;
                    }
                }
                total_score
            });
        });

        group.bench_with_input(BenchmarkId::new("raw_u32_files_top", k), &k, |b, &k| {
            let mut refs: Vec<&mut RawSegmentFile> = raw_segments.iter_mut().collect();
            b.iter(|| {
                let mut total_score = 0.0f32;
                for q in &fixture.raw_queries {
                    let results = top_k_weighted_u32_files(&mut refs, q, k).unwrap();
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
    let mut raw_agrees = 0;
    let mut raw_segment = RawSegmentFile::open(fixture.raw_path.as_path()).unwrap();
    let mut raw_segments: Vec<RawSegmentFile> = fixture
        .raw_paths
        .iter()
        .map(|path| RawSegmentFile::open(path.as_path()).unwrap())
        .collect();
    let mut raw_files_agrees = 0;
    let mut raw_refs: Vec<&mut RawSegmentFile> = raw_segments.iter_mut().collect();
    let mut raw_segments_seen = 0usize;
    let mut raw_segments_scored = 0usize;
    let mut raw_segments_pruned = 0usize;

    for (i, q) in fixture.queries.iter().take(n_check).enumerate() {
        let (wand, stats) = fixture.index.search_with_stats(q, k);
        let bf = brute_force(&fixture.docs, q, k);
        let raw = raw_segment
            .top_k_weighted_u32(&fixture.raw_queries[i], k)
            .unwrap();
        let raw_files_result =
            top_k_weighted_u32_files_with_stats(&mut raw_refs, &fixture.raw_queries[i], k).unwrap();
        let raw_files = raw_files_result.hits;
        raw_segments_seen += raw_files_result.stats.segments_seen;
        raw_segments_scored += raw_files_result.stats.segments_scored;
        raw_segments_pruned += raw_files_result.stats.segments_pruned;

        let wand_ids: std::collections::HashSet<u32> = wand.iter().map(|r| r.0).collect();
        let bf_ids: std::collections::HashSet<u32> = bf.iter().map(|r| r.0).collect();
        if wand_ids == bf_ids {
            wand_agrees += 1;
        }
        let raw_ids: std::collections::HashSet<u32> = raw.iter().map(|r| r.0).collect();
        if raw_ids == bf_ids {
            raw_agrees += 1;
        }
        let raw_files_ids: std::collections::HashSet<u32> = raw_files.iter().map(|r| r.0).collect();
        if raw_files_ids == bf_ids {
            raw_files_agrees += 1;
        }
        total_iterations += stats.iterations;
        total_scored += stats.docs_scored;
        total_skips += stats.cursor_skips;
    }

    let n = n_check as u64;
    let avg_iter = total_iterations / n;
    let avg_scored = total_scored / n;
    let avg_skips = total_skips / n;
    let advances_per_iter = if total_iterations > 0 {
        total_skips as f64 / total_iterations as f64
    } else {
        0.0
    };
    let avg_raw_seen = raw_segments_seen as f64 / n_check as f64;
    let avg_raw_scored = raw_segments_scored as f64 / n_check as f64;
    let avg_raw_pruned = raw_segments_pruned as f64 / n_check as f64;
    // WAND efficiency: fraction of the 10K collection actually scored
    let scored_frac = 100.0 * avg_scored as f64 / N_DOCS as f64;

    eprintln!(
        "\n[sporse diagnostics] WAND top-{k} agrees with brute force: {}/{} queries",
        wand_agrees, n_check
    );
    eprintln!(
        "[sporse diagnostics] quantized raw u32 file top-{k} agrees with brute force: {}/{} queries",
        raw_agrees, n_check
    );
    eprintln!(
        "[sporse diagnostics] quantized raw u32 files top-{k} agrees with brute force: {}/{} queries",
        raw_files_agrees, n_check
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
    eprintln!("  advance_to calls per iteration: {advances_per_iter:.2}");
    eprintln!(
        "  raw files searched/pruned segments: {avg_raw_scored:.1}/{avg_raw_pruned:.1} of {avg_raw_seen:.1}"
    );

    // Dummy bench so criterion doesn't complain about an unused group
    c.bench_function("diagnostics_noop", |b| b.iter(|| 0u64));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_build, bench_search, print_diagnostics
);
criterion_main!(benches);

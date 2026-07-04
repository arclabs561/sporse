//! Parity for the quantized-impact out-of-core path.
//!
//! `sporse` owns exact f32 sparse-vector search. `postings::raw` owns byte/file
//! backed u32 impact segments. This test pins the bridge between them for
//! fixed-point weights: if document weights are quantized by a declared scale,
//! a raw segment can reproduce `sporse` rankings without rebuilding an in-memory
//! `SporseIndex` from the segment payload.

use postings::raw::{
    top_k_weighted_u32_files, top_k_weighted_u32_files_and_index, write_u64_u32_segment,
    RawDocument, RawSegmentFile, RawTermId,
};
use postings::PostingsIndex;
use sporse::{SparseVec, SporseIndex};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const SCALE: f32 = 100.0;
const QUERY_SEED: u64 = 0x5eed_ba5e_1234_9876;
static TEMP_RAW_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn quantized_raw_impacts_match_sporse_ranking_for_fixed_point_weights() {
    let docs = vec![
        (10, SparseVec::new(vec![(1, 2.25), (2, 0.75), (7, 1.10)])),
        (20, SparseVec::new(vec![(1, 1.50), (3, 3.25), (7, 0.40)])),
        (30, SparseVec::new(vec![(2, 2.40), (3, 0.60), (5, 1.20)])),
        (40, SparseVec::new(vec![(8, 8.0)])),
    ];
    let query = SparseVec::new(vec![(1, 1.25), (2, 1.50), (3, 0.80)]);

    let mut sporse = SporseIndex::new();
    for (doc_id, vector) in &docs {
        sporse.insert(*doc_id, vector);
    }
    sporse.build();
    let expected = sporse.search(&query, 4);

    let raw_terms: Vec<Vec<(RawTermId, u32)>> = docs
        .iter()
        .map(|(_, vector)| vector.to_raw_impact_document(SCALE).unwrap())
        .collect();
    let raw_docs: Vec<_> = docs
        .iter()
        .zip(raw_terms.iter())
        .map(|((doc_id, _), terms)| RawDocument::new(*doc_id, terms))
        .collect();
    let bytes = write_u64_u32_segment(&raw_docs).unwrap();

    let path = TempRawPath::new();
    std::fs::write(path.as_path(), bytes).unwrap();
    let mut raw = RawSegmentFile::open(path.as_path()).unwrap();

    let raw_query = query.to_raw_impact_query(SCALE).unwrap();
    let got = raw.top_k_weighted_u32(&raw_query, 4).unwrap();

    assert_rankings_close(&expected, &got);
}

#[test]
fn quantized_raw_impacts_match_quantized_sparse_oracle_on_heavy_tailed_fixture() {
    let docs = generated_docs(384, 2_048, 48, 0x51ade_51ade);
    let path = write_raw_impact_file(&docs, SCALE);
    let mut raw = RawSegmentFile::open(path.as_path()).unwrap();

    let queries = generated_queries(32, 2_048, 16);
    for (query_id, query) in queries.iter().enumerate() {
        let expected = top_k_by_score(&docs, query, 10, |query, doc| {
            quantized_dot(query, doc, SCALE)
        });
        let got = raw
            .top_k_weighted_u32(&raw_query(query, SCALE), 10)
            .unwrap();
        assert_rankings_close(&expected, &got);
        assert!(
            got.len() <= 10,
            "query {query_id}: raw scorer returned more than k results"
        );
    }
}

#[test]
fn quantized_raw_impacts_match_quantized_sparse_oracle_across_files() {
    let docs = generated_docs(512, 2_048, 48, 0x517e_517e);
    let paths = write_raw_impact_files(&docs, SCALE, 4);
    let mut files: Vec<_> = paths
        .iter()
        .map(|path| RawSegmentFile::open(path.as_path()).unwrap())
        .collect();
    let mut segments: Vec<&mut RawSegmentFile> = files.iter_mut().collect();

    let queries = generated_queries(16, 2_048, 16);
    for (query_id, query) in queries.iter().enumerate() {
        let expected = top_k_by_score(&docs, query, 10, |query, doc| {
            quantized_dot(query, doc, SCALE)
        });
        let got = top_k_weighted_u32_files(&mut segments, &raw_query(query, SCALE), 10).unwrap();

        assert_rankings_close(&expected, &got);
        assert!(
            got.len() <= 10,
            "query {query_id}: multi-file raw scorer returned more than k results"
        );
    }
}

#[test]
fn quantized_raw_impacts_match_quantized_sparse_oracle_across_files_and_live_index() {
    let docs = generated_docs(512, 2_048, 48, 0x517e_11ee);
    let live_start = docs.len() - 64;
    let paths = write_raw_impact_files(&docs[..live_start], SCALE, 4);
    let live_index = build_raw_impact_index(&docs[live_start..], SCALE);
    let mut files: Vec<_> = paths
        .iter()
        .map(|path| RawSegmentFile::open(path.as_path()).unwrap())
        .collect();
    let mut segments: Vec<&mut RawSegmentFile> = files.iter_mut().collect();

    let queries = generated_queries(16, 2_048, 16);
    for (query_id, query) in queries.iter().enumerate() {
        let expected = top_k_by_score(&docs, query, 10, |query, doc| {
            quantized_dot(query, doc, SCALE)
        });
        let got = top_k_weighted_u32_files_and_index(
            &mut segments,
            &live_index,
            &raw_query(query, SCALE),
            10,
        )
        .unwrap();

        assert_rankings_close(&expected, &got);
        assert!(
            got.len() <= 10,
            "query {query_id}: raw files+live scorer returned more than k results"
        );
    }
}

#[test]
fn quantized_raw_impacts_preserve_exact_ranking_when_margin_exceeds_rounding_bound() {
    let docs = vec![
        (1, SparseVec::new(vec![(1, 3.014), (2, 2.019), (9, 0.331)])),
        (2, SparseVec::new(vec![(1, 2.512), (2, 1.901), (7, 1.101)])),
        (3, SparseVec::new(vec![(1, 1.403), (3, 3.307), (7, 0.221)])),
        (4, SparseVec::new(vec![(2, 1.104), (3, 1.101), (5, 2.229)])),
        (5, SparseVec::new(vec![(8, 9.0)])),
    ];
    let query = SparseVec::new(vec![(1, 1.70), (2, 0.80), (3, 0.40)]);
    let k = 3;

    let exact = top_k_by_score(&docs, &query, docs.len(), |query, doc| query.dot(doc));
    let margin = exact[k - 1].1 - exact[k].1;
    let error_bound = quantized_score_error_bound(&query, SCALE);
    assert!(
        margin > 2.0 * error_bound,
        "fixture must pin a stable ranking: margin={margin}, bound={error_bound}"
    );

    let path = write_raw_impact_file(&docs, SCALE);
    let mut raw = RawSegmentFile::open(path.as_path()).unwrap();
    let got = raw
        .top_k_weighted_u32(&raw_query(&query, SCALE), k)
        .unwrap();
    let expected = &exact[..k];

    assert_eq!(
        doc_ids(expected),
        doc_ids(&got),
        "ranking should be invariant when the exact score gap exceeds rounding error"
    );
    for (&(doc_id, exact_score), &(_, raw_score)) in expected.iter().zip(got.iter()) {
        assert!(
            (exact_score - raw_score).abs() <= error_bound + 1e-5,
            "doc {doc_id}: exact={exact_score}, raw={raw_score}, bound={error_bound}"
        );
    }
}

#[test]
fn quantized_raw_impacts_recall_sweep_improves_with_scale() {
    let docs = generated_docs(512, 2_048, 48, 0xacc0_7a1e);
    let queries = generated_queries(24, 2_048, 16);
    let k = 10usize;

    let recall_at_10 = scale_sweep_recall(&docs, &queries, k);

    assert!(
        recall_at_10[2] >= 0.95,
        "scale 100 should preserve high recall on the heavy-tailed fixture: {recall_at_10:?}"
    );
    assert!(
        recall_at_10[2] >= recall_at_10[0],
        "finer quantization should not underperform the coarse scale: {recall_at_10:?}"
    );
}

#[test]
fn quantized_raw_impacts_recall_sweep_covers_flat_weights() {
    let docs = generated_flat_docs(512, 2_048, 48, 0xf1a7_f1a7);
    let queries = generated_flat_queries(24, 2_048, 16);
    let k = 10usize;

    let recall_at_10 = scale_sweep_recall(&docs, &queries, k);

    assert!(
        recall_at_10[2] >= 0.90,
        "scale 100 should preserve useful recall on the flatter fixture: {recall_at_10:?}"
    );
    assert!(
        recall_at_10[2] >= recall_at_10[0],
        "finer quantization should not underperform the coarse scale: {recall_at_10:?}"
    );
}

#[test]
fn quantized_raw_impacts_recall_sweep_covers_query_density() {
    let docs = generated_docs(640, 4_096, 64, 0xd351_7e57);
    let k = 10usize;

    for (query_nnz, min_recall) in [(4, 0.95), (16, 0.92), (64, 0.85)] {
        let queries = generated_queries_with_seed(
            16,
            4_096,
            query_nnz,
            QUERY_SEED ^ (query_nnz as u64).wrapping_mul(0x9e37_79b9),
        );
        let recall_at_k = scale_sweep_recall_for_scales(&docs, &queries, k, &[25.0, 100.0, 250.0]);

        assert!(
            recall_at_k[1] >= min_recall,
            "scale 100 recall should hold for query_nnz={query_nnz}: {recall_at_k:?}"
        );
        assert!(
            recall_at_k[2] >= recall_at_k[0],
            "finer quantization should not underperform the coarse scale for query_nnz={query_nnz}: {recall_at_k:?}"
        );
    }
}

fn quantize_with_scale(weight: f32, scale: f32) -> u32 {
    assert!(weight.is_finite() && weight >= 0.0);
    let scaled = (weight * scale).round();
    assert!(scaled > 0.0 && scaled <= u32::MAX as f32);
    scaled as u32
}

fn write_raw_impact_file(docs: &[(u32, SparseVec)], scale: f32) -> TempRawPath {
    let raw_terms: Vec<Vec<(RawTermId, u32)>> = docs
        .iter()
        .map(|(_, vector)| vector.to_raw_impact_document(scale).unwrap())
        .collect();
    let raw_docs: Vec<_> = docs
        .iter()
        .zip(raw_terms.iter())
        .map(|((doc_id, _), terms)| RawDocument::new(*doc_id, terms))
        .collect();
    let bytes = write_u64_u32_segment(&raw_docs).unwrap();

    let path = TempRawPath::new();
    std::fs::write(path.as_path(), bytes).unwrap();
    path
}

fn write_raw_impact_files(
    docs: &[(u32, SparseVec)],
    scale: f32,
    n_files: usize,
) -> Vec<TempRawPath> {
    assert!(n_files > 0);
    let chunk_size = docs.len().div_ceil(n_files);
    docs.chunks(chunk_size)
        .map(|chunk| write_raw_impact_file(chunk, scale))
        .collect()
}

fn build_raw_impact_index(docs: &[(u32, SparseVec)], scale: f32) -> PostingsIndex<RawTermId, u32> {
    let mut index = PostingsIndex::new();
    for (doc_id, vector) in docs {
        let terms = vector.to_raw_impact_document(scale).unwrap();
        index.add_weighted_document(*doc_id, &terms).unwrap();
    }
    index
}

fn raw_query(query: &SparseVec, scale: f32) -> Vec<(RawTermId, f32)> {
    query.to_raw_impact_query(scale).unwrap()
}

fn quantized_dot(query: &SparseVec, doc: &SparseVec, scale: f32) -> f32 {
    let (mut qi, mut di) = (0, 0);
    let (q, d) = (query.pairs(), doc.pairs());
    let mut sum = 0.0;
    while qi < q.len() && di < d.len() {
        match q[qi].0.cmp(&d[di].0) {
            std::cmp::Ordering::Equal => {
                sum += q[qi].1 * quantize_with_scale(d[di].1, scale) as f32 / scale;
                qi += 1;
                di += 1;
            }
            std::cmp::Ordering::Less => qi += 1,
            std::cmp::Ordering::Greater => di += 1,
        }
    }
    sum
}

fn quantized_score_error_bound(query: &SparseVec, scale: f32) -> f32 {
    query
        .pairs()
        .iter()
        .map(|&(_, weight)| weight.abs() * 0.5 / scale)
        .sum()
}

fn top_k_by_score(
    docs: &[(u32, SparseVec)],
    query: &SparseVec,
    k: usize,
    mut score: impl FnMut(&SparseVec, &SparseVec) -> f32,
) -> Vec<(u32, f32)> {
    let mut ranked: Vec<_> = docs
        .iter()
        .filter_map(|(doc_id, doc)| {
            let score = score(query, doc);
            (score > 0.0 && score.is_finite()).then_some((*doc_id, score))
        })
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k);
    ranked
}

fn doc_ids(results: &[(u32, f32)]) -> Vec<u32> {
    results.iter().map(|(doc_id, _)| *doc_id).collect()
}

fn mean_raw_recall_at_k(
    docs: &[(u32, SparseVec)],
    queries: &[SparseVec],
    k: usize,
    scale: f32,
) -> f32 {
    let path = write_raw_impact_file(docs, scale);
    let mut raw = RawSegmentFile::open(path.as_path()).unwrap();
    let mut total = 0.0f32;
    for query in queries {
        let exact = top_k_by_score(docs, query, k, |query, doc| query.dot(doc));
        let got = raw.top_k_weighted_u32(&raw_query(query, scale), k).unwrap();
        total += recall_at_k(&doc_ids(&exact), &doc_ids(&got), k);
    }
    total / queries.len() as f32
}

fn scale_sweep_recall(docs: &[(u32, SparseVec)], queries: &[SparseVec], k: usize) -> Vec<f32> {
    scale_sweep_recall_for_scales(docs, queries, k, &[10.0, 25.0, 100.0])
}

fn scale_sweep_recall_for_scales(
    docs: &[(u32, SparseVec)],
    queries: &[SparseVec],
    k: usize,
    scales: &[f32],
) -> Vec<f32> {
    scales
        .iter()
        .map(|&scale| mean_raw_recall_at_k(docs, queries, k, scale))
        .collect()
}

fn recall_at_k(expected: &[u32], got: &[u32], k: usize) -> f32 {
    let got: std::collections::HashSet<_> = got.iter().copied().collect();
    let hits = expected
        .iter()
        .take(k)
        .filter(|doc_id| got.contains(doc_id))
        .count();
    hits as f32 / k as f32
}

fn generated_docs(n_docs: u32, vocab: u32, nnz: usize, seed: u64) -> Vec<(u32, SparseVec)> {
    let mut state = seed;
    (0..n_docs)
        .map(|doc_id| (doc_id, generated_sparse(&mut state, vocab, nnz)))
        .collect()
}

fn generated_queries(n_queries: usize, vocab: u32, nnz: usize) -> Vec<SparseVec> {
    generated_queries_with_seed(n_queries, vocab, nnz, QUERY_SEED)
}

fn generated_queries_with_seed(
    n_queries: usize,
    vocab: u32,
    nnz: usize,
    seed: u64,
) -> Vec<SparseVec> {
    let mut state = seed;
    (0..n_queries)
        .map(|_| generated_sparse(&mut state, vocab, nnz))
        .collect()
}

fn generated_flat_docs(n_docs: u32, vocab: u32, nnz: usize, seed: u64) -> Vec<(u32, SparseVec)> {
    let mut state = seed;
    (0..n_docs)
        .map(|doc_id| (doc_id, generated_flat_sparse(&mut state, vocab, nnz)))
        .collect()
}

fn generated_flat_queries(n_queries: usize, vocab: u32, nnz: usize) -> Vec<SparseVec> {
    let mut state = QUERY_SEED ^ 0x51a7_51a7;
    (0..n_queries)
        .map(|_| generated_flat_sparse(&mut state, vocab, nnz))
        .collect()
}

fn generated_sparse(state: &mut u64, vocab: u32, nnz: usize) -> SparseVec {
    let pairs = (0..nnz)
        .map(|_| {
            let dim = (xorshift(state) % vocab as u64) as u32;
            let weight = lognormal_weight(state);
            (dim, weight)
        })
        .collect();
    SparseVec::new(pairs)
}

fn generated_flat_sparse(state: &mut u64, vocab: u32, nnz: usize) -> SparseVec {
    let pairs = (0..nnz)
        .map(|_| {
            let dim = (xorshift(state) % vocab as u64) as u32;
            let weight = 0.05 + rand_f32(state) * 0.95;
            (dim, weight)
        })
        .collect();
    SparseVec::new(pairs)
}

fn lognormal_weight(state: &mut u64) -> f32 {
    let u1 = rand_f32(state).max(1e-9);
    let u2 = rand_f32(state);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    (0.3 + 0.8 * z).exp().clamp(0.01, 20.0)
}

fn rand_f32(state: &mut u64) -> f32 {
    (xorshift(state) >> 11) as f32 / (1u64 << 53) as f32
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn assert_rankings_close(expected: &[(u32, f32)], got: &[(u32, f32)]) {
    assert_eq!(expected.len(), got.len());
    for (&(expected_doc, expected_score), &(got_doc, got_score)) in expected.iter().zip(got) {
        assert_eq!(expected_doc, got_doc);
        assert!(
            (expected_score - got_score).abs() < 1e-5,
            "doc {expected_doc}: expected {expected_score}, got {got_score}"
        );
    }
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
        let sequence = TEMP_RAW_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            path: std::env::temp_dir().join(format!(
                "sporse-raw-impact-{}-{nanos}-{sequence}.raw",
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

use sporse::{SparseVec, SporseIndex};
use std::collections::BTreeSet;

const N_DOCS: u32 = 4_000;
const VOCAB: u32 = 24_000;
const DOC_NNZ: usize = 96;
const QUERY_NNZ: usize = 40;
const N_QUERIES: usize = 12;
const TOP_K: usize = 10;

fn main() {
    let mut rng = 0x51A5_EED5_0BAD_F00D;
    let docs: Vec<SparseVec> = (0..N_DOCS)
        .map(|_| gen_sparse(&mut rng, VOCAB, DOC_NNZ))
        .collect();

    let mut index = SporseIndex::new();
    for (doc_id, doc) in docs.iter().enumerate() {
        index.insert(doc_id as u32, doc);
    }
    index.build();

    let queries = make_queries(&docs);
    let mut exact_matches = 0usize;
    let mut total_iterations = 0u64;
    let mut total_scored = 0u64;
    let mut total_skips = 0u64;

    for query in &queries {
        let (wand, stats) = index.search_with_stats(query, TOP_K);
        let brute = brute_force(&docs, query, TOP_K);

        assert_same_top_k(&wand, &brute);
        exact_matches += 1;
        total_iterations += stats.iterations;
        total_scored += stats.docs_scored;
        total_skips += stats.cursor_skips;
    }

    let n = queries.len() as f64;
    let avg_scored = total_scored as f64 / n;
    let avg_iterations = total_iterations as f64 / n;
    let avg_skips = total_skips as f64 / n;
    let scored_pct = 100.0 * avg_scored / N_DOCS as f64;

    println!(
        "index: {} docs, {} sparse dimensions, {DOC_NNZ} nnz/doc",
        index.len(),
        index.num_dimensions()
    );
    println!("queries: {N_QUERIES}, {QUERY_NNZ} nnz/query, top-{TOP_K}");
    println!("exact top-k parity with brute force: {exact_matches}/{N_QUERIES}");
    println!("average WAND iterations: {avg_iterations:.1}");
    println!("average fully-scored docs: {avg_scored:.1} ({scored_pct:.2}% of collection)");
    println!("average cursor skips: {avg_skips:.1}");
}

fn make_queries(docs: &[SparseVec]) -> Vec<SparseVec> {
    (0..N_QUERIES)
        .map(|i| {
            let source = &docs[(i * 257 + 17) % docs.len()];
            let mut pairs: Vec<(u32, f32)> = source
                .pairs()
                .iter()
                .step_by(3)
                .take(QUERY_NNZ - 8)
                .map(|&(dim, weight)| (dim, 0.5 + weight.sqrt()))
                .collect();

            for j in 0..8 {
                let dim = (i as u32 * 1597 + j as u32 * 7919 + 101) % VOCAB;
                pairs.push((dim, 0.4 + j as f32 * 0.07));
            }
            SparseVec::new(pairs)
        })
        .collect()
}

fn brute_force(docs: &[SparseVec], query: &SparseVec, k: usize) -> Vec<(u32, f32)> {
    let mut scores: Vec<(u32, f32)> = docs
        .iter()
        .enumerate()
        .map(|(doc_id, doc)| (doc_id as u32, query.dot(doc)))
        .filter(|&(_, score)| score > 0.0)
        .collect();
    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scores.truncate(k);
    scores
}

fn assert_same_top_k(wand: &[(u32, f32)], brute: &[(u32, f32)]) {
    assert_eq!(wand.len(), brute.len());
    for (actual, expected) in wand.iter().zip(brute) {
        assert_eq!(actual.0, expected.0);
        assert!(
            (actual.1 - expected.1).abs() <= 1e-4,
            "doc {} score mismatch: WAND={} brute={}",
            actual.0,
            actual.1,
            expected.1
        );
    }
}

fn gen_sparse(rng: &mut u64, vocab: u32, nnz: usize) -> SparseVec {
    let mut dims = BTreeSet::new();
    while dims.len() < nnz {
        dims.insert((xorshift(rng) % vocab as u64) as u32);
    }

    SparseVec::from_sorted(dims.into_iter().map(|dim| (dim, lognormal(rng))).collect())
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn rand_f32(state: &mut u64) -> f32 {
    (xorshift(state) >> 11) as f32 / (1u64 << 53) as f32
}

fn lognormal(state: &mut u64) -> f32 {
    let u1 = rand_f32(state).max(1e-9);
    let u2 = rand_f32(state);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    (0.2 + 0.7 * z).exp().clamp(0.01, 12.0)
}

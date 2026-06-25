//! Hybrid retrieval: learned-sparse (`sporse`) + dense ANN (`vicinity`), fused
//! with reciprocal-rank fusion (`rankops`).
//!
//! The point of hybrid retrieval is that a document strong on BOTH a sparse
//! (lexical / learned-term) signal and a dense (semantic) signal should beat a
//! document strong on only one. This composes sporse's inverted index,
//! vicinity's HNSW, and rankops RRF to show exactly that: a "both-relevant"
//! document that is #1 on neither signal alone wins the fused ranking.
//!
//! Run: `cargo run --example hybrid_retrieval`

use rankops::rrf;
use sporse::{SparseVec, SporseIndex};
use vicinity::hnsw::HNSWIndex;

const DIM: usize = 8;
const DENSE_DECOY: u32 = 0; // #1 dense, absent from sparse
const SPARSE_DECOY: u32 = 1; // #1 sparse, low dense
const TARGET: u32 = 2; // #2 on both -> wins fusion

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}

fn dense(doc: u32) -> Vec<f32> {
    match doc {
        DENSE_DECOY => unit(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        TARGET => unit(&[0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        SPARSE_DECOY => unit(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        _ => {
            // Small query-direction component so noise ranks below TARGET but
            // above SPARSE_DECOY, keeping SPARSE_DECOY out of the dense top-k.
            let mut v = vec![0.0f32; DIM];
            v[0] = 0.3;
            v[2 + (doc as usize % (DIM - 2))] = 0.9;
            unit(&v)
        }
    }
}

fn sparse(doc: u32) -> SparseVec {
    match doc {
        SPARSE_DECOY => SparseVec::new(vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]),
        TARGET => SparseVec::new(vec![(0, 1.0), (1, 1.0)]),
        DENSE_DECOY => SparseVec::new(vec![(20, 1.0), (21, 1.0)]),
        _ => SparseVec::new(vec![(30 + doc, 1.0)]),
    }
}

fn main() {
    let n_docs = 20u32;

    let mut hnsw = HNSWIndex::new(DIM, 8, 16).expect("valid HNSW params");
    for d in 0..n_docs {
        hnsw.add(d, dense(d)).expect("add dense vector");
    }
    hnsw.build().expect("build HNSW");

    let mut sparse_idx = SporseIndex::new();
    for d in 0..n_docs {
        sparse_idx.insert(d, &sparse(d));
    }
    sparse_idx.build();

    // Query: dense direction matches DENSE_DECOY; sparse terms match SPARSE_DECOY.
    let dense_query = unit(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let sparse_query = SparseVec::new(vec![(0, 1.0), (1, 1.0), (2, 1.0)]);

    let dense_hits = hnsw.search(&dense_query, 5, 50).expect("dense search");
    let sparse_hits = sparse_idx.search(&sparse_query, 5);

    let show = |hits: &[(u32, f32)]| hits.iter().take(3).map(|h| h.0).collect::<Vec<_>>();
    println!("dense top-3:  {:?}", show(&dense_hits));
    println!("sparse top-3: {:?}", show(&sparse_hits));

    let fused = rrf(&sparse_hits, &dense_hits);
    println!("fused top-3:  {:?}", show(&fused));

    let dense_top = dense_hits[0].0;
    let sparse_top = sparse_hits[0].0;
    let fused_top = fused[0].0;

    // TARGET is #1 on neither signal but wins fusion: it is the only document
    // ranked highly by both the sparse and the dense retriever.
    assert_ne!(dense_top, TARGET, "TARGET should not be the dense #1");
    assert_ne!(sparse_top, TARGET, "TARGET should not be the sparse #1");
    assert_eq!(
        fused_top, TARGET,
        "the both-relevant document should win RRF fusion \
         (dense#1={dense_top}, sparse#1={sparse_top}, fused#1={fused_top})"
    );
    println!("  [PASS] doc {TARGET} wins fusion despite being #1 on neither signal alone");
}

//! Composite-Code Sparse Autoencoder (CCSA) end to end: train, encode, index,
//! retrieve.
//!
//! CCSA (Lassance, Formal, and Clinchant 2022, arXiv:2204.07023) learns a
//! C-hot composite code for each dense vector: one active dimension per chunk.
//! This example trains CCSA on clustered data, indexes the encoded documents,
//! and reports reconstruction error plus the retrieved document labels.
//!
//! Run: `cargo run --example ccsa_retrieval`

use sporse::sae::{standardize, train_ccsa, CcsaTrainConfig, CompositeCodeConfig};
use sporse::SporseIndex;

fn main() {
    let dim = 16;
    let n_clusters = 3;
    let per_cluster = 10;

    // Deterministic small noise.
    let mut state = 0x1234_5678u64;
    let mut noise = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };

    // Three clusters: cluster c has its signal in dimensions [c*4, c*4+4).
    let mut data: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for c in 0..n_clusters {
        for _ in 0..per_cluster {
            let mut v = vec![0.0f32; dim];
            for d in 0..4 {
                v[c * 4 + d] = 1.0 + 0.15 * noise();
            }
            for x in v.iter_mut() {
                *x += 0.05 * noise();
            }
            data.push(v);
            labels.push(c);
        }
    }

    // Standardize (BatchNorm's role in the paper), keeping the mean/std to apply
    // to the query.
    let (mean, std) = standardize(&mut data, dim);

    let cfg = CcsaTrainConfig {
        code: CompositeCodeConfig::new(4, 8), // 4 chunks of 8 -> 4-hot over 32 dims
        input_dim: dim,
        epochs: 300,
        lr: 0.1,
        temperature: 1.0,
        uniformity_weight: 0.1,
        gumbel_noise: false,
        seed: 42,
    };
    let model = train_ccsa(&data, &cfg);
    println!("reconstruction MSE: {:.4}", model.reconstruction_mse(&data));

    // Encode each document into its composite code and index it.
    let mut index = SporseIndex::new();
    for (i, x) in data.iter().enumerate() {
        let code = model.encode(x).expect("encode document");
        index.insert(i as u32, &code.to_sparse_vec());
    }
    index.build();

    // A fresh cluster-0 query, standardized with the training statistics.
    let mut q = vec![0.0f32; dim];
    for x in q.iter_mut().take(4) {
        *x = 1.0;
    }
    let q_std: Vec<f32> = (0..dim)
        .map(|d| (q[d] - mean[d]) / std[d].max(1e-6))
        .collect();
    let q_code = model.encode(&q_std).expect("encode query");

    let hits = index.search(&q_code.to_sparse_vec(), 3);
    let labeled: Vec<(u32, usize)> = hits
        .iter()
        .map(|&(id, _)| (id, labels[id as usize]))
        .collect();
    println!("top-3 (doc_id, cluster): {labeled:?}");

    // The assertion checks the top result's label for the cluster-0 query.
    let top = hits[0].0 as usize;
    assert_eq!(
        labels[top], 0,
        "expected a cluster-0 result via the encoded CCSA query"
    );
    println!("top result cluster: {}", labels[top]);
}

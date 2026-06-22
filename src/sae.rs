//! Composite-code sparse-autoencoder codes (CCSA; Lassance, Formal, and
//! Clinchant 2022, arXiv:2204.07023): the encoding and index-integration half.
//!
//! A CCSA encodes a dense vector into `C` chunks of size `L`, with exactly one
//! dimension active per chunk: a C-hot "composite code" over `D = C * L`
//! dimensions. Within each chunk the active dimension is the argmax of that
//! chunk's encoder logits (CCSA trains this with a Gumbel-Softmax
//! straight-through estimator; at inference the hard argmax is what the index
//! sees). The codes drive an inverted index, one posting list per dimension;
//! `CompositeCode::to_sparse_vec` adapts a code into the [`SparseVec`] the
//! existing [`SporseIndex`](crate::SporseIndex) already serves.
//!
//! Scope: this is the serving path (given encoder logits from a trained CCSA,
//! produce composite codes and index them). Training the encoder weights
//! (reconstruction + uniformity loss, Gumbel-Softmax straight-through, autodiff)
//! is a separate, heavier concern and is not implemented here.

use crate::SparseVec;

/// Composite-code geometry: `chunks` (C) chunks of `chunk_size` (L) each, so the
/// code lives over `D = C * L` dimensions and is C-hot (one active dim per chunk).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositeCodeConfig {
    /// Number of chunks C.
    pub chunks: usize,
    /// Dimensions per chunk L.
    pub chunk_size: usize,
}

impl CompositeCodeConfig {
    /// Create a config with `chunks` chunks of `chunk_size` dimensions each.
    pub fn new(chunks: usize, chunk_size: usize) -> Self {
        Self { chunks, chunk_size }
    }

    /// Total dimensionality `D = C * L`.
    pub fn dim(&self) -> usize {
        self.chunks * self.chunk_size
    }
}

/// A C-hot composite code: the active index (in `0..L`) for each of the C chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeCode {
    config: CompositeCodeConfig,
    active: Vec<usize>,
}

impl CompositeCode {
    /// Encode per-dimension encoder logits (length `D = C * L`) into a composite
    /// code by taking the argmax within each chunk. Returns `None` if `logits`
    /// has the wrong length or any chunk is empty.
    pub fn from_logits(logits: &[f32], config: CompositeCodeConfig) -> Option<Self> {
        if config.chunk_size == 0 || logits.len() != config.dim() {
            return None;
        }
        let l = config.chunk_size;
        let active = (0..config.chunks)
            .map(|c| argmax(&logits[c * l..(c + 1) * l]))
            .collect();
        Some(Self { config, active })
    }

    /// The active dimension index (`0..L`) for each chunk; length C.
    pub fn active(&self) -> &[usize] {
        &self.active
    }

    /// The code's geometry.
    pub fn config(&self) -> CompositeCodeConfig {
        self.config
    }

    /// The C-hot code as a [`SparseVec`] over the `D`-dimensional code space:
    /// active dim `= chunk * L + active_index`, weight `1.0`. Ready to index.
    pub fn to_sparse_vec(&self) -> SparseVec {
        let l = self.config.chunk_size;
        let pairs = self
            .active
            .iter()
            .enumerate()
            .map(|(c, &a)| ((c * l + a) as u32, 1.0))
            .collect();
        SparseVec::new(pairs)
    }
}

/// Index of the largest element; first on ties. Empty input returns 0 (callers
/// guarantee non-empty chunks via [`CompositeCode::from_logits`]).
fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Deterministic small-weight initializer (no `rand` dependency).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    /// Next value in roughly `[-0.1, 0.1]`.
    fn next_small(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
        u * 0.2 - 0.1
    }
}

/// Configuration for training a CCSA.
#[derive(Debug, Clone, Copy)]
pub struct CcsaTrainConfig {
    /// Composite-code geometry.
    pub code: CompositeCodeConfig,
    /// Dimensionality of the dense inputs.
    pub input_dim: usize,
    /// Number of full passes over the data.
    pub epochs: usize,
    /// Gradient-descent step size.
    pub lr: f32,
    /// Straight-through softmax temperature (the paper's tau).
    pub temperature: f32,
    /// Weight (lambda) on the uniformity / load-balance regularizer, which pushes
    /// per-dimension usage toward uniform and prevents code collapse. `0.0`
    /// disables it (reconstruction only).
    pub uniformity_weight: f32,
    /// Seed for deterministic weight initialization.
    pub seed: u64,
}

/// A trained Composite Code Sparse Autoencoder.
///
/// Holds the shallow linear encoder and decoder. [`encode`](Self::encode) maps a
/// dense input to a [`CompositeCode`] (the serving path); the decoder is kept for
/// reconstruction diagnostics.
///
/// This is the v1 core of CCSA (arXiv:2204.07023): a shallow linear
/// encoder/decoder, a hard composite code (argmax per chunk) on the forward
/// pass, a straight-through softmax on the backward pass, and an MSE
/// reconstruction objective trained by gradient descent, with an optional
/// uniformity / load-balance regularizer (`uniformity_weight`) that prevents
/// code collapse. Documented simplifications versus the paper: no input
/// BatchNorm (normalize externally) and no Gumbel sampling noise (deterministic
/// argmax). The gradients are hand-derived; the `train_reduces_mse` and
/// `uniformity_balances_usage` tests guard their correctness.
#[derive(Debug, Clone)]
pub struct CcsaModel {
    config: CompositeCodeConfig,
    input_dim: usize,
    w_enc: Vec<f32>, // D x input_dim, row-major
    w_dec: Vec<f32>, // D x input_dim, row-major
}

impl CcsaModel {
    /// Encode a dense input into a composite code with the trained encoder.
    /// Returns `None` if `x` has the wrong length.
    pub fn encode(&self, x: &[f32]) -> Option<CompositeCode> {
        if x.len() != self.input_dim {
            return None;
        }
        let m = self.input_dim;
        let logits: Vec<f32> = (0..self.config.dim())
            .map(|d| dot(&self.w_enc[d * m..d * m + m], x))
            .collect();
        CompositeCode::from_logits(&logits, self.config)
    }

    /// Reconstruct a dense input through encode then decode (sum of the active
    /// decoder rows).
    pub fn reconstruct(&self, x: &[f32]) -> Vec<f32> {
        let m = self.input_dim;
        let l = self.config.chunk_size;
        let code = self.encode(x).expect("input_dim matches");
        let mut out = vec![0.0f32; m];
        for (c, &a) in code.active().iter().enumerate() {
            let d = c * l + a;
            for (i, o) in out.iter_mut().enumerate() {
                *o += self.w_dec[d * m + i];
            }
        }
        out
    }

    /// Mean squared reconstruction error over `data`.
    pub fn reconstruction_mse(&self, data: &[Vec<f32>]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let m = self.input_dim;
        let mut se = 0.0f32;
        for x in data {
            let xh = self.reconstruct(x);
            for (xi, hi) in x.iter().zip(xh.iter()) {
                let delta = hi - xi;
                se += delta * delta;
            }
        }
        se / (data.len() * m) as f32
    }
}

/// Train a CCSA on dense inputs (each row length `cfg.input_dim`).
///
/// See [`CcsaModel`] for the method and its documented simplifications. The loss
/// is the mean squared reconstruction error; gradients flow through the decoder
/// from the hard code and through the encoder via the straight-through softmax.
// The inner loops index parallel arrays at an offset (`d * m + i`), which reads
// clearer with explicit indices than zipped iterators.
#[allow(clippy::needless_range_loop)]
pub fn train_ccsa(data: &[Vec<f32>], cfg: &CcsaTrainConfig) -> CcsaModel {
    let m = cfg.input_dim;
    let d_total = cfg.code.dim();
    let l = cfg.code.chunk_size;
    let c = cfg.code.chunks;
    let inv_tau = 1.0 / cfg.temperature;

    let mut rng = Lcg::new(cfg.seed);
    let mut w_enc: Vec<f32> = (0..d_total * m).map(|_| rng.next_small()).collect();
    let mut w_dec: Vec<f32> = (0..d_total * m).map(|_| rng.next_small()).collect();

    let n = data.len().max(1) as f32;
    let target_p = c as f32 / d_total as f32;
    for _ in 0..cfg.epochs {
        // Pass 1: per-dimension usage over the batch (for the uniformity term).
        let mut usage = vec![0.0f32; d_total];
        if cfg.uniformity_weight > 0.0 {
            for x in data {
                for cc in 0..c {
                    let z: Vec<f32> = (0..l)
                        .map(|j| {
                            let d = cc * l + j;
                            dot(&w_enc[d * m..d * m + m], x)
                        })
                        .collect();
                    usage[cc * l + argmax(&z)] += 1.0;
                }
            }
        }

        let mut g_enc = vec![0.0f32; d_total * m];
        let mut g_dec = vec![0.0f32; d_total * m];

        for x in data {
            // Forward: logits, per-chunk softmax (backward) and argmax (forward).
            let logits: Vec<f32> = (0..d_total)
                .map(|d| dot(&w_enc[d * m..d * m + m], x))
                .collect();
            let mut active = vec![0usize; c];
            let mut soft = vec![0.0f32; d_total];
            for cc in 0..c {
                let z = &logits[cc * l..cc * l + l];
                active[cc] = argmax(z);
                let mx = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = z.iter().map(|&v| ((v - mx) * inv_tau).exp()).collect();
                let s: f32 = exps.iter().sum::<f32>().max(f32::MIN_POSITIVE);
                for j in 0..l {
                    soft[cc * l + j] = exps[j] / s;
                }
            }

            // Decode: x_hat = sum of active decoder rows.
            let mut x_hat = vec![0.0f32; m];
            for (cc, &a) in active.iter().enumerate() {
                let d = cc * l + a;
                for i in 0..m {
                    x_hat[i] += w_dec[d * m + i];
                }
            }

            // MSE gradient w.r.t. x_hat.
            let scale = 2.0 / m as f32;
            let d_xhat: Vec<f32> = (0..m).map(|i| scale * (x_hat[i] - x[i])).collect();

            // Decoder gradient: only the active rows (hard code).
            for (cc, &a) in active.iter().enumerate() {
                let d = cc * l + a;
                for i in 0..m {
                    g_dec[d * m + i] += d_xhat[i];
                }
            }

            // d_code over all dimensions, then straight-through softmax to logits.
            let mut d_code: Vec<f32> = (0..d_total)
                .map(|d| dot(&w_dec[d * m..d * m + m], &d_xhat))
                .collect();
            // Uniformity gradient: push over-used dimensions down. The squared
            // deviation of usage from the uniform target C/D gives
            // dL_UR/dcode[j] = lambda * (2 / (D * B)) * (usage[j]/B - C/D).
            if cfg.uniformity_weight > 0.0 {
                let coef = cfg.uniformity_weight * 2.0 / (d_total as f32 * n);
                for j in 0..d_total {
                    d_code[j] += coef * (usage[j] / n - target_p);
                }
            }
            let mut d_logits = vec![0.0f32; d_total];
            for cc in 0..c {
                let g = &d_code[cc * l..cc * l + l];
                let s = &soft[cc * l..cc * l + l];
                let gs: f32 = g.iter().zip(s).map(|(a, b)| a * b).sum();
                for j in 0..l {
                    d_logits[cc * l + j] = inv_tau * s[j] * (g[j] - gs);
                }
            }

            // Encoder gradient.
            for d in 0..d_total {
                let dl = d_logits[d];
                if dl != 0.0 {
                    for i in 0..m {
                        g_enc[d * m + i] += dl * x[i];
                    }
                }
            }
        }

        for k in 0..d_total * m {
            w_enc[k] -= cfg.lr * g_enc[k] / n;
            w_dec[k] -= cfg.lr * g_dec[k] / n;
        }
    }

    CcsaModel {
        config: cfg.code,
        input_dim: m,
        w_enc,
        w_dec,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SporseIndex;

    #[test]
    fn composite_code_is_c_hot() {
        // C = 3 chunks of L = 4; argmax per chunk picks one dim each.
        let cfg = CompositeCodeConfig::new(3, 4);
        // chunk0 argmax @1, chunk1 @3, chunk2 @0.
        let logits = [
            0.0, 9.0, 1.0, 2.0, // chunk 0 -> 1
            1.0, 1.0, 2.0, 5.0, // chunk 1 -> 3
            8.0, 0.0, 0.0, 7.0, // chunk 2 -> 0
        ];
        let code = CompositeCode::from_logits(&logits, cfg).unwrap();
        assert_eq!(code.active(), &[1, 3, 0]);

        // Exactly C active dimensions at the right global positions.
        let sv = code.to_sparse_vec();
        // global dims: 0*4+1=1, 1*4+3=7, 2*4+0=8.
        let mut idx = SporseIndex::new();
        idx.insert(0, &sv);
        idx.build();
        // A query hitting one of the active dims retrieves the doc.
        let q = SparseVec::new(vec![(7, 1.0)]);
        let hits = idx.search(&q, 1);
        assert_eq!(hits.first().map(|(d, _)| *d), Some(0));
    }

    #[test]
    fn rejects_wrong_length() {
        let cfg = CompositeCodeConfig::new(2, 3); // dim = 6
        assert!(CompositeCode::from_logits(&[0.0; 5], cfg).is_none());
        assert!(CompositeCode::from_logits(&[0.0; 6], cfg).is_some());
        assert_eq!(cfg.dim(), 6);
    }

    #[test]
    fn round_trip_through_index_ranks_matching_doc_first() {
        let cfg = CompositeCodeConfig::new(2, 4);
        // doc A: chunks -> [0, 0]; doc B: chunks -> [3, 3].
        let a = CompositeCode::from_logits(&[5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0], cfg).unwrap();
        let b = CompositeCode::from_logits(&[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0], cfg).unwrap();
        let mut idx = SporseIndex::new();
        idx.insert(0, &a.to_sparse_vec());
        idx.insert(1, &b.to_sparse_vec());
        idx.build();
        // Query shaped like A retrieves A above B.
        let hits = idx.search(&a.to_sparse_vec(), 2);
        assert_eq!(hits.first().map(|(d, _)| *d), Some(0));
    }

    #[test]
    fn train_reduces_mse() {
        // Distinct dense vectors; an 8-dim composite code (C=2, L=4) has enough
        // capacity to reconstruct them. If the straight-through gradients are
        // wrong, training does not reduce reconstruction MSE.
        let data: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.2],
            vec![0.0, 1.0, 0.0, 0.0, 0.1, 0.9],
            vec![0.0, 0.0, 1.0, 0.0, 0.7, 0.3],
            vec![0.0, 0.0, 0.0, 1.0, 0.4, 0.6],
            vec![0.5, 0.5, 0.0, 0.0, 0.2, 0.2],
            vec![0.0, 0.0, 0.5, 0.5, 0.8, 0.1],
        ];
        let base = CcsaTrainConfig {
            code: CompositeCodeConfig::new(2, 4),
            input_dim: 6,
            epochs: 0,
            lr: 0.5,
            temperature: 1.0,
            uniformity_weight: 0.0,
            seed: 42,
        };
        let initial = train_ccsa(&data, &base).reconstruction_mse(&data);
        let trained = train_ccsa(
            &data,
            &CcsaTrainConfig {
                epochs: 400,
                ..base
            },
        )
        .reconstruction_mse(&data);
        assert!(
            trained < 0.5 * initial,
            "training should cut reconstruction MSE: {initial} -> {trained}"
        );
    }

    #[test]
    fn trained_encoder_feeds_the_index() {
        let data: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.2], vec![0.0, 1.0, 0.2, 0.0]];
        let cfg = CcsaTrainConfig {
            code: CompositeCodeConfig::new(2, 3),
            input_dim: 4,
            epochs: 50,
            lr: 0.3,
            temperature: 1.0,
            uniformity_weight: 0.0,
            seed: 7,
        };
        let model = train_ccsa(&data, &cfg);
        let code = model.encode(&data[0]).unwrap();
        assert_eq!(code.active().len(), 2); // C-hot
        let mut idx = SporseIndex::new();
        idx.insert(0, &code.to_sparse_vec());
        idx.build();
        let hits = idx.search(&code.to_sparse_vec(), 1);
        assert_eq!(hits.first().map(|(d, _)| *d), Some(0));
    }

    #[test]
    fn uniformity_balances_usage() {
        // Many similar inputs: reconstruction alone tends to map them to the same
        // code (collapse), concentrating dimension usage. The uniformity
        // regularizer should spread usage and lower its variance.
        let data: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                let t = i as f32 * 0.02;
                vec![1.0 + t, 0.5 - t, 0.2 + t, 0.8 - t]
            })
            .collect();
        let base = CcsaTrainConfig {
            code: CompositeCodeConfig::new(2, 4),
            input_dim: 4,
            epochs: 300,
            lr: 0.3,
            temperature: 1.0,
            uniformity_weight: 0.0,
            seed: 1,
        };
        let usage_variance = |w: f32| -> f32 {
            let model = train_ccsa(
                &data,
                &CcsaTrainConfig {
                    uniformity_weight: w,
                    ..base
                },
            );
            let d = base.code.dim();
            let l = base.code.chunk_size;
            let mut usage = vec![0.0f32; d];
            for x in &data {
                for (c, &a) in model.encode(x).unwrap().active().iter().enumerate() {
                    usage[c * l + a] += 1.0;
                }
            }
            let mean = usage.iter().sum::<f32>() / d as f32;
            usage.iter().map(|u| (u - mean).powi(2)).sum::<f32>() / d as f32
        };
        let var_plain = usage_variance(0.0);
        let var_uniform = usage_variance(5.0);
        assert!(
            var_uniform < var_plain,
            "uniformity should balance usage: variance {var_plain} -> {var_uniform}"
        );
    }
}

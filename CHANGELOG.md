# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `sae` module: Composite Code Sparse Autoencoder (CCSA; Lassance, Formal, and
  Clinchant 2022, arXiv:2204.07023), end to end. `CompositeCode::from_logits`
  turns per-chunk encoder logits into a C-hot code (argmax over `C` chunks of
  size `L`) and `to_sparse_vec` feeds the existing index. `train_ccsa` trains the
  shallow linear encoder/decoder with hand-derived straight-through gradients and
  gradient descent (dependency-free f32; no autodiff framework), returning a
  `CcsaModel` whose `encode` produces codes for the index. v1 core: documented
  simplifications versus the paper are no input BatchNorm, no Gumbel sampling
  noise (deterministic argmax), and no uniformity regularizer. A
  `train_reduces_mse` test guards the gradient derivation.
- Expanded examples for impact-score walkthroughs, WAND diagnostics, and serde
  round-trips.
- Restored the public `innr` feature as a compatibility no-op after the
  optional dependency was removed.

## [0.3.0] - 2026-04-20

### Added

- serde support for `SporseIndex` (full index serialization) and `SparseVec`, with round-trip tests.
- `SparseVec::dot` with a randomized brute-force parity test.
- Optional `innr` dependency.

### Changed

- Optimized WAND search with a `BinaryHeap` top-k path and fewer posting-list allocations.

## [0.2.0] - 2026-04-14

### Added

- Block-Max WAND inverted index.
- serde feature for `SparseVec` serialization.

## [0.1.0] - 2026-04-14

### Added

- Initial sparse vector index.

[0.3.0]: https://github.com/arclabs561/sporse/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/arclabs561/sporse/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/arclabs561/sporse/releases/tag/v0.1.0

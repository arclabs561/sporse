# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.2] - 2026-06-27

### Added

- `store::UpdatableIndex::compact_tiers()`: one round of size-tiered compaction
  (merge similarly-sized segments), keeping segment count bounded without a full
  `compact()`.

## [0.6.1] - 2026-06-27

### Added

- `store::UpdatableIndex::reclaim(min_live_ratio)` and `space_amplification()`
  (via the new `Store::live_len`): cheap tombstone reclamation, merging only the
  delete-heavy segments instead of a full compaction.

## [0.6.0] - 2026-06-27

### Changed

- `store::UpdatableIndex` now caches each segment's index by the segment's stable
  `Arc` identity (via segstore 0.2), so a mutation rebuilds only the new or
  changed segments instead of the whole corpus on the next query.
- Requires `segstore` 0.2 (only affects the optional `store` feature; the on-disk
  store format changed, so a `store` index written by 0.5.x is not read by 0.6.0).

## [0.5.1] - 2026-06-26

### Fixed

- `store::UpdatableIndex` no longer rebuilds the per-segment indexes on every
  query. They are cached and rebuilt only when a mutation (add/delete/compact)
  occurs, so query cost no longer grows with the corpus on each call.

## [0.5.0] - 2026-06-26

### Added

- Optional `store` feature: `store::UpdatableIndex`, an updatable, durable
  learned-sparse index backed by [`segstore`](https://crates.io/crates/segstore)
  (write-ahead log, checkpoint, compaction, crash recovery), driving the
  Block-Max WAND engine per segment. Opt-in; the default build does not depend on
  segstore.

## [0.4.0] - 2026-06-24

### Added

- `sae` module: Composite Code Sparse Autoencoder (CCSA; Lassance, Formal, and
  Clinchant 2022, arXiv:2204.07023), end to end. `CompositeCode::from_logits`
  turns per-chunk encoder logits into a C-hot code (argmax over `C` chunks of
  size `L`) and `to_sparse_vec` feeds the existing index. `train_ccsa` trains the
  shallow linear encoder/decoder with hand-derived straight-through gradients and
  gradient descent (dependency-free f32; no autodiff framework), returning a
  `CcsaModel` whose `encode` produces codes for the index. Optional
  uniformity / load-balance regularization (`uniformity_weight`) spreads
  per-dimension usage and prevents code collapse; optional Gumbel-Softmax
  exploration (`gumbel_noise`) adds stochastic code selection during training
  (reproducible given the seed). A `standardize` helper provides the input
  normalization that BatchNorm plays in the paper. `train_reduces_mse`,
  `uniformity_balances_usage`, and the Gumbel reproducibility test guard the
  gradient derivations.
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

[0.4.0]: https://github.com/arclabs561/sporse/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/arclabs561/sporse/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/arclabs561/sporse/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/arclabs561/sporse/releases/tag/v0.1.0

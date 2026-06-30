# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.5] - 2026-06-30

### Added

- The `store` feature now persists per-segment WAND sidecars containing finalized
  posting lists and block-max metadata, and loads them on restart when the recipe
  and live id set still match. On `benches/store.rs` (`N_DOCS=20_000`,
  `VOCAB=30_000`, `DOC_NNZ=80`, `FLUSH=2_000`), cold restart search with
  sidecars measured `[31.390 ms 31.937 ms 32.538 ms]` versus
  `[105.88 ms 107.20 ms 108.85 ms]` when sidecars were missing and every segment
  had to rebuild.

### Changed

- The `store` feature now requires `segstore = "0.4"` and `postcard`.

## [0.6.4] - 2026-06-28

### Added

- `store::UpdatableIndex::extend(docs)`: bulk ingest that syncs the write-ahead log
  once per batch instead of once per document. ~2.1x faster than a loop of `add`
  for a corpus load on a real filesystem (bench `ingest_fs`: 17.9ms vs 8.4ms / 4000
  docs).

### Changed

- The `store` feature now requires `segstore = "0.3"`; the internal `merge_segments`
  takes `&[&Segment]` (segstore 0.3's by-reference signature).

## [0.6.3] - 2026-06-27

### Changed

- A `delete` now invalidates only the cached index of the segment that holds the
  id, not the whole cache, so one delete no longer forces every segment to
  rebuild on the next query.

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

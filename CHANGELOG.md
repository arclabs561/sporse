# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a `raw_impact_generation` example showing quantized sparse-vector
  ingestion into sealed raw files plus one live shard, with a persisted
  quantizer scale sidecar.

## [0.6.7] - 2026-07-04

### Added

- Added `RawImpactQuantizer` as a checked raw-impact scale value for encoding
  document and query impacts with the same persisted scale.
- Added integration-test parity coverage for quantized raw-impact search across
  sealed raw files plus one live numeric postings shard.
- Added raw-impact benchmark coverage for building and sealing a live numeric
  postings shard, and for searching sealed raw files together with that live
  shard.
- Added search benchmark coverage for quantized `postings::raw` u32 impact
  files, including a diagnostic top-k agreement check against brute force.
- Added quantized raw-impact recall sweeps across query densities, so the
  byte-backed path is checked beyond one SPLADE-style query shape.
- Added `store::{UpdatableIndex, UpdatableReader, UpdatableView}::search_with_stats`
  for segment-level search diagnostics: sealed segments seen, segments searched,
  segments pruned, and writer-buffer pruning.
- Added store benchmark diagnostics that report writer and reader segment
  pruning on a partitioned multi-segment corpus.
- Added multi-file quantized raw-impact parity coverage against the quantized
  sparse oracle.
- Added multi-file raw-impact benchmark and diagnostics alongside the
  single-file raw-impact path.
- Added raw-impact multi-file segment-pruning counts to the search diagnostics
  benchmark, including a partitioned fixture that exercises segment pruning.
- Added raw-impact file-build benchmarks for single-file, multi-file, and
  partitioned segment construction policies.
- Added `store::SnapshotIndex`, a read-only checkpoint view that opens
  segstore's manifest and queries persisted per-segment WAND sidecars before
  falling back to one source sparse-vector segment decode on a sidecar miss.
- Added `SparseVec::to_raw_impact_document` and
  `SparseVec::to_raw_impact_query` for zero-dependency quantized-impact pairs
  that callers can write to `postings::raw`.
- Added a `raw_impact_file` example that writes quantized `SparseVec` impacts to
  a `postings::raw` file segment and queries it.

### Changed

- Changed raw-impact dev/bench coverage to use `postings` 0.2.11, exercising
  live-seeded sealed-file pruning in files-plus-live searches.
- Changed `SparseVec::new` to sum duplicate dimensions deterministically instead
  of keeping one duplicate after sorting.
- Changed store search to skip finite zero-bound segment and buffer indexes
  before calling per-segment WAND.
- Changed cold store writer searches to order sealed segments by query upper
  bound immediately after loading or building per-segment indexes.
- Changed store writer searches to build the temporary writer-buffer index from
  the buffer slice instead of cloning buffered sparse vectors first.
- Changed raw-impact benchmark setup to stream sorted postings raw documents
  directly to temp files instead of building a full raw segment byte vector
  before writing.
- Changed the `raw_impact_file` example to seal a live numeric postings shard
  into a raw segment file before querying it.
- The `store` feature now requires `segstore = "0.4.1"` for manifest-only
  snapshot reads. This remains fully optional; default builds do not depend on
  the storage stack.

## [0.6.6] - 2026-07-03

### Added

- `postings_bridge` example showing how to convert live weighted `postings`
  entries into `SparseVec` documents without introducing a shared cursor trait.
- `store::UpdatableIndex::reader()` with cloneable checkpoint-visible
  `UpdatableReader`/`UpdatableView` searches for concurrent read paths.

### Changed

- `SporseIndex::search` now falls back to exact sparse accumulation when either
  indexed vectors or query vectors contain negative or non-finite weights.
  Block-Max WAND remains the hot path for finite non-negative learned-sparse
  vectors, but the public API no longer silently relies on that precondition.
- Block-Max WAND now uses 32-doc blocks, tightening per-block upper bounds for
  SPLADE-style sparse search. Existing `store` sidecars rebuild once because the
  block size is part of the persisted sidecar recipe.
- `store::UpdatableIndex` now carries the current global top-k threshold into
  later per-segment WAND searches. Warm cached searches also visit higher-bound
  segments first and skip segments that cannot beat the already-known kth score.
- Block-Max WAND now advances every cursor behind the pivot on memory-resident
  postings, reducing repeated pivot/sort cycles without changing exact top-k
  results.

### Fixed

- Block-Max WAND could drop true top-k results: pivot selection used the
  current block's maximum as a term's upper bound while skips crossed block
  boundaries, so a high-weight posting in a later block could be skipped
  permanently (a document scoring 100.0 could lose to one scoring 0.2, and the
  long skewed lists of real learned-sparse corpora are exactly the triggering
  regime). Pivot selection now uses global per-term maxima, the sound classic
  WAND bound; per-block maxima remain as a scoring-time refinement on the
  pivot document, so block-max pruning still skips non-competitive scoring.
  Exactness is pinned by a minimal-counterexample regression test and a
  multi-block randomized brute-force parity test.
- Search results are now sorted by score descending (ties by doc id) even when
  fewer than `k` documents match, on both the writer and reader store paths;
  previously below-`k` results came back in segment order.
- `store::UpdatableIndex` now keys its in-memory per-segment cache by segstore's
  stable segment ids instead of `Arc` addresses, avoiding stale WAND indexes
  after compaction/reclaim if the allocator reuses a freed segment address.
- `store::UpdatableIndex::{compact, compact_tiers, reclaim}` now persist sidecars
  for newly merged segments immediately after segstore checkpoints them, instead
  of waiting for the next search to rebuild and write the sidecar lazily.

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

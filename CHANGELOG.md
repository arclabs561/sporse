# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Expanded examples for impact-score walkthroughs, WAND diagnostics, and serde
  round-trips.

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

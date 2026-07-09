# sporse

Sparse vector index for learned sparse retrieval.

Indexes sparse vectors using an inverted index, whether the dimensions are
lexical (SPLADE-style vocabulary-term weights) or latent (the composite codes
the `sae` module learns). Top-k inner product search uses Block-Max WAND, a
safe dynamic-pruning traversal: it skips
documents that provably cannot enter the top-k, so the result is identical to an
exhaustive scan over the same scores (the `wand_diagnostics` example verifies
this parity).

The `sae` module also *learns* sparse codes rather than only serving them: a
Composite-Code Sparse Autoencoder (CCSA, arXiv:2204.07023) encodes dense vectors
into C-hot composite codes for this same index (the `ccsa_retrieval` example).

## Install

```toml
[dependencies]
sporse = "0.7"
```

## Usage

```rust
use sporse::{SparseVec, SporseIndex};

let mut index = SporseIndex::new();

index.insert(0, &SparseVec::new(vec![(0, 1.0), (3, 2.5), (7, 0.8)]));
index.insert(1, &SparseVec::new(vec![(1, 3.0), (3, 1.0)]));
index.insert(2, &SparseVec::new(vec![(0, 0.5), (7, 2.0)]));

index.build();

let query = SparseVec::new(vec![(0, 1.0), (3, 1.0)]);
let results = index.search(&query, 2);
// [(0, 3.5), (1, 1.0)] -- doc 0 scores highest
```

## Examples

Runnable examples live in [`examples/`](examples/):

- `basic.rs` prints score contributions for a small impact-score collection.
- `postings_bridge.rs` converts live weighted `postings` entries into
  `SparseVec` documents and checks score parity against direct postings
  accumulation.
- `raw_impact_file.rs` writes quantized `SparseVec` impacts to a
  `postings::raw` file segment and queries the file-backed segment.
- `raw_impact_generation.rs` writes quantized sparse vectors into sealed raw
  files plus one live shard, reloads the persisted scale sidecar, and queries
  the combined generation.
- `wand_diagnostics.rs` checks exact top-k parity against brute force and reports
  how many documents Block-Max WAND scored.
- `serde_roundtrip.rs` verifies built-index serialization with the `serde`
  feature.
- `hybrid_retrieval.rs` composes this sparse index with vicinity's dense HNSW and
  rankops reciprocal-rank fusion.
- `ccsa_retrieval.rs` trains a Composite-Code Sparse Autoencoder (CCSA), encodes
  documents into C-hot codes, indexes them, and reports retrieved cluster
  labels.

## CLI

With the `cli` feature, the `sporse` binary builds and queries an index from
files (one `{"id": u32, "vec": [[dim, weight], ...]}` per line):

```sh
cargo run --features cli --bin sporse -- build docs.jsonl -o index.json
cargo run --features cli --bin sporse -- search index.json --query '[[0, 1.0], [3, 1.0]]' -k 5
```

## Features

- `serde` -- Serialize/Deserialize for `SparseVec` and built `SporseIndex`
- `cli` -- the `sporse` binary (build and query an index from files)
- `store` -- `store::UpdatableIndex`: an updatable, durable index (incremental
  add/delete, write-ahead log, checkpoint, compaction, crash recovery) backed by
  [`segstore`](https://crates.io/crates/segstore). Per-segment indexes are cached
  and persisted as sidecars, so restart loads finalized posting lists and
  block-max metadata instead of rebuilding unchanged segments. `reader()` returns
  cloneable checkpoint-visible snapshot views for concurrent searches, and
  `search_with_stats` reports segment-level pruning diagnostics.
  `store::SnapshotIndex` opens the last checkpoint manifest and queries sidecars
  first, so source sparse-vector batches are read only when a sidecar is missing
  or unusable. Fully out-of-core learned-sparse search needs byte-native sparse
  segment sidecars. `RawImpactQuantizer` carries the scale callers persist with
  a raw impact generation, and `SparseVec::to_raw_impact_document` /
  `SparseVec::to_raw_impact_query` remain available for direct conversion.
  `RawImpactQuantizer::score_error_bound` gives the per-query score bound from
  document-weight rounding, which is useful when choosing or evaluating a scale.
  Callers write those pairs to `postings::raw` without `sporse` owning the file
  lifecycle.
  `postings::raw` covers `u32` impact-score segments, while native `SparseVec`
  weights are `f32`. The current test suite covers recall sweeps across scales
  and query densities, but that path is not public storage API yet.

For store restart measurement, `cargo run --release --features store --example store_reopen_diagnostics`
prints first snapshot-query cost with persisted WAND sidecars present versus
after deleting those sidecars and forcing source-segment rebuilds.

For search-quality and pruning measurements, run:

```sh
cargo bench --bench search -- diagnostics_noop --warm-up-time 0.1 --measurement-time 0.1 --sample-size 10
```

The diagnostic pass checks WAND, raw-file, multi-file raw, and files-plus-live
raw top-k parity against brute force, then prints average documents scored,
cursor skips, and raw segment pruning counts. It also compares raw segment
layout policies on the same synthetic documents: in the current fixture,
random files search/prune 4.0/0.0 segments, vocab-local partitioned files
search/prune 1.0/3.0, and an interleaved partitioned control searches/prunes
4.0/0.0.

## References

- Broder, Carmel, Herscovici, Soffer, and Zien, "Efficient Query Evaluation
  using a Two-Level Retrieval Process" (CIKM 2003). The original WAND traversal.
- Ding and Suel, "Faster Top-k Document Retrieval Using Block-Max Indexes"
  (SIGIR 2011). The block-max refinement this crate implements.
- Formal, Piwowarski, and Clinchant, "SPLADE: Sparse Lexical and Expansion Model
  for First Stage Ranking" (arXiv:2107.05720). The lexical learned sparse
  representations this index is built to serve.
- Lassance, Formal, and Clinchant, "Composite Code Sparse Autoencoders for First
  Stage Retrieval" (arXiv:2204.07023). The latent C-hot composite codes the
  `sae` module learns.

## License

MIT OR Apache-2.0

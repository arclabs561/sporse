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

- `serde` -- Serialize/Deserialize for `SparseVec`
- `cli` -- the `sporse` binary (build and query an index from files)
- `store` -- `store::UpdatableIndex`: an updatable, durable index (incremental
  add/delete, write-ahead log, checkpoint, compaction, crash recovery) backed by
  [`segstore`](https://crates.io/crates/segstore). Per-segment indexes are cached
  and persisted as sidecars, so restart loads finalized posting lists and
  block-max metadata instead of rebuilding unchanged segments. `reader()` returns
  cloneable checkpoint-visible snapshot views for concurrent searches.

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

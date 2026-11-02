# sporse

Sparse vector index for learned sparse retrieval.

Indexes sparse vectors using an inverted index, whether the dimensions are
lexical (SPLADE-style vocabulary-term weights) or latent (the composite codes
the `sae` module learns). Top-k inner product search uses WAND with block-max
score bounds, but does not use Block-Max WAND’s shallow cursor moves. It skips
documents that provably cannot enter the top-k. Results match an exhaustive
scan over the same scores, with ties at the k-th score going to the lowest
document IDs. The `wand_diagnostics` example verifies this parity.

The `sae` module also *learns* sparse codes rather than only serving them: a
Composite-Code Sparse Autoencoder (CCSA, arXiv:2204.07023) encodes dense vectors
into C-hot composite codes for this same index (the `ccsa_retrieval` example).

## Install

```toml
[dependencies]
sporse = "0.9"
```

The default library supports Rust 1.80. The optional `store` feature requires
Rust 1.83 because its CRC dependency requires that compiler version.

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
assert_eq!(results, vec![(0, 3.5), (1, 1.0)]);
```

Before `build`, inserting an existing document ID replaces its entire vector.
`upsert` does the same and returns `InsertOutcome::Inserted` or
`InsertOutcome::Replaced`. Empty vectors still count as documents, and `len()`
counts distinct IDs. Built indexes are immutable. Search returns finite,
positive scores, preferring lower document IDs on ties.

With `serde`, only built indexes serialize; serializing or loading an unbuilt
index returns an error. Saved indexes that contain repeated IDs need
rebuilding from their source documents to get replacement semantics.

## Examples

[`examples/README.md`](https://github.com/arclabs561/sporse/blob/main/examples/README.md)
lists each example with its run command. `basic` prints the term contributions
behind each top result:

```sh
cargo run --release --example basic
```

```text
index: 6 documents, 13 sparse dimensions
query: sparse retrieval with WAND impact scoring

1. SPLADE retrieval with inverted indexes  score=11.220
   retrieval        4.620
   sparse           4.560
   wand             2.040
```

`wand_diagnostics` checks top-k parity against brute force and reports how many
documents WAND fully scored (12/12 queries match; 6.11% of 4,000 documents
scored on its synthetic collection).

## CLI

With the `cli` feature, the `sporse` binary builds and queries an index from
files (one `{"id": u32, "vec": [[dim, weight], ...]}` per line):

```sh
cargo run --features cli --bin sporse -- build docs.jsonl -o index.json
cargo run --features cli --bin sporse -- search index.json --query '[[0, 1.0], [3, 1.0]]' -k 5
```

## Features

| Feature | What it enables |
|---|---|
| `serde` | `Serialize`/`Deserialize` for `SparseVec` and built `SporseIndex` |
| `cli` | The `sporse` binary |
| `store` | `store::UpdatableIndex` and `store::SnapshotIndex` |

`store::UpdatableIndex` is an updatable, durable index backed by
[`segstore`](https://crates.io/crates/segstore): incremental add and delete,
write-ahead log, checkpoint, compaction, and recovery. Each segment's built
index is persisted as a sidecar, so a restart loads posting lists and block-max
metadata instead of rebuilding unchanged segments. `reader()` returns cloneable
snapshot views for concurrent searches. `store::SnapshotIndex` is a read-only
view of the last checkpoint.
`cargo run --release --features store --example store_reopen_diagnostics`
times the first query after reopening, with and without the sidecars.

Without any feature, `RawImpactQuantizer` and
`SparseVec::to_raw_impact_document`/`to_raw_impact_query` quantize document
weights to the `u32` impacts of a `postings::raw` segment and rescale queries
to match;
`score_error_bound` bounds the per-query score error from that rounding.
`sporse` does not write or own those files.

For pruning and parity measurements on synthetic data:

```sh
cargo bench --bench search -- diagnostics_noop --warm-up-time 0.1 --measurement-time 0.1 --sample-size 10
```

It checks WAND and raw-segment top-k against brute force, then prints average
documents scored, cursor skips, and raw-segment pruning counts.

## References

- Broder, Carmel, Herscovici, Soffer, and Zien, "Efficient Query Evaluation
  using a Two-Level Retrieval Process" (CIKM 2003). The original WAND traversal.
- Ding and Suel, "Faster Top-k Document Retrieval Using Block-Max Indexes"
  (SIGIR 2011). The source of the block-max bounds this crate uses; the
  traversal here skips per-document scoring with them but does not move
  cursors shallowly the way BMW does.
- Formal, Piwowarski, and Clinchant, "SPLADE: Sparse Lexical and Expansion Model
  for First Stage Ranking" (arXiv:2107.05720). The lexical learned sparse
  representations this index is built to serve.
- Lassance, Formal, and Clinchant, "Composite Code Sparse Autoencoders for First
  Stage Retrieval" (arXiv:2204.07023). The latent C-hot composite codes the
  `sae` module learns.

## License

MIT OR Apache-2.0

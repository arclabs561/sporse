# sporse

Sparse vector index for learned sparse retrieval.

Indexes sparse vectors (SPLADE, LADE) using an inverted index. Top-k inner
product search uses Block-Max WAND, a safe dynamic-pruning traversal: it skips
documents that provably cannot enter the top-k, so the result is identical to an
exhaustive scan over the same scores (the `wand_diagnostics` example verifies
this parity).

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
- `wand_diagnostics.rs` checks exact top-k parity against brute force and reports
  how many documents Block-Max WAND scored.
- `serde_roundtrip.rs` verifies built-index serialization with the `serde`
  feature.

## Features

- `serde` -- Serialize/Deserialize for `SparseVec`

## References

- Broder, Carmel, Herscovici, Soffer, and Zien, "Efficient Query Evaluation
  using a Two-Level Retrieval Process" (CIKM 2003). The original WAND traversal.
- Ding and Suel, "Faster Top-k Document Retrieval Using Block-Max Indexes"
  (SIGIR 2011). The block-max refinement this crate implements.
- Formal, Piwowarski, and Clinchant, "SPLADE: Sparse Lexical and Expansion Model
  for First Stage Ranking" (arXiv:2107.05720). The learned sparse
  representations this index is built to serve.

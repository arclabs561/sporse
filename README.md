# sporse

Sparse vector index for learned sparse retrieval.

Indexes sparse vectors (SPLADE, LADE) using an inverted index with
Block-Max WAND traversal for exact top-k inner product search.

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

# Examples

The examples use synthetic impact-score vectors: dimensions are token or latent
feature ids, weights are non-negative retrieval scores.

| Purpose | Example | Output |
| --- | --- | --- |
| See how scores are formed | `basic.rs` | Top results list per-term score contributions. |
| Check Block-Max WAND pruning | `wand_diagnostics.rs` | WAND top-k matches brute force while scoring a fraction of documents. |
| Verify serialization | `serde_roundtrip.rs` | Search results are identical after JSON round trip. |

## Score Contributions

`basic.rs` walks through a small learned-sparse collection and prints the term
contributions behind each top result. It is the shortest inspection path for
the inner-product scoring model.

```sh
cargo run --release --example basic
```

The output lists ranked documents and the matched sparse dimensions that
contribute to each score.

Expected excerpt:

```text
index: 6 documents, 13 sparse dimensions
query: sparse retrieval with WAND impact scoring

1. SPLADE retrieval with inverted indexes  score=11.220
   retrieval        4.620
   sparse           4.560
   wand             2.040
```

## WAND Diagnostics

`wand_diagnostics.rs` builds a larger synthetic index, checks exact top-k
parity against brute force, and prints how many documents Block-Max WAND fully
scored. The exact numbers are deterministic because the example uses a fixed
seed.

```sh
cargo run --release --example wand_diagnostics
```

The output reports exact top-k parity against brute force, average WAND loop
iterations, fully-scored document count, and cursor skips.

Expected output:

```text
index: 4000 docs, 24000 sparse dimensions, 96 nnz/doc
queries: 12, 40 nnz/query, top-10
exact top-k parity with brute force: 12/12
average WAND iterations: 576.8
average fully-scored docs: 244.4 (6.11% of collection)
average cursor skips: 331.5
```

## Serialization

`serde_roundtrip.rs` serializes a built index and verifies that deserialized
search results are identical.

```sh
cargo run --release --features serde --example serde_roundtrip
```

The output reports the serialized index size and the restored query results.

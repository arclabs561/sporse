# sporse examples

The examples use synthetic impact-score vectors: dimensions are token or latent
feature ids, weights are non-negative retrieval scores.

## Which example should I run?

| I want to... | Example | Run |
|---|---|---|
| See how inner-product scores are formed | `basic` | `cargo run --release --example basic` |
| Convert live weighted postings into sparse vectors | `postings_bridge` | `cargo run --release --example postings_bridge` |
| Write sparse vectors as raw impact files | `raw_impact_file` | `cargo run --release --example raw_impact_file` |
| Check Block-Max WAND against brute force | `wand_diagnostics` | `cargo run --release --example wand_diagnostics` |
| Combine sparse and dense retrieval | `hybrid_retrieval` | `cargo run --release --example hybrid_retrieval` |
| Train sparse codes, then index them | `ccsa_retrieval` | `cargo run --release --example ccsa_retrieval` |
| Verify built-index serialization | `serde_roundtrip` | `cargo run --release --features serde --example serde_roundtrip` |

## Score Contributions

`basic.rs` walks through a small learned-sparse collection and prints the term
contributions behind each top result. It is the shortest inspection path for
the inner-product scoring model.

```sh
cargo run --release --example basic
```

The output lists ranked documents and the matched sparse dimensions that
contribute to each score.

Output excerpt:

```text
index: 6 documents, 13 sparse dimensions
query: sparse retrieval with WAND impact scoring

1. SPLADE retrieval with inverted indexes  score=11.220
   retrieval        4.620
   sparse           4.560
   wand             2.040
```

## Postings Bridge

`postings_bridge.rs` adapts a mutable `postings::PostingsIndex<String, f32>`
into built `sporse::SparseVec` documents by assigning stable term dimensions
and reconstructing one sparse vector per live document.

The example checks that direct weighted-postings scoring and `sporse` search
produce the same top-k ranking. The adapter is intentionally local to the
example; a shared cursor trait still needs two real consumers before becoming
public API.

Output excerpt:

```text
postings top-k: [(2, 8.45), (3, 4.8), (0, 2.6999998)]
sporse top-k:   [(2, 8.45), (3, 4.8), (0, 2.6999998)]
```

## Raw Impact File

`raw_impact_file.rs` converts `SparseVec` documents with
`SparseVec::to_raw_impact_document`, writes a `postings::raw` segment to a temp
file, then queries that file-backed segment with
`SparseVec::to_raw_impact_query`.

The example keeps publication, fsync policy, manifests, deletes, and compaction
outside `sporse`; it only demonstrates the data conversion and raw segment
query boundary.

Output excerpt:

```text
top-k from postings::raw:
  doc 101: 5.120  learned sparse retrieval
```

## WAND Diagnostics

`wand_diagnostics.rs` builds a larger synthetic index, checks exact top-k
parity against brute force, and prints how many documents Block-Max WAND fully
scored. The exact numbers are deterministic because the example uses a fixed
seed.

The output reports exact top-k parity against brute force, average WAND loop
iterations, fully-scored document count, and cursor skips.

Output:

```text
index: 4000 docs, 24000 sparse dimensions, 96 nnz/doc
queries: 12, 40 nnz/query, top-10
exact top-k parity with brute force: 12/12
average WAND iterations: 576.8
average fully-scored docs: 244.4 (6.11% of collection)
average cursor skips: 331.5
```

## Hybrid Retrieval

`hybrid_retrieval.rs` composes sparse search with `vicinity` dense HNSW and
`rankops` reciprocal-rank fusion. The output reports the two source rankings and
the fused top document.

Output:

```text
dense top-2:  [0, 2]
sparse top-2: [1, 2]
fused rank-1 doc: 2
```

## CCSA Retrieval

`ccsa_retrieval.rs` trains a small Composite-Code Sparse Autoencoder, converts
dense vectors into C-hot sparse codes, and indexes those codes with the same
`SporseIndex` API used by SPLADE-style vectors.

Output:

```text
reconstruction MSE: 0.1933
top-3 (doc_id, cluster): [(1, 0), (4, 0), (3, 0)]
top result cluster: 0
```

## Serialization

`serde_roundtrip.rs` serializes a built index and verifies that deserialized
search results are identical.

The output reports the serialized index size and the restored query results.

Output:

```text
serialized index bytes: 559
query results after round trip:
  doc 10: 2.800
  doc 11: 1.600
```

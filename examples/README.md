# sporse examples

The examples use synthetic impact-score vectors: dimensions are token or latent
feature ids, weights are non-negative retrieval scores.

## Which example should I run?

| I want to... | Example | Run |
|---|---|---|
| See how inner-product scores are formed | `basic` | `cargo run --release --example basic` |
| Convert live weighted postings into sparse vectors | `postings_bridge` | `cargo run --release --example postings_bridge` |
| Write sparse vectors as raw impact files | `raw_impact_file` | `cargo run --release --example raw_impact_file` |
| Stream sparse vectors into raw impact generations | `raw_impact_generation` | `cargo run --release --example raw_impact_generation` |
| Measure store snapshot sidecars | `store_reopen_diagnostics` | `cargo run --release --features store --example store_reopen_diagnostics` |
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
`SparseVec::to_raw_impact_document`, seals a live numeric postings shard to a
`postings::raw` temp file, then queries that file-backed segment with
`SparseVec::to_raw_impact_query`.

`raw_impact_generation.rs` extends that boundary to multiple sealed raw files
plus one live raw postings shard. It writes and reloads the quantizer scale
sidecar before querying the combined generation.

The example keeps publication, fsync policy, manifests, deletes, and compaction
outside `sporse`; it only demonstrates the data conversion and raw segment
query boundary.

Output excerpt:

```text
top-k from postings::raw:
  doc 101: 5.120  learned sparse retrieval
```

## Store Diagnostics

`store_reopen_diagnostics.rs` builds a checkpointed segmented WAND store, opens
the read-only `SnapshotIndex`, then compares the first query with persisted
sidecars against the same query after deleting sidecars and rebuilding from
source sparse-vector segments.

```sh
cargo run --release --features store --example store_reopen_diagnostics
```

```text
documents: 1000, segments: 5, flush threshold: 200
sidecars loaded path: 5
sidecars rebuild path before/after delete: 5/0
first snapshot query with sidecars: 257 us
first snapshot query after deleting sidecars: 1246 us
with sidecars searched/pruned segments: 1/4
after rebuild searched/pruned segments: 1/4
top hit: Some((37, 320.0))
```

## Search Bench Diagnostics

The search benchmark's `diagnostics_noop` target checks WAND and raw-impact
top-k parity against brute force, then prints pruning counters for random raw
files, vocab-local partitioned files, an interleaved control over the same
synthetic documents, doc-order-local files, and impact-ordered files.

```sh
cargo bench --bench search -- diagnostics_noop --warm-up-time 0.1 --measurement-time 0.1 --sample-size 10
```

Current layout counters from that diagnostic:

```text
raw files searched/pruned segments: 4.0/0.0 of 4.0
partitioned raw files searched/pruned segments: 1.0/3.0 of 4.0
interleaved partitioned raw files searched/pruned segments: 4.0/0.0 of 4.0
doc-order-local raw files searched/pruned segments: 1.0/3.0 of 4.0
impact-ordered raw files searched/pruned segments: 1.0/3.0 of 4.0
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

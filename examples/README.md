# Examples

The examples use synthetic impact-score vectors: dimensions are token or latent
feature ids, weights are non-negative retrieval scores.

## `basic.rs`

Walks through a small learned-sparse collection and prints the term
contributions behind each top result.

```sh
cargo run --example basic
```

## `wand_diagnostics.rs`

Builds a larger synthetic index, checks exact top-k parity against brute force,
and prints how many documents Block-Max WAND fully scored.

```sh
cargo run --release --example wand_diagnostics
```

## `serde_roundtrip.rs`

Serializes a built index and verifies that deserialized search results are
identical.

```sh
cargo run --features serde --example serde_roundtrip
```

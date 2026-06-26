//! sporse command-line interface: build a sparse inverted index from a JSONL
//! file of sparse vectors and run top-k inner-product queries against it.
//!
//! Enabled by the `cli` feature.
//!
//!   sporse build docs.jsonl -o index.json
//!   sporse search index.json --query '[[3, 1.0], [21, 1.0]]' -k 5
//!
//! Each line of the docs file is one document:
//!   {"id": <u32>, "vec": [[<dim>, <weight>], ...]}
//! The query is a JSON array of `[dim, weight]` pairs. Scores are exact inner
//! products (Block-Max WAND, identical to a brute-force scan).

use std::fs;
use std::io::{BufRead, BufReader};

use clap::{Parser, Subcommand};
use serde::Deserialize;
use sporse::{SparseVec, SporseIndex};

#[derive(Parser)]
#[command(name = "sporse", about = "Build and query a sparse inverted index")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build an index from a JSONL file of sparse vectors and serialize it.
    Build {
        /// JSONL file: one {"id": u32, "vec": [[dim, weight], ...]} per line.
        docs: String,
        /// Output path for the serialized index.
        #[arg(short, long)]
        out: String,
    },
    /// Query a serialized index for the top-k documents.
    Search {
        /// Serialized index produced by `build`.
        index: String,
        /// Query as a JSON array of [dim, weight] pairs.
        #[arg(short, long)]
        query: String,
        /// Number of results to return.
        #[arg(short, default_value_t = 10)]
        k: usize,
    },
}

#[derive(Deserialize)]
struct DocLine {
    id: u32,
    vec: Vec<(u32, f32)>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().cmd {
        Cmd::Build { docs, out } => {
            let file = BufReader::new(fs::File::open(&docs)?);
            let mut index = SporseIndex::new();
            let mut n = 0usize;
            for line in file.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let doc: DocLine = serde_json::from_str(&line)?;
                index.insert(doc.id, &SparseVec::new(doc.vec));
                n += 1;
            }
            index.build();
            fs::write(&out, serde_json::to_string(&index)?)?;
            eprintln!("indexed {n} documents -> {out}");
        }
        Cmd::Search { index, query, k } => {
            let idx: SporseIndex = serde_json::from_str(&fs::read_to_string(&index)?)?;
            let q: Vec<(u32, f32)> = serde_json::from_str(&query)?;
            for (id, score) in idx.search(&SparseVec::new(q), k) {
                println!("{id}\t{score:.6}");
            }
        }
    }
    Ok(())
}

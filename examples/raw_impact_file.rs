//! Seal quantized `SparseVec` impacts from a live numeric postings shard into a
//! file-backed `postings::raw` segment and query it without building an
//! in-memory `SporseIndex`.
//!
//! Run with:
//! `cargo run --example raw_impact_file`

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use postings::raw::{write_u64_u32_segment_from_index_seekable_to, RawSegmentFile};
use postings::PostingsIndex;
use sporse::SparseVec;

const SCALE: f32 = 100.0;

struct Document {
    id: u32,
    title: &'static str,
    vector: SparseVec,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docs = [
        doc(
            101,
            "learned sparse retrieval",
            &[(10, 2.4), (20, 1.2), (30, 0.4)],
        ),
        doc(
            202,
            "dense vector service",
            &[(10, 0.7), (40, 3.1), (50, 1.0)],
        ),
        doc(
            303,
            "impact score pruning",
            &[(20, 2.0), (30, 1.8), (60, 0.5)],
        ),
    ];

    let mut live_shard = PostingsIndex::new();
    for doc in &docs {
        let terms = doc.vector.to_raw_impact_document(SCALE)?;
        live_shard.add_weighted_document(doc.id, &terms)?;
    }

    let temp = TempRawFile::new()?;
    let mut file = File::create(temp.path())?;
    write_u64_u32_segment_from_index_seekable_to(&live_shard, &mut file)?;
    file.sync_all()?;
    drop(file);

    let query = SparseVec::new(vec![(10, 1.5), (20, 1.0), (30, 0.8)]);
    let raw_query = query.to_raw_impact_query(SCALE)?;
    let mut segment = RawSegmentFile::open(temp.path())?;
    let hits = segment.top_k_weighted_u32(&raw_query, 3)?;

    assert_eq!(hits.first().map(|(doc_id, _)| *doc_id), Some(101));

    println!("raw segment path: {}", temp.path().display());
    println!("top-k from postings::raw:");
    for (doc_id, score) in hits {
        let title = docs
            .iter()
            .find(|doc| doc.id == doc_id)
            .map_or("unknown", |doc| doc.title);
        println!("  doc {doc_id}: {score:.3}  {title}");
    }

    Ok(())
}

fn doc(id: u32, title: &'static str, pairs: &[(u32, f32)]) -> Document {
    Document {
        id,
        title,
        vector: SparseVec::new(pairs.to_vec()),
    }
}

struct TempRawFile {
    path: PathBuf,
}

impl TempRawFile {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!(
            "sporse-raw-impact-{}-{nanos}.segment",
            std::process::id()
        ));
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempRawFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

//! Stream sparse vectors into raw impact segment files plus one live shard.
//!
//! Run with:
//! `cargo run --release --example raw_impact_generation`

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use postings::raw::{
    top_k_weighted_u32_files_and_index, write_u64_u32_segment_from_index_seekable_to,
    RawSegmentFile,
};
use postings::PostingsIndex;
use sporse::{RawImpactQuantizer, SparseVec};

const SCALE: f32 = 100.0;
const SEAL_AFTER_DOCS: usize = 2;

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
        doc(
            404,
            "raw file generation",
            &[(10, 1.2), (30, 1.5), (70, 0.4)],
        ),
        doc(
            505,
            "unsealed live tail",
            &[(10, 2.0), (20, 2.0), (30, 1.0)],
        ),
    ];

    let temp = TempDir::new()?;
    let scale_path = temp.path().join("impact-scale.txt");
    let mut scale_file = File::create(&scale_path)?;
    writeln!(scale_file, "{SCALE}")?;
    scale_file.sync_all()?;
    drop(scale_file);

    let persisted_scale: f32 = std::fs::read_to_string(&scale_path)?.trim().parse()?;
    let quantizer = RawImpactQuantizer::new(persisted_scale)?;
    let mut live = PostingsIndex::new();
    let mut live_docs = 0usize;
    let mut sealed_paths = Vec::new();

    for document in &docs {
        let terms = quantizer.document(&document.vector)?;
        live.add_weighted_document(document.id, &terms)?;
        live_docs += 1;

        if live_docs == SEAL_AFTER_DOCS {
            let path = temp
                .path()
                .join(format!("impact-generation-{}.raw", sealed_paths.len()));
            let mut file = File::create(&path)?;
            write_u64_u32_segment_from_index_seekable_to(&live, &mut file)?;
            file.sync_all()?;
            drop(file);

            sealed_paths.push(path);
            live = PostingsIndex::new();
            live_docs = 0;
        }
    }

    let query = SparseVec::new(vec![(10, 1.5), (20, 1.0), (30, 0.8)]);
    let raw_query = quantizer.query(&query)?;
    let mut sealed_segments = sealed_paths
        .iter()
        .map(RawSegmentFile::open)
        .collect::<Result<Vec<_>, _>>()?;
    let mut segment_refs: Vec<_> = sealed_segments.iter_mut().collect();
    let hits = top_k_weighted_u32_files_and_index(&mut segment_refs, &live, &raw_query, 5)?;

    assert_eq!(hits.first().map(|(doc_id, _)| *doc_id), Some(505));

    println!("raw impact scale: {}", quantizer.scale());
    println!("sealed raw files: {}", sealed_paths.len());
    println!("live docs: {live_docs}");
    println!("top-k from sealed files plus live shard:");
    for (doc_id, score) in hits {
        println!("  doc {doc_id}: {score:.3}  {}", title(doc_id, &docs));
    }

    Ok(())
}

struct Document {
    id: u32,
    title: &'static str,
    vector: SparseVec,
}

fn doc(id: u32, title: &'static str, pairs: &[(u32, f32)]) -> Document {
    Document {
        id,
        title,
        vector: SparseVec::new(pairs.to_vec()),
    }
}

fn title(doc_id: u32, docs: &[Document]) -> &'static str {
    docs.iter()
        .find(|doc| doc.id == doc_id)
        .map_or("unknown", |doc| doc.title)
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!(
            "sporse-raw-impact-generation-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

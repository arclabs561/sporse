//! Parity for the quantized-impact out-of-core path.
//!
//! `sporse` owns exact f32 sparse-vector search. `postings::raw` owns byte/file
//! backed u32 impact segments. This test pins the bridge between them for
//! fixed-point weights: if document weights are quantized by a declared scale,
//! a raw segment can reproduce `sporse` rankings without rebuilding an in-memory
//! `SporseIndex` from the segment payload.

use postings::raw::{write_u64_u32_segment, RawDocument, RawSegmentFile, RawTermId};
use sporse::{SparseVec, SporseIndex};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const SCALE: f32 = 100.0;

#[test]
fn quantized_raw_impacts_match_sporse_ranking_for_fixed_point_weights() {
    let docs = vec![
        (10, SparseVec::new(vec![(1, 2.25), (2, 0.75), (7, 1.10)])),
        (20, SparseVec::new(vec![(1, 1.50), (3, 3.25), (7, 0.40)])),
        (30, SparseVec::new(vec![(2, 2.40), (3, 0.60), (5, 1.20)])),
        (40, SparseVec::new(vec![(8, 8.0)])),
    ];
    let query = SparseVec::new(vec![(1, 1.25), (2, 1.50), (3, 0.80)]);

    let mut sporse = SporseIndex::new();
    for (doc_id, vector) in &docs {
        sporse.insert(*doc_id, vector);
    }
    sporse.build();
    let expected = sporse.search(&query, 4);

    let raw_terms: Vec<Vec<(RawTermId, u32)>> = docs
        .iter()
        .map(|(_, vector)| {
            vector
                .pairs()
                .iter()
                .map(|&(dim, weight)| (dim as RawTermId, quantize(weight)))
                .collect()
        })
        .collect();
    let raw_docs: Vec<_> = docs
        .iter()
        .zip(raw_terms.iter())
        .map(|((doc_id, _), terms)| RawDocument::new(*doc_id, terms))
        .collect();
    let bytes = write_u64_u32_segment(&raw_docs).unwrap();

    let path = TempRawPath::new();
    std::fs::write(path.as_path(), bytes).unwrap();
    let mut raw = RawSegmentFile::open(path.as_path()).unwrap();

    let raw_query: Vec<_> = query
        .pairs()
        .iter()
        .map(|&(dim, weight)| (dim as RawTermId, weight / SCALE))
        .collect();
    let got = raw.top_k_weighted_u32(&raw_query, 4).unwrap();

    assert_rankings_close(&expected, &got);
}

fn quantize(weight: f32) -> u32 {
    assert!(weight.is_finite() && weight >= 0.0);
    let scaled = (weight * SCALE).round();
    assert!(scaled > 0.0 && scaled <= u32::MAX as f32);
    scaled as u32
}

fn assert_rankings_close(expected: &[(u32, f32)], got: &[(u32, f32)]) {
    assert_eq!(expected.len(), got.len());
    for (&(expected_doc, expected_score), &(got_doc, got_score)) in expected.iter().zip(got) {
        assert_eq!(expected_doc, got_doc);
        assert!(
            (expected_score - got_score).abs() < 1e-5,
            "doc {expected_doc}: expected {expected_score}, got {got_score}"
        );
    }
}

struct TempRawPath {
    path: PathBuf,
}

impl TempRawPath {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self {
            path: std::env::temp_dir().join(format!(
                "sporse-raw-impact-{}-{nanos}.raw",
                std::process::id()
            )),
        }
    }

    fn as_path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempRawPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

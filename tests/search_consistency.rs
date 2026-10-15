//! Public-boundary parity: finite positive scores and global ID ordering on ties.
use sporse::{SparseVec, SporseIndex};

fn vector(weight: f32) -> SparseVec {
    SparseVec::new(vec![(0, weight)])
}

fn numerical_index(weight: f32, force_exact: bool) -> SporseIndex {
    let mut index = SporseIndex::new();
    index.insert(0, &vector(weight));
    index.insert(1, &vector(1.0));
    if force_exact {
        index.insert(9, &SparseVec::new(vec![(9, -1.0)]));
    }
    index.build();
    index
}

#[test]
fn finite_inputs_with_overflow_keep_the_finite_competitor() {
    let query = vector(2.0);
    for force_exact in [false, true] {
        let index = numerical_index(f32::MAX, force_exact);
        assert_eq!(index.search(&query, 2), vec![(1, 2.0)]);
        let (hits, stats) = index.search_with_stats(&query, 2);
        assert_eq!(hits, vec![(1, 2.0)]);
        assert_eq!(stats.docs_scored == 0, force_exact);
    }
}

#[test]
fn underflow_zero_scores_are_excluded_in_both_paths() {
    let query = vector(0.5);
    for force_exact in [false, true] {
        let index = numerical_index(f32::from_bits(1), force_exact);
        assert_eq!(index.search(&query, 2), vec![(1, 0.5)]);
        let (hits, stats) = index.search_with_stats(&query, 2);
        assert_eq!(hits, vec![(1, 0.5)]);
        assert_eq!(stats.docs_scored == 0, force_exact);
    }
}

#[cfg(feature = "store")]
mod store {
    use super::*;
    use durability::MemoryDirectory;
    use sporse::store::{SnapshotIndex, UpdatableIndex};

    fn assert_ties(surface: &str, weight: f32) {
        let dir = MemoryDirectory::arc();
        let mut store = UpdatableIndex::open(dir.clone(), 2).unwrap();
        // This first segment has bound 2 but each document scores only 1.
        // Later segments have bound 1 and must still compete by lower ID.
        store.add(90, vector(weight)).unwrap();
        store.add(91, SparseVec::new(vec![(1, weight)])).unwrap();
        store.add(1, vector(weight)).unwrap();
        store.add(2, vector(weight)).unwrap();
        store.checkpoint().unwrap();
        let reader = store.reader();
        let held = reader.view();
        store.add(0, vector(weight)).unwrap();
        let query = SparseVec::new(vec![(0, 1.0), (1, 1.0)]);
        let expected = vec![(0, weight), (1, weight)];
        match surface {
            "writer" => {
                assert_eq!(store.search(&query, 2), expected);
                assert_eq!(store.search_with_stats(&query, 2).0, expected);
            }
            "held" => {
                store.checkpoint().unwrap();
                assert_eq!(held.search(&query, 2), vec![(1, weight), (2, weight)]);
            }
            "reader" => {
                store.checkpoint().unwrap();
                assert_eq!(reader.search(&query, 2), expected);
                assert_eq!(reader.search_with_stats(&query, 2).0, expected);
            }
            "snapshot" => {
                store.checkpoint().unwrap();
                let snapshot = SnapshotIndex::open(dir).unwrap();
                assert_eq!(snapshot.search(&query, 2).unwrap(), expected);
                assert_eq!(snapshot.search_with_stats(&query, 2).unwrap().0, expected);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn writer_ties_across_segments_and_buffer() {
        assert_ties("writer", 1.0);
    }
    #[test]
    fn reader_ties_across_segments() {
        assert_ties("reader", 1.0);
    }
    #[test]
    fn held_view_ties_preserve_checkpoint() {
        assert_ties("held", 1.0);
    }
    #[test]
    fn snapshot_ties_across_segments() {
        assert_ties("snapshot", 1.0);
    }
    #[test]
    fn least_positive_score_ties_remain_visible() {
        for surface in ["writer", "reader", "held", "snapshot"] {
            assert_ties(surface, f32::from_bits(1));
        }
    }
}

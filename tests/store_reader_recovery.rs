//! Public reader searches after recovering the original segstore manifest layout.
#![cfg(feature = "store")]

use durability::{checkpoint::CheckpointFile, MemoryDirectory};
use segstore::{DefaultStore, SegmentedStore};
use serde::Serialize;
use sporse::{store::UpdatableIndex, SparseVec};

// Original segstore manifest layout, before the superseded-copy extension.
// Field order matches segstore's recovery Manifest and its legacy recovery tests.
#[derive(Serialize)]
struct OriginalManifest {
    next_seg_id: u64,
    segment_ids: Vec<u64>,
    tombstones: Vec<u32>,
}

fn vector(dim: u32, weight: f32) -> SparseVec {
    SparseVec::new(vec![(dim, weight)])
}

#[test]
fn recovered_reader_searches_checkpoint_owners_without_wal_leakage() {
    for legacy in [true, false] {
        for wal_replacement in [false, true] {
            let dir = MemoryDirectory::arc();
            let mut producer =
                SegmentedStore::open(dir.clone(), DefaultStore::<u32, SparseVec>::new(), 1)
                    .unwrap();
            producer.add(7, vector(1, 9.0)).unwrap();
            producer.add(7, vector(2, 2.0)).unwrap();
            producer.checkpoint().unwrap();
            assert_eq!(producer.segment_ids(), &[0, 1]);
            if legacy {
                CheckpointFile::new(dir.clone())
                    .write_postcard(
                        "segstore.manifest",
                        1,
                        &OriginalManifest {
                            next_seg_id: 2,
                            segment_ids: vec![0, 1],
                            tombstones: vec![],
                        },
                    )
                    .unwrap();
            }
            if wal_replacement {
                producer.add(7, vector(3, 4.0)).unwrap();
            }
            drop(producer);

            let mut store = UpdatableIndex::open(dir, 1).unwrap();
            let reader = store.reader();
            let held = reader.view();
            let obsolete = vector(1, 1.0);
            let checkpointed = vector(2, 1.0);
            let wal = vector(3, 1.0);
            assert!(reader.search(&obsolete, 10).is_empty(), "obsolete copy");
            assert_eq!(reader.search(&checkpointed, 10), vec![(7, 2.0)]);
            assert!(reader.search(&wal, 10).is_empty(), "WAL is unpublished");
            assert!(store.search(&obsolete, 10).is_empty());
            let (current_query, current_score) = if wal_replacement {
                assert!(store.search(&checkpointed, 10).is_empty());
                (&wal, 4.0)
            } else {
                assert!(store.search(&wal, 10).is_empty());
                (&checkpointed, 2.0)
            };
            assert_eq!(store.search(current_query, 10), vec![(7, current_score)]);

            store.checkpoint().unwrap();
            assert!(reader.search(&obsolete, 10).is_empty());
            assert_eq!(reader.search(current_query, 10), vec![(7, current_score)]);
            if wal_replacement {
                assert!(reader.search(&checkpointed, 10).is_empty());
            }
            assert!(held.search(&obsolete, 10).is_empty());
            assert_eq!(held.search(&checkpointed, 10), vec![(7, 2.0)]);
            assert!(held.search(&wal, 10).is_empty(), "held view is immutable");
        }
    }
}

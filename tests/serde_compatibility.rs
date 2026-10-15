//! Fixtures emitted by the unchanged 0.9.1 library, including real sidecars.
#![cfg(feature = "serde")]

use sporse::{SparseVec, SporseIndex};

fn query() -> SparseVec {
    SparseVec::new(vec![(0, 1.0), (3, 1.0)])
}

#[test]
fn named_ron_preserves_legacy_struct_name_and_self_roundtrip() {
    let legacy = include_str!("fixtures/v0.9.1/sparse-vector.ron");
    let decoded: SparseVec = ron::from_str(legacy).unwrap();
    assert_eq!(decoded.pairs(), &[(0, 2.0), (3, 1.0)]);
    let named = ron::ser::to_string_pretty(
        &decoded,
        ron::ser::PrettyConfig::default().struct_names(true),
    )
    .unwrap();
    assert!(named.starts_with("SparseVec("), "{named}");
    let roundtrip: SparseVec = ron::from_str(&named).unwrap();
    assert_eq!(roundtrip.pairs(), decoded.pairs());
    let unnamed = ron::to_string(&decoded).unwrap();
    let roundtrip: SparseVec = ron::from_str(&unnamed).unwrap();
    assert_eq!(roundtrip.pairs(), decoded.pairs());
}

#[test]
fn named_ron_normalizes_like_unnamed_input() {
    for prefix in ["SparseVec", ""] {
        let wire = format!("{prefix}(pairs:[(3,1.0),(0,2.0),(3,2.0),(1,0.0)])");
        let decoded: SparseVec = ron::from_str(&wire).unwrap();
        assert_eq!(decoded.pairs(), &[(0, 2.0), (3, 3.0)]);
    }
}
fn assert_index(index: &SporseIndex) {
    assert_eq!(index.len(), 3);
    assert_eq!(index.num_dimensions(), 3);
    assert_eq!(index.search(&query(), 10), vec![(10, 3.0), (20, 0.5)]);
}

#[test]
fn legacy_built_json_and_missing_safety_flag_remain_readable() {
    let bytes = include_bytes!("fixtures/v0.9.1/built-index.json");
    assert_index(&serde_json::from_slice(bytes).unwrap());
    let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    value.as_object_mut().unwrap().remove("wand_safe");
    let index: SporseIndex = serde_json::from_value(value).unwrap();
    assert_index(&index);
    assert_eq!(serde_json::to_value(&index).unwrap()["wand_safe"], false);
    let vector: SparseVec =
        serde_json::from_slice(include_bytes!("fixtures/v0.9.1/sparse-vector.json")).unwrap();
    assert_eq!(vector.pairs(), &[(0, 2.0), (3, 1.0)]);
}

fn replaced_index() -> SporseIndex {
    let mut index = SporseIndex::new();
    index.insert(10, &SparseVec::new(vec![(99, 9.0)]));
    assert!(index
        .upsert(10, &SparseVec::new(vec![(0, 2.0), (3, 1.0)]))
        .was_replaced());
    index.insert(20, &SparseVec::new(vec![(0, 0.5), (8, 4.0)]));
    index.insert(30, &SparseVec::new(vec![(100, 8.0)]));
    assert!(index.upsert(30, &SparseVec::new(vec![])).was_replaced());
    index.build();
    index
}

#[test]
fn replacement_json_roundtrip_and_unbuilt_json_boundary() {
    let index = replaced_index();
    assert_index(&index);
    assert_index(&serde_json::from_slice(&serde_json::to_vec(&index).unwrap()).unwrap());
    assert!(serde_json::from_slice::<SporseIndex>(include_bytes!(
        "fixtures/v0.9.1/unbuilt-index.json"
    ))
    .is_err());
    assert!(serde_json::to_vec(&SporseIndex::new()).is_err());
}

#[cfg(feature = "store")]
#[test]
fn postcard_preserves_legacy_and_replaced_built_indexes_only() {
    assert_index(
        &postcard::from_bytes(include_bytes!("fixtures/v0.9.1/built-index.postcard")).unwrap(),
    );
    assert_index(
        &postcard::from_bytes(&postcard::to_allocvec(&replaced_index()).unwrap()).unwrap(),
    );
    let vector: SparseVec =
        postcard::from_bytes(include_bytes!("fixtures/v0.9.1/sparse-vector.postcard")).unwrap();
    assert_eq!(vector.pairs(), &[(0, 2.0), (3, 1.0)]);
    assert!(postcard::from_bytes::<SporseIndex>(include_bytes!(
        "fixtures/v0.9.1/unbuilt-index.postcard"
    ))
    .is_err());
    assert!(postcard::to_allocvec(&SporseIndex::new()).is_err());
}

#[cfg(feature = "store")]
mod sidecars {
    use super::*;
    use durability::{Directory, MemoryDirectory, PersistenceResult};
    use sporse::store::{SnapshotIndex, UpdatableIndex};
    use std::{
        io::{Read, Write},
        path::PathBuf,
        sync::{Arc, Mutex},
    };

    struct RecordingDirectory {
        inner: Arc<dyn Directory>,
        opened: Arc<Mutex<Vec<String>>>,
        written: Arc<Mutex<Vec<String>>>,
    }
    impl Directory for RecordingDirectory {
        fn create_file(&self, p: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.written.lock().unwrap().push(p.into());
            self.inner.create_file(p)
        }
        fn open_file(&self, p: &str) -> PersistenceResult<Box<dyn Read + Send>> {
            self.opened.lock().unwrap().push(p.into());
            self.inner.open_file(p)
        }
        fn exists(&self, p: &str) -> bool {
            self.inner.exists(p)
        }
        fn delete(&self, p: &str) -> PersistenceResult<()> {
            self.inner.delete(p)
        }
        fn atomic_rename(&self, a: &str, b: &str) -> PersistenceResult<()> {
            self.inner.atomic_rename(a, b)
        }
        fn create_dir_all(&self, p: &str) -> PersistenceResult<()> {
            self.inner.create_dir_all(p)
        }
        fn list_dir(&self, p: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(p)
        }
        fn append_file(&self, p: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.append_file(p)
        }
        fn atomic_write(&self, p: &str, data: &[u8]) -> PersistenceResult<()> {
            self.written.lock().unwrap().push(p.into());
            self.inner.atomic_write(p, data)
        }
        fn file_path(&self, p: &str) -> Option<PathBuf> {
            self.inner.file_path(p)
        }
    }
    #[test]
    fn old_sidecars_are_loaded_without_rebuilding_for_writer_reader_and_snapshot() {
        for mode in ["writer", "reader", "snapshot"] {
            let inner = MemoryDirectory::arc();
            for name in [
                "segstore.manifest",
                "segstore.seg.0",
                "segstore.seg.1",
                "segstore.idx.0.wand",
                "segstore.idx.1.wand",
            ] {
                let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/v0.9.1/checkpoint")
                    .join(name);
                inner
                    .atomic_write(name, &std::fs::read(path).unwrap())
                    .unwrap();
            }
            let opened = Arc::new(Mutex::new(Vec::new()));
            let written = Arc::new(Mutex::new(Vec::new()));
            let watched: Arc<dyn Directory> = Arc::new(RecordingDirectory {
                inner,
                opened: opened.clone(),
                written: written.clone(),
            });
            let hits = if mode == "snapshot" {
                SnapshotIndex::open(watched)
                    .unwrap()
                    .search(&query(), 10)
                    .unwrap()
            } else {
                let store = UpdatableIndex::open(watched, 2).unwrap();
                // Writer recovery reads source segments. Measure search after recovery.
                opened.lock().unwrap().clear();
                written.lock().unwrap().clear();
                if mode == "writer" {
                    store.search(&query(), 10)
                } else {
                    store.reader().search(&query(), 10)
                }
            };
            assert_eq!(hits, vec![(10, 3.0), (20, 0.5)], "{mode}");
            let reads = opened.lock().unwrap();
            assert!(
                reads.iter().any(|p| p == "segstore.idx.0.wand"),
                "{mode}: {reads:?}"
            );
            assert!(
                !reads.iter().any(|p| p.starts_with("segstore.seg.")),
                "{mode}: {reads:?}"
            );
            // A rejected sidecar can rebuild from already resident writer segments;
            // capture writes as well so no new disk read cannot hide that fallback.
            assert!(
                !written.lock().unwrap().iter().any(|p| p.contains(".wand")),
                "{mode}: rebuilt a sidecar"
            );
        }
    }
}

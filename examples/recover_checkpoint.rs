//! Re-index one deleted document, restart the writer, and compare the views.
//! Run with `cargo run --features store --example recover_checkpoint`.
use sporse::store::{SnapshotIndex, UpdatableIndex};
use sporse::SparseVec;
use std::{error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    // The fixture is an actual on-disk checkpoint from sporse 0.7.1. Work on
    // a copy because opening a writer may upgrade a legacy store.
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v0.7.1/deleted-checkpoint");
    let copy = tempfile::tempdir()?;
    for entry in fs::read_dir(fixture)? {
        let entry = entry?;
        fs::copy(entry.path(), copy.path().join(entry.file_name()))?;
    }

    let dir = sporse::store::durability::FsDirectory::arc(copy.path())?;
    let query = SparseVec::new(vec![(1, 1.0)]);
    let mut writer = UpdatableIndex::open(dir.clone(), 2)?;
    let before = writer.reader().view();
    let original = writer.search(&query, 10);
    println!("original checkpoint: {original:?}");
    assert!(original.is_empty());

    writer.add(7, SparseVec::new(vec![(1, 1.0)]))?;
    let current = writer.search(&query, 10);
    println!("writer after replacement: {current:?}");
    assert_eq!(current, [(7, 1.0)]);
    drop(writer);

    let mut writer = UpdatableIndex::open(dir.clone(), 2)?;
    let replayed = writer.search(&query, 10);
    println!("writer after WAL replay: {replayed:?}");
    assert_eq!(replayed, [(7, 1.0)]);

    writer.checkpoint()?;
    let snapshot = SnapshotIndex::open(dir)?;
    let published = snapshot.search(&query, 10)?;
    println!("new checkpoint snapshot: {published:?}");
    assert_eq!(published, [(7, 1.0)]);
    let held = before.search(&query, 10);
    println!("reader held before replacement: {held:?}");
    assert!(held.is_empty());
    Ok(())
}

#[test]
fn deleted_legacy_checkpoint_returns_only_the_replacement_after_reopen() {
    main().unwrap();
}

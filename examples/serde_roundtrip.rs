use sporse::{SparseVec, SporseIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut index = SporseIndex::new();
    index.insert(10, &SparseVec::new(vec![(3, 2.0), (9, 1.5), (21, 0.8)]));
    index.insert(11, &SparseVec::new(vec![(3, 0.6), (8, 2.3), (21, 1.0)]));
    index.insert(12, &SparseVec::new(vec![(4, 1.8), (9, 2.5), (21, 0.4)]));
    index.build();

    let query = SparseVec::new(vec![(3, 1.0), (21, 1.0)]);
    let before = index.search(&query, 2);

    let json = serde_json::to_string(&index)?;
    let restored: SporseIndex = serde_json::from_str(&json)?;
    let after = restored.search(&query, 2);

    assert_eq!(before, after);

    println!("serialized index bytes: {}", json.len());
    println!("query results after round trip:");
    for (doc_id, score) in after {
        println!("  doc {doc_id}: {score:.3}");
    }

    Ok(())
}

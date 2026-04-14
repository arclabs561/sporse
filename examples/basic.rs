use sporse::{SparseVec, SporseIndex};

fn main() {
    let mut index = SporseIndex::new();

    // Simulate SPLADE-style sparse vectors: dimension = vocab token id, weight = impact score.
    let docs = [
        vec![(42, 2.1), (100, 1.5), (7, 0.3)], // "rust programming language"
        vec![(42, 1.0), (200, 3.0), (55, 0.8)], // "rust oxidation metal"
        vec![(100, 2.0), (7, 1.0), (300, 0.5)], // "programming tutorial basics"
        vec![(42, 0.5), (100, 2.5), (7, 1.2), (55, 0.1)], // "rust programming guide"
    ];

    for (id, pairs) in docs.iter().enumerate() {
        index.insert(id as u32, &SparseVec::new(pairs.clone()));
    }
    index.build();

    // Query: "rust programming"
    let query = SparseVec::new(vec![(42, 1.0), (100, 1.0)]);
    let results = index.search(&query, 3);

    println!("Query: rust programming");
    for (doc_id, score) in &results {
        println!("  doc {doc_id}: score {score:.2}");
    }
}

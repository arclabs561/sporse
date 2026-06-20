use sporse::{SparseVec, SporseIndex};

struct Document {
    title: &'static str,
    vector: SparseVec,
}

fn main() {
    let docs = [
        doc(
            "SPLADE retrieval with inverted indexes",
            &[(0, 2.4), (1, 2.2), (2, 1.7), (3, 1.4), (4, 0.8)],
        ),
        doc(
            "Dense vector HNSW service",
            &[(5, 2.8), (6, 2.2), (7, 2.0), (1, 0.4)],
        ),
        doc(
            "Block-Max WAND for impact scores",
            &[(2, 2.8), (8, 2.0), (1, 1.2), (3, 0.9)],
        ),
        doc(
            "Rust sparse index implementation",
            &[(9, 2.0), (0, 2.3), (3, 1.8), (1, 1.6)],
        ),
        doc(
            "BM25 lexical search baseline",
            &[(10, 2.3), (11, 2.0), (3, 1.2), (1, 0.8)],
        ),
        doc(
            "Learned expansion for code search",
            &[(0, 1.8), (12, 2.1), (4, 1.6), (1, 1.4)],
        ),
    ];

    let mut index = SporseIndex::new();
    for (id, doc) in docs.iter().enumerate() {
        index.insert(id as u32, &doc.vector);
    }
    index.build();

    let query = SparseVec::new(vec![(0, 1.9), (1, 2.1), (2, 1.2), (8, 0.9), (9, 0.8)]);
    let results = index.search(&query, 4);

    println!(
        "index: {} documents, {} sparse dimensions",
        index.len(),
        index.num_dimensions()
    );
    println!("query: sparse retrieval with WAND impact scoring");

    for (rank, (doc_id, score)) in results.iter().enumerate() {
        let doc = &docs[*doc_id as usize];
        println!("\n{}. {}  score={score:.3}", rank + 1, doc.title);
        for (term, contribution) in top_contributions(&query, &doc.vector) {
            println!("   {term:<16} {contribution:.3}");
        }
    }
}

fn doc(title: &'static str, pairs: &[(u32, f32)]) -> Document {
    Document {
        title,
        vector: SparseVec::new(pairs.to_vec()),
    }
}

fn top_contributions(query: &SparseVec, doc: &SparseVec) -> Vec<(&'static str, f32)> {
    let mut out = Vec::new();
    let (mut qi, mut di) = (0, 0);
    let (q, d) = (query.pairs(), doc.pairs());

    while qi < q.len() && di < d.len() {
        match q[qi].0.cmp(&d[di].0) {
            std::cmp::Ordering::Equal => {
                out.push((term_name(q[qi].0), q[qi].1 * d[di].1));
                qi += 1;
                di += 1;
            }
            std::cmp::Ordering::Less => qi += 1,
            std::cmp::Ordering::Greater => di += 1,
        }
    }

    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn term_name(dim: u32) -> &'static str {
    match dim {
        0 => "sparse",
        1 => "retrieval",
        2 => "wand",
        3 => "index",
        4 => "expansion",
        5 => "dense",
        6 => "ann",
        7 => "hnsw",
        8 => "impact",
        9 => "rust",
        10 => "bm25",
        11 => "lexical",
        12 => "code",
        _ => "latent",
    }
}

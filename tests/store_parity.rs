#![cfg(feature = "store")]
//! Randomized parity between the multi-segment store search and a brute-force
//! scan over the live corpus.
//!
//! The store path under test carries the segment-level threshold pruning
//! (skip a segment when its query upper bound cannot beat the running kth
//! score) plus per-segment WAND. The reuse-seam contract: pruning is an
//! optimization only, so `store.search(q, k)` must return exactly what an
//! exhaustive dot-product scan over the live documents returns, for any mix
//! of adds, deletes, checkpoints, and mid-stream compaction.

use durability::MemoryDirectory;
use sporse::store::UpdatableIndex;
use sporse::SparseVec;
use std::collections::BTreeMap;

/// Deterministic LCG so the test is reproducible without a rand dependency.
struct Lcg(u64);

impl Lcg {
    fn next_below(&mut self, bound: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) % bound
    }

    /// Positive weight in 0.01..=10.00 (WAND requires non-negative weights).
    fn weight(&mut self) -> f32 {
        (self.next_below(1000) + 1) as f32 / 100.0
    }
}

const DIM: u32 = 24;

fn random_vec(rng: &mut Lcg) -> Vec<(u32, f32)> {
    let nnz = 1 + rng.next_below(5) as usize;
    let mut pairs = BTreeMap::new();
    for _ in 0..nnz {
        pairs.insert(rng.next_below(DIM), rng.weight());
    }
    pairs.into_iter().collect()
}

fn dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let (mut i, mut j, mut sum) = (0, 0, 0.0f32);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                sum += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    sum
}

fn brute_force(
    live: &BTreeMap<u32, Vec<(u32, f32)>>,
    query: &[(u32, f32)],
    k: usize,
) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = live
        .iter()
        .map(|(&id, doc)| (id, dot(query, doc)))
        .filter(|&(_, s)| s > 0.0)
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

#[test]
fn store_search_matches_brute_force_across_segments() {
    let mut rng = Lcg(0x5eed_cafe);
    let dir = MemoryDirectory::arc();
    // flush_threshold 3 with ~90 adds and only one mid-stream compaction
    // leaves the store many segments deep at every query below, so the
    // segment-skip branch is exercised rather than a single-segment
    // degenerate case.
    let mut store = UpdatableIndex::open(dir, 3).unwrap();
    let mut live: BTreeMap<u32, Vec<(u32, f32)>> = BTreeMap::new();

    let check = |store: &UpdatableIndex, live: &BTreeMap<u32, Vec<(u32, f32)>>, rng: &mut Lcg| {
        for k in [1usize, 3, 10] {
            let query = random_vec(rng);
            let got = store.search(&SparseVec::new(query.clone()), k);
            let want = brute_force(live, &query, k);
            assert_eq!(
                got.len(),
                want.len(),
                "result count diverged for query {query:?} k={k}: got {got:?} want {want:?}"
            );
            for (rank, (&(gid, gscore), &(_, wscore))) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (gscore - wscore).abs() < 1e-4,
                    "rank-{rank} score diverged for query {query:?} k={k}: got {got:?} want {want:?}"
                );
                let doc = live
                    .get(&gid)
                    .expect("store returned a deleted or unknown doc id");
                assert!(
                    (gscore - dot(&query, doc)).abs() < 1e-4,
                    "reported score for doc {gid} does not match its content"
                );
            }
        }
    };

    for step in 0u32..90 {
        let id = step;
        let vec = random_vec(&mut rng);
        store.add(id, SparseVec::new(vec.clone())).unwrap();
        live.insert(id, vec);

        if step % 7 == 3 {
            let victim = rng.next_below(step + 1);
            if live.remove(&victim).is_some() {
                store.delete(victim).unwrap();
            }
        }
        if step == 40 {
            store.checkpoint().unwrap();
        }
        if step == 55 {
            store.compact().unwrap();
        }
        if step % 11 == 10 {
            check(&store, &live, &mut rng);
        }
    }
    check(&store, &live, &mut rng);
}

use crate::posting::{PostingList, BLOCK_SIZE};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ── Float ordering wrapper ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ── Cursor ───────────────────────────────────────────────────────────────────

pub(crate) struct Cursor<'a> {
    list: &'a PostingList,
    pos: usize,
    pub query_weight: f32,
}

impl<'a> Cursor<'a> {
    pub fn new(list: &'a PostingList, query_weight: f32) -> Self {
        Self {
            list,
            pos: 0,
            query_weight,
        }
    }

    #[inline]
    fn current_doc(&self) -> Option<u32> {
        self.list.entries().get(self.pos).map(|e| e.doc_id)
    }

    #[inline]
    fn current_weight(&self) -> f32 {
        self.list
            .entries()
            .get(self.pos)
            .map_or(0.0, |e| e.weight)
    }

    /// Upper bound contribution: block-max weight * query weight.
    #[inline]
    fn upper_bound(&self) -> f32 {
        self.list.block_max(self.pos / BLOCK_SIZE) * self.query_weight
    }

    #[inline]
    fn is_exhausted(&self) -> bool {
        self.pos >= self.list.len()
    }

    #[inline]
    fn advance(&mut self) {
        self.pos += 1;
    }

    /// Advance to the first entry with doc_id >= target.
    fn advance_to(&mut self, target: u32) {
        let entries = self.list.entries();
        let remaining = &entries[self.pos..];
        let offset = match remaining.binary_search_by_key(&target, |e| e.doc_id) {
            Ok(i) | Err(i) => i,
        };
        self.pos += offset;
    }
}

// ── Block-Max WAND ───────────────────────────────────────────────────────────

/// Block-Max WAND search. Returns `(doc_id, score)` in descending score order.
///
/// Assumes non-negative weights. Negative weights may cause missed results.
pub(crate) fn search_bmw(cursors: &mut Vec<Cursor>, k: usize) -> Vec<(u32, f32)> {
    let mut heap: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::with_capacity(k + 1);
    let mut threshold = 0.0f32;

    loop {
        cursors.retain(|c| !c.is_exhausted());
        if cursors.is_empty() {
            break;
        }

        // Sort cursors by current doc_id (ascending).
        cursors.sort_unstable_by_key(|c| c.current_doc().unwrap_or(u32::MAX));

        // Find pivot: accumulate upper bounds left-to-right until we can
        // potentially beat the threshold (or the heap has room).
        let mut acc = 0.0f32;
        let mut pivot_idx = None;
        for (i, cursor) in cursors.iter().enumerate() {
            acc += cursor.upper_bound();
            if acc > threshold || heap.len() < k {
                pivot_idx = Some(i);
                break;
            }
        }

        let pivot_idx = match pivot_idx {
            Some(p) => p,
            None => break, // no remaining docs can beat threshold
        };

        let pivot_doc = match cursors[pivot_idx].current_doc() {
            Some(d) => d,
            None => break,
        };

        // Check if all cursors up to the pivot point to pivot_doc.
        let all_at_pivot = cursors[..=pivot_idx]
            .iter()
            .all(|c| c.current_doc() == Some(pivot_doc));

        if all_at_pivot {
            // Fully score pivot_doc across ALL cursors (not just up to pivot).
            let mut score = 0.0f32;
            for cursor in cursors.iter() {
                if cursor.current_doc() == Some(pivot_doc) {
                    score += cursor.current_weight() * cursor.query_weight;
                }
            }

            if heap.len() < k || score > threshold {
                heap.push(Reverse((OrdF32(score), pivot_doc)));
                if heap.len() > k {
                    heap.pop();
                }
                if heap.len() >= k {
                    threshold = heap.peek().map_or(0.0, |r| r.0 .0 .0);
                }
            }

            // Advance all cursors pointing to pivot_doc.
            for cursor in cursors.iter_mut() {
                if cursor.current_doc() == Some(pivot_doc) {
                    cursor.advance();
                }
            }
        } else {
            // Advance the leftmost cursor that's behind pivot_doc.
            // Cursors are sorted by doc_id, so cursor[0] is the smallest.
            if let Some(cursor) = cursors[..pivot_idx]
                .iter_mut()
                .find(|c| c.current_doc().is_some_and(|d| d < pivot_doc))
            {
                cursor.advance_to(pivot_doc);
            }
        }
    }

    let mut results: Vec<(u32, f32)> = heap
        .into_iter()
        .map(|Reverse((OrdF32(score), doc_id))| (doc_id, score))
        .collect();
    results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    results
}

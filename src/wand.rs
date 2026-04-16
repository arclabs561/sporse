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
    /// query_weight * list.max_weight — cached for pivot selection.
    pub max_score: f32,
}

impl<'a> Cursor<'a> {
    pub fn new(list: &'a PostingList, query_weight: f32) -> Self {
        Self {
            list,
            pos: 0,
            query_weight,
            max_score: list.max_weight * query_weight,
        }
    }

    #[inline]
    fn current_doc(&self) -> Option<u32> {
        self.list.entries().get(self.pos).map(|e| e.doc_id)
    }

    #[inline]
    fn current_weight(&self) -> f32 {
        self.list.entries().get(self.pos).map_or(0.0, |e| e.weight)
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

// ── WAND search statistics (diagnostic, not in hot path) ────────────────────

/// Per-query statistics from a WAND search run.
#[derive(Debug, Default)]
pub struct WandStats {
    /// Number of WAND loop iterations (each iteration = one pivot evaluation).
    pub iterations: u64,
    /// Documents fully scored (pivot + all cursors at pivot_doc).
    pub docs_scored: u64,
    /// Pivot advances: times a cursor was skipped without scoring.
    pub cursor_skips: u64,
}

// ── Block-Max WAND ───────────────────────────────────────────────────────────

/// Block-Max WAND search. Returns `(doc_id, score)` in descending score order.
///
/// Assumes non-negative weights. Negative weights may cause missed results.
pub(crate) fn search_bmw(cursors: &mut Vec<Cursor>, k: usize) -> Vec<(u32, f32)> {
    search_bmw_impl(cursors, k, false).0
}

/// Block-Max WAND search with per-query statistics. Used for profiling/diagnosis.
pub(crate) fn search_bmw_with_stats(
    cursors: &mut Vec<Cursor>,
    k: usize,
) -> (Vec<(u32, f32)>, WandStats) {
    search_bmw_impl(cursors, k, true)
}

fn search_bmw_impl(
    cursors: &mut Vec<Cursor>,
    k: usize,
    collect_stats: bool,
) -> (Vec<(u32, f32)>, WandStats) {
    let mut heap: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::with_capacity(k + 1);
    let mut threshold = 0.0f32;
    let mut stats = WandStats::default();

    // Pre-sort cursors by max_score descending so the pivot accumulation loop
    // terminates as early as possible (high-impact terms reach threshold sooner).
    cursors.sort_unstable_by(|a, b| b.max_score.total_cmp(&a.max_score));

    loop {
        // Remove exhausted cursors in O(n) with swap-remove.
        {
            let mut i = 0;
            while i < cursors.len() {
                if cursors[i].is_exhausted() {
                    cursors.swap_remove(i);
                } else {
                    i += 1;
                }
            }
        }
        if cursors.is_empty() {
            break;
        }

        if collect_stats {
            stats.iterations += 1;
        }

        // Sort by current doc_id ascending for WAND pivot selection.
        cursors.sort_unstable_by_key(|c| c.current_doc().unwrap_or(u32::MAX));

        // Find pivot: accumulate upper bounds left-to-right until we exceed threshold.
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
            None => break,
        };

        let pivot_doc = match cursors[pivot_idx].current_doc() {
            Some(d) => d,
            None => break,
        };

        // Check if all cursors up to and including the pivot are at pivot_doc.
        let all_at_pivot = cursors[..=pivot_idx]
            .iter()
            .all(|c| c.current_doc() == Some(pivot_doc));

        if all_at_pivot {
            // Score pivot_doc across ALL cursors, advancing those at pivot_doc
            // in a single pass (avoids a second scan).
            let mut score = 0.0f32;
            for cursor in cursors.iter_mut() {
                if cursor.current_doc() == Some(pivot_doc) {
                    score += cursor.current_weight() * cursor.query_weight;
                    cursor.advance();
                }
            }

            if collect_stats {
                stats.docs_scored += 1;
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
        } else {
            // Advance the leftmost cursor behind pivot_doc.
            // Cursors are sorted ascending so cursor[0] is earliest.
            for cursor in cursors[..pivot_idx].iter_mut() {
                if cursor.current_doc().is_some_and(|d| d < pivot_doc) {
                    cursor.advance_to(pivot_doc);
                    if collect_stats {
                        stats.cursor_skips += 1;
                    }
                    break;
                }
            }
        }
    }

    let mut results: Vec<(u32, f32)> = heap
        .into_iter()
        .map(|Reverse((OrdF32(score), doc_id))| (doc_id, score))
        .collect();
    results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    (results, stats)
}

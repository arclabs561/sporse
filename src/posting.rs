pub(crate) const BLOCK_SIZE: usize = 128;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct PostingEntry {
    pub doc_id: u32,
    pub weight: f32,
}

/// A posting list for a single dimension, with block-max metadata for WAND pruning.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct PostingList {
    entries: Vec<PostingEntry>,
    block_maxes: Vec<f32>,
    /// Maximum weight across all entries in this list.
    pub max_weight: f32,
}

impl PostingList {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            block_maxes: Vec::new(),
            max_weight: 0.0,
        }
    }

    pub fn push(&mut self, doc_id: u32, weight: f32) {
        self.entries.push(PostingEntry { doc_id, weight });
    }

    /// Sort entries by doc_id and compute block-max metadata.
    pub fn finalize(&mut self) {
        self.entries.sort_unstable_by_key(|e| e.doc_id);

        self.block_maxes.clear();
        self.max_weight = 0.0;

        for chunk in self.entries.chunks(BLOCK_SIZE) {
            let bmax = chunk.iter().map(|e| e.weight).fold(0.0f32, f32::max);
            self.block_maxes.push(bmax);
            self.max_weight = self.max_weight.max(bmax);
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn entries(&self) -> &[PostingEntry] {
        &self.entries
    }

    #[inline]
    pub fn block_max(&self, block_idx: usize) -> f32 {
        self.block_maxes.get(block_idx).copied().unwrap_or(0.0)
    }
}

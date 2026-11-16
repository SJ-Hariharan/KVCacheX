//! Counters and point-in-time statistics.

/// Monotonic counters, updated by the manager on every operation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counters {
    pub sequences_created: u64,
    pub sequences_released: u64,
    pub forks: u64,
    /// Blocks handed out fresh (includes copy-on-write targets).
    pub blocks_allocated: u64,
    /// Operations rejected for lack of blocks.
    pub allocation_failures: u64,
    /// Cached blocks whose content was dropped to satisfy an allocation.
    pub evictions: u64,
    /// Full prompt blocks looked up in the prefix cache.
    pub prefix_query_blocks: u64,
    /// Of those, blocks served from the cache.
    pub prefix_hit_blocks: u64,
    /// Hits that revived a block from the free queue (ref_count was 0).
    pub prefix_revived_blocks: u64,
    /// Prompt tokens whose KV was found in the cache.
    pub prefix_hit_tokens: u64,
    /// Prompt tokens submitted.
    pub prompt_tokens: u64,
    pub cow_copies: u64,
    pub blocks_registered: u64,
}

impl Counters {
    pub fn prefix_hit_rate(&self) -> f64 {
        ratio(self.prefix_hit_blocks, self.prefix_query_blocks)
    }
    /// Fraction of prompt tokens served from the cache.
    pub fn token_hit_rate(&self) -> f64 {
        ratio(self.prefix_hit_tokens, self.prompt_tokens)
    }
}

/// Snapshot of pool occupancy.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheStats {
    pub block_size: u32,
    pub total_blocks: u32,
    pub used_blocks: u32,
    /// Free blocks (clean + cached).
    pub free_blocks: u32,
    /// Free blocks still holding reusable prefix content.
    pub cached_free_blocks: u32,
    pub registered_blocks: u32,
    pub peak_used_blocks: u32,
    pub live_sequences: usize,
    /// Sum of sequence lengths (shared tokens counted once per sequence).
    pub logical_tokens: u64,
    /// Block-table entries across sequences (shared blocks counted once per sequence).
    pub logical_blocks: u64,
    /// Token slots holding data in used blocks (each physical block counted once).
    pub physical_tokens: u64,
    pub counters: Counters,
}

impl CacheStats {
    /// Share of used slots that hold no token: the paged allocator's only waste.
    pub fn internal_fragmentation(&self) -> f64 {
        let cap = self.used_blocks as u64 * self.block_size as u64;
        if cap == 0 {
            0.0
        } else {
            1.0 - ratio(self.physical_tokens, cap)
        }
    }
    /// Logical blocks per physical used block; > 1 means sharing saves memory.
    pub fn sharing_factor(&self) -> f64 {
        if self.used_blocks == 0 {
            1.0
        } else {
            self.logical_blocks as f64 / self.used_blocks as f64
        }
    }
    pub fn utilization(&self) -> f64 {
        ratio(self.used_blocks as u64, self.total_blocks as u64)
    }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}

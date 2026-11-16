//! Sequence-level cache lifecycle on top of [`BlockPool`].
//!
//! # Lifecycle
//!
//! 1. [`CacheManager::create_sequence`] reserves blocks for a prompt. Leading full blocks
//!    found in the prefix cache are shared, and their tokens count as already computed.
//! 2. The engine computes KV for positions `computed..len` and writes them through the
//!    block table ([`CacheManager::slot`]).
//! 3. [`CacheManager::mark_computed`] records progress. Only *full, computed* blocks are
//!    published to the prefix cache, so no request can ever hit a block whose KV is unwritten.
//! 4. [`CacheManager::append_tokens`] reserves slots for decoded tokens, copying a shared
//!    partial tail block first (copy-on-write) when a fork shares it.
//! 5. [`CacheManager::fork`] shares every block with a new sequence (parallel sampling, beam search).
//! 6. [`CacheManager::release`] drops references. Blocks that reach zero references keep
//!    their cached content until the space is needed.
//!
//! Every fallible operation checks capacity before mutating, so on `Err` nothing changed.
//!
//! # Write safety
//!
//! A position may be written only while it is not yet computed. Uncomputed positions always
//! live in blocks owned by exactly one sequence (`ref_count == 1`) and not in the prefix
//! cache. [`CacheManager::check_invariants`] verifies this.

use std::collections::HashMap;
use std::fmt;

use crate::error::CacheError;
use crate::pool::{BlockId, BlockPool, ROOT_CID};
use crate::stats::{CacheStats, Counters};

/// Sequence handle. Ids are never reused, so a released handle can never resolve to a
/// different sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(u64);

impl SeqId {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seq#{}", self.0)
    }
}

/// Static configuration of a cache manager.
///
/// A manager serves exactly one model and KV layout, so model compatibility is structural.
/// Anything else that changes KV content for the same token ids (LoRA adapter, multimodal
/// inputs, a different system-level salt) must go into the `namespace` passed to
/// [`CacheManager::create_sequence`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheConfig {
    /// Token slots per block.
    pub block_size: u32,
    pub num_blocks: u32,
    pub enable_prefix_caching: bool,
}

/// A block copy the storage layer must perform before the next write (copy-on-write).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockCopy {
    pub src: BlockId,
    pub dst: BlockId,
    /// Number of leading token slots to copy.
    pub slots: u32,
}

/// Result of admitting a new sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Admission {
    pub seq: SeqId,
    /// Leading prompt tokens whose KV is already present (prefix-cache hits).
    /// The engine needs to compute only positions `cached_tokens..prompt_len`. If the
    /// engine needs logits for the last prompt token, it must still run that token; doing
    /// so writes nothing into the shared block if the attention kernel skips cached slots.
    pub cached_tokens: usize,
}

#[derive(Debug, Clone)]
struct Sequence {
    namespace: u64,
    tokens: Vec<u32>,
    blocks: Vec<BlockId>,
    computed: usize,
    /// Leading blocks already published to (or matched in) the prefix cache.
    published_blocks: usize,
    /// Content id of the last published block (chain parent for the next one).
    last_cid: u64,
}

/// Paged KV-cache manager: block tables, sharing, prefix caching and eviction.
///
/// Not internally synchronized; wrap it in [`crate::SharedCacheManager`] to share across threads.
#[derive(Debug)]
pub struct CacheManager {
    cfg: CacheConfig,
    pool: BlockPool,
    seqs: HashMap<u64, Sequence>,
    next_seq: u64,
    peak_used: u32,
    counters: Counters,
}

impl CacheManager {
    pub fn new(cfg: CacheConfig) -> Self {
        CacheManager {
            cfg,
            pool: BlockPool::new(cfg.num_blocks, cfg.block_size),
            seqs: HashMap::new(),
            next_seq: 1,
            peak_used: 0,
            counters: Counters::default(),
        }
    }

    pub fn config(&self) -> CacheConfig {
        self.cfg
    }
    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }
    pub fn num_free_blocks(&self) -> u32 {
        self.pool.num_free()
    }
    pub fn counters(&self) -> &Counters {
        &self.counters
    }
    pub fn num_sequences(&self) -> usize {
        self.seqs.len()
    }

    fn bs(&self) -> usize {
        self.cfg.block_size as usize
    }

    fn seq(&self, id: SeqId) -> Result<&Sequence, CacheError> {
        self.seqs.get(&id.0).ok_or(CacheError::UnknownSequence(id))
    }

    fn note_peak(&mut self) {
        self.peak_used = self.peak_used.max(self.pool.num_used());
    }

    fn take_fresh(&mut self) -> BlockId {
        let fresh = self
            .pool
            .alloc_fresh()
            .expect("capacity checked before allocation");
        self.counters.blocks_allocated += 1;
        if fresh.evicted {
            self.counters.evictions += 1;
        }
        fresh.id
    }

    /// Number of blocks a prompt of `prompt` tokens would take from the free queue right now,
    /// accounting for prefix hits. Read-only; useful for admission control.
    pub fn blocks_needed(&self, namespace: u64, prompt: &[u32]) -> u32 {
        let (hits, _) = self.match_prefix(namespace, prompt);
        let total = prompt.len().div_ceil(self.bs());
        let revive = hits.iter().filter(|&&i| self.pool.ref_count(i) == 0).count();
        (total - hits.len() + revive) as u32
    }

    /// Prompt tokens that would be served from the prefix cache right now. Read-only.
    pub fn cached_prefix_len(&self, namespace: u64, prompt: &[u32]) -> usize {
        self.match_prefix(namespace, prompt).0.len() * self.bs()
    }

    /// Longest chain of cached full blocks matching the start of `prompt`.
    fn match_prefix(&self, namespace: u64, prompt: &[u32]) -> (Vec<u32>, u64) {
        let mut hits = Vec::new();
        let mut cid = ROOT_CID;
        if self.cfg.enable_prefix_caching {
            for chunk in prompt.chunks_exact(self.bs()) {
                match self.pool.lookup(namespace, cid, chunk) {
                    Some((idx, c)) => {
                        hits.push(idx);
                        cid = c;
                    }
                    None => break,
                }
            }
        }
        (hits, cid)
    }

    /// Admits a sequence with the given prompt, sharing cached prefix blocks.
    pub fn create_sequence(&mut self, namespace: u64, prompt: &[u32]) -> Result<Admission, CacheError> {
        let bs = self.bs();
        let (hits, last_cid) = self.match_prefix(namespace, prompt);
        let total = prompt.len().div_ceil(bs);
        let revive = hits.iter().filter(|&&i| self.pool.ref_count(i) == 0).count();
        let needed = (total - hits.len() + revive) as u32;
        if needed > self.pool.num_free() {
            self.counters.allocation_failures += 1;
            return Err(CacheError::OutOfBlocks {
                needed,
                available: self.pool.num_free(),
            });
        }

        // Pin hits first so allocating fresh blocks cannot evict them.
        let mut blocks = Vec::with_capacity(total);
        for &i in &hits {
            if self.pool.incref(i) {
                self.counters.prefix_revived_blocks += 1;
            }
            blocks.push(self.pool.id_of(i));
        }
        for b in hits.len()..total {
            let id = self.take_fresh();
            let filled = (prompt.len() - b * bs).min(bs) as u32;
            self.pool.set_filled(id.index(), filled);
            blocks.push(id);
        }

        let cached_tokens = hits.len() * bs;
        let c = &mut self.counters;
        c.sequences_created += 1;
        c.prompt_tokens += prompt.len() as u64;
        if self.cfg.enable_prefix_caching {
            c.prefix_query_blocks += (prompt.len() / bs) as u64;
            c.prefix_hit_blocks += hits.len() as u64;
            c.prefix_hit_tokens += cached_tokens as u64;
        }

        let id = SeqId(self.next_seq);
        self.next_seq += 1;
        self.seqs.insert(
            id.0,
            Sequence {
                namespace,
                tokens: prompt.to_vec(),
                blocks,
                computed: cached_tokens,
                published_blocks: hits.len(),
                last_cid,
            },
        );
        self.note_peak();
        Ok(Admission {
            seq: id,
            cached_tokens,
        })
    }

    /// Reserves slots for `tokens` at the end of the sequence. Returns the block copies the
    /// storage must perform first (non-empty only when a fork shares the partial tail block).
    pub fn append_tokens(&mut self, seq: SeqId, tokens: &[u32]) -> Result<Vec<BlockCopy>, CacheError> {
        let bs = self.bs();
        let s = self.seq(seq)?;
        let len = s.tokens.len();
        let tail_partial_shared = len % bs != 0
            && s.blocks
                .last()
                .is_some_and(|b| self.pool.ref_count(b.index()) > 1)
            && !tokens.is_empty();
        let new_blocks = (len + tokens.len()).div_ceil(bs) - s.blocks.len();
        let needed = (new_blocks + tail_partial_shared as usize) as u32;
        if needed > self.pool.num_free() {
            self.counters.allocation_failures += 1;
            return Err(CacheError::OutOfBlocks {
                needed,
                available: self.pool.num_free(),
            });
        }

        let mut copies = Vec::new();
        if tail_partial_shared {
            let dst = self.take_fresh();
            let s = self.seqs.get_mut(&seq.0).expect("checked");
            let src = *s.blocks.last().expect("partial tail exists");
            let slots = (len % bs) as u32;
            *s.blocks.last_mut().expect("partial tail exists") = dst;
            self.pool.set_filled(dst.index(), slots);
            self.pool.decref(src);
            self.counters.cow_copies += 1;
            copies.push(BlockCopy { src, dst, slots });
        }
        for _ in 0..new_blocks {
            let id = self.take_fresh();
            self.seqs.get_mut(&seq.0).expect("checked").blocks.push(id);
        }
        let s = self.seqs.get_mut(&seq.0).expect("checked");
        s.tokens.extend_from_slice(tokens);
        // Update fill levels of every block touched by the append.
        let first = len / bs;
        let new_len = s.tokens.len();
        let touched: Vec<(u32, u32)> = (first..s.blocks.len())
            .map(|b| (s.blocks[b].index(), (new_len - b * bs).min(bs) as u32))
            .collect();
        for (idx, filled) in touched {
            self.pool.set_filled(idx, filled);
        }
        self.note_peak();
        Ok(copies)
    }

    /// Records that KV for positions `0..computed` is written, and publishes newly completed
    /// full blocks to the prefix cache.
    pub fn mark_computed(&mut self, seq: SeqId, computed: usize) -> Result<(), CacheError> {
        let s = self.seq(seq)?;
        if computed < s.computed || computed > s.tokens.len() {
            return Err(CacheError::InvalidComputed {
                seq,
                requested: computed,
                computed: s.computed,
                len: s.tokens.len(),
            });
        }
        let bs = self.bs();
        let s = self.seqs.get_mut(&seq.0).expect("checked");
        s.computed = computed;
        if !self.cfg.enable_prefix_caching {
            return Ok(());
        }
        let full = computed / bs;
        while s.published_blocks < full {
            let b = s.published_blocks;
            let idx = s.blocks[b].index();
            let toks = &s.tokens[b * bs..(b + 1) * bs];
            let was_cached = self.pool.is_cached(idx);
            s.last_cid = self.pool.register(idx, s.namespace, s.last_cid, toks);
            if !was_cached && self.pool.is_cached(idx) {
                self.counters.blocks_registered += 1;
            }
            s.published_blocks += 1;
        }
        Ok(())
    }

    /// Creates a new sequence sharing all of `parent`'s blocks. The parent must be fully computed.
    pub fn fork(&mut self, parent: SeqId) -> Result<SeqId, CacheError> {
        let s = self.seq(parent)?;
        if s.computed != s.tokens.len() {
            return Err(CacheError::NotFullyComputed {
                seq: parent,
                computed: s.computed,
                len: s.tokens.len(),
            });
        }
        let child = s.clone();
        for b in &child.blocks {
            let revived = self.pool.incref(b.index());
            debug_assert!(!revived, "parent holds a reference");
        }
        let id = SeqId(self.next_seq);
        self.next_seq += 1;
        self.seqs.insert(id.0, child);
        self.counters.forks += 1;
        Ok(id)
    }

    /// Releases the sequence. Blocks are released tail-first so that, among blocks of the
    /// same request, the deepest (least reusable) ones are evicted first.
    pub fn release(&mut self, seq: SeqId) -> Result<(), CacheError> {
        let s = self.seqs.remove(&seq.0).ok_or(CacheError::UnknownSequence(seq))?;
        for &b in s.blocks.iter().rev() {
            self.pool.decref(b);
        }
        self.counters.sequences_released += 1;
        Ok(())
    }

    pub fn block_table(&self, seq: SeqId) -> Result<&[BlockId], CacheError> {
        Ok(&self.seq(seq)?.blocks)
    }
    pub fn len(&self, seq: SeqId) -> Result<usize, CacheError> {
        Ok(self.seq(seq)?.tokens.len())
    }
    pub fn computed(&self, seq: SeqId) -> Result<usize, CacheError> {
        Ok(self.seq(seq)?.computed)
    }
    pub fn tokens(&self, seq: SeqId) -> Result<&[u32], CacheError> {
        Ok(&self.seq(seq)?.tokens)
    }

    /// Physical location `(block, slot)` of position `pos`.
    pub fn slot(&self, seq: SeqId, pos: usize) -> Result<(BlockId, u32), CacheError> {
        let s = self.seq(seq)?;
        if pos >= s.tokens.len() {
            return Err(CacheError::PositionOutOfRange {
                seq,
                pos,
                len: s.tokens.len(),
            });
        }
        let bs = self.bs();
        Ok((s.blocks[pos / bs], (pos % bs) as u32))
    }

    /// Like [`Self::slot`], but only for positions that may be written (not yet computed).
    pub fn writable_slot(&self, seq: SeqId, pos: usize) -> Result<(BlockId, u32), CacheError> {
        let (b, off) = self.slot(seq, pos)?;
        if pos < self.seq(seq)?.computed {
            return Err(CacheError::ReadOnlyPosition { seq, pos });
        }
        debug_assert_eq!(self.pool.ref_count(b.index()), 1);
        Ok((b, off))
    }

    /// Drops cached content from free blocks (e.g. after a model weight change).
    pub fn clear_prefix_cache(&mut self) -> u32 {
        self.pool.clear_cached_free()
    }

    pub fn stats(&self) -> CacheStats {
        let mut logical_tokens = 0u64;
        let mut logical_blocks = 0u64;
        for s in self.seqs.values() {
            logical_tokens += s.tokens.len() as u64;
            logical_blocks += s.blocks.len() as u64;
        }
        let physical_tokens = (0..self.pool.num_blocks())
            .filter(|&i| self.pool.ref_count(i) > 0)
            .map(|i| self.pool.filled(i) as u64)
            .sum();
        CacheStats {
            block_size: self.cfg.block_size,
            total_blocks: self.pool.num_blocks(),
            used_blocks: self.pool.num_used(),
            free_blocks: self.pool.num_free(),
            cached_free_blocks: self.pool.num_cached_free(),
            registered_blocks: self.pool.num_registered(),
            peak_used_blocks: self.peak_used,
            live_sequences: self.seqs.len(),
            logical_tokens,
            logical_blocks,
            physical_tokens,
            counters: self.counters.clone(),
        }
    }

    /// Verifies all manager and pool invariants. O(total blocks + total tokens); for tests.
    pub fn check_invariants(&self) -> Result<(), String> {
        self.pool.check_invariants()?;
        let bs = self.bs();
        let mut refs = vec![0u32; self.pool.num_blocks() as usize];
        for (&sid, s) in &self.seqs {
            if s.blocks.len() != s.tokens.len().div_ceil(bs) {
                return Err(format!(
                    "seq {sid}: {} blocks for {} tokens",
                    s.blocks.len(),
                    s.tokens.len()
                ));
            }
            if s.computed > s.tokens.len() || s.published_blocks > s.computed / bs {
                return Err(format!("seq {sid}: computed/published out of range"));
            }
            for (b, &id) in s.blocks.iter().enumerate() {
                if !self.pool.is_live(id) {
                    return Err(format!("seq {sid}: block {id:?} is not live (stale or freed)"));
                }
                refs[id.index() as usize] += 1;
                let want = (s.tokens.len() - b * bs).min(bs) as u32;
                if self.pool.filled(id.index()) != want {
                    return Err(format!(
                        "seq {sid}: block {b} filled {} != {want}",
                        self.pool.filled(id.index())
                    ));
                }
                // Does any position in [computed, len) fall inside this block?
                let has_uncomputed = (b * bs).max(s.computed) < ((b + 1) * bs).min(s.tokens.len());
                if has_uncomputed && (self.pool.ref_count(id.index()) != 1 || self.pool.is_cached(id.index()))
                {
                    return Err(format!("seq {sid}: writable block {b} is shared or cached"));
                }
            }
            // A block this sequence holds that is in the cache must hold this sequence's tokens.
            for (b, &id) in s.blocks.iter().enumerate().take(s.published_blocks) {
                if let Some(cached) = self.pool.cached_tokens(id.index()) {
                    if cached != &s.tokens[b * bs..(b + 1) * bs] {
                        return Err(format!("seq {sid}: shared block {b} holds different tokens"));
                    }
                }
            }
        }
        for (i, &r) in refs.iter().enumerate() {
            if self.pool.ref_count(i as u32) != r {
                return Err(format!(
                    "block {i}: ref_count {} but {r} block-table references",
                    self.pool.ref_count(i as u32)
                ));
            }
        }
        if self.pool.num_used() > self.pool.num_blocks() {
            return Err("used exceeds capacity".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr(blocks: u32, bs: u32, prefix: bool) -> CacheManager {
        CacheManager::new(CacheConfig {
            block_size: bs,
            num_blocks: blocks,
            enable_prefix_caching: prefix,
        })
    }

    fn prompt(n: usize, base: u32) -> Vec<u32> {
        (0..n as u32).map(|i| base + i).collect()
    }

    fn admit_computed(m: &mut CacheManager, ns: u64, p: &[u32]) -> Admission {
        let a = m.create_sequence(ns, p).unwrap();
        m.mark_computed(a.seq, p.len()).unwrap();
        a
    }

    #[test]
    fn allocates_ceil_blocks() {
        let mut m = mgr(10, 4, false);
        let a = m.create_sequence(0, &prompt(9, 0)).unwrap();
        assert_eq!(m.block_table(a.seq).unwrap().len(), 3);
        assert_eq!(m.num_free_blocks(), 7);
        m.check_invariants().unwrap();
    }

    #[test]
    fn out_of_blocks_is_atomic() {
        let mut m = mgr(2, 4, true);
        let before = m.stats();
        assert_eq!(
            m.create_sequence(0, &prompt(9, 0)),
            Err(CacheError::OutOfBlocks {
                needed: 3,
                available: 2
            })
        );
        assert_eq!(m.num_free_blocks(), 2);
        assert_eq!(m.num_sequences(), 0);
        assert_eq!(m.stats().used_blocks, before.used_blocks);
        m.check_invariants().unwrap();
    }

    #[test]
    fn prefix_hit_shares_full_blocks_only() {
        let mut m = mgr(16, 4, true);
        let p = prompt(10, 100); // 2 full blocks + 2 tokens
        let a = admit_computed(&mut m, 0, &p);
        assert_eq!(a.cached_tokens, 0);
        let b = m.create_sequence(0, &p).unwrap();
        assert_eq!(b.cached_tokens, 8);
        let ta = m.block_table(a.seq).unwrap().to_vec();
        let tb = m.block_table(b.seq).unwrap().to_vec();
        assert_eq!(ta[..2], tb[..2]);
        assert_ne!(ta[2], tb[2], "partial tail block is never shared");
        assert_eq!(m.computed(b.seq).unwrap(), 8);
        m.check_invariants().unwrap();
    }

    #[test]
    fn uncomputed_blocks_are_not_visible_to_lookups() {
        let mut m = mgr(16, 4, true);
        let p = prompt(8, 0);
        let a = m.create_sequence(0, &p).unwrap();
        // Not computed yet: a concurrent request must not hit.
        assert_eq!(m.create_sequence(0, &p).unwrap().cached_tokens, 0);
        m.mark_computed(a.seq, 4).unwrap();
        assert_eq!(m.create_sequence(0, &p).unwrap().cached_tokens, 4);
        m.check_invariants().unwrap();
    }

    #[test]
    fn namespaces_isolate_prefixes() {
        let mut m = mgr(16, 4, true);
        let p = prompt(8, 0);
        admit_computed(&mut m, 1, &p);
        assert_eq!(m.create_sequence(2, &p).unwrap().cached_tokens, 0);
        assert_eq!(m.create_sequence(1, &p).unwrap().cached_tokens, 8);
    }

    #[test]
    fn divergent_prompts_share_common_prefix() {
        let mut m = mgr(16, 4, true);
        let mut p1 = prompt(8, 0);
        p1.extend([50, 51, 52, 53]);
        let mut p2 = prompt(8, 0);
        p2.extend([60, 61, 62, 63]);
        admit_computed(&mut m, 0, &p1);
        assert_eq!(m.create_sequence(0, &p2).unwrap().cached_tokens, 8);
    }

    #[test]
    fn same_block_tokens_different_prefix_do_not_match() {
        let mut m = mgr(16, 2, true);
        admit_computed(&mut m, 0, &[1, 2, 9, 9]);
        // Second block [9,9] exists in the cache, but under a different parent.
        assert_eq!(m.create_sequence(0, &[3, 4, 9, 9]).unwrap().cached_tokens, 0);
    }

    #[test]
    fn released_blocks_stay_cached_until_evicted() {
        let mut m = mgr(4, 4, true);
        let p = prompt(8, 0);
        let a = admit_computed(&mut m, 0, &p);
        m.release(a.seq).unwrap();
        assert_eq!(m.stats().cached_free_blocks, 2);
        let b = m.create_sequence(0, &p).unwrap();
        assert_eq!(b.cached_tokens, 8);
        assert_eq!(m.counters().prefix_revived_blocks, 2);
        m.release(b.seq).unwrap();
        // Fill the pool with unrelated content → evicts.
        let c = m.create_sequence(0, &prompt(16, 1000)).unwrap();
        assert_eq!(m.counters().evictions, 2);
        m.release(c.seq).unwrap();
        assert_eq!(m.create_sequence(0, &p).unwrap().cached_tokens, 0);
        m.check_invariants().unwrap();
    }

    #[test]
    fn hits_are_pinned_before_fresh_allocation() {
        // 3 blocks; prefix cached in 2 free blocks; new prompt needs both hits + 1 fresh.
        let mut m = mgr(3, 2, true);
        let a = admit_computed(&mut m, 0, &[1, 2, 3, 4]);
        m.release(a.seq).unwrap();
        let b = m.create_sequence(0, &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(b.cached_tokens, 4);
        assert_eq!(m.counters().evictions, 0);
        m.check_invariants().unwrap();
    }

    #[test]
    fn revival_counts_against_capacity() {
        let mut m = mgr(3, 2, true);
        let a = admit_computed(&mut m, 0, &[1, 2, 3, 4]); // 2 blocks
        m.release(a.seq).unwrap();
        let _x = m.create_sequence(0, &[9]).unwrap(); // takes the clean block
                                                      // Hits revive 2 cached free blocks; one more fresh needed → 3 > 2 free.
        assert!(matches!(
            m.create_sequence(0, &[1, 2, 3, 4, 5]),
            Err(CacheError::OutOfBlocks {
                needed: 3,
                available: 2
            })
        ));
        assert_eq!(m.blocks_needed(0, &[1, 2, 3, 4]), 2);
        m.check_invariants().unwrap();
    }

    #[test]
    fn fork_shares_and_cow_on_partial_tail() {
        let mut m = mgr(16, 4, false);
        let a = admit_computed(&mut m, 0, &prompt(6, 0));
        let b = m.fork(a.seq).unwrap();
        assert_eq!(m.stats().used_blocks, 2);
        let copies = m.append_tokens(b, &[99]).unwrap();
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].slots, 2);
        let ta = m.block_table(a.seq).unwrap();
        let tb = m.block_table(b).unwrap();
        assert_eq!(ta[0], tb[0]);
        assert_ne!(ta[1], tb[1]);
        assert_eq!(copies[0].src, ta[1]);
        assert_eq!(copies[0].dst, tb[1]);
        // Parent now owns its tail alone: no copy needed.
        assert!(m.append_tokens(a.seq, &[7]).unwrap().is_empty());
        m.check_invariants().unwrap();
    }

    #[test]
    fn fork_requires_computed() {
        let mut m = mgr(16, 4, false);
        let a = m.create_sequence(0, &prompt(5, 0)).unwrap();
        assert!(matches!(m.fork(a.seq), Err(CacheError::NotFullyComputed { .. })));
    }

    #[test]
    fn append_across_block_boundary() {
        let mut m = mgr(16, 4, true);
        let a = admit_computed(&mut m, 0, &prompt(3, 0));
        m.append_tokens(a.seq, &[10, 11, 12]).unwrap();
        assert_eq!(m.block_table(a.seq).unwrap().len(), 2);
        m.mark_computed(a.seq, 6).unwrap();
        assert_eq!(m.stats().registered_blocks, 1);
        m.check_invariants().unwrap();
    }

    #[test]
    fn append_failure_is_atomic() {
        let mut m = mgr(1, 2, false);
        let a = admit_computed(&mut m, 0, &[1, 2]);
        assert!(m.append_tokens(a.seq, &[3]).is_err());
        assert_eq!(m.len(a.seq).unwrap(), 2);
        m.check_invariants().unwrap();
    }

    #[test]
    fn mark_computed_validation() {
        let mut m = mgr(4, 4, true);
        let a = m.create_sequence(0, &prompt(4, 0)).unwrap();
        m.mark_computed(a.seq, 2).unwrap();
        assert!(m.mark_computed(a.seq, 1).is_err());
        assert!(m.mark_computed(a.seq, 5).is_err());
    }

    #[test]
    fn stale_seq_handle_rejected() {
        let mut m = mgr(4, 4, false);
        let a = m.create_sequence(0, &[1]).unwrap();
        m.release(a.seq).unwrap();
        let b = m.create_sequence(0, &[1]).unwrap();
        assert_ne!(a.seq, b.seq);
        assert_eq!(m.release(a.seq), Err(CacheError::UnknownSequence(a.seq)));
        assert!(m.block_table(a.seq).is_err());
    }

    #[test]
    fn writable_slot_rejects_computed_positions() {
        let mut m = mgr(4, 4, false);
        let a = m.create_sequence(0, &prompt(4, 0)).unwrap();
        m.mark_computed(a.seq, 2).unwrap();
        assert!(m.writable_slot(a.seq, 1).is_err());
        assert!(m.writable_slot(a.seq, 2).is_ok());
        assert!(m.writable_slot(a.seq, 4).is_err());
    }

    #[test]
    fn stats_fragmentation_and_sharing() {
        let mut m = mgr(16, 4, true);
        let p = prompt(8, 0);
        admit_computed(&mut m, 0, &p);
        admit_computed(&mut m, 0, &p);
        let s = m.stats();
        assert_eq!(s.used_blocks, 2);
        assert_eq!(s.logical_blocks, 4);
        assert!((s.sharing_factor() - 2.0).abs() < 1e-12);
        assert_eq!(s.internal_fragmentation(), 0.0);
        let mut m2 = mgr(16, 4, false);
        m2.create_sequence(0, &prompt(5, 0)).unwrap();
        assert!((m2.stats().internal_fragmentation() - 3.0 / 8.0).abs() < 1e-12);
    }
}

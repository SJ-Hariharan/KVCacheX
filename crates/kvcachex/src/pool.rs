//! Fixed-capacity pool of KV blocks: reference counts, the free/LRU queue and the prefix index.
//!
//! # Block states
//!
//! ```text
//!            alloc_fresh                      register (block full + computed)
//!   FREE ─────────────────▶ OWNED (ref ≥ 1) ───────────────────────────▶ OWNED + CACHED
//!    ▲  (clean, at queue head)   │ decref→0                                   │ decref→0
//!    └───────────────────────────┘                                            ▼
//!    ▲                                                           FREE + CACHED (queue tail, LRU)
//!    │  alloc_fresh pops it from the queue head: unregister = eviction       │  prefix hit:
//!    └──────────────────────────────────────────────────────────────────────┘  incref revives
//! ```
//!
//! Every block with `ref_count == 0` is in one intrusive doubly linked free queue.
//! Clean (uncached) blocks are pushed to the head and cached blocks to the tail, so
//! allocation uses clean blocks first and evicts cached blocks in LRU order only when it
//! has to. A cached free block hit by a prefix lookup is unlinked in O(1).
//!
//! # Exact prefix matching
//!
//! A registered block records `(namespace, parent_cid, tokens)` and gets a fresh, never
//! reused content id (`cid`). The index maps `hash(namespace, parent_cid, tokens)` to a
//! bucket of blocks, and a lookup accepts a block only if all three fields match exactly.
//! Because `parent_cid` identifies one exact registered predecessor, a match at depth `i`
//! implies (by induction from the root, cid 0) that the whole prefix of `i` blocks is
//! identical. Hash collisions cannot cause wrong sharing.

use std::collections::HashMap;

use crate::hash::{block_hash, PassThroughBuild};

const NIL: u32 = u32::MAX;

/// Content id of the virtual root block that every prefix chain starts from.
pub const ROOT_CID: u64 = 0;

/// Handle to a physical block. `generation` changes each time the block is handed out
/// fresh, so a handle kept after its block was evicted and reused is detectably stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId {
    index: u32,
    generation: u32,
}

impl BlockId {
    /// Physical index, usable as an offset into KV storage.
    pub const fn index(self) -> u32 {
        self.index
    }
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheTag {
    hash: u64,
    namespace: u64,
    parent_cid: u64,
    cid: u64,
}

#[derive(Debug, Clone)]
struct Meta {
    ref_count: u32,
    generation: u32,
    /// Token slots holding data (bookkeeping for fragmentation stats).
    filled: u32,
    cached: Option<CacheTag>,
    in_free: bool,
    prev: u32,
    next: u32,
    /// Next block in the same prefix-index bucket.
    bucket_next: u32,
}

/// Outcome of handing out a fresh block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Fresh {
    pub id: BlockId,
    /// The block held cached content that was dropped to make room.
    pub evicted: bool,
}

/// Fixed pool of `num_blocks` blocks of `block_size` token slots each.
#[derive(Debug)]
pub struct BlockPool {
    block_size: u32,
    meta: Vec<Meta>,
    /// Token ids of registered blocks, `block_size` per block (used for exact matching).
    tokens: Vec<u32>,
    head: u32,
    tail: u32,
    free_len: u32,
    cached_free: u32,
    index: HashMap<u64, u32, PassThroughBuild>,
    registered: u32,
    next_cid: u64,
}

impl BlockPool {
    pub fn new(num_blocks: u32, block_size: u32) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        assert!(num_blocks < NIL, "num_blocks must be < u32::MAX");
        let mut pool = BlockPool {
            block_size,
            meta: Vec::with_capacity(num_blocks as usize),
            tokens: vec![0; num_blocks as usize * block_size as usize],
            head: NIL,
            tail: NIL,
            free_len: 0,
            cached_free: 0,
            index: HashMap::default(),
            registered: 0,
            next_cid: ROOT_CID + 1,
        };
        for _ in 0..num_blocks {
            pool.meta.push(Meta {
                ref_count: 0,
                generation: 0,
                filled: 0,
                cached: None,
                in_free: false,
                prev: NIL,
                next: NIL,
                bucket_next: NIL,
            });
        }
        for i in 0..num_blocks {
            pool.push_back(i);
        }
        pool
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }
    pub fn num_blocks(&self) -> u32 {
        self.meta.len() as u32
    }
    /// Blocks with `ref_count == 0` (clean or cached). All of them can be handed out.
    pub fn num_free(&self) -> u32 {
        self.free_len
    }
    pub fn num_used(&self) -> u32 {
        self.num_blocks() - self.free_len
    }
    /// Free blocks that still hold reusable cached content.
    pub fn num_cached_free(&self) -> u32 {
        self.cached_free
    }
    /// Blocks registered in the prefix index (owned or free).
    pub fn num_registered(&self) -> u32 {
        self.registered
    }

    /// True if `id` refers to a block that is currently owned and has not been reused.
    pub fn is_live(&self, id: BlockId) -> bool {
        self.meta
            .get(id.index as usize)
            .is_some_and(|m| m.generation == id.generation && m.ref_count > 0)
    }

    pub fn ref_count(&self, index: u32) -> u32 {
        self.meta[index as usize].ref_count
    }
    pub fn filled(&self, index: u32) -> u32 {
        self.meta[index as usize].filled
    }
    pub(crate) fn set_filled(&mut self, index: u32, filled: u32) {
        debug_assert!(filled <= self.block_size);
        self.meta[index as usize].filled = filled;
    }
    pub fn is_cached(&self, index: u32) -> bool {
        self.meta[index as usize].cached.is_some()
    }
    pub(crate) fn id_of(&self, index: u32) -> BlockId {
        BlockId {
            index,
            generation: self.meta[index as usize].generation,
        }
    }

    // ---- free queue -------------------------------------------------------------------

    fn push_back(&mut self, i: u32) {
        let tail = self.tail;
        let m = &mut self.meta[i as usize];
        debug_assert!(!m.in_free);
        m.in_free = true;
        m.prev = tail;
        m.next = NIL;
        if tail == NIL {
            self.head = i;
        } else {
            self.meta[tail as usize].next = i;
        }
        self.tail = i;
        self.free_len += 1;
        if self.meta[i as usize].cached.is_some() {
            self.cached_free += 1;
        }
    }

    fn push_front(&mut self, i: u32) {
        let head = self.head;
        let m = &mut self.meta[i as usize];
        debug_assert!(!m.in_free);
        m.in_free = true;
        m.prev = NIL;
        m.next = head;
        if head == NIL {
            self.tail = i;
        } else {
            self.meta[head as usize].prev = i;
        }
        self.head = i;
        self.free_len += 1;
        if self.meta[i as usize].cached.is_some() {
            self.cached_free += 1;
        }
    }

    fn unlink(&mut self, i: u32) {
        let (prev, next, cached) = {
            let m = &self.meta[i as usize];
            debug_assert!(m.in_free);
            (m.prev, m.next, m.cached.is_some())
        };
        if prev == NIL {
            self.head = next;
        } else {
            self.meta[prev as usize].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.meta[next as usize].prev = prev;
        }
        let m = &mut self.meta[i as usize];
        m.in_free = false;
        m.prev = NIL;
        m.next = NIL;
        self.free_len -= 1;
        if cached {
            self.cached_free -= 1;
        }
    }

    // ---- ownership --------------------------------------------------------------------

    /// Takes the block at the head of the free queue, evicting its cached content if any.
    pub(crate) fn alloc_fresh(&mut self) -> Option<Fresh> {
        let i = self.head;
        if i == NIL {
            return None;
        }
        self.unlink(i);
        let evicted = self.meta[i as usize].cached.is_some();
        if evicted {
            self.unregister(i);
        }
        let m = &mut self.meta[i as usize];
        m.generation = m.generation.wrapping_add(1);
        m.ref_count = 1;
        m.filled = 0;
        Some(Fresh {
            id: BlockId {
                index: i,
                generation: m.generation,
            },
            evicted,
        })
    }

    /// Adds a reference. Revives the block from the free queue if it had none.
    /// Returns `true` if the block was revived (was free).
    pub(crate) fn incref(&mut self, index: u32) -> bool {
        let revived = self.meta[index as usize].ref_count == 0;
        if revived {
            debug_assert!(
                self.meta[index as usize].cached.is_some(),
                "only cached blocks can be revived"
            );
            self.unlink(index);
        }
        let m = &mut self.meta[index as usize];
        m.ref_count = m.ref_count.checked_add(1).expect("block ref_count overflow");
        revived
    }

    /// Drops a reference held through `id`. Panics on stale handles or underflow: both
    /// are caller bugs that would otherwise corrupt shared state.
    pub(crate) fn decref(&mut self, id: BlockId) {
        let m = &mut self.meta[id.index as usize];
        assert_eq!(m.generation, id.generation, "decref through stale BlockId {id:?}");
        assert!(m.ref_count > 0, "ref_count underflow on block {id:?}");
        m.ref_count -= 1;
        if m.ref_count == 0 {
            if m.cached.is_some() {
                self.push_back(id.index); // keep content; evicted in LRU order
            } else {
                self.push_front(id.index); // nothing to keep; reuse first
            }
        }
    }

    // ---- prefix index -----------------------------------------------------------------

    fn block_tokens(&self, index: u32) -> &[u32] {
        let bs = self.block_size as usize;
        let start = index as usize * bs;
        &self.tokens[start..start + bs]
    }

    /// Token ids of a registered block, or `None` if it is not in the prefix cache.
    pub fn cached_tokens(&self, index: u32) -> Option<&[u32]> {
        self.meta[index as usize].cached.map(|_| self.block_tokens(index))
    }

    /// Finds a registered block whose content is exactly `(namespace, parent_cid, tokens)`.
    /// Returns `(index, cid)`.
    pub(crate) fn lookup(&self, namespace: u64, parent_cid: u64, tokens: &[u32]) -> Option<(u32, u64)> {
        debug_assert_eq!(tokens.len(), self.block_size as usize);
        let hash = block_hash(namespace, parent_cid, tokens);
        let mut cur = *self.index.get(&hash)?;
        while cur != NIL {
            let m = &self.meta[cur as usize];
            let tag = m.cached.expect("index points at uncached block");
            if tag.hash == hash
                && tag.namespace == namespace
                && tag.parent_cid == parent_cid
                && self.block_tokens(cur) == tokens
            {
                return Some((cur, tag.cid));
            }
            cur = m.bucket_next;
        }
        None
    }

    /// Registers owned block `index` as holding `tokens` after prefix `parent_cid`.
    ///
    /// If an identical block is already registered (for example, two requests computed the
    /// same prefix concurrently), `index` stays unregistered and the existing cid is returned,
    /// so the chain continues through the canonical copy.
    pub(crate) fn register(&mut self, index: u32, namespace: u64, parent_cid: u64, tokens: &[u32]) -> u64 {
        assert_eq!(
            tokens.len(),
            self.block_size as usize,
            "only full blocks are cached"
        );
        assert!(
            self.meta[index as usize].ref_count > 0,
            "register on unowned block"
        );
        if let Some(tag) = self.meta[index as usize].cached {
            assert!(
                tag.namespace == namespace
                    && tag.parent_cid == parent_cid
                    && self.block_tokens(index) == tokens,
                "re-registering block {index} with different content"
            );
            return tag.cid;
        }
        if let Some((_, cid)) = self.lookup(namespace, parent_cid, tokens) {
            return cid;
        }
        let hash = block_hash(namespace, parent_cid, tokens);
        let cid = self.next_cid;
        self.next_cid += 1;
        let bs = self.block_size as usize;
        let start = index as usize * bs;
        self.tokens[start..start + bs].copy_from_slice(tokens);
        let prev_head = self.index.insert(hash, index).unwrap_or(NIL);
        let m = &mut self.meta[index as usize];
        m.bucket_next = prev_head;
        m.cached = Some(CacheTag {
            hash,
            namespace,
            parent_cid,
            cid,
        });
        self.registered += 1;
        cid
    }

    fn unregister(&mut self, index: u32) {
        let tag = self.meta[index as usize]
            .cached
            .take()
            .expect("unregister uncached block");
        let next = std::mem::replace(&mut self.meta[index as usize].bucket_next, NIL);
        let head = *self
            .index
            .get(&tag.hash)
            .expect("cached block missing from index");
        if head == index {
            if next == NIL {
                self.index.remove(&tag.hash);
            } else {
                self.index.insert(tag.hash, next);
            }
        } else {
            let mut cur = head;
            loop {
                let n = self.meta[cur as usize].bucket_next;
                assert!(n != NIL, "cached block missing from its bucket");
                if n == index {
                    self.meta[cur as usize].bucket_next = next;
                    break;
                }
                cur = n;
            }
        }
        self.registered -= 1;
    }

    /// Drops all cached content held by free blocks. Owned blocks keep their registration.
    pub fn clear_cached_free(&mut self) -> u32 {
        let mut dropped = 0;
        let mut cur = self.head;
        while cur != NIL {
            let next = self.meta[cur as usize].next;
            if self.meta[cur as usize].cached.is_some() {
                self.unlink(cur);
                self.unregister(cur);
                self.push_front(cur);
                dropped += 1;
            }
            cur = next;
        }
        dropped
    }

    /// Verifies the pool's internal invariants. Intended for tests; O(num_blocks).
    pub fn check_invariants(&self) -> Result<(), String> {
        let n = self.meta.len();
        let mut seen = vec![false; n];
        let mut count = 0u32;
        let mut cached_free = 0u32;
        let mut prev = NIL;
        let mut cur = self.head;
        while cur != NIL {
            let i = cur as usize;
            if i >= n || seen[i] {
                return Err(format!("free queue cycle or bad index at {cur}"));
            }
            seen[i] = true;
            let m = &self.meta[i];
            if !m.in_free || m.ref_count != 0 {
                return Err(format!(
                    "block {cur} in free queue with ref_count {} in_free {}",
                    m.ref_count, m.in_free
                ));
            }
            if m.prev != prev {
                return Err(format!("block {cur} has broken prev link"));
            }
            if m.cached.is_some() {
                cached_free += 1;
            }
            count += 1;
            prev = cur;
            cur = m.next;
        }
        if prev != self.tail {
            return Err("free queue tail mismatch".into());
        }
        if count != self.free_len || cached_free != self.cached_free {
            return Err(format!(
                "free counters wrong: walked {count}/{cached_free}, recorded {}/{}",
                self.free_len, self.cached_free
            ));
        }
        let mut registered = 0u32;
        for (i, m) in self.meta.iter().enumerate() {
            if (m.ref_count == 0) != seen[i] || m.in_free != seen[i] {
                return Err(format!(
                    "block {i}: ref_count {} but in_free {}",
                    m.ref_count, m.in_free
                ));
            }
            if m.filled > self.block_size {
                return Err(format!("block {i} overfilled"));
            }
            if let Some(tag) = m.cached {
                registered += 1;
                if tag.cid == ROOT_CID || tag.cid >= self.next_cid {
                    return Err(format!("block {i} has invalid cid"));
                }
                if block_hash(tag.namespace, tag.parent_cid, self.block_tokens(i as u32)) != tag.hash {
                    return Err(format!("block {i} hash does not match stored tokens"));
                }
                let mut b = *self
                    .index
                    .get(&tag.hash)
                    .ok_or(format!("block {i} not indexed"))?;
                let mut found = false;
                while b != NIL {
                    if b as usize == i {
                        found = true;
                        break;
                    }
                    b = self.meta[b as usize].bucket_next;
                }
                if !found {
                    return Err(format!("block {i} not reachable from its bucket"));
                }
            }
        }
        if registered != self.registered {
            return Err(format!(
                "registered count {} != actual {registered}",
                self.registered
            ));
        }
        let mut indexed = 0u32;
        for (&h, &head) in &self.index {
            let mut b = head;
            while b != NIL {
                let tag = self.meta[b as usize]
                    .cached
                    .ok_or(format!("index entry points at uncached block {b}"))?;
                if tag.hash != h {
                    return Err(format!("block {b} in wrong bucket"));
                }
                indexed += 1;
                b = self.meta[b as usize].bucket_next;
            }
        }
        if indexed != registered {
            return Err(format!("index holds {indexed} entries, {registered} registered"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(base: u32, bs: u32) -> Vec<u32> {
        (base..base + bs).collect()
    }

    #[test]
    fn alloc_until_exhausted_then_none() {
        let mut p = BlockPool::new(3, 4);
        let a: Vec<_> = (0..3).map(|_| p.alloc_fresh().unwrap()).collect();
        assert!(p.alloc_fresh().is_none());
        assert_eq!(p.num_free(), 0);
        for f in &a {
            p.decref(f.id);
        }
        assert_eq!(p.num_free(), 3);
        p.check_invariants().unwrap();
    }

    #[test]
    fn generation_detects_stale_handle() {
        let mut p = BlockPool::new(1, 4);
        let a = p.alloc_fresh().unwrap().id;
        p.decref(a);
        let b = p.alloc_fresh().unwrap().id;
        assert_eq!(a.index(), b.index());
        assert!(!p.is_live(a));
        assert!(p.is_live(b));
    }

    #[test]
    #[should_panic(expected = "stale")]
    fn decref_stale_panics() {
        let mut p = BlockPool::new(1, 4);
        let a = p.alloc_fresh().unwrap().id;
        p.decref(a);
        let _b = p.alloc_fresh().unwrap();
        p.decref(a);
    }

    #[test]
    #[should_panic(expected = "underflow")]
    fn double_free_panics() {
        let mut p = BlockPool::new(2, 4);
        let a = p.alloc_fresh().unwrap().id;
        p.decref(a);
        p.decref(a);
    }

    #[test]
    fn clean_blocks_used_before_cached_ones() {
        let mut p = BlockPool::new(2, 4);
        let a = p.alloc_fresh().unwrap().id;
        let cid = p.register(a.index(), 0, ROOT_CID, &toks(0, 4));
        p.decref(a);
        let b = p.alloc_fresh().unwrap();
        assert!(!b.evicted);
        assert_ne!(b.id.index(), a.index());
        assert_eq!(p.lookup(0, ROOT_CID, &toks(0, 4)), Some((a.index(), cid)));
        let c = p.alloc_fresh().unwrap();
        assert!(c.evicted);
        assert_eq!(p.lookup(0, ROOT_CID, &toks(0, 4)), None);
        p.check_invariants().unwrap();
    }

    #[test]
    fn lookup_is_exact() {
        let mut p = BlockPool::new(4, 4);
        let a = p.alloc_fresh().unwrap().id;
        let cid = p.register(a.index(), 7, ROOT_CID, &toks(10, 4));
        assert!(p.lookup(7, ROOT_CID, &toks(10, 4)).is_some());
        assert!(
            p.lookup(8, ROOT_CID, &toks(10, 4)).is_none(),
            "namespace must match"
        );
        assert!(p.lookup(7, cid, &toks(10, 4)).is_none(), "parent must match");
        assert!(p.lookup(7, ROOT_CID, &toks(11, 4)).is_none(), "tokens must match");
    }

    #[test]
    fn duplicate_register_returns_canonical_cid() {
        let mut p = BlockPool::new(4, 2);
        let a = p.alloc_fresh().unwrap().id;
        let b = p.alloc_fresh().unwrap().id;
        let ca = p.register(a.index(), 0, ROOT_CID, &[1, 2]);
        let cb = p.register(b.index(), 0, ROOT_CID, &[1, 2]);
        assert_eq!(ca, cb);
        assert!(!p.is_cached(b.index()));
        assert_eq!(p.num_registered(), 1);
        p.check_invariants().unwrap();
    }

    #[test]
    fn revive_cached_free_block() {
        let mut p = BlockPool::new(2, 2);
        let a = p.alloc_fresh().unwrap().id;
        p.register(a.index(), 0, ROOT_CID, &[1, 2]);
        p.decref(a);
        assert_eq!(p.num_cached_free(), 1);
        let (i, _) = p.lookup(0, ROOT_CID, &[1, 2]).unwrap();
        assert!(p.incref(i));
        assert_eq!(p.num_free(), 1);
        assert_eq!(p.num_cached_free(), 0);
        assert!(p.is_live(a), "revived block keeps its generation: same content");
        p.check_invariants().unwrap();
    }

    #[test]
    fn eviction_is_lru_among_cached() {
        let mut p = BlockPool::new(3, 1);
        let ids: Vec<_> = (0..3).map(|_| p.alloc_fresh().unwrap().id).collect();
        for (k, id) in ids.iter().enumerate() {
            p.register(id.index(), 0, ROOT_CID, &[k as u32]);
        }
        // Free in order 2, 0, 1 → LRU order is 2, 0, 1.
        for &k in &[2usize, 0, 1] {
            p.decref(ids[k]);
        }
        assert_eq!(p.alloc_fresh().unwrap().id.index(), ids[2].index());
        assert_eq!(p.alloc_fresh().unwrap().id.index(), ids[0].index());
        assert!(p.lookup(0, ROOT_CID, &[1]).is_some());
        p.check_invariants().unwrap();
    }

    #[test]
    fn clear_cached_free_drops_only_free_blocks() {
        let mut p = BlockPool::new(3, 1);
        let a = p.alloc_fresh().unwrap().id;
        let b = p.alloc_fresh().unwrap().id;
        p.register(a.index(), 0, ROOT_CID, &[1]);
        p.register(b.index(), 0, ROOT_CID, &[2]);
        p.decref(a);
        assert_eq!(p.clear_cached_free(), 1);
        assert!(p.lookup(0, ROOT_CID, &[1]).is_none());
        assert!(p.lookup(0, ROOT_CID, &[2]).is_some());
        p.check_invariants().unwrap();
    }
}

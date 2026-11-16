//! Thread-safe handle: one mutex around a [`CacheManager`].
//!
//! Each operation is a few hundred nanoseconds of metadata work, so a coarse lock is the
//! simplest correct design. In a real engine, one scheduler thread makes all decisions for a
//! batch, so contention is rare. `kvx bench` measures the lock's throughput under contention.
//! The lock-free route was rejected for now: profiling has not shown a need (see ADR-0004).

use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::CacheError;
use crate::manager::{Admission, CacheConfig, CacheManager, SeqId};
use crate::stats::CacheStats;

#[derive(Debug, Clone)]
pub struct SharedCacheManager {
    inner: Arc<Mutex<CacheManager>>,
}

impl SharedCacheManager {
    pub fn new(cfg: CacheConfig) -> Self {
        SharedCacheManager {
            inner: Arc::new(Mutex::new(CacheManager::new(cfg))),
        }
    }

    /// Locks the manager. A panic while the lock is held poisons it. We recover the guard
    /// because every manager operation checks before it mutates; a panic means a bug that
    /// the invariant checker will report.
    pub fn lock(&self) -> MutexGuard<'_, CacheManager> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn create_sequence(&self, namespace: u64, prompt: &[u32]) -> Result<Admission, CacheError> {
        self.lock().create_sequence(namespace, prompt)
    }
    pub fn append_tokens(&self, seq: SeqId, tokens: &[u32]) -> Result<usize, CacheError> {
        self.lock().append_tokens(seq, tokens).map(|c| c.len())
    }
    pub fn mark_computed(&self, seq: SeqId, computed: usize) -> Result<(), CacheError> {
        self.lock().mark_computed(seq, computed)
    }
    pub fn fork(&self, seq: SeqId) -> Result<SeqId, CacheError> {
        self.lock().fork(seq)
    }
    pub fn release(&self, seq: SeqId) -> Result<(), CacheError> {
        self.lock().release(seq)
    }
    pub fn stats(&self) -> CacheStats {
        self.lock().stats()
    }
}

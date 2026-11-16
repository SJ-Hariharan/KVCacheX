//! # KVCacheX
//!
//! A paged KV-cache manager for LLM inference: fixed-size blocks, per-sequence block tables,
//! reference-counted sharing (fork and copy-on-write), exact prefix caching, and LRU eviction
//! of cached blocks.
//!
//! The crate manages **attention KV-cache memory**: the Key/Value tensors an inference engine
//! keeps per token. It is not API-level prompt caching.
//!
//! * [`geometry`]: per-token and per-block memory sizes for MHA, GQA and MQA models.
//! * [`BlockPool`]: block ownership, the free/LRU queue and the prefix index.
//! * [`CacheManager`]: sequence lifecycle (create, append, mark computed, fork, release).
//! * [`storage`]: tensor storage behind a trait: a CPU reference backend and, with feature
//!   `cuda`, a CUDA device backend (`CudaKvStorage`).
//! * [`baseline`]: a contiguous reservation allocator used as the comparison baseline.
//!
//! ```
//! use kvcachex::{CacheConfig, CacheManager};
//!
//! let mut m = CacheManager::new(CacheConfig { block_size: 4, num_blocks: 8, enable_prefix_caching: true });
//! let system_prompt: Vec<u32> = (0..8).collect();
//!
//! let a = m.create_sequence(0, &system_prompt).unwrap();
//! assert_eq!(a.cached_tokens, 0);
//! m.mark_computed(a.seq, system_prompt.len()).unwrap(); // engine wrote the KV
//!
//! let b = m.create_sequence(0, &system_prompt).unwrap();
//! assert_eq!(b.cached_tokens, 8);                       // both blocks shared
//! assert_eq!(m.stats().used_blocks, 2);
//! m.check_invariants().unwrap();
//! ```

pub mod baseline;
mod error;
pub mod geometry;
mod hash;
mod manager;
mod pool;
mod shared;
mod stats;
pub mod storage;

pub use error::CacheError;
pub use geometry::{DType, ModelGeometry};
pub use manager::{Admission, BlockCopy, CacheConfig, CacheManager, SeqId};
pub use pool::{BlockId, BlockPool};
pub use shared::SharedCacheManager;
pub use stats::{CacheStats, Counters};
#[cfg(feature = "cuda")]
pub use storage::cuda::{CudaDeviceInfo, CudaKvStorage};
pub use storage::{BlockLayout, CpuKvStorage, KvStorage, PagedKvCache, Residency, StorageError};

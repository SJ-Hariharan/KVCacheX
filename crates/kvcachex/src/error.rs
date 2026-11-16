use std::fmt;

use crate::manager::SeqId;
use crate::storage::StorageError;

/// Errors returned by [`crate::CacheManager`]. A failed operation leaves the manager unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// Not enough free blocks. `needed` counts blocks the operation must take from the free
    /// queue (fresh blocks plus cached free blocks it would revive).
    OutOfBlocks {
        needed: u32,
        available: u32,
    },
    UnknownSequence(SeqId),
    /// `mark_computed` went backwards or past the sequence length.
    InvalidComputed {
        seq: SeqId,
        requested: usize,
        computed: usize,
        len: usize,
    },
    /// `fork` requires all tokens of the parent to be computed, so the child never shares
    /// a block that is still being written.
    NotFullyComputed {
        seq: SeqId,
        computed: usize,
        len: usize,
    },
    /// Position is outside the sequence.
    PositionOutOfRange {
        seq: SeqId,
        pos: usize,
        len: usize,
    },
    /// Write to a position whose KV is already computed (it may be shared).
    ReadOnlyPosition {
        seq: SeqId,
        pos: usize,
    },
    /// The storage backend rejected or failed an operation.
    Storage(StorageError),
    /// An earlier fatal storage error poisoned the cache; its contents cannot be trusted.
    Poisoned(StorageError),
    /// Storage shape does not match the cache configuration.
    StorageMismatch {
        storage_block_size: usize,
        storage_blocks: usize,
        config_block_size: usize,
        config_blocks: usize,
    },
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::OutOfBlocks { needed, available } => {
                write!(f, "out of KV blocks: need {needed}, {available} available")
            }
            CacheError::UnknownSequence(s) => write!(f, "unknown sequence {s}"),
            CacheError::InvalidComputed {
                seq,
                requested,
                computed,
                len,
            } => write!(
                f,
                "invalid computed count {requested} for {seq} (computed {computed}, len {len})"
            ),
            CacheError::NotFullyComputed { seq, computed, len } => {
                write!(f, "cannot fork {seq}: only {computed}/{len} tokens computed")
            }
            CacheError::PositionOutOfRange { seq, pos, len } => {
                write!(f, "position {pos} out of range for {seq} (len {len})")
            }
            CacheError::ReadOnlyPosition { seq, pos } => {
                write!(f, "position {pos} of {seq} is already computed and read-only")
            }
            CacheError::Storage(e) => write!(f, "storage: {e}"),
            CacheError::Poisoned(e) => write!(f, "cache poisoned by earlier storage failure: {e}"),
            CacheError::StorageMismatch {
                storage_block_size,
                storage_blocks,
                config_block_size,
                config_blocks,
            } => write!(
                f,
                "storage has {storage_blocks} blocks of {storage_block_size}, config needs {config_blocks} of {config_block_size}"
            ),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<StorageError> for CacheError {
    fn from(e: StorageError) -> Self {
        CacheError::Storage(e)
    }
}

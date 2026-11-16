//! KV tensor storage, separate from cache metadata.
//!
//! [`CacheManager`] decides *where* each token's KV lives. A [`KvStorage`] backend holds the
//! values. Backends:
//!
//! * [`CpuKvStorage`]: host memory, the reference backend (always available).
//! * `CudaKvStorage` (feature `cuda`): one device allocation per cache on an NVIDIA GPU. KV values
//!   stay on the device. Copy-on-write block copies run device-to-device. Only explicit
//!   writes and reads cross PCIe.
//!
//! # Contract (see ADR-0005)
//!
//! * Layout of one block (row-major): `[layer][K=0|V=1][slot][kv_head * head_dim]`, `f32`
//!   elements whatever [`crate::DType`] the geometry names (dtype affects estimates only).
//! * Writes and reads cover `n >= 1` *consecutive slots of one block and one layer*, which are
//!   contiguous in this layout. A backend can move them with one transfer per K/V plane.
//! * Reads copy into caller buffers (`read_slots_into`). Borrowed access to stored values exists
//!   only where the values really are in host memory ([`CpuKvStorage::read`]).
//! * Every operation validates its arguments and returns [`StorageError`] instead of panicking.
//!   Device backends may queue work asynchronously; [`KvStorage::synchronize`] waits for it and
//!   reports deferred device errors. `read_slots_into` returns only after the data is in the
//!   caller's buffers.
//! * `copy_block` copies at least the first `copy.slots` slots of every (layer, K/V) plane from
//!   `src` to `dst`. A backend may copy the whole block. Slots at or beyond `copy.slots` in
//!   `dst` are unspecified afterwards (they are never read before being written).

mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

use std::fmt;

pub use cpu::CpuKvStorage;

use crate::error::CacheError;
use crate::geometry::ModelGeometry;
use crate::manager::{Admission, BlockCopy, CacheConfig, CacheManager, SeqId};
use crate::pool::BlockId;

/// Where a backend's KV values live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    Host,
    /// GPU device memory, by CUDA device ordinal.
    CudaDevice {
        ordinal: usize,
    },
}

/// Errors from a storage backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    /// An index or length was outside the storage's shape. Nothing was transferred.
    OutOfRange {
        what: &'static str,
        value: usize,
        limit: usize,
    },
    /// Buffer length does not match `n_slots * token_width`. Nothing was transferred.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// Requested storage size overflows `usize`.
    SizeOverflow,
    /// No usable device (driver library missing, no device, or invalid ordinal).
    Unavailable(String),
    /// The device or driver reported an error (allocation, copy, synchronization).
    /// The backend's state should be treated as lost.
    Device(String),
}

impl StorageError {
    /// True for errors after which the stored contents cannot be trusted.
    pub fn is_fatal(&self) -> bool {
        matches!(self, StorageError::Device(_))
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::OutOfRange { what, value, limit } => {
                write!(f, "{what} {value} out of range (limit {limit})")
            }
            StorageError::ShapeMismatch { what, expected, got } => {
                write!(f, "{what}: expected {expected} elements, got {got}")
            }
            StorageError::SizeOverflow => f.write_str("storage size overflows usize"),
            StorageError::Unavailable(m) => write!(f, "storage device unavailable: {m}"),
            StorageError::Device(m) => write!(f, "device error: {m}"),
        }
    }
}

impl std::error::Error for StorageError {}

/// Backend that stores KV tensors for physical blocks.
pub trait KvStorage {
    /// Shape of the storage.
    fn layout(&self) -> &BlockLayout;
    fn residency(&self) -> Residency;

    /// Elements of K (or V) for one token in one layer (`num_kv_heads * head_dim`).
    fn token_width(&self) -> usize {
        self.layout().width
    }

    /// Stores K and V for `n = k.len() / token_width` consecutive slots starting at `slot` in
    /// `block`, for one layer. For device backends this is a host-to-device transfer.
    fn write_slots(
        &mut self,
        block: BlockId,
        slot: u32,
        layer: u32,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), StorageError>;

    /// Copies K and V of `n = k_out.len() / token_width` consecutive slots into the caller's
    /// buffers. For device backends this is a device-to-host transfer, complete on return.
    fn read_slots_into(
        &self,
        block: BlockId,
        slot: u32,
        layer: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), StorageError>;

    /// Copies block contents storage-side (device-to-device for device backends).
    fn copy_block(&mut self, copy: BlockCopy) -> Result<(), StorageError>;

    /// Waits for queued operations and reports deferred device errors.
    fn synchronize(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

/// Shape of a block-structured KV store and checked offset arithmetic shared by backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    /// `num_kv_heads * head_dim`.
    pub width: usize,
    pub layers: usize,
    pub block_size: usize,
    pub num_blocks: usize,
}

impl BlockLayout {
    pub fn new(geom: &ModelGeometry, block_size: u32, num_blocks: u32) -> Self {
        BlockLayout {
            width: geom.kv_elems_per_token_per_layer() as usize,
            layers: geom.num_layers as usize,
            block_size: block_size as usize,
            num_blocks: num_blocks as usize,
        }
    }

    /// Elements in one (layer, K/V) plane of a block.
    pub fn plane_elems(&self) -> usize {
        self.block_size * self.width
    }

    /// Elements in one block.
    pub fn block_elems(&self) -> usize {
        self.layers * 2 * self.plane_elems()
    }

    /// Total elements, with overflow checking.
    pub fn total_elems(&self) -> Result<usize, StorageError> {
        self.layers
            .checked_mul(2)
            .and_then(|x| x.checked_mul(self.block_size))
            .and_then(|x| x.checked_mul(self.width))
            .and_then(|x| x.checked_mul(self.num_blocks))
            .ok_or(StorageError::SizeOverflow)
    }

    /// Total bytes as `f32`, with overflow checking.
    pub fn total_bytes(&self) -> Result<usize, StorageError> {
        self.total_elems()?
            .checked_mul(4)
            .ok_or(StorageError::SizeOverflow)
    }

    pub fn check_block(&self, block: BlockId) -> Result<usize, StorageError> {
        let b = block.index() as usize;
        if b >= self.num_blocks {
            return Err(StorageError::OutOfRange {
                what: "block",
                value: b,
                limit: self.num_blocks,
            });
        }
        Ok(b)
    }

    /// Validates a slot range and buffer pair, returning `(k_offset, v_offset, len)` in elements.
    pub fn slot_range(
        &self,
        block: BlockId,
        slot: u32,
        layer: u32,
        k_len: usize,
        v_len: usize,
    ) -> Result<(usize, usize, usize), StorageError> {
        let b = self.check_block(block)?;
        let layer = layer as usize;
        if layer >= self.layers {
            return Err(StorageError::OutOfRange {
                what: "layer",
                value: layer,
                limit: self.layers,
            });
        }
        if self.width == 0 || k_len == 0 || !k_len.is_multiple_of(self.width) {
            return Err(StorageError::ShapeMismatch {
                what: "k buffer (must be a non-zero multiple of token_width)",
                expected: self.width,
                got: k_len,
            });
        }
        if v_len != k_len {
            return Err(StorageError::ShapeMismatch {
                what: "v buffer",
                expected: k_len,
                got: v_len,
            });
        }
        let n = k_len / self.width;
        let slot = slot as usize;
        if slot >= self.block_size || n > self.block_size - slot {
            return Err(StorageError::OutOfRange {
                what: "slot range end",
                value: slot + n,
                limit: self.block_size,
            });
        }
        let base = b * self.block_elems() + slot * self.width;
        let k = base + (layer * 2) * self.plane_elems();
        let v = base + (layer * 2 + 1) * self.plane_elems();
        Ok((k, v, k_len))
    }

    /// Validates a block copy, returning `(src_block, dst_block)`.
    pub fn check_copy(&self, copy: BlockCopy) -> Result<(usize, usize), StorageError> {
        let s = self.check_block(copy.src)?;
        let d = self.check_block(copy.dst)?;
        if s == d {
            return Err(StorageError::OutOfRange {
                what: "copy dst (equals src)",
                value: d,
                limit: d,
            });
        }
        if copy.slots as usize > self.block_size {
            return Err(StorageError::OutOfRange {
                what: "copy slots",
                value: copy.slots as usize,
                limit: self.block_size,
            });
        }
        Ok((s, d))
    }
}

/// Metadata manager plus tensor storage, with writes checked against the block table.
///
/// This is the interface an inference runtime would use. It applies copy-on-write copies
/// automatically and refuses writes to computed (possibly shared) positions.
///
/// **Failure handling.** Argument errors from the storage are returned and change nothing.
/// After a *fatal* storage error ([`StorageError::is_fatal`]), the cache is **poisoned**. For
/// example, a failed device copy after the block table already points at the copy target. A
/// poisoned cache returns [`CacheError::Poisoned`] from every data or lifecycle call, rather
/// than serving possibly corrupt KV. Metadata accessors keep working for diagnosis.
#[derive(Debug)]
pub struct PagedKvCache<S: KvStorage = CpuKvStorage> {
    manager: CacheManager,
    storage: S,
    poisoned: Option<StorageError>,
}

impl PagedKvCache<CpuKvStorage> {
    pub fn new_cpu(geom: &ModelGeometry, cfg: CacheConfig) -> Self {
        PagedKvCache {
            storage: CpuKvStorage::new(geom, cfg.block_size, cfg.num_blocks),
            manager: CacheManager::new(cfg),
            poisoned: None,
        }
    }

    /// Zero-copy view of a position's K and V (host storage only).
    pub fn read(&self, seq: SeqId, pos: usize, layer: u32) -> Result<(&[f32], &[f32]), CacheError> {
        self.check_poison()?;
        let (block, slot) = self.manager.slot(seq, pos)?;
        Ok(self.storage.read(block, slot, layer)?)
    }
}

impl<S: KvStorage> PagedKvCache<S> {
    /// Combines a manager for `cfg` with `storage`. The storage must have the same block size
    /// and at least as many blocks as the manager.
    pub fn with_storage(cfg: CacheConfig, storage: S) -> Result<Self, CacheError> {
        let l = storage.layout();
        if l.block_size != cfg.block_size as usize || l.num_blocks < cfg.num_blocks as usize {
            return Err(CacheError::StorageMismatch {
                storage_block_size: l.block_size,
                storage_blocks: l.num_blocks,
                config_block_size: cfg.block_size as usize,
                config_blocks: cfg.num_blocks as usize,
            });
        }
        Ok(PagedKvCache {
            manager: CacheManager::new(cfg),
            storage,
            poisoned: None,
        })
    }

    pub fn manager(&self) -> &CacheManager {
        &self.manager
    }
    pub fn storage(&self) -> &S {
        &self.storage
    }
    /// The fatal storage error that poisoned this cache, if any.
    pub fn poison(&self) -> Option<&StorageError> {
        self.poisoned.as_ref()
    }

    fn check_poison(&self) -> Result<(), CacheError> {
        match &self.poisoned {
            Some(e) => Err(CacheError::Poisoned(e.clone())),
            None => Ok(()),
        }
    }

    fn storage_result<T>(&mut self, r: Result<T, StorageError>) -> Result<T, CacheError> {
        r.map_err(|e| {
            if e.is_fatal() {
                self.poisoned = Some(e.clone());
            }
            CacheError::Storage(e)
        })
    }

    pub fn create_sequence(&mut self, namespace: u64, prompt: &[u32]) -> Result<Admission, CacheError> {
        self.check_poison()?;
        self.manager.create_sequence(namespace, prompt)
    }

    /// Reserves slots for `tokens` and performs any copy-on-write block copies in storage.
    pub fn append_tokens(&mut self, seq: SeqId, tokens: &[u32]) -> Result<(), CacheError> {
        self.check_poison()?;
        for c in self.manager.append_tokens(seq, tokens)? {
            let r = self.storage.copy_block(c);
            self.storage_result(r)?;
        }
        Ok(())
    }

    /// Writes KV for position `pos` in `layer`. Fails if the position is already computed.
    pub fn write(
        &mut self,
        seq: SeqId,
        pos: usize,
        layer: u32,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), CacheError> {
        self.write_range(seq, pos, layer, k, v)
    }

    /// Writes KV for consecutive positions `start..start + n` of one layer, where
    /// `n = k.len() / token_width`. Positions may span blocks; each block's part is one
    /// storage write. All positions must be writable (not yet computed).
    pub fn write_range(
        &mut self,
        seq: SeqId,
        start: usize,
        layer: u32,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), CacheError> {
        self.check_poison()?;
        let w = self.storage.token_width();
        if w == 0 || k.is_empty() || !k.len().is_multiple_of(w) || v.len() != k.len() {
            let r = Err(StorageError::ShapeMismatch {
                what: "k/v buffers",
                expected: w,
                got: k.len(),
            });
            return self.storage_result(r);
        }
        let n = k.len() / w;
        // Validate every position before writing any of them.
        self.manager.writable_slot(seq, start)?;
        self.manager.writable_slot(seq, start + n - 1)?;
        let bs = self.manager.config().block_size as usize;
        let mut pos = start;
        while pos < start + n {
            let (block, slot) = self.manager.writable_slot(seq, pos)?;
            let run = (bs - slot as usize).min(start + n - pos);
            let (a, b) = ((pos - start) * w, (pos - start + run) * w);
            let r = self.storage.write_slots(block, slot, layer, &k[a..b], &v[a..b]);
            self.storage_result(r)?;
            pos += run;
        }
        Ok(())
    }

    /// Copies K and V of position `pos` in `layer` into the caller's buffers
    /// (a device-to-host transfer for device backends).
    pub fn read_into(
        &self,
        seq: SeqId,
        pos: usize,
        layer: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CacheError> {
        self.check_poison()?;
        let (block, slot) = self.manager.slot(seq, pos)?;
        self.storage
            .read_slots_into(block, slot, layer, k_out, v_out)
            .map_err(CacheError::Storage)
    }

    /// Gathers K and V of consecutive positions `start..start + n` of one layer into the caller's
    /// buffers, where `n = k_out.len() / token_width`, in position order (the layout a dense
    /// attention kernel expects). Positions may span blocks; each block's part is one storage
    /// read (a device-to-host transfer for device backends). Every position must be below the
    /// sequence length; reading a position that was never written returns stale slot contents.
    pub fn read_range_into(
        &self,
        seq: SeqId,
        start: usize,
        layer: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CacheError> {
        self.check_poison()?;
        let w = self.storage.token_width();
        if w == 0 || k_out.is_empty() || !k_out.len().is_multiple_of(w) || v_out.len() != k_out.len() {
            return Err(CacheError::Storage(StorageError::ShapeMismatch {
                what: "k/v buffers",
                expected: w,
                got: k_out.len(),
            }));
        }
        let n = k_out.len() / w;
        self.manager.slot(seq, start + n - 1)?;
        let bs = self.manager.config().block_size as usize;
        let mut pos = start;
        while pos < start + n {
            let (block, slot) = self.manager.slot(seq, pos)?;
            let run = (bs - slot as usize).min(start + n - pos);
            let (a, b) = ((pos - start) * w, (pos - start + run) * w);
            self.storage
                .read_slots_into(block, slot, layer, &mut k_out[a..b], &mut v_out[a..b])
                .map_err(CacheError::Storage)?;
            pos += run;
        }
        Ok(())
    }

    pub fn mark_computed(&mut self, seq: SeqId, computed: usize) -> Result<(), CacheError> {
        self.check_poison()?;
        self.manager.mark_computed(seq, computed)
    }

    pub fn fork(&mut self, seq: SeqId) -> Result<SeqId, CacheError> {
        self.check_poison()?;
        self.manager.fork(seq)
    }

    pub fn release(&mut self, seq: SeqId) -> Result<(), CacheError> {
        self.check_poison()?;
        self.manager.release(seq)
    }

    pub fn clear_prefix_cache(&mut self) -> u32 {
        self.manager.clear_prefix_cache()
    }

    /// Waits for queued storage work; a fatal error poisons the cache.
    pub fn synchronize(&mut self) -> Result<(), CacheError> {
        self.check_poison()?;
        let r = self.storage.synchronize();
        self.storage_result(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host storage that can be told to fail copies with a fatal error.
    #[derive(Debug)]
    struct FlakyStorage {
        inner: CpuKvStorage,
        fail_copies: bool,
    }

    impl KvStorage for FlakyStorage {
        fn layout(&self) -> &BlockLayout {
            self.inner.layout()
        }
        fn residency(&self) -> Residency {
            Residency::Host
        }
        fn write_slots(
            &mut self,
            b: BlockId,
            s: u32,
            l: u32,
            k: &[f32],
            v: &[f32],
        ) -> Result<(), StorageError> {
            self.inner.write_slots(b, s, l, k, v)
        }
        fn read_slots_into(
            &self,
            b: BlockId,
            s: u32,
            l: u32,
            k: &mut [f32],
            v: &mut [f32],
        ) -> Result<(), StorageError> {
            self.inner.read_slots_into(b, s, l, k, v)
        }
        fn copy_block(&mut self, c: BlockCopy) -> Result<(), StorageError> {
            if self.fail_copies {
                return Err(StorageError::Device("injected copy failure".into()));
            }
            self.inner.copy_block(c)
        }
    }

    fn cfg() -> CacheConfig {
        CacheConfig {
            block_size: 4,
            num_blocks: 8,
            enable_prefix_caching: false,
        }
    }

    #[test]
    fn fatal_copy_error_poisons_cache() {
        let g = ModelGeometry::preset("tiny").unwrap();
        let storage = FlakyStorage {
            inner: CpuKvStorage::new(&g, 4, 8),
            fail_copies: true,
        };
        let mut c = PagedKvCache::with_storage(cfg(), storage).unwrap();
        let a = c.create_sequence(0, &[1, 2]).unwrap();
        let w = c.storage().token_width();
        c.write_range(a.seq, 0, 0, &vec![1.0; 2 * w], &vec![2.0; 2 * w])
            .unwrap();
        c.mark_computed(a.seq, 2).unwrap();
        let b = c.fork(a.seq).unwrap();
        // Appending to the fork needs a COW copy of the shared partial tail; it fails.
        assert!(matches!(
            c.append_tokens(b, &[3]),
            Err(CacheError::Storage(StorageError::Device(_)))
        ));
        assert!(c.poison().is_some());
        assert!(matches!(c.create_sequence(0, &[9]), Err(CacheError::Poisoned(_))));
        let (mut k, mut v) = (vec![0.0; w], vec![0.0; w]);
        assert!(matches!(
            c.read_into(a.seq, 0, 0, &mut k, &mut v),
            Err(CacheError::Poisoned(_))
        ));
        c.manager().check_invariants().unwrap(); // metadata itself stays consistent
    }

    #[test]
    fn read_range_spans_blocks_and_matches_per_position_reads() {
        let g = ModelGeometry::preset("tiny").unwrap();
        let mut c = PagedKvCache::new_cpu(&g, cfg());
        let prompt: Vec<u32> = (0..11).collect(); // 3 blocks of 4, last one partial
        let a = c.create_sequence(0, &prompt).unwrap();
        let w = c.storage().token_width();
        let k: Vec<f32> = (0..11 * w).map(|i| i as f32).collect();
        let v: Vec<f32> = k.iter().map(|x| -x).collect();
        c.write_range(a.seq, 0, 1, &k, &v).unwrap();
        // Unaligned window crossing two block boundaries.
        let (mut ko, mut vo) = (vec![0.0; 8 * w], vec![0.0; 8 * w]);
        c.read_range_into(a.seq, 2, 1, &mut ko, &mut vo).unwrap();
        assert_eq!(ko, k[2 * w..10 * w]);
        assert_eq!(vo, v[2 * w..10 * w]);
        let (mut k1, mut v1) = (vec![0.0; w], vec![0.0; w]);
        c.read_into(a.seq, 9, 1, &mut k1, &mut v1).unwrap();
        assert_eq!(k1, k[9 * w..10 * w]);
        // Past the end and bad shapes are errors, and do not poison.
        let (mut kb, mut vb) = (vec![0.0; 2 * w], vec![0.0; 2 * w]);
        assert!(c.read_range_into(a.seq, 10, 1, &mut kb, &mut vb).is_err());
        assert!(matches!(
            c.read_range_into(a.seq, 0, 1, &mut [0.0; 3], &mut [0.0; 3]),
            Err(CacheError::Storage(StorageError::ShapeMismatch { .. }))
        ));
        assert!(c.read_range_into(a.seq, 0, 99, &mut k1, &mut v1).is_err());
        assert!(c.poison().is_none());
    }

    #[test]
    fn argument_errors_do_not_poison() {
        let g = ModelGeometry::preset("tiny").unwrap();
        let mut c = PagedKvCache::new_cpu(&g, cfg());
        let a = c.create_sequence(0, &[1, 2]).unwrap();
        assert!(matches!(
            c.write_range(a.seq, 0, 0, &[1.0; 3], &[1.0; 3]),
            Err(CacheError::Storage(StorageError::ShapeMismatch { .. }))
        ));
        let w = c.storage().token_width();
        assert!(matches!(
            c.write(a.seq, 0, 99, &vec![0.0; w], &vec![0.0; w]),
            Err(CacheError::Storage(StorageError::OutOfRange {
                what: "layer",
                ..
            }))
        ));
        assert!(c.poison().is_none());
        c.write(a.seq, 0, 0, &vec![0.0; w], &vec![0.0; w]).unwrap();
    }

    #[test]
    fn write_range_spans_blocks_and_validates_first() {
        let g = ModelGeometry::preset("tiny").unwrap();
        let mut c = PagedKvCache::new_cpu(&g, cfg());
        let a = c.create_sequence(0, &(0..10).collect::<Vec<_>>()).unwrap();
        let w = c.storage().token_width();
        let k: Vec<f32> = (0..10 * w).map(|i| i as f32).collect();
        let v: Vec<f32> = k.iter().map(|x| -x).collect();
        c.write_range(a.seq, 0, 1, &k, &v).unwrap();
        for p in 0..10 {
            let (rk, rv) = c.read(a.seq, p, 1).unwrap();
            assert_eq!(rk, &k[p * w..(p + 1) * w]);
            assert_eq!(rv, &v[p * w..(p + 1) * w]);
        }
        // Range past the end is rejected before anything is written.
        c.mark_computed(a.seq, 0).unwrap();
        assert!(c
            .write_range(a.seq, 8, 0, &vec![7.0; 3 * w], &vec![7.0; 3 * w])
            .is_err());
        assert_eq!(c.read(a.seq, 8, 0).unwrap().0, &vec![0.0; w][..]);
    }

    #[test]
    fn with_storage_rejects_mismatched_shape() {
        let g = ModelGeometry::preset("tiny").unwrap();
        assert!(matches!(
            PagedKvCache::with_storage(cfg(), CpuKvStorage::new(&g, 4, 4)),
            Err(CacheError::StorageMismatch { .. })
        ));
        assert!(matches!(
            PagedKvCache::with_storage(cfg(), CpuKvStorage::new(&g, 8, 8)),
            Err(CacheError::StorageMismatch { .. })
        ));
    }

    #[test]
    fn layout_rejects_bad_arguments() {
        let g = ModelGeometry::preset("tiny").unwrap();
        let l = BlockLayout::new(&g, 4, 3);
        let pool = crate::pool::BlockPool::new(4, 4);
        let id = |i| pool.id_of(i);
        let w = l.width;
        assert!(l.slot_range(id(3), 0, 0, w, w).is_err(), "block out of range");
        assert!(
            l.slot_range(id(0), 3, 0, 2 * w, 2 * w).is_err(),
            "slot range past block end"
        );
        assert!(l.slot_range(id(0), 0, 2, w, w).is_err(), "layer out of range");
        assert!(
            l.slot_range(id(0), 0, 0, w, w + 1).is_err(),
            "k/v length mismatch"
        );
        assert!(l.slot_range(id(0), 0, 0, 0, 0).is_err(), "empty buffers");
        assert!(l.slot_range(id(0), 0, 0, 4 * w, 4 * w).is_ok());
        let big = BlockLayout {
            width: usize::MAX / 2,
            layers: 4,
            block_size: 16,
            num_blocks: 8,
        };
        assert_eq!(big.total_bytes(), Err(StorageError::SizeOverflow));
    }
}

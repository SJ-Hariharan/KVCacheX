//! CUDA device storage (feature `cuda`).
//!
//! All KV values live in **one device allocation** (`CudaSlice<f32>`) with the same block layout
//! as [`super::CpuKvStorage`]. No host mirror exists.
//!
//! | operation | data movement |
//! |---|---|
//! | `write_slots` | host → device (`cuMemcpyHtoDAsync`), one transfer per K/V plane |
//! | `read_slots_into` | device → host (`cuMemcpyDtoHAsync`) + stream synchronize |
//! | `copy_block` (copy-on-write) | **device → device** (`cuMemcpyDtoDAsync`) of the whole block, no host round trip |
//! | `synchronize` | `cuStreamSynchronize`, surfaces deferred errors |
//!
//! Implemented with `cudarc`'s safe driver API, which loads `libcuda` at run time. This crate
//! keeps `unsafe_code = "forbid"`; the `unsafe` FFI lives in `cudarc`. `cudarc` panics if
//! `libcuda` cannot be loaded, and its presence probe is an `unsafe fn`. We therefore call
//! the first driver function under [`std::panic::catch_unwind`] and report
//! [`StorageError::Unavailable`] instead. The library load runs inside a `OnceLock` initializer that
//! panicked, so the lock stays uninitialized and is not poisoned. The default panic hook still
//! prints cudarc's message to stderr.
//!
//! Host buffers passed to writes are ordinary pageable memory. Per the CUDA driver API
//! synchronization rules, a host-to-device copy from pageable memory returns once the source
//! has been staged, so the caller may reuse the buffer immediately.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError};

use super::{BlockLayout, KvStorage, Residency, StorageError};
use crate::geometry::ModelGeometry;
use crate::manager::BlockCopy;
use crate::pool::BlockId;

/// Properties of a CUDA device, as reported by the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub ordinal: usize,
    pub device_count: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_mem_bytes: usize,
    pub free_mem_bytes: usize,
}

fn dev(what: &str, e: DriverError) -> StorageError {
    StorageError::Device(format!("{what}: {e}"))
}

/// Runs the first driver call(s), converting cudarc's "library not found" panic into an error.
fn guarded<T>(f: impl FnOnce() -> Result<T, DriverError>) -> Result<T, StorageError> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(StorageError::Unavailable(format!("CUDA driver error: {e}"))),
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".into());
            Err(StorageError::Unavailable(format!(
                "CUDA driver library could not be loaded: {msg}"
            )))
        }
    }
}

/// Number of CUDA devices visible to the driver. `Err(Unavailable)` if there is no driver.
pub fn device_count() -> Result<usize, StorageError> {
    guarded(CudaContext::device_count).map(|n| n.max(0) as usize)
}

fn open(ordinal: usize) -> Result<(Arc<CudaContext>, usize), StorageError> {
    let count = device_count()?;
    if count == 0 {
        return Err(StorageError::Unavailable(
            "driver loaded but no CUDA device is visible".into(),
        ));
    }
    if ordinal >= count {
        return Err(StorageError::Unavailable(format!(
            "device ordinal {ordinal} requested, {count} visible"
        )));
    }
    let ctx = guarded(|| CudaContext::new(ordinal))?;
    Ok((ctx, count))
}

/// Queries device `ordinal`. Fails with `Unavailable` when no driver or device is present.
pub fn device_info(ordinal: usize) -> Result<CudaDeviceInfo, StorageError> {
    let (ctx, device_count) = open(ordinal)?;
    let name = ctx.name().map_err(|e| dev("cuDeviceGetName", e))?;
    let compute_capability = ctx
        .compute_capability()
        .map_err(|e| dev("compute capability", e))?;
    let (free_mem_bytes, total_mem_bytes) = ctx.mem_get_info().map_err(|e| dev("cuMemGetInfo", e))?;
    Ok(CudaDeviceInfo {
        ordinal,
        device_count,
        name,
        compute_capability,
        total_mem_bytes,
        free_mem_bytes,
    })
}

/// KV storage resident in CUDA device memory.
pub struct CudaKvStorage {
    // Field order matters for drop: free the buffer, then the stream, then the context.
    buf: CudaSlice<f32>,
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    layout: BlockLayout,
    ordinal: usize,
}

impl std::fmt::Debug for CudaKvStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKvStorage")
            .field("ordinal", &self.ordinal)
            .field("layout", &self.layout)
            .finish()
    }
}

impl CudaKvStorage {
    /// Allocates zeroed device storage for `num_blocks` blocks on device `ordinal`.
    ///
    /// Errors: `Unavailable` (no driver/device), `SizeOverflow`, or `Device` (for example,
    /// `CUDA_ERROR_OUT_OF_MEMORY`). No partial allocation is left behind on error.
    pub fn new(
        geom: &ModelGeometry,
        block_size: u32,
        num_blocks: u32,
        ordinal: usize,
    ) -> Result<Self, StorageError> {
        let layout = BlockLayout::new(geom, block_size, num_blocks);
        let total = layout.total_elems()?;
        layout.total_bytes()?;
        if total == 0 {
            return Err(StorageError::OutOfRange {
                what: "storage elements",
                value: 0,
                limit: 0,
            });
        }
        let (ctx, _) = open(ordinal)?;
        let stream = ctx.default_stream();
        let buf = stream
            .alloc_zeros::<f32>(total)
            .map_err(|e| dev(&format!("allocating {} bytes", total * 4), e))?;
        stream
            .synchronize()
            .map_err(|e| dev("synchronize after allocation", e))?;
        Ok(CudaKvStorage {
            buf,
            stream,
            ctx,
            layout,
            ordinal,
        })
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Bytes of device memory held by this storage.
    pub fn device_bytes(&self) -> usize {
        self.buf.num_bytes()
    }

    /// Current `(free, total)` device memory, as reported by the driver.
    pub fn mem_info(&self) -> Result<(usize, usize), StorageError> {
        self.ctx.mem_get_info().map_err(|e| dev("cuMemGetInfo", e))
    }
}

impl KvStorage for CudaKvStorage {
    fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    fn residency(&self) -> Residency {
        Residency::CudaDevice {
            ordinal: self.ordinal,
        }
    }

    fn write_slots(
        &mut self,
        block: BlockId,
        slot: u32,
        layer: u32,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), StorageError> {
        let (ko, vo, n) = self.layout.slot_range(block, slot, layer, k.len(), v.len())?;
        let len = self.buf.len();
        for (off, src, what) in [(ko, k, "H2D copy (K)"), (vo, v, "H2D copy (V)")] {
            let mut dst = self
                .buf
                .try_slice_mut(off..off + n)
                .ok_or(StorageError::OutOfRange {
                    what: "device offset",
                    value: off + n,
                    limit: len,
                })?;
            self.stream.memcpy_htod(src, &mut dst).map_err(|e| dev(what, e))?;
        }
        Ok(())
    }

    fn read_slots_into(
        &self,
        block: BlockId,
        slot: u32,
        layer: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), StorageError> {
        let (ko, vo, n) = self
            .layout
            .slot_range(block, slot, layer, k_out.len(), v_out.len())?;
        for (off, out, what) in [(ko, k_out, "D2H copy (K)"), (vo, v_out, "D2H copy (V)")] {
            let src = self.buf.try_slice(off..off + n).ok_or(StorageError::OutOfRange {
                what: "device offset",
                value: off + n,
                limit: self.buf.len(),
            })?;
            self.stream.memcpy_dtoh(&src, out).map_err(|e| dev(what, e))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| dev("synchronize after D2H", e))
    }

    fn copy_block(&mut self, copy: BlockCopy) -> Result<(), StorageError> {
        let (s, d) = self.layout.check_copy(copy)?;
        let be = self.layout.block_elems();
        // src != dst, so the two blocks lie on opposite sides of the later block's start.
        // Split there to get a shared view of src and an exclusive view of dst.
        let split = s.max(d) * be;
        let len = self.buf.len();
        let (mut lo, mut hi) = self.buf.try_split_at_mut(split).ok_or(StorageError::OutOfRange {
            what: "device split",
            value: split,
            limit: len,
        })?;
        let oob = |v| StorageError::OutOfRange {
            what: "device block view",
            value: v,
            limit: split + be,
        };
        if s < d {
            let src = lo.try_slice(s * be..(s + 1) * be).ok_or(oob(s))?;
            let mut dst = hi.try_slice_mut(0..be).ok_or(oob(d))?;
            self.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| dev("D2D block copy", e))
        } else {
            let src = hi.try_slice(0..be).ok_or(oob(s))?;
            let mut dst = lo.try_slice_mut(d * be..(d + 1) * be).ok_or(oob(d))?;
            self.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| dev("D2D block copy", e))
        }
    }

    fn synchronize(&self) -> Result<(), StorageError> {
        self.stream
            .synchronize()
            .map_err(|e| dev("cuStreamSynchronize", e))
    }
}

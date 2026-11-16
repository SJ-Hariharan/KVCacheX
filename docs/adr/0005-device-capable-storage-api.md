# ADR-0005: Device-capable storage API and an optional CUDA backend

**Status:** accepted. The CUDA backend is validated on a Google Colab Tesla T4 (driver 580.82.07, CUDA 13.0):
9/9 device tests pass, with none skipped, in two recorded runs, the latest on a fresh runtime with an unmodified
notebook (`results/gpu/`).

## Context

The original `KvStorage` trait (ADR-0003) was shaped around host memory:

```rust
fn write(&mut self, block, slot, layer, k: &[f32], v: &[f32]);          // infallible, one slot
fn read(&self, block, slot, layer) -> (&[f32], &[f32]);                 // borrows stored values
fn copy_block(&mut self, copy: BlockCopy);                               // infallible
```

A GPU backend cannot implement this honestly:

1. **Borrowed reads.** `read` returns references into the stored data. Device memory cannot be
   borrowed as a host slice, so implementing it means keeping a host mirror, which is not GPU
   residency, or leaking a download buffer.
2. **Infallibility.** Device allocation and copies fail (out of memory, invalid context, sticky
   errors after a fault). The trait gave no way to report that.
3. **Panicking validation.** Bounds checks were `assert!`s, and cudarc's memcpy functions also
   `assert!` length relations. A bad argument would abort instead of producing an error.
4. **Granularity.** One write per token per layer means one host-to-device transfer each.
5. **Asynchrony.** Device copies are queued on a stream. Nothing in the trait said when data is
   in place or where deferred errors surface.
6. **Atomicity with metadata.** `PagedKvCache::append_tokens` commits the copy-on-write in the
   block table and then copies the data. If the device copy fails, metadata points at a block
   whose contents were never copied.

## Decision

1. The trait becomes fallible and transfer-explicit:
   - `write_slots(block, slot, layer, k, v) -> Result<(), StorageError>` writes `n` consecutive slots of one
     block and layer (contiguous in the layout, so one transfer per K/V plane);
   - `read_slots_into(.., k_out, v_out) -> Result` copies into caller buffers and is complete on return;
   - `copy_block(copy) -> Result` copies storage-side (device-to-device for GPUs);
   - `synchronize() -> Result` waits for queued work and surfaces deferred errors;
   - `layout()` and `residency()` describe the shape and where the bytes live (`Host` / `CudaDevice`).

   Borrowed access survives only as `CpuKvStorage::read` and `PagedKvCache<CpuKvStorage>::read`, where
   the values really are in host memory.
2. All argument validation is shared (`BlockLayout`) and returns `StorageError::{OutOfRange,
   ShapeMismatch, SizeOverflow}` before any transfer. Device failures are `StorageError::Device`;
   a missing driver or device is `StorageError::Unavailable`.
3. **Poisoning.** A fatal (`Device`) error inside `PagedKvCache` poisons the cache. Every later
   data or lifecycle call returns `CacheError::Poisoned`. This is preferred over rolling back
   metadata because CUDA context errors are typically sticky, and a cache that may hold corrupt KV
   must not keep serving it. Argument errors do not poison. Covered by a CPU unit test with an
   injected copy failure.
4. **`CudaKvStorage` (feature `cuda`)**, built on `cudarc` 0.19.10 (maintained, safe driver-API
   wrapper; features `std`, `driver`, `dynamic-loading`, `cuda-12000`):
   - one `CudaSlice<f32>` allocation per cache, same layout as the CPU backend, no host mirror;
   - writes are `cuMemcpyHtoDAsync`; reads are `cuMemcpyDtoHAsync` followed by a stream synchronize;
   - `copy_block` is one `cuMemcpyDtoDAsync` of the whole block. The source and destination views come
     from `try_split_at_mut` at the later block's start, so the borrow checker sees disjoint views;
   - `libcuda` is loaded at run time. Building needs no CUDA toolkit, and CPU-only users never compile
     cudarc. The bindings target the CUDA 12.0 driver API, so any driver reporting CUDA ≥ 12.0 is
     supported. The driver API is backward compatible.
5. `unsafe_code = "forbid"` stays. cudarc panics when `libcuda` is missing, and its presence probe
   is an `unsafe fn`, so the first driver call runs under `std::panic::catch_unwind` and maps the
   panic to `Unavailable`. The panicking initializer leaves cudarc's `OnceLock` uninitialized (not
   poisoned). The default panic hook still prints cudarc's message to stderr; that is documented
   and accepted.

## Consequences

- Breaking change to `KvStorage`, `PagedKvCache::with_storage` (now returns `Result`, checking
  shape) and `PagedKvCache::{write, read_into}`. There are no external users.
- The model-based integrity harness is generic over `KvStorage`. The same randomized operation
  model runs against CPU and CUDA storage and downloads every computed value after every step.
- GPU tests (`tests/gpu.rs`) are compiled only with `--features cuda`. They **fail**, never skip,
  without a usable device, and they print why.
- Not in scope: f16/bf16 device storage, pinned host buffers, multi-stream overlap, attention
  kernels. KVCacheX still runs no model compute, so nothing here measures or claims inference speed.

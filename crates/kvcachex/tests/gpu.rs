//! CUDA device-storage tests. Compiled only with `--features cuda`.
//!
//! These tests **fail** (they never skip) when no usable CUDA device is visible: building with
//! `--features cuda` is the explicit request for GPU validation. Select a device with
//! `KVX_CUDA_DEVICE=<ordinal>` (default 0). Run with `--test-threads=1` so memory accounting
//! tests are not disturbed by concurrent allocations.
#![cfg(feature = "cuda")]

mod common;

use kvcachex::storage::cuda::{device_count, device_info};
use kvcachex::{
    geometry::ModelGeometry, BlockCopy, CacheConfig, CacheManager, CudaKvStorage, KvStorage, PagedKvCache,
    Residency, StorageError,
};
use proptest::prelude::*;

fn ordinal() -> usize {
    std::env::var("KVX_CUDA_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn require_gpu() -> kvcachex::CudaDeviceInfo {
    device_info(ordinal()).unwrap_or_else(|e| {
        panic!(
            "GPU tests were requested (--features cuda) but no usable CUDA device {} is available: {e}. \
             On Colab select Runtime > Change runtime type > GPU.",
            ordinal()
        )
    })
}

fn tiny() -> ModelGeometry {
    ModelGeometry::preset("tiny").unwrap()
}

fn storage(geom: &ModelGeometry, block_size: u32, blocks: u32) -> CudaKvStorage {
    require_gpu();
    CudaKvStorage::new(geom, block_size, blocks, ordinal())
        .unwrap_or_else(|e| panic!("allocating device storage: {e}"))
}

/// Block handles with indices `0..n` for direct storage tests (only the manager mints them).
fn ids(n: u32) -> Vec<kvcachex::BlockId> {
    let mut m = CacheManager::new(CacheConfig {
        block_size: 1,
        num_blocks: n,
        enable_prefix_caching: false,
    });
    let a = m.create_sequence(0, &(0..n).collect::<Vec<_>>()).unwrap();
    let mut v = m.block_table(a.seq).unwrap().to_vec();
    v.sort_by_key(|b| b.index());
    assert!(v.iter().enumerate().all(|(i, b)| b.index() as usize == i));
    v
}

fn pattern(block: u32, layer: u32, slot: usize, kv: u32, i: usize) -> f32 {
    (block * 1_000_000 + layer * 10_000 + kv * 5_000) as f32 + (slot * 64 + i) as f32
}

#[test]
fn device_is_visible_and_reported() {
    let info = require_gpu();
    assert!(info.device_count >= 1);
    assert!(info.total_mem_bytes > 0 && info.free_mem_bytes <= info.total_mem_bytes);
    assert!(!info.name.is_empty());
    eprintln!(
        "CUDA device {}: {} (cc {}.{}), {} MiB total, {} MiB free, {} device(s)",
        info.ordinal,
        info.name,
        info.compute_capability.0,
        info.compute_capability.1,
        info.total_mem_bytes >> 20,
        info.free_mem_bytes >> 20,
        info.device_count
    );
}

#[test]
fn storage_is_device_resident() {
    let g = tiny();
    let s = storage(&g, 4, 8);
    assert_eq!(s.residency(), Residency::CudaDevice { ordinal: ordinal() });
    assert_eq!(s.device_bytes(), s.layout().total_bytes().unwrap());
}

#[test]
fn write_read_integrity_every_slot() {
    let g = tiny();
    let (bs, blocks) = (4u32, 6u32);
    let mut s = storage(&g, bs, blocks);
    let w = s.token_width();
    let b = ids(blocks);
    // Multi-slot writes of distinct values into every block/layer.
    for blk in 0..blocks {
        for l in 0..g.num_layers {
            let k: Vec<f32> = (0..bs as usize * w)
                .map(|i| pattern(blk, l, i / w, 0, i % w))
                .collect();
            let v: Vec<f32> = (0..bs as usize * w)
                .map(|i| pattern(blk, l, i / w, 1, i % w))
                .collect();
            s.write_slots(b[blk as usize], 0, l, &k, &v).unwrap();
        }
    }
    s.synchronize().unwrap();
    // Single-slot reads.
    let (mut k, mut v) = (vec![0.0; w], vec![0.0; w]);
    for blk in 0..blocks {
        for l in 0..g.num_layers {
            for slot in 0..bs {
                s.read_slots_into(b[blk as usize], slot, l, &mut k, &mut v)
                    .unwrap();
                for i in 0..w {
                    assert_eq!(
                        k[i],
                        pattern(blk, l, slot as usize, 0, i),
                        "K blk {blk} l {l} slot {slot}"
                    );
                    assert_eq!(
                        v[i],
                        pattern(blk, l, slot as usize, 1, i),
                        "V blk {blk} l {l} slot {slot}"
                    );
                }
            }
        }
    }
    // Overwrite a middle range and read it back as a multi-slot read.
    let k2 = vec![-1.5; 2 * w];
    let v2 = vec![-2.5; 2 * w];
    s.write_slots(b[3], 1, 1, &k2, &v2).unwrap();
    let (mut k3, mut v3) = (vec![0.0; 4 * w], vec![0.0; 4 * w]);
    s.read_slots_into(b[3], 0, 1, &mut k3, &mut v3).unwrap();
    for slot in 0..4 {
        let expect_k = if (1..3).contains(&slot) {
            -1.5
        } else {
            pattern(3, 1, slot, 0, 0)
        };
        assert_eq!(k3[slot * w], expect_k, "slot {slot}");
    }
}

#[test]
fn copy_block_device_to_device_both_directions() {
    let g = tiny();
    let (bs, blocks) = (4u32, 6u32);
    let mut s = storage(&g, bs, blocks);
    let w = s.token_width();
    let b = ids(blocks);
    for blk in 0..blocks {
        for l in 0..g.num_layers {
            let k: Vec<f32> = (0..bs as usize * w)
                .map(|i| pattern(blk, l, i / w, 0, i % w))
                .collect();
            let v: Vec<f32> = (0..bs as usize * w)
                .map(|i| pattern(blk, l, i / w, 1, i % w))
                .collect();
            s.write_slots(b[blk as usize], 0, l, &k, &v).unwrap();
        }
    }
    // src < dst (adjacent), src > dst (far), and a copy into the last block.
    for &(src, dst) in &[(1u32, 2u32), (5, 0), (0, 5)] {
        s.copy_block(BlockCopy {
            src: b[src as usize],
            dst: b[dst as usize],
            slots: 3,
        })
        .unwrap();
        s.synchronize().unwrap();
        let (mut k, mut v) = (vec![0.0; w], vec![0.0; w]);
        for l in 0..g.num_layers {
            for slot in 0..3u32 {
                s.read_slots_into(b[dst as usize], slot, l, &mut k, &mut v)
                    .unwrap();
                let (mut sk, mut sv) = (vec![0.0; w], vec![0.0; w]);
                s.read_slots_into(b[src as usize], slot, l, &mut sk, &mut sv)
                    .unwrap();
                assert_eq!(k, sk, "copy {src}->{dst} layer {l} slot {slot} K");
                assert_eq!(v, sv, "copy {src}->{dst} layer {l} slot {slot} V");
            }
        }
    }
    // Untouched blocks keep their original contents.
    let (mut k, mut v) = (vec![0.0; w], vec![0.0; w]);
    for blk in [3u32, 4] {
        s.read_slots_into(b[blk as usize], 2, 1, &mut k, &mut v).unwrap();
        assert_eq!(k[0], pattern(blk, 1, 2, 0, 0));
        assert_eq!(v[w - 1], pattern(blk, 1, 2, 1, w - 1));
    }
}

#[test]
fn invalid_arguments_are_errors_not_panics() {
    let g = tiny();
    let mut s = storage(&g, 4, 3);
    let w = s.token_width();
    let b = ids(4); // index 3 is out of range for this 3-block storage
    let one = vec![0.0; w];
    assert!(matches!(
        s.write_slots(b[3], 0, 0, &one, &one),
        Err(StorageError::OutOfRange { what: "block", .. })
    ));
    assert!(matches!(
        s.write_slots(b[0], 0, 9, &one, &one),
        Err(StorageError::OutOfRange { what: "layer", .. })
    ));
    let five = vec![0.0; 5 * w];
    assert!(matches!(
        s.write_slots(b[0], 0, 0, &five, &five),
        Err(StorageError::OutOfRange { .. })
    ));
    assert!(matches!(
        s.write_slots(b[0], 0, 0, &one, &five),
        Err(StorageError::ShapeMismatch { .. })
    ));
    let (mut k, mut v) = (vec![0.0; w + 1], vec![0.0; w + 1]);
    assert!(matches!(
        s.read_slots_into(b[0], 0, 0, &mut k, &mut v),
        Err(StorageError::ShapeMismatch { .. })
    ));
    assert!(s
        .copy_block(BlockCopy {
            src: b[0],
            dst: b[0],
            slots: 1
        })
        .is_err());
    assert!(s
        .copy_block(BlockCopy {
            src: b[0],
            dst: b[3],
            slots: 1
        })
        .is_err());
    s.synchronize().unwrap(); // the context is still healthy
    assert!(matches!(
        CudaKvStorage::new(&g, 4, 3, device_count().unwrap() + 7),
        Err(StorageError::Unavailable(_))
    ));
}

#[test]
fn oversized_allocation_fails_cleanly_and_context_survives() {
    let info = require_gpu();
    let g = ModelGeometry::preset("llama-3-8b").unwrap();
    // f32 block of 16 tokens = 2 * 32 * 16 * 1024 * 4 B = 4 MiB; ask for ~2x device memory.
    let block_bytes = 2 * 32 * 16 * 1024 * 4;
    let blocks = (2 * info.total_mem_bytes / block_bytes) as u32 + 1;
    match CudaKvStorage::new(&g, 16, blocks, ordinal()) {
        Err(StorageError::Device(msg)) => assert!(msg.contains("allocating"), "{msg}"),
        Err(e) => panic!("expected a Device allocation error, got {e}"),
        Ok(_) => panic!("allocation of ~2x device memory unexpectedly succeeded"),
    }
    // A normal allocation still works afterwards.
    let s = storage(&tiny(), 4, 4);
    s.synchronize().unwrap();
}

#[test]
fn device_memory_is_returned_on_drop() {
    require_gpu();
    let g = ModelGeometry::preset("llama-3-8b").unwrap();
    let probe = storage(&tiny(), 1, 1);
    let (free0, _) = probe.mem_info().unwrap();
    const MIB: usize = 1 << 20;
    for _ in 0..5 {
        let s = CudaKvStorage::new(&g, 16, 64, ordinal()).unwrap(); // 64 x 4 MiB = 256 MiB
        let (free1, _) = s.mem_info().unwrap();
        assert!(
            free0.saturating_sub(free1) >= 250 * MIB,
            "allocation not visible in free memory: {free0} -> {free1}"
        );
        drop(s);
        probe.synchronize().unwrap();
    }
    let (free2, _) = probe.mem_info().unwrap();
    assert!(
        free2 + 16 * MIB >= free0,
        "device memory not returned: free {free0} -> {free2} bytes"
    );
}

#[test]
fn paged_cache_cow_on_device() {
    let g = tiny();
    let cfg = CacheConfig {
        block_size: 4,
        num_blocks: 16,
        enable_prefix_caching: true,
    };
    let mut c = PagedKvCache::with_storage(cfg, storage(&g, 4, 16)).unwrap();
    let w = c.storage().token_width();
    let a = c.create_sequence(0, &[1, 2, 3, 4, 5, 6]).unwrap();
    for l in 0..g.num_layers {
        let k: Vec<f32> = (0..6 * w).map(|i| (l * 100) as f32 + i as f32).collect();
        c.write_range(a.seq, 0, l, &k, &k).unwrap();
    }
    c.mark_computed(a.seq, 6).unwrap();
    let b = c.fork(a.seq).unwrap();
    // Appending to the fork must copy the shared partial tail block device-to-device.
    c.append_tokens(b, &[7]).unwrap();
    assert_eq!(c.manager().counters().cow_copies, 1);
    let ta = c.manager().block_table(a.seq).unwrap().to_vec();
    let tb = c.manager().block_table(b).unwrap().to_vec();
    assert_ne!(ta[1], tb[1]);
    for l in 0..g.num_layers {
        c.write(b, 6, l, &vec![-9.0; w], &vec![-9.0; w]).unwrap();
    }
    c.mark_computed(b, 7).unwrap();
    c.synchronize().unwrap();
    let (mut k, mut v) = (vec![0.0; w], vec![0.0; w]);
    for l in 0..g.num_layers {
        for p in 0..6 {
            for seq in [a.seq, b] {
                c.read_into(seq, p, l, &mut k, &mut v).unwrap();
                let base = (l * 100) as f32 + (p * w) as f32;
                assert_eq!(k[0], base, "{seq} pos {p} layer {l}");
                assert_eq!(v[w - 1], base + (w - 1) as f32);
            }
        }
        c.read_into(b, 6, l, &mut k, &mut v).unwrap();
        assert!(k.iter().all(|&x| x == -9.0));
    }
    c.release(a.seq).unwrap();
    c.release(b).unwrap();
    assert!(c.poison().is_none());
}

proptest! {
    // Fewer cases than the CPU suite: every step downloads every computed position from the device.
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    #[test]
    fn random_ops_on_device_match_model(ops in prop::collection::vec(common::op(), 1..60)) {
        common::run(ops, 10, true, |g, cfg| {
            PagedKvCache::with_storage(cfg, storage(g, cfg.block_size, cfg.num_blocks)).unwrap()
        });
    }
}

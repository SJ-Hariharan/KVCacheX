//! `kvx gpu-info` and `kvx gpu-bench` (feature `cuda`).
//!
//! Storage-level microbenchmarks only: the cost of moving KV values in or out of a backend and of
//! copy-on-write block copies. Nothing here runs attention or a model, so the numbers are not
//! inference latency or throughput.

use std::hint::black_box;
use std::time::Instant;

use kvcachex::storage::cuda::{device_info, CudaDeviceInfo};
use kvcachex::{
    BlockCopy, BlockId, CacheConfig, CacheManager, CpuKvStorage, CudaKvStorage, KvStorage, ModelGeometry,
    StorageError,
};

use crate::util::{Args, Json, Summary};

pub fn info_json(i: &CudaDeviceInfo) -> Json {
    Json::obj(vec![
        ("ordinal", Json::Int(i.ordinal as i64)),
        ("device_count", Json::Int(i.device_count as i64)),
        ("name", Json::Str(i.name.clone())),
        (
            "compute_capability",
            Json::Str(format!("{}.{}", i.compute_capability.0, i.compute_capability.1)),
        ),
        ("total_mem_bytes", Json::Int(i.total_mem_bytes as i64)),
        ("free_mem_bytes", Json::Int(i.free_mem_bytes as i64)),
    ])
}

pub fn cmd_info(args: &Args) -> Result<(), String> {
    let ordinal: usize = args.num("device", 0)?;
    let i = device_info(ordinal).map_err(|e| e.to_string())?;
    print!("{}", info_json(&i).render());
    Ok(())
}

struct Op {
    name: &'static str,
    direction: &'static str,
    bytes: usize,
    ns: Summary,
}

fn block_ids(n: u32) -> Vec<BlockId> {
    let mut m = CacheManager::new(CacheConfig {
        block_size: 1,
        num_blocks: n,
        enable_prefix_caching: false,
    });
    let a = m
        .create_sequence(0, &(0..n).collect::<Vec<_>>())
        .expect("fresh manager has room");
    let mut v = m.block_table(a.seq).expect("live").to_vec();
    v.sort_by_key(|b| b.index());
    v
}

/// Times `f` (which must leave the backend synchronized) `reps` times after `warmup` runs.
fn time<F: FnMut(usize) -> Result<(), StorageError>>(
    warmup: usize,
    reps: usize,
    mut f: F,
) -> Result<Summary, String> {
    for i in 0..warmup {
        f(i).map_err(|e| e.to_string())?;
    }
    let mut xs = Vec::with_capacity(reps);
    for i in 0..reps {
        let t = Instant::now();
        f(warmup + i).map_err(|e| e.to_string())?;
        xs.push(t.elapsed().as_nanos() as f64);
    }
    Ok(Summary::of(&mut xs))
}

fn bench_backend<S: KvStorage>(
    s: &mut S,
    ids: &[BlockId],
    warmup: usize,
    reps: usize,
    dev: bool,
) -> Result<Vec<Op>, String> {
    let l = *s.layout();
    let (w, bs, n) = (l.width, l.block_size as u32, ids.len());
    let (h2d, d2h, d2d) = if dev {
        ("H2D", "D2H", "D2D")
    } else {
        ("host memcpy", "host memcpy", "host memcpy")
    };
    let one: Vec<f32> = (0..w).map(|i| i as f32).collect();
    let plane: Vec<f32> = (0..bs as usize * w).map(|i| i as f32).collect();
    let (mut k1, mut v1) = (vec![0.0; w], vec![0.0; w]);
    let (mut kp, mut vp) = (vec![0.0; bs as usize * w], vec![0.0; bs as usize * w]);
    let mut out = Vec::new();
    let mut push = |name, direction, bytes, ns| {
        out.push(Op {
            name,
            direction,
            bytes,
            ns,
        })
    };

    let ns = time(warmup, reps, |i| {
        s.write_slots(ids[i % n], (i as u32) % bs, 0, &one, &one)?;
        s.synchronize()
    })?;
    push("write_1_slot", h2d, 2 * w * 4, ns);
    let ns = time(warmup, reps, |i| {
        s.write_slots(ids[i % n], 0, (i % l.layers) as u32, &plane, &plane)?;
        s.synchronize()
    })?;
    push("write_block_plane", h2d, 2 * plane.len() * 4, ns);
    let ns = time(warmup, reps, |i| {
        s.read_slots_into(ids[i % n], (i as u32) % bs, 0, &mut k1, &mut v1)
    })?;
    black_box(&k1);
    push("read_1_slot", d2h, 2 * w * 4, ns);
    let ns = time(warmup, reps, |i| {
        s.read_slots_into(ids[i % n], 0, (i % l.layers) as u32, &mut kp, &mut vp)
    })?;
    black_box(&kp);
    push("read_block_plane", d2h, 2 * plane.len() * 4, ns);
    let ns = time(warmup, reps, |i| {
        let (src, dst) = (ids[i % n], ids[(i + 1) % n]);
        s.copy_block(BlockCopy { src, dst, slots: bs })?;
        s.synchronize()
    })?;
    push("copy_block_cow", d2d, l.block_elems() * 4, ns);

    // Sanity: the storage still holds what was written (bench must not measure broken ops).
    s.write_slots(ids[0], 0, 0, &plane, &plane)
        .map_err(|e| e.to_string())?;
    s.read_slots_into(ids[0], 0, 0, &mut kp, &mut vp)
        .map_err(|e| e.to_string())?;
    if kp != plane || vp != plane {
        return Err("readback after benchmark did not match written data".into());
    }
    Ok(out)
}

fn ops_json(ops: &[Op]) -> Json {
    Json::Arr(
        ops.iter()
            .map(|o| {
                Json::obj(vec![
                    ("name", Json::Str(o.name.into())),
                    ("direction", Json::Str(o.direction.into())),
                    ("bytes", Json::Int(o.bytes as i64)),
                    ("latency_ns", o.ns.json()),
                    ("gb_per_s_at_p50", Json::Num(o.bytes as f64 / o.ns.p50)),
                ])
            })
            .collect(),
    )
}

pub fn cmd_bench(args: &Args) -> Result<(), String> {
    let model = args.str("model", "llama-3-8b");
    let geom = ModelGeometry::preset(&model).ok_or_else(|| format!("unknown model '{model}'"))?;
    let block_size: u32 = args.num("block-size", 16)?;
    let blocks: u32 = args.num("blocks", 64)?;
    let reps: usize = args.num("reps", 200)?;
    let warmup: usize = args.num("warmup", 20)?;
    let ordinal: usize = args.num("device", 0)?;
    if blocks < 2 {
        return Err("--blocks must be >= 2".into());
    }
    let info = device_info(ordinal).map_err(|e| format!("no usable CUDA device: {e}"))?;
    let ids = block_ids(blocks);

    let mut gpu = CudaKvStorage::new(&geom, block_size, blocks, ordinal).map_err(|e| e.to_string())?;
    let gpu_ops = bench_backend(&mut gpu, &ids, warmup, reps, true)?;
    let gpu_bytes = gpu.device_bytes();
    drop(gpu);
    let mut cpu = CpuKvStorage::try_new(&geom, block_size, blocks).map_err(|e| e.to_string())?;
    let cpu_ops = bench_backend(&mut cpu, &ids, warmup, reps, false)?;

    println!(
        "device {} ({}), cc {}.{}; {model} f32 layout, block_size {block_size}, {blocks} blocks = {} MiB on device",
        info.ordinal,
        info.name,
        info.compute_capability.0,
        info.compute_capability.1,
        gpu_bytes >> 20
    );
    println!(
        "Each timed op includes a stream synchronize (CPU: no-op). {warmup} warm-up + {reps} timed reps.\n"
    );
    println!("| backend | operation | data movement | bytes | p50 µs | p95 µs | p99 µs | GB/s @p50 |");
    println!("|---|---|---|---:|---:|---:|---:|---:|");
    for (backend, ops) in [("cuda", &gpu_ops), ("cpu", &cpu_ops)] {
        for o in ops {
            println!(
                "| {backend} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.2} |",
                o.name,
                o.direction,
                o.bytes,
                o.ns.p50 / 1e3,
                o.ns.p95 / 1e3,
                o.ns.p99 / 1e3,
                o.bytes as f64 / o.ns.p50
            );
        }
    }
    if let Some(path) = args.opt("json") {
        let j = Json::obj(vec![
            (
                "kind",
                Json::Str("storage-level microbenchmark (not inference)".into()),
            ),
            ("device", info_json(&info)),
            ("model_geometry", Json::Str(model.clone())),
            ("element_type", Json::Str("f32".into())),
            ("block_size", Json::Int(block_size as i64)),
            ("blocks", Json::Int(blocks as i64)),
            ("device_bytes", Json::Int(gpu_bytes as i64)),
            ("warmup", Json::Int(warmup as i64)),
            ("reps", Json::Int(reps as i64)),
            (
                "timing",
                Json::Str("wall clock per op, including stream synchronize".into()),
            ),
            ("cuda", ops_json(&gpu_ops)),
            ("cpu", ops_json(&cpu_ops)),
        ]);
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, j.render()).map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}

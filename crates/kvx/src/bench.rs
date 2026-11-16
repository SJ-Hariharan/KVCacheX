//! Wall-clock microbenchmarks of cache-metadata operations.
//!
//! Each operation is timed one at a time with `Instant` (the timer cost is measured and
//! reported, not subtracted). Each scenario runs `warmup` untimed repetitions and then
//! `reps` timed ones; percentiles are over all timed operations.
//! These numbers cover CPU-side bookkeeping only. No KV tensors are moved.

use std::hint::black_box;
use std::time::Instant;

use kvcachex::baseline::ContiguousAllocator;
use kvcachex::{CacheConfig, CacheManager, SharedCacheManager};

use crate::util::{Json, Rng, Summary};

#[derive(Debug, Clone, Copy)]
pub struct BenchConfig {
    pub warmup: usize,
    pub reps: usize,
    pub block_size: u32,
}

#[derive(Debug)]
pub struct Scenario {
    pub name: &'static str,
    pub detail: String,
    pub ns: Summary,
}

fn timed<F: FnMut()>(samples: &mut Vec<f64>, mut f: F) {
    let t = Instant::now();
    f();
    samples.push(t.elapsed().as_nanos() as f64);
}

fn manager(bs: u32, prefix: bool) -> CacheManager {
    CacheManager::new(CacheConfig {
        block_size: bs,
        num_blocks: 1 << 16,
        enable_prefix_caching: prefix,
    })
}

fn unique_prompt(next: &mut u32, n: u32) -> Vec<u32> {
    let p = (*next..*next + n).collect();
    *next += n;
    p
}

/// Runs a scenario: `body(rep, record_or_not, samples)`.
fn scenario<F>(cfg: &BenchConfig, name: &'static str, detail: String, mut body: F) -> Scenario
where
    F: FnMut(&mut Vec<f64>),
{
    let mut scratch = Vec::new();
    for _ in 0..cfg.warmup {
        body(&mut scratch);
        scratch.clear();
    }
    let mut samples = Vec::new();
    for _ in 0..cfg.reps {
        body(&mut samples);
    }
    Scenario {
        name,
        detail,
        ns: Summary::of(&mut samples),
    }
}

pub fn timer_overhead() -> Summary {
    let mut s = Vec::with_capacity(100_000);
    for _ in 0..100_000 {
        timed(&mut s, || {});
    }
    Summary::of(&mut s)
}

pub fn run_ops(cfg: &BenchConfig) -> Vec<Scenario> {
    let bs = cfg.block_size;
    let mut out = Vec::new();
    let mut next_tok = 1u32;

    for prefix in [false, true] {
        let tag = if prefix { "on" } else { "off" };
        out.push(scenario(
            cfg,
            if prefix {
                "create_512_miss+prefix"
            } else {
                "create_512"
            },
            format!("create a 512-token unique prompt (prefix caching {tag}); 256 live seqs per rep"),
            |s| {
                let mut m = manager(bs, prefix);
                let mut ids = Vec::with_capacity(256);
                for _ in 0..256 {
                    let p = unique_prompt(&mut next_tok, 512);
                    timed(s, || ids.push(black_box(m.create_sequence(0, &p).unwrap()).seq));
                    m.mark_computed(*ids.last().unwrap(), 512).unwrap();
                }
            },
        ));
        out.push(scenario(
            cfg,
            if prefix {
                "release_512+prefix"
            } else {
                "release_512"
            },
            format!("release a 512-token sequence, tail-first (prefix caching {tag})"),
            |s| {
                let mut m = manager(bs, prefix);
                let ids: Vec<_> = (0..256)
                    .map(|_| {
                        let p = unique_prompt(&mut next_tok, 512);
                        let a = m.create_sequence(0, &p).unwrap();
                        m.mark_computed(a.seq, 512).unwrap();
                        a.seq
                    })
                    .collect();
                for id in ids {
                    timed(s, || m.release(black_box(id)).unwrap());
                }
            },
        ));
    }

    out.push(scenario(
        cfg,
        "create_1088_hit1024",
        "create 1024 shared + 64 unique tokens; 64 of 68 blocks served from prefix cache".into(),
        |s| {
            let mut m = manager(bs, true);
            let sys: Vec<u32> = (0..1024).map(|i| 0x4000_0000 + i).collect();
            let a = m.create_sequence(0, &sys).unwrap();
            m.mark_computed(a.seq, 1024).unwrap();
            for _ in 0..256 {
                let mut p = sys.clone();
                p.extend(unique_prompt(&mut next_tok, 64));
                timed(s, || {
                    black_box(m.create_sequence(0, &p).unwrap());
                });
            }
        },
    ));

    out.push(scenario(
        cfg,
        "decode_append_1",
        "append 1 token + mark_computed, 256 live seqs x 64 rounds (prefix caching on)".into(),
        |s| {
            let mut m = manager(bs, true);
            let ids: Vec<_> = (0..256)
                .map(|_| {
                    let p = unique_prompt(&mut next_tok, 200);
                    let a = m.create_sequence(0, &p).unwrap();
                    m.mark_computed(a.seq, 200).unwrap();
                    a.seq
                })
                .collect();
            for round in 0..64u32 {
                for &id in &ids {
                    timed(s, || {
                        m.append_tokens(id, &[round]).unwrap();
                        m.mark_computed(id, 201 + round as usize).unwrap();
                    });
                }
            }
        },
    ));

    out.push(scenario(
        cfg,
        "fork_1024",
        "fork a fully computed 1024-token sequence (64 block refs)".into(),
        |s| {
            let mut m = manager(bs, true);
            let p = unique_prompt(&mut next_tok, 1024);
            let a = m.create_sequence(0, &p).unwrap();
            m.mark_computed(a.seq, 1024).unwrap();
            for _ in 0..256 {
                timed(s, || {
                    black_box(m.fork(a.seq).unwrap());
                });
            }
        },
    ));

    out.push(scenario(
        cfg,
        "baseline_reserve",
        "contiguous first-fit reserve, 256 live extents with holes (baseline)".into(),
        |s| {
            let mut rng = Rng::new(9);
            let mut a = ContiguousAllocator::new(1 << 22);
            let mut live = Vec::new();
            for _ in 0..512 {
                live.push(a.reserve(rng.range(256, 4096)).unwrap());
            }
            for k in (0..live.len()).rev().step_by(2) {
                a.release(live.swap_remove(k));
            }
            for _ in 0..256 {
                let len = rng.range(256, 4096);
                timed(s, || {
                    if let Some(e) = black_box(a.reserve(len)) {
                        live.push(e);
                    }
                });
            }
        },
    ));

    out
}

#[derive(Debug)]
pub struct ThreadResult {
    pub threads: usize,
    /// `true`: one lock acquisition per 11-op cycle; `false`: one per operation.
    pub batched: bool,
    pub ops_per_sec: Summary,
}

/// Throughput of [`SharedCacheManager`] under contention. Each thread repeats a cycle of
/// create (128 tokens, 64 distinct prompts per thread so prefix hits occur), mark_computed,
/// 8 single-token appends, and release: 11 lock-protected operations per cycle.
pub fn run_threads(
    cfg: &BenchConfig,
    thread_counts: &[usize],
    cycles_per_thread: usize,
) -> Vec<ThreadResult> {
    let mut out = Vec::new();
    for batched in [false, true] {
        for &t in thread_counts {
            let mut rates = Vec::new();
            for rep in 0..cfg.warmup + cfg.reps {
                let shared = SharedCacheManager::new(CacheConfig {
                    block_size: cfg.block_size,
                    num_blocks: 1 << 16,
                    enable_prefix_caching: true,
                });
                let start = Instant::now();
                std::thread::scope(|s| {
                    for k in 0..t {
                        let shared = shared.clone();
                        s.spawn(move || {
                            let base = (k as u32 + 1) << 24;
                            for c in 0..cycles_per_thread as u32 {
                                let p: Vec<u32> = (0..128).map(|i| base + (c % 64) * 128 + i).collect();
                                if batched {
                                    let mut m = shared.lock();
                                    let a = m.create_sequence(k as u64, &p).unwrap();
                                    m.mark_computed(a.seq, 128).unwrap();
                                    for j in 0..8 {
                                        m.append_tokens(a.seq, &[j]).unwrap();
                                    }
                                    m.release(a.seq).unwrap();
                                } else {
                                    let a = shared.create_sequence(k as u64, &p).unwrap();
                                    shared.mark_computed(a.seq, 128).unwrap();
                                    for j in 0..8 {
                                        shared.append_tokens(a.seq, &[j]).unwrap();
                                    }
                                    shared.release(a.seq).unwrap();
                                }
                            }
                        });
                    }
                });
                let secs = start.elapsed().as_secs_f64();
                if rep >= cfg.warmup {
                    rates.push((t * cycles_per_thread * 11) as f64 / secs);
                }
            }
            out.push(ThreadResult {
                threads: t,
                batched,
                ops_per_sec: Summary::of(&mut rates),
            });
        }
    }
    out
}

pub fn json(timer: &Summary, ops: &[Scenario], threads: &[ThreadResult]) -> Json {
    Json::obj(vec![
        ("timer_overhead_ns", timer.json()),
        (
            "ops_ns",
            Json::Arr(
                ops.iter()
                    .map(|s| {
                        Json::obj(vec![
                            ("name", Json::Str(s.name.into())),
                            ("detail", Json::Str(s.detail.clone())),
                            ("latency_ns", s.ns.json()),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "shared_threads",
            Json::Arr(
                threads
                    .iter()
                    .map(|t| {
                        Json::obj(vec![
                            ("threads", Json::Int(t.threads as i64)),
                            ("batched_lock", Json::Bool(t.batched)),
                            ("ops_per_sec", t.ops_per_sec.json()),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

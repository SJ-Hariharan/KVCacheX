//! `kvx-candle`: run Candle's llama2.c model with its native KV cache or with KVCacheX.
//!
//!   kvx-candle generate --engine kvcachex-prefix --prompt "Once upon a time" --steps 64
//!   kvx-candle bench --workload shared-prefix --concurrency 4 --json out.json
//!
//! Everything runs on the CPU (Candle CPU backend, f32), one sequence per forward pass.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use candle_core::Result;
use candle_nn::VarBuilder;
use candle_transformers::models::llama2_c::Config;
use kvcachex::CacheConfig;
use kvcachex_candle::weights::load_checkpoint;
use kvcachex_candle::{argmax, Engine, KvxEngine, NativeEngine, Tokenizer};
use peak_alloc::PeakAlloc;

#[global_allocator]
static HEAP: PeakAlloc = PeakAlloc;

const ARMS: [&str; 3] = ["candle-native", "kvcachex", "kvcachex-prefix"];

struct Args(Vec<String>);

impl Args {
    fn get(&self, key: &str) -> Option<&str> {
        let flag = format!("--{key}");
        self.0
            .iter()
            .position(|a| *a == flag)
            .and_then(|i| self.0.get(i + 1))
            .map(|s| s.as_str())
    }
    fn str(&self, key: &str, default: &str) -> String {
        self.get(key).unwrap_or(default).to_string()
    }
    fn num(&self, key: &str, default: usize) -> std::result::Result<usize, String> {
        self.get(key).map_or(Ok(default), |v| {
            v.parse().map_err(|_| format!("--{key}: not a number: {v}"))
        })
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_default();
    let args = Args(argv);
    let r = match cmd.as_str() {
        "generate" => generate(&args),
        "bench" => bench(&args),
        _ => Err(USAGE.to_string()),
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

const USAGE: &str = "usage:
  kvx-candle generate [--engine candle-native|kvcachex|kvcachex-prefix] [--prompt TEXT] [--steps N]
  kvx-candle bench [--workload shared-prefix|unique] [--requests N] [--concurrency C] [--gen N]
                   [--warmup N] [--reps N] [--block-size N] [--json PATH]
common: [--model PATH] [--tokenizer PATH]  (defaults: models/stories15M.bin, models/tokenizer.bin)";

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn load(args: &Args) -> std::result::Result<(Config, VarBuilder<'static>, Tokenizer, PathBuf), String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let model = args
        .get("model")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("stories15M.bin"));
    let tok = args
        .get("tokenizer")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("tokenizer.bin"));
    if !model.exists() || !tok.exists() {
        return Err(format!(
            "model files not found ({} / {}); run ./scripts/fetch_tiny_models.sh from the repository root",
            model.display(),
            tok.display()
        ));
    }
    let (cfg, vb) = load_checkpoint(&model).map_err(err)?;
    let tokenizer =
        Tokenizer::from_reader(&mut std::fs::File::open(&tok).map_err(err)?, cfg.vocab_size).map_err(err)?;
    Ok((cfg, vb, tokenizer, model))
}

fn make_engine(
    arm: &str,
    vb: &VarBuilder<'static>,
    cfg: &Config,
    blocks: u32,
    bs: u32,
) -> Result<Box<dyn Engine>> {
    Ok(match arm {
        "candle-native" => Box::new(NativeEngine::new(vb.clone(), cfg.clone())?),
        "kvcachex" | "kvcachex-prefix" => {
            let kv = CacheConfig {
                block_size: bs,
                num_blocks: blocks,
                enable_prefix_caching: arm == "kvcachex-prefix",
            };
            Box::new(KvxEngine::new(vb.clone(), cfg.clone(), kv)?)
        }
        _ => return Err(candle_core::Error::Msg(format!("unknown engine {arm}"))),
    })
}

fn generate(args: &Args) -> std::result::Result<(), String> {
    let (cfg, vb, tok, _) = load(args)?;
    let arm = args.str("engine", "kvcachex-prefix");
    let steps = args.num("steps", 64)?;
    let prompt = tok.encode(&args.str("prompt", "Once upon a time"), true);
    if prompt.len() + steps > cfg.seq_len {
        return Err(format!(
            "prompt + steps must be <= {} (the model's context)",
            cfg.seq_len
        ));
    }
    let blocks = (cfg.seq_len as u32).div_ceil(16) + 1;
    let mut e = make_engine(&arm, &vb, &cfg, blocks, 16).map_err(err)?;
    let t0 = Instant::now();
    let (mut logits, _) = e.prefill(0, &prompt).map_err(err)?;
    let ttft = t0.elapsed();
    let mut out = prompt.clone();
    for _ in 0..steps {
        let t = argmax(&logits);
        out.push(t);
        logits = e.decode(0, t).map_err(err)?;
    }
    let total = t0.elapsed();
    println!("{}", tok.decode(&out));
    eprintln!(
        "[{arm}] prompt {} tokens, {steps} generated; TTFT {:.2} ms, {:.1} tok/s overall",
        prompt.len(),
        ttft.as_secs_f64() * 1e3,
        steps as f64 / total.as_secs_f64()
    );
    Ok(())
}

// ---------------------------------------------------------------- benchmark

const SYSTEM: &str = "Once upon a time, in a small village near a big forest, there lived a kind old \
    woman named Rose. Every morning she baked warm bread and shared it with the children who \
    lived nearby. The children loved her very much, and they often came to her house to hear \
    her stories. One sunny day, a little child knocked on her door and asked her a question.";
const NAMES: [&str; 8] = ["Tim", "Lily", "Ben", "Mia", "Sam", "Anna", "Max", "Zoe"];
const THINGS: [&str; 8] = [
    "a red ball",
    "a blue bird",
    "a shiny key",
    "a small boat",
    "a big tree",
    "a lost dog",
    "a funny hat",
    "a yellow kite",
];

/// Prompt trace. `shared-prefix`: one long shared opening plus a short request-specific line
/// (like a shared system prompt). `unique`: every prompt starts differently, so a prefix cache
/// can only match by accident.
fn trace(workload: &str, n: usize, tok: &Tokenizer) -> std::result::Result<Vec<Vec<u32>>, String> {
    (0..n)
        .map(|i| {
            let (name, thing) = (NAMES[i % 8], THINGS[(i / 8 + i) % 8]);
            let text = match workload {
                "shared-prefix" => format!("{SYSTEM} {name} asked about {thing}."),
                "unique" => format!(
                    "{name} had {thing} and went outside to play. {SYSTEM} The question was about {thing}."
                ),
                _ => return Err(format!("unknown workload {workload}")),
            };
            Ok(tok.encode(&text, true))
        })
        .collect()
}

#[derive(Debug, Default)]
struct RunResult {
    wall: Duration,
    ttft: Vec<f64>,
    prefill: Vec<f64>,
    itl: Vec<f64>,
    outputs: Vec<Vec<u32>>,
    computed_prompt_tokens: usize,
    cached_prompt_tokens: usize,
    peak_kv_in_use: usize,
    kv_allocated: usize,
    peak_heap_growth: usize,
}

/// Closed-loop run: all requests arrive at t=0; up to `conc` are in flight; each step decodes
/// one token for every in-flight request, round-robin, one sequence per forward pass.
fn run(e: &mut dyn Engine, prompts: &[Vec<u32>], conc: usize, gen: usize) -> Result<RunResult> {
    struct Live {
        id: usize,
        logits: Vec<f32>,
        out: Vec<u32>,
        last: Instant,
    }
    let mut r = RunResult {
        outputs: vec![Vec::new(); prompts.len()],
        ..RunResult::default()
    };
    let base = HEAP.current_usage();
    HEAP.reset_peak_usage();
    let mut pending: VecDeque<usize> = (0..prompts.len()).collect();
    let mut live: Vec<Live> = Vec::new();
    let t0 = Instant::now();
    while !pending.is_empty() || !live.is_empty() {
        while live.len() < conc {
            let Some(id) = pending.pop_front() else { break };
            let ts = Instant::now();
            let (logits, pre) = e.prefill(id, &prompts[id])?;
            let now = Instant::now();
            r.prefill.push((now - ts).as_secs_f64() * 1e3);
            r.ttft.push((now - t0).as_secs_f64() * 1e3);
            r.computed_prompt_tokens += pre.computed_tokens;
            r.cached_prompt_tokens += pre.cached_tokens;
            let first = argmax(&logits);
            live.push(Live {
                id,
                logits,
                out: vec![first],
                last: now,
            });
        }
        r.peak_kv_in_use = r.peak_kv_in_use.max(e.kv_bytes_in_use());
        let mut i = 0;
        while i < live.len() {
            if live[i].out.len() >= gen {
                let l = live.swap_remove(i);
                e.finish(l.id)?;
                r.outputs[l.id] = l.out;
                continue;
            }
            let l = &mut live[i];
            l.logits = e.decode(l.id, *l.out.last().unwrap())?;
            l.out.push(argmax(&l.logits));
            let now = Instant::now();
            r.itl.push((now - l.last).as_secs_f64() * 1e3);
            l.last = now;
            i += 1;
        }
        r.peak_kv_in_use = r.peak_kv_in_use.max(e.kv_bytes_in_use());
    }
    r.wall = t0.elapsed();
    // Candle's native cache allocates on demand (its peak allocation is its peak use);
    // KVCacheX preallocates its whole block pool.
    r.kv_allocated = e.kv_bytes_allocated().max(r.peak_kv_in_use);
    r.peak_heap_growth = HEAP.peak_usage().saturating_sub(base);
    Ok(r)
}

fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        return f64::NAN;
    }
    s[((p / 100.0) * (s.len() - 1) as f64).round() as usize]
}

fn median(v: &[f64]) -> f64 {
    pct(v, 50.0)
}

// Minimal JSON writer (no serde dependency).
fn js(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
fn jf(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.4}")
    } else {
        "null".into()
    }
}
fn jarr(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| jf(*x)).collect::<Vec<_>>().join(","))
}

fn bench(args: &Args) -> std::result::Result<(), String> {
    let (cfg, vb, tok, model_path) = load(args)?;
    let workload = args.str("workload", "shared-prefix");
    let n = args.num("requests", 16)?;
    let conc = args.num("concurrency", 1)?.max(1);
    let gen = args.num("gen", 48)?.max(1);
    let warmup = args.num("warmup", 1)?;
    let reps = args.num("reps", 5)?.max(1);
    let bs = args.num("block-size", 16)? as u32;
    let prompts = trace(&workload, n, &tok)?;
    let max_prompt = prompts.iter().map(Vec::len).max().unwrap_or(0);
    if max_prompt + gen > cfg.seq_len {
        return Err(format!(
            "prompt ({max_prompt}) + gen ({gen}) exceeds the model context {}",
            cfg.seq_len
        ));
    }
    // Pool: room for `conc` sequences at full length, plus one sequence's worth of cached blocks.
    let per_seq = ((max_prompt + gen) as u32).div_ceil(bs);
    let blocks = per_seq * (conc as u32 + 1);

    eprintln!(
        "bench: {workload}, {n} requests (prompt {}..{} tokens), gen {gen}, concurrency {conc}, \
         {warmup} warmup + {reps} reps per arm, interleaved; KVCacheX {blocks} blocks x {bs} tokens",
        prompts.iter().map(Vec::len).min().unwrap_or(0),
        max_prompt
    );
    let mut results: Vec<(String, usize, RunResult)> = Vec::new();
    for rep in 0..warmup + reps {
        // Rotate the arm order every repetition so no arm always runs first.
        for k in 0..ARMS.len() {
            let arm = ARMS[(rep + k) % ARMS.len()];
            let mut e = make_engine(arm, &vb, &cfg, blocks, bs).map_err(err)?;
            let r = run(e.as_mut(), &prompts, conc, gen).map_err(err)?;
            drop(e);
            if rep >= warmup {
                eprintln!(
                    "  rep {} {arm:16} wall {:7.1} ms  TTFT p50 {:6.2} ms  ITL p50 {:5.3} ms",
                    rep - warmup + 1,
                    r.wall.as_secs_f64() * 1e3,
                    median(&r.ttft),
                    median(&r.itl)
                );
                results.push((arm.to_string(), rep - warmup, r));
            }
        }
    }
    // Cache counters: one extra, untimed run per KVCacheX arm (counters are deterministic).
    let mut counters = Vec::new();
    for arm in ["kvcachex", "kvcachex-prefix"] {
        let kv = CacheConfig {
            block_size: bs,
            num_blocks: blocks,
            enable_prefix_caching: arm == "kvcachex-prefix",
        };
        let mut e = KvxEngine::new(vb.clone(), cfg.clone(), kv).map_err(err)?;
        run(&mut e, &prompts, conc, gen).map_err(err)?;
        e.cache().manager().check_invariants()?;
        counters.push((arm, e.stats()));
    }

    // Correctness: every arm's greedy outputs vs the native arm's, per repetition.
    let native_out: Vec<&Vec<Vec<u32>>> = results
        .iter()
        .filter(|(a, _, _)| a == "candle-native")
        .map(|(_, _, r)| &r.outputs)
        .collect();
    let mut identical = Vec::new();
    for arm in ARMS {
        let same = results
            .iter()
            .filter(|(a, _, _)| a == arm)
            .all(|(_, rep, r)| r.outputs == *native_out[*rep]);
        identical.push((arm, same));
    }

    // Summary.
    let summary = |arm: &str| {
        let rs: Vec<&RunResult> = results
            .iter()
            .filter(|(a, _, _)| a == arm)
            .map(|(_, _, r)| r)
            .collect();
        let m = |f: &dyn Fn(&RunResult) -> f64| median(&rs.iter().map(|r| f(r)).collect::<Vec<_>>());
        let gen_tokens = (n * gen) as f64;
        vec![
            ("wall_ms", m(&|r| r.wall.as_secs_f64() * 1e3)),
            ("throughput_tok_s", m(&|r| gen_tokens / r.wall.as_secs_f64())),
            ("ttft_queue_incl_p50_ms", m(&|r| pct(&r.ttft, 50.0))),
            ("ttft_queue_incl_p95_ms", m(&|r| pct(&r.ttft, 95.0))),
            ("ttft_queue_incl_p99_ms", m(&|r| pct(&r.ttft, 99.0))),
            ("prefill_latency_p50_ms", m(&|r| pct(&r.prefill, 50.0))),
            ("prefill_latency_p95_ms", m(&|r| pct(&r.prefill, 95.0))),
            ("itl_p50_ms", m(&|r| pct(&r.itl, 50.0))),
            ("itl_p95_ms", m(&|r| pct(&r.itl, 95.0))),
            ("itl_p99_ms", m(&|r| pct(&r.itl, 99.0))),
            ("computed_prompt_tokens", m(&|r| r.computed_prompt_tokens as f64)),
            ("cached_prompt_tokens", m(&|r| r.cached_prompt_tokens as f64)),
            ("peak_kv_in_use_bytes", m(&|r| r.peak_kv_in_use as f64)),
            ("kv_allocated_bytes", m(&|r| r.kv_allocated as f64)),
            ("peak_rust_heap_growth_bytes", m(&|r| r.peak_heap_growth as f64)),
        ]
    };
    eprintln!("\nmedian over {reps} reps:");
    for arm in ARMS {
        let s = summary(arm);
        let g = |k: &str| s.iter().find(|(n, _)| *n == k).unwrap().1;
        eprintln!(
            "  {arm:16} {:7.1} tok/s  TTFT(queue-incl) p50 {:6.2} p95 {:6.2} ms  prefill p50 {:6.2} ms  \
             ITL p50 {:5.3} p99 {:5.3} ms  prompt computed/cached {:.0}/{:.0}  KV peak {:.2} MiB \
             (alloc {:.2})  Rust heap peak +{:.1} MiB  same-as-native {}",
            g("throughput_tok_s"),
            g("ttft_queue_incl_p50_ms"),
            g("ttft_queue_incl_p95_ms"),
            g("prefill_latency_p50_ms"),
            g("itl_p50_ms"),
            g("itl_p99_ms"),
            g("computed_prompt_tokens"),
            g("cached_prompt_tokens"),
            g("peak_kv_in_use_bytes") / 1048576.0,
            g("kv_allocated_bytes") / 1048576.0,
            g("peak_rust_heap_growth_bytes") / 1048576.0,
            identical.iter().find(|(a, _)| *a == arm).unwrap().1
        );
    }

    if let Some(path) = args.get("json") {
        let mut o = String::from("{\n");
        o += "  \"schema\": \"kvx-candle-bench/1\",\n";
        o += "  \"method\": \"CPU reference integration. Closed-loop interleaved workload in one process: all requests arrive at t=0, up to `concurrency` are in flight, and each step runs one model forward per in-flight request (batch size 1, round-robin). Not batched execution and not an inference server. throughput_tok_s = generated tokens / wall time of this workload. ttft_queue_incl = time from t=0 to a request's first token, including waiting behind earlier prompts; prefill_latency = duration of that request's own prefill forward. peak_rust_heap_growth = peak Rust heap bytes above the pre-run level (global allocator), not process RSS. KV bytes: native = sum of Candle cache tensors; KVCacheX = whole blocks in use (16-token granularity) and its preallocated pool.\",\n";
        o += &format!(
            "  \"environment\": {{\"os\": {}, \"arch\": {}, \"available_parallelism\": {}, \"hardware\": {}, \"rustc\": {}, \"rayon_num_threads\": {}}},\n",
            js(std::env::consts::OS),
            js(std::env::consts::ARCH),
            std::thread::available_parallelism().map_or(0, |n| n.get()),
            std::env::var("KVX_HW").map_or("null".into(), |s| js(&s)),
            std::env::var("KVX_RUSTC").map_or("null".into(), |s| js(&s)),
            std::env::var("RAYON_NUM_THREADS").map_or("null".into(), |s| js(&s)),
        );
        o += &format!(
            "  \"model\": {{\"file\": {}, \"dim\": {}, \"n_layers\": {}, \"n_heads\": {}, \"n_kv_heads\": {}, \"vocab\": {}, \"context\": {}, \"dtype\": \"f32\", \"device\": \"cpu\", \"runtime\": \"candle 0.11.1\"}},\n",
            js(&model_path.file_name().unwrap_or_default().to_string_lossy()),
            cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len
        );
        o += &format!(
            "  \"config\": {{\"workload\": {}, \"requests\": {n}, \"concurrency\": {conc}, \"gen_tokens\": {gen}, \"warmup\": {warmup}, \"reps\": {reps}, \"block_size\": {bs}, \"kv_blocks\": {blocks}, \"prompt_tokens\": [{}], \"decoding\": \"greedy\", \"batch_per_forward\": 1}},\n",
            js(&workload),
            prompts.iter().map(|p| p.len().to_string()).collect::<Vec<_>>().join(",")
        );
        o += "  \"summary_median\": {\n";
        let arms: Vec<String> = ARMS
            .iter()
            .map(|arm| {
                let s = summary(arm);
                let same = identical.iter().find(|(a, _)| a == arm).unwrap().1;
                format!(
                    "    {}: {{{}, \"outputs_identical_to_native\": {same}}}",
                    js(arm),
                    s.iter()
                        .map(|(k, v)| format!("{}: {}", js(k), jf(*v)))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect();
        o += &arms.join(",\n");
        o += "\n  },\n  \"kvcachex_counters\": {\n";
        let cs: Vec<String> = counters
            .iter()
            .map(|(arm, s)| {
                let c = &s.counters;
                format!(
                    "    {}: {{\"prefix_query_blocks\": {}, \"prefix_hit_blocks\": {}, \"prefix_hit_tokens\": {}, \"prefix_revived_blocks\": {}, \"evictions\": {}, \"cow_copies\": {}, \"blocks_allocated\": {}, \"peak_used_blocks\": {}, \"total_blocks\": {}}}",
                    js(arm), c.prefix_query_blocks, c.prefix_hit_blocks, c.prefix_hit_tokens, c.prefix_revived_blocks,
                    c.evictions, c.cow_copies, c.blocks_allocated, s.peak_used_blocks, s.total_blocks
                )
            })
            .collect();
        o += &cs.join(",\n");
        o += "\n  },\n  \"runs\": [\n";
        let runs: Vec<String> = results
            .iter()
            .map(|(arm, rep, r)| {
                format!(
                    "    {{\"arm\": {}, \"rep\": {rep}, \"wall_ms\": {}, \"ttft_queue_incl_ms\": {}, \"prefill_latency_ms\": {}, \"itl_ms\": {}, \"computed_prompt_tokens\": {}, \"cached_prompt_tokens\": {}, \"peak_kv_in_use_bytes\": {}, \"kv_allocated_bytes\": {}, \"peak_rust_heap_growth_bytes\": {}}}",
                    js(arm),
                    jf(r.wall.as_secs_f64() * 1e3),
                    jarr(&r.ttft),
                    jarr(&r.prefill),
                    jarr(&r.itl),
                    r.computed_prompt_tokens,
                    r.cached_prompt_tokens,
                    r.peak_kv_in_use,
                    r.kv_allocated,
                    r.peak_heap_growth
                )
            })
            .collect();
        o += &runs.join(",\n");
        o += "\n  ],\n";
        let native = &native_out[0];
        o += &format!(
            "  \"outputs_native_rep0\": [{}]\n}}\n",
            native
                .iter()
                .map(|t| format!(
                    "[{}]",
                    t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
                ))
                .collect::<Vec<_>>()
                .join(",")
        );
        std::fs::write(path, o).map_err(|e| format!("{path}: {e}"))?;
        eprintln!("wrote {path}");
    }
    if identical.iter().any(|(_, same)| !same) {
        return Err("an arm's outputs differ from the native cache's outputs".into());
    }
    Ok(())
}

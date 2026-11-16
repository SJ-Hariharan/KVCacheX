//! `kvx`: KVCacheX command-line tool.

mod bench;
#[cfg(feature = "cuda")]
mod gpu;
mod sim;
mod util;
mod workload;

use std::fmt::Write as _;
use std::path::Path;

use kvcachex::geometry::{estimate, ModelGeometry};
use kvcachex::{CacheConfig, DType};

use sim::{Backend, ContiguousBackend, PagedBackend, SchedConfig, SimResult};
use util::{Args, Json};
use workload::{Kind, WorkloadSpec};

const USAGE: &str = "\
kvx: KVCacheX tools

USAGE:
  kvx estimate  [--model llama-3-8b] [--dtype f16] [--block-size 16] [--seq-len 4096] [--batch 32]
                [--reserve-len <tokens>] [--budget-gib <GiB>]
  kvx simulate  [--workload short|long|mixed|shared-prefix|parallel] [--requests 400]
                [--rate 2.0] [--seed 1] [--model llama-3-8b] [--kv-gib 2] [--block-size 16]
                [--max-batch-tokens 4096] [--max-seqs 256] [--check-every 0] [--json out.json]
  kvx bench     [--reps 20] [--warmup 3] [--block-size 16] [--json out.json]
  kvx report    [--out results] [--requests 400] [--seeds 3]   (all workloads + bench → REPORT.md)
  kvx models
  kvx gpu-info  [--device 0]                                   (build with --features cuda)
  kvx gpu-bench [--model llama-3-8b] [--block-size 16] [--blocks 64] [--reps 200] [--warmup 20]
                [--device 0] [--json out.json]                 (storage transfers only; not inference)

Simulation time is in scheduler steps; no model runs (see docs/benchmarking.md).
";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first() else {
        print!("{USAGE}");
        return;
    };
    let args = match Args::parse(&argv[1..]) {
        Ok(a) => a,
        Err(e) => fail(&e),
    };
    let r = match cmd.as_str() {
        "estimate" => cmd_estimate(&args),
        "simulate" => cmd_simulate(&args),
        "bench" => cmd_bench(&args),
        "report" => cmd_report(&args),
        #[cfg(feature = "cuda")]
        "gpu-info" => gpu::cmd_info(&args),
        #[cfg(feature = "cuda")]
        "gpu-bench" => gpu::cmd_bench(&args),
        #[cfg(not(feature = "cuda"))]
        "gpu-info" | "gpu-bench" => Err(format!("`{cmd}` needs a build with `--features cuda`")),
        "models" => {
            for p in ModelGeometry::PRESETS {
                let g = ModelGeometry::preset(p).expect("preset");
                println!(
                    "{p:<12} layers={:<3} q_heads={:<3} kv_heads={:<3} head_dim={:<4} {:?}  {} KiB/token (f16)",
                    g.num_layers,
                    g.num_query_heads,
                    g.num_kv_heads,
                    g.head_dim,
                    g.attention_kind(),
                    g.bytes_per_token() / 1024
                );
            }
            Ok(())
        }
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    };
    if let Err(e) = r {
        fail(&e);
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2)
}

fn geometry(args: &Args) -> Result<ModelGeometry, String> {
    let name = args.str("model", "llama-3-8b");
    let mut g =
        ModelGeometry::preset(&name).ok_or_else(|| format!("unknown model '{name}' (see `kvx models`)"))?;
    if let Some(d) = args.opt("dtype") {
        g.dtype = DType::parse(d).ok_or_else(|| format!("unknown dtype '{d}'"))?;
    }
    g.validate()?;
    Ok(g)
}

const GIB: f64 = (1u64 << 30) as f64;

fn cmd_estimate(args: &Args) -> Result<(), String> {
    let g = geometry(args)?;
    let bs: u32 = args.num("block-size", 16)?;
    let len: u64 = args.num("seq-len", 4096)?;
    let batch: usize = args.num("batch", 32)?;
    let reserve: u64 = args.num("reserve-len", len)?;
    let lens = vec![len; batch];
    let e = estimate(&g, bs, &lens, reserve);
    println!(
        "model            {} ({:?}, {})",
        g.name,
        g.attention_kind(),
        g.dtype
    );
    println!(
        "bytes/token      {} ({:.1} KiB)",
        g.bytes_per_token(),
        g.bytes_per_token() as f64 / 1024.0
    );
    println!("bytes/block      {} (block_size {bs})", g.bytes_per_block(bs));
    println!(
        "workload         {batch} x {len} tokens = {} tokens",
        e.total_tokens
    );
    println!("exact KV         {:.3} GiB", e.exact_bytes as f64 / GIB);
    println!(
        "paged KV         {:.3} GiB in {} blocks (slack {:.2}%)",
        e.paged_bytes as f64 / GIB,
        e.blocks,
        100.0 * e.paged_internal_fragmentation()
    );
    println!(
        "reserved ({reserve:>5})  {:.3} GiB (waste {:.2}%)",
        e.reserved_bytes as f64 / GIB,
        100.0 * e.reservation_waste()
    );
    println!(
        "block metadata   {:.2} MiB (approx.)",
        e.metadata_bytes as f64 / (1u64 << 20) as f64
    );
    if let Some(b) = args.opt("budget-gib") {
        let gib: f64 = b.parse().map_err(|_| "--budget-gib: not a number")?;
        let blocks = g.blocks_for_budget((gib * GIB) as u64, bs);
        println!(
            "budget {gib} GiB   {blocks} blocks = {} tokens = {} sequences of {len}",
            blocks * bs as u64,
            blocks / len.div_ceil(bs as u64).max(1)
        );
    }
    Ok(())
}

struct SimSetup {
    spec: WorkloadSpec,
    num_blocks: u32,
    block_size: u32,
    sched: SchedConfig,
    model: ModelGeometry,
    kv_gib: f64,
}

fn sim_setup(args: &Args) -> Result<SimSetup, String> {
    let model = geometry(args)?;
    let kind_s = args.str("workload", "mixed");
    let kind = Kind::parse(&kind_s).ok_or_else(|| format!("unknown workload '{kind_s}'"))?;
    let block_size: u32 = args.num("block-size", 16)?;
    let kv_gib: f64 = args.num("kv-gib", 2.0)?;
    let num_blocks = model.blocks_for_budget((kv_gib * GIB) as u64, block_size) as u32;
    if num_blocks == 0 {
        return Err("KV budget too small for one block".into());
    }
    Ok(SimSetup {
        spec: WorkloadSpec {
            kind,
            requests: args.num("requests", 400)?,
            arrival_rate: args.num("rate", 2.0)?,
            seed: args.num("seed", 1)?,
        },
        num_blocks,
        block_size,
        sched: SchedConfig {
            max_batch_tokens: args.num("max-batch-tokens", 4096)?,
            max_seqs: args.num("max-seqs", 256)?,
            check_every: args.num("check-every", 0)?,
            ..SchedConfig::default()
        },
        model,
        kv_gib,
    })
}

fn run_all_backends(s: &SimSetup) -> Result<Vec<SimResult>, String> {
    let reqs = workload::generate(&s.spec);
    let cfg = |prefix| CacheConfig {
        block_size: s.block_size,
        num_blocks: s.num_blocks,
        enable_prefix_caching: prefix,
    };
    let mut backends: Vec<Box<dyn Backend>> = vec![
        Box::new(ContiguousBackend::new(s.num_blocks as u64 * s.block_size as u64)),
        Box::new(PagedBackend::new(cfg(false))),
        Box::new(PagedBackend::new(cfg(true))),
    ];
    let results: Vec<SimResult> = backends
        .iter_mut()
        .map(|b| sim::run(&reqs, b.as_mut(), s.sched))
        .collect();
    if let Some(r) = results.iter().find(|r| r.truncated) {
        return Err(format!(
            "{}: hit max_steps ({}) with requests unfinished; results would be partial",
            r.backend, s.sched.max_steps
        ));
    }
    Ok(results)
}

fn sim_table(results: &[SimResult]) -> String {
    let mut t = String::new();
    t.push_str("| backend | steps | gen tok/step | mean batch | peak batch | prefill computed | prefix-cached | preemptions | mem held | mem waste | TTFT p50/p99 (steps) | E2E p50/p99 (steps) |\n");
    t.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in results {
        let _ = writeln!(
            t,
            "| {} | {} | {:.1} | {:.1} | {} | {} | {} | {} | {:.1}% | {:.1}% | {:.0} / {:.0} | {:.0} / {:.0} |",
            r.backend,
            r.steps,
            r.tokens_per_step(),
            r.mean_running,
            r.peak_running,
            r.prefill_computed_tokens,
            r.prefix_cached_tokens,
            r.preemptions,
            100.0 * r.mean_held_frac,
            100.0 * r.mean_waste_frac,
            r.ttft_steps.p50,
            r.ttft_steps.p99,
            r.e2e_steps.p50,
            r.e2e_steps.p99
        );
    }
    t
}

fn setup_json(s: &SimSetup) -> Json {
    Json::obj(vec![
        ("workload", Json::Str(s.spec.kind.name().into())),
        ("workload_detail", Json::Str(s.spec.kind.describe().into())),
        ("requests", Json::Int(s.spec.requests as i64)),
        ("arrival_rate_per_step", Json::Num(s.spec.arrival_rate)),
        ("seed", Json::Int(s.spec.seed as i64)),
        ("model", Json::Str(s.model.name.clone())),
        ("kv_budget_gib", Json::Num(s.kv_gib)),
        ("block_size", Json::Int(s.block_size as i64)),
        ("num_blocks", Json::Int(s.num_blocks as i64)),
        ("max_batch_tokens", Json::Int(s.sched.max_batch_tokens as i64)),
        ("max_seqs", Json::Int(s.sched.max_seqs as i64)),
    ])
}

fn cmd_simulate(args: &Args) -> Result<(), String> {
    let s = sim_setup(args)?;
    println!(
        "workload {} ({}), {} requests, rate {}/step, seed {}; {} {:.2} GiB KV = {} blocks x {} tokens\n",
        s.spec.kind.name(),
        s.spec.kind.describe(),
        s.spec.requests,
        s.spec.arrival_rate,
        s.spec.seed,
        s.model.name,
        s.kv_gib,
        s.num_blocks,
        s.block_size
    );
    let results = run_all_backends(&s)?;
    print!("{}", sim_table(&results));
    if let Some(path) = args.opt("json") {
        let j = Json::obj(vec![
            ("setup", setup_json(&s)),
            (
                "results",
                Json::Arr(results.iter().map(SimResult::json).collect()),
            ),
        ]);
        std::fs::write(path, j.render()).map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}

fn bench_config(args: &Args) -> Result<bench::BenchConfig, String> {
    Ok(bench::BenchConfig {
        warmup: args.num("warmup", 3)?,
        reps: args.num("reps", 20)?,
        block_size: args.num("block-size", 16)?,
    })
}

fn bench_tables(timer: &util::Summary, ops: &[bench::Scenario], threads: &[bench::ThreadResult]) -> String {
    let mut t = String::new();
    let _ = writeln!(
        t,
        "Timer overhead (empty closure): p50 {:.0} ns, p99 {:.0} ns. Not subtracted.\n",
        timer.p50, timer.p99
    );
    t.push_str("| operation | n | mean ns | p50 ns | p95 ns | p99 ns | max ns | description |\n|---|---:|---:|---:|---:|---:|---:|---|\n");
    for s in ops {
        let _ = writeln!(
            t,
            "| {} | {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {} |",
            s.name, s.ns.n, s.ns.mean, s.ns.p50, s.ns.p95, s.ns.p99, s.ns.max, s.detail
        );
    }
    t.push_str("\nSharedCacheManager throughput (11 ops per cycle; *per-op* locks each op, *batched* locks once per cycle):\n\n| threads | locking | ops/s mean | stdev | min | max |\n|---:|---|---:|---:|---:|---:|\n");
    for r in threads {
        let _ = writeln!(
            t,
            "| {} | {} | {:.0} | {:.0} | {:.0} | {:.0} |",
            r.threads,
            if r.batched { "batched" } else { "per-op" },
            r.ops_per_sec.mean,
            r.ops_per_sec.stdev,
            r.ops_per_sec.min,
            r.ops_per_sec.max
        );
    }
    t
}

fn run_bench(cfg: &bench::BenchConfig) -> (util::Summary, Vec<bench::Scenario>, Vec<bench::ThreadResult>) {
    let timer = bench::timer_overhead();
    let ops = bench::run_ops(cfg);
    let threads = bench::run_threads(cfg, &[1, 2, 4, 8], 2000);
    (timer, ops, threads)
}

fn cmd_bench(args: &Args) -> Result<(), String> {
    let cfg = bench_config(args)?;
    if cfg!(debug_assertions) {
        eprintln!("warning: debug build; use `cargo run --release` for meaningful numbers");
    }
    let (timer, ops, threads) = run_bench(&cfg);
    print!("{}", bench_tables(&timer, &ops, &threads));
    if let Some(path) = args.opt("json") {
        std::fs::write(path, bench::json(&timer, &ops, &threads).render())
            .map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}

fn env_json() -> Json {
    Json::obj(vec![
        ("os", Json::Str(std::env::consts::OS.into())),
        ("arch", Json::Str(std::env::consts::ARCH.into())),
        (
            "available_parallelism",
            Json::Int(std::thread::available_parallelism().map_or(0, |n| n.get()) as i64),
        ),
        ("debug_build", Json::Bool(cfg!(debug_assertions))),
        ("hardware", std::env::var("KVX_HW").map_or(Json::Null, Json::Str)),
        ("rustc", std::env::var("KVX_RUSTC").map_or(Json::Null, Json::Str)),
    ])
}

/// Runs every workload over several seeds plus the microbenchmarks, and writes JSON + REPORT.md.
fn cmd_report(args: &Args) -> Result<(), String> {
    let out = args.str("out", "results");
    let raw = Path::new(&out).join("raw");
    std::fs::create_dir_all(&raw).map_err(|e| format!("{}: {e}", raw.display()))?;
    let seeds: u64 = args.num("seeds", 3)?;
    let requests: usize = args.num("requests", 400)?;
    let cfg = bench_config(args)?;
    // Remove artifacts of earlier runs so raw/ always matches this REPORT.md.
    for entry in std::fs::read_dir(&raw).map_err(|e| format!("{}: {e}", raw.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().is_some_and(|x| x == "json") {
            std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    let full =
        requests == 400 && seeds == 3 && cfg.reps == 20 && cfg.warmup == 3 && args.opt("kv-gib").is_none();
    let mut md = String::new();
    md.push_str("# KVCacheX benchmark report\n\nGenerated by `kvx report`. Raw JSON is in `raw/`.\n");
    let _ = writeln!(
        md,
        "Parameters: {requests} requests per run, seeds 1..={seeds}, bench {} warm-up + {} reps.{}\n",
        cfg.warmup,
        cfg.reps,
        if full {
            " (full default configuration)"
        } else {
            " **REDUCED / NON-DEFAULT RUN: not the full report.**"
        }
    );
    md.push_str("Methodology: [docs/benchmarking.md](../docs/benchmarking.md).\n\n");
    let _ = writeln!(md, "Environment: `{}`\n", env_json().render().replace('\n', " "));
    md.push_str("## 1. Scheduler simulation (synthetic, time in steps)\n\n");
    md.push_str("Same requests, scheduler and KV byte budget for every backend. *mem held* = share of the KV budget allocated; *mem waste* = share of allocated slots holding no KV (reserved-but-unused or block slack). Not end-to-end inference results.\n\n");

    for kind in Kind::ALL {
        let mut sims = Vec::new();
        for seed in 1..=seeds {
            let mut a = vec![
                "--workload".to_string(),
                kind.name().into(),
                "--seed".into(),
                seed.to_string(),
                "--requests".into(),
                requests.to_string(),
            ];
            if let Some(g) = args.opt("kv-gib") {
                a.extend(["--kv-gib".into(), g.into()]);
            }
            let s = sim_setup(&Args::parse(&a)?)?;
            let results = run_all_backends(&s)?;
            let j = Json::obj(vec![
                ("setup", setup_json(&s)),
                (
                    "results",
                    Json::Arr(results.iter().map(SimResult::json).collect()),
                ),
            ]);
            let path = raw.join(format!("sim_{}_seed{seed}.json", kind.name()));
            std::fs::write(&path, j.render()).map_err(|e| format!("{}: {e}", path.display()))?;
            sims.push((s, results));
        }
        let (s0, r0) = &sims[0];
        let _ = writeln!(md, "### {} ({})\n", kind.name(), kind.describe());
        let _ = writeln!(
            md,
            "{} requests, rate {}/step, {} with {:.1} GiB KV = {} blocks x {}. Seed 1:\n",
            s0.spec.requests, s0.spec.arrival_rate, s0.model.name, s0.kv_gib, s0.num_blocks, s0.block_size
        );
        md.push_str(&sim_table(r0));
        // Cross-seed summary of the headline ratios.
        md.push_str("\nAcross seeds (mean ± stdev): steps to drain relative to contiguous, and mean memory waste.\n\n| backend | steps ratio | gen tok/step | prefill computed (k tok) | mem waste |\n|---|---:|---:|---:|---:|\n");
        for b in 0..r0.len() {
            let mut ratio: Vec<f64> = sims
                .iter()
                .map(|(_, r)| r[b].steps as f64 / r[0].steps as f64)
                .collect();
            let mut tps: Vec<f64> = sims.iter().map(|(_, r)| r[b].tokens_per_step()).collect();
            let mut pf: Vec<f64> = sims
                .iter()
                .map(|(_, r)| r[b].prefill_computed_tokens as f64 / 1e3)
                .collect();
            let mut waste: Vec<f64> = sims.iter().map(|(_, r)| 100.0 * r[b].mean_waste_frac).collect();
            let (ra, t, p, w) = (
                util::Summary::of(&mut ratio),
                util::Summary::of(&mut tps),
                util::Summary::of(&mut pf),
                util::Summary::of(&mut waste),
            );
            let _ = writeln!(
                md,
                "| {} | {:.3} ± {:.3} | {:.1} ± {:.1} | {:.1} ± {:.1} | {:.1}% ± {:.1} |",
                r0[b].backend, ra.mean, ra.stdev, t.mean, t.stdev, p.mean, p.stdev, w.mean, w.stdev
            );
        }
        md.push('\n');
        eprintln!("simulated {}", kind.name());
    }

    md.push_str("## 2. Metadata operation latency (wall clock, release build)\n\n");
    let _ = writeln!(
        md,
        "{} warm-up + {} measured repetitions per scenario; block size {}.\n",
        cfg.warmup, cfg.reps, cfg.block_size
    );
    let (timer, ops, threads) = run_bench(&cfg);
    md.push_str(&bench_tables(&timer, &ops, &threads));
    std::fs::write(
        raw.join("bench.json"),
        bench::json(&timer, &ops, &threads).render(),
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(raw.join("env.json"), env_json().render()).map_err(|e| e.to_string())?;
    let path = Path::new(&out).join("REPORT.md");
    std::fs::write(&path, md).map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

# KVCacheX

A Rust KV-cache manager with paged storage, copy-on-write prefix sharing, and optional CUDA support.

An LLM inference engine keeps the attention keys and values (the "KV cache") of every token it has
processed. KVCacheX manages that memory the way an operating system manages pages: it splits it into
fixed-size blocks, gives each sequence a block table, lets sequences with the same prompt prefix share
blocks (copying a block only when one of them writes to it), and evicts unused cached blocks in LRU order.

What you get:
* a library (`crates/kvcachex`, safe Rust, no dependencies by default) with CPU storage and an optional
  CUDA device backend (`--features cuda`);
* a CLI (`kvx`) to estimate KV memory for a model, simulate a scheduler, and run microbenchmarks;
* a contiguous-allocation baseline, so every result has something to compare against.

> Scope: this is the attention **KV-cache memory manager** of an inference engine, not an inference
> engine. Simulation results are synthetic (time in scheduler steps); see [what is measured](docs/benchmarking.md).

## Status

| | status |
|---|---|
| memory geometry (MHA / GQA / MQA, dtypes) | implemented, tested |
| paged block pool, ref counts, generation-checked handles | implemented, tested |
| sequence lifecycle: create / append / mark_computed / fork (COW) / release | implemented, tested |
| exact prefix cache, LRU eviction of cached free blocks | implemented, tested (incl. forced-collision mode) |
| CPU tensor storage with checked writes | implemented, tested (integrity verified under random ops) |
| CUDA device storage (`--features cuda`): device-resident KV, D2D copy-on-write | implemented; **validated on a Colab Tesla T4** (9/9 device tests, driver CUDA 13.0); see `results/gpu/` |
| thread-safe wrapper (`Mutex`) | implemented, stress-tested; does **not** scale with threads |
| contiguous baseline, scheduler simulator, benchmarks | implemented |
| attention kernels / model inference, CPU offload, KV quantization, H2O/StreamingLLM policies | **not implemented** (out of scope or future work) |
| MiniInfer integration | not started; integration point is `PagedKvCache` |

## Results (Apple M1, `results/REPORT.md`)

Synthetic scheduler simulation: fixed KV budget (2 GiB, Llama-3-8B geometry), identical requests
and scheduler for every backend, mean of seeds 1–3. Time is scheduler steps, not seconds:

| workload | steps vs. contiguous: paged | paged+prefix | memory waste: contiguous → paged |
|---|---:|---:|---|
| short | 0.36× | 0.36× | 64.6% → 3.1% |
| mixed | 0.54× | 0.54× | 37.4% → 0.8% |
| long | 0.65× | 0.65× | 23.3% → 0.3% |
| shared system prompts | 0.63× | **0.17×** | 25.3% → 0.6% (prefill tokens 492k → 72k with prefix caching) |
| parallel sampling (×4) | 0.54× | **0.26×** | 37.1% → 1.0% |

Wall-clock CPU cost of metadata operations, p50 in `results/REPORT.md`: decode append 0.21 µs,
512-token admission 0.58 µs (0.75 µs with prefix caching on), prefix-hit admission of a
1088-token prompt 6.9 µs (64 verified block lookups). Absolute latencies vary up to about 1.7× between
runs on this M1 (P/E cores); compare within one report only.

## Quick start

```bash
source ./env.sh                       # optional: Homebrew rustup / local CARGO_HOME if present
cargo test --workspace --release
cargo run --release -p kvx -- models
cargo run --release -p kvx -- estimate --model llama-3-8b --seq-len 4096 --batch 32 --budget-gib 16
cargo run --release -p kvx -- simulate --workload shared-prefix --requests 400
cargo run --release -p kvx -- bench
./scripts/reproduce.sh                # full report → results/
```

You need Rust; `rust-toolchain.toml` pins the version (1.99.1) and rustup installs it on first use.

## Development

CI (`.github/workflows/ci.yml`) runs on Linux and macOS: formatting, Clippy with warnings as errors, the
test suite in normal and forced-hash-collision modes, a check that the GPU tests fail clearly on a machine
without a GPU, Colab packaging, and a simulator invariant run.

Optional git hooks run the same checks before your changes leave your machine:

```bash
./scripts/install-hooks.sh            # enable for this clone
./scripts/install-hooks.sh --remove   # disable
```

* **pre-commit** checks formatting (`cargo fmt --all -- --check`), which takes a second or two.
* **pre-push** runs formatting, Clippy (`-D warnings`, all features) and both test suites, which takes a few
  minutes on a cold build.

A hook that fails stops the commit or push and prints the command that failed. If `cargo` isn't installed,
the hook fails and says so instead of skipping. To bypass a hook once, use `git commit --no-verify` or
`git push --no-verify`.

## GPU storage and Colab

`CudaKvStorage` (feature `cuda`) keeps KV values in one CUDA device allocation and runs copy-on-write
block copies device-to-device. Only explicit writes and reads cross PCIe. It uses `cudarc`'s safe
driver API with run-time loading of `libcuda`, so the default build and CI need no CUDA. The decision
record is [ADR-0005](docs/adr/0005-device-capable-storage-api.md).

To validate on a free Colab GPU: `./scripts/package_colab.sh`, then upload
`notebooks/kvcachex_colab.ipynb` and the ZIP to Colab. See [docs/colab.md](docs/colab.md). The GPU tests
fail rather than skip without a device.

Current recorded run ([results/gpu/](results/gpu/README.md), run 2 on a fresh runtime): Tesla T4, driver 580.82.07 /
CUDA 13.0. Device tests: 9 passed, 0 failed, 0 ignored (integrity, D2D copy and COW, error handling, memory lifecycle,
randomized model check). Storage microbenchmark: a 4 MiB COW block copy takes 45 µs p50 device-to-device, and an 8 KiB
single-slot write takes 13 µs (H2D); run 1 measured 49 µs and 19 µs. These are storage-transfer costs only, not
inference performance.

## Library example

```rust
use kvcachex::{geometry::ModelGeometry, CacheConfig, PagedKvCache};

let geom = ModelGeometry::preset("tiny").unwrap();
let mut cache = PagedKvCache::new_cpu(&geom, CacheConfig { block_size: 16, num_blocks: 1024, enable_prefix_caching: true });

let prompt: Vec<u32> = (0..40).collect();
let a = cache.create_sequence(/* namespace (e.g. LoRA id) */ 0, &prompt)?;
for pos in a.cached_tokens..prompt.len() {          // compute only what is not cached
    for layer in 0..geom.num_layers {
        let (k, v) = run_layer(pos, layer);          // your model
        cache.write(a.seq, pos, layer, &k, &v)?;
    }
}
cache.mark_computed(a.seq, prompt.len())?;          // full blocks become shareable
cache.append_tokens(a.seq, &[next_token])?;         // decode; COW handled internally
cache.release(a.seq)?;                              // blocks stay cached until evicted
```

## Documentation
* [docs/design.md](docs/design.md): problem, research survey with citations, architecture, invariants
* [docs/adr/](docs/adr): architecture decision records
* [docs/benchmarking.md](docs/benchmarking.md): methodology, what is and is not measured
* [progress.md](progress.md): status, decisions, next steps

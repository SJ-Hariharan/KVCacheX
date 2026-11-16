# Benchmark methodology

Reproduce everything: `./scripts/reproduce.sh` (tests, then `kvx report` → `results/REPORT.md` and `results/raw/*.json`).
Environment variables `REQUESTS`, `SEEDS` and `REPS` scale the run.

## What is and is not measured

| measured | how | not measured |
|---|---|---|
| batch size / concurrency a KV budget allows | scheduler simulation | real attention compute |
| memory held vs. KV actually stored (waste) | per-step sampling in simulation | GPU memory, PCIe/HBM traffic |
| prefill tokens avoided by prefix caching | simulation counters | TTFT/ITL in seconds |
| preemption and recompute | simulation | model quality |
| CPU cost of cache-metadata operations | wall-clock microbenchmarks | kernel cost of block-table indirection |

Simulation results are **synthetic**. A "step" is one scheduler iteration (see `crates/kvx/src/sim.rs`).
Fewer steps means the cache let more requests run at once. That is the mechanism by which PagedAttention
improves throughput, but steps are not seconds. No result here is an end-to-end inference speedup.

## Simulation setup
* Model geometry Llama-3-8B (32 layers, 8 KV heads, head_dim 128, f16 → 128 KiB/token). KV budget 2 GiB,
  so 1024 blocks × 16 tokens = 16,384 token slots. This is deliberately memory-constrained.
* Identical request streams (same seed) for all three backends. Poisson arrivals at 2 requests/step
  saturate the system, so results reflect capacity, not arrival rate.
* Scheduler: FCFS continuous batching, `max_batch_tokens` 4096, `max_seqs` 256. Paged backends keep a
  1% free-block watermark at admission. On out-of-memory during decode, the youngest sequence is preempted
  and re-queued for recompute (vLLM's default policy).
* Backends:
  * `contiguous`: reserves `prompt + max_tokens` contiguously, first-fit (`baseline::ContiguousAllocator`).
  * `paged`: KVCacheX with prefix caching off.
  * `paged+prefix`: KVCacheX with prefix caching on.
* Workloads (`crates/kvx/src/workload.rs`): short, long, mixed, shared-prefix (4 system prompts of 1024
  tokens with skewed popularity), parallel (groups of 4 identical prompts).
* Seeds 1..3. The report gives seed 1 in full, plus mean ± stdev across seeds of the headline ratios.
* Invariants: the simulator runs `check_invariants` at the end of every run (and every N steps with `--check-every N`).

Reading the columns: *prefill computed* counts prompt tokens whose KV had to be computed, including
recompute after preemption. *prefix-cached* counts prompt tokens served from the cache. On workloads without
shared prompts, the nonzero *prefix-cached* comes from preempted requests re-hitting their own
previously computed blocks, so prefix caching also cuts preemption recompute. *mem waste* is the
share of allocated slots holding no KV, averaged over steps in which the backend holds any memory.
*TTFT* is measured from arrival to the request's first generated token, even if the request is
later preempted and recomputed. A run that reaches `max_steps` before all requests finish is
reported as an error (`truncated` in JSON), never as partial results.

`kvx report` prints its parameters at the top of REPORT.md. Anything other than the defaults
(400 requests, seeds 1–3, 3 warm-up + 20 reps, 2 GiB) is labelled **REDUCED / NON-DEFAULT RUN**.
Each report run deletes the previous `raw/*.json` first, so `raw/` always matches `REPORT.md`.

## Microbenchmarks
* Release build (`lto = "thin"`, `codegen-units = 1`). Each operation timed individually with
  `std::time::Instant`. The timer's own cost is reported, not subtracted.
* 3 warm-up repetitions (untimed), then 20 measured repetitions per scenario. Percentiles use nearest rank
  over all timed operations.
* Threaded throughput: 1/2/4/8 threads, 2000 cycles per thread, 11 operations per cycle. Mean ± stdev over repetitions.

### Known measurement noise
Apple M1 has 4 performance and 4 efficiency cores, and macOS does not support pinning threads. Absolute
latencies varied **up to about 1.7× between runs on different occasions** (for example `create_512` p50
of 458 ns vs. 792 ns). Compare numbers within one report only. For A/B tests of code changes, we
interleave builds and compare medians (see `progress.md`, "prefix-hit hash optimization").

## GPU storage microbenchmark (`kvx gpu-bench`, feature `cuda`)
Times **storage operations only** on `CudaKvStorage`, with the same layout on `CpuKvStorage` as a reference:

| op | data movement (CUDA) | bytes (llama-3-8b geometry, f32, block 16) |
|---|---|---|
| `write_1_slot` | H2D, K and V | 8 KiB |
| `write_block_plane` | H2D, 16 slots of one layer | 128 KiB |
| `read_1_slot` | D2H + sync | 8 KiB |
| `read_block_plane` | D2H + sync | 128 KiB |
| `copy_block_cow` | **D2D**, whole block | 4 MiB |

Each timed operation ends with a stream synchronize, so the time is to completion, not to enqueue.
Default 20 warm-up + 200 timed repetitions, nearest-rank percentiles, and a readback check afterwards.
These are PCIe and device-memory transfer costs of a storage backend. They are **not** attention,
TTFT or tokens/s; KVCacheX runs no model. Results go to `results/gpu/gpu_bench.json` only via the
Colab notebook.

## Profiling
`crates/kvcachex/examples/prof_hit.rs` is a driver for macOS `sample` / Instruments:
```bash
cargo build --release --example prof_hit && ./target/release/examples/prof_hit & sample $! 3
```

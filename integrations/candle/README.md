# kvcachex-candle: Candle running on KVCacheX (CPU reference integration)

This crate runs a real language model in the [Candle](https://github.com/huggingface/candle) inference
runtime with its attention K/V state stored and managed by KVCacheX. It compares the result, token for
token, against Candle's own KV cache.

* **Model:** Candle's `llama2_c` implementation with the public TinyStories `stories15M` checkpoint
  (15M parameters, 6 layers, context 256 tokens, MIT license), plus llama2.c's tokenizer.
* **Two engines, same model, weights and Candle ops:**
  * `candle-native`: Candle's unmodified model and cache;
  * `kvcachex` / `kvcachex-prefix`: the same forward pass with K/V in KVCacheX paged blocks, with the
    exact prefix cache off or on. See [ADR-0006](../../docs/adr/0006-candle-reference-integration.md)
    for why the forward pass had to be re-expressed (`src/model.rs`).
* **CPU only, f32, one sequence per forward pass.** There is no GPU path and no paged-attention kernel:
  attention gathers the sequence's K/V from its blocks into a dense buffer, then runs Candle's attention.

It is a separate Cargo workspace, so Candle never enters the default KVCacheX build.

## Setup and commands

From the repository root:

```bash
source ./env.sh                       # optional: local toolchain setup (see the main README)
./scripts/fetch_tiny_models.sh        # ~61 MB into integrations/candle/models/, SHA-256 verified
cd integrations/candle
cargo test --release                  # correctness vs Candle's native cache (synthetic weights, no download needed)
cargo test --release -- --ignored     # the same against the real stories15M checkpoint
cargo run --release --bin kvx-candle -- generate --engine kvcachex-prefix --prompt "Once upon a time" --steps 64
cargo run --release --bin kvx-candle -- bench --workload shared-prefix --concurrency 4
```

Full benchmark report (4 configurations, writes `results/candle/`): `./scripts/candle_bench.sh` from the
repository root.

## What the tests check

| test | checks |
|---|---|
| `paged_cache_is_bit_identical_to_native_cache` | 23-token prompt + 40 greedy steps: every logit **bit-identical** to Candle's cache |
| `prefix_hit_skips_cached_positions_and_matches_native` | a prompt sharing 5 cached blocks runs only the 6 uncached positions; logits within 1e-4, same tokens |
| `fully_cached_prompt_reruns_only_last_position` | an entirely cached prompt computes 1 position and still matches |
| `fork_copy_on_write_matches_native_fork` | a fork with a shared partial block, diverging tokens, copy-on-write; bit-identical to native forks |
| `interleaved_requests_and_eviction_match_native` | 12 requests, 2 in flight, a pool small enough to evict; same tokens, invariants hold, no leaks |
| `out_of_blocks_is_an_error_not_a_crash` | pool exhaustion returns an error |
| `kv_byte_accounting` | the bytes reported for both caches |
| `stories15m_paged_matches_native` (opt-in) | real checkpoint: identical greedy text and prefill logits; same text again from the prefix cache |

The synthetic config uses grouped-query attention (4 query heads, 2 K/V heads); `stories15M` is MHA.

## What the benchmark is, and is not

`kvx-candle bench` is a **closed-loop, interleaved workload in one CPU process**. All requests arrive at
t=0, up to `--concurrency` are in flight, and each step runs one model forward pass per in-flight request
(batch size 1, round-robin). It is not batched execution and not an inference server.

* *Throughput* is generated tokens divided by the wall time of that workload.
* *TTFT* is **queue-inclusive**: it counts time spent waiting behind earlier prompts. *Prefill latency* is
  each request's own prefill forward pass.
* *Rust heap peak* is peak Rust heap growth (global-allocator count), **not process RSS**.

Prefix caching helps only when prompts share full blocks of tokens: KVCacheX then skips recomputing them.
Candle's native cache has no cross-request reuse. Without reuse (`--workload unique`), KVCacheX does the
same work plus a block-table gather. None of these numbers are GPU results or production-serving speedups.

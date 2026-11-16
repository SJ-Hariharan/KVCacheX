# ADR-0006: Candle reference integration (CPU) and the CUDA follow-up

**Status:** accepted for the CPU reference integration. The CUDA path below is planned, not built.

## Context

KVCacheX was a tested cache manager with no inference runtime using it. The goal of this step is the
smallest honest end-to-end proof: a real runtime runs a real model with its K/V state in KVCacheX,
the outputs are checked against that runtime's native cache, and an A/B benchmark compares the two.

Rust runtimes considered (versions as fetched on 2026-11-16):

| runtime | KV-cache extension point | fit |
|---|---|---|
| **Candle 0.11.1** (`candle-core`, `candle-nn`, `candle-transformers`) | None. `models::llama::Cache` keeps `kvs` private; `models::llama2_c::Cache` makes `kvs` public, but both attention layers are private and grow the cache with `Tensor::cat` inside `forward`. | Pure Rust, CPU without system libraries, small public models (llama2.c TinyStories), same `cudarc` (0.19.10) as `CudaKvStorage`. Chosen. |
| mistral.rs | Has its own paged-attention engine and scheduler. | Replacing its cache means working inside a full serving engine; far larger surface. |
| burn / tch-rs / llama.cpp bindings | No KV-cache hook; tch-rs and llama.cpp need native libraries. | Heavier build, less control. |

## Decision

1. **Model:** Candle's `llama2_c` model with the TinyStories `stories15M` checkpoint (60.8 MB,
   6 layers, dim 288, 6 heads, context 256, MIT), pinned by revision and SHA-256
   (`scripts/fetch_tiny_models.sh`). Tokenizer: llama2.c's `tokenizer.bin`, ported in
   `integrations/candle/src/tokenizer.rs` (no `tokenizers` dependency). `stories260K` (GQA) can't be
   used: Candle 0.11.1's `.bin` reader assumes `n_kv_heads == n_heads`.
2. **Baseline:** `candle_transformers::models::llama2_c::Llama` with its native `Cache`, unmodified.
3. **KVCacheX path:** because there is no injection point, `PagedLlama` (`src/model.rs`, about 200 lines)
   re-expresses that model's forward pass. It loads the same weights by name from the same `VarBuilder`
   and applies the same Candle layers and ops in the same order. Only the cache differs: new K/V go into
   the sequence's blocks (`PagedKvCache::write_range`), and attention reads the sequence's K/V back
   in position order (`PagedKvCache::read_range_into`, added for this) and runs Candle's dense attention.
   This port is scoped to this one tiny reference model; it isn't a general model library.
4. **Isolation:** `integrations/candle` is its own Cargo workspace with its own lockfile. The default
   KVCacheX build and lockfile do not change. CI builds and tests it with synthetic weights (no download).
5. **CPU only.** The integration never moves GPU K/V through host memory and never calls itself GPU work.

## Consequences

- Correctness is checked directly: with no prefix hits the KVCacheX path is **bit-identical** to
  Candle's native cache (same logits, synthetic and real weights). With prefix hits the logits match to
  within 1e-4 and greedy tokens are identical (cached K/V came from a different prefill shape).
  Forks (copy-on-write), eviction, interleaved requests and pool exhaustion are tested.
- The benchmark (`results/candle/`) can show the effect of prefix reuse (fewer prefill tokens) and the
  overhead of the paged gather. It can't show kernel-level gains: there is no paged-attention kernel.
- Each forward pass copies the sequence's K/V into a contiguous buffer. Candle's native cache does the
  same with `Tensor::cat`, so the two are comparable, but neither is how a production engine reads K/V.

## CUDA follow-up (planned work)

Verified facts: `candle-core` 0.11.1 depends on `cudarc` =0.19.10, the same version as `CudaKvStorage`.
`CudaDevice::cuda_stream()` exposes Candle's `Arc<CudaStream>`, `CudaStorage::wrap_cuda_slice` /
`as_cuda_slice` convert between Candle storage and `CudaSlice<f32>`, and `CustomOp1::cuda_fwd` lets
external code return new device storage (`Tensor::from_storage` is crate-private). So a device-resident
path needs **no host round trip and no new kernel** for a first version:

1. `kvcachex` (feature `cuda`): `CudaKvStorage::on_stream(stream: Arc<CudaStream>, ..)` so the cache
   shares Candle's context and stream. Add device-side
   `PagedKvCache<CudaKvStorage>::{write_range_device, gather_range_device}`, which copy between a
   sequence's blocks and caller `CudaView`/`CudaViewMut` buffers with device-to-device `memcpy`, one per
   block run. These are safe cudarc calls, so `unsafe_code = "forbid"` stays.
2. `integrations/candle` (feature `cuda` → `candle-core/cuda`): a `KvBackend` trait in `PagedLlama` with
   the CPU implementation from this ADR and a CUDA implementation that writes K/V from Candle's
   device tensors and gathers into a new device tensor through a `CustomOp1`.
3. Validate on the Colab T4 notebook: extend it to build `integrations/candle --features cuda`. Candle's
   CUDA kernels compile with `nvcc`; whether candle-kernels 0.11.1 builds with Colab's CUDA 13.0 toolkit
   is **unverified**. Run the correctness tests against Candle's native CUDA cache and the same A/B benchmark
   with `cudaEventRecord` timing and `cuMemGetInfo` for device memory.
4. Later, a real paged-attention decode kernel (reads K/V through the block table, no gather) needs
   kernel launches. cudarc's `launch` is `unsafe`, so that step needs a separate crate with a reviewed
   `unsafe` boundary and an amendment to this workspace's `unsafe_code = "forbid"` rule.

Until step 3 has run on real hardware, no document may claim GPU inference results for KVCacheX.

## Free GPU options for step 3 (researched 2026-11-16, not used yet)

From official documentation, collected for this project; nothing was signed up for, billed or started.

| option | what it offers | constraint |
|---|---|---|
| Google Colab (free tier) | T4 GPU in notebooks; the existing `notebooks/kvcachex_colab.ipynb` already passed the storage tests there (`results/gpu/`) | availability and usage limits vary ([FAQ](https://research.google.com/colaboratory/faq.html)) |
| Kaggle Notebooks | one NVIDIA P100 notebook GPU | weekly quota, queue and session limits ([docs](https://www.kaggle.com/docs/notebooks), [GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage)) |
| Google Cloud Free Trial | $300 / 90 days for eligible new accounts | a non-billable trial **cannot attach GPUs** or request quota; GPUs need a billing upgrade, which can incur charges ([free features](https://docs.cloud.google.com/free/docs/free-cloud-features), [GPUs](https://docs.cloud.google.com/compute/docs/gpus/about-gpus)) |
| NVIDIA LaunchPad | free, time-limited guided labs | not general-purpose GPU time ([page](https://www.nvidia.com/en-us/data-center/enterprise-software/)) |
| Modal (Starter) | $30/month of included compute credit | credit-based usage that can bill beyond credits; needs explicit spend controls and approval ([pricing](https://modal.com/pricing)) |

**Next practical path:** Colab first (same notebook, free T4), Kaggle's P100 as a fallback. Both need the
repository owner to start the notebook session; no paid option is used without explicit approval.

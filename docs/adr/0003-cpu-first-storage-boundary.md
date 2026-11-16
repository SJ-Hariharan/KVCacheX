# ADR-0003: CPU-first, metadata/storage separation

**Status:** accepted; storage API amended by [ADR-0005](0005-device-capable-storage-api.md)

**Context.** Development machine: Apple M1, 8 GB RAM, no CUDA. GPU memory management cannot be
validated here.

**Decision.** `CacheManager` is metadata-only and returns block ids and `BlockCopy` instructions.
Tensor bytes live behind the `KvStorage` trait. `CpuKvStorage` is a reference backend used to prove
data integrity (model-based tests write prefix-dependent KV values and read them back after every operation).

**Consequences.** No GPU or end-to-end inference claims. A GPU or offload backend implements
`KvStorage` and executes copies on device. MiniInfer would integrate through `PagedKvCache`.

**Update (milestone 3).** The original trait could not support device memory (borrowed host reads,
infallible operations). ADR-0005 makes it fallible and transfer-explicit, and adds an optional CUDA
backend (`--features cuda`). Development stays CPU-first: the default build and CI need no CUDA. This
machine still cannot validate the GPU path; that runs on Colab via `notebooks/kvcachex_colab.ipynb`.

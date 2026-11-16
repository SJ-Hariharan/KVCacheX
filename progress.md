# Progress

_Last updated: 2026-11-16 (Candle reference integration)_

## Done (milestone 1: core engine + measurement)
- Environment: Apple M1 (4P+4E), 8 GB, macOS 15.5, Rust 1.99.1, Python 3.15 (no numpy), **no CUDA** → CPU-first (ADR-0003).
- Research: PagedAttention, vLLM APC, RadixAttention, H2O, StreamingLLM, KIVI, GQA/MQA, cited in `docs/design.md`.
- Library: geometry; block pool (ref counts, generations, intrusive LRU free queue, clean-first allocation);
  exact prefix cache (namespace + parent content id + token verification); lifecycle with publish-after-compute,
  COW fork, tail-first release, atomic failure; CPU storage + `PagedKvCache`; `SharedCacheManager`; contiguous baseline.
- Tests (all green, both hash modes): 39 unit, 3 model-based property suites (512 cases each, KV integrity read-back),
  3 baseline/geometry property tests (written by a Haiku worker, reviewed), 8-thread stress, simulator tests, doctest.
  Mutation checks: skipping the COW copy → caught (corrupt KV); ignoring parent cid → caught only with `degenerate-hash`,
  which is why CI runs that mode.
- Bugs found: invariant checker wrongly flagged fully computed shared partial tails (fixed in the checker, not the manager).
- CLI + simulator + microbenchmarks; `results/REPORT.md` generated (3 seeds × 5 workloads).

## Done (milestone 2: release readiness, 2026-11-15)
- All documented checks green locally on 1.99.1 (fmt, clippy -D warnings, 57 workspace tests, 46 degenerate-hash tests,
  simulator invariants). Full `scripts/reproduce.sh` regenerated `results/`, and an audit of REPORT.md against the raw JSON found 0 mismatches.
- Fixed: TTFT after preemption, mean-waste averaging, silent `max_steps` truncation, stale raw artifacts,
  false MSRV (1.80 → removed; proptest needs 1.88), CI toolchain mismatch, env.sh portability, stale doc claims.
- Details and exact results: `RELEASE_NOTES.md`. Remote CI was not observed at the time (no remote yet).

## Done (milestone 3: device-capable storage + Colab path, 2026-11-15/16) — GPU validated on Colab T4
- Storage API redesigned for device memory (ADR-0005): fallible, transfer-explicit `KvStorage` (`write_slots`,
  `read_slots_into`, `copy_block`, `synchronize`, `layout`, `residency`), shared bounds checks, `PagedKvCache`
  poisoning on fatal device errors, `write_range` across blocks. Borrowed reads are kept only for host storage.
- `CudaKvStorage` behind `--features cuda` (cudarc =0.19.10, driver API, runtime-loaded libcuda, CUDA 12.0 bindings):
  one device allocation, H2D writes, D2H reads plus sync, D2D whole-block COW copies; `unsafe_code = "forbid"` kept
  (missing-driver panic converted to `Unavailable` via `catch_unwind`).
- `tests/gpu.rs` (9 tests incl. randomized model check sharing `tests/common` with the CPU suite): fail, never skip, without a GPU.
  `kvx gpu-info` / `kvx gpu-bench` (storage transfers only, CPU reference, readback check).
- `scripts/package_colab.sh` + `notebooks/kvcachex_colab.ipynb` (+ `docs/colab.md`). The notebook was dry-run locally from the ZIP:
  CPU checks pass, and the GPU cell raises as designed with no GPU.
- **Final validation, run 2 (2026-11-15 19:18–19:26 UTC, fresh runtime, unmodified notebook):** 17 checks, each once,
  all exit 0; CPU 63/0/0 and collision 52/0/1; GPU 9 passed / 0 failed / 0 ignored; benchmark regenerated. Source
  `dist/kvcachex-src-run2.zip`, sha256 `8b826ad2…85fe`. Run all's queue stopped after the upload cell, so the rest ran via
  "Run cell and below" in the same kernel without edits. Runtime deleted afterwards. Run 1 is kept in `results/gpu/history/`.
- **Colab GPU run 1** (user signed in to the built-in browser pane and uploaded the notebook and ZIP): Tesla T4,
  driver 580.82.07 / CUDA 13.0, rustc 1.99.1. CPU checks pass on Colab (63 tests; 52 + 1 ignored in collision mode).
  `tests/gpu.rs` 9/9 passed twice. `results/gpu/` holds the run record and benchmark, with provenance notes.
- The run found and fixed two notebook bugs: libcuda detection ignored `LD_LIBRARY_PATH` (false "no GPU" on Colab),
  and a stale blocker note survived a later successful GPU pass. Both fixes were applied byte-identically in Colab mid-run.

## Done (milestone 4: publication, 2026-11-16)
- Final local checks on the published tree (macOS, rustc 1.99.1): fmt pass; clippy `-D warnings` pass (default and
  `--all-features`); workspace 63/0/0; collision 52/0/1; simulator invariants pass.
- Optional git hooks (`.githooks/`, `scripts/install-hooks.sh`): pre-commit format check; pre-push fmt + clippy +
  both test suites. They fail loudly when `cargo` is missing.
- Machine-local agent configuration (`.claude/`, `CLAUDE.md`) is git-ignored and not published. The Colab archive now
  also leaves `CLAUDE.md` out (it followed `.gitignore`); the run-2 archive included it.
- Pushed to `github.com/SJ-Hariharan/KVCacheX` (`main`). CI results are on the Actions page; they were not known
  when this file was committed, so none are claimed here.

## Done (public-readiness audit, 2026-11-16)
- `LICENSE-MIT` / `LICENSE-APACHE` (2026 SJ-Hariharan) and a README License section, consistent with `MIT OR Apache-2.0`.
- `cargo audit --deny warnings`: 0 vulnerabilities, 0 warnings (54 dependencies, advisory-db `02e2517f`, 2026-11-15).
- History, tracked files, results and archive scanned for secrets, private paths and agent config: clean.
- The README example is now a compiled doctest; CI actions are pinned to commit SHAs with `persist-credentials: false`.
- Local: fmt and clippy pass; workspace 64/0/0; collision 53/0/1; simulator pass; both hooks pass. No new GPU run
  (the CUDA code is unchanged since the T4 validation). Details: `RELEASE_NOTES.md`.

## Done (Candle reference integration, 2026-11-16) — CPU/f32 only
- `integrations/candle` (separate workspace): Candle 0.11.1's llama2.c model with the public TinyStories `stories15M`
  checkpoint, running with K/V in a KVCacheX `PagedKvCache`, against Candle's unmodified native cache (ADR-0006).
  Candle has no KV-cache hook, so `PagedLlama` re-expresses that model's forward pass with the same weights and ops.
- Library: `PagedKvCache::read_range_into` (block-wise, position-ordered gather) plus a unit test.
- Correctness: 7 tests on synthetic GQA weights (bit-identical without prefix hits; within 1e-4 with prefix hits,
  full-prompt hits, COW forks, eviction and interleaving; pool exhaustion; byte accounting) and 1 opt-in
  real-checkpoint test (identical greedy text and prefill logits). A mutation (K stored in V) fails 5 of 7.
- Benchmark (`results/candle/`, M1, closed-loop interleaved, one forward per request, not batched, not a server):
  on shared-prefix prompts, prefix reuse cut prompt tokens computed by 69–78% and prefill p50 from 34.0/42.4 to
  15.5/17.3 ms (C=1/4), and peak KV in use from 7.05 to 5.06 MiB at C=4. Throughput changed by −8% to +6%, within
  ~13% noise. Without reuse, KVCacheX ran at 0.85–1.03× native throughput. Every arm's outputs matched native.
- Found: Candle 0.11.1 can't load the GQA `stories260K` `.bin` (its reader assumes n_kv_heads == n_heads).
- CI: a `candle-integration` job (fmt, clippy, synthetic-weight tests, missing-model error). Hooks cover the crate.

## Key measurements
- Paged vs contiguous: 0.36–0.65× steps to drain; waste 24–65% → 0.3–3%.
- Prefix caching on shared system prompts: 0.17× steps; prefill 492k → 72k tokens.
- Prefix-hit hash optimization: profiled with `sample`; `BlockPool::lookup` was about 55% of samples
  (a serial multiply chain through the parent). Token hash now independent of the parent, 4 lanes.
  Interleaved A/B, 5 runs each, create+release with a 64-block hit: median 6883 → 5137 ns (about 25%).
- Mutex scaling is negative (3–6 M ops/s on 1 thread across runs → about 0.9 M on 8); batching 11 ops/lock gives about 1.6× at 8 threads. See ADR-0004.

## Next steps (candidates, in priority order)
1. GPU path for the Candle integration (ADR-0006 step 1–3): device-side write/gather on Candle's CUDA stream
   (same cudarc 0.19.10), validated on the Colab T4 notebook (Kaggle P100 as fallback). Unverified: whether
   candle-kernels 0.11.1 builds with Colab's CUDA 13.0 toolkit.
2. Keep CI green on ubuntu-latest and macos-latest.
3. Cascading invalidation of unreachable cached children, and measuring the cache-capacity gain.
4. Eviction policy experiment: LRU vs. depth-aware/frequency-aware (only with a workload where LRU demonstrably loses).
5. CPU swap/offload tier behind `KvStorage` (copy cost measured with the real CPU backend).
6. MiniInfer adapter via `PagedKvCache` once a minimal attention kernel exists.

## Blockers
- No GPU locally: GPU validation depends on Colab and the user's sign-in and uploads. Storage validated on one GPU model (T4) only.
- No GPU *inference* path exists yet; the Candle integration is CPU-only.

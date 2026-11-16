# Progress

_Last updated: 2026-11-16 (milestone 4: published to GitHub)_

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

## Key measurements
- Paged vs contiguous: 0.36–0.65× steps to drain; waste 24–65% → 0.3–3%.
- Prefix caching on shared system prompts: 0.17× steps; prefill 492k → 72k tokens.
- Prefix-hit hash optimization: profiled with `sample`; `BlockPool::lookup` was about 55% of samples
  (a serial multiply chain through the parent). Token hash now independent of the parent, 4 lanes.
  Interleaved A/B, 5 runs each, create+release with a 64-block hit: median 6883 → 5137 ns (about 25%).
- Mutex scaling is negative (3–6 M ops/s on 1 thread across runs → about 0.9 M on 8); batching 11 ops/lock gives about 1.6× at 8 threads. See ADR-0004.

## Next steps (candidates, in priority order)
1. Keep CI green on ubuntu-latest and macos-latest (first remote run happens with the initial push).
2. Cascading invalidation of unreachable cached children, and measuring the cache-capacity gain.
3. Eviction policy experiment: LRU vs. depth-aware/frequency-aware (only with a workload where LRU demonstrably loses).
4. CPU swap/offload tier behind `KvStorage` (copy cost measured with the real CPU backend).
5. MiniInfer adapter via `PagedKvCache` once a minimal attention kernel exists.

## Blockers
- No GPU locally: GPU validation depends on Colab and the user's sign-in and uploads. Validated on one GPU model (T4) only.

# Candle reference integration (2026-11-16): CPU/f32 only

KVCacheX now holds the K/V state of a real model running in a real inference runtime. That runtime is
Candle 0.11.1's llama2.c model with the public TinyStories `stories15M` checkpoint (15M parameters, f32,
**CPU**). This is a reference integration, not a GPU result and not a production server.

**Added**
- `integrations/candle` (its own Cargo workspace and lockfile; the default build is unchanged):
  - `PagedLlama`: the llama2_c forward pass with K/V in KVCacheX, same weights and Candle ops;
  - engines for Candle's native cache and for KVCacheX, with and without the prefix cache;
  - a port of the llama2.c tokenizer;
  - `kvx-candle generate|bench`.
- `PagedKvCache::read_range_into` in the library (block-wise gather in position order).
- `scripts/fetch_tiny_models.sh`, which downloads files at pinned revisions and checks their SHA-256 hashes;
  `scripts/candle_bench.sh` and `scripts/candle_report.py`, which produce `results/candle/`.
- ADR-0006: runtime choice, design, the CUDA follow-up plan, and free-GPU options (none used).
- CI: a `candle-integration` job (synthetic weights, no download, no GPU). The hooks cover the integration.

**Correctness** (Candle's native cache as the reference)
| test | result |
|---|---|
| no prefix hits (synthetic GQA weights; 23-token prompt + 40 steps) | every logit bit-identical |
| prefix hit / fully cached prompt / COW fork / eviction with interleaving | same greedy tokens; logits within 1e-4 (bit-identical for forks) |
| `stories15M`, 64 greedy steps (opt-in, `--ignored`) | identical text and prefill logits; same text again from the prefix cache |
| benchmark, 4 configurations × 3 arms × 5 timed reps | every arm's outputs identical to native in every rep |

**Benchmark** (`results/candle/REPORT.md`). Setup:
- Apple M1 (4P+4E), 8 GB, macOS, rustc 1.99.1, `RAYON_NUM_THREADS=4`;
- 16 requests, 48 generated tokens each, block size 16, 1 warm-up + 5 timed repetitions, arm order rotated.

The workload is closed-loop and interleaved, in one process, with one forward pass per request per step. It isn't
batched and it isn't a server. TTFT is queue-inclusive and is reported separately from per-request prefill latency.
The heap figure is Rust heap growth, not RSS.

| configuration | native tok/s | kvcachex tok/s | kvcachex-prefix tok/s | prefill p50 native → prefix | prompt tokens computed native → prefix |
|---|---:|---:|---:|---:|---:|
| shared-prefix, C=1 | 192.8 | 181.0 | 205.3 | 34.0 → 15.5 ms | 1388 → 428 |
| shared-prefix, C=4 | 160.2 | 150.1 | 148.0 | 42.4 → 17.3 ms | 1388 → 300 |
| unique, C=1 | 148.6 | 126.8 | 141.5 | 50.1 → 51.4 ms | 1586 → 1586 |
| unique, C=4 | 134.9 | 138.5 | 120.5 | 55.7 → 58.6 ms | 1586 → 1586 |

How to read it:
- Prefix reuse reliably cuts prefill work and latency on shared prompts. At C=4 it also cuts peak KV in use
  (7.05 → 5.06 MiB).
- Throughput differences aren't reliable here. On `unique`, the two KVCacheX arms do identical work, yet they
  differ by 13%: that is the noise floor.
- Without reuse, KVCacheX ran at 0.85–1.03× native throughput, slower on 3 of 4 configurations. It adds a
  block-table gather on every step.

**Checks** (local, macOS, after `source ./env.sh`)
| check | result |
|---|---|
| root fmt; clippy `-D warnings` default and `--all-features` | pass |
| root workspace tests / collision mode | 65/0/0 / 54/0/1 |
| simulator invariants | pass |
| `integrations/candle`: fmt, clippy `-D warnings`, tests | pass; 7 passed, 1 ignored (opt-in real model) |
| `integrations/candle` real-model test (`--ignored`, `stories15M`) | 1 passed |
| hooks (pre-commit, pre-push) under `dash` | pass |
| `cargo audit --deny warnings`, root lockfile | 0 vulnerabilities, 0 warnings |
| `cargo audit`, `integrations/candle/Cargo.lock` (167 crates) | 0 vulnerabilities; 1 unmaintained warning: `paste` 1.0.15 (RUSTSEC-2024-0436), a compile-time proc-macro used by Candle's `gemm`, no fixed release |

**Limits**
- CPU only, f32, batch size 1 per forward pass. There is no GPU inference path and no paged-attention kernel:
  attention gathers K/V into a dense buffer, as Candle's own `Tensor::cat` cache does.
- The KVCacheX forward pass is a port of Candle's `llama2_c` forward pass, needed because Candle has no
  KV-cache hook. It covers this one tiny model.
- `stories260K` (GQA) can't be loaded by Candle 0.11.1. Real-checkpoint testing is therefore MHA only; GQA is
  covered by the synthetic-weight tests.
- Remote CI for this change hadn't finished when these notes were committed; see the Actions page.

# Public-readiness audit (2026-11-16)

Third commit on `main`. It adds licenses, compiles the README example as a test, and hardens CI. No library or CLI
behavior changed. The CUDA code is unchanged since the Colab T4 runs (`results/gpu/`); no new GPU run was made.

**Changes**
- `LICENSE-MIT` and `LICENSE-APACHE` (copyright 2026 SJ-Hariharan) match the existing `license = "MIT OR Apache-2.0"`
  in `Cargo.toml`. The Apache file is the official apache.org text (sha256 `cfc7749b…3d30` before editing) with only
  the appendix's copyright placeholder line filled in. The README gains a License section.
- The README library example called undefined helpers. It is now a complete program, compiled and run as a doctest
  (`ReadmeDoctests` in `lib.rs`, `cfg(doctest)` only).
- CI: `actions/checkout` and `actions/cache` are pinned to the full commit SHAs of their `v4` tags, and checkout sets
  `persist-credentials: false`. Permissions were already `contents: read`, there are no secrets, and there is no
  `pull_request_target`.

**Audit results**
| item | result |
|---|---|
| `cargo audit` (cargo-audit 0.23.0; it has no `--locked` flag, so `--deny warnings` was used on the committed `Cargo.lock`, which stayed unchanged) | 0 vulnerabilities, 0 warnings (unmaintained/yanked/unsound); 54 crate dependencies; 1509 advisories, advisory-db commit `02e2517f`, 2026-11-15 |
| lockfile | 56 packages, all from crates.io (no git or path dependencies besides the two workspace crates); `cudarc` only behind `cuda` |
| unsafe | `unsafe_code = "forbid"` workspace-wide, inherited by both crates; no `unsafe` in the code. The one `catch_unwind` (`storage/cuda.rs`) converts cudarc's missing-driver panic to `StorageError::Unavailable` |
| file writes | `kvx` writes only to paths it is given; `kvx report` deletes only `*.json` in `<out>/raw/` before regenerating |
| scripts / hooks | `set -eu`; no `eval`, no downloads, no `rm -rf` of a variable path except `package_colab.sh`'s own `mktemp -d` directory |
| tracked files and full git history | no tokens/keys (GitHub, AWS, OpenAI-style, Slack, Google, private-key patterns), no user email, no `/Users/` or host names, no `.claude/` or `CLAUDE.md`; one author/committer (the GitHub noreply address); no co-author trailers |
| result records | no IPs, UUIDs, serials or GPU UUIDs; Colab paths are the generic `/content/...` |
| upload archive (`scripts/package_colab.sh`) | 80 files, identical to the repository's file set; both license files byte-identical; no forbidden paths; secret scan clean |

**Checks (local, macOS, rustc 1.99.1, after `source ./env.sh`)**
| check | result |
|---|---|
| `cargo fmt --all -- --check` | pass |
| `cargo clippy --locked --all-targets -- -D warnings`, and with `--all-features` | pass |
| `cargo test --locked --workspace --release` | 64 passed, 0 failed, 0 ignored (includes the README doctest) |
| `cargo test --locked -p kvcachex --release --features degenerate-hash` | 53 passed, 0 failed, 1 ignored |
| simulator invariants | pass |
| `.githooks/pre-commit`, `.githooks/pre-push` under `dash` | pass |

Remote CI for this commit hadn't finished when these notes were committed; see the Actions page.

# Publication (2026-11-16)

First push to `github.com/SJ-Hariharan/KVCacheX`, branch `main`, two commits (implementation; user docs, hooks and
validation records). No tag. Changes since milestone 3: optional git hooks (`.githooks/`, `scripts/install-hooks.sh`),
a plain-language README, and `.gitignore` rules that keep machine-local agent configuration (`.claude/`, `CLAUDE.md`)
out of the repository. No library, CLI, test, CI or notebook code changed.

| check (local, macOS, rustc 1.99.1, on the published tree) | result |
|---|---|
| `cargo fmt --all -- --check` | pass |
| `cargo clippy --locked --all-targets -- -D warnings`, and with `--all-features` | pass |
| `cargo test --locked --workspace --release` | 63 passed, 0 failed, 0 ignored |
| `cargo test --locked -p kvcachex --release --features degenerate-hash` | 52 passed, 0 failed, 1 ignored |
| simulator invariants (`kvx simulate --workload shared-prefix --requests 100 --check-every 10`) | pass |
| hooks: pre-commit and pre-push run under `dash` | pass |

Remote CI runs for the first time on this push. Its result wasn't known when these notes were committed and isn't
recorded here; see the repository's Actions page.

# Milestone 3 (2026-11-15): device-capable storage, optional CUDA backend, Colab path

Committed with the publication above (it was not committed at the time). The CUDA backend is **validated on a Google Colab Tesla T4**, in two runs.
The current record is run 2 (below); run 1 is kept for provenance.

## Final validation (run 2, 2026-11-15 19:18–19:26 UTC)
| item | result |
|---|---|
| tree vs recorded run-1 ZIP (`031d77ee…`) | **byte comparison impossible**: that archive no longer exists (`dist/` was regenerated, and Colab recycled its VM). By file timestamps, no code, manifest, script, CI or toolchain file changed after run 1 started; only the notebook fixes, docs and results did |
| source for run 2 | `dist/kvcachex-src-run2.zip`, sha256 `8b826ad2103e23f075237dcf55ccc1d2fd854130b0657959ce7c3c29a9c785fe`, 72 files, kept locally |
| archive inspection | no `.git`, `target/`, `.cargo-home/`, `.claude/`, `.DS_Store`, `.env*`, keys or credentials; secret-pattern scan (cloud/API tokens, private keys, user email) found nothing |
| local checks (macOS, before the run) | fmt pass; clippy `-D warnings` pass (default and `--all-features`); workspace 63/0/0; collision 52/0/1; simulator invariants pass |
| Colab runtime | fresh T4 runtime (this notebook's previous runtime deleted first); deleted again after the run |
| device / toolchain | Tesla T4 cc 7.5, 15360 MiB; driver 580.82.07, driver CUDA 13.0; nvcc 13.0; Python 3.13.16; rustc/cargo 1.99.1 |
| notebook | identical to the repository version (all 10 code cells hash-matched) before and after the run; no edits during the run |
| Colab CPU checks | fmt, clippy, 63/0/0, 52/0/1, simulator invariants: all exit 0 |
| Colab GPU checks | `tests/gpu.rs` **9 passed, 0 failed, 0 ignored**; `gpu-info` and `gpu-bench` exit 0; `gpu_validated: true`, no blocker |
| benchmark (regenerated) | 4 MiB D2D COW copy 45.4 µs p50 (p99 249.6 µs outlier); 8 KiB H2D write 13.2 µs; 8 KiB D2H read 20.7 µs |
| execution note | Run all's queue stopped after the upload cell (remaining cells marked cancelled, never executed). They were started with "Run cell and below" in the same kernel, unedited, each running exactly once |


## What changed
- `KvStorage` redesigned for device memory (ADR-0005). It is fallible and transfer-explicit
  (`write_slots`, `read_slots_into`, `copy_block`, `synchronize`, `layout`, `residency`), and validates
  arguments before any transfer. Borrowed reads remain only for host storage (`CpuKvStorage::read`).
  `PagedKvCache` gains `write_range` (spans blocks), `read_into`, `synchronize`, and poisoning after fatal
  device errors. `with_storage` now checks the storage shape and returns `Result`. This is a breaking API change.
- New optional `CudaKvStorage` (`--features cuda`; `cudarc` =0.19.10 with `driver`, `dynamic-loading`, `cuda-12000`).
  KV values live in one device allocation; COW copies are device-to-device. The default build has no dependencies.
  `unsafe_code = "forbid"` is kept.
- `tests/gpu.rs` (9 device tests) and a shared generic model-based harness (`tests/common`). `kvx gpu-info` and `kvx gpu-bench`.
- `scripts/package_colab.sh`, `notebooks/kvcachex_colab.ipynb`, `docs/colab.md`, `results/gpu/README.md`.
- CI: negative GPU step (the cuda tests must fail clearly on GPU-less runners) and the packaging step.

## Verification (local: Apple M1, macOS 15.5, rustc 1.99.1, offline)
| check | result |
|---|---|
| `cargo fmt --all -- --check` | pass |
| `cargo clippy --offline --locked --all-targets -- -D warnings` | pass |
| `cargo clippy --offline --locked --all-targets --all-features -- -D warnings` (compiles the CUDA backend) | pass |
| `cargo test --offline --locked --workspace --release` | 7 suites, 63 passed, 0 failed |
| `cargo test --offline --locked -p kvcachex --release --features degenerate-hash` | 6 suites, 52 passed, 0 failed, 1 ignored (by design) |
| simulator invariants (`kvx simulate --workload shared-prefix --requests 100 --check-every 10`) | pass |
| `cargo test -p kvcachex --release --features cuda --test gpu -- --test-threads=1` (no GPU here) | **0 passed, 9 failed, as designed**: each fails with "GPU tests were requested … no usable CUDA device" |
| CI negative-GPU step, run from the workflow YAML | pass (found and fixed a SIGPIPE bug in the step before it ever ran remotely) |
| mutation: skip the COW copy in `PagedKvCache` | caught by the generic model harness (corrupt KV) |
| `scripts/package_colab.sh` | archive verified: no `.git`, `target`, `.cargo-home`, `.claude`, secrets |
| notebook dry run locally from the ZIP (CPU mode) | CPU checks pass (63 / 52 tests); GPU cell raises; run record says `gpu_validated: false` |
| `results/REPORT.md` | not regenerated (simulator/bench code unchanged); audit script still 0 mismatches |

## Colab GPU run (2026-11-15 UTC; details and provenance in `results/gpu/README.md`)
| item | value |
|---|---|
| runtime | Colab T4 GPU, Python 3.13.16, Linux 6.6.159+ x86_64 |
| device | Tesla T4, cc 7.5, 15360 MiB; driver 580.82.07 (CUDA 13.0); nvcc 13.0.88 |
| source | uploaded `kvcachex-src.zip`, sha256 `031d77ee…72be` |
| toolchain | rustc 1.99.1, cargo 1.99.1 (from `rust-toolchain.toml`) |
| CPU checks on Colab | fmt, clippy `--all-features -D warnings`, 63/0/0 workspace tests, 52/0/1 collision tests, simulator invariants: all exit 0 |
| `cargo test --locked -p kvcachex --release --features cuda --test gpu -- --test-threads=1 --nocapture` | **9 passed, 0 failed, 0 ignored** (run twice) |
| `kvx gpu-info`, `kvx gpu-bench --json results/gpu/gpu_bench.json` | exit 0; 4 MiB D2D COW block copy 49.4 µs p50; 8 KiB H2D write 18.7 µs p50 (storage transfers only) |

Found and fixed during the run (both in the repository notebook, then applied byte-identically in Colab):
the notebook's libcuda detection ignored `LD_LIBRARY_PATH`, so it falsely reported "no GPU" on Colab, and a stale
blocker note was left in the record after a successful GPU pass. The built-in browser could not save the results
download, so the archive was exported through a temporary print cell. Its SHA-256 was verified, and the cell was then deleted.

---

# KVCacheX 0.1.0 release candidate (local, 2026-11-15)

Not committed, tagged or pushed. Validated locally on Apple M1 (4P+4E, 8 GB), macOS 15.5,
rustc 1.99.1 (pinned by `rust-toolchain.toml`), offline with the project-local registry.

## What ran and the results

| check | command | result |
|---|---|---|
| format | `cargo fmt --all -- --check` | pass |
| lint | `cargo clippy --offline --locked --all-targets --all-features -- -D warnings` | pass (0 warnings) |
| workspace tests (release) | `cargo test --offline --locked --workspace --release` | pass: 6 suites, 57 passed, 0 failed |
| forced-collision tests | `cargo test --offline --locked -p kvcachex --release --features degenerate-hash` | pass: 5 suites, 46 passed, 0 failed, 1 ignored (the hash-distinctness unit test, meaningless when the hash is constant) |
| simulator invariants | `cargo run --offline --locked --release -p kvx -- simulate --workload shared-prefix --requests 100 --check-every 10` | pass (exit 0) |
| full reproduction | `CARGO_NET_OFFLINE=true ./scripts/reproduce.sh` (default parameters: 400 requests, seeds 1–3, 3+20 reps) | pass (exit 0, 21.7 s; 103 tests passed, 0 failed); regenerated `results/` |
| report audit | independent script comparing every REPORT.md cell to `results/raw/*.json` | 463 checks, 0 mismatches; 15/15 runs complete, none truncated or rejected |
| reduced-run labelling | `kvx report --requests 40 --seeds 1 --reps 2` into a scratch dir | labelled "REDUCED / NON-DEFAULT RUN", stale raw file removed |

**Remote CI: not run at the time.** The repository had no remote, and nothing was pushed. Every workflow step
was run locally with the same flags (plus `--offline`) on macOS only. The `ubuntu-latest` job has
not been executed anywhere.

## Fixes in this candidate
- **Simulator TTFT after preemption (metric defect).** A preempted request's first-token step was
  reset on re-admission, so its TTFT was overstated (only the paged backends preempt). The step is
  now carried through the waiting queue. Exact regression test `sim::tests::ttft_survives_preemption`
  (fails with 3 steps on the old code, passes with 1).
- **Mean memory waste (metric defect).** It was summed over steps holding memory but divided by all
  steps. It is now averaged over steps holding memory, as documented. Effect on reported numbers is ≤ 0.3 percentage points.
- **Silent truncation.** Runs that hit `max_steps` are now flagged (`truncated` in JSON), and
  `simulate` / `report` fail instead of printing partial results. Test `max_steps_truncation_is_reported`.
- **Report hygiene.** `kvx report` deletes old `raw/*.json` before writing, records its parameters,
  labels non-default runs as reduced, and adds cross-seed mean waste.
- **False MSRV.** `rust-version = "1.80"` was never tested and is impossible: the locked proptest 1.12
  needs Rust 1.88. Removed; the supported toolchain is the pinned 1.99.1. The removal surfaced three
  Clippy lints that the old MSRV had suppressed (`is_multiple_of`, `as_chunks`). They are fixed with no behavior change
  (the hash values are identical).
- **CI portability.** The workflow installed `stable` while `rust-toolchain.toml` overrides it to 1.99.1.
  It now installs the pinned toolchain explicitly, uses `--locked`, caches the registry, sets timeouts,
  and does not stop the other OS when one fails.
- **`env.sh` portability.** Homebrew's PATH and the local `CARGO_HOME` are applied only if those directories exist.
- **Docs.** The README waste figures now use the cross-seed mean (they were seed-1 values under a "mean"
  label). Latency figures now come from this report, with the noise caveat. The design doc no longer claims CI
  passed. The ADR-0004 numbers were updated, and the complexity claim was corrected.

Simulation step ratios reproduce the previous run exactly (deterministic seeds). Wall-clock
latencies differ from earlier runs within the documented about 1.7× noise (for example, prefix-hit
admission p50 6.9 µs vs 5.2 µs).

## Remaining blockers / known gaps
- Remote CI unobserved on both platforms at the time (resolved by the 2026-11-16 publication; see the Actions page).
- No GPU/CUDA on the development machine: no device-memory validation.
- No MSRV is declared or tested.

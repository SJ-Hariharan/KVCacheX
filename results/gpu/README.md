# GPU results

Output of `notebooks/kvcachex_colab.ipynb` (sections 6–7), copied here unmodified:

* `colab_run.json` / `colab_run.md`: environment (GPU, driver, CUDA, rustc), every command, exit codes, test counts;
* `gpu_bench.json`: `kvx gpu-bench` storage-transfer microbenchmark (H2D / D2H / D2D, with CPU reference).

These files are produced only by running the notebook; never write them by hand.

| run | files | status |
|---|---|---|
| **run 2** (current), 2026-11-15 19:18–19:26 UTC | `colab_run.*`, `gpu_bench.json` in this directory | fresh runtime, fixed notebook, no code edits during the run |
| run 1, 2026-11-15 18:41–18:51 UTC | `history/run1-2026-11-15T1841Z/` | superseded; needed two mid-run notebook fixes (see below) |

## Run 2 (current)

* **Runtime:** a fresh Colab T4 runtime. This notebook's previous runtime was deleted first (the runtime type was
  changed CPU → T4 GPU, with "disconnect and delete runtime"). The runtime was deleted again after the run.
* **Environment:** Tesla T4 (compute capability 7.5, 15360 MiB), driver 580.82.07 (driver CUDA 13.0), nvcc 13.0;
  Colab Python 3.13.16, Linux 6.6.159+ x86_64, glibc 2.39; rustc 1.99.1, cargo 1.99.1. `libcuda` was found at
  `/usr/lib64-nvidia/libcuda.so.580.82.07`.
* **Source:** `kvcachex-src-run2.zip` (146294 bytes, 91 entries), sha256
  `8b826ad2103e23f075237dcf55ccc1d2fd854130b0657959ce7c3c29a9c785fe`. The exact archive is kept locally as
  `dist/kvcachex-src-run2.zip` (git-ignored). It was built from this tree before the run; afterwards only docs and
  `results/` changed.
* **Notebook:** all 10 code cells were hash-identical to `notebooks/kvcachex_colab.ipynb` before and after the run.
* **Results** (17 checks, each recorded once, all exit 0):

| check | result |
|---|---|
| fmt, clippy `--all-targets --all-features -D warnings` | pass |
| `cargo test --locked --workspace --release` | 63 passed, 0 failed, 0 ignored (7 suites) |
| `cargo test --locked -p kvcachex --release --features degenerate-hash` | 52 passed, 0 failed, 1 ignored (6 suites; the ignored test is meaningless under a constant hash) |
| simulator invariants | pass |
| `cargo test --locked -p kvcachex --release --features cuda --test gpu -- --test-threads=1 --nocapture` | **9 passed, 0 failed, 0 ignored** (none skipped) |
| `kvx gpu-info`, `kvx gpu-bench --json results/gpu/gpu_bench.json` | exit 0; `gpu_validated: true` |

* **Execution note (not a code change):** Run all started all cells on the fresh runtime. After the upload cell
  completed (the user picked the ZIP), Colab marked the remaining queued cells as cancelled ("error, 0 s", never
  executed, no output). The cause isn't visible; the execution counts suggest the upload cell was started, cancelled
  and re-run once (count 4 is missing, and the upload cell shows count 5). The remaining cells were then started with
  Colab's "Run cell and below" in the same kernel session, without editing any cell. Every cell after the upload ran
  exactly once, in order. The kernel was not restarted: later cells used `sh()` and `RUN` defined by cell 0.
* **Export:** the browser pane cannot save the notebook's download. A temporary cell printed the results archive as
  base64 (5518 bytes, sha256 `10f869487ac59513529e895447e3df6978745bd34b92de61d3bf6dca252b244a`). It was decoded
  locally, the hash verified, and the cell deleted.

### Reading `gpu_bench.json` (run 2)
Storage operations only, each timed to completion (including a stream synchronize), f32 Llama-3-8B layout,
64 blocks (256 MiB) on the device, 20 warm-up + 200 timed reps:

| op | movement | bytes | p50 | p95 | p99 | GB/s @p50 |
|---|---|---:|---:|---:|---:|---:|
| write_1_slot | H2D | 8 KiB | 13.2 µs | 18.4 µs | 29.4 µs | 0.62 |
| write_block_plane | H2D | 128 KiB | 49.5 µs | 57.2 µs | 66.7 µs | 2.65 |
| read_1_slot | D2H + sync | 8 KiB | 20.7 µs | 28.4 µs | 32.9 µs | 0.40 |
| read_block_plane | D2H + sync | 128 KiB | 59.4 µs | 79.7 µs | 99.4 µs | 2.21 |
| copy_block_cow | **D2D** | 4 MiB | 45.4 µs | 50.2 µs | 249.6 µs | 92.39 |

Small transfers are dominated by per-call latency (pageable host memory, synchronize). The D2D p99 includes a rare
outlier about 5× the median. Run-to-run variation is visible: run 1 measured the same operations at 18.7 / 62.4 /
30.1 / 66.8 / 49.4 µs p50. The CPU backend's 4 MiB copy took 753 µs p50 on the same VM. That compares different
memories on a shared VM, not a speedup claim. None of this is attention, TTFT or tokens/s: KVCacheX runs no model.

## Run 1 (superseded; kept for provenance)

Same device, driver and toolchain. Source `kvcachex-src.zip`, sha256 `031d77eec0cc95164831d54bf3f974398f11b2c8d9fcc2e1e0fa17d9783772be`
(88 entries). That archive was not kept; `dist/` was regenerated before anyone thought to save it, and Colab recycled
its VM. File timestamps show that no code, manifest, script, CI or toolchain file has changed since run 1 started;
only the notebook (the two fixes below), docs and results have. Run 1 passed 63 + 52 CPU tests and 9/9 GPU tests
(twice). It is superseded because the notebook was edited during it:
1. The GPU-detection cell reported "no GPU" with a T4 attached: `libcuda.so` is on `LD_LIBRARY_PATH`
   (`/usr/lib64-nvidia`), not in the `ldconfig` cache, the only place the cell looked. The fix was made in the repository
   and applied byte-identically in Colab (checked by SHA-256), and the detection cell was re-run.
2. The first successful GPU pass left a stale "blocker" note in the run record. The GPU cell now resets that state.
   The fix was applied the same way, and the GPU section and record were re-run, so run 1's JSON lists the GPU
   commands twice.
3. Results were exported through a temporary print cell (archive sha256 `1ce379a4…6f3b`) that was then deleted.

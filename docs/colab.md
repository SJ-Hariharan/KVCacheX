# Running KVCacheX on Google Colab (CPU checks + CUDA storage validation)

`notebooks/kvcachex_colab.ipynb` runs on Colab's standard **Python** runtime. Rust commands run
through shell calls; no Rust Jupyter kernel is needed. It does not mount Drive, ask for secrets,
start background workers, or open tunnels.

KVCacheX manages KV-cache memory. It has no attention or model kernels, so the notebook validates
**storage correctness and transfer costs on the GPU**, not inference.

## 1. Package the source (on your machine)

```bash
./scripts/package_colab.sh            # → dist/kvcachex-src.zip and dist/kvcachex-src.zip.sha256
```

The archive takes the files git would track (`.gitignore` drops `target/` and `.cargo-home/`).
It also removes `.git`, `.claude/`, editor folders, `.DS_Store`, `.env*`, keys/credentials and
nested archives. The script then checks the archive and fails if a forbidden path got in. Uploading this ZIP is the primary path: the notebook needs no credentials for it. The optional clone
cell (3b) works only for a repository Colab can read without signing in, and the GitHub repository is
currently private.

## 2. Open the notebook in Colab

1. Go to https://colab.research.google.com → **File → Upload notebook** → choose `notebooks/kvcachex_colab.ipynb`.
   This creates a new notebook and leaves your other notebooks alone.
2. **Runtime → Change runtime type → Hardware accelerator: GPU** (for example T4) → Save.
   If Colab says no GPU is available (usage limits or capacity), keep the CPU runtime. The CPU
   checks still run, and the GPU section records the blocker instead of passing.
3. **Runtime → Run all**, or run the cells top to bottom. In section 3, pick `dist/kvcachex-src.zip`
   in the upload dialog and compare the printed SHA-256 with `dist/kvcachex-src.zip.sha256`.

## 3. What runs

| section | commands | needs GPU |
|---|---|---|
| 1. runtime | `nvidia-smi`, `nvidia-smi --query-gpu=…`, `ldconfig -p \| grep libcuda`, `nvcc --version` | no (reports) |
| 2. toolchain | official `rustup` installer, no default toolchain | no |
| 4. pinned toolchain | `rustup toolchain install` (reads `rust-toolchain.toml`), `rustc/cargo --version` | no |
| 5. CPU checks | `cargo fmt --check`, `cargo clippy --all-targets --all-features -D warnings`, `cargo test --workspace --release`, `cargo test -p kvcachex --release --features degenerate-hash`, simulator invariants | no |
| 6. GPU checks | `cargo test -p kvcachex --release --features cuda --test gpu -- --test-threads=1`, `kvx gpu-info`, `kvx gpu-bench --json results/gpu/gpu_bench.json` | **yes; fails without one** |
| 7. record | writes `results/gpu/colab_run.json` and `colab_run.md` | no |

The GPU section reports success only if every test in `tests/gpu.rs` ran and passed: the number of
tests run must equal the number declared, with 0 failed and 0 ignored.

## 4. Bring results back

Run the last cell to download `kvcachex-colab-results.zip`, then copy its `results/gpu/` into your
checkout. These files are the only evidence for GPU validation; nothing in this repository claims a
GPU pass without them.

## Compatibility notes

* Verified on 2026-11-15 (two runs; run 2 on a fresh runtime with this exact notebook): Colab T4 runtime, driver
  580.82.07 (driver CUDA 13.0), nvcc 13.0, Python 3.13.16 (`results/gpu/`).
* Keep a copy of the exact ZIP you upload (for example `cp dist/kvcachex-src.zip dist/kvcachex-src-<run>.zip`):
  `package_colab.sh` overwrites `dist/kvcachex-src.zip`, and the run record identifies the source only by its SHA-256.
* If Run all stops after the upload cell (queued cells shown as cancelled), focus the next cell and use
  **Runtime → Run cell and below**. The kernel state (`RUN`, `sh`) is kept, and no cell needs editing.
* On Colab, `libcuda.so` lives in `/usr/lib64-nvidia` on `LD_LIBRARY_PATH`, not in the `ldconfig`
  cache. The notebook's detection checks both.
* If the browser blocks the results download, use Colab's Files panel (`/content/kvcachex/results/gpu/`) to download the files.
* `cudarc` 0.19.10 loads `libcuda.so` at run time and binds the CUDA **12.0** driver API. Any driver
  whose `nvidia-smi` reports "CUDA Version" ≥ 12.0 works. The CUDA toolkit (`nvcc`) is not used.
* Rust 1.99.1 is pinned by `rust-toolchain.toml`. The first build on Colab downloads the toolchain and
  crates (a few minutes).

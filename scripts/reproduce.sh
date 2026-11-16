#!/usr/bin/env bash
# Reproduce every result in results/: tests, then the full benchmark report.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f env.sh ] && source ./env.sh
export KVX_RUSTC="$(rustc --version)"
if command -v system_profiler >/dev/null; then
  export KVX_HW="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip|Total Number of Cores|Memory:/{printf "%s; ", $2}')"
else
  export KVX_HW="$(uname -m) $(nproc 2>/dev/null) cpus"
fi
cargo test --workspace --release
cargo test -p kvcachex --release --features degenerate-hash
cargo build --release -p kvx
./target/release/kvx report --out results --requests "${REQUESTS:-400}" --seeds "${SEEDS:-3}" --reps "${REPS:-20}"

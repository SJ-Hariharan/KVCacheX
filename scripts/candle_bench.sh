#!/usr/bin/env bash
# Reproduce results/candle/: the Candle + KVCacheX CPU reference benchmark.
#   ./scripts/fetch_tiny_models.sh      # once (~61 MB)
#   ./scripts/candle_bench.sh           # builds integrations/candle, runs 4 configurations, writes the report
# Env: REQUESTS (16), GEN (48), WARMUP (1), REPS (5), RAYON_NUM_THREADS (4), KVX_HW (hardware label).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f env.sh ] && source ./env.sh
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export KVX_RUSTC="${KVX_RUSTC:-$(rustc --version)}"
if [ -z "${KVX_HW:-}" ]; then
  if command -v system_profiler >/dev/null; then
    KVX_HW="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip|Total Number of Cores|Memory:/{printf "%s; ", $2}')"
  else
    KVX_HW="$(uname -m) $(nproc 2>/dev/null) cpus"
  fi
  export KVX_HW
fi
out=results/candle
mkdir -p "$out/raw"
rm -f "$out"/raw/*.json
(cd integrations/candle && cargo build --locked --release --bin kvx-candle)
bin=integrations/candle/target/release/kvx-candle
for workload in shared-prefix unique; do
  for conc in 1 4; do
    "$bin" bench --workload "$workload" --concurrency "$conc" --requests "${REQUESTS:-16}" --gen "${GEN:-48}" \
      --warmup "${WARMUP:-1}" --reps "${REPS:-5}" --json "$out/raw/${workload}_c${conc}.json"
  done
done
python3 scripts/candle_report.py "$out"

#!/usr/bin/env bash
# Download the tiny public llama2.c TinyStories checkpoints used by integrations/candle
# (about 61 MB in total) and verify their SHA-256 hashes. Files are pinned to exact revisions.
#
#   ./scripts/fetch_tiny_models.sh                # into integrations/candle/models/
#   ./scripts/fetch_tiny_models.sh /some/dir      # elsewhere (then set KVX_MODELS_DIR)
#
# Sources: huggingface.co/karpathy/tinyllamas (MIT, trained by Andrej Karpathy on TinyStories)
# and github.com/karpathy/llama2.c (MIT) for tokenizer.bin.
set -euo pipefail
cd "$(dirname "$0")/.."
dir="${1:-integrations/candle/models}"
mkdir -p "$dir"
hf="https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add"
gh="https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb"

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

fetch() { # name url sha256
  local out="$dir/$1"
  if [ -f "$out" ] && [ "$(sha "$out")" = "$3" ]; then
    echo "ok       $1 (already present)"
    return
  fi
  curl -fL --retry 3 -o "$out.part" "$2"
  local got; got="$(sha "$out.part")"
  if [ "$got" != "$3" ]; then
    rm -f "$out.part"
    echo "error: $1 sha256 $got, expected $3" >&2
    exit 1
  fi
  mv "$out.part" "$out"
  echo "fetched  $1"
}

fetch stories15M.bin  "$hf/stories15M.bin"             cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a
fetch tokenizer.bin   "$gh/tokenizer.bin"              50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361
echo "models in $dir"

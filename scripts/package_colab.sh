#!/usr/bin/env bash
# Build an upload-ready source archive for notebooks/kvcachex_colab.ipynb.
#
#   ./scripts/package_colab.sh            # writes dist/kvcachex-src.zip (+ .sha256)
#   ./scripts/package_colab.sh out.zip    # custom output path
#
# Contents: every file git would track (tracked + untracked, minus .gitignore matches such as
# target/ and .cargo-home/), minus local tooling and anything that looks like a secret.
# The archive is verified after writing; the script fails if a forbidden path got in.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
out="${1:-dist/kvcachex-src.zip}"
case "$out" in /*) ;; *) out="$root/$out" ;; esac
mkdir -p "$(dirname "$out")"
rm -f "$out" "$out.sha256"

# Paths never shipped: VCS data, build output, local cargo cache, editor/agent config,
# OS junk, and secret-like files.
forbidden='(^|/)(\.git|target|\.cargo-home|dist|\.claude|\.vscode|\.idea|__pycache__|\.ipynb_checkpoints)(/|$)|(^|/)\.DS_Store$|(^|/)\.env(\..*)?$|\.(pem|key|p12|pfx|keystore)$|(^|/)(id_rsa|id_ed25519|\.netrc|\.npmrc|credentials(\.json)?)$|\.zip$'

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  files="$(git ls-files --cached --others --exclude-standard)"
else
  files="$(find . -type f | sed 's|^\./||')"
fi
list="$(printf '%s\n' "$files" | grep -Ev "$forbidden" | LC_ALL=C sort)"
[ -n "$list" ] || { echo "error: no files selected" >&2; exit 1; }
for required in Cargo.toml Cargo.lock rust-toolchain.toml crates/kvcachex/src/lib.rs notebooks/kvcachex_colab.ipynb; do
  printf '%s\n' "$list" | grep -qx "$required" || { echo "error: required file missing: $required" >&2; exit 1; }
done

# Store under a top-level kvcachex/ directory.
tmp="$(mktemp -d "${TMPDIR:-/tmp}/kvx-pkg.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/kvcachex"
printf '%s\n' "$list" | while IFS= read -r f; do
  mkdir -p "$tmp/kvcachex/$(dirname "$f")"
  cp -p "$f" "$tmp/kvcachex/$f"
done
(cd "$tmp" && zip -q -r -X "$out" kvcachex)

# Verify: nothing forbidden inside, and the file count matches.
inside="$(unzip -Z1 "$out" | grep -v '/$' | sed 's|^kvcachex/||')"
if printf '%s\n' "$inside" | grep -Eq "$forbidden"; then
  echo "error: forbidden path in archive:" >&2
  printf '%s\n' "$inside" | grep -E "$forbidden" >&2
  exit 1
fi
n_list="$(printf '%s\n' "$list" | wc -l | tr -d ' ')"
n_zip="$(printf '%s\n' "$inside" | wc -l | tr -d ' ')"
[ "$n_list" = "$n_zip" ] || { echo "error: archive has $n_zip files, expected $n_list" >&2; exit 1; }

if command -v sha256sum >/dev/null; then sum="$(sha256sum "$out" | cut -d' ' -f1)"; else sum="$(shasum -a 256 "$out" | cut -d' ' -f1)"; fi
echo "$sum  $(basename "$out")" > "$out.sha256"
echo "wrote $out ($n_zip files, $(du -h "$out" | cut -f1 | tr -d ' '))"
echo "sha256 $sum"
echo "Upload it in the notebook's 'Get the source' cell (Colab: Files > Upload also works)."

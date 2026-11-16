# Source before running cargo locally: `source ./env.sh`
# - Puts Homebrew's rustup first on PATH when it exists (macOS/Homebrew setups).
# - Uses a project-local CARGO_HOME only if `.cargo-home/` exists (pre-seeded registry for offline,
#   sandboxed builds). On a fresh clone, the default CARGO_HOME is kept.
_kvx_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ -d /opt/homebrew/opt/rustup/bin ]; then
  export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
fi
if [ -d "$_kvx_root/.cargo-home" ]; then
  export CARGO_HOME="$_kvx_root/.cargo-home"
fi
unset _kvx_root

# Shared setup for the hooks (POSIX sh). Mirrors env.sh without relying on bash.
if [ -d /opt/homebrew/opt/rustup/bin ]; then
  PATH="/opt/homebrew/opt/rustup/bin:$PATH"; export PATH
fi
if [ -d ./.cargo-home ]; then
  CARGO_HOME="$(pwd)/.cargo-home"; export CARGO_HOME
fi
kvx_require_cargo() {
  if ! command -v cargo >/dev/null 2>&1; then
    echo "git hook: 'cargo' was not found on PATH, so the checks cannot run." >&2
    echo "git hook: install Rust (https://rustup.rs) or bypass once with --no-verify." >&2
    exit 1
  fi
}

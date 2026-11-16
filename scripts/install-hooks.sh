#!/bin/sh
# Opt in to the KVCacheX git hooks (pre-commit: format check; pre-push: clippy + tests).
#   ./scripts/install-hooks.sh            # enable
#   ./scripts/install-hooks.sh --remove   # disable
# Works by pointing this clone's core.hooksPath at .githooks/; nothing is copied.
set -eu
cd "$(git rev-parse --show-toplevel)"
if [ "${1:-}" = "--remove" ]; then
  git config --unset core.hooksPath || true
  echo "hooks disabled (core.hooksPath unset)"
  exit 0
fi
chmod +x .githooks/pre-commit .githooks/pre-push
git config core.hooksPath .githooks
echo "hooks enabled: .githooks/pre-commit (cargo fmt --check), .githooks/pre-push (clippy + tests)"

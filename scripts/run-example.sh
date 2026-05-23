#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="${1:-vit_inference}"

if [ ! -f "$ROOT_DIR/.env.runtime" ]; then
  echo "Missing .env.runtime. Run scripts/setup-runtime.sh first." >&2
  exit 1
fi

source "$ROOT_DIR/.env.runtime"

cd "$ROOT_DIR"
cargo run --no-default-features --features runtime --example "$EXAMPLE"

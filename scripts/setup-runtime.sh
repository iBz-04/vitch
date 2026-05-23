#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VIT_TCH_VENV:-"$ROOT_DIR/.venv"}"
PYTHON_BIN="${PYTHON:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "torch==$TORCH_VERSION"

LIBTORCH_DIR="$("$VENV_DIR/bin/python" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')"

cat > "$ROOT_DIR/.env.runtime" <<EOF
export LIBTORCH="$LIBTORCH_DIR"
export DYLD_LIBRARY_PATH="\${LIBTORCH}/lib:\${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="\${LIBTORCH}/lib:\${LD_LIBRARY_PATH:-}"
EOF

echo "Runtime ready."
echo "Run: source .env.runtime"

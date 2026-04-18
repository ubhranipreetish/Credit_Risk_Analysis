#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Virtual environment not found at .venv/."
  echo "Create it and install deps with:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -r requirements.txt"
  exit 1
fi

cd "$ROOT_DIR"
exec "$VENV_PYTHON" -m streamlit run streamlit_app.py --server.headless true --server.port 8501

#!/usr/bin/env bash
# setup_env.sh
# Cross-platform helper to create a virtual environment and install project deps.
# Works on macOS / Linux / WSL. Use PowerShell script `setup_env.ps1` on Windows
# if you prefer PowerShell.

set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
NO_DEV=false

usage() {
  cat <<EOF
Usage: $0 [--no-dev] [--python /path/to/python]

Options:
  --no-dev           Skip installing dev-requirements.txt
  --python <path>    Use a specific python executable (overrides PYTHON env)
  -h, --help         Show this help message
EOF
}

while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --no-dev) NO_DEV=true; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON"
  exit 1
fi

echo "Creating virtual environment at ${VENV_DIR} using ${PYTHON}"
"$PYTHON" -m venv "${VENV_DIR}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip and installing runtime dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ "$NO_DEV" = false ]; then
  if [ -f dev-requirements.txt ]; then
    echo "Installing dev requirements"
    python -m pip install -r dev-requirements.txt
  else
    echo "dev-requirements.txt not found; skipping dev deps"
  fi
else
  echo "--no-dev set; skipping dev requirements"
fi

echo "Setup complete. Activate the venv with: source ${VENV_DIR}/bin/activate"

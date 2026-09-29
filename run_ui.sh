#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="/home/lovish/.gemini/antigravity/scratch/avido-browser-platform/.venv/bin/python"
fi
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
exec "$VENV_PYTHON" -m swapedev_service.ui_app "$@"

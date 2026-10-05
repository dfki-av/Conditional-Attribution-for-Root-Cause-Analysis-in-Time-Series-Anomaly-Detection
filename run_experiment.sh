#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Install Miniconda or Anaconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda run -n cond_attr python --version >/dev/null 2>&1; then
  echo "cond_attr is missing. Run ./installation.sh first." >&2
  exit 1
fi
conda activate cond_attr

python run_experiment.py "$@"

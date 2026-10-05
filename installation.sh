#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Install Miniconda or Anaconda, then rerun this script." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda run -n cond_attr python --version >/dev/null 2>&1; then
  conda env create -f environment.yml
fi
conda activate cond_attr

python -m pip install -r requirements.txt
python -m pip check
echo "cond_attr is ready. Prepare the SWaT CSVs, then start the experiment:"
echo "  conda run -n cond_attr python -m preprocessing.isolate_swat_attacks"
echo "  ./run_experiment.sh"

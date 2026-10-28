#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python "$ROOT/tests/make_synthetic_dataset.py" --output /tmp/syndromebench-smoke.p --n_syndromes 3 --subjects_per_syndrome 24
python "$ROOT/src/data_processing.py" --input /tmp/syndromebench-smoke.p --output_dir /tmp/syndromebench-smoke-output
python -m pytest -q "$ROOT/tests"
python "$ROOT/scripts/check_artifacts.py"
python "$ROOT/scripts/build_dashboard.py"
echo "Checks completed. Metrics shown in dashboard remain from the provided source archive."

#!/usr/bin/env bash
# Dataset-independent quick functional verification; results are not paper evidence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m pytest -q tests/test_alignment.py tests/test_integration.py
python scripts/smoke_test.py

#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
PYTHONPATH=. python tests/test_service.py
PYTHONPATH=. python -m pytest tests/test_api.py

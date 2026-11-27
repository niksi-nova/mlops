"""Thin wrapper kept for backwards compatibility.

The real code is in src/monitoring/run_drift_demo.py. Preferred command:
    uv run python -m src.monitoring.run_drift_demo --scenario all
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.monitoring.run_drift_demo import main  # noqa: E402

if __name__ == "__main__":
    main()

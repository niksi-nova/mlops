"""Unit tests for src/models/evaluate.py.

Small hand-made arrays only; no real data files are read.
"""

import numpy as np

from src.models.evaluate import bucket_delay, within_n_minutes_pct


# --------------------------------------------------------------------------
# bucket_delay
# --------------------------------------------------------------------------
def test_bucket_delay_edge_values():
    assert bucket_delay(0) == "0-5"
    assert bucket_delay(4.9) == "0-5"
    assert bucket_delay(5) == "5-10"       # left edge of next bucket
    assert bucket_delay(9.9) == "5-10"
    assert bucket_delay(10) == "10-20"
    assert bucket_delay(20) == "20-30"
    assert bucket_delay(30) == "30-60"
    assert bucket_delay(59.9) == "30-60"
    assert bucket_delay(60) == "60+"       # left edge of the open-ended bucket
    assert bucket_delay(1000) == "60+"


# --------------------------------------------------------------------------
# within_n_minutes_pct
# --------------------------------------------------------------------------
def test_within_n_minutes_pct_tiny_example():
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([10.0, 14.0, 16.0, 25.0])
    # errors: 0, 4, 6, 15 -> within 5: [True, True, False, False] = 50%
    assert within_n_minutes_pct(y_true, y_pred, 5) == 50.0
    # within 10: [True, True, True, False] = 75%
    assert within_n_minutes_pct(y_true, y_pred, 10) == 75.0

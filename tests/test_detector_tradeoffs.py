import numpy as np
import pandas as pd

from experiments.run_all import _detector_tradeoffs


def _summary(delays, missed):
    return pd.DataFrame({
        "seed": range(5),
        "strategy": ["drift_triggered"] * 5,
        "drift_type": ["harmful_abrupt"] * 5,
        "intensity": [1.0] * 5,
        "delay_windows": [1] * 5,
        "psi_detection_delay": delays,
        "psi_missed_drift": missed,
        "psi_false_alarms": [0] * 5,
    })


def _psi_row(summary):
    result = _detector_tradeoffs(summary)
    selected = result[result.detector == "psi"]
    assert len(selected) == 1
    return selected.iloc[0]


def test_detector_tradeoffs_counts_all_evaluated_runs_for_missed_rate():
    delays = [4, np.nan, np.nan, 6, np.nan]
    missed = [0, 1, 1, 0, 1]

    row = _psi_row(_summary(delays, missed))

    assert row.runs == 5
    assert row.detected_runs == 2
    assert row.missed_drift_rate == 0.6
    assert row.mean_detection_delay_windows == 5.0


def test_detector_tradeoffs_keeps_detector_that_never_fires():
    row = _psi_row(_summary([np.nan] * 5, [1] * 5))

    assert row.runs == 5
    assert row.detected_runs == 0
    assert row.missed_drift_rate == 1.0
    assert pd.isna(row.mean_detection_delay_windows)

"""Delayed-label ADWIN and empirically calibrated covariate-drift signals."""

import numpy as np
from river import drift
from scipy.stats import ks_2samp


def make_adwin(delta: float = 0.002):
    return drift.ADWIN(delta=delta)


def update_error_detector(detector, value: int | float) -> bool:
    detector.update(float(value))
    return bool(detector.drift_detected)


def population_stability_index(reference, current, bins: int = 10) -> float:
    # bins come from the reference quantiles; eps keeps empty bins from blowing up the log
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    eps = 1e-6
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift_report(reference_frame, current_frame, feature_names):
    report = {}
    for name in [*feature_names, "__score__"]:
        ref = np.asarray(reference_frame[name], dtype=float)
        cur = np.asarray(current_frame[name], dtype=float)
        result = ks_2samp(ref, cur, alternative="two-sided", method="asymp")
        report[name] = {
            "psi": population_stability_index(ref, cur),
            "ks_statistic": float(result.statistic),
            # Kept for inspection only; alarm logic never thresholds this value.
            "ks_pvalue": float(result.pvalue),
        }
    return report


def top_k_mean(values, k: int = 3) -> float:
    # a single max feature is too noisy, so average the top k
    vals = np.asarray(values, dtype=float)
    vals = np.sort(vals[np.isfinite(vals)])[::-1]
    if not len(vals):
        return float("nan")
    return float(np.mean(vals[:max(1, min(int(k), len(vals)))]))


def calibrate_drift_thresholds(no_drift_rows, quantile: float = 0.95):
    """Calibrate upper-tail PSI and KS-statistic cutoffs on held-out null runs."""
    if not 0.5 < quantile < 1:
        raise ValueError("quantile must be between 0.5 and 1.")
    frame = no_drift_rows

    def q(column, value):
        if column not in frame:
            return float("inf")
        values = np.asarray(frame[column], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.quantile(values, value)) if len(values) else float("inf")

    return {
        "feature_psi_top3_threshold": q("top3_mean_psi", quantile),
        "score_psi_threshold": q("score_psi", quantile),
        "feature_ks_stat_threshold": q("max_ks_statistic", quantile),
        "score_ks_stat_threshold": q("score_ks_statistic", quantile),
        "null_quantile": float(quantile),
        "consecutive_windows": 2,
    }


def alarm_evaluation(alarm_windows, drift_start_window, n_windows, is_benign=False):
    """Summarize false alarms, delay, and right-censored post-horizon alarms."""
    alarms = sorted(set(int(x) for x in alarm_windows))
    in_horizon = [x for x in alarms if x < n_windows]
    post_horizon = [x for x in alarms if x >= n_windows]
    if drift_start_window is None:
        return {
            "detection_delay": None,
            "false_alarms": len(in_horizon),
            "missed_drift": False,
            "post_horizon_alarms": len(post_horizon),
        }
    before = [x for x in in_horizon if x < drift_start_window]
    after = [x for x in alarms if x >= drift_start_window]
    if is_benign:
        return {
            "detection_delay": None,
            "false_alarms": len(in_horizon),
            "missed_drift": False,
            "benign_shift_alarms": len([x for x in in_horizon if x >= drift_start_window]),
            "post_horizon_alarms": len(post_horizon),
        }
    first = min(after) if after else None
    return {
        "detection_delay": (first - drift_start_window) if first is not None else None,
        "false_alarms": len(before),
        "missed_drift": first is None,
        "post_horizon_alarms": len(post_horizon),
    }

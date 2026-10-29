import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from molt.data import build_stream_pools, features_and_target, temporal_split
from molt.detectors import alarm_evaluation, calibrate_drift_thresholds, make_adwin, population_stability_index, update_error_detector
from molt.drift_injection import inject_drift
from molt.metrics import threshold_for_fpr
from molt.simulate import available_label_cutoff


def sample_frame(n=1000):
    rng = np.random.default_rng(7)
    time = np.arange(n, dtype=float)
    labels = np.zeros(n, dtype=int)
    fraud_indices = [i for i in [100, 180, 310, 480, 690, 850, 930] if i < n]
    labels[fraud_indices] = 1
    frame = pd.DataFrame({
        "Time": time,
        "Amount": rng.lognormal(2, 1, n),
        "V1": rng.normal(size=n),
        "V2": rng.normal(size=n),
        "Class": labels,
    })
    return frame


def test_temporal_split_is_ordered_and_features_exclude_metadata():
    train, validation, test = temporal_split(sample_frame(), 0.6, 0.2)
    assert train.Time.max() < validation.Time.min()
    assert validation.Time.max() < test.Time.min()
    X, y = features_and_target(train)
    assert list(X.columns) == ["Amount", "V1", "V2"]
    assert "Class" not in X and "Time" not in X
    assert len(X) == len(y)


def test_temporal_split_rejects_invalid_fractions():
    try:
        temporal_split(sample_frame(), 0.7, 0.3)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping train/validation fractions should fail")


def test_temporal_split_keeps_equal_timestamp_rows_in_one_partition():
    frame = sample_frame(1000)
    frame.loc[600, "Time"] = frame.loc[599, "Time"]
    train, validation, test = temporal_split(frame, 0.60, 0.20)
    assert train.Time.max() < validation.Time.min()
    assert validation.Time.max() < test.Time.min()


def test_threshold_for_fpr_respects_requested_validation_fpr():
    y = np.array([0] * 990 + [1] * 10)
    scores = np.r_[np.linspace(0, 0.6, 990), np.linspace(0.65, 1, 10)]
    threshold = threshold_for_fpr(y, scores, max_fpr=0.01)
    observed_fpr = np.mean(scores[:990] >= threshold)
    assert observed_fpr <= 0.01
    assert np.mean(scores[990:] >= threshold) > 0.9


def test_psi_is_near_zero_for_same_distribution_and_increases_after_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=10000)
    same = rng.normal(size=10000)
    shifted = rng.normal(loc=1.5, size=10000)
    assert population_stability_index(reference, same) < 0.05
    assert population_stability_index(reference, shifted) > 0.2


def test_detector_evaluator_reports_delay_and_benign_false_alarms():
    harmful = alarm_evaluation([1, 4, 6], drift_start_window=4, n_windows=10)
    assert harmful["detection_delay"] == 0
    assert harmful["false_alarms"] == 1
    assert not harmful["missed_drift"]
    benign = alarm_evaluation([1, 5, 7], drift_start_window=5, n_windows=10, is_benign=True)
    assert benign["benign_shift_alarms"] == 2
    assert benign["false_alarms"] == 3


def test_stream_pool_keeps_all_fraud_without_duplication():
    frame = sample_frame(1000)
    initial, calibration, stream = build_stream_pools(
        frame, seed=12, initial_fraction=0.2, calibration_fraction=0.1,
        legitimate_keep_fraction=0.25,
    )
    source_stream = frame.iloc[300:]
    assert int(stream.Class.sum()) == int(source_stream.Class.sum())
    assert int(initial.Class.sum()) == int(frame.iloc[:200].Class.sum())
    assert int(calibration.Class.sum()) == int(frame.iloc[200:300].Class.sum())
    assert stream["_source_index"].is_unique
    assert stream.Time.is_monotonic_increasing
    assert len(initial[initial.Class == 0]) < len(frame.iloc[:200][frame.iloc[:200].Class == 0])


def test_stream_pools_do_not_split_tied_timestamps():
    frame = sample_frame(1000)
    frame.loc[200, "Time"] = frame.loc[199, "Time"]
    initial, calibration, stream = build_stream_pools(
        frame, seed=1, initial_fraction=0.2, calibration_fraction=0.1,
        legitimate_keep_fraction=1.0,
    )
    assert initial.Time.max() < calibration.Time.min()
    assert calibration.Time.max() < stream.Time.min()


def test_harmful_and_benign_injections_start_at_recorded_index():
    frame = sample_frame(1000).iloc[500:].copy().reset_index(drop=True)
    reference = sample_frame(400)
    important = ["V1", "V2"]
    changed, start = inject_drift(frame, "harmful_abrupt", 20, 1.0, reference[reference.Class == 0], important)
    assert start == 20
    assert changed.loc[:19, ["V1", "V2"]].equals(frame.loc[:19, ["V1", "V2"]])
    fraud_mask = frame.Class == 1
    assert not changed.loc[fraud_mask, "V1"].equals(frame.loc[fraud_mask, "V1"])
    benign, benign_start = inject_drift(frame, "benign_covariate", 20, 1.0, reference[reference.Class == 0], important)
    assert benign_start == 20
    assert np.allclose(benign.loc[20:, "Amount"], frame.loc[20:, "Amount"] * 1.5)
    assert not benign.loc[20:, "V1"].equals(frame.loc[20:, "V1"])
    assert not benign.loc[20:, "V2"].equals(frame.loc[20:, "V2"])


def test_scoring_window_never_sees_its_own_labels_even_at_zero_delay():
    assert available_label_cutoff(5, 0) == 4
    assert available_label_cutoff(5, 1) == 4
    assert available_label_cutoff(5, 3) == 2
    assert available_label_cutoff(0, 0) == -1


def test_adwin_detects_a_large_step_in_fraud_miss_rate():
    detector = make_adwin(delta=0.05)
    for _ in range(400):
        update_error_detector(detector, 0)
    alarms = [update_error_detector(detector, 1) for _ in range(400)]
    assert any(alarms)


def test_null_calibration_returns_conservative_psi_ks_thresholds():
    frame = pd.DataFrame({
        "top3_mean_psi": np.linspace(0.001, 0.05, 100),
        "score_psi": np.linspace(0.001, 0.04, 100),
        "max_ks_statistic": np.linspace(0.01, 0.2, 100),
        "score_ks_statistic": np.linspace(0.02, 0.3, 100),
    })
    thresholds = calibrate_drift_thresholds(frame, quantile=0.95)
    assert thresholds["feature_psi_top3_threshold"] >= 0.045
    assert thresholds["feature_ks_stat_threshold"] >= 0.19
    assert thresholds["score_ks_stat_threshold"] >= 0.28
    assert thresholds["consecutive_windows"] == 2

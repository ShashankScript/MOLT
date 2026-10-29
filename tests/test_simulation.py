import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from molt.artifacts import require_files
from molt.data import build_stream_pools
from molt.detectors import calibrate_drift_thresholds
from molt.simulate import available_label_cutoff, run_windowed_simulation
import molt.simulate as simulate_module
from experiments.run_all import _paired_block_bootstrap_diff_ci


def fixture_frame(n=2000):
    rng = np.random.default_rng(123)
    labels = (np.arange(n) % 20 == 0).astype(int)
    return pd.DataFrame({
        "Time": np.arange(n, dtype=float),
        "Amount": rng.lognormal(2, 0.5, n),
        "V1": labels * 3.0 + rng.normal(0, 0.7, n),
        "V2": rng.normal(size=n),
        "Class": labels,
    })


def quiet_thresholds():
    return {
        "feature_psi_top3_threshold": float("inf"),
        "score_psi_threshold": float("inf"),
        "feature_ks_stat_threshold": float("inf"),
        "score_ks_stat_threshold": float("inf"),
        "consecutive_windows": 2,
    }


def test_simulation_records_label_arrival_without_same_window_leakage():
    pools = build_stream_pools(fixture_frame(), seed=2, legitimate_keep_fraction=0.25)
    initial, calibration, stream = pools
    rows = run_windowed_simulation(
        initial, calibration, stream, strategy="static", delay_windows=2,
        window_size=60, seed=3, drift_thresholds=quiet_thresholds(),
    )
    assert len(rows) > 3
    for row in rows:
        events = json.loads(row["events_json"])
        cutoff = available_label_cutoff(row["window"], 2)
        for event in events:
            if event["event"] == "labels_released" and event["phase"] == "before_score":
                assert event["source_window"] <= cutoff
                assert event["arrival_window"] == event["source_window"] + 2
    late = json.loads(rows[-1]["post_horizon_events_json"])
    assert all(event["phase"] == "post_horizon" for event in late)


def test_periodic_retraining_uses_recent_recalibration_and_reports_realized_fpr():
    initial, calibration, stream = build_stream_pools(fixture_frame(), seed=5, legitimate_keep_fraction=0.25)
    rows = run_windowed_simulation(
        initial, calibration, stream, strategy="periodic", delay_windows=1,
        periodic_interval=2, window_size=60, seed=6, drift_thresholds=quiet_thresholds(),
    )
    assert any(row["retrain_count"] > 0 for row in rows)
    assert all(0 <= row["realized_fpr"] <= 1 for row in rows)
    assert all("recall" in row and "false_positives" in row for row in rows)
    retrain_events = [e for row in rows for e in json.loads(row["events_json"]) if e["event"] == "retrain"]
    assert retrain_events
    assert all("threshold" in event for event in retrain_events)


@pytest.mark.parametrize("strategy", ["oracle", "online"])
def test_oracle_and_online_use_shared_threshold_recalibration(strategy):
    initial, calibration, stream = build_stream_pools(fixture_frame(), seed=7, legitimate_keep_fraction=0.25)
    original = simulate_module.recalibrate_threshold
    with patch("molt.simulate.recalibrate_threshold", wraps=original) as recalibrate:
        run_windowed_simulation(
            initial, calibration, stream, strategy=strategy, delay_windows=1,
            window_size=60, seed=8, drift_thresholds=quiet_thresholds(),
        )
    assert recalibrate.call_count >= 2


def test_missing_dashboard_artifact_has_actionable_error(tmp_path):
    missing = tmp_path / "window_metrics.parquet"
    with pytest.raises(FileNotFoundError, match="Missing precomputed artifacts"):
        require_files([missing])


def test_held_out_no_drift_psi_run_false_alarm_rate_stays_below_30_percent():
    frame = fixture_frame()
    seeds = list(range(10, 20))
    pools = {seed: build_stream_pools(frame, seed=seed, legitimate_keep_fraction=0.25) for seed in seeds}
    probes = {}
    for seed, (initial, calibration, stream) in pools.items():
        probes[seed] = run_windowed_simulation(
            initial, calibration, stream, strategy="static", drift_type="none",
            delay_windows=1, seed=seed, window_size=60, drift_thresholds=quiet_thresholds(),
        )
    false_alarm_runs = 0
    for seed, (initial, calibration, stream) in pools.items():
        null_rows = [row for other_seed, rows in probes.items() if other_seed != seed for row in rows]
        thresholds = calibrate_drift_thresholds(pd.DataFrame(null_rows), quantile=0.95)
        evaluated = run_windowed_simulation(
            initial, calibration, stream, strategy="static", drift_type="none",
            delay_windows=1, seed=seed, window_size=60, drift_thresholds=thresholds,
        )
        false_alarm_runs += int(any(row["psi_alarm"] for row in evaluated))
    observed_far = false_alarm_runs / len(seeds)
    assert observed_far <= 0.30, f"held-out no-drift PSI run FAR={observed_far:.2%} exceeded 30%"


def test_psi_ks_are_diagnostic_and_do_not_trigger_adaptation():
    initial, calibration, stream = build_stream_pools(fixture_frame(), seed=9, legitimate_keep_fraction=0.25)
    stream.loc[stream.index >= 2 * 100, "V2"] += 8.0
    thresholds = {
        "feature_psi_top3_threshold": -1.0,
        "score_psi_threshold": -1.0,
        "feature_ks_stat_threshold": -1.0,
        "score_ks_stat_threshold": -1.0,
        "consecutive_windows": 1,
    }
    with patch("molt.simulate.update_error_detector", return_value=False):
        rows = run_windowed_simulation(
            initial, calibration, stream, strategy="drift_triggered", drift_type="none",
            delay_windows=1, seed=9, window_size=100, drift_thresholds=thresholds,
            row_error_baseline=False,
        )
    events = [event for row in rows for event in json.loads(row["events_json"])]
    assert any(row["psi_alarm"] for row in rows)
    assert any(row["ks_alarm"] for row in rows)
    assert not any(row["warning"] for row in rows)
    assert not any(event["event"] in {"confirmation", "promotion"} for event in events)


def test_fraud_miss_adwin_drives_matched_fpr_challenger_promotion():
    class Challenger:
        pass

    initial, calibration, stream = build_stream_pools(
        fixture_frame(2500), seed=10, legitimate_keep_fraction=0.25,
    )

    def fake_score(model, frame, feature_names, online):
        fraud = frame["Class"].to_numpy(dtype=bool)
        probability = 0.9 if isinstance(model, Challenger) else 0.1
        return np.where(fraud, probability, 0.0)

    thresholds = {
        "feature_psi_top3_threshold": float("inf"),
        "score_psi_threshold": float("inf"),
        "feature_ks_stat_threshold": float("inf"),
        "score_ks_stat_threshold": float("inf"),
        "consecutive_windows": 2,
    }
    with (
        patch("molt.simulate.update_error_detector", return_value=True),
        patch("molt.simulate._score", side_effect=fake_score),
        patch("molt.simulate.recalibrate_threshold", return_value=(0.5, 0.0)),
        patch("molt.simulate._fit_recalibrated", return_value=(Challenger(), 0.5, 0.0)),
    ):
        rows = run_windowed_simulation(
            initial, calibration, stream, strategy="drift_triggered", drift_type="none",
            intensity=0, delay_windows=1, seed=10, window_size=100,
            drift_thresholds=thresholds, max_fpr=0.01, row_error_baseline=False,
        )

    events = [event for row in rows for event in json.loads(row["events_json"])]
    promotions = [event for event in events if event["event"] == "promotion"]
    assert promotions
    gate = promotions[0]
    assert gate["gate_passed"] and gate["promotion_applied"]
    assert gate["target_fpr"] == 0.01
    assert gate["calibration_fpr_ok"]
    assert gate["candidate_calibration_fpr"] <= 0.01
    assert gate["champion_calibration_fpr"] <= 0.01
    assert gate["holdout_fpr_is_diagnostic_only"]
    assert gate["candidate_recall"] > gate["champion_recall"]
    assert any(row["retrain_count"] > 0 for row in rows)


def test_paired_block_bootstrap_resamples_seed_clusters_and_preserves_effect_sign():
    rows = []
    for seed in [1, 2, 3, 4, 5]:
        for window in range(8):
            rows.extend([
                {"seed": seed, "window": window, "strategy": "drift_triggered",
                 "drift_type": "harmful_abrupt", "intensity": 1.0,
                 "delay_windows": 1, "recall": 0.6},
                {"seed": seed, "window": window, "strategy": "periodic",
                 "drift_type": "harmful_abrupt", "intensity": 1.0,
                 "delay_windows": 1, "recall": 0.5},
            ])
    result = _paired_block_bootstrap_diff_ci(
        pd.DataFrame(rows), "harmful_abrupt", 1.0, 1,
        "drift_triggered", "periodic", "recall", reps=200,
    )
    assert np.isclose(result["mean_difference"], 0.1)
    assert np.isclose(result["ci_low"], 0.1)
    assert np.isclose(result["ci_high"], 0.1)
    assert result["seed_clusters"] == 5

"""Windowed replay with calibrated drift warnings, delayed labels, and event logs."""

from collections import deque
import json

import numpy as np
import pandas as pd

from .data import TARGET, TIME, features_and_target, split_windows
from .detectors import alarm_evaluation, feature_drift_report, make_adwin, top_k_mean, update_error_detector
from .drift_injection import important_features, inject_drift
from .metrics import recall_at_fpr, threshold_for_fpr, window_metrics
from .strategies import make_batch_model, make_online_model

INITIAL_TRAIN_CAP = 12_000


def _river_features(row, feature_names):
    return {name: float(row[name]) for name in feature_names}


def _score(model, frame, feature_names, online):
    if online:
        return np.asarray([
            float(model.predict_proba_one(_river_features(row, feature_names)).get(1, 0.0))
            for _, row in frame.iterrows()
        ], dtype=float)
    X, _ = features_and_target(frame)
    return model.predict_proba(X)[:, 1]


def recalibrate_threshold(model, calibration_frame, feature_names, max_fpr, online=False):
    # every retraining strategy recalibrates the same way
    y = calibration_frame[TARGET].to_numpy(dtype=int)
    scores = _score(model, calibration_frame, feature_names, online=online)
    threshold = threshold_for_fpr(y, scores, max_fpr)
    negatives = y == 0
    observed_fpr = float(np.mean(scores[negatives] >= threshold)) if negatives.any() else float("nan")
    return threshold, observed_fpr


def _fit_recalibrated(initial, training_frames, calibration_frame, seed, feature_names, max_fpr):
    frames = [initial, *training_frames]
    training = pd.concat(frames, ignore_index=True)
    X, y = features_and_target(training)
    model = make_batch_model(seed)
    model.fit(X, y)
    threshold, calibration_fpr = recalibrate_threshold(
        model, calibration_frame, feature_names, max_fpr, online=False
    )
    return model, threshold, calibration_fpr


def _recent_training_frames(available, max_windows=6):
    # newest labeled window is held back for calibration
    return [frame for _, frame in available[-(max_windows + 1):-1]]


def _thresholded_rates(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(scores, dtype=float) >= float(threshold)
    positive = y == 1
    negative = y == 0
    return {
        "recall": float(np.mean(pred[positive])) if positive.any() else float("nan"),
        "fpr": float(np.mean(pred[negative])) if negative.any() else float("nan"),
        "misses": int(np.sum(~pred[positive])),
        "frauds": int(positive.sum()),
    }


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def available_label_cutoff(scoring_window: int, delay_windows: int) -> int:
    """Latest label-source window available before scoring this window.

    Even with zero delay, the current window must be scored before its own
    labels can be revealed; therefore the cutoff is at most t-1.
    """
    return int(scoring_window) - max(1, int(delay_windows))


def _pick_drift_start(seed, n_windows):
    if n_windows < 5:
        return max(1, n_windows // 2)
    low = max(1, n_windows // 4)
    high = max(low, min(n_windows - 3, n_windows // 2))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 8831]))
    return int(rng.integers(low, high + 1))


def run_windowed_simulation(
    initial_train: pd.DataFrame,
    calibration: pd.DataFrame,
    stream: pd.DataFrame,
    *,
    strategy: str = "static",
    drift_type: str = "none",
    intensity: float = 1.0,
    delay_windows: int = 1,
    seed: int = 42,
    window_size: int = 4000,
    max_fpr: float = 0.01,
    review_fee: float = 5.0,
    periodic_interval: int = 3,
    drift_thresholds: dict | None = None,
    adwin_delta: float = 0.002,
    row_error_baseline: bool = True,
) -> list[dict]:
    """Run one strategy. A window is scored before its own labels are released."""
    valid = {"static", "periodic", "drift_triggered", "online", "oracle"}
    if strategy not in valid:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {sorted(valid)}")
    if delay_windows < 0:
        raise ValueError("delay_windows cannot be negative")

    drop_internal = ["_stream_index", "_source_index"]
    initial = initial_train.drop(columns=drop_internal, errors="ignore").copy()
    calibration = calibration.drop(columns=drop_internal, errors="ignore").copy()
    initial = initial.tail(INITIAL_TRAIN_CAP).reset_index(drop=True)
    initial_X, initial_y = features_and_target(initial)
    feature_names = list(initial_X.columns)
    model = make_batch_model(seed)
    model.fit(initial_X, initial_y)
    threshold, initial_cal_fpr = recalibrate_threshold(
        model, calibration, feature_names, max_fpr, online=False
    )
    important = important_features(model, feature_names)
    reference_legit = initial.loc[initial[TARGET] == 0]

    n_windows = max(1, int(np.ceil(len(stream) / window_size)))
    true_drift_window = _pick_drift_start(seed, n_windows) if drift_type != "none" else None
    drift_row = min(true_drift_window * window_size, max(0, len(stream) - 1)) if true_drift_window is not None else 0
    altered_stream, true_drift_index = inject_drift(
        stream, drift_type, drift_row, intensity, reference_legit, important,
        ramp_rows=window_size * 2,
    )
    windows = split_windows(altered_stream, window_size)
    true_drift_window = drift_row // window_size if true_drift_index is not None else None

    online = strategy == "online"
    learner = None
    if online:
        learner = make_online_model()
        for _, row in initial.iterrows():
            learner.learn_one(_river_features(row, feature_names), int(row[TARGET]))
        model = learner
        threshold, initial_cal_fpr = recalibrate_threshold(
            model, calibration, feature_names, max_fpr, online=True
        )

    # The feature reference and score cohort stay fixed to the designated
    # calibration pool. Score values are recomputed under the currently active
    # model before each window; stream data never becomes reference data.
    reference_X, _ = features_and_target(calibration)

    thresholds = drift_thresholds or {
        "feature_psi_top3_threshold": float("inf"),
        "score_psi_threshold": float("inf"),
        "feature_ks_stat_threshold": float("inf"),
        "score_ks_stat_threshold": float("inf"),
        "consecutive_windows": 2,
    }
    consecutive_required = int(thresholds.get("consecutive_windows", 2))
    psi_run = 0
    ks_run = 0
    row_error_detector = make_adwin(adwin_delta)
    fraud_miss_detector = make_adwin(adwin_delta)
    pending = deque()
    available: list[tuple[int, pd.DataFrame]] = []
    online_learned: set[int] = set()
    records = []
    alarm_windows = {"adwin": [], "row_error_adwin": [], "psi": [], "ks": []}
    retrain_count = 0
    events_by_window: dict[int, list[dict]] = {}
    last_challenger_attempt = -10_000
    challenger_attempt_count = 0
    used_gate_holdout_windows: set[int] = set()
    recent_fpr = initial_cal_fpr

    def add_event(window_id, event_type, **details):
        events_by_window.setdefault(int(window_id), []).append({
            "window": int(window_id), "event": event_type, **details
        })

    def release(item, arrival_window, phase):
        source_window, labeled_frame, saved_scores, applied_threshold = item
        clean = labeled_frame.drop(columns=drop_internal, errors="ignore").copy()
        available.append((source_window, clean))
        add_event(arrival_window, "labels_released", source_window=source_window,
                  arrival_window=int(arrival_window), phase=phase,
                  delay_windows=int(arrival_window - source_window))
        actual = clean[TARGET].to_numpy(dtype=int)
        predicted = (saved_scores >= applied_threshold).astype(int)
        row_error_alarm = False
        fraud_miss_alarm = False
        for pred, truth in zip(predicted, actual):
            if row_error_baseline:
                row_error_alarm = update_error_detector(row_error_detector, int(pred != truth)) or row_error_alarm
            if truth == 1:
                fraud_miss_alarm = update_error_detector(fraud_miss_detector, int(pred == 0)) or fraud_miss_alarm
        if row_error_alarm:
            alarm_windows["row_error_adwin"].append(arrival_window)
            add_event(arrival_window, "row_error_adwin_alarm", source_window=source_window)
        if fraud_miss_alarm:
            alarm_windows["adwin"].append(arrival_window)
            add_event(arrival_window, "fraud_miss_adwin_alarm", source_window=source_window)

    for window_index, original_window in enumerate(windows):
        effective_delay = 0 if strategy == "oracle" else delay_windows
        cutoff = available_label_cutoff(window_index, effective_delay)
        while pending and pending[0][0] <= cutoff:
            source = pending[0][0]
            release(pending.popleft(), source + effective_delay, phase="before_score")

        # Online policy learns only on labeled windows older than its newest
        # calibration holdout; the held-out window is not used for fitting.
        if online and len(available) >= 2:
            for source_window, labeled in available[:-1]:
                if source_window not in online_learned:
                    for _, row in labeled.iterrows():
                        learner.learn_one(_river_features(row, feature_names), int(row[TARGET]))
                    online_learned.add(source_window)
            threshold, recent_fpr = recalibrate_threshold(
                learner, available[-1][1], feature_names, max_fpr, online=True
            )

        # Periodic and zero-delay reference retraining share the same fit and
        # latest-labeled-window recalibration function.
        if strategy in {"periodic", "oracle"} and window_index > 0 and available:
            should_retrain = strategy == "oracle" or window_index % periodic_interval == 0
            if should_retrain:
                calibration_frame = available[-1][1]
                train_frames = _recent_training_frames(available)
                model, threshold, recent_fpr = _fit_recalibrated(
                    initial, train_frames, calibration_frame, seed + window_index,
                    feature_names, max_fpr,
                )
                retrain_count += 1
                add_event(window_index, "retrain", strategy=strategy, train_windows=len(train_frames), threshold=float(threshold))

        retrains_used_for_this_window = retrain_count
        window = original_window.copy()
        y = window[TARGET].to_numpy(dtype=int)
        applied_threshold = float(threshold)
        scores = _score(model, window, feature_names, online)

        current_X, _ = features_and_target(window)
        current_data = current_X.copy()
        current_data["__score__"] = scores
        reference_scores = _score(model, calibration, feature_names, online)
        drift_reference = reference_X.copy()
        drift_reference["__score__"] = reference_scores
        drift_report = feature_drift_report(drift_reference, current_data, feature_names)
        feature_psis = [drift_report[name]["psi"] for name in feature_names]
        feature_pvalues = [drift_report[name]["ks_pvalue"] for name in feature_names]
        feature_ks_statistics = [drift_report[name]["ks_statistic"] for name in feature_names]
        max_psi = float(np.nanmax(feature_psis)) if feature_psis else float("nan")
        top3_mean_psi = top_k_mean(feature_psis, k=3)
        min_ks_p = float(np.nanmin(feature_pvalues)) if feature_pvalues else float("nan")
        max_ks_statistic = float(np.nanmax(feature_ks_statistics)) if feature_ks_statistics else float("nan")
        score_psi = float(drift_report["__score__"]["psi"])
        score_ks_p = float(drift_report["__score__"]["ks_pvalue"])
        score_ks_statistic = float(drift_report["__score__"]["ks_statistic"])
        psi_signal = (
            top3_mean_psi >= float(thresholds.get("feature_psi_top3_threshold", float("inf")))
            or score_psi >= float(thresholds.get("score_psi_threshold", float("inf")))
        )
        ks_signal = (
            max_ks_statistic >= float(thresholds.get("feature_ks_stat_threshold", float("inf")))
            or score_ks_statistic >= float(thresholds.get("score_ks_stat_threshold", float("inf")))
        )
        previous_psi_run, previous_ks_run = psi_run, ks_run
        psi_run = psi_run + 1 if psi_signal else 0
        ks_run = ks_run + 1 if ks_signal else 0
        psi_active = psi_run >= consecutive_required
        ks_active = ks_run >= consecutive_required
        psi_alarm = bool(psi_signal and previous_psi_run < consecutive_required and psi_active)
        ks_alarm = bool(ks_signal and previous_ks_run < consecutive_required and ks_active)
        if psi_alarm:
            alarm_windows["psi"].append(window_index)
        if ks_alarm:
            alarm_windows["ks"].append(window_index)
        # PSI/KS remain calibrated diagnostic alarms. The controller acts only
        # on a delayed-label fraud-miss ADWIN alarm, which is less prone to
        # reacting to model-specific score-scale changes.
        fraud_miss_adwin_alarm = any(
            event["event"] == "fraud_miss_adwin_alarm"
            for event in events_by_window.get(window_index, [])
        )
        warning = bool(fraud_miss_adwin_alarm)

        if strategy == "drift_triggered" and warning and window_index - last_challenger_attempt >= 2:
            last_challenger_attempt = window_index
            add_event(window_index, "warning", controller_signal="fraud_miss_adwin",
                      psi_active=bool(psi_active), ks_active=bool(ks_active),
                      max_psi=max_psi, top3_mean_psi=top3_mean_psi,
                      score_psi=score_psi, max_ks_statistic=max_ks_statistic,
                      score_ks_statistic=score_ks_statistic)
            if len(available) >= 3:
                train_frames = _recent_training_frames(available[:-1], max_windows=6)
                # Calibration and holdout are distinct; each holdout window is
                # consumed at most once by this sequential promotion policy.
                calibration_frame = available[-2][1]
                holdout_frame = available[-1][1]
                calibration_source_window = int(available[-2][0])
                holdout_source_window = int(available[-1][0])
                if holdout_source_window in used_gate_holdout_windows:
                    raise RuntimeError("Challenger gate attempted to reuse a holdout window.")
                used_gate_holdout_windows.add(holdout_source_window)
                challenger_attempt_count += 1
                try:
                    challenger, candidate_threshold, candidate_cal_fpr = _fit_recalibrated(
                        initial, train_frames, calibration_frame, seed + 10_000 + window_index,
                        feature_names, max_fpr,
                    )
                    champion_gate_threshold, champion_cal_fpr = recalibrate_threshold(
                        model, calibration_frame, feature_names, max_fpr, online=False
                    )
                    holdout_X, holdout_y = features_and_target(holdout_frame)
                    champion_scores = _score(model, holdout_frame, feature_names, online=False)
                    candidate_scores = _score(challenger, holdout_frame, feature_names, online=False)
                    champion_rates = _thresholded_rates(holdout_y, champion_scores, champion_gate_threshold)
                    candidate_rates = _thresholded_rates(holdout_y, candidate_scores, candidate_threshold)
                    champion_fpr = champion_rates["fpr"]
                    candidate_fpr = candidate_rates["fpr"]
                    enough_frauds = candidate_rates["frauds"] >= 3
                    calibration_fpr_ok = (
                        np.isfinite(candidate_cal_fpr) and np.isfinite(champion_cal_fpr)
                        and candidate_cal_fpr <= max_fpr + 1e-12
                        and champion_cal_fpr <= max_fpr + 1e-12
                    )
                    improves = candidate_rates["misses"] < champion_rates["misses"]
                    gate_passed = bool(enough_frauds and calibration_fpr_ok and improves)
                    can_apply = window_index < len(windows) - 1
                    promoted = bool(gate_passed and can_apply)
                    gate = {
                        "gate_attempt_index": int(challenger_attempt_count),
                        "calibration_window": calibration_source_window,
                        "holdout_window": holdout_source_window,
                        "holdout_window_used_once": True,
                        "target_fpr": float(max_fpr),
                        "holdout_frauds": int(candidate_rates["frauds"]),
                        "champion_recall": _finite_or_none(champion_rates["recall"]),
                        "candidate_recall": _finite_or_none(candidate_rates["recall"]),
                        "champion_fpr": _finite_or_none(champion_fpr),
                        "candidate_fpr": _finite_or_none(candidate_fpr),
                        "champion_misses": champion_rates["misses"],
                        "candidate_misses": candidate_rates["misses"],
                        "champion_calibration_fpr": _finite_or_none(champion_cal_fpr),
                        "candidate_calibration_fpr": _finite_or_none(candidate_cal_fpr),
                        "calibration_fpr_ok": bool(calibration_fpr_ok),
                        "holdout_fpr_is_diagnostic_only": True,
                        "gate_passed": gate_passed,
                        "promotion_applied": promoted,
                        "promoted": promoted,
                    }
                    add_event(window_index, "confirmation", **gate)
                    if promoted:
                        add_event(window_index, "promotion", **gate)
                    else:
                        reason = "no scored windows remain" if gate_passed and not can_apply else "challenger did not beat the champion gate"
                        add_event(window_index, "rejection", reason=reason, **gate)
                    if promoted:
                        model, threshold = challenger, candidate_threshold
                        retrain_count += 1
                except (ValueError, IndexError) as exc:
                    add_event(window_index, "rejection", reason=f"challenger_fit_failed: {type(exc).__name__}",
                              gate_attempt_index=int(challenger_attempt_count),
                              calibration_window=calibration_source_window,
                              holdout_window=holdout_source_window)
            else:
                add_event(window_index, "rejection", reason="fewer than three released labeled windows")

        metrics = window_metrics(
            y, scores, applied_threshold, window["Amount"].to_numpy(dtype=float), max_fpr, review_fee
        )
        n_legitimate = int(np.sum(y == 0))
        realized_fpr = metrics["false_positives"] / n_legitimate if n_legitimate else float("nan")
        records.append({
            "seed": int(seed), "strategy": strategy, "drift_type": drift_type,
            "intensity": float(intensity), "delay_windows": int(delay_windows),
            "window": int(window_index), "window_start": int(window_index * window_size),
            "time_start": float(window[TIME].min()), "time_end": float(window[TIME].max()),
            "true_drift_window": true_drift_window,
            "recall_at_fpr": metrics["recall_at_fpr"], "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"], "recall": metrics["recall"],
            "realized_fpr": float(realized_fpr), "cost": metrics["cost"],
            "false_positives": metrics["false_positives"], "false_negatives": metrics["false_negatives"],
            "true_positives": metrics["true_positives"], "alerts": metrics["alerts"],
            "max_psi": max_psi, "top3_mean_psi": top3_mean_psi, "score_psi": score_psi,
            "max_ks_statistic": max_ks_statistic, "score_ks_statistic": score_ks_statistic,
            "min_ks_pvalue": min_ks_p, "score_ks_pvalue": score_ks_p,
            "psi_alarm": bool(psi_alarm), "ks_alarm": bool(ks_alarm),
            "feature_psi_json": json.dumps({name: drift_report[name]["psi"] for name in feature_names}, allow_nan=False),
            "feature_ks_stat_json": json.dumps({name: drift_report[name]["ks_statistic"] for name in feature_names}, allow_nan=False),
            "score_reference_source": "fixed_calibration_cohort_rescored_by_active_model",
            "adwin_alarm": False, "adwin_fraud_miss_alarm": False, "adwin_row_error_alarm": False,
            "row_error_adwin_enabled": bool(row_error_baseline),
            "warning": bool(warning), "controller_signal": "fraud_miss_adwin" if warning else None,
            "psi_warning_active": bool(psi_active),
            "ks_warning_active": bool(ks_active), "retrain_count": retrains_used_for_this_window,
            "threshold": applied_threshold, "calibration_realized_fpr": float(recent_fpr),
            "n_rows": len(window), "n_frauds": int(y.sum()), "n_legitimate": n_legitimate,
            "events_json": "[]",
        })
        pending.append((window_index, window, scores, applied_threshold))
        if effective_delay == 0:
            source = pending[0][0]
            release(pending.popleft(), source, phase="after_score")
        # Populate events and delayed ADWIN alarm flags by arrival window.
        record = records[-1]
        arrived_events = events_by_window.get(window_index, [])
        record["events_json"] = json.dumps(arrived_events, allow_nan=False)
        record["adwin_fraud_miss_alarm"] = any(e["event"] == "fraud_miss_adwin_alarm" for e in arrived_events)
        record["adwin_row_error_alarm"] = any(e["event"] == "row_error_adwin_alarm" for e in arrived_events)
        record["adwin_alarm"] = record["adwin_fraud_miss_alarm"]

    # Drain at each label's actual arrival index; keep post-horizon detections
    # separate instead of clamping all pending alarms to the last scored window.
    while pending:
        source = pending[0][0]
        effective_delay = 0 if strategy == "oracle" else delay_windows
        arrival = source + max(0, effective_delay)
        release(pending.popleft(), arrival, phase="post_horizon")

    summaries = {}
    for name, alarms in alarm_windows.items():
        if name == "row_error_adwin" and not row_error_baseline:
            for key in ["detection_delay", "false_alarms", "missed_drift", "post_horizon_alarms"]:
                summaries[f"{name}_{key}"] = None
            continue
        evaluated = alarm_evaluation(
            alarms, true_drift_window, len(windows), is_benign=(drift_type == "benign_covariate")
        )
        for key, value in evaluated.items():
            summaries[f"{name}_{key}"] = value
    for record in records:
        record.update(summaries)
        record["events_json"] = json.dumps(events_by_window.get(record["window"], []), allow_nan=False)
        record["adwin_fraud_miss_alarm"] = any(
            e["event"] == "fraud_miss_adwin_alarm" for e in events_by_window.get(record["window"], [])
        )
        record["adwin_row_error_alarm"] = any(
            e["event"] == "row_error_adwin_alarm" for e in events_by_window.get(record["window"], [])
        )
        record["adwin_alarm"] = record["adwin_fraud_miss_alarm"]
        record["post_horizon_events_json"] = json.dumps(
            [event for idx, events in events_by_window.items() if idx >= len(windows) for event in events],
            allow_nan=False,
        )
    return records

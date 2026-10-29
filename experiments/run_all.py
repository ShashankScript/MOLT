"""Run calibrated semi-synthetic drift experiments and dashboard artifacts."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molt.data import DEFAULT_DATA_PATH, build_stream_pools, load_transactions
from molt.detectors import calibrate_drift_thresholds
from molt.simulate import run_windowed_simulation
from molt.strategies import STRATEGIES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
APP_DATA = ROOT / "app_data"
SEEDS = [17, 42, 101, 2026, 31415]
WINDOW_SIZE = 4000
ADWIN_DELTA = 0.002


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def scenario_grid():
    return [
        ("none", 0.0, 1),
        ("benign_covariate", 1.0, 1),
        ("harmful_abrupt", 1.0, 1),
        ("harmful_abrupt", 1.0, 3),
        ("harmful_gradual", 1.0, 1),
    ]


def _write_sample(df, seed=42):
    APP_DATA.mkdir(parents=True, exist_ok=True)
    fraud = df[df["Class"] == 1]
    legitimate = df[df["Class"] == 0].sample(
        n=min(5000, int((df["Class"] == 0).sum())), random_state=seed
    )
    sample = pd.concat([fraud, legitimate]).sort_values("Time", kind="mergesort").reset_index(drop=True)
    for column in sample.columns:
        sample[column] = sample[column].astype("int8" if column == "Class" else "float32")
    sample.to_parquet(APP_DATA / "creditcard_sample.parquet", index=False, compression="zstd")
    return len(sample)


def _run_level_rows(frame):
    group_cols = ["seed", "strategy", "drift_type", "intensity", "delay_windows"]
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        first = group.iloc[0]
        row = {
            **dict(zip(group_cols, key)),
            "n_windows": len(group),
            "n_frauds_total": int(group.n_frauds.sum()),
            "mean_recall_at_fpr": float(group.recall_at_fpr.mean()),
            "mean_threshold_recall": float(group.recall.mean()),
            "mean_realized_fpr": float(group.realized_fpr.mean()),
            "mean_pr_auc": float(group.pr_auc.mean()),
            "total_cost": float(group.cost.sum()),
            "total_false_positives": int(group.false_positives.sum()),
            "total_missed_frauds": int(group.false_negatives.sum()),
            "total_alerts": int(group.alerts.sum()),
            "retrain_count": int(group.retrain_count.max()),
            "row_error_adwin_enabled": bool(first.row_error_adwin_enabled),
            "true_drift_window": None if pd.isna(first.true_drift_window) else int(first.true_drift_window),
        }
        for detector in ["adwin", "row_error_adwin", "psi", "ks"]:
            for metric in ["detection_delay", "false_alarms", "missed_drift", "post_horizon_alarms"]:
                key_name = f"{detector}_{metric}"
                if key_name in group:
                    row[key_name] = first[key_name]
        static = frame[
            (frame.seed == key[0]) & (frame.strategy == "static")
            & (frame.drift_type == key[2]) & np.isclose(frame.intensity, key[3])
            & (frame.delay_windows == key[4])
        ]
        row["amount_units_saved_vs_static"] = float(static.cost.sum() - group.cost.sum()) if len(static) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(run_summary):
    group_cols = ["strategy", "drift_type", "intensity", "delay_windows"]
    numeric = ["mean_recall_at_fpr", "mean_threshold_recall", "mean_realized_fpr", "mean_pr_auc",
               "total_cost", "total_false_positives", "total_missed_frauds", "retrain_count",
               "amount_units_saved_vs_static", "adwin_detection_delay", "adwin_false_alarms",
               "row_error_adwin_detection_delay", "row_error_adwin_false_alarms",
               "psi_detection_delay", "psi_false_alarms", "ks_detection_delay", "ks_false_alarms"]
    rows = []
    for key, group in run_summary.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        for column in numeric:
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else None
                row[f"{column}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else None
        for detector in ["adwin", "row_error_adwin", "psi", "ks"]:
            col = f"{detector}_missed_drift"
            if col in group:
                valid = group[col].dropna()
                row[f"{detector}_missed_rate"] = float(valid.astype(bool).mean()) if len(valid) else None
        row["seeds"] = int(group.seed.nunique())
        rows.append(row)
    return rows


def _bootstrap_block_ci(frame, scenario, intensity, delay, strategy, metric, reps=1200, block_len=2, seed=804):
    """Moving-block bootstrap over contiguous windows; descriptive, not population inference."""
    subset = frame[(frame.drift_type == scenario) & np.isclose(frame.intensity, intensity)
                   & (frame.delay_windows == delay) & (frame.strategy == strategy)]
    rng = np.random.default_rng(seed)
    means = []
    for _, run in subset.groupby("seed"):
        values = run.sort_values("window")[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        blocks = [values[i:i + block_len] for i in range(max(1, len(values) - block_len + 1))]
        for _ in range(reps):
            draw = []
            while len(draw) < len(values):
                block = blocks[int(rng.integers(0, len(blocks)))]
                draw.extend(block.tolist())
            means.append(float(np.mean(draw[:len(values)])))
    if not means:
        return {"ci_low": None, "ci_high": None, "bootstrap_reps": reps, "block_len": block_len}
    return {"ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975)),
            "bootstrap_reps": reps, "block_len": block_len}


def _paired_block_bootstrap_diff_ci(
    frame, scenario, intensity, delay, strategy, comparator, metric,
    reps=2000, block_len=2, seed=1704,
):
    """Paired cluster bootstrap: resample seeds and within-seed temporal blocks."""
    subset = frame[(frame.drift_type == scenario) & np.isclose(frame.intensity, intensity)
                   & (frame.delay_windows == delay)
                   & frame.strategy.isin([strategy, comparator])]
    paired = subset.pivot_table(index=["seed", "window"], columns="strategy", values=metric, aggfunc="mean")
    if strategy not in paired or comparator not in paired:
        return {"mean_difference": None, "ci_low": None, "ci_high": None,
                "bootstrap_reps": reps, "block_len": block_len, "seed_clusters": 0}
    paired = paired[[strategy, comparator]].dropna()
    paired["difference"] = paired[strategy] - paired[comparator]
    by_seed = {
        int(seed_id): group.sort_index(level="window").difference.to_numpy(dtype=float)
        for seed_id, group in paired.groupby(level="seed")
    }
    by_seed = {s: x[np.isfinite(x)] for s, x in by_seed.items() if np.isfinite(x).any()}
    if not by_seed:
        return {"mean_difference": None, "ci_low": None, "ci_high": None,
                "bootstrap_reps": reps, "block_len": block_len, "seed_clusters": 0}
    point = float(np.mean([values.mean() for values in by_seed.values()]))
    seed_ids = np.asarray(sorted(by_seed), dtype=int)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(reps):
        sampled_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_means = []
        for seed_id in sampled_seeds:
            values = by_seed[int(seed_id)]
            draw = []
            while len(draw) < len(values):
                start = int(rng.integers(0, len(values)))
                indices = (np.arange(start, start + min(block_len, len(values))) % len(values))
                draw.extend(values[indices].tolist())
            seed_means.append(float(np.mean(draw[:len(values)])))
        boot.append(float(np.mean(seed_means)))
    return {
        "mean_difference": point,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "bootstrap_reps": reps,
        "block_len": block_len,
        "seed_clusters": int(len(seed_ids)),
    }


def _paired_effects(windows):
    rows = []
    scenario, intensity, delay, strategy = "harmful_abrupt", 1.0, 1, "drift_triggered"
    for comparator in ["static", "periodic", "online", "oracle"]:
        for metric in ["recall", "recall_at_fpr"]:
            interval = _paired_block_bootstrap_diff_ci(
                windows, scenario, intensity, delay, strategy, comparator, metric,
            )
            rows.append({
                "scenario": scenario, "intensity": intensity, "delay_windows": delay,
                "strategy": strategy, "comparator": comparator, "metric": metric,
                **interval,
            })
    return pd.DataFrame(rows)


def _headline_plot(windows, output):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="#0b1220", sharey=True)
    colors = {"static": "#94a3b8", "periodic": "#38bdf8", "drift_triggered": "#fb7185",
              "online": "#a3e635", "oracle": "#c084fc"}
    scenarios = [("none", 0.0, "No injected drift"),
                 ("harmful_abrupt", 1.0, "Harmful abrupt drift")]
    for ax, (scenario, intensity, title) in zip(axes, scenarios):
        ax.set_facecolor("#111b2b")
        subset = windows[(windows.drift_type == scenario) & np.isclose(windows.intensity, intensity)
                         & (windows.delay_windows == 1)]
        for strategy, group in subset.groupby("strategy"):
            stats = group.groupby("window").agg(
                mean=("recall", "mean"), sd=("recall", "std"), n=("seed", "nunique"),
                ranking=("recall_at_fpr", "mean"),
            )
            x = stats.index.to_numpy(dtype=float)
            y = stats["mean"].to_numpy(dtype=float)
            ci = 2.776 * stats.sd.fillna(0).to_numpy() / np.sqrt(stats.n.clip(lower=1).to_numpy())
            color = colors.get(strategy, "#94a3b8")
            strategy_label = "Zero-delay reference" if strategy == "oracle" else strategy.replace("_", " ").title()
            ax.plot(x, y, marker="o", ms=3, lw=2, label=strategy_label, color=color)
            ax.fill_between(x, np.clip(y - ci, 0, 1), np.clip(y + ci, 0, 1), alpha=.12, color=color)
            ax.plot(x, stats.ranking, ls="--", lw=1, alpha=.65, color=color)
        drift = subset.true_drift_window.dropna()
        if len(drift):
            ax.axvline(float(drift.mean()), color="#f87171", ls=":", lw=1.5, label="Mean injected start")
        ax.set_title(title, color="#f8fafc", fontsize=13, weight="bold")
        ax.set_xlabel("Scored stream window", color="#cbd5e1")
        ax.grid(alpha=.17, color="#94a3b8")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values(): spine.set_color("#334155")
    axes[0].set_ylabel("Recall", color="#cbd5e1")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=6, frameon=False, labelcolor="#e2e8f0")
    figure.suptitle("Fixed-threshold recall (solid); per-window recall@1% FPR (dashed)", color="#5eead4", fontsize=12)
    figure.tight_layout(rect=[0, .12, 1, .94])
    figure.savefig(output, dpi=165, facecolor="#0b1220")
    plt.close(figure)


def _detector_tradeoffs(run_summary):
    rows = []
    no_drift = run_summary[(run_summary.drift_type == "none")]
    for strategy, strategy_runs in no_drift.groupby("strategy"):
        for detector in ["adwin", "row_error_adwin", "psi", "ks"]:
            alarm_col = f"{detector}_false_alarms"
            if alarm_col not in strategy_runs:
                continue
            by_run = strategy_runs.groupby("seed")[alarm_col].first().dropna()
            if not len(by_run):
                continue
            rows.append({"scenario": "no_drift", "strategy": strategy, "detector": detector,
                         "runs": len(by_run), "false_alarm_rate": float((by_run > 0).mean()),
                         "mean_false_alarms_per_run": float(by_run.mean()),
                         "mean_detection_delay_windows": None})
    grouped = run_summary[run_summary.drift_type.isin(["harmful_abrupt", "harmful_gradual"])].groupby(
        ["drift_type", "intensity", "delay_windows", "strategy"], dropna=False
    )
    for (scenario, intensity, delay, strategy), harmful in grouped:
        for detector in ["adwin", "row_error_adwin", "psi", "ks"]:
            if detector in {"adwin", "row_error_adwin"} and strategy not in {"static", "drift_triggered"}:
                # ADWIN is a passive monitor outside the static and drift-triggered policies.
                continue
            delay_col = f"{detector}_detection_delay"
            missed_col = f"{detector}_missed_drift"
            if delay_col not in harmful or missed_col not in harmful:
                continue
            evaluated = harmful[harmful[missed_col].notna()]
            if evaluated.empty:
                continue
            detected = evaluated[evaluated[delay_col].notna()]
            vals = pd.to_numeric(detected[delay_col], errors="coerce")
            alarm_col = f"{detector}_false_alarms"
            rows.append({"scenario": scenario, "intensity": intensity, "delay_windows": delay,
                         "strategy": strategy, "detector": detector,
                         "runs": len(evaluated), "detected_runs": len(detected), "false_alarm_rate": None,
                         "mean_false_alarms_per_run": float(pd.to_numeric(evaluated[alarm_col], errors="coerce").mean()) if alarm_col in evaluated else None,
                         "mean_detection_delay_windows": float(vals.mean()) if vals.notna().any() else None,
                         "missed_drift_rate": float(evaluated[missed_col].astype(float).mean())})
    return pd.DataFrame(rows)


def _diagnosis(frame):
    subset = frame[(frame.strategy == "drift_triggered") & (frame.drift_type == "harmful_abrupt")
                   & np.isclose(frame.intensity, 1.0) & (frame.delay_windows == 1)].copy()
    keep = ["seed", "window", "true_drift_window", "n_frauds", "warning", "psi_alarm", "ks_alarm",
            "adwin_fraud_miss_alarm", "adwin_row_error_alarm", "row_error_adwin_enabled",
            "retrain_count", "events_json"]
    return subset[keep].sort_values(["seed", "window"])


def _write_findings(windows, run_summary, tradeoffs, paired_effects):
    comparison = run_summary[
        (((run_summary.drift_type == "none") & np.isclose(run_summary.intensity, 0.0))
         | ((run_summary.drift_type == "harmful_abrupt") & np.isclose(run_summary.intensity, 1.0)))
        & (run_summary.delay_windows == 1)
    ].groupby(["drift_type", "strategy"], as_index=False)[
        ["mean_threshold_recall", "mean_realized_fpr", "mean_recall_at_fpr",
         "total_false_positives", "total_missed_frauds", "retrain_count"]
    ].mean()
    lines = [
        "# Regenerated experiment findings",
        "",
        "This note is generated by `experiments.run_all` from the artifacts written in the same run. "
        "It describes this semi-synthetic, sampled experiment only; it is not population-level evidence.",
        "",
        "## Headline conditions",
        "",
        "Mean run-level outcomes, averaged across the five configured seeds for each strategy/condition:",
        "",
        "| Scenario | Strategy | Threshold recall | Realized FPR | Recall@1% FPR | False positives/run | Missed fraud/run | Retrains/run |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.sort_values(["drift_type", "strategy"]).iterrows():
        scenario = "No drift" if row.drift_type == "none" else "Harmful abrupt, intensity 1"
        strategy_label = "Zero-delay reference (oracle)" if row.strategy == "oracle" else row.strategy
        fmt = lambda value: "n/a" if pd.isna(value) else f"{float(value):.3f}"
        lines.append(
            f"| {scenario} | {strategy_label} | {fmt(row.mean_threshold_recall)} | {fmt(row.mean_realized_fpr)} | "
            f"{fmt(row.mean_recall_at_fpr)} | {fmt(row.total_false_positives)} | {fmt(row.total_missed_frauds)} | {fmt(row.retrain_count)} |"
        )
    lines.extend([
        "", "## Paired strategy differences under harmful abrupt drift", "",
        "Differences are drift-triggered minus comparator. Intervals resample seed clusters and contiguous within-run blocks; they are exploratory, unadjusted comparisons for this sampled benchmark, not population-level confidence. Only five seed clusters are available, and intervals do not adjust for multiple comparisons. A comparison is not described as a win when its interval includes zero.",
        "| Comparator | Metric | Mean difference | 95% paired block interval | Seed clusters |",
        "|---|---|---:|---:|---:|",
    ])
    for _, row in paired_effects.iterrows():
        comparator = "zero-delay reference" if row.comparator == "oracle" else row.comparator
        if pd.isna(row.mean_difference):
            point, interval = "n/a", "n/a"
        else:
            point = f"{float(row.mean_difference):+.3f}"
            interval = f"[{float(row.ci_low):+.3f}, {float(row.ci_high):+.3f}]"
        metric = "Threshold recall" if row.metric == "recall" else "Recall@1% FPR"
        lines.append(f"| {comparator} | {metric} | {point} | {interval} | {int(row.seed_clusters)} |")
    lines.extend([
        "", "## Detector trade-offs", "",
        "No-drift false-alarm rates are stratified by strategy and measured across held-out seed runs; the row-error ADWIN baseline is static-only.",
        "| Scenario | Strategy | Intensity | Delay | Detector | Runs | Detected runs | False-alarm rate | Mean detection delay (windows) | Missed-drift rate |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in tradeoffs.iterrows():
        fmt = lambda value: "n/a" if pd.isna(value) else f"{float(value):.3f}"
        detected_runs = "n/a" if pd.isna(row.get("detected_runs")) else str(int(row.detected_runs))
        lines.append(
            f"| {row.scenario} | {row.get('strategy', 'all')} | {fmt(row.get('intensity'))} | {fmt(row.get('delay_windows'))} | {row.detector} | "
            f"{int(row.runs)} | {detected_runs} | {fmt(row.get('false_alarm_rate'))} | {fmt(row.get('mean_detection_delay_windows'))} | "
            f"{fmt(row.get('missed_drift_rate'))} |"
        )
    null_rows = windows[windows.drift_type == "none"]
    null_runs = null_rows.groupby(["seed", "strategy"], as_index=False).agg(
        psi=("psi_alarm", "any"), ks=("ks_alarm", "any"), fraud_miss_adwin=("adwin_fraud_miss_alarm", "any")
    )
    strategy_rates = null_runs.groupby("strategy")[['psi', 'ks', 'fraud_miss_adwin']].mean()
    row_error_runs = null_rows[null_rows.row_error_adwin_enabled].groupby("seed").adwin_row_error_alarm.any()
    row_error_far = float(row_error_runs.mean()) if len(row_error_runs) else float("nan")
    lines.extend([
        "", "### No-drift false-alarm rate by strategy", "",
        "Rates are run-level across the five seeds; row-error ADWIN is evaluated only for static.",
        "| Strategy | PSI | KS | Fraud-miss ADWIN | Row-error ADWIN baseline |",
        "|---|---:|---:|---:|---:|",
    ])
    for strategy, row in strategy_rates.iterrows():
        row_error_text = f"{row_error_far:.3f}" if strategy == "static" else "n/a"
        lines.append(f"| {strategy} | {row.psi:.3f} | {row.ks:.3f} | {row.fraud_miss_adwin:.3f} | {row_error_text} |")
    controller = windows[
        (windows.strategy == "drift_triggered") & (windows.drift_type == "harmful_abrupt")
        & np.isclose(windows.intensity, 1.0) & (windows.delay_windows == 1)
    ]
    warnings = int(controller.warning.sum()) if len(controller) else 0
    adwin = int(controller.adwin_fraud_miss_alarm.sum()) if len(controller) else 0
    static_baseline = windows[
        (windows.strategy == "static") & (windows.drift_type == "harmful_abrupt")
        & np.isclose(windows.intensity, 1.0) & (windows.delay_windows == 1)
    ]
    row_adwin = int(static_baseline.adwin_row_error_alarm.sum()) if len(static_baseline) else 0
    run_subset = run_summary[
        (run_summary.strategy == "drift_triggered") & (run_summary.drift_type == "harmful_abrupt")
        & np.isclose(run_summary.intensity, 1.0) & (run_summary.delay_windows == 1)
    ]
    promotions = int(run_subset.retrain_count.fillna(0).sum()) if len(run_subset) else 0
    lines.extend([
        "",
        "## Drift-triggered diagnosis",
        "",
        f"Across the selected harmful-abrupt condition, {warnings} drift-triggered scored windows carried an active calibrated warning; "
        f"the fraud-miss ADWIN alarmed in {adwin} such windows. The static-only row-error ADWIN baseline alarmed in "
        f"{row_adwin} scored windows. "
        f"The run-level total of applied challenger promotions was {promotions}.",
        "See `drift_triggered_diagnosis.csv` for each window’s fraud count, detector states, gate metrics, and event log. "
        "A warning is not itself evidence of successful adaptation; an unapplied or rejected candidate is reported as such.",
        "",
        "False-alarm rates are measured on held-out no-drift seeds in the full run. If the run used `--quick`, the thresholds "
        "were calibrated on the same single seed and the false-alarm figures are only a smoke-test diagnostic.",
        "",
    ])
    (RESULTS / "experiment_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="one-seed smoke run; thresholds are calibrated on that seed only")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = load_transactions(args.data)
    seeds = SEEDS[:1] if args.quick else SEEDS[:max(1, min(args.seeds, len(SEEDS)))]
    scenarios = scenario_grid()
    pools = {seed: build_stream_pools(frame, seed=seed) for seed in seeds}

    # Null-signal pass. Calibrate each strategy's score diagnostics separately;
    # model updates change score distributions. Full runs use leave-one-seed-out
    # cutoffs so the held-out seed is not used to set its detector thresholds.
    null_signals = {}
    for seed in seeds:
        initial, calibration, stream = pools[seed]
        for strategy in STRATEGIES:
            probe = run_windowed_simulation(
                initial, calibration, stream, strategy=strategy, drift_type="none", seed=seed,
                window_size=args.window_size, drift_thresholds={
                    "feature_psi_top3_threshold": float("inf"), "score_psi_threshold": float("inf"),
                    "feature_ks_stat_threshold": float("inf"), "score_ks_stat_threshold": float("inf"),
                    "consecutive_windows": 2,
                },
                adwin_delta=ADWIN_DELTA,
                row_error_baseline=(strategy == "static"),
            )
            null_signals[(seed, strategy)] = pd.DataFrame(probe)

    thresholds_by_seed = {}
    for seed in seeds:
        calibration_seeds = [s for s in seeds if s != seed] or [seed]
        thresholds_by_seed[seed] = {}
        for strategy in STRATEGIES:
            calibration_rows = pd.concat(
                [null_signals[(s, strategy)] for s in calibration_seeds], ignore_index=True
            )
            thresholds_by_seed[seed][strategy] = calibrate_drift_thresholds(calibration_rows, quantile=0.95)

    all_rows = []
    for seed in seeds:
        initial, calibration, stream = pools[seed]
        print(f"Seed {seed}: initial={len(initial):,}, calibration={len(calibration):,}, stream={len(stream):,}; stream frauds={int(stream.Class.sum())}", flush=True)
        for drift_type, intensity, delay in scenarios:
            for strategy in STRATEGIES:
                print(f"  {drift_type:18} intensity={intensity:.1f} delay={delay} strategy={strategy}", flush=True)
                all_rows.extend(run_windowed_simulation(
                    initial, calibration, stream, strategy=strategy, drift_type=drift_type,
                    intensity=intensity, delay_windows=delay, seed=seed,
                    window_size=args.window_size, drift_thresholds=thresholds_by_seed[seed][strategy],
                    adwin_delta=ADWIN_DELTA,
                    row_error_baseline=(strategy == "static"),
                ))

    windows = pd.DataFrame(all_rows)
    windows.to_parquet(RESULTS / "window_metrics.parquet", index=False, compression="zstd")
    run_summary = _run_level_rows(windows)
    run_summary.to_csv(RESULTS / "run_summary.csv", index=False)
    aggregate = _aggregate(run_summary)
    (RESULTS / "strategy_summary.json").write_text(json.dumps(_json_safe(aggregate), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tradeoffs = _detector_tradeoffs(run_summary)
    tradeoffs.to_csv(RESULTS / "detector_tradeoffs.csv", index=False)
    diagnosis = _diagnosis(windows)
    diagnosis.to_csv(RESULTS / "drift_triggered_diagnosis.csv", index=False)
    paired_effects = _paired_effects(windows)
    paired_effects.to_csv(RESULTS / "paired_block_bootstrap_differences.csv", index=False)
    _write_findings(windows, run_summary, tradeoffs, paired_effects)
    sample_rows = _write_sample(frame)
    _headline_plot(windows, RESULTS / "headline_performance.png")

    bootstrap_rows = []
    for scenario, intensity, delay in [("none", 0.0, 1), ("harmful_abrupt", 1.0, 1)]:
        for strategy in STRATEGIES:
            interval = _bootstrap_block_ci(windows, scenario, intensity, delay, strategy, "recall")
            bootstrap_rows.append({"scenario": scenario, "intensity": intensity, "delay_windows": delay,
                                   "strategy": strategy, "metric": "threshold_recall", **interval})
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS / "block_bootstrap.csv", index=False)

    config = {
        "source": "creditcard.csv (ULB/Worldline via Kaggle)", "input_rows": len(frame),
        "input_frauds": int(frame.Class.sum()), "initial_fraction": 0.20,
        "calibration_fraction": 0.05, "stream_fraction": 0.75,
        "legitimate_keep_fraction": 0.25, "sampling_applied_to": ["initial", "calibration", "stream"],
        "window_size": args.window_size, "n_seeds": len(seeds), "seeds": seeds,
        "scenario_count": len(scenarios), "strategy_count": len(STRATEGIES),
        "simulated_runs": len(run_summary), "sample_rows": sample_rows,
        "max_fpr": 0.01, "review_fee": 5.0,
        "drift_threshold_calibration": "strategy-matched leave-one-seed-out no-drift runs" if len(seeds) > 1 else "single-seed, strategy-matched smoke calibration; not an independent FAR estimate",
        "drift_thresholds_by_seed_and_strategy": thresholds_by_seed,
        "detector_consecutive_windows": 2, "adwin_delta": ADWIN_DELTA,
        "controller_signal": "fraud-miss ADWIN only; PSI and KS are diagnostic",
        "ks_alarm_statistic": "two-sample KS D statistic; p-values are descriptive only",
        "score_reference": "fixed calibration cohort rescored under each active model; stream windows never refresh reference",
        "feature_psi_summary": "mean of top three feature PSI values, empirically calibrated by strategy",
        "row_error_adwin_baseline": "static strategy only; NA elsewhere, not zero alarms",
        "window_tail_policy": "merge final remainder into prior window if below half the target size; row-count only",
        "initial_train_cap_rows": 12000, "oracle_definition": "zero-delay retraining reference; not a guaranteed upper bound",
        "drift_start_policy": "seed-specific start sampled from the middle portion of the stream",
        "uncertainty": "five-seed descriptive t intervals and paired seed-clustered moving-block bootstrap differences; five seed clusters, exploratory and unadjusted, neither is population-level certainty",
        "paired_comparison_adjustment": "none; exploratory intervals, no multiple-comparison correction",
        "limitation": "semi-synthetic drift; sampled stream has elevated fraud prevalence; amount costs use dataset units, not realized savings",
    }
    (RESULTS / "experiment_config.json").write_text(json.dumps(_json_safe(config), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nSaved {len(windows):,} window rows from {len(run_summary):,} simulation runs.")
    print("Artifacts: results/window_metrics.parquet, results/strategy_summary.json, results/detector_tradeoffs.csv, results/paired_block_bootstrap_differences.csv, results/drift_triggered_diagnosis.csv, results/headline_performance.png")


if __name__ == "__main__":
    main()

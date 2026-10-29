"""Streamlit dashboard for precomputed MOLT semi-synthetic experiments."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from molt.artifacts import require_files

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SAMPLE_PATH = ROOT / "app_data" / "creditcard_sample.parquet"
TRADEOFF_PATH = RESULTS / "detector_tradeoffs.csv"
HEADLINE_PATH = RESULTS / "headline_performance.png"
MODEL_PATH = ROOT / "models" / "molt_model.joblib"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

st.set_page_config(page_title="MOLT | Shed the old. Adapt to the new.", page_icon="M", layout="wide")
# styles live in style.css so this file stays readable
st.markdown(f"<style>{(ROOT / 'style.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_artifacts():
    needed = [RESULTS / "window_metrics.parquet", RESULTS / "strategy_summary.json", SAMPLE_PATH, TRADEOFF_PATH, HEADLINE_PATH]
    require_files(needed)
    windows = pd.read_parquet(needed[0])
    summary = json.loads(needed[1].read_text(encoding="utf-8"))
    sample = pd.read_parquet(SAMPLE_PATH)
    tradeoffs = pd.read_csv(TRADEOFF_PATH)
    config_path = RESULTS / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return windows, pd.DataFrame(summary), sample, config, tradeoffs


try:
    windows, summary, sample, config, tradeoffs = load_artifacts()
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    threshold_settings = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8")) if THRESHOLD_PATH.exists() else {}
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    """
    <section class="molt-publication" aria-label="MOLT research workbench">
      <div class="molt-pub-grid">
        <div class="molt-pub-title">
          <div class="molt-pub-kicker">
            Fraud detection under controlled distribution shift
          </div>
          <div class="molt-pub-wordmark">MOLT</div>
          <div class="molt-pub-tagline">
            Comparing model adaptation strategies under delayed labels.
          </div>
        </div>

       
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(
        '<div class="molt-sidebar-brand"><div class="molt-sidebar-sigil">M</div>'
        '<div><div class="molt-sidebar-name">MOLT</div><div class="molt-sidebar-sub">Research console</div></div></div>',
        unsafe_allow_html=True,
    )
    st.header("Experiment protocol")
    st.write(f"**Dataset rows:** {config.get('input_rows', '—'):,}" if config.get("input_rows") else "**Dataset rows:** —")
    st.write(f"**Frauds in source:** {config.get('input_frauds', '—'):,}" if config.get("input_frauds") else "**Frauds in source:** —")
    st.write(f"**Seeds:** {config.get('n_seeds', '—')}")
    st.write(f"**Window size:** {config.get('window_size', '—'):,} rows" if config.get("window_size") else "**Window size:** —")
    st.write("**Subsampling:** all frauds + 25% of legitimate rows in every pool, no replacement")
    st.write("**Threshold rule:** validation/calibration FPR ≤ 1%")
    st.write("**Adaptation trigger:** fraud-miss ADWIN (δ = 0.002); PSI/KS are diagnostic")
    st.caption("Sampling raises fraud prevalence. Amount-based cost uses dataset units, not realized savings.")

main_tab, drift_tab, strategies_tab, attack_tab, replay_tab, analyzer_tab = st.tabs(
    ["Headline", "Drift monitor", "Strategy comparison", "Attack simulator", "Replay", "Transaction analyzer"]
)

scenario_labels = {
    "none": "No injected drift",
    "benign_covariate": "Benign mixed covariate shift",
    "harmful_abrupt": "Harmful abrupt drift",
    "harmful_gradual": "Harmful gradual drift",
}


def scenario_filter(frame, drift_type, intensity, delay):
    if drift_type == "none":
        intensity = 0.0
    return frame[
        (frame["drift_type"] == drift_type)
        & np.isclose(frame["intensity"].astype(float), float(intensity))
        & (frame["delay_windows"].astype(int) == int(delay))
    ].copy()


def scenario_controls(frame, drift_type, key_prefix):
    available = frame[frame.drift_type == drift_type]
    intensity_options = sorted(available.intensity.astype(float).unique().tolist())
    default_intensity = 0.0 if drift_type == "none" else (1.0 if 1.0 in intensity_options else intensity_options[-1])
    intensity = st.select_slider(
        "Injection intensity", options=intensity_options, value=default_intensity,
        key=f"{key_prefix}_intensity",
    )
    delay_options = sorted(
        available[np.isclose(available.intensity.astype(float), float(intensity))]
        .delay_windows.astype(int).unique().tolist()
    )
    default_delay = 1 if 1 in delay_options else delay_options[0]
    delay = st.select_slider(
        "Label delay (windows)", options=delay_options, value=default_delay,
        key=f"{key_prefix}_delay",
    )
    return float(intensity), int(delay)


def performance_chart(frame, title):
    frame = frame.sort_values(["seed", "strategy", "window"]).copy()
    for metric, column in [("fixed", "recall"), ("ranking", "recall_at_fpr")]:
        frame[f"rolling_{metric}"] = frame.groupby(["seed", "strategy"])[column].transform(
            lambda values: values.rolling(3, min_periods=1).mean()
        )
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="#f3f0e7")
    ax.set_facecolor("#fffdf7")
    colors = {"static": "#78837d", "periodic": "#357b78", "drift_triggered": "#c84f36", "online": "#708355", "oracle": "#77668e"}
    for strategy, group in frame.groupby("strategy"):
        color = colors.get(strategy, "#555555")
        strategy_label = "Zero-delay reference" if strategy == "oracle" else strategy.replace("_", " ").title()
        for metric, style, label in [("fixed", "-", "threshold recall"), ("ranking", "--", "recall@1% FPR")]:
            stats = group.groupby("window")[f"rolling_{metric}"].agg(["mean", "std", "count"])
            x = stats.index.to_numpy(dtype=float)
            mean = stats["mean"].to_numpy(dtype=float)
            ci = (2.776 * stats["std"].fillna(0) / np.sqrt(stats["count"].clip(lower=1))).to_numpy(dtype=float)
            ax.plot(x, mean, linestyle=style, marker="o" if metric == "fixed" else None, ms=3, lw=2 if metric == "fixed" else 1.2,
                    alpha=1 if metric == "fixed" else .7, label=f"{strategy_label} · {label}", color=color)
            if metric == "fixed":
                ax.fill_between(x, np.clip(mean-ci, 0, 1), np.clip(mean+ci, 0, 1), alpha=.1, color=color)
    drift_starts = frame["true_drift_window"].dropna()
    if not drift_starts.empty:
        ax.axvline(float(drift_starts.mean()), color="#c84f36", ls="--", label="Mean injected drift start")
    ax.set(title=title, xlabel="Chronological stream window", ylabel="3-window rolling recall", ylim=(0, 1))
    ax.tick_params(colors="#68736d")
    ax.xaxis.label.set_color("#4c5953")
    ax.yaxis.label.set_color("#4c5953")
    ax.title.set_color("#202a27")
    ax.title.set_fontfamily("serif")
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color("#cbc9bd")
    ax.grid(alpha=.45, color="#d4d0c5")
    ax.legend(ncol=3, frameon=False, fontsize=8, labelcolor="#384641")
    fig.tight_layout()
    return fig


def headline_chart(frame):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor="#f3f0e7", sharey=True)
    colors = {
        "static": "#78837d",
        "periodic": "#357b78",
        "drift_triggered": "#c84f36",
        "online": "#708355",
        "oracle": "#77668e",
    }
    scenarios = [("none", 0.0, "No injected drift"), ("harmful_abrupt", 1.0, "Harmful abrupt drift")]
    for ax, (scenario, intensity, title) in zip(axes, scenarios):
        ax.set_facecolor("#fffdf7")
        subset = frame[
            (frame.drift_type == scenario)
            & np.isclose(frame.intensity.astype(float), intensity)
            & (frame.delay_windows == 1)
        ]
        for strategy, group in subset.groupby("strategy"):
            stats = group.groupby("window").agg(
                mean=("recall", "mean"),
                sd=("recall", "std"),
                n=("seed", "nunique"),
                ranking=("recall_at_fpr", "mean"),
            )
            x = stats.index.to_numpy(dtype=float)
            y = stats["mean"].to_numpy(dtype=float)
            ci = 2.776 * stats.sd.fillna(0).to_numpy() / np.sqrt(stats.n.clip(lower=1).to_numpy())
            color = colors.get(strategy, "#78837d")
            label = "Zero-delay reference" if strategy == "oracle" else strategy.replace("_", " ").title()
            ax.plot(x, y, marker="o", ms=3, lw=2, label=label, color=color)
            ax.fill_between(x, np.clip(y - ci, 0, 1), np.clip(y + ci, 0, 1), alpha=.12, color=color)
            ax.plot(x, stats.ranking, ls="--", lw=1, alpha=.65, color=color)
        drift = subset.true_drift_window.dropna()
        if not drift.empty:
            ax.axvline(float(drift.mean()), color="#c84f36", ls=":", lw=1.5, label="Mean injected start")
        ax.set_title(title, color="#202a27", fontsize=14, weight="bold", loc="left", pad=12, fontfamily="serif")
        ax.set_xlabel("Scored stream window", color="#4c5953")
        ax.set_ylim(0, 1)
        ax.grid(alpha=.42, color="#d4d0c5")
        ax.tick_params(colors="#68736d")
        for spine in ax.spines.values():
            spine.set_color("#cbc9bd")
    axes[0].set_ylabel("Recall", color="#4c5953")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=6, frameon=False, labelcolor="#384641", fontsize=8)
    figure.suptitle("Fixed-threshold recall · dashed = recall@1% FPR", color="#c84f36", fontsize=10, y=.98)
    figure.tight_layout(rect=[0, .11, 1, .92])
    return figure


with main_tab:
    st.header("Headline performance")
    st.write(
        "The headline plot compares the no-drift control with harmful abrupt drift. Solid lines show recall at each strategy’s "
        "fixed, available-label-calibrated threshold; dashed lines show the separate per-window ranking metric recall@1% FPR. "
        "The dashed metric is not the recall achieved by the decision threshold. Confidence summaries are descriptive across "
        "five legitimate-row subsamples and moving blocks, not real-world certainty."
    )
    st.pyplot(headline_chart(windows), width="stretch")
    st.caption("Solid = recall at the applied threshold; dashed = the separate per-window ranking metric. Shading uses the five-seed interval; the dotted marker shows mean injected-drift start.")
    headline_summary = summary[
        (((summary.drift_type == "none") & np.isclose(summary.intensity, 0.0))
         | ((summary.drift_type == "harmful_abrupt") & np.isclose(summary.intensity, 1.0)))
        & (summary.delay_windows == 1)
    ].copy()
    headline_cols = ["drift_type", "strategy", "mean_threshold_recall_mean", "mean_realized_fpr_mean",
                     "mean_recall_at_fpr_mean", "total_false_positives_mean", "total_missed_frauds_mean"]
    headline_cols = [column for column in headline_cols if column in headline_summary.columns]
    if not headline_summary.empty:
        st.caption("Run-averaged threshold outcomes for the two scenarios shown above; counts are mean totals per run.")
        headline_display = headline_summary[headline_cols].round(3).rename(columns={
            "drift_type": "Scenario", "strategy": "Strategy",
            "mean_threshold_recall_mean": "Threshold recall",
            "mean_realized_fpr_mean": "Realized FPR",
            "mean_recall_at_fpr_mean": "Ranking recall@1% FPR",
            "total_false_positives_mean": "False positives / run",
            "total_missed_frauds_mean": "Missed fraud / run",
        })
        headline_display["Strategy"] = headline_display["Strategy"].replace({"oracle": "Zero-delay reference"})
        st.dataframe(headline_display, hide_index=True, width="stretch")
    cols = st.columns(4)
    cols[0].metric("Simulation runs", f"{config.get('simulated_runs', 0):,}")
    cols[1].metric("Seeds", f"{config.get('n_seeds', 0)}")
    cols[2].metric("Rows per window", f"{config.get('window_size', 0):,}")
    cols[3].metric("Sample rows", f"{config.get('sample_rows', len(sample)):,}")
    st.info(
        "The source covers only two days and has few frauds. Drift is injected into later historical rows; "
        "the resulting curve demonstrates controlled experiments, not naturally occurring drift."
    )
    fraud_counts = windows[(windows.drift_type == "harmful_abrupt") & np.isclose(windows.intensity, 1.0) & (windows.delay_windows == 1)]
    if not fraud_counts.empty:
        counts = fraud_counts.groupby("window").n_frauds.agg(["mean", "min", "max"]).reset_index()
        st.caption("Fraud counts per scored window vary across the sampled streams:")
        st.dataframe(counts.rename(columns={"mean": "Mean frauds", "min": "Minimum", "max": "Maximum"}).round(1), hide_index=True, width="stretch")
    bootstrap_path = RESULTS / "block_bootstrap.csv"
    if bootstrap_path.exists():
        bootstrap = pd.read_csv(bootstrap_path)
        st.caption("Moving-block bootstrap intervals are descriptive temporal-window uncertainty, not population-level confidence.")
        st.dataframe(bootstrap.round(3), hide_index=True, width="stretch")

with drift_tab:
    st.header("Drift monitor")
    scenario = st.selectbox("Scenario", list(scenario_labels), format_func=lambda x: scenario_labels[x], key="monitor_scenario")
    intensity, delay = scenario_controls(windows, scenario, "monitor")
    strategy = st.selectbox("Strategy", sorted(windows.strategy.unique()), key="monitor_strategy")
    sub = scenario_filter(windows, scenario, intensity, delay)
    sub = sub[sub.strategy == strategy]
    if sub.empty:
        st.info("No precomputed run for this combination. Run `python -m experiments.run_all` to create the full grid.")
    else:
        available_windows = sorted(sub.window.unique().tolist())
        selected_window = st.select_slider("Window", options=available_windows, value=available_windows[0])
        selected = sub[sub.window == selected_window]
        psi_rows, ks_rows = [], []
        for _, row in selected.iterrows():
            psi = json.loads(row.feature_psi_json)
            ks = json.loads(row.feature_ks_stat_json)
            psi_rows.append(psi)
            ks_rows.append(ks)
        psi_mean = pd.DataFrame(psi_rows).mean().sort_values(ascending=False)
        ks_max = pd.DataFrame(ks_rows).max()
        feature_report = pd.DataFrame({"PSI": psi_mean, "KS D statistic": ks_max}).sort_values("PSI", ascending=False).head(15)
        st.subheader(f"Top feature shifts · stream window {selected_window}")
        st.dataframe(feature_report.round(5), width="stretch")
        row = selected.iloc[0]
        metrics_cols = st.columns(5)
        metrics_cols[0].metric("Max feature/score PSI", f"{max(row.max_psi, row.score_psi):.3f}")
        metrics_cols[1].metric("Maximum KS D statistic", f"{max(row.max_ks_statistic, row.score_ks_statistic):.3f}")
        metrics_cols[2].metric("Fraud-miss ADWIN", f"{selected.adwin_fraud_miss_alarm.mean():.0%}")
        if selected.row_error_adwin_enabled.any():
            metrics_cols[3].metric("Row-error ADWIN baseline", f"{selected.adwin_row_error_alarm.mean():.0%}")
        else:
            metrics_cols[3].metric("Row-error ADWIN baseline", "Static-only")
        metrics_cols[4].metric("Injected drift starts", str(row.true_drift_window) if pd.notna(row.true_drift_window) else "None")
        timeline_input = sub.copy()
        if not timeline_input.row_error_adwin_enabled.any():
            timeline_input["adwin_row_error_alarm"] = np.nan
        timeline = timeline_input.groupby("window").agg(PSI=("max_psi", "mean"), Score_PSI=("score_psi", "mean"), Fraud_miss_ADWIN=("adwin_fraud_miss_alarm", "mean"), Row_error_baseline=("adwin_row_error_alarm", "mean"), KS_alarm_rate=("ks_alarm", "mean"), PSI_alarm_rate=("psi_alarm", "mean"))
        st.line_chart(timeline[["PSI", "Score_PSI", "Fraud_miss_ADWIN", "Row_error_baseline"]])
        st.dataframe(timeline.reset_index(), hide_index=True, width="stretch")
        st.caption("PSI uses calibration quantile bins. KS D statistics summarize empirical-distribution differences; p-values are descriptive only. PSI and KS are diagnostics, not adaptation triggers.")
        st.subheader("Detector trade-offs")
        st.caption("No-drift false-alarm rates use the held-out seed controls; harmful detection delays are reported by scenario, intensity, and label delay.")
        st.dataframe(tradeoffs.round(3), hide_index=True, width="stretch")
        null_rows = windows[windows.drift_type == "none"]
        if not null_rows.empty:
            null_runs = null_rows.groupby(["seed", "strategy"], as_index=False).agg(
                PSI=("psi_alarm", "any"), KS=("ks_alarm", "any"),
                Fraud_miss_ADWIN=("adwin_fraud_miss_alarm", "any"),
            )
            by_strategy = null_runs.groupby("strategy")[["PSI", "KS", "Fraud_miss_ADWIN"]].mean()
            baseline = null_rows[null_rows.row_error_adwin_enabled].groupby("seed").adwin_row_error_alarm.any()
            psi_rate, ks_rate = float(null_runs.PSI.mean()), float(null_runs.KS.mean())
            st.warning(
                f"Observed no-drift run-level false alarms: PSI {psi_rate:.0%} and KS {ks_rate:.0%} "
                f"across {len(null_runs)} seed-strategy runs. These diagnostics do not confirm harmful drift "
                "and do not trigger adaptation."
            )
            st.caption(
                f"Static-only row-error ADWIN baseline false-alarm rate: "
                f"{(float(baseline.mean()) if len(baseline) else float('nan')):.0%} across {len(baseline)} runs."
            )
            st.dataframe(by_strategy.rename(columns={
                "PSI": "PSI false-alarm rate", "KS": "KS false-alarm rate",
                "Fraud_miss_ADWIN": "Fraud-miss ADWIN false-alarm rate",
            }).reset_index().round(3), hide_index=True, width="stretch")

with strategies_tab:
    st.header("Strategy comparison")
    scenario = st.selectbox("Scenario", list(scenario_labels), format_func=lambda x: scenario_labels[x], key="compare_scenario")
    intensity, delay = scenario_controls(windows, scenario, "compare")
    report = summary[
        (summary.drift_type == scenario)
        & np.isclose(summary.intensity.astype(float), float(intensity))
        & (summary.delay_windows.astype(int) == int(delay))
    ].copy()
    if report.empty:
        st.info("No precomputed results for this combination.")
    else:
        for optional_metric in ["benign_adwin_alarms_mean", "benign_psi_alarms_mean", "benign_ks_alarms_mean"]:
            if optional_metric not in report:
                report[optional_metric] = 0.0
        if "adwin_missed_rate" not in report:
            report["adwin_missed_rate"] = report.get("any_drift_missed_rate", 0.0)
        cols = ["strategy", "mean_threshold_recall_mean", "mean_realized_fpr_mean", "mean_recall_at_fpr_mean", "mean_pr_auc_mean", "adwin_detection_delay_mean", "adwin_false_alarms_mean", "retrain_count_mean", "amount_units_saved_vs_static_mean"]
        show = report[cols].rename(columns={
            "strategy": "Strategy", "mean_threshold_recall_mean": "Recall at applied threshold",
            "mean_realized_fpr_mean": "Realized FPR",
            "mean_recall_at_fpr_mean": "Ranking recall@1% FPR",
            "mean_pr_auc_mean": "Mean PR-AUC", "adwin_detection_delay_mean": "ADWIN delay (windows)",
            "adwin_false_alarms_mean": "ADWIN false alarms", "benign_adwin_alarms_mean": "Benign ADWIN alarms",
            "benign_psi_alarms_mean": "Benign PSI alarms", "benign_ks_alarms_mean": "Benign KS alarms",
            "retrain_count_mean": "Retrains",
            "amount_units_saved_vs_static_mean": "Amount units saved vs static",
        })
        show["Strategy"] = show["Strategy"].replace({"oracle": "Zero-delay reference"})
        st.dataframe(show.round(3), hide_index=True, width="stretch")
        paired_path = RESULTS / "paired_block_bootstrap_differences.csv"
        if scenario == "harmful_abrupt" and np.isclose(float(intensity), 1.0) and int(delay) == 1 and paired_path.exists():
            effects = pd.read_csv(paired_path)
            effects = effects[["comparator", "metric", "mean_difference", "ci_low", "ci_high", "seed_clusters"]].copy()
            effects["comparator"] = effects["comparator"].replace({"oracle": "Zero-delay reference"})
            effects["metric"] = effects["metric"].replace({"recall": "Threshold recall", "recall_at_fpr": "Recall@1% FPR"})
            effects = effects.rename(columns={
                "comparator": "Comparator", "metric": "Metric", "mean_difference": "Mean difference",
                "ci_low": "95% interval — low", "ci_high": "95% interval — high", "seed_clusters": "Seed clusters",
            })
            st.subheader("Paired differences: drift-triggered minus comparator")
            st.caption("Positive differences favor drift-triggered. Intervals resample five seed clusters and contiguous windows; they are exploratory, unadjusted, and not population-level certainty. Intervals containing zero do not establish a strategy win.")
            st.dataframe(effects.round(3), hide_index=True, width="stretch")
        st.caption("Cost savings are paired against the static strategy per seed and use dataset Amount units plus an assumed review fee. They are not realized savings.")

with attack_tab:
    st.header("Attack and shift simulator")
    st.write("Select an already-run injection configuration. The chart shows precomputed results; sliders do not initiate training on the hosted app.")
    attack_type = st.selectbox("Injected change", ["harmful_abrupt", "harmful_gradual", "benign_covariate"], format_func=lambda x: scenario_labels[x])
    intensity, delay = scenario_controls(windows, attack_type, "attack")
    selected = scenario_filter(windows, attack_type, intensity, delay)
    if selected.empty:
        st.warning("This combination is not in the precomputed result grid. Run the experiment suite locally and redeploy its outputs.")
    else:
        st.pyplot(performance_chart(selected, f"{scenario_labels[attack_type]} · intensity {intensity:g} · label delay {delay}"), width="stretch")
        if attack_type.startswith("harmful"):
            st.caption("Only fraud rows are moved toward legitimate reference medians on high-importance features; labels remain unchanged.")
        else:
            st.caption("Amount and several important V features shift for both labels. Labels remain unchanged; detectors may still alarm.")

with replay_tab:
    st.header("Window replay")
    replay_scenario = st.selectbox("Replay scenario", list(scenario_labels), format_func=lambda x: scenario_labels[x], key="replay_scenario")
    replay_intensity, replay_delay = scenario_controls(windows, replay_scenario, "replay")
    replay_strategy = st.selectbox("Replay strategy", sorted(windows.strategy.unique()), key="replay_strategy")
    replay = scenario_filter(windows, replay_scenario, replay_intensity, replay_delay)
    replay = replay[replay.strategy == replay_strategy].sort_values(["seed", "window"])
    if replay.empty:
        st.info("No matching replay is available.")
    else:
        seed_choice = st.selectbox("Seed", sorted(replay.seed.unique().tolist()))
        replay = replay[replay.seed == seed_choice].reset_index(drop=True)
        autoplay = st.checkbox("Auto-play windows", value=False)
        if "replay_index" not in st.session_state:
            st.session_state.replay_index = 0

        @st.fragment(run_every="1s")
        def replay_frame(frame, enabled):
            if enabled:
                st.session_state.replay_index = (st.session_state.replay_index + 1) % len(frame)
            st.session_state.replay_index = min(st.session_state.replay_index, len(frame) - 1)
            index = st.slider("Window index", 0, len(frame) - 1, key="replay_index")
            row = frame.iloc[index]
            a, b, c, d, e = st.columns(5)
            a.metric("Recall at applied threshold", "—" if pd.isna(row.recall) else f"{row.recall:.1%}")
            b.metric("Realized FPR", "—" if pd.isna(row.realized_fpr) else f"{row.realized_fpr:.2%}")
            c.metric("Ranking recall@1% FPR", "—" if pd.isna(row.recall_at_fpr) else f"{row.recall_at_fpr:.1%}")
            d.metric("False positives", int(row.false_positives))
            e.metric("Window cost proxy", f"{row.cost:,.1f} units")
            st.dataframe(frame.iloc[[index]][["window", "time_start", "time_end", "n_rows", "n_frauds", "recall", "realized_fpr", "recall_at_fpr", "pr_auc", "alerts", "cost", "psi_alarm", "ks_alarm", "adwin_alarm"]], hide_index=True, width="stretch")
        replay_frame(replay, autoplay)

with analyzer_tab:
    st.header("Transaction analyzer")
    st.caption("The included sample has all dataset fraud rows plus a seeded subset of legitimate rows. The score is a model output, not a calibrated real-world probability.")
    if model is None:
        st.warning("Saved model not found; run `python threshold_analysis.py` locally and include the artifact.")
    else:
        idx = st.number_input("Sample row index", min_value=0, max_value=len(sample) - 1, value=0, step=1)
        row = sample.iloc[int(idx)]
        feature_names = list(model.feature_names_in_)
        sample_X = pd.DataFrame([{name: float(row[name]) for name in feature_names}], columns=feature_names)
        score = float(model.predict_proba(sample_X)[0, 1])
        threshold = float(threshold_settings.get("threshold", 0.5))
        a, b, c = st.columns(3)
        a.metric("Model score", f"{score:.2%}")
        b.metric("Fixed-FPR threshold", f"{threshold:.2%}")
        c.metric("Model decision", "Flagged" if score >= threshold else "Below threshold")
        st.write(f"Historical label: **{'Fraud' if int(row['Class']) else 'Legitimate'}** (available only because this is labeled historical data).")
        st.dataframe(pd.DataFrame([{"Time (ordering metadata)": float(row["Time"]), "Amount": float(row["Amount"]), "Score": score, "Threshold": threshold}]), hide_index=True, width="stretch")

st.divider()
st.caption("MOLT research prototype · ULB/Worldline credit-card dataset · semi-synthetic injected drift · not a production fraud decision system")

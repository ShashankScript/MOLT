"""Print and export the harmful-abrupt drift-triggered diagnosis."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    path = args.results / "window_metrics.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run python -m experiments.run_all first.")
    frame = pd.read_parquet(path)
    subset = frame[(frame.strategy == "drift_triggered") & (frame.drift_type == "harmful_abrupt")
                   & np.isclose(frame.intensity, 1.0) & (frame.delay_windows == 1)].copy()
    if subset.empty:
        raise SystemExit("No harmful_abrupt intensity=1.0 delay=1 drift_triggered rows found.")
    static = frame[(frame.strategy == "static") & (frame.drift_type == "harmful_abrupt")
                   & np.isclose(frame.intensity, 1.0) & (frame.delay_windows == 1)]
    static_row_adwin = static.set_index(["seed", "window"])["adwin_row_error_alarm"].to_dict()
    output_rows = []
    for _, row in subset.sort_values(["seed", "window"]).iterrows():
        events = json.loads(row.events_json) if isinstance(row.events_json, str) else []
        gates = [event for event in events if event["event"] in {"confirmation", "promotion", "rejection"}]
        output_rows.append({
            "seed": int(row.seed), "window": int(row.window),
            "drift_start_window": int(row.true_drift_window), "n_frauds": int(row.n_frauds),
            "psi_alarm": bool(row.psi_alarm), "ks_alarm": bool(row.ks_alarm),
            "fraud_miss_adwin_alarm": bool(row.adwin_fraud_miss_alarm),
            "row_error_adwin_alarm": bool(row.adwin_row_error_alarm),
            "row_error_adwin_enabled": bool(row.row_error_adwin_enabled),
            "row_error_adwin_static_baseline": bool(static_row_adwin.get((row.seed, row.window), False)),
            "warning": bool(row.warning), "retrain_count": int(row.retrain_count),
            "gate_events": json.dumps(gates, allow_nan=False),
            "all_events": row.events_json,
        })
    report = pd.DataFrame(output_rows)
    target = args.results / "drift_triggered_diagnosis.csv"
    report.to_csv(target, index=False)
    print("Harmful abrupt drift-triggered diagnosis (intensity=1.0, delay=1)")
    for _, row in report.iterrows():
        row_adwin = int(row.row_error_adwin_static_baseline)
        print(f"seed={row.seed:5d} window={row.window:2d} start={row.drift_start_window:2d} "
              f"frauds={row.n_frauds:2d} PSI={int(row.psi_alarm)} KS={int(row.ks_alarm)} "
              f"ADWIN_miss={int(row.fraud_miss_adwin_alarm)} ADWIN_row_static={row_adwin} "
              f"warning={int(row.warning)} retrains={row.retrain_count} gates={row.gate_events}")
    print(f"\nSaved {target}")


if __name__ == "__main__":
    main()

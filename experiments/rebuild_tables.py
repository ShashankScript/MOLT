"""Rebuild detector trade-off tables from saved experiment artifacts without simulation."""

import pandas as pd

from experiments.run_all import RESULTS, _detector_tradeoffs, _write_findings


def main():
    run_summary_path = RESULTS / "run_summary.csv"
    windows_path = RESULTS / "window_metrics.parquet"
    paired_effects_path = RESULTS / "paired_block_bootstrap_differences.csv"
    required = [run_summary_path, windows_path, paired_effects_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot rebuild tables; missing saved experiment artifacts: "
            + ", ".join(str(path) for path in missing)
        )

    run_summary = pd.read_csv(run_summary_path)
    windows = pd.read_parquet(windows_path)
    paired_effects = pd.read_csv(paired_effects_path)
    tradeoffs = _detector_tradeoffs(run_summary)
    tradeoffs.to_csv(RESULTS / "detector_tradeoffs.csv", index=False)
    _write_findings(windows, run_summary, tradeoffs, paired_effects)
    print("Rebuilt results/detector_tradeoffs.csv and results/experiment_findings.md from saved artifacts.")


if __name__ == "__main__":
    main()

"""Train the batch analyzer model on chronological 60/20/20 partitions."""

import json
from pathlib import Path

import joblib

from molt.data import DEFAULT_DATA_PATH, features_and_target, load_transactions, temporal_split
from molt.metrics import calculate_metrics, threshold_for_fpr
from molt.strategies import make_batch_model

ROOT = Path(__file__).resolve().parent


def main():
    frame = load_transactions(DEFAULT_DATA_PATH)
    train, validation, test = temporal_split(frame)
    X_train, y_train = features_and_target(train)
    X_val, y_val = features_and_target(validation)
    X_test, y_test = features_and_target(test)
    model = make_batch_model()
    model.fit(X_train, y_train)
    threshold = threshold_for_fpr(y_val, model.predict_proba(X_val)[:, 1], max_fpr=0.01)
    metrics = calculate_metrics(y_test, model.predict_proba(X_test)[:, 1], threshold)
    settings = {
        "threshold": threshold,
        "target_fpr": 0.01,
        "split": "chronological 60% train / 20% validation / 20% test",
        "features_excluded": ["Time", "Class"],
        "validation_rows": len(validation),
        "test_rows": len(test),
        "threshold_selected_on": "validation only",
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "molt_model.joblib")
    (ROOT / "models" / "threshold.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Saved threshold {threshold:.6f} at validation FPR <=1%")
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()

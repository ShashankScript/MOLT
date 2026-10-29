"""Compare batch classifiers on the same chronological holdout."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from molt.data import DEFAULT_DATA_PATH, features_and_target, load_transactions, temporal_split
from molt.metrics import calculate_metrics, threshold_for_fpr
from molt.strategies import make_batch_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    frame = load_transactions(DEFAULT_DATA_PATH)
    train, validation, test = temporal_split(frame)
    X_train, y_train = features_and_target(train)
    X_val, y_val = features_and_target(validation)
    X_test, y_test = features_and_target(test)
    candidates = [
        ("Logistic Regression", make_batch_model()),
        ("Random Forest", RandomForestClassifier(n_estimators=80, max_depth=18, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=80, max_depth=18, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42)),
    ]
    rows = []
    for name, model in candidates:
        model.fit(X_train, y_train)
        threshold = threshold_for_fpr(y_val, model.predict_proba(X_val)[:, 1], 0.01)
        scores = model.predict_proba(X_test)[:, 1]
        rows.append({"Model": name, "Threshold": threshold, "Max FPR": 0.01, **calculate_metrics(y_test, scores, threshold)})
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "results" / "batch_model_comparison.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()

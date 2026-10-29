"""Evaluate the batch model on the future-period holdout at 1% FPR."""

from molt.data import DEFAULT_DATA_PATH, features_and_target, load_transactions, temporal_split
from molt.metrics import calculate_metrics, threshold_for_fpr
from molt.strategies import make_batch_model


def main():
    frame = load_transactions(DEFAULT_DATA_PATH)
    train, validation, test = temporal_split(frame)
    X_train, y_train = features_and_target(train)
    X_val, y_val = features_and_target(validation)
    X_test, y_test = features_and_target(test)
    model = make_batch_model()
    model.fit(X_train, y_train)
    threshold = threshold_for_fpr(y_val, model.predict_proba(X_val)[:, 1], 0.01)
    metrics = calculate_metrics(y_test, model.predict_proba(X_test)[:, 1], threshold)
    print(f"Chronological 60/20/20; threshold chosen for validation FPR <=1%; threshold={threshold:.6f}")
    for name, value in metrics.items():
        print(f"{name:12} {value:.4f}")


if __name__ == "__main__":
    main()

"""Shared model factories and experiment strategy identifiers."""

from river import linear_model, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STRATEGIES = ("static", "periodic", "drift_triggered", "online", "oracle")
RANDOM_STATE = 42
STRATEGY_INFO = {
    "static": "Keep the initial batch model and threshold; no updates.",
    "periodic": "Retrain every three scored windows using released labels and a recent calibration window.",
    "drift_triggered": "Use calibrated PSI/KS warnings; confirm challenger promotion on a separate labeled holdout.",
    "online": "Update River only with released labels; reserve the newest labeled window for threshold calibration.",
    "oracle": "Zero-delay retraining reference; an empirical comparator, not a guaranteed upper bound.",
}


def make_batch_model(seed: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed,
        )),
    ])


def make_online_model():
    return preprocessing.StandardScaler() | linear_model.LogisticRegression()

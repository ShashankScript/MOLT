"""Semi-synthetic feature shifts for controlled, labeled drift experiments."""

import numpy as np
import pandas as pd

from .data import TARGET, features_and_target


def important_features(model, feature_names, top_n: int = 8) -> list[str]:
    # biggest standardized LR weights first
    classifier = model.named_steps["classifier"]
    order = np.argsort(np.abs(classifier.coef_[0]))[::-1]
    return [feature_names[int(i)] for i in order[: min(top_n, len(order))]]


def inject_drift(
    stream: pd.DataFrame,
    drift_type: str,
    drift_start_index: int,
    intensity: float,
    legitimate_reference: pd.DataFrame,
    important: list[str],
    ramp_rows: int = 8000,
) -> tuple[pd.DataFrame, int | None]:
    """Copy and alter rows; preserve labels/source ids and return exact start.

    Harmful shifts move fraud rows toward legitimate medians on high-weight
    features. ``benign_covariate`` scales Amount and shifts three important V
    features for every class, preserving labels while changing covariates.
    """
    if not 0 <= intensity <= 1:
        raise ValueError("intensity must be between 0 and 1")
    supported = {"none", "harmful_abrupt", "harmful_gradual", "benign_covariate"}
    if drift_type not in supported:
        raise ValueError(f"Unknown drift type: {drift_type}")
    out = stream.copy()
    if drift_type == "none":
        return out, None
    if not 0 <= drift_start_index < len(out):
        raise ValueError("drift_start_index is outside the stream")

    indices = np.arange(len(out))
    affected = indices >= drift_start_index
    if drift_type == "benign_covariate":
        out.loc[affected, "Amount"] = out.loc[affected, "Amount"] * (1 + 0.5 * intensity)
        v_features = [name for name in important if name.startswith("V")][:3]
        for name in v_features:
            scale = float(legitimate_reference[name].std(ddof=0))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            values = out[name].to_numpy(dtype=float)
            values[affected] += 0.5 * intensity * scale
            out[name] = values
        return out, int(drift_start_index)

    fraud_mask = out[TARGET].to_numpy(dtype=int) == 1
    reference_medians = legitimate_reference[important].median(numeric_only=True)
    for name in important:
        original = out[name].to_numpy(dtype=float)
        if drift_type == "harmful_abrupt":
            alpha = np.full(len(out), intensity, dtype=float)
        else:
            ramp = np.clip((indices - drift_start_index) / max(1, ramp_rows), 0, 1)
            alpha = ramp * intensity
        mask = affected & fraud_mask
        original[mask] = original[mask] * (1 - alpha[mask]) + float(reference_medians[name]) * alpha[mask]
        out[name] = original
    return out, int(drift_start_index)


def score_stream(model, frame: pd.DataFrame) -> np.ndarray:
    X, _ = features_and_target(frame)
    return model.predict_proba(X)[:, 1]

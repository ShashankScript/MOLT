"""Evaluation metrics for rare-event detection and review-cost trade-offs."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_for_fpr(y_true, scores, max_fpr: float = 0.01) -> float:
    # threshold from validation negatives so FPR stays under the ceiling
    if not 0 < max_fpr < 1:
        raise ValueError("max_fpr must be between 0 and 1.")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Threshold selection requires positive and negative validation rows.")
    fpr, tpr, thresholds = roc_curve(y, p)
    allowed = np.flatnonzero((fpr <= max_fpr) & np.isfinite(thresholds))
    if not len(allowed):
        return float(np.nextafter(np.max(p), np.inf))
    best = allowed[np.argmax(tpr[allowed])]
    return float(thresholds[best])


def recall_at_fpr(y_true, scores, max_fpr: float = 0.01) -> float:
    # best recall on the ROC curve without going over the FPR limit
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.sum() == 0 or len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def expected_cost(y_true, scores, threshold, amounts, review_fee: float = 5.0) -> float:
    # missed fraud costs its Amount, every alert costs the review fee
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    amount = np.asarray(amounts, dtype=float)
    predictions = p >= threshold
    missed_fraud_loss = amount[(y == 1) & (~predictions)].sum()
    review_cost = float(predictions.sum()) * float(review_fee)
    return float(missed_fraud_loss + review_cost)


def window_metrics(y_true, scores, threshold, amounts, max_fpr=0.01, review_fee=5.0):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    predictions = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "recall_at_fpr": recall_at_fpr(y, p, max_fpr),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "false_positives": int(fp),
        "realized_fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts": int(predictions.sum()),
        "cost": expected_cost(y, p, threshold, amounts, review_fee),
    }


def calculate_metrics(y_true, scores, threshold):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1 Score": float(f1_score(y, pred, zero_division=0)),
        "PR-AUC": float(average_precision_score(y, p)),
        "ROC-AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }

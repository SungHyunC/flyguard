"""Shared evaluation helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y, score, threshold: float) -> dict[str, float]:
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    pred = score > threshold
    return {
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "auc": roc_auc_score(y, score),
    }


def best_threshold(y, score) -> float:
    """Threshold that maximises accuracy on (y, score). Only ever call this on validation."""
    score = np.asarray(score, dtype=float)
    candidates = np.unique(np.quantile(score, np.linspace(0.005, 0.995, 399)))
    accs = [accuracy_score(y, score > t) for t in candidates]
    return float(candidates[int(np.argmax(accs))])

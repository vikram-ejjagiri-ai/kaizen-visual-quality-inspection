"""Metrics and the cost-based decision threshold."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.config import COST_FALSE_REJECT, COST_MISSED_DEFECT


def image_auroc(labels, scores):
    """Threshold-free: probability that a random defective part scores higher than a random good one."""
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks, maps):
    """Same idea per pixel: does the heatmap light up exactly where the defect is?"""
    return float(roc_auc_score(np.asarray(masks).ravel(), np.asarray(maps).ravel()))


def decisions_summary(labels, scores, threshold):
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    n_defect, n_good = int(tp + fn), int(tn + fp)
    return {
        "threshold": float(threshold),
        "defects_caught": int(tp), "defects_missed": int(fn),
        "good_passed": int(tn), "good_rejected": int(fp),
        "defect_detection_rate": tp / n_defect if n_defect else float("nan"),
        "false_reject_rate": fp / n_good if n_good else float("nan"),
        "business_cost": int(fn * COST_MISSED_DEFECT + fp * COST_FALSE_REJECT),
    }


def cost_curve(labels, scores, n_points=400):
    scores = np.asarray(scores)
    candidates = np.unique(np.concatenate([
        scores, np.linspace(scores.min(), scores.max(), n_points)]))
    costs = np.array([decisions_summary(labels, scores, t)["business_cost"] for t in candidates])
    return candidates, costs


def best_threshold(labels, scores):
    """Lowest business cost; among ties, the highest threshold (fewest false rejects).

    Note: a threshold placed exactly on the highest-scoring defect is fragile. We
    move it halfway down to the next lower score, so it has a safety margin.
    """
    thresholds, costs = cost_curve(labels, scores)
    best = np.flatnonzero(costs == costs.min())
    t = thresholds[best.max()]
    lower = np.asarray(scores)[np.asarray(scores) < t]
    if lower.size:
        t = (t + lower.max()) / 2
    return float(t), int(costs.min())

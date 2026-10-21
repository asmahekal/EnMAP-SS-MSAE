"""Evaluation metrics. Ground truth is used only here, after scores are computed."""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def metrics_from_score(y, score):
    y = np.asarray(y).astype(np.uint8).ravel(); s = np.asarray(score, float).ravel()
    if len(np.unique(y)) < 2:
        return None
    thr99 = float(np.quantile(s, 0.99)); p99 = (s >= thr99).astype(np.uint8)
    prec, rec, _ = precision_recall_curve(y, s)
    f = 2 * prec * rec / (prec + rec + 1e-12)
    fpr, tpr, _ = roc_curve(y, s)
    ids = np.where(tpr >= 0.95)[0]
    return dict(roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
                f1_p99=float(f1_score(y, p99, zero_division=0)),
                precision_p99=float(precision_score(y, p99, zero_division=0)),
                recall_p99=float(recall_score(y, p99, zero_division=0)),
                oracle_f1_posthoc=float(np.nanmax(f)),
                fpr_at_95_tpr=float(fpr[ids[0]]) if len(ids) else float('nan'))

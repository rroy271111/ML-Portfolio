"""
Metrics utilities
"""

from typing import Sequence

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, average_precision_score


def compute_pr_auc(y_true: Sequence, y_scores: Sequence) -> float:
    """
    Compute Precision-Recall AUC (area under PR curve).
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    precision, recall, __ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return float(pr_auc)


def classification_accuracy(y_true, y_pred):
    """
    Overall classification accuracy.
    Note:
     Not suitable as a primary metric for imbalanced datasets.
     Intended as a secondary / sanity-check metric.
    """

    return accuracy_score(y_true, y_pred)

"""Evaluation module for CodeGhost Risk Scoring (Milestone 4).

Computes comprehensive classification metrics using pure NumPy:
Precision, Recall, F1, ROC-AUC, PR-AUC, False Positive Rate (FPR),
Confusion Matrix, Class Balance, and Categorical Score Distributions.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from backend.risk.baseline_model import BaselineRiskModel, map_probability_to_score
from backend.risk.schemas import RiskEvaluationMetrics, RiskExample, RiskPrediction, RiskThresholds

logger = logging.getLogger(__name__)


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Compute exact ROC-AUC via Mann-Whitney U test statistic."""
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    if n_pos == 0 or n_neg == 0:
        return None

    # Rank with tie-handling (ascending)
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)

    # Average ranks for ties
    vals, inv, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
    if len(vals) < len(y_prob):
        tie_ranks = np.zeros(len(vals), dtype=float)
        np.add.at(tie_ranks, inv, ranks)
        tie_ranks /= counts
        ranks = tie_ranks[inv]

    sum_ranks_pos = np.sum(ranks[pos_mask])
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Compute PR-AUC (Average Precision) across precision-recall thresholds."""
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return None

    # Sort descending
    desc_idx = np.argsort(-y_prob)
    y_sorted = y_true[desc_idx]

    tp_cumsum = np.cumsum(y_sorted == 1)
    fp_cumsum = np.cumsum(y_sorted == 0)

    recalls = tp_cumsum / n_pos
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Add (0, 1) and calculate trapezoidal area
    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([1.0], precisions))

    # Trapezoid rule on recall diff
    area = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return float(area)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "f1",
    min_recall: float = 0.80,
) -> Tuple[float, float]:
    """Find the optimal decision threshold that maximizes F1 or minimizes FPR subject to min_recall."""
    if len(y_true) == 0:
        return 0.50, 0.0

    thresholds = np.linspace(0.10, 0.95, 86)
    best_thresh = 0.50
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        if criterion == "f1":
            if f1 > best_score:
                best_score = f1
                best_thresh = float(t)
        elif criterion == "min_fpr":
            if rec >= min_recall:
                score = 1.0 - fpr
                if score > best_score:
                    best_score = score
                    best_thresh = float(t)

    return round(best_thresh, 3), round(best_score, 4)


class RiskEvaluator:
    """Evaluates risk scoring models on test or validation splits."""

    def __init__(self, thresholds: Optional[RiskThresholds] = None):
        self.thresholds = thresholds or RiskThresholds()

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        split_name: str = "test",
        decision_threshold: float = 0.5,
        latency_ms: float = 0.0,
    ) -> RiskEvaluationMetrics:
        """
        Compute all classification metrics from ground-truth labels and predicted probabilities.

        Args:
            y_true: Binary ground-truth labels (1 = positive/buggy, 0 = negative/stable).
            y_prob: Predicted positive-class probabilities in [0.0, 1.0].
            model_name: Name of the model evaluated.
            split_name: Name of the dataset split (e.g. 'test', 'validation').
            decision_threshold: Probability threshold for binary classification.
            latency_ms: Average inference latency in milliseconds.

        Returns:
            RiskEvaluationMetrics object.
        """
        total = len(y_true)
        if total == 0:
            return RiskEvaluationMetrics(
                model_name=model_name,
                split_name=split_name,
                total_samples=0,
                positive_samples=0,
                negative_samples=0,
                class_balance_ratio=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                roc_auc=None,
                pr_auc=None,
                false_positive_rate=0.0,
                confusion_matrix=[[0, 0], [0, 0]],
                score_distribution={"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
                threshold=decision_threshold,
                latency_ms=latency_ms,
            )

        positives = int(np.sum(y_true == 1))
        negatives = int(np.sum(y_true == 0))
        balance_ratio = round(positives / total, 4)

        y_pred = (y_prob >= decision_threshold).astype(int)

        # Confusion Matrix [[TN, FP], [FN, TP]]
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        # Precision, Recall, F1
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # ROC-AUC & PR-AUC
        roc_val = compute_roc_auc(y_true, y_prob)
        pr_val = compute_pr_auc(y_true, y_prob)

        # Categorical Score Distribution
        scores = [map_probability_to_score(p) for p in y_prob]
        distribution: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for s in scores:
            cat = self.thresholds.categorize(s).value
            distribution[cat] = distribution.get(cat, 0) + 1

        return RiskEvaluationMetrics(
            model_name=model_name,
            split_name=split_name,
            total_samples=total,
            positive_samples=positives,
            negative_samples=negatives,
            class_balance_ratio=balance_ratio,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            roc_auc=round(roc_val, 4) if roc_val is not None else None,
            pr_auc=round(pr_val, 4) if pr_val is not None else None,
            false_positive_rate=round(fpr, 4),
            confusion_matrix=[[tn, fp], [fn, tp]],
            score_distribution=distribution,
            threshold=decision_threshold,
            latency_ms=round(latency_ms, 2),
        )

    def evaluate_model(
        self,
        model: BaselineRiskModel,
        test_examples: List[RiskExample],
        split_name: str = "test",
        decision_threshold: float = 0.5,
    ) -> RiskEvaluationMetrics:
        """Evaluate a BaselineRiskModel on a list of test RiskExamples."""
        if not test_examples:
            return self.evaluate_predictions(
                y_true=np.empty((0,)),
                y_prob=np.empty((0,)),
                model_name=model.name,
                split_name=split_name,
            )

        start_time = time.time()
        X_test, y_true, _ = model.feature_extractor.extract_dataset_matrix(test_examples)
        y_prob = model.predict_proba(X_test)
        elapsed_ms = ((time.time() - start_time) / len(test_examples)) * 1000.0

        return self.evaluate_predictions(
            y_true=y_true,
            y_prob=y_prob,
            model_name=model.name,
            split_name=split_name,
            decision_threshold=decision_threshold,
            latency_ms=elapsed_ms,
        )

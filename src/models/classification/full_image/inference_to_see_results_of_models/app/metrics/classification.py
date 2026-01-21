"""
Classification metrics for model evaluation.

Provides F-beta score, precision, recall, and confusion matrix utilities
for binary and multiclass classification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    """
    Container for classification metrics.

    Attributes:
        accuracy: Overall accuracy
        precision: Precision (per-class for multiclass)
        recall: Recall (per-class for multiclass)
        f1: F1 score
        f2: F2 score (emphasizes recall)
        fbeta: F-beta score with custom beta
        support: Number of samples per class
        confusion_matrix: Confusion matrix
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    fbeta: Optional[float] = None
    beta: Optional[float] = None
    support: Optional[Dict[Any, int]] = None
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
        }
        if self.fbeta is not None:
            result["fbeta"] = self.fbeta
            result["beta"] = self.beta
        if self.support is not None:
            result["support"] = self.support
        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()
        return result


def compute_fbeta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 2.0,
    pos_label: Any = 1,
    average: str = "binary"
) -> float:
    """
    Compute F-beta score.

    F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        beta: Beta value (2.0 = F2 which emphasizes recall)
        pos_label: Positive class label (for binary classification)
        average: Averaging method ('binary', 'macro', 'micro', 'weighted')

    Returns:
        F-beta score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    if len(y_true) == 0:
        return 0.0

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_labels)

    if average == "binary" or n_classes == 2:
        # Binary classification
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        beta_sq = beta ** 2
        fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
        return fbeta

    elif average == "macro":
        # Macro-average: unweighted mean across classes
        fbeta_scores = []
        for label in unique_labels:
            fbeta_scores.append(compute_fbeta(y_true, y_pred, beta, pos_label=label, average="binary"))
        return float(np.mean(fbeta_scores))

    elif average == "micro":
        # Micro-average: global TP, FP, FN
        tp_total = 0
        fp_total = 0
        fn_total = 0
        for label in unique_labels:
            tp_total += np.sum((y_true == label) & (y_pred == label))
            fp_total += np.sum((y_true != label) & (y_pred == label))
            fn_total += np.sum((y_true == label) & (y_pred != label))

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        beta_sq = beta ** 2
        return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

    elif average == "weighted":
        # Weighted average by support
        fbeta_scores = []
        weights = []
        for label in unique_labels:
            fbeta_scores.append(compute_fbeta(y_true, y_pred, beta, pos_label=label, average="binary"))
            weights.append(np.sum(y_true == label))

        weights = np.array(weights)
        if np.sum(weights) == 0:
            return 0.0

        return float(np.average(fbeta_scores, weights=weights))

    else:
        raise ValueError(f"Unknown average method: {average}")


def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Any = 1
) -> Tuple[float, float]:
    """
    Compute precision and recall for binary classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label

    Returns:
        Tuple of (precision, recall)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return float(precision), float(recall)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0.0

    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[Any]] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of labels (for ordering)

    Returns:
        Confusion matrix (n_classes, n_classes)
        Rows = true labels, columns = predicted labels
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((n_labels, n_labels), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            i = label_to_idx[true_label]
            j = label_to_idx[pred_label]
            cm[i, j] += 1

    return cm


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Any = 1,
    beta: float = 2.0,
    average: str = "binary"
) -> ClassificationMetrics:
    """
    Compute all classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label (for binary)
        beta: Beta value for F-beta score
        average: Averaging method for multiclass

    Returns:
        ClassificationMetrics instance
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_labels)

    # Determine if binary or multiclass
    is_binary = n_classes == 2 or average == "binary"

    # Compute basic metrics
    accuracy = compute_accuracy(y_true, y_pred)

    if is_binary:
        precision, recall = compute_precision_recall(y_true, y_pred, pos_label)
    else:
        # Macro average for multiclass
        precisions, recalls = [], []
        for label in unique_labels:
            p, r = compute_precision_recall(y_true, y_pred, label)
            precisions.append(p)
            recalls.append(r)
        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))

    # F-scores
    f1 = compute_fbeta(y_true, y_pred, beta=1.0, pos_label=pos_label, average=average)
    f2 = compute_fbeta(y_true, y_pred, beta=2.0, pos_label=pos_label, average=average)
    fbeta = compute_fbeta(y_true, y_pred, beta=beta, pos_label=pos_label, average=average)

    # Support
    support = {label: int(np.sum(y_true == label)) for label in unique_labels}

    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, labels=list(unique_labels))

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        f2=f2,
        fbeta=fbeta,
        beta=beta,
        support=support,
        confusion_matrix=cm
    )


def get_metric_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f2",
    pos_label: Any = 1,
    beta: float = 2.0
) -> float:
    """
    Get a single metric value by name.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric: Metric name ('accuracy', 'precision', 'recall', 'f1', 'f2', 'fbeta', 'macro_f1')
        pos_label: Positive class label
        beta: Beta for fbeta metric

    Returns:
        Metric value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metric = metric.lower()

    if metric == "accuracy":
        return compute_accuracy(y_true, y_pred)

    elif metric == "precision":
        p, _ = compute_precision_recall(y_true, y_pred, pos_label)
        return p

    elif metric == "recall":
        _, r = compute_precision_recall(y_true, y_pred, pos_label)
        return r

    elif metric == "f1":
        return compute_fbeta(y_true, y_pred, beta=1.0, pos_label=pos_label, average="binary")

    elif metric == "f2":
        return compute_fbeta(y_true, y_pred, beta=2.0, pos_label=pos_label, average="binary")

    elif metric == "fbeta":
        return compute_fbeta(y_true, y_pred, beta=beta, pos_label=pos_label, average="binary")

    elif metric == "macro_f1":
        return compute_fbeta(y_true, y_pred, beta=1.0, pos_label=pos_label, average="macro")

    elif metric == "macro_f2":
        return compute_fbeta(y_true, y_pred, beta=2.0, pos_label=pos_label, average="macro")

    else:
        raise ValueError(
            f"Unknown metric: {metric}. "
            f"Supported: accuracy, precision, recall, f1, f2, fbeta, macro_f1, macro_f2"
        )


# ============================================================================
# Sanity Checks
# ============================================================================

def _run_sanity_checks() -> bool:
    """Run sanity checks on metric functions."""
    print("Running classification metrics sanity checks...")

    # Test 1: Perfect predictions
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])

    assert compute_accuracy(y_true, y_pred) == 1.0
    assert compute_fbeta(y_true, y_pred, beta=1.0, pos_label=1) == 1.0
    print("  ✓ Perfect predictions give 1.0 scores")

    # Test 2: All wrong predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    assert compute_accuracy(y_true, y_pred) == 0.0
    assert compute_fbeta(y_true, y_pred, beta=1.0, pos_label=1) == 0.0
    print("  ✓ All wrong predictions give 0.0 scores")

    # Test 3: F2 emphasizes recall
    # High recall, low precision case
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 0, 0])  # 4 TP, 2 FP, 0 FN

    f1 = compute_fbeta(y_true, y_pred, beta=1.0, pos_label=1)
    f2 = compute_fbeta(y_true, y_pred, beta=2.0, pos_label=1)

    # With perfect recall (1.0) and precision 4/6 = 0.667
    # F2 should be higher than F1 because it weights recall more
    assert f2 > f1, f"F2 ({f2}) should be > F1 ({f1}) when recall is higher"
    print(f"  ✓ F2 ({f2:.3f}) > F1 ({f1:.3f}) when recall is high")

    # Test 4: Confusion matrix
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 2, 2])

    cm = compute_confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # True 0, pred 0
    assert cm[0, 1] == 1  # True 0, pred 1
    assert cm[1, 1] == 1  # True 1, pred 1
    assert cm[1, 2] == 1  # True 1, pred 2
    assert cm[2, 2] == 1  # True 2, pred 2
    print("  ✓ Confusion matrix computed correctly")

    # Test 5: Macro F1
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    macro_f1 = compute_fbeta(y_true, y_pred, beta=1.0, average="macro")
    assert macro_f1 == 1.0
    print("  ✓ Macro F1 works for multiclass")

    # Test 6: get_metric_value
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    assert get_metric_value(y_true, y_pred, "accuracy") == 1.0
    assert get_metric_value(y_true, y_pred, "f1") == 1.0
    assert get_metric_value(y_true, y_pred, "f2") == 1.0
    print("  ✓ get_metric_value works correctly")

    # Test 7: compute_all_metrics
    metrics = compute_all_metrics(y_true, y_pred, pos_label=1)

    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0
    assert metrics.confusion_matrix is not None
    print("  ✓ compute_all_metrics returns complete results")

    print("\n✅ All classification metrics sanity checks passed!")
    return True


if __name__ == "__main__":
    _run_sanity_checks()

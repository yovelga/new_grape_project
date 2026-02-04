"""
Comprehensive metrics reporting for binary classification.

Provides extensive evaluation metrics and serialization.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfusionMatrix:
    """Confusion matrix components."""
    TP: int  # True Positives
    FP: int  # False Positives
    TN: int  # True Negatives
    FN: int  # False Negatives
    
    @property
    def total(self) -> int:
        """Total samples."""
        return self.TP + self.FP + self.TN + self.FN
    
    @property
    def num_positive(self) -> int:
        """Total positive samples (CRACK)."""
        return self.TP + self.FN
    
    @property
    def num_negative(self) -> int:
        """Total negative samples (HEALTHY)."""
        return self.TN + self.FP
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_array(self) -> np.ndarray:
        """Convert to 2x2 array [[TN, FP], [FN, TP]]."""
        return np.array([[self.TN, self.FP], [self.FN, self.TP]])


@dataclass
class BinaryMetrics:
    """
    Comprehensive binary classification metrics.
    
    All metrics for a single dataset split (TRAIN/VAL/TEST).
    """
    # Confusion matrix
    confusion_matrix: ConfusionMatrix
    
    # Basic metrics
    accuracy: float
    balanced_accuracy: float
    
    # Positive class (CRACK) metrics
    precision: float  # PPV
    recall: float  # Sensitivity, TPR
    f1_score: float
    f2_score: float
    
    # Negative class (HEALTHY) metrics
    specificity: float  # TNR
    npv: float  # Negative Predictive Value
    
    # Correlation
    mcc: float  # Matthews Correlation Coefficient
    
    # Optional probabilistic metrics (if available)
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'confusion_matrix': self.confusion_matrix.to_dict(),
            'accuracy': float(self.accuracy),
            'balanced_accuracy': float(self.balanced_accuracy),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'f2_score': float(self.f2_score),
            'specificity': float(self.specificity),
            'npv': float(self.npv),
            'mcc': float(self.mcc),
        }
        
        if self.roc_auc is not None:
            result['roc_auc'] = float(self.roc_auc)
        if self.pr_auc is not None:
            result['pr_auc'] = float(self.pr_auc)
        
        return result
    
    def __str__(self) -> str:
        """String representation with formatted metrics."""
        lines = [
            "Binary Classification Metrics",
            "=" * 50,
            f"Accuracy:          {self.accuracy:.4f}",
            f"Balanced Accuracy: {self.balanced_accuracy:.4f}",
            "",
            "Positive Class (CRACK):",
            f"  Precision:       {self.precision:.4f}",
            f"  Recall:          {self.recall:.4f}",
            f"  F1 Score:        {self.f1_score:.4f}",
            f"  F2 Score:        {self.f2_score:.4f}",
            "",
            "Negative Class (HEALTHY):",
            f"  Specificity:     {self.specificity:.4f}",
            f"  NPV:             {self.npv:.4f}",
            "",
            f"MCC:               {self.mcc:.4f}",
        ]
        
        if self.roc_auc is not None:
            lines.append(f"ROC-AUC:           {self.roc_auc:.4f}")
        if self.pr_auc is not None:
            lines.append(f"PR-AUC:            {self.pr_auc:.4f}")
        
        lines.extend([
            "",
            "Confusion Matrix:",
            f"  TP: {self.confusion_matrix.TP:4d}  FP: {self.confusion_matrix.FP:4d}",
            f"  FN: {self.confusion_matrix.FN:4d}  TN: {self.confusion_matrix.TN:4d}",
            f"  Support - CRACK: {self.confusion_matrix.num_positive}, "
            f"HEALTHY: {self.confusion_matrix.num_negative}",
        ])
        
        return "\n".join(lines)


class MetricsCalculator:
    """Calculate comprehensive binary classification metrics."""
    
    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
        """
        Compute confusion matrix.
        
        Args:
            y_true: True labels (0=HEALTHY, 1=CRACK)
            y_pred: Predicted labels (0=HEALTHY, 1=CRACK)
            
        Returns:
            ConfusionMatrix object
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        return ConfusionMatrix(TP=int(TP), FP=int(FP), TN=int(TN), FN=int(FN))
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       y_prob: Optional[np.ndarray] = None) -> BinaryMetrics:
        """
        Compute all binary classification metrics.
        
        Args:
            y_true: True labels (0=HEALTHY, 1=CRACK)
            y_pred: Predicted labels (0=HEALTHY, 1=CRACK)
            y_prob: Optional predicted probabilities for class 1 (for ROC-AUC, PR-AUC)
            
        Returns:
            BinaryMetrics object with all metrics
        """
        # Confusion matrix
        cm = MetricsCalculator.compute_confusion_matrix(y_true, y_pred)
        
        # Handle edge cases
        if cm.total == 0:
            raise ValueError("No samples to evaluate")
        
        # Accuracy
        accuracy = (cm.TP + cm.TN) / cm.total if cm.total > 0 else 0.0
        
        # Balanced accuracy
        tpr = cm.TP / cm.num_positive if cm.num_positive > 0 else 0.0
        tnr = cm.TN / cm.num_negative if cm.num_negative > 0 else 0.0
        balanced_accuracy = (tpr + tnr) / 2.0
        
        # Precision (PPV)
        precision = cm.TP / (cm.TP + cm.FP) if (cm.TP + cm.FP) > 0 else 0.0
        
        # Recall (Sensitivity, TPR)
        recall = cm.TP / (cm.TP + cm.FN) if (cm.TP + cm.FN) > 0 else 0.0
        
        # F1 Score
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # F2 Score (weights recall higher than precision)
        beta = 2.0
        f2_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) \
            if (beta**2 * precision + recall) > 0 else 0.0
        
        # Specificity (TNR)
        specificity = cm.TN / (cm.TN + cm.FP) if (cm.TN + cm.FP) > 0 else 0.0
        
        # NPV (Negative Predictive Value)
        npv = cm.TN / (cm.TN + cm.FN) if (cm.TN + cm.FN) > 0 else 0.0
        
        # MCC (Matthews Correlation Coefficient)
        mcc_numerator = (cm.TP * cm.TN) - (cm.FP * cm.FN)
        mcc_denominator = np.sqrt((cm.TP + cm.FP) * (cm.TP + cm.FN) * 
                                 (cm.TN + cm.FP) * (cm.TN + cm.FN))
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0.0
        
        # Probabilistic metrics (optional)
        roc_auc = None
        pr_auc = None
        
        if y_prob is not None:
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                
                # ROC-AUC
                if len(np.unique(y_true)) > 1:  # Need at least 2 classes
                    roc_auc = roc_auc_score(y_true, y_prob)
                    pr_auc = average_precision_score(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Could not compute probabilistic metrics: {e}")
        
        return BinaryMetrics(
            confusion_matrix=cm,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            f2_score=f2_score,
            specificity=specificity,
            npv=npv,
            mcc=mcc,
            roc_auc=roc_auc,
            pr_auc=pr_auc
        )


class MetricsReport:
    """
    Comprehensive metrics report for multiple dataset splits.
    
    Manages metrics for TRAIN/VAL/TEST and provides serialization.
    """
    
    def __init__(self):
        """Initialize empty report."""
        self.metrics: Dict[str, BinaryMetrics] = {}
        self.metadata: Dict = {}
    
    def add_split(self, 
                  split_name: str,
                  y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_prob: Optional[np.ndarray] = None) -> BinaryMetrics:
        """
        Add metrics for a dataset split.
        
        Args:
            split_name: Name of split (e.g., 'train', 'val', 'test')
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Optional predicted probabilities
            
        Returns:
            Computed BinaryMetrics
        """
        metrics = MetricsCalculator.compute_metrics(y_true, y_pred, y_prob)
        self.metrics[split_name] = metrics
        logger.info(f"Added metrics for split '{split_name}'")
        return metrics
    
    def get_split(self, split_name: str) -> Optional[BinaryMetrics]:
        """Get metrics for a specific split."""
        return self.metrics.get(split_name)
    
    def set_metadata(self, **kwargs) -> None:
        """Set metadata fields."""
        self.metadata.update(kwargs)
    
    def to_dict(self) -> Dict:
        """Convert entire report to dictionary."""
        result = {
            'metadata': self.metadata,
            'splits': {}
        }
        
        for split_name, metrics in self.metrics.items():
            result['splits'][split_name] = metrics.to_dict()
        
        return result
    
    def save_json(self, filepath: str) -> None:
        """
        Save report to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved metrics report to {filepath}")
    
    def save_confusion_matrices_csv(self, filepath: str) -> None:
        """
        Save confusion matrices to CSV.
        
        Args:
            filepath: Path to output CSV file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for split_name, metrics in self.metrics.items():
            cm = metrics.confusion_matrix
            rows.append({
                'split': split_name,
                'TP': cm.TP,
                'FP': cm.FP,
                'TN': cm.TN,
                'FN': cm.FN,
                'support_crack': cm.num_positive,
                'support_healthy': cm.num_negative
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved confusion matrices to {filepath}")
    
    def __str__(self) -> str:
        """String representation of full report."""
        lines = ["Metrics Report", "=" * 70, ""]
        
        if self.metadata:
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        for split_name, metrics in self.metrics.items():
            lines.append(f"Split: {split_name.upper()}")
            lines.append("-" * 70)
            lines.append(str(metrics))
            lines.append("")
        
        return "\n".join(lines)

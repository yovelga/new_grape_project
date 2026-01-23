"""
Optuna-based hyperparameter tuner for patch-based classifier.

Tunes postprocessing and patch decision hyperparameters using Optuna.
"""

import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
import logging
import json

from .dataset_loader import Sample
from .postprocess_patch import PatchClassifierParams, PostprocessPatchClassifier, ClassificationResult
from .metrics_report import MetricsReport, MetricsCalculator
from .inference_cache import InferenceCache
from .search_space import HyperparameterSearchSpace

logger = logging.getLogger(__name__)


class OptunaTuner:
    """
    Optuna hyperparameter tuner for postprocess + patch classifier.
    
    Tunes:
    - pixel_threshold
    - min_blob_area
    - morph_size
    - patch_size
    - patch_crack_pct_threshold
    
    to maximize F1 or F2 score on validation set.
    """
    
    def __init__(self,
                 train_samples: List[Sample],
                 val_samples: List[Sample],
                 test_samples: List[Sample],
                 inference_fn: Callable[[str], np.ndarray],
                 optimize_metric: str = 'f1',
                 output_dir: Optional[str] = None,
                 search_space: Optional[HyperparameterSearchSpace] = None):
        """
        Initialize Optuna tuner.
        
        Args:
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
            inference_fn: Function that takes image path and returns prob_map (H, W) or (H, W, C)
            optimize_metric: Metric to optimize ('f1' or 'f2')
            output_dir: Directory to save results
            search_space: Hyperparameter search space (uses default if None)
        """
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.inference_fn = inference_fn
        self.optimize_metric = optimize_metric.lower()
        
        if self.optimize_metric not in ['f1', 'f2']:
            raise ValueError(f"optimize_metric must be 'f1' or 'f2', got '{optimize_metric}'")
        
        # Setup search space
        self.search_space = search_space if search_space is not None else HyperparameterSearchSpace()
        self.search_space.validate_all()
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"outputs/optuna/run_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache = InferenceCache(max_cache_size_mb=2048)  # 2GB cache
        
        # Trial history
        self.trial_history: List[Dict] = []
        
        # Best results
        self.best_params: Optional[Dict] = None
        self.best_score: float = 0.0
        self.best_trial_number: int = -1
        
        # Study object
        self.study: Optional[optuna.Study] = None
        
        logger.info(f"OptunaTuner initialized")
        logger.info(f"  Train: {len(train_samples)} samples")
        logger.info(f"  Val: {len(val_samples)} samples")
        logger.info(f"  Test: {len(test_samples)} samples")
        logger.info(f"  Optimize: {optimize_metric.upper()}")
        logger.info(f"  Output: {self.output_dir}")
    
    def _get_prob_map(self, sample: Sample) -> np.ndarray:
        """
        Get probability map for sample (with caching).
        
        Args:
            sample: Sample object
            
        Returns:
            Probability map (H, W) or (H, W, C)
        """
        # Check cache
        prob_map = self.cache.get(sample.path)
        
        if prob_map is None:
            # Run inference
            prob_map = self.inference_fn(sample.path)
            
            # Cache result
            self.cache.put(sample.path, prob_map)
        
        return prob_map
    
    def _evaluate_samples(self, 
                         samples: List[Sample],
                         params: PatchClassifierParams) -> Dict[str, Any]:
        """
        Evaluate samples with given parameters.
        
        Args:
            samples: List of samples to evaluate
            params: Classifier parameters
            
        Returns:
            Dictionary with predictions and labels
        """
        classifier = PostprocessPatchClassifier(params)
        
        y_true = []
        y_pred = []
        results = []
        
        for sample in samples:
            # Get probability map
            prob_map = self._get_prob_map(sample)
            
            # Classify
            result = classifier.classify(prob_map)
            
            # Store
            y_true.append(sample.label)
            y_pred.append(result.predicted_label)
            results.append({
                'sample_id': sample.sample_id,
                'path': sample.path,
                'true_label': sample.label,
                'pred_label': result.predicted_label,
                'max_patch_crack_pct': result.max_patch_crack_pct,
                'num_flagged_patches': result.num_flagged_patches,
                'total_patches': result.total_patches,
                'num_blobs': result.num_blobs,
                'max_blob_area': result.max_blob_area,
                'total_crack_pixels': result.total_crack_pixels
            })
        
        return {
            'y_true': np.array(y_true),
            'y_pred': np.array(y_pred),
            'results': results
        }
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Metric value to maximize
        """
        # Sample hyperparameters using search space
        suggested = self.search_space.suggest_all(trial)
        
        # Create params
        params = PatchClassifierParams(
            pixel_threshold=suggested['pixel_threshold'],
            min_blob_area=suggested['min_blob_area'],
            morph_size=suggested['morph_size'],
            patch_size=suggested['patch_size'],
            patch_crack_pct_threshold=suggested['patch_crack_pct_threshold']
        )
        
        # Evaluate on validation set
        eval_result = self._evaluate_samples(self.val_samples, params)
        
        # Compute metrics
        metrics = MetricsCalculator.compute_metrics(
            eval_result['y_true'],
            eval_result['y_pred']
        )
        
        # Get score based on optimize_metric
        if self.optimize_metric == 'f1':
            score = metrics.f1_score
        else:  # f2
            score = metrics.f2_score
        
        # Store comprehensive trial info with all metrics
        trial_info = {
            'trial_number': trial.number,
            'params': suggested.copy(),
            'score': float(score),
            'metric': self.optimize_metric,
            # All validation metrics
            'val_accuracy': float(metrics.accuracy),
            'val_balanced_accuracy': float(metrics.balanced_accuracy),
            'val_precision': float(metrics.precision),
            'val_recall': float(metrics.recall),
            'val_f1': float(metrics.f1_score),
            'val_f2': float(metrics.f2_score),
            'val_specificity': float(metrics.specificity),
            'val_npv': float(metrics.npv),
            'val_mcc': float(metrics.mcc),
            # Confusion matrix
            'val_TP': metrics.confusion_matrix.TP,
            'val_FP': metrics.confusion_matrix.FP,
            'val_TN': metrics.confusion_matrix.TN,
            'val_FN': metrics.confusion_matrix.FN
        }
        
        # Add probabilistic metrics if available
        if metrics.roc_auc is not None:
            trial_info['val_roc_auc'] = float(metrics.roc_auc)
        if metrics.pr_auc is not None:
            trial_info['val_pr_auc'] = float(metrics.pr_auc)
        
        self.trial_history.append(trial_info)
        
        # Enhanced logging with key metrics
        logger.info(
            f"Trial {trial.number}: {self.optimize_metric.upper()}={score:.4f} | "
            f"Acc={metrics.accuracy:.3f} | Prec={metrics.precision:.3f} | "
            f"Rec={metrics.recall:.3f} | F1={metrics.f1_score:.3f} | F2={metrics.f2_score:.3f}"
        )
        
        return score
    
    def run_study(self,
                  n_trials: int = 100,
                  seed: Optional[int] = None,
                  n_jobs: int = 1,
                  timeout: Optional[int] = None,
                  callbacks: Optional[List[Callable]] = None) -> optuna.Study:
        """
        Run Optuna study.
        
        Args:
            n_trials: Number of trials to run
            seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (1 = sequential)
            timeout: Timeout in seconds (None = no limit)
            callbacks: Optional list of callback functions
            
        Returns:
            Completed Optuna study
        """
        logger.info(f"Starting Optuna study with {n_trials} trials")
        
        # Create study
        sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else None
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'postprocess_patch_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=False  # We'll handle progress in UI
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        self.best_trial_number = self.study.best_trial.number
        
        # Convert morph_size_val back to actual morph_size
        if 'morph_size_val' in self.best_params:
            morph_size_val = self.best_params.pop('morph_size_val')
            self.best_params['morph_size'] = morph_size_val * 2 + 1 if morph_size_val > 0 else 0
        
        logger.info(f"Study complete. Best trial: {self.best_trial_number}, "
                   f"Best {self.optimize_metric.upper()}: {self.best_score:.4f}")
        
        return self.study
    
    def evaluate_final(self) -> MetricsReport:
        """
        Evaluate final model on all splits using best parameters.
        
        Returns:
            MetricsReport with results for TRAIN/VAL/TEST
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run study first.")
        
        logger.info("Evaluating with best parameters on all splits...")
        
        # Create params from best
        params = PatchClassifierParams(**self.best_params)
        
        # Create metrics report
        report = MetricsReport()
        report.set_metadata(
            optimize_metric=self.optimize_metric,
            best_trial=self.best_trial_number,
            best_score=self.best_score,
            best_params=self.best_params,
            timestamp=datetime.now().isoformat()
        )
        
        # Evaluate each split
        all_predictions = {}
        
        for split_name, samples in [('train', self.train_samples),
                                     ('val', self.val_samples),
                                     ('test', self.test_samples)]:
            logger.info(f"Evaluating {split_name} split ({len(samples)} samples)...")
            
            eval_result = self._evaluate_samples(samples, params)
            
            # Add metrics
            report.add_split(
                split_name,
                eval_result['y_true'],
                eval_result['y_pred']
            )
            
            # Store predictions
            all_predictions[split_name] = eval_result['results']
        
        # Save detailed predictions for test set
        test_df = pd.DataFrame(all_predictions['test'])
        test_predictions_path = self.output_dir / 'predictions_test.csv'
        test_df.to_csv(test_predictions_path, index=False)
        logger.info(f"Saved test predictions to {test_predictions_path}")
        
        return report
    
    def save_results(self, report: MetricsReport) -> None:
        """
        Save all results to disk.
        
        Args:
            report: MetricsReport to save
        """
        logger.info(f"Saving results to {self.output_dir}")
        
        # Save best params
        best_params_path = self.output_dir / 'study_best_params.json'
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_trial': self.best_trial_number,
                'best_score': self.best_score,
                'best_params': self.best_params,
                'optimize_metric': self.optimize_metric
            }, f, indent=2)
        
        # Save trial history
        trial_history_path = self.output_dir / 'trial_history.csv'
        trial_df = pd.DataFrame(self.trial_history)
        # Flatten params dict
        for key in ['pixel_threshold', 'min_blob_area', 'morph_size', 'patch_size', 'patch_crack_pct_threshold']:
            trial_df[key] = trial_df['params'].apply(lambda x: x.get(key))
        trial_df = trial_df.drop('params', axis=1)
        trial_df.to_csv(trial_history_path, index=False)
        
        # Save metrics report
        report.save_json(self.output_dir / 'final_metrics.json')
        report.save_confusion_matrices_csv(self.output_dir / 'confusion_matrices.csv')
        
        # Save cache stats
        cache_stats_path = self.output_dir / 'cache_stats.json'
        with open(cache_stats_path, 'w') as f:
            json.dump(self.cache.get_stats(), f, indent=2)
        
        logger.info(f"All results saved to {self.output_dir}")
    
    def get_summary(self) -> str:
        """
        Get text summary of tuning results.
        
        Returns:
            Formatted summary string
        """
        if self.best_params is None:
            return "No tuning results available. Run study first."
        
        lines = [
            "Optuna Hyperparameter Tuning Summary",
            "=" * 70,
            "",
            f"Optimization Metric: {self.optimize_metric.upper()}",
            f"Number of Trials: {len(self.trial_history)}",
            f"Best Trial: {self.best_trial_number}",
            f"Best Score: {self.best_score:.4f}",
            "",
            "Best Parameters:",
            "-" * 70
        ]
        
        for key, value in self.best_params.items():
            if isinstance(value, float):
                lines.append(f"  {key:30s}: {value:.6f}")
            else:
                lines.append(f"  {key:30s}: {value}")
        
        lines.extend([
            "",
            f"Cache Statistics:",
            "-" * 70,
            f"  {str(self.cache)}",
            "",
            f"Results saved to: {self.output_dir}",
        ])
        
        return "\n".join(lines)

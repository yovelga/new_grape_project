"""
Optuna worker thread for background execution.

Runs Optuna study in a QThread to keep UI responsive.
"""

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import logging
from typing import List, Optional, Callable
import traceback

from .dataset_loader import Sample
from .optuna_tuner import OptunaTuner
from .search_space import HyperparameterSearchSpace

logger = logging.getLogger(__name__)


class OptunaWorker(QThread):
    """
    Worker thread for running Optuna study.
    
    Signals:
        trial_started: Emitted when trial starts (trial_number: int)
        trial_completed: Emitted when trial completes (trial_number: int, score: float, params: dict)
        progress_update: Emitted for progress (message: str)
        study_completed: Emitted when study finishes successfully
        error_occurred: Emitted when error occurs (error_msg: str)
        final_results: Emitted with final metrics report (report_str: str)
    """
    
    # Signals
    trial_started = pyqtSignal(int)  # trial_number
    trial_completed = pyqtSignal(int, float, dict)  # trial_number, score, params
    progress_update = pyqtSignal(str)  # message
    study_completed = pyqtSignal()
    error_occurred = pyqtSignal(str)  # error_msg
    final_results = pyqtSignal(str)  # report_str
    
    def __init__(self,
                 train_samples: List[Sample],
                 val_samples: List[Sample],
                 test_samples: List[Sample],
                 inference_fn: Callable[[str], np.ndarray],
                 optimize_metric: str = 'f1',
                 n_trials: int = 100,
                 seed: Optional[int] = None,
                 n_jobs: int = 1,
                 timeout: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 search_space: Optional[HyperparameterSearchSpace] = None):
        """
        Initialize Optuna worker.
        
        Args:
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
            inference_fn: Function that takes image path and returns prob_map
            optimize_metric: Metric to optimize ('f1' or 'f2')
            n_trials: Number of trials to run
            seed: Random seed
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds
            output_dir: Output directory for results
            search_space: Hyperparameter search space (uses default if None)
        """
        super().__init__()
        
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.inference_fn = inference_fn
        self.optimize_metric = optimize_metric
        self.n_trials = n_trials
        self.seed = seed
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.output_dir = output_dir
        self.search_space = search_space
        
        self._stop_requested = False
        self.tuner: Optional[OptunaTuner] = None
    
    def request_stop(self):
        """Request worker to stop (graceful cancellation)."""
        self._stop_requested = True
        logger.info("Stop requested for Optuna worker")
    
    def _trial_callback(self, study, trial):
        """
        Callback called after each trial.
        
        Args:
            study: Optuna study
            trial: Completed trial
        """
        # Check if stop requested
        if self._stop_requested:
            study.stop()
            return
        
        # Emit signals
        self.trial_completed.emit(
            trial.number,
            trial.value,
            trial.params
        )
        
        # Get comprehensive metrics from trial history
        if self.tuner and len(self.tuner.trial_history) > 0:
            trial_info = self.tuner.trial_history[-1]  # Latest trial
            
            # Build single-line compact message with train+val metrics and parameters
            msg = (
                f"T{trial.number + 1}/{self.n_trials}: {self.optimize_metric.upper()}={trial.value:.4f} (Best={study.best_value:.4f}@T{study.best_trial.number + 1}) | "
                f"VAL[Acc={trial_info.get('val_accuracy', 0):.3f} P={trial_info.get('val_precision', 0):.3f} "
                f"R={trial_info.get('val_recall', 0):.3f} F1={trial_info.get('val_f1', 0):.3f} "
                f"F2={trial_info.get('val_f2', 0):.3f} MCC={trial_info.get('val_mcc', 0):.3f} "
                f"TP={trial_info.get('val_TP', 0)} FP={trial_info.get('val_FP', 0)} "
                f"FN={trial_info.get('val_FN', 0)} TN={trial_info.get('val_TN', 0)}] | "
                f"TRAIN[Acc={trial_info.get('train_accuracy', 0):.3f} P={trial_info.get('train_precision', 0):.3f} "
                f"R={trial_info.get('train_recall', 0):.3f} F1={trial_info.get('train_f1', 0):.3f} "
                f"F2={trial_info.get('train_f2', 0):.3f} MCC={trial_info.get('train_mcc', 0):.3f}] | "
                f"PARAMS[pxl_th={trial.params.get('pixel_threshold', 0):.4f} "
                f"area={trial.params.get('min_blob_area', 0)} "
                f"morph={trial.params.get('morph_size', 0)} "
                f"patch={trial.params.get('patch_size', 0)} "
                f"patch_th={trial.params.get('patch_crack_pct_threshold', 0):.2f} "
                f"global_th={trial.params.get('global_crack_pct_threshold', 0):.2f}]"
            )
        else:
            # Fallback if trial history not available
            best_value = study.best_value
            msg = (f"Trial {trial.number + 1}/{self.n_trials} | "
                   f"Score: {trial.value:.4f} | "
                   f"Best: {best_value:.4f}")
        
        self.progress_update.emit(msg)
    
    def run(self):
        """Run Optuna study in thread."""
        try:
            # Emit start message
            self.progress_update.emit(f"Initializing Optuna tuner...")
            self.progress_update.emit(f"Train: {len(self.train_samples)} samples")
            self.progress_update.emit(f"Val: {len(self.val_samples)} samples")
            self.progress_update.emit(f"Test: {len(self.test_samples)} samples")
            self.progress_update.emit(f"Optimizing: {self.optimize_metric.upper()}")
            self.progress_update.emit(f"Trials: {self.n_trials}")
            self.progress_update.emit("")
            
            # Create tuner
            self.tuner = OptunaTuner(
                train_samples=self.train_samples,
                val_samples=self.val_samples,
                test_samples=self.test_samples,
                inference_fn=self.inference_fn,
                optimize_metric=self.optimize_metric,
                output_dir=self.output_dir,
                search_space=self.search_space
            )
            
            # Pre-cache all probability maps
            self.progress_update.emit("Pre-caching inference results...")
            total_samples = len(self.train_samples) + len(self.val_samples) + len(self.test_samples)
            cached = 0
            
            for samples in [self.train_samples, self.val_samples, self.test_samples]:
                for sample in samples:
                    if self._stop_requested:
                        self.progress_update.emit("Caching interrupted by stop request")
                        return
                    
                    # Get and cache
                    self.tuner._get_prob_map(sample)
                    cached += 1
                    
                    if cached % 10 == 0:
                        self.progress_update.emit(f"Cached {cached}/{total_samples} samples...")
            
            self.progress_update.emit(f"Cached all {total_samples} samples")
            self.progress_update.emit(f"Cache stats: {self.tuner.cache}")
            self.progress_update.emit("")
            self.progress_update.emit("Starting Optuna optimization...")
            self.progress_update.emit("-" * 60)
            
            # Run study with callback
            callbacks = [self._trial_callback]
            
            self.tuner.run_study(
                n_trials=self.n_trials,
                seed=self.seed,
                n_jobs=self.n_jobs,
                timeout=self.timeout,
                callbacks=callbacks
            )
            
            # Check if stopped
            if self._stop_requested:
                self.progress_update.emit("")
                self.progress_update.emit("Study stopped by user request")
                return
            
            # Study completed
            self.progress_update.emit("")
            self.progress_update.emit("-" * 60)
            self.progress_update.emit("Study completed!")
            self.progress_update.emit("")
            self.progress_update.emit(self.tuner.get_summary())
            self.progress_update.emit("")
            
            self.study_completed.emit()
            
            # Evaluate on all splits
            self.progress_update.emit("=" * 60)
            self.progress_update.emit("Evaluating best parameters on all splits...")
            self.progress_update.emit("")
            
            report = self.tuner.evaluate_final()
            
            # Save results
            self.progress_update.emit("Saving results...")
            self.tuner.save_results(report)
            
            # Emit final results
            self.progress_update.emit("")
            self.progress_update.emit("=" * 60)
            self.progress_update.emit("FINAL RESULTS")
            self.progress_update.emit("=" * 60)
            
            report_str = str(report)
            self.progress_update.emit(report_str)
            self.final_results.emit(report_str)
            
            self.progress_update.emit("")
            self.progress_update.emit(f"All results saved to: {self.tuner.output_dir}")
            self.progress_update.emit("")
            self.progress_update.emit("Done!")
            
        except Exception as e:
            error_msg = f"Error in Optuna worker: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

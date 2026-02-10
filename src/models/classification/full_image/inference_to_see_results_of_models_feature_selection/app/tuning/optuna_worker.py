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
                 calibration_samples: List[Sample],
                 test_samples: List[Sample],
                 inference_fn: Callable[[str], np.ndarray],
                 optimize_metric: str = 'f1',
                 n_trials: int = 100,
                 seed: Optional[int] = None,
                 n_jobs: int = 1,
                 timeout: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 search_space: Optional[HyperparameterSearchSpace] = None,
                 model_path: Optional[str] = None,
                 calibration_csv_path: Optional[str] = None,
                 test_csv_path: Optional[str] = None):
        """
        Initialize Optuna worker.
        
        Args:
            calibration_samples: Calibration samples (used for hyperparameter tuning)
            test_samples: Test samples (used for final evaluation)
            inference_fn: Function that takes image path and returns prob_map
            optimize_metric: Metric to optimize ('f1' or 'f2')
            n_trials: Number of trials to run
            seed: Random seed
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds
            output_dir: Output directory for results
            search_space: Hyperparameter search space (uses default if None)
            model_path: Full path to the model file used for inference
            calibration_csv_path: Path to the calibration CSV dataset file
            test_csv_path: Path to the test CSV dataset file
        """
        super().__init__()
        
        self.calibration_samples = calibration_samples
        self.test_samples = test_samples
        self.inference_fn = inference_fn
        self.optimize_metric = optimize_metric
        self.n_trials = n_trials
        self.seed = seed
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.output_dir = output_dir
        self.search_space = search_space
        
        # Model and dataset paths for metadata
        self.model_path = model_path
        self.calibration_csv_path = calibration_csv_path
        self.test_csv_path = test_csv_path
        
        # Collect log messages for saving to file
        self.log_messages: List[str] = []
        
        self._stop_requested = False
        self.tuner: Optional[OptunaTuner] = None
    
    def request_stop(self):
        """Request worker to stop (graceful cancellation)."""
        self._stop_requested = True
        logger.info("Stop requested for Optuna worker")
    
    def _log_and_emit(self, message: str):
        """Log message and emit to UI, also store for file saving."""
        self.log_messages.append(message)
        self.progress_update.emit(message)
    
    def _save_log_file(self):
        """Save collected log messages to a file in the output directory."""
        if self.tuner is None or not self.log_messages:
            return
        
        try:
            from pathlib import Path
            log_path = Path(self.tuner.output_dir) / 'optuna_run.log'
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log_messages))
            logger.info(f"Log saved to {log_path}")
        except Exception as e:
            logger.error(f"Failed to save log file: {e}")
    
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
            
            # Build single-line compact message with calibration metrics and parameters
            msg = (
                f"T{trial.number + 1}/{self.n_trials}: {self.optimize_metric.upper()}={trial.value:.4f} (Best={study.best_value:.4f}@T{study.best_trial.number + 1}) | "
                f"CAL[Acc={trial_info.get('calibration_accuracy', 0):.3f} P={trial_info.get('calibration_precision', 0):.3f} "
                f"R={trial_info.get('calibration_recall', 0):.3f} F1={trial_info.get('calibration_f1', 0):.3f} "
                f"F2={trial_info.get('calibration_f2', 0):.3f} MCC={trial_info.get('calibration_mcc', 0):.3f} "
                f"TP={trial_info.get('calibration_TP', 0)} FP={trial_info.get('calibration_FP', 0)} "
                f"FN={trial_info.get('calibration_FN', 0)} TN={trial_info.get('calibration_TN', 0)}] | "
                f"PARAMS[pxl_th={trial.params.get('pixel_threshold', 0):.4f} "
                f"min_area={trial.params.get('min_blob_area', 0)} "
                f"max_area={trial.params.get('max_blob_area', 'None')} "
                f"morph={trial.params.get('morph_size', 0)} "
                f"patch={trial.params.get('patch_size', 0)} "
                f"patch_th={trial.params.get('patch_crack_pct_threshold', 0):.2f} "
                f"global_th={trial.params.get('global_crack_pct_threshold', 0):.2f}]"
            )
            
            # Add test metrics if available (every 20 trials - monitoring only)
            if 'test_f1' in trial_info:
                msg += (
                    f" | 🔍TEST[Acc={trial_info.get('test_accuracy', 0):.3f} "
                    f"P={trial_info.get('test_precision', 0):.3f} "
                    f"R={trial_info.get('test_recall', 0):.3f} "
                    f"F1={trial_info.get('test_f1', 0):.3f} "
                    f"F2={trial_info.get('test_f2', 0):.3f}]"
                )
        else:
            # Fallback if trial history not available
            best_value = study.best_value
            msg = (f"Trial {trial.number + 1}/{self.n_trials} | "
                   f"Score: {trial.value:.4f} | "
                   f"Best: {best_value:.4f}")
        
        self._log_and_emit(msg)
    
    def run(self):
        """Run Optuna study in thread."""
        try:
            # Emit start message
            self._log_and_emit(f"Initializing Optuna tuner...")
            self._log_and_emit(f"Model: {self.model_path}")
            self._log_and_emit(f"Calibration CSV: {self.calibration_csv_path}")
            self._log_and_emit(f"Test CSV: {self.test_csv_path}")
            self._log_and_emit(f"Calibration: {len(self.calibration_samples)} samples")
            self._log_and_emit(f"Test: {len(self.test_samples)} samples")
            self._log_and_emit(f"Optimizing: {self.optimize_metric.upper()}")
            self._log_and_emit(f"Trials: {self.n_trials}")
            self._log_and_emit("")
            
            # Create tuner
            self.tuner = OptunaTuner(
                calibration_samples=self.calibration_samples,
                test_samples=self.test_samples,
                inference_fn=self.inference_fn,
                optimize_metric=self.optimize_metric,
                output_dir=self.output_dir,
                search_space=self.search_space,
                model_path=self.model_path,
                calibration_csv_path=self.calibration_csv_path,
                test_csv_path=self.test_csv_path
            )
            
            # Pre-cache all probability maps
            self._log_and_emit("Pre-caching inference results...")
            total_samples = len(self.calibration_samples) + len(self.test_samples)
            cached = 0
            
            for samples in [self.calibration_samples, self.test_samples]:
                for sample in samples:
                    if self._stop_requested:
                        self._log_and_emit("Caching interrupted by stop request")
                        self._save_log_file()
                        return
                    
                    # Get and cache
                    self.tuner._get_prob_map(sample)
                    cached += 1
                    
                    if cached % 10 == 0:
                        self._log_and_emit(f"Cached {cached}/{total_samples} samples...")
            
            self._log_and_emit(f"Cached all {total_samples} samples")
            self._log_and_emit(f"Cache stats: {self.tuner.cache}")
            self._log_and_emit("")
            self._log_and_emit("Starting Optuna optimization...")
            self._log_and_emit("-" * 60)
            
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
                self._log_and_emit("")
                self._log_and_emit("Study stopped by user request")
                self._save_log_file()
                return
            
            # Study completed
            self._log_and_emit("")
            self._log_and_emit("-" * 60)
            self._log_and_emit("Study completed!")
            self._log_and_emit("")
            self._log_and_emit(self.tuner.get_summary())
            self._log_and_emit("")
            
            self.study_completed.emit()
            
            # Evaluate on all splits
            self._log_and_emit("=" * 60)
            self._log_and_emit("Evaluating best parameters on all splits...")
            self._log_and_emit("")
            
            report = self.tuner.evaluate_final()
            
            # Save results
            self._log_and_emit("Saving results...")
            self.tuner.save_results(report)
            
            # Emit final results
            self._log_and_emit("")
            self._log_and_emit("=" * 60)
            self._log_and_emit("FINAL RESULTS")
            self._log_and_emit("=" * 60)
            
            report_str = str(report)
            self._log_and_emit(report_str)
            self.final_results.emit(report_str)
            
            self._log_and_emit("")
            self._log_and_emit(f"All results saved to: {self.tuner.output_dir}")
            self._log_and_emit("")
            self._log_and_emit("Done!")
            
            # Save log file to output directory
            self._save_log_file()
            
        except Exception as e:
            error_msg = f"Error in Optuna worker: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self._log_and_emit(f"ERROR: {error_msg}")
            self._save_log_file()
            self.error_occurred.emit(error_msg)

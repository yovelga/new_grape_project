"""
Optuna hyperparameter tuning tab widget - Enhanced version.

Provides UI for CSV-based Optuna hyperparameter tuning with:
- Improved column mapping with help text
- Smart file dialog defaults
- Editable hyperparameter search space
- Comprehensive metrics output
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QLineEdit, QTextEdit,
    QFileDialog, QMessageBox, QSplitter, QDoubleSpinBox, QGridLayout,
    QScrollArea, QSizePolicy, QMainWindow
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QFont

from pathlib import Path
from typing import Optional, List
import logging
import traceback
import numpy as np

from app.tuning.dataset_loader import DatasetCSVLoader, Sample
from app.tuning.optuna_worker import OptunaWorker
from app.tuning.search_space import HyperparameterSearchSpace
from app.ui.hyperparam_cards import FloatRangeCard, IntRangeCard, CategoricalCard
from app.models import ModelManager
from app.config.settings import settings

logger = logging.getLogger(__name__)


class OptunaTabWidget(QWidget):
    """
    Optuna hyperparameter tuning tab - Enhanced.
    
    Features:
    - CSV loading with smart defaults
    - Column mapping with help text
    - Editable hyperparameter search space
    - Live progress and comprehensive results
    """
    
    # Default directories
    DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
    DEFAULT_EXPERIMENTS_DIR = Path("C:/Users/yovel/Desktop/Grape_Project/experiments")
    
    def __init__(self, model_manager: ModelManager, parent=None):
        """
        Initialize Optuna tab.
        
        Args:
            model_manager: Shared model manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Model manager (shared across tabs)
        self.model_manager = model_manager
        
        # State
        self.calibration_loader: Optional[DatasetCSVLoader] = None
        self.test_loader: Optional[DatasetCSVLoader] = None
        
        self.calibration_samples: List[Sample] = []
        self.test_samples: List[Sample] = []
        
        self.worker: Optional[OptunaWorker] = None
        
        # Search space for hyperparameters
        self.search_space = HyperparameterSearchSpace()
        
        # Last directories for file dialogs
        self.last_csv_dir = str(self.DEFAULT_DATA_DIR) if self.DEFAULT_DATA_DIR.exists() else str(Path.home())
        self.last_model_dir = str(settings.models_dir) if settings.models_dir.exists() else str(Path.home())
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Horizontal splitter: Controls (left) | Progress (right)
        splitter = QSplitter(Qt.Horizontal)
        
        # === LEFT PANEL: Controls wrapped in QScrollArea ===
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(1050)  # Prevent squeezing on ultrawide
        
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        
        # === Model Selection (NEW - at top) ===
        controls_layout.addWidget(self._create_model_selection_group())
        
        # === Dataset Loading ===
        controls_layout.addWidget(self._create_dataset_group())
        
        # === Column Mapping with Help ===
        controls_layout.addWidget(self._create_mapping_group())
        
        # === Hyperparameter Search Space ===
        controls_layout.addWidget(self._create_search_space_group())
        
        # === Tuning Configuration ===
        controls_layout.addWidget(self._create_tuning_config_group())
        
        # === Run Controls ===
        controls_layout.addLayout(self._create_run_controls())
        
        # No vertical stretch - let scroll area handle overflow
        scroll_area.setWidget(controls_widget)
        
        # === RIGHT PANEL: Progress and Results ===
        progress_widget = QWidget()
        progress_widget.setMinimumWidth(1000)  # Wider for single-line logs
        
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(8, 8, 8, 8)
        progress_layout.setSpacing(6)
        
        # Progress title
        progress_title = QLabel("Tuning Progress and Results")
        progress_title.setFont(QFont("Arial", 11, QFont.Bold))
        progress_layout.addWidget(progress_title)
        
        # Progress text area
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setFont(QFont("Consolas", 9))
        self.progress_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        self.progress_text.setLineWrapMode(QTextEdit.NoWrap)  # No wrap for single-line logs
        self.progress_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        progress_layout.addWidget(self.progress_text, stretch=1)
        
        # Clear button (compact)
        clear_btn = QPushButton("Clear Log")
        clear_btn.setMaximumWidth(120)
        clear_btn.clicked.connect(self.progress_text.clear)
        progress_layout.addWidget(clear_btn)
        
        # Add to splitter
        splitter.addWidget(scroll_area)  # Left: controls in scroll area
        splitter.addWidget(progress_widget)  # Right: log panel
        
        # Stretch factors: give more space to right panel for wide single-line logs
        splitter.setStretchFactor(0, 2)  # Left - less space
        splitter.setStretchFactor(1, 5)  # Right - more space for wide logs
        
        # Restore splitter position from settings
        settings = QSettings("GrapeProject", "OptunaUI")
        if settings.contains("optuna_splitter_state"):
            splitter.restoreState(settings.value("optuna_splitter_state"))
        
        # Save splitter state on change
        self.splitter = splitter
        splitter.splitterMoved.connect(self._save_splitter_state)
        
        main_layout.addWidget(splitter)
    
    def _save_splitter_state(self):
        """Save splitter position to settings."""
        settings = QSettings("GrapeProject", "OptunaUI")
        settings.setValue("optuna_splitter_state", self.splitter.saveState())
    
    def _create_model_selection_group(self) -> QGroupBox:
        """Create model selection group."""
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(8, 10, 8, 8)
        model_layout.setSpacing(8)
        
        # Model path display
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Model:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("No model selected")
        self.model_path_edit.setReadOnly(True)
        path_layout.addWidget(self.model_path_edit, stretch=1)
        
        browse_model_btn = QPushButton("Browse...")
        browse_model_btn.clicked.connect(self._browse_model)
        path_layout.addWidget(browse_model_btn)
        
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self._load_selected_model)
        path_layout.addWidget(load_model_btn)
        
        model_layout.addLayout(path_layout)
        
        # Model status label
        self.model_status_label = QLabel("⚠ No model loaded")
        self.model_status_label.setStyleSheet(
            "color: #e67e22; font-weight: bold; font-size: 11px; "
            "padding: 6px; background-color: #fef5e7; border-radius: 4px;"
        )
        model_layout.addWidget(self.model_status_label)
        
        # Target class selection
        target_class_row = QHBoxLayout()
        target_class_row.setSpacing(4)
        target_class_label = QLabel("Target Class:")
        target_class_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        target_class_row.addWidget(target_class_label)
        
        self.target_class_combo = QComboBox()
        self.target_class_combo.setStyleSheet("font-size: 10px;")
        self.target_class_combo.setToolTip("Select which class to optimize for (usually class 1 = CRACK)")
        self._update_target_class_options(2)  # Default binary
        target_class_row.addWidget(self.target_class_combo, stretch=1)
        
        model_layout.addLayout(target_class_row)
        
        # Model info (collapsed details)
        self.model_info_label = QLabel()
        self.model_info_label.setStyleSheet("color: #555; font-size: 10px;")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.hide()
        model_layout.addWidget(self.model_info_label)
        
        # Check if model already loaded
        self._update_model_status()
        
        return model_group
    
    def _create_dataset_group(self) -> QGroupBox:
        """Create dataset loading group."""
        dataset_group = QGroupBox("Dataset Loading")
        dataset_layout = QFormLayout(dataset_group)
        dataset_layout.setContentsMargins(8, 10, 8, 8)
        dataset_layout.setSpacing(6)
        
        # CALIBRATION CSV (used for hyperparameter tuning)
        calibration_row = QHBoxLayout()
        self.calibration_path_edit = QLineEdit()
        self.calibration_path_edit.setPlaceholderText("No file selected")
        self.calibration_path_edit.setReadOnly(True)
        calibration_row.addWidget(self.calibration_path_edit, stretch=1)
        calibration_browse_btn = QPushButton("Browse...")
        calibration_browse_btn.clicked.connect(lambda: self._browse_csv('calibration'))
        calibration_row.addWidget(calibration_browse_btn)
        dataset_layout.addRow("CALIBRATION CSV:", calibration_row)
        
        # TEST CSV
        test_row = QHBoxLayout()
        self.test_path_edit = QLineEdit()
        self.test_path_edit.setPlaceholderText("No file selected")
        self.test_path_edit.setReadOnly(True)
        test_row.addWidget(self.test_path_edit, stretch=1)
        test_browse_btn = QPushButton("Browse...")
        test_browse_btn.clicked.connect(lambda: self._browse_csv('test'))
        test_row.addWidget(test_browse_btn)
        dataset_layout.addRow("TEST CSV:", test_row)
        
        # Dataset summary
        self.dataset_summary_label = QLabel("No datasets loaded")
        self.dataset_summary_label.setStyleSheet("color: #666; font-size: 10px;")
        dataset_layout.addRow("", self.dataset_summary_label)
        
        return dataset_group
    
    def _create_mapping_group(self) -> QGroupBox:
        """Create column mapping group with help text."""
        mapping_group = QGroupBox("Column Mapping")
        mapping_layout = QVBoxLayout(mapping_group)
        mapping_layout.setContentsMargins(8, 10, 8, 8)
        mapping_layout.setSpacing(8)
        
        # Help text explaining column mapping
        help_text = QLabel(
            "<b>Column Mapping Help:</b><br>"
            "• <b>Label Column:</b> Binary labels (0=Healthy, 1=Crack) or text ('healthy'/'crack')<br>"
            "• <b>Path Column:</b> Full path to HSI cube file (.hdr file)<br>"
            "• <b>ID Column:</b> Optional unique identifier (auto-generated if not provided)<br>"
            "<i>Example CSV: grape_id,row,week_date,label,image_path</i>"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #555; font-size: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 4px;")
        mapping_layout.addWidget(help_text)
        
        # Column selectors
        form_layout = QFormLayout()
        
        self.label_col_combo = QComboBox()
        self.label_col_combo.setEditable(True)
        form_layout.addRow("Label Column:", self.label_col_combo)
        
        self.path_col_combo = QComboBox()
        self.path_col_combo.setEditable(True)
        form_layout.addRow("Path Column:", self.path_col_combo)
        
        self.id_col_combo = QComboBox()
        self.id_col_combo.setEditable(True)
        self.id_col_combo.addItem("(auto)")
        form_layout.addRow("ID Column:", self.id_col_combo)
        
        apply_mapping_btn = QPushButton("Apply Mapping")
        apply_mapping_btn.clicked.connect(self._apply_column_mapping)
        form_layout.addRow("", apply_mapping_btn)
        
        mapping_layout.addLayout(form_layout)
        
        return mapping_group
    
    def _create_search_space_group(self) -> QGroupBox:
        """Create hyperparameter search space editor with modern card UI."""
        space_group = QGroupBox()
        space_layout = QVBoxLayout(space_group)
        space_layout.setContentsMargins(8, 10, 8, 8)
        space_layout.setSpacing(10)
        
        # Header
        header_label = QLabel("Hyperparameter Search Space")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        space_layout.addWidget(header_label)
        
        subtitle_label = QLabel("Optuna will optimize these parameters during tuning")
        subtitle_label.setStyleSheet("font-size: 11px; color: #7f8c8d; font-style: italic;")
        space_layout.addWidget(subtitle_label)
        
        # Cards container with 2-column grid layout
        cards_widget = QWidget()
        cards_layout = QGridLayout(cards_widget)
        cards_layout.setContentsMargins(10, 10, 10, 10)
        cards_layout.setSpacing(12)
        
        # Create parameter cards with proper sizing
        self.param_cards = {}
        
        # Row 0: pixel_threshold, min_blob_area
        self.param_cards['pixel_threshold'] = FloatRangeCard(
            param_name='pixel_threshold',
            display_name='Pixel Threshold',
            min_value=0.970,
            max_value=0.999,
            step=0.005,
            decimals=3
        )
        self.param_cards['pixel_threshold'].setMinimumWidth(420)
        self.param_cards['pixel_threshold'].setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        cards_layout.addWidget(self.param_cards['pixel_threshold'], 0, 0)
        
        self.param_cards['min_blob_area'] = IntRangeCard(
            param_name='min_blob_area',
            display_name='Min Blob Area',
            min_value=1,
            max_value=300,
            step=1
        )
        self.param_cards['min_blob_area'].setMinimumWidth(420)
        self.param_cards['min_blob_area'].setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        cards_layout.addWidget(self.param_cards['min_blob_area'], 0, 1)
        
        # Row 1: max_blob_area, morph_size
        self.param_cards['max_blob_area'] = IntRangeCard(
            param_name='max_blob_area',
            display_name='Max Blob Area',
            min_value=301,
            max_value=9000,
            step=100
        )
        self.param_cards['max_blob_area'].setMinimumWidth(420)
        self.param_cards['max_blob_area'].setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        cards_layout.addWidget(self.param_cards['max_blob_area'], 1, 0)
        
        self.param_cards['morph_size'] = CategoricalCard(
            param_name='morph_size',
            display_name='Morphological Size',
            choices=[0, 3, 5, 7, 9, 11]
        )
        self.param_cards['morph_size'].setMinimumWidth(420)
        self.param_cards['morph_size'].setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        cards_layout.addWidget(self.param_cards['morph_size'], 1, 1)
        
        # Row 2: patch_size, patch_crack_pct_threshold
        self.param_cards['patch_size'] = CategoricalCard(
            param_name='patch_size',
            display_name='Patch Size',
            choices=[4, 8, 16, 24, 32, 40, 48, 64]
        )
        self.param_cards['patch_size'].setMinimumWidth(420)
        self.param_cards['patch_size'].setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        cards_layout.addWidget(self.param_cards['patch_size'], 2, 0)
        
        self.param_cards['patch_crack_pct_threshold'] = FloatRangeCard(
            param_name='patch_crack_pct_threshold',
            display_name='Patch Crack % Threshold',
            min_value=0.1,
            max_value=100.0,
            step=0.5,
            decimals=1
        )
        self.param_cards['patch_crack_pct_threshold'].setMinimumWidth(420)
        self.param_cards['patch_crack_pct_threshold'].setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        cards_layout.addWidget(self.param_cards['patch_crack_pct_threshold'], 2, 1)
        
        # Row 3: global_crack_pct_threshold
        self.param_cards['global_crack_pct_threshold'] = FloatRangeCard(
            param_name='global_crack_pct_threshold',
            display_name='Global Crack % Threshold',
            min_value=0.1,
            max_value=5.0,
            step=0.1,
            decimals=1
        )
        self.param_cards['global_crack_pct_threshold'].setMinimumWidth(420)
        self.param_cards['global_crack_pct_threshold'].setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        cards_layout.addWidget(self.param_cards['global_crack_pct_threshold'], 3, 0)
        
        space_layout.addWidget(cards_widget)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        reset_btn.clicked.connect(self._reset_search_space)
        btn_layout.addWidget(reset_btn)
        
        apply_btn = QPushButton("✓ Apply Search Space")
        apply_btn.setStyleSheet("font-weight: bold; padding: 6px 12px; background-color: #3498db; color: white;")
        apply_btn.clicked.connect(self._apply_search_space)
        btn_layout.addWidget(apply_btn)
        
        btn_layout.addStretch()
        space_layout.addLayout(btn_layout)
        
        # Summary box (read-only, compact JSON view with MAXIMUM height)
        summary_label = QLabel("Search Space Summary (read-only):")
        summary_label.setStyleSheet("font-weight: bold; font-size: 11px; margin-top: 4px;")
        space_layout.addWidget(summary_label)
        
        self.search_space_summary = QTextEdit()
        self.search_space_summary.setReadOnly(True)
        self.search_space_summary.setMaximumHeight(100)  # Cap height to avoid dead space
        self.search_space_summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.search_space_summary.setStyleSheet(
            "font-family: 'Courier New'; font-size: 10px; "
            "background-color: #ecf0f1; border: 1px solid #bdc3c7; "
            "border-radius: 4px; padding: 6px;"
        )
        space_layout.addWidget(self.search_space_summary)
        
        # Initial summary update
        self._update_search_space_summary()
        
        return space_group
    
    def _reset_search_space(self):
        """Reset all cards to default values."""
        # Reset search space model
        self.search_space = HyperparameterSearchSpace()
        
        # Reset each card to defaults
        for param_name, card in self.param_cards.items():
            param = self.search_space.get_param(param_name)
            if param:
                spec = {
                    'name': param.name,
                    'type': param.type
                }
                if param.type in ['float', 'int']:
                    spec['min_value'] = param.min_value
                    spec['max_value'] = param.max_value
                else:  # categorical
                    spec['choices'] = param.choices
                
                card.set_spec(spec)
        
        self._update_search_space_summary()
        self._log("Search space reset to defaults")
    
    def _apply_search_space(self):
        """Apply search space from cards to model."""
        # Validate all cards first
        errors = []
        for param_name, card in self.param_cards.items():
            is_valid, error_msg = card.is_valid()
            if not is_valid:
                errors.append(f"{param_name}: {error_msg}")
        
        # Validate that max_blob_area range is greater than min_blob_area range
        if 'min_blob_area' in self.param_cards and 'max_blob_area' in self.param_cards:
            min_spec = self.param_cards['min_blob_area'].get_spec()
            max_spec = self.param_cards['max_blob_area'].get_spec()
            # The minimum of max_blob_area range should be > maximum of min_blob_area range
            if max_spec['min_value'] <= min_spec['max_value']:
                errors.append(
                    f"max_blob_area min ({max_spec['min_value']}) must be > min_blob_area max ({min_spec['max_value']})"
                )
        
        if errors:
            QMessageBox.warning(
                self, 
                "Invalid Configuration", 
                "Please fix the following errors:\n\n" + "\n".join(errors)
            )
            return
        
        # Update search space model from cards
        try:
            for param_name, card in self.param_cards.items():
                spec = card.get_spec()
                
                if spec['type'] in ['float', 'int']:
                    self.search_space.update_param(
                        param_name, 
                        min_value=spec['min_value'],
                        max_value=spec['max_value']
                    )
                else:  # categorical
                    self.search_space.update_param(
                        param_name,
                        choices=spec['choices']
                    )
            
            # Final validation
            self.search_space.validate_all()
            
            # Update summary
            self._update_search_space_summary()
            
            self._log("Search space applied successfully")
            QMessageBox.information(self, "Success", "Search space applied and ready for tuning")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply search space: {e}")
            logger.error(f"Error applying search space: {e}\n{traceback.format_exc()}")
    
    def _update_search_space_summary(self):
        """Update the search space summary text box."""
        summary_lines = []
        summary_lines.append("{")
        
        for param_name in ['pixel_threshold', 'min_blob_area', 'max_blob_area', 'morph_size', 'patch_size', 'patch_crack_pct_threshold', 'global_crack_pct_threshold']:
            param = self.search_space.get_param(param_name)
            if param:
                if param.type in ['float', 'int']:
                    summary_lines.append(f'  "{param_name}": [{param.min_value}, {param.max_value}],')
                else:  # categorical
                    choices_str = ', '.join(map(str, param.choices))
                    summary_lines.append(f'  "{param_name}": [{choices_str}],')
        
        # Remove trailing comma from last line
        if len(summary_lines) > 1:
            summary_lines[-1] = summary_lines[-1].rstrip(',')
        
        summary_lines.append("}")
        
        self.search_space_summary.setPlainText('\n'.join(summary_lines))
    
    def _create_tuning_config_group(self) -> QGroupBox:
        """Create tuning configuration group."""
        tuning_group = QGroupBox("Tuning Configuration")
        tuning_layout = QFormLayout(tuning_group)
        tuning_layout.setContentsMargins(8, 10, 8, 8)
        tuning_layout.setSpacing(6)
        
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["F1", "F2"])
        tuning_layout.addRow("Optimize Metric:", self.metric_combo)
        
        self.n_trials_spin = QSpinBox()
        self.n_trials_spin.setRange(1, 1000)
        self.n_trials_spin.setValue(50)
        tuning_layout.addRow("Number of Trials:", self.n_trials_spin)
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.seed_spin.setSpecialValueText("Random")
        tuning_layout.addRow("Random Seed:", self.seed_spin)
        
        self.output_dir_edit = QLineEdit()
        default_output = str(self.DEFAULT_EXPERIMENTS_DIR / "optuna_full_image" / "<timestamp>")
        self.output_dir_edit.setPlaceholderText(default_output)
        tuning_layout.addRow("Output Directory:", self.output_dir_edit)
        
        return tuning_group
    
    def _create_run_controls(self) -> QHBoxLayout:
        """Create run control buttons."""
        run_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Tuning")
        self.start_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.start_btn.setEnabled(False)  # Initially disabled until model and datasets loaded
        self.start_btn.setToolTip("Please load a model and datasets first")
        self.start_btn.clicked.connect(self._start_tuning)
        run_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_tuning)
        run_layout.addWidget(self.stop_btn)
        
        return run_layout
    
    def _browse_model(self):
        """Browse for model file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            self.last_model_dir,
            "Model Files (*.joblib *.pkl *.pth);;All Files (*.*)"
        )
        
        if not filepath:
            return
        
        # Update last directory
        self.last_model_dir = str(Path(filepath).parent)
        
        # Update path display
        self.model_path_edit.setText(filepath)
        
        # Auto-load model
        self._load_selected_model()
    
    def _load_selected_model(self):
        """Load model from selected path."""
        model_path = self.model_path_edit.text()
        
        if not model_path or model_path == "No model selected":
            QMessageBox.warning(self, "No Model Selected", "Please select a model file first")
            return
        
        try:
            # Load model using model manager (automatically applies default preprocessing)
            model_info = self.model_manager.load_model(model_path)
            preprocess_cfg = self.model_manager.get_preprocess_cfg()
            
            # Update UI
            self._update_model_status()
            
            # Log success with preprocessing details
            self._log("=" * 60)
            self._log("Model Loaded Successfully")
            self._log("=" * 60)
            self._log(f"Path: {model_info.path}")
            self._log(f"Type: {model_info.model_type}")
            self._log(f"Classes: {model_info.n_classes}")
            self._log("")
            self._log("Preprocessing Configuration:")
            if preprocess_cfg:
                self._log(f"  - SNV normalization: {preprocess_cfg.use_snv}")
                if preprocess_cfg.wl_min or preprocess_cfg.wl_max:
                    self._log(f"  - Wavelength filter: [{preprocess_cfg.wl_min or 'min'}-{preprocess_cfg.wl_max or 'max'}] nm")
                if preprocess_cfg.use_l2_norm:
                    self._log(f"  - L2 normalization: {preprocess_cfg.use_l2_norm}")
            else:
                self._log("  (No preprocessing)")
            self._log("")
            
        except Exception as e:
            # Get full traceback
            full_traceback = traceback.format_exc()
            error_msg = f"Failed to load model: {str(e)}"
            
            # Log detailed error
            self._log("=" * 60)
            self._log("✗ MODEL LOAD ERROR")
            self._log("=" * 60)
            self._log(f"Path: {model_path}")
            self._log(f"Error: {str(e)}")
            self._log("")
            self._log("Full traceback:")
            self._log(full_traceback)
            self._log("=" * 60)
            
            # Update status label
            self.model_status_label.setText("⚠ Model load failed")
            self.model_status_label.setStyleSheet(
                "color: #c0392b; font-weight: bold; font-size: 11px; "
                "padding: 6px; background-color: #fadbd8; border-radius: 4px;"
            )
            
            # Show error dialog with more context
            QMessageBox.critical(
                self, 
                "Model Load Error", 
                f"{error_msg}\n\nSee log panel for full traceback."
            )
            logger.error(f"Model load error: {e}\n{full_traceback}")
    
    def _update_target_class_options(self, n_classes: int):
        """Update target class combo box based on number of classes."""
        current = self.target_class_combo.currentData()
        self.target_class_combo.blockSignals(True)
        self.target_class_combo.clear()
        
        for idx in range(n_classes):
            if n_classes == 2:
                label = f"Class {idx} (CRACK)" if idx == 1 else f"Class {idx} (REGULAR)"
            else:
                label = f"Class {idx}"
            self.target_class_combo.addItem(label, idx)
        
        # Restore previous selection or default to class 1
        if current is not None and 0 <= int(current) < n_classes:
            self.target_class_combo.setCurrentIndex(int(current))
        else:
            default_idx = 1 if n_classes > 1 else 0
            self.target_class_combo.setCurrentIndex(default_idx)
        
        self.target_class_combo.blockSignals(False)
    
    def _update_model_status(self):
        """Update model status display."""
        if self.model_manager.is_loaded():
            model_info = self.model_manager.get_model_info()
            preprocess_cfg = self.model_manager.get_preprocess_cfg()
            
            # Update target class options
            self._update_target_class_options(model_info.n_classes)
            
            # Update status label (success)
            self.model_status_label.setText(f"✓ Model loaded: {model_info.model_type}")
            self.model_status_label.setStyleSheet(
                "color: #27ae60; font-weight: bold; font-size: 11px; "
                "padding: 6px; background-color: #d5f4e6; border-radius: 4px;"
            )
            
            # Update info label with preprocessing details
            preprocess_info = ""
            if preprocess_cfg:
                preprocess_details = []
                if preprocess_cfg.use_snv:
                    preprocess_details.append("SNV")
                if preprocess_cfg.wl_min or preprocess_cfg.wl_max:
                    wl_range = f"[{preprocess_cfg.wl_min or 'min'}-{preprocess_cfg.wl_max or 'max'}]nm"
                    preprocess_details.append(f"Wavelength: {wl_range}")
                if preprocess_cfg.use_l2_norm:
                    preprocess_details.append("L2-norm")
                
                if preprocess_details:
                    preprocess_info = f" | Preprocessing: {', '.join(preprocess_details)}"
            
            self.model_info_label.setText(
                f"Type: {model_info.model_type} | Classes: {model_info.n_classes} | "
                f"Path: {model_info.name}{preprocess_info}"
            )
            self.model_info_label.show()
            
            # Update path display if not already set
            if not self.model_path_edit.text() or self.model_path_edit.text() == "No model selected":
                self.model_path_edit.setText(model_info.path)
            
            # Enable start button (if other conditions met)
            self._update_start_button_state()
            
        else:
            # No model loaded
            self.model_status_label.setText("⚠ No model loaded - Please select and load a model")
            self.model_status_label.setStyleSheet(
                "color: #e67e22; font-weight: bold; font-size: 11px; "
                "padding: 6px; background-color: #fef5e7; border-radius: 4px;"
            )
            self.model_info_label.hide()
            
            # Disable start button
            self._update_start_button_state()
    
    def _update_start_button_state(self):
        """Update start button enabled state based on readiness."""
        # Check if UI is fully initialized
        if not hasattr(self, 'start_btn'):
            return
        
        # Check all prerequisites
        has_model = self.model_manager.is_loaded()
        has_datasets = (self.calibration_samples and self.test_samples)
        
        # Ensure ready is strictly boolean
        ready = bool(has_model and has_datasets)
        
        # Debug logging
        logger.debug(f"_update_start_button_state: ready={ready} (type={type(ready).__name__})")
        logger.debug(f"  has_model={has_model}, has_datasets={has_datasets}")
        
        # Strict type check before calling setEnabled
        if not isinstance(ready, bool):
            logger.error(f"ERROR: ready is not bool, it's {type(ready).__name__}: {ready}")
            ready = False
        
        self.start_btn.setEnabled(ready)
        
        # Update button tooltip with status
        if not has_model:
            self.start_btn.setToolTip("Please load a model first")
        elif not has_datasets:
            self.start_btn.setToolTip("Please load and map calibration and test datasets")
        else:
            self.start_btn.setToolTip("Start Optuna hyperparameter tuning")
    
    def _browse_csv(self, split: str):
        """
        Browse for CSV file.
        
        Opens file dialog in data directory by default, remembers last location.
        """
        filepath, _ = QFileDialog.getOpenFileName(
            self, 
            f"Select {split.upper()} CSV", 
            self.last_csv_dir,  # Start in last used directory
            "CSV Files (*.csv)"
        )
        
        if not filepath:
            return
        
        # Update last directory
        self.last_csv_dir = str(Path(filepath).parent)
        
        try:
            loader = DatasetCSVLoader()
            loader.load_csv(filepath)
            
            # Store loader
            if split == 'calibration':
                self.calibration_loader = loader
                self.calibration_path_edit.setText(filepath)
            else:  # test
                self.test_loader = loader
                self.test_path_edit.setText(filepath)
            
            # Update column combos (use first loaded CSV as reference)
            if self.label_col_combo.count() == 0:
                columns = loader.get_available_columns()
                self.label_col_combo.clear()
                self.label_col_combo.addItems(columns)
                self.path_col_combo.clear()
                self.path_col_combo.addItems(columns)
                self.id_col_combo.clear()
                self.id_col_combo.addItem("(auto)")
                self.id_col_combo.addItems(columns)
                
                # Enhanced auto-detection for typical column names
                # Look for 'label' column
                for col in columns:
                    if col.lower() in ['label', 'y', 'target', 'class']:
                        idx = self.label_col_combo.findText(col)
                        if idx >= 0:
                            self.label_col_combo.setCurrentIndex(idx)
                        break
                
                # Look for 'image_path' or 'path' column
                for col in columns:
                    if col.lower() in ['image_path', 'path', 'filepath', 'file_path']:
                        idx = self.path_col_combo.findText(col)
                        if idx >= 0:
                            self.path_col_combo.setCurrentIndex(idx)
                        break
                
                # Look for 'grape_id' or 'id' column
                for col in columns:
                    if col.lower() in ['grape_id', 'id', 'sample_id', 'name']:
                        idx = self.id_col_combo.findText(col)
                        if idx >= 0:
                            self.id_col_combo.setCurrentIndex(idx)
                        break
            
            self._update_dataset_summary()
            self._log(f"Loaded {split.upper()} CSV: {Path(filepath).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading CSV", str(e))
            logger.error(f"Error loading CSV: {e}")
    
    def _apply_column_mapping(self):
        """Apply column mapping to all loaded datasets."""
        label_col = self.label_col_combo.currentText()
        path_col = self.path_col_combo.currentText()
        id_col = self.id_col_combo.currentText() if self.id_col_combo.currentText() != "(auto)" else None
        
        if not label_col or not path_col:
            QMessageBox.warning(self, "Missing Columns", "Please select label and path columns")
            return
        
        try:
            # Apply to all loaded datasets
            for split, loader in [('calibration', self.calibration_loader), 
                                  ('test', self.test_loader)]:
                if loader is not None:
                    loader.set_column_mapping(label_col, path_col, id_col)
                    
                    # Convert to samples
                    samples = loader.to_samples()
                    
                    if split == 'calibration':
                        self.calibration_samples = samples
                    else:
                        self.test_samples = samples
                    
                    self._log(f"{split.upper()}: Loaded {len(samples)} samples")
            
            self._update_dataset_summary()
            self._update_start_button_state()  # Update button state after loading datasets
            QMessageBox.information(self, "Success", "Column mapping applied successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply mapping: {e}")
            logger.error(f"Error applying mapping: {e}\n{traceback.format_exc()}")
    
    def _update_dataset_summary(self):
        """Update dataset summary label."""
        parts = []
        
        if self.calibration_samples:
            crack = sum(1 for s in self.calibration_samples if s.label == 1)
            parts.append(f"CALIBRATION: {len(self.calibration_samples)} ({crack} CRACK)")
        
        if self.test_samples:
            crack = sum(1 for s in self.test_samples if s.label == 1)
            parts.append(f"TEST: {len(self.test_samples)} ({crack} CRACK)")
        
        if parts:
            self.dataset_summary_label.setText(" | ".join(parts))
        else:
            self.dataset_summary_label.setText("No datasets loaded")
    
    def _start_tuning(self):
        """Start Optuna tuning."""
        # Validate (button should already be disabled, but double-check)
        if not self.model_manager.is_loaded():
            # This shouldn't happen if button state is correct, but handle gracefully
            self._log("⚠ ERROR: No model loaded")
            return
        
        if not self.calibration_samples or not self.test_samples:
            self._log("⚠ ERROR: Missing datasets")
            return
        
        # Get parameters
        optimize_metric = self.metric_combo.currentText().lower()
        n_trials = self.n_trials_spin.value()
        seed = self.seed_spin.value() if self.seed_spin.value() > 0 else None
        
        # Construct output directory
        if self.output_dir_edit.text():
            output_dir = self.output_dir_edit.text()
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = str(self.DEFAULT_EXPERIMENTS_DIR / "optuna_full_image" / f"run_{timestamp}")
        
        # Log configuration
        model_info = self.model_manager.get_model_info()
        target_class_idx = int(self.target_class_combo.currentData())
        
        self._log("=" * 60)
        self._log("Starting Optuna Hyperparameter Tuning")
        self._log("=" * 60)
        self._log(f"Model: {model_info.name}")
        self._log(f"Model Type: {model_info.model_type}")
        self._log(f"Classes: {model_info.n_classes}")
        self._log(f"Target Class: {target_class_idx} ({self.target_class_combo.currentText()})")
        self._log(f"Metric: {optimize_metric.upper()}")
        self._log(f"Trials: {n_trials}")
        self._log(f"Seed: {seed}")
        self._log(f"Output: {output_dir}")
        self._log("")
        self._log(self.search_space.get_summary())
        self._log("")
        
        # Create inference function using model manager with selected target class
        inference_fn = self.model_manager.create_inference_fn(target_class_index=target_class_idx)
        
        # Create and start worker
        self.worker = OptunaWorker(
            calibration_samples=self.calibration_samples,
            test_samples=self.test_samples,
            inference_fn=inference_fn,
            optimize_metric=optimize_metric,
            n_trials=n_trials,
            seed=seed,
            n_jobs=1,
            timeout=None,
            output_dir=output_dir,
            search_space=self.search_space
        )
        
        # Connect signals
        self.worker.progress_update.connect(self._log)
        self.worker.trial_completed.connect(self._on_trial_completed)
        self.worker.study_completed.connect(self._on_study_completed)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.final_results.connect(self._on_final_results)
        self.worker.finished.connect(self._on_worker_finished)
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Start
        self.worker.start()
    
    def _stop_tuning(self):
        """Stop ongoing tuning."""
        if self.worker is not None:
            self._log("\nStop requested...")
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
    
    def _on_trial_completed(self, trial_number: int, score: float, params: dict):
        """Handle trial completion."""
        pass  # Progress is already logged via progress_update
    
    def _on_study_completed(self):
        """Handle study completion."""
        self._log("\n✓ Study completed successfully!")
    
    def _on_error(self, error_msg: str):
        """Handle error."""
        self._log(f"\n✗ ERROR: {error_msg}")
        QMessageBox.critical(self, "Tuning Error", f"An error occurred:\n{error_msg[:500]}")
    
    def _on_final_results(self, report_str: str):
        """Handle final results."""
        # Already logged via progress_update
        pass
    
    def _on_worker_finished(self):
        """Handle worker finished."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
    
    def _log(self, message: str):
        """Append message to progress log."""
        self.progress_text.append(message)
        # Auto-scroll to bottom
        self.progress_text.verticalScrollBar().setValue(
            self.progress_text.verticalScrollBar().maximum()
        )
    
    def on_model_loaded_externally(self):
        """
        Callback when model is loaded externally (e.g., from Visual Debug tab).
        
        Updates UI to reflect the newly loaded model.
        """
        self._update_model_status()
        logger.info("Model loaded externally, Optuna tab updated")
    
    def reset_state(self):
        """Reset tab state."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait()
        
        self.calibration_loader = None
        self.test_loader = None
        self.calibration_samples = []
        self.test_samples = []
        
        self.calibration_path_edit.clear()
        self.test_path_edit.clear()
        self.progress_text.clear()
        
        # Reset search space and cards
        self._reset_search_space()
        
        self._update_dataset_summary()
        self._update_model_status()
        
        logger.info("Optuna tab state reset")
    
    def stop_workers(self):
        """Stop any running workers."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait()

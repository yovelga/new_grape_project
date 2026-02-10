"""
Hyperparameter search space configuration.

Defines the search space for Optuna optimization with UI-editable ranges.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import optuna


@dataclass
class HyperparameterSpec:
    """
    Specification for a single hyperparameter.
    
    Attributes:
        name: Parameter name
        type: Parameter type ('float', 'int', 'categorical')
        min_value: Minimum value (for float/int)
        max_value: Maximum value (for float/int)
        step: Step size (optional, for int)
        choices: List of choices (for categorical)
        description: Human-readable description
    """
    name: str
    type: str  # 'float', 'int', 'categorical'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[int] = None
    choices: Optional[List[Any]] = None
    description: str = ""
    
    def validate(self):
        """Validate hyperparameter specification."""
        if self.type not in ['float', 'int', 'categorical']:
            raise ValueError(f"Invalid type: {self.type}")
        
        if self.type in ['float', 'int']:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"{self.name}: min_value and max_value required for {self.type}")
            if self.min_value >= self.max_value:
                raise ValueError(f"{self.name}: min_value must be < max_value")
        
        if self.type == 'categorical':
            if not self.choices or len(self.choices) == 0:
                raise ValueError(f"{self.name}: choices required for categorical")
    
    def suggest(self, trial: optuna.Trial) -> Any:
        """
        Suggest a value for this hyperparameter using Optuna trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Suggested value
        """
        if self.type == 'float':
            return trial.suggest_float(self.name, self.min_value, self.max_value)
        elif self.type == 'int':
            if self.step is not None:
                return trial.suggest_int(self.name, int(self.min_value), int(self.max_value), step=self.step)
            else:
                return trial.suggest_int(self.name, int(self.min_value), int(self.max_value))
        elif self.type == 'categorical':
            return trial.suggest_categorical(self.name, self.choices)
        else:
            raise ValueError(f"Unknown type: {self.type}")


class HyperparameterSearchSpace:
    """
    Manages the hyperparameter search space for Optuna tuning.
    
    Provides default search space and allows UI customization.
    Supports different presets for sklearn/XGBoost vs autoencoder models.
    """
    
    def __init__(self, preset: str = "default"):
        """
        Initialize with search space preset.
        
        Args:
            preset: 'default' for sklearn/XGBoost, 'autoencoder' for autoencoder models
        """
        self.params: Dict[str, HyperparameterSpec] = {}
        self.preset = preset
        if preset == "autoencoder":
            self._init_autoencoder_search_space()
        else:
            self._init_default_search_space()
    
    @classmethod
    def create_default(cls) -> 'HyperparameterSearchSpace':
        """Create default search space (sklearn/XGBoost models)."""
        return cls(preset="default")
    
    @classmethod
    def create_autoencoder(cls) -> 'HyperparameterSearchSpace':
        """
        Create autoencoder-specific search space.
        
        Key difference: pixel_threshold range is [0.10, 0.95] instead of
        [0.970, 0.999] because autoencoder probability maps have a very
        different distribution than sklearn/XGBoost models.
        """
        return cls(preset="autoencoder")
    
    def _init_autoencoder_search_space(self):
        """Initialize search space tuned for autoencoder probability maps."""
        self.params = {
            'pixel_threshold': HyperparameterSpec(
                name='pixel_threshold',
                type='float',
                min_value=0.10,
                max_value=0.95,
                description='Probability threshold for binarization (autoencoder range)'
            ),
            'min_blob_area': HyperparameterSpec(
                name='min_blob_area',
                type='int',
                min_value=1,
                max_value=300,
                description='Minimum blob area in pixels (smaller removed)'
            ),
            'max_blob_area': HyperparameterSpec(
                name='max_blob_area',
                type='int',
                min_value=301,
                max_value=9000,
                description='Maximum blob area in pixels (larger removed). Must be > min_blob_area'
            ),
            'morph_size': HyperparameterSpec(
                name='morph_size',
                type='categorical',
                choices=[0, 3, 5, 7, 9, 11, 13],
                description='Morphological closing kernel size (0=disabled, must be odd)'
            ),
            'patch_size': HyperparameterSpec(
                name='patch_size',
                type='categorical',
                choices=[4, 8, 16, 24, 32, 40, 48, 64],
                description='Square patch size in pixels'
            ),
            'patch_crack_pct_threshold': HyperparameterSpec(
                name='patch_crack_pct_threshold',
                type='float',
                min_value=0.1,
                max_value=100.0,
                description='Crack percentage threshold for patch flagging (0-100%)'
            ),
            'global_crack_pct_threshold': HyperparameterSpec(
                name='global_crack_pct_threshold',
                type='float',
                min_value=0.1,
                max_value=5.0,
                description='Global image crack % threshold - classify as CRACK if >= this% (0-100%)'
            ),
        }
    
    def _init_default_search_space(self):
        """Initialize default search space for patch-based classifier."""
        self.params = {
            'pixel_threshold': HyperparameterSpec(
                name='pixel_threshold',
                type='float',
                min_value=0.970,
                max_value=0.999,
                description='Probability threshold for binarization'
            ),
            'min_blob_area': HyperparameterSpec(
                name='min_blob_area',
                type='int',
                min_value=1,
                max_value=300,
                description='Minimum blob area in pixels (smaller removed)'
            ),
            'max_blob_area': HyperparameterSpec(
                name='max_blob_area',
                type='int',
                min_value=301,
                max_value=9000,
                description='Maximum blob area in pixels (larger removed). Must be > min_blob_area'
            ),
            'morph_size': HyperparameterSpec(
                name='morph_size',
                type='categorical',
                choices=[0, 3, 5, 7, 9, 11, 13],
                description='Morphological closing kernel size (0=disabled, must be odd)'
            ),
            'patch_size': HyperparameterSpec(
                name='patch_size',
                type='categorical',
                choices=[4, 8, 16, 24, 32, 40, 48, 64],
                description='Square patch size in pixels'
            ),
            'patch_crack_pct_threshold': HyperparameterSpec(
                name='patch_crack_pct_threshold',
                type='float',
                min_value=0.1,
                max_value=100.0,
                description='Crack percentage threshold for patch flagging (0-100%)'
            ),
            'global_crack_pct_threshold': HyperparameterSpec(
                name='global_crack_pct_threshold',
                type='float',
                min_value=0.1,
                max_value=5.0,
                description='Global image crack % threshold - classify as CRACK if >= this% (0-100%)'
            ),
        }
    
    def get_param(self, name: str) -> Optional[HyperparameterSpec]:
        """Get hyperparameter specification by name."""
        return self.params.get(name)
    
    def update_param(self, name: str, **kwargs):
        """
        Update hyperparameter specification.
        
        Args:
            name: Parameter name
            **kwargs: Fields to update (min_value, max_value, choices, etc.)
        """
        if name not in self.params:
            raise ValueError(f"Unknown parameter: {name}")
        
        param = self.params[name]
        for key, value in kwargs.items():
            if hasattr(param, key):
                setattr(param, key, value)
        
        param.validate()
    
    def validate_all(self):
        """Validate all hyperparameter specifications."""
        for param in self.params.values():
            param.validate()
    
    def suggest_all(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest values for all hyperparameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of parameter names to suggested values
        """
        suggestions = {}
        for name, param in self.params.items():
            suggestions[name] = param.suggest(trial)
        return suggestions
    
    def get_summary(self) -> str:
        """Get text summary of search space."""
        lines = ["Hyperparameter Search Space:", "-" * 50]
        for name, param in self.params.items():
            if param.type in ['float', 'int']:
                lines.append(f"{name:30s}: [{param.min_value}, {param.max_value}]")
            else:
                lines.append(f"{name:30s}: {param.choices}")
        return "\n".join(lines)

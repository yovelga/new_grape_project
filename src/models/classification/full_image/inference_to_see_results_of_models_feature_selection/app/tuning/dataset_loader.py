"""
Dataset loader for CSV-based hyperparameter tuning.

Provides CSV loading, column mapping, and sample normalization for Optuna tuning.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """
    Single sample for inference and evaluation.
    
    Attributes:
        path: Path to HSI cube file
        label: Binary label (0=HEALTHY, 1=CRACK)
        sample_id: Optional identifier for the sample
        metadata: Additional metadata
    """
    path: str
    label: int
    sample_id: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate sample after initialization."""
        if self.label not in [0, 1]:
            raise ValueError(f"Label must be 0 or 1, got {self.label}")


class DatasetCSVLoader:
    """
    Loader for CSV datasets with automatic column detection and mapping.
    
    Handles label normalization (CRACK=1, HEALTHY=0) and path resolution.
    """
    
    # Common column name patterns
    LABEL_PATTERNS = ['label', 'y', 'target', 'class', 'crack', 'is_crack', 
                      'Label', 'Y', 'Target', 'Class', 'Crack', 'IsCrack']
    PATH_PATTERNS = ['path', 'filepath', 'file_path', 'file', 'image_path', 
                     'Path', 'FilePath', 'File', 'ImagePath']
    ID_PATTERNS = ['id', 'sample_id', 'name', 'filename', 'ID', 'SampleID', 'Name']
    
    def __init__(self):
        """Initialize dataset loader."""
        self.df: Optional[pd.DataFrame] = None
        self.csv_path: Optional[Path] = None
        self.label_col: Optional[str] = None
        self.path_col: Optional[str] = None
        self.id_col: Optional[str] = None
        
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV is empty
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        if self.df.empty:
            raise ValueError(f"CSV file is empty: {csv_path}")
        
        logger.info(f"Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
        logger.info(f"Columns: {list(self.df.columns)}")
        
        # Auto-detect columns
        self._auto_detect_columns()
        
        return self.df
    
    def _auto_detect_columns(self) -> None:
        """Auto-detect label, path, and ID columns."""
        if self.df is None:
            return
        
        columns = list(self.df.columns)
        
        # Detect label column
        for pattern in self.LABEL_PATTERNS:
            if pattern in columns:
                self.label_col = pattern
                logger.info(f"Auto-detected label column: {pattern}")
                break
        
        # Detect path column
        for pattern in self.PATH_PATTERNS:
            if pattern in columns:
                self.path_col = pattern
                logger.info(f"Auto-detected path column: {pattern}")
                break
        
        # Detect ID column
        for pattern in self.ID_PATTERNS:
            if pattern in columns:
                self.id_col = pattern
                logger.info(f"Auto-detected ID column: {pattern}")
                break
    
    def get_available_columns(self) -> List[str]:
        """
        Get list of available columns.
        
        Returns:
            List of column names
        """
        if self.df is None:
            return []
        return list(self.df.columns)
    
    def set_column_mapping(self, 
                          label_col: str, 
                          path_col: str,
                          id_col: Optional[str] = None) -> None:
        """
        Manually set column mapping.
        
        Args:
            label_col: Column name for labels
            path_col: Column name for file paths
            id_col: Optional column name for sample IDs
        """
        if self.df is None:
            raise ValueError("No CSV loaded. Call load_csv() first.")
        
        columns = list(self.df.columns)
        
        if label_col not in columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV")
        if path_col not in columns:
            raise ValueError(f"Path column '{path_col}' not found in CSV")
        if id_col is not None and id_col not in columns:
            raise ValueError(f"ID column '{id_col}' not found in CSV")
        
        self.label_col = label_col
        self.path_col = path_col
        self.id_col = id_col
        
        logger.info(f"Column mapping set: label={label_col}, path={path_col}, id={id_col}")
    
    def _normalize_label(self, label) -> int:
        """
        Normalize label to binary (0=HEALTHY, 1=CRACK).
        
        Args:
            label: Raw label value (int, str, float)
            
        Returns:
            Normalized binary label
            
        Raises:
            ValueError: If label cannot be normalized
        """
        # Handle numeric labels
        if isinstance(label, (int, float)):
            if label in [0, 1]:
                return int(label)
            elif label == 0.0:
                return 0
            elif label == 1.0:
                return 1
            else:
                raise ValueError(f"Invalid numeric label: {label}. Expected 0 or 1.")
        
        # Handle string labels
        if isinstance(label, str):
            label_lower = label.lower().strip()
            
            # CRACK variants
            if label_lower in ['crack', 'cracked', '1', 'yes', 'positive', 'true']:
                return 1
            
            # HEALTHY variants
            elif label_lower in ['healthy', 'normal', '0', 'no', 'negative', 'false']:
                return 0
            
            else:
                raise ValueError(f"Unknown string label: '{label}'")
        
        raise ValueError(f"Unsupported label type: {type(label)}")
    
    def to_samples(self, base_path: Optional[str] = None) -> List[Sample]:
        """
        Convert DataFrame to list of Sample objects.
        
        Args:
            base_path: Optional base directory to resolve relative paths
            
        Returns:
            List of Sample objects
            
        Raises:
            ValueError: If columns are not set or labels cannot be normalized
        """
        if self.df is None:
            raise ValueError("No CSV loaded. Call load_csv() first.")
        
        if self.label_col is None or self.path_col is None:
            raise ValueError("Column mapping not set. Call set_column_mapping() first.")
        
        samples = []
        errors = []
        
        for idx, row in self.df.iterrows():
            try:
                # Get raw values
                raw_label = row[self.label_col]
                raw_path = row[self.path_col]
                
                # Normalize label
                label = self._normalize_label(raw_label)
                
                # Resolve path
                path = str(raw_path)
                if base_path is not None:
                    path_obj = Path(path)
                    if not path_obj.is_absolute():
                        path = str(Path(base_path) / path_obj)
                
                # Get sample ID
                sample_id = None
                if self.id_col is not None:
                    sample_id = str(row[self.id_col])
                else:
                    sample_id = f"sample_{idx}"
                
                # Create sample
                sample = Sample(
                    path=path,
                    label=label,
                    sample_id=sample_id,
                    metadata={'row_index': idx}
                )
                samples.append(sample)
                
            except Exception as e:
                error_msg = f"Row {idx}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during sample conversion")
            if len(errors) == len(self.df):
                raise ValueError("All rows failed to convert. Check column mapping and label format.")
        
        logger.info(f"Created {len(samples)} samples from CSV")
        logger.info(f"Label distribution: HEALTHY={sum(1 for s in samples if s.label == 0)}, "
                   f"CRACK={sum(1 for s in samples if s.label == 1)}")
        
        return samples
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of loaded dataset.
        
        Returns:
            Dictionary with summary info
        """
        if self.df is None:
            return {'loaded': False}
        
        summary = {
            'loaded': True,
            'num_rows': len(self.df),
            'num_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'label_col': self.label_col,
            'path_col': self.path_col,
            'id_col': self.id_col,
        }
        
        if self.label_col is not None:
            try:
                label_counts = self.df[self.label_col].value_counts().to_dict()
                summary['label_distribution'] = label_counts
            except:
                pass
        
        return summary

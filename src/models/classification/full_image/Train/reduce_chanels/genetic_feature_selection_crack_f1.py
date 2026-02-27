"""
Genetic Algorithm Feature Selection for Crack Detection in Hyperspectral Imaging
===================================================================================

This script implements a Genetic Algorithm (GA) to optimize feature (wavelength) selection
for multi-class classification of grape hyperspectral images, with the primary objective
of maximizing F1-score for the CRACK class.

Author: ML Research Team
Date: 2025-12-09
Purpose: Thesis research - Optimal wavelength selection for crack detection

Key Features:
- Multi-class classification with focus on CRACK class F1 optimization
- Genetic Algorithm with tournament selection, elitism, crossover & mutation
- Handles class imbalance via RandomOverSampler on training set
- Comprehensive logging and visualization for research thesis
- High-quality plots (300 DPI) for publication
- Threshold optimization for CRACK class
- Feature stability and importance analysis

Dependencies:
- numpy, pandas, scikit-learn, imblearn, matplotlib, seaborn, joblib, tqdm
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    fbeta_score
)

from imblearn.over_sampling import RandomOverSampler
_PROJECT_ROOT = Path(__file__).resolve().parents[6]

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_PATH = str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/dataset/cleaned_0.001/all_classes_cleaned_2025-11-01.csv")
RESULTS_FOLDER = str(_PROJECT_ROOT / r"results/ga_feature_selection_crack_f1")

# Wavelengths mapping (from wavelengths.py)
WAVELENGTHS = {
    1: 397.32, 2: 400.20, 3: 403.09, 4: 405.97, 5: 408.85,
    6: 411.74, 7: 414.63, 8: 417.52, 9: 420.40, 10: 423.29,
    11: 426.19, 12: 429.08, 13: 431.97, 14: 434.87, 15: 437.76,
    16: 440.66, 17: 443.56, 18: 446.45, 19: 449.35, 20: 452.25,
    21: 455.16, 22: 458.06, 23: 460.96, 24: 463.87, 25: 466.77,
    26: 469.68, 27: 472.59, 28: 475.50, 29: 478.41, 30: 481.32,
    31: 484.23, 32: 487.14, 33: 490.06, 34: 492.97, 35: 495.89,
    36: 498.80, 37: 501.72, 38: 504.64, 39: 507.56, 40: 510.48,
    41: 513.40, 42: 516.33, 43: 519.25, 44: 522.18, 45: 525.10,
    46: 528.03, 47: 530.96, 48: 533.89, 49: 536.82, 50: 539.75,
    51: 542.68, 52: 545.62, 53: 548.55, 54: 551.49, 55: 554.43,
    56: 557.36, 57: 560.30, 58: 563.24, 59: 566.18, 60: 569.12,
    61: 572.07, 62: 575.01, 63: 577.96, 64: 580.90, 65: 583.85,
    66: 586.80, 67: 589.75, 68: 592.70, 69: 595.65, 70: 598.60,
    71: 601.55, 72: 604.51, 73: 607.46, 74: 610.42, 75: 613.38,
    76: 616.34, 77: 619.30, 78: 622.26, 79: 625.22, 80: 628.18,
    81: 631.15, 82: 634.11, 83: 637.08, 84: 640.04, 85: 643.01,
    86: 645.98, 87: 648.95, 88: 651.92, 89: 654.89, 90: 657.87,
    91: 660.84, 92: 663.81, 93: 666.79, 94: 669.77, 95: 672.75,
    96: 675.73, 97: 678.71, 98: 681.69, 99: 684.67, 100: 687.65,
    101: 690.64, 102: 693.62, 103: 696.61, 104: 699.60, 105: 702.58,
    106: 705.57, 107: 708.57, 108: 711.56, 109: 714.55, 110: 717.54,
    111: 720.54, 112: 723.53, 113: 726.53, 114: 729.53, 115: 732.53,
    116: 735.53, 117: 738.53, 118: 741.53, 119: 744.53, 120: 747.54,
    121: 750.54, 122: 753.55, 123: 756.56, 124: 759.56, 125: 762.57,
    126: 765.58, 127: 768.60, 128: 771.61, 129: 774.62, 130: 777.64,
    131: 780.65, 132: 783.67, 133: 786.68, 134: 789.70, 135: 792.72,
    136: 795.74, 137: 798.77, 138: 801.79, 139: 804.81, 140: 807.84,
    141: 810.86, 142: 813.89, 143: 816.92, 144: 819.95, 145: 822.98,
    146: 826.01, 147: 829.04, 148: 832.07, 149: 835.11, 150: 838.14,
    151: 841.18, 152: 844.22, 153: 847.25, 154: 850.29, 155: 853.33,
    156: 856.37, 157: 859.42, 158: 862.46, 159: 865.50, 160: 868.55,
    161: 871.60, 162: 874.64, 163: 877.69, 164: 880.74, 165: 883.79,
    166: 886.84, 167: 889.90, 168: 892.95, 169: 896.01, 170: 899.06,
    171: 902.12, 172: 905.18, 173: 908.24, 174: 911.30, 175: 914.36,
    176: 917.42, 177: 920.48, 178: 923.55, 179: 926.61, 180: 929.68,
    181: 932.74, 182: 935.81, 183: 938.88, 184: 941.95, 185: 945.02,
    186: 948.10, 187: 951.17, 188: 954.24, 189: 957.32, 190: 960.40,
    191: 963.47, 192: 966.55, 193: 969.63, 194: 972.71, 195: 975.79,
    196: 978.88, 197: 981.96, 198: 985.05, 199: 988.13, 200: 991.22,
    201: 994.31, 202: 997.40, 203: 1000.49, 204: 1003.58,
}

# Columns to exclude from features
EXCLUDE_COLUMNS = ['label', 'is_outlier', 'json_file', 'hs_dir', 'x', 'y', 'timestamp', 'mask_path']

# GA Hyperparameters
GA_CONFIG = {
    'population_size': 60,
    'num_generations': 40,
    'crossover_rate': 0.8,
    'mutation_rate': 0.05,  # Reduced for 204 features
    'tournament_size': 4,
    'elitism_count': 2,  # Number of top individuals to carry over
    'min_features': 5,
    'max_features': 50,
    'random_seed': 42,
    'feature_penalty_alpha': 0.002  # Soft penalty for too many features
}

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.20,
    'random_state': 42,
    'target_class': 'CRACK',
    'classifier': 'LDA'  # LinearDiscriminantAnalysis
}

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_results_folder(results_folder: str) -> None:
    """Create results folder structure."""
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_folder, 'plots')).mkdir(exist_ok=True)
    Path(os.path.join(results_folder, 'logs')).mkdir(exist_ok=True)
    Path(os.path.join(results_folder, 'models')).mkdir(exist_ok=True)
    print(f"✓ Results folder created: {results_folder}")


def load_and_prepare_data(data_path: str, exclude_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Load dataset and separate features from labels.

    Returns:
        X: Feature DataFrame
        y: Label Series
        hs_dir: Series of hs_dir grouping column (for group-based splitting)
        feature_names: List of feature column names
    """
    print("\n" + "="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)

    df = pd.read_csv(data_path)
    print(f"✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    # Remove outliers BEFORE splitting
    if 'is_outlier' in df.columns:
        initial_count = len(df)
        df = df[df['is_outlier'] != True]
        removed_count = initial_count - len(df)
        print(f"✓ Removed {removed_count} outliers (is_outlier=True)")
        print(f"✓ Dataset after outlier removal: {df.shape[0]} samples")

    # Separate labels
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain 'label' column")

    # Check for hs_dir column (required for group-based splitting)
    if 'hs_dir' not in df.columns:
        raise ValueError("Dataset must contain 'hs_dir' column for group-based splitting")

    y = df['label']
    hs_dir = df['hs_dir']
    print(f"✓ Label distribution:\n{y.value_counts()}")
    print(f"✓ Unique hs_dir groups: {hs_dir.nunique()}")

    # Extract features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]

    print(f"✓ Features extracted: {len(feature_cols)} wavelength channels")
    print(f"✓ Feature range: {X.min().min():.2f} to {X.max().max():.2f}")

    return X, y, hs_dir, feature_cols


def create_wavelength_mapping(feature_names: List[str]) -> Dict[str, float]:
    """
    Create mapping from feature names to wavelength values.
    Tries to match with WAVELENGTHS dict, fallback to column names.
    """
    wavelength_map = {}

    for fname in feature_names:
        # Try to extract band number from column name (e.g., "band_1" -> 1)
        try:
            if 'band_' in fname.lower():
                band_num = int(fname.lower().split('band_')[-1])
                wavelength_map[fname] = WAVELENGTHS.get(band_num, None)
            elif fname.replace('.', '').replace('nm', '').replace('_', '').isdigit():
                # Column name is already a wavelength
                wavelength_map[fname] = float(fname.replace('nm', '').replace('_', ''))
            else:
                wavelength_map[fname] = None
        except:
            wavelength_map[fname] = None

    # Fill missing with generic names
    for i, (fname, wl) in enumerate(wavelength_map.items()):
        if wl is None:
            wavelength_map[fname] = f"Feature_{i+1}"

    return wavelength_map


def split_and_balance_data(X: pd.DataFrame, y: pd.Series, hs_dir: pd.Series,
                           test_size: float, random_state: int,
                           validation_size: float = 0.2) -> Tuple:
    """
    Split data into train/val/test using GROUP-BASED splitting to eliminate data leakage.

    CRITICAL: Pixels from the same hs_dir (same hyperspectral image) are NEVER split
    across TRAIN, VALIDATION, and TEST sets.

    Apply oversampling ONLY to the training set AFTER grouping.

    Returns:
        X_train_sub, X_val, y_train_sub, y_val, X_train_balanced, X_test, y_train_balanced, y_test
    """
    print("\n" + "="*80)
    print("SPLITTING AND BALANCING DATA (GROUP-BASED - NO DATA LEAKAGE)")
    print("="*80)

    # Get unique hs_dir groups
    unique_groups = hs_dir.unique()
    print(f"✓ Total unique hs_dir groups: {len(unique_groups)}")

    # Create a temporary dataframe to facilitate group-based splitting
    # We'll reset indices to ensure proper alignment
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    hs_dir_reset = hs_dir.reset_index(drop=True)

    # Step 1: Split hs_dir groups into train and temp (val+test)
    # Use GroupShuffleSplit: first split is train vs (val+test)
    gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=test_size + validation_size, random_state=random_state)

    train_idx, temp_idx = next(gss_train_temp.split(X_reset, y_reset, groups=hs_dir_reset))

    # Extract train set
    X_train = X_reset.iloc[train_idx]
    y_train = y_reset.iloc[train_idx]
    train_hs_dir = hs_dir_reset.iloc[train_idx]

    # Extract temp set (will be split into val and test)
    X_temp = X_reset.iloc[temp_idx]
    y_temp = y_reset.iloc[temp_idx]
    temp_hs_dir = hs_dir_reset.iloc[temp_idx]

    print(f"✓ Train set: {len(X_train)} samples from {train_hs_dir.nunique()} hs_dir groups")
    print(f"✓ Temp set (val+test): {len(X_temp)} samples from {temp_hs_dir.nunique()} hs_dir groups")

    # Step 2: Split temp into validation and test sets
    # Calculate the proportion of test within temp
    test_proportion = test_size / (test_size + validation_size)

    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=test_proportion, random_state=random_state)

    val_idx, test_idx = next(gss_val_test.split(X_temp, y_temp, groups=temp_hs_dir))

    # Extract validation and test sets
    X_val = X_temp.iloc[val_idx].reset_index(drop=True)
    y_val = y_temp.iloc[val_idx].reset_index(drop=True)
    val_hs_dir = temp_hs_dir.iloc[val_idx]

    X_test = X_temp.iloc[test_idx].reset_index(drop=True)
    y_test = y_temp.iloc[test_idx].reset_index(drop=True)
    test_hs_dir = temp_hs_dir.iloc[test_idx]

    print(f"✓ Validation set: {len(X_val)} samples from {val_hs_dir.nunique()} hs_dir groups")
    print(f"✓ Test set (WITHHELD): {len(X_test)} samples from {test_hs_dir.nunique()} hs_dir groups")

    # CRITICAL ASSERTION: Verify no overlap in hs_dir groups
    train_hs_dir_set = set(train_hs_dir.unique())
    val_hs_dir_set = set(val_hs_dir.unique())
    test_hs_dir_set = set(test_hs_dir.unique())

    assert train_hs_dir_set.isdisjoint(val_hs_dir_set), "LEAKAGE DETECTED: Train and Val share hs_dir groups!"
    assert train_hs_dir_set.isdisjoint(test_hs_dir_set), "LEAKAGE DETECTED: Train and Test share hs_dir groups!"
    assert val_hs_dir_set.isdisjoint(test_hs_dir_set), "LEAKAGE DETECTED: Val and Test share hs_dir groups!"

    print("\n✓✓✓ GROUP-BASED SPLIT VERIFIED - NO LEAKAGE ✓✓✓")
    print(f"  ✓ Train hs_dir: {len(train_hs_dir_set)} unique groups")
    print(f"  ✓ Val hs_dir: {len(val_hs_dir_set)} unique groups")
    print(f"  ✓ Test hs_dir: {len(test_hs_dir_set)} unique groups")
    print(f"  ✓ No overlap between splits confirmed!")

    # Print class distributions BEFORE balancing
    print(f"\n✓ Class distribution (Train - BEFORE balancing):")
    print(f"{y_train.value_counts()}")
    print(f"\n✓ Class distribution (Validation):")
    print(f"{y_val.value_counts()}")
    print(f"\n✓ Class distribution (Test):")
    print(f"{y_test.value_counts()}")

    # Step 3: Balance training set ONLY using RandomOverSampler
    # This is applied AFTER group-based splitting
    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)

    ros = RandomOverSampler(random_state=random_state)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train_reset, y_train_reset)

    print(f"\n✓ Training set AFTER oversampling: {len(X_train_balanced)} samples")
    print(f"  Class distribution (balanced):\n{pd.Series(y_train_balanced).value_counts()}")

    # Step 4: Create internal train_sub/validation split for GA from balanced training set
    # Use standard train_test_split here (not group-based) since we're working within the already-split training set
    X_train_sub, X_val_from_train, y_train_sub, y_val_from_train = train_test_split(
        X_train_balanced, y_train_balanced,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_balanced
    )

    print(f"\n✓ Internal GA splits (from balanced training set):")
    print(f"  Train_sub (for LDA training): {X_train_sub.shape[0]} samples")
    print(f"  Internal validation (for fitness): {X_val_from_train.shape[0]} samples")

    # Use the original validation set (from group split) as the actual validation for GA
    print(f"\n✓ Using group-based validation set for GA fitness evaluation:")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Class distribution:\n{y_val.value_counts()}")

    print(f"\n⚠️  NOTE: Test set will be used ONLY ONCE at the end for final evaluation.")
    print("="*80)

    # Return train_sub and the GROUP-BASED validation set (not the one from train split)
    return X_train_sub, X_val, y_train_sub, y_val, X_train_balanced, X_test, y_train_balanced, y_test


# ============================================================================
# GENETIC ALGORITHM FUNCTIONS
# ============================================================================

class GAIndividual:
    """Represents a GA individual (chromosome) with a binary feature mask."""

    def __init__(self, mask: np.ndarray, fitness: float = 0.0, metrics: Dict = None):
        self.mask = mask.astype(bool)
        self.fitness = fitness
        self.metrics = metrics or {}
        self.num_features = int(np.sum(mask))

    def __repr__(self):
        return f"Individual(features={self.num_features}, fitness={self.fitness:.4f})"


def create_random_individual(n_features: int, min_features: int, max_features: int) -> np.ndarray:
    """Create a random binary mask with constraints."""
    num_selected = np.random.randint(min_features, max_features + 1)
    mask = np.zeros(n_features, dtype=bool)
    selected_indices = np.random.choice(n_features, size=num_selected, replace=False)
    mask[selected_indices] = True
    return mask


def create_initial_population(population_size: int, n_features: int,
                              min_features: int, max_features: int) -> List[GAIndividual]:
    """Initialize GA population with random individuals."""
    population = []
    for _ in range(population_size):
        mask = create_random_individual(n_features, min_features, max_features)
        population.append(GAIndividual(mask))
    return population


def evaluate_individual(individual: GAIndividual,
                       X_train_sub: pd.DataFrame, y_train_sub: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       target_class: str,
                       feature_penalty_alpha: float = 0.0,
                       cache: Dict[str, Dict] = None) -> GAIndividual:
    """
    Evaluate fitness of an individual by training and validating a classifier.

    IMPORTANT: NO TEST LEAKAGE
    - Trains on X_train_sub, y_train_sub
    - Evaluates fitness on X_val, y_val
    - Test set is never used during GA evolution

    Fitness = F1-score for CRACK class (on validation) - alpha * num_features
    Uses cache to avoid redundant evaluations.
    """
    # Check cache
    mask_key = ''.join(['1' if m else '0' for m in individual.mask])
    if cache is not None and mask_key in cache:
        cached = cache[mask_key]
        individual.fitness = cached['fitness']
        individual.metrics = cached['metrics']
        return individual

    # Check minimum features constraint
    if individual.num_features < GA_CONFIG['min_features']:
        individual.fitness = 0.0
        individual.metrics = {
            'f1_crack_val': 0.0,
            'accuracy_val': 0.0,
            'f1_weighted_val': 0.0,
            'precision_crack_val': 0.0,
            'recall_crack_val': 0.0,
            'f2_crack_val': 0.0,
            'raw_f1_crack': 0.0
        }
        return individual

    try:
        # Subset features
        X_train_subset = X_train_sub.iloc[:, individual.mask]
        X_val_subset = X_val.iloc[:, individual.mask]

        # Train classifier on train_sub
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_subset, y_train_sub)

        # Predict on VALIDATION set (not test!)
        y_pred_val = clf.predict(X_val_subset)

        # Compute metrics on validation set
        accuracy = accuracy_score(y_val, y_pred_val)
        f1_weighted = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)

        # Target class metrics on validation
        labels = sorted(y_val.unique())
        if target_class in labels:
            target_idx = labels.index(target_class)
            f1_crack = f1_score(y_val, y_pred_val, labels=labels, average=None, zero_division=0)[target_idx]
            precision_crack = precision_score(y_val, y_pred_val, labels=labels, average=None, zero_division=0)[target_idx]
            recall_crack = recall_score(y_val, y_pred_val, labels=labels, average=None, zero_division=0)[target_idx]
            f2_crack = fbeta_score(y_val, y_pred_val, beta=2, labels=labels, average=None, zero_division=0)[target_idx]
        else:
            f1_crack = precision_crack = recall_crack = f2_crack = 0.0

        # Apply feature penalty to fitness
        raw_f1_crack = f1_crack
        penalized_fitness = f1_crack - feature_penalty_alpha * individual.num_features

        # Store metrics
        individual.fitness = penalized_fitness
        individual.metrics = {
            'f1_crack_val': f1_crack,
            'accuracy_val': accuracy,
            'f1_weighted_val': f1_weighted,
            'precision_crack_val': precision_crack,
            'recall_crack_val': recall_crack,
            'f2_crack_val': f2_crack,
            'raw_f1_crack': raw_f1_crack,
            'penalized_fitness': penalized_fitness
        }

        # Cache result
        if cache is not None:
            cache[mask_key] = {
                'fitness': individual.fitness,
                'metrics': individual.metrics.copy()
            }

    except Exception as e:
        # Handle edge cases (e.g., perfect separation, singular matrix)
        individual.fitness = 0.0
        individual.metrics = {
            'f1_crack_val': 0.0,
            'accuracy_val': 0.0,
            'f1_weighted_val': 0.0,
            'precision_crack_val': 0.0,
            'recall_crack_val': 0.0,
            'f2_crack_val': 0.0,
            'raw_f1_crack': 0.0,
            'penalized_fitness': 0.0
        }

    return individual


def tournament_selection(population: List[GAIndividual], tournament_size: int,
                        num_parents: int) -> List[GAIndividual]:
    """Select parents using tournament selection."""
    selected = []
    for _ in range(num_parents):
        tournament = np.random.choice(population, size=tournament_size, replace=False)
        winner = max(tournament, key=lambda ind: ind.fitness)
        selected.append(winner)
    return selected


def crossover(parent1: GAIndividual, parent2: GAIndividual,
             crossover_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Single-point crossover between two parents."""
    if np.random.rand() > crossover_rate:
        return parent1.mask.copy(), parent2.mask.copy()

    n_features = len(parent1.mask)
    crossover_point = np.random.randint(1, n_features)

    child1_mask = np.concatenate([parent1.mask[:crossover_point], parent2.mask[crossover_point:]])
    child2_mask = np.concatenate([parent2.mask[:crossover_point], parent1.mask[crossover_point:]])

    return child1_mask, child2_mask


def mutate(mask: np.ndarray, mutation_rate: float,
          min_features: int, max_features: int) -> np.ndarray:
    """Flip random bits in the mask with probability mutation_rate."""
    mutated_mask = mask.copy()

    for i in range(len(mutated_mask)):
        if np.random.rand() < mutation_rate:
            mutated_mask[i] = not mutated_mask[i]

    # Enforce constraints
    num_selected = int(np.sum(mutated_mask))
    if num_selected < min_features:
        # Add random features
        available = np.where(~mutated_mask)[0]
        to_add = min_features - num_selected
        if len(available) >= to_add:
            add_indices = np.random.choice(available, size=to_add, replace=False)
            mutated_mask[add_indices] = True
    elif num_selected > max_features:
        # Remove random features
        selected = np.where(mutated_mask)[0]
        to_remove = num_selected - max_features
        remove_indices = np.random.choice(selected, size=to_remove, replace=False)
        mutated_mask[remove_indices] = False

    return mutated_mask


def run_ga_feature_selection(X_train_sub: pd.DataFrame, y_train_sub: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             target_class: str,
                             ga_config: Dict,
                             results_folder: str) -> Tuple[GAIndividual, pd.DataFrame, pd.DataFrame]:
    """
    Main GA loop for feature selection.

    IMPORTANT: NO TEST LEAKAGE
    - Uses only train_sub for training
    - Uses only validation for fitness evaluation
    - Test set is never touched during GA evolution

    Returns:
        best_individual: Best individual found
        generation_log: DataFrame with per-generation statistics
        all_individuals_log: DataFrame with all evaluated individuals
    """
    print("\n" + "="*80)
    print("GENETIC ALGORITHM FEATURE SELECTION")
    print("="*80)
    print(f"⚠️  GA FITNESS EVALUATED ON VALIDATION SET ONLY – TEST WITHHELD")
    print("="*80)
    print(f"Configuration:")
    for key, value in ga_config.items():
        print(f"  {key}: {value}")
    print("="*80)

    np.random.seed(ga_config['random_seed'])

    n_features = X_train_sub.shape[1]
    population_size = ga_config['population_size']
    num_generations = ga_config['num_generations']
    crossover_rate = ga_config['crossover_rate']
    mutation_rate = ga_config['mutation_rate']
    tournament_size = ga_config['tournament_size']
    elitism_count = ga_config['elitism_count']
    min_features = ga_config['min_features']
    max_features = ga_config['max_features']
    feature_penalty_alpha = ga_config.get('feature_penalty_alpha', 0.0)

    # Initialize
    population = create_initial_population(population_size, n_features, min_features, max_features)
    cache = {}  # Cache for evaluated masks

    global_best = None
    generation_stats = []
    all_individuals_data = []

    # Evolution loop
    for generation in tqdm(range(num_generations), desc="GA Evolution"):
        # Evaluate fitness (on validation set only!)
        for individual in population:
            evaluate_individual(individual, X_train_sub, y_train_sub, X_val, y_val,
                              target_class, feature_penalty_alpha, cache)

        # Sort by fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Update global best
        gen_best = population[0]
        if global_best is None or gen_best.fitness > global_best.fitness:
            global_best = GAIndividual(
                mask=gen_best.mask.copy(),
                fitness=gen_best.fitness,
                metrics=gen_best.metrics.copy()
            )

        # Log generation statistics
        fitnesses = [ind.fitness for ind in population]
        raw_f1_scores = [ind.metrics.get('raw_f1_crack', ind.fitness) for ind in population]

        gen_stats = {
            'generation': generation,
            'best_penalized_fitness': gen_best.fitness,
            'best_raw_f1_crack_val': gen_best.metrics.get('raw_f1_crack', gen_best.fitness),
            'mean_penalized_fitness': np.mean(fitnesses),
            'mean_raw_f1_crack_val': np.mean(raw_f1_scores),
            'std_penalized_fitness': np.std(fitnesses),
            'std_raw_f1_crack_val': np.std(raw_f1_scores),
            'best_num_features': gen_best.num_features,
            'best_accuracy_val': gen_best.metrics.get('accuracy_val', 0.0),
            'best_f1_weighted_val': gen_best.metrics.get('f1_weighted_val', 0.0),
            'best_precision_crack_val': gen_best.metrics.get('precision_crack_val', 0.0),
            'best_recall_crack_val': gen_best.metrics.get('recall_crack_val', 0.0),
            'best_f2_crack_val': gen_best.metrics.get('f2_crack_val', 0.0)
        }
        generation_stats.append(gen_stats)

        # Log all individuals
        for idx, ind in enumerate(population):
            ind_data = {
                'generation': generation,
                'individual_id': idx,
                'mask_string': ''.join(['1' if m else '0' for m in ind.mask]),
                'num_features': ind.num_features,
                'penalized_fitness': ind.fitness,
                'raw_f1_crack_val': ind.metrics.get('raw_f1_crack', ind.fitness),
                'accuracy_val': ind.metrics.get('accuracy_val', 0.0),
                'f1_weighted_val': ind.metrics.get('f1_weighted_val', 0.0),
                'precision_crack_val': ind.metrics.get('precision_crack_val', 0.0),
                'recall_crack_val': ind.metrics.get('recall_crack_val', 0.0),
                'f2_crack_val': ind.metrics.get('f2_crack_val', 0.0)
            }
            all_individuals_data.append(ind_data)

        # Print progress
        if generation % 5 == 0 or generation == num_generations - 1:
            print(f"\nGeneration {generation}:")
            print(f"  Best Raw F1_CRACK (val): {gen_best.metrics.get('raw_f1_crack', 0):.4f} ({gen_best.num_features} features)")
            print(f"  Best Penalized Fitness: {gen_best.fitness:.4f}")
            print(f"  Mean Raw F1_CRACK (val): {np.mean(raw_f1_scores):.4f} ± {np.std(raw_f1_scores):.4f}")
            print(f"  Global Best Penalized Fitness: {global_best.fitness:.4f}")

        # Create next generation
        if generation < num_generations - 1:
            next_population = []

            # Elitism: carry over top individuals
            next_population.extend([
                GAIndividual(ind.mask.copy(), ind.fitness, ind.metrics.copy())
                for ind in population[:elitism_count]
            ])

            # Generate offspring
            while len(next_population) < population_size:
                # Selection
                parents = tournament_selection(population, tournament_size, 2)

                # Crossover
                child1_mask, child2_mask = crossover(parents[0], parents[1], crossover_rate)

                # Mutation
                child1_mask = mutate(child1_mask, mutation_rate, min_features, max_features)
                child2_mask = mutate(child2_mask, mutation_rate, min_features, max_features)

                # Add to next generation
                next_population.append(GAIndividual(child1_mask))
                if len(next_population) < population_size:
                    next_population.append(GAIndividual(child2_mask))

            population = next_population[:population_size]

    print("\n" + "="*80)
    print("GA EVOLUTION COMPLETE")
    print("="*80)
    print(f"✓ Global Best Raw F1_CRACK (validation): {global_best.metrics.get('raw_f1_crack', 0):.4f}")
    print(f"✓ Global Best Penalized Fitness: {global_best.fitness:.4f}")
    print(f"✓ Number of features selected: {global_best.num_features}")
    print(f"✓ Unique masks evaluated: {len(cache)}")

    # Create DataFrames
    generation_log = pd.DataFrame(generation_stats)
    all_individuals_log = pd.DataFrame(all_individuals_data)

    # Save logs
    generation_log.to_csv(os.path.join(results_folder, 'logs', 'ga_generation_log.csv'), index=False)
    all_individuals_log.to_csv(os.path.join(results_folder, 'logs', 'ga_all_individuals_log.csv'), index=False)

    print(f"✓ Logs saved to: {os.path.join(results_folder, 'logs')}")

    return global_best, generation_log, all_individuals_log


# ============================================================================
# FINAL MODEL EVALUATION
# ============================================================================

def evaluate_best_model(best_individual: GAIndividual,
                       X_train_balanced: pd.DataFrame, y_train_balanced: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       feature_names: List[str],
                       target_class: str,
                       results_folder: str) -> Tuple[Any, Dict]:
    """
    Train final model with best features on FULL balanced training set.
    Evaluate on test set (USED ONLY ONCE HERE - NO LEAKAGE).
    """
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION ON TEST SET (USED ONLY ONCE)")
    print("="*80)

    # Get selected features
    selected_features = [feature_names[i] for i, selected in enumerate(best_individual.mask) if selected]
    print(f"✓ Selected {len(selected_features)} features")

    # Subset data using best GA mask
    X_train_subset = X_train_balanced.iloc[:, best_individual.mask]
    X_test_subset = X_test.iloc[:, best_individual.mask]

    print(f"\n✓ Training final model on FULL balanced training set:")
    print(f"  Training samples: {X_train_subset.shape[0]}")
    print(f"  Selected features: {X_train_subset.shape[1]}")

    # Train final model on full balanced training set
    final_model = LinearDiscriminantAnalysis()
    final_model.fit(X_train_subset, y_train_balanced)

    print(f"\n✓ Evaluating on test set:")
    print(f"  Test samples: {X_test_subset.shape[0]}")

    # Predictions on test set
    y_pred = final_model.predict(X_test_subset)
    y_pred_proba = final_model.predict_proba(X_test_subset)

    # Metrics on test set
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"\n✓ Final Model Performance on Test Set:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # CRACK-specific metrics on test set
    crack_metrics = report.get(target_class, {})
    print(f"\n✓ {target_class} Class Metrics (Test Set):")
    print(f"  Precision: {crack_metrics.get('precision', 0.0):.4f}")
    print(f"  Recall: {crack_metrics.get('recall', 0.0):.4f}")
    print(f"  F1-score: {crack_metrics.get('f1-score', 0.0):.4f}")

    # Compute F2 score for CRACK
    labels_test = sorted(y_test.unique())
    f2_crack_test = 0.0
    if target_class in labels_test:
        target_idx = labels_test.index(target_class)
        f2_crack_test = fbeta_score(y_test, y_pred, beta=2, labels=labels_test, average=None, zero_division=0)[target_idx]
        print(f"  F2-score: {f2_crack_test:.4f}")

    # Save model
    model_path = os.path.join(results_folder, 'models', 'ga_selected_lda_model.pkl')
    joblib.dump({
        'model': final_model,
        'selected_features': selected_features,
        'feature_mask': best_individual.mask,
        'ga_validation_metrics': best_individual.metrics,  # Metrics from validation during GA
        'test_metrics': {  # Metrics from final test evaluation
            'accuracy': accuracy,
            'crack_precision': crack_metrics.get('precision', 0.0),
            'crack_recall': crack_metrics.get('recall', 0.0),
            'crack_f1': crack_metrics.get('f1-score', 0.0),
            'crack_f2': f2_crack_test
        },
        'classes': final_model.classes_
    }, model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Save selected features list
    wavelength_map = create_wavelength_mapping(feature_names)
    selected_features_df = pd.DataFrame({
        'feature_name': selected_features,
        'feature_index': [i for i, sel in enumerate(best_individual.mask) if sel],
        'wavelength_nm': [wavelength_map.get(f, 'N/A') for f in selected_features]
    })

    # Add LDA coefficients if available
    if hasattr(final_model, 'coef_') and final_model.coef_ is not None:
        # For multi-class, take mean absolute coefficient across all discriminants
        coefs = np.abs(final_model.coef_).mean(axis=0)
        selected_features_df['lda_coefficient'] = coefs
        selected_features_df['importance_rank'] = selected_features_df['lda_coefficient'].rank(ascending=False)

    selected_features_df.to_csv(
        os.path.join(results_folder, 'logs', 'ga_selected_features.csv'),
        index=False
    )
    print(f"✓ Selected features saved to: {os.path.join(results_folder, 'logs', 'ga_selected_features.csv')}")

    results = {
        'model': final_model,
        'selected_features': selected_features,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy
    }

    return final_model, results


def optimize_crack_threshold(model: Any,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             target_class: str,
                             results_folder: str) -> Tuple[float, pd.DataFrame]:
    """
    Optimize decision threshold for CRACK class to maximize F1.
    """
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION FOR CRACK CLASS")
    print("="*80)

    # Get probability predictions
    y_proba = model.predict_proba(X_test)
    classes = model.classes_

    if target_class not in classes:
        print(f"✗ {target_class} not in classes: {classes}")
        return 0.5, pd.DataFrame()

    crack_idx = list(classes).index(target_class)
    crack_proba = y_proba[:, crack_idx]

    # Test thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold_results = []

    for threshold in thresholds:
        # Apply threshold: if CRACK probability >= threshold, predict CRACK
        # Otherwise, predict argmax of remaining classes
        y_pred_thresh = []
        for i, sample_proba in enumerate(y_proba):
            if sample_proba[crack_idx] >= threshold:
                y_pred_thresh.append(target_class)
            else:
                # Choose best among non-CRACK classes
                other_proba = sample_proba.copy()
                other_proba[crack_idx] = -1  # Exclude CRACK
                pred_idx = np.argmax(other_proba)
                y_pred_thresh.append(classes[pred_idx])

        # Compute metrics
        labels = sorted(y_test.unique())
        if target_class in labels:
            target_idx_labels = labels.index(target_class)
            f1_crack = f1_score(y_test, y_pred_thresh, labels=labels, average=None, zero_division=0)[target_idx_labels]
            precision_crack = precision_score(y_test, y_pred_thresh, labels=labels, average=None, zero_division=0)[target_idx_labels]
            recall_crack = recall_score(y_test, y_pred_thresh, labels=labels, average=None, zero_division=0)[target_idx_labels]
        else:
            f1_crack = precision_crack = recall_crack = 0.0

        threshold_results.append({
            'threshold': threshold,
            'f1_crack': f1_crack,
            'precision_crack': precision_crack,
            'recall_crack': recall_crack
        })

    threshold_df = pd.DataFrame(threshold_results)

    # Find best threshold
    best_idx = threshold_df['f1_crack'].idxmax()
    best_threshold = threshold_df.loc[best_idx, 'threshold']
    best_f1 = threshold_df.loc[best_idx, 'f1_crack']

    print(f"\n✓ Best Threshold: {best_threshold:.3f}")
    print(f"✓ F1_CRACK at best threshold: {best_f1:.4f}")
    print(f"✓ Precision at best threshold: {threshold_df.loc[best_idx, 'precision_crack']:.4f}")
    print(f"✓ Recall at best threshold: {threshold_df.loc[best_idx, 'recall_crack']:.4f}")

    # Save threshold optimization log
    threshold_df.to_csv(
        os.path.join(results_folder, 'logs', 'threshold_optimization_log.csv'),
        index=False
    )
    print(f"✓ Threshold optimization log saved")

    return best_threshold, threshold_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_ga_convergence(generation_log: pd.DataFrame, results_folder: str) -> None:
    """Plot GA convergence: best and mean F1_CRACK (validation) over generations."""
    print("\n✓ Generating convergence plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot best raw F1 (validation)
    ax.plot(generation_log['generation'], generation_log['best_raw_f1_crack_val'],
            marker='o', linewidth=2.5, markersize=6, label='Best F1_CRACK (Validation)', color='#d62728')

    # Plot mean raw F1 (validation)
    ax.plot(generation_log['generation'], generation_log['mean_raw_f1_crack_val'],
            marker='s', linewidth=2, markersize=5, alpha=0.7, label='Mean F1_CRACK (Validation)', color='#1f77b4')

    # Fill between for std
    ax.fill_between(
        generation_log['generation'],
        generation_log['mean_raw_f1_crack_val'] - generation_log['std_raw_f1_crack_val'],
        generation_log['mean_raw_f1_crack_val'] + generation_log['std_raw_f1_crack_val'],
        alpha=0.2, color='#1f77b4'
    )

    # Mark final best
    final_gen = generation_log['generation'].max()
    final_best = generation_log['best_raw_f1_crack_val'].max()
    ax.scatter([final_gen], [final_best], s=300, c='gold', edgecolor='black',
               linewidth=2, zorder=5, label='Global Best', marker='*')

    ax.set_xlabel('Generation', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score (CRACK Class on Validation)', fontsize=14, fontweight='bold')
    ax.set_title('Genetic Algorithm Convergence: CRACK F1 Optimization (Validation Set)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'ga_convergence.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_feature_selection_heatmap(generation_log: pd.DataFrame,
                                   all_individuals_log: pd.DataFrame,
                                   feature_names: List[str],
                                   results_folder: str) -> None:
    """
    Heatmap showing which features were selected in the best individual per generation.
    """
    print("\n✓ Generating feature selection heatmap...")

    n_features = len(feature_names)
    n_generations = len(generation_log)

    # Build selection matrix: rows = generations, cols = features
    selection_matrix = np.zeros((n_generations, n_features), dtype=int)

    for gen in range(n_generations):
        # Get best individual of this generation
        gen_individuals = all_individuals_log[all_individuals_log['generation'] == gen]
        best_ind = gen_individuals.loc[gen_individuals['raw_f1_crack_val'].idxmax()]
        mask_str = best_ind['mask_string']

        for i, bit in enumerate(mask_str):
            if i < n_features:
                selection_matrix[gen, i] = int(bit)

    # Create wavelength labels
    wavelength_map = create_wavelength_mapping(feature_names)
    wavelength_labels = []
    for fname in feature_names:
        wl = wavelength_map[fname]
        if isinstance(wl, (int, float)):
            wavelength_labels.append(f"{wl:.1f}nm")
        else:
            wavelength_labels.append(str(wl)[:15])  # Truncate long names

    # Plot heatmap (subsample features if too many)
    max_features_display = 50
    if n_features > max_features_display:
        # Show only features that were selected at least once
        feature_selection_counts = selection_matrix.sum(axis=0)
        top_feature_indices = np.argsort(feature_selection_counts)[-max_features_display:]
        selection_matrix = selection_matrix[:, top_feature_indices]
        wavelength_labels = [wavelength_labels[i] for i in top_feature_indices]

    fig, ax = plt.subplots(figsize=(16, 10))

    sns.heatmap(selection_matrix.T, cmap='RdYlGn', cbar_kws={'label': 'Selected (1) / Not Selected (0)'},
                xticklabels=5, yticklabels=wavelength_labels, linewidths=0.1, linecolor='gray',
                ax=ax, vmin=0, vmax=1)

    ax.set_xlabel('Generation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Wavelength', fontsize=14, fontweight='bold')
    ax.set_title('GA Feature Selection Heatmap: Best Individual per Generation',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'ga_feature_selection_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_feature_frequency_bar(all_individuals_log: pd.DataFrame,
                               feature_names: List[str],
                               results_folder: str,
                               top_n: int = 20) -> None:
    """
    Bar plot showing how often each feature was selected in the best individuals.
    """
    print("\n✓ Generating feature frequency bar plot...")

    n_features = len(feature_names)

    # Count selection frequency across all best individuals per generation
    generation_best_masks = []
    for gen in all_individuals_log['generation'].unique():
        gen_data = all_individuals_log[all_individuals_log['generation'] == gen]
        best_mask_str = gen_data.loc[gen_data['raw_f1_crack_val'].idxmax(), 'mask_string']
        generation_best_masks.append(best_mask_str)

    # Count frequency
    feature_counts = np.zeros(n_features, dtype=int)
    for mask_str in generation_best_masks:
        for i, bit in enumerate(mask_str):
            if i < n_features and bit == '1':
                feature_counts[i] += 1

    # Create DataFrame
    wavelength_map = create_wavelength_mapping(feature_names)
    feature_freq_df = pd.DataFrame({
        'feature_name': feature_names,
        'wavelength_nm': [wavelength_map[f] for f in feature_names],
        'selection_count': feature_counts,
        'selection_percentage': (feature_counts / len(generation_best_masks)) * 100
    })

    # Sort by frequency
    feature_freq_df = feature_freq_df.sort_values('selection_count', ascending=False)

    # Take top N
    top_features = feature_freq_df.head(top_n)

    # Create labels
    labels = []
    for _, row in top_features.iterrows():
        wl = row['wavelength_nm']
        if isinstance(wl, (int, float)):
            labels.append(f"{wl:.1f}nm")
        else:
            labels.append(str(row['feature_name'])[:15])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(range(len(top_features)), top_features['selection_percentage'],
                   color='steelblue', edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['selection_percentage'])):
        ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Selection Frequency (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Wavelength', fontsize=14, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Frequently Selected Wavelengths (GA Feature Selection)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'ga_feature_frequency_bar.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")

    # Save frequency data
    feature_freq_df.to_csv(
        os.path.join(results_folder, 'logs', 'feature_selection_frequency.csv'),
        index=False
    )


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], results_folder: str) -> None:
    """Plot confusion matrix heatmap."""
    print("\n✓ Generating confusion matrix...")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=1, linecolor='gray', ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix: GA-Selected LDA Model on Test Set',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_performance_vs_features(all_individuals_log: pd.DataFrame,
                                 best_individual: GAIndividual,
                                 results_folder: str) -> None:
    """Scatter plot: F1_CRACK (validation) vs number of selected features."""
    print("\n✓ Generating performance vs. features plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Scatter all individuals (use raw F1 from validation)
    ax.scatter(all_individuals_log['num_features'],
               all_individuals_log['raw_f1_crack_val'],
               alpha=0.4, s=30, c='steelblue', edgecolor='none', label='All Individuals')

    # Highlight best (use raw F1, not penalized)
    best_raw_f1 = best_individual.metrics.get('raw_f1_crack', best_individual.fitness)
    ax.scatter([best_individual.num_features], [best_raw_f1],
               s=500, c='gold', edgecolor='black', linewidth=2.5,
               marker='*', zorder=5, label='Best Individual')

    ax.set_xlabel('Number of Selected Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score (CRACK Class on Validation)', fontsize=14, fontweight='bold')
    ax.set_title('Performance vs. Feature Subset Size (GA Feature Selection - Validation)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'performance_vs_features.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


def plot_threshold_optimization(threshold_df: pd.DataFrame,
                                best_threshold: float,
                                results_folder: str) -> None:
    """Plot threshold optimization curve."""
    print("\n✓ Generating threshold optimization plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(threshold_df['threshold'], threshold_df['f1_crack'],
            linewidth=2.5, label='F1-Score', color='#d62728', marker='o', markersize=4)
    ax.plot(threshold_df['threshold'], threshold_df['precision_crack'],
            linewidth=2, label='Precision', color='#2ca02c', alpha=0.7, linestyle='--')
    ax.plot(threshold_df['threshold'], threshold_df['recall_crack'],
            linewidth=2, label='Recall', color='#ff7f0e', alpha=0.7, linestyle='--')

    # Mark best threshold
    best_f1 = threshold_df[threshold_df['threshold'] == best_threshold]['f1_crack'].values[0]
    ax.axvline(best_threshold, color='black', linestyle=':', linewidth=2,
               label=f'Best Threshold = {best_threshold:.3f}')
    ax.scatter([best_threshold], [best_f1], s=300, c='gold', edgecolor='black',
               linewidth=2, zorder=5, marker='*')

    ax.set_xlabel('Decision Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Threshold Optimization for CRACK Class Detection',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'plots', 'threshold_optimization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {plot_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    start_time = time.time()

    print("\n" + "="*80)
    print(" GENETIC ALGORITHM FEATURE SELECTION FOR CRACK DETECTION ")
    print(" Hyperspectral Imaging - Multi-class Classification ")
    print("="*80)

    # Setup
    setup_results_folder(RESULTS_FOLDER)

    # Load data
    X, y, hs_dir, feature_names = load_and_prepare_data(DATA_PATH, EXCLUDE_COLUMNS)

    # Split and balance (creates train_sub/val/test splits using GROUP-BASED splitting)
    X_train_sub, X_val, y_train_sub, y_val, X_train_bal, X_test, y_train_bal, y_test = split_and_balance_data(
        X, y, hs_dir, MODEL_CONFIG['test_size'], MODEL_CONFIG['random_state'], validation_size=0.2
    )

    # Run GA feature selection (uses ONLY train_sub and validation - NO TEST!)
    best_individual, generation_log, all_individuals_log = run_ga_feature_selection(
        X_train_sub, y_train_sub,
        X_val, y_val,
        MODEL_CONFIG['target_class'],
        GA_CONFIG,
        RESULTS_FOLDER
    )

    # Evaluate best model on TEST SET (used only ONCE here)
    final_model, results = evaluate_best_model(
        best_individual,
        X_train_bal, y_train_bal,  # Train on full balanced training set
        X_test, y_test,  # Evaluate on test set
        feature_names,
        MODEL_CONFIG['target_class'],
        RESULTS_FOLDER
    )

    # Threshold optimization
    best_threshold, threshold_df = optimize_crack_threshold(
        final_model,
        X_test.iloc[:, best_individual.mask],
        y_test,
        MODEL_CONFIG['target_class'],
        RESULTS_FOLDER
    )

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_ga_convergence(generation_log, RESULTS_FOLDER)
    plot_feature_selection_heatmap(generation_log, all_individuals_log, feature_names, RESULTS_FOLDER)
    plot_feature_frequency_bar(all_individuals_log, feature_names, RESULTS_FOLDER, top_n=20)
    plot_confusion_matrix(results['confusion_matrix'], final_model.classes_, RESULTS_FOLDER)
    plot_performance_vs_features(all_individuals_log, best_individual, RESULTS_FOLDER)

    if not threshold_df.empty:
        plot_threshold_optimization(threshold_df, best_threshold, RESULTS_FOLDER)

    # Save summary report
    print("\n" + "="*80)
    print("SAVING SUMMARY REPORT")
    print("="*80)

    report_path = os.path.join(RESULTS_FOLDER, 'GA_FEATURE_SELECTION_SUMMARY.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GENETIC ALGORITHM FEATURE SELECTION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Total samples: {len(X)}\n")
        f.write(f"  Total features: {len(feature_names)}\n")
        f.write(f"  Training samples (balanced): {len(X_train_bal)}\n")
        f.write(f"  Test samples: {len(X_test)}\n\n")

        f.write("GA Configuration:\n")
        for key, value in GA_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Best Individual Found by GA (Validation Set Performance):\n")
        f.write(f"  Raw F1-Score (CRACK, validation): {best_individual.metrics.get('raw_f1_crack', 0):.4f}\n")
        f.write(f"  Penalized Fitness: {best_individual.fitness:.4f}\n")
        f.write(f"  Number of features: {best_individual.num_features}\n")
        f.write(f"  Accuracy (validation): {best_individual.metrics.get('accuracy_val', 0):.4f}\n")
        f.write(f"  Weighted F1 (validation): {best_individual.metrics.get('f1_weighted_val', 0):.4f}\n")
        f.write(f"  CRACK Precision (validation): {best_individual.metrics.get('precision_crack_val', 0):.4f}\n")
        f.write(f"  CRACK Recall (validation): {best_individual.metrics.get('recall_crack_val', 0):.4f}\n")
        f.write(f"  CRACK F2-Score (validation): {best_individual.metrics.get('f2_crack_val', 0):.4f}\n\n")

        f.write("Final Model Performance on Test Set (NO LEAKAGE - Used Only Once):\n")
        f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
        crack_report = results['classification_report'].get(MODEL_CONFIG['target_class'], {})
        f.write(f"  CRACK Precision: {crack_report.get('precision', 0):.4f}\n")
        f.write(f"  CRACK Recall: {crack_report.get('recall', 0):.4f}\n")
        f.write(f"  CRACK F1-Score: {crack_report.get('f1-score', 0):.4f}\n\n")

        f.write("Threshold Optimization:\n")
        f.write(f"  Best threshold: {best_threshold:.3f}\n\n")

        f.write("Selected Wavelengths:\n")
        selected_feats = [feature_names[i] for i, sel in enumerate(best_individual.mask) if sel]
        wavelength_map = create_wavelength_mapping(feature_names)
        for feat in selected_feats:
            wl = wavelength_map[feat]
            if isinstance(wl, (int, float)):
                f.write(f"  - {feat}: {wl:.2f} nm\n")
            else:
                f.write(f"  - {feat}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Classification Report:\n")
        f.write("="*80 + "\n")
        f.write(classification_report(y_test, results['y_pred'], zero_division=0))

        f.write("\n" + "="*80 + "\n")
        f.write(f"Total execution time: {time.time() - start_time:.2f} seconds\n")
        f.write("="*80 + "\n")

    print(f"✓ Summary report saved to: {report_path}")

    # Final summary
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"✓ Best F1_CRACK (validation): {best_individual.metrics.get('raw_f1_crack', 0):.4f}")
    print(f"✓ Best F1_CRACK (test set): {results['classification_report'].get(MODEL_CONFIG['target_class'], {}).get('f1-score', 0):.4f}")
    print(f"✓ Selected features: {best_individual.num_features}/{len(feature_names)}")
    print(f"✓ Best threshold: {best_threshold:.3f}")
    print(f"✓ Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"✓ All results saved to: {RESULTS_FOLDER}")
    print("\n⚠️  IMPORTANT: Test set was used ONLY ONCE for final evaluation - NO LEAKAGE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


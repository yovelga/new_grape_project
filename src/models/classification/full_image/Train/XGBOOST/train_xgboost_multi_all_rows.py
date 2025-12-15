import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib  # Add this at the top

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

from wavelengths import WAVELENGTHS  # Import wavelengths mapping

# path to cleaned dataset
DATA_PATH = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\dataset\cleaned_0.001\all_classes_cleaned_2025-11-01.csv"

# folder to save results
RESULTS_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\results\xgboost_multi_2_rows"

# global constants
RANDOM_STATE = 42
TARGET_CLASS = 'CRACK'

# Columns to exclude from features
EXCLUDE_COLUMNS = ['label', 'is_outlier', 'json_file', 'hs_dir', 'x', 'y', 'timestamp', 'mask_path']

#  Stronger regularization to prevent overfitting
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'min_child_weight': 4,  # Crucial for imbalance
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'objective': 'multi:softprob',  # for multi-class classification
    'eval_metric': 'mlogloss', # multi-class logloss
    'n_jobs': -1,   # utilize all CPU cores
    'tree_method': 'hist', # faster histogram-based algorithm
    'random_state': RANDOM_STATE
}

plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# DATA LOADING & FILTERING
# ============================================================================

def load_and_filter_data(path):
    print(f"✓ Loading data from: {path}")
    df = pd.read_csv(path)

    # 1. Filter Outliers
    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == 0].copy()

    # 2. Filter Row 1 only (1_01 to 1_60)
    # The hs_dir contains full paths like: C:\...\data\raw\1_01\25.09.24\HS
    # We need to check if the path contains any of these patterns

    initial_len = len(df)

    # Create a filter function that checks if any row1 identifier is in the path
    def is_row1(hs_dir_path):
        if pd.isna(hs_dir_path):
            return False
        path_str = str(hs_dir_path)
        # Check if path contains \1_01\ to \1_60\ or /1_01/ to /1_60/
        for i in range(1, 61):
            pattern = f"1_{i:02d}"
            if f"\\{pattern}\\" in path_str or f"/{pattern}/" in path_str:
                return True
        return False

    df = df[df['hs_dir'].apply(is_row1)].copy()
    filtered_len = len(df)

    print(f"✓ Filtering Row 1 (1_01 to 1_60): Kept {filtered_len} out of {initial_len} samples.")

    # Check if we have data
    if filtered_len == 0:
        print("\n❌ ERROR: No data found after filtering!")
        print("Sample hs_dir values from original data:")
        # Show first 5 unique hs_dir values to help debug
        sample_dirs = df.head(10)['hs_dir'].unique() if 'hs_dir' in df.columns else []
        for idx, dir_val in enumerate(sample_dirs[:5], 1):
            print(f"  {idx}. {dir_val}")
        return df, [], None

    # 3. Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS and c != 'label_encoded']

    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Classes: {list(le.classes_)}")
    print(f"✓ Class distribution:")
    for label in le.classes_:
        count = (df['label'] == label).sum()
        print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")

    # Log all unique hs_dir groups found
    unique_groups = sorted(df['hs_dir'].unique())
    print(f"\n✓ Found {len(unique_groups)} unique grape bunches (hs_dir) in Row 1:")
    for idx, group in enumerate(unique_groups, 1):
        samples_count = len(df[df['hs_dir'] == group])
        print(f"    {idx:2d}. {group} ({samples_count} samples)")

    return df, feature_cols, le


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_wavelength_mapping(feature_names):
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


# ============================================================================
# LOGO GROUP FILTERING
# ============================================================================

def filter_logo_groups(df, le):
    """
    Filter groups for LOGO validation:
    - Only groups with BOTH CRACK and REGULAR samples can be validation folds
    - Groups with only one class are always included in training (never left out)

    Returns:
        logo_validation_groups: Groups that can serve as validation (both classes)
        always_train_groups: Groups that always stay in training (single class)
    """
    crack_code = le.transform(['CRACK'])[0] if 'CRACK' in le.classes_ else -1
    regular_code = le.transform(['REGULAR'])[0] if 'REGULAR' in le.classes_ else -1

    all_groups = sorted(df['hs_dir'].unique())
    logo_validation_groups = []
    always_train_groups = []

    print(f"\n{'='*80}")
    print("FILTERING GROUPS FOR LOGO VALIDATION")
    print(f"{'='*80}")
    print(f"✓ Total unique groups found: {len(all_groups)}")
    print(f"\n✓ Checking each group for CRACK and REGULAR presence:\n")

    for group in all_groups:
        group_data = df[df['hs_dir'] == group]
        group_classes = set(group_data['label_encoded'].unique())

        # Check if group has both CRACK and REGULAR
        has_both = (crack_code in group_classes and regular_code in group_classes)

        if has_both:
            crack_count = len(group_data[group_data['label_encoded'] == crack_code])
            regular_count = len(group_data[group_data['label_encoded'] == regular_code])
            logo_validation_groups.append(group)
            print(f"  ✓ {group}")
            print(f"    → BOTH classes (CRACK={crack_count}, REGULAR={regular_count}) → Validation fold")
        else:
            # Get class names present in this group
            class_names = [le.inverse_transform([cls])[0] for cls in group_classes]
            class_counts = ', '.join([f"{le.inverse_transform([cls])[0]}={len(group_data[group_data['label_encoded'] == cls])}"
                                     for cls in group_classes])
            always_train_groups.append(group)
            print(f"  ✗ {group}")
            print(f"    → Single class only ({class_counts}) → Always in training")

    print(f"\n{'='*80}")
    print("LOGO STRATEGY SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Validation groups (both CRACK and REGULAR): {len(logo_validation_groups)}")
    print(f"✓ Always-train groups (single class only): {len(always_train_groups)}")
    print(f"\n⚠️  IMPORTANT:")
    print(f"  - Only {len(logo_validation_groups)} groups will be used as validation folds")
    print(f"  - {len(always_train_groups)} groups will ALWAYS stay in training (never wasted)")
    print(f"  - This ensures meaningful validation (requires both classes)")
    print(f"{'='*80}\n")

    return logo_validation_groups, always_train_groups


# ============================================================================
# LOGO VALIDATION & PLOTTING
# ============================================================================

def run_logo_validation(df, feature_cols, le, logo_validation_groups, always_train_groups):
    """
    Run LOGO (Leave-One-Group-Out) validation with filtered groups:
    - Only validate on groups that have BOTH CRACK and REGULAR
    - Always include single-class groups in training (never wasted)

    UPDATED: Now includes MCC, Specificity, and per-fold AUC for comprehensive thesis metrics.
    """
    crack_code = le.transform(['CRACK'])[0] if 'CRACK' in le.classes_ else -1

    if crack_code == -1:
        print("⚠️  Warning: CRACK class not found in dataset!")

    results = []

    # Store data for global plots
    global_y_true = []
    global_y_pred = []
    global_probs = []  # For CRACK class ROC curve
    feature_importance_list = []

    print(f"\n✓ Running LOGO Validation on {len(logo_validation_groups)} filtered groups...")
    print(f"✓ Each fold trains on {len(logo_validation_groups)-1 + len(always_train_groups)} groups:")
    print(f"   - {len(logo_validation_groups)-1} other validation groups")
    print(f"   - {len(always_train_groups)} always-train groups")
    print(f"✓ Validates on 1 group (from validation groups only)\n")

    # Create a detailed log file for LOGO splits
    logo_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo_splits_log.txt')
    logo_log = open(logo_log_path, 'w', encoding='utf-8')
    logo_log.write("="*100 + "\n")
    logo_log.write("LOGO CROSS-VALIDATION SPLITS LOG (FILTERED GROUPS)\n")
    logo_log.write("="*100 + "\n\n")
    logo_log.write(f"Total validation groups: {len(logo_validation_groups)}\n")
    logo_log.write(f"Total always-train groups: {len(always_train_groups)}\n")
    logo_log.write(f"Total samples in dataset: {len(df)}\n\n")
    logo_log.write("VALIDATION GROUPS (both CRACK and REGULAR):\n")
    for idx, group in enumerate(logo_validation_groups, 1):
        samples = len(df[df['hs_dir'] == group])
        logo_log.write(f"  {idx:2d}. {group} ({samples} samples)\n")
    logo_log.write(f"\nALWAYS-TRAIN GROUPS (single class only):\n")
    for idx, group in enumerate(always_train_groups, 1):
        samples = len(df[df['hs_dir'] == group])
        logo_log.write(f"  {idx:2d}. {group} ({samples} samples)\n")
    logo_log.write("\n")

    print(f"✓ Detailed LOGO splits log will be saved to: {logo_log_path}\n")

    for fold_idx, val_group in enumerate(tqdm(logo_validation_groups, desc="LOGO Folds"), 1):
        # Training includes:
        # 1. All always_train_groups (never left out)
        # 2. All logo_validation_groups EXCEPT the current validation group
        train_groups = always_train_groups + [g for g in logo_validation_groups if g != val_group]

        val_df = df[df['hs_dir'] == val_group]
        train_df = df[df['hs_dir'].isin(train_groups)]

        # Log to file
        logo_log.write("-"*100 + "\n")
        logo_log.write(f"FOLD {fold_idx}/{len(logo_validation_groups)}\n")
        logo_log.write("-"*100 + "\n")
        logo_log.write(f"VALIDATION GROUP:\n")
        logo_log.write(f"  {val_group} ({len(val_df)} samples)\n\n")
        logo_log.write(f"TRAINING GROUPS ({len(train_groups)} groups, {len(train_df)} samples):\n")
        logo_log.write(f"\n  Always-train groups ({len(always_train_groups)} groups):\n")
        for idx, train_group in enumerate(always_train_groups, 1):
            train_group_samples = len(df[df['hs_dir'] == train_group])
            logo_log.write(f"    {idx:2d}. {train_group} ({train_group_samples} samples)\n")
        logo_log.write(f"\n  Other validation groups ({len(logo_validation_groups)-1} groups):\n")
        other_val_groups = [g for g in logo_validation_groups if g != val_group]
        for idx, train_group in enumerate(other_val_groups, 1):
            train_group_samples = len(df[df['hs_dir'] == train_group])
            logo_log.write(f"    {idx:2d}. {train_group} ({train_group_samples} samples)\n")
        logo_log.write("\n")

        # Also print to console for first 3 folds and last fold
        if fold_idx <= 3 or fold_idx == len(logo_validation_groups):
            print(f"\n{'='*80}")
            print(f"FOLD {fold_idx}/{len(logo_validation_groups)}")
            print(f"{'='*80}")
            print(f"Validation: {val_group} ({len(val_df)} samples)")
            print(f"Training: {len(train_groups)} groups ({len(train_df)} samples)")
            print(f"  - {len(always_train_groups)} always-train groups")
            print(f"  - {len(logo_validation_groups)-1} other validation groups")
            if fold_idx <= 2:  # Show some training groups for first 2 folds
                print(f"Sample training groups:")
                for idx, tg in enumerate(train_groups[:3], 1):  # Show first 3
                    print(f"  {idx}. {tg}")
                if len(train_groups) > 3:
                    print(f"  ... and {len(train_groups) - 3} more groups")

        X_train = train_df[feature_cols]
        y_train = train_df['label_encoded']
        X_val = val_df[feature_cols]
        y_val = val_df['label_encoded']

        # Calculate sample weights for training (handles class imbalance)
        weights = compute_sample_weight('balanced', y_train)

        # Manually boost CRACK class weights by 1.5x to force model focus
        weights[y_train == crack_code] *= 1.5

        # Train XGBoost
        xgb = XGBClassifier(**XGB_PARAMS)
        xgb.fit(X_train, y_train, sample_weight=weights, verbose=False)

        # Predict with custom threshold for CRACK class (Threshold Moving)
        y_prob = xgb.predict_proba(X_val)

        # Custom threshold: 0.35 for CRACK class
        CRACK_THRESHOLD = 0.35

        # Get default predictions (argmax)
        y_pred = xgb.predict(X_val)

        # Override predictions if CRACK probability > threshold
        if crack_code != -1 and crack_code in xgb.classes_:
            crack_idx = list(xgb.classes_).index(crack_code)
            crack_probs = y_prob[:, crack_idx]

            # Apply custom threshold: classify as CRACK if prob > 0.35
            y_pred = np.where(crack_probs > CRACK_THRESHOLD, crack_code, y_pred)

        # Store data for global plots
        global_y_true.extend(y_val)
        global_y_pred.extend(y_pred)

        # Store probabilities for CRACK class (for ROC curve)
        if crack_code != -1 and crack_code in xgb.classes_:
            crack_idx = list(xgb.classes_).index(crack_code)
            current_probs = y_prob[:, crack_idx]
            global_probs.extend(current_probs)
        else:
            current_probs = np.zeros(len(y_val))
            global_probs.extend(current_probs)

        # Store feature importances
        feature_importance_list.append(pd.DataFrame({
            'Feature': feature_cols,
            'Importance': xgb.feature_importances_
        }))

        # --- UPDATED METRICS CALCULATION ---
        try:
            # 1. Basic Metrics for CRACK
            f1_crack = f1_score(y_val, y_pred, labels=[crack_code], average=None, zero_division=0)[0]
            prec_crack = precision_score(y_val, y_pred, labels=[crack_code], average=None, zero_division=0)[0]
            rec_crack = recall_score(y_val, y_pred, labels=[crack_code], average=None, zero_division=0)[0]

            # 2. MCC (Matthews Correlation Coefficient)
            mcc = matthews_corrcoef(y_val, y_pred)

            # 3. Specificity (True Negative Rate)
            # Calculate manually from confusion matrix for the specific class
            # Treat as One-vs-Rest: Crack vs Non-Crack
            y_val_binary = (y_val == crack_code).astype(int)
            y_pred_binary = (y_pred == crack_code).astype(int)
            cm_binary = confusion_matrix(y_val_binary, y_pred_binary, labels=[0, 1])
            if cm_binary.shape == (2, 2):
                tn, fp, fn, tp = cm_binary.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                specificity = 0

            # 4. AUC for this specific fold
            try:
                fold_auc = roc_auc_score(y_val_binary, current_probs)
            except ValueError:
                # Happens if fold contains only one class
                fold_auc = 0.5

        except Exception as e:
            print(f"⚠️  Error calculating metrics for fold {fold_idx}: {e}")
            f1_crack, prec_crack, rec_crack, mcc, specificity, fold_auc = 0, 0, 0, 0, 0, 0

        # Overall metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        results.append({
            'Fold': fold_idx,
            'Group': val_group,
            'F1_Crack': f1_crack,
            'Precision_Crack': prec_crack,
            'Recall_Crack': rec_crack,
            'Specificity': specificity,
            'MCC': mcc,
            'AUC': fold_auc,
            'Accuracy': accuracy,
            'F1_Weighted': f1_weighted,
            'Train_Size': len(train_df),
            'Val_Size': len(val_df)
        })

    # Close the log file and add summary
    logo_log.write("="*100 + "\n")
    logo_log.write("SUMMARY\n")
    logo_log.write("="*100 + "\n")
    logo_log.write(f"Total validation folds completed: {len(results)}\n")
    logo_log.write(f"Each fold used {len(always_train_groups) + len(logo_validation_groups) - 1} groups for training:\n")
    logo_log.write(f"  - {len(always_train_groups)} always-train groups (single class)\n")
    logo_log.write(f"  - {len(logo_validation_groups) - 1} other validation groups (both classes)\n")
    logo_log.write(f"All {len(logo_validation_groups)} validation groups were used as validation exactly once\n")
    logo_log.write(f"{len(always_train_groups)} always-train groups were NEVER left out (always in training)\n")
    logo_log.close()

    print(f"\n✓ Detailed LOGO splits saved to: {logo_log_path}")

    return pd.DataFrame(results), {
        'y_true': global_y_true,
        'y_pred': global_y_pred,
        'probs': global_probs,
        'importances': pd.concat(feature_importance_list, ignore_index=True)
    }


def save_plots(global_data, output_folder, le):
    """
    Generate comprehensive publication-ready plots from LOGO validation
    """
    print("\n✓ Generating plots...")

    crack_code = le.transform(['CRACK'])[0] if 'CRACK' in le.classes_ else -1
    classes = le.classes_

    # 1. Global Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(global_data['y_true'], global_data['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap='Greens', values_format='d')
    ax.set_title('Global Confusion Matrix\n(Aggregated LOGO Cross-Validation)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix_global.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Confusion matrix saved")

    # 2. ROC Curve (CRACK vs Others)
    if crack_code != -1:
        fig, ax = plt.subplots(figsize=(8, 7))

        # Convert to binary: 1 if CRACK, 0 otherwise
        y_true_binary = (np.array(global_data['y_true']) == crack_code).astype(int)

        try:
            fpr, tpr, _ = roc_curve(y_true_binary, global_data['probs'])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color='darkorange', lw=2.5,
                   label=f'XGBoost (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                   alpha=0.5, label='Random Classifier')

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title('ROC Curve: CRACK Detection\n(Aggregated LOGO Cross-Validation)',
                        fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ ROC curve saved")
        except Exception as e:
            print(f"  ⚠️  Could not generate ROC curve: {e}")
    else:
        print("  ⚠️  Skipping ROC curve (CRACK class not found)")

    # 3. Feature Importance (Top 20) with Wavelengths
    if not global_data['importances'].empty:
        fig, ax = plt.subplots(figsize=(12, 10))

        # Get average importance
        avg_imp = global_data['importances'].groupby('Feature')['Importance'].mean()
        avg_imp = avg_imp.sort_values(ascending=False).head(20)

        # Create wavelength mapping
        wavelength_map = create_wavelength_mapping(list(avg_imp.index))

        # Create labels with wavelengths
        labels = []
        for feat in avg_imp.index:
            wl = wavelength_map.get(feat, feat)
            if isinstance(wl, (int, float)):
                labels.append(f"{wl:.1f} nm")
            else:
                labels.append(str(feat)[:20])  # Fallback to feature name

        # Create horizontal bar plot with gradient colors
        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0.3, 0.9, len(avg_imp)))

        bars = ax.barh(range(len(avg_imp)), avg_imp.values,
                      color=colors, edgecolor='black', linewidth=0.8)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, avg_imp.values)):
            ax.text(val + val*0.02, i, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

        ax.set_yticks(range(len(avg_imp)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Mean Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Wavelengths: XGBoost Importance\n(Averaged over LOGO Cross-Validation)',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'feature_importance_validation.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Feature importance plot saved (with wavelengths)")
    else:
        print("  ⚠️  No feature importance data available")

    print(f"\n✓ All plots saved to: {output_folder}")


# ============================================================================
# FINAL TRAINING
# ============================================================================

def train_final_model(df, feature_cols, le, output_folder):
    print("\n=== Training FINAL Model on ALL Data (Row 1) ===")

    X = df[feature_cols]
    y = df['label_encoded']

    weights = compute_sample_weight('balanced', y)

    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y, sample_weight=weights)

    # Save Model using joblib
    model_path = os.path.join(output_folder, 'xgboost_row1_final.joblib')
    joblib.dump(final_model, model_path)
    print(f"✓ Final model saved to: {model_path}")

    # Save Label Encoder using joblib (Critical for inference)
    le_path = os.path.join(output_folder, 'label_encoder.joblib')
    joblib.dump(le, le_path)
    print(f"✓ Label Encoder saved to: {le_path}")

    # Save feature names
    pd.DataFrame(feature_cols, columns=['feature_name']).to_csv(
        os.path.join(output_folder, 'feature_names.csv'), index=False
    )

    return final_model


# ============================================================================
# MAIN execution
# ============================================================================

if __name__ == "__main__":
    # Create results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print(f"✓ Created results folder: {RESULTS_FOLDER}")

    # 1. Load and filter Row 1 data
    df, features, le = load_and_filter_data(DATA_PATH)

    # Check if we have data and label encoder
    if len(df) == 0 or le is None:
        print("\n❌ ERROR: No data found after filtering! Check the hs_dir names in your CSV.")
        print("\nTroubleshooting:")
        print("1. Verify that hs_dir column contains paths with patterns like '\\1_01\\' to '\\1_60\\'")
        print("2. Check that the CSV file has data after outlier removal")
        print("3. Ensure the paths match the structure: ...\\data\\raw\\1_XX\\...")
        exit(1)

    print(f"\n✓ Successfully loaded {len(df)} samples from Row 1")
    print(f"✓ Number of unique grapes (hs_dir): {df['hs_dir'].nunique()}")

    # 2. Filter groups for LOGO validation
    logo_validation_groups, always_train_groups = filter_logo_groups(df, le)

    if len(logo_validation_groups) == 0:
        print("\n❌ ERROR: No groups with both CRACK and REGULAR samples found!")
        print("Cannot perform LOGO validation without groups containing both classes.")
        exit(1)

    # 3. Run LOGO validation and generate plots
    print("\n" + "="*80)
    print("STARTING LOGO CROSS-VALIDATION (FILTERED GROUPS)")
    print("="*80)

    results_df, global_data = run_logo_validation(df, features, le, logo_validation_groups, always_train_groups)

    # Save validation results
    results_path = os.path.join(RESULTS_FOLDER, 'validation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Validation results saved to: {results_path}")

    # Print validation summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY (Comprehensive Metrics)")
    print("="*80)
    print(f"✓ Validation folds completed: {len(results_df)}")
    print(f"✓ Always-train groups: {len(always_train_groups)}")

    print(f"\n{'='*80}")
    print("CRACK DETECTION PERFORMANCE (Primary Metrics)")
    print(f"{'='*80}")
    print(f"  F1-Score:    {results_df['F1_Crack'].mean():.4f} ± {results_df['F1_Crack'].std():.4f} (range: {results_df['F1_Crack'].min():.4f} - {results_df['F1_Crack'].max():.4f})")
    print(f"  Precision:   {results_df['Precision_Crack'].mean():.4f} ± {results_df['Precision_Crack'].std():.4f}")
    print(f"  Recall:      {results_df['Recall_Crack'].mean():.4f} ± {results_df['Recall_Crack'].std():.4f}")
    print(f"  Specificity: {results_df['Specificity'].mean():.4f} ± {results_df['Specificity'].std():.4f}")

    print(f"\n{'='*80}")
    print("BALANCED METRICS (Handles Class Imbalance)")
    print(f"{'='*80}")
    print(f"  MCC (Matthews):  {results_df['MCC'].mean():.4f} ± {results_df['MCC'].std():.4f} (range: -1 to +1)")
    print(f"  AUC (ROC):       {results_df['AUC'].mean():.4f} ± {results_df['AUC'].std():.4f}")

    print(f"\n{'='*80}")
    print("OVERALL CLASSIFICATION PERFORMANCE")
    print(f"{'='*80}")
    print(f"  Global Accuracy: {results_df['Accuracy'].mean():.4f} ± {results_df['Accuracy'].std():.4f}")
    print(f"  Weighted F1:     {results_df['F1_Weighted'].mean():.4f} ± {results_df['F1_Weighted'].std():.4f}")

    # Generate plots
    save_plots(global_data, RESULTS_FOLDER, le)

    # 3. Train final model on all Row 1 data
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL")
    print("="*80)

    final_model = train_final_model(df, features, le, RESULTS_FOLDER)

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"All results saved to: {RESULTS_FOLDER}")
    print(f"\nLOGO Strategy Summary:")
    print(f"  - Validation folds: {len(logo_validation_groups)} (groups with both CRACK and REGULAR)")
    print(f"  - Always-train groups: {len(always_train_groups)} (single class only)")
    print(f"  - Total training groups per fold: {len(always_train_groups) + len(logo_validation_groups) - 1}")
    print("\nGenerated files:")
    print(f"  1. validation_results.csv - Per-fold F1 scores")
    print(f"  2. confusion_matrix_global.png - Confusion matrix")
    print(f"  3. roc_curve.png - ROC curve for CRACK detection")
    print(f"  4. feature_importance_validation.png - Top 20 features")
    print(f"  5. xgboost_row1_final.json - Trained model")
    print(f"  6. feature_names.csv - Feature names for inference")
    print(f"  7. logo_splits_log.txt - Detailed LOGO splits log")
    print("="*80)

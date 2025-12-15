import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from wavelengths import WAVELENGTHS
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\dataset\cleaned_0.001\all_classes_cleaned_2025-11-01.csv"
RESULTS_FOLDER = r"C:\Users\yovel\Desktop\Grape_Project\results\xgboost_multi_2_rows"

RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 80000  # Minimum samples per class (will use oversampling if needed)
MAX_SAMPLES_PER_CLASS = 150000  # Maximum samples per class (will downsample if above)
EXCLUDE_COLUMNS = ['label', 'is_outlier', 'json_file', 'hs_dir', 'x', 'y', 'timestamp', 'mask_path']

XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'min_child_weight': 4,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'n_jobs': -1,
    'tree_method': 'hist',
    'random_state': RANDOM_STATE
}

plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# DATA LOADING & FILTERING
# ============================================================================

def load_and_filter_data(path, min_samples=MIN_SAMPLES_PER_CLASS, max_samples=MAX_SAMPLES_PER_CLASS):
    print(f"✓ Loading data from: {path}")
    df = pd.read_csv(path)

    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == 0]

    initial_len = len(df)
    # Use all rows (no hs_dir Row-1 filtering)
    print(f"✓ Using all rows after outlier removal: {initial_len} samples")

    if len(df) == 0:
        print("\n❌ ERROR: No data found after loading/cleaning!")
        return None, None, None

    # Find class sizes
    class_sizes = {}
    for label in sorted(df['label'].unique()):
        class_sizes[label] = len(df[df['label'] == label])

    min_class_size = min(class_sizes.values())
    max_class_size = max(class_sizes.values())

    print(f"\n✓ Class sizes before sampling:")
    for label, size in sorted(class_sizes.items()):
        print(f"    {label}: {size}")
    print(f"\n✓ Sampling strategy:")
    print(f"    Min class size found: {min_class_size}")
    print(f"    Max class size found: {max_class_size}")
    print(f"    Min threshold: {min_samples}")
    print(f"    Max threshold: {max_samples}")
    print(f"    Strategy: Clamp each class to [{min_samples}, {max_samples}] range")

    # Sample each class independently based on its size
    sampled_dfs = []
    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        n = len(class_df)
        if n == 0:
            print(f"  ⚠️  Class '{label}' has 0 samples — skipping.")
            continue

        if n < min_samples:
            # Oversample to min_samples
            needed = min_samples - n
            extra = class_df.sample(n=needed, replace=True, random_state=RANDOM_STATE)
            sampled = pd.concat([class_df, extra], ignore_index=True)
            action = 'oversampled'
            target = min_samples
        elif n > max_samples:
            # Downsample to max_samples
            sampled = class_df.sample(n=max_samples, random_state=RANDOM_STATE)
            action = 'downsampled'
            target = max_samples
        else:
            # Keep all samples (within range)
            sampled = class_df
            action = 'kept all'
            target = n

        sampled_dfs.append(sampled)
        print(f"  {action}: {label}: {n} -> {target}")

    df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\n✓ Prepared dataset: {len(df)} total samples")

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS and c != 'label_encoded']

    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Classes: {list(le.classes_)}")
    print(f"\n✓ Final class distribution:")
    for label in le.classes_:
        count = (df['label'] == label).sum()
        print(f"    {label}: {count} ({count / len(df) * 100:.1f}%)")

    return df, feature_cols, le


def create_wavelength_mapping(feature_names):
    wavelength_map = {}
    for fname in feature_names:
        try:
            if 'band_' in fname.lower():
                band_num = int(fname.lower().split('band_')[-1])
                wavelength_map[fname] = WAVELENGTHS.get(band_num, None)
            elif fname.replace('.', '').replace('nm', '').replace('_', '').isdigit():
                wavelength_map[fname] = float(fname.replace('nm', '').replace('_', ''))
            else:
                wavelength_map[fname] = None
        except:
            wavelength_map[fname] = None

    for i, (fname, wl) in enumerate(wavelength_map.items()):
        if wl is None:
            wavelength_map[fname] = f"Feature_{i + 1}"

    return wavelength_map


def save_feature_importance_plot(model, feature_cols, output_folder):
    print("\n✓ Generating feature importance plot...")

    fig, ax = plt.subplots(figsize=(12, 10))

    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(20)

    wavelength_map = create_wavelength_mapping(list(importances['Feature']))

    labels = []
    for feat in importances['Feature']:
        wl = wavelength_map.get(feat, feat)
        labels.append(f"{wl:.1f} nm" if isinstance(wl, (int, float)) else str(feat)[:20])

    cmap = plt.cm.get_cmap('viridis')
    colors = cmap(np.linspace(0.3, 0.9, len(importances)))

    bars = ax.barh(range(len(importances)), importances['Importance'].values,
                   color=colors, edgecolor='black', linewidth=0.8)

    for i, (bar, val) in enumerate(zip(bars, importances['Importance'].values)):
        ax.text(val + val * 0.02, i, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Wavelengths: XGBoost Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Feature importance plot saved")


def save_confusion_matrix(model, X, y, le, output_folder):
    print("\n✓ Generating confusion matrix...")

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)

    ax.set_title(f'Confusion Matrix - Training Data\nOverall Accuracy: {accuracy:.2%}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print class-wise accuracy
    print(f"\n✓ Overall Training Accuracy: {accuracy:.2%}")
    print("\n✓ Per-Class Accuracy:")
    for i, label in enumerate(le.classes_):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"    {label}: {class_acc:.2%} ({cm[i, i]}/{cm[i].sum()})")

    print(f"  ✓ Confusion matrix saved")


def train_final_model(df, feature_cols, le, output_folder):
    print("\n=== Training XGBoost Model on All Data ===")

    X = df[feature_cols]
    y = df['label_encoded']

    weights = compute_sample_weight('balanced', y)

    crack_code = le.transform(['CRACK'])[0] if 'CRACK' in le.classes_ else -1
    if crack_code != -1:
        weights[y == crack_code] *= 1.5

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, sample_weight=weights, verbose=True)

    model_path = os.path.join(output_folder, 'xgboost_row1_final.joblib')
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")

    le_path = os.path.join(output_folder, 'label_encoder.joblib')
    joblib.dump(le, le_path)
    print(f"✓ Label Encoder saved to: {le_path}")

    pd.DataFrame(feature_cols, columns=['feature_name']).to_csv(
        os.path.join(output_folder, 'feature_names.csv'), index=False
    )

    save_feature_importance_plot(model, feature_cols, output_folder)
    save_confusion_matrix(model, X, y, le, output_folder)

    return model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print(f"✓ Created results folder: {RESULTS_FOLDER}")

    df, features, le = load_and_filter_data(DATA_PATH)

    if df is None or len(df) == 0:
        print("\n❌ ERROR: No data available!")
        exit(1)

    print(f"\n✓ Successfully loaded {len(df)} samples")
    print(f"✓ Number of unique grapes (hs_dir): {df['hs_dir'].nunique()}")

    model = train_final_model(df, features, le, RESULTS_FOLDER)

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"All results saved to: {RESULTS_FOLDER}")
    print("\nGenerated files:")
    print(f"  1. xgboost_row1_final.joblib - Trained model")
    print(f"  2. label_encoder.joblib - Label encoder")
    print(f"  3. feature_names.csv - Feature names")
    print(f"  4. feature_importance.png - Top 20 features")
    print(f"  5. confusion_matrix.png - Training confusion matrix")
    print("=" * 80)

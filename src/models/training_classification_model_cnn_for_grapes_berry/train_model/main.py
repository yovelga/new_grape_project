# main.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os
from typing import Dict, List
from datetime import datetime
from train import train_model
from model import get_model_gray
from dataset_multi import GrapeDataset
from data_transforms import get_train_transforms, get_test_transforms
from config import TRAIN_DIR, TEST_DIR, BATCH_SIZE, NUM_EPOCHS

from torch.utils.data import DataLoader


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history["accuracy"], label="Accuracy")
    ax2.plot(history["f1"], label="F1 Score")
    ax2.plot(history["auc"], label="AUC")
    ax2.set_title("Metrics Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_history.png"))
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "roc_curve.png"))
    plt.close()


def plot_input_mode_comparison(results_summary: List[Dict], save_path: str):
    modes = [r["mode"] for r in results_summary]
    metrics = {
        "F1 Score": [r["best_metrics"]["f1"] for r in results_summary],
        "Accuracy": [r["best_metrics"]["accuracy"] for r in results_summary],
        "AUC": [r["best_metrics"]["auc"] for r in results_summary],
    }
    plt.figure(figsize=(10, 6))
    x = np.arange(len(modes))
    width = 0.25
    for i, (metric, values) in enumerate(metrics.items()):
        plt.bar(x + i * width, values, width, label=metric)
    plt.xlabel("Input Mode")
    plt.ylabel("Score")
    plt.title("Performance Comparison Between Input Modes")
    plt.xticks(x + width, modes)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # GPU setup and debugging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"DEVICE INFORMATION:")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        # Set GPU to be more efficient
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: CUDA is not available. Training will be slow on CPU.")
    print(f"{'='*60}\n")

    results_summary = []

    # Create dated folder for this training session
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"new_models_{current_date}"
    os.makedirs(base_save_dir, exist_ok=True)
    print(f"Models and results will be saved to: {base_save_dir}")

    # for mode in ["all", "original", "enlarged", "segmentation"]:
    # for mode in ["enlarged"]:  # OLD: Train with enlarged mode
    for mode in ["context_square_segmentation"]:  # NEW: Context-aware square crops with mask (BEST)
        # Use get_model instead of build_model
        model = get_model_gray(num_classes=2)
        model.to(device)

        train_dataset = GrapeDataset(
            root_dir=TRAIN_DIR,
            input_mode=mode,
            transform=get_train_transforms(),
            balance_mode="oversample"  # Oversample minority class for better balance
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=9,  # Reduced for Windows stability
            pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer
            persistent_workers=True  # Faster data loading
        )

        val_dataset = GrapeDataset(
            root_dir=TEST_DIR, input_mode=mode, transform=get_test_transforms()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,  # Reduced for Windows stability
            pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer
            persistent_workers=True  # Faster data loading
        )

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        num_epochs = NUM_EPOCHS

        model, results = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs
        )

        # Save model and results in the dated folder
        save_path = os.path.join(base_save_dir, f"{mode}_gray")
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, "best_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

        plot_training_history(results["history"], save_path)
        plot_confusion_matrix(
            results["predictions"]["labels"],
            results["predictions"]["predictions"],
            save_path,
        )
        plot_roc_curve(
            results["predictions"]["labels"],
            results["predictions"]["probabilities"][:, 1],
            save_path,
        )
        results["mode"] = mode
        results_summary.append(results)

    # Save comparison plot in the base dated folder
    comparison_path = os.path.join(base_save_dir, "input_mode_comparison.png")
    plot_input_mode_comparison(results_summary, comparison_path)


if __name__ == "__main__":
    main()

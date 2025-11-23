# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def calculate_metrics(labels, preds, probs) -> Dict[str, float]:
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = float("nan")
    return {"accuracy": accuracy, "f1": f1, "auc": auc}


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return running_loss / len(train_loader)


def evaluate_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    val_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)
    metrics = calculate_metrics(all_labels_np, all_preds_np, all_probs_np)
    metrics["val_loss"] = val_loss / len(test_loader)
    metrics["labels"] = all_labels_np
    metrics["predictions"] = all_preds_np
    metrics["probabilities"] = all_probs_np
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    scheduler: Any = None,
    patience: int = 5,  # Early stopping patience
) -> Tuple[nn.Module, Dict[str, Any]]:
    history = {"train_loss": [], "val_loss": [], "accuracy": [], "f1": [], "auc": []}
    best_f1, best_metrics, best_model = 0.0, None, None
    best_preds, best_labels, best_probs = None, None, None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_epoch(model, test_loader, criterion, device)
        if scheduler:
            scheduler.step(metrics["val_loss"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["val_loss"])
        history["accuracy"].append(metrics["accuracy"])
        history["f1"].append(metrics["f1"])
        history["auc"].append(metrics["auc"])
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = {
                k: metrics[k] for k in ["accuracy", "f1", "auc", "val_loss"]
            }
            best_model = model.state_dict().copy()
            best_preds = metrics["predictions"]
            best_labels = metrics["labels"]
            best_probs = metrics["probabilities"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {metrics['val_loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}"
        )
        # Only allow early stopping after 10 epochs
        if epoch + 1 >= 10 and epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs with no improvement for {patience} epochs."
            )
            break

    model.load_state_dict(best_model)
    return model, {
        "history": history,
        "best_metrics": best_metrics,
        "predictions": {
            "labels": best_labels,
            "predictions": best_preds,
            "probabilities": best_probs,
        },
    }

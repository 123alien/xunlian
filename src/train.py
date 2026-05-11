"""Training loop, pretraining, fine-tuning, and evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.models import LSTMForecaster


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_windows(df, feature_cols: list[str], target_col: str = "energy",
                  window_size: int = 24, horizon: int = 1):
    """Convert a chronological DataFrame into (X, y) window arrays.

    Args:
        df: sorted DataFrame with feature columns
        feature_cols: columns to use as model input
        target_col: column to predict
        window_size: number of past timesteps
        horizon: how far ahead to predict (1 = next step)

    Returns:
        X: (n_windows, window_size, n_features)
        y: (n_windows,)
    """
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)

    X, y = [], []
    for i in range(len(df) - window_size - horizon + 1):
        X.append(features[i:i + window_size])
        y.append(targets[i + window_size + horizon - 1])

    return np.array(X), np.array(y)


def fit_scaler(X_train: np.ndarray):
    """Fit a Z-score scaler (mean, std) on training data only."""
    # X_train: (n, window_size, n_features)
    n_samples, window_size, n_features = X_train.shape
    X_flat = X_train.reshape(-1, n_features)
    mean = X_flat.mean(axis=0, keepdims=True)
    std = X_flat.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch).squeeze(-1)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, config: Config, device,
                verbose: bool = True) -> dict:
    """Standard training loop with early stopping.

    Returns training history dict.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_weights = deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.train.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.train.early_stopping_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.6f}")
            break

    model.load_state_dict(best_weights)
    history["best_val_loss"] = best_val_loss
    history["stopped_epoch"] = epoch + 1
    return history


# ---------------------------------------------------------------------------
# Pretrain + Fine-tune
# ---------------------------------------------------------------------------

def pretrain(model, source_loaders: list[tuple[DataLoader, DataLoader]],
             config: Config, device, verbose: bool = True) -> dict:
    """Pretrain model on multiple source buildings.

    Args:
        model: fresh LSTMForecaster instance
        source_loaders: list of (train_loader, val_loader) for each source building
        config: Config
        device: torch device
        verbose: print progress

    Returns:
        training history dict
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_weights = deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.train.pretrain_epochs):
        # Train on all source buildings (concatenate batches across buildings)
        model.train()
        epoch_train_loss = 0.0
        n_train_samples = 0
        for train_ldr, _ in source_loaders:
            for X_batch, y_batch in train_ldr:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch).squeeze(-1)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * X_batch.size(0)
                n_train_samples += X_batch.size(0)
        avg_train_loss = epoch_train_loss / n_train_samples

        # Validate on all source validation sets
        model.eval()
        epoch_val_loss = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for _, val_ldr in source_loaders:
                for X_batch, y_batch in val_ldr:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = model(X_batch).squeeze(-1)
                    loss = criterion(pred, y_batch)
                    epoch_val_loss += loss.item() * X_batch.size(0)
                    n_val_samples += X_batch.size(0)
        avg_val_loss = epoch_val_loss / n_val_samples

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.train.early_stopping_patience:
            if verbose:
                print(f"  Pretrain early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.6f}")
            break

    model.load_state_dict(best_weights)
    history["best_val_loss"] = best_val_loss
    history["stopped_epoch"] = epoch + 1
    return history


def fine_tune(model, train_loader, val_loader, config: Config, device,
              verbose: bool = True) -> dict:
    """Fine-tune a pretrained model on target-building data."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.finetune_lr,
        weight_decay=config.train.weight_decay,
    )
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_weights = deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.train.finetune_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.train.early_stopping_patience:
            if verbose:
                print(f"  Fine-tune early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.6f}")
            break

    model.load_state_dict(best_weights)
    history["best_val_loss"] = best_val_loss
    history["stopped_epoch"] = epoch + 1
    return history


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions. Returns (y_true, y_pred)."""
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).squeeze(-1).cpu().numpy()
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(pred)
    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int,
                    shuffle: bool = False) -> DataLoader:
    """Create DataLoader from numpy arrays."""
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def prepare_building_data(split: dict, feature_cols: list[str],
                          config: Config):
    """Prepare train/val/test loaders for one building.

    Returns:
        train_loader, val_loader, test_loader, scaler_params, X_test, y_test
    """
    # Build windows
    X_train, y_train = build_windows(split["train"], feature_cols, "energy",
                                     config.data.window_size, config.data.horizon)
    X_val, y_val = build_windows(split["val"], feature_cols, "energy",
                                 config.data.window_size, config.data.horizon)
    X_test, y_test = build_windows(split["test"], feature_cols, "energy",
                                   config.data.window_size, config.data.horizon)

    if len(X_train) == 0 or len(X_test) == 0:
        return None  # insufficient data

    # Scale
    mean, std = fit_scaler(X_train)
    X_train_s = apply_scaler(X_train, mean, std)
    X_val_s = apply_scaler(X_val, mean, std)
    X_test_s = apply_scaler(X_test, mean, std)

    train_ldr = make_dataloader(X_train_s, y_train, config.train.batch_size, shuffle=True)
    val_ldr = make_dataloader(X_val_s, y_val, config.train.batch_size, shuffle=False)
    test_ldr = make_dataloader(X_test_s, y_test, config.train.batch_size, shuffle=False)

    return {
        "train_loader": train_ldr,
        "val_loader": val_ldr,
        "test_loader": test_ldr,
        "scaler": (mean, std),
        "X_train": X_train_s,
        "y_train": y_train,
        "X_test": X_test_s,
        "y_test": y_test,
    }


def save_run(output_dir: str, config: Config, metrics: dict,
             predictions: np.ndarray | None = None,
             anomaly_scores: np.ndarray | None = None,
             history: dict | None = None):
    """Save experiment outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Config as YAML
    with open(out / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    # Metrics as JSON
    with open(out / "metrics.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in metrics.items()}, f, indent=2)

    # Predictions
    if predictions is not None:
        pred_df = predictions if hasattr(predictions, "to_csv") else None
        if anomaly_scores is not None:
            np.savez(out / "predictions.npz",
                     y_true=predictions[0] if isinstance(predictions, tuple) else None,
                     y_pred=predictions[1] if isinstance(predictions, tuple) else None,
                     anomaly_scores=anomaly_scores)
        else:
            np.savez(out / "predictions.npz",
                     y_true=predictions[0] if isinstance(predictions, tuple) else None,
                     y_pred=predictions[1] if isinstance(predictions, tuple) else None)

    # History
    if history is not None:
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)

"""Modèle LSTM multi-stations pour la prédiction de crues.

Architecture : LSTM 2 couches (128→64) + couche dense → 5 sorties multi-horizon.
Entraîné avec PyTorch + CUDA si disponible.

Usage:
    python train_lstm.py
    python train_lstm.py --epochs 200 --hidden 256 --lr 0.0005
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EPOCHS,
    FORECAST_HORIZONS,
    LEARNING_RATE,
    PATIENCE,
    PROCESSED_DIR,
)


class FloodLSTM(nn.Module):
    """LSTM multi-couches pour la prédiction de hauteur d'eau."""

    def __init__(self, n_features: int, n_horizons: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # dernier pas de temps
        out = self.dropout(out)
        return self.fc(out)


def create_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Charge les données et crée les DataLoaders."""
    X_train = torch.from_numpy(np.load(PROCESSED_DIR / "X_train.npy"))
    y_train = torch.from_numpy(np.load(PROCESSED_DIR / "y_train.npy"))
    X_val = torch.from_numpy(np.load(PROCESSED_DIR / "X_val.npy"))
    y_val = torch.from_numpy(np.load(PROCESSED_DIR / "y_val.npy"))
    X_test = torch.from_numpy(np.load(PROCESSED_DIR / "X_test.npy"))
    y_test = torch.from_numpy(np.load(PROCESSED_DIR / "y_test.npy"))

    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, meta


def nash_sutcliffe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Évalue le modèle sur un DataLoader."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    metrics = {}
    for i, h in enumerate(FORECAST_HORIZONS):
        rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
        mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        nse = nash_sutcliffe(y_true[:, i], y_pred[:, i])
        metrics[f"t+{h}h"] = {"rmse": float(rmse), "mae": float(mae), "nse": float(nse)}

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Entraînement LSTM")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Données
    print("Chargement des données...")
    train_loader, val_loader, test_loader, meta = create_dataloaders(args.batch_size)
    n_features = meta["n_features"]
    n_horizons = len(meta["forecast_horizons"])
    print(f"Features: {n_features}, Horizons: {n_horizons}")

    # Modèle
    model = FloodLSTM(n_features, n_horizons, args.hidden1, args.hidden2, args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    # Entraînement
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nEntraînement — {args.epochs} epochs max, patience={args.patience}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'LR':>10} {'Time':>8}")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                loss = criterion(model(X_batch), y_batch)
                val_loss += loss.item()
                n_val += 1
        val_loss /= n_val

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "lstm_best.pt")
            marker = " ★"
        else:
            patience_counter += 1

        print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>12.6f} {lr:>10.2e} {elapsed:>7.1f}s{marker}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping à l'epoch {epoch}")
            break

    # Évaluation finale
    print(f"\n{'='*50}")
    print("Évaluation sur le meilleur modèle")
    print(f"{'='*50}")

    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "lstm_best.pt", weights_only=True))

    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    print("\nValidation:")
    for k, v in val_metrics.items():
        print(f"  {k}: NSE={v['nse']:.4f}, RMSE={v['rmse']:.6f}, MAE={v['mae']:.6f}")

    print("\nTest:")
    for k, v in test_metrics.items():
        print(f"  {k}: NSE={v['nse']:.4f}, RMSE={v['rmse']:.6f}, MAE={v['mae']:.6f}")

    # Sauvegarder les résultats
    results = {
        "model": "LSTM",
        "params": {
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "total_params": total_params,
        },
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(CHECKPOINTS_DIR / "lstm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Sauvegarder aussi les hyperparamètres du modèle pour l'export ONNX
    model_config = {
        "n_features": n_features,
        "n_horizons": n_horizons,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "dropout": args.dropout,
    }
    with open(CHECKPOINTS_DIR / "lstm_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n✓ Modèle et résultats dans {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()

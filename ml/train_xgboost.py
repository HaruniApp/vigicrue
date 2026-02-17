"""Baseline XGBoost pour la prédiction de crues.

XGBoost travaille sur des features tabulaires : on aplatit la fenêtre temporelle
en un vecteur de lags. Plus simple mais souvent compétitif sur les séries temporelles.

Usage:
    python train_xgboost.py
    python train_xgboost.py --max-lags 48
"""

import argparse
import json
import pickle
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import CHECKPOINTS_DIR, FORECAST_HORIZONS, PROCESSED_DIR


def nash_sutcliffe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (NSE). 1 = parfait, <0 = pire que la moyenne."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


def flatten_windows(X: np.ndarray, max_lags: int | None = None) -> np.ndarray:
    """Aplatit les fenêtres 3D (N, T, F) en 2D (N, T*F) pour XGBoost.

    Si max_lags est spécifié, ne garde que les max_lags derniers pas de temps.
    """
    if max_lags is not None and max_lags < X.shape[1]:
        X = X[:, -max_lags:, :]
    return X.reshape(X.shape[0], -1)


def main():
    parser = argparse.ArgumentParser(description="Entraînement XGBoost baseline")
    parser.add_argument("--max-lags", type=int, default=48, help="Nombre max de lags à utiliser")
    parser.add_argument("--n-estimators", type=int, default=500, help="Nombre d'arbres")
    parser.add_argument("--max-depth", type=int, default=8, help="Profondeur max des arbres")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Charger les données
    print("Chargement des données...")
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Max lags: {args.max_lags}")

    # Aplatir pour XGBoost
    X_train_flat = flatten_windows(X_train, args.max_lags)
    X_val_flat = flatten_windows(X_val, args.max_lags)
    X_test_flat = flatten_windows(X_test, args.max_lags)

    print(f"Features XGBoost: {X_train_flat.shape[1]}")

    horizons = meta["forecast_horizons"]
    results = {}

    # Entraîner un modèle par horizon
    for i, horizon in enumerate(horizons):
        print(f"\n{'='*50}")
        print(f"Horizon t+{horizon}h")
        print(f"{'='*50}")

        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            early_stopping_rounds=20,
            random_state=42,
        )

        t0 = time.time()
        model.fit(
            X_train_flat,
            y_train[:, i],
            eval_set=[(X_val_flat, y_val[:, i])],
            verbose=False,
        )
        train_time = time.time() - t0

        # Prédictions
        y_pred_val = model.predict(X_val_flat)
        y_pred_test = model.predict(X_test_flat)

        # Métriques
        val_rmse = np.sqrt(mean_squared_error(y_val[:, i], y_pred_val))
        val_mae = mean_absolute_error(y_val[:, i], y_pred_val)
        val_nse = nash_sutcliffe(y_val[:, i], y_pred_val)

        test_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred_test))
        test_mae = mean_absolute_error(y_test[:, i], y_pred_test)
        test_nse = nash_sutcliffe(y_test[:, i], y_pred_test)

        print(f"  Temps d'entraînement: {train_time:.1f}s")
        print(f"  Val  — RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, NSE: {val_nse:.4f}")
        print(f"  Test — RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, NSE: {test_nse:.4f}")

        results[f"t+{horizon}h"] = {
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "val_nse": float(val_nse),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_nse": float(test_nse),
            "train_time_s": round(train_time, 1),
            "best_iteration": model.best_iteration,
        }

        # Sauvegarder le modèle
        model_path = CHECKPOINTS_DIR / f"xgboost_h{horizon}.json"
        model.save_model(str(model_path))

    # Sauvegarder les résultats
    results_path = CHECKPOINTS_DIR / "xgboost_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("Résumé XGBoost")
    print(f"{'='*50}")
    for horizon_key, metrics in results.items():
        print(f"  {horizon_key}: NSE={metrics['test_nse']:.4f}, RMSE={metrics['test_rmse']:.6f}")

    print(f"\n✓ Modèles et résultats dans {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()

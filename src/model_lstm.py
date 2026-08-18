"""
src/model_lstm.py
=================
Phase 5 — LSTM (Model 3) for the Economic Policy Impact Simulator.

What it does
------------
1. Loads ``data/processed/india_macro_quarterly.csv`` and selects the three
   indicators available over the full sample (2010-Q1 → 2025-Q4):
       CPI_Inflation, GDP_Growth, Unemployment_Rate
   Repo_Rate is available only through 2021-Q1 (NaN thereafter), so the model
   is run in two configurations — see ``COLUMNS_FULL`` and ``COLUMNS_WITH_REPO``.

2. MinMaxScaler normalisation across all selected columns (levels, not diffs —
   LSTM learns the trend directly).

3. Sliding-window sequence creation:
       X[t] = scaled[t : t+LOOK_BACK]   shape (LOOK_BACK, n_features)
       y[t] = scaled[t + LOOK_BACK]     shape (n_features,)

4. Chronological 80/20 train-test split.

5. Two-layer stacked LSTM + Dense(n_features) head.

6. Training with EarlyStopping + ReduceLROnPlateau.

7. Evaluation: RMSE and MAPE per indicator.

8. Saves:
       models/lstm_model.keras   — trained model (Keras v3 native format)
       models/lstm_results.json  — metrics dict

Run from the project root:
    python src/model_lstm.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Project paths ─────────────────────────────────────────────────────────────
_SRC_DIR    = Path(__file__).parent
_ROOT       = _SRC_DIR.parent
_DATA_PATH  = _ROOT / "data" / "processed" / "india_macro_quarterly.csv"
_MODELS_DIR = _ROOT / "models"
_MODELS_DIR.mkdir(exist_ok=True)

# ── Column sets ───────────────────────────────────────────────────────────────
# Full-sample: 3 indicators, 2010-Q1 to 2025-Q4 (64 quarters)
COLUMNS_FULL = ["CPI_Inflation", "GDP_Growth", "Unemployment_Rate"]

# Partial-sample: 4 indicators, capped at 2021-Q1 when Repo_Rate ends (~45 rows)
COLUMNS_WITH_REPO = ["Repo_Rate", "CPI_Inflation", "GDP_Growth", "Unemployment_Rate"]

# Default hyper-parameters
LOOK_BACK   = 4     # quarters of history used as input window
TEST_RATIO  = 0.20  # fraction of sequences held out for testing
EPOCHS      = 200
BATCH_SIZE  = 8
PATIENCE    = 25


# =============================================================================
# Step 0 — Parse quarter index
# =============================================================================

def _parse_quarter_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'YYYY-Qn' string index to a quarterly DatetimeIndex."""
    def _q_to_date(label: str) -> pd.Timestamp:
        year, q = label.split("-Q")
        month = (int(q) - 1) * 3 + 1
        return pd.Timestamp(int(year), month, 1)

    out = df.copy()
    out.index = pd.DatetimeIndex([_q_to_date(q) for q in out.index])
    out.index.freq = "QS"
    return out


# =============================================================================
# Step 1 — Load data
# =============================================================================

def load_data(
    data_path: Path = _DATA_PATH,
    include_repo_rate: bool = False,
) -> pd.DataFrame:
    """
    Load the quarterly macro CSV and return a clean DataFrame of level values
    (not first-differenced — the LSTM learns trend patterns directly).

    Parameters
    ----------
    data_path         : path to india_macro_quarterly.csv
    include_repo_rate : if True, include Repo_Rate (sample truncated to 2021-Q1)

    Returns
    -------
    pd.DataFrame — selected columns, NaN rows dropped, DatetimeIndex.
    """
    raw = pd.read_csv(data_path, index_col=0)
    raw = _parse_quarter_index(raw)

    cols = COLUMNS_WITH_REPO if include_repo_rate else COLUMNS_FULL
    df = raw[cols].dropna()

    print(f"[load_data] Columns : {df.columns.tolist()}")
    print(f"[load_data] Sample  : {df.index[0].date()} -> {df.index[-1].date()} "
          f"({len(df)} quarters)")
    return df


# =============================================================================
# Step 2 — Scale
# =============================================================================

def scale_data(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler(0, 1) on all columns and return scaled array + scaler.

    Parameters
    ----------
    df : raw DataFrame (T, n_features)

    Returns
    -------
    scaled : np.ndarray (T, n_features)
    scaler : fitted MinMaxScaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values.astype(np.float32))
    print(f"[scale_data] Scaled array shape: {scaled.shape}")
    return scaled.astype(np.float32), scaler


# =============================================================================
# Step 3 — Sliding-window sequences
# =============================================================================

def make_sequences(
    scaled: np.ndarray,
    look_back: int = LOOK_BACK,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised (X, y) pairs for multi-output regression.

    For every time step t:
        X[t] = scaled[t : t+look_back]   shape (look_back, n_features)
        y[t] = scaled[t + look_back]     shape (n_features,)

    Parameters
    ----------
    scaled    : np.ndarray (T, n_features)
    look_back : window length in quarters

    Returns
    -------
    X : np.ndarray (N, look_back, n_features)
    y : np.ndarray (N, n_features)
    """
    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i : i + look_back])
        y.append(scaled[i + look_back])
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    print(f"[make_sequences] X: {X_arr.shape}  y: {y_arr.shape}")
    return X_arr, y_arr


# =============================================================================
# Step 4 — Chronological split
# =============================================================================

def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = TEST_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological 80/20 split — no shuffling.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    split = int(len(X) * (1 - test_ratio))
    print(f"[split_sequences] Train: {split} | Test: {len(X) - split}")
    return X[:split], X[split:], y[:split], y[split:]


# =============================================================================
# Step 5 — Model architecture
# =============================================================================

def build_model(look_back: int, n_features: int) -> keras.Model:
    """
    Stacked two-layer LSTM with a Dense multi-output head.

    Architecture
    ────────────
    Input(look_back, n_features)
      → LSTM(64, return_sequences=True)  — captures temporal patterns
      → Dropout(0.2)
      → LSTM(32)                          — compresses to latent vector
      → Dropout(0.2)
      → Dense(n_features, activation='linear')  — predict all indicators

    Loss     : MSE
    Optimizer: Adam (lr = 1e-3)
    Metrics  : MAE

    Parameters
    ----------
    look_back  : number of input time steps
    n_features : number of macro variables (= output width)

    Returns
    -------
    Compiled keras.Sequential model.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(look_back, n_features), name="input_window"),
            layers.LSTM(64, return_sequences=True, name="lstm_1"),
            layers.Dropout(0.2, name="dropout_1"),
            layers.LSTM(32, return_sequences=False, name="lstm_2"),
            layers.Dropout(0.2, name="dropout_2"),
            layers.Dense(n_features, activation="linear", name="output"),
        ],
        name="macro_lstm",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# =============================================================================
# Step 6 — Training function
# =============================================================================

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
) -> keras.callbacks.History:
    """
    Fit the LSTM with early stopping and learning-rate reduction.

    Callbacks
    ---------
    EarlyStopping      : stops when val_loss stops improving (patience=25)
    ReduceLROnPlateau  : halves LR after 10 epochs of stagnant val_loss

    Parameters
    ----------
    model      : compiled keras model (from build_model)
    X_train / y_train : training sequences
    X_test  / y_test  : validation / test sequences
    epochs     : maximum number of epochs
    batch_size : mini-batch size
    patience   : EarlyStopping patience (epochs without improvement)

    Returns
    -------
    keras History object
    """
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        shuffle=False,    # preserve temporal order within epochs
        verbose=1,
    )
    print(f"\n[train_model] Stopped at epoch {len(history.history['loss'])}")
    return history


# =============================================================================
# Step 7 — Evaluation metrics
# =============================================================================

def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, pred)))


def _mape(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((actual - pred) / (np.abs(actual) + eps))) * 100)


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    col_names: List[str],
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Generate predictions on the test set, inverse-transform to original scale,
    and compute RMSE + MAPE per indicator.

    Parameters
    ----------
    model     : trained keras model
    X_test    : scaled test sequences (N, look_back, n_features)
    y_test    : scaled test targets   (N, n_features)
    scaler    : fitted MinMaxScaler (to reverse scaling)
    col_names : list of n_features indicator names

    Returns
    -------
    metrics_dict : {indicator: {RMSE: float, MAPE: float}}
    y_pred_real  : np.ndarray (N, n_features) — original-scale predictions
    y_true_real  : np.ndarray (N, n_features) — original-scale actuals
    """
    y_pred_scaled = model.predict(X_test, verbose=0).astype(np.float32)
    y_pred_real = scaler.inverse_transform(y_pred_scaled)
    y_true_real = scaler.inverse_transform(y_test.astype(np.float32))

    metrics = {}
    print("\n" + "=" * 55)
    print("  LSTM Test-Set Metrics (original scale)")
    print("=" * 55)
    print(f"{'Indicator':<22}  {'RMSE':>10}  {'MAPE (%)':>10}")
    print("-" * 55)

    for i, name in enumerate(col_names):
        rmse_val = _rmse(y_true_real[:, i], y_pred_real[:, i])
        mape_val = _mape(y_true_real[:, i], y_pred_real[:, i])
        metrics[name] = {"RMSE": round(rmse_val, 6), "MAPE": round(mape_val, 4)}
        print(f"  {name:<20}  {rmse_val:>10.4f}  {mape_val:>9.2f}%")

    mean_rmse = np.mean([v["RMSE"] for v in metrics.values()])
    mean_mape = np.mean([v["MAPE"] for v in metrics.values()])
    print("-" * 55)
    print(f"  {'MEAN':<20}  {mean_rmse:>10.4f}  {mean_mape:>9.2f}%")
    print("=" * 55)

    return metrics, y_pred_real, y_true_real


# =============================================================================
# Step 8 — Save artefacts
# =============================================================================

def save_model(
    model: keras.Model,
    metrics: dict,
    col_names: List[str],
    look_back: int,
    n_train: int,
    n_test: int,
    model_path: Path = _MODELS_DIR / "lstm_model.keras",
    results_path: Path = _MODELS_DIR / "lstm_results.json",
) -> None:
    """
    Persist the trained Keras model and a JSON metrics file.

    Model format
    ------------
    ``.keras``  — Keras v3 native format (recommended over legacy .h5 for TF >= 2.12)
    Legacy     : pass  ``model_path.with_suffix('.h5')`` for HDF5 format.

    Parameters
    ----------
    model        : trained keras.Model
    metrics      : dict returned by evaluate_model
    col_names    : list of indicator names
    look_back    : window length used during training
    n_train / n_test : number of training / test sequences
    model_path   : where to save the .keras file
    results_path : where to save lstm_results.json
    """
    # ── Model ────────────────────────────────────────────────────────────────
    model.save(str(model_path))
    print(f"\n[save_model] Model saved  -> {model_path}")

    # ── JSON metrics ─────────────────────────────────────────────────────────
    results = {
        "model"       : "LSTM",
        "architecture": {
            "layers"      : ["LSTM(64)", "Dropout(0.2)", "LSTM(32)",
                             "Dropout(0.2)", f"Dense({len(col_names)})"],
            "look_back"   : look_back,
            "n_features"  : len(col_names),
            "loss"        : "MSE",
            "optimizer"   : "Adam(lr=1e-3)",
        },
        "data": {
            "columns"     : col_names,
            "n_train"     : n_train,
            "n_test"      : n_test,
        },
        "metrics"     : metrics,
    }

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[save_model] Results saved -> {results_path}")


# =============================================================================
# Public pipeline
# =============================================================================

def run_lstm_pipeline(
    data_path: Path = _DATA_PATH,
    include_repo_rate: bool = False,
    look_back: int = LOOK_BACK,
    test_ratio: float = TEST_RATIO,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
    save: bool = True,
) -> dict:
    """
    End-to-end Phase 5 pipeline.

    Parameters
    ----------
    data_path         : path to india_macro_quarterly.csv
    include_repo_rate : if True, adds Repo_Rate (sample truncated to 2021-Q1)
    look_back         : sliding-window length (quarters)
    test_ratio        : fraction of sequences for test set
    epochs            : maximum training epochs
    batch_size        : mini-batch size
    patience          : early-stopping patience
    save              : if True, write model + results JSON to models/

    Returns
    -------
    dict with keys: model, history, scaler, df,
                    X_train, X_test, y_train, y_test,
                    y_pred_real, y_true_real, metrics
    """
    tf.random.set_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("  PHASE 5 — LSTM PIPELINE")
    print("=" * 60)

    # 1. Load
    df = load_data(data_path, include_repo_rate=include_repo_rate)
    col_names  = df.columns.tolist()
    n_features = len(col_names)

    # 2. Scale
    scaled, scaler = scale_data(df)

    # 3. Sequences
    X, y = make_sequences(scaled, look_back=look_back)

    # 4. Split
    X_train, X_test, y_train, y_test = split_sequences(X, y, test_ratio)

    # 5. Build
    model = build_model(look_back=look_back, n_features=n_features)
    model.summary()

    # 6. Train
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=epochs, batch_size=batch_size, patience=patience,
    )

    # 7. Evaluate
    metrics, y_pred_real, y_true_real = evaluate_model(
        model, X_test, y_test, scaler, col_names
    )

    # 8. Save
    if save:
        save_model(
            model, metrics, col_names,
            look_back=look_back,
            n_train=len(X_train),
            n_test=len(X_test),
        )

    return {
        "model"      : model,
        "history"    : history,
        "scaler"     : scaler,
        "df"         : df,
        "col_names"  : col_names,
        "X_train"    : X_train,
        "X_test"     : X_test,
        "y_train"    : y_train,
        "y_test"     : y_test,
        "y_pred_real": y_pred_real,
        "y_true_real": y_true_real,
        "metrics"    : metrics,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    results = run_lstm_pipeline(include_repo_rate=False, save=True)

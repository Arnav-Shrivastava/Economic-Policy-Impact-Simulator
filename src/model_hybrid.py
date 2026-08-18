"""
src/model_hybrid.py
===================
Phase 6 — Hybrid VAR + LSTM model for the Economic Policy Impact Simulator.

Architecture overview
---------------------
    VAR(p)  →  in-sample fitted values  →  residuals  (actual − fitted)
                                               ↓
                               Sliding-window LSTM on residuals
                                               ↓
    hybrid_forecast(h) = VAR point forecast  +  LSTM residual correction

Why this hybrid works
---------------------
A VAR captures linear dynamics between first-differenced macro variables.
Its residuals contain the non-linear, regime-change, and surprise components
that a pure VAR cannot model.  The residual LSTM learns those patterns so
that, at forecast time, the predicted residual is *added back* on top of the
VAR point forecast — correcting systematic biases while preserving the
structural interpretation of the VAR.

What each section does
----------------------
1.  compute_var_residuals(var_results, df)
        Extract in-sample fitted values and residuals from a fitted
        VARResults object.  Returns DataFrames in the same column order
        as the VAR.

2.  make_residual_sequences(residuals_df, window=4)
        Build sliding-window (X, y) pairs from the residual series so
        that the LSTM maps:
            X[t] = residuals[t : t+window]   → y[t] = residuals[t+window]
        Operates purely on raw (unscaled) residuals since their magnitude
        is already small and consistent across columns.

3.  build_residual_lstm(window, n_features)
        Lightweight stacked LSTM identical in spirit to model_lstm.py but
        sized for residuals (smaller hidden units, stronger regularisation).

4.  train_residual_lstm(...)
        Train the residual LSTM with EarlyStopping + ReduceLROnPlateau,
        mirroring the training protocol in model_lstm.py.

5.  hybrid_forecast(periods)
        Step-by-step h-step hybrid forecast:
        a.  Obtain VAR h-step point forecast (in first-differenced space).
        b.  Predict residual corrections via the LSTM.
        c.  Return VAR forecast + residual correction, then cumulate back
            to level-space using the last observed raw level.

6.  compare_models(...)
        Evaluate standalone VAR, standalone LSTM (residual-LSTM only), and
        the hybrid model on the same hold-out test window.  Prints a
        side-by-side RMSE / MAPE table for every variable.

Run from the project root:
    python src/model_hybrid.py

Dependencies
------------
    statsmodels  >= 0.14
    tensorflow   >= 2.12
    scikit-learn >= 1.3
    numpy, pandas, matplotlib
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# -- Project paths -------------------------------------------------------------
_SRC_DIR    = Path(__file__).parent
_ROOT       = _SRC_DIR.parent
_DATA_PATH  = _ROOT / "data" / "processed" / "india_macro_quarterly.csv"
_MODELS_DIR = _ROOT / "models"
_MODELS_DIR.mkdir(exist_ok=True)

# -- Hyper-parameters (residual LSTM) -----------------------------------------
WINDOW      = 4      # quarters of residual history used as input
TEST_RATIO  = 0.20   # fraction of residual sequences held out for evaluation

# -- Roadmap output paths ------------------------------------------------------
_HYBRID_MODEL_PATH   = _MODELS_DIR / "hybrid_model.h5"
_HYBRID_RESULTS_PATH = _MODELS_DIR / "hybrid_results.json"
EPOCHS      = 300
BATCH_SIZE  = 8
PATIENCE    = 30


# =============================================================================
# Metric helpers
# =============================================================================

def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Root-Mean-Square Error."""
    return float(np.sqrt(mean_squared_error(actual.ravel(), pred.ravel())))


def _mape(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(
        np.mean(np.abs((actual - pred) / (np.abs(actual) + eps))) * 100
    )


def _per_column_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    col_names: List[str],
    label: str = "Model",
) -> Dict[str, Dict[str, float]]:
    """Return {col: {RMSE, MAPE}} and pretty-print a summary table."""
    metrics: Dict[str, Dict[str, float]] = {}
    print(f"\n{'=' * 58}")
    print(f"  {label} -- Test-Set Metrics")
    print(f"{'=' * 58}")
    print(f"  {'Variable':<26}  {'RMSE':>10}  {'MAPE (%)':>10}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*10}")
    for i, name in enumerate(col_names):
        r = _rmse(actual[:, i], pred[:, i])
        m = _mape(actual[:, i], pred[:, i])
        metrics[name] = {"RMSE": round(r, 6), "MAPE": round(m, 4)}
        print(f"  {name:<26}  {r:>10.4f}  {m:>9.2f}%")
    mean_r = np.mean([v["RMSE"] for v in metrics.values()])
    mean_m = np.mean([v["MAPE"] for v in metrics.values()])
    print(f"  {'-'*26}  {'-'*10}  {'-'*10}")
    print(f"  {'MEAN':<26}  {mean_r:>10.4f}  {mean_m:>9.2f}%")
    print(f"{'=' * 58}")
    return metrics


# =============================================================================
# Step 1 -- Compute VAR in-sample fitted values and residuals
# =============================================================================

def compute_var_residuals(
    var_results,
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract in-sample fitted values and residuals from a fitted VARResults
    object.

    The VAR model of order p can only produce fitted values starting at
    row p (the first p observations are used as initial conditions).  This
    function aligns everything to the same DatetimeIndex slice.

    Parameters
    ----------
    var_results : statsmodels VARResults
        A fitted model obtained from VAR(df).fit(p).
    df : pd.DataFrame
        The stationarity-adjusted DataFrame used to fit the model.
        Must have columns matching var_results.model.endog_names.

    Returns
    -------
    actual_df  : pd.DataFrame  -- rows [p:] of df (shape T-p x k)
    fitted_df  : pd.DataFrame  -- in-sample fitted values (shape T-p x k)
    resid_df   : pd.DataFrame  -- actual minus fitted  (shape T-p x k)
    """
    p          = var_results.k_ar
    col_names  = list(var_results.model.endog_names)

    # statsmodels stores residuals directly on the result object
    resid_arr  = var_results.resid          # shape (T-p, k)
    fitted_arr = var_results.fittedvalues   # shape (T-p, k)

    # Align index: VAR can only fit from row p onwards
    idx = df.index[p:]

    actual_df = df.iloc[p:].copy()
    fitted_df = pd.DataFrame(fitted_arr, index=idx, columns=col_names)
    resid_df  = pd.DataFrame(resid_arr,  index=idx, columns=col_names)

    print(f"[compute_var_residuals] p = {p}")
    print(f"[compute_var_residuals] Actual/fitted/residual rows: {len(resid_df)}")
    print(f"[compute_var_residuals] Residual stats (per column):")
    for col in col_names:
        r = resid_df[col]
        print(f"   {col:<30}  mean={r.mean():+.4f}  std={r.std():.4f}")

    return actual_df, fitted_df, resid_df


# =============================================================================
# Step 2 -- Prepare sliding-window residual sequences for LSTM training
# =============================================================================

def make_residual_sequences(
    resid_df: pd.DataFrame,
    window: int = WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised (X, y) pairs from the residual time-series.

    For every time step t in [0, N-window):
        X[t] = resid_arr[t : t+window]   shape (window, k)
        y[t] = resid_arr[t + window]     shape (k,)

    The residuals are used raw (no additional scaling):
      - They are already centred near zero.
      - Their magnitude is naturally small (model errors, not levels).
      - Avoiding an extra scaler simplifies the forecast step.

    Parameters
    ----------
    resid_df : pd.DataFrame -- VAR residuals (T-p, k)
    window   : int -- look-back window in quarters (default 4)

    Returns
    -------
    X : np.ndarray (N, window, k)
    y : np.ndarray (N, k)
    """
    arr  = resid_df.values.astype(np.float32)   # (T-p, k)
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i : i + window])
        y.append(arr[i + window])

    X_arr = np.array(X, dtype=np.float32)  # (N, window, k)
    y_arr = np.array(y, dtype=np.float32)  # (N, k)

    print(f"[make_residual_sequences] window={window}")
    print(f"[make_residual_sequences] X: {X_arr.shape}   y: {y_arr.shape}")
    return X_arr, y_arr


# =============================================================================
# Step 3 -- Build the residual-correction LSTM
# =============================================================================

def build_residual_lstm(window: int, n_features: int) -> keras.Model:
    """
    Lightweight stacked LSTM for residual correction.

    Architecture
    ------------
    Input(window, n_features)
      -> LSTM(32, return_sequences=True)
      -> Dropout(0.3)
      -> LSTM(16)
      -> Dropout(0.3)
      -> Dense(n_features, activation='linear')

    Smaller units than the main LSTM in model_lstm.py because residuals
    carry less structured information than level series.

    Parameters
    ----------
    window     : look-back window (quarters)
    n_features : number of VAR variables (= output width)

    Returns
    -------
    Compiled keras.Sequential model.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(window, n_features), name="resid_input"),
            layers.LSTM(32, return_sequences=True,  name="resid_lstm_1"),
            layers.Dropout(0.3,                     name="resid_drop_1"),
            layers.LSTM(16, return_sequences=False,  name="resid_lstm_2"),
            layers.Dropout(0.3,                     name="resid_drop_2"),
            layers.Dense(n_features, activation="linear", name="resid_output"),
        ],
        name="residual_correction_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse",
        metrics=["mae"],
    )
    return model


# =============================================================================
# Step 4 -- Train the residual LSTM
# =============================================================================

def train_residual_lstm(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
) -> keras.callbacks.History:
    """
    Train the residual-correction LSTM with early stopping and LR reduction.

    Parameters mirror train_model() in model_lstm.py for consistency.

    Returns
    -------
    keras History object.
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
        patience=12,
        min_lr=1e-7,
        verbose=0,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        shuffle=False,   # preserve temporal order
        verbose=1,
    )
    print(f"\n[train_residual_lstm] Stopped at epoch {len(history.history['loss'])}")
    return history


# =============================================================================
# Step 5 -- hybrid_forecast(periods)
# =============================================================================

def hybrid_forecast(
    periods: int,
    var_results,
    resid_lstm: keras.Model,
    df_diff: pd.DataFrame,
    resid_df: pd.DataFrame,
    df_raw: Optional[pd.DataFrame] = None,
    window: int = WINDOW,
    cumulate_levels: bool = True,
) -> pd.DataFrame:
    """
    Produce a ``periods``-step ahead hybrid forecast:

        hybrid_t = VAR_forecast_t  +  LSTM_residual_correction_t

    The LSTM predicts the expected residual for each future quarter using the
    last ``window`` observed residuals as its input window.  Because the VAR
    output is in first-differenced space, the hybrid forecast is also in that
    space; optionally it is cumulated back to level space using the last
    observed raw levels from ``df_raw``.

    Step-by-step algorithm
    ----------------------
    For h = 1 ... periods:

      VAR step
        y_hat_h = VAR point forecast h quarters ahead (standard recursive
                  multi-step forecast via var_results.forecast).

      LSTM residual step
        At step h=1 the seed window is the last ``window`` observed residuals.
        At step h>1 the window is updated by appending the previously predicted
        residual (pseudo-recursive -- we do not have true future residuals).

      Combination
        combined_h = y_hat_h + resid_pred_h

    Parameters
    ----------
    periods        : forecast horizon (quarters)
    var_results    : fitted statsmodels VARResults
    resid_lstm     : trained residual-correction keras Model
    df_diff        : the first-differenced DataFrame used to fit the VAR
    resid_df       : VAR in-sample residuals (output of compute_var_residuals)
    df_raw         : optional raw-levels DataFrame (used to cumulate levels)
    window         : residual LSTM look-back (must match training)
    cumulate_levels: if True and df_raw is provided, add a second DataFrame
                     showing cumulated level-path forecasts

    Returns
    -------
    pd.DataFrame with columns for each variable, index = forecast quarter dates.
    If cumulate_levels=True the returned DataFrame has a MultiIndex column:
        (variable, 'diff')  for first-differenced -> comparable to VAR output
        (variable, 'level') for cumulated levels  -> interpretable as actual values
    Otherwise: simple column = variable names (diff-space only).
    """
    p         = var_results.k_ar
    col_names = list(var_results.model.endog_names)
    k         = len(col_names)

    # 1. VAR point forecast (all h steps at once)
    seed_window = df_diff.values[-p:].astype(np.float64)   # last p obs
    var_fc      = var_results.forecast(seed_window, steps=periods)  # (h, k)

    # 2. LSTM residual forecast (recursive, one step at a time)
    # Seed the LSTM window with the last `window` observed residuals
    resid_window = resid_df.values[-window:].astype(np.float32).copy()  # (window, k)
    resid_preds  = []

    for _ in range(periods):
        # Shape the LSTM input: (1, window, k)
        x_in  = resid_window[np.newaxis, :, :]          # (1, window, k)
        r_hat = resid_lstm.predict(x_in, verbose=0)     # (1, k)
        resid_preds.append(r_hat[0])

        # Slide the window forward: drop oldest, append predicted residual
        resid_window = np.vstack([resid_window[1:], r_hat])  # (window, k)

    resid_fc = np.array(resid_preds, dtype=np.float32)  # (h, k)

    # 3. Combine
    hybrid_diff = var_fc + resid_fc.astype(np.float64)     # (h, k)

    # 4. Build date index for forecast quarters
    forecast_dates = pd.date_range(
        start=df_diff.index[-1] + pd.DateOffset(months=3),
        periods=periods,
        freq="QS",
    )

    # 5. Return results
    if cumulate_levels and df_raw is not None:
        records: dict = {}
        for i, col in enumerate(col_names):
            raw_col = col[2:] if col.startswith("d_") else col   # strip "d_"
            if raw_col in df_raw.columns:
                last_level = df_raw[raw_col].dropna().iloc[-1]
            else:
                last_level = 0.0
            records[(col, "diff")]  = pd.Series(hybrid_diff[:, i], index=forecast_dates)
            records[(col, "level")] = pd.Series(
                last_level + np.cumsum(hybrid_diff[:, i]), index=forecast_dates
            )

        result = pd.DataFrame(records)
        result.index.name = "Quarter"
    else:
        result = pd.DataFrame(hybrid_diff, index=forecast_dates, columns=col_names)

    print(f"\n[hybrid_forecast] {periods}-step hybrid forecast")
    print(f"  VAR component    shape : {var_fc.shape}")
    print(f"  Residual corr    shape : {resid_fc.shape}")
    print(f"  Combined         shape : {hybrid_diff.shape}")
    print(result.to_string())

    return result


# =============================================================================
# Step 6 -- Compare hybrid vs standalone VAR vs standalone residual LSTM
# =============================================================================

def compare_models(
    var_results,
    resid_lstm: keras.Model,
    df_diff: pd.DataFrame,
    actual_df: pd.DataFrame,
    resid_df: pd.DataFrame,
    test_ratio: float = TEST_RATIO,
    window: int = WINDOW,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate three models on the same chronological test window and print a
    side-by-side RMSE / MAPE comparison table.

    Models compared
    ---------------
    VAR          : raw VAR in-sample fitted values on the test slice.
    Residual LSTM: LSTM predictions of the residual on the test slice
                   (standalone; predictions compared against actual residuals).
    Hybrid       : VAR fitted + LSTM predicted residual (test slice).

    Parameters
    ----------
    var_results  : fitted statsmodels VARResults
    resid_lstm   : trained residual-correction keras Model
    df_diff      : first-differenced DataFrame used to fit the VAR
    actual_df    : actual values aligned with the VAR fitted span (T-p rows)
    resid_df     : VAR in-sample residuals (T-p rows)
    test_ratio   : fraction of residual sequences held out for testing
    window       : residual LSTM look-back window

    Returns
    -------
    dict with keys 'VAR', 'Residual_LSTM', 'Hybrid', each mapping to a
    {variable: {RMSE, MAPE}} metrics dict.
    """
    col_names = list(var_results.model.endog_names)

    # Build residual sequences (same as training prep)
    X_seq, y_seq = make_residual_sequences(resid_df, window=window)
    n_total      = len(X_seq)
    split_idx    = int(n_total * (1 - test_ratio))

    X_test       = X_seq[split_idx:]   # (n_test, window, k)
    y_test_resid = y_seq[split_idx:]   # (n_test, k)  <- actual residuals

    # The y_seq[i] corresponds to resid_df row (i + window).
    # y_test_resid[j] = resid_df.iloc[split_idx + window + j]
    test_row_start = split_idx + window
    test_row_end   = test_row_start + len(y_test_resid)

    # Actual macro values on test window
    y_actual_test  = actual_df.values[test_row_start:test_row_end]   # (n_test, k)

    # VAR fitted values on test window
    fitted_arr     = var_results.fittedvalues  # (T-p, k); aligned to actual_df
    y_var_test     = fitted_arr[test_row_start:test_row_end]          # (n_test, k)

    # LSTM residual predictions on test sequences
    y_resid_pred   = resid_lstm.predict(X_test, verbose=0).astype(np.float64)

    # Hybrid = VAR fitted + predicted residual
    y_hybrid_test  = y_var_test + y_resid_pred

    # Compute metrics
    print(f"\n{'#' * 60}")
    print("  MODEL COMPARISON: VAR  |  Residual LSTM  |  Hybrid")
    print(f"{'#' * 60}")
    print(f"  Test rows : {test_row_start} to {test_row_end - 1}  ({len(y_test_resid)} quarters)")

    var_metrics    = _per_column_metrics(y_actual_test, y_var_test,
                                         col_names, label="Standalone VAR")
    resid_metrics  = _per_column_metrics(y_test_resid, y_resid_pred,
                                         col_names, label="Residual LSTM (residual space)")
    hybrid_metrics = _per_column_metrics(y_actual_test, y_hybrid_test,
                                         col_names, label="Hybrid VAR+LSTM")

    # Summary comparison table
    print(f"\n{'=' * 76}")
    print("  SUMMARY -- RMSE comparison (first-differenced scale)")
    print(f"{'=' * 76}")
    print(f"  {'Variable':<26}  {'VAR RMSE':>10}  {'Hybrid RMSE':>12}  {'Delta RMSE':>12}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*12}  {'-'*12}")
    for col in col_names:
        v = var_metrics[col]["RMSE"]
        h = hybrid_metrics[col]["RMSE"]
        delta = h - v
        sign  = "down" if delta < 0 else "up  "
        print(f"  {col:<26}  {v:>10.4f}  {h:>12.4f}  [{sign}] {abs(delta):>8.4f}")
    print(f"{'=' * 76}")

    print(f"\n{'=' * 76}")
    print("  SUMMARY -- MAPE comparison (%)")
    print(f"{'=' * 76}")
    print(f"  {'Variable':<26}  {'VAR MAPE':>10}  {'Hybrid MAPE':>12}  {'Delta MAPE':>12}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*12}  {'-'*12}")
    for col in col_names:
        v = var_metrics[col]["MAPE"]
        h = hybrid_metrics[col]["MAPE"]
        delta = h - v
        sign  = "down" if delta < 0 else "up  "
        print(f"  {col:<26}  {v:>10.2f}%  {h:>11.2f}%  [{sign}] {abs(delta):>7.2f}%")
    print(f"{'=' * 76}")

    return {
        "VAR":           var_metrics,
        "Residual_LSTM": resid_metrics,
        "Hybrid":        hybrid_metrics,
    }


# =============================================================================
# Full end-to-end pipeline
# =============================================================================

def _q_to_date(label: str) -> pd.Timestamp:
    """Convert 'YYYY-Qn' string index entry to a monthly pd.Timestamp."""
    year, q = label.split("-Q")
    return pd.Timestamp(int(year), (int(q) - 1) * 3 + 1, 1)


def run_hybrid_pipeline(
    data_path: Path                   = _DATA_PATH,
    include_repo_rate: bool           = False,  # False = Scenario A (3 vars, full sample)
    var_lag_override: Optional[int]   = None,
    window: int                       = WINDOW,
    test_ratio: float                 = TEST_RATIO,
    epochs: int                       = EPOCHS,
    batch_size: int                   = BATCH_SIZE,
    patience: int                     = PATIENCE,
    forecast_periods: int             = 8,
    save_model_path: Optional[Path]   = _HYBRID_MODEL_PATH,
    save_results_path: Optional[Path] = _HYBRID_RESULTS_PATH,
    seed: int                         = 42,
) -> dict:
    """
    End-to-end Phase 6 hybrid pipeline.

    Parameters
    ----------
    data_path         : path to india_macro_quarterly.csv
    include_repo_rate : if True, include d_Repo_Rate (sample ~44 quarters)
    var_lag_override  : fix the VAR lag order (None = auto-select via BIC)
    window            : residual LSTM look-back window in quarters
    test_ratio        : fraction of residual sequences held out for testing
    epochs            : maximum training epochs for the residual LSTM
    batch_size        : mini-batch size
    patience          : early-stopping patience
    forecast_periods  : number of future quarters to forecast
    save_model_path   : where to save the trained residual LSTM (.keras)
    save_results_path : where to save the results JSON
    seed              : random seed for reproducibility

    Returns
    -------
    dict with keys:
        var_results, df_diff, df_raw,
        actual_df, fitted_df, resid_df,
        resid_lstm, X_train, X_test, y_train, y_test,
        comparison_metrics, forecast_df, col_names
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 65)
    print("  PHASE 6 -- HYBRID VAR + LSTM PIPELINE")
    print("=" * 65)

    # Import VAR helpers from model_var (already in this project)
    sys.path.insert(0, str(_SRC_DIR))
    from model_var import build_stationary_df, select_lag_order, fit_var

    # A. Load raw data (for level cumulation later)
    raw = pd.read_csv(data_path, index_col=0)
    raw.index = pd.DatetimeIndex([_q_to_date(q) for q in raw.index])
    raw.index.freq = "QS"
    df_raw = raw.copy()

    # B. Build stationary (first-differenced) DataFrame
    df_diff = build_stationary_df(
        data_path=data_path,
        include_repo_rate=include_repo_rate,
    )
    col_names  = df_diff.columns.tolist()
    n_features = len(col_names)

    # C. Fit VAR
    if var_lag_override is None:
        lag_order = select_lag_order(df_diff, maxlags=8, verbose=False)
    else:
        lag_order = var_lag_override
        print(f"[run_hybrid_pipeline] Using fixed lag order p={lag_order}")

    var_results = fit_var(df_diff, lag_order)

    # D. Extract residuals
    actual_df, fitted_df, resid_df = compute_var_residuals(var_results, df_diff)

    # E. Build residual sequences
    X_seq, y_seq = make_residual_sequences(resid_df, window=window)
    split        = int(len(X_seq) * (1 - test_ratio))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    print(f"\n[run_hybrid_pipeline] Residual sequences -- "
          f"train: {split}  test: {len(X_test)}")

    # F. Build + train residual LSTM
    resid_lstm = build_residual_lstm(window=window, n_features=n_features)
    resid_lstm.summary()
    history = train_residual_lstm(
        resid_lstm, X_train, y_train, X_test, y_test,
        epochs=epochs, batch_size=batch_size, patience=patience,
    )

    # G. Compare models on test set
    comparison_metrics = compare_models(
        var_results, resid_lstm, df_diff,
        actual_df, resid_df,
        test_ratio=test_ratio, window=window,
    )

    # H. Produce hybrid forecast
    forecast_df = hybrid_forecast(
        periods=forecast_periods,
        var_results=var_results,
        resid_lstm=resid_lstm,
        df_diff=df_diff,
        resid_df=resid_df,
        df_raw=df_raw,
        window=window,
        cumulate_levels=True,
    )

    # I. Save artefacts
    if save_model_path:
        # save_format='h5' produces hybrid_model.h5 as required by the roadmap
        resid_lstm.save(str(save_model_path), save_format="h5")
        print(f"\n[run_hybrid_pipeline] Residual LSTM saved -> {save_model_path}")

    if save_results_path:
        out = {
            "model":              "Hybrid VAR+LSTM",
            "var_lag_order":      lag_order,
            "window":             window,
            "columns":            col_names,
            "comparison":         comparison_metrics,
            "forecast_quarters":  [str(d.date()) for d in forecast_df.index],
        }
        with open(save_results_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"[run_hybrid_pipeline] Results JSON saved -> {save_results_path}")

    return {
        "var_results":         var_results,
        "df_diff":             df_diff,
        "df_raw":              df_raw,
        "actual_df":           actual_df,
        "fitted_df":           fitted_df,
        "resid_df":            resid_df,
        "resid_lstm":          resid_lstm,
        "history":             history,
        "X_train":             X_train,
        "X_test":              X_test,
        "y_train":             y_train,
        "y_test":              y_test,
        "comparison_metrics":  comparison_metrics,
        "forecast_df":         forecast_df,
        "col_names":           col_names,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import pickle

    # ── Option A: re-fit VAR on Scenario A (3 vars, full sample) from scratch
    # out = run_hybrid_pipeline(
    #     include_repo_rate=False,   # Scenario A — matches Phase-5 LSTM columns
    #     forecast_periods=8,
    # )

    # ── Option B: load the already-fitted Scenario-A VAR from disk and pass it
    #    in directly via var_lag_override + run_hybrid_pipeline.
    #    (The pkl was produced by build_04_var_notebook.py.)
    _pkl = _MODELS_DIR / "var_model.pkl"
    if _pkl.exists():
        with open(_pkl, "rb") as _fh:
            _var_loaded = pickle.load(_fh)
        print(f"[__main__] Loaded VAR from {_pkl}")
        print(f"           Columns  : {_var_loaded.model.endog_names}")
        print(f"           Lag order: {_var_loaded.k_ar}")
        # Note: var_model.pkl contains Scenario B (4 vars, with Repo Rate).
        # For a like-for-like comparison with Phase 5 LSTM (3 vars, full sample)
        # re-fit from scratch with include_repo_rate=False.

    out = run_hybrid_pipeline(
        include_repo_rate=False,   # Scenario A — same columns as Phase-5 LSTM
        forecast_periods=8,
    )
    print("\n[Done] Hybrid pipeline complete.")
    print(f"  Forecast head:\n{out['forecast_df'].head()}")

"""
src/model_arima.py
==================
Production ARIMA module for the Economic Policy Impact Simulator.

Public API
----------
  fit_arima(series, order=None)         → ARIMAResultsWrapper
  forecast_arima(result, steps, ...)    → dict

Run as a script to execute the full GDP_Growth forecasting pipeline and save
metrics to models/arima_results.json.

  python src/model_arima.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────────────────────
_SRC_DIR    = Path(__file__).parent
_ROOT       = _SRC_DIR.parent
_DATA_PATH  = _ROOT / "data" / "processed" / "india_macro_quarterly.csv"
_MODELS_DIR = _ROOT / "models"
_DOCS_DIR   = _ROOT / "docs"


# =============================================================================
# Helper — parse "YYYY-Qn" index into a proper DatetimeIndex
# =============================================================================

def _parse_quarter_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a 'YYYY-Qn' string index (e.g. '2010-Q1') to a DatetimeIndex
    with quarterly-start frequency ('QS').  Returns a copy of df with the
    new index.
    """
    def _q_to_date(label: str) -> pd.Timestamp:
        year, q = label.split("-Q")
        month = (int(q) - 1) * 3 + 1          # Q1→Jan, Q2→Apr, Q3→Jul, Q4→Oct
        return pd.Timestamp(int(year), month, 1)

    df2 = df.copy()
    df2.index = pd.DatetimeIndex([_q_to_date(q) for q in df2.index])
    df2.index.freq = "QS"
    return df2


# =============================================================================
# Public helper — load & prepare the target series
# =============================================================================

def load_series(
    column: str = "GDP_Growth",
    data_path: Path = _DATA_PATH,
) -> pd.Series:
    """
    Load `india_macro_quarterly.csv`, convert the quarter index to a
    DatetimeIndex, and return the requested column as a clean Series
    (NaN-forward-filled so ARIMA receives a complete, evenly-spaced series).

    Parameters
    ----------
    column    : column to extract (default "GDP_Growth")
    data_path : override for the CSV path (useful in tests)
    """
    df = pd.read_csv(data_path, index_col=0)
    df = _parse_quarter_index(df)
    series = df[column].copy().ffill().dropna()
    series.name = column
    return series


# =============================================================================
# Public API — fit_arima
# =============================================================================

def fit_arima(
    series: pd.Series,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal: bool = False,
    max_p: int = 5,
    max_q: int = 5,
) -> object:
    """
    Fit an ARIMA model on *series*.

    If *order* is None (default), ``auto_arima`` is used to select the
    optimal (p, d, q) via AIC minimisation.  If *order* is specified, the
    model is fitted directly with those parameters.

    Parameters
    ----------
    series   : pd.Series with a DatetimeIndex at quarterly frequency
    order    : (p, d, q) tuple, or None to auto-select
    seasonal : whether to also search seasonal (P, D, Q, m=4) components
    max_p    : upper bound on p in auto_arima grid search
    max_q    : upper bound on q in auto_arima grid search

    Returns
    -------
    Fitted ``statsmodels.tsa.arima.model.ARIMAResultsWrapper`` object.
    Also stores the selected order as `result.model.order`.
    """
    if order is None:
        print("Running auto_arima to select optimal (p, d, q) …")
        auto = auto_arima(
            series,
            start_p=0, max_p=max_p,
            start_q=0, max_q=max_q,
            d=None,                   # auto-determine via ADF
            seasonal=seasonal,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=True,
        )
        order = auto.order
        print(f"\nBest order → p={order[0]}, d={order[1]}, q={order[2]}")
        print(f"AIC = {auto.aic():.4f}")

    print(f"\nFitting statsmodels ARIMA{order} …")
    model  = ARIMA(series, order=order)
    result = model.fit()
    print(result.summary())
    return result


# =============================================================================
# Public API — forecast_arima
# =============================================================================

def forecast_arima(
    result,
    steps: int,
    conf_level: float = 0.95,
    index: Optional[pd.DatetimeIndex] = None,
) -> dict:
    """
    Produce an out-of-sample forecast from a fitted ARIMA result.

    Parameters
    ----------
    result     : fitted ARIMAResultsWrapper (output of fit_arima)
    steps      : number of periods to forecast
    conf_level : confidence level for the interval (default 0.95)
    index      : optional DatetimeIndex to attach to the forecast series;
                 if None, statsmodels default integer index is used

    Returns
    -------
    dict with keys:
        forecast  (pd.Series) – point forecasts
        lower_ci  (pd.Series) – lower confidence bound
        upper_ci  (pd.Series) – upper confidence bound
        conf_int  (pd.DataFrame) – full CI DataFrame
        dates     (pd.DatetimeIndex or None)
    """
    alpha = 1 - conf_level
    fc_obj  = result.get_forecast(steps=steps)
    fc_mean = fc_obj.predicted_mean
    fc_ci   = fc_obj.conf_int(alpha=alpha)

    if index is not None:
        fc_mean.index = index
        fc_ci.index   = index

    return {
        "forecast": fc_mean,
        "lower_ci": fc_ci.iloc[:, 0],
        "upper_ci": fc_ci.iloc[:, 1],
        "conf_int": fc_ci,
        "dates":    index,
    }


# =============================================================================
# Evaluation helper
# =============================================================================

def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Compute RMSE and MAPE between actual and predicted series.

    MAPE is computed only on non-zero actual values to avoid division-by-zero.

    Returns
    -------
    dict with keys: RMSE, MAPE
    """
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    non_zero = actual != 0
    if non_zero.sum() < len(actual):
        print("Warning: some actual values are zero — MAPE on non-zero subset.")
    mape = float(
        np.mean(np.abs(
            (actual[non_zero] - predicted[non_zero]) / actual[non_zero]
        )) * 100
    )

    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f} %")
    return {"RMSE": rmse, "MAPE": mape}


# =============================================================================
# Plot helper
# =============================================================================

def plot_arima_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    conf_int: pd.DataFrame,
    metrics: dict,
    order: tuple,
    save_path: Optional[Path] = None,
) -> None:
    """
    Dark-themed plot: training history, test actuals, ARIMA forecast + 95 % CI.
    Saves to *save_path* if provided.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3d4d")

    # Training
    ax.plot(train.index, train.values,
            color="#7eb8f7", linewidth=1.8, label="Train (actual)", zorder=3)

    # Test actuals
    ax.plot(test.index, test.values,
            color="#a8e6cf", linewidth=2.0, linestyle="--",
            label="Test (actual)", zorder=4)

    # Forecast
    ax.plot(forecast.index, forecast.values,
            color="#ff9f7f", linewidth=2.2, marker="o", markersize=4,
            label=f"ARIMA{order} forecast", zorder=5)

    # 95 % CI ribbon
    ax.fill_between(
        conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
        color="#ff9f7f", alpha=0.18, label="95 % CI",
    )

    # Train | Test divider
    split_date = test.index[0]
    ax.axvline(split_date, color="#888", linestyle=":", linewidth=1.2, alpha=0.8)
    ylim = ax.get_ylim()
    ax.text(split_date, ylim[1] - (ylim[1] - ylim[0]) * 0.05,
            "  Train │ Test", color="#aaa", fontsize=9, va="top")

    # Metrics annotation
    metrics_txt = f"RMSE: {metrics['RMSE']:.3f}\nMAPE: {metrics['MAPE']:.2f}%"
    ax.text(
        0.02, 0.97, metrics_txt,
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3a",
                  edgecolor="#ff9f7f", alpha=0.85),
        color="white",
    )

    ax.set_title(f"GDP Growth — ARIMA{order} Forecast  (India, 2010-Q1 → 2025-Q4)",
                 fontsize=14, color="white", pad=14, fontweight="bold")
    ax.set_xlabel("Quarter", color="#ccc", fontsize=11)
    ax.set_ylabel("GDP Annual Growth (%)", color="#ccc", fontsize=11)
    ax.tick_params(colors="#bbb", labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=35, ha="right")
    ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#3a3d4d",
              labelcolor="white", loc="upper left")
    ax.grid(True, color="#2a2d3a", linewidth=0.6, zorder=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Plot saved → {save_path}")

    plt.show()


# =============================================================================
# Full pipeline  (called by __main__ and by the notebook)
# =============================================================================

def run_arima_pipeline(
    column: str = "GDP_Growth",
    train_ratio: float = 0.80,
    data_path: Path = _DATA_PATH,
    save_plot: Optional[Path] = None,
    save_results: Optional[Path] = None,
) -> dict:
    """
    End-to-end ARIMA pipeline on the real India macro dataset.

    Steps:
      1. Load & prepare the series
      2. Chronological train/test split (80 / 20)
      3. Auto-select (p, d, q) via AIC
      4. Fit statsmodels ARIMA
      5. Out-of-sample forecast
      6. Evaluate (RMSE, MAPE)
      7. Plot actual vs forecast
      8. Persist metrics to JSON

    Returns
    -------
    dict with keys: order, metrics, forecast, conf_int, model_result, series
    """
    # 1. Series
    series = load_series(column=column, data_path=data_path)
    print(f"Series loaded: {series.index[0].date()} → {series.index[-1].date()}"
          f"  ({len(series)} observations)")

    # 2. Chronological split
    n_train = int(len(series) * train_ratio)
    train   = series.iloc[:n_train]
    test    = series.iloc[n_train:]
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} quarters)")
    print(f"Test : {test.index[0].date()}  → {test.index[-1].date()}   ({len(test)} quarters)")

    # 3 & 4. Fit (auto-select order)
    result = fit_arima(train)
    order  = result.model.order

    # 5. Forecast
    fc = forecast_arima(result, steps=len(test), index=test.index)

    # 6. Evaluate
    print("\n── Evaluation ──")
    metrics = evaluate_forecast(test, fc["forecast"])

    # 7. Plot
    if save_plot is None:
        save_plot = _DOCS_DIR / "arima_forecast.png"
    matplotlib.use("Agg")   # non-interactive when run as a script
    plot_arima_forecast(
        train, test, fc["forecast"], fc["conf_int"],
        metrics, order, save_path=save_plot,
    )

    # 8. Save JSON
    results_dict = {
        "model":        "ARIMA",
        "target":       column,
        "order":        list(order),
        "train_period": {
            "start": str(train.index[0].date()),
            "end":   str(train.index[-1].date()),
        },
        "test_period": {
            "start": str(test.index[0].date()),
            "end":   str(test.index[-1].date()),
        },
        "n_train": len(train),
        "n_test":  len(test),
        "metrics": metrics,
        "aic":     float(result.aic),
        "bic":     float(result.bic),
    }

    if save_results is None:
        save_results = _MODELS_DIR / "arima_results.json"
    save_results.parent.mkdir(parents=True, exist_ok=True)
    with open(save_results, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved → {save_results}")

    return {
        "order":        order,
        "metrics":      metrics,
        "forecast":     fc["forecast"],
        "conf_int":     fc["conf_int"],
        "model_result": result,
        "series":       series,
        "train":        train,
        "test":         test,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_arima_pipeline()

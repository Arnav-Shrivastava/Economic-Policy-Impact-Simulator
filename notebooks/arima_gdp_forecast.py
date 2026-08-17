"""
ARIMA Forecasting for GDP_Growth
=================================
Uses pmdarima.auto_arima to select optimal (p, d, q) parameters via AIC,
then fits a statsmodels ARIMA model, forecasts the test period, evaluates
with RMSE & MAPE, and plots actual vs. predicted values.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ── 0. Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ Prepare the series
# ─────────────────────────────────────────────────────────────────────────────

def prepare_series(df: pd.DataFrame, column: str = "GDP_Growth") -> pd.Series:
    """
    Extract the target column and ensure the index is a DatetimeIndex with
    quarterly frequency (QS = Quarter Start).  Any missing values are forward-
    filled so that ARIMA receives a complete, evenly-spaced series.
    """
    series = df[column].copy()

    # Convert index to DatetimeIndex if it isn't already
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    series = series.asfreq("QS")           # enforce quarterly frequency
    series = series.ffill().dropna()       # fill any gaps
    return series


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ Train / Test split
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(series: pd.Series, train_ratio: float = 0.80):
    """
    Split a time series into training and test sets **chronologically**.

    WHY NOT RANDOM SPLIT?
    ---------------------
    Time series observations are not i.i.d. — each value depends on its past
    values (autocorrelation).  Randomly shuffling the data would:
      1. Let the model "see" future observations during training, causing
         data leakage and artificially optimistic metrics.
      2. Destroy the temporal ordering that ARIMA's autoregressive and
         moving-average components are built to model.
    A strict chronological split (earliest 80 % → train, latest 20 % → test)
    is the correct approach and mirrors real-world deployment where the model
    only has access to historical data when making future predictions.
    """
    split_idx = int(len(series) * train_ratio)
    train = series.iloc[:split_idx]
    test  = series.iloc[split_idx:]
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} quarters)")
    print(f"Test : {test.index[0].date()}  → {test.index[-1].date()}   ({len(test)} quarters)")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ Automatic (p, d, q) selection via AIC
# ─────────────────────────────────────────────────────────────────────────────

def select_arima_order(train: pd.Series, seasonal: bool = False) -> tuple:
    """
    Use pmdarima.auto_arima to exhaustively search over a grid of (p, d, q)
    values and return the combination that minimises the Akaike Information
    Criterion (AIC).

    AIC balances goodness-of-fit against model complexity:
        AIC = 2k − 2 ln(L)
    where k = number of free parameters and L = maximised likelihood.
    A lower AIC indicates a better model after penalising for extra parameters.

    Parameters
    ----------
    seasonal : bool
        Set True to additionally search seasonal (P, D, Q, m=4) parameters
        (i.e., SARIMA).  Disabled here because we are fitting a plain ARIMA.
    """
    print("\nRunning auto_arima to select optimal (p, d, q) …")
    auto_model = auto_arima(
        train,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=None,                  # let auto_arima determine d via ADF test
        seasonal=seasonal,
        information_criterion="aic",
        stepwise=True,           # stepwise search is faster than exhaustive
        suppress_warnings=True,
        error_action="ignore",
        trace=True,              # print each candidate model's AIC
    )
    order = auto_model.order
    print(f"\nBest order selected → p={order[0]}, d={order[1]}, q={order[2]}")
    print(f"AIC of selected model: {auto_model.aic():.4f}")
    return order


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ Fit statsmodels ARIMA & forecast
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_forecast(
    train: pd.Series,
    test: pd.Series,
    order: tuple,
):
    """
    Fit a statsmodels ARIMA model on the training data using the order found
    by auto_arima, then produce an out-of-sample forecast for the test horizon.

    Returns
    -------
    forecast_series : pd.Series  – point forecasts aligned to the test index
    result          : ARIMAResults – fitted model object (access .summary(), etc.)
    """
    print(f"\nFitting statsmodels ARIMA{order} on training data …")
    model  = ARIMA(train, order=order)
    result = model.fit()

    # --- Out-of-sample forecast for exactly len(test) steps ---
    forecast_obj    = result.get_forecast(steps=len(test))
    forecast_mean   = forecast_obj.predicted_mean
    forecast_conf   = forecast_obj.conf_int(alpha=0.05)   # 95 % CI

    # Align index to test dates
    forecast_mean.index = test.index
    forecast_conf.index = test.index

    print(result.summary())
    return forecast_mean, forecast_conf, result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Compute RMSE and MAPE.

    RMSE (Root Mean Squared Error)
        sqrt[ (1/n) * sum( (y_i - y_hat_i)^2 ) ]
        Same units as GDP_Growth; penalises large errors more heavily.

    MAPE (Mean Absolute Percentage Error)
        (100/n) * sum( |y_i - y_hat_i| / |y_i| )
        Scale-independent; expressed as a percentage.
        NOTE: undefined / unstable when actual values are at or near zero.
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Guard against zero actuals to avoid division by zero in MAPE
    non_zero_mask = actual != 0
    if non_zero_mask.sum() < len(actual):
        print("Warning: Some actual values are zero — MAPE computed on non-zero observations only.")
    mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask])
                           / actual[non_zero_mask])) * 100

    print(f"\n{'─'*40}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f} %")
    print(f"{'─'*40}")
    return {"RMSE": rmse, "MAPE": mape}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    conf_int: pd.DataFrame,
    metrics: dict,
    order: tuple,
    save_path=None,
) -> None:
    """
    Plot the full GDP_Growth series with the ARIMA forecast and 95 % CI ribbon.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Styling
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3d4d")

    # Training history
    ax.plot(train.index, train.values,
            color="#7eb8f7", linewidth=1.6, label="Train (actual)", zorder=3)

    # Test actuals
    ax.plot(test.index, test.values,
            color="#a8e6cf", linewidth=2.0, linestyle="--",
            label="Test (actual)", zorder=4)

    # Forecast
    ax.plot(forecast.index, forecast.values,
            color="#ff9f7f", linewidth=2.2, marker="o", markersize=4,
            label=f"ARIMA{order} forecast", zorder=5)

    # 95 % Confidence interval ribbon
    ax.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color="#ff9f7f", alpha=0.18, label="95 % CI",
    )

    # Vertical split line
    split_date = test.index[0]
    ax.axvline(split_date, color="#888", linestyle=":", linewidth=1.2, alpha=0.8)
    ylim = ax.get_ylim()
    ax.text(split_date, ylim[1] - (ylim[1] - ylim[0]) * 0.05,
            " Train | Test", color="#aaa", fontsize=9, va="top")

    # Metrics annotation
    metrics_text = f"RMSE: {metrics['RMSE']:.3f}\nMAPE: {metrics['MAPE']:.2f}%"
    ax.text(
        0.02, 0.97, metrics_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3a",
                  edgecolor="#ff9f7f", alpha=0.85),
        color="white",
    )

    # Labels & formatting
    ax.set_title(f"GDP Growth — ARIMA{order} Forecast", fontsize=15,
                 color="white", pad=14, fontweight="bold")
    ax.set_xlabel("Quarter", color="#ccc", fontsize=11)
    ax.set_ylabel("GDP Growth (%)", color="#ccc", fontsize=11)
    ax.tick_params(colors="#bbb", labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=35, ha="right")
    ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#3a3d4d",
              labelcolor="white", loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nPlot saved → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ─ Main pipeline  (call this with your DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def run_arima_pipeline(
    df: pd.DataFrame,
    column: str = "GDP_Growth",
    train_ratio: float = 0.80,
    save_plot=None,
) -> dict:
    """
    End-to-end ARIMA forecasting pipeline.

    Parameters
    ----------
    df          : pd.DataFrame with a DatetimeIndex (quarterly) and GDP_Growth column
    column      : name of the target column
    train_ratio : fraction of observations used for training (default 0.80)
    save_plot   : file path to save the plot; None to skip saving

    Returns
    -------
    dict with keys: order, metrics, forecast, conf_int, model_result
    """
    # 1. Prepare
    series = prepare_series(df, column)

    # 2. Split — chronologically (see docstring inside the function for the reason)
    train, test = chronological_split(series, train_ratio)

    # 3. Auto-select (p, d, q)
    order = select_arima_order(train)

    # 4. Fit & forecast
    forecast, conf_int, result = fit_and_forecast(train, test, order)

    # 5. Evaluate
    metrics = evaluate(test, forecast)

    # 6. Plot
    plot_forecast(train, test, forecast, conf_int, metrics, order,
                  save_path=save_plot)

    return {
        "order":        order,
        "metrics":      metrics,
        "forecast":     forecast,
        "conf_int":     conf_int,
        "model_result": result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo with synthetic data  (remove / replace with your real `df`)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Build a realistic synthetic quarterly GDP growth series (2000-2024) ---
    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=100, freq="QS")
    gdp   = (
        2.5                                             # long-run mean
        + np.cumsum(np.random.normal(0, 0.25, 100))    # random walk component
        + np.sin(np.linspace(0, 6 * np.pi, 100)) * 0.6 # mild cyclicality
        + np.random.normal(0, 0.3, 100)                 # noise
    )

    df_demo = pd.DataFrame({"GDP_Growth": gdp}, index=dates)

    results = run_arima_pipeline(df_demo, column="GDP_Growth", save_plot=None)

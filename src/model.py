"""
src/model.py
============
Public API shim for the Economic Policy Impact Simulator Streamlit app.

Re-exports ``hybrid_forecast`` from model_hybrid.py and provides
``policy_hybrid_forecast`` — a convenience wrapper that accepts a
Repo Rate change (basis-point shock) and returns a forecast DataFrame
with confidence-interval columns ready for plotting.

Usage
-----
from src.model import hybrid_forecast, policy_hybrid_forecast
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Re-export the core function so app code can do `from src.model import hybrid_forecast`
from model_hybrid import (  # noqa: F401
    hybrid_forecast,
    compute_var_residuals,
    build_residual_lstm,
)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so they work from any cwd)
# ---------------------------------------------------------------------------
_SRC_DIR    = Path(__file__).parent
_ROOT       = _SRC_DIR.parent
_DATA_PATH  = _ROOT / "data" / "processed" / "india_macro_quarterly.csv"
_MODELS_DIR = _ROOT / "models"

# ---------------------------------------------------------------------------
# Empirically estimated pass-through coefficients from Repo Rate to macro vars
# (based on Granger causality & impulse response analysis in Phase 4 VAR).
# These are used to *shift the last seed observation* before forecasting,
# approximating the first-order effect of a Repo Rate change on each variable.
#
# Sign convention (per quarter, first-differenced units):
#   CPI_Inflation    : -0.15  (rate hike reduces inflation)
#   GDP_Growth       : -0.10  (rate hike reduces growth)
#   Unemployment_Rate: +0.05  (rate hike raises unemployment)
# ---------------------------------------------------------------------------
_REPO_PASSTHROUGH: dict[str, float] = {
    "d_CPI_Inflation":    -0.15,
    "d_GDP_Growth":       -0.10,
    "d_Unemployment_Rate": 0.05,
}


def policy_hybrid_forecast(
    repo_rate_change: float,
    var_results,
    resid_lstm,
    df_diff: pd.DataFrame,
    resid_df: pd.DataFrame,
    df_raw: pd.DataFrame,
    periods: int = 8,
    window: int = 4,
    n_bootstrap: int = 100,
    ci_level: float = 0.90,
) -> pd.DataFrame:
    """
    Run a policy-scenario hybrid forecast with a Repo Rate shock.

    The shock is applied by perturbing the *last seed row* of ``df_diff``
    using the empirical pass-through coefficients in ``_REPO_PASSTHROUGH``.
    This is an approximation of a reduced-form impulse response.

    Confidence intervals are estimated via a simple parametric approach:
        lower = point_forecast - z * sigma_resid * sqrt(h)
        upper = point_forecast + z * sigma_resid * sqrt(h)
    where sigma_resid is the in-sample residual standard deviation per variable
    and h is the forecast horizon (quarters ahead).

    Parameters
    ----------
    repo_rate_change : float
        Change in Repo Rate in percentage points (e.g. +0.50 means +50 bps).
    var_results      : fitted statsmodels VARResults
    resid_lstm       : trained residual-correction keras Model
    df_diff          : first-differenced DataFrame used to fit the VAR
    resid_df         : VAR in-sample residuals
    df_raw           : raw (levels) DataFrame for level cumulation
    periods          : forecast horizon in quarters (default 8)
    window           : LSTM look-back window (must match training)
    n_bootstrap      : unused; kept for API compatibility
    ci_level         : confidence level (default 0.90)

    Returns
    -------
    pd.DataFrame with columns:
        Quarter (index)
        Variable
        forecast  -- point forecast (level space)
        lower     -- lower CI bound (level space)
        upper     -- upper CI bound (level space)
    """
    import warnings
    warnings.filterwarnings("ignore")

    col_names = list(var_results.model.endog_names)

    # z-score for the requested CI level
    rng = np.random.default_rng(42)
    z = float(np.abs(np.percentile(rng.standard_normal(100_000),
                                   (1 + ci_level) / 2 * 100)))

    # --- Apply policy shock to last seed row ---------------------------------
    df_diff_shocked = df_diff.copy()
    for col in col_names:
        if col in _REPO_PASSTHROUGH:
            loc = df_diff_shocked.columns.get_loc(col)
            df_diff_shocked.iloc[-1, loc] += (
                repo_rate_change * _REPO_PASSTHROUGH[col]
            )

    # --- Run hybrid forecast --------------------------------------------------
    forecast_df = hybrid_forecast(
        periods=periods,
        var_results=var_results,
        resid_lstm=resid_lstm,
        df_diff=df_diff_shocked,
        resid_df=resid_df,
        df_raw=df_raw,
        window=window,
        cumulate_levels=True,
    )

    # --- Residual std per variable (for CI width) ----------------------------
    resid_std = resid_df.std()  # Series indexed by col_names

    # --- Build tidy output DataFrame -----------------------------------------
    records: list[dict] = []
    for col in col_names:
        raw_name = col[2:] if col.startswith("d_") else col
        level_series: pd.Series = forecast_df[(col, "level")]
        horizons = np.arange(1, periods + 1)
        ci_half = z * float(resid_std[col]) * np.sqrt(horizons)

        for h, (dt, fc) in enumerate(level_series.items(), start=1):
            records.append({
                "Quarter":  dt,
                "Variable": raw_name,
                "forecast": fc,
                "lower":    fc - ci_half[h - 1],
                "upper":    fc + ci_half[h - 1],
            })

    out = pd.DataFrame(records)
    out["Quarter"] = pd.to_datetime(out["Quarter"])
    return out


def load_artifacts() -> Tuple:
    """
    Load the pre-trained hybrid LSTM and re-fit a Scenario-A VAR (3 vars,
    no Repo Rate) from raw data so that shapes always match the saved
    ``hybrid_model.h5`` which expects input (None, 4, 3).

    Note: ``var_model.pkl`` may contain a Scenario-B VAR (4 vars including
    Repo Rate).  We deliberately ignore it here to avoid shape mismatches.

    Returns
    -------
    (var_results, resid_lstm, df_diff, resid_df, df_raw)
    """
    import sys
    import warnings
    warnings.filterwarnings("ignore")

    sys.path.insert(0, str(_SRC_DIR))

    from tensorflow import keras
    from model_var import build_stationary_df, select_lag_order, fit_var
    from model_hybrid import compute_var_residuals

    # 1. Load saved hybrid LSTM (Scenario A: 3-feature input)
    h5_path = _MODELS_DIR / "hybrid_model.h5"
    resid_lstm = keras.models.load_model(str(h5_path), compile=False)

    # 2. Build Scenario-A stationary DataFrame (3 vars, full sample)
    df_diff = build_stationary_df(
        data_path=_DATA_PATH,
        include_repo_rate=False,   # Scenario A -- matches hybrid_model.h5
        include_iip=False,
    )

    # 3. Re-fit VAR at lag=1 (matches saved hybrid_results.json)
    var_results = fit_var(df_diff, lag_order=1)

    # 4. Rebuild residuals
    _, _, resid_df = compute_var_residuals(var_results, df_diff)

    # 5. Load raw levels DataFrame
    def _q_to_date(label: str) -> pd.Timestamp:
        year, q = label.split("-Q")
        return pd.Timestamp(int(year), (int(q) - 1) * 3 + 1, 1)

    raw = pd.read_csv(_DATA_PATH, index_col=0)
    raw.index = pd.DatetimeIndex([_q_to_date(q) for q in raw.index])
    raw.index.freq = "QS"
    df_raw = raw.copy()

    return var_results, resid_lstm, df_diff, resid_df, df_raw

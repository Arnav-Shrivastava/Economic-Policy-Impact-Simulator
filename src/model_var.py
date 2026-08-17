"""
src/model_var.py
================
VAR (Vector AutoRegression) pipeline for the Economic Policy Impact Simulator.

What it does
------------
1. Loads `data/processed/india_macro_quarterly.csv` and builds the
   stationarity-adjusted DataFrame (first differences of all I(1) series,
   as established in Phase 2 EDA).
2. Uses ``model.select_order(maxlags=8)`` to pick the optimal lag via AIC/BIC.
3. Fits the VAR model at that lag order.
4. Prints the full model summary.
5. Runs Granger causality tests between Repo_Rate and every other variable,
   with plain-English interpretation of the p-values.

Dataset caveats handled here
-----------------------------
• Repo_Rate : ends 2020-Q1 -> 19 quarters of NaN (2021-Q2 -> 2025-Q4).
  Only included when ``include_repo_rate=True``.  If included, rows after
  2021-Q1 are dropped, shrinking the usable sample to ~44 quarters.
• IIP_Growth : starts 2023-Q2 -> 53 quarters of NaN.
  Only included when ``include_iip=True``.  If included, sample shrinks to
  ~11 quarters -- far too small for a reliable VAR -- so it is excluded by
  default.

Run from the project root:
    python src/model_var.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

# Force UTF-8 output so Unicode in statsmodels summaries and print statements
# doesn't crash on Windows (which defaults to cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────────────────────
_SRC_DIR   = Path(__file__).parent
_ROOT      = _SRC_DIR.parent
_DATA_PATH = _ROOT / "data" / "processed" / "india_macro_quarterly.csv"


# =============================================================================
# Step 0 -- Parse quarter index
# =============================================================================

def _parse_quarter_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'YYYY-Qn' string index to a quarterly DatetimeIndex (QS freq)."""
    def _q_to_date(label: str) -> pd.Timestamp:
        year, q = label.split("-Q")
        month = (int(q) - 1) * 3 + 1
        return pd.Timestamp(int(year), month, 1)

    out = df.copy()
    out.index = pd.DatetimeIndex([_q_to_date(q) for q in out.index])
    out.index.freq = "QS"
    return out


# =============================================================================
# Step 1 -- Build stationarity-adjusted DataFrame
# =============================================================================

def build_stationary_df(
    data_path: Path = _DATA_PATH,
    include_repo_rate: bool = True,
    include_iip: bool = False,
) -> pd.DataFrame:
    """
    Load the raw quarterly CSV and produce a stationarity-adjusted DataFrame
    suitable for VAR estimation.

    All five indicators are I(1) per Phase 2 ADF tests, so the transformation
    is first-differencing:   d_col[t] = col[t] - col[t-1]

    Parameters
    ----------
    include_repo_rate : bool
        If True, include d_Repo_Rate.  This limits the sample to rows where
        Repo_Rate is non-null (up to 2021-Q1, ~44 quarters after differencing).
    include_iip : bool
        If True, include d_IIP_Growth.  This limits the sample to the 11
        quarters available (2023-Q2 -> 2025-Q4), which is too small for VAR.
        Excluded by default.

    Returns
    -------
    pd.DataFrame -- first-differenced, NaN rows dropped, DatetimeIndex.
    """
    raw = pd.read_csv(data_path, index_col=0)
    raw = _parse_quarter_index(raw)

    VALUE_COLS = ["Repo_Rate", "CPI_Inflation", "GDP_Growth",
                  "Unemployment_Rate", "IIP_Growth"]
    df = raw[VALUE_COLS].copy()

    # ── First-difference every column ─────────────────────────────────────────
    df_diff = df.diff()
    df_diff.columns = [f"d_{c}" for c in df_diff.columns]

    # ── Select which columns to include ───────────────────────────────────────
    cols = ["d_CPI_Inflation", "d_GDP_Growth", "d_Unemployment_Rate"]
    if include_repo_rate:
        cols = ["d_Repo_Rate"] + cols
    if include_iip:
        cols = cols + ["d_IIP_Growth"]

    df_model = df_diff[cols].dropna()

    print(f"[build_stationary_df] Columns   : {df_model.columns.tolist()}")
    print(f"[build_stationary_df] Sample     : {df_model.index[0].date()}"
          f" -> {df_model.index[-1].date()} ({len(df_model)} quarters)")
    return df_model


# =============================================================================
# Step 2 -- Lag order selection
# =============================================================================

def select_lag_order(
    df: pd.DataFrame,
    maxlags: int = 8,
    verbose: bool = True,
) -> int:
    """
    Fit a VAR on ``df`` and select the optimal number of lags using
    ``select_order``.  Returns the AIC-optimal lag (falls back to BIC if AIC
    gives an implausibly large value relative to the sample).

    Why AIC vs BIC?
    ---------------
    AIC tends to select richer models (more lags), which is desirable for
    macro data where impulse responses can persist several quarters.
    BIC applies a heavier penalty for extra parameters and is preferred when
    the sample is small (n < 60).  With ~44 usable quarters here, BIC is
    the safer choice, but AIC is shown for comparison.

    Parameters
    ----------
    df      : stationarity-adjusted DataFrame (output of build_stationary_df)
    maxlags : maximum number of lags to search (default 8 = 2 years of quarters)

    Returns
    -------
    int -- selected lag order
    """
    model = VAR(df)
    lag_results = model.select_order(maxlags=maxlags)

    if verbose:
        print("\n" + "=" * 70)
        print("  LAG ORDER SELECTION RESULTS")
        print("=" * 70)
        print(lag_results.summary())
        print(f"\n  AIC optimal lag : {lag_results.aic}")
        print(f"  BIC optimal lag : {lag_results.bic}")
        print(f"  FPE optimal lag : {lag_results.fpe}")
        print(f"  HQIC optimal lag: {lag_results.hqic}")

    # With small samples prefer BIC; fallback to 1 if 0 is selected
    selected = max(lag_results.bic, 1)
    print(f"\n  >> Selected lag order (BIC): {selected}")
    return selected


# =============================================================================
# Step 3 -- Fit VAR
# =============================================================================

def fit_var(df: pd.DataFrame, lag_order: int) -> object:
    """
    Fit a VAR(p) model on ``df`` at ``lag_order`` lags.

    The VAR(p) model jointly models k endogenous variables:
        Y_t = A_1 * Y_{t-1} + A_2 * Y_{t-2} + ... + A_p * Y_{t-p} + u_t

    where Y_t is the k-vector of first-differenced macro variables,
    A_i are (k x k) coefficient matrices, and u_t is white noise.

    Returns
    -------
    Fitted VARResults object (statsmodels).
    """
    print(f"\n{'=' * 70}")
    print(f"  FITTING VAR({lag_order})")
    print("=" * 70)

    model  = VAR(df)
    result = model.fit(lag_order)

    print(result.summary())
    return result


# =============================================================================
# Step 4 -- Granger causality tests
# =============================================================================

def granger_causality_analysis(
    df: pd.DataFrame,
    lag_order: int,
    caused_by: str = "d_Repo_Rate",
    alpha: float = 0.05,
) -> dict:
    """
    Run Granger causality tests: does ``caused_by`` Granger-cause each of the
    other variables in ``df``?

    What is Granger causality?
    --------------------------
    Variable X is said to *Granger-cause* variable Y if past values of X
    contain information that helps predict Y, **over and above** what Y's own
    past values already tell us.  Formally the test compares two regressions:

        Restricted   : Y_t = f(Y_{t-1}, ..., Y_{t-p})
        Unrestricted : Y_t = f(Y_{t-1}, ..., Y_{t-p}, X_{t-1}, ..., X_{t-p})

    H0: X does NOT Granger-cause Y  (the X lags add no predictive power)
    H1: X DOES Granger-cause Y

    p <= alpha -> reject H0 -> X Granger-causes Y.

    Important caveat: Granger causality is a *predictive* concept, not
    structural causality.  It tells you about temporal precedence and
    information content, not about the underlying economic mechanism.

    Parameters
    ----------
    df         : stationarity-adjusted DataFrame
    lag_order  : the VAR lag order (same p used in the VAR)
    caused_by  : column name of the "cause" variable (default "d_Repo_Rate")
    alpha      : significance level (default 0.05)

    Returns
    -------
    dict mapping target_column -> {"p_value": float, "significant": bool}
    """
    if caused_by not in df.columns:
        print(f"  [Granger] '{caused_by}' not in DataFrame -- skipping.")
        return {}

    target_cols = [c for c in df.columns if c != caused_by]
    results = {}

    print(f"\n{'=' * 70}")
    print(f"  GRANGER CAUSALITY: Does '{caused_by}' Granger-cause each variable?")
    print(f"  H0: '{caused_by}' does NOT Granger-cause the target")
    print(f"  Significance level: alpha = {alpha}  |  Lag order: {lag_order}")
    print("=" * 70)

    for target in target_cols:
        # grangercausalitytests expects a 2-col DataFrame: [Y, X]
        pair_df = df[[target, caused_by]].dropna()

        print(f"\n  -- Target: {target} --")
        gc_out = grangercausalitytests(pair_df, maxlag=lag_order, verbose=False)

        # Collect the F-test p-value at the selected lag order
        if lag_order in gc_out:
            ftest   = gc_out[lag_order][0]["ssr_ftest"]
            p_value = ftest[1]
        else:
            # Fallback: use the lag with the smallest p-value
            p_value = min(
                gc_out[lag][0]["ssr_ftest"][1]
                for lag in gc_out
            )

        significant = p_value <= alpha
        results[target] = {"p_value": p_value, "significant": significant}

        # ── Plain-English interpretation ───────────────────────────────────
        cause_clean  = caused_by.replace("d_", "").replace("_", " ")
        target_clean = target.replace("d_", "").replace("_", " ")

        print(f"  F-test p-value : {p_value:.4f}")

        if significant:
            print(
                f"  Result         : SIGNIFICANT (p={p_value:.4f} <= {alpha})\n"
                f"\n"
                f"  Plain English  : Past changes in {cause_clean} do contain\n"
                f"                   statistically useful information for predicting\n"
                f"                   future changes in {target_clean}, even after\n"
                f"                   accounting for {target_clean}'s own history.\n"
                f"                   In macro terms: {cause_clean} movements\n"
                f"                   *lead* {target_clean} by at least one quarter.\n"
                f"                   (This is consistent with monetary policy\n"
                f"                   transmission with a lag.)"
            )
        else:
            print(
                f"  Result         : NOT significant (p={p_value:.4f} > {alpha})\n"
                f"\n"
                f"  Plain English  : Past changes in {cause_clean} do NOT add\n"
                f"                   meaningful predictive power for future\n"
                f"                   {target_clean} movements once we already\n"
                f"                   account for {target_clean}'s own history.\n"
                f"                   This could mean: (a) the transmission\n"
                f"                   lag is longer than {lag_order} quarter(s),\n"
                f"                   (b) the relationship is non-linear, or\n"
                f"                   (c) there genuinely is no Granger-causal link."
            )

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  SUMMARY -- Granger causality from '{caused_by}'")
    print(f"{'─' * 70}")
    print(f"  {'Target':<30} {'p-value':>10}  {'Granger-causes?':>15}")
    print(f"  {'─'*30} {'─'*10}  {'─'*15}")
    for target, r in results.items():
        sig_str = "YES ✓" if r["significant"] else "NO  ✗"
        print(f"  {target:<30} {r['p_value']:>10.4f}  {sig_str:>15}")
    print(f"{'─' * 70}\n")

    return results


# =============================================================================
# Full pipeline
# =============================================================================

def run_var_pipeline(
    data_path: Path = _DATA_PATH,
    maxlags: int = 8,
    include_repo_rate: bool = True,
    include_iip: bool = False,
    granger_source: Optional[str] = None,
    alpha: float = 0.05,
) -> dict:
    """
    End-to-end VAR pipeline.

    Steps:
      1. Build stationarity-adjusted DataFrame (first differences)
      2. Select optimal lag order via AIC/BIC
      3. Fit VAR(p)
      4. Print model summary
      5. Granger causality tests

    Parameters
    ----------
    maxlags           : max lags in select_order search (default 8)
    include_repo_rate : include d_Repo_Rate (limits sample to <= 2021-Q1)
    include_iip       : include d_IIP_Growth (n≈9 -- not recommended)
    granger_source    : column name to test as Granger-cause; auto-detected
                        (uses d_Repo_Rate if present, else d_CPI_Inflation)
    alpha             : significance level for Granger tests (default 0.05)

    Returns
    -------
    dict with keys: df, lag_order, var_result, granger_results
    """
    print("\n" + "=" * 70)
    print("  ECONOMIC POLICY IMPACT SIMULATOR -- VAR PIPELINE")
    print("=" * 70)

    # 1. Build stationary df
    df = build_stationary_df(
        data_path=data_path,
        include_repo_rate=include_repo_rate,
        include_iip=include_iip,
    )

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} complete rows -- too few for a reliable VAR. "
            "Try include_repo_rate=False (uses full 2010-2025 sample) or "
            "include_iip=False."
        )

    # 2. Lag order
    lag_order = select_lag_order(df, maxlags=maxlags)

    # 3 & 4. Fit + summary
    var_result = fit_var(df, lag_order)

    # 5. Granger causality
    if granger_source is None:
        granger_source = "d_Repo_Rate" if "d_Repo_Rate" in df.columns \
                         else "d_CPI_Inflation"

    granger_results = granger_causality_analysis(
        df, lag_order, caused_by=granger_source, alpha=alpha
    )

    return {
        "df":              df,
        "lag_order":       lag_order,
        "var_result":      var_result,
        "granger_results": granger_results,
    }

# =============================================================================
# Impulse Response Function (IRF)
# =============================================================================

def plot_irf(
    var_result,
    impulse_var: str = "d_Repo_Rate",
    periods: int = 10,
    save_path=None,
    orth: bool = True,
) -> object:
    """
    Compute and plot the Impulse Response Function (IRF) showing how a
    one-unit shock to ``impulse_var`` propagates to every other variable
    over ``periods`` quarters.

    What is an IRF?
    ---------------
    An IRF traces the effect of a single, temporary one-unit shock to one
    variable (here: d_Repo_Rate) on all other variables in the system,
    holding everything else constant.

    Because the VAR operates on *first-differenced* series, the responses
    are changes in the rate-of-change.  Cumulate (sum) them to recover
    the effect on *levels*.

    Orthogonalised IRF (orth=True, default)
    ----------------------------------------
    The Cholesky decomposition of the residual covariance matrix isolates
    each shock structurally.  The variable ordering in the DataFrame sets
    the causal chain:
        d_Repo_Rate -> d_CPI_Inflation -> d_GDP_Growth -> d_Unemployment_Rate
    This means monetary policy is assumed to affect output and prices only
    with a lag, not in the same quarter.

    Parameters
    ----------
    var_result  : fitted VARResults object
    impulse_var : column whose shock we trace (default 'd_Repo_Rate')
    periods     : forecast horizon in quarters (default 10)
    save_path   : optional path to save the figure
    orth        : use orthogonalised IRF (default True)

    Returns
    -------
    irf_obj : statsmodels IRAnalysis object (.irfs / .orth_irfs arrays)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    col_names = var_result.model.endog_names
    if impulse_var not in col_names:
        raise ValueError(
            f"'{impulse_var}' not found in model variables: {col_names}.\n"
            "Re-run with include_repo_rate=True."
        )

    irf_obj       = var_result.irf(periods=periods)
    responses     = irf_obj.orth_irfs if orth else irf_obj.irfs
    impulse_idx   = col_names.index(impulse_var)
    response_cols = [c for c in col_names if c != impulse_var]
    n_resp        = len(response_cols)
    quarters      = np.arange(periods + 1)

    # Confidence intervals (bootstrap; skipped gracefully if it fails)
    has_ci = False
    try:
        ci = irf_obj.orth_errband_mc() if orth else irf_obj.errband_mc()
        lower_ci, upper_ci = ci
        has_ci = True
    except Exception:
        pass

    # ── Dark theme ────────────────────────────────────────────────────────
    BG     = "#0f1117"
    GRID   = "#2a2d3a"
    TEXT   = "#e0e0e0"
    COLORS = ["#7eb8f7", "#a8e6cf", "#ff9f7f", "#c3a6ff", "#ffd580"]

    fig, axes = plt.subplots(1, n_resp, figsize=(5 * n_resp, 5), squeeze=False)
    fig.patch.set_facecolor(BG)
    impulse_label = impulse_var.replace("d_", "").replace("_", " ")
    fig.suptitle(
        f"Orthogonalised IRF: 1-unit shock to {impulse_label}  "
        f"({periods}-quarter horizon)",
        fontsize=13, fontweight="bold", color=TEXT, y=1.02,
    )

    for ax_i, (resp_col, color) in enumerate(zip(response_cols, COLORS)):
        ax = axes[0][ax_i]
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

        resp_idx = col_names.index(resp_col)
        irf_vals = responses[:, resp_idx, impulse_idx]

        ax.plot(quarters, irf_vals, color=color, linewidth=2.2,
                marker="o", markersize=4, zorder=4, label="IRF")
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", alpha=0.6)

        if has_ci:
            try:
                lo = lower_ci[:, resp_idx, impulse_idx]
                hi = upper_ci[:, resp_idx, impulse_idx]
                ax.fill_between(quarters, lo, hi, color=color,
                                alpha=0.18, label="95% CI", zorder=3)
            except Exception:
                pass

        ax.fill_between(quarters, irf_vals, 0,
                        where=(irf_vals >= 0), alpha=0.10, color=color)
        ax.fill_between(quarters, irf_vals, 0,
                        where=(irf_vals < 0),  alpha=0.10, color="#ff4444")

        resp_label = resp_col.replace("d_", "D").replace("_", " ")
        ax.set_title(resp_label, fontsize=11, color=TEXT,
                     fontweight="bold", pad=8)
        ax.set_xlabel("Quarters after shock", color="#ccc", fontsize=9)
        ax.set_ylabel("Response (first diff)", color="#ccc", fontsize=9)
        ax.tick_params(colors="#bbb", labelsize=8)
        ax.set_xlim(0, periods)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID,
                  labelcolor="white", loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"IRF plot saved -> {save_path}")
    plt.show()

    # Numeric table
    irf_df = pd.DataFrame(
        {resp: responses[:, col_names.index(resp), impulse_idx]
         for resp in response_cols},
        index=[f"Q+{i}" for i in range(periods + 1)],
    ).round(6)
    print("\nOrthogonalised IRF values (rows = quarters after shock):")
    print(irf_df.to_string())
    return irf_obj


# =============================================================================
# Policy shock simulation
# =============================================================================

def simulate_policy_shock(
    var_result,
    df: "pd.DataFrame",
    repo_rate_change: float = 0.5,
    periods: int = 8,
    impulse_var: str = "d_Repo_Rate",
    save_path=None,
) -> "pd.DataFrame":
    """
    Project how all macro indicators would evolve after a hypothetical
    one-time change in the Repo Rate, compared to a no-shock baseline.

    Approach: IRF scaling
    ---------------------
    Because the VAR is linear, a shock of size ``s`` produces exactly
    ``s`` times the IRF response.  Steps:
      1. Compute the orthogonalised IRF.
      2. Scale by ``repo_rate_change`` to get the additional response of
         every variable above the no-shock baseline.
      3. Add those scaled IRF values to the baseline VAR.forecast.
      4. Cumulate the first-differenced projections back to level paths
         using the last observed raw level from the CSV.

    Parameters
    ----------
    var_result       : fitted VARResults (from fit_var / run_var_pipeline)
    df               : stationarity-adjusted DataFrame used to fit the model
    repo_rate_change : repo rate change in pp (e.g. +0.50 = 50 bps hike,
                       -0.25 = 25 bps cut)
    periods          : quarters to project (default 8)
    impulse_var      : column to shock (default 'd_Repo_Rate')
    save_path        : optional path to save the comparison plot

    Returns
    -------
    pd.DataFrame with MultiIndex columns (d_col, scenario) where scenario
    is one of 'baseline', 'shocked', or 'delta'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    col_names   = var_result.model.endog_names
    lag_order   = var_result.k_ar

    if impulse_var not in col_names:
        raise ValueError(
            f"'{impulse_var}' not in model variables: {col_names}.\n"
            "Re-run the VAR with include_repo_rate=True."
        )
    impulse_idx = col_names.index(impulse_var)

    # ── Baseline VAR forecast ─────────────────────────────────────────────
    last_obs    = df.values[-lag_order:]
    baseline_fc = var_result.forecast(last_obs, steps=periods)
    baseline_df = pd.DataFrame(baseline_fc, columns=col_names)

    # ── Scaled IRF shock  ─────────────────────────────────────────────────
    irf_obj    = var_result.irf(periods=periods)
    orth_irfs  = irf_obj.orth_irfs                            # (periods+1, k, k)
    irf_scaled = orth_irfs[1:periods+1, :, impulse_idx] * repo_rate_change

    shocked_df = baseline_df + pd.DataFrame(irf_scaled, columns=col_names)

    # ── Reconstruct level paths from raw CSV ──────────────────────────────
    raw = pd.read_csv(_DATA_PATH, index_col=0)

    def _q_to_date(label):
        year, q = label.split("-Q")
        return pd.Timestamp(int(year), (int(q) - 1) * 3 + 1, 1)

    raw_idx        = pd.DatetimeIndex([_q_to_date(q) for q in raw.index])
    forecast_dates = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=3),
        periods=periods, freq="QS",
    )

    records = {}
    for d_col in col_names:
        raw_col = d_col[2:]        # strip "d_" prefix
        if raw_col in raw.columns:
            raw_series = pd.Series(raw[raw_col].values, index=raw_idx)
            last_level = raw_series.dropna().iloc[-1]
        else:
            last_level = 0.0

        base_lev  = last_level + np.cumsum(baseline_df[d_col].values)
        shock_lev = last_level + np.cumsum(shocked_df[d_col].values)

        records[(d_col, "baseline")] = pd.Series(base_lev,  index=forecast_dates)
        records[(d_col, "shocked")]  = pd.Series(shock_lev, index=forecast_dates)
        records[(d_col, "delta")]    = pd.Series(
            shock_lev - base_lev, index=forecast_dates)

    result = pd.DataFrame(records)
    result.index.name = "Quarter"

    # ── Print summary ─────────────────────────────────────────────────────
    sign_str = ("+" if repo_rate_change >= 0 else "") + str(repo_rate_change)
    print(f"\n{'='*68}")
    print(f"  POLICY SIMULATION: {sign_str} pp Repo Rate shock")
    print(f"  Horizon: {periods} quarters from {df.index[-1].date()}")
    print(f"{'='*68}")
    for d_col in col_names:
        label = d_col.replace("d_", "").replace("_", " ")
        tbl = pd.DataFrame({
            "Baseline": result[(d_col, "baseline")].round(4),
            "Shocked":  result[(d_col, "shocked")].round(4),
            "Delta":    result[(d_col, "delta")].round(4),
        })
        tbl.index = [f"Q+{i+1}" for i in range(len(tbl))]
        print(f"\n  {label}:")
        print(tbl.to_string(col_space=12))

    # ── Plot ──────────────────────────────────────────────────────────────
    BG      = "#0f1117"
    GRID    = "#2a2d3a"
    TEXT    = "#e0e0e0"
    BASE_C  = "#7eb8f7"
    SHOCK_C = "#ff9f7f" if repo_rate_change >= 0 else "#a8e6cf"

    n_cols = len(col_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Policy Simulation: {sign_str} pp Repo Rate shock  "
        f"({periods}-quarter projection)",
        fontsize=13, fontweight="bold", color=TEXT, y=1.02,
    )

    for ax_i, d_col in enumerate(col_names):
        ax = axes[0][ax_i]
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

        base  = result[(d_col, "baseline")]
        shock = result[(d_col, "shocked")]

        ax.plot(forecast_dates, base.values,  color=BASE_C,  linewidth=2.0,
                linestyle="--", label="Baseline", zorder=4)
        ax.plot(forecast_dates, shock.values, color=SHOCK_C, linewidth=2.2,
                marker="o", markersize=4, label="Shocked", zorder=5)
        ax.fill_between(forecast_dates, base.values, shock.values,
                        alpha=0.15, color=SHOCK_C, zorder=3)

        label = d_col.replace("d_", "").replace("_", " ")
        ax.set_title(label, fontsize=11, color=TEXT, fontweight="bold", pad=8)
        ax.set_xlabel("Quarter", color="#ccc", fontsize=9)
        ax.tick_params(colors="#bbb", labelsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nPolicy shock plot saved -> {save_path}")
    plt.show()

    return result


# =============================================================================
# Entry point
# =============================================================================


# =============================================================================
# Bootstrap Forecast Intervals
# =============================================================================

def bootstrap_forecast_intervals(
    var_result,
    df,
    periods: int = 8,
    n_boot: int = 1000,
    ci: float = 0.90,
    seed: int = 42,
    save_path=None,
):
    """
    Construct empirical forecast intervals for a fitted VAR model using
    residual resampling (non-parametric bootstrap).

    Why residual bootstrap instead of parametric intervals?
    -------------------------------------------------------
    statsmodels VAR.forecast_interval() uses asymptotic Gaussian theory,
    which assumes normally distributed residuals.  Macro residuals often
    exhibit fat tails (e.g. the COVID quarter) and skewness.  Residual
    resampling makes no distributional assumption: it lets the actual
    observed residuals drive the uncertainty, which is more honest for
    short macro time-series.

    Algorithm (per replicate)
    -------------------------
    1.  Draw ``periods`` rows from ``var_result.resid`` WITH REPLACEMENT,
        giving a bootstrapped shock sequence u*_1 ... u*_T.
    2.  Propagate the VAR forward step-by-step starting from the last
        ``lag_order`` observed rows (the seed window):

            y_hat_t = intercept + A_1 @ y_{t-1} + ... + A_p @ y_{t-p}
            y_boot_t = y_hat_t + u*_t

        where A_l are the fitted coefficient matrices.  The simulated
        y_boot_t becomes the new history for the next step.
    3.  Repeat n_boot times. Compute percentile bands across replicates.

    Parameters
    ----------
    var_result : fitted VARResults  (from fit_var / run_var_pipeline)
    df         : stationarity-adjusted pd.DataFrame used to fit the model;
                 the last ``k_ar`` rows serve as the simulation seed
    periods    : forecast horizon in quarters (default 8)
    n_boot     : number of bootstrap replicates (default 1000)
    ci         : confidence level, e.g. 0.90 yields 5th-95th pct band
    seed       : random seed for reproducibility (default 42)
    save_path  : optional Path/str to save the fan chart figure

    Returns
    -------
    pd.DataFrame with MultiIndex columns (variable, band):
        band in {'lower', 'point', 'upper'}
    Index is a DatetimeIndex of the forecast quarters.

    Example
    -------
    out = run_var_pipeline(include_repo_rate=True, maxlags=6)
    bdf = bootstrap_forecast_intervals(out['var_result'], out['df'],
                                       periods=8, n_boot=1000, ci=0.90)
    # Access one variable's bands:
    print(bdf[('d_GDP_Growth', 'lower')])   # 5th pct
    print(bdf[('d_GDP_Growth', 'point')])   # deterministic point forecast
    print(bdf[('d_GDP_Growth', 'upper')])   # 95th pct
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    col_names  = list(var_result.model.endog_names)
    k          = len(col_names)
    lag_order  = var_result.k_ar
    intercept  = var_result.intercept    # (k,)
    coefs      = var_result.coefs        # (lag_order, k, k)
    resid_pool = np.asarray(var_result.resid)   # (n_obs - lag_order, k)

    # Seed buffer: last lag_order rows in REVERSE order so buf[0] = y_{t-1}
    seed_buf = df.values[-lag_order:][::-1].copy()   # (lag_order, k)

    # ── Deterministic point forecast (zero noise) ──────────────────────────
    point_paths = np.zeros((periods, k))
    buf = seed_buf.copy()
    for t in range(periods):
        y_hat = intercept.copy().astype(float)
        for l in range(lag_order):
            y_hat += coefs[l] @ buf[l]
        point_paths[t] = y_hat
        buf = np.vstack([y_hat, buf[:-1]])

    # ── Bootstrap replicates ───────────────────────────────────────────────
    boot_paths = np.zeros((n_boot, periods, k))

    for b in range(n_boot):
        # Sample residual rows WITH REPLACEMENT
        idx        = rng.integers(0, len(resid_pool), size=periods)
        boot_resid = resid_pool[idx]          # (periods, k)

        buf = seed_buf.copy()
        for t in range(periods):
            y_hat = intercept.copy().astype(float)
            for l in range(lag_order):
                y_hat += coefs[l] @ buf[l]
            y_hat += boot_resid[t]            # add resampled shock
            boot_paths[b, t] = y_hat
            buf = np.vstack([y_hat, buf[:-1]])

    # ── Percentile bands ───────────────────────────────────────────────────
    alpha    = (1.0 - ci) / 2.0
    lo_pct   = alpha * 100         # e.g. 5.0
    hi_pct   = (1.0 - alpha) * 100 # e.g. 95.0
    lo_band  = np.percentile(boot_paths, lo_pct, axis=0)   # (periods, k)
    hi_band  = np.percentile(boot_paths, hi_pct, axis=0)   # (periods, k)

    # Also compute inner quartile bands for the fan chart
    q25_band = np.percentile(boot_paths, 25, axis=0)
    q75_band = np.percentile(boot_paths, 75, axis=0)
    q10_band = np.percentile(boot_paths, 10, axis=0)
    q90_band = np.percentile(boot_paths, 90, axis=0)

    # ── Build output DataFrame ─────────────────────────────────────────────
    forecast_dates = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=3),
        periods=periods,
        freq="QS",
    )

    records = {}
    for j, col in enumerate(col_names):
        records[(col, "lower")] = pd.Series(lo_band[:, j],     index=forecast_dates)
        records[(col, "point")] = pd.Series(point_paths[:, j], index=forecast_dates)
        records[(col, "upper")] = pd.Series(hi_band[:, j],     index=forecast_dates)

    result = pd.DataFrame(records)
    result.index.name = "Quarter"

    # ── Print summary ──────────────────────────────────────────────────────
    lo_label = int(lo_pct)
    hi_label = int(hi_pct)
    print(f"\n{'='*72}")
    print(f"  BOOTSTRAP FORECAST INTERVALS")
    print(f"  Replicates : {n_boot}   |   CI level : {int(ci*100)}%  "
          f"({lo_label}th-{hi_label}th percentile)")
    print(f"  Horizon    : {periods} quarters from {df.index[-1].date()}")
    print(f"  Residuals  : {len(resid_pool)} rows in pool")
    print(f"{'='*72}")

    for col in col_names:
        label = col.replace("d_", "").replace("_", " ")
        tbl = pd.DataFrame({
            f"  {lo_label}th pct": result[(col, "lower")].round(4),
            "  Point":             result[(col, "point")].round(4),
            f"  {hi_label}th pct": result[(col, "upper")].round(4),
        })
        tbl.index = [f"Q+{i+1}" for i in range(len(tbl))]
        print(f"\n  {label}:")
        print(tbl.to_string())

    # ── Fan chart ──────────────────────────────────────────────────────────
    BG     = "#0f1117"
    GRID   = "#2a2d3a"
    TEXT   = "#e0e0e0"
    COLORS = ["#7eb8f7", "#a8e6cf", "#ff9f7f", "#c3a6ff", "#ffd580"]

    n_cols = len(col_names)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(5 * n_cols, 5), squeeze=False
    )
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"VAR Bootstrap Fan Chart  "
        f"({n_boot} replicates, {int(ci*100)}% CI — {lo_label}th to {hi_label}th pct)",
        fontsize=13, fontweight="bold", color=TEXT, y=1.02,
    )

    for ax_i, (col, color) in enumerate(zip(col_names, COLORS)):
        ax = axes[0][ax_i]
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

        j     = col_names.index(col)
        lo    = lo_band[:, j]
        hi    = hi_band[:, j]
        pt    = point_paths[:, j]
        dates = result.index

        # Outermost fan layer: 10th-90th pct
        ax.fill_between(dates, q10_band[:, j], q90_band[:, j],
                        color=color, alpha=0.08, label="10-90 pct")
        # Inner quartile range
        ax.fill_between(dates, q25_band[:, j], q75_band[:, j],
                        color=color, alpha=0.16, label="25-75 pct")
        # Requested CI band
        ax.fill_between(dates, lo, hi,
                        color=color, alpha=0.24,
                        label=f"{lo_label}-{hi_label} pct")
        # Point forecast line
        ax.plot(dates, pt, color=color, linewidth=2.2,
                marker="o", markersize=4, zorder=5, label="Point")
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", alpha=0.5)

        label = col.replace("d_", "D").replace("_", " ")
        ax.set_title(label, fontsize=11, color=TEXT, fontweight="bold", pad=8)
        ax.set_xlabel("Quarter", color="#ccc", fontsize=9)
        ax.set_ylabel("First diff (pp)", color="#ccc", fontsize=9)
        ax.tick_params(colors="#bbb", labelsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.legend(fontsize=7, facecolor=BG, edgecolor=GRID,
                  labelcolor="white", loc="upper right",
                  framealpha=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nFan chart saved -> {save_path}")
    plt.show()

    return result


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    ROOT     = Path(__file__).parent.parent
    DOCS_DIR = ROOT / "docs"
    DOCS_DIR.mkdir(exist_ok=True)

    print("\n" + "#" * 70)
    print("# SCENARIO A -- 3 core variables, full sample (2010-Q2 -> 2025-Q4)")
    print("#" * 70)
    out_a = run_var_pipeline(
        include_repo_rate=False,
        include_iip=False,
        granger_source="d_CPI_Inflation",
        maxlags=8,
    )

    print("\n" + "#" * 70)
    print("# SCENARIO B -- 4 variables incl. Repo_Rate (2011-Q1 -> 2021-Q1)")
    print("#" * 70)
    out_b = run_var_pipeline(
        include_repo_rate=True,
        include_iip=False,
        granger_source="d_Repo_Rate",
        maxlags=6,
    )

    # ── IRF: 10-quarter horizon, shock to Repo_Rate ──────────────────────
    print("\n" + "#" * 70)
    print("# IRF: 10-quarter shock to Repo_Rate (Scenario B)")
    print("#" * 70)
    irf = plot_irf(
        out_b["var_result"],
        impulse_var="d_Repo_Rate",
        periods=10,
        orth=True,
        save_path=DOCS_DIR / "var_irf_repo_rate.png",
    )

    # ── Policy simulation: +50 bps hike ──────────────────────────────────
    print("\n" + "#" * 70)
    print("# POLICY SIMULATION: +0.50 pp Repo Rate hike")
    print("#" * 70)
    hike = simulate_policy_shock(
        out_b["var_result"],
        out_b["df"],
        repo_rate_change=+0.50,
        periods=8,
        impulse_var="d_Repo_Rate",
        save_path=DOCS_DIR / "var_shock_hike_50bps.png",
    )

    # ── Policy simulation: -25 bps cut ───────────────────────────────────
    print("\n" + "#" * 70)
    print("# POLICY SIMULATION: -0.25 pp Repo Rate cut")
    print("#" * 70)
    cut = simulate_policy_shock(
        out_b["var_result"],
        out_b["df"],
        repo_rate_change=-0.25,
        periods=8,
        impulse_var="d_Repo_Rate",
        save_path=DOCS_DIR / "var_shock_cut_25bps.png",
    )

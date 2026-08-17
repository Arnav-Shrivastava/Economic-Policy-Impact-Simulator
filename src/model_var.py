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
# Entry point
# =============================================================================

if __name__ == "__main__":
    # ── Configuration ──────────────────────────────────────────────────────
    # include_repo_rate=True  -> sample capped at 2021-Q1 (~44 quarters)
    # include_repo_rate=False -> full sample 2010-Q2 -> 2025-Q4 (~62 quarters)
    #
    # We run BOTH scenarios so you can compare:

    print("\n" + "#" * 70)
    print("# SCENARIO A -- 3 core variables, full sample (2010-Q2 -> 2025-Q4)")
    print("#" * 70)
    out_a = run_var_pipeline(
        include_repo_rate=False,
        include_iip=False,
        granger_source="d_CPI_Inflation",   # test CPI as leading indicator
        maxlags=8,
    )

    print("\n" + "#" * 70)
    print("# SCENARIO B -- 4 variables including Repo_Rate (2011-Q1 -> 2021-Q1)")
    print("#" * 70)
    out_b = run_var_pipeline(
        include_repo_rate=True,
        include_iip=False,
        granger_source="d_Repo_Rate",       # test monetary policy lead
        maxlags=6,   # lower max because sample is shorter
    )

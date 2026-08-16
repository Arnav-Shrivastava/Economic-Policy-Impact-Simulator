"""
resample_quarterly.py
---------------------
Resamples a monthly-indexed DataFrame (CPI_Inflation, IIP_Growth) to
quarterly frequency using the mean of each quarter.

Handles:
  - Gaps / missing months via forward-fill before aggregation
  - Quarter labels formatted as "YYYY-Qn" (e.g. 2023-Q1)
  - Optional logging of which months were imputed
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Core utility
# ---------------------------------------------------------------------------

def resample_to_quarterly(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fill_method: str = "ffill",
    min_periods: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Resample a monthly DatetimeIndex DataFrame to quarterly frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a monthly DatetimeIndex and at least the
        columns [CPI_Inflation, IIP_Growth].
    columns : list[str] | None
        Columns to include in the output.  Defaults to all numeric columns.
    fill_method : {"ffill", "bfill", "interpolate", None}
        Strategy to fill missing months *before* resampling:
          - "ffill"        – forward-fill (carry last known value)
          - "bfill"        – backward-fill
          - "interpolate"  – linear interpolation
          - None           – leave NaN as-is (quarter mean ignores them)
    min_periods : int
        Minimum number of non-NaN months required to compute a quarter mean.
        Quarters with fewer observations are set to NaN.
    verbose : bool
        Print a summary of missing months that were imputed.

    Returns
    -------
    pd.DataFrame
        Quarterly DataFrame indexed by period strings "YYYY-Qn".
    """

    # ── 1. Validate index ────────────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a DatetimeIndex. "
            f"Got {type(df.index).__name__} instead."
        )

    # ── 2. Select columns ───────────────────────────────────────────────────
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Column(s) not found in DataFrame: {missing_cols}")

    df = df[columns].copy()

    # ── 3. Reindex to a complete monthly grid ───────────────────────────────
    full_monthly_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS",          # Month-Start keeps things unambiguous
    )
    missing_months = full_monthly_index.difference(df.index)

    if verbose and len(missing_months):
        print(
            f"[resample_quarterly] WARNING: {len(missing_months)} missing month(s) "
            f"detected - imputing with '{fill_method}':\n"
            f"  {missing_months.strftime('%Y-%m').tolist()}"
        )

    df = df.reindex(full_monthly_index)   # introduces NaN for absent months

    # -- 4. Fill missing months ----------------------------------------------
    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    elif fill_method == "interpolate":
        df = df.interpolate(method="linear", limit_direction="both")
    elif fill_method is not None:
        raise ValueError(
            f"Unknown fill_method '{fill_method}'. "
            "Choose from: 'ffill', 'bfill', 'interpolate', None."
        )

    # -- 5. Resample -> quarterly mean ---------------------------------------
    quarterly = (
        df.resample("QS")                      # Quarter-Start anchors
          .agg(lambda x: x.mean() if x.notna().sum() >= min_periods else np.nan)
    )

    # -- 6. Format index as "YYYY-Qn" strings --------------------------------
    quarterly.index = quarterly.index.to_period("Q").strftime("%Y-Q%q")
    quarterly.index.name = "Quarter"

    if verbose:
        print(
            f"[resample_quarterly] OK: Resampled {len(df)} monthly rows "
            f"-> {len(quarterly)} quarters."
        )

    return quarterly


# ---------------------------------------------------------------------------
# Demo / quick-start
# ---------------------------------------------------------------------------

def _build_sample_dataframe() -> pd.DataFrame:
    """Create a realistic sample with intentional gaps for demonstration."""
    np.random.seed(42)

    dates = pd.date_range("2021-01-01", periods=36, freq="MS")   # 3 years

    cpi = 4.0 + np.cumsum(np.random.normal(0, 0.3, len(dates)))  # random walk
    iip = 3.0 + np.cumsum(np.random.normal(0, 0.5, len(dates)))

    df = pd.DataFrame({"CPI_Inflation": cpi, "IIP_Growth": iip}, index=dates)

    # Simulate missing months (gaps a downstream pipeline might produce)
    df.loc["2021-04-01"] = np.nan   # single missing month
    df = df.drop("2022-08-01")      # entirely absent row

    return df


if __name__ == "__main__":
    # -- Build sample data ---------------------------------------------------
    monthly_df = _build_sample_dataframe()

    print("-- Monthly DataFrame (first 8 rows) ------------------------------")
    print(monthly_df.head(8).to_string())
    print()

    # -- Resample ------------------------------------------------------------
    quarterly_df = resample_to_quarterly(
        monthly_df,
        columns=["CPI_Inflation", "IIP_Growth"],
        fill_method="ffill",    # change to "interpolate" or None as needed
        min_periods=1,
        verbose=True,
    )

    print()
    print("-- Quarterly DataFrame -------------------------------------------")
    print(quarterly_df.round(4).to_string())

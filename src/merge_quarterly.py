"""
merge_quarterly.py
------------------
Merges four economic indicator DataFrames onto a canonical quarterly
DatetimeIndex spanning 2010-Q1 to 2025-Q4.

Sources
-------
  - repo_rate    : quarterly DatetimeIndex
  - cpi          : quarterly DatetimeIndex
  - gdp          : quarterly DatetimeIndex
  - unemployment : mixed -- some years quarterly, some years annual.
                   Annual rows are forward-filled across the four quarters
                   of that year, and a companion boolean flag column
                   (unemployment_is_interpolated) marks imputed values.

Output
------
  A single pd.DataFrame with index "YYYY-Qn" strings and columns:
    repo_rate | cpi | gdp | unemployment | unemployment_is_interpolated
"""

import pandas as pd
import numpy as np
from typing import Literal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUARTER_START = "2010Q1"
QUARTER_END   = "2025Q4"

# Canonical quarterly DatetimeIndex (Quarter-Start anchored)
CANONICAL_QS_INDEX: pd.DatetimeIndex = pd.period_range(
    start=QUARTER_START, end=QUARTER_END, freq="Q"
).to_timestamp(how="start")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_quarter_start(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Snap any DatetimeIndex to Quarter-Start timestamps."""
    return index.to_period("Q").to_timestamp(how="start")


def _validate_frame(df: pd.DataFrame, name: str, col: str) -> None:
    """Basic sanity checks on an input DataFrame."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"'{name}': index must be a DatetimeIndex, "
            f"got {type(df.index).__name__}."
        )
    if col not in df.columns:
        raise ValueError(f"'{name}': expected column '{col}' not found. "
                         f"Available: {df.columns.tolist()}")


def _align_to_canonical(
    series: pd.Series,
    name: str,
) -> pd.Series:
    """
    Reindex a quarterly Series onto CANONICAL_QS_INDEX.

    - Snaps the input index to Quarter-Start.
    - Warns about duplicates (keeps last).
    - Forward-fills up to 1 quarter for minor gaps; leaves larger gaps NaN.
    """
    s = series.copy()
    s.index = _to_quarter_start(s.index)
    s = s[~s.index.duplicated(keep="last")]            # drop dupe quarters
    s = s.reindex(CANONICAL_QS_INDEX)                  # align to canon grid

    n_missing = s.isna().sum()
    if n_missing:
        print(f"  [merge] '{name}': {n_missing} quarter(s) missing after "
              "reindex -- forward-filled (limit=1).")
    s = s.ffill(limit=1)                               # mild gap fill only
    return s


# ---------------------------------------------------------------------------
# Unemployment: mixed-frequency handler
# ---------------------------------------------------------------------------

def _expand_unemployment(
    df_unemp: pd.DataFrame,
    col: str = "unemployment",
) -> pd.DataFrame:
    """
    Expand a mixed-frequency unemployment DataFrame onto CANONICAL_QS_INDEX.

    Logic
    -----
    1.  Snap all timestamps to Quarter-Start.
    2.  For years where ONLY ONE observation exists (assumed annual):
          - Assign it to Q1 of that year.
          - Forward-fill to Q2, Q3, Q4 within the same year.
          - Mark Q2–Q4 as interpolated.
    3.  For years where 2–4 observations exist (assumed quarterly):
          - Reindex normally; any missing quarter within the year is
            forward-filled from the previous quarter within the same year
            and flagged as interpolated.
    4.  Reindex to the full canonical grid; remaining NaN rows get
        ``interpolated=False`` (they are truly absent, not imputed).

    Returns
    -------
    pd.DataFrame with columns [col, f"{col}_is_interpolated"].
    """
    _validate_frame(df_unemp, "unemployment", col)

    s = df_unemp[col].copy()
    s.index = _to_quarter_start(s.index)
    s = s[~s.index.duplicated(keep="last")]

    # Build output containers
    values: dict[pd.Timestamp, float] = {}
    flags:  dict[pd.Timestamp, bool]  = {}

    # Group by calendar year
    for year, grp in s.groupby(s.index.year):
        obs_count = len(grp)

        if obs_count == 1:
            # --- Annual observation: spread across 4 quarters ---------------
            annual_val = grp.iloc[0]
            q_dates = pd.period_range(
                start=f"{year}Q1", end=f"{year}Q4", freq="Q"
            ).to_timestamp(how="start")

            for i, qdate in enumerate(q_dates):
                values[qdate] = annual_val
                flags[qdate]  = (i > 0)          # Q1 = actual, Q2-Q4 = imputed

        else:
            # --- Quarterly (or partial): fill gaps within year only ----------
            q_dates = pd.period_range(
                start=f"{year}Q1", end=f"{year}Q4", freq="Q"
            ).to_timestamp(how="start")

            year_series = grp.reindex(q_dates)
            # Forward-fill within the year; backward-fill only if Q1 missing
            year_filled = year_series.ffill().bfill()

            for qdate in q_dates:
                orig_val   = year_series.get(qdate, np.nan)
                filled_val = year_filled.get(qdate, np.nan)
                was_filled = pd.isna(orig_val) and not pd.isna(filled_val)
                values[qdate] = filled_val
                flags[qdate]  = bool(was_filled)

    # Assemble into a DataFrame; use explicit bool dtype to avoid
    # FutureWarning from fillna on object-dtype boolean columns.
    result = pd.DataFrame(
        {
            col: pd.array(list(values.values()), dtype="Float64"),
            f"{col}_is_interpolated": pd.array(list(flags.values()), dtype="boolean"),
        },
        index=pd.DatetimeIndex(list(values.keys())),
    )
    result = result.reindex(CANONICAL_QS_INDEX)

    # Rows genuinely absent from source get flag=False (not imputed)
    result[f"{col}_is_interpolated"] = (
        result[f"{col}_is_interpolated"].fillna(False).astype(bool)
    )

    n_imputed = result[f"{col}_is_interpolated"].sum()
    n_actual  = result[col].notna().sum() - n_imputed
    n_absent  = result[col].isna().sum()
    print(
        f"  [merge] unemployment: {n_actual} actual quarters, "
        f"{n_imputed} interpolated, {n_absent} absent (NaN)."
    )
    return result


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------

def merge_quarterly_indicators(
    repo_rate:    pd.DataFrame,
    cpi:          pd.DataFrame,
    gdp:          pd.DataFrame,
    unemployment: pd.DataFrame,
    cols: dict[str, str] | None = None,
    sort_index: bool = True,
) -> pd.DataFrame:
    """
    Merge repo_rate, cpi, gdp, and unemployment into one quarterly DataFrame.

    Parameters
    ----------
    repo_rate, cpi, gdp, unemployment : pd.DataFrame
        Each must have a DatetimeIndex and at least the value column named
        in `cols`.
    cols : dict, optional
        Maps DataFrame name -> column to extract.  Defaults to:
          {"repo_rate": "repo_rate", "cpi": "cpi",
           "gdp": "gdp", "unemployment": "unemployment"}
    sort_index : bool
        If True, sort the final DataFrame by quarter ascending.

    Returns
    -------
    pd.DataFrame
        Quarterly DataFrame, indexed by "YYYY-Qn" strings, with columns:
        [repo_rate, cpi, gdp, unemployment, unemployment_is_interpolated]
    """
    _cols = {
        "repo_rate":    "repo_rate",
        "cpi":          "cpi",
        "gdp":          "gdp",
        "unemployment": "unemployment",
    }
    if cols:
        _cols.update(cols)

    print("[merge] Starting quarterly merge ...")
    print(f"        Canonical range : {QUARTER_START} -> {QUARTER_END} "
          f"({len(CANONICAL_QS_INDEX)} quarters)")

    # -- Validate simple quarterly inputs ------------------------------------
    for name, df in [("repo_rate", repo_rate), ("cpi", cpi), ("gdp", gdp)]:
        _validate_frame(df, name, _cols[name])

    # -- Align simple quarterly series --------------------------------------
    aligned: dict[str, pd.Series] = {}
    for name, df in [("repo_rate", repo_rate), ("cpi", cpi), ("gdp", gdp)]:
        aligned[name] = _align_to_canonical(df[_cols[name]], name)

    # -- Handle mixed-frequency unemployment --------------------------------
    unemp_df = _expand_unemployment(unemployment, col=_cols["unemployment"])

    # -- Combine everything -------------------------------------------------
    merged = pd.DataFrame(aligned, index=CANONICAL_QS_INDEX)
    merged = merged.join(unemp_df, how="left")

    if sort_index:
        merged.sort_index(inplace=True)

    # -- Format index as "YYYY-Qn" strings ----------------------------------
    merged.index = merged.index.to_period("Q").strftime("%Y-Q%q")
    merged.index.name = "Quarter"

    print(
        f"[merge] Done. Output shape: {merged.shape}  "
        f"| NaN counts:\n{merged.isna().sum().to_string()}"
    )
    return merged


# ---------------------------------------------------------------------------
# Sample data builders (for demo / testing)
# ---------------------------------------------------------------------------

def _make_quarterly_df(
    col: str,
    start: str = "2010-01-01",
    periods: int = 64,
    seed: int = 0,
    base: float = 5.0,
    noise: float = 0.3,
) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.date_range(start, periods=periods, freq="QS")
    vals = base + np.cumsum(np.random.normal(0, noise, periods))
    return pd.DataFrame({col: vals}, index=idx)


def _make_mixed_unemployment(seed: int = 42) -> pd.DataFrame:
    """
    Simulate a realistic mixed-frequency unemployment DataFrame:
      - 2010-2014 : annual (one row per year, date = Jan-01)
      - 2015-2019 : quarterly
      - 2020-2025 : quarterly (with one entirely missing year: 2022)
    """
    np.random.seed(seed)
    rows: list[dict] = []

    # Annual block: 2010-2014
    for year in range(2010, 2015):
        rows.append({
            "date":         pd.Timestamp(f"{year}-01-01"),
            "unemployment": round(6.0 + np.random.normal(0, 0.4), 2),
        })

    # Quarterly block: 2015-2021
    for period in pd.period_range("2015Q1", "2021Q4", freq="Q"):
        rows.append({
            "date":         period.to_timestamp(how="start"),
            "unemployment": round(5.0 + np.random.normal(0, 0.6), 2),
        })

    # 2022 entirely absent (simulate data gap)

    # Quarterly block: 2023-2025
    for period in pd.period_range("2023Q1", "2025Q4", freq="Q"):
        rows.append({
            "date":         period.to_timestamp(how="start"),
            "unemployment": round(4.5 + np.random.normal(0, 0.5), 2),
        })

    df = pd.DataFrame(rows).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -- Build sample inputs -------------------------------------------------
    repo_rate_df    = _make_quarterly_df("repo_rate",    seed=1, base=6.0, noise=0.2)
    cpi_df          = _make_quarterly_df("cpi",          seed=2, base=4.5, noise=0.3)
    gdp_df          = _make_quarterly_df("gdp",          seed=3, base=7.0, noise=0.5)
    unemployment_df = _make_mixed_unemployment(seed=42)

    print("-- Unemployment source (first 12 rows) ---------------------------")
    print(unemployment_df.head(12).to_string())
    print()

    # -- Merge ---------------------------------------------------------------
    master = merge_quarterly_indicators(
        repo_rate    = repo_rate_df,
        cpi          = cpi_df,
        gdp          = gdp_df,
        unemployment = unemployment_df,
    )

    print()
    print("-- Merged quarterly master DataFrame -----------------------------")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(master.round(4).to_string())

    # -- Interpolation summary -----------------------------------------------
    print()
    print("-- Unemployment flag summary -------------------------------------")
    flag_col = "unemployment_is_interpolated"
    summary = master.groupby(flag_col)["unemployment"].count().rename("count")
    print(summary.to_string())

    # Quick spot-check: show the annual-expansion block (2010-2014)
    print()
    print("-- Spot-check: 2010-2014 (annual -> quarterly expansion) ---------")
    print(master.loc["2010-Q1":"2014-Q4", ["unemployment", flag_col]].to_string())

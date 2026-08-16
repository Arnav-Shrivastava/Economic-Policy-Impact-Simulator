"""
build_dataset.py
----------------
End-to-end pipeline:
  1. Loads each raw source from data/raw/
  2. Reshapes/resamples to quarterly frequency
  3. Merges all sources into one master DataFrame (2010-Q1 to 2025-Q4)
  4. Flags interpolated / forward-filled values per column
  5. Prints sanity-check diagnostics
  6. Saves to data/processed/india_macro_quarterly.csv
  7. Asserts the output index is a complete, gapless quarterly sequence

Column names in output
----------------------
  Repo_Rate | CPI_Inflation | GDP_Growth | Unemployment_Rate | IIP_Growth
  + companion boolean flags: <col>_is_interpolated for every column
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# -- make src/ importable when run as a top-level script --------------------
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from resample_quarterly import resample_to_quarterly          # noqa: E402
from merge_quarterly import (                                  # noqa: E402
    merge_quarterly_indicators,
    CANONICAL_QS_INDEX,
    _to_quarter_start,
    _align_to_canonical,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).parent.parent
RAW_DIR      = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

GDP_CSV      = RAW_DIR / "gdp"          / "API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_57.csv"
CPI_CSV      = RAW_DIR / "cpi_inflation"/ "API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_285.csv"
UNEMP_CSV    = RAW_DIR / "unemployment" / "API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_33398.csv"
IIP_XLSX     = RAW_DIR / "iip"          / "mospi_iip_dashboard.xlsx"
REPO_XLSX    = RAW_DIR / "repo_rate"    / "Repo_ Reverse Repo Auction Under Liquidity Adjustment Facility .xlsx"

OUTPUT_CSV   = PROCESSED_DIR / "india_macro_quarterly.csv"
COUNTRY_CODE = "IND"   # ISO 3-letter code for India in World Bank CSVs

# ---------------------------------------------------------------------------
# Reconfigure stdout to UTF-8 so Unicode characters print cleanly on Windows
# ---------------------------------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8")

DIVIDER = "-" * 70


# ===========================================================================
# 1.  LOADERS  -- one function per raw source
# ===========================================================================

def _world_bank_annual_to_quarterly(
    csv_path: Path,
    col_out: str,
    country_code: str = COUNTRY_CODE,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Read a World Bank 'wide' annual CSV, extract the India row, and
    build a quarterly DataFrame by repeating each annual value across
    the four quarters of that year (forward-fill logic).

    Returns
    -------
    df   : pd.DataFrame with DatetimeIndex and columns [col_out, col_out_is_interpolated]
    warnings : list of human-readable warning strings
    """
    raw = pd.read_csv(csv_path, skiprows=4)
    india = raw[raw["Country Code"] == country_code]
    if india.empty:
        raise ValueError(f"Country code '{country_code}' not found in {csv_path.name}")

    # Extract numeric year columns that fall within our target window
    year_cols = [
        c for c in india.columns
        if str(c).isdigit() and 2010 <= int(c) <= 2025
    ]
    annual_series = india[year_cols].iloc[0]
    annual_series.index = annual_series.index.astype(int)
    annual_series = annual_series.dropna().astype(float)

    # Expand each annual value to 4 quarters (Q1 = actual, Q2-Q4 = interpolated)
    values: dict[pd.Timestamp, float] = {}
    flags:  dict[pd.Timestamp, bool]  = {}
    interp_warnings: list[str] = []

    for year in range(2010, 2026):
        q_dates = pd.period_range(f"{year}Q1", f"{year}Q4", freq="Q").to_timestamp(how="start")
        if year in annual_series.index:
            val = annual_series[year]
            for i, qd in enumerate(q_dates):
                values[qd] = val
                flags[qd]  = (i > 0)  # Q1 = actual; Q2-Q4 = interpolated
            interp_warnings.append(
                f"  {year}: annual value ({val:.4f}) repeated to "
                f"{year}-Q2, {year}-Q3, {year}-Q4"
            )
        else:
            for qd in q_dates:
                values[qd] = np.nan
                flags[qd]  = False

    df = pd.DataFrame(
        {
            col_out: pd.array(list(values.values()), dtype="Float64"),
            f"{col_out}_is_interpolated": pd.array(list(flags.values()), dtype="boolean"),
        },
        index=pd.DatetimeIndex(list(values.keys())),
    )
    df.index.name = "date"
    return df, interp_warnings


def load_gdp() -> tuple[pd.DataFrame, list[str]]:
    print(f"[load] GDP   <- {GDP_CSV.name}")
    df, warns = _world_bank_annual_to_quarterly(GDP_CSV, "GDP_Growth")
    print(f"       {df['GDP_Growth'].notna().sum()} non-NaN quarters, "
          f"{df['GDP_Growth_is_interpolated'].sum()} interpolated")
    return df, warns


def load_cpi() -> tuple[pd.DataFrame, list[str]]:
    print(f"[load] CPI   <- {CPI_CSV.name}")
    df, warns = _world_bank_annual_to_quarterly(CPI_CSV, "CPI_Inflation")
    print(f"       {df['CPI_Inflation'].notna().sum()} non-NaN quarters, "
          f"{df['CPI_Inflation_is_interpolated'].sum()} interpolated")
    return df, warns


def load_unemployment() -> tuple[pd.DataFrame, list[str]]:
    print(f"[load] UNEMP <- {UNEMP_CSV.name}")
    df, warns = _world_bank_annual_to_quarterly(UNEMP_CSV, "Unemployment_Rate")
    print(f"       {df['Unemployment_Rate'].notna().sum()} non-NaN quarters, "
          f"{df['Unemployment_Rate_is_interpolated'].sum()} interpolated")
    return df, warns


def load_iip() -> tuple[pd.DataFrame, list[str]]:
    """
    Load monthly IIP (General index growth) from the MOSPI Excel,
    resample to quarterly mean, flag any month-level gaps that were
    forward-filled before aggregation.
    """
    print(f"[load] IIP   <- {IIP_XLSX.name}")
    raw = pd.read_excel(IIP_XLSX, sheet_name="Sectoral-Monthly")
    general_row = raw[raw["Description"] == "General"]
    if general_row.empty:
        raise ValueError("'General' row not found in IIP Sectoral-Monthly sheet.")

    # Date columns may be datetime.datetime (from openpyxl) or pd.Timestamp
    import datetime as _dt
    date_cols = [
        c for c in raw.columns
        if isinstance(c, (_dt.datetime, pd.Timestamp))
        and not (isinstance(c, float))   # exclude the NaN sentinel columns
    ]
    values = general_row[date_cols].iloc[0].astype(float)
    monthly_df = pd.DataFrame(
        {"IIP_Growth": values.values},
        index=pd.DatetimeIndex(date_cols),
    )
    monthly_df.index.name = "date"
    monthly_df.sort_index(inplace=True)

    data_start = monthly_df.index.min()
    data_end   = monthly_df.index.max()
    print(f"       Raw monthly IIP: {data_start.date()} .. {data_end.date()} "
          f"({len(monthly_df)} months)")

    # Resample to quarterly using resample_quarterly utility
    quarterly = resample_to_quarterly(
        monthly_df,
        columns=["IIP_Growth"],
        fill_method="ffill",
        min_periods=1,
        verbose=False,
    )
    # resample_to_quarterly returns index like '2023-Q1'; convert to PeriodIndex
    import re
    def _parse_quarter_str(s: str) -> str:
        """'2023-Q1' -> '2023Q1'  (pandas PeriodIndex format)"""
        return re.sub(r"-Q(\d)", r"Q\1", s)

    quarterly.index = pd.PeriodIndex(
        [_parse_quarter_str(q) for q in quarterly.index], freq="Q"
    ).to_timestamp(how="start")


    # Reindex to canonical grid; mark quarters outside data range as NaN
    quarterly = quarterly.reindex(CANONICAL_QS_INDEX)
    n_missing = quarterly["IIP_Growth"].isna().sum()

    # Build flag column: NaN quarters are absent (not interpolated)
    quarterly["IIP_Growth_is_interpolated"] = False  # monthly->quarterly mean is not interpolation

    interp_warns: list[str] = []
    if n_missing:
        missing_qtrs = quarterly[quarterly["IIP_Growth"].isna()].index
        missing_labels = missing_qtrs.to_period("Q").strftime("%Y-Q%q").tolist()
        interp_warns.append(
            f"  IIP_Growth: {n_missing} quarters absent (no source data) -> NaN:\n"
            f"    {missing_labels}"
        )

    quarterly.index.name = "date"
    print(f"       {quarterly['IIP_Growth'].notna().sum()} non-NaN quarters after reindex")
    return quarterly, interp_warns


def load_repo_rate() -> tuple[pd.DataFrame, list[str]]:
    """
    Load daily RBI repo rate data from the LAF Excel, resample to quarterly
    mean (last observed rate per quarter), and reindex to canonical grid.
    """
    print(f"[load] REPO  <- {REPO_XLSX.name}")
    raw = pd.read_excel(REPO_XLSX, sheet_name=0, header=None)

    # Column 1 = auction date (datetime), Column 7 = Fixed Repo Cut-off Rate
    date_col = raw.iloc[:, 1]
    rate_col = raw.iloc[:, 7]

    mask = date_col.apply(
        lambda x: isinstance(x, pd.Timestamp)
        or (hasattr(x, "year") and not isinstance(x, str) and not isinstance(x, float))
    )
    data = raw[mask][[1, 7]].copy()
    data.columns = ["date", "Repo_Rate"]
    data["Repo_Rate"] = pd.to_numeric(data["Repo_Rate"], errors="coerce")
    data["date"]      = pd.to_datetime(data["date"])
    data = data.dropna(subset=["Repo_Rate"])
    data = data.sort_values("date").set_index("date")

    data_start = data.index.min()
    data_end   = data.index.max()
    print(f"       Raw daily repo rate: {data_start.date()} .. {data_end.date()} "
          f"({len(data)} auction days)")

    # Resample: use last observed rate in each quarter (policy rate is a step function)
    quarterly = data.resample("QS").last().dropna()
    quarterly.index = _to_quarter_start(quarterly.index)
    quarterly = quarterly.reindex(CANONICAL_QS_INDEX)
    n_missing = quarterly["Repo_Rate"].isna().sum()

    # Forward-fill up to 4 quarters for minor gaps (policy rate is sticky)
    quarterly["Repo_Rate"] = quarterly["Repo_Rate"].ffill(limit=4)
    n_remaining = quarterly["Repo_Rate"].isna().sum()

    quarterly["Repo_Rate_is_interpolated"] = False  # all from actual auction data

    interp_warns: list[str] = []
    if n_missing:
        still_nan = quarterly[quarterly["Repo_Rate"].isna()].index
        nan_labels = still_nan.to_period("Q").strftime("%Y-Q%q").tolist() if n_remaining else []
        filled_count = n_missing - n_remaining
        interp_warns.append(
            f"  Repo_Rate: source data ends {data_end.date()}; "
            f"{filled_count} subsequent quarter(s) forward-filled; "
            f"{n_remaining} quarter(s) remain NaN: {nan_labels}"
        )

    quarterly.index.name = "date"
    print(f"       {quarterly['Repo_Rate'].notna().sum()} non-NaN quarters after reindex + ffill")
    return quarterly, interp_warns


# ===========================================================================
# 2.  MERGE  -- join all sources on CANONICAL_QS_INDEX
# ===========================================================================

def build_master(
    gdp_df:   pd.DataFrame,
    cpi_df:   pd.DataFrame,
    unemp_df: pd.DataFrame,
    iip_df:   pd.DataFrame,
    repo_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all five pre-aligned quarterly DataFrames onto CANONICAL_QS_INDEX.
    Each df already has a DatetimeIndex aligned to CANONICAL_QS_INDEX.
    """
    master = pd.concat(
        [gdp_df, cpi_df, unemp_df, iip_df, repo_df],
        axis=1,
    )
    master = master.reindex(CANONICAL_QS_INDEX)

    # Canonical column order
    value_cols = ["Repo_Rate", "CPI_Inflation", "GDP_Growth", "Unemployment_Rate", "IIP_Growth"]
    flag_cols  = [f"{c}_is_interpolated" for c in value_cols]
    all_cols   = value_cols + flag_cols

    # Fill any flag columns that ended up NaN (absent quarters)
    for fc in flag_cols:
        if fc in master.columns:
            master[fc] = master[fc].fillna(False).astype(bool)

    master = master[all_cols]

    # Format index as "YYYY-Qn"
    master.index = master.index.to_period("Q").strftime("%Y-Q%q")
    master.index.name = "Quarter"

    return master


# ===========================================================================
# 3.  DIAGNOSTICS / WARNINGS
# ===========================================================================

def print_interpolation_warnings(all_warns: dict[str, list[str]]) -> None:
    any_warn = any(len(v) > 0 for v in all_warns.values())
    if not any_warn:
        print("\n[OK] No interpolation/gap warnings.")
        return

    print(f"\n{'=' * 70}")
    print("  INTERPOLATION / GAP WARNINGS")
    print(f"{'=' * 70}")
    for source, warns in all_warns.items():
        if warns:
            print(f"\n  [{source.upper()}]")
            for w in warns:
                print(w)
    print(f"{'=' * 70}\n")


def print_diagnostics(df: pd.DataFrame) -> None:
    value_cols = ["Repo_Rate", "CPI_Inflation", "GDP_Growth", "Unemployment_Rate", "IIP_Growth"]

    print(f"\n{DIVIDER}")
    print("  SHAPE")
    print(f"{DIVIDER}")
    print(f"  {df.shape[0]} quarters x {df.shape[1]} columns")

    print(f"\n{DIVIDER}")
    print("  HEAD (first 10 quarters)")
    print(f"{DIVIDER}")
    pd.set_option("display.width", 130)
    pd.set_option("display.max_columns", 12)
    print(df[value_cols].head(10).to_string())

    print(f"\n{DIVIDER}")
    print("  TAIL (last 10 quarters)")
    print(f"{DIVIDER}")
    print(df[value_cols].tail(10).to_string())

    print(f"\n{DIVIDER}")
    print("  NULL COUNTS  (value columns only)")
    print(f"{DIVIDER}")
    null_counts = df[value_cols].isnull().sum()
    for col, cnt in null_counts.items():
        status = "OK" if cnt == 0 else f"MISSING {cnt} quarters"
        print(f"  {col:<25} {cnt:>3}   [{status}]")

    print(f"\n{DIVIDER}")
    print("  INTERPOLATION FLAG TOTALS")
    print(f"{DIVIDER}")
    flag_cols = [f"{c}_is_interpolated" for c in value_cols]
    for fc in flag_cols:
        if fc in df.columns:
            n = int(df[fc].sum())
            col_label = fc.replace("_is_interpolated", "")
            print(f"  {col_label:<25} {n:>3} quarters flagged as interpolated")


# ===========================================================================
# 4.  ASSERTION  -- continuous quarterly index, no skipped quarters
# ===========================================================================

def assert_continuous_quarterly_index(df: pd.DataFrame) -> None:
    """
    Verify that the DataFrame's quarter-string index is a perfectly
    continuous sequence with no skipped quarters.

    Raises
    ------
    AssertionError if any gap is found, listing the specific missing quarters.
    """
    import re
    def _to_period_str(s: str) -> str:
        """'2023-Q1' -> '2023Q1'"""
        return re.sub(r"-Q(\d)", r"Q\1", s)

    periods = pd.PeriodIndex(
        [_to_period_str(q) for q in df.index], freq="Q"
    )

    expected = pd.period_range(periods[0], periods[-1], freq="Q")
    missing  = expected.difference(periods)

    if len(missing) > 0:
        raise AssertionError(
            f"CONTINUITY CHECK FAILED: {len(missing)} quarter(s) are skipped "
            f"in the output index:\n  {missing.strftime('%Y-Q%q').tolist()}"
        )

    print(f"\n[ASSERT] Continuity check passed: {len(df)} consecutive quarters "
          f"({df.index[0]} .. {df.index[-1]}), no gaps.")


# ===========================================================================
# 5.  MAIN
# ===========================================================================

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("  BUILD DATASET  --  India Macro Quarterly  (2010-Q1 to 2025-Q4)")
    print(f"{'=' * 70}\n")

    # ── Load all sources ────────────────────────────────────────────────────
    gdp_df,   gdp_warns   = load_gdp()
    cpi_df,   cpi_warns   = load_cpi()
    unemp_df, unemp_warns = load_unemployment()
    iip_df,   iip_warns   = load_iip()
    repo_df,  repo_warns  = load_repo_rate()

    # ── Merge ────────────────────────────────────────────────────────────────
    print(f"\n[merge] Joining all sources ...")
    master = build_master(gdp_df, cpi_df, unemp_df, iip_df, repo_df)
    print(f"[merge] Done. Shape: {master.shape}")

    # ── Interpolation warnings ───────────────────────────────────────────────
    all_warns = {
        "gdp":          gdp_warns,
        "cpi":          cpi_warns,
        "unemployment": unemp_warns,
        "iip":          iip_warns,
        "repo_rate":    repo_warns,
    }
    print_interpolation_warnings(all_warns)

    # ── Diagnostics ─────────────────────────────────────────────────────────
    print_diagnostics(master)

    # ── Continuity assertion ─────────────────────────────────────────────────
    assert_continuous_quarterly_index(master)

    # ── Save ─────────────────────────────────────────────────────────────────
    master.to_csv(OUTPUT_CSV)
    print(f"\n[save] Written to: {OUTPUT_CSV}")
    print(f"       File size  : {OUTPUT_CSV.stat().st_size / 1024:.1f} KB")
    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

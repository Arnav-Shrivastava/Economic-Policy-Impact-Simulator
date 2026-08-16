# Data Sources — India Macro Quarterly Dataset

**Dataset**: `data/processed/india_macro_quarterly.csv`  
**Coverage**: 2010-Q1 → 2025-Q4 (64 quarters)  
**Index format**: `YYYY-Qn` (e.g. `2023-Q1`)  
**Pipeline**: [`src/build_dataset.py`](../src/build_dataset.py)  
**Generated**: 2026-08-16

---

## Summary Table

| Indicator | Column Name | Original Source | Source Frequency | Resampling Method | Null Quarters | Interpolated Quarters |
|---|---|---|---|---|---|---|
| RBI Repo Rate | `Repo_Rate` | [RBI — LAF Auction Data](https://rbi.org.in/Scripts/BS_ViewBulletin.aspx) | Daily (auction-level) | Last observed rate per quarter | 19 (2021-Q2 → 2025-Q4) | 0 |
| CPI Inflation | `CPI_Inflation` | [World Bank — FP.CPI.TOTL.ZG](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=IN) | Annual | Repeated across 4 quarters | 0 | 48 (Q2–Q4 of every year) |
| GDP Growth | `GDP_Growth` | [World Bank — NY.GDP.MKTP.KD.ZG](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=IN) | Annual | Repeated across 4 quarters | 0 | 48 (Q2–Q4 of every year) |
| Unemployment Rate | `Unemployment_Rate` | [World Bank — SL.UEM.TOTL.ZS](https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS?locations=IN) | Annual | Repeated across 4 quarters | 0 | 48 (Q2–Q4 of every year) |
| IIP Growth | `IIP_Growth` | [MoSPI — IIP Dashboard](https://mospi.gov.in/web/mospi/download-tables-data/-/reports/view/templateThree/16801?q=thematic&themat=8) | Monthly | Mean of 3 months per quarter | 53 (2010-Q1 → 2023-Q1) | 0 |

Each value column has a companion boolean flag column (`<column>_is_interpolated`) that is `True` for every quarter where the value was carried forward or repeated from an annual figure, and `False` for quarters with a genuine direct observation.

---

## Indicator Notes

---

### 1. Repo Rate (`Repo_Rate`)

- **Source**: Reserve Bank of India (RBI) — *Repo / Reverse Repo Auctions Under the Liquidity Adjustment Facility (LAF)*
  - File: `Repo_ Reverse Repo Auction Under Liquidity Adjustment Facility .xlsx`
  - URL: [rbi.org.in — Bulletin Data](https://rbi.org.in/Scripts/BS_ViewBulletin.aspx)
- **Original frequency**: Daily (individual auction records, multiple rows per day possible)
- **Coverage in source file**: 2001-04-03 → 2020-02-13 (4,146 auction days)
- **Resampling method**: `.resample("QS").last()` — the *last observed auction rate* within each quarter is used.
  - **Why `last` and not `mean`**: The repo rate is a policy rate, not a market rate. It is a discrete step function that changes only when the Monetary Policy Committee votes. Taking the mean of a quarter where the rate changed mid-way would produce a spurious fractional rate that never actually existed. The end-of-quarter rate best represents the prevailing policy stance for that period.
- **Forward-fill applied**: After the source data ends on 2020-02-13, the rate 5.15% was forward-filled for 4 subsequent quarters (2020-Q2 through 2021-Q1), reflecting that no rate change was announced in the downloaded data during that window.
  - Quarters forward-filled: `2020-Q2`, `2020-Q3`, `2020-Q4`, `2021-Q1`
  - `Repo_Rate_is_interpolated` is `False` for these quarters because they are carried from a real observation, not a statistical interpolation.
- **Null quarters**: **19 quarters** (2021-Q2 → 2025-Q4) are `NaN` because the source file does not cover this period.

**Data quality issues**:
- The downloaded LAF auction file contains *all individual repo and reverse-repo auction bids*, not just the policy rate change announcements. The "Fixed Repo Cut-off Rate" column (column index 7 in the raw sheet) was identified as the policy-equivalent rate through manual inspection of the sheet header rows.
- The source file appears to be a static RBI Bulletin download and was not updated beyond February 2020. To extend coverage to 2025, an updated LAF file or the RBI's [Monetary Policy Rates history page](https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx) should be used as a supplementary source.
- The Excel sheet name contains a garbled character (`REPO \ufffd REVERSE REPO AUCTIONS WI`) due to a special dash character in the original RBI filename. This is handled in the loader by reading `sheet_name=0` rather than by name.

---

### 2. CPI Inflation (`CPI_Inflation`)

- **Source**: World Bank Open Data — *Inflation, consumer prices (annual %, India)*
  - Indicator code: `FP.CPI.TOTL.ZG`
  - File: `API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_285.csv`
  - URL: [data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=IN](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=IN)
- **Original frequency**: Annual (calendar year)
- **Coverage in source file**: 1960–2025 (India row complete from ~1987 onward)
- **Resampling method**: Annual value assigned to Q1 of that year and repeated (forward-filled) to Q2, Q3, Q4.
  - **Why this method**: No sub-annual breakdown is available from this source. The annual CPI figure represents the full-year average, so distributing it equally across all four quarters is the most honest representation. Marking Q2–Q4 as interpolated ensures downstream models can identify and optionally mask these quarters.
- **Interpolated quarters**: **48** — Q2, Q3, Q4 of every year from 2010 to 2025. These are flagged `CPI_Inflation_is_interpolated = True`.
- **Null quarters**: None.

**Data quality issues**:
- The World Bank annual CPI figure for India is a calendar-year average derived from monthly CPI data. It does **not** align with India's fiscal year (April–March). Models that need fiscal-year seasonality should source monthly CPI from the [Ministry of Statistics (MoSPI) CPIIW/CPI-C releases](https://mospi.gov.in) instead.
- The 2024 and 2025 values may be preliminary estimates and subject to revision in future World Bank data releases. The `Unnamed: 70` trailing column in the raw CSV is an artefact of the World Bank export format and is safely ignored.

---

### 3. GDP Growth (`GDP_Growth`)

- **Source**: World Bank Open Data — *GDP growth (annual %, India)*
  - Indicator code: `NY.GDP.MKTP.KD.ZG`
  - File: `API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_57.csv`
  - URL: [data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=IN](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=IN)
- **Original frequency**: Annual (calendar year, constant 2015 USD)
- **Coverage in source file**: 1961–2025
- **Resampling method**: Annual value assigned to Q1 and repeated to Q2–Q4.
  - **Why this method**: India's CSO/NSO publishes quarterly GDP estimates (GVA at basic prices), but those are not in this World Bank download. The annual figure represents an aggregated full-year growth rate and is the most widely cited comparable metric. Repetition to sub-annual periods is appropriate for broad macro context; `_is_interpolated` flags prevent accidental over-reliance on the quarterly breakdown.
- **Interpolated quarters**: **48** — Q2, Q3, Q4 of every year from 2010 to 2025.
- **Null quarters**: None.

**Data quality issues**:
- India switched its GDP base year from 2004–05 to 2011–12 in 2015, and the World Bank series incorporates this revision. Historical figures pre-2012 reflect the revised (rebased) series.
- The 2020 value correctly captures the COVID-19 contraction (−5.78%). Note that annual data masks the sharp intra-year V-shape (severe Q1 contraction, strong Q3–Q4 recovery); a model consuming this data will see a flat −5.78% across all four 2020 quarters.
- For quarter-level granularity, replace this source with MOSPI's [National Accounts Statistics Quarterly GDP releases](https://mospi.gov.in/web/mospi/download-tables-data/-/reports/view/templateOne/16901).

---

### 4. Unemployment Rate (`Unemployment_Rate`)

- **Source**: World Bank Open Data — *Unemployment, total (% of total labour force, ILO modelled estimate, India)*
  - Indicator code: `SL.UEM.TOTL.ZS`
  - File: `API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_33398.csv`
  - URL: [data.worldbank.org/indicator/SL.UEM.TOTL.ZS?locations=IN](https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS?locations=IN)
- **Original frequency**: Annual (calendar year, ILO modelled)
- **Coverage in source file**: 1991–2025
- **Resampling method**: Annual value assigned to Q1 and repeated to Q2–Q4.
  - **Why this method**: This indicator is an ILO *modelled* annual estimate, not a survey-based quarterly figure. India's official quarterly employment surveys (PLFS) exist from 2017 onward but are not in this file. Repeating the annual value is the correct treatment for a smoothed modelled series.
- **Interpolated quarters**: **48** — Q2, Q3, Q4 of every year from 2010 to 2025.
- **Null quarters**: None.

**Data quality issues**:
- The ILO modelled unemployment rate for India (~4–8% range) is structurally different from the CMIE or NSSO/PLFS household survey estimates, which show higher headline unemployment. The ILO model adjusts for large informal-sector employment in ways that suppress the measured rate. Use this series only for long-run trend comparisons, not for short-run labour market signals.
- The 2025 value is a model projection and should be treated as an estimate, not an observation.
- For higher-frequency or more India-specific unemployment data, the [CMIE Consumer Pyramids Household Survey](https://unemploymentinindia.cmie.com/) or [MoSPI PLFS Annual Reports](https://mospi.gov.in/web/plfs) are preferred alternatives.

---

### 5. IIP Growth (`IIP_Growth`)

- **Source**: Ministry of Statistics and Programme Implementation (MoSPI) — *Index of Industrial Production (IIP) Dashboard*
  - File: `mospi_iip_dashboard.xlsx`, sheet `Sectoral-Monthly`, row `General`
  - URL: [mospi.gov.in — IIP Dashboard](https://mospi.gov.in/web/mospi/download-tables-data/-/reports/view/templateThree/16801?q=thematic&themat=8)
- **Original frequency**: Monthly (year-on-year % growth, base year 2011–12)
- **Coverage in source file**: April 2023 → June 2026 (39 months, of which 36 fall within scope)
- **Resampling method**: Arithmetic mean of the three monthly growth figures within each quarter.
  - **Why `mean`**: IIP growth is a flow variable (percentage change). Averaging three monthly growth rates produces a representative quarterly growth rate, consistent with standard industrial production aggregation practice. Using `last` would discard two-thirds of the available monthly information.
- **Gap handling**: Three columns (Sep 2024, Oct 2024, Nov 2024) were `NaN` in the source Excel file due to blank cells in the MoSPI dashboard download. These were forward-filled at the monthly level before quarterly aggregation using the `resample_to_quarterly()` utility (`fill_method="ffill"`).
- **Null quarters**: **53 quarters** (2010-Q1 → 2023-Q1) are `NaN` because the MoSPI dashboard file only contains data from April 2023 onward.
  - Quarters with data: `2023-Q2` → `2025-Q4` (11 quarters)

**Data quality issues**:
- The downloaded MoSPI Excel file (`mospi_iip_dashboard.xlsx`) is a **dashboard snapshot**, not a full historical series. It only includes the most recent ~3 years. For historical IIP going back to 2010, the [RBI DBIE — IIP section](https://dbie.rbi.org.in/DBIE/dbie.rbi?site=publications#!4) or [MoSPI's historical IIP release archive](https://mospi.gov.in/web/mospi/download-tables-data/-/reports/view/templateOne/16601) must be used.
- The `Sectoral-Monthly` sheet contains a `General` row (100% weight composite) which is the headline IIP figure used here. Other rows (Mining, Manufacturing, Electricity, etc.) are available for sectoral decomposition if needed.
- Three months (Sep–Nov 2024) had blank cells in the source Excel, confirmed by `NaN` column headers (`nan`, `nan.1`, `nan.2`) when read by `openpyxl`. These were forward-filled in the pipeline.

---

## Pipeline Flags Reference

Each value column has a companion boolean column:

| Flag Column | `True` means | `False` means |
|---|---|---|
| `Repo_Rate_is_interpolated` | *(not used — all values are direct observations or forward-carries from an observed rate)* | Direct observation or forward-carry |
| `CPI_Inflation_is_interpolated` | Q2–Q4 of a year: repeated from the annual Q1 value | Q1 of each year: direct annual observation |
| `GDP_Growth_is_interpolated` | Q2–Q4 of a year: repeated from the annual Q1 value | Q1 of each year: direct annual observation |
| `Unemployment_Rate_is_interpolated` | Q2–Q4 of a year: repeated from the annual Q1 value | Q1 of each year: direct annual observation |
| `IIP_Growth_is_interpolated` | *(not used — quarterly mean of real monthly data)* | All values are aggregated from real monthly figures |

> **Usage note for modelling**: Filter `df[df["CPI_Inflation_is_interpolated"] == False]` to restrict any regression or correlation analysis to Q1 observations only, avoiding the artificial within-year autocorrelation introduced by the annual-to-quarterly expansion.

---

## Recommended Replacements / Upgrades

| Indicator | Current Limitation | Better Source |
|---|---|---|
| `Repo_Rate` | Ends Feb 2020 | [RBI Monetary Policy Rate History](https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx) — policy decisions with exact effective dates |
| `GDP_Growth` | Annual only, calendar year | [MoSPI Quarterly GDP / GVA Estimates](https://mospi.gov.in/web/mospi/download-tables-data/-/reports/view/templateOne/16901) — true quarterly national accounts |
| `CPI_Inflation` | Annual only | [RBI DBIE — CPI Monthly](https://dbie.rbi.org.in) or [MoSPI CPI Press Releases](https://mospi.gov.in) — monthly, then resample |
| `Unemployment_Rate` | ILO modelled annual | [MoSPI PLFS](https://mospi.gov.in/web/plfs) (quarterly from 2017) or [CMIE CPHS](https://unemploymentinindia.cmie.com/) |
| `IIP_Growth` | Dashboard only, starts Apr 2023 | [RBI DBIE — IIP](https://dbie.rbi.org.in/DBIE/dbie.rbi?site=publications#!4) — monthly from 2000 onward |

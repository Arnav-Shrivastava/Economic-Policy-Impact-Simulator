# Methodology — Economic Policy Impact Simulator

## Overview

This document describes the analytical methodology applied across all phases of the
Economic Policy Impact Simulator project. The dataset covers India's macroeconomic
indicators at a quarterly frequency from 2010-Q1 to 2025-Q4.

---

## Phase 1 — Data Collection & Pipeline

**Script**: [`src/build_dataset.py`](../src/build_dataset.py)

Five macro indicators were sourced from public databases and resampled to quarterly frequency:

| Indicator | Source | Frequency | Resampling |
|-----------|--------|-----------|------------|
| Repo Rate | RBI LAF Auction Data | Daily | Last rate per quarter |
| CPI Inflation | World Bank (FP.CPI.TOTL.ZG) | Annual | Q1 direct; Q2–Q4 repeated |
| GDP Growth | World Bank (NY.GDP.MKTP.KD.ZG) | Annual | Q1 direct; Q2–Q4 repeated |
| Unemployment Rate | World Bank (SL.UEM.TOTL.ZS) | Annual | Q1 direct; Q2–Q4 repeated |
| IIP Growth | MoSPI IIP Dashboard | Monthly | Mean of 3 months per quarter |

See [`docs/data_sources.md`](data_sources.md) for full source documentation.

---

## Phase 2 — Exploratory Data Analysis

**Notebook**: [`notebooks/02_eda.ipynb`](../notebooks/02_eda.ipynb)

### Descriptive Statistics Summary

| Indicator | N | Mean | Std Dev | Min | Max | Skewness | Kurtosis |
|-----------|---|------|---------|-----|-----|----------|----------|
| Repo_Rate | 45 | 6.67 | 1.07 | 5.00 | 8.50 | +0.06 | −1.18 |
| CPI_Inflation | 64 | 6.21 | 2.61 | 2.40 | 11.99 | +0.70 | −0.35 |
| GDP_Growth | 64 | 6.24 | 3.41 | −5.78 | 9.69 | −2.77 | +7.77 |
| Unemployment_Rate | 64 | 6.68 | 1.42 | 4.17 | 7.86 | −0.99 | −0.81 |
| IIP_Growth | 11 | 5.45 | 2.84 | 3.40 | 13.27 | +2.43 | +6.58 |

**Notes**:
- GDP_Growth is strongly left-skewed (skew = −2.77) with high kurtosis (+7.77) due to the COVID-19 contraction in 2020 (−5.78%), which is a 3σ outlier. A binary COVID dummy variable is recommended for time-series models.
- IIP_Growth has only n = 11 observations — treat all results for this indicator with caution.

---

## Stationarity Analysis — ADF Results & VAR Implications

**Augmented Dickey-Fuller (ADF) Test** — H₀: unit root present (non-stationary). Reject H₀ at p ≤ 0.05.

### Level Series Results

| Indicator | Regression | N | ADF Statistic | p-value | Crit. 5% | Stationary at Level? |
|-----------|-----------|---|--------------|---------|----------|----------------------|
| Repo_Rate | ct | 38 | −2.8713 | 0.1720 | −3.5331 | ❌ **NO** |
| CPI_Inflation | ct | 63 | −2.1841 | 0.4988 | −3.4826 | ❌ **NO** |
| GDP_Growth | c | 59 | −2.3569 | 0.1543 | −2.9119 | ❌ **NO** |
| Unemployment_Rate | ct | 63 | −1.9230 | 0.6427 | −3.4826 | ❌ **NO** |
| IIP_Growth | c | 10 | −2.2933 | 0.1742 | −3.2330 | ❌ **NO** |

> Regression: `c` = constant only; `ct` = constant + trend

### First-Differenced Series Results

| Indicator | N | ADF Statistic (Δ) | p-value (Δ) | Stationary After Diff? | Integration Order |
|-----------|---|------------------|------------|------------------------|-------------------|
| d_Repo_Rate | 38 | −5.1780 | 0.0000 | ✅ **YES** | **I(1)** |
| d_CPI_Inflation | 62 | −7.9937 | 0.0000 | ✅ **YES** | **I(1)** |
| d_GDP_Growth | 59 | −5.9097 | 0.0000 | ✅ **YES** | **I(1)** |
| d_Unemployment_Rate | 62 | −7.9304 | 0.0000 | ✅ **YES** | **I(1)** |
| d_IIP_Growth | 9 | −3.3718 | 0.0120 | ✅ **YES** | **I(1)** ⚠️ |

> ⚠️ IIP_Growth result is borderline reliable due to very small sample (n=9 after differencing).

### Conclusion: All Series Are I(1)

**Every indicator is integrated of order 1** — non-stationary at level, stationary after first differencing.

---

### VAR Implications (Phase 4)

| Indicator | VAR Column | Treatment |
|-----------|-----------|-----------|
| Repo_Rate | `d_Repo_Rate` | First difference ⚠️ 19-quarter gap post-2021 |
| CPI_Inflation | `d_CPI_Inflation` | First difference |
| GDP_Growth | `d_GDP_Growth` | First difference + COVID dummy recommended |
| Unemployment_Rate | `d_Unemployment_Rate` | First difference |
| IIP_Growth | `d_IIP_Growth` | First difference (consider excluding; n=11) |

**Cointegration check**: Because all series are I(1), run the **Johansen cointegration test** in Phase 4.
- If ≥ 1 cointegrating vector is detected → use **VECM** (Vector Error Correction Model)
- If no cointegration → use **VAR in differences** on the `d_<col>` columns

**Recommended approach for Phase 4**:
1. Use `d_CPI_Inflation`, `d_GDP_Growth`, `d_Unemployment_Rate` as the core VAR variables (complete coverage)
2. Add `d_Repo_Rate` as exogenous regressor (due to the 19-quarter gap)
3. Exclude `d_IIP_Growth` from the main VAR unless the historical IIP series is backfilled

---

## Phase 3 — ARIMA Forecasting

**Module**: [`src/model_arima.py`](../src/model_arima.py)  
**Notebook**: [`notebooks/03_arima.ipynb`](../notebooks/03_arima.ipynb)  
**Results**: [`models/arima_results.json`](../models/arima_results.json)

ARIMA (AutoRegressive Integrated Moving Average) was applied to forecast `GDP_Growth`.

- **Order selection**: `pmdarima.auto_arima` with AIC minimisation, stepwise search, p ∈ [0,5], q ∈ [0,5]
- **Train/test split**: Chronological 80/20 (no random shuffle — would cause data leakage)
- **Evaluation**: RMSE (root mean squared error) and MAPE (mean absolute percentage error)
- **Note on differencing**: ARIMA handles I(1) series internally via the *d* parameter; the raw `GDP_Growth` series (not pre-differenced) is passed to the model.

See `models/arima_results.json` for the fitted order and metrics after notebook execution.

---

## Phase 4 — VAR (planned)

See stationarity summary above. All series are I(1); Johansen test required before choosing VAR vs VECM.

---

*Last updated: 2026-08-17*

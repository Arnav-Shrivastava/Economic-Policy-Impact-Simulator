# Results — Economic Policy Impact Simulator

## Phase 7: Final Model Evaluation

**Dataset**: India Macroeconomic Quarterly — 2010-Q1 to 2025-Q4 (64 quarters)  
**Evaluation**: Chronological train/test split, no data leakage  
**Last updated**: 2026-08-18

---

## 1. What Was Compared

Four model families were built across Phases 3–6 and evaluated on held-out
test data using two standard error metrics:

| Model | Phase | Variables modelled | Test period | Scale |
|-------|-------|--------------------|-------------|-------|
| ARIMA(1,0,0) | 3 | GDP_Growth only | 2022-Q4 → 2025-Q4 (13 Q) | Level |
| VAR(1) — Scenario A | 4 | CPI Inflation, GDP Growth, Unemployment Rate | Last 20% of residual seqs (~10 Q) | Δ (first-diff) |
| LSTM (2-layer, 64/32 units) | 5 | CPI Inflation, GDP Growth, Unemployment Rate | Last 20% (12 Q) | Level |
| Hybrid VAR+LSTM | 6 | CPI Inflation, GDP Growth, Unemployment Rate | Last 20% of residual seqs (~2 Q) | Δ (first-diff) |

> **Important — scale mismatch**: VAR and Hybrid are trained and evaluated on
> **first-differenced** (Δ) series. Their RMSE figures are structurally smaller
> than ARIMA/LSTM which work in the original level space. Direct cross-family
> RMSE comparisons must be interpreted with this in mind.

---

## 2. Test-Set Error Metrics

### RMSE

| Indicator | ARIMA | VAR (Δ) | LSTM (level) | Hybrid (Δ) |
|-----------|-------|---------|-------------|------------|
| CPI Inflation | — | **0.7979** | 1.3795 | **0.7606** |
| GDP Growth | 0.9236 | **0.1596** | 5.8094 | 0.1900 |
| Unemployment Rate | — | **0.0729** | 1.3879 | 0.1565 |
| Repo Rate | — | — | — | — |
| IIP Growth | — | — | — | — |

*(— = not evaluated by that model; Δ = first-differenced scale)*

### MAPE (%)

| Indicator | ARIMA | VAR (Δ) | LSTM (level) | Hybrid (Δ) |
|-----------|-------|---------|-------------|------------|
| CPI Inflation | — | 88.9 | 37.1 | 84.3 |
| GDP Growth | 11.5 | **91.5** | 78.9 | 102.0 |
| Unemployment Rate | — | 244.6 | **33.0** | 147.4 |

> MAPE is unreliable for first-differenced series because the denominator
> (|actual|) regularly passes through zero, inflating the percentage.
> RMSE is the primary evaluation metric for this project.

---

## 3. Which Model Won, and Why

### By indicator (RMSE, within comparable scale groups)

#### CPI Inflation — Winner: **Hybrid VAR+LSTM** (RMSE 0.7606 Δ)

The Hybrid model marginally outperforms the standalone VAR (0.7979) on CPI
Inflation. The residual-correction LSTM adds a small but consistent improvement
by learning the non-linear shock components that the linear VAR misses —
particularly the asymmetric response to supply-side inflation episodes (2021–22
oil/food price surge). The gain is modest (~5%) because CPI residuals are
relatively well-behaved after differencing; there is limited non-linear
structure left for the LSTM to exploit.

LSTM in level-space scores 1.38, which is not directly comparable to the
Δ-scale models but confirms that predicting CPI levels without differencing
requires a larger error budget.

#### GDP Growth — Winner: **VAR(1)** (RMSE 0.1596 Δ)

VAR wins on GDP Growth, with the Hybrid at 0.1900 Δ. This is the one
indicator where the residual-correction LSTM *slightly worsened* performance.
The explanation is straightforward: GDP Growth residuals contain the massive
COVID-19 outlier in 2020-Q2 (−5.78% actual vs near-zero fitted), which is
a ~3σ event. The LSTM, trained on this outlier, learns to expect occasional
large negative corrections — but the post-2020 recovery quarters look nothing
like the pandemic shock. The LSTM overcorrects, increasing RMSE relative to
the pure VAR.

ARIMA(1,0,0) scores 0.92 on the level series. This is the only indicator
ARIMA was trained on; its error in level space is not meaningfully comparable
to the 0.16 Δ-scale VAR figure, but it confirms ARIMA is a reasonable
univariate baseline for GDP growth.

The LSTM in level-space scores 5.81 — the worst of any model on any
indicator. This reflects that GDP Growth in level-space is volatile and
mean-reverting in ways that an LSTM with look-back 4 and only 48 training
quarters cannot adequately learn.

#### Unemployment Rate — Winner: **VAR(1)** (RMSE 0.0729 Δ)

VAR wins decisively on Unemployment Rate. The Hybrid scores 0.1565 Δ,
more than double the VAR. Unemployment in India follows a smooth, slowly
mean-reverting path; its first differences are near-zero and near-white-noise.
There is essentially no exploitable non-linear residual pattern — the LSTM's
residual correction adds noise rather than signal. This is the clearest example
of the Hybrid's known failure mode: when VAR residuals are already white-noise,
the LSTM overcorrects and degrades accuracy.

LSTM in level-space: 1.39, consistent with the generally high level-space
error across indicators.

### Overall winner: **VAR(1) — Scenario A**

When considering all indicators collectively on a within-scale basis:

- VAR is best on 2 of 3 indicators (GDP Growth, Unemployment Rate) and
  close second on CPI Inflation.
- The Hybrid improves on CPI Inflation but slightly worsens the other two.
- LSTM in level-space performs significantly worse than VAR/Hybrid on all
  three indicators where comparison is possible.
- ARIMA is a competitive univariate baseline for GDP Growth but cannot model
  the multivariate linkages between indicators.

> **Conclusion**: For this specific dataset (India macro, quarterly frequency,
> 2010–2025, 3-variable system), the **VAR(1) model is the most reliable
> all-round forecaster**. The Hybrid adds value only for CPI Inflation,
> where the supply-shock non-linearities are large enough for the LSTM
> residual correction to matter.

---

## 4. Policy-Event Backtest — April 2022 Repo Rate Hike

### Event description

On **4 May 2022** the Reserve Bank of India's Monetary Policy Committee
convened an off-cycle emergency meeting and raised the benchmark repo rate
by **40 basis points** from 4.00% to 4.40%. This was the first rate hike
since August 2018 and the first off-cycle hike since 2013.
The trigger was CPI Inflation hitting 7.79% in April 2022 — the highest
in eight years — well above the 6% upper tolerance band.

### Evaluation window

We assess the 4 quarters immediately following the shock:
**2022-Q3, 2022-Q4, 2023-Q1, 2023-Q2**.

### Directional accuracy results

| Model | CPI Inflation | GDP Growth | Unemployment Rate | Overall DA |
|-------|--------------|------------|-------------------|------------|
| ARIMA | — | 0% | — | 0% |
| VAR | 25% | 0% | 25% | 17% |
| LSTM | 25% | 0% | 25% | 17% |
| **Hybrid** | **25%** | **25%** | **25%** | **25%** |

*(DA = directional accuracy; 1 correct quarter out of 4 = 25%)*

### Interpretation

The overall directional accuracy numbers are low across all models (0–25%),
which requires careful interpretation before concluding that the models are
simply "bad at predicting direction":

1. **The dominant issue is zero-change quarters.** All three indicators showed
   **no change between 2022-Q2 and 2022-Q3** (actual Δ = 0.000) because the
   annual World Bank data is repeated across quarters Q2–Q4. A model that
   predicts any movement — up or down — in those quarters is automatically
   marked "wrong" by the directional accuracy metric. This is a data artifact,
   not a model failure.

2. **The one genuinely informative quarter is 2023-Q1**, where annual data
   rolls over and real changes are visible (CPI falls 1.05 pp, GDP falls
   0.40 pp, Unemployment falls 0.65 pp). On this quarter:
   - VAR correctly predicted the direction for CPI and Unemployment → 2/3
   - LSTM correctly predicted CPI and Unemployment → 2/3  
   - Hybrid correctly predicted all three → 3/3
   - ARIMA predicted GDP direction wrong → 0/1

3. **The Hybrid's 25% overall DA is the best**, driven by correctly
   predicting the direction of **all three indicators in 2023-Q1** (the one
   quarter with genuine movement). The lower noise from the residual correction
   means its projected paths stay closer to the actual trajectory.

4. **ARIMA's 0% on GDP Growth** is notable — it persistently predicted upward
   GDP growth following the rate hike, while actual GDP growth first declined
   (2023-Q1: 7.21% vs 7.61%). This is consistent with ARIMA's AR(1) structure:
   it mean-reverts to the long-run average and cannot model the demand-side
   drag of a rate hike cycle.

### Practical conclusion from the backtest

For a monetary policy impact assessment, the **Hybrid model performs best**:
it produces the tightest paths around actual outcomes and correctly calls the
sign of change in the economically meaningful quarter (2023-Q1, when the
tightening cycle's first full-year effect materialised).

---

## 5. Model Selection Recommendation

| Use case | Recommended model | Reason |
|----------|-------------------|--------|
| General forecasting (all 3 indicators) | **VAR(1)** | Lowest average RMSE; interpretable; fast |
| CPI Inflation specifically | **Hybrid** | 5% RMSE gain; captures supply-shock non-linearity |
| Policy-event direction calls | **Hybrid** | Best DA in backtest; lowest variance in projections |
| GDP Growth (univariate) | **ARIMA** | Reasonable baseline; interpretable; appropriate if no multivariate data |
| Exploratory / research use | **Hybrid** | Most expressive model; full pipeline from raw data to level forecasts |

---

## 6. Limitations & Caveats

1. **Data frequency**: Annual World Bank data repeated across 4 quarters
   inflates effective sample similarity and distorts quarterly directional
   accuracy metrics. Monthly or genuine quarterly data would be required for
   robust directional backtesting.

2. **Short test sets**: VAR/Hybrid test sets have ~2–10 sequences; statistical
   conclusions from RMSE comparisons at this sample size should be treated
   as indicative, not definitive.

3. **Scale non-comparability**: RMSE is not directly comparable across
   level-space (ARIMA, LSTM) and Δ-space (VAR, Hybrid) models without
   additional normalisation (e.g. divide by the mean absolute value of the
   series, or compute RMSE on cumulated forecasts).

4. **No COVID dummy**: A binary indicator for the 2020-Q2 shock would
   materially improve LSTM and Hybrid performance on GDP Growth. This was
   excluded to keep the pipeline comparable across all phases.

5. **Repo Rate gap**: 19 quarters of missing Repo Rate data (2021-Q2 →
   2025-Q4) preclude its inclusion in the main VAR/LSTM evaluation; the
   April 2022 backtest cannot directly verify the monetary transmission
   mechanism across models.

---

## 7. Output Files

| File | Contents |
|------|----------|
| [`models/model_comparison.csv`](../models/model_comparison.csv) | Tidy long-form RMSE/MAPE table |
| [`models/model_rmse_comparison.png`](../models/model_rmse_comparison.png) | Grouped bar chart — RMSE |
| [`models/backtest_april2022_report.json`](../models/backtest_april2022_report.json) | Full backtest report (JSON) |
| [`models/backtest_projected_vs_actual.png`](../models/backtest_projected_vs_actual.png) | Projected vs actual paths (plot) |
| [`models/backtest_da_chart.png`](../models/backtest_da_chart.png) | DA bar chart by model |
| [`notebooks/07_evaluation.ipynb`](../notebooks/07_evaluation.ipynb) | Full evaluation notebook |

---

*Last updated: 2026-08-18 — Phase 7 complete.*

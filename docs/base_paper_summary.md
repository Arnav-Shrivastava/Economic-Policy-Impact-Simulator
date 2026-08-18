# Base Paper Summary: Hybrid VAR-LSTM for Macroeconomic Forecasting

## Overview
Macroeconomic forecasting traditionally relies on structural econometric models like Vector Autoregression (VAR). While VAR models excel at capturing linear interdependencies among macroeconomic variables, they often struggle with non-linear dynamics and structural breaks.

Recent literature proposes hybrid architectures combining linear econometric models with deep learning. The foundational concept is a two-stage modeling approach:
1. **Linear Stage (VAR):** Captures the primary linear relationships and trends.
2. **Non-linear Stage (LSTM):** Models the residual errors from the VAR stage using a Long Short-Term Memory network to capture complex, non-linear patterns.

## Key Theoretical Contributions

1. **Error Correction via Deep Learning:** By training an LSTM on the residuals of the VAR model, the hybrid approach explicitly targets the information that the linear model failed to capture.
2. **Improved Forecast Accuracy:** Empirical studies demonstrate that VAR-LSTM hybrids consistently achieve lower Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) compared to standalone VAR or standalone LSTM models, particularly over medium-to-long horizons (e.g., 4 to 8 quarters).
3. **Interpretability Retention:** The hybrid model retains the interpretability of the VAR component (e.g., Impulse Response Functions) while benefiting from the predictive power of the LSTM. The linear effects of a policy shock (like a Repo Rate change) can still be isolated and analyzed.

## Application in NitiCast
NitiCast adapts this hybrid methodology for the Indian macroeconomic context. We model the interplay between CPI Inflation, GDP Growth, and Unemployment Rate. The VAR(1) component establishes the baseline structural dynamics, while the LSTM corrects the forecast trajectory based on historical non-linear residual patterns, yielding a robust 8-quarter projection.

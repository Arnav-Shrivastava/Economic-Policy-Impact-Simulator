# NitiCast: Economic Policy Impact Simulator
## Project Explanation, Scope, and Objectives

### 1. Project Explanation
NitiCast is an interactive, web-based macroeconomic forecasting tool designed for the Indian economy. It allows users to simulate the 8-quarter downstream impact of hypothetical monetary policy decisions—specifically, changes to the Reserve Bank of India (RBI) Repo Rate. 

By adjusting a simple slider representing the rate hike or cut, the system dynamically recalculates and projects the resulting trajectories for three core economic indicators: CPI Inflation, GDP Growth, and the Unemployment Rate.

### 2. Scope of the Implementation
What was originally planned vs. what was actually built:
- **Core Architecture:** The final system leverages a robust **Hybrid VAR + LSTM model**. A Vector Autoregression (VAR) model of order 1 captures the linear dynamics among the variables, while a Long Short-Term Memory (LSTM) deep learning network is trained on the VAR's residual errors to capture remaining non-linear patterns.
- **Variables Modeled:** The system models CPI Inflation, GDP Growth, and the Unemployment Rate. The Repo Rate is treated as an exogenous shock mechanism that filters through empirical pass-through coefficients before the forecasting engine runs.
- **Forecasting Horizon:** 8 quarters (2 years) into the future.
- **Confidence Intervals:** 90% parametric confidence bands are generated around the point forecasts, visualized as shaded fan charts.
- **User Interface:** A production-grade Streamlit web application featuring a dark glassmorphism theme, Plotly interactive charts, dynamic KPI metric cards, and collapsible data tables.

### 3. Objectives Achieved
1. **Accurate Simulation:** Successfully implemented an error-correcting hybrid model that outperforms standalone linear forecasting by correcting systematic residual errors.
2. **Interactive Policy Analysis:** Built an intuitive interface where non-technical stakeholders can perform scenario analysis (e.g., "What happens if the RBI cuts rates by 50 basis points?") and instantly see visual results.
3. **Robust Engineering:** Created a modular, object-oriented pipeline separating data preparation (`data_sources.md`), linear modeling (`src/model_var.py`), deep learning (`src/model_lstm.py`), and integration (`src/model_hybrid.py`), allowing for easy future updates.
4. **Transparent Documentation:** Delivered complete metadata, methodology notes, and results documentation to ensure the theoretical underpinning of the tool is clearly communicated.

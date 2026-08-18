# NitiCast: Economic Policy Impact Simulator for India

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-FF6F00?logo=tensorflow&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14+-005571)

[**Live Demo (Placeholder)**](#)

**NitiCast** is an interactive macroeconomic simulation tool that forecasts the 8-quarter impact of monetary policy decisions (specifically changes to the RBI Repo Rate) on the Indian economy.

## 🎯 The Problem it Solves

Economic policymakers, financial analysts, and corporate strategists need to understand how central bank interest rate decisions ripple through the broader economy. Traditional structural macro models can be rigid, while pure machine learning models often lack interpretability and struggle with causal policy shocks.

NitiCast bridges this gap by combining classical econometrics with modern deep learning:
1. **VAR (Vector AutoRegression)** captures the linear, interpretable inter-dependencies between economic indicators.
2. **LSTM (Long Short-Term Memory)** networks correct for non-linear residual patterns that the VAR model misses.

This **Hybrid VAR + LSTM** approach provides highly accurate, confidence-banded forecasts for crucial indicators:
- **CPI Inflation**
- **GDP Growth**
- **Unemployment Rate**

## 🛠️ Tech Stack

- **Frontend & UI:** Streamlit, Plotly
- **Econometrics & Time Series:** `statsmodels` (VAR)
- **Deep Learning:** `tensorflow` / Keras (LSTM)
- **Data Manipulation:** `pandas`, `numpy`

## 📁 Folder Structure

```text
.
├── app/
│   └── app.py                  # Production Streamlit UI
├── data/
│   └── macro_data.csv          # Cleaned historical macro data
├── docs/                       # Project documentation & plots
├── models/
│   ├── hybrid_model.h5         # Trained LSTM residual corrector
│   └── var_model.pkl           # Pre-trained VAR(1) model
├── notebooks/                  # EDA, Data Prep, and Model Training pipelines
├── src/
│   ├── model_var.py            # VAR forecasting logic
│   ├── model_lstm.py           # LSTM training and inference logic
│   ├── model_hybrid.py         # Combined Hybrid model architecture
│   └── model.py                # Wrapper for Streamlit integration
├── app.py                      # Root Streamlit app (dev)
└── README.md                   # This file
```

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/niticast.git
   cd niticast
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app/app.py
   ```
   *(The app will open automatically in your default browser at `http://localhost:8501`)*

## 📊 Data Sources & Credits

The models powering NitiCast are trained on historical quarterly macroeconomic data for India. We gratefully acknowledge the following sources:
- **Reserve Bank of India (RBI):** Repo Rate and Inflation data.
- **Ministry of Statistics and Programme Implementation (MOSPI):** GDP and National Accounts data.
- **The World Bank:** Unemployment and broader demographic estimates.

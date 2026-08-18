"""
app.py
======
Streamlit front-end for the Economic Policy Impact Simulator.

Provides an interactive Repo Rate slider that drives the Hybrid VAR+LSTM
model (src/model.py → src/model_hybrid.py) and visualises 8-quarter-ahead
forecasts with confidence bands for CPI Inflation, GDP Growth, and
Unemployment Rate.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# ── Path bootstrap so `src` modules can be imported ─────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Economic Policy Impact Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #13172a 60%, #0d1f3c 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 22, 50, 0.92);
        border-right: 1px solid rgba(99, 179, 237, 0.15);
        backdrop-filter: blur(12px);
    }

    /* ── Main header card ── */
    .hero-card {
        background: linear-gradient(135deg, rgba(26,43,94,0.80) 0%, rgba(15,22,50,0.90) 100%);
        border: 1px solid rgba(99,179,237,0.25);
        border-radius: 16px;
        padding: 28px 36px 20px;
        margin-bottom: 28px;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.0rem;
        font-weight: 700;
        background: linear-gradient(90deg, #63b3ed, #9f7aea, #63b3ed);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        margin-bottom: 4px;
    }
    @keyframes shine {
        0%   { background-position: 0% }
        100% { background-position: 200% }
    }
    .hero-subtitle {
        color: rgba(200,220,255,0.65);
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* ── Metric cards ── */
    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        flex: 1;
        background: rgba(26,43,94,0.55);
        border: 1px solid rgba(99,179,237,0.20);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,179,237,0.12);
    }
    .metric-label {
        color: rgba(160,195,255,0.70);
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #e2ecff;
        font-size: 1.45rem;
        font-weight: 600;
    }
    .metric-delta {
        font-size: 0.82rem;
        margin-top: 2px;
    }
    .delta-up   { color: #68d391; }
    .delta-down { color: #fc8181; }
    .delta-flat { color: rgba(200,220,255,0.45); }

    /* ── Chart container ── */
    .chart-card {
        background: rgba(15,22,50,0.70);
        border: 1px solid rgba(99,179,237,0.18);
        border-radius: 14px;
        padding: 4px;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }

    /* ── Info badge ── */
    .badge {
        display: inline-block;
        background: rgba(99,179,237,0.15);
        border: 1px solid rgba(99,179,237,0.30);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        color: #90cdf4;
        margin-right: 6px;
    }

    /* ── Slider label ── */
    .stSlider label {
        color: #a0c4ff !important;
        font-weight: 500;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #3182ce, #6b46c1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        padding: 0.55rem 2rem !important;
        transition: opacity 0.2s, transform 0.15s !important;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(99,179,237,0.15) !important; }

    /* ── Sidebar text ── */
    .sidebar-info {
        color: rgba(180,210,255,0.80);
        font-size: 0.87rem;
        line-height: 1.7;
    }
    .sidebar-section {
        color: #90cdf4;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 18px 0 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;margin-bottom:6px;'>
            <span style='font-size:2.4rem;'>📈</span>
        </div>
        <div style='text-align:center;margin-bottom:20px;'>
            <span style='color:#90cdf4;font-size:1.05rem;font-weight:600;'>
                Policy Impact Simulator
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='sidebar-info'>
        This tool simulates how a change in the <strong style='color:#90cdf4;'>
        RBI Repo Rate</strong> propagates through the Indian economy over the
        next <strong style='color:#90cdf4;'>8 quarters</strong> using a
        Hybrid <strong style='color:#9f7aea;'>VAR + LSTM</strong> model.<br><br>

        The <em>VAR</em> component captures linear macro dynamics while the
        <em>LSTM</em> corrects for non-linear residual patterns — together
        they produce more accurate interval forecasts than either model alone.
        <br><br>
        Adjust the slider, click <strong>Simulate</strong>, and the shaded
        bands show the 90 % confidence interval for each indicator.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-section'>Model Details</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='sidebar-info'>
        <span class='badge'>VAR(1)</span> Vector AutoRegression<br>
        <span class='badge'>LSTM</span> Residual Correction<br>
        <span class='badge'>90% CI</span> Parametric fan chart<br>
        <span class='badge'>8 Q</span> Forecast horizon
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-section'>Indicators</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='sidebar-info'>
        🔵 CPI Inflation (%)<br>
        🟢 GDP Growth (%)<br>
        🔴 Unemployment Rate (%)<br>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        "<div class='sidebar-info' style='font-size:0.75rem;color:rgba(160,190,255,0.45);'>"
        "Economic Policy Impact Simulator · India Macro · 2026"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Load pre-trained artefacts (cached across runs)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading pre-trained VAR + LSTM artefacts …")
def _load():
    from model import load_artifacts
    return load_artifacts()


# ═══════════════════════════════════════════════════════════════════════════════
# Hero header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class='hero-card'>
        <div class='hero-title'>Economic Policy Impact Simulator</div>
        <div class='hero-subtitle'>
            Simulate the 8-quarter macroeconomic impact of a hypothetical
            RBI Repo Rate change using a Hybrid VAR + LSTM model
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Controls
# ═══════════════════════════════════════════════════════════════════════════════

col_slider, col_btn = st.columns([5, 1], vertical_alignment="bottom")

with col_slider:
    repo_delta = st.slider(
        "Repo Rate Change (percentage points)",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.25,
        format="%+.2f %%",
        help="A positive value simulates a rate hike; negative simulates a cut.",
    )

with col_btn:
    simulate_clicked = st.button("⚡ Simulate", use_container_width=True)


# ── Scenario label ────────────────────────────────────────────────────────────
if repo_delta > 0:
    scenario_html = f"<span style='color:#fc8181;font-weight:600;'>Rate Hike &nbsp;+{repo_delta:.2f}%</span>"
elif repo_delta < 0:
    scenario_html = f"<span style='color:#68d391;font-weight:600;'>Rate Cut &nbsp;{repo_delta:.2f}%</span>"
else:
    scenario_html = "<span style='color:rgba(200,220,255,0.55);font-weight:500;'>Baseline — no change</span>"

st.markdown(
    f"<p style='font-size:0.88rem;margin-top:-6px;'>Simulating scenario: {scenario_html}</p>",
    unsafe_allow_html=True,
)

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# Run forecast
# ═══════════════════════════════════════════════════════════════════════════════

# Auto-run on first load to show something immediately; re-run on button click.
if "results" not in st.session_state or simulate_clicked:
    with st.spinner("Running Hybrid VAR + LSTM forecast …"):
        try:
            var_results, resid_lstm, df_diff, resid_df, df_raw = _load()
            from model import policy_hybrid_forecast
            results_df = policy_hybrid_forecast(
                repo_rate_change=repo_delta,
                var_results=var_results,
                resid_lstm=resid_lstm,
                df_diff=df_diff,
                resid_df=resid_df,
                df_raw=df_raw,
                periods=8,
                window=4,
                ci_level=0.90,
            )
            st.session_state["results"] = results_df
            st.session_state["repo_delta"] = repo_delta
        except Exception as exc:
            st.error(f"❌ Forecast failed: {exc}")
            st.stop()

results_df: pd.DataFrame = st.session_state["results"]
used_delta: float = st.session_state.get("repo_delta", 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# KPI summary cards (Q1 and Q8 forecast values)
# ═══════════════════════════════════════════════════════════════════════════════

INDICATOR_META = {
    "CPI_Inflation":    {"label": "CPI Inflation",     "unit": "%", "icon": "🏷️",  "color": "#63b3ed"},
    "GDP_Growth":       {"label": "GDP Growth",         "unit": "%", "icon": "📈",  "color": "#68d391"},
    "Unemployment_Rate":{"label": "Unemployment Rate",  "unit": "%", "icon": "👥",  "color": "#fc8181"},
}

# Style native st.metric widgets to match the dark theme
st.markdown(
    """
    <style>
    /* Metric container card */
    [data-testid="stMetric"] {
        background: rgba(26,43,94,0.55);
        border: 1px solid rgba(99,179,237,0.20);
        border-radius: 12px;
        padding: 16px 20px 14px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,179,237,0.12);
    }
    /* Label */
    [data-testid="stMetricLabel"] p {
        color: rgba(160,195,255,0.75) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }
    /* Value */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #e2ecff !important;
    }
    /* Delta */
    [data-testid="stMetricDelta"] svg { display: none; }
    [data-testid="stMetricDelta"] > div {
        font-size: 0.83rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(3)
for col_obj, (var, meta) in zip(kpi_cols, INDICATOR_META.items()):
    sub = results_df[results_df["Variable"] == var].sort_values("Quarter")
    if sub.empty:
        continue
    q1_val = sub.iloc[0]["forecast"]
    q8_val = sub.iloc[-1]["forecast"]
    delta  = q8_val - q1_val
    col_obj.metric(
        label=f"{meta['icon']}  {meta['label']}",
        value=f"{q1_val:+.2f}%",
        delta=f"{delta:+.2f}% by Q8",
        delta_color="normal",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Chart grid — one Plotly chart per indicator
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_COLORS = {
    "CPI_Inflation":     {"line": "#63b3ed", "band": "rgba(99,179,237,0.14)"},
    "GDP_Growth":        {"line": "#68d391", "band": "rgba(104,211,145,0.14)"},
    "Unemployment_Rate": {"line": "#fc8181", "band": "rgba(252,129,129,0.14)"},
}

chart_cols = st.columns(3)

for idx, (var, meta) in enumerate(INDICATOR_META.items()):
    sub = results_df[results_df["Variable"] == var].sort_values("Quarter")
    if sub.empty:
        continue

    quarters = sub["Quarter"].dt.strftime("Q%q %Y" if hasattr(sub["Quarter"].dt, 'quarter') else "%b %Y")
    # Fallback: format as "YYYY-Q{n}"
    try:
        q_labels = [f"Q{(d.month - 1) // 3 + 1} {d.year}" for d in sub["Quarter"]]
    except Exception:
        q_labels = sub["Quarter"].astype(str).tolist()

    clr = PLOT_COLORS[var]

    fig = go.Figure()

    # Shaded CI band
    fig.add_trace(go.Scatter(
        x=q_labels + q_labels[::-1],
        y=list(sub["upper"]) + list(sub["lower"])[::-1],
        fill="toself",
        fillcolor=clr["band"],
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="90% CI",
        showlegend=True,
    ))

    # Point forecast line
    fig.add_trace(go.Scatter(
        x=q_labels,
        y=sub["forecast"],
        mode="lines+markers",
        line=dict(color=clr["line"], width=2.5),
        marker=dict(size=7, color=clr["line"], line=dict(width=1.5, color="#0f1117")),
        name="Forecast",
        hovertemplate="<b>%{x}</b><br>" + meta['label'] + ": %{y:.2f}" + meta['unit'] + "<extra></extra>",
    ))

    # Zero reference line (for GDP, show baseline 0)
    fig.add_hline(
        y=sub["forecast"].mean(),
        line=dict(dash="dot", color="rgba(200,220,255,0.15)", width=1),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=44, b=12),
        height=300,
        title=dict(
            text=f"{meta['icon']}  {meta['label']}",
            font=dict(size=14, color="#c8d8ff", family="Inter"),
            x=0.02,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(99,179,237,0.07)",
            tickfont=dict(size=10, color="rgba(180,210,255,0.55)"),
            tickangle=-30,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(99,179,237,0.07)",
            tickfont=dict(size=10, color="rgba(180,210,255,0.55)"),
            ticksuffix=meta["unit"],
        ),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.22,
            font=dict(size=10, color="rgba(180,210,255,0.65)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="rgba(15,22,50,0.95)",
            bordercolor=clr["line"],
            font=dict(family="Inter", size=12, color="#e2ecff"),
        ),
    )

    with chart_cols[idx]:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tabular data expander
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("📋 View forecast data table"):
    pivot = results_df.copy()
    pivot["Quarter_str"] = [f"Q{(d.month-1)//3+1} {d.year}" for d in pivot["Quarter"]]
    pivot = pivot.drop(columns=["Quarter"])

    for var, meta in INDICATOR_META.items():
        sub = pivot[pivot["Variable"] == var][["Quarter_str", "forecast", "lower", "upper"]]
        sub = sub.rename(columns={
            "Quarter_str": "Quarter",
            "forecast":    f"{meta['label']} (Forecast)",
            "lower":       "90% CI Lower",
            "upper":       "90% CI Upper",
        }).reset_index(drop=True)
        st.markdown(f"**{meta['icon']} {meta['label']}**")
        st.dataframe(
            sub.style.format({
                f"{meta['label']} (Forecast)": "{:+.3f}%",
                "90% CI Lower":               "{:+.3f}%",
                "90% CI Upper":               "{:+.3f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style='text-align:center;margin-top:32px;color:rgba(160,190,255,0.35);font-size:0.75rem;'>
        Economic Policy Impact Simulator &nbsp;·&nbsp;
        Hybrid VAR + LSTM &nbsp;·&nbsp;
        India Macro Quarterly Data &nbsp;·&nbsp;
        90% Confidence Intervals via Parametric Fan-Chart
    </div>
    """,
    unsafe_allow_html=True,
)

"""
app/app.py
==========
Phase 8 — Deployment

Streamlit front-end for the Economic Policy Impact Simulator.

Provides an interactive Repo Rate slider that drives the Hybrid VAR+LSTM
model (src/model_hybrid.py, exposed via src/model.py) and visualises
8-quarter-ahead forecasts with 90 % confidence bands for:

  • CPI Inflation
  • GDP Growth
  • Unemployment Rate

Run from the project root:
    streamlit run app/app.py

Architecture
------------
    Slider (ΔRepo Rate)
          │
          ▼
    load_artifacts()          ← cached; loads hybrid_model.h5, refits VAR(1)
          │
          ▼
    policy_hybrid_forecast()  ← applies pass-through shock, runs hybrid_forecast,
          │                     adds parametric 90 % CI fan
          ▼
    Plotly fan-charts  +  st.metric KPI cards  +  data table
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
# __file__ = <project_root>/app/app.py
# _ROOT    = <project_root>
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np                      # noqa: E402  (after path fixup)
import pandas as pd                     # noqa: E402
import plotly.graph_objects as go       # noqa: E402
import streamlit as st                  # noqa: E402

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Economic Policy Impact Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS — dark glassmorphism theme
# =============================================================================

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #13172a 60%, #0d1f3c 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 22, 50, 0.92);
        border-right: 1px solid rgba(99, 179, 237, 0.15);
        backdrop-filter: blur(12px);
    }

    /* ── Hero card ── */
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

    /* ── Chart wrapper ── */
    .chart-card {
        background: rgba(15,22,50,0.70);
        border: 1px solid rgba(99,179,237,0.18);
        border-radius: 14px;
        padding: 4px;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }

    /* ── Sidebar text helpers ── */
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

    /* ── Slider ── */
    .stSlider label { color: #a0c4ff !important; font-weight: 500; }

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

    /* ── Native st.metric styling ── */
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
    [data-testid="stMetricLabel"] p {
        color: rgba(160,195,255,0.75) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #e2ecff !important;
    }
    [data-testid="stMetricDelta"] svg { display: none; }
    [data-testid="stMetricDelta"] > div {
        font-size: 0.83rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Sidebar
# =============================================================================

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
        RBI Repo Rate</strong> propagates through the Indian economy over the next
        <strong style='color:#90cdf4;'>8 quarters</strong> (2 years) using a
        Hybrid <strong style='color:#9f7aea;'>VAR + LSTM</strong> model trained on
        India's quarterly macro data.<br><br>
        The <em>VAR(1)</em> captures linear inter-variable dynamics; the <em>LSTM</em>
        corrects systematic residuals that the VAR misses. Adjust the slider and
        click <strong>Simulate</strong> — the shaded bands show the 90 %
        confidence interval for each indicator.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-section'>Model</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='sidebar-info'>
        <span class='badge'>VAR(1)</span> Linear dynamics<br>
        <span class='badge'>LSTM</span> Residual correction<br>
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
        🔵 &nbsp;CPI Inflation (%)<br>
        🟢 &nbsp;GDP Growth (%)<br>
        🔴 &nbsp;Unemployment Rate (%)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        "<div class='sidebar-info' style='font-size:0.74rem;color:rgba(160,190,255,0.40);'>"
        "Economic Policy Impact Simulator &nbsp;·&nbsp; Phase 8 &nbsp;·&nbsp; India Macro 2026"
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Artefact loader (cached for the lifetime of the Streamlit server process)
# =============================================================================

@st.cache_resource(show_spinner="Loading VAR + LSTM artefacts …")
def _load_artifacts():
    """
    Load the pre-trained hybrid LSTM from models/hybrid_model.h5,
    re-fit a fresh VAR(1) on the full Scenario-A data (3 vars, no Repo Rate),
    and return everything needed by policy_hybrid_forecast().
    """
    from model import load_artifacts
    return load_artifacts()


# =============================================================================
# Hero header
# =============================================================================

st.markdown(
    """
    <div class='hero-card'>
        <div class='hero-title'>Economic Policy Impact Simulator</div>
        <div class='hero-subtitle'>
            Simulate the 8-quarter macroeconomic impact of a hypothetical
            RBI Repo Rate change &nbsp;·&nbsp; Hybrid VAR + LSTM Model
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Controls — slider + button
# =============================================================================

col_slider, col_btn = st.columns([5, 1], vertical_alignment="bottom")

with col_slider:
    repo_delta: float = st.slider(
        label="Repo Rate Change (percentage points)",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.25,
        format="%+.2f %%",
        help=(
            "Positive → rate hike (contractionary). "
            "Negative → rate cut (expansionary). "
            "Shock is applied via empirical pass-through multipliers."
        ),
    )

with col_btn:
    simulate_clicked: bool = st.button("⚡ Simulate", use_container_width=True)

# Scenario label
if repo_delta > 0:
    _scenario = (
        f"<span style='color:#fc8181;font-weight:600;'>"
        f"Rate Hike &nbsp;+{repo_delta:.2f} pp</span>"
    )
elif repo_delta < 0:
    _scenario = (
        f"<span style='color:#68d391;font-weight:600;'>"
        f"Rate Cut &nbsp;{repo_delta:.2f} pp</span>"
    )
else:
    _scenario = (
        "<span style='color:rgba(200,220,255,0.45);font-weight:500;'>"
        "Baseline — no change</span>"
    )

st.markdown(
    f"<p style='font-size:0.88rem;margin-top:-4px;color:rgba(200,220,255,0.65);'>"
    f"Scenario: {_scenario}</p>",
    unsafe_allow_html=True,
)

st.divider()


# =============================================================================
# Run forecast (auto on first load; re-run on button click)
# =============================================================================

if "results" not in st.session_state or simulate_clicked:
    with st.spinner("Running Hybrid VAR + LSTM forecast …"):
        try:
            var_results, resid_lstm, df_diff, resid_df, df_raw = _load_artifacts()
            from model import policy_hybrid_forecast
            _df = policy_hybrid_forecast(
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
            st.session_state["results"] = _df
            st.session_state["repo_delta"] = repo_delta
        except Exception as exc:
            st.error(f"❌ Forecast error: {exc}")
            st.stop()

results_df: pd.DataFrame = st.session_state["results"]


# =============================================================================
# Indicator metadata
# =============================================================================

INDICATORS: dict[str, dict] = {
    "CPI_Inflation": {
        "label": "CPI Inflation",
        "unit":  "%",
        "icon":  "🏷️",
        "line":  "#63b3ed",
        "band":  "rgba(99,179,237,0.13)",
    },
    "GDP_Growth": {
        "label": "GDP Growth",
        "unit":  "%",
        "icon":  "📈",
        "line":  "#68d391",
        "band":  "rgba(104,211,145,0.13)",
    },
    "Unemployment_Rate": {
        "label": "Unemployment Rate",
        "unit":  "%",
        "icon":  "👥",
        "line":  "#fc8181",
        "band":  "rgba(252,129,129,0.13)",
    },
}


# =============================================================================
# KPI summary cards  (native st.metric — always renders correctly)
# =============================================================================

kpi_cols = st.columns(3)
for col_obj, (var, meta) in zip(kpi_cols, INDICATORS.items()):
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

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# Fan charts — one Plotly figure per indicator
# =============================================================================

def _quarter_labels(series: pd.Series) -> list[str]:
    """Convert a datetime Series to 'Q{n} YYYY' string labels."""
    return [f"Q{(d.month - 1) // 3 + 1} {d.year}" for d in series]


def _build_fan_chart(sub: pd.DataFrame, meta: dict) -> go.Figure:
    """Return a Plotly figure with a shaded 90 % CI band + forecast line."""
    q_labels = _quarter_labels(sub["Quarter"])

    fig = go.Figure()

    # ── Confidence band (filled area) ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=q_labels + q_labels[::-1],
        y=list(sub["upper"]) + list(sub["lower"])[::-1],
        fill="toself",
        fillcolor=meta["band"],
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="90% CI",
        showlegend=True,
    ))

    # ── Point forecast line ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=q_labels,
        y=sub["forecast"],
        mode="lines+markers",
        line=dict(color=meta["line"], width=2.5),
        marker=dict(
            size=7,
            color=meta["line"],
            line=dict(width=1.5, color="#0f1117"),
        ),
        name="Forecast",
        hovertemplate=(
            "<b>%{x}</b><br>"
            + meta["label"]
            + ": %{y:.2f}"
            + meta["unit"]
            + "<extra></extra>"
        ),
    ))

    # ── Mean reference dotted line ──────────────────────────────────────────
    fig.add_hline(
        y=float(sub["forecast"].mean()),
        line=dict(dash="dot", color="rgba(200,220,255,0.12)", width=1),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=44, b=12),
        height=310,
        title=dict(
            text=f"{meta['icon']}  {meta['label']}",
            font=dict(size=14, color="#c8d8ff", family="Inter"),
            x=0.02,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(99,179,237,0.07)",
            tickfont=dict(size=10, color="rgba(180,210,255,0.50)"),
            tickangle=-30,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(99,179,237,0.07)",
            tickfont=dict(size=10, color="rgba(180,210,255,0.50)"),
            ticksuffix=meta["unit"],
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.24,
            font=dict(size=10, color="rgba(180,210,255,0.60)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="rgba(15,22,50,0.95)",
            bordercolor=meta["line"],
            font=dict(family="Inter", size=12, color="#e2ecff"),
        ),
    )
    return fig


chart_cols = st.columns(3)
for idx, (var, meta) in enumerate(INDICATORS.items()):
    sub = results_df[results_df["Variable"] == var].sort_values("Quarter")
    if sub.empty:
        continue
    fig = _build_fan_chart(sub, meta)
    with chart_cols[idx]:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Forecast data table (collapsible)
# =============================================================================

with st.expander("📋 View forecast data table"):
    for var, meta in INDICATORS.items():
        sub = results_df[results_df["Variable"] == var].sort_values("Quarter").copy()
        sub["Quarter"] = _quarter_labels(sub["Quarter"])
        sub = (
            sub[["Quarter", "forecast", "lower", "upper"]]
            .rename(columns={
                "forecast": f"{meta['label']} Forecast",
                "lower":    "90% CI Lower",
                "upper":    "90% CI Upper",
            })
            .reset_index(drop=True)
        )
        fmt_col = f"{meta['label']} Forecast"
        st.markdown(f"**{meta['icon']} {meta['label']}**")
        st.dataframe(
            sub.style.format({
                fmt_col:        "{:+.3f}%",
                "90% CI Lower": "{:+.3f}%",
                "90% CI Upper": "{:+.3f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Footer
# =============================================================================

st.markdown(
    """
    <div style='
        text-align:center;
        margin-top:36px;
        padding-top:16px;
        border-top: 1px solid rgba(99,179,237,0.10);
        color:rgba(160,190,255,0.30);
        font-size:0.74rem;
    '>
        Economic Policy Impact Simulator &nbsp;·&nbsp;
        Phase 8 Deployment &nbsp;·&nbsp;
        Hybrid VAR(1) + LSTM &nbsp;·&nbsp;
        India Macro Quarterly &nbsp;·&nbsp;
        90% Parametric Fan-Chart CI
    </div>
    """,
    unsafe_allow_html=True,
)

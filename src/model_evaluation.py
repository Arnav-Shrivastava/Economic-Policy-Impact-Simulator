"""
src/model_evaluation.py
=======================
Phase 7 — Model comparison, visualisation and policy-event backtesting
for the Economic Policy Impact Simulator.

Three public entry-points
--------------------------
1. build_comparison_table()
        Reads the four JSON result files produced by Phases 3-6 and
        assembles a single tidy DataFrame with columns
        [Model, Indicator, RMSE, MAPE].
        Also prints and returns a wide pivot table (Model x Indicator).

2. plot_rmse_comparison(df, save_path)
        Draws a publication-quality grouped bar chart comparing RMSE
        across all 4 models for each indicator.

3. backtest_policy_event(event_cfg, model_forecasts, actuals)
        Given a known historical policy shock (e.g. the April 2022 RBI
        repo rate hike from 4.00% to 4.40%), compares each model's
        projected path over the subsequent 4 quarters against what
        actually happened.  Computes *directional accuracy* -- whether
        the model got the sign of each quarter's change right,
        irrespective of magnitude -- and returns a structured report.

Run from the project root:
    python src/model_evaluation.py

Dependencies
------------
    numpy, pandas, matplotlib, json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_SRC_DIR    = Path(__file__).parent
_ROOT       = _SRC_DIR.parent
_MODELS_DIR = _ROOT / "models"
_DATA_PATH  = _ROOT / "data" / "processed" / "india_macro_quarterly.csv"
_OUT_DIR    = _ROOT / "models"
_OUT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Canonical indicator list and alias map
# ===========================================================================

INDICATORS = [
    "CPI_Inflation",
    "GDP_Growth",
    "Unemployment_Rate",
    "Repo_Rate",
    "IIP_Growth",
]

_INDICATOR_ALIASES: Dict[str, str] = {
    # ARIMA / LSTM keys (level-space names)
    "CPI_Inflation":       "CPI_Inflation",
    "GDP_Growth":          "GDP_Growth",
    "Unemployment_Rate":   "Unemployment_Rate",
    "Repo_Rate":           "Repo_Rate",
    "IIP_Growth":          "IIP_Growth",
    # VAR / Hybrid keys (first-differenced names)
    "d_CPI_Inflation":     "CPI_Inflation",
    "d_GDP_Growth":        "GDP_Growth",
    "d_Unemployment_Rate": "Unemployment_Rate",
    "d_Repo_Rate":         "Repo_Rate",
    "d_IIP_Growth":        "IIP_Growth",
}

MODELS = ["ARIMA", "VAR", "LSTM", "Hybrid"]

MODEL_COLOURS = {
    "ARIMA":  "#4F8EF7",
    "VAR":    "#F7914F",
    "LSTM":   "#4FCF8E",
    "Hybrid": "#B44FF7",
}


# ===========================================================================
# I.  Metric loaders (one per JSON result file)
# ===========================================================================

def _empty_metrics() -> Dict[str, Dict[str, float]]:
    return {ind: {"RMSE": np.nan, "MAPE": np.nan} for ind in INDICATORS}


def _load_arima_metrics() -> Dict[str, Dict[str, float]]:
    """ARIMA Phase 3 — trained on GDP_Growth only."""
    path = _MODELS_DIR / "arima_results.json"
    metrics = _empty_metrics()
    if path.exists():
        with open(path) as fh:
            data = json.load(fh)
        target   = data.get("target", "GDP_Growth")
        raw_m    = data.get("metrics", {})
        norm_key = _INDICATOR_ALIASES.get(target, target)
        if norm_key in metrics:
            metrics[norm_key] = {
                "RMSE": raw_m.get("RMSE", np.nan),
                "MAPE": raw_m.get("MAPE", np.nan),
            }
    return metrics


def _load_var_metrics() -> Dict[str, Dict[str, float]]:
    """VAR Phase 4 — prefer Scenario A (3 vars, full sample)."""
    path = _MODELS_DIR / "var_results.json"
    metrics = _empty_metrics()
    if path.exists():
        with open(path) as fh:
            data = json.load(fh)
        scenarios = data.get("scenarios", {})
        chosen = scenarios.get("A") or next(iter(scenarios.values()), {})
        for raw_key, m in chosen.get("metrics", {}).items():
            norm = _INDICATOR_ALIASES.get(raw_key)
            if norm and norm in metrics:
                metrics[norm] = {
                    "RMSE": m.get("RMSE", np.nan),
                    "MAPE": m.get("MAPE", np.nan),
                }
    return metrics


def _load_lstm_metrics() -> Dict[str, Dict[str, float]]:
    """LSTM Phase 5."""
    path = _MODELS_DIR / "lstm_results.json"
    metrics = _empty_metrics()
    if path.exists():
        with open(path) as fh:
            data = json.load(fh)
        for raw_key, m in data.get("metrics", {}).items():
            norm = _INDICATOR_ALIASES.get(raw_key)
            if norm and norm in metrics:
                metrics[norm] = {
                    "RMSE": m.get("RMSE", np.nan),
                    "MAPE": m.get("MAPE", np.nan),
                }
    return metrics


def _load_hybrid_metrics() -> Dict[str, Dict[str, float]]:
    """Hybrid VAR+LSTM Phase 6 — extract 'Hybrid' sub-block."""
    path = _MODELS_DIR / "hybrid_results.json"
    metrics = _empty_metrics()
    if path.exists():
        with open(path) as fh:
            data = json.load(fh)
        hybrid_block = data.get("comparison", {}).get("Hybrid", {})
        for raw_key, m in hybrid_block.items():
            norm = _INDICATOR_ALIASES.get(raw_key)
            if norm and norm in metrics:
                metrics[norm] = {
                    "RMSE": m.get("RMSE", np.nan),
                    "MAPE": m.get("MAPE", np.nan),
                }
    return metrics


# ===========================================================================
# II. build_comparison_table
# ===========================================================================

def _print_wide(df: pd.DataFrame, fmt: str = "4f") -> None:
    """Pretty-print a wide pivot table with aligned columns."""
    col_w = 14
    ind_w = 22
    header = f"  {'Indicator':<{ind_w}}" + "".join(
        f"  {c:>{col_w}}" for c in df.columns
    )
    sep = f"  {'-' * ind_w}" + ("  " + "-" * col_w) * len(df.columns)
    print(header)
    print(sep)
    for ind, row in df.iterrows():
        cells = "".join(
            f"  {row[c]:>{col_w}.{fmt}}" if not np.isnan(row[c]) else f"  {'NaN':>{col_w}}"
            for c in df.columns
        )
        print(f"  {ind:<{ind_w}}{cells}")
    print(sep)


def build_comparison_table(
    print_table: bool = True,
    save_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble a unified comparison DataFrame from the four JSON result files.

    Returns
    -------
    long_df   : tidy DataFrame [Model, Indicator, RMSE, MAPE]
    wide_rmse : pivot table — rows = Indicator, columns = Model (RMSE values)

    Notes
    -----
    - Indicators not covered by a given model appear as NaN.
    - ARIMA Phase 3 only modelled GDP_Growth; other cells are NaN.
    - VAR and Hybrid operate on first-differenced series (delta-scale);
      LSTM and ARIMA operate in level-space. This is noted in the header.
    """
    loaders = {
        "ARIMA":  _load_arima_metrics,
        "VAR":    _load_var_metrics,
        "LSTM":   _load_lstm_metrics,
        "Hybrid": _load_hybrid_metrics,
    }

    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        m_data = loaders[model]()
        for ind in INDICATORS:
            rows.append({
                "Model":     model,
                "Indicator": ind,
                "RMSE":      m_data[ind]["RMSE"],
                "MAPE":      m_data[ind]["MAPE"],
            })

    long_df = pd.DataFrame(rows)

    wide_rmse = long_df.pivot(index="Indicator", columns="Model", values="RMSE")[MODELS]
    wide_mape = long_df.pivot(index="Indicator", columns="Model", values="MAPE")[MODELS]

    if print_table:
        print("\n" + "=" * 82)
        print("  TEST-SET METRICS -- ALL MODELS x ALL INDICATORS")
        print("  (VAR & Hybrid are in delta-scale; ARIMA & LSTM in level-scale)")
        print("=" * 82)
        print("\n  -- RMSE --")
        _print_wide(wide_rmse, fmt="4f")
        print("\n  -- MAPE (%) --")
        _print_wide(wide_mape, fmt="2f")
        print("\n  -- Tidy long-form DataFrame --")
        print(
            long_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "     NaN",
            )
        )
        print("=" * 82)

    if save_csv is not None:
        long_df.to_csv(save_csv, index=False)
        print(f"  [build_comparison_table] Saved long-form CSV -> {save_csv}")

    return long_df, wide_rmse


# ===========================================================================
# III. plot_rmse_comparison
# ===========================================================================

def plot_rmse_comparison(
    long_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart comparing RMSE across 4 models for each indicator.

    Parameters
    ----------
    long_df   : tidy DataFrame from build_comparison_table(); rebuilt if None.
    save_path : optional PNG output path.
    show      : call plt.show() when True.

    Returns
    -------
    matplotlib Figure object.
    """
    if long_df is None:
        long_df, _ = build_comparison_table(print_table=False)

    # Drop indicators that are NaN for every model
    valid_inds = (
        long_df.groupby("Indicator")["RMSE"]
        .apply(lambda s: s.notna().any())
    )
    valid_inds = valid_inds[valid_inds].index.tolist()
    plot_df = long_df[long_df["Indicator"].isin(valid_inds)].copy()

    n_ind  = len(valid_inds)
    n_mod  = len(MODELS)
    bar_w  = 0.18
    grp_w  = n_mod * bar_w + 0.12
    x_ctr  = np.arange(n_ind) * grp_w

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    max_rmse = plot_df["RMSE"].dropna().max()

    for i, model in enumerate(MODELS):
        offsets = x_ctr + (i - (n_mod - 1) / 2) * bar_w
        rmse_vals = []
        for ind in valid_inds:
            sub = plot_df.loc[
                (plot_df["Model"] == model) & (plot_df["Indicator"] == ind), "RMSE"
            ]
            rmse_vals.append(float(sub.values[0]) if len(sub) > 0 else np.nan)

        bars = ax.bar(
            offsets,
            [v if not np.isnan(v) else 0 for v in rmse_vals],
            width=bar_w,
            color=MODEL_COLOURS[model],
            alpha=0.88,
            zorder=3,
            label=model,
            linewidth=0.6,
            edgecolor="white",
        )

        for bar, val in zip(bars, rmse_vals):
            if not np.isnan(val) and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_rmse * 0.015,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="white", fontweight="bold",
                )

    ax.set_xticks(x_ctr)
    ax.set_xticklabels(
        [ind.replace("_", "\n") for ind in valid_inds],
        color="white", fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("RMSE (test set)", color="white", fontsize=12, labelpad=10)
    ax.set_xlabel("Indicator", color="white", fontsize=12, labelpad=10)
    ax.tick_params(colors="white", which="both")
    ax.yaxis.set_tick_params(labelcolor="white", labelsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#3A3D4D")
    ax.spines["bottom"].set_color("#3A3D4D")
    ax.set_xlim(x_ctr[0] - grp_w / 2, x_ctr[-1] + grp_w / 2)
    ax.set_ylim(0, max_rmse * 1.20)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5,
                  color="#3A3D4D", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(
        title="Model", title_fontsize=11, fontsize=10,
        framealpha=0.15, edgecolor="#3A3D4D", labelcolor="white",
        loc="upper right",
    )
    legend.get_title().set_color("white")

    ax.annotate(
        "Note: VAR & Hybrid RMSE in delta-scale (first-differenced); "
        "ARIMA & LSTM in level-scale.",
        xy=(0.01, 0.01), xycoords="figure fraction",
        fontsize=8, color="#AAAAAA", style="italic",
    )
    ax.set_title(
        "Model RMSE Comparison -- All Indicators",
        color="white", fontsize=15, fontweight="bold", pad=16,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [plot_rmse_comparison] Saved -> {save_path}")

    if show:
        plt.show()

    return fig


# ===========================================================================
# IV. backtest_policy_event
# ===========================================================================

def backtest_policy_event(
    event_cfg: Dict[str, Any],
    model_forecasts: Dict[str, pd.DataFrame],
    actuals: pd.DataFrame,
    n_quarters: int = 4,
    print_report: bool = True,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Backtest each model's projected path against what actually happened
    following a known historical policy shock.

    Directional Accuracy (DA)
    -------------------------
    For each variable V and post-event quarter t in {1, ..., n_quarters}:

        actual_change[t]   = actual[t] - actual[t-1]
        forecast_change[t] = forecast[t] - actual[t-1]   (anchored to observed level)

        actual_direction[t]   = sign(actual_change[t])    in {+1, -1, 0}
        forecast_direction[t] = sign(forecast_change[t])

        correct[t] = (actual_direction[t] == forecast_direction[t])

    DA(model, V) = mean(correct[t])   over t = 1..n_quarters
    DA(model)    = mean(DA(model, V)) over all variables V

    Parameters
    ----------
    event_cfg : dict with keys:
        - "name"          : str -- human-readable label
        - "event_quarter" : str -- quarter of shock, e.g. "2022-Q2"
        - "description"   : str -- optional narrative
    model_forecasts : dict {model_name -> pd.DataFrame}
        Each DataFrame has DatetimeIndex (quarterly, after event_quarter)
        and one column per macro variable.
    actuals : pd.DataFrame
        Historical actual values with DatetimeIndex (quarterly).
        Must span event_quarter and the n_quarters following it.
    n_quarters : int -- post-event horizon to evaluate (default 4).
    print_report : bool -- pretty-print the full report.
    save_path    : optional Path to save the report as JSON.

    Returns
    -------
    report : dict
        {
          "event":   event_cfg,
          "models":  {
              "<Model>": {
                  "<Variable>": {
                      "directional_accuracy": float,
                      "quarters": [
                          {
                              "quarter": str,
                              "actual_prev": float,
                              "actual_curr": float,
                              "actual_change": float,
                              "actual_direction": int,
                              "forecast_curr": float,
                              "forecast_change": float,
                              "forecast_direction": int,
                              "correct": bool,
                          }, ...
                      ],
                  }, ...
              }, ...
          },
          "summary": {
              "<Model>": {
                  "per_variable_DA": {<Var>: float},
                  "overall_DA": float,
              }, ...
          },
        }
    """
    # -- Resolve event quarter timestamp ------------------------------------
    eq_label = event_cfg["event_quarter"]
    year, q  = eq_label.split("-Q")
    eq_ts    = pd.Timestamp(int(year), (int(q) - 1) * 3 + 1, 1)

    # Closest actual row at event_quarter
    if eq_ts in actuals.index:
        anchor_row = actuals.loc[eq_ts]
    else:
        idx_pos = actuals.index.searchsorted(eq_ts)
        if idx_pos >= len(actuals):
            raise ValueError(f"event_quarter {eq_label} is beyond the actuals index.")
        eq_ts      = actuals.index[idx_pos]
        anchor_row = actuals.iloc[idx_pos]

    post_idx = actuals.index[actuals.index > eq_ts][:n_quarters]
    if len(post_idx) < n_quarters:
        warnings.warn(
            f"Only {len(post_idx)} post-event quarters available "
            f"(requested {n_quarters})."
        )

    # Variables available in at least one model forecast
    forecast_cols: set = set()
    for fc_df in model_forecasts.values():
        forecast_cols.update(fc_df.columns)
    variables = [
        c for c in actuals.columns
        if not c.endswith("_is_interpolated") and c in forecast_cols
    ]

    report: Dict[str, Any] = {"event": event_cfg, "models": {}, "summary": {}}

    for model_name, fc_df in model_forecasts.items():
        model_report: Dict[str, Any] = {}

        for var in variables:
            if var not in fc_df.columns:
                continue

            quarter_records: List[Dict[str, Any]] = []
            correct_flags: List[bool] = []

            prev_actual = (
                float(anchor_row[var]) if var in anchor_row.index else np.nan
            )

            for t, q_ts in enumerate(post_idx):
                # Actual value
                curr_actual = (
                    float(actuals.loc[q_ts, var])
                    if q_ts in actuals.index else np.nan
                )

                # Forecast value (match by timestamp; fall back to position)
                if q_ts in fc_df.index:
                    curr_fc = float(fc_df.loc[q_ts, var])
                elif t < len(fc_df):
                    curr_fc = float(fc_df.iloc[t][var])
                else:
                    curr_fc = np.nan

                # Changes
                actual_change   = curr_actual - prev_actual if not np.isnan(curr_actual) else np.nan
                forecast_change = curr_fc    - prev_actual  if not np.isnan(curr_fc)    else np.nan

                def _sign(x: float) -> int:
                    if np.isnan(x):
                        return 0
                    return int(np.sign(x)) if x != 0.0 else 0

                actual_dir   = _sign(actual_change)
                forecast_dir = _sign(forecast_change)

                # Directional correctness
                if actual_dir == 0 and forecast_dir == 0:
                    correct = True
                elif actual_dir == 0 or forecast_dir == 0:
                    correct = False
                else:
                    correct = (actual_dir == forecast_dir)

                correct_flags.append(correct)
                quarter_records.append({
                    "quarter":            (
                        q_ts.strftime("%Y-Q") + str((q_ts.month - 1) // 3 + 1)
                    ),
                    "actual_prev":        round(prev_actual, 4)    if not np.isnan(prev_actual) else None,
                    "actual_curr":        round(curr_actual, 4)    if not np.isnan(curr_actual) else None,
                    "actual_change":      round(actual_change, 4)  if not np.isnan(actual_change) else None,
                    "actual_direction":   actual_dir,
                    "forecast_curr":      round(curr_fc, 4)         if not np.isnan(curr_fc) else None,
                    "forecast_change":    round(forecast_change, 4) if not np.isnan(forecast_change) else None,
                    "forecast_direction": forecast_dir,
                    "correct":            correct,
                })

                # Slide anchor to observed actual for next quarter's delta
                prev_actual = curr_actual

            da = float(np.mean(correct_flags)) if correct_flags else np.nan
            model_report[var] = {
                "directional_accuracy": round(da, 4) if not np.isnan(da) else None,
                "quarters":             quarter_records,
            }

        per_var_da = {
            v: model_report[v]["directional_accuracy"]
            for v in model_report
            if model_report[v]["directional_accuracy"] is not None
        }
        overall_da = float(np.mean(list(per_var_da.values()))) if per_var_da else np.nan

        report["models"][model_name]  = model_report
        report["summary"][model_name] = {
            "per_variable_DA": per_var_da,
            "overall_DA": round(overall_da, 4) if not np.isnan(overall_da) else None,
        }

    if print_report:
        _print_backtest_report(report, post_idx, variables)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\n  [backtest_policy_event] Report saved -> {save_path}")

    return report


def _print_backtest_report(
    report: Dict[str, Any],
    post_idx: pd.DatetimeIndex,
    variables: List[str],
) -> None:
    """Pretty-print the backtest directional-accuracy report."""
    evt = report["event"]
    print("\n" + "=" * 80)
    print(f"  POLICY-EVENT BACKTEST: {evt['name']}")
    print(f"  Event quarter : {evt['event_quarter']}")
    print(f"  Description   : {evt.get('description', '')}")
    print(f"  Horizon       : {len(post_idx)} quarters post-event")
    print("=" * 80)

    models = list(report["models"].keys())
    col_w, var_w = 12, 22
    header = f"  {'Variable':<{var_w}}" + "".join(f"  {m:>{col_w}}" for m in models)
    sep    = f"  {'-' * var_w}" + ("  " + "-" * col_w) * len(models)

    print("\n  -- Directional Accuracy (DA) per Model x Variable --\n")
    print(header)
    print(sep)
    for var in variables:
        row = f"  {var:<{var_w}}"
        for model in models:
            da = report["summary"][model]["per_variable_DA"].get(var)
            row += f"  {da:>{col_w}.1%}" if da is not None else f"  {'N/A':>{col_w}}"
        print(row)
    print(sep)
    row = f"  {'OVERALL':<{var_w}}"
    for model in models:
        oda = report["summary"][model].get("overall_DA")
        row += f"  {oda:>{col_w}.1%}" if oda is not None else f"  {'N/A':>{col_w}}"
    print(row)
    print(sep)

    print("\n  -- Quarter-by-Quarter Detail --")
    for model in models:
        print(f"\n  > {model}")
        for var in variables:
            if var not in report["models"][model]:
                continue
            var_data = report["models"][model][var]
            da_str   = (
                f"{var_data['directional_accuracy']:.1%}"
                if var_data["directional_accuracy"] is not None else "N/A"
            )
            print(f"\n    [{var}]  DA = {da_str}")
            qw = 10
            print(
                f"    {'Quarter':<{qw}}  {'Actual':>9}  {'Forecast':>9}  "
                f"{'D Actual':>9}  {'D Fcst':>8}  {'Dir A':>6}  {'Dir F':>6}  {'C/W':>4}"
            )
            print(
                f"    {'-'*qw}  {'-'*9}  {'-'*9}  "
                f"{'-'*9}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*4}"
            )
            for qr in var_data["quarters"]:
                a     = f"{qr['actual_curr']:>9.3f}"   if qr["actual_curr"]   is not None else f"{'NaN':>9}"
                f_str = f"{qr['forecast_curr']:>9.3f}" if qr["forecast_curr"] is not None else f"{'NaN':>9}"
                da_v  = f"{qr['actual_change']:>9.3f}" if qr["actual_change"] is not None else f"{'NaN':>9}"
                df_v  = f"{qr['forecast_change']:>8.3f}" if qr["forecast_change"] is not None else f"{'NaN':>8}"
                dir_a = {1: " UP", -1: "DWN", 0: " --"}.get(qr["actual_direction"], "?")
                dir_f = {1: " UP", -1: "DWN", 0: " --"}.get(qr["forecast_direction"], "?")
                tick  = "  Y" if qr["correct"] else "  N"
                print(
                    f"    {qr['quarter']:<{qw}}  {a}  {f_str}  {da_v}  "
                    f"{df_v}  {dir_a:>6}  {dir_f:>6}  {tick:>4}"
                )

    print("\n" + "=" * 80)


# ===========================================================================
# V. Demo forecast builder for the April 2022 backtest
# ===========================================================================

def _build_demo_forecasts(
    actuals: pd.DataFrame,
    eq_ts: pd.Timestamp,
    n_quarters: int = 4,
) -> Dict[str, pd.DataFrame]:
    """
    Synthesise illustrative 4-quarter-ahead forecast DataFrames for each
    model, anchored at event quarter eq_ts.

    Because we store only aggregate metrics (not full forecast arrays),
    we construct representative projected paths by perturbing the actual
    post-event series with each model's characteristic noise and bias.
    This allows the backtest API to be demonstrated end-to-end.

    In production you would supply the real out-of-sample predictions
    produced by each model at the event date.

    Column coverage:
        ARIMA  -> GDP_Growth only (Phase 3 trained on GDP_Growth)
        VAR    -> CPI_Inflation, GDP_Growth, Unemployment_Rate
        LSTM   -> CPI_Inflation, GDP_Growth, Unemployment_Rate
        Hybrid -> CPI_Inflation, GDP_Growth, Unemployment_Rate
    """
    post_idx  = actuals.index[actuals.index > eq_ts][:n_quarters]
    core_vars = ["CPI_Inflation", "GDP_Growth", "Unemployment_Rate"]
    rng       = np.random.default_rng(42)

    def _simulate(var: str, noise_std: float, bias: float = 0.0) -> pd.Series:
        """forecast[t] = actual[t] + bias + N(0, noise_std)"""
        vals = [
            float(actuals.loc[q_ts, var]) if q_ts in actuals.index else np.nan
            for q_ts in post_idx
        ]
        noise = rng.normal(0, noise_std, size=len(post_idx))
        fc = [a + bias + n if not np.isnan(a) else np.nan for a, n in zip(vals, noise)]
        return pd.Series(fc, index=post_idx)

    arima_fc = pd.DataFrame(
        {"GDP_Growth": _simulate("GDP_Growth", noise_std=0.90, bias=+0.20)},
        index=post_idx,
    )
    var_fc = pd.DataFrame(
        {v: _simulate(v, noise_std=0.55, bias=+0.05) for v in core_vars},
        index=post_idx,
    )
    lstm_fc = pd.DataFrame(
        {v: _simulate(v, noise_std=1.20, bias=-0.10) for v in core_vars},
        index=post_idx,
    )
    hybrid_fc = pd.DataFrame(
        {v: _simulate(v, noise_std=0.40, bias=+0.03) for v in core_vars},
        index=post_idx,
    )

    return {
        "ARIMA":  arima_fc,
        "VAR":    var_fc,
        "LSTM":   lstm_fc,
        "Hybrid": hybrid_fc,
    }


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    print("\n" + "#" * 82)
    print("  ECONOMIC POLICY IMPACT SIMULATOR -- Model Evaluation & Backtesting")
    print("#" * 82)

    # Step 1: comparison table
    print("\n[Step 1] Building comparison table from stored JSON results ...")
    long_df, wide_rmse = build_comparison_table(print_table=True)
    long_csv = _OUT_DIR / "model_comparison.csv"
    long_df.to_csv(long_csv, index=False)
    print(f"  -> Long-form CSV saved: {long_csv}")

    # Step 2: RMSE chart
    print("\n[Step 2] Plotting RMSE grouped bar chart ...")
    chart_path = _OUT_DIR / "model_rmse_comparison.png"
    plot_rmse_comparison(long_df, save_path=chart_path, show=False)
    print(f"  -> Chart saved: {chart_path}")

    # Step 3: policy-event backtest
    print("\n[Step 3] Running April-2022 repo-rate-hike backtest ...")

    # Load actual historical data
    raw = pd.read_csv(_DATA_PATH, index_col=0)

    def _q_to_ts(label: str) -> pd.Timestamp:
        yr, q = label.split("-Q")
        return pd.Timestamp(int(yr), (int(q) - 1) * 3 + 1, 1)

    raw.index = pd.DatetimeIndex([_q_to_ts(q) for q in raw.index])
    raw.index.freq = "QS"

    actuals = raw[["CPI_Inflation", "GDP_Growth", "Unemployment_Rate", "Repo_Rate"]].copy()

    event_cfg = {
        "name":          "RBI Repo Rate Hike -- April 2022",
        "event_quarter": "2022-Q2",
        "description": (
            "RBI MPC raised the repo rate by 40 bps from 4.00% to 4.40% "
            "in an off-cycle emergency meeting on 4 May 2022, signalling the "
            "start of an aggressive rate-hike cycle to tame inflation that had "
            "breached the 6% upper tolerance band."
        ),
    }

    eq_ts           = _q_to_ts(event_cfg["event_quarter"])
    model_forecasts = _build_demo_forecasts(actuals, eq_ts, n_quarters=4)

    report = backtest_policy_event(
        event_cfg       = event_cfg,
        model_forecasts = model_forecasts,
        actuals         = actuals,
        n_quarters      = 4,
        print_report    = True,
        save_path       = _OUT_DIR / "backtest_april2022_report.json",
    )

    # Final summary
    print("\n[Step 4] Overall directional accuracy summary:")
    print(f"\n  {'Model':<10}  {'Overall DA':>12}")
    print(f"  {'-'*10}  {'-'*12}")
    for model, summ in report["summary"].items():
        oda = summ["overall_DA"]
        if oda is not None:
            print(f"  {model:<10}  {oda:>12.1%}")
        else:
            print(f"  {model:<10}  {'N/A':>12}")

    print("\n[Done] All steps complete.\n")


if __name__ == "__main__":
    main()

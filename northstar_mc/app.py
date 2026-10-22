"""Streamlit UI: slider scenario + Monte Carlo results.  Run:  streamlit run app.py"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import BENCHMARKS, PROJECTS, CaseData, load_case
from model import ProjectResult, run_all
from shocks import ShockConfig, Shocks, base_shocks
from simulate import SimResults, excel_bytes, run_monte_carlo, tornado

warnings.filterwarnings("ignore", category=UserWarning)  # openpyxl conditional-format notice

COLORS = {"Bison": "#2a78d6", "Fennec": "#eb6834", "Puma": "#1baf7a"}
CEILING_COLOR = "#e34948"
BENCH_LABEL = {"WTI": "WTI", "Brent": "Brent", "HH": "Henry Hub", "TTF": "TTF", "OPIS": "OPIS NGL"}
BUILD_NOTE = "Peak falls in a build year (no production yet): it is driven by build capex, not by Exhibit 5-9 inputs."

st.set_page_config(page_title="NorthStar leverage Monte Carlo", layout="wide")


@st.cache_resource
def get_case() -> CaseData:
    return load_case()


@st.cache_data
def base_ratios() -> dict[str, np.ndarray]:
    case = get_case()
    return {k: r.ratio[0] for k, r in run_all(case, base_shocks(case)).items()}


# ---------- sidebar ----------
def slider_shocks(case: CaseData) -> Shocks:
    sb = st.sidebar
    s = base_shocks(case)
    sb.header("Scenario sliders")
    sb.caption("Prices (multiplier on the Exhibit 9 curve)")
    for b in BENCHMARKS:
        s.price[b][:] = sb.slider(BENCH_LABEL[b], 0.3, 2.0, 1.0, 0.05, key=f"px_{b}")
    sb.caption("Production volume (multiplier on Exhibits 5-7)")
    for p in PROJECTS:
        s.volume[p][:] = sb.slider(p, 0.5, 1.5, 1.0, 0.05, key=f"vol_{p}")
    s.crude_shift[:] = sb.slider("Crude share shift (pp, Exhibit 8)", -10, 10, 0, 1) / 100
    with sb.expander("Extra levers (outside Exhibits 5-9)", expanded=False):
        for p in PROJECTS:
            s.capex[p][:] = 1 + st.slider(f"{p} capex overrun %", -10, 50, 0, 1) / 100
        for p in PROJECTS:
            s.delay[p][:] = st.slider(f"{p} first-production delay (years)", 0, 2, 0)
        s.ebitda_growth[:] = st.slider("Baseline EBITDA growth %", 0.0, 6.0, 3.0, 0.25) / 100
    return s


def mc_config() -> tuple[ShockConfig, bool]:
    sb = st.sidebar
    sb.header("Monte Carlo settings")
    cfg = ShockConfig()
    cfg.n_scenarios = int(sb.number_input("Scenarios", 100, 100_000, 10_000, 1_000))
    cfg.seed = int(sb.number_input("Seed", 0, 10**9, cfg.seed))
    with sb.expander("Distribution σ values", expanded=False):
        cfg.volume_sigma = {p: st.number_input(f"Volume σ {p}", 0.0, 0.5, v, 0.01) for p, v in cfg.volume_sigma.items()}
        cfg.crude_shift_sigma = st.number_input("Crude shift σ (pp)", 0.0, 10.0, 3.0, 0.5) / 100
        cfg.price_vol = {b: st.number_input(f"{BENCH_LABEL[b]} vol", 0.0, 1.0, v, 0.05) for b, v in cfg.price_vol.items()}
        cfg.volume_year_sigma = st.number_input("Year-by-year volume σ", 0.0, 0.2, cfg.volume_year_sigma, 0.01)
        cfg.ebitda_growth_sigma = st.number_input("EBITDA growth σ (pp)", 0.0, 5.0, 1.5, 0.25) / 100
    cfg.volume_year_noise = sb.toggle("Year-by-year volume noise", False)
    cfg.mean_reverting_prices = sb.toggle("Mean-reverting prices (OU, 3y half-life)", False)
    cfg.extra_levers = sb.toggle("Include extra levers (capex, delay, EBITDA growth)", False)
    return cfg, sb.button("Run simulation", type="primary", width="stretch")


# ---------- charts ----------
def ceiling_line(fig: go.Figure, ceiling: float, vertical: bool = False) -> None:
    kw = dict(line_dash="dash", line_color=CEILING_COLOR, line_width=2, annotation_text=f"Ceiling {ceiling:.2f}x")
    if vertical:
        fig.add_vline(x=ceiling, **kw)
    else:
        fig.add_hline(y=ceiling, annotation_position="top left", **kw)


def layout(fig: go.Figure, title: str, height: int = 420, ytitle: str = "Net debt / Adj. EBITDA (x)") -> go.Figure:
    fig.update_layout(title=title, height=height, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.15), yaxis_title=ytitle)
    fig.update_yaxes(tickformat=".2f", gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
    return fig


def scenario_chart(results: dict[str, ProjectResult], base: dict[str, np.ndarray], ceiling: float) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        for y in r.years[r.breach[0]]:
            fig.add_vrect(x0=y - 0.5, x1=y + 0.5, fillcolor=COLORS[name], opacity=0.10, line_width=0)
        fig.add_scatter(x=r.years, y=base[name], name=f"{name} base", mode="lines",
                        line=dict(color=COLORS[name], dash="dot", width=1.5), opacity=0.45, hoverinfo="skip")
        fig.add_scatter(x=r.years, y=r.ratio[0], name=name, mode="lines+markers",
                        line=dict(color=COLORS[name], width=2), marker=dict(size=8),
                        hovertemplate=f"{name}: %{{y:.3f}}x<extra></extra>")
    ceiling_line(fig, ceiling)
    return layout(fig, "Slider scenario: leverage by year (shaded = breach year, dotted = base case)")


def fan_chart(sim: SimResults, name: str, ceiling: float) -> go.Figure:
    d = sim.by_year[sim.by_year.Project == name]
    c = COLORS[name]
    fig = go.Figure()
    fig.add_scatter(x=d.Year, y=d.P95, mode="lines", line=dict(width=0, color=c), name="P95", showlegend=False, hovertemplate="P95 %{y:.3f}x<extra></extra>")
    fig.add_scatter(x=d.Year, y=d.P5, mode="lines", line=dict(width=0, color=c), fill="tonexty",
                    fillcolor=_rgba(c, 0.22), name="P5-P95 band", hovertemplate="P5 %{y:.3f}x<extra></extra>")
    fig.add_scatter(x=d.Year, y=d.P50, mode="lines", line=dict(color=c, width=2), name="P50",
                    hovertemplate="P50 %{y:.3f}x<extra></extra>")
    ceiling_line(fig, ceiling)
    return layout(fig, f"{name}: ratio fan chart", height=360)


def histogram(sim: SimResults, metric: str, ceiling: float) -> go.Figure:
    fig = go.Figure()
    for name, r in sim.results.items():
        x = r.peak if metric == "Peak ratio" else r.post_build_peak
        fig.add_histogram(x=x, name=name, marker_color=COLORS[name], opacity=0.7, xbins=dict(size=0.01),
                          marker_line=dict(color="white", width=0.5),
                          hovertemplate=f"{name}<br>%{{x}}<br>%{{y}} scenarios<extra></extra>")
    fig.update_layout(barmode="overlay")
    ceiling_line(fig, ceiling, vertical=True)
    fig = layout(fig, f"Distribution of {metric.lower()} across scenarios", ytitle="Scenarios")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title=f"{metric} (x)", tickformat=".2f")
    fig.update_yaxes(tickformat=",d")
    return fig


def tornado_chart(df: pd.DataFrame, name: str, metric: str) -> go.Figure:
    lo, hi = (("Peak at P10", "Peak at P90"), "Base peak") if metric == "Peak ratio" else \
             (("Post-build peak at P10", "Post-build peak at P90"), "Base post-build peak")
    (lo_col, hi_col), base_col = lo, hi
    d = df[df.Project == name].copy()
    base = d[base_col].iloc[0]
    d["swing"] = (d[hi_col] - d[lo_col]).abs()
    d = d.sort_values("swing")
    labels = [f"{i}{' (extra)' if e else ''}" for i, e in zip(d.Input, d["Extra lever"])]
    fig = go.Figure()
    for col, lab, color in ((lo_col, "Input at P10", "#6250d6"), (hi_col, "Input at P90", "#eda100")):
        fig.add_bar(y=labels, x=d[col] - base, base=base, orientation="h", name=lab, marker_color=color,
                    customdata=np.c_[d[col], d["P10 value" if col == lo_col else "P90 value"]],
                    hovertemplate="%{y}<br>input = %{customdata[1]:.3f}<br>ratio = %{customdata[0]:.3f}x<extra>" + lab + "</extra>")
    fig.add_vline(x=base, line_color="gray", line_width=1)
    fig.update_layout(barmode="group", bargap=0.3, height=140 + 40 * len(d),
                      title=f"{name} tornado (base {base:.3f}x)",
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    fig.update_xaxes(title=f"{metric} (x)", tickformat=".2f")
    return fig


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# ---------- page ----------
def badges(results: dict[str, ProjectResult]) -> None:
    for col, (name, r) in zip(st.columns(len(results)), results.items()):
        ok = bool(r.passed[0])
        col.metric(f"{name}: {'✅ PASS' if ok else '❌ FAIL'}", f"{r.peak[0]:.3f}x peak in {r.peak_year[0]}")
        col.markdown(f"Post-build peak: **{r.post_build_peak[0]:.3f}x**")
        if r.peak_in_build[0]:
            col.caption(BUILD_NOTE)


SUMMARY_FMT = {"Pass rate": "{:.1%}", "Fail rate": "{:.1%}", "Peak in build year (share)": "{:.1%}",
               "Post-build breach rate": "{:.1%}", "Most common peak year": "{:.0f}", "Most common breach year": "{:.0f}"}


def mc_section(sim: SimResults, tor: pd.DataFrame, ceiling: float) -> None:
    st.subheader(f"Monte Carlo results ({sim.config.n_scenarios:,} scenarios, seed {sim.config.seed})")
    cols = ["Project", "Pass rate", "Peak P50", "Peak P95", "Most common breach year", "Peak in build year (share)",
            "Post-build peak P50", "Post-build peak P95", "Post-build breach rate"]
    st.dataframe(sim.summary[cols].style.format(SUMMARY_FMT | {c: "{:.3f}" for c in cols if "P50" in c or "P95" in c},
                                               na_rep="none"), hide_index=True, width="stretch")
    for name, r in sim.results.items():
        if r.peak_in_build.mean() > 0.5:
            st.caption(f"**{name}**: {r.peak_in_build.mean():.0%} of scenarios peak in a build year. {BUILD_NOTE}")
    for col, name in zip(st.columns(3), PROJECTS):
        col.plotly_chart(fan_chart(sim, name, ceiling), width="stretch")
    metric = st.radio("Metric for histogram and tornado", ["Peak ratio", "Post-build peak"], horizontal=True)
    st.plotly_chart(histogram(sim, metric, ceiling), width="stretch")
    st.caption("Tornado: everything at base, one input moved to its P10 (purple bar) or P90 (gold bar). "
               "Hover a bar for the input value. (extra) = lever outside Exhibits 5-9.")
    for col, name in zip(st.columns(3), PROJECTS):
        col.plotly_chart(tornado_chart(tor, name, metric), width="stretch")
    with st.expander("Probability of breach by year and full summary"):
        st.dataframe(sim.by_year, hide_index=True, width="stretch")
        st.dataframe(sim.summary, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.download_button("Download scenarios (CSV)", sim.scenarios.to_csv(index=False).encode(), "scenarios.csv", "text/csv")
    c2.download_button("Download summary (Excel)", excel_bytes(sim, tor), "summary.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def main() -> None:
    case = get_case()
    ceiling = case.corporate.ceiling
    st.title("NorthStar leverage Monte Carlo")
    st.caption(f"Net debt / Adj. EBITDA must stay at or below {ceiling:.2f}x in every year. "
               "Sliders update the scenario chart live; the Monte Carlo runs the same model function.")
    shocks = slider_shocks(case)
    cfg, run = mc_config()
    results = run_all(case, shocks)
    badges(results)
    st.plotly_chart(scenario_chart(results, base_ratios(), ceiling), width="stretch")
    if run:
        with st.spinner("Running scenarios..."):
            st.session_state["sim"] = run_monte_carlo(cfg, case)
            st.session_state["tornado"] = tornado(cfg, case)
    if "sim" in st.session_state:
        mc_section(st.session_state["sim"], st.session_state["tornado"], ceiling)
    else:
        st.info("Set the Monte Carlo options in the sidebar and press **Run simulation**.")


main()

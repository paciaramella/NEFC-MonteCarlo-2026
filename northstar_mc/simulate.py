"""Run N scenarios and compute the section 5 outputs.

    python simulate.py                 # 10,000 scenarios, defaults, writes outputs/
    python simulate.py --extra-levers  # also vary capex, delay and EBITDA growth
"""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import BENCHMARKS, PROJECTS, CaseData, load_case
from model import ProjectResult, run_all
from shocks import NO_TARIFF, ShockConfig, Shocks, base_shocks, draw_shocks, input_quantiles, set_tariff

PERCENTILES = (5, 25, 50, 75, 95)
EXTRA_KEYS = ("Baseline EBITDA growth", "Opex growth", "Tariff event")


@dataclass
class SimResults:
    config: ShockConfig
    shocks: Shocks
    results: dict[str, ProjectResult]
    summary: pd.DataFrame
    by_year: pd.DataFrame
    scenarios: pd.DataFrame


def _mode(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    vals, counts = np.unique(values, return_counts=True)
    return vals[counts.argmax()]


def summarise(r: ProjectResult) -> dict[str, float]:
    breach_years = np.broadcast_to(r.years, r.ratio.shape)[r.breach]
    row = {"Project": r.name, "Scenarios": len(r.ratio), "Pass rate": r.passed.mean(), "Fail rate": 1 - r.passed.mean()}
    row |= {f"Peak P{q}": np.percentile(r.peak, q) for q in PERCENTILES}
    row["Peak mean"] = r.peak.mean()
    row["Most common peak year"] = _mode(r.peak_year)
    row["Peak in build year (share)"] = r.peak_in_build.mean()
    row |= {f"Post-build peak P{q}": np.percentile(r.post_build_peak, q) for q in PERCENTILES}
    row["Post-build peak mean"] = r.post_build_peak.mean()
    row["Post-build breach rate"] = (r.post_build_peak > r.ceiling).mean()
    row["Most common breach year"] = _mode(breach_years)
    return row


def per_year(r: ProjectResult) -> pd.DataFrame:
    df = pd.DataFrame({"Project": r.name, "Year": r.years})
    for q in (5, 50, 95):
        df[f"P{q}"] = np.percentile(r.ratio, q, axis=0)
    df["Mean"] = r.ratio.mean(axis=0)
    df["P(breach)"] = r.breach.mean(axis=0)
    return df


def scenario_table(s: Shocks, results: dict[str, ProjectResult]) -> pd.DataFrame:
    """One row per scenario per project: inputs drawn and leverage results."""
    frames = []
    for name, r in results.items():
        t = len(r.years)
        df = pd.DataFrame({"Scenario": np.arange(s.n), "Project": name})
        df["Volume mult (avg)"] = s.volume[name][:, :t].mean(axis=1)
        df["Crude share shift (pp)"] = s.crude_shift * 100
        for b in BENCHMARKS:
            df[f"{b} mult (avg)"] = s.price[b][:, :t].mean(axis=1)
        df["Capex mult"] = s.capex[name]
        df["Delay (years)"] = s.delay[name]
        df["Baseline EBITDA growth"] = s.ebitda_growth
        df["Tariff start year"] = np.where(s.tariff_year == NO_TARIFF, np.nan, s.tariff_year)
        df["Opex growth"] = s.opex_growth
        df["Peak ratio"], df["Peak year"] = r.peak, r.peak_year
        df["Post-build peak"] = r.post_build_peak
        df["Result"] = np.where(r.passed, "PASS", "FAIL")
        ratios = pd.DataFrame(r.ratio, columns=[f"Ratio {y}" for y in r.years])
        frames.append(pd.concat([df, ratios], axis=1))
    return pd.concat(frames, ignore_index=True)


def run_monte_carlo(cfg: ShockConfig, case: CaseData | None = None) -> SimResults:
    case = case or load_case()
    s = draw_shocks(case, cfg)
    results = run_all(case, s)
    return SimResults(
        config=cfg,
        shocks=s,
        results=results,
        summary=pd.DataFrame([summarise(r) for r in results.values()]),
        by_year=pd.concat([per_year(r) for r in results.values()], ignore_index=True),
        scenarios=scenario_table(s, results),
    )


def _set_input(s: Shocks, key: str, value: float, cfg: ShockConfig) -> None:
    kind, _, target = key.partition(" ")
    if kind == "Volume":
        s.volume[target][:] = value
    elif kind == "Price":
        s.price[target][:] = value
    elif kind == "Capex":
        s.capex[target][:] = value
    elif kind == "Delay":
        s.delay[target][:] = int(value)
    elif key == "Crude share shift":
        s.crude_shift[:] = value
    elif key == "Baseline EBITDA growth":
        s.ebitda_growth[:] = value
    elif key == "Opex growth":
        s.opex_growth[:] = value
    elif key == "Tariff event" and value:
        set_tariff(s, cfg, cfg.tariff_window[0], cfg.tariff_rate)


def _relevant(key: str, project: str, case: CaseData) -> bool:
    kind, _, target = key.partition(" ")
    if kind in ("Volume", "Capex", "Delay"):
        return target == project
    if kind == "Price":
        b = case.projects[project].benchmarks
        return target in b or (target == "OPIS" and "EU_NGL" in b)
    return True


def tornado(cfg: ShockConfig, case: CaseData | None = None, lo_q: float = 0.10, hi_q: float = 0.90) -> pd.DataFrame:
    """Base case, one input at a time moved to its P10 and P90; change in peak ratios."""
    case = case or load_case()
    base = run_all(case, base_shocks(case))
    lows, highs = input_quantiles(cfg, lo_q), input_quantiles(cfg, hi_q)
    rows = []
    for key in lows:
        extra = key.split(" ")[0] in ("Capex", "Delay") or key in EXTRA_KEYS
        for project in PROJECTS:
            if not _relevant(key, project, case):
                continue
            out = {}
            for side, value in (("low", lows[key]), ("high", highs[key])):
                s = base_shocks(case)
                _set_input(s, key, value, cfg)
                r = run_all(case, s)[project]
                out[side] = (value, r.peak[0], r.post_build_peak[0])
            b = base[project]
            rows.append({
                "Project": project, "Input": key, "Extra lever": extra,
                "P10 value": out["low"][0], "P90 value": out["high"][0],
                "Base peak": b.peak[0], "Peak at P10": out["low"][1], "Peak at P90": out["high"][1],
                "Base post-build peak": b.post_build_peak[0],
                "Post-build peak at P10": out["low"][2], "Post-build peak at P90": out["high"][2],
            })
    df = pd.DataFrame(rows)
    df["Peak swing"] = (df["Peak at P90"] - df["Peak at P10"]).abs()
    df["Post-build swing"] = (df["Post-build peak at P90"] - df["Post-build peak at P10"]).abs()
    return df


def excel_bytes(sim: SimResults, tornado_df: pd.DataFrame | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        sim.summary.to_excel(xl, sheet_name="Summary", index=False)
        sim.by_year.to_excel(xl, sheet_name="By year", index=False)
        if tornado_df is not None:
            tornado_df.to_excel(xl, sheet_name="Tornado", index=False)
        pd.DataFrame(
            [(k, str(v)) for k, v in vars(sim.config).items()], columns=["Setting", "Value"]
        ).to_excel(xl, sheet_name="Settings", index=False)
    return buf.getvalue()


def print_summary(sim: SimResults) -> None:
    cols = ["Project", "Pass rate", "Peak P5", "Peak P50", "Peak P95", "Peak mean", "Most common peak year",
            "Peak in build year (share)", "Post-build peak P50", "Post-build peak P95", "Post-build breach rate", "Most common breach year"]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(sim.summary[cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--scenarios", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--extra-levers", action="store_true")
    ap.add_argument("--tariff", action="store_true", help="tariff event risk on steel / materials")
    ap.add_argument("--opex-growth", action="store_true", help="unexpected opex escalation")
    ap.add_argument("--year-noise", action="store_true")
    ap.add_argument("--mean-reverting", action="store_true")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    cfg = ShockConfig(n_scenarios=a.scenarios, seed=a.seed, extra_levers=a.extra_levers,
                      volume_year_noise=a.year_noise, mean_reverting_prices=a.mean_reverting,
                      tariff_event=a.tariff, opex_growth=a.opex_growth)
    sim = run_monte_carlo(cfg)
    tor = tornado(cfg)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    sim.scenarios.to_csv(out / "scenarios.csv", index=False)
    (out / "summary.xlsx").write_bytes(excel_bytes(sim, tor))
    print_summary(sim)
    print(f"\nWrote {out / 'scenarios.csv'} and {out / 'summary.xlsx'}")


if __name__ == "__main__":
    main()

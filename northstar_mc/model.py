"""Vectorised project model and leverage test (brief section 3).

Replicates rows 17-40 (project operating model) and 54-65 (leverage test) of the
Bison / Fennec / Puma DCF tabs. Arrays are [n_scenarios, n_years]; one scenario
is just n = 1, so the slider view and the Monte Carlo run the same code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_loader import BASE_YEAR, BASELINE_PRICE_MAP, FIRST_YEAR, CaseData, Project
from shocks import Shocks

RATIO_CAP = 10.0  # ratio reported when total EBITDA <= 0 (no meaningful multiple; always a breach)


@dataclass
class ProjectResult:
    name: str
    years: np.ndarray
    production: np.ndarray       # MBOED after volume shock and delay
    revenue: np.ndarray
    ebitda: np.ndarray           # project EBITDA
    unlevered_fcf: np.ndarray
    total_ebitda: np.ndarray     # baseline + project
    net_debt: np.ndarray         # ending net debt
    ratio: np.ndarray            # net debt / adj. EBITDA
    headroom: np.ndarray
    ceiling: float

    @property
    def breach(self) -> np.ndarray:
        return self.ratio > self.ceiling

    @property
    def passed(self) -> np.ndarray:
        return ~self.breach.any(axis=1)

    @property
    def peak(self) -> np.ndarray:
        return self.ratio.max(axis=1)

    @property
    def peak_year(self) -> np.ndarray:
        return self.years[self.ratio.argmax(axis=1)]

    @property
    def producing(self) -> np.ndarray:
        """True from the first producing year onward."""
        return np.cumsum(self.production > 0, axis=1) > 0

    @property
    def post_build_peak(self) -> np.ndarray:
        return np.where(self.producing, self.ratio, -np.inf).max(axis=1)

    @property
    def peak_in_build(self) -> np.ndarray:
        """Peak falls before first production (driven by build capex, not Exhibits 5-9)."""
        return ~self.producing[np.arange(len(self.ratio)), self.ratio.argmax(axis=1)]


def shift_right(a: np.ndarray, delay: np.ndarray) -> np.ndarray:
    """Shift each row right by its delay (0-2 years), truncating the tail."""
    out = a.copy()
    for d in np.unique(delay):
        if d:
            rows = delay == d
            out[rows] = 0.0
            out[rows, d:] = a[rows, :-d]
    return out


def project_prices(case: CaseData, p: Project, s: Shocks) -> tuple[np.ndarray, ...]:
    """Crude, gas ($/MMBtu) and NGL price paths for the project's benchmarks."""
    t = p.n_years
    curve = lambda b: case.prices[b][:t]  # noqa: E731
    mult = lambda b: s.price[b][:, :t]  # noqa: E731
    crude_b, gas_b, ngl_b = p.benchmarks
    crude, gas = curve(crude_b) * mult(crude_b), curve(gas_b) * mult(gas_b)
    if ngl_b == "EU_NGL":  # Europe NGL = OPIS x multiplier + spread
        ngl = curve("OPIS") * mult("OPIS") + case.eu_ngl_spread[:t]
    else:
        ngl = curve(ngl_b) * mult(ngl_b)
    return crude, gas, ngl


def opex_index(p: Project, s: Shocks) -> np.ndarray:
    """Unexpected opex escalation, compounding from 2027 (1.0 in the base case)."""
    return (1 + s.opex_growth[:, None]) ** (p.years[None, :] - FIRST_YEAR)


def operating_model(case: CaseData, p: Project, s: Shocks) -> dict[str, np.ndarray]:
    """DCF rows 17-40: production through unlevered FCF."""
    c, t, n = case.constants, p.n_years, s.n
    delay = s.delay[p.name]
    production = shift_right(p.mboed[None, :] * s.volume[p.name][:, :t], delay)
    volume = production * 1000 * c.days / 1e6
    crude_sh, gas_sh, ngl_sh = (shift_right(np.broadcast_to(x, (n, t)), delay)
                                for x in (p.crude_share, p.gas_share, p.ngl_share))
    crude_px, gas_px, ngl_px = project_prices(case, p, s)
    gas_boe = gas_px * c.mcf_per_boe * c.mmbtu_per_mcf
    revenue = volume * (crude_sh * crude_px + gas_sh * gas_boe + ngl_sh * ngl_px)
    tariff_on = p.years[None, :] >= s.tariff_year[:, None]
    unit_cost = p.unit_cost * opex_index(p, s) * (1 + s.tariff_opex_uplift[:, None] * tariff_on)
    ebitda = revenue * (1 - p.royalty) - volume * unit_cost

    cost = p.dev_cost * s.capex[p.name][:, None]
    idx = np.arange(t)[None, :]
    capex = np.where(idx < p.dev_years, cost / p.dev_years, 0.0) * (1 + s.tariff_capex_uplift[:, None] * tariff_on)
    depreciation = np.where(production > 0, capex.sum(axis=1, keepdims=True) / p.dep_life, 0.0)
    tax = case.corporate.tax_rate * (ebitda - depreciation)  # negative = credit, as in workbook
    decom = np.where(idx == p.dev_years + p.dep_life - 1, p.decom, 0.0) * np.ones((n, 1))
    fcf = ebitda - tax - capex - decom
    return {"production": production, "revenue": revenue, "ebitda": ebitda, "fcf": fcf}


def baseline_price_impact(case: CaseData, p: Project, s: Shocks) -> np.ndarray:
    """Change in NorthStar's baseline EBITDA when prices leave the Exhibit 9 curve.

    Exhibit 2 2026 revenue by product x the price deviation of its benchmarks, grown with
    baseline EBITDA. Zero unless baseline financials are factored in (passthrough > 0).
    """
    t = p.n_years
    dev = sum(case.corporate.revenue_mix[k] * w * (s.price[b][:, :t] - 1)
              for k, weights in BASELINE_PRICE_MAP.items() for b, w in weights.items())
    growth = (1 + s.ebitda_growth[:, None]) ** (p.years[None, :] - BASE_YEAR)
    return s.baseline_passthrough * growth * dev


def leverage_test(case: CaseData, p: Project, s: Shocks, ebitda: np.ndarray, fcf: np.ndarray) -> dict[str, np.ndarray]:
    """DCF rows 54-65, plus the price impact on the baseline business when it is factored in."""
    corp = case.corporate
    years = p.years[None, :]
    impact = baseline_price_impact(case, p, s)
    baseline = corp.adj_ebitda_0 * (1 + s.ebitda_growth[:, None]) ** (years - BASE_YEAR) + impact
    total_ebitda = baseline + ebitda
    base_fcf = corp.base_fcf * (1 + corp.fcf_growth) ** (years - FIRST_YEAR) + (1 - corp.tax_rate) * impact
    dividends = corp.dividends_0 * (1 + corp.dividend_growth) ** (years - BASE_YEAR)
    net_debt = corp.net_debt_0 + np.cumsum(-base_fcf + dividends - fcf, axis=1)
    ratio = np.divide(net_debt, total_ebitda, out=np.full_like(net_debt, RATIO_CAP), where=total_ebitda > 0)
    return {
        "total_ebitda": total_ebitda,
        "net_debt": net_debt,
        "ratio": np.minimum(ratio, RATIO_CAP),
        "headroom": corp.ceiling * total_ebitda - net_debt,
    }


def run_project(case: CaseData, name: str, s: Shocks) -> ProjectResult:
    p = case.projects[name]
    op = operating_model(case, p, s)
    lev = leverage_test(case, p, s, op["ebitda"], op["fcf"])
    return ProjectResult(
        name=name,
        years=p.years,
        production=op["production"],
        revenue=op["revenue"],
        ebitda=op["ebitda"],
        unlevered_fcf=op["fcf"],
        total_ebitda=lev["total_ebitda"],
        net_debt=lev["net_debt"],
        ratio=lev["ratio"],
        headroom=lev["headroom"],
        ceiling=case.corporate.ceiling,
    )


def run_all(case: CaseData, s: Shocks) -> dict[str, ProjectResult]:
    return {name: run_project(case, name, s) for name in case.projects}

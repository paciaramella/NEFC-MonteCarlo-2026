"""Read the NorthStar case workbook cells listed in the brief into dataclasses.

Every number the model uses comes from the workbook (cached values, data_only=True),
so the simulation follows the workbook if an exhibit changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.utils import column_index_from_string, get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

WORKBOOK_NAME = "2026 NorthStarEnergy_Case_Exhibits_with_Puma_Analysis.xlsx"
FIRST_YEAR = 2027
BASE_YEAR = 2026

PROJECTS = ("Bison", "Fennec", "Puma")
BENCHMARKS = ("WTI", "Brent", "HH", "TTF", "OPIS")

# Exhibit 9 rows per benchmark. "EU_NGL" is OPIS + spread; the spread is read from row 10.
PRICE_ROWS = {"WTI": 5, "HH": 6, "OPIS": 7, "Brent": 8, "TTF": 9, "EU_NGL": 10}

# Price benchmark each project sells into (DCF tabs rows 22-24): (crude, gas, ngl).
PRICE_MAP = {
    "Bison": ("WTI", "HH", "OPIS"),
    "Fennec": ("Brent", "TTF", "EU_NGL"),
    "Puma": ("Brent", "HH", "OPIS"),
}

# [ASSUMPTION] benchmarks NorthStar's existing business sells into, weight per Exhibit 2 revenue line.
# Used only when baseline financials are factored in. Exhibit 2 rows 6-8 give the revenue split.
BASELINE_PRICE_MAP = {
    "oil": {"WTI": 0.5, "Brent": 0.5},
    "gas": {"HH": 1.0},
    "ngl": {"OPIS": 1.0},
}
REVENUE_ROWS = {"oil": 6, "gas": 7, "ngl": 8}

# Production sheet + last column; Exhibit 8 crude/gas/NGL rows; Exhibit 1 first assumption row.
PRODUCTION_SHEETS = {"Bison": ("Exhibit 5", "N"), "Fennec": ("Exhibit 6", "U"), "Puma": ("Exhibit 7", "AG")}
MIX_ROWS = {"Bison": (5, 6, 7), "Fennec": (8, 9, 10), "Puma": (11, 12, 13)}
ASSUMPTION_ROW = {"Bison": 11, "Fennec": 22, "Puma": 33}


@dataclass(frozen=True)
class Project:
    name: str
    years: np.ndarray          # calendar years, 2027..last
    mboed: np.ndarray          # Exhibit 5/6/7
    crude_share: np.ndarray    # Exhibit 8, 0 in pre-production years
    gas_share: np.ndarray
    ngl_share: np.ndarray
    dev_cost: float            # $MM
    royalty: float             # % of revenue
    opex: float                # $/boe
    overhead: float
    midstream: float
    dep_life: int              # years
    decom: float               # $MM, undiscounted
    dev_years: int
    benchmarks: tuple[str, str, str]

    @property
    def n_years(self) -> int:
        return len(self.years)

    @property
    def unit_cost(self) -> float:
        return self.opex + self.overhead + self.midstream


@dataclass(frozen=True)
class Corporate:
    net_debt_0: float          # Exhibit 2 E73
    adj_ebitda_0: float        # Exhibit 2 E76
    dividends_0: float         # Exhibit 2 E64, stored positive here
    ebitda_growth: float       # Exhibit 12 D5
    fcf_growth: float          # D6
    dividend_growth: float     # D7
    base_fcf: float            # D8, 2027 value
    ceiling: float             # D9, "1.75x"
    tax_rate: float            # Exhibit 1 C6
    revenue_mix: dict[str, float]  # Exhibit 2 E6:E8, 2026 oil / gas / NGL revenue


@dataclass(frozen=True)
class Constants:
    mcf_per_boe: float
    mmbtu_per_mcf: float
    days: float


@dataclass(frozen=True)
class CaseData:
    projects: dict[str, Project]
    corporate: Corporate
    constants: Constants
    price_years: np.ndarray            # 2027..last Exhibit 9 year
    prices: dict[str, np.ndarray]      # base curves by benchmark (incl. EU_NGL)
    eu_ngl_spread: np.ndarray          # EU_NGL - OPIS per year

    @property
    def max_years(self) -> int:
        return max(p.n_years for p in self.projects.values())


def find_workbook(path: str | os.PathLike | None = None) -> Path:
    """Explicit path, then $NORTHSTAR_WORKBOOK, then this folder and its two parents."""
    if path:
        return Path(path)
    if os.environ.get("NORTHSTAR_WORKBOOK"):
        return Path(os.environ["NORTHSTAR_WORKBOOK"])
    here = Path(__file__).resolve().parent
    for folder in (here, here.parent, here.parent.parent):
        if (folder / WORKBOOK_NAME).exists():
            return folder / WORKBOOK_NAME
    raise FileNotFoundError(
        f"Could not find '{WORKBOOK_NAME}'. Put it next to this folder or set NORTHSTAR_WORKBOOK."
    )


def _row(ws: Worksheet, row: int, first_col: str, n: int) -> np.ndarray:
    """n cells across a row starting at first_col, blanks read as 0."""
    start = column_index_from_string(first_col)
    vals = [ws.cell(row, start + i).value for i in range(n)]
    return np.array([float(v) if isinstance(v, (int, float)) else 0.0 for v in vals])


def _year_count(ws: Worksheet, row: int, first_col: str) -> int:
    start, n = column_index_from_string(first_col), 0
    while isinstance(ws.cell(row, start + n).value, (int, float)):
        n += 1
    return n


def _load_project(wb: openpyxl.Workbook, name: str) -> Project:
    sheet, last_col = PRODUCTION_SHEETS[name]
    ws = wb[sheet]
    n = column_index_from_string(last_col) - column_index_from_string("D") + 1
    years = _row(ws, 4, "D", n).astype(int)
    crude_r, gas_r, ngl_r = MIX_ROWS[name]
    mix = wb["Exhibit 8"]
    a = wb["Exhibit 1"]
    r = ASSUMPTION_ROW[name]
    val = lambda off: float(a[f"C{r + off}"].value)  # noqa: E731
    return Project(
        name=name,
        years=years,
        mboed=_row(ws, 5, "D", n),
        crude_share=_row(mix, crude_r, "E", n),
        gas_share=_row(mix, gas_r, "E", n),
        ngl_share=_row(mix, ngl_r, "E", n),
        dev_cost=val(0) * (1000.0 if "billion" in str(a[f"D{r}"].value).lower() else 1.0),
        royalty=val(1),
        opex=val(2),
        overhead=val(3),
        dep_life=int(val(4)),
        midstream=val(5),
        decom=val(6),
        dev_years=int(val(7)),
        benchmarks=PRICE_MAP[name],
    )


def _load_corporate(wb: openpyxl.Workbook) -> Corporate:
    e2, e12 = wb["Exhibit 2"], wb["Exhibit 12"]
    return Corporate(
        net_debt_0=float(e2["E73"].value),
        adj_ebitda_0=float(e2["E76"].value),
        dividends_0=-float(e2["E64"].value),
        ebitda_growth=float(e12["D5"].value),
        fcf_growth=float(e12["D6"].value),
        dividend_growth=float(e12["D7"].value),
        base_fcf=float(e12["D8"].value),
        ceiling=float(str(e12["D9"].value).lower().replace("x", "").strip()),
        tax_rate=float(wb["Exhibit 1"]["C6"].value),
        revenue_mix={k: float(e2[f"E{r}"].value) for k, r in REVENUE_ROWS.items()},
    )


def _load_prices(wb: openpyxl.Workbook) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ws = wb["Exhibit 9"]
    n = _year_count(ws, 4, "F")
    years = _row(ws, 4, "F", n).astype(int)
    return years, {k: _row(ws, r, "F", n) for k, r in PRICE_ROWS.items()}


@lru_cache(maxsize=4)
def load_case(path: str | None = None) -> CaseData:
    wb = openpyxl.load_workbook(find_workbook(path), data_only=True)
    c = wb["WACC & Inputs"]
    price_years, prices = _load_prices(wb)
    projects = {name: _load_project(wb, name) for name in PROJECTS}
    for p in projects.values():
        if p.years[0] != FIRST_YEAR or p.years[-1] > price_years[-1]:
            raise ValueError(f"{p.name}: production years fall outside the Exhibit 9 price curve")
    return CaseData(
        projects=projects,
        corporate=_load_corporate(wb),
        constants=Constants(float(c["C6"].value), float(c["C7"].value), float(c["C8"].value)),
        price_years=price_years,
        prices=prices,
        eu_ngl_spread=prices["EU_NGL"] - prices["OPIS"],
    )

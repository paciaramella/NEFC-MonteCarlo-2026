"""Factor baseline financials toggle: prices also move NorthStar's own EBITDA and FCF."""
import numpy as np
import pytest

from data_loader import BASE_YEAR, FIRST_YEAR, load_case
from model import RATIO_CAP, run_all
from shocks import ShockConfig, base_shocks
from simulate import run_monte_carlo

N = 2_000


@pytest.fixture(scope="module")
def case():
    return load_case()


@pytest.fixture(scope="module")
def sim_off(case):
    return run_monte_carlo(ShockConfig(n_scenarios=N, baseline_financials=False), case)


@pytest.fixture(scope="module")
def sim_on(case):
    return run_monte_carlo(ShockConfig(n_scenarios=N, baseline_financials=True), case)


def _plan(case, years):
    """Exhibit 12 baseline EBITDA and base FCF, before any price impact."""
    c = case.corporate
    return c.adj_ebitda_0 * (1 + c.ebitda_growth) ** (years - BASE_YEAR), c.base_fcf * (1 + c.fcf_growth) ** (years - FIRST_YEAR)


def test_toggle_off_baseline_ignores_prices(case, sim_off):
    c = case.corporate
    assert sim_off.shocks.baseline_passthrough == 0.0
    assert sim_off.shocks.price["Brent"][:, 0].std() > 0.2  # prices really do vary across scenarios
    for name, r in sim_off.results.items():
        ebitda_plan, fcf_plan = _plan(case, r.years)
        # baseline EBITDA is the Exhibit 12 path in every scenario, whatever prices did
        assert np.allclose(r.total_ebitda - r.ebitda, ebitda_plan)
        dividends = c.dividends_0 * (1 + c.dividend_growth) ** (r.years - BASE_YEAR)
        assert np.allclose(r.net_debt, c.net_debt_0 + np.cumsum(-fcf_plan + dividends - r.unlevered_fcf, axis=1))
    # with the baseline fixed, build-year peaks cannot move: Exhibit 5-9 inputs never flip pass/fail
    base = run_all(case, base_shocks(case))
    for name, r in sim_off.results.items():
        assert np.allclose(r.peak, base[name].peak[0])
    assert sim_off.summary.set_index("Project")["Pass rate"].to_dict() == {"Bison": 1.0, "Fennec": 1.0, "Puma": 0.0}


def test_toggle_on_baseline_moves_with_prices(case, sim_off, sim_on):
    c, s = case.corporate, sim_on.shocks
    assert s.baseline_passthrough == 1.0
    assert all(np.array_equal(s.price[b], sim_off.shocks.price[b]) for b in s.price)  # same draws as toggle off
    rev = c.revenue_mix
    for name, r in sim_on.results.items():
        off, t = sim_off.results[name], len(r.years)
        px = {b: s.price[b][:, :t] for b in s.price}
        # hand-built: Exhibit 2 revenue x price deviation (oil on 50/50 WTI-Brent), grown with baseline EBITDA
        impact = (rev["oil"] * (0.5 * (px["WTI"] - 1) + 0.5 * (px["Brent"] - 1))
                  + rev["gas"] * (px["HH"] - 1) + rev["ngl"] * (px["OPIS"] - 1)) * (1 + c.ebitda_growth) ** (r.years - BASE_YEAR)
        assert np.allclose(r.ebitda, off.ebitda) and np.allclose(r.unlevered_fcf, off.unlevered_fcf)  # project unchanged
        assert np.allclose(r.total_ebitda - off.total_ebitda, impact)
        assert np.allclose(r.net_debt - off.net_debt, -np.cumsum((1 - c.tax_rate) * impact, axis=1))
        # low baseline-weighted prices now raise the peak; high prices lower it
        brent = s.price["Brent"][:, 0]
        assert r.peak[brent < 0.8].mean() > off.peak[brent < 0.8].mean()
        assert r.peak[brent > 1.2].mean() < off.peak[brent > 1.2].mean()
    rates = sim_on.summary.set_index("Project")["Pass rate"]
    assert rates["Bison"] < 1.0 and rates["Fennec"] < 1.0 and rates["Puma"] > 0.0  # prices now flip pass/fail

    # at base prices the toggle changes nothing: the workbook base case still holds
    on_cfg = ShockConfig(baseline_financials=True)
    base_on, base_off = run_all(case, base_shocks(case, cfg=on_cfg)), run_all(case, base_shocks(case))
    assert all(np.allclose(base_on[k].ratio, base_off[k].ratio) for k in base_on)
    # a price collapse wipes out total EBITDA: ratio is capped, not negative, and the test fails
    crash = base_shocks(case, cfg=on_cfg)
    for b in crash.price:
        crash.price[b][:] = 0.3
    for r in run_all(case, crash).values():
        assert (r.total_ebitda[0] <= 0).any()
        assert (r.ratio[0][r.total_ebitda[0] <= 0] == RATIO_CAP).all() and not r.passed[0]

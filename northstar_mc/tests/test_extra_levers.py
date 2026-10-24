"""Tariff event and unexpected opex growth levers."""
import numpy as np
import pytest

from data_loader import load_case
from model import run_all
from shocks import ShockConfig, base_shocks, draw_shocks, set_tariff


@pytest.fixture(scope="module")
def case():
    return load_case()


def test_tariff_raises_build_capex(case):
    cfg = ShockConfig()
    base, s = run_all(case, base_shocks(case)), base_shocks(case)
    set_tariff(s, cfg, 2027, 0.25)
    hit = run_all(case, s)
    extra = 2000 * cfg.capex_material_share * 0.25  # Bison builds in 2027
    assert base["Bison"].net_debt[0, 0] + extra == pytest.approx(hit["Bison"].net_debt[0, 0])
    assert hit["Fennec"].peak[0] > base["Fennec"].peak[0]


def test_tariff_after_build_only_hits_opex(case):
    cfg = ShockConfig()
    s = base_shocks(case)
    set_tariff(s, cfg, 2035, 0.25)
    base, hit = run_all(case, base_shocks(case))["Puma"], run_all(case, s)["Puma"]
    assert np.allclose(base.ratio[0, :8], hit.ratio[0, :8])
    assert (hit.ebitda[0, 8:] < base.ebitda[0, 8:]).all()


def test_opex_growth_lowers_ebitda(case):
    s = base_shocks(case)
    s.opex_growth[:] = 0.03
    base, hit = run_all(case, base_shocks(case))["Bison"], run_all(case, s)["Bison"]
    assert (hit.ebitda[0, 1:] < base.ebitda[0, 1:]).all()
    assert hit.post_build_peak[0] > base.post_build_peak[0]


def test_draws_switched_off_by_default(case):
    s = draw_shocks(case, ShockConfig(n_scenarios=500))
    assert (s.tariff_capex_uplift == 0).all() and (s.opex_growth == 0).all()
    on = draw_shocks(case, ShockConfig(n_scenarios=20_000, tariff_event=True, opex_growth=True))
    assert 0.28 < (on.tariff_year < 9999).mean() < 0.32
    # turning the new levers on leaves every other draw unchanged
    assert np.array_equal(draw_shocks(case, ShockConfig(n_scenarios=20_000)).price["Brent"], on.price["Brent"])

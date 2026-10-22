"""Deterministic base case must match the workbook DCF tabs before any simulation."""
import numpy as np
import openpyxl
import pytest

from data_loader import find_workbook, load_case
from model import run_all
from shocks import base_shocks

# Brief section 3: peak ratio, peak year, 2027-2032 ratios (4 dp as given).
EXPECTED = {
    "Bison": (1.636311, 2027, [1.6363, 1.4649, 1.3721, 1.2808, 1.1946, 1.1131]),
    "Fennec": (1.669736, 2029, [1.6152, 1.6437, 1.6697, 1.3683, 1.2363, 1.1111]),
    "Puma": (2.090148, 2031, [1.6996, 1.8074, 1.9082, 2.0023, 2.0901, 1.4882]),
}


@pytest.fixture(scope="module")
def results():
    case = load_case()
    return run_all(case, base_shocks(case))


@pytest.mark.parametrize("name", EXPECTED)
def test_peak_and_early_years(results, name):
    peak, year, ratios = EXPECTED[name]
    r = results[name]
    assert r.peak[0] == pytest.approx(peak, abs=1e-6)
    assert r.peak_year[0] == year
    assert np.allclose(r.ratio[0, :6], ratios, atol=5e-5)


@pytest.mark.parametrize("name", EXPECTED)
def test_every_year_matches_workbook(results, name):
    """Compare every year to the cached DCF tab rows 34, 40, 61, 62 to 1e-6."""
    ws = openpyxl.load_workbook(find_workbook(), data_only=True)[f"{name} DCF"]
    r = results[name]
    n = len(r.years)
    row = lambda i: np.array([ws.cell(i, 4 + j).value for j in range(n)], dtype=float)  # noqa: E731
    assert np.allclose(r.ebitda[0], row(34), atol=1e-6)
    assert np.allclose(r.unlevered_fcf[0], row(40), atol=1e-6)
    assert np.allclose(r.net_debt[0], row(61), atol=1e-6)
    assert np.allclose(r.ratio[0], row(62), atol=1e-6)


def test_spot_values(results):
    assert results["Bison"].ebitda[0, 1] == pytest.approx(680.99279, abs=1e-5)
    assert results["Puma"].unlevered_fcf[0, 5] == pytest.approx(3388.26161, abs=1e-5)


def test_pass_fail(results):
    assert results["Bison"].passed[0] and results["Fennec"].passed[0]
    assert not results["Puma"].passed[0]
    puma = results["Puma"]
    assert list(puma.years[puma.breach[0]]) == [2028, 2029, 2030, 2031]

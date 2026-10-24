"""Random draws for the Monte Carlo (brief section 4).

Every [ASSUMPTION] in the brief is a field on ShockConfig. All standard normals are
drawn every run in a fixed order, so switching a lever on or off never reshuffles the
draws used by the other levers, and all three projects share the same price draws
(common random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from data_loader import BENCHMARKS, PROJECTS, CaseData

NO_TARIFF = 9999  # tariff_year value meaning no tariff in this scenario

# [ASSUMPTION] price correlation, order = BENCHMARKS (WTI, Brent, HH, TTF, OPIS)
DEFAULT_CORRELATION = np.array([
    [1.00, 0.95, 0.30, 0.30, 0.70],
    [0.95, 1.00, 0.30, 0.35, 0.70],
    [0.30, 0.30, 1.00, 0.40, 0.40],
    [0.30, 0.35, 0.40, 1.00, 0.35],
    [0.70, 0.70, 0.40, 0.35, 1.00],
])


@dataclass
class ShockConfig:
    n_scenarios: int = 10_000
    seed: int = 2026
    # A. production volume
    volume_sigma: dict[str, float] = field(default_factory=lambda: {"Bison": 0.10, "Fennec": 0.12, "Puma": 0.15})
    volume_year_noise: bool = False
    volume_year_sigma: float = 0.03
    volume_clip: tuple[float, float] = (0.5, 1.5)
    # B. product mix (percentage points as fractions)
    crude_shift_sigma: float = 0.03
    crude_shift_clip: float = 0.10
    # C. prices
    price_vol: dict[str, float] = field(default_factory=lambda: {"WTI": 0.25, "Brent": 0.25, "HH": 0.35, "TTF": 0.40, "OPIS": 0.25})
    price_correlation: np.ndarray = field(default_factory=lambda: DEFAULT_CORRELATION.copy())
    mean_reverting_prices: bool = False
    price_half_life: float = 3.0
    price_clip: tuple[float, float] = (0.3, 2.5)
    # D. extra levers outside Exhibits 5-9
    extra_levers: bool = False
    capex_triangular: tuple[float, float, float] = (0.95, 1.05, 1.40)  # min, mode, max
    delay_probs: tuple[float, float, float] = (0.70, 0.20, 0.10)       # 0, 1, 2 years
    ebitda_growth_mean: float = 0.03
    ebitda_growth_sigma: float = 0.015
    # E. tariff event and unexpected opex growth (extra levers, each with its own toggle)
    tariff_event: bool = False
    tariff_probability: float = 0.30           # chance a tariff starts during the window
    tariff_window: tuple[int, int] = (2027, 2036)  # first / last year it can start (uniform)
    tariff_rate: float = 0.25                  # duty on steel and other imported materials
    capex_material_share: float = 0.35         # share of development capex that is steel / materials
    opex_material_share: float = 0.10          # share of opex that is materials and spares
    opex_growth: bool = False
    opex_growth_mean: float = 0.01             # unexpected opex escalation per year, above plan
    opex_growth_sigma: float = 0.015
    opex_growth_clip: tuple[float, float] = (-0.02, 0.08)
    # F. factor baseline financials: prices also move NorthStar's own EBITDA and FCF (off = workbook)
    baseline_financials: bool = False
    baseline_passthrough: float = 1.0          # share of the baseline revenue change that reaches EBITDA


@dataclass
class Shocks:
    """One row per scenario. Price and volume multipliers are [n, years from 2027]."""
    volume: dict[str, np.ndarray]
    crude_shift: np.ndarray
    price: dict[str, np.ndarray]
    capex: dict[str, np.ndarray]
    delay: dict[str, np.ndarray]
    ebitda_growth: np.ndarray
    tariff_year: np.ndarray        # calendar year the tariff starts; NO_TARIFF = never
    tariff_capex_uplift: np.ndarray  # tariff rate x capex material share, from tariff_year on
    tariff_opex_uplift: np.ndarray   # tariff rate x opex material share, from tariff_year on
    opex_growth: np.ndarray        # unexpected opex escalation per year from 2027
    baseline_passthrough: float = 0.0  # 0 = baseline EBITDA and FCF ignore prices, as in the workbook

    @property
    def n(self) -> int:
        return len(self.crude_shift)


def lognormal_mean_one(z: np.ndarray, sigma: float) -> np.ndarray:
    """exp(sigma*z - sigma^2/2): lognormal with mean exactly 1."""
    return np.exp(sigma * z - 0.5 * sigma**2)


def base_shocks(case: CaseData, n: int = 1, cfg: ShockConfig | None = None) -> Shocks:
    """All multipliers at 1, no shift, no delay, Exhibit 12 growth. cfg only sets the baseline link."""
    t = case.max_years
    return Shocks(
        volume={p: np.ones((n, t)) for p in PROJECTS},
        crude_shift=np.zeros(n),
        price={b: np.ones((n, t)) for b in BENCHMARKS},
        capex={p: np.ones(n) for p in PROJECTS},
        delay={p: np.zeros(n, dtype=int) for p in PROJECTS},
        ebitda_growth=np.full(n, case.corporate.ebitda_growth),
        tariff_year=np.full(n, NO_TARIFF),
        tariff_capex_uplift=np.zeros(n),
        tariff_opex_uplift=np.zeros(n),
        opex_growth=np.zeros(n),
        baseline_passthrough=cfg.baseline_passthrough if cfg and cfg.baseline_financials else 0.0,
    )


def set_tariff(s: Shocks, cfg: ShockConfig, start_year: np.ndarray | int, rate: np.ndarray | float) -> None:
    """Tariff from start_year onward; rate 0 or NO_TARIFF means no tariff."""
    s.tariff_year[:] = start_year
    s.tariff_capex_uplift[:] = rate * cfg.capex_material_share
    s.tariff_opex_uplift[:] = rate * cfg.opex_material_share


def _volume(cfg: ShockConfig, z_level: np.ndarray, z_year: np.ndarray, sigma: float) -> np.ndarray:
    m = lognormal_mean_one(z_level, sigma)[:, None] * np.ones_like(z_year)
    if cfg.volume_year_noise:
        m = m * lognormal_mean_one(z_year, cfg.volume_year_sigma)
    return np.clip(m, *cfg.volume_clip)


def _price_paths(cfg: ShockConfig, z: np.ndarray) -> dict[str, np.ndarray]:
    """z: [n, years, 5] independent normals -> correlated multipliers per benchmark."""
    corr = np.linalg.cholesky(cfg.price_correlation)
    vols = np.array([cfg.price_vol[b] for b in BENCHMARKS])
    eps = z @ corr.T  # correlated across benchmarks
    if cfg.mean_reverting_prices:
        x = _ou(eps, vols, np.log(2) / cfg.price_half_life)
        var = _ou_variance(vols, np.log(2) / cfg.price_half_life, z.shape[1])
        mult = np.exp(x - 0.5 * var)
    else:
        level = eps[:, :1, :] * vols  # one draw per scenario, whole curve
        mult = np.exp(level - 0.5 * vols**2) * np.ones_like(eps)
    mult = np.clip(mult, *cfg.price_clip)
    return {b: mult[:, :, i] for i, b in enumerate(BENCHMARKS)}


def _ou(eps: np.ndarray, vols: np.ndarray, kappa: float) -> np.ndarray:
    """Ornstein-Uhlenbeck log deviation from the curve, starting at 0 in 2026."""
    decay = np.exp(-kappa)
    step = vols * np.sqrt((1 - decay**2) / (2 * kappa))
    x = np.zeros_like(eps)
    prev = np.zeros(eps[:, 0, :].shape)
    for t in range(eps.shape[1]):
        prev = prev * decay + step * eps[:, t, :]
        x[:, t, :] = prev
    return x


def _ou_variance(vols: np.ndarray, kappa: float, n_years: int) -> np.ndarray:
    t = np.arange(1, n_years + 1)[:, None]
    return vols**2 * (1 - np.exp(-2 * kappa * t)) / (2 * kappa)


def _delay_cdf(probs: tuple[float, float, float]) -> np.ndarray:
    return np.round(np.cumsum(probs) / sum(probs), 12)  # 0.7 + 0.2 must equal 0.9


def _delay(rng_u: np.ndarray, probs: tuple[float, float, float]) -> np.ndarray:
    return np.searchsorted(_delay_cdf(probs), rng_u, side="right").clip(0, 2)


def draw_shocks(case: CaseData, cfg: ShockConfig) -> Shocks:
    n, t = cfg.n_scenarios, case.max_years
    rng = np.random.default_rng(cfg.seed)
    z_vol = rng.standard_normal((n, len(PROJECTS)))
    z_vol_year = rng.standard_normal((n, t, len(PROJECTS)))
    z_mix = rng.standard_normal(n)
    z_price = rng.standard_normal((n, t, len(BENCHMARKS)))
    u_capex = rng.random((n, len(PROJECTS)))
    u_delay = rng.random((n, len(PROJECTS)))
    z_growth = rng.standard_normal(n)
    u_tariff = rng.random(n)        # drawn last, so earlier draws keep their values
    u_tariff_year = rng.random(n)
    z_opex = rng.standard_normal(n)

    shocks = base_shocks(case, n, cfg)
    shocks.volume = {
        p: _volume(cfg, z_vol[:, i], z_vol_year[:, :, i], cfg.volume_sigma[p]) for i, p in enumerate(PROJECTS)
    }
    lim = cfg.crude_shift_clip
    shocks.crude_shift = np.clip(z_mix * cfg.crude_shift_sigma, -lim, lim)
    shocks.price = _price_paths(cfg, z_price)
    if cfg.extra_levers:
        lo, mode, hi = cfg.capex_triangular
        shocks.capex = {p: _triangular_ppf(u_capex[:, i], lo, mode, hi) for i, p in enumerate(PROJECTS)}
        shocks.delay = {p: _delay(u_delay[:, i], cfg.delay_probs) for i, p in enumerate(PROJECTS)}
        shocks.ebitda_growth = cfg.ebitda_growth_mean + cfg.ebitda_growth_sigma * z_growth
    if cfg.tariff_event:  # one macro event per scenario, shared by all three projects
        first, last = cfg.tariff_window
        year = first + np.floor(u_tariff_year * (last - first + 1)).astype(int)
        set_tariff(shocks, cfg, np.where(u_tariff < cfg.tariff_probability, year, NO_TARIFF), cfg.tariff_rate)
    if cfg.opex_growth:
        shocks.opex_growth = np.clip(cfg.opex_growth_mean + cfg.opex_growth_sigma * z_opex, *cfg.opex_growth_clip)
    return shocks


def _triangular_ppf(u: np.ndarray, lo: float, mode: float, hi: float) -> np.ndarray:
    """Inverse CDF of the triangular distribution (lets the tornado use exact P10/P90)."""
    u = np.asarray(u, dtype=float)
    f = (mode - lo) / (hi - lo)
    left = lo + np.sqrt(u * (hi - lo) * (mode - lo))
    right = hi - np.sqrt((1 - u) * (hi - lo) * (hi - mode))
    return np.where(u < f, left, right)


def input_quantiles(cfg: ShockConfig, q: float) -> dict[str, float]:
    """Quantile q of each single input's own distribution, for the tornado chart."""
    z = NormalDist().inv_cdf(q)
    out = {f"Volume {p}": float(np.clip(lognormal_mean_one(z, s), *cfg.volume_clip)) for p, s in cfg.volume_sigma.items()}
    out["Crude share shift"] = float(np.clip(z * cfg.crude_shift_sigma, -cfg.crude_shift_clip, cfg.crude_shift_clip))
    for b in BENCHMARKS:
        out[f"Price {b}"] = float(np.clip(lognormal_mean_one(z, cfg.price_vol[b]), *cfg.price_clip))
    for p in PROJECTS:
        out[f"Capex {p}"] = float(_triangular_ppf(q, *cfg.capex_triangular))
        out[f"Delay {p}"] = float(np.searchsorted(_delay_cdf(cfg.delay_probs), q, side="left"))
    out["Baseline EBITDA growth"] = cfg.ebitda_growth_mean + z * cfg.ebitda_growth_sigma
    out["Opex growth"] = float(np.clip(cfg.opex_growth_mean + z * cfg.opex_growth_sigma, *cfg.opex_growth_clip))
    out["Tariff event"] = float(q > 1 - cfg.tariff_probability)  # 0 = none, 1 = tariff from first window year
    return out

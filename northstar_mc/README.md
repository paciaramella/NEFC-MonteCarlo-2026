# NorthStar leverage Monte Carlo

Tests how sensitive each project's **Net debt / Adj. EBITDA** is to the Exhibit 5-9 inputs
(production, product mix, prices), plus optional capex, delay and EBITDA-growth levers.
A scenario **FAILS** if the ratio is above **1.75x** in any year (exactly 1.75x passes).

## Run it

```bash
pip install -r requirements.txt
python -m pytest                 # baseline must match the workbook DCF tabs, plus lever and toggle tests
python simulate.py               # 10,000 scenarios, writes outputs/scenarios.csv + outputs/summary.xlsx
python simulate.py --extra-levers   # also vary capex overrun, first-production delay, EBITDA growth
python simulate.py --tariff --opex-growth   # tariff event risk and unexpected opex growth (combine freely)
python simulate.py --baseline-financials    # prices also move NorthStar's baseline EBITDA and FCF
streamlit run app.py             # slider UI + Monte Carlo
```

The code reads `2026 NorthStarEnergy_Case_Exhibits_with_Puma_Analysis.xlsx` from this folder or
either of the two folders above it. To point somewhere else, set `NORTHSTAR_WORKBOOK=/path/to/file.xlsx`.
All exhibit numbers come from the workbook, so editing an exhibit changes the model.

## Files

| File | What it does |
|---|---|
| `data_loader.py` | Reads the workbook cells (Exhibits 1, 2, 5-9, 12, WACC & Inputs) into dataclasses |
| `model.py` | Vectorised project model (DCF rows 17-40) and leverage test (rows 54-65) |
| `shocks.py` | Random draws. Every assumption below is a field on `ShockConfig` |
| `simulate.py` | Runs N scenarios, summary stats, per-year fan data, tornado, CSV / Excel export |
| `app.py` | Streamlit UI. The slider scenario uses the same `run_all` function as the Monte Carlo |
| `theme.py`, `.streamlit/config.toml` | Look matched to the NEFC 2026 deck: Cambria headings, Arial body, teal accent, Bison teal / Fennec blue / Puma red, gray dashed ceiling |
| `tests/test_baseline.py` | Base case vs. workbook: every year's EBITDA, FCF, net debt and ratio to 1e-6 |
| `tests/test_baseline_financials.py` | Monte Carlo with the baseline financials toggle off and on, checked against hand-built values |

## Assumptions (defaults, all in `shocks.ShockConfig`)

| Assumption | Default |
|---|---|
| Scenarios / seed | 10,000 / 2026 |
| Volume multiplier (one per project, all producing years) | lognormal, mean 1, σ = 10% Bison, 12% Fennec, 15% Puma, clipped to [0.5, 1.5] |
| Year-by-year volume noise | off; σ = 3% per year when on |
| Product mix | fixed at Exhibit 8 (the crude share shift was removed: negligible effect on every result) |
| Price level multipliers | lognormal, mean 1; vol WTI 25%, Brent 25%, Henry Hub 35%, TTF 40%, OPIS 25%; clipped [0.3, 2.5] |
| Price correlation | WTI-Brent 0.95, crude-OPIS 0.70, crude-HH 0.30, WTI-TTF 0.30, Brent-TTF 0.35, HH-TTF 0.40, HH-OPIS 0.40, TTF-OPIS 0.35 (Cholesky) |
| Europe NGL | OPIS × OPIS multiplier + the Exhibit 9 spread ($3) |
| Mean-reverting prices | off; Ornstein-Uhlenbeck on log price around the Exhibit 9 curve, 3-year half-life, same vol/correlation |
| Common random numbers | the same price draws are used for all three projects in a scenario |
| Capex overrun (extra lever) | off; triangular(0.95, 1.05, 1.40) × Exhibit 1 cost. Also scales depreciation |
| First-production delay (extra lever) | off; 0 / 1 / 2 years with 70% / 20% / 10%. Shifts production, mix and depreciation right; capex keeps its schedule; tail truncated; decommissioning stays in the original final year |
| Baseline EBITDA growth (extra lever) | off; normal(3%, 1.5 pp) |
| Tariff event (own toggle) | off; 30% chance per scenario, start year uniform 2027-2036, one event shared by all projects, lasts for the rest of the project life |
| Tariff impact | 25% duty on steel and materials: 35% of development capex and 10% of opex are materials, so capex spent from the start year rises 8.75% and opex 2.5%. Higher capex also raises depreciation (tax) |
| Unexpected opex growth (own toggle) | off; normal(1%/yr, 1.5 pp), clipped to [-2%, 8%]; compounds from 2027 on direct opex, overhead and midstream $/boe, shared by all projects |
| Factor baseline financials (own toggle, applies to sliders and Monte Carlo) | off; when on, NorthStar's baseline EBITDA changes by Exhibit 2 2026 revenue × (price multiplier − 1), grown with baseline EBITDA: oil $21,135MM on 50% WTI / 50% Brent, gas $2,382MM on Henry Hub, NGL $8,361MM on OPIS, × pass-through (default 100%). Base FCF changes by (1 − 21% tax) × that amount. No management response (capex and dividends unchanged) |
| Ratio when total EBITDA ≤ 0 | reported as 10.0x (`model.RATIO_CAP`), always a breach; ratios above 10x are capped at 10x |
| Tornado | each input moved alone to its own P10 and P90, everything else at base |

Workbook simplifications kept: baseline EBITDA and base FCF do not move with prices (unless the toggle is on), no interest
on project debt, no new equity, no buybacks after 2026, tax can go negative (credit) in loss years,
carbon cost excluded from the leverage test.

## Results with the defaults (10,000 scenarios, seed 2026)

Exhibits 5-9 only:

| Project | Pass rate | Peak P50 | Peak P95 | Peak in build year | Post-build peak P50 | Post-build peak P95 | Post-build breach rate |
|---|---|---|---|---|---|---|---|
| Bison | 100.0% | 1.636x | 1.636x | 100% | 1.470x | 1.521x | 0.0% |
| Fennec | 100.0% | 1.670x | 1.670x | 100% | 1.390x | 1.528x | 0.0% |
| Puma | 0.0% | 2.090x | 2.090x | 100% | 1.512x | 1.731x | 3.5% |

With the extra levers on:

| Project | Pass rate | Peak P50 | Peak P95 | Most common breach year | Post-build peak P95 | Post-build breach rate |
|---|---|---|---|---|---|---|
| Bison | 99.97% | 1.653x | 1.699x | 2027 | 1.571x | 0.01% |
| Fennec | 68.9% | 1.710x | 1.848x | 2029 | 1.620x | 0.5% |
| Puma | 0.16% | 2.203x | 2.520x | 2031 | 1.935x | 20.1% |

Tariff event and opex growth levers (10,000 scenarios, seed 2026):

| Run | Bison pass | Fennec pass | Puma pass | Puma post-build breach rate |
|---|---|---|---|---|
| Exhibits 5-9 only | 100% | 100% | 0% | 3.5% |
| + tariff event | 100% | 100% | 0% | 4.3% |
| + opex growth | 100% | 100% | 0% | 4.4% |
| + tariff and opex growth | 100% | 100% | 0% | 5.2% |
| All extra levers on | 99.96% | 68.1% | 0.16% | 22.6% |

A 25% tariff starting in 2027 lifts Fennec's 2029 peak from 1.670x to 1.696x and Puma's from 2.090x to
2.164x; Fennec only fails at a tariff of roughly 75%+ on materials. Opex growth hits after first oil, so
it moves the post-build peak (Puma 1.488x to 1.537x at 5%/yr), not the build-year peak. Above about
8%/yr, Puma's late-life opex outruns revenue and its ratio climbs again.

**Reading it:** every project's peak falls in a build year, before first production, so the
Exhibit 5-9 inputs cannot move pass/fail. The peak is set by build capex and NorthStar's baseline.
Exhibit 5-9 risk shows up in the post-build peak: Puma's post-build ratio crosses 1.75x in 3.5% of
scenarios (mostly low Brent), while Bison and Fennec never do. Capex overrun and baseline EBITDA growth
are what flip Fennec: a 30% Fennec overrun takes its 2029 ratio to 1.759x (FAIL), and 1% baseline
growth takes it to 1.767x. A first-production delay barely changes the peak, because the peak is at
the end of the build and the base business keeps paying debt down while production waits.

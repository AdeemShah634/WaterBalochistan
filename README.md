# Groundwater Sustainability Monitoring & Forecasting System for Balochistan

District-level groundwater monitoring, forecasting and risk classification for
all 26 districts of Balochistan, delivered as an interactive web GIS dashboard.

Groundwater storage comes from NASA **GLDAS-2.2 CLSM GRACE-DA** (`GWS_tavg`,
0.25°) and rainfall from **CHIRPS** (0.05°), aggregated per district by zonal
statistics in Google Earth Engine. The record runs **2003-02 → 2026-05**
(280 monthly observations per district) and the system forecasts three years
beyond it.

---

## Quick start

```bash
pip install -r requirements.txt
python build_data.py            # extract -> validate -> forecast -> risk index
python -m streamlit run app/dashboard.py
```

`build_data.py` takes about half a minute; it fits 26 Prophet models, benchmarks
each against two baselines, computes the risk index and writes everything to
`data/groundwater.db`. The dashboard only reads that database, so it starts
instantly.

Optional REST layer (same analysis code, different presentation):

```bash
pip install fastapi uvicorn
python -m uvicorn src.api:app --reload
```

---

## What the dashboard shows

| Tab | Contents |
| --- | --- |
| **Risk Map** | Colour-coded district risk choropleth (Folium) with a timeline slider spanning the observed record *and* the forecast horizon, province-wide band counts, a worst-first district ranking, and a stacked risk-over-time chart. |
| **District Detail** | Observed groundwater with the Prophet forecast and its 80% uncertainty band, the risk percentile against the district's own same-month history, monthly rainfall with a 12-month rolling total, and a one-click PDF summary. |
| **Model Validation** | RMSE/MAE for Prophet, seasonal-naive and SARIMA over 5 rolling test windows — province-wide, per district, per window, plus the Prophet configurations that were considered and rejected. |
| **Data & Method** | The data quality report, the method in plain language, declared scope limitations, GeoPandas-derived district geometry, and CSV downloads. |

Observed and forecast periods are labelled everywhere they appear — on the
slider, in the map header, on the charts and in the PDF.

---

## Architecture (three-tier)

```
Tier 1  Presentation      app/dashboard.py   (Streamlit + Folium + Plotly)
                          src/api.py         (FastAPI - optional, same data)
                                  |
                          src/analysis.py    <- presentation-independent core
                                  |
Tier 2  Application       src/pipeline.py    extract, validate, load
                          src/forecasting.py Prophet + baselines
                          src/risk.py        Groundwater Risk Index
                          src/report.py      FPDF2 district summary
                                  |
Tier 3  Data              src/db.py -> data/groundwater.db  +  district GeoJSON
```

The dashboard contains **no analysis logic**. Every number on screen is returned
by `src/analysis.py`, which is also what the REST service calls — so the two
interfaces can never disagree, and the UI is interchangeable.

---

## The Groundwater Risk Index

Each district is judged against **its own history**, for the **same calendar
month**, so that ordinary seasonal variation is never mistaken for depletion.
The percentile rank of a value within that reference distribution maps onto four
bands whose cut-offs follow the U.S. Drought Monitor convention:

| Band | Percentile | USDM equivalent |
| --- | --- | --- |
| Severe Deficit | ≤ 10th | D2–D4, severe to exceptional drought |
| Critical | 10–20th | D1, moderate drought |
| Moderate Stress | 20–30th | D0, abnormally dry |
| Stable | > 30th | no drought classification |

A **secondary indicator** flags any district whose trailing 12-month rainfall
total falls below the 20th percentile of its own same-month history. Rainfall is
measured rather than forecast, so this indicator is only defined over the
observed period.

Forecast values are scored against the *observed* historical distribution, so a
band means exactly the same thing on both sides of the timeline.

---

## Forecasting and how it was validated

Prophet is fitted per district (additive yearly seasonality, 80% intervals) and
projects 60 months ahead, of which the first **36 are presented as reliable**.

### Accuracy is measured over several test windows, not one

A single holdout is not enough evidence. Scored only on the final 24 months, the
seasonal-naive baseline looks competitive — that particular window happens to
suit it. Accuracy is therefore measured by **rolling-origin backtest**:
5 separate 24-month test windows,
12 months apart, each forecast from a training set that
ends before it.

Mean RMSE per model, per test window:

| Test window starts | arima | prophet | seasonal_naive |
| --- | --- | --- | --- |
| 2020-06 | 10.86 | 7.27 | 11.67 |
| 2021-06 | 24.10 | 22.27 | 20.65 |
| 2022-06 | 25.52 | 22.42 | 23.52 |
| 2023-06 | 7.43 | 7.35 | 26.65 |
| 2024-06 | 12.10 | 7.72 | 8.73 |

The two hard windows in the middle are real events the models miss, not a bug —
and they are exactly why one window is not enough to rank forecasters.

### Results

Averaged across the 5 windows and all 26 districts,
against a **seasonal-naive** baseline (repeat the value 12 months earlier) and a
**SARIMA(1, 1, 1)(1, 1, 1, 12)** baseline:

| Model | Mean RMSE (mm) | Mean MAE (mm) | Districts where best |
| --- | --- | --- | --- |
| **Prophet (constrained trend)** | 13.41 | 8.61 | 21 |
| SARIMA | 16.00 | 11.42 | 5 |
| seasonal-naive | 18.24 | 12.07 | 0 |

Prophet beats the seasonal-naive baseline by **27%** on mean RMSE.

### Why this Prophet configuration

Prophet's stock `changepoint_prior_scale=0.05` lets the trend bend freely at 25
changepoints. Extrapolated one to three years, that produces a spurious slope on
these strongly seasonal, weakly trending series. The trend is therefore
constrained to `0.005` — a stated
modelling assumption, scored against the alternatives over the same windows:

| Configuration | Mean RMSE (mm) | Mean MAE (mm) |
| --- | --- | --- |
| **chosen (cps=0.005)** | 13.41 | 8.61 |
| flat trend | 13.72 | 8.94 |
| rigid trend (cps=0.0005) | 14.32 | 9.48 |
| Prophet default (cps=0.05) | 15.30 | 10.62 |

Per-district metrics, the per-window breakdown and this comparison are all on
the dashboard's Model Validation tab, so the choice is visible and falsifiable
rather than buried in a config file.

---

## Data notes

The source data lands in `data/raw/`:

| File | Contents |
| --- | --- |
| `balochistan_groundwater_monthly.csv` | district, month, `gws_mm` |
| `balochistan_rainfall_monthly.csv` | district, month, `rain_mm` |
| `balochistan_districts.geojson` | 26 district boundaries (`ADM2_NAME`) |

`build_data.py` prints a data quality report covering district-name matching
against the boundary file, coverage and gap checks, nulls, value ranges,
duplicates, time-span alignment and geometry validity. All checks pass on the
current data, with one warning: **seven districts arrive as GeometryCollections**
carrying stray Points and LineStrings alongside their real polygons. Those
zero-area digitizing artifacts are stripped in `pipeline.polygonal_only()` —
left in place they break Folium's bounds calculation and would distort the
computed district areas.

### Scope limitation

Groundwater values derive from a 0.25° GRACE-assimilated land surface model and
represent **district-scale averages, not individual well or aquifer
measurements**. The system is a screening and prioritization tool, not a
replacement for ground observation. District boundaries come from a public
dataset whose vintage is a declared limitation.

---

## Tests

```bash
python -m pytest
```

75 tests covering data validation, geometry cleaning, SQLite operations,
risk-band threshold boundaries, the rolling-origin backtest (including that a
model failing at one origin is recorded rather than aborting the run) and the
core analysis interface. The `test_analysis.py` integration tests skip
automatically when the database has not been built yet.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fpdf'` (or geopandas, folium, statsmodels)**

These live in the per-user site-packages directory
(`%APPDATA%\Python\Python314\site-packages`), which some shells do not put on
`sys.path`. Check which interpreter your shell is actually using:

```bash
python -c "import fpdf, geopandas, folium, statsmodels; print('all imports OK')"
```

If that succeeds but the dashboard still fails, you are looking at a **stale
Streamlit process** — stop it with `Ctrl+C` in its terminal and start a fresh
one. If the import genuinely fails, reinstall into the interpreter your shell
resolves to:

```bash
python -m pip install -r requirements.txt
```

The PDF export is the only feature that needs `fpdf2`, and a missing copy now
disables just that button rather than taking the whole dashboard down.

**`Program 'streamlit.exe' failed to run: An Application Control policy has blocked this file`**

Windows Smart App Control blocks unsigned executables, and the `streamlit.exe`
launcher that pip generates is unsigned. `python.exe` is signed, so launch
through it instead — this works on every machine:

```bash
python -m streamlit run app/dashboard.py
```

The same applies to any pip-installed command (`python -m pytest`,
`python -m uvicorn ...`).

**PowerShell rejects `&&`** — Windows PowerShell 5.1 has no `&&` operator. Run
the two commands separately:

```bash
cd H:\c-water
python -m streamlit run app/dashboard.py
```

---

## Refreshing with new satellite data

1. Re-run the Earth Engine extraction and drop the new CSVs into `data/raw/`.
2. `python build_data.py` — revalidates, refits every model, rebuilds the index.
3. Restart the dashboard (or press **R** in it) to clear the Streamlit cache.

Use `python build_data.py --skip-forecast` to reload and revalidate the data
without refitting the models.

"""Central configuration for the Groundwater Sustainability Monitoring System.

Keeps every path, threshold and model parameter in one place so the pipeline,
the analysis layer and the dashboard all agree on the same constants.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"

GROUNDWATER_CSV = RAW_DIR / "balochistan_groundwater_monthly.csv"
RAINFALL_CSV = RAW_DIR / "balochistan_rainfall_monthly.csv"
DISTRICTS_GEOJSON = RAW_DIR / "balochistan_districts.geojson"

DB_PATH = DATA_DIR / "groundwater.db"

# Property in the GeoJSON that carries the district name.
GEOJSON_NAME_FIELD = "ADM2_NAME"

# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
FORECAST_MONTHS = 60          # Prophet projects 5 years ahead ...
RELIABLE_FORECAST_MONTHS = 36  # ... of which only 3 years are presented as reliable.
HOLDOUT_MONTHS = 24            # final 24 observed months held out for benchmarking.
SEASONAL_PERIOD = 12

# SARIMA order used for the ARIMA baseline benchmark.
ARIMA_ORDER = (1, 1, 1)
ARIMA_SEASONAL_ORDER = (1, 1, 1, 12)

# Prophet configuration.
#
# Prophet's stock trend flexibility (changepoint_prior_scale=0.05) lets the trend
# bend freely at 25 changepoints. Over a 1-3 year extrapolation of a strongly
# seasonal, weakly trending series that produces a spurious slope, so the trend
# is deliberately constrained. This is a stated modelling assumption, not a value
# fitted to the test data - and `PROPHET_CANDIDATES` below is evaluated against
# it by rolling-origin backtest so the choice is visible and falsifiable.
PROPHET_PARAMS = {"changepoint_prior_scale": 0.005, "yearly_seasonality": 10}
PROPHET_INTERVAL_WIDTH = 0.8

# Alternative configurations scored alongside the chosen one, purely so the
# Model Validation tab can show why this one was picked.
PROPHET_CANDIDATES = {
    "chosen (cps=0.005)": {"changepoint_prior_scale": 0.005, "yearly_seasonality": 10},
    "rigid trend (cps=0.0005)": {"changepoint_prior_scale": 0.0005, "yearly_seasonality": 10},
    "flat trend": {"growth": "flat", "yearly_seasonality": 10},
    "Prophet default (cps=0.05)": {"changepoint_prior_scale": 0.05, "yearly_seasonality": 10},
}

# Rolling-origin backtest.
#
# A single holdout window is not enough evidence: on this data the final 24
# months happen to be a period the seasonal-naive baseline handles unusually
# well, which flatters it. Accuracy is therefore measured over several test
# periods, each forecast from a training set that ends before it.
BACKTEST_ORIGINS = 5     # number of test windows
BACKTEST_STRIDE = 12     # months between consecutive test windows

# --------------------------------------------------------------------------
# Groundwater Risk Index
# --------------------------------------------------------------------------
# Percentile bands aligned to the U.S. Drought Monitor (USDM) convention, where
# a *low* percentile of the historical distribution means *drier than usual*.
#   USDM: D0 <=30th, D1 <=20th, D2 <=10th, D3 <=5th, D4 <=2nd percentile.
# The four project classes collapse those bands as follows.
RISK_BANDS = [
    # (label, upper percentile bound inclusive, colour, USDM equivalent)
    ("Severe Deficit", 10.0, "#a50f15", "D2-D4 (severe to exceptional)"),
    ("Critical", 20.0, "#fb6a4a", "D1 (moderate drought)"),
    ("Moderate Stress", 30.0, "#fdae6b", "D0 (abnormally dry)"),
    ("Stable", 100.0, "#41ab5d", "No drought classification"),
]
RISK_ORDER = [b[0] for b in RISK_BANDS]
RISK_COLORS = {b[0]: b[2] for b in RISK_BANDS}
RISK_USDM = {b[0]: b[3] for b in RISK_BANDS}
# Numeric severity, handy for sorting / choropleth scales (higher = worse).
RISK_SEVERITY = {"Stable": 0, "Moderate Stress": 1, "Critical": 2, "Severe Deficit": 3}

# Secondary indicator: a district is flagged for rainfall deficit when its
# trailing 12-month rainfall total falls below this percentile of its own history.
RAINFALL_DEFICIT_PERCENTILE = 20.0
RAINFALL_WINDOW_MONTHS = 12

# --------------------------------------------------------------------------
# Data validation expectations
# --------------------------------------------------------------------------
GWS_VALID_RANGE = (0.0, 5000.0)     # mm of groundwater storage
RAIN_VALID_RANGE = (0.0, 2000.0)    # mm of monthly rainfall
EXPECTED_DISTRICT_COUNT = 26

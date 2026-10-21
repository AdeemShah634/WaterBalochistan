"""Risk-index unit tests, with an emphasis on the band threshold boundaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, risk


# --------------------------------------------------------------------------
# percentile_of
# --------------------------------------------------------------------------
def test_percentile_of_extremes():
    baseline = np.arange(0, 100, dtype=float)
    assert risk.percentile_of(-1, baseline) == 0.0
    assert risk.percentile_of(200, baseline) == 100.0


def test_percentile_of_ties_count_half():
    baseline = np.array([1.0, 1.0, 1.0, 1.0])
    assert risk.percentile_of(1.0, baseline) == 50.0


def test_percentile_of_midpoint():
    baseline = np.array([10.0, 20.0, 30.0, 40.0])
    assert risk.percentile_of(25.0, baseline) == 50.0


def test_percentile_of_empty_baseline_is_nan():
    assert np.isnan(risk.percentile_of(5.0, np.array([])))


def test_percentile_of_ignores_nans_in_baseline():
    assert risk.percentile_of(5.0, np.array([1.0, np.nan, 9.0])) == 50.0


# --------------------------------------------------------------------------
# classify - threshold boundaries are inclusive of the upper bound
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "percentile,expected",
    [
        (0.0, "Severe Deficit"),
        (9.99, "Severe Deficit"),
        (10.0, "Severe Deficit"),   # boundary belongs to the worse band
        (10.01, "Critical"),
        (20.0, "Critical"),
        (20.01, "Moderate Stress"),
        (30.0, "Moderate Stress"),
        (30.01, "Stable"),
        (100.0, "Stable"),
    ],
)
def test_classify_boundaries(percentile, expected):
    assert risk.classify(percentile) == expected


def test_classify_nan_defaults_to_stable():
    assert risk.classify(float("nan")) == "Stable"


def test_every_band_has_a_colour_and_severity():
    for label in config.RISK_ORDER:
        assert label in config.RISK_COLORS
        assert label in config.RISK_SEVERITY
        assert label in config.RISK_USDM


# --------------------------------------------------------------------------
# build_risk_index
# --------------------------------------------------------------------------
def _synthetic(n_years: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = pd.period_range("2010-01", periods=n_years * 12, freq="M")
    rows_gw, rows_rain = [], []
    for i, m in enumerate(months):
        rows_gw.append({"district": "A", "month": str(m), "gws_mm": 500 + (i % 12) - i * 0.1})
        rows_rain.append({"district": "A", "month": str(m), "rain_mm": 10 + (i % 12)})
    return pd.DataFrame(rows_gw), pd.DataFrame(rows_rain)


def test_build_risk_index_scores_every_month():
    gw, rain = _synthetic()
    fc = pd.DataFrame(
        {
            "district": ["A", "A"],
            "month": ["2018-01", "2018-02"],
            "yhat": [400.0, 600.0],
        }
    )
    out = risk.build_risk_index(gw, fc, rain)
    assert len(out) == len(gw) + len(fc)
    assert set(out["period"]) == {"observed", "forecast"}
    assert out["percentile"].between(0, 100).all()


def test_forecast_rows_overlapping_observed_months_are_dropped():
    gw, rain = _synthetic()
    overlapping = gw["month"].iloc[0]
    fc = pd.DataFrame({"district": ["A"], "month": [overlapping], "yhat": [1.0]})
    out = risk.build_risk_index(gw, fc, rain)
    assert len(out) == len(gw)
    assert (out["period"] == "observed").all()


def test_declining_series_ends_in_severe_deficit():
    """A district trending steadily down must eventually score at the bottom."""
    gw, rain = _synthetic()
    fc = pd.DataFrame({"district": ["A"], "month": ["2018-06"], "yhat": [100.0]})
    out = risk.build_risk_index(gw, fc, rain)
    last = out[out["month"] == "2018-06"].iloc[0]
    assert last["risk_band"] == "Severe Deficit"
    assert last["severity"] == config.RISK_SEVERITY["Severe Deficit"]


def test_seasonality_is_not_mistaken_for_depletion():
    """A perfectly repeating seasonal cycle must never register as a deficit."""
    months = pd.period_range("2010-01", periods=120, freq="M")
    gw = pd.DataFrame(
        {"district": "A", "month": [str(m) for m in months], "gws_mm": [500 + (i % 12) * 20 for i in range(120)]}
    )
    rain = pd.DataFrame(
        {"district": "A", "month": [str(m) for m in months], "rain_mm": [5.0] * 120}
    )
    out = risk.build_risk_index(gw, pd.DataFrame(columns=["district", "month", "yhat"]), rain)
    assert (out["risk_band"] == "Stable").all()


# --------------------------------------------------------------------------
# rainfall indicator
# --------------------------------------------------------------------------
def test_rainfall_indicator_rolls_12_months():
    _, rain = _synthetic()
    out = risk.rainfall_indicator(rain)
    assert out["rain_12m_mm"].isna().sum() == config.RAINFALL_WINDOW_MONTHS - 1
    expected = rain["rain_mm"].iloc[:12].sum()
    assert out["rain_12m_mm"].iloc[11] == pytest.approx(expected)


def test_rainfall_deficit_flag_is_binary():
    _, rain = _synthetic()
    out = risk.rainfall_indicator(rain)
    assert set(out["rain_deficit"].unique()) <= {0, 1}

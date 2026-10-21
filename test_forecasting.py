"""Forecasting and baseline tests (Prophet itself is exercised only lightly -
fitting is slow, so the heavy check is marked `slow`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, forecasting


@pytest.fixture
def ts() -> pd.DataFrame:
    months = pd.period_range("2010-01", periods=120, freq="M")
    df = pd.DataFrame(
        {
            "district": "A",
            "month": [str(m) for m in months],
            "gws_mm": [500 + 15 * np.sin(2 * np.pi * i / 12) - 0.2 * i for i in range(120)],
        }
    )
    return forecasting.to_timeseries(df)


def test_to_timeseries_shape_and_order(ts):
    assert list(ts.columns) == ["ds", "y"]
    assert ts["ds"].is_monotonic_increasing
    assert len(ts) == 120


def test_rmse_and_mae_are_zero_for_a_perfect_prediction():
    a = np.array([1.0, 2.0, 3.0])
    assert forecasting.rmse(a, a) == 0.0
    assert forecasting.mae(a, a) == 0.0


def test_rmse_penalises_large_errors_more_than_mae():
    actual = np.array([0.0, 0.0, 0.0, 0.0])
    predicted = np.array([0.0, 0.0, 0.0, 8.0])
    assert forecasting.rmse(actual, predicted) > forecasting.mae(actual, predicted)


def test_seasonal_naive_repeats_the_previous_year(ts):
    preds = forecasting.seasonal_naive_predict(ts, config.SEASONAL_PERIOD)
    expected = ts["y"].to_numpy()[-config.SEASONAL_PERIOD :]
    np.testing.assert_allclose(preds, expected)


def test_seasonal_naive_wraps_beyond_one_year(ts):
    preds = forecasting.seasonal_naive_predict(ts, 18)
    assert len(preds) == 18
    np.testing.assert_allclose(preds[:6], preds[12:18])


def test_arima_returns_the_requested_horizon(ts):
    preds = forecasting.arima_predict(ts.iloc[:96], 12)
    assert len(preds) == 12
    assert np.isfinite(preds).all()


def test_every_model_is_scored_over_one_window(ts):
    detail = forecasting.backtest_district(ts, horizon=12, origins=[len(ts) - 12])
    metrics = forecasting.aggregate_backtest(detail)
    assert set(metrics["model"]) == {"prophet", "seasonal_naive", "arima"}
    assert (metrics[["rmse", "mae"]].dropna() >= 0).all().all()


@pytest.mark.slow
def test_forecast_district_horizon_and_intervals(ts):
    fcst = forecasting.forecast_district(ts, periods=24)
    assert len(fcst) == 24
    assert (fcst["yhat_lower"] <= fcst["yhat"]).all()
    assert (fcst["yhat"] <= fcst["yhat_upper"]).all()
    # Only the first RELIABLE_FORECAST_MONTHS are flagged reliable.
    assert fcst["is_reliable"].sum() == min(24, config.RELIABLE_FORECAST_MONTHS)
    assert fcst["month"].is_monotonic_increasing


# --------------------------------------------------------------------------
# Rolling-origin backtest
# --------------------------------------------------------------------------
def test_backtest_origins_are_spaced_and_leave_room_for_the_test_window():
    origins = forecasting.backtest_origins(280, horizon=24, n_origins=5, stride=12)
    assert origins == [256, 244, 232, 220, 208]
    # every window must fit inside the record
    assert all(o + 24 <= 280 for o in origins)


def test_backtest_origins_drops_windows_without_enough_history():
    origins = forecasting.backtest_origins(80, horizon=24, n_origins=5, stride=12)
    assert all(o >= 24 + 36 for o in origins)
    assert len(origins) < 5


def test_backtest_district_scores_every_model_in_every_window(ts):
    origins = forecasting.backtest_origins(len(ts), horizon=12, n_origins=2, stride=12)
    detail = forecasting.backtest_district(ts, horizon=12, origins=origins)
    assert set(detail["model"]) == {"prophet", "seasonal_naive", "arima"}
    assert detail["origin_month"].nunique() == len(origins)
    assert len(detail) == 3 * len(origins)
    assert (detail[["rmse", "mae"]].dropna() >= 0).all().all()


def test_backtest_records_a_failure_as_nan_without_aborting(ts):
    def explode(train, periods):
        raise RuntimeError("solver blew up")

    detail = forecasting.backtest_district(
        ts,
        horizon=12,
        origins=[len(ts) - 12],
        predictors={"broken": explode, "seasonal_naive": forecasting.seasonal_naive_predict},
    )
    broken = detail[detail["model"] == "broken"]
    assert len(broken) == 1 and np.isnan(broken["rmse"].iloc[0])
    # the healthy model still produced a real score
    assert detail[detail["model"] == "seasonal_naive"]["rmse"].notna().all()


def test_aggregate_backtest_averages_across_windows():
    detail = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "origin_month": ["2020-01", "2021-01"] * 2,
            "rmse": [10.0, 20.0, 1.0, 3.0],
            "mae": [5.0, 15.0, 1.0, 1.0],
        }
    )
    agg = forecasting.aggregate_backtest(detail).set_index("model")
    assert agg.loc["a", "rmse"] == 15.0
    assert agg.loc["b", "mae"] == 1.0
    assert agg.loc["a", "origins"] == 2


def test_compare_configs_scores_every_candidate(ts):
    candidates = {
        "tight": {"changepoint_prior_scale": 0.0005, "yearly_seasonality": 4},
        "loose": {"changepoint_prior_scale": 0.05, "yearly_seasonality": 4},
    }
    out = forecasting.compare_configs(
        ts, horizon=12, origins=[len(ts) - 12], candidates=candidates
    )
    assert set(out["config"]) == {"tight", "loose"}


def test_constrained_trend_beats_the_stock_default(ts):
    """The modelling assumption behind config.PROPHET_PARAMS: over a long
    horizon Prophet's stock trend flexibility overshoots."""
    horizon = 24
    train, test = ts.iloc[:-horizon], ts.iloc[-horizon:]
    actual = test["y"].to_numpy()

    loose = forecasting.prophet_predict(
        train, horizon, {"changepoint_prior_scale": 0.05, "yearly_seasonality": 10}
    )["yhat"].to_numpy()
    chosen = forecasting.prophet_predict(train, horizon, config.PROPHET_PARAMS)[
        "yhat"
    ].to_numpy()

    assert forecasting.rmse(actual, chosen) <= forecasting.rmse(actual, loose)


@pytest.mark.slow
def test_run_all_returns_forecasts_metrics_detail_and_configs():
    months = pd.period_range("2010-01", periods=140, freq="M")
    gw = pd.DataFrame(
        [
            {"district": d, "month": str(m), "gws_mm": 500 + 10 * np.sin(2 * np.pi * i / 12)}
            for d in ("A", "B")
            for i, m in enumerate(months)
        ]
    )
    forecasts, metrics, detail, configs = forecasting.run_all(gw, compare=False)
    assert set(metrics["model"]) == {"prophet", "seasonal_naive", "arima"}
    assert len(forecasts) == 2 * config.FORECAST_MONTHS
    assert set(detail["district"]) == {"A", "B"}
    assert "origins" in metrics.columns
    assert configs.empty  # compare=False

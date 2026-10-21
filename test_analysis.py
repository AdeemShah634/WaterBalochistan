"""Integration tests over the core analysis interface.

These run against the built database, so they are skipped when the pipeline
has not been run yet (`python build_data.py`).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import analysis, config, db, report

pytestmark = pytest.mark.skipif(
    not db.is_built(), reason="database not built - run `python build_data.py` first"
)


@pytest.fixture(scope="module", autouse=True)
def _fresh_cache():
    analysis.refresh()
    yield
    analysis.refresh()


def test_district_list_matches_the_boundary_file():
    assert len(analysis.list_districts()) == config.EXPECTED_DISTRICT_COUNT


def test_timeline_is_sorted_and_covers_observed_plus_forecast():
    tl = analysis.timeline_months()
    assert tl == sorted(tl)
    assert len(set(tl)) == len(tl)
    assert analysis.latest_observed_month() in tl
    assert len(tl) == len(analysis.observed_months()) + config.RELIABLE_FORECAST_MONTHS


def test_period_of_labels_both_sides_of_the_split():
    assert analysis.period_of(analysis.latest_observed_month()) == "observed"
    assert analysis.period_of(analysis.forecast_months()[0]) == "forecast"


def test_risk_snapshot_covers_every_district_once():
    snap = analysis.risk_snapshot(analysis.latest_observed_month())
    assert len(snap) == config.EXPECTED_DISTRICT_COUNT
    assert snap["district"].is_unique
    assert set(snap["risk_band"]) <= set(config.RISK_ORDER)


def test_risk_snapshot_is_ordered_worst_first():
    snap = analysis.risk_snapshot(analysis.latest_observed_month())
    assert snap["severity"].is_monotonic_decreasing


def test_province_summary_band_counts_add_up():
    month = analysis.latest_observed_month()
    s = analysis.province_summary(month)
    assert sum(s["band_counts"].values()) == s["districts"]
    assert s["period"] == "observed"


def test_forecast_snapshot_is_labelled_forecast():
    month = analysis.forecast_months()[0]
    snap = analysis.risk_snapshot(month)
    assert (snap["period"] == "forecast").all()
    assert len(snap) == config.EXPECTED_DISTRICT_COUNT


def test_combined_series_joins_observed_and_forecast_without_gaps():
    d = analysis.list_districts()[0]
    combined = analysis.combined_series(d)
    assert combined["month"].is_monotonic_increasing
    assert combined["observed"].notna().any()
    assert combined["forecast"].notna().any()


def test_forecast_intervals_bracket_the_point_forecast():
    d = analysis.list_districts()[0]
    fc = analysis.forecast_series(d)
    assert (fc["yhat_lower"] <= fc["yhat"]).all()
    assert (fc["yhat"] <= fc["yhat_upper"]).all()


def test_district_summary_is_self_consistent():
    d = analysis.list_districts()[0]
    month = analysis.latest_observed_month()
    s = analysis.district_summary(d, month)
    assert s["district"] == d and s["month"] == month
    assert s["min_gws_mm"] <= s["gws_mm"] <= s["max_gws_mm"]
    assert 0 <= s["percentile"] <= 100
    assert s["risk_band"] in config.RISK_ORDER
    assert s["months_severe_observed"] <= s["months_observed"]


def test_district_summary_rejects_an_unknown_month():
    with pytest.raises(KeyError):
        analysis.district_summary(analysis.list_districts()[0], "1900-01")


def test_metrics_summary_lists_every_benchmarked_model():
    summary = analysis.metrics_summary()
    assert set(summary["model"]) == {"prophet", "seasonal_naive", "arima"}
    assert summary["districts_best_rmse"].sum() == config.EXPECTED_DISTRICT_COUNT


def test_data_quality_report_has_no_failures():
    q = analysis.data_quality_report()
    assert not q.empty
    assert (q["status"] != "FAIL").all()


def test_rainfall_indicator_only_defined_over_the_observed_period():
    risk = db.read_table("risk_index")
    assert risk[risk["period"] == "forecast"]["rain_12m_mm"].isna().all()


def test_pdf_report_renders_for_observed_and_forecast_months():
    d = analysis.list_districts()[0]
    for month in (analysis.latest_observed_month(), analysis.forecast_months()[0]):
        pdf = report.build_district_report(d, month)
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 1500


def test_backtest_covers_every_district_model_and_window():
    detail = analysis.backtest_detail()
    assert detail["district"].nunique() == config.EXPECTED_DISTRICT_COUNT
    assert set(detail["model"]) == {"prophet", "seasonal_naive", "arima"}
    assert detail["origin_month"].nunique() == config.BACKTEST_ORIGINS


def test_metrics_average_over_every_test_window():
    m = analysis.model_metrics()
    assert (m["origins"] == config.BACKTEST_ORIGINS).all()


def test_backtest_by_origin_has_one_row_per_window():
    by_origin = analysis.backtest_by_origin()
    assert len(by_origin) == config.BACKTEST_ORIGINS
    assert {"prophet", "seasonal_naive", "arima"} <= set(by_origin.columns)


def test_prophet_beats_the_seasonal_naive_baseline_across_the_backtest():
    """The headline claim on the Model Validation tab - if this regresses, the
    dashboard is telling the user something different."""
    summary = analysis.metrics_summary().set_index("model")
    assert summary.loc["prophet", "rmse"] < summary.loc["seasonal_naive", "rmse"]


def test_config_comparison_includes_the_chosen_configuration_and_ranks_it_first():
    configs = analysis.config_comparison()
    assert len(configs) == len(config.PROPHET_CANDIDATES)
    assert set(configs["config"]) == set(config.PROPHET_CANDIDATES)
    # The configuration the system actually runs must be the one that won.
    assert "chosen" in configs.iloc[0]["config"]


def test_rainfall_indicator_is_populated_over_the_observed_period():
    """Guards the display bug where every rain-deficit cell rendered blank."""
    risk = db.read_table("risk_index")
    observed = risk[risk["period"] == "observed"]
    scored = observed[observed["rain_percentile"].notna()]
    assert len(scored) > 0
    # The 20th-percentile threshold should flag roughly a fifth of scored months.
    share = scored["rain_deficit"].mean()
    assert 0.1 < share < 0.3, f"unexpected deficit share {share:.3f}"

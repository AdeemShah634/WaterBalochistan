"""Groundwater Risk Index.

A district is judged against *its own* history, not against other districts, and
against the *same calendar month*, so that ordinary seasonal variation is never
mistaken for depletion. The percentile of a value within that reference
distribution is mapped onto four bands whose cut-offs follow the U.S. Drought
Monitor convention (see `config.RISK_BANDS`).

A trailing 12-month rainfall total, scored the same way, is carried as a
secondary indicator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def percentile_of(value: float, baseline: np.ndarray) -> float:
    """Percentile rank of `value` within `baseline` (0-100, ties counted half).

    This is the "mean" convention of `scipy.stats.percentileofscore`, written
    out so the risk index has no extra dependency.
    """
    baseline = np.asarray(baseline, dtype=float)
    baseline = baseline[~np.isnan(baseline)]
    if baseline.size == 0 or np.isnan(value):
        return float("nan")
    below = np.count_nonzero(baseline < value)
    equal = np.count_nonzero(baseline == value)
    return 100.0 * (below + 0.5 * equal) / baseline.size


def classify(percentile: float) -> str:
    """Map a percentile onto a risk band label."""
    if percentile is None or (isinstance(percentile, float) and np.isnan(percentile)):
        return "Stable"
    for label, upper, _colour, _usdm in config.RISK_BANDS:
        if percentile <= upper:
            return label
    return config.RISK_BANDS[-1][0]


def _calendar_month(months: pd.Series) -> pd.Series:
    return months.str.slice(5, 7).astype(int)


def _monthly_baselines(observed: pd.DataFrame, value_col: str) -> dict[tuple[str, int], np.ndarray]:
    """Reference distribution per (district, calendar month) from observed data."""
    obs = observed.copy()
    obs["cal_month"] = _calendar_month(obs["month"])
    return {
        key: grp[value_col].to_numpy(dtype=float)
        for key, grp in obs.groupby(["district", "cal_month"])
    }


def rainfall_indicator(rainfall: pd.DataFrame) -> pd.DataFrame:
    """Trailing 12-month rainfall total and its percentile, per district-month.

    The percentile compares each 12-month total against that district's other
    12-month totals ending in the same calendar month, so the comparison stays
    like-for-like.
    """
    window = config.RAINFALL_WINDOW_MONTHS
    frames = []
    for name, grp in rainfall.sort_values("month").groupby("district"):
        grp = grp.copy()
        grp["rain_12m_mm"] = grp["rain_mm"].rolling(window, min_periods=window).sum()
        frames.append(grp)
    roll = pd.concat(frames, ignore_index=True)

    baselines = _monthly_baselines(roll.dropna(subset=["rain_12m_mm"]), "rain_12m_mm")
    roll["cal_month"] = _calendar_month(roll["month"])
    roll["rain_percentile"] = [
        percentile_of(v, baselines.get((d, m), np.array([])))
        for d, m, v in zip(roll["district"], roll["cal_month"], roll["rain_12m_mm"])
    ]
    roll["rain_deficit"] = (
        roll["rain_percentile"] <= config.RAINFALL_DEFICIT_PERCENTILE
    ).fillna(False).astype(int)
    return roll[["district", "month", "rain_12m_mm", "rain_percentile", "rain_deficit"]]


def build_risk_index(
    groundwater: pd.DataFrame,
    forecasts: pd.DataFrame,
    rainfall: pd.DataFrame,
) -> pd.DataFrame:
    """Score every observed and forecast district-month.

    Forecast values are scored against the *observed* historical distribution,
    so the bands mean the same thing on both sides of the timeline.
    """
    observed = groundwater[["district", "month", "gws_mm"]].copy()
    observed["period"] = "observed"

    projected = forecasts[["district", "month", "yhat"]].rename(columns={"yhat": "gws_mm"}).copy()
    projected["period"] = "forecast"
    # Drop any forecast month that overlaps the observed record.
    seen = set(zip(observed["district"], observed["month"]))
    keep = [pair not in seen for pair in zip(projected["district"], projected["month"])]
    projected = projected[keep]

    combined = pd.concat([observed, projected], ignore_index=True)
    combined["cal_month"] = _calendar_month(combined["month"])

    baselines = _monthly_baselines(groundwater, "gws_mm")
    combined["percentile"] = [
        percentile_of(v, baselines.get((d, m), np.array([])))
        for d, m, v in zip(combined["district"], combined["cal_month"], combined["gws_mm"])
    ]
    combined["risk_band"] = combined["percentile"].apply(classify)
    combined["severity"] = combined["risk_band"].map(config.RISK_SEVERITY).astype(int)

    rain = rainfall_indicator(rainfall)
    out = combined.merge(rain, on=["district", "month"], how="left")
    out["rain_deficit"] = out["rain_deficit"].fillna(0).astype(int)

    return out[
        [
            "district",
            "month",
            "gws_mm",
            "percentile",
            "risk_band",
            "severity",
            "period",
            "rain_12m_mm",
            "rain_percentile",
            "rain_deficit",
        ]
    ].sort_values(["district", "month"]).reset_index(drop=True)

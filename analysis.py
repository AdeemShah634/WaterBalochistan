"""Core analysis interface - Tier 2, presentation-independent.

Every question the system can answer is exposed here as a plain Python
function returning DataFrames / dicts. The Streamlit dashboard and the FastAPI
service both consume this module, so no analysis logic lives in either of them
and the user interface stays interchangeable.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config, db


# --------------------------------------------------------------------------
# Cached loads
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load(db_path: str | None = None) -> dict[str, pd.DataFrame]:
    path = Path(db_path) if db_path else config.DB_PATH
    return {
        "districts": db.read_table("districts", path),
        "groundwater": db.read_table("groundwater", path),
        "rainfall": db.read_table("rainfall", path),
        "forecasts": db.read_table("forecasts", path),
        "risk": db.read_table("risk_index", path),
        "metrics": db.read_table("model_metrics", path),
        "backtest": db.read_table("backtest_detail", path),
        "configs": db.read_table("config_comparison", path),
        "quality": db.read_table("data_quality", path),
    }


def refresh() -> None:
    """Drop the in-process cache (call after rebuilding the database)."""
    _load.cache_clear()


def _t(name: str) -> pd.DataFrame:
    return _load()[name].copy()


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------
def list_districts() -> list[str]:
    """Every district in the study area, alphabetically."""
    return sorted(_load()["groundwater"]["district"].unique())


def observed_months() -> list[str]:
    return sorted(_load()["groundwater"]["month"].unique())


def forecast_months(reliable_only: bool = True) -> list[str]:
    f = _load()["forecasts"]
    if reliable_only:
        f = f[f["is_reliable"] == 1]
    return sorted(f["month"].unique())


def timeline_months(reliable_only: bool = True) -> list[str]:
    """The full slider domain: observed history followed by the forecast horizon."""
    obs = set(observed_months())
    return sorted(obs | (set(forecast_months(reliable_only)) - obs))


def latest_observed_month() -> str:
    return observed_months()[-1]


def period_of(month: str) -> str:
    """'observed' or 'forecast' for a given month."""
    return "observed" if month in set(observed_months()) else "forecast"


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------
def risk_snapshot(month: str) -> pd.DataFrame:
    """Risk band for every district in one month - the map's data source."""
    risk = _t("risk")
    snap = risk[risk["month"] == month].copy()
    return snap.sort_values("severity", ascending=False).reset_index(drop=True)

def district_risk_history(district: str) -> pd.DataFrame:
    """Risk percentile and band over the whole timeline for one district."""
    risk = _t("risk")
    out = risk[risk["district"] == district].sort_values("month")
    return out.reset_index(drop=True)


def province_summary(month: str) -> dict:
    """Headline numbers for one month across the province."""
    snap = risk_snapshot(month)
    counts = snap["risk_band"].value_counts().to_dict()
    return {
        "month": month,
        "period": period_of(month),
        "districts": int(len(snap)),
        "band_counts": {band: int(counts.get(band, 0)) for band in config.RISK_ORDER},
        "at_risk": int((snap["severity"] >= config.RISK_SEVERITY["Critical"]).sum()),
        "rainfall_deficit_districts": int(snap["rain_deficit"].sum()),
        "mean_percentile": float(snap["percentile"].mean()) if len(snap) else float("nan"),
        "worst_districts": snap.head(5)["district"].tolist(),
    }


def risk_timeline(reliable_only: bool = True) -> pd.DataFrame:
    """Count of districts in each risk band, per month - a province-wide trend."""
    risk = _t("risk")
    if reliable_only:
        risk = risk[risk["month"].isin(timeline_months())]
    counts = (
        risk.groupby(["month", "risk_band"]).size().unstack(fill_value=0).reindex(
            columns=config.RISK_ORDER, fill_value=0
        )
    )
    return counts.reset_index()


# --------------------------------------------------------------------------
# Per-district series
# --------------------------------------------------------------------------
def groundwater_series(district: str) -> pd.DataFrame:
    gw = _t("groundwater")
    return gw[gw["district"] == district].sort_values("month").reset_index(drop=True)


def rainfall_series(district: str) -> pd.DataFrame:
    rain = _t("rainfall")
    out = rain[rain["district"] == district].sort_values("month").copy()
    out["rain_12m_mm"] = out["rain_mm"].rolling(
        config.RAINFALL_WINDOW_MONTHS, min_periods=config.RAINFALL_WINDOW_MONTHS
    ).sum()
    return out.reset_index(drop=True)


def forecast_series(district: str, reliable_only: bool = True) -> pd.DataFrame:
    f = _t("forecasts")
    out = f[f["district"] == district]
    if reliable_only:
        out = out[out["is_reliable"] == 1]
    return out.sort_values("month").reset_index(drop=True)


def combined_series(district: str, reliable_only: bool = True) -> pd.DataFrame:
    """Observed groundwater joined to the forecast, ready to plot as one line."""
    obs = groundwater_series(district).rename(columns={"gws_mm": "observed"})
    fc = forecast_series(district, reliable_only).rename(
        columns={"yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"}
    )
    merged = pd.merge(
        obs[["month", "observed"]],
        fc[["month", "forecast", "lower", "upper"]],
        on="month",
        how="outer",
    ).sort_values("month")
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------
# District summary (what the PDF report and the map popup both use)
# --------------------------------------------------------------------------
def district_summary(district: str, month: str | None = None) -> dict:
    """Everything a stakeholder needs about one district at one point in time."""
    month = month or latest_observed_month()
    history = district_risk_history(district)
    row = history[history["month"] == month]
    if row.empty:
        raise KeyError(f"No risk record for {district} in {month}")
    row = row.iloc[0]

    obs = groundwater_series(district)
    recent = obs.tail(config.SEASONAL_PERIOD)["gws_mm"].mean()
    earliest = obs.head(config.SEASONAL_PERIOD)["gws_mm"].mean()
    span_years = max((len(obs) - config.SEASONAL_PERIOD) / 12.0, 1e-9)

    observed_only = history[history["period"] == "observed"]
    fc = forecast_series(district)

    attrs = _t("districts")
    attr_row = attrs[attrs["district"] == district]

    return {
        "district": district,
        "month": month,
        "period": row["period"],
        "gws_mm": float(row["gws_mm"]),
        "percentile": float(row["percentile"]),
        "risk_band": row["risk_band"],
        "usdm_equivalent": config.RISK_USDM[row["risk_band"]],
        "rain_12m_mm": None if pd.isna(row["rain_12m_mm"]) else float(row["rain_12m_mm"]),
        "rain_percentile": None if pd.isna(row["rain_percentile"]) else float(row["rain_percentile"]),
        "rain_deficit": bool(row["rain_deficit"]),
        "area_km2": float(attr_row["area_km2"].iloc[0]) if not attr_row.empty else None,
        "record_start": obs["month"].iloc[0],
        "record_end": obs["month"].iloc[-1],
        "mean_gws_mm": float(obs["gws_mm"].mean()),
        "min_gws_mm": float(obs["gws_mm"].min()),
        "max_gws_mm": float(obs["gws_mm"].max()),
        "trend_mm_per_year": float((recent - earliest) / span_years),
        "months_severe_observed": int((observed_only["risk_band"] == "Severe Deficit").sum()),
        "months_observed": int(len(observed_only)),
        "forecast_end": fc["month"].iloc[-1] if not fc.empty else None,
        "forecast_end_gws_mm": float(fc["yhat"].iloc[-1]) if not fc.empty else None,
        "forecast_end_band": (
            history[history["month"] == fc["month"].iloc[-1]]["risk_band"].iloc[0]
            if not fc.empty and (history["month"] == fc["month"].iloc[-1]).any()
            else None
        ),
    }


# --------------------------------------------------------------------------
# Model validation & data quality
# --------------------------------------------------------------------------
def model_metrics(district: str | None = None) -> pd.DataFrame:
    m = _t("metrics")
    if district:
        m = m[m["district"] == district]
    return m.sort_values(["district", "model"]).reset_index(drop=True)


def metrics_summary() -> pd.DataFrame:
    """Province-wide mean RMSE / MAE per model, plus how often each model wins."""
    m = _t("metrics")
    wins = (
        m.dropna(subset=["rmse"])
        .loc[lambda d: d.groupby("district")["rmse"].idxmin()]["model"]
        .value_counts()
    )
    summary = m.groupby("model")[["rmse", "mae"]].mean().reset_index()
    summary["districts_best_rmse"] = summary["model"].map(wins).fillna(0).astype(int)
    order = {"prophet": 0, "arima": 1, "seasonal_naive": 2}
    return summary.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)


def backtest_detail(district: str | None = None) -> pd.DataFrame:
    """Per-model, per-test-window scores behind the averaged metrics."""
    b = _t("backtest")
    if district:
        b = b[b["district"] == district]
    return b.sort_values(["origin_month", "model"]).reset_index(drop=True)


def backtest_by_origin() -> pd.DataFrame:
    """Mean RMSE per model for each test window - shows how much the ranking
    depends on which period you happen to score on."""
    b = _t("backtest")
    return (
        b.groupby(["origin_month", "model"], as_index=False)["rmse"]
        .mean()
        .pivot(index="origin_month", columns="model", values="rmse")
        .reset_index()
    )


def config_comparison() -> pd.DataFrame:
    """Rolling-origin scores for the Prophet configurations that were
    considered, including Prophet's own default."""
    return _t("configs").sort_values("rmse").reset_index(drop=True)


def data_quality_report() -> pd.DataFrame:
    return _t("quality")


def districts_table() -> pd.DataFrame:
    return _t("districts")

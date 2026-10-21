"""FastAPI REST service - the second consumer of the core analysis module.

It exists to demonstrate the architectural point in the proposal: the analysis
layer is presentation-independent, so the dashboard and the API answer the same
questions from the same code. Every endpoint below is a thin wrapper.

    pip install fastapi uvicorn
    uvicorn src.api:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response

from . import analysis, config, db, report

app = FastAPI(
    title="Balochistan Groundwater Monitoring API",
    description=(
        "District-level groundwater risk, forecasts and rainfall for Balochistan, "
        "derived from NASA GLDAS-2.2 CLSM GRACE-DA and CHIRPS."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    if not db.is_built():
        raise RuntimeError("Database not built - run `python build_data.py` first.")
    analysis.refresh()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "database_built": db.is_built()}


@app.get("/districts")
def districts() -> dict:
    return {"districts": analysis.list_districts()}


@app.get("/timeline")
def timeline() -> dict:
    return {
        "observed": analysis.observed_months(),
        "forecast": analysis.forecast_months(),
        "reliable_forecast_months": config.RELIABLE_FORECAST_MONTHS,
    }


@app.get("/risk/{month}")
def risk(month: str) -> dict:
    snap = analysis.risk_snapshot(month)
    if snap.empty:
        raise HTTPException(404, f"No risk data for {month}")
    return {
        "month": month,
        "period": analysis.period_of(month),
        "districts": snap.to_dict(orient="records"),
    }


@app.get("/summary/{month}")
def summary(month: str) -> dict:
    try:
        return analysis.province_summary(month)
    except (KeyError, IndexError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/district/{district}")
def district_detail(district: str, month: str | None = None) -> dict:
    try:
        return analysis.district_summary(district, month)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/district/{district}/series")
def district_series(district: str) -> dict:
    if district not in analysis.list_districts():
        raise HTTPException(404, f"Unknown district: {district}")
    return {
        "district": district,
        "groundwater": analysis.groundwater_series(district).to_dict(orient="records"),
        "rainfall": analysis.rainfall_series(district).to_dict(orient="records"),
        "forecast": analysis.forecast_series(district).to_dict(orient="records"),
    }


@app.get("/district/{district}/report.pdf")
def district_report(district: str, month: str | None = None) -> Response:
    try:
        pdf = report.build_district_report(district, month)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    filename = f"{district.replace(' ', '_')}_{month or 'latest'}.pdf"
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/metrics")
def metrics() -> dict:
    return {
        "summary": analysis.metrics_summary().to_dict(orient="records"),
        "per_district": analysis.model_metrics().to_dict(orient="records"),
        "by_test_window": analysis.backtest_by_origin().to_dict(orient="records"),
        "config_comparison": analysis.config_comparison().to_dict(orient="records"),
        "horizon_months": config.HOLDOUT_MONTHS,
        "test_windows": config.BACKTEST_ORIGINS,
    }


@app.get("/data-quality")
def data_quality() -> dict:
    return {"checks": analysis.data_quality_report().to_dict(orient="records")}

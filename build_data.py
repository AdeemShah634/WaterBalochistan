"""One-shot pipeline runner: extract -> process -> forecast -> risk index.

    python build_data.py            # full rebuild
    python build_data.py --skip-forecast   # reload data only (fast)

Everything lands in `data/groundwater.db`, which the dashboard then reads.
"""
from __future__ import annotations

import argparse
import sys
import time

from src import analysis, config, db, forecasting, pipeline, risk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="reload and validate the source data without refitting the models",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="continue even if a data validation check fails",
    )
    args = parser.parse_args()

    started = time.time()

    print("[1/4] Extract & validate source data ...")
    report = pipeline.build_database(strict=not args.lenient)
    for row in report.itertuples():
        marker = {"PASS": "ok ", "WARN": "warn", "FAIL": "FAIL"}[row.status]
        print(f"      [{marker}] {row.check_name}: {row.detail}")

    groundwater = db.read_table("groundwater")
    rainfall = db.read_table("rainfall")
    print(
        f"      loaded {len(groundwater):,} groundwater and {len(rainfall):,} rainfall rows "
        f"for {groundwater['district'].nunique()} districts "
        f"({groundwater['month'].min()} to {groundwater['month'].max()})"
    )

    if args.skip_forecast and db.table_exists("forecasts"):
        forecasts = db.read_table("forecasts")
        print("[2/4] Forecast ... skipped (reusing stored forecasts)")
    else:
        print(
            f"[2/4] Forecast: fitting Prophet per district "
            f"({config.FORECAST_MONTHS} months ahead) and scoring it against "
            f"seasonal-naive + ARIMA over {config.BACKTEST_ORIGINS} rolling "
            f"{config.HOLDOUT_MONTHS}-month test windows ..."
        )

        def progress(i: int, total: int, name: str) -> None:
            print(f"      ({i:2d}/{total}) {name}", flush=True)

        forecasts, metrics, detail, configs = forecasting.run_all(
            groundwater, progress=progress
        )
        db.replace_table(forecasts, "forecasts")
        db.replace_table(metrics, "model_metrics")
        db.replace_table(detail, "backtest_detail")
        db.replace_table(configs, "config_comparison")

        best = metrics.dropna(subset=["rmse"]).groupby("model")["rmse"].mean()
        print("      mean RMSE by model: " + ", ".join(f"{k}={v:.2f}" for k, v in best.items()))
        if not configs.empty:
            print(
                "      Prophet configs: "
                + ", ".join(f"{r.config}={r.rmse:.2f}" for r in configs.itertuples())
            )

    print("[3/4] Compute the Groundwater Risk Index ...")
    risk_df = risk.build_risk_index(groundwater, forecasts, rainfall)
    db.replace_table(risk_df, "risk_index")
    counts = risk_df[risk_df["period"] == "observed"]["risk_band"].value_counts().to_dict()
    print("      observed band counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    print("[4/4] Done.")
    analysis.refresh()
    latest = analysis.latest_observed_month()
    summary = analysis.province_summary(latest)
    print(
        f"      {latest}: {summary['at_risk']} of {summary['districts']} districts at "
        f"Critical or worse; database at {config.DB_PATH}"
    )
    print(f"      elapsed {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

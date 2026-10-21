"""Forecast stage - per-district Prophet models, benchmarked against
seasonal-naive and ARIMA baselines by rolling-origin backtest.

Why a rolling-origin backtest rather than one holdout: on this data the final
24 months are a period the seasonal-naive baseline happens to handle unusually
well, so scoring on that window alone ranks the models wrongly. Measuring over
several test windows - each forecast from a training set that ends before it -
gives a stable ranking and is the standard way to evaluate a forecaster.

Three things happen here:

1. `backtest_district` scores every model over `BACKTEST_ORIGINS` test windows.
2. `compare_configs` scores the chosen Prophet configuration against the
   alternatives, so the Model Validation tab can show why it was chosen.
3. `forecast_district` refits Prophet on the *full* series and projects
   `FORECAST_MONTHS` ahead with uncertainty intervals, flagging the first
   `RELIABLE_FORECAST_MONTHS` as the horizon presented as reliable.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from . import config

warnings.filterwarnings("ignore")


def _quieten_solver_logs() -> None:
    """Prophet's Stan backend chatters on every fit; keep the console readable."""
    for name in ("prophet", "cmdstanpy", "prophet.plot"):
        log = logging.getLogger(name)
        log.setLevel(logging.CRITICAL)
        log.propagate = False
        log.handlers = [logging.NullHandler()]


_quieten_solver_logs()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def to_timeseries(df: pd.DataFrame, value_col: str = "gws_mm") -> pd.DataFrame:
    """Convert a district's monthly rows into a Prophet-style ds/y frame."""
    out = df.copy()
    out["ds"] = pd.PeriodIndex(out["month"], freq="M").to_timestamp()
    out = out.sort_values("ds").reset_index(drop=True)
    return out[["ds", value_col]].rename(columns={value_col: "y"})


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def backtest_origins(
    n_obs: int,
    horizon: int | None = None,
    n_origins: int | None = None,
    stride: int | None = None,
) -> list[int]:
    """Training-set end indices for the rolling-origin backtest, newest first.

    Each origin `o` means: train on `ts[:o]`, forecast `horizon` months, score
    against `ts[o : o + horizon]`. Origins that would leave too little history
    to fit a seasonal model are dropped.
    """
    horizon = horizon or config.HOLDOUT_MONTHS
    n_origins = n_origins or config.BACKTEST_ORIGINS
    stride = stride or config.BACKTEST_STRIDE

    origins = [n_obs - horizon - stride * k for k in range(n_origins)]
    floor = horizon + 3 * config.SEASONAL_PERIOD
    return [o for o in origins if o >= floor]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def _fit_prophet(train: pd.DataFrame, params: dict | None = None):
    from prophet import Prophet

    _quieten_solver_logs()
    model = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=config.PROPHET_INTERVAL_WIDTH,
        **(params or config.PROPHET_PARAMS),
    )
    model.fit(train)
    return model


def prophet_predict(
    train: pd.DataFrame, periods: int, params: dict | None = None
) -> pd.DataFrame:
    """Fit Prophet on `train` and return `periods` months of monthly forecast."""
    model = _fit_prophet(train, params)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    fcst = model.predict(future)
    return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)


def seasonal_naive_predict(train: pd.DataFrame, periods: int) -> np.ndarray:
    """Baseline: repeat the value observed `SEASONAL_PERIOD` months earlier."""
    m = config.SEASONAL_PERIOD
    history = list(train["y"].to_numpy())
    preds = []
    for i in range(periods):
        preds.append(history[-m + (i % m)] if len(history) >= m else history[-1])
    return np.asarray(preds, dtype=float)


def arima_predict(train: pd.DataFrame, periods: int) -> np.ndarray:
    """Baseline: SARIMA(1,1,1)(1,1,1,12) via statsmodels."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = pd.Series(train["y"].to_numpy(), index=pd.DatetimeIndex(train["ds"]))
    series.index.freq = "MS"
    model = SARIMAX(
        series,
        order=config.ARIMA_ORDER,
        seasonal_order=config.ARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return np.asarray(fitted.forecast(steps=periods), dtype=float)


PREDICTORS = {
    "prophet": lambda train, periods: prophet_predict(train, periods)["yhat"].to_numpy(),
    "seasonal_naive": seasonal_naive_predict,
    "arima": arima_predict,
}


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def backtest_district(
    ts: pd.DataFrame,
    horizon: int | None = None,
    origins: list[int] | None = None,
    predictors: dict | None = None,
) -> pd.DataFrame:
    """Score each model over every rolling test window for one district.

    Returns one row per (model, origin) with RMSE and MAE. A model that fails
    to fit at one origin is recorded as NaN there rather than aborting the run.
    """
    horizon = horizon or config.HOLDOUT_MONTHS
    origins = origins if origins is not None else backtest_origins(len(ts), horizon)
    predictors = predictors or PREDICTORS

    rows = []
    for cut in origins:
        train, test = ts.iloc[:cut], ts.iloc[cut : cut + horizon]
        actual = test["y"].to_numpy()
        origin_month = test["ds"].iloc[0].strftime("%Y-%m")
        for name, predict in predictors.items():
            try:
                pred = np.asarray(predict(train, len(test)), dtype=float)
                rows.append(
                    {
                        "model": name,
                        "origin_month": origin_month,
                        "rmse": rmse(actual, pred),
                        "mae": mae(actual, pred),
                    }
                )
            except Exception as exc:  # pragma: no cover - solver dependent
                logging.warning("%s failed at origin %s: %s", name, origin_month, exc)
                rows.append(
                    {
                        "model": name,
                        "origin_month": origin_month,
                        "rmse": float("nan"),
                        "mae": float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_backtest(detail: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-origin scores into one row per model."""
    return detail.groupby("model", as_index=False).agg(
        rmse=("rmse", "mean"), mae=("mae", "mean"), origins=("rmse", "count")
    )


def compare_configs(
    ts: pd.DataFrame,
    horizon: int | None = None,
    origins: list[int] | None = None,
    candidates: dict | None = None,
) -> pd.DataFrame:
    """Score the candidate Prophet configurations over the same test windows.

    This is what justifies `config.PROPHET_PARAMS` on the Model Validation tab:
    the chosen setting is shown next to the alternatives it was chosen over,
    including Prophet's own default.
    """
    candidates = candidates or config.PROPHET_CANDIDATES
    predictors = {
        name: (
            lambda train, periods, p=params: prophet_predict(train, periods, p)[
                "yhat"
            ].to_numpy()
        )
        for name, params in candidates.items()
    }
    detail = backtest_district(ts, horizon, origins, predictors)
    return detail.rename(columns={"model": "config"})


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------
def forecast_district(
    ts: pd.DataFrame, periods: int | None = None, params: dict | None = None
) -> pd.DataFrame:
    """Refit Prophet on the full series and project ahead with intervals."""
    periods = periods or config.FORECAST_MONTHS
    fcst = prophet_predict(ts, periods, params)
    fcst["month"] = fcst["ds"].dt.strftime("%Y-%m")
    fcst["is_reliable"] = (
        np.arange(1, len(fcst) + 1) <= config.RELIABLE_FORECAST_MONTHS
    ).astype(int)
    return fcst[["month", "yhat", "yhat_lower", "yhat_upper", "is_reliable"]]


def run_all(
    groundwater: pd.DataFrame, progress=None, compare: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest and forecast every district.

    Returns `(forecasts, metrics, backtest_detail, config_comparison)`, all
    ready to write to the data layer. `progress` is an optional callable
    `(i, total, district)` for CLI/UI feedback; `compare=False` skips the
    configuration comparison, which roughly halves the runtime.
    """
    districts = sorted(groundwater["district"].unique())
    all_fcst, all_metrics, all_detail, all_configs = [], [], [], []

    for i, name in enumerate(districts, start=1):
        if progress:
            progress(i, len(districts), name)
        ts = to_timeseries(groundwater[groundwater["district"] == name])
        origins = backtest_origins(len(ts))

        detail = backtest_district(ts, origins=origins)
        detail.insert(0, "district", name)
        all_detail.append(detail)

        metrics = aggregate_backtest(detail)
        metrics.insert(0, "district", name)
        all_metrics.append(metrics)

        if compare:
            comparison = compare_configs(ts, origins=origins)
            comparison.insert(0, "district", name)
            all_configs.append(comparison)

        fcst = forecast_district(ts)
        fcst.insert(0, "district", name)
        all_fcst.append(fcst)

    config_summary = (
        pd.concat(all_configs, ignore_index=True)
        .groupby("config", as_index=False)
        .agg(rmse=("rmse", "mean"), mae=("mae", "mean"))
        .sort_values("rmse")
        .reset_index(drop=True)
        if all_configs
        else pd.DataFrame(columns=["config", "rmse", "mae"])
    )

    return (
        pd.concat(all_fcst, ignore_index=True),
        pd.concat(all_metrics, ignore_index=True),
        pd.concat(all_detail, ignore_index=True),
        config_summary,
    )

"""Data Layer - SQLite schema and read/write helpers.

Tier 3 of the three-tier design. Nothing in this module knows anything about
Streamlit, Folium or FastAPI; it only stores and returns tabular data.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS districts (
    district        TEXT PRIMARY KEY,
    area_km2        REAL,
    centroid_lat    REAL,
    centroid_lon    REAL
);

CREATE TABLE IF NOT EXISTS groundwater (
    district        TEXT NOT NULL,
    month           TEXT NOT NULL,          -- 'YYYY-MM'
    gws_mm          REAL NOT NULL,
    PRIMARY KEY (district, month)
);

CREATE TABLE IF NOT EXISTS rainfall (
    district        TEXT NOT NULL,
    month           TEXT NOT NULL,
    rain_mm         REAL NOT NULL,
    PRIMARY KEY (district, month)
);

CREATE TABLE IF NOT EXISTS forecasts (
    district        TEXT NOT NULL,
    month           TEXT NOT NULL,
    yhat            REAL NOT NULL,
    yhat_lower      REAL,
    yhat_upper      REAL,
    is_reliable     INTEGER NOT NULL DEFAULT 1,   -- within the 3-year horizon
    PRIMARY KEY (district, month)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    district        TEXT NOT NULL,
    model           TEXT NOT NULL,          -- prophet | seasonal_naive | arima
    rmse            REAL,                   -- averaged over the backtest origins
    mae             REAL,
    origins         INTEGER,                -- how many test windows were scored
    PRIMARY KEY (district, model)
);

CREATE TABLE IF NOT EXISTS backtest_detail (
    district        TEXT NOT NULL,
    model           TEXT NOT NULL,
    origin_month    TEXT NOT NULL,          -- first month of that test window
    rmse            REAL,
    mae             REAL,
    PRIMARY KEY (district, model, origin_month)
);

CREATE TABLE IF NOT EXISTS config_comparison (
    config          TEXT PRIMARY KEY,       -- Prophet configuration label
    rmse            REAL,
    mae             REAL
);

CREATE TABLE IF NOT EXISTS risk_index (
    district        TEXT NOT NULL,
    month           TEXT NOT NULL,
    gws_mm          REAL NOT NULL,
    percentile      REAL NOT NULL,
    risk_band       TEXT NOT NULL,
    severity        INTEGER NOT NULL,
    period          TEXT NOT NULL,          -- observed | forecast
    rain_12m_mm     REAL,
    rain_percentile REAL,
    rain_deficit    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (district, month)
);

CREATE TABLE IF NOT EXISTS data_quality (
    check_name      TEXT PRIMARY KEY,
    status          TEXT NOT NULL,          -- PASS | WARN | FAIL
    detail          TEXT
);

CREATE INDEX IF NOT EXISTS idx_gw_district ON groundwater(district);
CREATE INDEX IF NOT EXISTS idx_rain_district ON rainfall(district);
CREATE INDEX IF NOT EXISTS idx_risk_month ON risk_index(month);
CREATE INDEX IF NOT EXISTS idx_risk_district ON risk_index(district);
"""


@contextmanager
def connect(db_path: Path | str | None = None):
    """Yield a SQLite connection with foreign keys on, closing it afterwards."""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(db_path: Path | str | None = None) -> None:
    """Create every table if it does not already exist."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def replace_table(df: pd.DataFrame, table: str, db_path: Path | str | None = None) -> int:
    """Overwrite `table` with `df`. Returns the number of rows written."""
    with connect(db_path) as conn:
        conn.execute(f"DELETE FROM {table}")
        df.to_sql(table, conn, if_exists="append", index=False)
    return len(df)


def read_table(table: str, db_path: Path | str | None = None) -> pd.DataFrame:
    """Read a whole table into a DataFrame."""
    with connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def query(sql: str, params: tuple = (), db_path: Path | str | None = None) -> pd.DataFrame:
    """Run an arbitrary read query and return a DataFrame."""
    with connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def table_exists(table: str, db_path: Path | str | None = None) -> bool:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
    return row is not None


def is_built(db_path: Path | str | None = None) -> bool:
    """True when the database exists and the pipeline has populated it."""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    if not path.exists():
        return False
    try:
        with connect(path) as conn:
            for table in ("groundwater", "rainfall", "risk_index", "forecasts"):
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone()
                if row is None:
                    return False
                if conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0:
                    return False
    except sqlite3.DatabaseError:
        return False
    return True

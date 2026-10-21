"""Extract & Process stages - load the Earth Engine exports, validate them and
populate the SQLite data layer.

The Earth Engine extraction itself (GLDAS-2.2 CLSM GRACE-DA `GWS_tavg` and
CHIRPS, aggregated per district by server-side zonal statistics) is run
separately and lands in `data/raw/` as CSV + GeoJSON. This module is the
reproducible, version-controlled step that turns those exports into a validated
database, and it emits a data quality report while doing so.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon
from shapely.validation import make_valid

from . import config, db


# --------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------
def polygonal_only(geom):
    """Reduce a geometry to its polygonal parts as a MultiPolygon.

    Seven districts in the source boundary file arrive as GeometryCollections
    carrying stray Points and LineStrings alongside the real polygons -
    digitizing artifacts. Those parts have no area, break Folium's bounds
    calculation, and would silently distort a zonal-statistics area figure, so
    they are dropped here and the fact is recorded in the data quality report.
    """
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    parts = geom.geoms if geom.geom_type.startswith(("Multi", "Geometry")) else [geom]
    polygons: list = []
    for part in parts:
        if part.geom_type == "Polygon":
            polygons.append(part)
        elif part.geom_type == "MultiPolygon":
            polygons.extend(part.geoms)
    return MultiPolygon(polygons) if polygons else None


def load_districts(path: Path | None = None) -> gpd.GeoDataFrame:
    """Read the district boundary GeoJSON into a clean GeoDataFrame (EPSG:4326)."""
    path = path or config.DISTRICTS_GEOJSON
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.rename(columns={config.GEOJSON_NAME_FIELD: "district"})
    gdf["district"] = gdf["district"].astype(str).str.strip()
    gdf["raw_geom_type"] = gdf.geometry.geom_type
    gdf["geometry"] = gdf.geometry.apply(polygonal_only)
    gdf = gdf[gdf.geometry.notna()].set_geometry("geometry")
    return gdf[["district", "raw_geom_type", "geometry"]]


def district_attributes(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Derive per-district area and centroid with geopandas.

    Area is computed in an equal-area projection (World Cylindrical Equal Area)
    so square-kilometre figures are meaningful; centroids are taken in the same
    projected CRS and converted back to lat/lon for map labelling.
    """
    projected = gdf.to_crs("ESRI:54034")  # World Cylindrical Equal Area
    centroids = projected.geometry.centroid.to_crs("EPSG:4326")
    return pd.DataFrame(
        {
            "district": gdf["district"].to_numpy(),
            "area_km2": (projected.geometry.area / 1e6).to_numpy(),
            "centroid_lat": centroids.y.to_numpy(),
            "centroid_lon": centroids.x.to_numpy(),
        }
    )


def load_series(path: Path, value_col: str) -> pd.DataFrame:
    """Read one of the monthly CSV exports."""
    df = pd.read_csv(path)
    df["district"] = df["district"].astype(str).str.strip()
    df["month"] = df["month"].astype(str).str.strip()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df[["district", "month", value_col]].sort_values(["district", "month"])


# --------------------------------------------------------------------------
# Validate
# --------------------------------------------------------------------------
def _check(name: str, ok: bool, detail: str, warn_only: bool = False) -> dict:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    return {"check_name": name, "status": status, "detail": detail}


def validate(
    gw: pd.DataFrame, rain: pd.DataFrame, gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Run the data quality checks described in the methodology.

    Covers district-name matching against the boundary file, coverage and gap
    checks, and null / value-range checks. Returns a tidy report.
    """
    checks: list[dict] = []

    geo_names = set(gdf["district"])
    gw_names = set(gw["district"])
    rain_names = set(rain["district"])

    checks.append(
        _check(
            "district_count",
            len(geo_names) == config.EXPECTED_DISTRICT_COUNT,
            f"{len(geo_names)} districts in boundary file "
            f"(expected {config.EXPECTED_DISTRICT_COUNT})",
        )
    )
    missing_in_gw = sorted(geo_names - gw_names)
    extra_in_gw = sorted(gw_names - geo_names)
    checks.append(
        _check(
            "name_match_groundwater",
            not missing_in_gw and not extra_in_gw,
            "All district names match the boundary file"
            if not missing_in_gw and not extra_in_gw
            else f"missing={missing_in_gw} unexpected={extra_in_gw}",
        )
    )
    missing_in_rain = sorted(geo_names - rain_names)
    extra_in_rain = sorted(rain_names - geo_names)
    checks.append(
        _check(
            "name_match_rainfall",
            not missing_in_rain and not extra_in_rain,
            "All district names match the boundary file"
            if not missing_in_rain and not extra_in_rain
            else f"missing={missing_in_rain} unexpected={extra_in_rain}",
        )
    )

    # Nulls
    checks.append(
        _check(
            "nulls_groundwater",
            gw["gws_mm"].isna().sum() == 0,
            f"{int(gw['gws_mm'].isna().sum())} null groundwater values",
        )
    )
    checks.append(
        _check(
            "nulls_rainfall",
            rain["rain_mm"].isna().sum() == 0,
            f"{int(rain['rain_mm'].isna().sum())} null rainfall values",
        )
    )

    # Value ranges
    lo, hi = config.GWS_VALID_RANGE
    out_gw = int(((gw["gws_mm"] < lo) | (gw["gws_mm"] > hi)).sum())
    checks.append(
        _check("range_groundwater", out_gw == 0, f"{out_gw} values outside {lo}-{hi} mm")
    )
    lo, hi = config.RAIN_VALID_RANGE
    out_rain = int(((rain["rain_mm"] < lo) | (rain["rain_mm"] > hi)).sum())
    checks.append(
        _check("range_rainfall", out_rain == 0, f"{out_rain} values outside {lo}-{hi} mm")
    )

    # Duplicates
    dup_gw = int(gw.duplicated(["district", "month"]).sum())
    checks.append(_check("duplicates_groundwater", dup_gw == 0, f"{dup_gw} duplicate rows"))
    dup_rain = int(rain.duplicated(["district", "month"]).sum())
    checks.append(_check("duplicates_rainfall", dup_rain == 0, f"{dup_rain} duplicate rows"))

    # Coverage / gaps: every district must carry an unbroken monthly series.
    gaps = []
    for name, grp in gw.groupby("district"):
        periods = pd.PeriodIndex(grp["month"], freq="M").sort_values()
        expected = pd.period_range(periods.min(), periods.max(), freq="M")
        missing = expected.difference(periods)
        if len(missing):
            gaps.append(f"{name}: {len(missing)} missing months")
    checks.append(
        _check(
            "coverage_groundwater",
            not gaps,
            "; ".join(gaps) if gaps else "Continuous monthly coverage for every district",
        )
    )

    # Aligned span between the two variables
    same_span = (
        gw["month"].min() == rain["month"].min() and gw["month"].max() == rain["month"].max()
    )
    checks.append(
        _check(
            "aligned_time_span",
            same_span,
            f"groundwater {gw['month'].min()}..{gw['month'].max()} / "
            f"rainfall {rain['month'].min()}..{rain['month'].max()}",
        )
    )

    # Geometry hygiene
    cleaned = (
        sorted(gdf.loc[gdf["raw_geom_type"] == "GeometryCollection", "district"])
        if "raw_geom_type" in gdf.columns
        else []
    )
    checks.append(
        _check(
            "geometry_cleaned",
            not cleaned,
            f"{len(cleaned)} district(s) arrived as GeometryCollections; non-polygonal "
            f"parts were dropped: {', '.join(cleaned)}"
            if cleaned
            else "All boundaries are polygonal as supplied",
            warn_only=True,
        )
    )
    checks.append(
        _check(
            "geometry_valid",
            bool(gdf.geometry.is_valid.all()),
            f"{int((~gdf.geometry.is_valid).sum())} invalid geometries after cleaning",
        )
    )

    # Series length (Prophet needs a reasonable history plus the holdout)
    min_len = int(gw.groupby("district").size().min())
    need = config.HOLDOUT_MONTHS + 3 * config.SEASONAL_PERIOD
    checks.append(
        _check(
            "series_length",
            min_len >= need,
            f"shortest district series is {min_len} months (need >= {need})",
        )
    )

    return pd.DataFrame(checks)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def build_database(db_path: Path | None = None, strict: bool = True) -> pd.DataFrame:
    """Run extract -> validate -> load and return the data quality report."""
    gdf = load_districts()
    gw = load_series(config.GROUNDWATER_CSV, "gws_mm")
    rain = load_series(config.RAINFALL_CSV, "rain_mm")

    report = validate(gw, rain, gdf)
    failures = report[report["status"] == "FAIL"]
    if strict and not failures.empty:
        raise ValueError(
            "Data validation failed:\n"
            + "\n".join(f"  - {r.check_name}: {r.detail}" for r in failures.itertuples())
        )

    db.init_schema(db_path)
    db.replace_table(district_attributes(gdf), "districts", db_path)
    db.replace_table(gw, "groundwater", db_path)
    db.replace_table(rain, "rainfall", db_path)
    db.replace_table(report, "data_quality", db_path)
    return report


def geojson_dict(path: Path | None = None) -> dict:
    """Load the boundary file as a plain GeoJSON dict (for Folium)."""
    path = path or config.DISTRICTS_GEOJSON
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    for feature in gj["features"]:
        props = feature.setdefault("properties", {})
        props["district"] = str(props.get(config.GEOJSON_NAME_FIELD, "")).strip()
    return gj

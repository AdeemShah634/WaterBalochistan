"""Data validation and data-layer tests."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src import config, db, pipeline


@pytest.fixture
def gdf() -> gpd.GeoDataFrame:
    polys = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(2)]
    return gpd.GeoDataFrame(
        {"district": ["A District", "B District"]}, geometry=polys, crs="EPSG:4326"
    )


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    months = [str(m) for m in pd.period_range("2010-01", periods=80, freq="M")]
    gw = pd.DataFrame(
        [{"district": d, "month": m, "gws_mm": 500.0} for d in ("A District", "B District") for m in months]
    )
    rain = pd.DataFrame(
        [{"district": d, "month": m, "rain_mm": 10.0} for d in ("A District", "B District") for m in months]
    )
    return gw, rain


def _status(report: pd.DataFrame, name: str) -> str:
    return report.loc[report["check_name"] == name, "status"].iloc[0]


def test_validate_clean_data_passes_every_name_and_value_check(frames, gdf):
    gw, rain = frames
    report = pipeline.validate(gw, rain, gdf)
    for check in (
        "name_match_groundwater",
        "name_match_rainfall",
        "nulls_groundwater",
        "range_groundwater",
        "duplicates_groundwater",
        "coverage_groundwater",
        "aligned_time_span",
        "series_length",
    ):
        assert _status(report, check) == "PASS", check


def test_validate_catches_a_district_name_mismatch(frames, gdf):
    gw, rain = frames
    gw.loc[gw["district"] == "A District", "district"] = "A Distrikt"
    report = pipeline.validate(gw, rain, gdf)
    assert _status(report, "name_match_groundwater") == "FAIL"


def test_validate_catches_a_gap_in_coverage(frames, gdf):
    gw, rain = frames
    gw = gw[gw["month"] != "2012-05"]
    report = pipeline.validate(gw, rain, gdf)
    assert _status(report, "coverage_groundwater") == "FAIL"


def test_validate_catches_out_of_range_values(frames, gdf):
    gw, rain = frames
    gw.loc[0, "gws_mm"] = -5.0
    report = pipeline.validate(gw, rain, gdf)
    assert _status(report, "range_groundwater") == "FAIL"


def test_validate_catches_nulls(frames, gdf):
    gw, rain = frames
    gw.loc[0, "gws_mm"] = None
    report = pipeline.validate(gw, rain, gdf)
    assert _status(report, "nulls_groundwater") == "FAIL"


def test_validate_catches_duplicates(frames, gdf):
    gw, rain = frames
    gw = pd.concat([gw, gw.iloc[[0]]], ignore_index=True)
    report = pipeline.validate(gw, rain, gdf)
    assert _status(report, "duplicates_groundwater") == "FAIL"


def test_district_attributes_are_positive_and_inside_bounds(gdf):
    attrs = pipeline.district_attributes(gdf)
    assert len(attrs) == 2
    assert (attrs["area_km2"] > 0).all()
    assert attrs["centroid_lon"].between(0, 2).all()
    assert attrs["centroid_lat"].between(0, 1).all()


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------
def test_roundtrip_through_sqlite(tmp_path, frames):
    path = tmp_path / "test.db"
    gw, _rain = frames
    db.init_schema(path)
    assert db.table_exists("groundwater", path)
    db.replace_table(gw, "groundwater", path)
    back = db.read_table("groundwater", path)
    assert len(back) == len(gw)
    pd.testing.assert_frame_equal(
        back.sort_values(["district", "month"]).reset_index(drop=True),
        gw.sort_values(["district", "month"]).reset_index(drop=True),
    )


def test_replace_table_overwrites_rather_than_appends(tmp_path, frames):
    path = tmp_path / "test.db"
    gw, _ = frames
    db.init_schema(path)
    db.replace_table(gw, "groundwater", path)
    db.replace_table(gw.head(3), "groundwater", path)
    assert len(db.read_table("groundwater", path)) == 3


def test_is_built_false_for_an_empty_database(tmp_path):
    path = tmp_path / "empty.db"
    db.init_schema(path)
    assert db.is_built(path) is False


def test_is_built_false_when_file_missing(tmp_path):
    assert db.is_built(tmp_path / "nope.db") is False


def test_query_accepts_parameters(tmp_path, frames):
    path = tmp_path / "test.db"
    gw, _ = frames
    db.init_schema(path)
    db.replace_table(gw, "groundwater", path)
    out = db.query(
        "SELECT * FROM groundwater WHERE district = ?", ("A District",), path
    )
    assert set(out["district"]) == {"A District"}


def test_real_boundary_file_loads_with_expected_districts():
    gdf = pipeline.load_districts()
    assert len(gdf) == config.EXPECTED_DISTRICT_COUNT
    assert gdf.crs.to_string() == "EPSG:4326"
    assert gdf["geometry"].is_valid.all()


# --------------------------------------------------------------------------
# Geometry cleaning
# --------------------------------------------------------------------------
def test_polygonal_only_keeps_polygons_and_drops_stray_parts():
    from shapely.geometry import GeometryCollection, LineString, Point

    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    messy = GeometryCollection([Point(5, 5), LineString([(0, 0), (1, 1)]), square])
    cleaned = pipeline.polygonal_only(messy)
    assert cleaned.geom_type == "MultiPolygon"
    assert cleaned.area == pytest.approx(square.area)


def test_polygonal_only_returns_none_when_nothing_polygonal_remains():
    from shapely.geometry import LineString

    assert pipeline.polygonal_only(LineString([(0, 0), (1, 1)])) is None


def test_real_boundary_file_is_polygonal_after_cleaning():
    gdf = pipeline.load_districts()
    assert set(gdf.geometry.geom_type) == {"MultiPolygon"}
    assert (gdf.to_crs("ESRI:54034").geometry.area > 0).all()

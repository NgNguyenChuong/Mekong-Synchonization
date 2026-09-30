"""Kiem thu sinh luoi (T1-03): 4 khung, quy tac chon o chung, khop dien tich.

Luoi duoc sinh truc tiep tu ranh gioi chuan (khong doc data/grids/, vi thu
muc do bi .gitignore) o do phan giai tho de chay nhanh.
"""
import os
import sys

import geopandas as gpd
import h3
import pytest
from pyproj import Geod
from shapely.geometry import Point, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import (  # noqa: E402
    CANONICAL_BOUNDARY,
    generate_h3_grid,
    generate_latlon_grid,
    generate_s2_grid,
    generate_square_grid,
)

GEOD = Geod(ellps="WGS84")


def _km2(geom):
    return abs(GEOD.geometry_area_perimeter(geom)[0]) / 1e6


@pytest.fixture(scope="module")
def boundary_union():
    return gpd.read_file(CANONICAL_BOUNDARY).to_crs(4326).geometry.union_all()


@pytest.fixture(scope="module")
def grids_res5():
    return {
        "h3": generate_h3_grid(CANONICAL_BOUNDARY, 5),
        "s2": generate_s2_grid(CANONICAL_BOUNDARY, 9),
        "square": generate_square_grid(CANONICAL_BOUNDARY, 17087),
        "latlon": generate_latlon_grid(CANONICAL_BOUNDARY, 0.1552),
    }


@pytest.mark.parametrize("name", ["h3", "s2", "square", "latlon"])
def test_schema_ids_and_geometry(grids_res5, name):
    gdf = grids_res5[name]
    assert list(gdf.columns) == ["cell_id", "overlap_frac", "geometry"]
    assert len(gdf) > 0
    assert gdf["cell_id"].is_unique
    assert all(isinstance(v, str) and v for v in gdf["cell_id"])
    assert gdf.geometry.is_valid.all()
    assert gdf["overlap_frac"].between(0.0, 1.0).all()
    assert str(gdf.crs) == "EPSG:4326"


@pytest.mark.parametrize("name", ["h3", "s2", "square", "latlon"])
def test_centroids_inside_boundary(grids_res5, boundary_union, name):
    gdf = grids_res5[name]
    assert all(boundary_union.contains(g.centroid) for g in gdf.geometry)


@pytest.mark.parametrize("name", ["h3", "s2", "square", "latlon"])
def test_cells_do_not_overlap(grids_res5, name):
    gdf = grids_res5[name]
    total = sum(_km2(g) for g in gdf.geometry)
    union = _km2(gdf.geometry.union_all())
    assert total == pytest.approx(union, rel=1e-3)


def test_same_selection_rule_gives_similar_coverage(grids_res5, boundary_union):
    boundary_km2 = _km2(boundary_union)
    for name, gdf in grids_res5.items():
        covered = sum(_km2(g) for g in gdf.geometry)
        assert covered / boundary_km2 == pytest.approx(1.0, abs=0.05), name


def test_square_and_latlon_match_local_h3_area(grids_res5):
    h3_mean = gpd.GeoSeries(grids_res5["h3"].geometry).apply(_km2).mean()
    for name in ("square", "latlon"):
        mean = gpd.GeoSeries(grids_res5[name].geometry).apply(_km2).mean()
        assert mean / h3_mean == pytest.approx(1.0, abs=0.05), name


def test_h3_matches_independent_polyfill(boundary_union):
    gdf = generate_h3_grid(CANONICAL_BOUNDARY, 6)
    parts = boundary_union.geoms if boundary_union.geom_type == "MultiPolygon" else [boundary_union]
    expected = set()
    for part in parts:
        if len(set(part.exterior.coords)) < 3:
            continue
        outer = [(lat, lon) for lon, lat in part.exterior.coords]
        holes = [[(lat, lon) for lon, lat in r.coords] for r in part.interiors if len(set(r.coords)) >= 3]
        for cell in h3.polygon_to_cells(h3.LatLngPoly(outer, *holes), 6):
            lat, lon = h3.cell_to_latlng(cell)
            if boundary_union.contains(Point(lon, lat)):
                expected.add(cell)
    assert set(gdf["cell_id"]) == expected


def test_square_grid_on_synthetic_boundary(tmp_path):
    # Hinh vuong 4 x 4 km trong UTM 48N, neo dung boi so 1000 m -> dung 16 o, khong o nao bi cat.
    square = gpd.GeoDataFrame(geometry=[box(500000, 1100000, 504000, 1104000)], crs="EPSG:32648").to_crs(4326)
    path = tmp_path / "square.geojson"
    square.to_file(path, driver="GeoJSON")
    gdf = generate_square_grid(str(path), 1000)
    assert len(gdf) == 16
    assert (gdf["overlap_frac"] > 0.99).all()


def test_latlon_grid_is_anchored_at_origin(tmp_path):
    step = 0.05
    region = gpd.GeoDataFrame(geometry=[box(105.0, 10.0, 105.2, 10.1)], crs="EPSG:4326")
    path = tmp_path / "region.geojson"
    region.to_file(path, driver="GeoJSON")
    gdf = generate_latlon_grid(str(path), step)
    assert len(gdf) == 8
    for geom in gdf.geometry:
        minx, miny, _, _ = geom.bounds
        assert round(minx / step, 6) == pytest.approx(round(minx / step))
        assert round(miny / step, 6) == pytest.approx(round(miny / step))

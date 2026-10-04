"""Kiem thu sinh luoi (T1-03): 4 khung, quy tac chon o chung, khop dien tich.

Quy tac chon o (An duyet 2026-10-03): o thuoc luoi neu GIAO ranh gioi (overlap_frac > 0),
giu nguyen hinh o -> moi diem trong ranh gioi thuoc it nhat 1 o (truoc day: theo tam o).

Luoi duoc sinh truc tiep tu ranh gioi chuan (khong doc data/grids/, vi thu
muc do bi .gitignore) o do phan giai tho de chay nhanh.
"""
import os
import sys

import geopandas as gpd
import h3
import numpy as np
import pytest
from pyproj import Geod
from shapely.geometry import Point, Polygon, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import (  # noqa: E402
    CANONICAL_BOUNDARY,
    MIN_OVERLAP_FRAC,
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
def test_every_cell_intersects_boundary(grids_res5, boundary_union, name):
    gdf = grids_res5[name]
    assert (gdf["overlap_frac"] > MIN_OVERLAP_FRAC).all()
    assert all(boundary_union.intersection(g).area > 0 for g in gdf.geometry)


@pytest.mark.parametrize("name", ["h3", "s2", "square", "latlon"])
def test_every_boundary_point_is_in_some_cell(grids_res5, boundary_union, name):
    """Quy tac giao: khong con dai ven ranh gioi bi bo sot (quy tac tam o tung bo sot)."""
    gdf = grids_res5[name]
    cells = gdf.geometry.union_all()
    assert boundary_union.difference(cells).area / boundary_union.area < 1e-6
    # Diem ngau nhien trong ranh gioi + dinh ranh gioi (nam ngay tren duong bien).
    rng = np.random.default_rng(0)
    minx, miny, maxx, maxy = boundary_union.bounds
    xs, ys = rng.uniform(minx, maxx, 20000), rng.uniform(miny, maxy, 20000)
    pts = [Point(x, y) for x, y in zip(xs, ys) if boundary_union.contains(Point(x, y))]
    parts = boundary_union.geoms if boundary_union.geom_type == "MultiPolygon" else [boundary_union]
    pts += [Point(c) for p in parts for c in p.exterior.coords[::5]]
    pts = gpd.GeoDataFrame(geometry=pts, crs=4326)
    hit = gpd.sjoin(pts, gdf[["geometry"]], predicate="intersects", how="left")
    missing = hit.loc[hit["index_right"].isna()]
    assert len(missing) == 0, f"{len(missing)} diem khong thuoc o nao"


@pytest.mark.parametrize("name", ["h3", "s2", "square", "latlon"])
def test_cells_do_not_overlap(grids_res5, name):
    gdf = grids_res5[name]
    total = sum(_km2(g) for g in gdf.geometry)
    union = _km2(gdf.geometry.union_all())
    assert total == pytest.approx(union, rel=1e-3)


def test_overlap_frac_sums_to_boundary_area(grids_res5, boundary_union):
    """O khong cat + khong chong nhau -> tong (overlap_frac x dien tich o) = dien tich ranh gioi;
    tong dien tich o >= ranh gioi (o ven bien giu nguyen hinh)."""
    boundary_km2 = _km2(boundary_union)
    for name, gdf in grids_res5.items():
        areas = np.array([_km2(g) for g in gdf.geometry])
        assert (areas * gdf["overlap_frac"].to_numpy()).sum() / boundary_km2 == pytest.approx(1.0, abs=0.01), name
        assert areas.sum() >= boundary_km2, name


def test_square_and_latlon_match_local_h3_area(grids_res5):
    h3_mean = gpd.GeoSeries(grids_res5["h3"].geometry).apply(_km2).mean()
    for name in ("square", "latlon"):
        mean = gpd.GeoSeries(grids_res5[name].geometry).apply(_km2).mean()
        assert mean / h3_mean == pytest.approx(1.0, abs=0.05), name


def test_h3_matches_independent_overlap_polyfill(boundary_union):
    """Doi chieu voi polyfill 'overlap' cua chinh H3 (doc lap voi cach sinh ung vien bang buffer)."""
    gdf = generate_h3_grid(CANONICAL_BOUNDARY, 6)
    parts = boundary_union.geoms if boundary_union.geom_type == "MultiPolygon" else [boundary_union]
    expected = set()
    for part in parts:
        if len(set(part.exterior.coords)) < 3:
            continue
        outer = [(lat, lon) for lon, lat in part.exterior.coords]
        holes = [[(lat, lon) for lon, lat in r.coords] for r in part.interiors if len(set(r.coords)) >= 3]
        poly = h3.LatLngPoly(outer, *holes)
        for cell in h3.polygon_to_cells_experimental(poly, 6, contain="overlap"):
            hexagon = Polygon([(p[1], p[0]) for p in h3.cell_to_boundary(cell)])
            if boundary_union.intersection(hexagon).area / hexagon.area > MIN_OVERLAP_FRAC:
                expected.add(cell)
    got = set(gdf["cell_id"])
    # H3 'overlap' xet canh o theo cung tron lon, ta xet duong thang lon/lat -> cho phep lech vai o
    # sat bien, nhung tap cua ta phai chua tap cua H3.
    assert expected <= got
    assert len(got - expected) <= 0.002 * len(got)


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


# --- Quy tac giao: o chi giao mot phan nho van duoc giu (chong tai phat quy tac tam o) ---
def _write_region(tmp_path, geom, crs, name="region.geojson"):
    path = tmp_path / name
    gpd.GeoDataFrame(geometry=[geom], crs=crs).to_crs(4326).to_file(path, driver="GeoJSON")
    return str(path)


def test_square_cell_with_1pct_overlap_is_kept(tmp_path):
    # O 1000 m [500000, 501000] x [1100000, 1101000]; ranh gioi = 2 o day du ben phai + dai 10 m (1%)
    # lan vao o ben trai -> o ben trai (tam nam NGOAI ranh gioi) van phai duoc giu, overlap_frac ~ 0,01.
    region = box(500990, 1100000, 503000, 1101000)
    gdf = generate_square_grid(_write_region(tmp_path, region, "EPSG:32648"), 1000)
    by_id = dict(zip(gdf["cell_id"], gdf["overlap_frac"]))
    assert by_id["sq_1000m_500000_1100000"] == pytest.approx(0.01, abs=0.002)
    assert by_id["sq_1000m_501000_1100000"] == pytest.approx(1.0, abs=1e-3)
    assert len(gdf) == 3  # o cham canh (tren/duoi) khong duoc tinh


@pytest.mark.parametrize("gen,arg", [(generate_h3_grid, 5), (generate_s2_grid, 9),
                                     (generate_square_grid, 17087), (generate_latlon_grid, 0.1552)])
def test_tiny_region_without_any_cell_centre_still_gets_cells(tmp_path, gen, arg):
    # Vung 300 x 300 m: nho hon nhieu so voi o -> khong chua tam o nao; quy tac tam o tra 0 o.
    region = box(560000, 1110000, 560300, 1110300)
    path = _write_region(tmp_path, region, "EPSG:32648")
    gdf = gen(path, arg)
    assert 1 <= len(gdf) <= 4
    assert (gdf["overlap_frac"] > 0).all() and (gdf["overlap_frac"] < 0.01).all()
    region_wgs = gpd.read_file(path).geometry.iloc[0]
    assert region_wgs.difference(gdf.geometry.union_all()).area / region_wgs.area < 1e-9

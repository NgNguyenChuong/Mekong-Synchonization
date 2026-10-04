"""Kiem thu ranh gioi v2 (noi 3 km phia bien, tru lang gieng) va ghi provenance.

Hinh hoc TONG HOP trong EPSG:32648, khong doc du lieu that.
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_boundary_v2 import build_v2, sample_on_grid  # noqa: E402
from preprocessing import file_sha256, write_provenance  # noqa: E402


def _wgs(geom):
    return gpd.GeoSeries([geom], crs="EPSG:32648").to_crs(4326).iloc[0]


def _m(geom):
    return gpd.GeoSeries([geom], crs=4326).to_crs("EPSG:32648").iloc[0]


@pytest.fixture(scope="module")
def setup():
    # v1 = o vuong 20 x 20 km; lang gieng = o vuong ke ben trai (chung canh x = 500000).
    v1_m = box(500000, 1100000, 520000, 1120000)
    nb_m = box(480000, 1100000, 500000, 1120000)
    v2_wgs, added_m = build_v2(_wgs(v1_m), nb_m, 3000.0)
    return v1_m, nb_m, _m(v2_wgs), added_m


def test_v1_is_never_reduced(setup):
    v1_m, _, v2_m, _ = setup
    assert v1_m.difference(v2_m).area < 1.0  # m2 (nhieu so hoc do chieu qua lai)


def test_added_part_does_not_enter_neighbor_or_its_buffer(setup):
    _, nb_m, v2_m, added_m = setup
    assert added_m.intersection(nb_m).area == 0
    assert added_m.intersection(nb_m.buffer(3000)).area == pytest.approx(0, abs=1e-6)
    # v2 khong lan sang lang gieng (ngoai phan v1 san co - o day v1 chung canh nen = 0).
    assert v2_m.intersection(nb_m).area < 1.0


def test_buffer_reaches_3km_away_from_neighbor(setup):
    v1_m, nb_m, v2_m, _ = setup
    # Phia bien (phai): them dung 3 km; phia lang gieng (trai): khong them.
    assert v2_m.bounds[2] == pytest.approx(523000, abs=1.0)
    assert v2_m.bounds[0] == pytest.approx(500000, abs=1.0)
    added_km2 = (v2_m.area - v1_m.area) / 1e6
    # Dai 3 km quanh 3 canh (tren, duoi, phai) + goc tron, tru phan trong buffer lang gieng.
    assert 150 < added_km2 < 200


def test_neighbor_overlapping_v1_does_not_cut_v1():
    # Lang gieng CHONG len v1 (nhu Campuchia geoBoundaries chong 182 km2 len v1 that): v1 van giu nguyen.
    v1_m = box(500000, 1100000, 520000, 1120000)
    nb_m = box(490000, 1100000, 505000, 1120000)
    v2_wgs, _ = build_v2(_wgs(v1_m), nb_m, 3000.0)
    assert v1_m.difference(_m(v2_wgs)).area < 1.0


def test_sample_on_grid_nearest_and_fill(tmp_path):
    # Raster 10 m 6 x 6, gia tri = chi so pixel; luoi dich 30 m lech 10 m -> tam o dich roi vao pixel biet truoc.
    path = tmp_path / "lc.tif"
    data = np.arange(36, dtype="uint8").reshape(6, 6)
    with rasterio.open(path, "w", driver="GTiff", width=6, height=6, count=1, dtype="uint8",
                       crs="EPSG:32648", transform=from_origin(1000, 2060, 10, 10)) as ds:
        ds.write(data, 1)
        ds.descriptions = ("Map",)
    out = sample_on_grid(str(path), "Map", from_origin(1010, 2050, 30, 30), (3, 3), fill=255)
    # Tam o dich (0,0) = (1025, 2035) -> pixel nguon hang 2, cot 2 = 14.
    assert out[0, 0] == data[2, 2]
    assert out[0, 1] == data[2, 5]
    assert out[1, 0] == data[5, 2]
    assert out[2, 2] == 255  # ngoai khung nguon


def test_write_provenance_records_boundary_hash(tmp_path):
    b = tmp_path / "b.geojson"
    gpd.GeoDataFrame(geometry=[box(105, 10, 105.1, 10.1)], crs=4326).to_file(b, driver="GeoJSON")
    out = tmp_path / "grid.geojson"
    out.write_text("{}")
    p = write_provenance(str(out), str(b), cells=3)
    info = json.loads(open(p, encoding="utf-8").read())
    assert p.endswith("grid.geojson.provenance.json")
    assert info["boundary_sha256"] == file_sha256(str(b)) and info["cells"] == 3

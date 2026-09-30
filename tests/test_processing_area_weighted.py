"""Kiem thu trich dac trung theo dien tich (tuan 2, viec 3) va chay main.py nhieu luoi (viec 1).

Du lieu la raster TONG HOP voi gia tri biet truoc, khong phai du lieu that.
"""
import os
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import processing  # noqa: E402

# Raster 4 x 2 pixel, moi pixel 0.1 do, goc tren trai (105.0, 10.2).
TRANSFORM = from_origin(105.0, 10.2, 0.1, 0.1)
BASE = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="float32")


def _write(path, bands, dtype="float32", nodata=None):
    bands = [np.asarray(b, dtype=dtype) for b in bands]
    with rasterio.open(path, "w", driver="GTiff", width=4, height=2, count=len(bands), dtype=dtype,
                       crs="EPSG:4326", transform=TRANSFORM, nodata=nodata) as ds:
        for i, b in enumerate(bands, start=1):
            ds.write(b, i)


def _cells(geoms, ids=None):
    ids = ids or [f"c{i}" for i in range(len(geoms))]
    return (ids, None, geoms)


def test_area_weighted_mean_uses_coverage(tmp_path):
    tif = tmp_path / "x.tif"
    _write(tif, [BASE])
    cells = _cells([
        box(105.0, 10.1, 105.1, 10.2),    # dung pixel gia tri 1
        box(105.05, 10.1, 105.15, 10.2),  # nua pixel 1 + nua pixel 2 -> 1.5
        box(105.02, 10.12, 105.04, 10.14),  # nho hon mot pixel -> gia tri pixel chua no
        box(106.0, 11.0, 106.1, 11.1),    # ngoai raster -> NaN
    ])
    out = processing.area_weighted_means(str(tif), cells)
    assert list(out.columns) == ["cell_id", "b1"]
    assert out["b1"].iloc[:3].tolist() == pytest.approx([1.0, 1.5, 1.0])
    assert np.isnan(out["b1"].iloc[3])


def test_nan_and_nodata_are_ignored_not_zero(tmp_path):
    tif = tmp_path / "x.tif"
    b = BASE.copy()
    b[0, 0] = -9999
    b[0, 1] = np.nan
    _write(tif, [b], nodata=-9999)
    cells = _cells([box(105.0, 10.1, 105.2, 10.2), box(105.0, 10.1, 105.3, 10.2)])
    out = processing.area_weighted_means(str(tif), cells)
    assert np.isnan(out["b1"].iloc[0])           # ca hai pixel deu thieu -> NaN, khong phai 0
    assert out["b1"].iloc[1] == pytest.approx(3.0)  # chi con pixel gia tri 3


def test_dynamic_monthly_file_maps_bands_to_days(tmp_path):
    folder = tmp_path / "daily_rain"
    folder.mkdir()
    _write(folder / "rain_2020_02.tif", [BASE, BASE * 10, BASE * 100])
    cells = _cells([box(105.05, 10.1, 105.15, 10.2)], ids=["a"])
    df = processing.extract_generic("rain", {"folder": "daily_rain", "col_name": "rain_mm"}, cells, str(tmp_path))
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].tolist() == ["2020-02-01", "2020-02-02", "2020-02-03"]
    assert df["rain_mm"].tolist() == pytest.approx([1.5, 15.0, 150.0])


def test_landcover_fractions_weighted_and_nan_without_data(tmp_path):
    folder = tmp_path / "landcover"
    folder.mkdir()
    lc = np.array([[10, 40, 40, 80], [10, 10, 40, 80]], dtype="uint8")
    _write(folder / "lc.tif", [lc], dtype="uint8", nodata=0)
    cells = _cells([box(105.05, 10.0, 105.15, 10.2), box(106.0, 11.0, 106.1, 11.1)], ids=["a", "outside"])
    spec = {"folder": "landcover", "file": "lc.tif", "col_name": "lc", "method": "all_classes",
            "class_names": {10: "Trees", 40: "Cropland", 80: "Water"}}
    df = processing.extract_static_generic("landcover", spec, cells, str(tmp_path)).set_index("cell_id")
    # O "a" phu nua cot 0 (10, 10) va nua cot 1 (40, 10): Trees 3/4, Cropland 1/4.
    assert df.loc["a", "lc_Trees"] == pytest.approx(0.75)
    assert df.loc["a", "lc_Cropland"] == pytest.approx(0.25)
    assert df.loc["a"].sum() == pytest.approx(1.0)
    assert df.loc["outside"].isna().all()


def test_min_distance_zero_for_water_cells(tmp_path):
    folder = tmp_path / "river"
    folder.mkdir()
    water = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype="float32")
    _write(folder / "w.tif", [water])
    geoms = [box(105.0 + 0.1 * i, 10.0, 105.1 + 0.1 * i, 10.2) for i in range(4)]
    spec = {"folder": "river", "file": "w.tif", "col_name": "river_proximity", "method": "min_distance"}
    df = processing.extract_static_generic("river", spec, _cells(geoms), str(tmp_path))
    d = df["river_proximity"].to_numpy()
    assert d[0] == 0.0
    assert np.all(np.diff(d) > 0)                  # xa dan theo huong dong
    assert d[1] == pytest.approx(10.9, abs=0.3)    # 0.1 do kinh o vi do 10 ~ 10.9 km


def test_periodic_file_per_date(tmp_path):
    folder = tmp_path / "ndvi"
    folder.mkdir()
    _write(folder / "NDVI_2020-03-15.tif", [BASE])
    _write(folder / "NDVI_2020-04-15.tif", [BASE * 2])
    spec = {"folder": "ndvi", "col_name": "ndvi", "file_pattern": "NDVI_{date}.tif", "date_pattern": "%Y-%m-%d"}
    cells = _cells([box(105.05, 10.1, 105.15, 10.2)], ids=["a"])
    df = processing.extract_periodic_generic("ndvi", spec, cells, str(tmp_path)).sort_values("date")
    assert df["date"].tolist() == ["2020-03-15", "2020-04-15"]
    assert df["ndvi"].tolist() == pytest.approx([1.5, 3.0])


def test_merge_skips_missing_file_without_reusing_previous(tmp_path, monkeypatch):
    monkeypatch.setattr(processing, "DATA_PROCESSED", str(tmp_path))
    pd.DataFrame({"cell_id": ["a", "b"], "dem": [1.0, 2.0]}).to_csv(tmp_path / "dem.csv", index=False)
    specs = {"dem": {"output_file": "dem.csv"}, "missing": {"output_file": "khong_co.csv"}}
    merged = processing._merge(specs, ["cell_id"], "OUT.csv", "TEST")
    assert list(merged.columns) == ["cell_id", "dem"]
    assert len(merged) == 2


def test_main_runs_on_given_grid_and_output_dir(tmp_path):
    raw = tmp_path / "raw"
    (raw / "daily_rain").mkdir(parents=True)
    _write(raw / "daily_rain" / "rain_2020_01.tif", [BASE, BASE * 2])
    grid = gpd.GeoDataFrame({"cell_id": ["sq_a", "sq_b"]},
                            geometry=[box(105.0, 10.1, 105.1, 10.2), box(105.05, 10.1, 105.15, 10.2)],
                            crs="EPSG:4326")
    grid_path = tmp_path / "grid.geojson"
    grid.to_file(grid_path, driver="GeoJSON")
    out = tmp_path / "out"
    out.mkdir()

    default_out = os.path.join(ROOT, "data", "processed", "DYNAMIC_MERGE.csv")
    before = os.path.getmtime(default_out) if os.path.exists(default_out) else None

    env = dict(os.environ, GRID_GEOJSON=str(grid_path), OUTPUT_DIR=str(out), RAW_DIR=str(raw),
               PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    df = pd.read_csv(out / "DYNAMIC_MERGE.csv").set_index(["cell_id", "date"])
    assert set(df.index.get_level_values("cell_id")) == {"sq_a", "sq_b"}
    assert df.loc[("sq_b", "2020-01-02"), "rain_mm"] == pytest.approx(3.0)
    # Thu muc ket qua mac dinh khong bi dong vao.
    after = os.path.getmtime(default_out) if os.path.exists(default_out) else None
    assert after == before

"""Script kiem cua harness (.claude/skills/*/scripts) - CHG-22 (3 trang thai): loi/thieu khong bao gio thanh "dat".

.claude/ nam ngoai git (gitignore) -> test tu bo qua neu ban sao khong co script. Du lieu gia nho (tmp_path).
"""
import os
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SK = os.path.join(ROOT, ".claude", "skills")
CHECK_SPLIT = os.path.join(SK, "method-review", "scripts", "check_split.py")
CHECK_FEATURES = os.path.join(SK, "method-review", "scripts", "check_features.py")
AUDIT_RASTER = os.path.join(SK, "data-audit", "scripts", "audit_raster.py")
AUDIT_GRID = os.path.join(SK, "data-audit", "scripts", "audit_grid_table.py")


def _run(script, *args):
    if not os.path.exists(script):
        pytest.skip(f"khong co {script} (thu muc .claude/ khong nam trong git)")
    r = subprocess.run([sys.executable, script, *map(str, args)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


# ---------------- check_split ----------------
@pytest.fixture()
def split_world(tmp_path):
    size = 50_000.0
    blocks = gpd.GeoDataFrame({"block_id": ["blk_10_22", "blk_11_22", "blk_12_22"], "is_holdout": [True, False, False]},
                              geometry=[box(i * size, 22 * size, (i + 1) * size, 23 * size) for i in (10, 11, 12)],
                              crs=32648).to_crs(4326)
    cells = {"c_h": (510_000, 1_110_000), "c1": (560_000, 1_110_000), "c2": (610_000, 1_110_000),
             "c3": (620_000, 1_120_000)}
    grid = gpd.GeoDataFrame({"cell_id": list(cells)}, geometry=[box(x, y, x + 2000, y + 2000) for x, y in cells.values()],
                            crs=32648).to_crs(4326)
    bp, gp = tmp_path / "blocks.geojson", tmp_path / "grid.geojson"
    blocks.to_file(bp, driver="GeoJSON")
    grid.to_file(gp, driver="GeoJSON")
    table = pd.DataFrame({"cell_id": ["c1", "c2", "c3", "c_h"], "season": [2015, 2015, 2016, 2015],
                          "split": ["train", "val", "train", "test"], "fold": [0, 1, 1, np.nan],
                          "block_id": ["blk_11_22", "blk_12_22", "blk_12_22", "blk_10_22"]})
    return tmp_path, gp, bp, table


def _split(tmp_path, gp, bp, table, *extra):
    p = tmp_path / "t.csv"
    table.to_csv(p, index=False)
    return _run(CHECK_SPLIT, p, "--grid", gp, "--blocks", bp, "--fold-col", "fold", "--block-col", "block_id", *extra)


def test_check_split_dung_thi_dat(split_world):
    code, out = _split(*split_world)
    assert code == 0 and "OK" in out, out


@pytest.mark.parametrize("hong, khop", [
    (lambda t: t.assign(split=["train", "Train", "train", "test"]), "khong thuoc"),   # gia tri split la
    (lambda t: t.assign(split=["train", None, "train", "test"]), "khong thuoc"),      # split NaN
    (lambda t: t.assign(fold=[0, 1, np.nan, np.nan]), "fold NaN"),                   # train fold NaN
    (lambda t: t.assign(block_id=["blk_11_22", None, "blk_12_22", "blk_10_22"]), "khong co khoi"),
])
def test_check_split_loi_khong_bi_bo_qua(split_world, hong, khop):
    tmp_path, gp, bp, table = split_world
    code, out = _split(tmp_path, gp, bp, hong(table))
    assert code == 1 and khop in out, out


# ---------------- check_features ----------------
def test_check_features_khong_cot_la_loi_va_doc_du_tieu_de(tmp_path):
    code, out = _run(CHECK_FEATURES)
    assert code == 1 and "0 cot" in out
    p = tmp_path / "f.csv"
    pd.DataFrame({"cell_id": ["a"] * 300, "dem_mean": [1.0] * 300, "ndvi_lop": ["cao"] * 300}).to_csv(p, index=False)
    code, out = _run(CHECK_FEATURES, "--from-csv", p, "--exclude", "cell_id")
    assert code == 1 and "ndvi_lop" in out, out  # cot chuoi truoc day bi bo khoi danh sach ten
    code, out = _run(CHECK_FEATURES, "--from-csv", p, "--exclude", "cell_id", "dem_mean", "ndvi_lop")
    assert code == 1 and "0 cot" in out
    code, out = _run(CHECK_FEATURES, "--from-csv", p, "--exclude", "cell_id", "ndvi_lop")
    assert code == 0 and "1 cot" in out


# ---------------- audit_raster ----------------
def _tif(path, arr, x0=500_000.0, y0=1_100_000.0):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs="EPSG:32648", transform=from_origin(x0, y0, 30, 30)) as w:
        w.write(arr, 1)
        w.set_band_description(1, "Salinity")


def test_audit_raster_inf_dem_rieng_va_vung_rong(tmp_path):
    a = np.ones((20, 20), "float32")
    a[0, 0] = np.nan
    ok = tmp_path / "ok.tif"
    _tif(ok, a)
    code, out = _run(AUDIT_RASTER, ok)
    assert code == 0, out
    a[1, 1], a[2, 2] = np.inf, -np.inf
    bad = tmp_path / "inf.tif"
    _tif(bad, a)
    code, out = _run(AUDIT_RASTER, bad, "--range", 0, 30)
    assert code == 1 and "2 pixel +-inf" in out, out
    # vung khong giao raster -> loi "band rong", khong crash, khong thoat 0
    far = gpd.GeoDataFrame(geometry=[box(900_000, 1_500_000, 910_000, 1_510_000)], crs=32648)
    rp = tmp_path / "far.geojson"
    far.to_file(rp, driver="GeoJSON")
    code, out = _run(AUDIT_RASTER, ok, "--region", rp)
    assert code == 1 and "band rong trong vung" in out and "Traceback" not in out, out
    code, out = _run(AUDIT_RASTER, ok, "--region", rp, "--categorical")
    assert code == 1 and "band rong trong vung" in out and "Traceback" not in out, out


# ---------------- audit_grid_table ----------------
def test_audit_grid_table_cot_range_vang_hoac_chuoi_va_inf(tmp_path):
    gd = tmp_path / "grids"
    gd.mkdir()
    gpd.GeoDataFrame({"cell_id": ["a", "b"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
                     crs=4326).to_file(gd / "gx.geojson", driver="GeoJSON")
    p = tmp_path / "gx_feat.csv"
    pd.DataFrame({"cell_id": ["a", "b"], "rain_mm": [10.0, 20.0], "lop": ["x", "y"]}).to_csv(p, index=False)
    code, out = _run(AUDIT_GRID, p, "--grids-dir", gd, "--range", "rain_mm", 0, 100)
    assert code == 0, out
    code, out = _run(AUDIT_GRID, p, "--grids-dir", gd, "--range", "rain_mmm", 0, 100)
    assert code == 1 and "khong co trong bang" in out, out
    code, out = _run(AUDIT_GRID, p, "--grids-dir", gd, "--range", "lop", 0, 100)
    assert code == 1 and "khong phai cot so" in out, out
    pd.DataFrame({"cell_id": ["a", "b"], "rain_mm": [10.0, np.inf]}).to_csv(p, index=False)
    code, out = _run(AUDIT_GRID, p, "--grids-dir", gd)
    assert code == 1 and "+-inf" in out, out

"""Kiem thu sua georeference ERA5 lech nua pixel (known-pitfalls 1a).

Raster TONG HOP mo phong dung loi GEE `scale=11132`: luoi goc 0,1 do co tam pixel o boi so 0,1 do;
file xuat co goc (103.700474, 11.200051), phan giai 0.10000046 -> tam pixel xuat roi sat mep goc,
gia tri la cua pixel goc Dong-Bac. Transform dung = (0.1, 103.75, 11.25).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import era5_georef as eg  # noqa: E402
import fix_era5_transform as fx  # noqa: E402

RES_GEE = 0.10000045742818513
T_OLD = Affine(RES_GEE, 0.0, 103.70047435302797, 0.0, -RES_GEE, 11.200051231956735)
T_NEW = Affine(0.1, 0.0, 103.75, 0.0, -0.1, 11.25)


def _native_value(lon_c, lat_c):
    """Gia tri biet truoc cua pixel goc co tam (lon_c, lat_c): ma hoa vi tri de nhan ra pixel bi chon."""
    return round(lon_c * 10) * 1000 + round(lat_c * 10)


def _simulate_gee_export(width=33, height=29):
    """Mo phong GEE: moi pixel xuat lay gia tri pixel goc chua TAM pixel xuat (nearest)."""
    arr = np.zeros((height, width), "float32")
    for r in range(height):
        for c in range(width):
            x, y = T_OLD @ (c + 0.5, r + 0.5)
            k = np.floor((x - eg.NATIVE_X0) / 0.1)
            j = np.floor((eg.NATIVE_Y0 - y) / 0.1)
            arr[r, c] = _native_value(eg.NATIVE_X0 + (k + 0.5) * 0.1, eg.NATIVE_Y0 - (j + 0.5) * 0.1)
    return arr


def _write(path, arr, transform, count=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=count,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=-np.inf,
                       compress="deflate") as ds:
        for b in range(count):
            ds.write(arr + b, b + 1)


def test_cong_thuc_transform_dung_va_pixel_goc_khop():
    t = eg.native_transform(T_OLD, 33, 29)
    assert t.almost_equals(T_NEW, precision=1e-9)
    # sau sua: tam pixel (r, c) theo transform moi = tam pixel goc da cap gia tri cho (r, c)
    arr = _simulate_gee_export()
    for r, c in [(0, 0), (0, 32), (28, 0), (28, 32), (14, 17)]:
        x, y = t @ (c + 0.5, r + 0.5)
        assert arr[r, c] == _native_value(x, y)
    # transform cu: tam pixel lech 0,05 do Tay va Nam so voi pixel goc giu gia tri
    x, y = T_OLD @ (0.5, 0.5)
    assert arr[0, 0] == _native_value(103.8, 11.2)
    assert abs((x - 103.8) + 0.05) < 1e-3 and abs((y - 11.2) + 0.05) < 1e-3


def test_native_transform_idempotent_va_tu_choi_mo_ho():
    assert eg.is_native_aligned(T_NEW) and not eg.is_native_aligned(T_OLD)
    assert eg.native_transform(T_NEW, 33, 29) == T_NEW
    # tam pixel roi dung mep luoi goc -> khong biet GEE chon pixel nao -> loi
    with pytest.raises(ValueError, match="mo ho"):
        eg.native_transform(Affine(0.1, 0, 103.70, 0, -0.1, 11.20), 33, 29)
    # phan giai lech nhieu -> pixel cuoi nhay sang pixel goc khong lien tiep -> loi
    with pytest.raises(ValueError, match="khong lien tiep"):
        eg.native_transform(Affine(0.105, 0, 103.7005, 0, -0.1, 11.2005), 33, 29)


def test_sua_header_khong_doi_du_lieu_idempotent_va_dao_nguoc(tmp_path):
    raw = tmp_path / "raw"
    arr = _simulate_gee_export()
    files = [raw / "daily_rain" / "rain_2016_03.tif", raw / "daily_temp_max" / "temp_max_2016_02.tif"]
    for f in files:
        _write(str(f), arr, T_OLD)
    log = str(tmp_path / "fix.json")
    with rasterio.open(files[0]) as ds:
        digest0 = eg.data_digest(ds)

    fx.fix_files(str(raw), log)
    rec = json.load(open(log, encoding="utf-8"))
    assert sorted(rec["files"]) == ["daily_rain/rain_2016_03.tif", "daily_temp_max/temp_max_2016_02.tif"]
    for key, r in rec["files"].items():
        assert r["status"] == "fixed"
        assert np.allclose(r["transform_old"], list(T_OLD)[:6]) and np.allclose(r["transform_new"], list(T_NEW)[:6])
        assert r["sha256_before"] != r["sha256_after"]
    for f in files:
        with rasterio.open(f) as ds:
            assert ds.transform.almost_equals(T_NEW, precision=1e-12)
            assert eg.data_digest(ds) == digest0          # du lieu khong doi
            assert ds.nodata == -np.inf and ds.count == 3

    # lan 2: khong dich them, khong ghi lai file, JSON giu nguyen
    sha_after = {k: r["sha256_after"] for k, r in rec["files"].items()}
    fx.fix_files(str(raw), log)
    for f in files:
        with rasterio.open(f) as ds:
            assert ds.transform.almost_equals(T_NEW, precision=1e-12)
    rec2 = json.load(open(log, encoding="utf-8"))
    assert {k: fx.file_sha256(str(raw / k)) for k in rec2["files"]} == sha_after

    # dao nguoc
    fx.revert_files(str(raw), log)
    for f in files:
        with rasterio.open(f) as ds:
            assert ds.transform.almost_equals(T_OLD, precision=1e-12)
            assert eg.data_digest(ds) == digest0


def test_dry_run_khong_ghi(tmp_path):
    raw = tmp_path / "raw"
    f = raw / "daily_rain" / "rain_2016_03.tif"
    _write(str(f), _simulate_gee_export(), T_OLD, count=1)
    sha = fx.file_sha256(str(f))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fix_era5_transform.py"), "--raw-dir", str(raw),
                        "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "can sua: 1" in r.stdout
    assert fx.file_sha256(str(f)) == sha and not (raw / "era5_transform_fix.json").exists()


def test_compare_on_grid_phan_biet_bang_dien_tich():
    """So theo tam pixel khop ca 2 transform; dien tich trung chi 100% voi transform moi."""
    arr = _simulate_gee_export()
    # anh "goc" GEE: luoi native phu rong hon
    t_ref = Affine(0.1, 0, 103.45, 0, -0.1, 11.45)
    ref = np.zeros((35, 40))
    for j in range(35):
        for k in range(40):
            x, y = t_ref @ (k + 0.5, j + 0.5)
            ref[j, k] = _native_value(x, y)
    old = fx.compare_on_grid(arr.astype(float), T_OLD, ref, t_ref)
    new = fx.compare_on_grid(arr.astype(float), T_NEW, ref, t_ref)
    assert old["match_frac_local_land"] == 1.0 and new["match_frac_local_land"] == 1.0
    assert new["footprint_overlap"] == pytest.approx(1.0)
    assert old["footprint_overlap"] == pytest.approx(0.25, abs=0.01)
    assert fx.compare_on_grid(arr.astype(float), T_NEW, ref, t_ref, shift=(0, 1))["match"] == 0

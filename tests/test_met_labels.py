"""Kiem thu ham thuan cho nhan khi tuong Dot 7 (CHIRPS v3 mua, MCD18A1 buc xa) - KHONG goi GEE.

Chong tai phat: cua so = seasons.py (01/11..29/04), nam nhuan, chon pentad/thang, ten file,
TB co/khong thang 12, NaN/-9999 khong bi dien 0, doc band theo ten.
"""
import datetime as dt
import importlib
import os
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import met_labels as ml  # noqa: E402

D = dt.date


# ---------------------------------------------------------------- cua so mua
def test_cua_so_mua_het_29_04_va_nam_nhuan():
    assert ml.met_season_window(2020) == (D(2019, 11, 1), D(2020, 4, 29))
    assert ml.met_season_days(2020) == 181          # 2020 nhuan (29/02)
    assert ml.met_season_days(2021) == 180
    assert ml.met_season_days(2024) == 181
    days = ml.met_season_dates(2020)
    assert days[0] == D(2019, 11, 1) and days[-1] == D(2020, 4, 29)
    assert D(2019, 10, 31) not in days and D(2020, 4, 30) not in days
    assert D(2020, 2, 29) in days and D(2020, 4, 29) in days
    assert len(set(days)) == len(days)


def test_cua_so_trung_seasons_py():
    # Nhan khi tuong dung CHUNG cua so voi do man/dac trung (An chot 2026-10-06), khong tu dinh nghia lai.
    from seasons import season_days, season_window
    for y in range(2014, 2027):
        s, e = season_window(y)
        assert ml.met_season_window(y) == (s.date(), e.date())
        assert ml.met_season_days(y) == season_days(y)
    assert not hasattr(ml, "MET_SEASON_END")


def test_filter_date_gee_loai_ngay_cuoi():
    # giong filterDate cua tac gia nhan Zenodo ('Y-1-11-01', 'Y-04-30')
    assert ml.met_season_filter_dates(2014) == ("2013-11-01", "2014-04-30")
    assert ml.met_season_filter_dates(2026) == ("2025-11-01", "2026-04-30")


def test_bo_thang_12():
    nd = ml.met_season_dates(2020, exclude_months=(12,))
    assert len(nd) == 181 - 31
    assert all(d.month != 12 for d in nd)
    assert D(2019, 11, 30) in nd and D(2020, 1, 1) in nd
    assert len(ml.mcd18_indices(2021, exclude_months=(12,))) == 149


# ---------------------------------------------------------------- chon thang / pentad / chi so anh
def test_thang_cua_mua():
    assert ml.season_year_months(2014) == [(2013, 11), (2013, 12), (2014, 1), (2014, 2), (2014, 3), (2014, 4)]


def test_pentad_chirps():
    idx = ml.chirps_pentad_indices(2020)
    assert len(idx) == 36 and len(set(idx)) == 36
    assert idx[0] == "20191101" and idx[-1] == "20200426"
    assert "20200226" in idx            # pentad 6 thang 2 (gom 29/02)
    assert "20191026" not in idx and "20200501" not in idx
    assert all(i[6:] in {"01", "06", "11", "16", "21", "26"} for i in idx)


def test_chi_so_mcd18_va_ngay_thieu():
    idx = ml.mcd18_indices(2020)
    assert idx[0] == "2019_11_01" and idx[-1] == "2020_04_29" and len(idx) == 181
    avail = [i for i in idx if not ("2019_12_07" <= i <= "2019_12_31")]
    miss = ml.missing_indices(idx, avail)
    assert len(miss) == 25
    runs = ml.date_runs([ml.parse_mcd18_index(i) for i in miss])
    assert runs == ["2019-12-07..2019-12-31"]
    assert ml.date_runs([D(2025, 2, 20), D(2025, 2, 21), D(2025, 3, 1)]) == ["2025-02-20..2025-02-21", "2025-03-01"]


def test_ten_file():
    assert ml.output_name("chirps3", 2014) == "chirps3_rain_2014.tif"
    assert ml.output_name("mcd18a1", 2026) == "mcd18a1_dsr_2026.tif"
    with pytest.raises(KeyError):
        ml.output_name("era5", 2014)


def _us(y, m, d):
    return int((dt.datetime(y, m, d) - dt.datetime(1970, 1, 1)).total_seconds() * 1e6)


def test_final_prelim_theo_thoi_diem_nap():
    # pentad 01/11/2025 nap 12/12/2025 (sau khi CHC phat hanh final 11/12) -> final
    assert ml.chirps_status_from_version("20251101", _us(2025, 12, 12)) == "final"
    # nap 05/11/2025 (chi co the la prelim)
    assert ml.chirps_status_from_version("20251126", _us(2025, 12, 3)) == "prelim"
    # thang 12 -> moc la thang 1 nam sau
    assert ml.chirps_status_from_version("20251226", _us(2026, 1, 5)) == "prelim"
    assert ml.chirps_status_from_version("20251226", _us(2026, 1, 15)) == "final"


# ---------------------------------------------------------------- cong thuc
def test_dsr_ngay_thieu_moc_gio_la_nan():
    v = np.array([[300.0, 300.0], [800, 800], [600, 600], [80, np.nan], [0, 0], [0, 0], [0, 0], [20, 20]])
    out = ml.daily_dsr(v)
    assert out[0] == pytest.approx(1800 / 8)
    assert np.isnan(out[1])                    # thieu 1 moc gio -> ngay khong hop le (khong dien 0)
    with pytest.raises(ValueError):
        ml.daily_dsr(v[:7])


def test_tb_mua_co_va_khong_thang_12():
    dates = [D(2019, 11, 30), D(2019, 12, 1), D(2019, 12, 2), D(2020, 1, 1)]
    # 2 pixel: pixel 0 du ngay; pixel 1 thieu ngay 01/01 (NaN) va ngay 30/11
    daily = np.array([[100.0, np.nan], [200, 200], [300, 400], [400, np.nan]])
    qual = np.array([[1, 1], [2, 6], [4, 2], [2, 0]])     # 6 = 0b110 -> bit 0-1 = 2; 4 -> bit 0-1 = 0
    s = ml.season_dsr_stats(daily, qual, dates)
    assert s["dsr_mean"][0] == pytest.approx(250.0)
    assert s["dsr_mean"][1] == pytest.approx(300.0)       # NaN khong tinh nhu 0 (khong phai 150)
    assert list(s["n_days_valid"]) == [4, 2]
    assert s["frac_quality2"][0] == pytest.approx(2 / 4)  # ngay 2 (q=2) va ngay 4 (q=2)
    assert s["frac_quality2"][1] == pytest.approx(2 / 2)  # q=6 va q=2 tren 2 ngay hop le
    assert s["dsr_mean_no_dec"][0] == pytest.approx(250.0)   # (100 + 400) / 2
    assert np.isnan(s["dsr_mean_no_dec"][1])               # pixel 1 chi co ngay thang 12


def test_tb_mua_pixel_khong_ngay_hop_le():
    s = ml.season_dsr_stats(np.full((3, 1), np.nan), np.zeros((3, 1)), [D(2020, 1, k) for k in (1, 2, 3)])
    assert s["n_days_valid"][0] == 0 and np.isnan(s["dsr_mean"][0]) and np.isnan(s["frac_quality2"][0])


def test_tong_mua_pentad_nodata_khong_thanh_0():
    p = np.zeros((36, 3))
    p[:, 0] = 2.0                 # 72 mm
    p[5, 1] = -9999.0             # NoData cua CHIRPS
    p[7, 2] = np.nan
    out = ml.season_rain_sum(p)
    assert out[0] == pytest.approx(72.0)
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert ml.season_rain_sum(np.zeros((36, 1)))[0] == 0.0     # 0 that van la 0
    with pytest.raises(ValueError):
        ml.season_rain_sum(np.zeros((35, 1)))


# ---------------------------------------------------------------- doc file / tom tat
def _write(path, bands: dict, transform=from_origin(105.0, 10.2, 0.05, 0.05), crs="EPSG:4326"):
    names = list(bands)
    a0 = next(iter(bands.values()))
    with rasterio.open(path, "w", driver="GTiff", width=a0.shape[1], height=a0.shape[0], count=len(names),
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan) as ds:
        for i, n in enumerate(names):
            ds.write(bands[n].astype("float32"), i + 1)
        ds.descriptions = tuple(names)


def _boundary(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    p = tmp_path / "b.geojson"
    # phu 2 x 2 pixel dau (105.0-105.1, 10.1-10.2)
    gpd.GeoDataFrame(geometry=[box(105.0, 10.1, 105.1, 10.2)], crs=4326).to_file(p, driver="GeoJSON")
    return str(p)


def test_tom_tat_chirps_nan_va_mat_na_kho(tmp_path):
    sh = (2, 3)
    rs = np.array([[100.0, np.nan, 5.0], [200.0, 300.0, 7.0]])
    m = {f"rain_m{k}": np.full(sh, 10.0) for k in ("11", "12", "01", "02", "03", "04")}
    m["rain_m01"] = np.array([[0.0, np.nan, 0], [3.0, 0.0, 0]])
    bands = {"rain_sum": rs, **m, "rain_jfm": np.array([[0.0, np.nan, 0], [0.5, 4.0, 0]]),
             "n_pentad_valid": np.array([[36.0, 35, 36], [36, 36, 36]]), "n_pentad_neg": np.array([[0.0, 1, 0], [0, 0, 0]])}
    f = tmp_path / "chirps3_rain_2020.tif"
    _write(f, bands)
    s = ml.chirps_season_summary(str(f), _boundary(tmp_path))
    assert s["n_px_center"] == 4
    assert s["n_nodata_center"] == 1
    assert s["n_px_pentad_neg_touched"] == 1 and s["n_px_pentad_missing_touched"] == 1
    assert s["n_zero_m01"] == 2          # NaN khong dem la 0; cot 3 ngoai vung
    assert s["n_jfm_zero"] == 1 and s["n_jfm_lt1mm"] == 2
    assert s["rain_sum_mean"] == pytest.approx(200.0)   # TB(100, 200, 300), NaN bo qua
    assert s["rain_sum_min"] == 100.0 and s["rain_sum_max"] == 300.0


def test_tom_tat_mcd18_va_doc_band_theo_ten(tmp_path):
    sh = (2, 3)
    # thu tu band dao nguoc so voi MCD18_BANDS -> phai doc theo ten
    bands = {"dsr_mean_no_dec": np.array([[190.0, 210, 0], [200, np.nan, 0]]),
             "frac_quality2": np.array([[0.5, 1.0, 0], [0.0, np.nan, 0]]),
             "n_days_valid": np.array([[180.0, 150, 0], [182, 0, 0]]),
             "dsr_mean": np.array([[200.0, 200, 0], [220, np.nan, 0]])}
    f = tmp_path / "mcd18a1_dsr_2020.tif"
    _write(f, bands)
    s = ml.mcd18_season_summary(str(f), _boundary(tmp_path))
    assert s["n_px_center"] == 4 and s["n_px_no_valid_day_center"] == 1
    assert s["n_days_valid_min"] == 0 and s["n_days_valid_mean"] == pytest.approx((180 + 150 + 182 + 0) / 4)
    assert s["n_days_valid_min_pos"] == 150 and s["n_days_valid_mean_pos"] == pytest.approx((180 + 150 + 182) / 3)
    assert s["dsr_mean_mean"] == pytest.approx(620 / 3)
    assert s["diff_dec_mean"] == pytest.approx((10 - 10 + 20) / 3)
    assert s["frac_quality2_mean"] == pytest.approx(0.5)


def test_doc_band_thieu_ten_bao_loi(tmp_path):
    f = tmp_path / "x.tif"
    with rasterio.open(f, "w", driver="GTiff", width=1, height=1, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(105, 10, 0.05, 0.05)) as ds:
        ds.write(np.ones((1, 1, 1), dtype="float32"))
    with pytest.raises(ValueError):
        ml.read_bands(str(f))


# ---------------------------------------------------------------- hau xu ly file tai tu GEE
def test_hau_xu_ly_file_gee_inf_thanh_nan(tmp_path):
    gf = importlib.import_module("gee_fetch")
    tmp = tmp_path / "a.tif.dl.tif"
    tr = from_origin(105.0, 10.2, 0.05, 0.05)
    with rasterio.open(tmp, "w", driver="GTiff", width=2, height=1, count=2, dtype="float32", crs="EPSG:4326",
                       transform=tr, nodata=-np.inf) as ds:
        ds.write(np.array([[[1.0, -np.inf]], [[0.0, 36.0]]], dtype="float32"))
    out = tmp_path / "a.tif"
    gf._finish_download(str(tmp), str(out), ("rain_sum", "n_pentad_valid"))
    assert not tmp.exists()
    with rasterio.open(out) as ds:
        assert ds.descriptions == ("rain_sum", "n_pentad_valid")
        assert np.isnan(ds.nodata) and ds.transform == tr
        a = ds.read()
    assert a[0, 0, 0] == 1.0 and np.isnan(a[0, 0, 1])
    assert a[1, 0, 0] == 0.0                      # 0 that giu nguyen
    with pytest.raises(RuntimeError):
        gf._finish_download(str(out), str(tmp_path / "b.tif"), ("chi_mot_band",))


def test_lenh_con_co_trong_cli():
    for cmd in ("chirps3", "mcd18"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gee_fetch.py"), cmd, "--help"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        assert "--years" in r.stdout and "--buffer-km" in r.stdout


# ---------------------------------------------------------------- file cu khac cua so -> LOI
def test_file_cu_khac_cua_so_la_loi(tmp_path):
    import json
    gf = importlib.import_module("gee_fetch")
    path = str(tmp_path / "mcd18a1_dsr_2021.tif")
    with open(path + ".provenance.json", "w", encoding="utf-8") as f:
        json.dump({"window": ["2020-11-01", "2021-04-30"]}, f)       # ban cu gom 30/04
    with pytest.raises(SystemExit, match="LOI"):
        gf._check_window(path, 2021)
    with open(path + ".provenance.json", "w", encoding="utf-8") as f:
        json.dump({"window": ["2020-11-01", "2021-04-29"]}, f)
    assert gf._check_window(path, 2021)["window"][1] == "2021-04-29"


# ---------------------------------------------------------------- CRS sinusoidal MODIS (hinh cau, khong ellipsoid)
def _tif(path, crs):
    with rasterio.open(path, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32", crs=crs,
                       transform=from_origin(11_000_000, 1_100_000, 926.6, 926.6)) as ds:
        ds.write(np.ones((1, 2, 2), "float32"))


def test_crs_modis_hinh_cau(tmp_path):
    ellip = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"   # nhu WKT GEE ghi
    assert not ml.is_modis_sphere(ellip)
    assert ml.is_modis_sphere(ml.MODIS_SINU_PROJ4)
    p = str(tmp_path / "m.tif")
    _tif(p, ellip)
    assert ml.set_modis_sphere_crs(p) is True
    with rasterio.open(p) as ds:
        assert ml.is_modis_sphere(ds.crs)
        assert ds.transform == from_origin(11_000_000, 1_100_000, 926.6, 926.6)   # khong doi pixel
        assert ds.read(1).sum() == 4
    assert ml.set_modis_sphere_crs(p) is False                                     # lan 2: khong sua nua
    q = str(tmp_path / "w.tif")
    _tif(q, "EPSG:4326")
    with pytest.raises(ValueError):
        ml.set_modis_sphere_crs(q)


def test_crs_hinh_cau_doi_vi_tri_dung_bac():
    # cung toa do lon/lat, ellipsoid vs hinh cau lech ~14 km x, ~6 km y o 105,5E 10N (bac do lech da do 2026-10-06)
    from pyproj import Transformer
    e = Transformer.from_crs("EPSG:4326", "+proj=sinu +datum=WGS84", always_xy=True).transform(105.5, 10.0)
    s = Transformer.from_crs("EPSG:4326", ml.MODIS_SINU_PROJ4, always_xy=True).transform(105.5, 10.0)
    assert 12_000 < e[0] - s[0] < 16_000 and 4_000 < abs(e[1] - s[1]) < 8_000

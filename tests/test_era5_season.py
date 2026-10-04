"""Kiem thu gop ERA5 theo mua kho va trich theo o (P3).

Raster TONG HOP gia tri biet truoc: 1 hang x 3 pixel 0,1 do, goc (105.0, 10.1):
pixel A (105.0-105.1), pixel B (105.1-105.2), pixel C (105.2-105.3) = bien (-inf, nhu file ERA5 tai ve).
"""
import json
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

import era5_season as es  # noqa: E402

TRANSFORM = from_origin(105.0, 10.1, 0.1, 0.1)
SEA = -np.inf
PREFIX = {"rain": "rain", "solar": "solar", "temp_avg": "temp_avg", "temp_max": "temp_max",
          "temp_min": "temp_min", "humid": "humidity"}   # file do am that ten humidity_YYYY_MM


def _write_month(folder, prefix, year, month, day_values, n_bands=None):
    """day_values(date) -> [A, B, C]. n_bands < so ngay thang -> file thieu ngay cuoi."""
    p = pd.Period(f"{year}-{month:02d}", freq="M")
    n = n_bands or p.days_in_month
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{prefix}_{year}_{month:02d}.tif")
    with rasterio.open(path, "w", driver="GTiff", width=3, height=1, count=n, dtype="float32",
                       crs="EPSG:4326", transform=TRANSFORM, nodata=SEA) as ds:
        for k in range(n):
            d = p.start_time + pd.Timedelta(days=k)
            ds.write(np.asarray([day_values(d)], dtype="float32"), k + 1)
    return path


def _land(a, b):
    return lambda d: [a, b, SEA]


def _make_raw(root, year, values=None, months=None, n_bands=None):
    """Ghi file thang cho ca 6 bien tu 10/(year-1) den 05/year (gom thang ngoai mua de kiem loai tru)."""
    values = values or {}
    months = months or [(year - 1, 10), (year - 1, 11), (year - 1, 12)] + [(year, m) for m in range(1, 6)]
    for var, (folder, _, _) in es.ERA5_SEASON_SPECS.items():
        fn = values.get(var, _land(1.0, 2.0))
        for (y, m) in months:
            nb = (n_bands or {}).get((var, y, m))
            _write_month(os.path.join(root, folder), PREFIX[var], y, m, fn, nb)
    return str(root)


def test_ranh_gioi_mua_va_nam_nhuan(tmp_path):
    # mua 2016 = 01/11/2015 .. 29/04/2016, co 29/02 -> 181 ngay
    marks = {(10, 31): 1e4, (11, 1): 10.0, (4, 29): 100.0, (4, 30): 1e5, (5, 1): 1e6}

    def rain(d):
        v = marks.get((d.month, d.day), 1e6 if d.month in (5, 10) else 1.0)
        return [v, v, SEA]

    def temp(d):
        v = 1000.0 if (d.month in (5, 10) or (d.month, d.day) == (4, 30)) else 1.0
        return [v, 3.0 if v == 1.0 else 1000.0, SEA]

    raw = _make_raw(tmp_path, 2016, {"rain": rain, "temp_avg": temp})
    layers, counts, valid, _, diag = es.season_layers(raw, 2016)
    assert counts["rain"][0, 0] == 181 and counts["temp_avg"][0, 0] == 181
    # 179 ngay x 1 + 01/11 (10) + 29/04 (100); 31/10, 30/04, thang 5 khong tinh
    assert layers["rain"][0, 0] == pytest.approx(179 + 10 + 100)
    assert layers["temp_avg"][0].tolist()[:2] == pytest.approx([1.0, 3.0])
    assert np.isnan(layers["temp_avg"][0, 2])
    assert valid[0].tolist() == [True, True, False]
    assert diag["rain"]["px_full"] == 2 and diag["rain"]["px_partial"] == 0


def test_mua_khong_nhuan_180_ngay(tmp_path):
    raw = _make_raw(tmp_path, 2017)
    layers, counts, _, _, _ = es.season_layers(raw, 2017)
    assert counts["rain"][0, 0] == 180
    assert layers["rain"][0, :2].tolist() == pytest.approx([180.0, 360.0])   # mua = TONG
    assert layers["humid"][0, :2].tolist() == pytest.approx([1.0, 2.0])      # bien khac = TRUNG BINH


def test_mua_am_cat_ve_0_nhiet_am_khong_cat(tmp_path):
    def rain(d):
        v = 5.0 if (d.month, d.day) == (1, 15) else -4e-5
        return [v, -1.0, SEA]

    raw = _make_raw(tmp_path, 2017, {"rain": rain, "temp_min": _land(-2.0, -3.0)})
    layers, _, _, _, _ = es.season_layers(raw, 2017)
    assert layers["rain"][0, 0] == pytest.approx(5.0)
    assert layers["rain"][0, 1] == 0.0
    assert layers["temp_min"][0, :2].tolist() == pytest.approx([-2.0, -3.0])


def test_thieu_mot_ngay_thanh_nan_khong_lap(tmp_path):
    def rain(d):
        return [np.nan if (d.month, d.day) == (2, 10) else 1.0, 1.0, SEA]   # pixel A thieu 1 ngay (NaN)

    raw = _make_raw(tmp_path, 2017, {"rain": rain}, n_bands={("solar", 2017, 3): 30})  # file thang 3 thieu 31/03
    layers, counts, valid, _, diag = es.season_layers(raw, 2017)
    assert counts["rain"][0, 0] == 179 and np.isnan(layers["rain"][0, 0])
    assert layers["rain"][0, 1] == pytest.approx(180.0)
    assert layers["temp_avg"][0, 0] == pytest.approx(1.0)        # bien khac cua pixel A khong bi anh huong
    assert np.isnan(layers["solar"][0, :2]).all()                 # thieu 31/03 o moi pixel
    assert diag["rain"]["px_partial"] == 1
    assert not valid.any()                                        # solar thieu -> khong pixel nao du moi bien


def test_thieu_file_thang_thanh_nan(tmp_path):
    months = [(2016, 11), (2016, 12), (2017, 1), (2017, 3), (2017, 4)]   # thieu 02/2017
    raw = _make_raw(tmp_path, 2017, months=months)
    layers, _, _, _, diag = es.season_layers(raw, 2017)
    assert np.isnan(layers["rain"]).all()
    assert diag["rain"]["missing_months"] == ["2017-02"]


def test_mat_na_dat_temp_ap_cho_solar_bang_0_o_bien(tmp_path):
    raw = _make_raw(tmp_path, 2017, {"solar": lambda d: [15.0, 17.0, 0.0]})   # loi ERA5: solar = 0 o bien
    layers, _, _, _, diag = es.season_layers(raw, 2017)
    assert np.isnan(layers["solar"][0, 2])
    assert diag["solar"]["px_outside_land"] == 1


def test_file_thang_trung_bao_loi(tmp_path):
    d = tmp_path / "daily_rain"
    _write_month(str(d), "rain", 2017, 1, _land(1.0, 1.0))
    _write_month(str(d), "rain_v2", 2017, 1, _land(1.0, 1.0))
    with pytest.raises(ValueError, match="cung thang"):
        es.index_monthly_files(str(d))


def test_file_thua_band_bao_loi(tmp_path):
    raw = _make_raw(tmp_path, 2017, n_bands={("rain", 2017, 2): 29})   # thang 2/2017 chi co 28 ngay
    with pytest.raises(ValueError, match="band"):
        es.season_layers(raw, 2017)


def _cells():
    geoms = [
        box(105.05, 10.0, 105.15, 10.1),   # nua A nua B -> trung binh
        box(105.15, 10.0, 105.25, 10.1),   # nua B nua bien -> gia tri B, phu 0,5
        box(105.2, 10.0, 105.3, 10.1),     # toan bien -> NaN, phu 0
        box(104.95, 10.0, 105.05, 10.1),   # nua ngoai vung tai ve (o ven), nua A -> gia tri A, phu 0,5
    ]
    return (["ab", "bs", "sea", "edge"], None, geoms)


def test_trich_theo_dien_tich_nua_A_nua_B(tmp_path):
    raw = _make_raw(tmp_path, 2017)
    _make_raw(tmp_path, 2018)
    by_season = {}
    for y in (2017, 2018):
        layers, _, valid, profile, _ = es.season_layers(raw, y)
        by_season[y] = {es.ERA5_SEASON_SPECS[v][1]: layers[v] for v in es.ERA5_SEASON_SPECS}
        by_season[y][es.COVER_COL] = valid.astype("float32")
    out = es.season_table(by_season, profile, _cells()).set_index(["cell_id", "season"])
    assert len(out) == 4 * 2
    r = out.loc[("ab", 2017)]
    assert r["temp_c"] == pytest.approx(1.5)
    assert r["rain_mm"] == pytest.approx((180 + 360) / 2)   # trung binh dien tich cua TONG mua tung pixel
    assert r[es.COVER_COL] == pytest.approx(1.0)
    assert out.loc[("bs", 2017), "temp_c"] == pytest.approx(2.0)
    assert out.loc[("bs", 2017), es.COVER_COL] == pytest.approx(0.5)
    assert np.isnan(out.loc[("sea", 2018), "rh_percent"])
    assert out.loc[("sea", 2018), es.COVER_COL] == 0.0
    assert out.loc[("edge", 2017), "temp_c"] == pytest.approx(1.0)
    assert out.loc[("edge", 2017), es.COVER_COL] == pytest.approx(0.5)


def test_o_vuot_vung_dem_bao_loi(tmp_path):
    raw = _make_raw(tmp_path, 2017)
    layers, _, valid, profile, _ = es.season_layers(raw, 2017)
    by = {2017: {"temp_c": layers["temp_avg"], es.COVER_COL: valid.astype("float32")}}
    with pytest.raises(ValueError, match="vuot"):
        es.season_table(by, profile, (["x"], None, [box(110.0, 10.0, 110.1, 10.1)]))


def test_script_chay_dau_cuoi(tmp_path):
    raw = _make_raw(tmp_path / "raw", 2017)
    grids = tmp_path / "grids"
    grids.mkdir()
    ids, _, geoms = _cells()
    gpd.GeoDataFrame({"cell_id": ids}, geometry=geoms, crs="EPSG:4326").to_file(grids / "g.geojson")
    boundary = tmp_path / "b.geojson"
    gpd.GeoDataFrame({"n": [1]}, geometry=[box(105.0, 10.0, 105.3, 10.1)], crs="EPSG:4326").to_file(boundary)
    out = tmp_path / "out"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "build_era5_season.py"), "--raw-dir", raw,
           "--grids-dir", str(grids), "--out-dir", str(out), "--boundary", str(boundary), "--years", "2017", "2017"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stdout + res.stderr
    df = pd.read_csv(out / "g_era5_season.csv", dtype={"cell_id": str})
    assert list(df.columns) == ["cell_id", "season", "rain_mm", "solar", "temp_c", "temp_max_c", "temp_min_c",
                                "rh_percent", "era5_cover_frac"]
    assert len(df) == 4 and df["season"].eq(2017).all()
    assert df.loc[df.cell_id == "sea", "rain_mm"].isna().all()
    prov = json.loads((out / "g_era5_season.csv.provenance.json").read_text(encoding="utf-8"))
    assert prov["grid_sha256"] and prov["boundary_sha256"]
    with rasterio.open(out / "rain_2017.tif") as ds:
        assert ds.count == 2 and ds.read(2)[0, 0] == 180
    assert (out / "era5_valid_2017.tif").exists()


# ---------- Cau K: lap pixel ERA5 khong hop le trong 1 pixel ----------

def test_lap_1_pixel_canh_uu_tien_cheo_va_cach_2_pixel_khong_lap():
    """Luoi 5x5, pixel hop le: (2,0)=10 va (0,4)=100 (goc tren phai). Moi pixel la dich (giao ranh gioi)."""
    v = np.full((5, 5), np.nan)
    v[2, 0], v[0, 4] = 10.0, 100.0
    valid = np.isfinite(v)
    out, filled, n_src = es.fill_nearest_1px({"x": v, "y": v * 2}, valid, np.ones((5, 5), bool))
    x = out["x"]
    assert x[2, 0] == 10.0 and x[0, 4] == 100.0                  # pixel hop le giu nguyen
    assert x[2, 1] == 10.0 and filled[2, 1] and n_src[2, 1] == 1  # canh
    assert x[1, 1] == 10.0 and filled[1, 1]                       # chi co cheo -> lay cheo
    assert np.isnan(x[2, 2]) and not filled[2, 2] and n_src[2, 2] == 0   # cach 2 pixel -> KHONG lap
    assert np.isnan(x[4, 2]) and not filled[4, 2]
    assert out["y"][2, 1] == 20.0                                 # moi bien cung pixel nguon
    assert not filled[valid].any()
    # pixel vua lap khong lam nguon: (2,2) ke (2,1) da lap nhung van NaN (da kiem o tren)


def test_lap_canh_truoc_cheo_va_trung_binh_khi_hoa():
    v = np.full((3, 3), np.nan)
    v[1, 0], v[1, 2] = 10.0, 20.0     # 2 canh cua tam (1,1)
    v[0, 0] = 1000.0                  # cheo cua tam - phai bi bo qua vi da co canh
    out, filled, n_src = es.fill_nearest_1px({"x": v}, np.isfinite(v), np.ones((3, 3), bool))
    assert out["x"][1, 1] == pytest.approx(15.0) and n_src[1, 1] == 2
    # (0,1): canh (0,0)=1000 -> 1000 (canh), khong tron cheo (1,0),(1,2)
    assert out["x"][0, 1] == pytest.approx(1000.0) and n_src[0, 1] == 1
    # (2,1): canh hop le khong co ((2,0),(2,2) NaN, (1,1) khong hop le goc) -> trung binh 2 cheo (1,0),(1,2)
    assert out["x"][2, 1] == pytest.approx(15.0) and n_src[2, 1] == 2


def test_chi_lap_pixel_giao_ranh_gioi():
    v = np.array([[5.0, np.nan, np.nan]])
    target = np.array([[False, False, True]])   # pixel giua KHONG giao ranh gioi
    out, filled, _ = es.fill_nearest_1px({"x": v}, np.isfinite(v), target)
    assert np.isnan(out["x"][0, 1]) and not filled[0, 1]
    assert np.isnan(out["x"][0, 2]) and not filled[0, 2]          # cach 2 pixel -> khong lap
    t2 = np.array([[False, True, True]])
    out, filled, _ = es.fill_nearest_1px({"x": v}, np.isfinite(v), t2)
    assert out["x"][0, 1] == 5.0 and filled.tolist() == [[False, True, False]]


def test_pixel_hop_le_ma_gia_tri_nan_bao_loi():
    v = np.array([[np.nan, 1.0]])
    with pytest.raises(ValueError, match="khong huu han"):
        es.fill_nearest_1px({"x": v}, np.array([[True, True]]), np.ones((1, 2), bool))


def test_boundary_touch_mask_giao_dien_tich(tmp_path):
    b = tmp_path / "b.geojson"
    # da giac phu A, va lan 0,01 do vao B; khong cham C
    gpd.GeoDataFrame({"n": [1]}, geometry=[box(105.0, 10.0, 105.11, 10.1)], crs="EPSG:4326").to_file(b)
    prof = {"crs": "EPSG:4326", "transform": TRANSFORM, "width": 3, "height": 1}
    assert es.boundary_touch_mask(str(b), prof).tolist() == [[True, True, False]]


def test_trich_fill_frac_va_dem_0(tmp_path):
    raw = _make_raw(tmp_path, 2017)
    layers, _, valid, profile, _ = es.season_layers(raw, 2017)
    target = np.ones_like(valid)
    fl, filled, _ = es.fill_nearest_1px(layers, valid, target)
    assert filled.tolist() == [[False, False, True]]               # C (bien) lap tu B (canh)
    assert fl["temp_avg"][0, 2] == pytest.approx(2.0) and fl["rain"][0, 2] == pytest.approx(360.0)
    by = {2017: {"temp_c": fl["temp_avg"], es.COVER_COL: valid.astype("float32"),
                 es.FILL_COL: filled.astype("float32")}}
    out = es.season_table(by, profile, _cells()).set_index("cell_id")
    assert out.loc["sea", "temp_c"] == pytest.approx(2.0)
    assert out.loc["sea", es.FILL_COL] == pytest.approx(1.0) and out.loc["sea", es.COVER_COL] == 0.0
    assert out.loc["bs", es.FILL_COL] == pytest.approx(0.5) and out.loc["bs", es.COVER_COL] == pytest.approx(0.5)
    assert out.loc["bs", "temp_c"] == pytest.approx(2.0)
    assert out.loc["ab", es.FILL_COL] == 0.0
    # o ven: nua ngoai raster (dem 0 cho FILL_COL, khong NaN) -> fill 0, cover 0,5
    assert out.loc["edge", es.FILL_COL] == 0.0 and out.loc["edge", es.COVER_COL] == pytest.approx(0.5)


def test_script_ban_lap_song_song(tmp_path):
    raw = _make_raw(tmp_path / "raw", 2017)
    grids = tmp_path / "grids"
    grids.mkdir()
    ids, _, geoms = _cells()
    gpd.GeoDataFrame({"cell_id": ids}, geometry=geoms, crs="EPSG:4326").to_file(grids / "g.geojson")
    boundary = tmp_path / "b.geojson"
    gpd.GeoDataFrame({"n": [1]}, geometry=[box(105.0, 10.0, 105.3, 10.1)], crs="EPSG:4326").to_file(boundary)
    out = tmp_path / "out"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "build_era5_season.py"), "--raw-dir", raw,
           "--grids-dir", str(grids), "--out-dir", str(out), "--boundary", str(boundary), "--years", "2017", "2017"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stdout + res.stderr
    df = pd.read_csv(out / "g_era5_season.csv", dtype={"cell_id": str}).set_index("cell_id")
    dff = pd.read_csv(out / "g_era5_season_filled.csv", dtype={"cell_id": str}).set_index("cell_id")
    assert list(dff.columns) == ["season", "rain_mm", "solar", "temp_c", "temp_max_c", "temp_min_c",
                                 "rh_percent", "era5_cover_frac", "era5_fill_frac"]
    assert np.isnan(df.loc["sea", "rain_mm"])                       # ban goc khong lap
    assert dff.loc["sea", "rain_mm"] == pytest.approx(360.0) and dff.loc["sea", "era5_fill_frac"] == 1.0
    assert dff.loc["ab", "rain_mm"] == df.loc["ab", "rain_mm"]      # o khong dinh pixel lap: giong het
    with rasterio.open(out / "era5_filled_2017.tif") as ds:
        assert ds.read(1).tolist() == [[0, 0, 1]] and ds.read(2).tolist() == [[0, 0, 1]]
        assert ds.descriptions == ("era5_filled", "n_source_px")
    with rasterio.open(out / "rain_2017_filled.tif") as ds:
        assert ds.read(1)[0, 2] == pytest.approx(360.0) and ds.read(2)[0, 2] == 0
    prov = json.loads((out / "g_era5_season_filled.csv.provenance.json").read_text(encoding="utf-8"))
    assert "fill_rule" in prov

"""Kiem thu nhan theo o (TB co trong so pham vi) + dap an diem cho 4 bien khi tuong Dot 7 - raster TONG HOP."""
import json
import os
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin
from shapely.geometry import Point, box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import met_cell_labels as mc  # noqa: E402
from met_labels import MODIS_SINU_PROJ4  # noqa: E402

UTM = "EPSG:32648"
X0, Y0 = 438540.0, 1219680.0


def direct_weighted_mean(values, weights, transform, poly):
    """Doi chung DOC LAP voi exactextract: cov = dien tich (pixel ∩ o) / dien tich pixel bang shapely;
    TB = sum(cov*w*v)/sum(cov*w) tren pixel v huu han. Tra (TB, sum_valid(cov*w), sum_all(cov*w))."""
    h, w = values.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    x0 = transform.c + cc * transform.a
    y0 = transform.f + rr * transform.e
    px = shapely.box(x0, y0 + transform.e, x0 + transform.a, y0)
    cov = shapely.area(shapely.intersection(px, poly)) / abs(transform.a * transform.e)
    v, wt = values[rr, cc], weights[rr, cc]
    ok = np.isfinite(v)
    s_all = float((cov * wt).sum())
    s_ok = float((cov * wt)[ok].sum())
    mean = float((cov * wt * np.where(ok, v, 0)).sum() / s_ok) if s_ok > 0 else np.nan
    return mean, s_ok, s_all


def _write(path, bands, transform, crs, names=None, dtype="float32", nodata=None):
    arr = np.stack([np.asarray(b) for b in bands]).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata) as d:
        d.write(arr)
        for i, n in enumerate(names or [], start=1):
            d.set_band_description(i, n)


def _scope(path, scope):
    """scope_mask_v3 tong hop (3 band theo ten) tren luoi 30 m UTM goc X0, Y0."""
    s = np.asarray(scope, np.uint8)
    _write(path, [s, s, np.where(s == 1, 40, 0)], from_origin(X0, Y0, 30, 30), UTM,
           ["scope", "zenodo_n_years", "wc_class"], dtype="uint8")


# ---------------------------------------------------------------- trong so pham vi
def test_scope_weights_luoi_chieu_dem_tam_pixel(tmp_path):
    sc = np.zeros((6, 6), np.uint8)
    sc[0:3, 0:3] = 1          # pixel nguon (0,0): 9/9
    sc[0, 3] = 1              # pixel nguon (0,1): 1/9
    sc[4, 4] = sc[5, 5] = 1   # pixel nguon (1,1): 2/9
    _scope(tmp_path / "s.tif", sc)
    w, info = mc.scope_weights(str(tmp_path / "s.tif"), UTM, from_origin(X0, Y0, 90, 90), (3, 3))
    np.testing.assert_allclose(w[:2, :2], [[1.0, 1 / 9], [0.0, 2 / 9]], atol=1e-9)
    assert w[2].sum() == 0 and w[:, 2].sum() == 0
    assert info["n_scope_px"] == 12 and info["n_scope_outside"] == 0 and info["n_px_w_gt0"] == 3
    # luoi nguon nho hon pham vi -> dem pixel scope nam ngoai (nguoi goi bao LOI)
    _, info2 = mc.scope_weights(str(tmp_path / "s.tif"), UTM, from_origin(X0, Y0, 90, 90), (1, 1))
    assert info2["n_scope_outside"] == 3


def test_pixel_areas_kinh_vi_giam_theo_cos_vi_do():
    a = mc.pixel_areas_m2("EPSG:4326", from_origin(105.0, 11.0, 0.1, 0.1), (30, 2))
    assert a[0, 0] == pytest.approx(a[0, 1])
    assert a[0, 0] == pytest.approx(11057 * 11119 * np.cos(np.radians(10.95)), rel=0.01)
    assert a[29, 0] / a[0, 0] == pytest.approx(np.cos(np.radians(8.05)) / np.cos(np.radians(10.95)), rel=1e-3)
    s = mc.pixel_areas_m2(MODIS_SINU_PROJ4, from_origin(0, 0, 926.625, 926.625), (2, 2))
    assert (s == pytest.approx(926.625 ** 2))


# ---------------------------------------------------------------- TB co trong so theo o
def test_cell_weighted_labels_khop_doi_chung_doc_lap(tmp_path):
    tr = from_origin(X0, Y0, 100, 100)
    v = np.array([[1, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], float)
    wt = np.array([[1, 0, 0.5, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1, 0.3, 1, 1]], float)
    bands, names, seasons = mc.stack_seasons({2019: v, 2020: v * 2}, nan_seasons=(2020,))
    mc.write_raster(str(tmp_path / "st.tif"), bands, UTM, tr, names)
    mc.write_raster(str(tmp_path / "w.tif"), [wt], UTM, tr, ["scope_weight"], nodata=None)
    polys = {"a": box(X0, Y0 - 200, X0 + 200, Y0),            # 1,2,5,NaN -> (1*1 + 5*1) / 2 = 3
             "c": box(X0, Y0 - 300, X0 + 400, Y0 - 200),      # trong so 0 toan bo -> NaN (khong dien 0)
             "d": box(X0 + 150, Y0 - 250, X0 + 300, Y0),      # cat ngang pixel
             "e": box(X0 + 30, Y0 - 390, X0 + 370, Y0 - 120)}  # da giac lech, phu nhieu hang
    g = gpd.GeoSeries(list(polys.values()), crs=UTM).to_crs(4326)
    out = mc.cell_weighted_labels(str(tmp_path / "st.tif"), str(tmp_path / "w.tif"),
                                  (list(polys), None, list(g)), seasons, "x")
    assert list(out.columns) == ["cell_id", "season", "x", "lbl_cover_frac", "lbl_valid_px", "lbl_ok"]
    o = out.set_index(["cell_id", "season"])
    assert o.loc[("a", 2019), "x"] == pytest.approx(3.0)
    assert o.loc[("a", 2019), "lbl_cover_frac"] == pytest.approx(2 / 3)   # pixel NaN co w = 1
    assert np.isnan(o.loc[("c", 2019), "x"]) and o.loc[("c", 2019), "lbl_valid_px"] == 0
    assert not o.loc[("c", 2019), "lbl_ok"]
    for cid, poly in polys.items():
        ref, s_ok, s_all = direct_weighted_mean(v, wt, tr, poly)
        got = o.loc[(cid, 2019)]
        if np.isnan(ref):
            assert np.isnan(got["x"])
        else:
            # da giac di qua EPSG:4326 roi chieu lai -> sai so dinh nho
            assert got["x"] == pytest.approx(ref, rel=1e-3)
            assert got["lbl_valid_px"] == pytest.approx(s_ok, rel=1e-3)
            assert got["lbl_cover_frac"] == pytest.approx(s_ok / s_all, rel=1e-3)
    # mua NaN theo quy tac: moi o NaN, phu 0
    assert o.xs(2020, level="season")["x"].isna().all()
    assert (o.xs(2020, level="season")["lbl_valid_px"] == 0).all()


def test_point_pixel_values_pixel_chua_diem_va_ngoai_luoi():
    tr = from_origin(0, 30, 10, 10)
    v = np.arange(9, dtype=float).reshape(3, 3)
    v[1, 1] = np.nan
    pv = mc.point_pixel_values({2019: v, 2020: v}, tr, np.array([5.0, 15.0, 29.9]), np.array([25.0, 15.0, 0.1]),
                               nan_seasons=(2020,))
    p19 = pv[pv["season"] == 2019]
    assert p19["value"].iloc[0] == 0.0 and np.isnan(p19["value"].iloc[1]) and p19["value"].iloc[2] == 8.0
    assert pv[pv["season"] == 2020]["value"].isna().all()
    with pytest.raises(ValueError):
        mc.point_pixel_values({2019: v}, tr, np.array([31.0]), np.array([5.0]))


def test_nan_pattern_changes_chi_tinh_pixel_co_trong_so():
    a = np.array([[1.0, np.nan, 1.0]])
    b = np.array([[np.nan, np.nan, 1.0]])
    w = np.array([[1.0, 1.0, 0.0]])
    assert mc.nan_pattern_changes({1: a, 2: b}, w) == (1, 1)
    assert mc.nan_pattern_changes({1: a, 2: b}, w, nan_seasons=(2,)) == (0, 1)


# ---------------------------------------------------------------- kiem nguon
def test_check_source_tu_choi_ban_lap_crs_sai_va_band_thieu(tmp_path):
    tr = from_origin(105, 10, 0.1, 0.1)
    _write(tmp_path / "temp_avg_2019.tif", [np.ones((2, 2)), np.ones((2, 2))], tr, "EPSG:4326",
           ["temp_c_mean", "n_days"])
    sig, b = mc.check_source(str(tmp_path / "temp_avg_2019.tif"), mc.MET_TARGETS["temp"])
    assert b == 1 and sig[2:] == (2, 2)
    _write(tmp_path / "temp_avg_2019_filled.tif", [np.ones((2, 2))], tr, "EPSG:4326", ["temp_c_mean"])
    with pytest.raises(ValueError, match="LAP"):
        mc.check_source(str(tmp_path / "temp_avg_2019_filled.tif"), mc.MET_TARGETS["temp"])
    _write(tmp_path / "humid_2019.tif", [np.ones((2, 2))], tr, "EPSG:4326", ["rh_mean"])
    with pytest.raises(ValueError, match="band"):
        mc.check_source(str(tmp_path / "humid_2019.tif"), mc.MET_TARGETS["rh"])
    with open(tmp_path / "temp_avg_2019.tif.provenance.json", "w", encoding="utf-8") as f:
        json.dump({"fill_rule": "x"}, f)
    with pytest.raises(ValueError, match="lap"):
        mc.check_source(str(tmp_path / "temp_avg_2019.tif"), mc.MET_TARGETS["temp"])
    str_tr = from_origin(11.3e6, 1.24e6, 926.625, 926.625)
    _write(tmp_path / "ell.tif", [np.ones((2, 2))], str_tr, "+proj=sinu +lon_0=0 +datum=WGS84 +units=m",
           ["dsr_mean"])
    with pytest.raises(ValueError, match="HINH CAU"):
        mc.check_source(str(tmp_path / "ell.tif"), mc.MET_TARGETS["dsr"])
    _write(tmp_path / "sph.tif", [np.ones((2, 2))], str_tr, MODIS_SINU_PROJ4, ["dsr_mean"])
    assert mc.check_source(str(tmp_path / "sph.tif"), mc.MET_TARGETS["dsr"])[1] == 1


# ---------------------------------------------------------------- script tren du lieu tong hop
def test_script_mua_buc_xa_end_to_end(tmp_path):
    from pyproj import Transformer

    n = 60
    sc = np.zeros((n, n), np.uint8)
    sc[:, :30] = 1                                             # pham vi chi nua trai
    _scope(tmp_path / "scope.tif", sc)
    to_ll = Transformer.from_crs(UTM, 4326, always_xy=True)
    lon0, lat1 = to_ll.transform(X0 - 300, Y0 + 300)
    lon1, lat0 = to_ll.transform(X0 + n * 30 + 300, Y0 - n * 30 - 300)
    res = 0.004
    tr_ll = from_origin(np.floor(lon0 / res) * res, np.ceil(lat1 / res) * res, res, res)
    h, w = int(np.ceil((tr_ll.f - lat0) / res)), int(np.ceil((lon1 - tr_ll.c) / res))
    data = tmp_path / "data"
    (data / "raw" / "chirps3").mkdir(parents=True)
    (data / "raw" / "mcd18a1").mkdir(parents=True)
    vals = {}
    for s in (2019, 2020):
        v = (np.arange(h * w, dtype=float).reshape(h, w) + 100 * (s - 2019))
        v[1, 1] = np.nan                                       # 1 pixel thieu (moi mua -> hop le theo quy tac E4)
        vals[s] = v
        _write(data / "raw" / "chirps3" / f"chirps3_rain_{s}.tif", [v, np.full_like(v, 36)], tr_ll, "EPSG:4326",
               ["rain_sum", "n_pentad_valid"], nodata=np.nan)
    to_sin = Transformer.from_crs(UTM, MODIS_SINU_PROJ4, always_xy=True)
    sx0, sy1 = to_sin.transform(X0 - 1000, Y0 + 1000)
    tr_s = from_origin(sx0, sy1, 500, 500)
    for s in (2019, 2020):
        _write(data / "raw" / "mcd18a1" / f"mcd18a1_dsr_{s}.tif", [np.full((8, 8), 200.0 + s - 2019)], tr_s,
               MODIS_SINU_PROJ4, ["dsr_mean"], nodata=np.nan)
    gdir = tmp_path / "grids"
    gdir.mkdir()
    cells = {"L": box(X0, Y0 - n * 30, X0 + 900, Y0), "R": box(X0 + 900, Y0 - n * 30, X0 + n * 30, Y0)}
    gpd.GeoDataFrame({"cell_id": list(cells)}, geometry=list(cells.values()), crs=UTM).to_crs(4326).to_file(
        gdir / "toy.geojson", driver="GeoJSON")
    pts = gpd.GeoDataFrame({"point_id": ["p1", "p2", "p3"], "block_id": ["b1", "b1", "b2"],
                            "is_holdout": [False, False, True]},
                           geometry=[Point(X0 + 100, Y0 - 100), Point(X0 + 1500, Y0 - 900), Point(X0 + 500, Y0 - 1700)],
                           crs=UTM).to_crs(4326)
    pts.to_file(tmp_path / "pts.geojson", driver="GeoJSON")
    pd.DataFrame({"point_id": ["p1", "p2", "p3", "p1", "p2", "p3"], "season": [2019] * 3 + [2020] * 3,
                  "ref_salinity": [1.0, np.nan, np.nan, 1.0, np.nan, 2.0]}).to_csv(tmp_path / "pref.csv", index=False)
    bnd = gpd.GeoDataFrame(geometry=[box(X0 - 1, Y0 - n * 30 - 1, X0 + n * 30 + 1, Y0 + 1)], crs=UTM).to_crs(4326)
    bnd.to_file(tmp_path / "bnd.geojson", driver="GeoJSON")
    out, rep = tmp_path / "out", tmp_path / "rep"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "build_met_labels.py"), "--targets", "rain", "dsr",
           "--data-root", str(data), "--scope", str(tmp_path / "scope.tif"), "--boundary", str(tmp_path / "bnd.geojson"),
           "--points", str(tmp_path / "pts.geojson"), "--points-ref-sal", str(tmp_path / "pref.csv"),
           "--n-points-expected", "2", "--grids-dir", str(gdir), "--years", "2019", "2020", "--out-dir", str(out),
           "--work-dir", str(tmp_path / "work"), "--report-dir", str(rep)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr

    rain = pd.read_csv(out / "toy_labels_season_rain_chirps.csv", dtype={"cell_id": str})
    assert list(rain.columns) == ["cell_id", "season", "rain_chirps", "lbl_cover_frac", "lbl_valid_px", "lbl_ok"]
    assert len(rain) == 4 and not rain.duplicated(["cell_id", "season"]).any()
    with rasterio.open(out / "scope_weight_chirps3.tif") as ds:
        wt = ds.read(1).astype(float)
    assert wt.max() <= 1.0 and wt.sum() > 0
    rr = rain.set_index(["cell_id", "season"])
    for cid, poly in cells.items():
        p_ll = gpd.GeoSeries([poly], crs=UTM).to_crs(4326).iloc[0]
        for s in (2019, 2020):
            ref, s_ok, _ = direct_weighted_mean(vals[s], wt, tr_ll, p_ll)
            got = rr.loc[(cid, s)]
            assert got["rain_chirps"] == pytest.approx(ref, rel=1e-5)   # CSV ghi %.6g
            assert got["lbl_valid_px"] == pytest.approx(s_ok, rel=1e-4)
    # mua 2020 cua buc xa = NaN toan bo (o + diem), 2019 = 200 (raster hang so)
    dsr = pd.read_csv(out / "toy_labels_season_dsr_mcd18.csv", dtype={"cell_id": str}).set_index(["cell_id", "season"])
    assert dsr.xs(2020, level="season")["dsr_mcd18"].isna().all()
    assert dsr.loc[("L", 2019), "dsr_mcd18"] == pytest.approx(200.0)

    # diem: chi p1, p3 (co ref_salinity >= 1 mua); gia tri = pixel chua diem
    pr = pd.read_csv(out / "points_reference_rain_chirps.csv", dtype={"point_id": str})
    assert sorted(pr["point_id"].unique()) == ["p1", "p3"]
    assert {"ref_rain_chirps", "src_px_valid", "block_id", "is_holdout"} <= set(pr.columns)
    p1 = pts.set_index("point_id").loc["p1"].geometry
    c = int(np.floor((p1.x - tr_ll.c) / res))
    rw = int(np.floor((tr_ll.f - p1.y) / res))
    got = pr[(pr["point_id"] == "p1") & (pr["season"] == 2019)]["ref_rain_chirps"].iloc[0]
    assert got == pytest.approx(vals[2019][rw, c])
    pd_ = pd.read_csv(out / "points_reference_dsr_mcd18.csv", dtype={"point_id": str})
    assert pd_[pd_["season"] == 2020]["ref_dsr_mcd18"].isna().all()
    assert not pd_[pd_["season"] == 2020]["src_px_valid"].any()
    prov = json.load(open(out / "points_reference_dsr_mcd18.csv.provenance.json", encoding="utf-8"))
    assert prov["variant"] == "main" and prov["ref_rule_kind"] == "pixel" and prov["target"] == "dsr_mcd18"
    assert prov["nan_seasons"] == [2020]
    assert (rep / "dot7_e5_khi_tuong_luoi_rain_dsr.csv").exists()

    # chay lai: bo qua phan da co, ket qua khong doi
    r2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r2.returncode == 0 and "da co - bo qua" in r2.stdout, r2.stdout + r2.stderr
    pd.testing.assert_frame_equal(pd.read_csv(out / "toy_labels_season_rain_chirps.csv", dtype={"cell_id": str}), rain)

    # MCD18 sai CRS (ellipsoid) -> LOI
    for s in (2019, 2020):
        with rasterio.open(data / "raw" / "mcd18a1" / f"mcd18a1_dsr_{s}.tif", "r+") as ds:
            ds.crs = rasterio.crs.CRS.from_proj4("+proj=sinu +lon_0=0 +datum=WGS84 +units=m")
    r3 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r3.returncode != 0 and "HINH CAU" in (r3.stdout + r3.stderr)

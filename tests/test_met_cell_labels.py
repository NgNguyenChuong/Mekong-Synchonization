"""Kiem thu nhan theo o (cach giong dac trung tinh: TB tren tam pixel pham vi 30 m) + dap an diem cho 4 bien
khi tuong Dot 7 - raster TONG HOP, doi chung tinh truc tiep bang numpy/shapely."""
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


def direct_scope_mean(values, src_transform, scope, poly, to_src=None):
    """Doi chung DOC LAP (khong rasterize, khong bincount): duyet tung pixel pham vi 30 m, tam nam TRONG da giac o
    (shapely, CRS UTM) -> lay gia tri pixel nguon chua tam (floor). TB tren pixel co gia tri huu han.
    Tra (TB, so pixel hop le, so pixel pham vi trong o). to_src: ham (x, y) UTM -> CRS nguon (mac dinh dong nhat)."""
    rr, cc = np.nonzero(np.asarray(scope) == 1)
    x = X0 + (cc + 0.5) * 30.0
    y = Y0 - (rr + 0.5) * 30.0
    inside = shapely.contains_xy(poly, x, y)
    x, y = x[inside], y[inside]
    if to_src is not None:
        x, y = to_src(x, y)
    inv = ~src_transform
    col = np.floor(inv.a * x + inv.b * y + inv.c).astype(int)
    row = np.floor(inv.d * x + inv.e * y + inv.f).astype(int)
    v = values[row, col]
    ok = np.isfinite(v)
    return (float(v[ok].mean()) if ok.any() else np.nan), int(ok.sum()), int(inside.sum())


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


def _cells(polys):
    """(cell_ids, None, geoms EPSG:4326) tu {id: da giac UTM}."""
    g = gpd.GeoSeries(list(polys.values()), crs=UTM).to_crs(4326)
    return list(polys), None, list(g)


# ---------------------------------------------------------------- ma tran dem o x pixel nguon
def test_scope_source_counts_dem_tam_pixel_pham_vi(tmp_path):
    sc = np.zeros((6, 6), np.uint8)
    sc[0:3, 0:3] = 1          # pixel nguon 90 m (0,0): 9 pixel pham vi
    sc[0, 3] = 1              # pixel nguon (0,1): 1
    sc[4, 4] = sc[5, 5] = 1   # pixel nguon (1,1): 2
    _scope(tmp_path / "s.tif", sc)
    tr = from_origin(X0, Y0, 90, 90)
    polys = {"A": box(X0, Y0 - 180, X0 + 180, Y0), "B": box(X0, Y0 - 180 - 90, X0 + 180, Y0 - 180),
             "C": box(X0 - 90, Y0 - 90, X0, Y0)}
    ids, counts, info = mc.scope_source_counts(str(tmp_path / "s.tif"), _cells(polys), {"k": (UTM, tr, (3, 3))})
    assert list(ids) == ["A", "B", "C"]
    got = {(ids[c], p): n for c, p, n in counts["k"][["cell", "px", "n"]].itertuples(index=False)}
    # o A (hang/cot 30 m 0-5) chua ca 12 pixel pham vi; B, C nam ngoai raster pham vi -> khong co dong
    assert got == {("A", 0): 9, ("A", 1): 1, ("A", 4): 2}
    assert info["k"]["n_scope_in_cells"] == 12 and info["k"]["n_scope_outside"] == 0
    # luoi nguon nho hon pham vi -> dem pixel pham vi nam ngoai (nguoi goi bao LOI)
    _, _, info2 = mc.scope_source_counts(str(tmp_path / "s.tif"), _cells(polys), {"k": (UTM, tr, (1, 1))})
    assert info2["k"]["n_scope_outside"] == 3


def test_tong_dem_moi_o_trung_scope_n_px_cua_dac_trung_tinh(tmp_path):
    """Gan pixel 30 m cho o GIONG HET scope_centroids (scope_n_px trong bang dac trung tinh)."""
    from static_features import scope_centroids

    rng = np.random.default_rng(0)
    sc = (rng.random((40, 40)) < 0.6).astype(np.uint8)
    _scope(tmp_path / "s.tif", sc)
    polys = {"h": shapely.Polygon([(X0 + 37, Y0 - 11), (X0 + 610, Y0 - 140), (X0 + 500, Y0 - 900),
                                    (X0 + 20, Y0 - 700)]),
             "q": box(X0 + 610, Y0 - 1200, X0 + 1200, Y0 - 140)}
    cells = _cells(polys)
    ids, counts, _ = mc.scope_source_counts(str(tmp_path / "s.tif"), cells,
                                            {"k": (UTM, from_origin(X0, Y0, 300, 300), (4, 4))})
    tot = counts["k"].groupby("cell")["n"].sum()
    ref = scope_centroids(str(tmp_path / "s.tif"), cells).set_index("cell_id")["scope_n_px"]
    for i, cid in enumerate(ids):
        assert tot.get(i, 0) == ref[cid]


# ---------------------------------------------------------------- nhan o tu ma tran dem
def _labels(tmp_path, scope, values, src_tr, polys, nan_seasons=()):
    _scope(tmp_path / "s.tif", scope)
    shape = next(iter(values.values())).shape
    ids, counts, _ = mc.scope_source_counts(str(tmp_path / "s.tif"), _cells(polys), {"k": (UTM, src_tr, shape)})
    return mc.cell_count_labels(counts["k"], ids, values, "x", nan_seasons).set_index(["cell_id", "season"])


def test_o_khong_pham_vi_la_nan(tmp_path):
    sc = np.zeros((6, 6), np.uint8)
    sc[:, :3] = 1                                       # pham vi chi nua trai
    v = np.array([[1.0, 2.0], [3.0, 4.0]])
    o = _labels(tmp_path, sc, {2019: v}, from_origin(X0, Y0, 90, 90),
                {"trai": box(X0, Y0 - 180, X0 + 90, Y0), "phai": box(X0 + 90, Y0 - 180, X0 + 180, Y0)})
    assert o.loc[("trai", 2019), "x"] == pytest.approx(2.0)          # (9*1 + 9*3) / 18
    r = o.loc[("phai", 2019)]
    assert np.isnan(r["x"]) and np.isnan(r["lbl_cover_frac"]) and r["lbl_valid_px"] == 0 and not r["lbl_ok"]


def test_o_pham_vi_chi_tren_pixel_nguon_nan_la_nan_khong_muon_lang_gieng(tmp_path):
    """(iii): pham vi cua o chi nam tren pixel nguon NaN -> NaN, du pixel nguon HOP LE lang gieng cham/phu mot phan
    o (cach exactextract weighted_mean cu se muon gia tri cua no)."""
    sc = np.zeros((6, 6), np.uint8)
    sc[0:3, 0:3] = 1          # pham vi o chi trong pixel nguon (0,0) = NaN
    sc[0:3, 3:6] = 1          # pham vi o pixel lang gieng (0,1) = 7 - NHUNG nam ngoai o
    v = np.array([[np.nan, 7.0], [5.0, 6.0]])
    # o phu pixel (0,0) + 1/9 pixel (0,1); tam cot 30 m thu 3 (X0+105) nam ngoai o (mep X0+100)
    o = _labels(tmp_path, sc, {2019: v}, from_origin(X0, Y0, 90, 90), {"ven": box(X0, Y0 - 90, X0 + 100, Y0)})
    r = o.loc[("ven", 2019)]
    assert np.isnan(r["x"]) and not r["lbl_ok"]
    assert r["lbl_cover_frac"] == 0.0 and r["lbl_valid_px"] == 0
    # doi chung: pixel lang gieng that su phu o va CO pham vi trong o -> chi lay phan do
    sc2 = sc.copy()
    o2 = _labels(tmp_path, sc2, {2019: v}, from_origin(X0, Y0, 90, 90), {"ven": box(X0, Y0 - 90, X0 + 150, Y0)})
    assert o2.loc[("ven", 2019), "x"] == pytest.approx(7.0)          # 9 pixel NaN bi bo, 2 cot (6 px) gia tri 7
    assert o2.loc[("ven", 2019), "lbl_cover_frac"] == pytest.approx(6 / 15)


def test_tb_theo_dien_tich_pixel_30m_va_o_trong_mot_pixel(tmp_path):
    """TB theo so pixel pham vi 30 m (cung dien tich), KHONG theo dien tich pixel nguon; o nam tron 1 pixel nguon
    = gia tri pixel do; mua nan_seasons -> NaN."""
    sc = np.zeros((6, 6), np.uint8)
    sc[0:3, 0:3] = 1          # 9 pixel pham vi trong pixel nguon (0,0) = 10
    sc[0, 3] = 1              # 1 pixel pham vi trong pixel nguon (0,1) = 20
    sc[3:6, 3:6] = 1
    v = np.array([[10.0, 20.0], [30.0, 40.0]])
    polys = {"hai": box(X0, Y0 - 90, X0 + 180, Y0),           # TB dien tich pham vi = (9*10 + 1*20) / 10 = 11
             "mot": box(X0 + 100, Y0 - 175, X0 + 175, Y0 - 95)}       # tron trong pixel nguon (1,1)
    o = _labels(tmp_path, sc, {2019: v, 2020: v * 2}, from_origin(X0, Y0, 90, 90), polys, nan_seasons=(2020,))
    assert o.loc[("hai", 2019), "x"] == pytest.approx(11.0)            # TB pixel nguon thuong = 15 -> sai
    assert o.loc[("hai", 2019), "lbl_valid_px"] == 10 and o.loc[("hai", 2019), "lbl_cover_frac"] == 1.0
    assert o.loc[("mot", 2019), "x"] == 40.0
    assert o.xs(2020, level="season")["x"].isna().all()
    assert (o.xs(2020, level="season")["lbl_valid_px"] == 0).all()


def test_cell_count_labels_khop_doi_chung_truc_tiep(tmp_path):
    """Da giac lech + pixel nguon NaN + pham vi ngau nhien: khop phep tinh truc tiep tung pixel 30 m."""
    rng = np.random.default_rng(1)
    sc = (rng.random((30, 30)) < 0.5).astype(np.uint8)
    tr = from_origin(X0 - 40, Y0 + 25, 110, 110)                       # luoi nguon KHONG can voi luoi 30 m
    v = rng.random((10, 10)) * 100
    v[1, 2] = v[4, 4] = np.nan
    polys = {"a": shapely.Polygon([(X0 + 17, Y0 - 3), (X0 + 433, Y0 - 61), (X0 + 380, Y0 - 512), (X0 + 9, Y0 - 470)]),
             "b": box(X0 + 433, Y0 - 899, X0 + 899, Y0 - 300), "c": box(X0 + 100, Y0 - 899, X0 + 433, Y0 - 512)}
    o = _labels(tmp_path, sc, {2019: v}, tr, polys)
    for cid, poly in polys.items():
        ref, n_ok, n_all = direct_scope_mean(v, tr, sc, poly)
        got = o.loc[(cid, 2019)]
        assert got["lbl_valid_px"] == n_ok and got["lbl_cover_frac"] == pytest.approx(n_ok / n_all)
        assert got["x"] == pytest.approx(ref, rel=1e-12)


def test_cell_count_labels_tu_choi_ma_tran_loi():
    c = pd.DataFrame({"cell": [0, 0], "px": [1, 1], "n": [2, 3]})
    with pytest.raises(ValueError, match="trung"):
        mc.cell_count_labels(c, np.array(["a"]), {2019: np.ones((2, 2))}, "x")
    with pytest.raises(ValueError, match="n <= 0"):
        mc.cell_count_labels(c.assign(n=[0, 1], px=[0, 1]), np.array(["a"]), {2019: np.ones((2, 2))}, "x")
    with pytest.raises(ValueError, match="vuot"):
        mc.cell_count_labels(pd.DataFrame({"cell": [0], "px": [9], "n": [1]}), np.array(["a"]),
                             {2019: np.ones((2, 2))}, "x")


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
           "--report-dir", str(rep)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr

    rain = pd.read_csv(out / "toy_labels_season_rain_chirps.csv", dtype={"cell_id": str})
    assert list(rain.columns) == ["cell_id", "season", "rain_chirps", "lbl_cover_frac", "lbl_valid_px", "lbl_ok"]
    assert len(rain) == 4 and not rain.duplicated(["cell_id", "season"]).any()
    assert not list(out.glob("scope_weight_*"))                  # cach cu (raster trong so) da bo
    rr = rain.set_index(["cell_id", "season"])
    for cid, poly in cells.items():
        for s in (2019, 2020):
            ref, n_ok, n_all = direct_scope_mean(vals[s], tr_ll, sc, poly, to_src=to_ll.transform)
            got = rr.loc[(cid, s)]
            if n_all == 0:                                         # o R: khong co pixel pham vi -> NaN
                assert cid == "R" and np.isnan(got["rain_chirps"]) and np.isnan(got["lbl_cover_frac"])
                continue
            assert got["rain_chirps"] == pytest.approx(ref, rel=1e-5)   # CSV ghi %.6g
            assert got["lbl_valid_px"] == n_ok and got["lbl_cover_frac"] == pytest.approx(n_ok / n_all, rel=1e-5)
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

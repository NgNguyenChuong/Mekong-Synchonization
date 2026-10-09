"""Kiem thu nhan theo (o, mua) va tham chieu diem (P5) - raster TONG HOP gia tri biet truoc."""
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
from shapely.geometry import Point, box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import label_season as ls  # noqa: E402
from scope_mask import build_scope_mask, write_scope_mask  # noqa: E402

CRS = "EPSG:32648"
X0, Y0 = 438540.0, 1219680.0


# ---------------------------------------------------------------- quy tac NaN
def test_nan_rule_three_conditions_and_no_fill():
    ndwi = np.array([0.2, np.nan, 0.2, 1.0, 0.2, -1.0, 0.2, -0.5], np.float32)
    sal = np.array([1.5, 1.5, 28.5, 1.5, 28.013, 1.5, np.nan, 27.9], np.float32)
    out = ls.nan_rule(ndwi, sal)
    # 0 hop le; 1 NDWI NaN; 2 S > 28,013; 3 NDWI = 1 -> B7 = 0 (<= 0); 4 S = 28,013 -> B5 = 0 -> B7 = 0;
    # 5 NDWI = -1 -> chia 0 (B7 khong huu han); 6 S NaN; 7 S < 28,013 va NDWI am -> hop le (B7 > 0)
    assert out[0] == pytest.approx(1.5) and out[7] == pytest.approx(27.9)
    assert np.isnan(out[[1, 2, 3, 4, 5, 6]]).all()
    assert out.dtype == np.float32
    assert not np.any(out[np.isnan(out)] == 0)


def test_nan_rule_b7_le_zero_only():
    # S hop le (< 28,013), NDWI hop le nhung > 1 (phi vat ly) -> (1 - NDWI) < 0 -> B7 < 0 -> NaN
    out = ls.nan_rule(np.array([1.2, 0.99]), np.array([3.0, 3.0]))
    assert np.isnan(out[0]) and out[1] == pytest.approx(3.0)


# ---------------------------------------------------------------- mask nuoc
def test_water_mask_threshold_255_and_wc80():
    wf = np.array([49, 50, 100, 255, 0, 255], np.uint8)
    wc = np.array([40, 40, 40, 40, 80, 80], np.uint8)
    assert ls.water_mask(wf, wc).tolist() == [False, True, True, False, True, True]


def test_clear_ok_chg08():
    # CHG-08: n_clear 0, 1 -> khong tin cay; 2 -> giu; water_freq 255 -> khong tin cay du n_clear lon
    wf = np.array([0, 0, 0, 30, 255, 100], np.uint8)
    ncl = np.array([0, 1, 2, 7, 9, 2], np.uint8)
    assert ls.clear_ok(wf, ncl).tolist() == [False, False, True, True, False, True]
    assert ls.MIN_CLEAR == 2


def test_variant_masks_membership():
    #   dat  WC80  WC95  WC0  nuoc dong  ngoai v2  ngoai dau chan  NaN  WC60  WC90  WC50  n_clear 1  n_clear 2  wf255
    finite = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], bool)
    in_v2 = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], bool)
    fp = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], bool)
    wc = np.array([40, 80, 95, 0, 40, 40, 40, 40, 60, 90, 50, 10, 30, 20], np.uint8)
    wf = np.array([0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 255], np.uint8)
    ncl = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 2, 0], np.uint8)
    m = ls.variant_masks(finite, in_v2, fp, wc, ls.water_mask(wf, wc), ls.clear_ok(wf, ncl))
    assert m["before"].astype(int).tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert m["main"].astype(int).tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert m["keepwater"].astype(int).tolist() == [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert m["keepmangrove"].astype(int).tolist() == [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert m["keep6090"].astype(int).tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]


def _all_combos():
    """Moi to hop (lop WC, nuoc dong, n_clear, water_freq 255, huu han, v2, dau chan)."""
    import itertools
    classes = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
    rows = list(itertools.product(classes, (0, 49, 50, 100, 255), (0, 1, 2, 3), (0, 1), (0, 1), (0, 1)))
    a = np.array(rows)
    wc, wf, ncl = a[:, 0].astype(np.uint8), a[:, 1].astype(np.uint8), a[:, 2].astype(np.uint8)
    fin, v2, fp = a[:, 3].astype(bool), a[:, 4].astype(bool), a[:, 5].astype(bool)
    return wc, wf, ncl, fin, v2, fp


def test_variants_superset_of_main_and_differ_by_one_condition():
    wc, wf, ncl, fin, v2, fp = _all_combos()
    water, clear = ls.water_mask(wf, wc), ls.clear_ok(wf, ncl)
    m = ls.variant_masks(fin, v2, fp, wc, water, clear)
    base = fin & v2 & fp & clear
    in_main_lc = np.isin(wc, (10, 20, 30, 40))
    assert np.array_equal(m["main"], base & in_main_lc & ~water)
    for v in ("keepwater", "keepmangrove", "keep6090"):
        assert not (m["main"] & ~m[v]).any(), f"{v} khong chua bo chinh"
        assert not (m[v] & ~clear).any(), f"{v}: CHG-08 khong ap"
        assert not (m[v] & np.isin(wc, (0, 50, 70, 100))).any(), f"{v}: lop khong duoc phep"
    extra = {v: m[v] & ~m["main"] for v in ("keepwater", "keepmangrove", "keep6090")}
    # phan them chi thuoc dung dieu kien cua bo
    assert np.array_equal(extra["keepwater"], base & ((wc == 80) | (in_main_lc & water)))
    assert np.array_equal(extra["keepmangrove"], base & (wc == 95) & ~water)
    assert np.array_equal(extra["keep6090"], base & np.isin(wc, (60, 90)) & ~water)
    assert extra["keepwater"].any() and extra["keepmangrove"].any() and extra["keep6090"].any()


def test_comparison_masks_split_effects():
    wc, wf, ncl, fin, v2, fp = _all_combos()
    water, clear = ls.water_mask(wf, wc), ls.clear_ok(wf, ncl)
    m = ls.variant_masks(fin, v2, fp, wc, water, clear)
    c = ls.comparison_masks(fin, v2, fp, wc, water, wf, ncl)
    base = fin & v2 & fp
    assert np.array_equal(c["v2_old"], base & ~np.isin(wc, (0, 80, 95)) & ~water)   # quy tac P5 cu
    # hai buoc noi tiep: v2_old -(a)-> v3_noclear -(b)-> main
    assert np.array_equal(c["v3_noclear"] & clear, m["main"])
    assert np.array_equal(c["v2_old"] & ~c["drop_lc"], c["v3_noclear"])
    assert np.array_equal(c["v3_noclear"] & ~c["drop_clear"], m["main"])
    assert np.array_equal(c["v2_clear"] & np.isin(wc, (10, 20, 30, 40)), m["main"])   # thu tu nguoc
    assert np.array_equal(c["drop_lc"], c["v2_old"] & np.isin(wc, (50, 60, 70, 90, 100)))
    assert np.array_equal(c["drop_lc50"] | c["drop_lc6090"], c["drop_lc"] & np.isin(wc, (50, 60, 90)))
    assert not (c["drop_clear_n0"] & c["drop_clear_n1"]).any()
    assert np.array_equal(c["drop_clear_n0"] | c["drop_clear_n1"], c["drop_clear"])
    assert set(ncl[c["drop_clear_n1"]].tolist()) == {1}
    assert np.all((ncl[c["drop_clear_n0"]] == 0) | (wf[c["drop_clear_n0"]] == 255))


def test_variant_point_sets():
    main = pd.DataFrame({"point_id": ["a", "b"], "block_id": ["k1", "k2"], "wc_class": [40, 10]})
    src = pd.DataFrame({"point_id": ["a", "b", "c", "d", "e", "f", "g"], "block_id": ["k1", "k2", "k3", "k3", "k4",
                                                                                        "k4", "k5"],
                        "wc_class": [40, 10, 60, 90, 50, 95, 40]})
    out = ls.variant_point_sets(main, src)
    assert out["point_id"].tolist() == ["a", "b", "c", "d", "f"]          # 50 va diem 40 bi loc (g) khong them
    assert out.loc[out["in_keep6090"], "point_id"].tolist() == ["a", "b", "c", "d"]
    assert out.loc[out["in_keepmangrove"], "point_id"].tolist() == ["a", "b", "f"]
    assert out.loc[out["in_keepwater"], "point_id"].tolist() == ["a", "b"]
    assert out.loc[out["in_main"], "point_id"].tolist() == ["a", "b"]
    assert out.set_index("point_id").loc["c", "block_id"] == "k3"         # giu thuoc tinh goc
    with pytest.raises(ValueError):
        ls.variant_point_sets(pd.concat([main, main]), src)


# ---------------------------------------------------------------- can luoi
def _write(path, bands, transform, names=None, dtype="float32"):
    bands = [np.asarray(b, dtype=dtype) for b in bands]
    h, w = bands[0].shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=len(bands), dtype=dtype, crs=CRS,
                       transform=transform) as ds:
        for i, b in enumerate(bands, start=1):
            ds.write(b, i)
            if names:
                ds.set_band_description(i, names[i - 1])


def test_read_aligned_integer_offset_and_fill(tmp_path):
    src = np.arange(5 * 6, dtype="float32").reshape(5, 6)
    # nguon goc lech 2 cot TAY, 1 hang BAC so voi luoi dich
    p = tmp_path / "src.tif"
    _write(p, [src, src * 10], from_origin(X0 - 60, Y0 + 30, 30, 30), names=["NDWIchen", "Salinity"])
    dst_tr = from_origin(X0, Y0, 30, 30)
    with rasterio.open(p) as s:
        out = ls.read_aligned(s, "Salinity", dst_tr, (5, 5), row0=0, fill=np.nan)
        out1 = ls.read_aligned(s, "Salinity", dst_tr, (2, 5), row0=3, fill=np.nan)
    # pixel dich (r, c) = nguon (r + 1, c + 2); nguon chi co 6 cot -> cot dich 4 ngoai khung; hang dich 4 ngoai
    exp = np.full((5, 5), np.nan, np.float32)
    exp[:4, :4] = src[1:5, 2:6] * 10
    np.testing.assert_array_equal(out, exp)
    np.testing.assert_array_equal(out1, exp[3:5])


def test_read_aligned_rejects_non_integer_offset(tmp_path):
    p = tmp_path / "src.tif"
    _write(p, [np.zeros((3, 3))], from_origin(X0 + 15, Y0, 30, 30), names=["Salinity"])
    with rasterio.open(p) as s, pytest.raises(ValueError):
        ls.read_aligned(s, "Salinity", from_origin(X0, Y0, 30, 30), (3, 3))


# ---------------------------------------------------------------- median 3x3
def test_median_3x3_min_valid_and_edges():
    a = np.arange(1, 26, dtype=float).reshape(5, 5)
    b = a.copy()
    b[0:3, 0:2] = np.nan            # quanh (1, 1): con 3 hop le (cot 2) -> NaN
    c = a.copy()
    c[0:2, 0:2] = np.nan            # quanh (1, 1): con 5 hop le -> median cua 5
    med, n = ls.median_3x3(a, [2, 0], [2, 0])
    assert med[0] == pytest.approx(13.0) and n[0] == 9
    assert np.isnan(med[1]) and n[1] == 4       # goc: chi 4 pixel trong khung -> < 5 -> NaN
    med, n = ls.median_3x3(b, [1], [1])
    assert np.isnan(med[0]) and n[0] == 3
    med, n = ls.median_3x3(c, [1], [1])
    assert n[0] == 5 and med[0] == pytest.approx(np.median([3, 8, 11, 12, 13]))
    med, n = ls.median_3x3(a, [-5], [2])        # pixel ngoai khung hoan toan
    assert np.isnan(med[0]) and n[0] == 0


# ---------------------------------------------------------------- nguong 100 px
def test_cell_label_rows_threshold_and_nan():
    out = ls.cell_label_rows(mean=[2.0, 3.0, 5.0], count=[99.99, 100.0, 0.0],
                             cell_area_m2=[1000 * 900.0, 100 * 900.0, 900.0], px_area_m2=900.0)
    assert out["train_ok"].tolist() == [False, True, False]
    assert out["valid_frac"].tolist() == pytest.approx([0.09999, 1.0, 0.0])
    assert out["train_ok_10pct"].tolist() == [False, True, False]
    assert out["salinity"].iloc[:2].tolist() == [2.0, 3.0]
    assert np.isnan(out["salinity"].iloc[2])    # khong co pixel hop le -> NaN, khong phai 0 / 5


def test_area_weighted_mean_count_with_nan(tmp_path):
    a = np.array([[1, 2, np.nan, 4], [5, 6, np.nan, 8]], np.float32)
    p = tmp_path / "x.tif"
    _write(p, [a, a * 10], from_origin(X0, Y0, 30, 30))
    g = gpd.GeoSeries([box(X0, Y0 - 60, X0 + 60, Y0),           # 4 pixel hop le: mean 3.5
                       box(X0 + 45, Y0 - 60, X0 + 105, Y0),     # nua cot 1 (2,6), cot 2 NaN, nua cot 3 (4,8)
                       box(X0 + 62, Y0 - 58, X0 + 88, Y0 - 2),  # chi cot NaN -> NaN, count 0
                       box(X0 + 900, Y0, X0 + 960, Y0 + 60)], crs=CRS).to_crs(4326)
    df = ls.area_weighted_mean_count(str(p), (["a", "b", "c", "d"], None, list(g)))
    assert df["b1_mean"].iloc[0] == pytest.approx(3.5, abs=1e-4)
    assert df["b1_count"].iloc[0] == pytest.approx(4.0, abs=0.05)
    assert df["b1_mean"].iloc[1] == pytest.approx(5.0, abs=1e-4)       # (2+6+4+8)/4, NaN bi bo qua
    assert df["b1_count"].iloc[1] == pytest.approx(2.0, abs=1e-3)      # 4 nua pixel
    assert df["b2_mean"].iloc[0] == pytest.approx(35.0, abs=0.05)
    assert np.isnan(df["b1_mean"].iloc[2]) and df["b1_count"].iloc[2] == 0
    assert np.isnan(df["b1_mean"].iloc[3]) and df["b1_count"].iloc[3] == 0
    assert df["cell_area_m2"].iloc[0] == pytest.approx(3600.0, rel=1e-3)


def test_value_summary():
    d = ls.value_summary(np.array([1.0, 2.0, 3.0, 5.0]))
    assert d["n_px"] == 4 and d["n_gt2"] == 2 and d["n_gt4"] == 1 and d["pct_gt4"] == 25.0
    assert d["min"] == 1.0 and d["max"] == 5.0 and d["p50"] == 2.5
    assert d["n_gt21"] == 0 and d["pct_gt21"] == 0.0


# ---------------------------------------------------------------- chay script tren du lieu tong hop
def _px_point(r, c):
    return Point(X0 + c * 30 + 15, Y0 - r * 30 - 15)


def _toy_e2e(tmp_path, ndwi_zen=None, ndwi_gee=None):
    """Du lieu tong hop cho script (dung chung test do man va E5a NDWI). Mac dinh NDWI = 0,2 moi pixel.
    Tra (cmd chay script, thu muc out, work, rep)."""
    n = 12
    tr = from_origin(X0, Y0, 30, 30)
    zen, gee, out, work, rep = (tmp_path / d for d in ("zen", "gee", "out", "work", "rep"))
    for d in (zen, gee):
        d.mkdir()
    sal = np.full((n, n), 1.0, np.float32)
    sal[:, n // 2:] = 3.0                                      # trai 1, phai 3
    sal[0, 0] = 30.0                                           # S > 28,013 -> NaN
    sal[n - 1, n - 1] = np.nan                                 # ngoai dau chan (khong co nam nao)
    ndwi = np.full((n, n), 0.2, np.float32) if ndwi_zen is None else ndwi_zen.astype(np.float32)
    _write(zen / "2023_MD_dry_NDWIchen_Salinity.tif", [ndwi, sal], tr, ["NDWIchen", "Salinity"])
    # GEE 2024: luoi lech 2 cot tay, 1 hang bac; gia tri = Zenodo + 1; phu ca ngoai dau chan
    g_tr = from_origin(X0 - 60, Y0 + 30, 30, 30)
    gs = np.full((n + 2, n + 3), 9.0, np.float32)
    gs[1:n + 1, 2:n + 2] = np.where(np.isfinite(sal), sal + 1, 7.0)
    gs[1, 2] = 2.0                                             # pixel (0,0): 2024 hop le
    gn = np.full_like(gs, 0.2) if ndwi_gee is None else ndwi_gee.astype(np.float32)
    _write(gee / "2024_MD_dry_NDWIchen_Salinity_l8_v2.tif", [gn, gs], g_tr, ["NDWIchen", "Salinity"])
    for y in (2023, 2024):
        wf = np.zeros((n + 2, n + 3), np.uint8)
        ncl = np.full_like(wf, 5)
        wf[1 + 5, 2 + 1] = 60                                  # pixel (5,1) nuoc dong
        wf[1 + 6, 2 + 1] = 255                                 # pixel (6,1) khong quan sat -> CHG-08 NaN
        ncl[1 + 6, 2 + 1] = 0
        ncl[1 + 9, 2 + 1] = 1                                  # pixel (9,1) 1 lan chup -> CHG-08 NaN
        ncl[1 + 10, 2 + 1] = 0                                 # pixel (10,1) 0 lan chup -> NaN
        ncl[1 + 9, 2 + 2] = 2                                  # pixel (9,2) 2 lan chup -> giu
        _write(gee / f"{y}_MD_dry_watermask_l8_v2.tif", [wf, ncl], g_tr, ["water_freq", "n_clear"], dtype="uint8")
    wc = np.full((n * 3, n * 3), 40, np.uint8)
    for (r, c), cls in {(2, 1): 80, (3, 1): 95, (4, 1): 60, (7, 1): 90, (8, 1): 50}.items():
        wc[3 * r:3 * r + 3, 3 * c:3 * c + 3] = cls
    _write(tmp_path / "wc.tif", [wc], from_origin(X0, Y0, 10, 10), ["Map"], dtype="uint8")
    bnd = gpd.GeoDataFrame(geometry=[box(X0 - 1, Y0 - n * 30 - 1, X0 + n * 30 + 1, Y0 + 1)], crs=CRS)
    bnd.to_crs(4326).to_file(tmp_path / "bnd.geojson", driver="GeoJSON")
    res = build_scope_mask(bnd.geometry.iloc[0], str(tmp_path / "wc.tif"),
                           [str(zen / "2023_MD_dry_NDWIchen_Salinity.tif")])   # mac dinh v3 {10,20,30,40}
    write_scope_mask(str(tmp_path / "scope.tif"), res)
    d = np.full((10, 10), 50.0, np.float32)
    d[:, :2] = 5.0                                             # cot trai (x < X0 + 180) ven bien
    _write(tmp_path / "dist.tif", [d, d, d], from_origin(X0, Y0, 90, 90),
           ["dist_main_river_km", "dist_any_water_km", "dist_coast_km"])
    src_pts = gpd.GeoDataFrame(
        {"point_id": ["p_mid", "p_edge", "p_60", "p_50", "p_95"], "block_id": ["b1", "b1", "b2", "b2", "b3"],
         "is_holdout": [False, True, True, False, False]},
        geometry=[_px_point(4, 4), Point(X0 + 15, Y0 - 15), _px_point(4, 1), _px_point(8, 1), _px_point(3, 1)],
        crs=CRS).to_crs(4326)
    src_pts.to_file(tmp_path / "pts_src.geojson", driver="GeoJSON")
    src_pts.iloc[:2].to_file(tmp_path / "pts.geojson", driver="GeoJSON")           # bo chinh (da loc v3)
    gdir = tmp_path / "grids"
    gdir.mkdir()
    cells = gpd.GeoDataFrame({"cell_id": ["L", "R"]},
                             geometry=[box(X0, Y0 - n * 30, X0 + n * 15, Y0), box(X0 + n * 15, Y0 - n * 30,
                                                                                    X0 + n * 30, Y0)], crs=CRS)
    cells.to_crs(4326).to_file(gdir / "toy.geojson", driver="GeoJSON")

    cmd = [sys.executable, os.path.join(ROOT, "scripts", "build_labels_season.py"), "--zenodo-dir", str(zen),
           "--gee-dir", str(gee), "--scope", str(tmp_path / "scope.tif"), "--worldcover", str(tmp_path / "wc.tif"),
           "--dist-raster", str(tmp_path / "dist.tif"), "--boundary", str(tmp_path / "bnd.geojson"),
           "--points", str(tmp_path / "pts.geojson"), "--points-source", str(tmp_path / "pts_src.geojson"),
           "--grids-dir", str(gdir), "--years", "2023", "2024",
           "--out-dir", str(out), "--work-dir", str(work), "--report-dir", str(rep)]
    return cmd, out, work, rep


def test_script_end_to_end(tmp_path):
    cmd, out, work, rep = _toy_e2e(tmp_path)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr

    main = pd.read_csv(out / "toy_labels_season.csv", dtype={"cell_id": str})
    assert list(main.columns) == ["cell_id", "season", "salinity", "n_valid_px", "valid_frac", "train_ok",
                                  "train_ok_10pct"]
    assert not main.duplicated(["cell_id", "season"]).any() and len(main) == 4
    m = main.set_index(["cell_id", "season"])
    # o trai 2023: 72 px, bo (0,0) NaN-rule, (2,1) WC80, (3,1) WC95, (4,1) WC60, (7,1) WC90, (8,1) WC50,
    # (5,1) nuoc dong, CHG-08: (6,1) wf 255, (9,1) n_clear 1, (10,1) n_clear 0 -> 62 px gia tri 1; (9,2) n_clear 2 giu
    assert m.loc[("L", 2023), "n_valid_px"] == pytest.approx(62, abs=0.05)
    assert m.loc[("L", 2023), "salinity"] == pytest.approx(1.0)
    # o phai 2023: 72 px tru (11,11) ngoai dau chan -> 71 px gia tri 3
    assert m.loc[("R", 2023), "n_valid_px"] == pytest.approx(71, abs=0.05)
    assert m.loc[("R", 2023), "salinity"] == pytest.approx(3.0)
    # 2024 (GEE lech luoi): o trai = 62 px gia tri 2 + pixel (0,0) = 2 -> 63 px; o phai gia tri 4, (11,11) bi cat
    assert m.loc[("L", 2024), "n_valid_px"] == pytest.approx(63, abs=0.05)
    assert m.loc[("L", 2024), "salinity"] == pytest.approx(2.0)
    assert m.loc[("R", 2024), "salinity"] == pytest.approx(4.0)
    assert m.loc[("R", 2024), "n_valid_px"] == pytest.approx(71, abs=0.05)
    assert m["train_ok"].eq(False).all()                       # < 100 px
    assert m["train_ok_10pct"].all()

    def nvp(sfx):
        t = pd.read_csv(out / f"toy_labels_season{sfx}.csv", dtype={"cell_id": str}).set_index(["cell_id", "season"])
        return t.loc[("L", 2023), "n_valid_px"]
    assert nvp("_keepwater") == pytest.approx(64, abs=0.05)      # + WC80 (2,1) + nuoc dong (5,1); (6,1) van NaN
    assert nvp("_keepmangrove") == pytest.approx(63, abs=0.05)   # + WC95
    assert nvp("_keep6090") == pytest.approx(64, abs=0.05)       # + WC60 + WC90; WC50 khong bao gio

    cols = ["point_id", "block_id", "is_holdout", "wc_class", "added", "season", "ref_salinity", "n_valid_3x3"]
    pr = pd.read_csv(out / "points_reference.csv", dtype={"point_id": str})
    assert list(pr.columns) == cols and sorted(pr["point_id"].unique()) == ["p_edge", "p_mid"]
    pr = pr.set_index(["point_id", "season"])
    assert pr.loc[("p_mid", 2023), "ref_salinity"] == pytest.approx(1.0)
    assert pr.loc[("p_mid", 2023), "n_valid_3x3"] == 9
    assert np.isnan(pr.loc[("p_edge", 2023), "ref_salinity"])                 # goc: 3/9 hop le (2023)
    assert pr.loc[("p_edge", 2023), "n_valid_3x3"] == 3
    assert pr.loc[("p_edge", 2024), "n_valid_3x3"] == 4                       # (0,0) hop le 2024, van < 5
    p6 = pd.read_csv(out / "points_reference_keep6090.csv", dtype={"point_id": str, "block_id": str})
    assert list(p6.columns) == cols and len(p6) == 3 * 2
    assert sorted(p6["point_id"].unique()) == ["p_60", "p_edge", "p_mid"]     # WC50, WC95 khong vao keep6090
    a60 = p6[p6["point_id"] == "p_60"].set_index("season")
    assert a60["added"].all() and (a60["block_id"] == "b2").all() and a60["is_holdout"].all()
    assert (a60["wc_class"] == 60).all()
    # cua so quanh (4,1) theo keep6090: bo (3,1) WC95 va (5,1) nuoc dong -> 7 pixel gia tri 1
    assert a60.loc[2023, "n_valid_3x3"] == 7 and a60.loc[2023, "ref_salinity"] == pytest.approx(1.0)
    assert not p6.loc[p6["point_id"] != "p_60", "added"].any()
    pm = pd.read_csv(out / "points_reference_keepmangrove.csv", dtype={"point_id": str})
    assert sorted(pm["point_id"].unique()) == ["p_95", "p_edge", "p_mid"]
    pw = pd.read_csv(out / "points_reference_keepwater.csv", dtype={"point_id": str})
    assert sorted(pw["point_id"].unique()) == ["p_edge", "p_mid"]
    for f in ("toy_labels_season.csv", "points_reference.csv", "points_reference_keep6090.csv"):
        prov = json.load(open(out / f"{f}.provenance.json", encoding="utf-8"))
        assert "2023_MD_dry_NDWIchen_Salinity.tif" in prov["labels_sha256"]
        assert prov["scope_sha256"] and prov["boundary_sha256"] and "EC1:5" in prov["unit"]
        assert prov["min_clear"] == 2 and prov["rules"]["rules_version"] == ls.RULES_VERSION

    ry = pd.read_csv(rep / "dot4_phan_bo_nhan_v3.csv").set_index(["season", "set"])
    assert ry.loc[(2023, "before"), "n_px"] == 144 - 2         # bo (0,0) NaN-rule va (11,11) ngoai dau chan
    assert ry.loc[(2023, "before"), "max"] == pytest.approx(3.0)
    assert ry.loc[(2023, "main"), "n_px"] == 62 + 71
    assert ry.loc[(2023, "v2_old"), "n_px"] == 142 - 3         # P5 cu: tru WC80, WC95, nuoc dong
    assert ry.loc[(2023, "v3_noclear"), "n_px"] == 139 - 3     # (a): tru WC 50, 60, 90
    assert ry.loc[(2023, "drop_lc50"), "n_px"] == 1 and ry.loc[(2023, "drop_lc6090"), "n_px"] == 2
    assert ry.loc[(2023, "drop_clear"), "n_px"] == 3           # (b): (6,1), (9,1), (10,1)
    assert ry.loc[(2023, "drop_clear_n0"), "n_px"] == 2 and ry.loc[(2023, "drop_clear_n1"), "n_px"] == 1
    assert ry.loc[(2023, "keep6090"), "n_px"] == 133 + 2
    assert ry.loc[(2023, "keep6090"), "n_points"] == 3 and ry.loc[(2023, "main"), "n_points"] == 2
    dec = pd.read_csv(rep / "dot4_phan_bo_nhan_v3_tach_tac_dong.csv").set_index("season")
    assert dec.loc[2023, "delta_a_n_px"] == -3 and dec.loc[2023, "delta_b_n_px"] == -3
    assert dec.loc[2023, "delta_b_truoc_n_px"] == -3 and dec.loc[2023, "delta_a_sau_n_px"] == -3
    rg = pd.read_csv(rep / "dot4_phan_bo_nhan_v3_luoi.csv")
    assert set(rg["variant"]) == {"main", "keepwater", "keepmangrove", "keep6090"}
    g = rg[(rg["variant"] == "main") & (rg["season"] == 2023)].iloc[0]
    assert g["n_cells"] == 2 and g["n_cells_coastal"] == 1 and g["n_train_ok"] == 0
    assert g["n_train_ok_10pct"] == 2 and g["n_train_ok_10pct_coastal"] == 1

    # chay lai: bo qua phan da co, ket qua khong doi
    r2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r2.returncode == 0 and "bo qua" in r2.stdout, r2.stdout + r2.stderr
    pd.testing.assert_frame_equal(pd.read_csv(out / "toy_labels_season.csv", dtype={"cell_id": str}),
                                  main)


def test_script_rejects_scope_v2(tmp_path):
    """Scope khong co band wc_class (ban v2) -> script dung, khong am tham dung quy tac cu."""
    n = 4
    tr = from_origin(X0, Y0, 30, 30)
    (tmp_path / "zen").mkdir()
    (tmp_path / "gee").mkdir()
    _write(tmp_path / "zen" / "2023_MD_dry_NDWIchen_Salinity.tif", [np.full((n, n), 0.2), np.ones((n, n))], tr,
           ["NDWIchen", "Salinity"])
    _write(tmp_path / "gee" / "2023_MD_dry_watermask_l8_v2.tif", [np.zeros((n, n)), np.full((n, n), 5)], tr,
           ["water_freq", "n_clear"], dtype="uint8")
    _write(tmp_path / "wc.tif", [np.full((3 * n, 3 * n), 40)], from_origin(X0, Y0, 10, 10), ["Map"], dtype="uint8")
    bnd = gpd.GeoDataFrame(geometry=[box(X0 - 1, Y0 - n * 30 - 1, X0 + n * 30 + 1, Y0 + 1)], crs=CRS)
    bnd.to_crs(4326).to_file(tmp_path / "bnd.geojson", driver="GeoJSON")
    res = build_scope_mask(bnd.geometry.iloc[0], str(tmp_path / "wc.tif"),
                           [str(tmp_path / "zen" / "2023_MD_dry_NDWIchen_Salinity.tif")], exclude=(0, 80, 95))
    write_scope_mask(str(tmp_path / "scope.tif"), res, with_wc_class=False)
    gdir = tmp_path / "grids"
    gdir.mkdir()
    gpd.GeoDataFrame({"cell_id": ["A"]}, geometry=[box(X0, Y0 - 60, X0 + 60, Y0)], crs=CRS).to_crs(4326).to_file(
        gdir / "toy.geojson", driver="GeoJSON")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "build_labels_season.py"), "--zenodo-dir",
           str(tmp_path / "zen"), "--gee-dir", str(tmp_path / "gee"), "--scope", str(tmp_path / "scope.tif"),
           "--worldcover", str(tmp_path / "wc.tif"), "--boundary", str(tmp_path / "bnd.geojson"),
           "--grids-dir", str(gdir), "--years", "2023", "2023", "--out-dir", str(tmp_path / "out"),
           "--work-dir", str(tmp_path / "work"), "--report-dir", str(tmp_path / "rep")]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode != 0 and "wc_class" in (r.stdout + r.stderr)


# ---------------------------------------------------------------- E5a Dot 7: NDWI cung raster nhan
def test_label_values_ndwi_cung_tap_pixel_voi_do_man():
    ndwi = np.array([0.2, np.nan, 0.2, 1.0, 0.3, -1.0, 0.2, -0.5], np.float32)
    sal = np.array([1.5, 1.5, 28.5, 1.5, 2.0, 1.5, np.nan, 27.9], np.float32)
    s = ls.label_values(ndwi, sal, "salinity")
    np.testing.assert_array_equal(s, ls.nan_rule(ndwi, sal))         # duong do man khong doi
    v = ls.label_values(ndwi, sal, "ndwi")
    assert v.dtype == np.float32
    np.testing.assert_array_equal(np.isfinite(v), np.isfinite(s))      # cung tap pixel hop le
    assert v[0] == pytest.approx(0.2) and v[4] == pytest.approx(0.3) and v[7] == pytest.approx(-0.5)
    assert not np.any(v[np.isnan(s)] == 0)                             # NaN khong bi dien 0
    with pytest.raises(ValueError):
        ls.label_values(ndwi, sal, "rain")


def test_script_ndwi_khong_doi_duong_do_man(tmp_path):
    """--target ndwi: cung mat na (n_valid_px, NaN diem giong do man), gia tri = NDWI; dau ra do man khong doi;
    raster cache trong work-dir dung chung KHONG bi dung lan giua hai bien."""
    n = 12
    nz = np.full((n, n), 0.1, np.float32)
    nz[:, n // 2:] = 0.3                                                # trai 0,1 / phai 0,3
    ng = np.full((n + 2, n + 3), 0.5, np.float32)
    ng[1:n + 1, 2:n + 2] = nz + 0.05                                    # 2024: trai 0,15 / phai 0,35
    cmd, out, work, rep = _toy_e2e(tmp_path, ndwi_zen=nz, ndwi_gee=ng)
    run = lambda c: subprocess.run(c, capture_output=True, text=True, encoding="utf-8", errors="replace")  # noqa: E731
    r = run(cmd)
    assert r.returncode == 0, r.stdout + r.stderr
    files = ["toy_labels_season.csv", "points_reference.csv", "toy_labels_season_keepwater.csv"]
    before = {f: (out / f).read_bytes() for f in files}
    rep_before = (rep / "dot4_phan_bo_nhan_v3.csv").read_bytes()

    out_n = tmp_path / "out_ndwi"
    r = run(cmd + ["--target", "ndwi", "--out-dir", str(out_n)])       # CUNG work-dir voi lan do man
    assert r.returncode == 0, r.stdout + r.stderr
    assert "bo qua" not in r.stdout                                     # khong dung lai raster do man
    for f in files:
        assert (out / f).read_bytes() == before[f]
    assert (rep / "dot4_phan_bo_nhan_v3.csv").read_bytes() == rep_before
    assert (rep / "dot7_e5_ndwi_phan_bo.csv").exists() and (rep / "dot7_e5_ndwi_luoi.csv").exists()
    assert sorted(p.name for p in out_n.glob("*.csv")) == ["points_reference_ndwi.csv", "toy_labels_season_ndwi.csv"]

    sal = pd.read_csv(out / "toy_labels_season.csv", dtype={"cell_id": str}).set_index(["cell_id", "season"])
    nd = pd.read_csv(out_n / "toy_labels_season_ndwi.csv", dtype={"cell_id": str})
    assert list(nd.columns) == ["cell_id", "season", "ndwi", "n_valid_px", "valid_frac", "train_ok", "train_ok_10pct"]
    nd = nd.set_index(["cell_id", "season"])
    pd.testing.assert_series_equal(nd["n_valid_px"], sal["n_valid_px"])
    pd.testing.assert_series_equal(nd["train_ok"], sal["train_ok"])
    assert nd.loc[("L", 2023), "ndwi"] == pytest.approx(0.1, abs=1e-6)
    assert nd.loc[("R", 2023), "ndwi"] == pytest.approx(0.3, abs=1e-6)
    assert nd.loc[("L", 2024), "ndwi"] == pytest.approx(0.15, abs=1e-6)
    assert nd.loc[("R", 2024), "ndwi"] == pytest.approx(0.35, abs=1e-6)

    ps = pd.read_csv(out / "points_reference.csv", dtype={"point_id": str}).set_index(["point_id", "season"])
    pn = pd.read_csv(out_n / "points_reference_ndwi.csv", dtype={"point_id": str})
    assert "ref_ndwi" in pn.columns and "ref_salinity" not in pn.columns
    pn = pn.set_index(["point_id", "season"])
    assert (pn["n_valid_3x3"] == ps["n_valid_3x3"]).all()
    assert (pn["ref_ndwi"].isna() == ps["ref_salinity"].isna()).all()
    assert pn.loc[("p_mid", 2023), "ref_ndwi"] == pytest.approx(0.1, abs=1e-6)
    prov = json.load(open(out_n / "points_reference_ndwi.csv.provenance.json", encoding="utf-8"))
    assert prov["target"] == "ndwi" and prov["variant"] == "main" and prov["ref_rule_kind"] == "3x3"
    assert prov["value_band"] == "NDWIchen" and "EC1:5" not in prov["unit"]
    tprov = json.load(open(out_n / "toy_labels_season_ndwi.csv.provenance.json", encoding="utf-8"))
    assert tprov["target"] == "ndwi"
    assert "target" not in json.load(open(out / "toy_labels_season.csv.provenance.json", encoding="utf-8"))

    # chay lai do man tren work-dir vua bi NDWI ghi: phai tinh lai raster (khong lay raster NDWI), ket qua khong doi
    r = run(cmd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "khac quy tac/tap diem - tinh lai" in r.stdout
    for f in files:
        assert (out / f).read_bytes() == before[f]

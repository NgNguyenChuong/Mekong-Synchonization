"""Chuoi thiet ke danh gia tren du lieu TONG HOP gia tri biet truoc (gop 2026-10-04 tu 4 file):
  - mat na pham vi tinh (M), gop khoi it diem thanh don vi (J), gan fold can bang 2 tieu chi (I, D);
  - thiet ke danh gia tren ranh gioi v2: giu khoi giu rieng, diem tren mat na dat (cu: test_eval_design_v2.py);
  - CHG-10: loc diem theo mat na pham vi v3, kiem lai J/I/D (cu: test_scope_v3_filter.py);
  - CHG-14: nguong TUYET DOI cho phuong an fold phu, pha hoa tai lap, sao luu khi --force, diem CV ven bien
    theo fold (cu: test_cv_folds_chg14.py).
"""
import json
import os
import subprocess
import sys
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_design import (  # noqa: E402
    assign_unit_folds,
    dist_coast_at_points,
    fold_balance_objective,
    fold_max_rel_dev,
    fold_overlap,
    fold_pair_stats,
    fold_schemes,
    mask_area_by_block,
    merge_small_blocks,
    pick_alternates,
    rank_pool,
    raster_values_at,
    refresh_blocks,
    sample_points_in_mask,
    scheme_code,
    scheme_conditions,
    scheme_rank_key,
    unit_folds_to_blocks,
    unit_pool,
)
from filter_eval_points_scope import coast_band  # noqa: E402
from scope_mask import (  # noqa: E402
    SCOPE_EXCLUDE_LC,
    SCOPE_INCLUDE_LC,
    build_scope_mask,
    compare_worldcover,
    scope_area_by_block,
    scope_change_by_class,
    scope_values_at_points,
    write_scope_mask,
)

UTM = "EPSG:32648"
X0, Y1 = 500000.0, 1200000.0


def _write(path, arrays, res, names, dtype, x0=X0, y1=Y1):
    arrays = [np.asarray(a, dtype) for a in arrays]
    h, w = arrays[0].shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=len(arrays), dtype=dtype, crs=UTM,
                       transform=from_origin(x0, y1, res, res)) as ds:
        for i, (a, n) in enumerate(zip(arrays, names), start=1):
            ds.write(a, i)
            ds.set_band_description(i, n)
    return str(path)


@pytest.fixture()
def scope_inputs(tmp_path):
    """Luoi nhan 4 x 4 pixel 30 m. WorldCover 10 m 12 x 12 (moi pixel 30 m = 3 x 3 pixel 10 m).

    - Nhan: band 1 NDWIchen HUU HAN moi noi (bay doc nham band), band 2 Salinity:
        nam A: NaN o cot 3; nam B: NaN o cot 3 va hang 0 -> pixel (0, c<3) huu han 1 nam, cot 3 0 nam.
    - WorldCover: tam pixel 30 m (0,0) = 80 nhung 8 pixel 10 m xung quanh = 40 -> loai (theo TAM);
      pixel (0,1): tam 40, xung quanh 80 -> giu; pixel (1,1) tam 95 -> loai; pixel (1,2) tam 0 -> loai.
    - Ranh gioi: bo hang 3 (y < Y1 - 90).
    """
    nd = np.full((4, 4), 0.1)
    sa = np.full((4, 4), 1.0)
    sa_a = sa.copy()
    sa_a[:, 3] = np.nan
    sa_b = sa_a.copy()
    sa_b[0, :] = np.nan
    la = _write(tmp_path / "a.tif", [nd, sa_a], 30, ["NDWIchen", "Salinity"], "float32")
    lb = _write(tmp_path / "b.tif", [nd, sa_b], 30, ["NDWIchen", "Salinity"], "float32")
    wc = np.full((12, 12), 40)
    wc[1, 1] = 80                          # tam pixel 30 m (0,0)
    wc[0:3, 3:6] = 80
    wc[1, 4] = 40                          # tam pixel (0,1) la dat, xung quanh nuoc
    wc[4, 4] = 95                          # tam pixel (1,1)
    wc[4, 7] = 0                           # tam pixel (1,2)
    wcp = _write(tmp_path / "wc.tif", [wc], 10, ["Map"], "uint8")
    boundary = box(X0, Y1 - 90, X0 + 120, Y1)
    return la, lb, wcp, boundary


def test_scope_mask_gia_tri_biet_truoc(scope_inputs, tmp_path):
    """Quy tac v2 (truoc CHG-10: bo 0/80/95), dinh dang 2 band - chay lai v2 phai giu nguyen."""
    la, lb, wcp, boundary = scope_inputs
    r = build_scope_mask(boundary, wcp, [la, lb], exclude=SCOPE_EXCLUDE_LC)
    want_ny = np.array([[1, 1, 1, 0], [2, 2, 2, 0], [2, 2, 2, 0], [2, 2, 2, 0]])
    assert r["zenodo_n_years"].tolist() == want_ny.tolist()   # doc Salinity theo TEN, khong band 1
    want = np.array([[0, 1, 1, 0],      # (0,0) tam WC 80; (0,1) tam 40 du xung quanh 80; cot 3 khong nhan
                     [1, 0, 0, 0],      # (1,1) WC 95, (1,2) WC 0
                     [1, 1, 1, 0],
                     [0, 0, 0, 0]])     # ngoai ranh gioi
    assert r["scope"].tolist() == want.tolist()
    out = tmp_path / "scope.tif"
    write_scope_mask(str(out), r, with_wc_class=False)
    with rasterio.open(out) as s:
        assert s.descriptions == ("scope", "zenodo_n_years")
        assert s.read(1).tolist() == want.tolist()
        assert s.nodata is None


@pytest.fixture()
def scope_v3_inputs(tmp_path):
    """Luoi nhan 3 x 4 pixel 30 m, Salinity huu han moi noi (1 nam). WorldCover 10 m: moi pixel 30 m mang MOT lop
    o TAM, 8 pixel 10 m xung quanh mang lop KHAC (bay: phai lay theo tam, khong theo da so):
        hang 0: 10, 20, 30, 40     -> giu ca 4 (xung quanh 50)
        hang 1: 50, 60, 90, 95     -> loai ca 4 (xung quanh 40)
        hang 2: 80, 0, 100, 70     -> loai ca 4 (xung quanh 10)
    Ranh gioi phu het."""
    nd = np.full((3, 4), 0.1)
    la = _write(tmp_path / "a.tif", [nd, np.ones((3, 4))], 30, ["NDWIchen", "Salinity"], "float32")
    centers = np.array([[10, 20, 30, 40], [50, 60, 90, 95], [80, 0, 100, 70]])
    around = np.array([[50] * 4, [40] * 4, [10] * 4])
    wc = np.kron(around, np.ones((3, 3), int))
    wc[1::3, 1::3] = centers
    wcp = _write(tmp_path / "wc.tif", [wc], 10, ["Map"], "uint8")
    return la, wcp, centers, box(X0, Y1 - 90, X0 + 120, Y1)


def test_scope_v3_chi_giu_wc_10_20_30_40(scope_v3_inputs, tmp_path):
    la, wcp, centers, boundary = scope_v3_inputs
    r = build_scope_mask(boundary, wcp, [la])                      # mac dinh = v3 (CHG-10)
    assert SCOPE_INCLUDE_LC == (10, 20, 30, 40)
    assert r["include"] == (10, 20, 30, 40) and r["exclude"] is None
    assert r["scope"].tolist() == [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert r["wc_class"].tolist() == centers.tolist()              # lop tai TAM pixel
    out = tmp_path / "scope_v3.tif"
    write_scope_mask(str(out), r)
    with rasterio.open(out) as s:
        assert s.descriptions == ("scope", "zenodo_n_years", "wc_class")
        assert s.read(1).tolist() == r["scope"].tolist()           # band 1 van la scope (code doc band=1)
        assert s.read(2).tolist() == [[1] * 4] * 3
        assert s.read(3).tolist() == centers.tolist()
        assert "10,20,30,40" in s.tags()["scope"]
    # cung du lieu, quy tac v2 -> 50, 60, 90, 100, 70 duoc giu (khac v3 dung 5 pixel)
    r2 = build_scope_mask(boundary, wcp, [la], exclude=SCOPE_EXCLUDE_LC)
    assert r2["scope"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 1, 1]]
    ch = scope_change_by_class(r2["scope"], r).set_index("wc_class")
    assert ch["lost_px"].to_dict() == {10: 0, 20: 0, 30: 0, 40: 0, 50: 1, 60: 1, 70: 1, 90: 1, 100: 1}
    assert ch["gained_px"].sum() == 0
    with pytest.raises(ValueError, match="MOT"):
        build_scope_mask(boundary, wcp, [la], include=(10,), exclude=(80,))


def test_scope_v3_wc_class_0_ngoai_ranh_gioi(scope_v3_inputs):
    la, wcp, centers, _ = scope_v3_inputs
    r = build_scope_mask(box(X0, Y1 - 30, X0 + 120, Y1), wcp, [la])  # chi hang 0 trong ranh gioi
    assert r["wc_class"].tolist() == [centers[0].tolist(), [0] * 4, [0] * 4]
    assert r["scope"].tolist() == [[1, 1, 1, 1], [0] * 4, [0] * 4]


def test_compare_worldcover_lech_nguyen_pixel(tmp_path):
    # ban cu 4 x 4 goc (X0, Y1); ban moi 6 x 6 goc lech 1 cot tay, 2 hang bac; trung gia tri tru 1 pixel
    old = np.arange(16).reshape(4, 4) % 5 * 10 + 10
    new = np.full((6, 6), 80)
    new[2:6, 1:5] = old
    new[2 + 3, 1 + 0] = 95                                         # pixel cu (3,0) khac
    op = _write(tmp_path / "old.tif", [old], 10, ["Map"], "uint8")
    npth = _write(tmp_path / "new.tif", [new], 10, ["Map"], "uint8", x0=X0 - 10, y1=Y1 + 20)
    c = compare_worldcover(npth, op, box(X0, Y1 - 40, X0 + 40, Y1))
    assert c["offset"] == (1, 2)
    assert c["n_px_in_boundary"] == 16 and c["n_diff"] == 1 and c["n_outside_new"] == 0
    assert c["diff_pairs"] == {(int(old[3, 0]), 95): 1}
    bad = _write(tmp_path / "bad.tif", [new], 10, ["Map"], "uint8", x0=X0 - 5)
    with pytest.raises(ValueError, match="nguyen"):
        compare_worldcover(bad, op, box(X0, Y1 - 40, X0 + 40, Y1))


def test_scope_mask_luoi_nhan_khac_nhau_bao_loi(scope_inputs, tmp_path):
    la, _, wcp, boundary = scope_inputs
    other = _write(tmp_path / "c.tif", [np.ones((4, 4)), np.ones((4, 4))], 30, ["NDWIchen", "Salinity"],
                   "float32", x0=X0 + 30)
    with pytest.raises(ValueError, match="khac luoi"):
        build_scope_mask(boundary, wcp, [la, other])
    nos = _write(tmp_path / "d.tif", [np.ones((4, 4))], 30, ["NDWIchen"], "float32")
    with pytest.raises(ValueError, match="Salinity"):
        build_scope_mask(boundary, wcp, [nos])


def test_scope_area_by_block_va_ven_bien(tmp_path):
    # scope 4 x 4 pixel 30 m, 1 o moi noi tru hang 0 cot 0; khoi trai = cot 0-1, khoi phai = cot 2-3.
    sc = np.ones((4, 4))
    sc[0, 0] = 0
    sp = _write(tmp_path / "s.tif", [sc, np.full((4, 4), 9)], 30, ["scope", "zenodo_n_years"], "uint8")
    # dist_coast 90 m (1 pixel phu het 120 x 120 m? -> dung 60 m de co 2 x 2): cot trai 20.0 (= nguong,
    # TINH), phai-tren 20.01 (khong), phai-duoi NaN (khong tinh, dem rieng)
    dc = np.array([[20.0, 20.01], [20.0, np.nan]])
    dp = _write(tmp_path / "dc.tif", [dc, dc, dc], 60, ["dist_main_river_km", "dist_any_water_km", "dist_coast_km"],
                "float32")
    blocks = gpd.GeoDataFrame({"block_id": ["L", "R"]},
                              geometry=[box(X0, Y1 - 120, X0 + 60, Y1), box(X0 + 60, Y1 - 120, X0 + 120, Y1)],
                              crs=UTM)
    a = scope_area_by_block(blocks, sp, dp, coast_km=20.0).set_index("block_id")
    px = 0.0009
    assert a.loc["L", "scope_km2"] == pytest.approx(7 * px)
    assert a.loc["R", "scope_km2"] == pytest.approx(8 * px)
    assert a.loc["L", "coast_scope_km2"] == pytest.approx(7 * px)   # 20.0 <= 20 tinh
    assert a.loc["R", "coast_scope_km2"] == pytest.approx(0.0)      # 20.01 va NaN khong tinh
    assert a.loc["R", "coast_nan_km2"] == pytest.approx(4 * px)


def _tab(spec):
    """spec: [(i, j, n_points, is_holdout)] -> bang khoi."""
    return pd.DataFrame([{"block_id": f"blk_{i}_{j}", "is_holdout": h, "land_km2": 100.0, "n_points": n}
                         for i, j, n, h in spec])


def test_gop_khoi_vao_ke_chung_canh_nhieu_diem_nhat():
    #   j=2:  [0_2: 50]  [1_2: 5]   [2_2: 200 GIU RIENG]
    #   j=1:  [0_1: 80]  [1_1: 40]  [2_1: 300]
    #   j=0:                         [2_0: 3]  ; [4_4: 2] co lap (chi gan nhat)
    t = _tab([(0, 2, 50, False), (1, 2, 5, False), (2, 2, 200, True), (0, 1, 80, False), (1, 1, 40, False),
              (2, 1, 300, False), (2, 0, 3, False), (4, 4, 2, False), (3, 3, 0, True)])
    t.loc[t["block_id"] == "blk_3_3", "land_km2"] = 0.0
    df, log = merge_small_blocks(t, min_points=30)
    u = df.set_index("block_id")["unit_id"]
    # blk_1_2 (5): ke chung canh 0_2 (50), 1_1 (40); 2_2 la GIU RIENG -> khong gop; chon 0_2 (nhieu diem nhat)
    assert u["blk_1_2"] == "blk_0_2"
    assert u["blk_2_0"] == "blk_2_1"          # chi ke 2_1 (1_0 khong ton tai)
    # blk_4_4 (2): khong ke chung canh (3_4, 4_3 khong co; 3_3 khong dat) -> gan nhat Chebyshev (2_2 giu
    # rieng bi bo): khoang cach 3 = {1_2 (5), 1_1 (40), 2_1 (300)} -> 2_1 (nhieu diem nhat)
    assert log.set_index("from_unit").loc["blk_4_4", "rule"] == "gan_nhat"
    assert u["blk_4_4"] == "blk_2_1"
    assert u["blk_2_2"] == "blk_2_2" and u["blk_3_3"] == "blk_3_3"   # giu rieng / khong dat khong gop
    assert u["blk_1_1"] == "blk_1_1" and u["blk_0_1"] == "blk_0_1"  # >= 30 khong gop
    pts = df.groupby("unit_id")["n_points"].sum()
    cv_units = df.loc[~df["is_holdout"] & (df["land_km2"] > 0), "unit_id"].unique()
    assert (pts[cv_units] >= 30).all()
    # thu tu: it diem nhat truoc (blk_4_4: 2, roi blk_2_0: 3, roi blk_1_2: 5)
    assert log["from_unit"].tolist()[:3] == ["blk_4_4", "blk_2_0", "blk_1_2"]


def test_gop_khong_tinh_cham_goc_va_hoa_chon_id_nho():
    # blk_1_1 (10) cham goc blk_0_0 (500) nhung chung canh blk_1_0 (60) va blk_0_1 (60) -> hoa -> blk_0_1
    t = _tab([(1, 1, 10, False), (0, 0, 500, False), (1, 0, 60, False), (0, 1, 60, False)])
    df, log = merge_small_blocks(t, min_points=30)
    assert df.set_index("block_id").loc["blk_1_1", "unit_id"] == "blk_0_1"
    assert log["rule"].tolist() == ["chung_canh"]


def test_gop_day_chuyen_don_vi_nho_nhan_them():
    # blk_0_0 (1) -> ke duy nhat blk_1_0 (2) -> don vi (3) van < 30 -> gop tiep vao blk_2_0 (100)
    t = _tab([(0, 0, 1, False), (1, 0, 2, False), (2, 0, 100, False)])
    df, log = merge_small_blocks(t, min_points=30)
    assert set(df["unit_id"]) == {"blk_2_0"}
    assert len(log) == 2


@pytest.fixture()
def units():
    rng = np.random.default_rng(0)
    n = 24
    return pd.DataFrame({"unit_id": [f"blk_{i}_0" for i in range(n)],
                         "scope_km2": rng.uniform(200, 2500, n).round(1),
                         "coast_scope_km2": np.where(rng.random(n) < 0.4, rng.uniform(100, 1500, n), 0.0).round(1)})


def test_gan_fold_can_bang_hai_tieu_chi_va_tat_dinh(units):
    f1 = assign_unit_folds(units, n_folds=5, seed=42, n_starts=30)
    f2 = assign_unit_folds(units, n_folds=5, seed=42, n_starts=30)
    pd.testing.assert_frame_equal(f1, f2)
    tot = f1.groupby("cv_fold")[["scope_km2", "coast_scope_km2"]].sum()
    assert sorted(tot.index) == [0, 1, 2, 3, 4]
    obj = fold_balance_objective(tot.to_numpy())
    assert obj == pytest.approx(f1.attrs["objective"])
    # dien tich pham vi gan nhu deu (ven bien chi ~10 don vi khac 0 -> chenh lon hon, bi gioi han boi do thoi)
    assert (tot["scope_km2"].max() - tot["scope_km2"].min()) / tot["scope_km2"].mean() < 0.03
    # so voi gan ngau nhien: chenh ven bien phai nho hon nhieu
    rnd = np.random.default_rng(1).permutation(np.arange(len(units)) % 5)
    t_rnd = units.assign(k=rnd).groupby("k")[["scope_km2", "coast_scope_km2"]].sum().to_numpy()
    assert obj < fold_balance_objective(t_rnd)
    # nhan fold danh theo unit_id nho nhat trong fold
    first = f1.groupby("cv_fold")["unit_id"].min()
    assert first.is_monotonic_increasing


def test_fold_overlap_hungarian_va_rand_ari():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert fold_overlap(a, np.array([2, 2, 0, 0, 1, 1])) == pytest.approx(1.0)   # chi doi nhan -> trung 100%
    assert fold_overlap(a, np.array([1, 1, 0, 2, 2, 0])) == pytest.approx(4 / 6)  # ghep 0->1, 1->0, 2->2: 4 giu
    st = fold_pair_stats(a, np.array([2, 2, 0, 0, 1, 1]))
    assert st["rand"] == pytest.approx(1.0) and st["ari"] == pytest.approx(1.0)
    assert fold_max_rel_dev(a, [1, 1, 2, 2, 3, 3], 3) == pytest.approx(0.5)      # tong 2,4,6; TB 4


@pytest.fixture()
def units_coast():
    rng = np.random.default_rng(5)
    n = 20
    return pd.DataFrame({"unit_id": [f"blk_{i}_0" for i in range(n)],
                         "scope_km2": rng.uniform(300, 2500, n).round(1),
                         "coast_scope_km2": rng.uniform(100, 700, n).round(1)})


def test_fold_schemes_rang_buoc_D(units_coast):
    kw = dict(n_folds=5, n_starts=60, pool_seed=0, tols=(0.10, 0.15, 0.5), max_overlap=0.60, coast_tol=0.15)
    picked, pool, info = fold_schemes(units_coast, [42, 43, 44], **kw)
    again, _, _ = fold_schemes(units_coast, [42, 43, 44], **kw)
    best = info["best_objective"]
    assert picked[42].attrs["rank"] == 1 and picked[42].attrs["objective"] == pytest.approx(best)
    assert pool["objective"].min() == pytest.approx(best)
    tol = info["tol_used"]
    assert tol in (0.10, 0.15, 0.5)
    lab = {s: picked[s].sort_values("unit_id")["cv_fold"].to_numpy() for s in picked}
    coast = picked[42].sort_values("unit_id")["coast_scope_km2"].to_numpy()
    for s in (42, 43, 44):
        pd.testing.assert_frame_equal(picked[s], again[s])                       # tat dinh
        assert fold_max_rel_dev(lab[s], coast, 5) <= 0.15 + 1e-12                 # ven bien +-15% moi phuong an
        assert picked[s].attrs["objective"] <= best * (1 + tol) + 1e-12
    for x, y in ((42, 43), (42, 44), (43, 44)):
        assert fold_overlap(lab[x], lab[y]) <= 0.60 + 1e-12                       # do trung <= 60% tung cap
    assert picked[43].attrs["objective"] <= picked[44].attrs["objective"]        # tie-break: muc tieu thap truoc
    assert (pool["chon"] != "").sum() == 3


def test_fold_schemes_noi_nguong_muc_tieu_khong_noi_do_trung(units_coast):
    # tol 0,0 -> chi phuong an tot nhat -> khong co phuong an phu -> phai sang tol thu hai
    _, _, info = fold_schemes(units_coast, [42, 43, 44], n_folds=5, n_starts=60, pool_seed=0, tols=(0.0, 1.0))
    assert info["tol_used"] == 1.0
    with pytest.raises(ValueError, match="Khong du"):                            # do trung 0 khong the -> loi
        fold_schemes(units_coast, [42, 43, 44], n_folds=5, n_starts=60, pool_seed=0, tols=(0.10, 0.15),
                     max_overlap=0.0)
    with pytest.raises(ValueError, match="ven bien"):                            # chinh vi pham -> loi, khong noi
        fold_schemes(units_coast, [42, 43, 44], n_folds=5, n_starts=20, pool_seed=0, coast_tol=0.0)


def test_fold_balance_objective_gia_tri():
    tot = np.array([[10.0, 0.0], [20.0, 4.0]])   # (20-10)/15 + (4-0)/2
    assert fold_balance_objective(tot) == pytest.approx(10 / 15 + 2.0)


def test_unit_folds_to_blocks_giu_rieng_va_cung_fold_trong_don_vi():
    t = _tab([(0, 0, 1, False), (1, 0, 100, False), (2, 0, 100, False), (3, 0, 100, False), (4, 0, 50, True)])
    df, _ = merge_small_blocks(t, min_points=30)
    uf = pd.DataFrame({"unit_id": ["blk_1_0", "blk_2_0", "blk_3_0"], "cv_fold": [0, 1, 2]})
    b = unit_folds_to_blocks(df, uf).set_index("block_id")
    assert b.loc["blk_0_0", "cv_fold"] == b.loc["blk_1_0", "cv_fold"] == 0
    assert b.loc["blk_4_0", "cv_fold"] == -1
    bad = uf.copy()
    bad.loc[len(bad)] = ["blk_4_0", 1]
    with pytest.raises(ValueError, match="giu rieng"):
        unit_folds_to_blocks(df, bad)


# ---------- thiet ke danh gia v2 (cu: test_eval_design_v2.py)

@pytest.fixture()
def lc_raster(tmp_path):
    """WorldCover gia 1 km/pixel, 100 x 100 km tu (500000, 1100000): nua trai = 80 (nuoc),
    nua phai = 40 (dat), rieng cot cuoi = 95 (ngap man)."""
    data = np.full((100, 100), 40, "uint8")
    data[:, :50] = 80
    data[:, 99] = 95
    path = tmp_path / "lc.tif"
    with rasterio.open(path, "w", driver="GTiff", width=100, height=100, count=1, dtype="uint8",
                       crs=UTM, transform=from_origin(500000, 1200000, 1000, 1000)) as ds:
        ds.write(data, 1)
    return str(path)


def _gdf(geoms, crs=UTM):
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def test_raster_values_at_known_pixels_and_fill(lc_raster):
    xs = np.array([500500, 560500, 599500, 700000])
    ys = np.array([1150000, 1150000, 1100500, 1150000])
    assert raster_values_at(lc_raster, xs, ys, fill=0).tolist() == [80, 40, 95, 0]


def test_points_only_on_mask_and_uniform(lc_raster):
    region = box(500000, 1100000, 600000, 1200000)

    def accept(x, y):
        return ~np.isin(raster_values_at(lc_raster, x, y, fill=0), [0, 80, 95])

    pts = sample_points_in_mask(region, UTM, 2000, seed=42, accept=accept).to_crs(UTM)
    x = pts.geometry.x.to_numpy()
    assert len(pts) == 2000
    assert (x >= 550000).all() and (x < 599000).all()  # khong diem nao tren nuoc / ngap man
    # Deu theo dien tich: hai nua cua phan dat (550-574,5 km va 574,5-599 km) ~ 50/50.
    assert abs((x < 574500).mean() - 0.5) < 0.05
    again = sample_points_in_mask(region, UTM, 2000, seed=42, accept=accept).to_crs(UTM)
    assert np.allclose(again.geometry.x, pts.geometry.x)  # tat dinh theo seed


def test_mask_area_by_block_counts_land_only(lc_raster):
    boundary = _gdf([box(500000, 1100000, 600000, 1200000)]).to_crs(4326)
    blocks = gpd.GeoDataFrame({"block_id": ["w", "e"], "is_holdout": [False, True]},
                              geometry=[box(500000, 1100000, 550000, 1200000), box(550000, 1100000, 600000, 1200000)],
                              crs=UTM).to_crs(4326)
    km2 = mask_area_by_block(blocks, boundary, lc_raster)
    assert km2["w"] == pytest.approx(0.0)          # toan nuoc
    assert km2["e"] == pytest.approx(49 * 100.0)   # 49 cot dat x 100 hang (cot 95 bi loai)


def test_refresh_blocks_keeps_ids_and_holdout_and_adds_touched_block():
    old = gpd.GeoDataFrame(
        {"block_id": ["blk_10_22", "blk_11_22"], "land_km2": [1.0, 2.0], "is_holdout": [True, False]},
        geometry=[box(500000, 1100000, 550000, 1150000), box(550000, 1100000, 600000, 1150000)], crs=UTM,
    ).to_crs(4326)
    # Ranh gioi moi: phu 2 khoi cu + lan 3 km sang khoi phia bac (blk_10_23) chua co trong file.
    boundary = _gdf([box(510000, 1100000, 590000, 1153000)]).to_crs(4326)
    blocks, added = refresh_blocks(old, boundary, 50.0)
    assert added == ["blk_10_23", "blk_11_23"]
    b = blocks.set_index("block_id")
    assert bool(b.loc["blk_10_22", "is_holdout"]) and not bool(b.loc["blk_11_22", "is_holdout"])
    assert not b.loc[added, "is_holdout"].any()
    assert b.loc["blk_10_22", "land_km2"] == pytest.approx(40 * 50, rel=1e-3)
    assert b.loc["blk_10_23", "land_km2"] == pytest.approx(40 * 3, rel=1e-3)


# ---------- CHG-10 loc diem scope v3 (cu: test_scope_v3_filter.py)

def test_coast_band_canh_tren_tinh_vao_dai_duoi_va_nan():
    got = coast_band([0.0, 5.0, 5.0001, 20.0, 20.01, np.nan]).tolist()
    assert got == ["<=5", "<=5", "5-20", "5-20", ">20", "nan"]


@pytest.fixture()
def scope_files(tmp_path):
    """scope v3 1 x 6 pixel 30 m, lop tam pixel: 10, 20, 30, 40 (scope 1) | 50, 60 (scope 0); them 1 pixel 90.
    scope v2 (2 band): 1 o moi pixel. dist_coast 90 m: pixel trai 3 km, pixel phai 25 km."""
    cls = np.array([[10, 20, 30, 40, 50, 60, 90]])
    sc3 = np.isin(cls, [10, 20, 30, 40]).astype(int)
    v3 = _write(tmp_path / "scope_v3.tif", [sc3, np.full_like(cls, 10), cls], 30,
                ["scope", "zenodo_n_years", "wc_class"], "uint8")
    v2 = _write(tmp_path / "scope_v2.tif", [np.ones_like(cls), np.full_like(cls, 10)], 30,
                ["scope", "zenodo_n_years"], "uint8")
    d = np.array([[3.0, 3.0, 25.0]])
    dist = _write(tmp_path / "dist.tif", [d, d, d], 90, ["dist_main_river_km", "dist_any_water_km", "dist_coast_km"],
                  "float32")
    return v3, v2, dist, cls


def _points(tmp_path):
    """Moi pixel mot diem (lech khoi tam de bat loi lam tron), xen ke giu rieng / CV, thu tu khong theo id."""
    xs = [X0 + 30 * c + 7.3 for c in range(7)]
    pts = gpd.GeoDataFrame({"point_id": [f"pt_{i:05d}" for i in (6, 0, 1, 2, 3, 4, 5)],
                            "block_id": ["blk_0_0"] * 4 + ["blk_0_1"] * 3,
                            "is_holdout": [True, False, True, False, True, False, False]},
                           geometry=[Point(x, Y1 - 21.9) for x in xs], crs=UTM).to_crs(4326)
    path = tmp_path / "eval" / "eval_points.geojson"
    path.parent.mkdir()
    pts.to_file(path, driver="GeoJSON")
    with open(str(path) + ".provenance.json", "w", encoding="utf-8") as f:
        json.dump({"output": "eval_points.geojson", "seed_holdout": 42, "seed_cv": 43}, f)
    return pts, path


def test_scope_values_at_points_lay_pixel_chua_diem(scope_files, tmp_path):
    v3, _, _, cls = scope_files
    pts, _ = _points(tmp_path)
    v = scope_values_at_points(pts, v3)
    assert v["wc_class"].tolist() == cls[0].tolist()
    assert v["scope"].tolist() == [1, 1, 1, 1, 0, 0, 0]
    out = gpd.GeoDataFrame(geometry=[Point(X0 - 100, Y1 - 10)], crs=UTM)   # ngoai khung -> 0
    assert scope_values_at_points(out, v3)["scope"].tolist() == [0]


def test_filter_script_chi_loc_khong_sinh_lai_va_chay_lai_on_dinh(scope_files, tmp_path):
    v3, v2, dist, _ = scope_files
    pts, path = _points(tmp_path)
    bnd = tmp_path / "bnd.geojson"
    gpd.GeoDataFrame(geometry=[box(X0, Y1 - 30, X0 + 210, Y1)], crs=UTM).to_crs(4326).to_file(bnd, driver="GeoJSON")
    rep = tmp_path / "rep"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "filter_eval_points_scope.py"), "--points", str(path),
           "--backup-dir", str(tmp_path / "eval" / "deprecated_v2b"), "--scope", v3, "--compare-scope", v2,
           "--dist-raster", dist, "--boundary", str(bnd), "--report-dir", str(rep)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    out = gpd.read_file(path)
    # giu dung 4 diem lop 10/20/30/40, id/thu tu/toa do/khoi/giu rieng khong doi
    assert out["point_id"].tolist() == ["pt_00006", "pt_00000", "pt_00001", "pt_00002"]
    assert np.array_equal(out.geometry.x.to_numpy(), pts.geometry.x.to_numpy()[:4])
    assert out["is_holdout"].astype(bool).tolist() == [True, False, True, False]
    assert os.path.exists(tmp_path / "eval" / "deprecated_v2b" / "eval_points.geojson")
    prov = json.load(open(str(path) + ".provenance.json", encoding="utf-8"))
    assert prov["n_before"] == 7 and prov["n_after"] == 4 and prov["n_holdout"] == 2 and prov["n_cv"] == 2
    assert prov["dropped_by_wc_class"] == {"50": 1, "60": 1, "90": 1}
    assert prov["parent"]["seed_cv"] == 43
    by_cls = pd.read_csv(rep / "dot4_scope_v3_diem_loai_theo_lop.csv")
    row50 = by_cls[by_cls["wc_class"].astype(str) == "50"].iloc[0]
    assert (row50["giu_rieng"], row50["cv"]) == (1, 0)
    by_coast = pd.read_csv(rep / "dot4_scope_v3_diem_theo_ven_bien.csv")
    allc = by_coast[by_coast["nhom"] == "tat_ca"].set_index("coast_band")
    # pixel 30 m cot 0-5 nam trong pixel 90 m 0-1 (3 km), cot 6 trong pixel 90 m 2 (25 km)
    assert allc.loc["<=5", "n_truoc"] == 6 and allc.loc["<=5", "n_loai"] == 2
    assert allc.loc[">20", "n_truoc"] == 1 and allc.loc[">20", "n_loai"] == 1
    # chay lai: nguon = ban sao luu -> cung ket qua, khong loc chong / khong ghi de ban sao luu
    sha_bak = open(tmp_path / "eval" / "deprecated_v2b" / "eval_points.geojson", "rb").read()
    r2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert gpd.read_file(path)["point_id"].tolist() == out["point_id"].tolist()
    assert open(tmp_path / "eval" / "deprecated_v2b" / "eval_points.geojson", "rb").read() == sha_bak


def _units(scope, coast, pts):
    return pd.DataFrame({"unit_id": [f"u{i:02d}" for i in range(len(scope))], "scope_km2": scope,
                         "coast_scope_km2": coast, "n_points": pts})


def test_scheme_conditions_gia_tri_biet_truoc():
    # 4 don vi, 2 fold. A = {u00,u01 | u02,u03} can bang tuyet doi; B = {u00,u02 | u01,u03} lech.
    u = _units([10, 10, 10, 10], [5, 1, 5, 1], [40, 40, 40, 29])
    A = pd.Series({"u00": 0, "u01": 0, "u02": 1, "u03": 1})
    B = pd.Series({"u00": 0, "u02": 0, "u01": 1, "u03": 1})
    c = scheme_conditions(u, {42: A, 43: B}, best_objective=0.0 + 1e-9, n_folds=2, min_points=30, tol=0.10,
                          coast_tol=0.15, max_overlap=0.60).set_index("seed")
    assert c.loc[42, "objective"] == pytest.approx(0.0)
    assert c.loc[43, "objective"] == pytest.approx(8 / 6)              # dien tich 20/20 -> 0; ven bien (10-2)/6
    assert bool(c.loc[42, "ok_I_coast"]) and not bool(c.loc[43, "ok_I_coast"])   # 10 vs 2: lech 66,7%
    assert c.loc[43, "coast_max_dev"] == pytest.approx(4 / 6)
    assert not bool(c.loc[42, "ok_J_min_points"])                       # u03 chi 29 diem
    assert not bool(c.loc[43, "ok_D_ratio"]) and bool(c.loc[42, "ok_D_ratio"])
    assert c.loc[42, "overlap_vs_s43"] == pytest.approx(0.5) and bool(c.loc[42, "ok_D_overlap"])
    assert not c["ok_all"].any()
    with pytest.raises(ValueError, match="tap don vi"):
        scheme_conditions(u, {42: A.drop("u03")}, best_objective=1.0, n_folds=2)


def test_unit_pool_dung_lai_cho_fold_schemes_cung_ket_qua():
    rng = np.random.default_rng(3)
    u = _units(rng.uniform(50, 150, 12).round(1), rng.uniform(5, 60, 12).round(1), [100] * 12)
    a = fold_schemes(u, [42, 43, 44], n_folds=3, n_starts=200, pool_seed=0, tols=(0.5, 1.0), max_overlap=0.8)
    pool = unit_pool(u, n_folds=3, n_starts=200, pool_seed=0)
    b = fold_schemes(u, [42, 43, 44], n_folds=3, n_starts=200, pool_seed=0, tols=(0.5, 1.0), max_overlap=0.8,
                     pool=pool)
    for s in (42, 43, 44):
        assert a[0][s]["cv_fold"].tolist() == b[0][s]["cv_fold"].tolist()
    assert a[2] == b[2]
    best = min(v[0] for v in pool.values())
    c = scheme_conditions(u, {s: a[0][s].set_index("unit_id")["cv_fold"] for s in (42, 43, 44)}, best, n_folds=3,
                          tol=0.5, max_overlap=0.8).set_index("seed")
    assert bool(c.loc[42, "main_is_best"]) and c.loc[42, "ratio_vs_best"] == pytest.approx(1.0)
    assert c["ok_D_ratio"].all() and c["ok_D_overlap"].all()


# ---------- CHG-14 fold phu (cu: test_cv_folds_chg14.py)

def _u(scope, coast, ids=None):
    ids = ids or [f"u{i}" for i in range(len(scope))]
    return pd.DataFrame({"unit_id": ids, "scope_km2": scope, "coast_scope_km2": coast})


def _obj(units, lab, n_folds=2):
    tot = np.zeros((n_folds, 2))
    np.add.at(tot, np.asarray(lab), units[["scope_km2", "coast_scope_km2"]].to_numpy(float))
    return fold_balance_objective(tot)


# ------------------------------------------------------------------ pha hoa
def test_pha_hoa_muc_tieu_chenh_nhieu_so_uu_tien_ven_bien():
    """A = {u0,u1}|{u2,u3}: dien tich lech 0,1, ven bien 0 -> muc tieu 0,1, lech ven bien 0.
    B = {u0,u2}|{u1,u3}: dien tich 0, ven bien lech 0,1 -> muc tieu 0,1, lech ven bien 0,05.
    B nho hon A 1e-12 (nhieu so, nhu hang 1-2 chenh 5e-17 tren v3) -> sau lam tron 1e-9 hoa -> A (ven bien tot)."""
    u = _u([11, 10, 9, 10], [11, 9, 10, 10])
    A, B = (0, 0, 1, 1), (0, 1, 0, 1)
    assert _obj(u, A) == pytest.approx(0.1) and _obj(u, B) == pytest.approx(0.1)
    assert fold_max_rel_dev(A, u["coast_scope_km2"], 2) == pytest.approx(0.0)
    assert fold_max_rel_dev(B, u["coast_scope_km2"], 2) == pytest.approx(0.05)
    pool = {B: [0.1 - 1e-12, 5], A: [0.1, 1]}
    assert sorted(pool, key=lambda k: (pool[k][0], k))[0] == B          # quy tac cu: B thang vi nhieu so
    keys, dev, code, rkey = rank_pool(u, pool, n_folds=2)
    assert keys[0] == A
    picked, table, info = fold_schemes(u, [42], n_folds=2, pool=pool, max_objective=1.0, tols=None)
    assert tuple(picked[42]["cv_fold"]) == A and picked[42].attrs["rank"] == 1
    assert table.loc[0, "ma_phan_chia"] == "0-0-1-1" and table["objective_round"].nunique() == 1
    # sai khac THAT (> 1e-9) thi muc tieu van quyet
    keys2, *_ = rank_pool(u, {B: [0.1 - 1e-6, 1], A: [0.1, 1]}, n_folds=2)
    assert keys2[0] == B


def test_pha_hoa_cung_muc_tieu_cung_ven_bien_theo_ma_khong_phu_thuoc_thu_tu_dong():
    u = _u([10, 10, 10, 10], [5, 5, 5, 5])
    A, B, C = (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0)
    pool = {C: [0.0, 9], B: [0.0, 3], A: [0.0, 1]}
    keys, *_ = rank_pool(u, pool, n_folds=2)
    assert keys == [A, B, C]
    # dao thu tu dong bang don vi: khoa pool viet theo dong moi (u3, u2, u1, u0) -> van chon {u0,u1}|{u2,u3}
    r = u.iloc[::-1].reset_index(drop=True)
    pool_r = {(1, 0, 1, 0): [0.0, 3], (0, 1, 1, 0): [0.0, 9], (1, 1, 0, 0): [0.0, 1]}
    picked, _, _ = fold_schemes(r, [42], n_folds=2, pool=pool_r, max_objective=1.0, tols=None)
    f = picked[42].set_index("unit_id")["cv_fold"]
    assert f["u0"] == f["u1"] != f["u2"] == f["u3"]
    assert picked[42].attrs["code"] == "0-0-1-1"
    assert scheme_code([1, 1, 0, 0], ["u3", "u2", "u1", "u0"]) == (0, 0, 1, 1)


def test_pick_alternates_tong_muc_tieu_nho_nhat_khong_theo_thu_tu_tu_dien():
    rk = {"a1": scheme_rank_key(0.10, 0.0, (1,)), "a2": scheme_rank_key(0.11, 0.0, (2,)),
          "a3": scheme_rank_key(0.115, 0.0, (3,)), "a4": scheme_rank_key(0.20, 0.0, (4,))}
    bad = {frozenset(("a1", "a2")), frozenset(("a1", "a3"))}

    def ov(a, b):
        return 0.8 if frozenset((a, b)) in bad else 0.5

    got, n = pick_alternates(["a1", "a2", "a3", "a4"], rk, ov, 2, 0.6)
    assert got == ["a2", "a3"]          # tong 0,225 < (a1, a4) 0,30 (quy tac tu dien cu se chon a1, a4)
    assert n == 4                       # 6 cap - 2 cap trung > 0,6
    assert pick_alternates(["a1", "a2"], rk, ov, 2, 0.6) == (None, 0)


def test_pick_alternates_hoa_tong_roi_ven_bien_roi_ma():
    ov = lambda a, b: 0.5  # noqa: E731
    rk = {"b1": scheme_rank_key(0.1, 0.05, (0, 0, 1)), "b2": scheme_rank_key(0.1, 0.02, (0, 1, 0)),
          "b3": scheme_rank_key(0.1 + 1e-13, 0.02, (0, 1, 1))}
    assert pick_alternates(["b1", "b2", "b3"], rk, ov, 2, 0.6)[0] == ["b2", "b3"]   # cung tong -> ven bien nho
    rk = {"c1": scheme_rank_key(0.1, 0.02, (0, 1, 1)), "c2": scheme_rank_key(0.1, 0.02, (0, 0, 1)),
          "c3": scheme_rank_key(0.1, 0.02, (0, 1, 0))}

    def ov2(a, b):
        return 0.9 if {a, b} == {"c2", "c3"} else 0.5

    # hop le: (c1, c2) ma {001, 011}; (c1, c3) ma {010, 011} -> chon ma nho hon; xep theo hang -> c2 truoc
    assert pick_alternates(["c1", "c2", "c3"], rk, ov2, 2, 0.6)[0] == ["c2", "c1"]


def test_fold_schemes_nguong_tuyet_doi_cap_tong_nho_nhat(units_coast):
    from itertools import combinations

    kw = dict(n_folds=5, n_starts=80, pool_seed=0, max_overlap=0.60, coast_tol=0.15, tols=None)
    _, table0, info0 = fold_schemes(units_coast, [42], max_objective=10.0, **kw)
    best = info0["best_objective"]
    thr = round(float(table0["objective"].quantile(0.6)), 6)
    picked, table, info = fold_schemes(units_coast, [42, 43, 44], max_objective=thr, **kw)
    again, _, _ = fold_schemes(units_coast, [42, 43, 44], max_objective=thr, **kw)
    assert info["mode"] == "tuyet_doi" and info["tol_used"] is None and info["best_objective"] == best
    lab = {s: picked[s].sort_values("unit_id")["cv_fold"].to_numpy() for s in picked}
    coast = picked[42].sort_values("unit_id")["coast_scope_km2"].to_numpy()
    for s in (42, 43, 44):
        pd.testing.assert_frame_equal(picked[s], again[s])
        assert picked[s].attrs["objective"] <= thr + 1e-12
        assert fold_max_rel_dev(lab[s], coast, 5) <= 0.15 + 1e-12
    assert picked[42].attrs["rank"] == 1
    for x, y in ((42, 43), (42, 44), (43, 44)):
        assert fold_overlap(lab[x], lab[y]) <= 0.60 + 1e-12
    # vet can: khong cap hop le nao co tong muc tieu nho hon
    t = table.set_index("rank")
    main = t.loc[1, "ma_phan_chia"]
    el = t[(t.index > 1) & (t["objective_round"] <= thr) & (t["coast_max_dev"] <= 0.15) & (t["overlap_vs_main"] <= 0.6)]
    labs = {r: np.array([int(x) for x in t.loc[r, "ma_phan_chia"].split("-")]) for r in el.index}
    assert main == "-".join(str(x) for x in lab[42])
    sums = [t.loc[a, "objective_round"] + t.loc[b, "objective_round"] for a, b in combinations(el.index, 2)
            if fold_overlap(labs[a], labs[b]) <= 0.6]
    assert len(sums) == info["n_valid_combos"] and len(el) == info["n_eligible"]
    got = picked[43].attrs["objective"] + picked[44].attrs["objective"]
    assert got == pytest.approx(min(sums), abs=1e-9)
    assert picked[43].attrs["rank"] < picked[44].attrs["rank"]
    assert (table["chon"] != "").sum() == 3
    with pytest.raises(ValueError, match="nguong tuyet doi"):                   # chinh vuot nguong -> loi
        fold_schemes(units_coast, [42, 43, 44], max_objective=best * 0.5, **kw)
    with pytest.raises(ValueError, match="Khong du"):
        fold_schemes(units_coast, [42, 43, 44], max_objective=best + 1e-10, **kw)
    with pytest.raises(ValueError, match="max_objective"):
        fold_schemes(units_coast, [42, 43, 44], **kw)


# ------------------------------------------------------------------ diem ven bien
def test_dist_coast_at_points_pixel_chua_diem_va_ngoai_khung_nan(tmp_path):
    p = tmp_path / "dist.tif"
    d = np.array([[3.0, 19.9, 20.0, 25.0]], "float32")
    with rasterio.open(p, "w", driver="GTiff", width=4, height=1, count=3, dtype="float32", crs="EPSG:32648",
                       transform=from_origin(500000, 1200000, 90, 90)) as ds:
        for i, n in enumerate(("dist_main_river_km", "dist_any_water_km", "dist_coast_km"), start=1):
            ds.write(d * (10 if i < 3 else 1), i)
            ds.set_band_description(i, n)
    xs = [500000 + 90 * c + 89.0 for c in range(4)] + [500000 + 400]
    pts = gpd.GeoDataFrame(geometry=[Point(x, 1200000 - 45) for x in xs], crs="EPSG:32648").to_crs(4326)
    v = dist_coast_at_points(pts, p)
    assert v[:4].tolist() == pytest.approx([3.0, 19.9, 20.0, 25.0], abs=1e-5)   # band theo TEN, khong band 1
    assert np.isnan(v[4])
    assert int((v <= 20.0).sum()) == 3                                          # 20 km tinh la ven bien


# ------------------------------------------------------------------ sao luu --force
def test_backup_before_force_khong_ghi_de_ban_sao_luu(tmp_path):
    import build_cv_folds as b

    out, rep, bk = tmp_path / "eval", tmp_path / "ket-qua", tmp_path / "eval" / "deprecated_v2b"
    (out / "cell_blocks" / "s42").mkdir(parents=True)
    rep.mkdir()
    files = {out / "cv_folds.csv": "a", out / "cv_folds.csv.provenance.json": "{}", out / "cv_folds_s42.csv": "b",
             out / "cell_blocks" / "h3.csv": "c", out / "cell_blocks" / "s42" / "h3.csv": "d",
             rep / "dot4_cv_tap_phuong_an.csv": "e", rep / "khac.csv": "khong sao luu"}
    for p, txt in files.items():
        p.write_text(txt)
    (bk).mkdir()
    (bk / "eval_points.geojson").write_text("diem cu")                        # file co san trong thu muc sao luu
    t1 = b.backup_before_force(str(out), str(rep), str(bk))
    assert t1 == str(bk)
    assert (bk / "cv_folds_s42.csv").read_text() == "b"
    assert (bk / "cell_blocks" / "s42" / "h3.csv").read_text() == "d"
    assert (bk / "ket-qua" / "dot4_cv_tap_phuong_an.csv").read_text() == "e"
    assert not (bk / "ket-qua" / "khac.csv").exists() and not (bk / "deprecated_v2b").exists()
    man = json.loads((bk / "SAO_LUU_cv_folds.json").read_text(encoding="utf-8"))
    assert len(man["files"]) == 6
    # chay lai, nguon khong doi -> khong chep them
    assert b.backup_before_force(str(out), str(rep), str(bk)) == str(bk)
    assert not [d for d in os.listdir(bk) if d.startswith("sao_luu_")]
    # nguon doi (fold moi da ghi) -> thu muc con co dau thoi gian; ban sao luu cu giu nguyen
    (out / "cv_folds_s42.csv").write_text("moi")
    t2 = b.backup_before_force(str(out), str(rep), str(bk), now=datetime(2026, 10, 4, 9, 0, 0))
    assert t2 == str(bk / "sao_luu_20261004_090000")
    assert (bk / "cv_folds_s42.csv").read_text() == "b"
    assert (bk / "sao_luu_20261004_090000" / "cv_folds_s42.csv").read_text() == "moi"
    assert (bk / "eval_points.geojson").read_text() == "diem cu"


def test_cli_tols_va_max_objective_loai_tru():
    import argparse

    import build_cv_folds as b

    a = b.resolve_rule(argparse.Namespace(tols=None, max_objective=None))
    assert a.max_objective == 0.183
    assert b.resolve_rule(argparse.Namespace(tols=[0.1, 0.15], max_objective=None)).max_objective is None
    with pytest.raises(SystemExit):
        b.resolve_rule(argparse.Namespace(tols=[0.1], max_objective=0.183))

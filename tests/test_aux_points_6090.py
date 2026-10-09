"""Tap cham PHU 349 diem 60/90 (scripts/build_aux_points_6090.py, S2 2026-10-06).

Du lieu TONG HOP (raster scope gia tri biet truoc, 2 khoi 50 km) cho logic chon / kiem; test cuoi dung file that
(skip khi thieu): tap con cua tap goc, giao tap chinh rong, 183 + 166, 37 giu rieng, trung dap an keep6090.
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from preprocessing import file_sha256
from scope_mask import select_filtered_points
from settings import data_path  # (.env: DATA_ROOT)
from training.point_eval import point_blocks, point_frame, restrict_reference

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_aux_points_6090 as bap  # noqa: E402

UTM = "EPSG:32648"
PIX = 10000.0  # pixel 10 km de moi diem mot pixel (ham doc theo transform, khong phu thuoc 30 m)


# ------------------------------------------------------------------ ham thuan: chon diem
def _pts_attrs():
    pts = gpd.GeoDataFrame({"point_id": [f"p{i}" for i in range(6)], "block_id": "b", "is_holdout": False},
                           geometry=[Point(i, 0) for i in range(6)], crs=UTM)
    attrs = pd.DataFrame({"scope": [1, 0, 0, 0, 1, 0], "wc_class": [40, 60, 50, 90, 10, 60],
                          "zenodo_n_years": [10, 9, 8, 7, 6, 5]}, index=pts.index)
    return pts, attrs


def test_chon_diem_bi_loc_dung_lop_giu_thu_tu_va_cot():
    pts, attrs = _pts_attrs()
    sel = select_filtered_points(pts, attrs, [90, 60])
    assert sel["point_id"].tolist() == ["p1", "p3", "p5"]       # lop 50 bi loc nhung KHONG lay; scope 1 khong lay
    assert sel["wc_class"].tolist() == [60, 90, 60] and sel["zenodo_n_years"].tolist() == [9, 7, 5]
    assert list(sel.geometry.x) == [1, 3, 5] and {"block_id", "is_holdout"} <= set(sel.columns)
    assert select_filtered_points(pts, attrs, [50])["point_id"].tolist() == ["p2"]


def test_chon_diem_loi_mat_na_va_index():
    pts, attrs = _pts_attrs()
    bad = attrs.copy()
    bad.loc[1, "scope"] = 1  # diem lop 60 ma scope = 1 -> mat na khong loai 60 -> sai mat na
    with pytest.raises(ValueError, match="van co scope = 1"):
        select_filtered_points(pts, bad, [60, 90])
    with pytest.raises(ValueError, match="cung index"):
        select_filtered_points(pts, attrs.reset_index(drop=True).set_index(attrs.index + 10), [60])
    with pytest.raises(ValueError, match="rong"):
        select_filtered_points(pts, attrs, [])


# ------------------------------------------------------------------ dap an la tap cha cua file diem
def test_restrict_reference_tap_con_va_loi():
    ref = pd.DataFrame({"point_id": ["a", "a", "b", "c"], "season": [1, 2, 1, 1], "x": [1, 2, 3, 4]})
    out, n_drop = restrict_reference(ref, ["a", "c"])
    assert out["point_id"].tolist() == ["a", "a", "c"] and n_drop == 1
    with pytest.raises(ValueError, match="khong co dong nao trong dap an"):
        restrict_reference(ref, ["a", "z"])
    with pytest.raises(ValueError, match="trung"):
        restrict_reference(ref, ["a", "a"])


def test_point_frame_mac_dinh_van_loi_khi_dap_an_thua_diem():
    """Khong noi long mac dinh: dap an co diem khong co trong file diem -> LOI (chi --points-subset-of-ref moi bo)."""
    blocks = gpd.GeoDataFrame({"block_id": ["blk_10_20"], "is_holdout": [False]},
                              geometry=[box(500000, 1000000, 550000, 1050000)], crs=UTM).to_crs(4326)
    folds = pd.DataFrame({"block_id": ["blk_10_20"], "cv_fold": [0]})
    grid = gpd.GeoDataFrame({"cell_id": ["c1"]}, geometry=[box(500000, 1000000, 550000, 1050000)], crs=UTM)
    pts = gpd.GeoDataFrame({"point_id": ["p1"]}, geometry=[Point(510000, 1010000)], crs=UTM)
    ref = pd.DataFrame({"point_id": ["p1", "p2"], "season": [2018, 2018], "ref_salinity": [1.0, 2.0],
                        "n_valid_3x3": [9, 9]})
    with pytest.raises(ValueError, match="khong co trong file diem"):
        point_frame(pts, grid, blocks, folds, ref, (2020,))
    r, _ = restrict_reference(ref, pts["point_id"])
    pf = point_frame(pts, grid, blocks, folds, r, (2020,))
    assert pf["point_id"].tolist() == ["p1"]


# ------------------------------------------------------------------ script tren du lieu tong hop
# Diem (UTM): (x, y, scope, wc_class). Khoi blk_10_20 (giu rieng) x 500-550 km, blk_11_20 (CV) x 550-600 km.
TOY = [("p0", 505000, 1045000, 1, 40), ("p1", 515000, 1045000, 0, 60), ("p2", 525000, 1045000, 0, 50),
       ("p3", 565000, 1045000, 0, 90), ("p4", 575000, 1045000, 1, 10), ("p5", 585000, 1035000, 0, 60)]


def _write_scope(path, toy):
    import rasterio
    from rasterio.transform import from_origin

    w, h = 12, 6
    tr = from_origin(500000, 1060000, PIX, PIX)
    bands = {"scope": np.zeros((h, w), "uint8"), "zenodo_n_years": np.full((h, w), 10, "uint8"),
             "wc_class": np.zeros((h, w), "uint8")}
    inv = ~tr
    for _, x, y, sc, wc in toy:
        c, r = (int(np.floor(v)) for v in inv * (x, y))
        bands["scope"][r, c], bands["wc_class"][r, c] = sc, wc
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=UTM,
                       transform=tr) as dst:
        for i, (name, arr) in enumerate(bands.items(), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, name)


def _block_of(x):
    return "blk_10_20" if x < 550000 else "blk_11_20"


@pytest.fixture()
def toy(tmp_path):
    def build(toy=TOY, ref_added=None, main_ids=None, block_override=None):
        d = tmp_path
        scope = d / "scope.tif"
        _write_scope(scope, toy)
        blocks = gpd.GeoDataFrame({"block_id": ["blk_10_20", "blk_11_20"], "is_holdout": [True, False]},
                                  geometry=[box(500000, 1000000, 550000, 1050000), box(550000, 1000000, 600000, 1050000)],
                                  crs=UTM).to_crs(4326)
        blocks.to_file(d / "blocks.geojson", driver="GeoJSON")
        pd.DataFrame({"block_id": ["blk_10_20", "blk_11_20"], "unit_id": ["blk_10_20", "blk_11_20"],
                      "cv_fold": [-1, 0]}).to_csv(d / "cv_folds_s42.csv", index=False)
        bid = [(block_override or {}).get(p, _block_of(x)) for p, x, *_ in toy]
        src = gpd.GeoDataFrame({"point_id": [t[0] for t in toy], "block_id": bid,
                                "is_holdout": [b == "blk_10_20" for b in bid]},
                               geometry=[Point(x, y) for _, x, y, *_ in toy], crs=UTM).to_crs(4326)
        src.to_file(d / "src.geojson", driver="GeoJSON")
        src = gpd.read_file(d / "src.geojson")  # toa do sau ghi/doc (giong file that)
        keep_ids = main_ids if main_ids is not None else [t[0] for t in toy if t[3] == 1]
        src[src["point_id"].isin(keep_ids)].to_file(d / "main.geojson", driver="GeoJSON")
        dropped = pd.Series([t[4] for t in toy if t[3] != 1]).value_counts().sort_index()
        (d / "main.geojson.provenance.json").write_text(json.dumps({
            "filtered_from_sha256": file_sha256(d / "src.geojson"), "scope_mask_sha256": file_sha256(scope),
            "dropped_by_wc_class": {str(k): int(v) for k, v in dropped.items()}}), encoding="utf-8")
        added = ref_added if ref_added is not None else [t[0] for t in toy if t[3] != 1 and t[4] in (60, 90)]
        rows = []
        for (p, x, _, sc, wc), b in zip(toy, bid):  # dap an mang block_id/is_holdout CUA NGUON (nhu file that)
            if sc == 1 or p in added:
                for s in (2018, 2019):
                    rows.append({"point_id": p, "block_id": b, "is_holdout": b == "blk_10_20",
                                 "wc_class": wc, "added": p in added, "season": s, "ref_salinity": 1.0,
                                 "n_valid_3x3": 9})
        pd.DataFrame(rows).to_csv(d / "ref.csv", index=False)
        (d / "boundary.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
        args = bap.build_parser().parse_args([
            "--source", str(d / "src.geojson"), "--main-points", str(d / "main.geojson"), "--scope", str(scope),
            "--points-ref", str(d / "ref.csv"), "--blocks", str(d / "blocks.geojson"), "--folds-dir", str(d),
            "--schemes", "42", "--tables-dir", "", "--out", str(d / "aux" / "eval_points_6090.geojson"),
            "--boundary", str(d / "boundary.geojson")])
        return args
    return build


def test_script_tong_hop_ghi_dung_diem_khoi_fold(toy):
    a = toy()
    bap.main(a)
    out = gpd.read_file(a.out)
    src = gpd.read_file(a.source).set_index("point_id")
    assert out["point_id"].tolist() == ["p1", "p3", "p5"]                    # thu tu nguon, khong p2 (lop 50)
    assert out["wc_class"].tolist() == [60, 90, 60]
    assert out["is_holdout"].tolist() == [True, False, False]
    assert out["cv_fold_s42"].tolist() == [-1, 0, 0] and out["unit_id_s42"].tolist() == ["blk_10_20", "blk_11_20",
                                                                                         "blk_11_20"]
    assert np.array_equal(out.geometry.x.to_numpy(), src.loc[out["point_id"]].geometry.x.to_numpy())
    assert np.array_equal(out.geometry.y.to_numpy(), src.loc[out["point_id"]].geometry.y.to_numpy())
    prov = json.loads(open(a.out + ".provenance.json", encoding="utf-8").read())
    assert prov["n_points"] == 3 and prov["n_by_wc_class"] == {"60": 2, "90": 1} and prov["n_holdout"] == 1
    assert prov["source_sha256"] == file_sha256(a.source)


def test_script_diem_ngoai_moi_khoi_la_loi(toy):
    t = list(TOY)
    t[5] = ("p5", 605000, 1035000, 0, 60)  # ngoai ca 2 khoi
    a = toy(toy=t, block_override={"p5": "blk_11_20"})
    with pytest.raises(SystemExit, match="ngoai moi khoi"):
        bap.main(a)
    assert not os.path.exists(a.out)


def test_script_dap_an_khac_tap_diem_la_loi(toy):
    a = toy(ref_added=["p1", "p3"])  # thieu p5 trong dap an added=True
    with pytest.raises(SystemExit, match="khac dap an added=True"):
        bap.main(a)


def test_script_tai_lap_bo_loc_khac_tap_chinh_la_loi(toy):
    a = toy(main_ids=["p0"])  # tap chinh khong phai ket qua loc scope = 1 (thieu p4)
    with pytest.raises(SystemExit, match="KHONG trung tap chinh"):
        bap.main(a)


def test_script_block_id_nguon_khac_vi_tri_la_loi(toy):
    a = toy(block_override={"p3": "blk_10_20"})  # nguon (va dap an) ghi sai khoi cho p3 -> point_blocks bat
    with pytest.raises(SystemExit, match="khac khoi theo vi tri"):
        bap.main(a)


# ------------------------------------------------------------------ du lieu that (skip khi thieu)
SRC = os.path.join(ROOT, "data", "eval", "deprecated_v2b", "eval_points.geojson")
MAIN = os.path.join(ROOT, "data", "eval", "eval_points.geojson")
OUT = os.path.join(ROOT, "data", "eval", "aux_6090", "eval_points_6090.geojson")
REF = data_path("labels", "points_reference_keep6090.csv")
BLOCKS = os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson")
need_real = pytest.mark.skipif(not all(os.path.exists(p) for p in (SRC, MAIN, OUT, REF, BLOCKS)),
                               reason="thieu du lieu that (diem / dap an keep6090)")


@pytest.fixture(scope="module")
def real():
    return gpd.read_file(OUT), gpd.read_file(SRC), gpd.read_file(MAIN)


@need_real
def test_that_tap_con_cua_tap_goc(real):
    out, src, _ = real
    s = src.set_index("point_id")
    assert out["point_id"].is_unique and set(out["point_id"]) <= set(s.index)
    assert np.array_equal(out.geometry.x.to_numpy(), s.loc[out["point_id"]].geometry.x.to_numpy())
    assert np.array_equal(out.geometry.y.to_numpy(), s.loc[out["point_id"]].geometry.y.to_numpy())
    assert (out["block_id"].to_numpy() == s.loc[out["point_id"], "block_id"].to_numpy()).all()
    assert (out["is_holdout"].astype(bool).to_numpy() == s.loc[out["point_id"], "is_holdout"].astype(bool).to_numpy()).all()


@need_real
def test_that_giao_tap_chinh_rong_va_hop_dung_tap_goc_tru_lop_50(real):
    out, src, main = real
    assert len(main) == 10801 and not set(out["point_id"]) & set(main["point_id"])
    assert len(src) == 11766 and len(src) - len(main) - len(out) == 616  # phan con lai = lop 50


@need_real
def test_that_dung_183_lop_60_va_166_lop_90(real):
    out, _, _ = real
    assert len(out) == 349 and out["wc_class"].value_counts().to_dict() == {60: 183, 90: 166}


@need_real
def test_that_37_diem_giu_rieng_va_fold_theo_vi_tri(real):
    out, _, _ = real
    assert int(out["is_holdout"].astype(bool).sum()) == 37
    blocks = gpd.read_file(BLOCKS)
    for s in (42, 43, 44):
        fp = os.path.join(ROOT, "data", "eval", f"cv_folds_s{s}.csv")
        if not os.path.exists(fp):
            continue
        hb = out["is_holdout"].astype(bool).to_numpy()
        assert ((out[f"cv_fold_s{s}"] == -1).to_numpy() == hb).all()
        pb = point_blocks(out, blocks, pd.read_csv(fp, dtype={"block_id": str})).set_index("point_id")
        assert (pb.loc[out["point_id"], "cv_fold"].to_numpy() == out[f"cv_fold_s{s}"].to_numpy()).all()


@need_real
def test_that_trung_tap_added_cua_dap_an_keep6090(real):
    out, _, _ = real
    ref = pd.read_csv(REF, dtype={"point_id": str}, usecols=["point_id", "added", "wc_class"])
    added = ref[ref["added"].astype(str).str.lower() == "true"].drop_duplicates("point_id").set_index("point_id")
    assert set(added.index) == set(out["point_id"])
    assert (added.loc[out["point_id"], "wc_class"].to_numpy() == out["wc_class"].to_numpy()).all()

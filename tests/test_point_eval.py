"""Phuong an C (src/training/point_eval.py): khoi theo vi tri diem, dap an median 3x3 >= 5/9, moi diem du doan
mot lan, khong du doan bang mo hinh da hoc o chua diem. Du lieu TONG HOP; test cuoi dung du lieu that nhung KHONG
fit mo hinh (chi kiem phu diem/dac trung)."""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from training.point_eval import (REF_MIN_VALID, attach_features, check_point_not_in_train, load_reference,
                                 point_blocks, point_frame)
from settings import data_path  # (.env: DATA_ROOT)

UTM = "EPSG:32648"


def _blocks():
    # 2 khoi 50 km canh nhau theo truc x: blk_10_20 (giu rieng), blk_11_20; luu EPSG:4326 nhu file that
    b = gpd.GeoDataFrame({"block_id": ["blk_10_20", "blk_11_20"], "is_holdout": [True, False]},
                         geometry=[box(500000, 1000000, 550000, 1050000), box(550000, 1000000, 600000, 1050000)],
                         crs=UTM)
    return b.to_crs(4326)


def _folds():
    return pd.DataFrame({"block_id": ["blk_10_20", "blk_11_20"], "cv_fold": [-1, 0]})


def _points(xy, ids=None, block_col=None, wgs84=True):
    g = gpd.GeoDataFrame({"point_id": ids or [f"p{i}" for i in range(len(xy))]},
                         geometry=[Point(x, y) for x, y in xy], crs=UTM)
    if block_col is not None:
        g["block_id"] = block_col
    return g.to_crs(4326) if wgs84 else g


def test_khoi_theo_vi_tri_diem_va_canh_chung():
    # Giu UTM (khong qua 4326): diem p2 nam DUNG tren canh chung x = 550000.
    pts = _points([(520000, 1020000), (570000, 1020000), (550000, 1020000)], wgs84=False)
    pb = point_blocks(pts, _blocks(), _folds()).set_index("point_id")
    assert pb.loc["p0", "block_id"] == "blk_10_20" and pb.loc["p0", "cv_fold"] == -1 and pb.loc["p0", "is_holdout"]
    assert pb.loc["p1", "block_id"] == "blk_11_20" and pb.loc["p1", "cv_fold"] == 0
    assert pb.loc["p2", "block_id"] == "blk_10_20"  # tren canh chung -> block_id nho nhat (co dinh)


def test_block_id_trong_file_khac_vi_tri_bao_loi():
    pts = _points([(520000, 1020000)], block_col=["blk_11_20"])  # ghi sai khoi
    with pytest.raises(ValueError, match="khac khoi theo vi tri"):
        point_blocks(pts, _blocks(), _folds())


def test_fold_khong_nhat_quan_bao_loi():
    with pytest.raises(ValueError, match="khong nhat quan"):
        point_blocks(_points([(570000, 1020000)]), _blocks(), pd.DataFrame({"block_id": ["blk_10_20", "blk_11_20"],
                                                                           "cv_fold": [-1, -1]}))


def test_dap_an_quy_tac_median_3x3_5_tren_9():
    import label_season
    assert REF_MIN_VALID == label_season.REF_MIN_VALID == 5
    ref = pd.DataFrame({"point_id": ["a", "a", "b"], "season": [2019, 2020, 2019],
                        "ref_salinity": [1.0, np.nan, 2.0], "n_valid_3x3": [9, 4, 5]})
    out = load_reference(ref)
    assert out[["point_id", "season"]].values.tolist() == [["a", 2019], ["b", 2019]]
    with pytest.raises(ValueError, match="sai quy tac"):
        load_reference(ref.assign(ref_salinity=[1.0, 3.0, 2.0]))  # 4/9 ma co gia tri
    with pytest.raises(ValueError, match="du 5/9 pixel nhung NaN"):
        load_reference(ref.assign(ref_salinity=[1.0, np.nan, np.nan]))


def _grid():
    # 4 o 25 km: c0, c1 trong khoi giu rieng; c2, c3 trong khoi fold 0
    cells = [box(500000 + i * 25000, 1000000, 525000 + i * 25000, 1050000) for i in range(4)]
    return gpd.GeoDataFrame({"cell_id": [f"c{i}" for i in range(4)]}, geometry=cells, crs=UTM)


def test_point_frame_vai_tro_va_diem_ngoai_luoi():
    pts = _points([(510000, 1020000), (560000, 1020000), (590000, 1020000)])
    ref = pd.DataFrame([(p, s, 1.0, 9) for p in ["p0", "p1", "p2"] for s in (2019, 2020)],
                       columns=["point_id", "season", "ref_salinity", "n_valid_3x3"])
    pf = point_frame(pts, _grid(), _blocks(), _folds(), ref, holdout_seasons=(2020,)).set_index(["point_id", "season"])
    assert pf.loc[("p1", 2019), "role"] == "cv" and pf.loc[("p1", 2019), "cell_id"] == "c2"
    assert pf.loc[("p1", 2020), "role"] == "thoi_gian"
    assert pf.loc[("p0", 2019), "role"] == "khong_gian" and pf.loc[("p0", 2020), "role"] == "ca_hai"
    far = _points([(700000, 1020000)], ids=["px"])
    with pytest.raises(ValueError):
        point_frame(far, _grid(), _blocks(), _folds(), ref.assign(point_id="px").drop_duplicates("season"), (2020,))


def test_gan_dac_trung_o_chua_diem_va_kiem_khong_hoc_o_cua_diem():
    pf = pd.DataFrame({"point_id": ["p1"], "season": [2019], "cell_id": ["c2"], "y_ref": [1.0], "cv_fold": [0]})
    feats = pd.DataFrame({"cell_id": ["c2", "c3"], "season": [2019, 2019], "dem_mean": [1.5, 2.0]})
    assert attach_features(pf, feats, ["dem_mean"])["dem_mean"].tolist() == [1.5]
    with pytest.raises(ValueError, match="khong co dong"):
        attach_features(pf.assign(season=2018), feats, ["dem_mean"])
    check_point_not_in_train(pf, ["c3"], 0)
    with pytest.raises(ValueError, match="vi pham"):
        check_point_not_in_train(pf, ["c2", "c3"], 0)


# ---------------------------------------------------------------- du lieu that, KHONG fit mo hinh
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIFIED = data_path("features/unified")
REF = data_path("labels/points_reference.csv")
POINTS = os.path.join(ROOT, "data", "eval", "eval_points.geojson")
need_real = pytest.mark.skipif(not (os.path.isdir(UNIFIED) and os.path.exists(REF) and os.path.exists(POINTS)),
                               reason="khong co du lieu that")


@need_real
def test_du_10801_diem_moi_luoi_ke_ca_o_khong_huan_luyen():
    """Moi luoi: du 10.801 diem gan duoc khoi + o; moi (diem, mua) co dap an hop le co dong dac trung trong bang
    hop nhat (ke ca o train_ok_scope = False). 347 diem KHONG mua nao co >= 5/9 pixel (quy tac CHG-13) -> khong
    cham duoc o MOI luoi (giong nhau giua khung); con lai 10.454 diem duoc cham."""
    import glob

    from training.evaluate import assign_points_to_cells

    points = gpd.read_file(POINTS)
    blocks = gpd.read_file(os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    folds = pd.read_csv(os.path.join(ROOT, "data", "eval", "cv_folds.csv"), dtype={"block_id": str})
    ref = pd.read_csv(REF, dtype={"point_id": str})
    assert points["point_id"].nunique() == 10801
    excluded = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "grids", "*.geojson"))):
        g = os.path.basename(path)[:-8]
        grid = gpd.read_file(path)
        grid["cell_id"] = grid["cell_id"].astype(str)
        assert assign_points_to_cells(points, grid).notna().all(), g          # 10.801 diem deu nam trong o
        assert len(point_blocks(points, blocks, folds)) == 10801, g
        pf = point_frame(points, grid, blocks, folds, ref, (2020,))
        assert pf["point_id"].nunique() == 10454, g
        tab = pd.read_csv(os.path.join(UNIFIED, f"{g}_unified.csv"), dtype={"cell_id": str},
                          usecols=["cell_id", "season", "train_ok_scope", "dem_mean"])
        out = attach_features(pf, tab, ["train_ok_scope", "dem_mean"])  # loi neu thieu dong (o, mua)
        assert len(out) == len(pf), g
        assert out["dem_mean"].notna().mean() > 0.99, g  # o chua diem (dat) co dac trung
        assert (pf["role"] == "cv").any() and set(pf["role"]) == {"cv", "khong_gian", "thoi_gian", "ca_hai"}, g
        excluded[g] = int((~out["train_ok_scope"].astype(bool)).sum())
    # Luoi min co (diem, mua) o o KHONG huan luyen (it dat) - van gan duoc dac trung de cham
    assert sum(excluded.values()) > 0, excluded


def test_fold_diem_phai_cung_cach_chia_voi_o():
    from training.train import _check_point_folds_match_cells

    pf = pd.DataFrame({"block_id": ["b1", "b2"], "cv_fold": [0, 1]})
    ct = pd.DataFrame({"block_id": ["b1", "b2", "b2"], "cv_fold": [0, 1, 1]})
    _check_point_folds_match_cells(pf, ct)
    with pytest.raises(ValueError, match="cach chia fold khac nhau"):
        _check_point_folds_match_cells(pf, ct.assign(cv_fold=[1, 0, 0]))

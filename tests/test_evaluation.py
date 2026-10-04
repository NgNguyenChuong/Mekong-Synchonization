"""Kiem thu dot 1 (khong can du lieu that): xu ly thieu du lieu, thiet ke danh gia,
duong co so, danh gia theo diem va kiem dinh. Du lieu la TONG HOP."""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_design import make_holdout_blocks, sample_eval_points, training_mask  # noqa: E402
from training.baselines import GlobalMeanBaseline, IDWBaseline, PersistenceBaseline  # noqa: E402
from training.evaluate import (  # noqa: E402
    assign_points_to_cells, bootstrap_median_diff_ci, holm_adjust, paired_wilcoxon, point_errors,
)
from training.features import MissingValueHandler, prepare_matrices  # noqa: E402


# ---------------- A. Thieu du lieu ----------------
def test_missing_handler_uses_train_median_only_and_flags():
    train = pd.DataFrame({"rain": [0.0, 10.0, np.nan, 20.0], "temp": [25.0, 26.0, 27.0, 28.0]})
    test = pd.DataFrame({"rain": [np.nan, 1000.0], "temp": [np.nan, 30.0]})
    h = MissingValueHandler().fit(train)
    out = h.transform(test)
    assert out.loc[0, "rain"] == 10.0          # trung vi train (0, 10, 20), khong phai 0
    assert out.loc[0, "temp"] == 26.5          # trung vi train
    assert list(out["rain__missing"]) == [1, 0]
    assert "temp__missing" not in out.columns  # train khong thieu temp -> khong them co
    # Doi gia tri test khong lam doi trung vi (khong ro ri tu tap kiem tra).
    h2 = MissingValueHandler().fit(train)
    assert h2.medians_.equals(h.medians_)


def test_missing_handler_drops_all_nan_column_and_hist_gb_passthrough():
    train = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan]})
    h = MissingValueHandler().fit(train)
    assert h.dropped_ == ["b"] and "b" not in h.transform(train).columns
    tr, te, handler = prepare_matrices("hist_gb", train, train)
    assert handler is None and tr["b"].isna().all()


# ---------------- B. Thiet ke danh gia ----------------
@pytest.fixture(scope="module")
def square_region():
    # Hinh vuong ~200 x 200 km trong UTM 48N quanh DBSCL.
    return gpd.GeoDataFrame(geometry=[box(500000, 1000000, 700000, 1200000)], crs="EPSG:32648").to_crs(4326)


def test_holdout_blocks_reach_target_area_and_are_deterministic(square_region):
    blocks = make_holdout_blocks(square_region, block_km=50, holdout_frac=0.2, seed=1)
    frac = blocks.loc[blocks["is_holdout"], "land_km2"].sum() / blocks["land_km2"].sum()
    assert 0.2 <= frac < 0.2 + blocks["land_km2"].max() / blocks["land_km2"].sum() + 1e-9
    again = make_holdout_blocks(square_region, block_km=50, holdout_frac=0.2, seed=1)
    assert list(again["is_holdout"]) == list(blocks["is_holdout"])


def test_eval_points_inside_holdout_land(square_region):
    blocks = make_holdout_blocks(square_region, block_km=50, holdout_frac=0.2, seed=1)
    pts = sample_eval_points(square_region, blocks, n_points=500, seed=3)
    holdout = blocks[blocks["is_holdout"]].geometry.union_all()
    assert len(pts) == 500 and pts["point_id"].is_unique
    assert pts.geometry.within(holdout.buffer(1e-9)).all()
    assert pts.geometry.within(square_region.geometry.union_all().buffer(1e-9)).all()


def test_training_mask_excludes_cells_touching_holdout():
    blocks = gpd.GeoDataFrame({"is_holdout": [True, False]},
                              geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326")
    grid = gpd.GeoDataFrame({"cell_id": ["in", "edge", "out"]},
                            geometry=[box(0.2, 0.2, 0.4, 0.4), box(0.9, 0.2, 1.1, 0.4), box(1.5, 0.2, 1.7, 0.4)],
                            crs="EPSG:4326")
    assert list(training_mask(grid, blocks)) == [False, False, True]


# ---------------- C. Duong co so ----------------
def test_global_mean_baseline():
    assert list(GlobalMeanBaseline().fit([1, 2, np.nan, 3]).predict(2)) == [2.0, 2.0]


def test_idw_exact_match_and_equidistant_mean():
    m = IDWBaseline(p=2, k=2).fit([[0, 0], [10, 0]], [1.0, 3.0])
    assert m.predict([[0, 0]])[0] == 1.0
    assert m.predict([[5, 0]])[0] == pytest.approx(2.0)
    assert m.predict([[1, 0]])[0] < 1.5  # gan diem 1 hon


def test_idw_tune_picks_from_grid_on_train_only():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, (200, 2))
    y = xy[:, 0] / 100.0
    groups = (xy[:, 0] // 250).astype(int) * 10 + (xy[:, 1] // 250).astype(int)
    # groups + GroupKFold chi de test (allow_group_kfold); chay that dung cv_splits.
    m = IDWBaseline.tune(xy, y, groups=groups, n_splits=4, allow_group_kfold=True)
    assert (m.p, m.k) in m.cv_scores_ and len(m.cv_scores_) == 9


def test_idw_tune_khong_roi_ve_kfold_ngau_nhien():
    # Loi V4 (soat 2026-10-03): groups=None hoac it nhom -> truoc day am tham dung KFold ngau nhien.
    xy = np.random.default_rng(1).uniform(0, 1000, (40, 2))
    y = xy[:, 0]
    with pytest.raises(ValueError, match="cv_splits"):
        IDWBaseline.tune(xy, y, groups=None)
    with pytest.raises(ValueError, match="nhom"):
        IDWBaseline.tune(xy, y, groups=np.repeat([0, 1, 2], [14, 13, 13]), n_splits=5, allow_group_kfold=True)


def test_idw_tune_chay_that_bat_buoc_cv_splits():
    # Muc 6 soat 2026-10-03: GroupKFold chia theo so mau -> fold khac nhau giua luoi; chi cho phep trong test.
    xy = np.random.default_rng(2).uniform(0, 1000, (40, 2))
    groups = np.repeat(np.arange(8), 5)
    with pytest.raises(ValueError, match="cv_splits"):
        IDWBaseline.tune(xy, xy[:, 0], groups=groups, n_splits=4)


def test_idw_tune_fold_toan_nan_bao_loi_khong_nanmean():
    # Fold 2 kiem mua 2021 nhung huan luyen chi co 2020 -> du doan toan NaN.
    # Truoc day np.nanmean bo fold nay am tham va diem CV chi con tu fold 1.
    xy = np.array([[0, 0], [10, 0], [20, 0], [0, 5], [10, 5], [20, 5]], dtype=float)
    y = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    seasons = np.array([2020, 2020, 2020, 2020, 2021, 2021])
    splits = [(np.array([0, 1, 2]), np.array([3])), (np.array([0, 1, 2]), np.array([4, 5]))]
    with pytest.raises(ValueError, match="fold 1"):
        IDWBaseline.tune(xy, y, cv_splits=splits, p_grid=(2.0,), k_grid=(1,), seasons=seasons)
    # Doi chung: bo fold hong -> chay, dem du fold
    m = IDWBaseline.tune(xy, y, cv_splits=splits[:1], p_grid=(2.0,), k_grid=(1,), seasons=seasons)
    assert m.cv_n_folds_ == 1 and m.cv_scores_[(2.0, 1)] == pytest.approx(0.5)


def test_idw_tune_nhan_cv_splits_dung_san():
    xy = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]], dtype=float)
    y = xy[:, 0] / 10.0
    splits = [(np.array([0, 1, 2]), np.array([3, 4, 5])), (np.array([3, 4, 5]), np.array([0, 1, 2]))]
    m = IDWBaseline.tune(xy, y, cv_splits=splits, p_grid=(2.0,), k_grid=(1,))
    # k=1: du doan diem 3,4,5 bang y[2]=2 -> sai 1,2,3; nguoc lai 0,1,2 bang y[3]=3 -> 3,2,1 -> MAE 2
    assert m.cv_scores_[(2.0, 1)] == pytest.approx(2.0)


def test_idw_toa_do_lap_qua_mua_bao_loi_va_fit_theo_mua():
    # Cung o qua 2 mua: fit chung -> d = 0 lay nhan nam khac cua chinh o (V4b) -> phai loi.
    xy = np.array([[0, 0], [100, 0], [0, 0], [100, 0]], dtype=float)
    y = np.array([1.0, 3.0, 10.0, 30.0])
    seasons = np.array([2019, 2019, 2020, 2020])
    with pytest.raises(ValueError, match="trung lap"):
        IDWBaseline(p=2, k=2).fit(xy, y)
    pred = IDWBaseline(p=2, k=2).predict_by_season(
        xy[[0, 1]], y[[0, 1]], seasons[[0, 1]], [[50, 0], [50, 0], [50, 0]], [2019, 2020, 2021])
    # mua 2019 chi dung nhan 2019 (trung binh 2); mua 2020/2021 khong co diem huan luyen -> NaN
    assert pred[0] == pytest.approx(2.0)
    assert np.isnan(pred[1]) and np.isnan(pred[2])


def test_persistence_uses_value_strictly_before_each_date():
    hist_cells = ["a", "a", "a", "b"]
    hist_dates = ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-01"]
    hist_y = [1.0, 2.0, 5.0, 9.0]
    m = PersistenceBaseline().fit(hist_cells, hist_dates, hist_y)
    pred = m.predict(["a", "a", "a", "c"], ["2020-01-02", "2020-01-04", "2020-01-06", "2020-01-03"])
    # ngay 02 -> lay 01 (khong lay chinh ngay 02); ngay 04 -> 02; ngay 06 -> 05; o 'c' -> trung binh
    assert list(pred) == [1.0, 2.0, 5.0, pytest.approx(np.mean(hist_y))]


# ---------------- D. Danh gia theo diem + kiem dinh ----------------
def test_assign_points_tie_rule_and_outside():
    grid = gpd.GeoDataFrame({"cell_id": ["z_right", "a_left"]},
                            geometry=[box(1, 0, 2, 1), box(0, 0, 1, 1)], crs="EPSG:4326")
    pts = gpd.GeoDataFrame({"point_id": ["edge", "left", "far"]},
                           geometry=[Point(1, 0.5), Point(0.5, 0.5), Point(5, 5)], crs="EPSG:4326")
    cells = assign_points_to_cells(pts, grid)
    assert cells["edge"] == "a_left"   # tren canh chung -> id nho nhat
    assert cells["left"] == "a_left"
    assert pd.isna(cells["far"])


def test_point_errors_co_dau_chi_muc_point_season():
    point_cell = pd.Series({"p1": "c1", "p2": "c2", "p3": np.nan})
    cell_pred = pd.Series({("c1", 2020): 5.0, ("c2", 2020): 1.0, ("c1", 2021): 2.0})
    truth = pd.Series({("p1", 2020): 4.0, ("p2", 2020): 3.0, ("p3", 2020): 1.0,
                       ("p1", 2021): 4.0, ("p2", 2021): 3.0})
    err = point_errors(point_cell, cell_pred, truth)
    assert list(err.index.names) == ["point_id", "season"]
    assert err[("p1", 2020)] == 1.0      # 5 - 4
    assert err[("p2", 2020)] == -2.0     # 1 - 3: giu dau, khong lay tri tuyet doi
    assert err[("p1", 2021)] == -2.0     # du doan dung mua 2021 cua o c1
    assert pd.isna(err[("p3", 2020)])    # ngoai luoi -> NaN, khong dien 0
    assert pd.isna(err[("p2", 2021)])    # o c2 khong co du doan mua 2021 -> NaN
    with pytest.raises(ValueError, match="2 cap"):
        point_errors(point_cell, pd.Series({"c1": 5.0}), truth)


def test_holm_matches_hand_computation():
    # sap xep 0.01, 0.03, 0.04 -> 0.03, 0.06, max(0.06, 0.04) = 0.06
    assert list(holm_adjust([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    assert list(holm_adjust([0.5, 0.9])) == pytest.approx([1.0, 1.0])


def test_holm_nan_bao_loi():
    # Truoc day: [0.01, nan, 0.04] -> [0.03, 0.08, 0.08] (NaN bi tinh nhu p hop le, sai m va thu hang).
    with pytest.raises(ValueError, match="NaN"):
        holm_adjust([0.01, np.nan, 0.04])


def _series(vals, ids):
    return pd.Series(vals, index=pd.MultiIndex.from_tuples(ids, names=["point_id", "season"]))


def test_wilcoxon_and_bootstrap_detect_shift():
    rng = np.random.default_rng(0)
    ids = [(f"p{i}", 2020) for i in range(300)]
    a = _series(rng.uniform(0, 1, 300), ids)
    b = a + 0.2 + rng.normal(0, 0.01, 300)
    w = paired_wilcoxon(a, b)
    assert w["n"] == 300 and w["p_value"] < 1e-10
    ci = bootstrap_median_diff_ci(a, b, n_boot=500, seed=1)
    assert ci["ci_low"] <= ci["median_diff"] <= ci["ci_high"] < 0
    assert ci == bootstrap_median_diff_ci(a, b, n_boot=500, seed=1)  # co dinh seed -> tai lap
    assert paired_wilcoxon(a, a)["p_value"] == 1.0


def test_ghep_cap_theo_chi_muc_khi_dao_thu_tu():
    ids = [("p1", 2020), ("p2", 2020), ("p3", 2020), ("p1", 2021), ("p2", 2021), ("p3", 2021)]
    a = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ids)
    b = a + np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.05])  # b - a that, trung vi 0.25
    b_rev = b.iloc[::-1]                                 # dao thu tu dong
    ci = bootstrap_median_diff_ci(a, b_rev, n_boot=200, seed=1)
    # ghep theo vi tri (ban cu): a - b_rev = [-5.05, -3.3, -1.8, 0.8, 2.1, 4.9] -> trung vi -0.5
    assert ci["median_diff"] == pytest.approx(-0.25)
    assert ci == bootstrap_median_diff_ci(a, b, n_boot=200, seed=1)
    assert paired_wilcoxon(a, b_rev) == paired_wilcoxon(a, b)
    # chi muc khac nhau -> loi; mang khong chi muc -> tu choi
    with pytest.raises(ValueError, match="khac tap chi muc"):
        paired_wilcoxon(a, b.rename(index={"p3": "p9"}))
    with pytest.raises(ValueError, match="khac tap chi muc"):
        bootstrap_median_diff_ci(a, b.iloc[:5])
    with pytest.raises(TypeError):
        paired_wilcoxon(a.to_numpy(), b.to_numpy())

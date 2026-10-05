"""Kiem thu kiem dinh theo khoi (src/training/block_stats.py). Du lieu TONG HOP, gia tri biet truoc."""
import itertools
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.block_stats import (  # noqa: E402
    area_levels, cluster_t_ci, common_point_set, compare_family, comparison_seed, family_verdict,
    pairs_by_area, seed_mean_errors, signflip_test, unit_metrics,
)


# ---------------- sign-flip ----------------
def test_signflip_g3_liet_ke_tay_du_8_to_hop():
    # d = [3, -1, 2], w = [1, 1, 2] -> w chuan hoa [.25, .25, .5], w*d = [.75, -.25, 1.0]
    # Delta = 1.5. Liet ke tay |sum s*w*d|:
    #  (+,+,+)  .75-.25+1 = 1.5   >= 1.5
    #  (+,+,-)  .75-.25-1 = -.5
    #  (+,-,+)  .75+.25+1 = 2.0   >= 1.5
    #  (+,-,-)  .75+.25-1 = 0.0
    #  (-,+,+) -.75-.25+1 = 0.0
    #  (-,+,-) -.75-.25-1 = -2.0  >= 1.5
    #  (-,-,+) -.75+.25+1 = .5
    #  (-,-,-) -.75+.25-1 = -1.5  >= 1.5
    # -> 4/8 = 0.5
    res = signflip_test([3.0, -1.0, 2.0], [1.0, 1.0, 2.0])
    assert res["delta_hat"] == pytest.approx(1.5)
    assert res["p_value"] == pytest.approx(0.5)
    assert res["exact"] and res["n_perm"] == 8
    # doi chieu bang vong lap doc lap
    wd = np.array([0.75, -0.25, 1.0])
    hand = np.mean([abs(np.dot(s, wd)) >= 1.5 - 1e-12 for s in itertools.product([-1, 1], repeat=3)])
    assert hand == 0.5
    # cung dau het -> p nho nhat = 2/2^G
    assert signflip_test([1.0, 2.0, 4.0], [1, 1, 1])["p_value"] == pytest.approx(2 / 8)
    # d = 0 -> p = 1
    assert signflip_test([0.0, 0.0, 0.0], [1, 2, 3])["p_value"] == 1.0


def test_signflip_bao_loi_nan_va_trong_so():
    with pytest.raises(ValueError):
        signflip_test([1.0, np.nan], [1, 1])
    with pytest.raises(ValueError):
        signflip_test([1.0, 2.0], [1, 0])
    with pytest.raises(ValueError):
        signflip_test([], [])


def test_signflip_monte_carlo_khi_g_lon_hon_20_tai_lap_theo_seed():
    rng = np.random.default_rng(5)
    d = rng.normal(0, 1, 25)
    w = rng.integers(100, 900, 25).astype(float)
    with pytest.raises(ValueError, match="seed"):
        signflip_test(d, w)
    r1 = signflip_test(d, w, seed=comparison_seed("F1:a-vs-b"), n_perm=5000)
    r2 = signflip_test(d, w, seed=comparison_seed("F1:a-vs-b"), n_perm=5000)
    assert not r1["exact"] and r1["n_perm"] == 5000
    assert r1["p_value"] == r2["p_value"] and 0 < r1["p_value"] <= 1
    # cung dau het -> p = 1/(n+1)
    assert signflip_test(np.abs(d) + 0.1, w, seed=1, n_perm=5000)["p_value"] == pytest.approx(1 / 5001)
    # G = 20 van liet ke du (2^20)
    assert signflip_test(d[:20], w[:20])["exact"]


def test_comparison_seed_khoa_theo_ma_khong_theo_thu_tu():
    a1 = comparison_seed("F1:h3_res_5|hgb-vs-latlon|hgb")
    a2 = comparison_seed("F1:h3_res_5|hgb-vs-latlon|hgb")
    b = comparison_seed("F1:h3_res_5|hgb-vs-utm|hgb")
    assert a1.generate_state(4).tolist() == a2.generate_state(4).tolist()
    assert a1.generate_state(4).tolist() != b.generate_state(4).tolist()
    assert a1.entropy == 42


def test_signflip_delta_0_co_hieu_ung_khoi_bac_bo_khong_qua_alpha():
    # Du lieu gia: 8 khoi, hieu ung khoi rieng cho moi khung (u ~ N(0, 0,3)) + nhieu diem;
    # A va B hoan doi duoc trong khoi -> Delta that = 0. Lap 500 lan, seed co dinh.
    rng = np.random.default_rng(20261003)
    sizes = np.array([837, 835, 833, 340, 119, 60, 200, 400])
    alpha, R = 0.05, 500
    rej_block = rej_point = 0
    for _ in range(R):
        d = np.empty(len(sizes))
        diff_pts = []
        for b, n in enumerate(sizes):
            base = rng.normal(0, 1, n)
            ea = np.abs(base + rng.normal(0, 0.3) + rng.normal(0, 0.3, n))
            eb = np.abs(base + rng.normal(0, 0.3) + rng.normal(0, 0.3, n))
            d[b] = ea.mean() - eb.mean()
            diff_pts.append(ea - eb)
        rej_block += signflip_test(d, sizes.astype(float))["p_value"] < alpha
        rej_point += wilcoxon(np.concatenate(diff_pts)).pvalue < alpha
    rate = rej_block / R
    # sai so Monte Carlo 500 lan: 2 SE ~ 0,0195
    assert rate <= alpha + 2 * np.sqrt(alpha * (1 - alpha) / R)
    # doi chung: Wilcoxon cap diem bac bo qua muc (ly do khong dung o cap diem)
    assert rej_point / R > 0.3


# ---------------- CI t robust theo cum ----------------
def test_cluster_t_ci_khop_tinh_tay():
    # d = [1, 2, 4], w = [1, 1, 2] -> w = [.25, .25, .5]; Delta = .25 + .5 + 2 = 2.75
    # d - Delta = [-1.75, -.75, 1.25]; w(d - Delta) = [-.4375, -.1875, .625]
    # tong binh phuong = .19140625 + .03515625 + .390625 = .6171875; x G/(G-1) = 1.5 -> .92578125
    # se = sqrt(.92578125) = 0.96217527; q = t_{0.975, 2} = 4.30265273
    res = cluster_t_ci([1.0, 2.0, 4.0], [1.0, 1.0, 2.0], alpha=0.05)
    se = np.sqrt(0.92578125)
    q = 4.302652729749462
    assert res["delta_hat"] == pytest.approx(2.75)
    assert res["se"] == pytest.approx(se)
    assert res["df"] == 2
    assert res["ci_low"] == pytest.approx(2.75 - q * se)
    assert res["ci_high"] == pytest.approx(2.75 + q * se)
    with pytest.raises(ValueError):
        cluster_t_ci([1.0], [1.0], alpha=0.05)


# ---------------- tap diem chung ----------------
def test_common_point_set_loai_diem_thieu_o_mot_luoi_cho_moi_luoi():
    rows = []
    for g in ("g1", "g2", "g3"):
        for p, blk in (("p1", "A"), ("p2", "A"), ("p3", "B")):
            for s in (2020, 2021):
                if g == "g2" and p == "p3" and s == 2020:
                    continue  # thieu dong o g2 -> khong hop le o g2
                valid = not (g == "g3" and p == "p2" and s == 2021)
                rows.append({"grid": g, "point_id": p, "season": s, "valid": valid, "block": blk})
    common, dropped = common_point_set(pd.DataFrame(rows))
    assert list(common) == [("p1", 2020), ("p1", 2021), ("p2", 2020), ("p3", 2021)]
    t = dropped.set_index(["grid", "season", "unit"])
    # p2/2021 khong hop le o g3 -> bi loai o MOI luoi
    assert t.loc[("g1", 2021, "A"), "n_invalid_grid"] == 0
    assert t.loc[("g1", 2021, "A"), "n_dropped_common"] == 1
    assert t.loc[("g3", 2021, "A"), "n_invalid_grid"] == 1
    # p3/2020 thieu o g2
    assert t.loc[("g2", 2020, "B"), "n_invalid_grid"] == 1
    assert t.loc[("g1", 2020, "B"), "n_dropped_common"] == 1
    assert t.loc[("g1", 2020, "B"), "n_common"] == 0
    with pytest.raises(ValueError, match="trung lap"):
        common_point_set(pd.DataFrame(rows + [rows[0]]))


# ---------------- seed + don vi ----------------
def test_seed_mean_errors_va_bao_loi():
    df = pd.DataFrame({"grid": "g", "model": "m", "seed": [1, 2, 1, 2], "point_id": ["p", "p", "q", "q"],
                       "season": 2020, "err": [1.0, -3.0, 2.0, 2.0]})
    out = seed_mean_errors(df).set_index("point_id")
    assert out.loc["p", "abs_err"] == 2.0 and out.loc["p", "sq_err"] == 5.0 and out.loc["p", "mean_err"] == -1.0
    with pytest.raises(ValueError, match="NaN"):
        seed_mean_errors(df.assign(err=[1.0, np.nan, 2.0, 2.0]))
    with pytest.raises(ValueError, match="thieu seed"):
        seed_mean_errors(df.iloc[:3])
    with pytest.raises(ValueError, match="trung lap"):
        seed_mean_errors(pd.concat([df, df.iloc[:1]]))



def test_unit_metrics_bo_unit_mua_it_diem_cho_moi_luoi_va_trung_binh_deu_theo_mua():
    rows = []
    # u1: mua 2020 co 40 diem (|e| A = 1), mua 2021 co 30 diem (|e| A = 3) -> trung binh deu = 2 (khong phai 1,86)
    # u2: mua 2020 chi 10 diem -> bo cho ca A va B; mua 2021 35 diem
    # u3: 5 diem ca hai mua -> bi bo HOAN TOAN (vd khoi giu rieng nho) -> phai duoc bao
    for season, n, ea in ((2020, 40, 1.0), (2021, 30, 3.0)):
        for i in range(n):
            rows += [("A", "m", f"u1_{i}", season, ea, ea ** 2), ("B", "m", f"u1_{i}", season, 1.0, 1.0)]
    for season, n in ((2020, 10), (2021, 35)):
        for i in range(n):
            rows += [("A", "m", f"u2_{i}", season, 2.0, 4.0), ("B", "m", f"u2_{i}", season, 2.0, 4.0)]
    for season in (2020, 2021):
        for i in range(5):
            rows += [("A", "m", f"u3_{i}", season, 2.0, 4.0), ("B", "m", f"u3_{i}", season, 2.0, 4.0)]
    err = pd.DataFrame(rows, columns=["grid", "model", "point_id", "season", "abs_err", "sq_err"])
    pu = pd.Series({p: p.split("_")[0] for p in err["point_id"].unique()})
    units, seasons, dropped = unit_metrics(err, pu, min_pts=30)
    u = units.set_index(["grid", "unit"])
    assert u.loc[("A", "u1"), "mae"] == pytest.approx(2.0)
    assert u.loc[("A", "u1"), "mse"] == pytest.approx(5.0)
    assert u.loc[("A", "u1"), "n_pts_mean"] == pytest.approx(35.0)
    assert u.loc[("A", "u2"), "n_seasons"] == 1 and u.loc[("B", "u2"), "n_seasons"] == 1
    assert u.loc[("A", "u2"), "n_pts_mean"] == 35
    assert "u3" not in units["unit"].tolist()
    assert not seasons.query("unit == 'u2' and season == 2020")["kept"].any()
    # bang bi loai: moi (unit, mua) mot dong (khong nhan theo luoi)
    dd = dropped.set_index(["unit", "season"])
    assert sorted(dd.index) == [("u2", 2020), ("u3", 2020), ("u3", 2021)]
    assert dd.loc[("u2", 2020), "n_pts"] == 10 and not dd.loc[("u2", 2020), "unit_fully_dropped"]
    assert dd.loc[("u3", 2021), "unit_fully_dropped"]
    # mot luoi thieu 1 (diem, mua) -> assert
    with pytest.raises(AssertionError, match="khac tap"):
        unit_metrics(err.iloc[1:], pu, min_pts=30)


# ---------------- cap theo dien tich ----------------
AREA_CSV = os.path.join(os.path.dirname(__file__), "..", "KE_HOACH", "ket-qua", "tuan1_doi_chieu_dien_tich.csv")


def _area_df():
    # trich bang dien tich that (13 luoi); khong phu thuoc file de test chay ca khi thieu KE_HOACH
    rows = [("h3_res_5", 5, 291.9751), ("h3_res_6", 6, 41.7066), ("h3_res_7", 7, 5.9578),
            ("s2_level_9", 5, 394.4811), ("s2_level_10", 6, 98.64), ("s2_level_11", 6, 24.6601),
            ("s2_level_12", 7, 6.1649), ("square_utm_17087m", 5, 292.1366), ("square_utm_6458m", 6, 41.7304),
            ("square_utm_2441m", 7, 5.962), ("latlon_0.1552deg", 5, 292.1067), ("latlon_0.0586deg", 6, 41.6432),
            ("latlon_0.0222deg", 7, 5.9765)]
    return pd.DataFrame([{"file": f"data/grids/{g}.geojson", "tier_h3_res": t, "mean_area_km2": a}
                         for g, t, a in rows])


def test_pairs_by_area_15_cap_voi_ti_le_1_5():
    pairs = pairs_by_area(_area_df(), max_ratio=1.5, expected_m=15)
    assert len(pairs) == 15
    by_tier = pairs.groupby("tier").size().to_dict()
    # muc 5: H3/S2 L9/UTM/LatLon (S2 L9 / H3 = 1,35) -> 6; muc 6: S2 L10 (98,6) va L11 (24,7) lech > 1,5 -> 3; muc 7 -> 6
    assert by_tier == {5: 6, 6: 3, 7: 6}
    names = set(pairs["grid_a"]) | set(pairs["grid_b"])
    assert "s2_level_10" not in names and "s2_level_11" not in names
    assert (pairs["area_ratio"] <= 1.5).all()
    assert pairs.query("grid_a == 'h3_res_5' and grid_b == 's2_level_9'")["area_ratio"].iloc[0] == pytest.approx(
        394.4811 / 291.9751)
    with pytest.raises(AssertionError, match="ky vong 18"):
        pairs_by_area(_area_df(), max_ratio=1.5, expected_m=18)
    # nguong chat hon: S2 L9 (1,35) roi ra -> 15 - 3 = 12
    assert len(pairs_by_area(_area_df(), max_ratio=1.2)) == 12
    # muc tu bang: muc 6 gom ca S2 L10/L11 neu khong loc theo cap
    lv = area_levels(_area_df())
    assert lv["s2_level_10"] == 6 and len(lv) == 13
    assert "s2_level_10" not in area_levels(_area_df(), pairs=pairs).index


@pytest.mark.skipif(not os.path.exists(AREA_CSV), reason="thieu bang dien tich KE_HOACH")
def test_pairs_by_area_doc_file_that():
    assert len(pairs_by_area(AREA_CSV, max_ratio=1.5, expected_m=15)) == 15


# ---------------- ho so sanh ----------------
def _mk(unit_n, err_of, ho_units=(), grids=("A", "B"), seeds=(1, 2, 3),
        cv_seasons=(2019, 2021), ho_seasons=(2019, 2020, 2021)):
    """unit_n: {unit: so_diem}; err_of(grid, unit, seed) -> sai so co dau.

    Khoi CV: du doan 'oof', khong co mua giu rieng 2020. Khoi giu rieng: 'final', moi mua.
    """
    rows, unit_of = [], {}
    for u, n in unit_n.items():
        ho = u in ho_units
        for i in range(n):
            pid = f"{u}_{i}"
            unit_of[pid] = u
            for season in (ho_seasons if ho else cv_seasons):
                for s in seeds:
                    for g in grids:
                        rows.append((g, "hgb", s, pid, season, err_of(g, u, s), "final" if ho else "oof"))
    df = pd.DataFrame(rows, columns=["grid", "model", "seed", "point_id", "season", "err", "pred_source"])
    cv = [u for u in unit_n if u not in ho_units]
    return df, pd.Series(unit_of), cv, list(ho_units)


def _ab(d, shift=None, other=None):
    """A: |e| = 1 + d[u] (+ lech theo seed); B: |e| = 1 (dau am); luoi khac: other[g](u, s)."""
    shift = shift or {}
    other = other or {}

    def f(g, u, s):
        if g == "A":
            return 1.0 + d[u] + shift.get(s, 0.0)
        if g in other:
            return other[g](u, s)
        return -1.0
    return f


def _cv_only(df, pu, ho):
    return df[~df["point_id"].map(pu).isin(ho)]


KW = dict(delta_min=0.05, delta_min_kind="abs", alpha=0.05, holdout_seasons=(2020,), model="hgb")


def test_compare_family_moi_suy_dien_chi_tren_khoi_cv():
    # V1: truoc day khoi giu rieng nam trong d/w/p -> "giu rieng cung chieu" tu thoa.
    d_cv = [-0.1, -0.2, -0.3, -0.15, -0.25, -0.05]
    n_cv = [30 + 5 * i for i in range(6)]
    d = {f"u{i}": x for i, x in enumerate(d_cv)}
    d.update({"h0": -0.2, "h1": -0.1})
    unit_n = {f"u{i}": n for i, n in enumerate(n_cv)}
    unit_n.update({"h0": 40, "h1": 35})
    df, pu, cv, ho = _mk(unit_n, _ab(d), ho_units=("h0", "h1"))
    r = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW).iloc[0]
    w = np.array(n_cv, dtype=float)
    assert r["G"] == 6 and r["k_over_G"] == "6/6"
    assert r["delta_hat"] == pytest.approx((w / w.sum() * np.array(d_cv)).sum())  # -0,1725, chi CV
    assert r["p_value"] == pytest.approx(2 / 64) and r["p_holm"] == pytest.approx(2 / 64)
    assert r["delta_holdout"] == pytest.approx((40 * -0.2 + 35 * -0.1) / 75)
    assert r["n_holdout_units"] == 2 and r["holdout_same_dir"]
    assert r["seeds_same_dir"] and r["better"] == "A" and r["ci_high"] < 0
    assert r["label"] == "xac_nhan"
    assert r["seasons_cv"] == "2019;2021" and r["seasons_holdout"] == "2019;2020;2021"
    assert r["holdout_seasons"] == "2020"


HO6 = {f"h{i}": n for i, n in enumerate((60, 60, 60, 40, 35, 31))}


@pytest.mark.parametrize("cv_sign, ho_d", [(+1, -0.30), (-1, +0.25)])
def test_compare_family_kb1_kb2_khong_tai_lap(cv_sign, ho_d):
    # KB1: CV noi B tot hon (d > 0, co y nghia), giu rieng noi A tot hon manh.
    # KB2: CV noi A tot hon, giu rieng nguoc chieu manh. Truoc khi sua: gop 26 khoi -> Delta
    # bi khoi giu rieng keo, p ~0,2 -> "chua_phan_dinh" (che mat mau thuan). Dung: khong_tai_lap.
    d = {f"u{i}": cv_sign * (0.03 + 0.01 * np.sin(i)) for i in range(20)}
    d.update({h: ho_d for h in HO6})
    unit_n = {f"u{i}": 30 for i in range(20)}
    unit_n.update(HO6)
    df, pu, cv, ho = _mk(unit_n, _ab(d), ho_units=tuple(HO6), cv_seasons=(2019,), ho_seasons=(2020,))
    r = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW).iloc[0]
    assert r["G"] == 20
    assert np.sign(r["delta_hat"]) == cv_sign and r["p_holm"] < 0.05
    assert np.sign(r["delta_holdout"]) == -cv_sign
    assert r["delta_hat"] == pytest.approx(np.mean([d[f"u{i}"] for i in range(20)]))
    assert r["label"] == "khong_tai_lap"


def test_compare_family_p_cv_khong_doi_khi_them_bot_khoi_giu_rieng():
    d = {f"u{i}": x for i, x in enumerate([-0.1, 0.05, -0.2, -0.12, 0.03, -0.08, -0.15, -0.02])}
    d.update({"h1": 0.2, "h2": -0.4})
    unit_n = {f"u{i}": 30 + 3 * i for i in range(8)}
    unit_n.update({"h1": 40, "h2": 35})
    df, pu, cv, _ = _mk(unit_n, _ab(d), ho_units=("h1", "h2"))
    kw = dict(KW, delta_min_kind="rel", delta_ref="level_mean", levels={"A": 7, "B": 7})
    r_both = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=["h1", "h2"], mode="final", **kw)
    keep = pu[pu != "h2"]
    r_one = compare_family(df[df["point_id"].isin(keep.index)], keep, [("A", "B")], cv_units=cv,
                           holdout_units=["h1"], mode="final", **kw)
    r_cv = compare_family(_cv_only(df, pu, ["h1", "h2"]), pu, [("A", "B")], cv_units=cv,
                          holdout_units=["h1", "h2"], mode="cv", **kw)
    cols = ["G", "delta_hat", "p_value", "p_holm", "ci_low", "ci_high", "ci_tost_low", "ci_tost_high", "se",
            "k_over_G", "seed_deltas", "delta_ref", "delta_min_thr"]
    pd.testing.assert_frame_equal(r_both[cols], r_one[cols])
    pd.testing.assert_frame_equal(r_both[cols], r_cv[cols])
    # nhung Delta_holdout thi khac (khoi giu rieng chi xac nhan chieu)
    assert r_both.iloc[0]["delta_holdout"] != pytest.approx(r_one.iloc[0]["delta_holdout"])
    assert np.isnan(r_cv.iloc[0]["delta_holdout"])


def test_compare_family_mode_cv_tu_choi_vung_giu_rieng_va_nhan_cho():
    d = {f"u{i}": -0.1 - 0.01 * i for i in range(6)}
    d["h0"] = -0.1
    unit_n = {f"u{i}": 30 for i in range(6)}
    unit_n["h0"] = 30
    df, pu, cv, ho = _mk(unit_n, _ab(d), ho_units=("h0",))
    with pytest.raises(ValueError, match="mode='cv'"):
        compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="cv", **KW)
    r = compare_family(_cv_only(df, pu, ho), pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="cv", **KW)
    assert r.iloc[0]["label"] == "cho_giu_rieng" and r.iloc[0]["seasons_holdout"] == ""
    with pytest.raises(ValueError, match="mode phai"):
        compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="test", **KW)
    with pytest.raises(ValueError, match="mode='final'"):
        compare_family(_cv_only(df, pu, ho), pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW)


def test_compare_family_kiem_nguon_du_doan_mua_va_don_vi():
    d = {f"u{i}": -0.1 for i in range(6)}
    d["h0"] = -0.1
    unit_n = {f"u{i}": 30 for i in range(6)}
    unit_n["h0"] = 30
    df, pu, cv, ho = _mk(unit_n, _ab(d), ho_units=("h0",))
    is_cv = df["point_id"].map(pu).isin(cv)
    call = lambda frame, **k: compare_family(frame, pu, [("A", "B")], mode="final",  # noqa: E731
                                             **{**dict(cv_units=cv, holdout_units=ho), **KW, **k})
    # khoi CV co mua giu rieng 2020
    with pytest.raises(ValueError, match="mua giu rieng"):
        call(pd.concat([df, df[is_cv & (df["season"] == 2019)].assign(season=2020)]))
    # nguon du doan sai
    with pytest.raises(ValueError, match="oof"):
        call(df.assign(pred_source=np.where(is_cv & (df["point_id"] == "u0_0"), "final", df["pred_source"])))
    with pytest.raises(ValueError, match="final"):
        call(df.assign(pred_source="oof"))
    with pytest.raises(ValueError, match="pred_source phai"):
        call(df.assign(pred_source="test"))
    with pytest.raises(ValueError, match="pred_source"):
        call(df.drop(columns="pred_source"))
    # don vi: giao nhau / bo sot / khong ton tai
    with pytest.raises(ValueError, match="giao nhau"):
        call(df, holdout_units=["h0", "u0"])
    with pytest.raises(ValueError, match="khong thuoc"):
        call(df, cv_units=cv[1:])
    with pytest.raises(ValueError, match="khong co trong point_unit"):
        call(df, holdout_units=["h0", "h_xyz"])
    with pytest.raises(ValueError, match="holdout_seasons"):
        call(df, holdout_seasons=())


def test_compare_family_bao_khoi_giu_rieng_bi_loai_vi_it_diem():
    d = {f"u{i}": -0.1 - 0.01 * i for i in range(6)}
    d.update({"h0": -0.1, "h_nho": 0.5})
    unit_n = {f"u{i}": 30 for i in range(6)}
    unit_n.update({"h0": 30, "h_nho": 10})  # < min_pts -> bi loai, khong lam doi chieu
    df, pu, cv, ho = _mk(unit_n, _ab(d), ho_units=("h0", "h_nho"))
    with pytest.warns(UserWarning, match="khoi giu rieng bi loai.*h_nho"):
        r = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW).iloc[0]
    assert r["holdout_units_dropped"] == "h_nho" and r["n_holdout_units"] == 1
    assert r["delta_holdout"] == pytest.approx(-0.1) and r["cv_units_dropped"] == ""


def test_compare_family_chua_phan_dinh_khi_seed_khong_cung_chieu():
    d = {f"u{i}": x for i, x in enumerate([-0.1, -0.2, -0.3, -0.15, -0.25, -0.05])}
    d["h0"] = -0.2
    unit_n = {u: 30 for u in d}
    df, pu, cv, ho = _mk(unit_n, _ab(d, shift={1: -0.3, 2: -0.3, 3: 0.6}), ho_units=("h0",))  # seed 3: A te hon
    r = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW).iloc[0]
    assert r["p_holm"] < 0.05 and not r["seeds_same_dir"] and r["holdout_same_dir"]
    assert r["label"] == "chua_phan_dinh"


def test_compare_family_co_y_nghia_duoi_nguong_theo_require_practical():
    d = {f"u{i}": -0.01 for i in range(6)}
    d["h0"] = -0.01
    df, pu, cv, ho = _mk({u: 30 for u in d}, _ab(d), ho_units=("h0",))
    r = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW).iloc[0]
    assert r["p_holm"] < 0.05 and abs(r["delta_hat"]) < r["delta_min_thr"] and not r["practical_ok"]
    assert r["label"] == "co_y_nghia_duoi_nguong"
    r2 = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final",
                        require_practical=False, **KW).iloc[0]
    assert r2["label"] == "xac_nhan"


def test_compare_family_tuong_duong_tost_va_nguong_tuong_doi_theo_muc():
    xs = [0.004, -0.003, 0.002, -0.004, 0.001, -0.002, 0.003, -0.001, 0.002, -0.002]
    d = {f"u{i}": x for i, x in enumerate(xs)}
    d["h0"] = 0.0
    other = {"C": lambda u, s: 2.0}  # luoi thu 3 cung muc, khong vao cap: |e| = 2
    df, pu, cv, ho = _mk({u: 40 for u in d}, _ab(d, other=other), ho_units=("h0",), grids=("A", "B", "C"),
                         cv_seasons=(2019,))
    r = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW).iloc[0]
    assert r["p_holm"] > 0.05 and r["tost_tuong_duong"] and r["label"] == "tuong_duong"
    assert np.isnan(r["delta_ref"])
    # rel + level_mean: MOT tham chieu cho muc = trung binh MAE (CV) cua MOI luoi trong muc (A, B, C)
    kw = dict(KW, delta_min_kind="rel")
    r2 = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final",
                        levels={"A": 5, "B": 5, "C": 5}, **kw).iloc[0]
    ref = ((1 + r2["delta_hat"]) + 1.0 + 2.0) / 3
    assert r2["delta_ref"] == pytest.approx(ref) and r2["delta_min_thr"] == pytest.approx(0.05 * ref)
    assert r2["delta_ref_level"] == 5 and r2["delta_ref_grids"] == "A|hgb;B|hgb;C|hgb"
    # rel + so truyen thang
    r3 = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", delta_ref=2.0,
                        **kw).iloc[0]
    assert r3["delta_min_thr"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="levels"):
        compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **kw)
    with pytest.raises(ValueError, match="khac muc"):
        compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final",
                       levels={"A": 5, "B": 6}, **kw)
    # nguong qua nho -> khong con tuong duong
    r4 = compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final",
                        **dict(KW, delta_min=1e-4)).iloc[0]
    assert r4["label"] == "chua_phan_dinh"


def test_compare_family_tham_so_bat_buoc_expected_m_va_holm_trong_ho():
    d = {f"u{i}": -0.1 - 0.01 * i for i in range(6)}
    d["h0"] = -0.1
    other = {"C": lambda u, s: -1.0}  # C == B
    df, pu, cv, ho = _mk({u: 30 for u in d}, _ab(d, other=other), ho_units=("h0",), grids=("A", "B", "C"),
                         seeds=(1,))
    with pytest.raises(TypeError):
        compare_family(df, pu, [("A", "B")], model="hgb")  # thieu cv_units/holdout_units/mode/delta_min/alpha
    with pytest.raises(TypeError):
        compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", model="hgb",
                       delta_min=0.05, delta_min_kind="abs", alpha=0.05)  # thieu holdout_seasons
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    out = compare_family(df, pu, pairs, cv_units=cv, holdout_units=ho, mode="final", expected_m=3, **KW)
    # p = [2/64, 2/64, 1 (B == C)] -> Holm [6/64, 6/64, 1]
    assert out["p_value"].tolist() == pytest.approx([2 / 64, 2 / 64, 1.0])
    assert out["p_holm"].tolist() == pytest.approx([6 / 64, 6 / 64, 1.0])
    assert out["label"].tolist() == ["chua_phan_dinh", "chua_phan_dinh", "tuong_duong"]
    with pytest.raises(AssertionError, match="expected_m=4"):
        compare_family(df, pu, pairs, cv_units=cv, holdout_units=ho, mode="final", expected_m=4, **KW)


def test_compare_family_nhan_cap_tu_pairs_by_area():
    d = {f"u{i}": -0.1 for i in range(6)}
    d["h0"] = -0.1
    df, pu, cv, ho = _mk({u: 30 for u in d}, _ab(d), ho_units=("h0",), seeds=(1,))
    pairs = pd.DataFrame({"grid_a": ["A"], "grid_b": ["B"], "area_ratio": [1.35]})
    r = compare_family(df, pu, pairs, cv_units=cv, holdout_units=ho, mode="final", **KW).iloc[0]
    assert r["area_ratio"] == 1.35 and "flag" not in r.index


def test_compare_family_assert_hai_luoi_cung_tap_chi_muc():
    d = {f"u{i}": -0.1 for i in range(6)}
    d["h0"] = -0.1
    df, pu, cv, ho = _mk({u: 30 for u in d}, _ab(d), ho_units=("h0",), seeds=(1,))
    bad = df.drop(df[(df["grid"] == "B") & (df["point_id"] == "u0_0")].index)
    with pytest.raises(AssertionError, match="khac tap"):
        compare_family(bad, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW)
    bad_ho = df.drop(df[(df["grid"] == "B") & (df["point_id"] == "h0_0")].index)
    with pytest.raises(AssertionError, match="khac tap"):
        compare_family(bad_ho, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW)
    bad_seed = pd.concat([df, df[df["grid"] == "A"].assign(seed=2)])
    with pytest.raises(ValueError, match="seed"):
        compare_family(bad_seed, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW)


# ---------------- ket luan cap ho ----------------
def _family_out(seed_shift_a=None, d_a=-0.2, d_d=-0.2, ho_a=-0.2, ho_d=0.3):
    # 8 khoi CV (p = 2/256, Holm m = 2 -> 1/64 < 0,05); 2 khoi giu rieng.
    # (A, B): giu rieng cung chieu; (D, C): giu rieng nguoc chieu.
    units = [f"u{i}" for i in range(8)]
    da = {u: d_a for u in units}
    da.update({"h0": ho_a, "h1": ho_a})
    dd = {u: d_d for u in units}
    dd.update({"h0": ho_d, "h1": ho_d})
    other = {"C": lambda u, s: -1.0, "D": lambda u, s: 1.0 + dd[u]}
    df, pu, cv, ho = _mk({u: 30 for u in da}, _ab(da, shift=seed_shift_a, other=other), ho_units=("h0", "h1"),
                         grids=("A", "B", "C", "D"))
    return compare_family(df, pu, [("A", "B"), ("D", "C")], cv_units=cv, holdout_units=ho, mode="final",
                          expected_m=2, **KW)


def test_family_verdict_ti_le_tai_lap_va_ghi_de_nhan():
    out = _family_out()
    assert out["label"].tolist() == ["xac_nhan", "khong_tai_lap"]
    res, sm = family_verdict(out, min_frac=0.8)
    assert (sm["numerator"], sm["denominator"]) == (1, 2) and sm["frac"] == 0.5
    assert sm["label"] == "khong_tai_lap" and (res["family_label"] == "khong_tai_lap").all()
    assert res["label"].tolist() == ["khong_tai_lap_cap_ho", "khong_tai_lap"]
    assert res["label_cap"].tolist() == ["xac_nhan", "khong_tai_lap"]
    assert out["label"].tolist() == ["xac_nhan", "khong_tai_lap"]  # khong sua dau vao
    res2, sm2 = family_verdict(out, min_frac=0.5)
    assert sm2["label"] == "tai_lap" and res2["label"].tolist() == ["xac_nhan", "khong_tai_lap"]


def test_family_verdict_seed_khong_cung_chieu_va_ho_rong():
    # seed 3 cua A: A te hon B -> (A, B) Holm dat, giu rieng cung chieu nhung seed khong cung chieu
    out = _family_out(seed_shift_a={3: 0.5})
    assert out.iloc[0]["holdout_same_dir"] and not out.iloc[0]["seeds_same_dir"]
    assert family_verdict(out, min_frac=0.5)[1]["numerator"] == 0
    assert family_verdict(out, min_frac=0.5)[1]["label"] == "khong_tai_lap"
    sm = family_verdict(out, min_frac=0.5, seed_inconsistent_counts_as_fail=False)[1]
    assert sm["numerator"] == 1 and sm["label"] == "tai_lap"
    # khong cap nao co y nghia -> nhan rong
    flat = _family_out(d_a=0.0, d_d=0.0, ho_a=0.0, ho_d=0.0)
    res, sm = family_verdict(flat)
    assert sm["denominator"] == 0 and sm["label"] == "khong_co_khac_biet_xac_nhan"
    assert family_verdict(flat, empty_label="rong")[1]["label"] == "rong"
    # mode cv -> khong dung duoc
    with pytest.raises(ValueError, match="mode='final'"):
        family_verdict(out.assign(mode="cv"))


def test_cap_co_y_nghia_khong_con_khoi_giu_rieng_la_loi():
    """CHG-22 (An 2026-10-05 21:22): Holm dat nhung moi khoi giu rieng bi loai (< min_pts) -> n_holdout_units = 0.
    Truoc day cap nay nhan "chua_phan_dinh" (cap) va vao mau so family_verdict -> keo ho ve "khong_tai_lap". Nay LOI."""
    d = {f"u{i}": -0.1 - 0.01 * i for i in range(6)}
    d["h_nho"] = -0.1
    unit_n = {f"u{i}": 30 for i in range(6)}
    unit_n["h_nho"] = 10  # < min_pts -> khoi giu rieng duy nhat bi loai
    df, pu, cv, ho = _mk(unit_n, _ab(d), ho_units=("h_nho",))
    with pytest.warns(UserWarning, match="h_nho"):
        with pytest.raises(ValueError, match="khong con khoi giu rieng"):
            compare_family(df, pu, [("A", "B")], cv_units=cv, holdout_units=ho, mode="final", **KW)
    # cap KHONG co y nghia ma khong con khoi giu rieng -> khong can xac nhan chieu -> van chay
    flat = {u: 0.0 for u in d}
    df0, pu0, cv0, ho0 = _mk(unit_n, _ab(flat), ho_units=("h_nho",))
    with pytest.warns(UserWarning, match="h_nho"):
        r = compare_family(df0, pu0, [("A", "B")], cv_units=cv0, holdout_units=ho0, mode="final", **KW).iloc[0]
    assert r["n_holdout_units"] == 0 and r["p_holm"] >= 0.05 and r["label"] in ("tuong_duong", "chua_phan_dinh")


def test_family_verdict_cap_co_y_nghia_khong_con_khoi_giu_rieng_la_loi():
    """CHG-22: family_verdict gap cap trong mau so co n_holdout_units == 0 -> LOI (khong dem vao mau so)."""
    out = _family_out()
    assert (out["p_holm"] < 0.05).all()
    bad = out.copy()
    bad.loc[1, "n_holdout_units"] = 0
    with pytest.raises(ValueError, match="khong con khoi giu rieng"):
        family_verdict(bad, min_frac=0.5)
    # cap khong co y nghia voi n_holdout_units = 0 khong anh huong
    flat = _family_out(d_a=0.0, d_d=0.0, ho_a=0.0, ho_d=0.0).assign(n_holdout_units=0)
    assert family_verdict(flat)[1]["denominator"] == 0

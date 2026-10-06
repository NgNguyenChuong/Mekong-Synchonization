"""scripts/analyze_subsets.py - S4 phan tich bo nhan phu (thiet ke An chot 2026-10-06 14:19).

Du lieu gia nho (tmp_path): 2 luoi x (bo chinh + 3 bo phu) x 3 cach chia, 8 don vi x 2 mua x 30 diem nong nghiep;
cham phu 60/90: 4 don vi CV x 10 diem (3 cach chia) + 2 don vi giu rieng x 5 diem (final s42).
Sai so bo chinh duong (0,5-1,5) -> cong hang so c vao bo phu thi |e| tang dung c -> Delta biet truoc.
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_subsets", os.path.join(ROOT, "scripts", "analyze_subsets.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GRIDS = ("g1", "g2")
UNITS = [f"blk_{i}" for i in range(8)]
SCHEMES = (42, 43, 44)
SEASONS = (2014, 2015)
SUBSETS = ("keepwater", "keepmangrove", "keep6090")
AUX_CV_UNITS = UNITS[:4]
AUX_HO_UNITS = ["ho_0", "ho_1"]
TIER = pd.Series({"g1": 5, "g2": 5})
PAIRS = pd.DataFrame({"grid_a": ["g1"], "grid_b": ["g2"]})


def _pu():
    return pd.Series({f"{u}_p{p}": u for u in UNITS for p in range(30)})


def _points(prefix, units, n):
    return [(f"{prefix}{u}_p{p}", u) for u in units for p in range(n)]


def _frame(rng, pts, src, err=None):
    rows = [(pid, s, u, 3.0, src) for pid, u in pts for s in SEASONS]
    d = pd.DataFrame(rows, columns=["point_id", "season", "unit_id", "y_ref", "pred_source"])
    d["err"] = rng.uniform(0.5, 1.5, len(d)) if err is None else err(d)
    return d


def _meta(label_set, mode, scheme, variant=None):
    return {"label_set": label_set, "mode": mode, "returncode": 0, "points_ref_variant": variant,
            "cv_folds_sha256": f"folds{scheme}", "points_ref_sha256": "ref_chinh" if variant is None else "ref_6090",
            "points_sha256": "pts_chinh" if variant is None else "pts_6090"}


def _put(exp, name, mode, df, meta, fname):
    d = exp / name / mode
    d.mkdir(parents=True)
    df.to_csv(d / fname, index=False)
    (d / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def make_exp(tmp_path, shift=None, mutate=None, meta_mut=None, aux_err=None):
    """shift(bo, scheme, unit) -> hang so cong vao err bo phu; mutate(name, df) / meta_mut(name, meta) lam hong 1 luot."""
    rng = np.random.default_rng(7)
    exp = tmp_path / "experiments"
    mutate = mutate or (lambda n, d: d)
    meta_mut = meta_mut or (lambda n, m: m)
    agri = [(f"{u}_p{p}", u) for u in UNITS for p in range(30)]
    for g in GRIDS:
        for s in SCHEMES:
            base = _frame(rng, agri, "oof")
            for bo in ("chinh",) + SUBSETS:
                d = base.copy()
                if bo != "chinh" and shift is not None:
                    d["err"] = d["err"] + d["unit_id"].map(lambda u: shift(bo, s, u))
                name = f"cv1__{g}__hist_gb__s{s}" + ("" if bo == "chinh" else f"__{bo}")
                _put(exp, name, "cv", mutate(name, d), meta_mut(name, _meta(bo, "cv", s)), "oof_points.csv")
            name = f"phu6090__{g}__hist_gb__s{s}__keep6090"
            d = _frame(rng, _points("a_", AUX_CV_UNITS, 10), "oof", aux_err)
            _put(exp, name, "cv", mutate(name, d), meta_mut(name, _meta("keep6090", "cv", s, "keep6090")),
                 "oof_points.csv")
        name = f"phu6090__{g}__hist_gb__s42__keep6090"
        d = _frame(rng, _points("f_", AUX_HO_UNITS, 5), "final", aux_err)
        _put(exp, name, "final", mutate(name, d), meta_mut(name, _meta("keep6090", "final", 42, "keep6090")),
             "final_points.csv")
    return str(exp)


def run(mod, exp, **kw):
    kw.setdefault("expected_g", 8)
    return mod.analyze(exp, "cv1", "phu6090", list(GRIDS), list(SCHEMES), list(SUBSETS), PAIRS, TIER, TIER, _pu(),
                       n_boot=500, **kw)


def test_tu_so_bo_phu_bang_bo_chinh_delta_0_p_1(mod, tmp_path):
    out = run(mod, make_exp(tmp_path))
    cc = out["cham_chinh"]
    assert len(cc) == len(SUBSETS) * len(GRIDS) and (cc["holm_m"] == len(GRIDS)).all()
    assert np.allclose(cc["delta"], 0) and (cc["p_value"] == 1).all() and (cc["p_holm"] == 1).all()
    assert (cc["nhan"] == "tuong_duong").all() and cc["giu_ca_3_cach_chia"].all()
    assert (cc["ket_luan_D"] == "tuong_duong").all() and (cc["G"] == 8).all()
    per = out["theo_cach_chia"]
    assert len(per) == len(SUBSETS) * len(GRIDS) * len(SCHEMES) and np.allclose(per["delta"], 0)
    f1 = out["f1"]
    assert len(f1) == len(SUBSETS) and np.allclose(f1["delta_bo"], f1["delta_chinh"]) and np.allclose(f1["chenh"], 0)


def test_delta_biet_truoc(mod, tmp_path):
    c = {u: 0.05 * (i + 1) for i, u in enumerate(UNITS)}
    exp = make_exp(tmp_path, shift=lambda bo, s, u: c[u] if bo == "keepwater" else 0.0)
    cc = run(mod, exp)["cham_chinh"].set_index(["bo", "grid"])
    kw = cc.loc["keepwater"]
    # moi don vi cung so diem -> trong so deu -> Delta = TB c_u
    assert np.allclose(kw["delta"], np.mean(list(c.values())))
    assert np.allclose(kw["mae_bo"] - kw["mae_chinh"], kw["delta"])
    # moi d_b > 0 -> chi 2/2^8 to hop dau dat |thong ke| quan sat; Holm m = 2 luoi
    assert np.allclose(kw["p_value"], 2 / 256) and np.allclose(kw["p_holm"], 2 * 2 / 256)
    assert (kw["nhan"] == "khac_co_y_nghia").all() and (kw["ci90_low"] > 0).all() and kw["giu_ca_3_cach_chia"].all()
    # nguong = 5% x TB MAE bo chinh cua cac luoi trong muc (ca hai luoi cung muc -> cung nguong)
    ref = cc.loc["keepmangrove", "mae_chinh"].mean()
    assert np.allclose(kw["nguong"], 0.05 * ref) and np.allclose(kw["muc_tham_chieu"], ref)
    assert (cc.loc["keepmangrove", "nhan"] == "tuong_duong").all()


def test_quy_tac_D_doi_chieu_o_mot_cach_chia(mod, tmp_path):
    sh = {42: 0.3, 43: 0.3, 44: -0.1}
    exp = make_exp(tmp_path, shift=lambda bo, s, u: sh[s] if bo == "keepwater" else 0.0)
    cc = run(mod, exp)["cham_chinh"].set_index(["bo", "grid"]).loc["keepwater"]
    assert np.allclose(cc["delta"], np.mean(list(sh.values())))
    assert (cc["nhan"] == "khac_co_y_nghia").all()
    assert (cc["nhan_s44"] == "khac_co_y_nghia").all() and (cc["delta_s44"] < 0).all()
    assert (~cc["giu_ca_3_cach_chia"]).all() and (cc["ket_luan_D"] == "khong_vung_theo_cach_chia").all()


def test_quy_tac_D_tuong_duong_mot_cach_chia_chua_phan_dinh(mod):
    tb = pd.DataFrame({"bo": ["b"] * 3, "grid": ["g1", "g2", "g3"], "delta": [0.0, 0.2, 0.1],
                       "nhan": ["tuong_duong", "khac_co_y_nghia", "chua_phan_dinh"]})
    per = pd.DataFrame([("b", g, s, n, dl) for g, n, dl in
                        (("g1", "tuong_duong", 0.0), ("g2", "khac_co_y_nghia", 0.2), ("g3", "chua_phan_dinh", 0.1))
                        for s in ("s42", "s43", "s44")], columns=["bo", "grid", "cach_chia", "nhan", "delta"])
    per.loc[(per["grid"] == "g1") & (per["cach_chia"] == "s43"), "nhan"] = "chua_phan_dinh"
    r = mod.apply_rule_d(tb, per).set_index("grid")
    assert r.loc["g1", "giu_ca_3_cach_chia"] == False and r.loc["g1", "ket_luan_D"] == "khong_vung_theo_cach_chia"  # noqa: E712
    assert r.loc["g2", "giu_ca_3_cach_chia"] == True and r.loc["g2", "ket_luan_D"] == "khac_co_y_nghia"  # noqa: E712
    assert pd.isna(r.loc["g3", "giu_ca_3_cach_chia"]) and r.loc["g3", "ket_luan_D"] == "chua_phan_dinh"


BAD = "cv1__g2__hist_gb__s43__keepwater"


@pytest.mark.parametrize("mutate, meta_mut, khop", [
    (lambda n, d: d.iloc[1:] if n == BAD else d, None, "khac tap"),                       # thieu 1 (diem, mua)
    (lambda n, d: d.assign(unit_id=np.where(d.index == 0, "blk_7", d["unit_id"])) if n == BAD else d, None,
     "khac tap"),                                                                           # don vi khac
    (lambda n, d: d.assign(y_ref=np.where(d.index == 0, 9.0, d["y_ref"])) if n == BAD else d, None, "y_ref"),
    (lambda n, d: d.assign(err=np.where(d.index == 3, np.nan, d["err"])) if n == BAD else d, None, "khong huu han"),
    (lambda n, d: d.assign(pred_source="final") if n == BAD else d, None, "pred_source"),
    (lambda n, d: pd.concat([d, d.iloc[[0]]]) if n == BAD else d, None, "trung"),
    (None, lambda n, m: {**m, "points_ref_variant": "keep6090"} if n == BAD else m, "run_meta"),
    (None, lambda n, m: {**m, "label_set": "keepmangrove"} if n == BAD else m, "run_meta"),
    (None, lambda n, m: {**m, "cv_folds_sha256": "khac"} if n == BAD else m, "cv_folds_sha256"),
    (None, lambda n, m: {**m, "returncode": 1} if n == BAD else m, "run_meta"),
])
def test_dau_vao_hong_la_loi_khong_cat_giao(mod, tmp_path, mutate, meta_mut, khop):
    exp = make_exp(tmp_path, mutate=mutate, meta_mut=meta_mut)
    with pytest.raises(SystemExit, match=khop):
        run(mod, exp)


def test_thieu_file_va_G_khac_ky_vong_la_loi(mod, tmp_path):
    exp = make_exp(tmp_path)
    with pytest.raises(SystemExit, match="ky vong"):
        run(mod, exp, expected_g=19)
    os.remove(os.path.join(exp, BAD, "cv", "oof_points.csv"))
    with pytest.raises(SystemExit, match="thieu file"):
        run(mod, exp)


def test_luoi_khac_tap_diem_la_loi(mod, tmp_path):
    exp = make_exp(tmp_path, mutate=lambda n, d: d.iloc[1:] if n.startswith("cv1__g2__") else d)
    with pytest.raises(SystemExit, match="khac tap"):
        run(mod, exp)


def test_cham_phu_6090_mae_ngu_canh_va_ktc_tai_lap(mod, tmp_path):
    exp = make_exp(tmp_path, aux_err=lambda d: d["unit_id"].map(lambda u: 0.1 * (1 + int(u[-1]))).to_numpy())
    p1 = run(mod, exp)["phu_6090"]
    p2 = run(mod, exp)["phu_6090"]
    pd.testing.assert_frame_equal(p1, p2)  # cung seed -> cung KTC
    cv = p1[p1["phan"] == "cv_oof"].set_index("grid")
    fin = p1[p1["phan"] == "giu_rieng_final"].set_index("grid")
    assert (cv["n_diem"] == 40).all() and (cv["n_don_vi"] == 4).all() and (fin["n_diem"] == 10).all()
    assert np.allclose(cv["mae"], np.mean([0.1, 0.2, 0.3, 0.4])) and np.allclose(fin["mae"], np.mean([0.1, 0.2]))
    assert ((cv["ci95_low"] <= cv["mae"]) & (cv["mae"] <= cv["ci95_high"])).all()
    assert ((cv["ci95_low"] >= 0.1 - 1e-12) & (cv["ci95_high"] <= 0.4 + 1e-12)).all()
    assert cv["ci95_high"].gt(cv["ci95_low"]).all()
    assert cv["ghi_chu"].str.contains("lac quan").all() and fin["mae_nong_nghiep_cung_don_vi"].isna().all()
    # ngu canh = MAE luot cv1 keep6090 (dap an bo chinh) tren diem nong nghiep CUNG don vi, TB 3 cach chia
    for g in GRIDS:
        parts = [pd.read_csv(os.path.join(exp, f"cv1__{g}__hist_gb__s{s}__keep6090", "cv", "oof_points.csv"))
                 for s in SCHEMES]
        e = pd.concat(parts)
        e = e[e["unit_id"].isin(AUX_CV_UNITS)]
        assert np.isclose(cv.loc[g, "mae_nong_nghiep_cung_don_vi"], e["err"].abs().mean())
        assert cv.loc[g, "n_diem_nong_nghiep_cung_don_vi"] == 4 * 30


def test_bootstrap_tai_lap_theo_seed(mod):
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2, 300)
    u = np.repeat([f"u{i}" for i in range(10)], 30)
    r1 = mod.bootstrap_mae(a, u, 2000, 42)
    assert r1 == mod.bootstrap_mae(a, u, 2000, 42)
    assert r1 != mod.bootstrap_mae(a, u, 2000, 43)
    assert r1[0] < a.mean() < r1[1]


def test_6090_trung_diem_giua_cv_va_giu_rieng_la_loi(mod, tmp_path):
    exp = make_exp(tmp_path)
    # chep diem giu rieng sang ten diem CV
    for g in GRIDS:
        p = os.path.join(exp, f"phu6090__{g}__hist_gb__s42__keep6090", "final", "final_points.csv")
        d = pd.read_csv(p)
        d.loc[d.index[:2], "point_id"] = "a_blk_0_p0"
        d.to_csv(p, index=False)
    with pytest.raises(SystemExit, match="vua o CV vua o giu rieng"):
        run(mod, exp)


def test_xoa_ket_qua_cu_va_ghi_qua_file_tam(mod, tmp_path):
    out = tmp_path / "ket-qua"
    out.mkdir()
    for f in mod.OUT_FILES.values():
        (out / f).write_text("cu", encoding="utf-8")
    mod.clear_outputs(str(out))
    assert not any((out / f).exists() for f in mod.OUT_FILES.values())
    mod.write_outputs(str(out), {"f1": pd.DataFrame({"x": [1]})})
    assert (out / mod.OUT_FILES["f1"]).exists() and not list(out.glob("*.tmp"))

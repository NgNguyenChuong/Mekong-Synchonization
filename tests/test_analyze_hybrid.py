"""scripts/analyze_hybrid.py - CHG-22 (3 trang thai): kiem tap khoa bang nhau thay cho giao/merge inner am tham.

Du lieu gia nho (tmp_path): oof_points cua 2 luoi x (a)/(b), 2 cach chia, 8 don vi x 2 mua x 30 diem.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_hybrid", os.path.join(ROOT, "scripts", "analyze_hybrid.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


UNITS = [f"blk_{i}" for i in range(8)]
SEEDS = (42, 43)


def _oof(rng, shift=0.0):
    rows = []
    for u in UNITS:
        for p in range(30):
            for season in (2014, 2015):
                rows.append((f"{u}_p{p}", season, u, rng.normal(0, 1) + shift, "oof"))
    return pd.DataFrame(rows, columns=["point_id", "season", "unit_id", "err", "pred_source"])


def _write_runs(tmp_path, grids=("g1", "g2"), mutate=None):
    """Ghi oof_points gia; mutate(name, df) -> df de lam hong mot lan chay."""
    rng = np.random.default_rng(1)
    exp = tmp_path / "experiments"
    for g in grids:
        for s in SEEDS:
            for fs, shift in ((None, 0.0), ("khong_diem", 0.05)):
                name = f"cv1__{g}__hist_gb__s{s}" + (f"__fs-{fs}" if fs else "")
                d = _oof(rng, shift)
                if mutate is not None:
                    d = mutate(name, d)
                (exp / name / "cv").mkdir(parents=True)
                d.to_csv(exp / name / "cv" / "oof_points.csv", index=False)
    return str(exp)


def _pu():
    return pd.Series({f"{u}_p{p}": u for u in UNITS for p in range(30)})


def test_du_lieu_dung_tinh_duoc_va_cot_ci90(mod, tmp_path):
    exp = _write_runs(tmp_path)
    fulls = {g: mod.load(exp, "cv1", g, SEEDS) for g in ("g1", "g2")}
    abls = {g: mod.load(exp, "cv1", g, SEEDS, "khong_diem") for g in ("g1", "g2")}
    ref = mod.assert_same_keys({**fulls, **{f"b{g}": v for g, v in abls.items()}}, _pu())
    assert len(ref) == len(UNITS) * 30 * 2 * len(SEEDS)
    imp = {g: mod.unit_improvement(fulls[g], abls[g], _pu()) for g in fulls}
    s1 = mod.step1(imp, "khong_diem")
    assert {"ci90_low", "ci90_high"} <= set(s1.columns) and not any(c.startswith("ci_tost") for c in s1.columns)
    assert (s1["ci90_low"] < s1["I"]).all() and (s1["I"] < s1["ci90_high"]).all()
    levels = pd.Series({"g1": 5, "g2": 5})
    pairs = pd.DataFrame({"grid_a": ["g1"], "grid_b": ["g2"]})
    s2 = mod.step2(imp, pairs, levels, s1, "khong_diem")
    assert {"ci_tost_low", "ci_tost_high"} <= set(s2.columns)  # buoc 2 giu ten (co Delta_min_D dang ky)


@pytest.mark.parametrize("hong, khop", [
    (lambda d: d.assign(pred_source=np.where(d.index == 3, "final", "oof")), "pred_source"),
    (lambda d: d.assign(err=np.where(d.index == 3, np.nan, d["err"])), "khong huu han"),
    (lambda d: d.assign(err=np.where(d.index == 3, np.inf, d["err"])), "khong huu han"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "trung"),
])
def test_load_tu_choi_oof_hong(mod, tmp_path, hong, khop):
    exp = _write_runs(tmp_path, grids=("g1",), mutate=lambda n, d: hong(d) if n == "cv1__g1__hist_gb__s43" else d)
    with pytest.raises(SystemExit, match=khop):
        mod.load(exp, "cv1", "g1", SEEDS)


def test_khac_tap_diem_hoac_don_vi_la_loi_khong_cat_giao(mod, tmp_path):
    """Truoc day: merge inner / intersection bo lang. Nay: khac tap (diem, mua, don vi) -> LOI."""
    drop1 = lambda n, d: d.iloc[1:] if n == "cv1__g2__hist_gb__s42__fs-khong_diem" else d  # noqa: E731
    exp = _write_runs(tmp_path, mutate=drop1)
    f2, b2 = mod.load(exp, "cv1", "g2", SEEDS), mod.load(exp, "cv1", "g2", SEEDS, "khong_diem")
    with pytest.raises(SystemExit, match="khac tap"):
        mod.unit_improvement(f2, b2, _pu())
    f1 = mod.load(exp, "cv1", "g1", SEEDS)
    with pytest.raises(SystemExit, match="khac tap"):
        mod.assert_same_keys({"g1": f1, "b g2": b2})
    # unit_id trong oof khac don vi tinh tu vi tri diem
    pu_sai = _pu().copy()
    pu_sai.iloc[0] = "blk_7"
    with pytest.raises(SystemExit, match="unit_id"):
        mod.assert_same_keys({"g1": f1}, pu_sai)


def test_buoc2_khac_tap_don_vi_va_mean_I_khong_huu_han_la_loi(mod, tmp_path):
    exp = _write_runs(tmp_path)
    imp = {g: mod.unit_improvement(mod.load(exp, "cv1", g, SEEDS), mod.load(exp, "cv1", g, SEEDS, "khong_diem"),
                                   _pu()) for g in ("g1", "g2")}
    s1 = mod.step1(imp, "x")
    levels = pd.Series({"g1": 5, "g2": 5})
    pairs = pd.DataFrame({"grid_a": ["g1"], "grid_b": ["g2"]})
    bad = dict(imp, g2=imp["g2"].iloc[1:])
    with pytest.raises(SystemExit, match="khac tap don vi"):
        mod.step2(bad, pairs, levels, s1, "x")
    with pytest.raises(SystemExit, match="khong huu han"):  # luoi cua muc khong co trong buoc 1 -> mean NaN
        mod.step2(imp, pairs, pd.Series({"g1": 5, "g2": 5, "g3": 6}), s1, "x")
    with pytest.raises(SystemExit, match="khong huu han"):
        mod.step2(imp, pairs, levels, s1.assign(I=np.nan), "x")


def test_strata_diem_nan_hoac_ngoai_khoang_la_loi(mod, tmp_path, monkeypatch):
    exp = _write_runs(tmp_path)
    monkeypatch.setattr(mod, "GRIDS", ["g1", "g2"])
    full = pd.concat([mod.load(exp, "cv1", g, SEEDS) for g in ("g1", "g2")])
    abl = pd.concat([mod.load(exp, "cv1", g, SEEDS, "khong_diem") for g in ("g1", "g2")])
    pts = _pu().index
    lat = pd.Series(np.linspace(0, 30, len(pts)), index=pts)
    t = mod.strata_table(full, abl, lat, "x")
    assert len(t) == 3 * 2 and t.groupby("grid")["n_diem"].sum().eq(len(pts)).all()  # moi diem vao dung 1 nhom
    for hong in (lat.where(lat.index != pts[0]), lat.drop(pts[0]), lat.where(lat.index != pts[0], -5.0)):
        with pytest.raises(SystemExit, match="graph_lateral_km"):
            mod.strata_table(full, abl, hong, "x")

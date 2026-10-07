"""Kiem thu doi chung duong PC-1 (src/training/pc1.py, scripts/analyze_pc1.py). Du lieu gia 19 don vi, khong can artifacts."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from training import pc1  # noqa: E402
from training.block_stats import (_arm_level, compare_family, seed_mean_errors, signflip_test,  # noqa: E402
                                  unit_metrics)
from training.evaluate import holm_adjust  # noqa: E402

M = "hgb"
UNITS = [f"u{i:02d}" for i in range(19)]
GRIDS = ["F1", "F2", "F3", "C1", "C2"]
PAIRS = [("F1", "F2"), ("F1", "F3"), ("F2", "F3"), ("C1", "C2")]
LEVELS = pd.Series({"F1": 7, "F2": 7, "F3": 7, "C1": 5, "C2": 5})
SEEDS = (42, 43, 44)
KW = dict(cv_units=UNITS, holdout_units=["h1"], holdout_seasons=(2020,), mode="cv", delta_min=0.05,
          delta_min_kind="rel", alpha=0.05, delta_ref="level_mean", levels=LEVELS, model=M, family="F1")


def _make(grids=GRIDS, model=M):
    rng = np.random.default_rng(7)
    n_pt = 32  # >= min_pts 30 moi (don vi, mua)
    pid = [f"{u}_p{j}" for u in UNITS for j in range(n_pt)]
    pu = pd.Series([p.split("_")[0] for p in pid] + ["h1"], index=pid + ["h1_p0"])
    base = rng.normal(0, 0.8, (len(pid), 2))
    off = {g: rng.normal(0, 0.05, len(UNITS)) for g in grids}
    rows = []
    for g in grids:
        for s in SEEDS:
            noise = rng.normal(0, 0.3, base.shape)
            for j, season in enumerate((2014, 2015)):
                e = base[:, j] + noise[:, j] + np.repeat(off[g], n_pt) * np.sign(base[:, j])
                rows.append(pd.DataFrame({"grid": g, "model": model, "seed": s, "point_id": pid, "season": season,
                                          "err": e, "pred_source": "oof"}))
    return pd.concat(rows, ignore_index=True), pu


@pytest.fixture(scope="module")
def data():
    err, pu = _make()
    frozen = compare_family(err, pu, PAIRS, expected_m=len(PAIRS), **KW)
    inputs = [pc1.pair_unit_deltas(err, pu, (a, M), (b, M)) for a, b in PAIRS]
    return err, pu, frozen, inputs


def _level_mae(err, pu, grid):
    sub = err[err["grid"] == grid]
    return _arm_level(unit_metrics(seed_mean_errors(sub), pu)[0], (grid, M), "mae")


# ---------------- tiem ----------------
@pytest.mark.parametrize("kind", ["deu", "lognormal"])
@pytest.mark.parametrize("k", [1.0, 2.0])
def test_tiem_muc_diem_lam_mae_muc_tang_dung_k_thr(data, kind, k):
    err, pu, frozen, inputs = data
    thr = float(frozen.loc[0, "delta_min_thr"])
    d, w, ds = inputs[0]
    wn = (w / w.sum()).to_numpy()
    x = pc1.injection_pattern(kind, wn, [3], "F1:test")[0]
    delta = pd.Series(k * thr * x, index=d.index)
    inj = pc1.inject_abs_err(err, ("F2", M), pu, delta)
    assert _level_mae(inj, pu, "F2") - _level_mae(err, pu, "F2") == pytest.approx(k * thr, abs=1e-9)
    assert _level_mae(inj, pu, "F1") == _level_mae(err, pu, "F1")
    d2, _, ds2 = pc1.pair_unit_deltas(inj, pu, ("F1", M), ("F2", M))
    np.testing.assert_allclose(d2.to_numpy(), (d - delta).to_numpy(), atol=1e-12)
    np.testing.assert_allclose(ds2.to_numpy(), ds.sub(delta, axis=0).to_numpy(), atol=1e-12)
    # dau sai so giu nguyen, |e| tang dung delta
    m = inj["grid"] == "F2"
    assert (np.sign(inj.loc[m, "err"]) * np.sign(err.loc[m, "err"]) >= 0).all()


def test_tiem_muc_diem_khop_duong_don_vi_qua_compare_family(data):
    err, pu, frozen, inputs = data
    thr = float(frozen.loc[0, "delta_min_thr"])
    d, w, ds = inputs[0]
    wn = (w / w.sum()).to_numpy()
    x = pc1.injection_pattern("lognormal", wn, [11], "F1:test")[0]
    delta = pd.Series(2 * thr * x, index=d.index)
    cf = compare_family(pc1.inject_abs_err(err, ("F2", M), pu, delta), pu, PAIRS, **KW)
    sim = pc1.simulate_pair(d, ds, w, thr, cf["p_value"].to_numpy(), 0, 0.05, np.ones((1, 19)),
                            delta.to_numpy()[None]).iloc[0]
    r = cf.iloc[0]
    for a, b in (("delta_star", "delta_hat"), ("p_value", "p_value"), ("p_holm", "p_holm"),
                 ("ci_tost_low", "ci_tost_low"), ("ci_tost_high", "ci_tost_high")):
        assert sim[a] == pytest.approx(r[b], abs=1e-12)
    assert sim["seed_deltas"] == r["seed_deltas"]
    assert sim["seeds_same_dir"] == r["seeds_same_dir"]
    # compare_family tinh lai thr tu du lieu da tiem -> lon hon; PC-1 giu thr dong bang
    assert r["delta_min_thr"] > thr + 1e-6
    assert sim["delta_min_thr"] == thr


def test_tiem_am_khong_cai_o_muc_diem(data):
    err, pu, _, inputs = data
    with pytest.raises(ValueError, match=">= 0"):
        pc1.inject_abs_err(err, ("F2", M), pu, pd.Series(-0.01, index=inputs[0][0].index))


def test_he_so_tiem_tb_trong_so_dung_1():
    w = np.random.default_rng(1).integers(100, 2000, 19).astype(float)
    wn = w / w.sum()
    seeds = list(range(300))
    for kind in pc1.KINDS:
        x = pc1.injection_pattern(kind, wn, seeds, "F1:a-vs-b")
        np.testing.assert_allclose(x @ wn, 1.0, atol=1e-12)
        np.testing.assert_array_equal(x, pc1.injection_pattern(kind, wn, seeds, "F1:a-vs-b"))
    assert (pc1.injection_pattern("deu", wn, seeds, "x") == 1).all()
    ln = pc1.injection_pattern("lognormal", wn, seeds, "x")
    assert (ln > 0).all()
    cv = np.median(ln.std(axis=1, ddof=1) / ln.mean(axis=1))
    assert 0.6 < cv < 0.85
    assert (pc1.injection_pattern("tau_mu", wn, seeds, "x") < 0).any()
    with pytest.raises(ValueError):
        pc1.injection_pattern("chuan", wn, seeds, "x")


# ---------------- kiem dinh ----------------
def test_signflip_p_batch_khop_signflip_test():
    rng = np.random.default_rng(3)
    w = rng.integers(200, 900, 19).astype(float)
    D = rng.normal(0.02, 0.05, (6, 19))
    D[4] = 0.0
    D[5] = 0.3
    p = pc1.signflip_p_batch(D, w / w.sum())
    exp = [signflip_test(r, w)["p_value"] for r in D]
    np.testing.assert_array_equal(p, exp)
    assert p[4] == 1.0 and p[5] == pytest.approx(2 / 2 ** 19)
    mc = pc1.signflip_p_batch(D[:4], w / w.sum(), n_perm=40_000, rng=np.random.default_rng(0))
    np.testing.assert_allclose(mc, exp[:4], atol=0.01)


def test_dong_nhat_giong_dong_bang(data):
    _, _, frozen, inputs = data
    p = frozen["p_value"].to_numpy()
    ident = []
    for i, (d, w, ds) in enumerate(inputs):
        r = pc1.simulate_pair(d, ds, w, float(frozen.loc[i, "delta_min_thr"]), p, i, 0.05,
                              np.ones((1, 19)), np.zeros((1, 19))).iloc[0]
        ident.append(r.rename({"delta_star": "delta_hat"}))
    ident = pd.DataFrame(ident).reset_index(drop=True)
    assert pc1.compare_identity(ident, frozen) == []
    assert list(ident["label"]) == list(frozen["label"])
    # lech mot gia tri -> bao
    bad = frozen.copy()
    bad.loc[1, "p_value"] += 1e-4
    assert any("p_value" in x for x in pc1.compare_identity(ident, bad))


def test_wild_giu_delta_hat_va_dau_chung_3_cach_chia(data):
    _, _, frozen, inputs = data
    d, w, ds = inputs[1]
    wn = (w / w.sum()).to_numpy()
    thr = float(frozen.loc[1, "delta_min_thr"])
    s = pc1.wild_signs(range(50), 19, "F1:x")
    assert set(np.unique(s)) == {-1.0, 1.0}
    z = np.zeros_like(s)
    a = pc1.simulate_pair(d, ds, w, thr, frozen["p_value"].to_numpy(), 1, 0.05, s, z)
    b = pc1.simulate_pair(d, ds, w, thr, frozen["p_value"].to_numpy(), 1, 0.05, -s, z)
    dhat = float(wn @ d.to_numpy())
    np.testing.assert_allclose(a["delta_star"] + b["delta_star"], 2 * dhat, atol=1e-12)
    # tung cach chia: Delta*_s = Delta_s + sum wn s (d_s - Delta_s) voi CUNG s
    Ds = ds.to_numpy().T
    dhs = Ds @ wn
    got = np.array([[float(x) for x in r.split(";")] for r in a["seed_deltas"]])
    exp = dhs[None] + (s[:, None, :] * (Ds - dhs[:, None])[None]) @ wn
    np.testing.assert_allclose(got, exp, rtol=1e-5)  # seed_deltas ghi 6 chu so


# ---------------- chay mot cap ----------------
def test_chi_cap_duoc_tiem_thay_doi_va_thr_khong_doi(data):
    _, _, frozen, inputs = data
    p_fz = frozen["p_value"].to_numpy().copy()
    p_keep = p_fz.copy()
    d, w, ds = inputs[2]
    thr = float(frozen.loc[2, "delta_min_thr"])
    out = pc1.run_pair(2, "F2", "F3", 7, True, d, ds, w, thr, p_fz, 0.05, [1.0, 2.0], list(pc1.KINDS),
                       range(20), "F1:F2-vs-F3")
    np.testing.assert_array_equal(p_fz, p_keep)  # khong sua p dong bang
    assert set(out["pair_index"]) == {2} and len(out) == 2 * 3 * 20
    for r in out.itertuples():
        pf = p_keep.copy()
        pf[2] = r.p_value
        assert r.p_holm == holm_adjust(pf)[2]
    assert (out["delta_min_thr"] == thr).all()
    assert out["inj_err"].max() < 1e-12
    # cung dau wild + cung he so giua k: Delta*(k=2) - Delta*(k=1) = -thr
    for kind in pc1.KINDS:
        k1 = out[(out["kind"] == kind) & (out["k"] == 1.0)]["delta_star"].to_numpy()
        k2 = out[(out["kind"] == kind) & (out["k"] == 2.0)]["delta_star"].to_numpy()
        np.testing.assert_allclose(k2 - k1, -thr, atol=1e-12)


def test_cuc_doan_tiem_lon_phat_hien_tiem_0_giong_dong_bang(data):
    _, _, frozen, inputs = data
    p = frozen["p_value"].to_numpy()
    for i, (a, b) in enumerate(PAIRS):
        d, w, ds = inputs[i]
        thr = float(frozen.loc[i, "delta_min_thr"])
        big = pc1.run_pair(i, a, b, LEVELS[a], LEVELS[a] == 7, d, ds, w, thr, p, 0.05, [50.0], ["deu", "lognormal"],
                           range(10), f"F1:{a}-vs-{b}")
        assert big["phat_hien_vung_D"].all() and big["phat_hien_tren_nguong"].all()
        assert (big["label"] == "cho_giu_rieng").all() and not big["tost_tuong_duong"].any()
        zero = pc1.simulate_pair(d, ds, w, thr, p, i, 0.05, np.ones((3, 19)), np.zeros((3, 19)))
        assert (zero["label"] == frozen.loc[i, "label"]).all()
        assert zero["delta_star"].to_numpy() == pytest.approx(frozen.loc[i, "delta_hat"], abs=1e-12)


# ---------------- tong hop, ket luan, loi ----------------
def _summary(rate=0.9, tost=0.0, n_fine=6, drop_kind=None):
    rows = []
    for j in range(n_fine + 3):
        for kind in ("deu", "lognormal", "tau_mu"):
            if kind == drop_kind:
                continue
            for k in (1.0, 2.0):
                rows.append({"pair_index": j, "grid_a": f"a{j}", "grid_b": f"b{j}", "level": 7 if j < n_fine else 5,
                             "is_fine": j < n_fine, "k": k, "kind": kind, "n_rep": 200,
                             "ty_le_phat_hien": rate if k == 2 else 0.3, "ty_le_tost": tost if k == 2 else 0.06})
    return pd.DataFrame(rows)


def test_ket_luan_ba_trang_thai():
    assert pc1.verdict(_summary(0.9, 0.0), n_rep=200)["trang_thai"] == "DAT"
    s = _summary(0.9, 0.0)
    s.loc[(s["pair_index"] == 3) & (s["k"] == 2) & (s["kind"] == "lognormal"), "ty_le_phat_hien"] = 0.79
    v = pc1.verdict(s, n_rep=200)
    assert v["trang_thai"] == "KHONG_DAT" and v["cap_yeu_nhat"] == "a3 vs b3 (lognormal)"
    assert pc1.verdict(_summary(0.9, 0.06), n_rep=200)["trang_thai"] == "KHONG_DAT"
    assert pc1.verdict(_summary(n_fine=5), n_rep=200)["trang_thai"] == "LOI"
    assert pc1.verdict(_summary(drop_kind="lognormal"), n_rep=200)["trang_thai"] == "LOI"
    assert pc1.verdict(_summary(), n_rep=100)["trang_thai"] == "LOI"
    assert pc1.verdict(_summary().query("k == 1"), n_rep=200)["trang_thai"] == "LOI"


def test_thieu_cap_trong_ban_dong_bang_la_loi(data):
    _, _, frozen, _ = data
    got = pc1.match_frozen(PAIRS[::-1], frozen)
    assert list(zip(got["grid_a"], got["grid_b"])) == PAIRS[::-1]
    with pytest.raises(ValueError, match="LOI.*thieu"):
        pc1.match_frozen(PAIRS, frozen.iloc[:3])
    with pytest.raises(ValueError, match="LOI.*thua"):
        pc1.match_frozen(PAIRS[:3], frozen)
    with pytest.raises(ValueError, match="LOI.*trung"):
        pc1.match_frozen(PAIRS, pd.concat([frozen, frozen.iloc[[0]]]))


def test_tong_hop_ti_le(data):
    _, _, frozen, inputs = data
    d, w, ds = inputs[0]
    out = pc1.run_pair(0, "F1", "F2", 7, True, d, ds, w, float(frozen.loc[0, "delta_min_thr"]),
                       frozen["p_value"].to_numpy(), 0.05, [2.0], ["deu"], range(30), "F1:F1-vs-F2")
    s = pc1.summarize(out).iloc[0]
    assert s["n_rep"] == 30
    assert s["ty_le_phat_hien"] == pytest.approx(out["phat_hien_vung_D"].mean())
    assert s["ty_le_tost"] == pytest.approx(out["tost_tuong_duong"].mean())
    assert s["trung_vi_delta_star"] == pytest.approx(out["delta_star"].median())


def test_script_thieu_ban_dong_bang_tra_loi(tmp_path, capsys):
    import analyze_pc1
    rc = analyze_pc1.cli(["--frozen", str(tmp_path / "khong_co.csv"), "--out", str(tmp_path)])
    assert rc == 2
    assert "LOI" in capsys.readouterr().out


# ---------------- script dau-cuoi tren du lieu gia (13 ten luoi that) ----------------
AREAS = {"h3_res_5": (5, 290.0), "latlon_0.1552deg": (5, 290.1), "square_utm_17087m": (5, 290.2),
         "s2_level_9": (5, 400.0), "h3_res_6": (6, 41.7), "latlon_0.0586deg": (6, 41.75),
         "square_utm_6458m": (6, 41.8), "s2_level_10": (6, 80.0), "s2_level_11": (6, 20.0),
         "h3_res_7": (7, 5.96), "s2_level_12": (7, 5.97), "square_utm_2441m": (7, 5.98),
         "latlon_0.0222deg": (7, 5.99)}


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory):
    import analyze_cv_family as acf
    from training.block_stats import area_levels, pairs_by_area
    at = pd.DataFrame([{"file": f"data/grids/{g}.geojson", "tier_h3_res": t, "mean_area_km2": a}
                       for g, (t, a) in AREAS.items()])
    at["grid"] = at["file"].map(lambda f: os.path.splitext(os.path.basename(f))[0])
    err, pu = _make(grids=list(AREAS), model="hist_gb")
    pairs = pairs_by_area(at, max_ratio=1.2, expected_m=12)
    fz = compare_family(err, pu, pairs, cv_units=UNITS, holdout_units=["h1"], holdout_seasons=(2020,), mode="cv",
                        delta_min=0.05, delta_min_kind="rel", alpha=0.05, delta_ref="level_mean",
                        levels=area_levels(at, pairs=pairs), require_practical=True, model="hist_gb",
                        family="F1_chinh", expected_m=12)
    fz.insert(0, "n_common_point_seasons", err[["point_id", "season"]].drop_duplicates().shape[0])
    d = tmp_path_factory.mktemp("pc1")
    fz.to_csv(d / "frozen.csv", index=False)
    patches = {"area_table": lambda path: at, "load_errors": lambda *a, **k: err,
               "point_units": lambda e, f: (pu, UNITS, ["h1"])}
    return acf, patches, fz, d


def _run_cli(monkeypatch, fake_run, frozen_df, extra=()):
    import analyze_pc1
    acf, patches, _, d = fake_run
    for name, fn in patches.items():
        monkeypatch.setattr(acf, name, fn)
    f = d / "fz_case.csv"
    frozen_df.to_csv(f, index=False)
    out = d / "out"
    rc = analyze_pc1.cli(["--frozen", str(f), "--out", str(out), "--n-rep", "4", *extra])
    return rc, out


def test_script_dau_cuoi_du_lieu_gia(monkeypatch, fake_run, capsys):
    rc, out = _run_cli(monkeypatch, fake_run, fake_run[2])
    log = capsys.readouterr().out
    assert rc == 0, log
    assert "khop ban dong bang (12/12 cap)" in log
    reps = pd.read_csv(out / "dot7_pc1_lan_lap.csv")
    summ = pd.read_csv(out / "dot7_pc1_tong_hop.csv")
    ver = pd.read_csv(out / "dot7_pc1_ket_luan.csv").iloc[0]
    assert len(reps) == 12 * 2 * 3 * 4 and len(summ) == 12 * 2 * 3
    assert summ["is_fine"].sum() == 6 * 2 * 3
    assert ver["trang_thai"] in ("DAT", "KHONG_DAT") and ver["n_cap_min"] == 6
    fz = fake_run[2].set_index(["grid_a", "grid_b"])
    for r in summ.itertuples():
        assert r.thr == pytest.approx(fz.loc[(r.grid_a, r.grid_b), "delta_min_thr"], rel=1e-12)  # qua CSV


def test_script_ban_dong_bang_lech_hoac_thieu_cap_la_loi(monkeypatch, fake_run, capsys):
    bad = fake_run[2].copy()
    bad.loc[3, "p_value"] = min(1.0, bad.loc[3, "p_value"] + 0.01)
    rc, _ = _run_cli(monkeypatch, fake_run, bad)
    assert rc == 2 and "lech ban dong bang" in capsys.readouterr().out
    bad = fake_run[2].copy()
    bad.loc[5, "delta_min_thr"] *= 1.01
    rc, _ = _run_cli(monkeypatch, fake_run, bad)
    assert rc == 2 and "thr cap" in capsys.readouterr().out
    rc, _ = _run_cli(monkeypatch, fake_run, fake_run[2].drop(index=7))
    assert rc == 2 and "LOI" in capsys.readouterr().out
    # chi chay k = 1 -> khong tinh duoc tieu chi -> LOI
    rc, out = _run_cli(monkeypatch, fake_run, fake_run[2], ["--k", "1"])
    assert rc == 2 and pd.read_csv(out / "dot7_pc1_ket_luan.csv").iloc[0]["trang_thai"] == "LOI"

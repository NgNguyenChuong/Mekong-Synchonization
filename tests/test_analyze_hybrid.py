"""scripts/analyze_hybrid.py - CHG-22 (3 trang thai): kiem tap khoa bang nhau thay cho giao/merge inner am tham.

Du lieu gia nho (tmp_path): oof_points cua 2 luoi x (a)/(b), 2 cach chia, 8 don vi x 2 mua x 30 diem.
"""
import argparse
import importlib.util
import json
import os
import subprocess

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


# ------------------------------------------------------------------ --target (Dot 7, NHAT_KY 2026-10-10)
G4 = ["g1", "g2", "g3", "g4"]


def _name(g, s, fs, target):
    """Ten luot ky vong (runner): (a) cv1__g__hist_gb__sN__t-<t>, (b) ...__fs-khong_diem__t-<t>; do man khong __t-."""
    return f"cv1__{g}__hist_gb__s{s}" + (f"__fs-{fs}" if fs else "") + ("" if target == "salinity" else f"__t-{target}")


def _write_target_runs(exp, target, tags=("nckh-dot7-e6a", "nckh-dot7-hyb"), mutate_meta=None):
    rng = np.random.default_rng(2)
    cfgs = [(None, 0.0, tags[0]), ("khong_diem", 0.05, tags[1])]
    if target == "salinity":
        cfgs.append(("zos_vung", 0.02, tags[1]))
    for g in G4:
        for s in SEEDS:
            for fs, shift, tag in cfgs:
                name = _name(g, s, fs, target)
                d = exp / name / "cv"
                d.mkdir(parents=True)
                _oof(rng, shift).to_csv(d / "oof_points.csv", index=False)
                meta = {"returncode": 0, "mode": "cv", "target": target, "feature_set": fs, "git_tag": tag,
                        "points_ref_sha256": "ref1", "features_sha256": "f1" if fs else None}
                if mutate_meta is not None:
                    meta = mutate_meta(name, meta)
                (d / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
                (d / "config.json").write_text(json.dumps({"target": target}), encoding="utf-8")


def _patch(m, monkeypatch):
    monkeypatch.setattr(m, "GRIDS", list(G4))
    monkeypatch.setattr(m, "area_table", lambda p: None)
    monkeypatch.setattr(m, "pairs_by_area",
                        lambda at, **kw: pd.DataFrame({"grid_a": ["g1", "g3"], "grid_b": ["g2", "g4"]}))
    monkeypatch.setattr(m, "area_levels", lambda at, pairs=None: pd.Series({"g1": 5, "g2": 5, "g3": 6, "g4": 6}))
    pts = _pu().index
    monkeypatch.setattr(m, "point_lateral", lambda: pd.Series(np.linspace(0, 30, len(pts)), index=pts))
    monkeypatch.setattr(m, "_point_unit", _pu)


def _args(tmp_path, target, frozen_files=("KE_HOACH/ket-qua/khac.csv",), **kw):
    man = tmp_path / "KE_HOACH" / "ket-qua" / "manifest_gia.csv"   # cot file tuong doi tmp_path (2 cap tren)
    man.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file": list(frozen_files)}).to_csv(man, index=False)
    (tmp_path / "at.csv").write_text("file\n", encoding="utf-8")
    base = dict(prefix="cv1", target=target, schemes=list(SEEDS), area_table=str(tmp_path / "at.csv"),
                out_dir=str(tmp_path / "out"), exp_root=str(tmp_path / "artifacts" / "experiments"),
                frozen_manifest=[str(man)], allowed_tags=["nckh-dot7-e6a", "nckh-dot7-hyb"], tags_reason="r",
                no_table=True)
    return argparse.Namespace(**{**base, **kw})


def test_dot7_ten_luot_buoc2_mo_ta_va_provenance(mod, tmp_path, monkeypatch):
    _patch(mod, monkeypatch)
    _write_target_runs(tmp_path / "artifacts" / "experiments", "rain_chirps")   # khong co luot (c) zos_vung
    mod.main(_args(tmp_path, "rain_chirps"))
    out = tmp_path / "out"
    for k in ("buoc1", "buoc2", "nhom_song", "do_nhay_cach_chia"):
        assert (out / f"dot7_rain_chirps_hybrid_{k}.csv").is_file()
        assert (out / f"dot7_rain_chirps_hybrid_{k}.csv.provenance.json").is_file()
    assert not list(out.glob("dot6_*"))
    s1 = pd.read_csv(out / "dot7_rain_chirps_hybrid_buoc1.csv")
    assert set(s1["cau_hinh"]) == {"khong_diem"} and len(s1) == 4 and s1["p_holm"].notna().all()
    s2 = pd.read_csv(out / "dot7_rain_chirps_hybrid_buoc2.csv").set_index("grid_a")
    # rain_chirps: TESTED_TIERS {5} -> cap muc 6 chi mo ta (p/Holm NaN, giu D), Holm m = 1
    assert bool(s2.loc["g1", "kiem_dinh"]) and s2.loc["g1", "nhan"] != "mo_ta" and np.isfinite(s2.loc["g1", "p_holm"])
    assert not bool(s2.loc["g3", "kiem_dinh"]) and s2.loc["g3", "nhan"] == "mo_ta"
    assert np.isnan(s2.loc["g3", "p_holm"]) and np.isnan(s2.loc["g3", "p_value"]) and np.isfinite(s2.loc["g3", "D"])
    assert (s2["holm_m"] == 1).all()
    sens = pd.read_csv(out / "dot7_rain_chirps_hybrid_do_nhay_cach_chia.csv")
    s2s = sens[sens["grid_a"].notna()]
    assert set(s2s.loc[s2s["muc"] == 6, "nhan"]) == {"mo_ta"} and set(s2s["scheme"]) == set(SEEDS)
    prov = json.loads((out / "dot7_rain_chirps_hybrid_buoc2.csv.provenance.json").read_text(encoding="utf-8"))
    assert prov["git_tags_seen"] == ["nckh-dot7-e6a", "nckh-dot7-hyb"] and prov["n_luot"] == 4 * 2 * len(SEEDS)
    assert prov["muc_kiem_dinh"] == [5] and prov["holm_m_buoc2"] == 1 and prov["holm_m_buoc1"] == 4
    assert prov["cau_hinh"] == ["khong_diem"] and prov["features_sha256_b"] == ["f1"]


def test_dot7_nhiet_am_moi_cap_mo_ta(mod, tmp_path, monkeypatch):
    _patch(mod, monkeypatch)
    _write_target_runs(tmp_path / "artifacts" / "experiments", "t2m_era5")
    mod.main(_args(tmp_path, "t2m_era5"))
    s2 = pd.read_csv(tmp_path / "out" / "dot7_t2m_era5_hybrid_buoc2.csv")
    assert (s2["nhan"] == "mo_ta").all() and s2["p_holm"].isna().all() and (s2["holm_m"] == 0).all()


def test_dot7_tag_meta_va_manifest_la_loi_khong_ghi(mod, tmp_path, monkeypatch):
    _patch(mod, monkeypatch)
    _write_target_runs(tmp_path / "artifacts" / "experiments", "ndwi")
    with pytest.raises(SystemExit, match="git_tag"):                       # 2 tag ma khong khai bao
        mod.main(_args(tmp_path, "ndwi", allowed_tags=None))
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit) as e:                                   # file ra nam trong manifest dong bang
        mod.main(_args(tmp_path, "ndwi", frozen_files=("out/dot7_ndwi_hybrid_buoc2.csv",)))
    assert e.value.code == 2 and not (tmp_path / "out").exists()
    with pytest.raises(SystemExit, match="TESTED_TIERS"):
        mod.main(_args(tmp_path, "khong_co"))
    bad = tmp_path / "bad"
    _write_target_runs(bad / "artifacts" / "experiments", "ndwi", mutate_meta=lambda n, m: (
        {**m, "feature_set": None} if n == _name("g2", 43, "khong_diem", "ndwi") else m))
    with pytest.raises(SystemExit, match="feature_set"):
        mod.main(_args(bad, "ndwi"))
    bad2 = tmp_path / "bad2"
    _write_target_runs(bad2 / "artifacts" / "experiments", "ndwi", mutate_meta=lambda n, m: (
        {**m, "target": "salinity"} if n == _name("g1", 42, None, "ndwi") else m))
    with pytest.raises(SystemExit, match="target"):
        mod.main(_args(bad2, "ndwi"))


def test_do_man_giu_byte_nhu_ban_cu(mod, tmp_path, monkeypatch):
    """Mac dinh salinity: 4 file dot6_hybrid_* trung tung byte voi analyze_hybrid.py truoc --target (b123d93f)."""
    old_src = subprocess.run(["git", "show", "b123d93f:scripts/analyze_hybrid.py"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8")
    if old_src.returncode != 0:
        pytest.skip("khong doc duoc ban cu tu git")
    (tmp_path / "scripts").mkdir()
    old_path = tmp_path / "scripts" / "analyze_hybrid_cu.py"   # ROOT ban cu = tmp_path -> exp = tmp/artifacts/experiments
    old_path.write_text(old_src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("analyze_hybrid_cu", str(old_path))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    _patch(old, monkeypatch)
    _patch(mod, monkeypatch)
    _write_target_runs(tmp_path / "artifacts" / "experiments", "salinity")
    a = _args(tmp_path, "salinity", out_dir=str(tmp_path / "out_moi"), allowed_tags=None, no_table=False)
    old.main(argparse.Namespace(prefix="cv1", schemes=list(SEEDS), area_table=a.area_table,
                                out_dir=str(tmp_path / "out_cu")))
    mod.main(a)
    names = ["dot6_hybrid_buoc1.csv", "dot6_hybrid_buoc2.csv", "dot6_hybrid_theo_nhom_song.csv",
             "dot6_hybrid_do_nhay_cach_chia.csv"]
    assert sorted(os.listdir(tmp_path / "out_moi")) == sorted(names)   # khong sidecar, khong file moi
    for n in names:
        assert (tmp_path / "out_moi" / n).read_bytes() == (tmp_path / "out_cu" / n).read_bytes(), n

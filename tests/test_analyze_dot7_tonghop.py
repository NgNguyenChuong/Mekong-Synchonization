"""Script tong hop Dot 7 (CHG-25): analyze_learnability -> analyze_cv_family -> analyze_block_size -> analyze_final.

Du lieu gia (tmp_path): 13 luoi that (ten + dien tich nhu bang doi chieu), 10 khoi CV + 2 khoi giu rieng x 30 diem x
2 mua; |e| HistGB = U(0,5; 1,5); season_mean = HistGB + 1 (hoc duoc ro) tru luoi BAD (chenh doi dau theo khoi ~ 0).
Pham vi Holm: so voi analyze_cv_family + dot7_rules NGUYEN VAN o commit 114bb63f (git show), cung du lieu gia.
"""
import hashlib
import json
import os
import subprocess
import sys
import types
import zlib

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import analyze_block_size as abs_  # noqa: E402
import analyze_cv_family as acf  # noqa: E402
import analyze_final as afin  # noqa: E402
import analyze_learnability as alr  # noqa: E402
import check_sealed  # noqa: E402
from run_experiments import run_name  # noqa: E402
from training.dot7_rules import (EXPECTED_F1_M, MO_TA, SENS_SUFFIX, TESTED_TIERS, compare_tiered,  # noqa: E402
                                 expected_f1_m, holm_tiers, scopes_differ)
from training.evaluate import holm_adjust  # noqa: E402
from training.split import EMPTY_LABEL  # noqa: E402

AREA = {"h3_res_5": (5, 292.2), "h3_res_6": (6, 41.72), "h3_res_7": (7, 5.960), "s2_level_9": (5, 394.3),
        "s2_level_10": (6, 98.59), "s2_level_11": (6, 24.65), "s2_level_12": (7, 6.163),
        "square_utm_17087m": (5, 292.1), "square_utm_6458m": (6, 41.73), "square_utm_2441m": (7, 5.962),
        "latlon_0.1552deg": (5, 292.1), "latlon_0.0586deg": (6, 41.65), "latlon_0.0222deg": (7, 5.977)}
CV_UNITS = [f"u{i}" for i in range(10)]
HO_UNITS = ["h0", "h1"]
SEASONS = (2014, 2015)
NPT = 30
BAD = "square_utm_2441m"
TAG, REF = "nckh-test", "ref_sha"
MODELS = ("hist_gb", "linear", "idw", "season_mean")
TARGETS = ("salinity", "ndwi", "rain_chirps", "dsr_mcd18", "t2m_era5")


def _pts(units):
    return [(f"{u}_p{i}", u) for u in units for i in range(NPT)]


def _abs_err(target, grid, model, scheme, pts, seasons, offset=0):
    """|e| tat dinh theo (target, luoi, diem, mua); season_mean lech +1 (BAD: +-0,05 doi dau theo khoi)."""
    seed = zlib.crc32(repr((target, grid, scheme, offset)).encode())
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 1.5, len(pts) * len(seasons))
    if model == "linear":
        base = base * 1.1
    if model == "idw":
        base = base * 1.2
    if model == "season_mean":
        units = [u for _, u in pts for _ in seasons]
        if grid == BAD:
            bump = np.array([0.05 if int(u[1:]) % 2 else -0.05 for u in units])
        else:
            bump = np.ones(len(units))
        base = base + bump
    return base


def _write_run(exp, name, mode, target, extra_meta=None, cfg=None):
    d = exp / name / mode
    d.mkdir(parents=True, exist_ok=True)
    meta = {"run": name, "returncode": 0, "mode": mode, "git_tag": TAG, "points_ref_sha256": REF,
            "allow_dirty": False, "allow_untagged": False}
    if target != "salinity":
        meta["target"] = target  # luot do man cu khong co khoa target
    meta.update(extra_meta or {})
    (d / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    c = {"target": target} if target != "salinity" else {}
    c.update(cfg or {})
    (d / "config.json").write_text(json.dumps(c), encoding="utf-8")
    return d


def _oof(target, grid, model, scheme):
    pts = _pts(CV_UNITS)
    rows = [(grid, p, s, u, "oof") for p, u in pts for s in SEASONS]
    df = pd.DataFrame(rows, columns=["grid", "point_id", "season", "unit_id", "pred_source"])
    df["err"] = _abs_err(target, grid, model, scheme, pts, SEASONS)
    return df


def _final(target, grid, model, other_err=0.0, empty_groups=()):
    sp = _pts(HO_UNITS)
    rows = [(p, s, u, "final", "khong_gian") for p, u in sp for s in SEASONS]
    df = pd.DataFrame(rows, columns=["point_id", "season", "unit_id", "pred_source", "test_group"])
    df["err"] = _abs_err(target, grid, model, 42, sp, SEASONS, offset=1)
    extra = []
    if "thoi_gian" not in empty_groups:
        extra += [(p, 2020, u, "final", "thoi_gian") for p, u in _pts(CV_UNITS)]
    if "ca_hai" not in empty_groups:
        extra += [(p, 2020, u, "final", "ca_hai") for p, u in sp]
    if extra:
        x = pd.DataFrame(extra, columns=df.columns[:5])
        x["err"] = other_err
        df = pd.concat([df, x], ignore_index=True)
    return df


def _final_cfg(df, empty_groups):
    pm, n, st = {}, {}, {}
    for g in ("khong_gian", "thoi_gian", "ca_hai"):
        sub = df[df["test_group"] == g]
        n[g] = len(sub)
        m = {"n": len(sub), "mae": float(sub["err"].abs().mean()), "rmse": 1.0, "r2": 0.5}
        pm[g] = {"42": m} if len(sub) else {}
        st[g] = {"n_seeds": 1 if len(sub) else 0, "mae": None}
        if g in empty_groups:
            st[g]["status"] = EMPTY_LABEL
    cfg = {"point_metrics_by_group": pm, "n_point_rows_by_group": n, "test_metrics_by_group_mean_over_seeds": st}
    if empty_groups:
        cfg["notes"] = {"nhom_test_rong": f"{list(empty_groups)}: {EMPTY_LABEL} - mua 2020 NaN"}
    return cfg


def build_exp(exp, targets=TARGETS, other_err=0.0):
    for t in targets:
        empty = ("thoi_gian", "ca_hai") if t == "dsr_mcd18" else ()
        for g in AREA:
            for m in MODELS:
                for s in (42, 43, 44):
                    name = run_name("cv1", g, m, s, "", None, t)
                    d = _write_run(exp, name, "cv", t)
                    _oof(t, g, m, s).to_csv(d / "oof_points.csv", index=False)
                name = run_name("cv1", g, m, 42, "", None, t)
                f = _final(t, g, m, other_err, empty)
                d = _write_run(exp, name, "final", t, {"empty_test_groups_declared": list(empty)}, _final_cfg(f, empty))
                f.to_csv(d / "final_points.csv", index=False)
            for s in (42, 43):
                name = run_name("k100", g, "hist_gb", s, "", None, t)
                d = _write_run(exp, name, "cv", t)
                _oof(t, g, "hist_gb", s + 100).to_csv(d / "oof_points.csv", index=False)


def _fake_point_units(err, folds_csv):
    return pd.Series({p: u for p, u in _pts(CV_UNITS + HO_UNITS)}), CV_UNITS, HO_UNITS


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(acf, "point_units", _fake_point_units)  # don vi lay tu du lieu gia, khong doc data/eval
    base = tmp_path_factory.mktemp("dot7")
    exp = base / "exp"
    build_exp(exp)
    at = pd.DataFrame([{"file": f"data/grids/{g}.geojson", "tier_h3_res": t, "mean_area_km2": a}
                       for g, (t, a) in AREA.items()])
    at_path = base / "dien_tich.csv"
    at.to_csv(at_path, index=False)
    man = base / "repo" / "KE_HOACH" / "ket-qua" / "dot_dong_bang_do_man_manifest.csv"
    man.parent.mkdir(parents=True)
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/dot5_cv1_kiem_dinh_khoi.csv", "KE_HOACH/ket-qua/dot6_khoi100_mae.csv",
                           "KE_HOACH/ket-qua/dot6_khoi100_cap_min.csv"]}).to_csv(man, index=False)
    yield {"base": base, "exp": exp, "at": at_path, "man": man}
    mp.undo()


def _common(world, out, extra=(), area=True):
    return ["--exp-root", str(world["exp"]), "--out-dir", str(out), "--frozen-manifest", str(world["man"]),
            *(["--area-table", str(world["at"])] if area else []), *extra]


def _run(mod, args):
    mod.main(mod.build_parser().parse_args(args))


def _gate(path, rows):
    pd.DataFrame(rows, columns=["bien", "muc", "hop_le"]).to_csv(path, index=False)


def _f1(out, target, prefix="cv1"):
    r = pd.read_csv(out / acf.out_name(target, prefix))
    return r[r["family"] == "F1_chinh"].reset_index(drop=True)


def _sens_name(target, prefix="cv1"):
    root, ext = os.path.splitext(acf.out_name(target, prefix))
    return f"{root}{SENS_SUFFIX}{ext}"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prov(path):
    return json.loads((path.parent / f"{path.name}.provenance.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------
# C1: ten luot, doc nham bien, file dong bang
# ---------------------------------------------------------
def test_ten_luot_hau_to_bien():
    assert run_name("cv1", "h3_res_5", "hist_gb", 42, "", None, "ndwi") == "cv1__h3_res_5__hist_gb__s42__t-ndwi"
    assert run_name("cv1", "h3_res_5", "hist_gb", 42, "", None, "salinity") == "cv1__h3_res_5__hist_gb__s42"
    assert acf.out_name("salinity", "cv1") == "dot5_cv1_kiem_dinh_khoi.csv"
    assert acf.out_name("ndwi", "cv1") == "dot7_ndwi_cv1_kiem_dinh_khoi.csv"
    assert abs_.out_names("rain_chirps") == ("dot7_rain_chirps_khoi100_mae.csv", "dot7_rain_chirps_khoi100_cap_min.csv")
    assert abs_.out_names("salinity") == ("dot6_khoi100_mae.csv", "dot6_khoi100_cap_min.csv")


def test_ndwi_khong_doc_nham_luot_do_man(world, tmp_path):
    exp = tmp_path / "exp"
    build_exp(exp, targets=("salinity",))
    out = tmp_path / "out"
    out.mkdir()
    _gate(out / acf.GATE_FILE, [("ndwi", 5, True), ("ndwi", 6, True), ("ndwi", 7, True)])
    args = _common(world, out, ["--target", "ndwi"])
    args[1] = str(exp)
    with pytest.raises(SystemExit) as e:
        _run(acf, args)
    assert "LOI" in str(e.value) and "__t-ndwi" in str(e.value)
    assert not (out / acf.out_name("ndwi", "cv1")).exists()


def test_config_target_lech_loi(world, tmp_path):
    exp = tmp_path / "exp"
    build_exp(exp, targets=("ndwi",))
    cfg = exp / run_name("cv1", "h3_res_6", "linear", 43, "", None, "ndwi") / "cv" / "config.json"
    cfg.write_text(json.dumps({"target": "salinity"}), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    _gate(out / acf.GATE_FILE, [("ndwi", 5, True), ("ndwi", 6, True), ("ndwi", 7, True)])
    args = _common(world, out, ["--target", "ndwi"])
    args[1] = str(exp)
    with pytest.raises(SystemExit) as e:
        _run(acf, args)
    assert "config.target 'salinity'" in str(e.value)


@pytest.mark.parametrize("key,val", [("git_tag", "tag_khac"), ("points_ref_sha256", "ref_khac"), ("git_tag", None)])
def test_tag_hoac_dap_an_khac_nhau_loi(world, tmp_path, key, val):
    exp = tmp_path / "exp"
    build_exp(exp, targets=("rain_chirps",))
    p = exp / run_name("cv1", "s2_level_10", "idw", 44, "", None, "rain_chirps") / "cv" / "run_meta.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m[key] = val
    p.write_text(json.dumps(m), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    _gate(out / acf.GATE_FILE, [("rain_chirps", 5, True)])
    args = _common(world, out, ["--target", "rain_chirps"])
    args[1] = str(exp)
    with pytest.raises(SystemExit) as e:
        _run(acf, args)
    assert key in str(e.value)


def test_ghi_de_file_dong_bang_ma_2(world, capsys):
    frozen_dir = world["man"].parent  # <repo>/KE_HOACH/ket-qua
    for mod, extra in ((acf, []), (abs_, []), (alr, ["--targets", "ndwi"]), (afin, ["--target", "ndwi"])):
        args = _common(world, frozen_dir, extra, area=mod is not abs_)
        if mod is afin:  # final: file dau ra khong trong manifest -> dua vao manifest tam
            man = pd.read_csv(world["man"])
            pd.concat([man, pd.DataFrame({"file": ["KE_HOACH/ket-qua/dot7_ndwi_cv1_final_cap.csv"]})]).to_csv(
                world["man"], index=False)
        if mod is alr:
            man = pd.read_csv(world["man"])
            pd.concat([man, pd.DataFrame({"file": [f"KE_HOACH/ket-qua/{acf.GATE_FILE}"]})]).to_csv(
                world["man"], index=False)
        with pytest.raises(SystemExit) as e:
            _run(mod, args)
        assert e.value.code == 2, mod.__name__
        assert "tu choi ghi de file dong bang" in capsys.readouterr().out, mod.__name__
    assert [p.name for p in frozen_dir.iterdir()] == [world["man"].name]


def test_thieu_manifest_dong_bang_ma_2(world, tmp_path):
    args = _common(world, tmp_path, ["--target", "ndwi"])
    args[args.index("--frozen-manifest") + 1] = str(tmp_path / "khong_co.csv")
    with pytest.raises(SystemExit) as e:
        _run(acf, args)
    assert e.value.code == 2


# ---------------------------------------------------------
# C3: cong hoc duoc
# ---------------------------------------------------------
@pytest.fixture(scope="module")
def learned(world):
    out = world["base"] / "out_learn"
    out.mkdir()
    _gate(out / acf.GATE_FILE, [("salinity", 5, True)])  # dong bien khac phai duoc giu
    _run(alr, _common(world, out, ["--targets", "ndwi", "rain_chirps", "t2m_era5", "dsr_mcd18"]))
    return out


def test_hoc_duoc_ndwi_holm_13(learned):
    h = pd.read_csv(learned / "dot7_ndwi_hoc_duoc.csv")
    assert len(h) == 13 and (h["holm_m"] == 13).all() and h["kiem_dinh"].all()
    assert np.allclose(h["p_holm"], holm_adjust(h["p_value"].to_numpy()))
    bad = h.set_index("grid_a").loc[BAD]
    assert bad["label"] == "khong_hoc_duoc" and not bad["hoc_duoc"]
    good = h[h["grid_a"] != BAD]
    assert good["hoc_duoc"].all() and (good["delta_hat"] < 0).all() and good["vung_D"].all()
    assert (h["model_a"] == "hist_gb").all() and (h["model_b"] == "season_mean").all()
    assert h["delta_min_thr"].isna().all() and h["tost_tuong_duong"].isna().all()
    assert (h["git_tag"] == TAG).all() and (h["points_ref_sha256"] == REF).all()


def test_hoc_duoc_mua_m4_gom_s2_9(learned):
    h = pd.read_csv(learned / "dot7_rain_chirps_hoc_duoc.csv")
    k = h[h["kiem_dinh"]]
    assert sorted(k["grid_a"]) == sorted(g for g, (t, _) in AREA.items() if t == 5)
    assert "s2_level_9" in set(k["grid_a"]) and (k["holm_m"] == 4).all()
    assert np.allclose(k["p_holm"], holm_adjust(k["p_value"].to_numpy()))
    d = h[~h["kiem_dinh"]]
    assert len(d) == 9 and (d["label"] == MO_TA).all() and d["p_holm"].isna().all() and d["hoc_duoc"].isna().all()


def test_cong_kiem_dinh(learned):
    g = pd.read_csv(learned / acf.GATE_FILE).set_index(["bien", "muc"])
    assert bool(g.loc[("ndwi", 5), "hop_le"]) and bool(g.loc[("ndwi", 6), "hop_le"])
    assert not bool(g.loc[("ndwi", 7), "hop_le"])          # BAD o muc 7
    assert bool(g.loc[("rain_chirps", 5), "hop_le"])
    assert not g.loc[("rain_chirps", 6), "hop_le"] and not g.loc[("rain_chirps", 7), "hop_le"]
    assert g.loc[("rain_chirps", 6), "ly_do"].startswith("khong_kiem_dinh")
    assert not g.loc["t2m_era5", "hop_le"].any()
    assert ("salinity", 5) in g.index                        # dong bien khac giu nguyen


def test_cong_pham_vi_cap_f1(world, tmp_path):
    _run(alr, _common(world, tmp_path, ["--targets", "rain_chirps", "--scope", "cap_f1"]))
    h = pd.read_csv(tmp_path / "dot7_rain_chirps_hoc_duoc.csv")
    k = h[h["kiem_dinh"]]
    assert "s2_level_9" not in set(k["grid_a"]) and (k["holm_m"] == 3).all()


# ---------------------------------------------------------
# C2 + cong: analyze_cv_family
# ---------------------------------------------------------
def test_mua_holm_m3_giua_min_mo_ta(world, learned):
    _run(acf, _common(world, learned, ["--target", "rain_chirps"]))
    f = _f1(learned, "rain_chirps")
    k = f[f["kiem_dinh"]]
    assert len(k) == 3 and (k["muc"] == 5).all() and (k["holm_m"] == 3).all()
    assert np.allclose(k["p_holm"], holm_adjust(k["p_value"].to_numpy()))
    d = f[~f["kiem_dinh"]]
    assert len(d) == 9 and set(d["muc"]) == {6, 7} and (d["label"] == MO_TA).all()
    assert d["p_holm"].isna().all() and d["tost_tuong_duong"].isna().all() and d["delta_hat"].notna().all()
    assert d["ci_low"].notna().all()


def test_t2m_toan_mo_ta(world, learned):
    _run(acf, _common(world, learned, ["--target", "t2m_era5"]))
    r = pd.read_csv(learned / acf.out_name("t2m_era5", "cv1"))
    assert (r["label"] == MO_TA).all() and not r["kiem_dinh"].any()
    assert not r["label"].isin(["tuong_duong", "cho_giu_rieng"]).any()
    assert r["p_holm"].isna().all() and r["delta_hat"].notna().all()
    assert (r["git_tag"] == TAG).all()


def test_cong_khong_hop_le_ep_mo_ta(world, learned):
    _run(acf, _common(world, learned, ["--target", "ndwi"]))   # cong tu learned: muc 7 khong hop le
    f = _f1(learned, "ndwi")
    assert len(f) == 12 and f["muc_kiem_dinh"].all()
    m7 = f[f["muc"] == 7]
    assert len(m7) == 6 and (m7["label"] == MO_TA).all() and not m7["kiem_dinh"].any()
    assert (~m7["cong_hoc_duoc"].astype(bool)).all() and m7["p_holm"].isna().all()
    assert (m7["holm_m"] == 0).all()                         # pham vi cong: muc truot cong ra khoi ho Holm
    ok = f[f["muc"] != 7]
    assert ok["kiem_dinh"].all() and ok["cong_hoc_duoc"].astype(bool).all() and (ok["label"] != MO_TA).all()
    assert (ok["holm_m"] == 6).all()
    s = pd.read_csv(learned / _sens_name("ndwi"))           # do nhay: ho = moi muc kiem dinh (m = 12)
    s = s[s["family"] == "F1_chinh"]
    assert (s["holm_m"] == 12).all() and (s.loc[s["muc"] == 7, "label"] == MO_TA).all()


def test_thieu_cong_loi(world, tmp_path):
    with pytest.raises(SystemExit) as e:
        _run(acf, _common(world, tmp_path, ["--target", "ndwi"]))
    assert "cong" in str(e.value)
    _gate(tmp_path / acf.GATE_FILE, [("ndwi", 5, True), ("ndwi", 6, True)])  # thieu muc 7
    with pytest.raises(SystemExit) as e:
        _run(acf, _common(world, tmp_path, ["--target", "ndwi"]))
    assert "[7]" in str(e.value)


def test_do_man_khong_can_cong(world, tmp_path):
    _run(acf, _common(world, tmp_path, ["--target", "salinity"]))
    f = _f1(tmp_path, "salinity")
    assert len(f) == 12 and f["kiem_dinh"].all() and (f["holm_m"] == 12).all()
    assert f["cong_hoc_duoc"].isna().all()


# ---------------------------------------------------------
# Khoi 100 km
# ---------------------------------------------------------
def test_block_size_theo_bien(world, learned):
    _run(acf, _common(world, learned, ["--target", "rain_chirps"]))
    _run(abs_, _common(world, learned, ["--target", "rain_chirps"], area=False))
    cap = pd.read_csv(learned / "dot7_rain_chirps_khoi100_cap_min.csv")
    mae = pd.read_csv(learned / "dot7_rain_chirps_khoi100_mae.csv")
    assert len(mae) == 13 and (mae["target"] == "rain_chirps").all()
    assert not cap["kiem_dinh"].any()                       # muc min khong kiem dinh cho mua
    f = _f1(learned, "rain_chirps")
    assert np.allclose(cap["delta_min"], f.loc[f["muc"] == 7, "delta_min_thr"].iloc[0])
    assert (cap["git_tag"] == TAG).all()


def test_block_size_thieu_ket_qua_cv_loi(world, tmp_path):
    _gate(tmp_path / acf.GATE_FILE, [("ndwi", 5, True), ("ndwi", 6, True), ("ndwi", 7, True)])
    with pytest.raises(SystemExit) as e:
        _run(abs_, _common(world, tmp_path, ["--target", "ndwi"], area=False))
    assert "analyze_cv_family" in str(e.value)


# ---------------------------------------------------------
# C4: final
# ---------------------------------------------------------
def test_final_loc_khong_gian(world, learned, tmp_path):
    _run(acf, _common(world, learned, ["--target", "ndwi"]))
    _run(afin, _common(world, learned, ["--target", "ndwi"]))
    a = pd.read_csv(learned / "dot7_ndwi_cv1_final_cap.csv")
    assert a["doi_chieu_cv"].all() and a["seasons_holdout"].astype(str).str.contains("2020").sum() == 0
    exp2 = tmp_path / "exp"
    build_exp(exp2, targets=("ndwi",), other_err=1000.0)     # thoi_gian / ca_hai cuc lon: phai bi loc
    args = _common(world, learned, ["--target", "ndwi"])
    args[1] = str(exp2)
    _run(afin, args)
    b = pd.read_csv(learned / "dot7_ndwi_cv1_final_cap.csv")
    assert np.allclose(a["delta_holdout"].fillna(-9), b["delta_holdout"].fillna(-9))
    m7 = b[b["muc"] == 7]
    assert (m7["label"] == MO_TA).all()                     # muc khong qua cong: chi mo ta
    assert (b.loc[~b["kiem_dinh"], "label"] == MO_TA).all()


def test_final_dsr_nhom_rong_khai_bao(world, learned, capsys):
    _gate(learned / "gate_dsr.csv", [("dsr_mcd18", 5, True), ("dsr_mcd18", 6, True), ("dsr_mcd18", 7, True)])
    _run(afin, _common(world, learned, ["--target", "dsr_mcd18", "--gate", str(learned / "gate_dsr.csv")]))
    n = pd.read_csv(learned / "dot7_dsr_mcd18_cv1_final_nhom.csv")
    e = n[n["nhom"].isin(["thoi_gian", "ca_hai"])]
    assert len(e) == 2 * 13 * 4 and (e["trang_thai"] == EMPTY_LABEL).all() and e["MAE_diem"].isna().all()
    k = n[n["nhom"] == "khong_gian"]
    assert (k["trang_thai"] == "co_du_lieu").all() and k["MAE_diem"].notna().all()
    assert EMPTY_LABEL in capsys.readouterr().out


def test_final_nhom_rong_khong_khai_bao_loi(world, tmp_path):
    exp = tmp_path / "exp"
    build_exp(exp, targets=("dsr_mcd18",))
    p = exp / run_name("cv1", "h3_res_7", "idw", 42, "", None, "dsr_mcd18") / "final"
    m = json.loads((p / "run_meta.json").read_text(encoding="utf-8"))
    m["empty_test_groups_declared"] = []
    (p / "run_meta.json").write_text(json.dumps(m), encoding="utf-8")
    c = json.loads((p / "config.json").read_text(encoding="utf-8"))
    for g in ("thoi_gian", "ca_hai"):
        c["test_metrics_by_group_mean_over_seeds"][g].pop("status", None)
    (p / "config.json").write_text(json.dumps(c), encoding="utf-8")
    _gate(tmp_path / acf.GATE_FILE, [("dsr_mcd18", 5, True), ("dsr_mcd18", 6, True), ("dsr_mcd18", 7, True)])
    args = _common(world, tmp_path, ["--target", "dsr_mcd18"])
    args[1] = str(exp)
    with pytest.raises(SystemExit) as e:
        _run(afin, args)
    assert "khong khai bao" in str(e.value)


def test_tested_tiers_bang_co_dinh():
    assert TESTED_TIERS["rain_chirps"] == {5} and TESTED_TIERS["t2m_era5"] == set() == TESTED_TIERS["rh_era5"]
    assert all(TESTED_TIERS[t] == {5, 6, 7} for t in ("salinity", "ndwi", "dsr_mcd18"))


# ---------------------------------------------------------
# Pham vi Holm F1: cong (mac dinh) / muc_kiem_dinh (do nhay) - so voi code cu 114bb63f
# ---------------------------------------------------------
OLD_COMMIT = "114bb63f"
NO_M_COLS = ["delta_hat", "ci_low", "ci_high", "ci_tost_low", "ci_tost_high", "se", "seed_deltas", "delta_min_thr",
             "delta_ref_level"]


def _git_show(rel):
    r = subprocess.run(["git", "show", f"{OLD_COMMIT}:{rel}"], cwd=ROOT, capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    return r.stdout.decode("utf-8")


def _exec_module(name, rel):
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(ROOT, rel)  # ROOT cua script = goc repo
    exec(compile(_git_show(rel), f"<{OLD_COMMIT}>/{rel}", "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def old_acf():
    """analyze_cv_family + dot7_rules nguyen van o commit truoc khi doi pham vi Holm."""
    rules = _exec_module("dot7_rules_old", "src/training/dot7_rules.py")
    assert not hasattr(rules, "holm_tiers")
    saved = sys.modules.get("training.dot7_rules")
    sys.modules["training.dot7_rules"] = rules
    try:
        mod = _exec_module("analyze_cv_family_old", "scripts/analyze_cv_family.py")
    finally:
        if saved is None:
            sys.modules.pop("training.dot7_rules", None)
        else:
            sys.modules["training.dot7_rules"] = saved
    assert mod.compare_tiered is rules.compare_tiered
    mod.point_units = _fake_point_units
    return mod


@pytest.fixture(scope="module")
def scoped(world, old_acf):
    """ndwi, cong muc 7 truot: code cu va code moi (mac dinh) cung --gate (cot cong_file trung)."""
    base = world["base"] / "scoped"
    base.mkdir()
    gate = base / "gate_ndwi.csv"
    _gate(gate, [("ndwi", 5, True), ("ndwi", 6, True), ("ndwi", 7, False)])
    out = {"gate": gate}
    for name, mod in (("old", old_acf), ("new", acf)):
        out[name] = base / name
        out[name].mkdir()
        _run(mod, _common(world, out[name], ["--target", "ndwi", "--gate", str(gate)]))
    return out


def test_holm_m_suy_tu_bang_cong():
    real = {"dsr_mcd18": {5: False, 6: True, 7: True}, "ndwi": {5: True, 6: True, 7: True},
            "rain_chirps": {5: True, 6: False, 7: False}, "t2m_era5": {5: False, 6: False, 7: False},
            "rh_era5": {5: False, 6: False, 7: False}, "salinity": None}
    want = {"dsr_mcd18": 9, "ndwi": 12, "rain_chirps": 3, "t2m_era5": 0, "rh_era5": 0, "salinity": 12}
    for t, g in real.items():
        assert expected_f1_m(holm_tiers(t, g, "cong")) == want[t], t
        assert expected_f1_m(holm_tiers(t, g, "muc_kiem_dinh")) == EXPECTED_F1_M[t], t
        assert scopes_differ(t, g) == (t == "dsr_mcd18"), t
    assert holm_tiers("dsr_mcd18", real["dsr_mcd18"]) == {6, 7}
    with pytest.raises(ValueError):
        holm_tiers("ndwi", None, "cap_f1")
    with pytest.raises(ValueError):  # muc Holm ngoai muc kiem dinh
        compare_tiered(None, None, pd.DataFrame(), "rain_chirps", levels=None, holm_tiers={6})
    pairs = pd.DataFrame({"grid_a": list("abcde"), "grid_b": list("fghij"), "tier": [7] * 5})
    with pytest.raises(AssertionError):  # so cap trong ho khac m suy tu cong
        compare_tiered(None, None, pairs, "ndwi", levels=None, expected_m=expected_f1_m({7}), holm_tiers={7})


def test_do_nhay_byte_giong_code_cu(scoped):
    """File do nhay (m = moi muc kiem dinh) = dung byte file chinh cua code cu."""
    old = scoped["old"] / acf.out_name("ndwi", "cv1")
    assert (scoped["new"] / _sens_name("ndwi")).read_bytes() == old.read_bytes()
    assert not (scoped["old"] / _sens_name("ndwi")).exists()
    assert (scoped["new"] / acf.out_name("ndwi", "cv1")).read_bytes() != old.read_bytes()


def test_holm_cong_bo_muc_truot_cong(scoped):
    main = pd.read_csv(scoped["new"] / acf.out_name("ndwi", "cv1"))
    sens = pd.read_csv(scoped["new"] / _sens_name("ndwi"))
    assert list(main.columns) == list(sens.columns)
    assert main[["family", "cmp_id"]].equals(sens[["family", "cmp_id"]])
    for fam in ("F1_chinh", "phu_linear", "phu_idw", "do_nhay_chi_s42", "do_nhay_chi_s43", "do_nhay_chi_s44"):
        f = main[main["family"] == fam]
        k = f[f["kiem_dinh"]]
        assert len(k) == 6 and set(k["muc"]) == {5, 6} and (k["holm_m"] == 6).all(), fam
        assert np.allclose(k["p_holm"], holm_adjust(k["p_value"].to_numpy())), fam
        m7 = f[f["muc"] == 7]
        assert len(m7) == 6 and (m7["label"] == MO_TA).all() and (m7["holm_m"] == 0).all(), fam
        assert m7["muc_kiem_dinh"].all() and (~m7["cong_hoc_duoc"].astype(bool)).all(), fam
        assert m7[["p_value", "p_holm", "tost_tuong_duong"]].isna().all().all(), fam
        s = sens[sens["family"] == fam].set_index("cmp_id").loc[k["cmp_id"]]
        assert np.allclose(k["p_value"], s["p_value"], rtol=0, atol=0), fam
        assert (k["p_holm"].to_numpy() <= s["p_holm"].to_numpy()).all(), fam   # ho nho hon -> Holm khong chat hon
    s2 = main[main["family"] == "phu_S2_ti_le_1.2-1.5"]
    assert len(s2) and (s2["muc"] == 5).all() and s2["kiem_dinh"].all()
    for c in NO_M_COLS:  # khong phu thuoc m
        assert main[c].astype(str).equals(sens[c].astype(str)), c


def test_check_sealed_cot_khong_phu_thuoc_m(scoped, capsys):
    main = str(scoped["new"] / acf.out_name("ndwi", "cv1"))
    sens = str(scoped["new"] / _sens_name("ndwi"))
    assert check_sealed.main([main, sens, "--cols", *NO_M_COLS]) == 0
    assert check_sealed.main([main, sens]) == 2
    out = capsys.readouterr().out
    assert "p_holm" in out and "holm_m" in out and "KHONG DAT" in out


def test_provenance_pham_vi_holm(scoped):
    main = scoped["new"] / acf.out_name("ndwi", "cv1")
    sens = scoped["new"] / _sens_name("ndwi")
    pm, ps = _prov(main), _prov(sens)
    assert pm["holm_scope"] == "cong" and pm["holm_m_f1"] == 6 and pm["muc_holm"] == [5, 6]
    assert pm["muc_truot_cong"] == [7] and pm["holm_scopes_khac_nhau"] is True
    assert pm["holm_m_theo_ho"]["F1_chinh"] == 6 and pm["holm_m_theo_ho"]["phu_S2_ti_le_1.2-1.5"] == 3
    assert ps["holm_scope"] == "muc_kiem_dinh" and ps["holm_m_f1"] == 12 and ps["holm_m_theo_ho"]["F1_chinh"] == 12
    for p, info in ((main, pm), (sens, ps)):
        assert info["csv_sha256"] == _sha(p) and info["gate_sha256"] == _sha(scoped["gate"])
        assert info["gate_hop_le"] == {"5": True, "6": True, "7": False} and info["n_luot"] == 117


@pytest.mark.parametrize("target,rows", [
    ("ndwi", [("ndwi", 5, True), ("ndwi", 6, True), ("ndwi", 7, True)]),
    ("rain_chirps", [("rain_chirps", 5, True), ("rain_chirps", 6, False), ("rain_chirps", 7, False)]),
    ("t2m_era5", [("t2m_era5", m, False) for m in (5, 6, 7)]),
    ("salinity", None),
])
def test_moi_muc_qua_cong_file_chinh_byte_giong_code_cu(world, old_acf, tmp_path, target, rows):
    extra = ["--target", target]
    if rows is not None:
        _gate(tmp_path / "gate.csv", rows)
        extra += ["--gate", str(tmp_path / "gate.csv")]
    outs = {}
    for name, mod in (("old", old_acf), ("new", acf)):
        outs[name] = tmp_path / name
        outs[name].mkdir()
        _run(mod, _common(world, outs[name], extra))
    fn = acf.out_name(target, "cv1")
    assert (outs["new"] / fn).read_bytes() == (outs["old"] / fn).read_bytes()
    assert not (outs["new"] / _sens_name(target)).exists()
    pv = _prov(outs["new"] / fn)
    assert pv["holm_scopes_khac_nhau"] is False and pv["holm_m_f1"] == EXPECTED_F1_M[target]


def test_holm_scope_muc_kiem_dinh_chi_ghi_file_do_nhay(world, scoped, tmp_path, capsys):
    capsys.readouterr()
    _run(acf, _common(world, tmp_path, ["--target", "ndwi", "--gate", str(scoped["gate"]),
                                        "--holm-scope", "muc_kiem_dinh", "--no-table"]))
    assert not (tmp_path / acf.out_name("ndwi", "cv1")).exists()
    assert (tmp_path / _sens_name("ndwi")).read_bytes() == (scoped["new"] / _sens_name("ndwi")).read_bytes()
    out = capsys.readouterr().out
    assert "delta_hat" not in out and "cho_giu_rieng" not in out and "tuong_duong" not in out


def test_final_theo_pham_vi_holm(world, scoped):
    d, g = scoped["new"], ["--gate", str(scoped["gate"])]
    _run(afin, _common(world, d, ["--target", "ndwi", *g]))
    c = pd.read_csv(d / "dot7_ndwi_cv1_final_cap.csv")
    assert c["doi_chieu_cv"].all() and (c.loc[c["kiem_dinh"], "holm_m"] == 6).all()
    assert (c.loc[c["muc"] == 7, "label"] == MO_TA).all() and (c.loc[c["muc"] == 7, "holm_m"] == 0).all()
    assert _prov(d / "dot7_ndwi_cv1_final_cap.csv")["holm_scope"] == "cong"
    _run(afin, _common(world, d, ["--target", "ndwi", *g, "--holm-scope", "muc_kiem_dinh"]))
    s = pd.read_csv(d / f"dot7_ndwi_cv1_final_cap{SENS_SUFFIX}.csv")
    assert (d / f"dot7_ndwi_cv1_final_nhom{SENS_SUFFIX}.csv").exists()
    assert s["doi_chieu_cv"].all() and (s["holm_m"] == 12).all() and (s.loc[s["muc"] == 7, "label"] == MO_TA).all()
    assert c["delta_holdout"].astype(str).equals(s["delta_holdout"].astype(str))


def test_final_khac_pham_vi_cv_loi(world, scoped, tmp_path):
    """CV tinh voi muc 7 truot cong, final voi cong khac -> tap cap kiem dinh khac -> LOI."""
    _gate(tmp_path / "gate_all.csv", [("ndwi", 5, True), ("ndwi", 6, True), ("ndwi", 7, True)])
    args = _common(world, tmp_path, ["--target", "ndwi", "--gate", str(tmp_path / "gate_all.csv"),
                                     "--cv-dir", str(scoped["new"])])
    with pytest.raises(SystemExit) as e:
        _run(afin, args)
    assert "kiem dinh" in str(e.value)


def test_block_size_theo_pham_vi_holm(world, scoped, tmp_path):
    d, g = scoped["new"], ["--gate", str(scoped["gate"])]
    _run(abs_, _common(world, d, ["--target", "ndwi", *g, "--no-table"], area=False))
    _run(abs_, _common(world, d, ["--target", "ndwi", *g, "--holm-scope", "muc_kiem_dinh"], area=False))
    a = pd.read_csv(d / "dot7_ndwi_khoi100_cap_min.csv")
    b = pd.read_csv(d / f"dot7_ndwi_khoi100_cap_min{SENS_SUFFIX}.csv")
    assert not a["kiem_dinh"].any() and not b["kiem_dinh"].any()   # muc min truot cong o ca hai pham vi
    assert a.equals(b)
    pa, pb = _prov(d / "dot7_ndwi_khoi100_cap_min.csv"), _prov(d / f"dot7_ndwi_khoi100_cap_min{SENS_SUFFIX}.csv")
    assert pa["cv_ref"].endswith("dot7_ndwi_cv1_kiem_dinh_khoi.csv")
    assert pb["cv_ref"].endswith(f"dot7_ndwi_cv1_kiem_dinh_khoi{SENS_SUFFIX}.csv")
    _gate(tmp_path / "gate_all.csv", [("ndwi", 5, True), ("ndwi", 6, True), ("ndwi", 7, True)])
    with pytest.raises(SystemExit) as e:  # cong khac cong cua file CV
        _run(abs_, _common(world, tmp_path, ["--target", "ndwi", "--gate", str(tmp_path / "gate_all.csv"),
                                             "--ref-dir", str(d)], area=False))
    assert "kiem_dinh muc min" in str(e.value)

"""Script tong hop Dot 7 (CHG-25): analyze_learnability -> analyze_cv_family -> analyze_block_size -> analyze_final.

Du lieu gia (tmp_path): 13 luoi that (ten + dien tich nhu bang doi chieu), 10 khoi CV + 2 khoi giu rieng x 30 diem x
2 mua; |e| HistGB = U(0,5; 1,5); season_mean = HistGB + 1 (hoc duoc ro) tru luoi BAD (chenh doi dau theo khoi ~ 0).
"""
import json
import os
import sys
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
from run_experiments import run_name  # noqa: E402
from training.dot7_rules import MO_TA, TESTED_TIERS  # noqa: E402
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
    assert len(f) == 12 and (f["holm_m"][f["muc_kiem_dinh"]] == 12).all()
    m7 = f[f["muc"] == 7]
    assert len(m7) == 6 and (m7["label"] == MO_TA).all() and not m7["kiem_dinh"].any()
    assert (~m7["cong_hoc_duoc"].astype(bool)).all() and m7["p_holm"].isna().all()
    ok = f[f["muc"] != 7]
    assert ok["kiem_dinh"].all() and ok["cong_hoc_duoc"].astype(bool).all() and (ok["label"] != MO_TA).all()


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

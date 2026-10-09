"""Kiem tuong tac mo hinh x khung (analyze_model_interaction). Du lieu gia: 13 luoi x 10 khoi CV x 30 diem x 2 mua
(test_analyze_dot7_tonghop); rf = HistGB + nhieu doi xung theo khoi (+ tuong tac cong vao mot luoi). File F1 gia tinh
bang compare_family tren 4 cap roi nhau; mot test tich hop dung file F1 that cua analyze_cv_family (do man, 12 cap).
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
sys.path.insert(0, os.path.dirname(__file__))

import analyze_cv_family as acf  # noqa: E402
import analyze_model_interaction as ami  # noqa: E402
import test_analyze_dot7_tonghop as base  # noqa: E402
from run_experiments import run_name  # noqa: E402
from training.block_stats import compare_family  # noqa: E402
from training.dot7_rules import main_family, model_name  # noqa: E402
from training.model_interaction import COLS, NONE, SIG, interaction_family, verdict  # noqa: E402

RF = "random_forest"
E6E = "nckh-test-e6e"
T = "ndwi"
SCHEMES = (42, 43, 44)
PAIRS = [("h3_res_5", "square_utm_17087m", 5), ("h3_res_6", "square_utm_6458m", 6),
         ("h3_res_7", "s2_level_12", 7), ("latlon_0.0222deg", "square_utm_2441m", 7)]  # luoi roi nhau
BUMP = "h3_res_6"  # grid_a cua cap muc 6
SIGNS = np.array([1, 1, -1, 1, -1, -1, 1, -1, 1, -1])  # dau theo khoi: sum s_u (1 + 0,1 u) = -0,9 ~ 0
ALL = {5: True, 6: True, 7: True}
NO6 = {5: True, 6: False, 7: True}
TAGS = [base.TAG, E6E]


def _noise(grid, units):
    """Nhieu doi xung theo khoi 0,05 c_g s_u (1 + 0,1 u); |nhieu| < 0,15 nen |e| giu dau."""
    c = 0.5 + (zlib.crc32(grid.encode()) % 1000) / 1000
    idx = np.array([int(u[1:]) for u in units])
    return 0.05 * c * SIGNS[idx] * (1 + 0.1 * idx)


def build_exp(exp, target=T, bump=None, drop=None, with_histgb=True):
    """Luot cv HistGB (tag base.TAG) + rf (tag E6E); bump {grid: {cach chia: x}} cong vao sai so rf; drop (grid, diem)
    bo khoi rf."""
    for g in base.AREA:
        for s in SCHEMES:
            h = base._oof(target, g, "hist_gb", s)
            if with_histgb:
                d = base._write_run(exp, run_name("cv1", g, "hist_gb", s, "", None, target), "cv", target)
                h.to_csv(d / "oof_points.csv", index=False)
            r = h.copy()
            r["err"] = r["err"] + _noise(g, r["unit_id"])
            r["err"] += (bump or {}).get(g, {}).get(s, 0.0)
            if drop and drop[0] == g:
                r = r[r["point_id"] != drop[1]]
            d = base._write_run(exp, run_name("cv1", g, RF, s, "", None, target), "cv", target, {"git_tag": E6E})
            r.to_csv(d / "oof_points.csv", index=False)


def _gate(path, target, hop, model=None):
    g = pd.DataFrame([(target, m, v) for m, v in hop.items()], columns=["bien", "muc", "hop_le"])
    if model:
        g.insert(1, "model", model)
    g.to_csv(path, index=False)


def write_f1(exp, out, target=T, h_gate=ALL, m_gate=ALL, pairs=PAIRS):
    """File F1 gia hai ben bang compare_family (cung tap chung theo mo hinh nhu analyze_cv_family) + bang cong."""
    out.mkdir(parents=True, exist_ok=True)
    pu, cv, ho = base._fake_point_units(None, None)
    for model, rm, gate, tag in (("hist_gb", "hist_gb", h_gate, base.TAG), ("rf", RF, m_gate, E6E)):
        e, n = acf.common_subset(acf.load_errors(str(exp), "cv1", rm, list(SCHEMES), target=target))
        o = compare_family(e, pu, [(a, b) for a, b, _ in pairs], cv_units=cv, holdout_units=ho,
                           holdout_seasons=(2020,), mode="cv", delta_min=0.05, delta_min_kind="abs", alpha=0.05,
                           model=rm, family=main_family(model))
        o.insert(0, "n_common_point_seasons", n)
        o["target"], o["git_tag"], o["points_ref_sha256"] = target, tag, base.REF
        o["muc"] = o["delta_ref_level"] = [m for *_, m in pairs]
        o["muc_kiem_dinh"] = o["muc"].isin(base.TESTED_TIERS[target])
        o["cong_hoc_duoc"] = [gate[m] for m in o["muc"]] if gate is not None else np.nan
        o.to_csv(out / model_name(acf.out_name(target, "cv1"), model), index=False)
        if gate is not None:
            _gate(out / acf.gate_name(model), target, gate, None if model == "hist_gb" else "rf")


def run_ami(exp, out, target=T, allowed=TAGS, extra=(), man=None):
    if man is None:
        man = out.parent / "man.csv"
        pd.DataFrame({"file": ["KE_HOACH/ket-qua/khong_co.csv"]}).to_csv(man, index=False)
    args = ["--model", "rf", "--targets", target, "--exp-root", str(exp), "--out-dir", str(out),
            "--frozen-manifest", str(man), "--no-table", *(["--allowed-tags", *allowed] if allowed else []), *extra]
    ami.main(ami.build_parser().parse_args(args))
    return (pd.read_csv(out / f"dot7_{target}_tuong_tac_mo_hinh__rf.csv"),
            pd.read_csv(out / "dot7_tuong_tac_mo_hinh_tong_ket__rf.csv").iloc[0])


def _prov(p):
    return json.loads(open(f"{p}.provenance.json", encoding="utf-8").read())


@pytest.fixture(autouse=True)
def _units(monkeypatch):
    monkeypatch.setattr(acf, "point_units", base._fake_point_units)


@pytest.fixture(scope="module")
def exps(tmp_path_factory):
    root = tmp_path_factory.mktemp("tuong_tac")
    out = {}
    for k, bump in (("nhieu", None), ("cong", {BUMP: {s: 0.3 for s in SCHEMES}}),
                    ("lech_cach_chia", {BUMP: {42: 0.3, 43: 0.3, 44: -0.1}})):
        build_exp(root / k, bump=bump)
        out[k] = root / k
    return out


# ---------------------------------------------------------
# (a) nhieu doi xung -> khong tuong tac
# ---------------------------------------------------------
def test_nhieu_doi_xung_khong_tuong_tac(exps, tmp_path):
    out = tmp_path / "kq"
    write_f1(exps["nhieu"], out)
    x, s = run_ami(exps["nhieu"], out)
    assert len(x) == 4 and (x["label"] == NONE).all() and not x["chua_ro"].any()
    assert (x["p_value"] > 0.2).all() and (x["holm_m"] == 4).all()
    assert (x["G"] == 10).all() and x["p_exact"].all() and (x["n_perm"] == 1024).all()
    assert (x["n_seeds"] == 3).all() and x["seed_deltas"].str.count(";").eq(2).all()
    assert s["ket_luan"] == "khong_phu_thuoc_mo_hinh" and s["n_cap"] == 4 and s["n_tuong_tac_y_nghia"] == 0
    h = pd.read_csv(out / acf.out_name(T, "cv1")).set_index(["grid_a", "grid_b"])
    assert np.allclose(x["delta_hat_histgb"], h.loc[list(zip(x["grid_a"], x["grid_b"])), "delta_hat"], rtol=0,
                       atol=1e-12)
    assert np.allclose(x["I_hat"], x["delta_hat_model"] - x["delta_hat_histgb"], rtol=0, atol=1e-12)
    assert (x["n_common_point_seasons"] == 10 * base.NPT * 2).all() and list(x.columns) == ami.OUT_COLS


# ---------------------------------------------------------
# (b) tuong tac lon cung dau moi khoi o mot cap
# ---------------------------------------------------------
def test_tuong_tac_lon_mot_cap(exps, tmp_path):
    out = tmp_path / "kq"
    write_f1(exps["cong"], out)
    x, s = run_ami(exps["cong"], out)
    hit = x["grid_a"] == BUMP
    r = x[hit].iloc[0]
    assert r["label"] == SIG and not r["chua_ro"] and r["k_over_G"] == "10/10"
    assert 0.29 < r["I_hat"] < 0.31 and r["ci_low"] > 0 and not r["tost_tuong_tac"]
    assert np.isclose(r["p_value"], 2 / 1024) and np.isclose(r["p_holm"], 4 * 2 / 1024)
    assert all(float(v) > 0 for v in r["seed_deltas"].split(";")) and r["seeds_same_dir"]
    assert (x.loc[~hit, "label"] == NONE).all() and not x.loc[~hit, "chua_ro"].any()
    assert s["ket_luan"] == "co_tuong_tac" and s["n_tuong_tac_y_nghia"] == 1 and s["n_chua_ro"] == 0


def test_cach_chia_khong_cung_dau_chua_ro(exps, tmp_path):
    out = tmp_path / "kq"
    write_f1(exps["lech_cach_chia"], out)
    x, s = run_ami(exps["lech_cach_chia"], out)
    r = x[x["grid_a"] == BUMP].iloc[0]
    seeds = [float(v) for v in r["seed_deltas"].split(";")]
    assert r["p_holm"] < 0.05 and r["I_hat"] > 0 and seeds[2] < 0 < min(seeds[:2])
    assert r["label"] == NONE and r["chua_ro"] and not r["seeds_same_dir"]
    assert s["ket_luan"] == "chua_ro" and s["n_chua_ro"] == 1 and s["n_tuong_tac_y_nghia"] == 0


# ---------------------------------------------------------
# (c) cap truot cong mot ben bi loai
# ---------------------------------------------------------
@pytest.mark.parametrize("h_gate,m_gate", [(NO6, ALL), (ALL, NO6)])
def test_cap_truot_cong_mot_ben_bi_loai(exps, tmp_path, h_gate, m_gate):
    out = tmp_path / "kq"
    write_f1(exps["cong"], out, h_gate=h_gate, m_gate=m_gate)
    x, s = run_ami(exps["cong"], out)
    assert 6 not in set(x["muc"]) and BUMP not in set(x["grid_a"])  # cap co tuong tac nam o muc 6
    assert len(x) == 3 and (x["holm_m"] == 3).all() and (x["label"] == NONE).all()
    assert s["n_cap_kiem_dinh"] == 4 and s["n_cap"] == 3 and s["ket_luan"] == "khong_phu_thuoc_mo_hinh"
    pv = _prov(out / "dot7_ndwi_tuong_tac_mo_hinh__rf.csv")
    assert pv["holm_m"] == 3
    assert pv["dau_vao"]["hist_gb"]["gate_hop_le"]["6"] is h_gate[6]
    assert pv["dau_vao"]["mo_hinh"]["gate_hop_le"]["6"] is m_gate[6]


@pytest.mark.parametrize("target,n_tested", [(T, 4), ("t2m_era5", 0)])
def test_khong_co_cap_khong_doc_sai_so(exps, tmp_path, target, n_tested):
    exp = exps["nhieu"] if target == T else tmp_path / "exp"
    if target != T:
        build_exp(exp, target=target)
    out = tmp_path / "kq"
    off = {5: False, 6: False, 7: False}
    write_f1(exp, out, target=target, h_gate=ALL if target == T else off, m_gate=off)
    empty = tmp_path / "exp_rong"
    empty.mkdir()
    x, s = run_ami(empty, out, target=target)  # exp rong: neu doc sai so se LOI
    assert len(x) == 0 and list(x.columns) == ami.OUT_COLS
    assert s["n_cap"] == 0 and s["ket_luan"] == "khong_co_cap" and s["n_cap_kiem_dinh"] == n_tested


# ---------------------------------------------------------
# (d) tap diem lech -> LOI
# ---------------------------------------------------------
@pytest.mark.filterwarnings("ignore:do_nhay_rf")  # khoi u3 con 29 diem -> bi loai o file F1 rf gia
def test_tap_diem_lech_loi(tmp_path):
    exp = tmp_path / "exp"
    build_exp(exp, drop=("s2_level_10", "u3_p0"))  # luoi khong nam trong cap nao van lam lech tap chung
    out = tmp_path / "kq"
    write_f1(exp, out)
    with pytest.raises(SystemExit) as e:
        run_ami(exp, out)
    assert "tap (diem, mua) lech" in str(e.value)
    assert not (out / "dot7_ndwi_tuong_tac_mo_hinh__rf.csv").exists()


def test_loi_bien_sau_khong_ghi_bien_truoc(exps, tmp_path):
    out = tmp_path / "kq"
    write_f1(exps["nhieu"], out)
    with pytest.raises(SystemExit) as e:  # salinity khong co file F1 -> LOI sau khi ndwi da tinh xong
        run_ami(exps["nhieu"], out, extra=["--targets", T, "salinity"])
    assert "thieu ket qua analyze_cv_family" in str(e.value)
    assert not any("tuong_tac" in p.name for p in out.iterdir())


# ---------------------------------------------------------
# (e) khong ghi file nam trong manifest dong bang
# ---------------------------------------------------------
@pytest.mark.parametrize("frozen", ["dot7_ndwi_tuong_tac_mo_hinh__rf.csv",
                                    "dot7_tuong_tac_mo_hinh_tong_ket__rf.csv.provenance.json"])
def test_khong_ghi_file_dong_bang(exps, tmp_path, capsys, frozen):
    out = tmp_path / "repo" / "KE_HOACH" / "ket-qua"
    write_f1(exps["nhieu"], out)
    man = out / "man.csv"
    pd.DataFrame({"file": [f"KE_HOACH/ket-qua/{frozen}"]}).to_csv(man, index=False)
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(SystemExit) as e:
        run_ami(exps["nhieu"], out, man=man)
    assert e.value.code == 2 and "tu choi ghi de file dong bang" in capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == before
    with pytest.raises(SystemExit) as e:  # thieu manifest -> ma 2
        run_ami(exps["nhieu"], out, man=out / "khong_co_manifest.csv")
    assert e.value.code == 2 and sorted(p.name for p in out.iterdir()) == before


def test_mac_dinh_hai_manifest():
    assert ami.build_parser().parse_args(["--model", "rf"]).frozen_manifest == acf.FROZEN_MANIFESTS
    assert len(acf.FROZEN_MANIFESTS) == 2
    assert ami.build_parser().parse_args(["--model", "mlp"]).targets == list(ami.ams.TARGETS)
    with pytest.raises(SystemExit):
        ami.build_parser().parse_args(["--model", "hist_gb"])


# ---------------------------------------------------------
# Tag, doi chieu file F1
# ---------------------------------------------------------
@pytest.mark.parametrize("allowed,msg", [(None, "git_tag khac nhau"), ([E6E], "ngoai --allowed-tags")])
def test_hai_mo_hinh_khac_tag_can_allowed(exps, tmp_path, allowed, msg):
    out = tmp_path / "kq"
    write_f1(exps["nhieu"], out)
    with pytest.raises(SystemExit) as e:
        run_ami(exps["nhieu"], out, allowed=allowed)
    assert msg in str(e.value)
    assert not (out / "dot7_ndwi_tuong_tac_mo_hinh__rf.csv").exists()


def test_tag_va_provenance(exps, tmp_path):
    out = tmp_path / "kq"
    write_f1(exps["nhieu"], out)
    x, s = run_ami(exps["nhieu"], out, extra=["--tags-reason", "ly do thu"])
    assert (x["git_tags"] == f"hist_gb:{base.TAG};rf:{E6E}").all() and s["git_tags"] == x["git_tags"].iloc[0]
    assert (x["points_ref_sha256"] == base.REF).all()
    pv = _prov(out / "dot7_ndwi_tuong_tac_mo_hinh__rf.csv")
    assert pv["git_tags"] == {"hist_gb": [base.TAG], "mo_hinh": [E6E]} and pv["git_tags_seen"] == sorted(TAGS)
    assert pv["git_tags_allowed"] == sorted(TAGS) and pv["ly_do_nhieu_tag"] == "ly do thu"
    assert pv["model"] == "rf" and pv["run_model"] == RF and pv["holm_m"] == 4 and pv["bao_cao"] == "kiem_dinh"
    assert pv["dau_vao"]["mo_hinh"]["cv_file"].endswith("dot7_ndwi_cv1_kiem_dinh_khoi__rf.csv")
    assert pv["dau_vao"]["hist_gb"]["n_luot"] == 39 and pv["dau_vao"]["mo_hinh"]["git_tags"] == [E6E]
    ps = _prov(out / "dot7_tuong_tac_mo_hinh_tong_ket__rf.csv")
    assert ps["holm_m"] == {T: 4} and ps["bang_cap"][T].endswith("dot7_ndwi_tuong_tac_mo_hinh__rf.csv")


def test_delta_f1_lech_loi(exps, tmp_path):
    out = tmp_path / "kq"
    write_f1(exps["nhieu"], out)
    p = out / acf.out_name(T, "cv1")
    h = pd.read_csv(p)
    h.loc[1, "delta_hat"] += 0.01  # file F1 khong tinh tu cac luot dang doc
    h.to_csv(p, index=False)
    with pytest.raises(SystemExit) as e:
        run_ami(exps["nhieu"], out)
    assert "Delta_hat tinh lai khac file F1" in str(e.value) and "hist_gb" in str(e.value)


def test_n_common_f1_lech_loi(exps, tmp_path):
    out = tmp_path / "kq"
    write_f1(exps["nhieu"], out)
    p = out / model_name(acf.out_name(T, "cv1"), "rf")
    m = pd.read_csv(p)
    m["n_common_point_seasons"] -= 2
    m.to_csv(p, index=False)
    with pytest.raises(SystemExit) as e:
        run_ami(exps["nhieu"], out)
    assert "so (diem, mua) chung" in str(e.value) and "mo_hinh" in str(e.value)


# ---------------------------------------------------------
# Ham thuan
# ---------------------------------------------------------
def test_verdict():
    assert verdict(0, 0, 0) == "khong_co_cap"
    assert verdict(3, 0, 0) == "khong_phu_thuoc_mo_hinh"
    assert verdict(3, 1, 1) == "co_tuong_tac"
    assert verdict(3, 0, 2) == "chua_ro"


def _err(exp, rm):
    return acf.common_subset(acf.load_errors(str(exp), "cv1", rm, list(SCHEMES), target=T))[0]


def test_interaction_family_kiem_dau_vao(exps):
    pu, cv, ho = base._fake_point_units(None, None)
    eh, em = _err(exps["nhieu"], "hist_gb"), _err(exps["nhieu"], RF)
    kw = dict(ref_model="hist_gb", model=RF, cv_units=cv, holdout_units=ho, holdout_seasons=(2020,))
    pairs = [(a, b) for a, b, _ in PAIRS]
    assert list(interaction_family(eh, em, pu, [], [], **kw).columns) == COLS
    with pytest.raises(ValueError, match="mua giu rieng"):
        interaction_family(eh, em.assign(season=np.where(em["season"] == 2015, 2020, em["season"])), pu, pairs,
                           [0.05] * 4, **kw)
    with pytest.raises(ValueError, match="cach chia"):
        interaction_family(eh, em[em["seed"] != 44], pu, pairs, [0.05] * 4, **kw)
    with pytest.raises(ValueError, match="can 4 cach chia"):
        interaction_family(eh, em, pu, pairs, [0.05] * 4, n_schemes=4, **kw)
    with pytest.raises(ValueError, match="delta_min_thr"):
        interaction_family(eh, em, pu, pairs, [0.05, np.nan, 0.05, 0.05], **kw)
    with pytest.raises(ValueError, match="chi co mo hinh"):
        interaction_family(eh, eh, pu, pairs, [0.05] * 4, **kw)
    same = interaction_family(eh, eh.assign(model=RF), pu, pairs, [0.05] * 4, **kw)  # mo hinh = HistGB
    assert np.allclose(same["I_hat"], 0) and (same["p_value"] == 1).all() and same["tost_tuong_tac"].all()


# ---------------------------------------------------------
# Tich hop: file F1 that cua analyze_cv_family (do man: HistGB khong cong, rf cong rieng)
# ---------------------------------------------------------
def test_tich_hop_do_man_voi_analyze_cv_family(tmp_path):
    exp = tmp_path / "exp"
    base.build_exp(exp, targets=("salinity",))
    build_exp(exp, target="salinity", with_histgb=False)
    at = pd.DataFrame([{"file": f"data/grids/{g}.geojson", "tier_h3_res": t, "mean_area_km2": a}
                       for g, (t, a) in base.AREA.items()])
    at.to_csv(tmp_path / "dien_tich.csv", index=False)
    out = tmp_path / "kq"
    out.mkdir()
    man = tmp_path / "man.csv"
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/khong_co.csv"]}).to_csv(man, index=False)
    common = ["--target", "salinity", "--exp-root", str(exp), "--out-dir", str(out), "--frozen-manifest", str(man),
              "--area-table", str(tmp_path / "dien_tich.csv"), "--no-table"]
    acf.main(acf.build_parser().parse_args(common))
    _gate(out / acf.gate_name("rf"), "salinity", ALL, "rf")
    acf.main(acf.build_parser().parse_args(common + ["--model", "rf"]))
    x, s = run_ami(exp, out, target="salinity", man=man)
    f1 = pd.read_csv(out / acf.out_name("salinity", "cv1"))
    f1 = f1[f1["family"] == "F1_chinh"].set_index(["grid_a", "grid_b"])
    assert len(x) == 12 and (x["holm_m"] == 12).all() and (x["label"] == NONE).all()
    assert np.allclose(x["delta_hat_histgb"], f1.loc[list(zip(x["grid_a"], x["grid_b"])), "delta_hat"], **ami.TOL)
    assert np.allclose(x["delta_min_thr_histgb"], f1.loc[list(zip(x["grid_a"], x["grid_b"])), "delta_min_thr"])
    assert s["ket_luan"] == "khong_phu_thuoc_mo_hinh" and s["n_cap"] == 12
    pv = _prov(out / "dot7_salinity_tuong_tac_mo_hinh__rf.csv")
    assert pv["dau_vao"]["hist_gb"]["gate_file"] is None and pv["dau_vao"]["mo_hinh"]["gate_hop_le"]["7"] is True

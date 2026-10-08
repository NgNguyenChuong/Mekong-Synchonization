"""Tong hop Dot 7 cho rf/mlp (do nhay theo mo hinh): --model, hai manifest dong bang, cong rieng tung mo hinh,
--allowed-tags, analyze_model_sensitivity. Du lieu gia dung lai test_analyze_dot7_tonghop; sai so rf = HistGB (cung
hat giong) nen moi chi so rf phai trung HistGB, chi khac ten.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

import analyze_block_size as abs_  # noqa: E402
import analyze_cv_family as acf  # noqa: E402
import analyze_final as afin  # noqa: E402
import analyze_learnability as alr  # noqa: E402
import analyze_model_sensitivity as ams  # noqa: E402
import test_analyze_dot7_tonghop as base  # noqa: E402
from run_experiments import run_name  # noqa: E402
from training.dot7_rules import (GATE_COLS, MULTI_TAG_REASON, SENS_SUFFIX, all_scoped_names,  # noqa: E402
                                 frozen_paths, guard_frozen, merge_gate, model_name)

E6E = "nckh-test-e6e"
RF = "random_forest"
T = "ndwi"
DO_MAN = "dot_dong_bang_do_man_manifest.csv"
HISTGB = "dot7_dong_bang_histgb_manifest.csv"


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _prov(p):
    return json.loads(open(f"{p}.provenance.json", encoding="utf-8").read())


def _run(mod, args):
    mod.main(mod.build_parser().parse_args(args))


def build_rf(exp, target=T, run_model=RF, tag=E6E):
    """Luot rf (cv 3 cach chia, final s42, k100 P1/P2) cung sai so voi HistGB, tag rieng."""
    meta = {"git_tag": tag}
    for g in base.AREA:
        for s in (42, 43, 44):
            d = base._write_run(exp, run_name("cv1", g, run_model, s, "", None, target), "cv", target, meta)
            base._oof(target, g, run_model, s).to_csv(d / "oof_points.csv", index=False)
        f = base._final(target, g, run_model)
        d = base._write_run(exp, run_name("cv1", g, run_model, 42, "", None, target), "final", target,
                            {**meta, "empty_test_groups_declared": []}, base._final_cfg(f, ()))
        f.to_csv(d / "final_points.csv", index=False)
        for s in (42, 43):
            d = base._write_run(exp, run_name("k100", g, run_model, s, "", None, target), "cv", target, meta)
            base._oof(target, g, run_model, s + 100).to_csv(d / "oof_points.csv", index=False)


def histgb_outputs(out):
    """Moi file HistGB cua T ma 4 script co the ghi (ke ca __holm_muc_kiem_dinh va provenance)."""
    names = [acf.out_name(T, "cv1"), *abs_.out_names(T), *afin.out_names(T, "cv1"), alr.out_name(T)]
    paths = [p for n in names for p in all_scoped_names(os.path.join(out, n))] + [os.path.join(out, acf.GATE_FILE)]
    return paths + [f"{p}.provenance.json" for p in paths]


@pytest.fixture(scope="module")
def rf_world(tmp_path_factory):
    """HistGB chay truoc vao <repo>/KE_HOACH/ket-qua -> dong bang bang manifest HistGB -> rf chay vao CUNG thu muc."""
    mp = pytest.MonkeyPatch()
    mp.setattr(acf, "point_units", base._fake_point_units)
    root = tmp_path_factory.mktemp("rfmlp")
    exp = root / "exp"
    base.build_exp(exp, targets=(T,))
    build_rf(exp)
    at = pd.DataFrame([{"file": f"data/grids/{g}.geojson", "tier_h3_res": t, "mean_area_km2": a}
                       for g, (t, a) in base.AREA.items()])
    at.to_csv(root / "dien_tich.csv", index=False)
    out = root / "repo" / "KE_HOACH" / "ket-qua"
    out.mkdir(parents=True)
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/dot5_cv1_kiem_dinh_khoi.csv"]}).to_csv(out / DO_MAN, index=False)
    w = {"root": root, "exp": exp, "at": root / "dien_tich.csv", "out": out,
         "mans": [str(out / DO_MAN), str(out / HISTGB)]}

    def args(extra=(), mans=None, area=True, out_dir=out):
        return ["--exp-root", str(exp), "--out-dir", str(out_dir), "--frozen-manifest", *(mans or w["mans"]),
                *(["--area-table", str(w["at"])] if area else []), *extra]
    w["args"] = args

    hist_args = dict(mans=[str(out / DO_MAN)])
    _run(alr, args(["--targets", T, "--no-table"], **hist_args))
    _run(acf, args(["--target", T, "--no-table"], **hist_args))
    _run(abs_, args(["--target", T, "--no-table"], area=False, **hist_args))
    _run(afin, args(["--target", T, "--no-table"], **hist_args))
    frozen = sorted(p for p in histgb_outputs(str(out)) if os.path.isfile(p))
    rel = [os.path.relpath(p, out.parent.parent).replace("\\", "/") for p in frozen]
    pd.DataFrame({"file": rel, "sha256": [_sha(p) for p in frozen]}).to_csv(out / HISTGB, index=False)
    w["frozen"] = {p: _sha(p) for p in frozen}

    rf = ["--model", "rf", "--no-table"]
    _run(alr, args(["--targets", T, *rf, "--allowed-tags", base.TAG, E6E]))
    _run(acf, args(["--target", T, *rf]))
    _run(abs_, args(["--target", T, *rf], area=False))
    _run(afin, args(["--target", T, *rf]))
    _run(ams, ["--model", "rf", "--targets", T, "--out-dir", str(out), "--frozen-manifest", *w["mans"], "--no-table"])
    yield w
    mp.undo()


# ---------------------------------------------------------
# (i) file ra rf khong trung manifest HistGB; guard_frozen 2 manifest
# ---------------------------------------------------------
RF_FILES = ["dot7_ndwi_hoc_duoc__rf.csv", "dot7_cong_kiem_dinh__rf.csv", "dot7_ndwi_cv1_kiem_dinh_khoi__rf.csv",
            f"dot7_ndwi_cv1_kiem_dinh_khoi{SENS_SUFFIX}__rf.csv", "dot7_ndwi_khoi100_mae__rf.csv",
            "dot7_ndwi_khoi100_cap_min__rf.csv", "dot7_ndwi_cv1_final_cap__rf.csv", "dot7_ndwi_cv1_final_nhom__rf.csv",
            "dot7_ndwi_do_nhay_mo_hinh__rf.csv", "dot7_do_nhay_mo_hinh_tong_ket__rf.csv"]


def test_rf_khong_cham_file_histgb(rf_world):
    out = rf_world["out"]
    for p, h in rf_world["frozen"].items():
        assert _sha(p) == h, p
    frozen = frozen_paths(rf_world["mans"])
    assert len(rf_world["frozen"]) >= 10 and all(os.path.normcase(p) in frozen for p in rf_world["frozen"])
    for n in RF_FILES:
        p = out / n
        assert p.is_file() and (out / f"{n}.provenance.json").is_file(), n
        assert os.path.normcase(str(p)) not in frozen, n
    new = {f.name for f in out.iterdir()} - {os.path.basename(p) for p in rf_world["frozen"]} - {DO_MAN, HISTGB}
    assert all("__rf" in n for n in new), sorted(new)


def test_ten_file_theo_mo_hinh():
    assert model_name("dot7_ndwi_khoi100_mae.csv", "hist_gb") == "dot7_ndwi_khoi100_mae.csv"
    assert model_name("dot7_ndwi_khoi100_mae.csv", "mlp") == "dot7_ndwi_khoi100_mae__mlp.csv"
    assert all_scoped_names("dot5_cv1_kiem_dinh_khoi.csv", "rf") == [
        "dot5_cv1_kiem_dinh_khoi__rf.csv", f"dot5_cv1_kiem_dinh_khoi{SENS_SUFFIX}__rf.csv"]
    assert alr.out_name("ndwi", "rf") == "dot7_ndwi_hoc_duoc__rf.csv" and acf.gate_name("mlp") == \
        "dot7_cong_kiem_dinh__mlp.csv"
    with pytest.raises(ValueError):
        model_name("x.csv", "random_forest")


def test_guard_frozen_hai_manifest(rf_world, tmp_path, capsys):
    out = str(rf_world["out"])
    histgb_file = os.path.join(out, acf.out_name(T, "cv1"))
    salinity_file = os.path.join(out, "dot5_cv1_kiem_dinh_khoi.csv")
    for p in (histgb_file, salinity_file):  # moi manifest chan dung file cua no
        with pytest.raises(SystemExit) as e:
            guard_frozen([os.path.join(out, "khac.csv"), p], rf_world["mans"])
        assert e.value.code == 2
    guard_frozen([os.path.join(out, RF_FILES[2])], rf_world["mans"])
    with pytest.raises(SystemExit) as e:  # thieu manifest thu hai -> ma 2
        guard_frozen([os.path.join(out, RF_FILES[2])], [rf_world["mans"][0], str(tmp_path / HISTGB)])
    assert e.value.code == 2
    capsys.readouterr()
    for mod, extra in ((acf, ["--target", T]), (abs_, ["--target", T]), (afin, ["--target", T]),
                       (alr, ["--targets", T])):  # HistGB chay lai mac dinh 2 manifest -> ma 2, khong ghi gi
        with pytest.raises(SystemExit) as e:
            _run(mod, rf_world["args"](extra, area=mod is not abs_))
        assert e.value.code == 2, mod.__name__
        assert "tu choi ghi de file dong bang" in capsys.readouterr().out, mod.__name__
    for p, h in rf_world["frozen"].items():
        assert _sha(p) == h, p


def test_mac_dinh_hai_manifest():
    want = [os.path.join(acf.RESULTS, DO_MAN), os.path.join(acf.RESULTS, HISTGB)]
    assert acf.FROZEN_MANIFESTS == want
    for mod in (acf, abs_, afin, alr, ams):
        args = ["--model", "rf"] if mod is ams else []
        assert mod.build_parser().parse_args(args).frozen_manifest == want, mod.__name__


# ---------------------------------------------------------
# (ii) cong rieng tung mo hinh
# ---------------------------------------------------------
def test_learnability_rf_cong_rieng(rf_world):
    out = rf_world["out"]
    g = pd.read_csv(out / "dot7_cong_kiem_dinh__rf.csv")
    assert set(GATE_COLS) | {"model"} <= set(g.columns) and (g["model"] == "rf").all()
    h = pd.read_csv(out / acf.GATE_FILE)
    assert "model" not in h.columns
    key = ["bien", "muc", "n_luoi", "n_hoc_duoc", "hop_le", "ly_do", "holm_m", "pham_vi_cong"]
    assert g[key].equals(h[key])  # sai so rf = HistGB -> cong trung
    assert (g["git_tag"] == f"{base.TAG};{E6E}").all()
    hoc = pd.read_csv(out / "dot7_ndwi_hoc_duoc__rf.csv")
    assert (hoc["model_a"] == RF).all() and (hoc["model_b"] == "season_mean").all() and len(hoc) == 13
    pv = _prov(out / "dot7_cong_kiem_dinh__rf.csv")
    assert pv["model"] == "rf" and pv["run_model"] == RF and pv["theo_bien"][T]["git_tags_seen"] == [base.TAG, E6E]
    ph = _prov(out / "dot7_ndwi_hoc_duoc__rf.csv")
    assert ph["git_tags_allowed"] == [base.TAG, E6E] and ph["ly_do_nhieu_tag"] == MULTI_TAG_REASON


def test_cv_family_rf_doc_cong_rieng_va_trung_histgb(rf_world):
    out = rf_world["out"]
    r = pd.read_csv(out / "dot7_ndwi_cv1_kiem_dinh_khoi__rf.csv")
    h = pd.read_csv(out / acf.out_name(T, "cv1"))
    assert set(r["family"]) == {"do_nhay_rf", "do_nhay_chi_s42", "do_nhay_chi_s43", "do_nhay_chi_s44"}
    assert (r["vai_tro"] == "do_nhay_mo_hinh").all() and "vai_tro" not in h.columns
    assert (r["model_a"] == RF).all() and (r["git_tag"] == E6E).all() and (r["n_luot"] == 39).all()
    assert r["cong_file"].str.endswith("dot7_cong_kiem_dinh__rf.csv").all()
    rm = r[r["family"] == "do_nhay_rf"].reset_index(drop=True)
    hm = h[h["family"] == "F1_chinh"].reset_index(drop=True)
    for c in ("grid_a", "grid_b", "muc", "kiem_dinh", "holm_m", "delta_hat", "p_value", "p_holm", "label",
              "cong_hoc_duoc"):
        assert rm[c].astype(str).equals(hm[c].astype(str)), c
    for s in (42, 43, 44):
        f = f"do_nhay_chi_s{s}"
        assert r.loc[r["family"] == f, "delta_hat"].round(12).tolist() == \
            h.loc[h["family"] == f, "delta_hat"].round(12).tolist(), f
    pv = _prov(out / "dot7_ndwi_cv1_kiem_dinh_khoi__rf.csv")
    assert pv["vai_tro"] == "do_nhay_mo_hinh" and pv["gate_file"].endswith("dot7_cong_kiem_dinh__rf.csv")
    assert "git_tags_allowed" not in pv


def test_cv_family_rf_tu_choi_cong_histgb(rf_world, tmp_path):
    with pytest.raises(SystemExit) as e:
        _run(acf, rf_world["args"](["--target", T, "--model", "rf", "--gate", str(rf_world["out"] / acf.GATE_FILE)],
                                   out_dir=tmp_path))
    assert "can 'rf'" in str(e.value)
    with pytest.raises(SystemExit) as e:  # thieu cong rf (co cong HistGB cung thu muc) -> LOI
        (tmp_path / acf.GATE_FILE).write_bytes((rf_world["out"] / acf.GATE_FILE).read_bytes())
        _run(acf, rf_world["args"](["--target", T, "--model", "rf"], out_dir=tmp_path))
    assert "dot7_cong_kiem_dinh__rf.csv" in str(e.value)
    assert not any("kiem_dinh_khoi" in p.name for p in tmp_path.iterdir())


def test_cong_do_man_rf_bat_buoc(rf_world, tmp_path):
    with pytest.raises(SystemExit) as e:
        _run(acf, rf_world["args"](["--target", "salinity", "--model", "rf"], out_dir=tmp_path))
    assert "cong" in str(e.value)


def test_merge_gate_khong_tron_mo_hinh(tmp_path):
    p = tmp_path / "g.csv"
    base._gate(p, [("ndwi", 5, True)])
    new = pd.DataFrame({"bien": ["rain_chirps"], "model": ["rf"], "muc": [5], "hop_le": [True]})
    with pytest.raises(ValueError):
        merge_gate(p, new)
    pd.DataFrame({"bien": ["ndwi"], "model": ["rf"], "muc": [5], "hop_le": [True]}).to_csv(p, index=False)
    assert set(merge_gate(p, new)["bien"]) == {"ndwi", "rain_chirps"}


# ---------------------------------------------------------
# block_size / final rf
# ---------------------------------------------------------
def test_block_size_rf_delta_min_cung_mo_hinh(rf_world, tmp_path):
    out = rf_world["out"]
    cap = pd.read_csv(out / "dot7_ndwi_khoi100_cap_min__rf.csv")
    r = pd.read_csv(out / "dot7_ndwi_cv1_kiem_dinh_khoi__rf.csv")
    r = r[r["family"] == "do_nhay_rf"]
    assert np.allclose(cap["delta_min"], r.loc[r["muc"] == 7, "delta_min_thr"].iloc[0])
    h_cap = pd.read_csv(out / "dot7_ndwi_khoi100_cap_min.csv")
    assert np.allclose(cap["delta_hat_cv1"], h_cap["delta_hat_cv1"]) and cap["kiem_dinh"].equals(h_cap["kiem_dinh"])
    assert _prov(out / "dot7_ndwi_khoi100_cap_min__rf.csv")["cv_ref"].endswith("dot7_ndwi_cv1_kiem_dinh_khoi__rf.csv")
    mae = pd.read_csv(out / "dot7_ndwi_khoi100_mae__rf.csv")
    assert len(mae) == 13 and (mae["git_tag"] == E6E).all()
    for n in (acf.GATE_FILE, acf.gate_name("rf"), acf.out_name(T, "cv1")):  # chi co file HistGB -> LOI
        (tmp_path / n).write_bytes((out / n).read_bytes())
    with pytest.raises(SystemExit) as e:
        _run(abs_, rf_world["args"](["--target", T, "--model", "rf"], area=False, out_dir=tmp_path))
    assert "kiem_dinh_khoi__rf.csv" in str(e.value)


def test_final_rf_chi_mo_hinh_do(rf_world, tmp_path):
    out = rf_world["out"]
    c = pd.read_csv(out / "dot7_ndwi_cv1_final_cap__rf.csv")
    assert c["doi_chieu_cv"].all() and (c["family"] == "do_nhay_rf").all() and (c["model_a"] == RF).all()
    h = pd.read_csv(out / "dot7_ndwi_cv1_final_cap.csv")
    for col in ("delta_hat", "delta_holdout", "label", "kiem_dinh"):
        assert c[col].astype(str).equals(h[col].astype(str)), col
    n = pd.read_csv(out / "dot7_ndwi_cv1_final_nhom__rf.csv")
    assert set(n["model"]) == {RF} and len(n) == 13 * 3
    with pytest.raises(SystemExit) as e:
        _run(afin, rf_world["args"](["--target", T, "--model", "rf", "--models", "hist_gb"], out_dir=tmp_path))
    assert "--models" in str(e.value)


# ---------------------------------------------------------
# (iii) check_consistent nhieu tag
# ---------------------------------------------------------
def _infos(tags):
    return [{"run": f"r{i}", "git_tag": t, "points_ref_sha256": "x"} for i, t in enumerate(tags)]


def test_check_consistent_nhieu_tag():
    two = _infos(["a", "b", "a"])
    with pytest.raises(SystemExit) as e:
        acf.check_consistent(two)
    assert "git_tag khac nhau" in str(e.value)
    same = acf.check_consistent(two, ["a", "b", "c"])
    assert same["git_tag"] == "a;b" and same["git_tags_seen"] == ["a", "b"] and same["points_ref_sha256"] == "x"
    assert acf.tag_info(same, ["c", "b", "a"])["git_tags_allowed"] == ["a", "b", "c"]
    assert acf.tag_info(same, None) == {}
    with pytest.raises(SystemExit) as e:
        acf.check_consistent(two, ["a"])
    assert "['b'] ngoai --allowed-tags" in str(e.value)
    assert acf.check_consistent(_infos(["a", "a"]))["git_tag"] == "a"
    ref = _infos(["a", "b"])
    ref[1]["points_ref_sha256"] = "y"
    with pytest.raises(SystemExit) as e:  # dap an van phai duy nhat
        acf.check_consistent(ref, ["a", "b"])
    assert "points_ref_sha256" in str(e.value)


@pytest.mark.parametrize("tags,msg", [(None, "git_tag khac nhau"), ([E6E, "nckh-khac"], "ngoai --allowed-tags")])
def test_learnability_rf_nhieu_tag_loi(rf_world, tmp_path, tags, msg):
    extra = ["--targets", T, "--model", "rf", "--no-table"] + (["--allowed-tags", *tags] if tags else [])
    with pytest.raises(SystemExit) as e:
        _run(alr, rf_world["args"](extra, out_dir=tmp_path))
    assert msg in str(e.value)
    assert not (tmp_path / acf.gate_name("rf")).exists()


# ---------------------------------------------------------
# (v) --model hist_gb = hanh vi cu
# ---------------------------------------------------------
def test_hist_gb_tuong_minh_trung_mac_dinh(rf_world, tmp_path):
    g = ["--gate", str(rf_world["out"] / acf.GATE_FILE)]
    names = {}
    for k, extra in (("mac_dinh", []), ("tuong_minh", ["--model", "hist_gb"])):
        d = tmp_path / k
        d.mkdir()
        _run(acf, rf_world["args"](["--target", T, "--no-table", *g, *extra], out_dir=d))
        _run(abs_, rf_world["args"](["--target", T, "--no-table", *g, *extra], area=False, out_dir=d))
        names[k] = sorted(p.name for p in d.iterdir())
    assert names["mac_dinh"] == names["tuong_minh"]
    assert set(names["mac_dinh"]) == {f"{n}{x}" for n in (acf.out_name(T, "cv1"), f"dot7_ndwi_cv1_kiem_dinh_khoi"
                                                         f"{SENS_SUFFIX}.csv", *abs_.out_names(T))
                                      for x in ("", ".provenance.json")}
    for n in names["mac_dinh"]:
        if n.endswith(".csv"):
            assert (tmp_path / "mac_dinh" / n).read_bytes() == (tmp_path / "tuong_minh" / n).read_bytes(), n
            assert _sha(tmp_path / "mac_dinh" / n) == rf_world["frozen"][str(rf_world["out"] / n)], n
            pv = _prov(tmp_path / "mac_dinh" / n)
            assert "model" not in pv and "git_tags_allowed" not in pv, n


# ---------------------------------------------------------
# (iv) analyze_model_sensitivity
# ---------------------------------------------------------
def test_do_nhay_mo_hinh_ndwi_tich_hop(rf_world):
    out = rf_world["out"]
    x = pd.read_csv(out / "dot7_ndwi_do_nhay_mo_hinh__rf.csv")
    s = pd.read_csv(out / "dot7_do_nhay_mo_hinh_tong_ket__rf.csv").iloc[0]
    assert len(x) == 12 and x["cung_dau"].all() and not x["co_y_nghia_them"].any()
    assert sorted(x.loc[x["qua_cong_ca_hai"], "muc"].unique()) == [5, 6]       # BAD o muc 7 truot cong ca hai
    assert s["n_cap_kiem_dinh"] == 12 and s["n_cap"] == 6 and s["ti_le_cung_dau"] == 1.0 and bool(s["ket_luan_giu"])
    pv = _prov(out / "dot7_ndwi_do_nhay_mo_hinh__rf.csv")
    assert pv["dau_vao"]["mo_hinh"]["cv_file"].endswith("dot7_ndwi_cv1_kiem_dinh_khoi__rf.csv")
    assert pv["dau_vao"]["hist_gb"]["cv_sha256"] == _sha(out / acf.out_name(T, "cv1"))


def _cv(path, family, rows, target="ndwi", gate=None, old=False):
    """File kiem_dinh_khoi gia: rows = (grid_a, grid_b, muc, delta_hat, label)."""
    d = pd.DataFrame(rows, columns=["grid_a", "grid_b", "muc", "delta_hat", "label"])
    d["family"], d["mode"], d["delta_ref_level"], d["p_holm"] = family, "cv", d["muc"], 0.01
    if old:  # file do man cu: khong co target / muc / muc_kiem_dinh / cong_hoc_duoc
        d = d.drop(columns="muc")
    else:
        d["target"], d["git_tag"] = target, "tag"
        d["muc_kiem_dinh"] = d["muc"].isin(base.TESTED_TIERS[target])
        d["cong_hoc_duoc"] = [gate[m] for m in d["muc"]] if gate else np.nan
    d.to_csv(path, index=False)


def _gate_m(path, target, hop, model=None):
    g = pd.DataFrame([(target, m, v) for m, v in hop.items()], columns=["bien", "muc", "hop_le"])
    if model:
        g.insert(1, "model", model)
    g.to_csv(path, index=False)


PAIRS5 = [("a1", "b1", 5), ("a2", "b2", 5), ("a3", "b3", 6), ("a4", "b4", 6), ("a5", "b5", 6)]
ALL_T = {5: True, 6: True, 7: True}


def _sens(tmp_path, h_rows, m_rows, h_gate=ALL_T, m_gate=ALL_T, target="ndwi", min_frac=None, old=False):
    d = tmp_path / "kq"
    d.mkdir(exist_ok=True)
    if h_gate is not None:
        _gate_m(d / acf.GATE_FILE, target, h_gate)
    _gate_m(d / acf.gate_name("rf"), target, m_gate, "rf")
    _cv(d / acf.out_name(target, "cv1"), "F1_chinh", h_rows, target, h_gate, old)
    _cv(d / model_name(acf.out_name(target, "cv1"), "rf"), "do_nhay_rf", m_rows, target, m_gate)
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/khong_co.csv"]}).to_csv(tmp_path / "man.csv", index=False)
    args = ["--model", "rf", "--targets", target, "--out-dir", str(d), "--frozen-manifest", str(tmp_path / "man.csv"),
            "--no-table", *(["--min-frac", str(min_frac)] if min_frac else [])]
    _run(ams, args)
    return (pd.read_csv(d / f"dot7_{target}_do_nhay_mo_hinh__rf.csv"),
            pd.read_csv(d / "dot7_do_nhay_mo_hinh_tong_ket__rf.csv").iloc[0])


def _rows(signs, labels=None, pairs=PAIRS5):
    labels = labels or ["chua_phan_dinh"] * len(pairs)
    return [(a, b, m, 0.1 * s, lab) for (a, b, m), s, lab in zip(pairs, signs, labels)]


@pytest.mark.parametrize("m_signs,m_labels,h_labels,want,frac,extra", [
    ([1, -1, 1, 1, -1], None, None, True, 1.0, 0),                                         # cung dau, khong them
    ([1, -1, 1, 1, -1], ["cho_giu_rieng"] + ["chua_phan_dinh"] * 4, None, False, 1.0, 1),  # co y nghia them
    ([1, -1, 1, 1, 1], None, None, True, 0.8, 0),                                          # bien 0,8 -> giu
    ([-1, -1, 1, 1, 1], None, None, False, 0.6, 0),                                        # 0,6 < 0,8
    ([1, -1, 1, 1, -1], ["cho_giu_rieng"] * 5, ["cho_giu_rieng"] * 5, True, 1.0, 0),       # ca hai co y nghia
    ([1, -1, 1, 1, -1], None, ["cho_giu_rieng"] * 5, True, 1.0, 0),                        # mat y nghia: chi mo ta
])
def test_ket_luan_giu(tmp_path, m_signs, m_labels, h_labels, want, frac, extra):
    x, s = _sens(tmp_path, _rows([1, -1, 1, 1, -1], h_labels), _rows(m_signs, m_labels))
    assert bool(s["ket_luan_giu"]) is want and np.isclose(s["ti_le_cung_dau"], frac)
    assert s["n_co_y_nghia_them"] == extra and s["n_cap"] == 5
    assert x["co_y_nghia_them"].sum() == extra
    if h_labels and not m_labels:
        assert s["n_mat_y_nghia"] == 5


def test_min_frac_tham_so(tmp_path):
    _, s = _sens(tmp_path, _rows([1] * 5), _rows([1, 1, 1, 1, -1]), min_frac=0.9)
    assert not bool(s["ket_luan_giu"]) and s["min_frac"] == 0.9


def test_cap_truot_cong_mot_ben_khong_tinh(tmp_path):
    pairs = PAIRS5 + [("c1", "d1", 7), ("c2", "d2", 7)]
    h = _rows([1] * 7, pairs=pairs)
    m = _rows([1] * 5 + [-1, -1], ["chua_phan_dinh"] * 5 + ["cho_giu_rieng"] * 2, pairs=pairs)
    x, s = _sens(tmp_path, h, m, h_gate={5: True, 6: True, 7: False})
    assert s["n_cap_kiem_dinh"] == 7 and s["n_cap"] == 5 and bool(s["ket_luan_giu"])
    m7 = x[x["muc"] == 7]
    assert not m7["qua_cong_ca_hai"].any() and m7["qua_cong_mo_hinh"].all() and m7["co_y_nghia_them"].all()


def test_do_man_file_cu_khong_cong_histgb(tmp_path):
    x, s = _sens(tmp_path, _rows([1] * 5), _rows([1] * 5), h_gate=None, target="salinity", old=True)
    assert x["qua_cong_hist_gb"].all() and s["n_cap"] == 5 and bool(s["ket_luan_giu"])
    pv = _prov(tmp_path / "kq" / "dot7_salinity_do_nhay_mo_hinh__rf.csv")
    assert pv["dau_vao"]["hist_gb"]["gate_file"] is None and pv["dau_vao"]["mo_hinh"]["gate_hop_le"]["7"] is True


def test_khong_co_cap_kiem_dinh(tmp_path):
    rows = _rows([1] * 5)
    x, s = _sens(tmp_path, rows, rows, h_gate={5: False, 6: False, 7: False},
                 m_gate={5: False, 6: False, 7: False}, target="t2m_era5")
    assert len(x) == 0 and s["n_cap"] == 0 and pd.isna(s["ket_luan_giu"])
    assert s["trang_thai"] == "khong_co_cap_qua_cong_ca_hai"


def test_tap_cap_lech_loi(tmp_path):
    with pytest.raises(SystemExit) as e:
        _sens(tmp_path, _rows([1] * 5), _rows([1] * 4))
    assert "khac nhau" in str(e.value)


def test_cong_hoc_duoc_khong_khop_cong_loi(tmp_path):
    d = tmp_path / "kq"
    d.mkdir()
    _cv(d / acf.out_name("ndwi", "cv1"), "F1_chinh", _rows([1] * 5), gate={5: True, 6: False, 7: True})
    with pytest.raises(SystemExit) as e:  # file CV tinh voi cong khac cong dang doc
        _gate_m(d / acf.GATE_FILE, "ndwi", ALL_T)
        _gate_m(d / acf.gate_name("rf"), "ndwi", ALL_T, "rf")
        _cv(d / model_name(acf.out_name("ndwi", "cv1"), "rf"), "do_nhay_rf", _rows([1] * 5), gate=ALL_T)
        pd.DataFrame({"file": ["x.csv"]}).to_csv(tmp_path / "man.csv", index=False)
        _run(ams, ["--model", "rf", "--targets", "ndwi", "--out-dir", str(d), "--frozen-manifest",
                   str(tmp_path / "man.csv"), "--no-table"])
    assert "cong_hoc_duoc" in str(e.value)

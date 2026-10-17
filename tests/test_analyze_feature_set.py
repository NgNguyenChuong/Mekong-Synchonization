"""--feature-set khong_diem (DBSCL bo (b)): analyze_learnability / cv_family / final / ledger doc luot __fs-khong_diem.

Du lieu gia: HistGB chi co luot __fs-khong_diem (luot (a) HistGB la bay: returncode 1); season_mean la luot (a)
(khong dung dac trung). Kiem: ten file hau to, provenance, khong ho phu linear/idw, tu choi luot sai bo.
"""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

import analyze_cv_family as acf  # noqa: E402
import analyze_final as afin  # noqa: E402
import analyze_learnability as alr  # noqa: E402
import build_comparison_ledger as bcl  # noqa: E402
import test_analyze_dot7_tonghop as base  # noqa: E402
from run_experiments import run_name  # noqa: E402

T, FS, SHA = "ndwi", "khong_diem", "feat_sha"


def _build(exp):
    for g in base.AREA:
        for s in (42, 43, 44):
            name = run_name("cv1", g, "hist_gb", s, "", FS, T)
            d = base._write_run(exp, name, "cv", T, {"feature_set": FS, "features_sha256": SHA})
            base._oof(T, g, "hist_gb", s).to_csv(d / "oof_points.csv", index=False)
            bad = base._write_run(exp, run_name("cv1", g, "hist_gb", s, "", None, T), "cv", T, {"returncode": 1})
            base._oof(T, g, "hist_gb", s).to_csv(bad / "oof_points.csv", index=False)
            d = base._write_run(exp, run_name("cv1", g, "season_mean", s, "", None, T), "cv", T)
            base._oof(T, g, "season_mean", s).to_csv(d / "oof_points.csv", index=False)
        for m, fs in (("hist_gb", FS), ("season_mean", None)):
            f = base._final(T, g, m)
            d = base._write_run(exp, run_name("cv1", g, m, 42, "", fs, T), "final", T,
                                {"empty_test_groups_declared": [], **({"feature_set": fs, "features_sha256": SHA} if fs else {})},
                                base._final_cfg(f, ()))
            f.to_csv(d / "final_points.csv", index=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(acf, "point_units", base._fake_point_units)
    b = tmp_path_factory.mktemp("fs")
    _build(b / "exp")
    at = pd.DataFrame([{"file": f"data/grids/{g}.geojson", "tier_h3_res": t, "mean_area_km2": a,
                        "framework": bcl.grid_area_key(g)[0], "resolution": bcl.grid_area_key(g)[1]}
                       for g, (t, a) in base.AREA.items()])
    at.to_csv(b / "dien_tich.csv", index=False)
    man = b / "repo" / "KE_HOACH" / "ket-qua" / "man.csv"
    man.parent.mkdir(parents=True)
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/x.csv"]}).to_csv(man, index=False)
    out = b / "out"
    args = ["--exp-root", str(b / "exp"), "--area-table", str(b / "dien_tich.csv"), "--out-dir", str(out),
            "--frozen-manifest", str(man), "--folds-csv", "x.csv", "--feature-set", FS, "--no-table",
            "--allowed-tags", base.TAG]
    for mod, extra in ((alr, ["--targets", T]), (acf, ["--target", T]), (afin, ["--target", T])):
        mod.main(mod.build_parser().parse_args(args + extra))
    yield {"b": b, "out": out, "exp": b / "exp", "args": args}
    mp.undo()


def _prov(p):
    return json.loads(open(f"{p}.provenance.json", encoding="utf-8").read())


def test_ten_file_hau_to_va_khong_ghi_file_khong_hau_to(world):
    names = sorted(f for f in os.listdir(world["out"]) if not f.endswith(".provenance.json"))
    assert names == sorted([f"dot7_{T}_cv1_final_cap__fs-{FS}.csv", f"dot7_{T}_cv1_final_nhom__fs-{FS}.csv",
                            f"dot7_{T}_cv1_kiem_dinh_khoi__fs-{FS}.csv", f"dot7_{T}_hoc_duoc__fs-{FS}.csv",
                            f"dot7_{T}_cv1_kiem_dinh_khoi__fs-{FS}__holm_muc_kiem_dinh.csv",  # muc 7 truot cong
                            f"dot7_cong_kiem_dinh__fs-{FS}.csv"])


def test_provenance_ghi_bo_dac_trung(world):
    for n in (f"dot7_{T}_cv1_final_cap__fs-{FS}.csv", f"dot7_{T}_cv1_kiem_dinh_khoi__fs-{FS}.csv",
              f"dot7_{T}_hoc_duoc__fs-{FS}.csv"):
        pv = _prov(world["out"] / n)
        assert pv["feature_set"] == FS and (pv.get("features_sha256") or pv["theo_bien"]) is not None
    assert _prov(world["out"] / f"dot7_{T}_cv1_kiem_dinh_khoi__fs-{FS}.csv")["features_sha256"] == SHA
    assert _prov(world["out"] / f"dot7_cong_kiem_dinh__fs-{FS}.csv")["feature_set"] == FS


def test_cv_family_khong_ho_phu_linear_idw_va_doc_dung_luot(world):
    r = pd.read_csv(world["out"] / f"dot7_{T}_cv1_kiem_dinh_khoi__fs-{FS}.csv")
    assert not set(r["family"]) & {"phu_linear", "phu_idw"} and "F1_chinh" in set(r["family"])
    f1 = r[r["family"] == "F1_chinh"]
    assert len(f1) == 12 and (f1["n_common_point_seasons"] == len(base.CV_UNITS) * base.NPT * 2).all()
    cap = pd.read_csv(world["out"] / f"dot7_{T}_cv1_final_cap__fs-{FS}.csv")
    assert len(cap) == 12 and cap["doi_chieu_cv"].all()


def test_final_nhom_chi_hist_gb_va_season_mean(world):
    n = pd.read_csv(world["out"] / f"dot7_{T}_cv1_final_nhom__fs-{FS}.csv")
    assert set(n["model"]) == {"hist_gb", "season_mean"}


def test_final_tu_choi_linear_idw_khi_co_bo_dac_trung(world, tmp_path):
    a = afin.build_parser().parse_args(world["args"][:-2] + ["--target", T, "--models", "hist_gb", "linear"])
    a.out_dir = str(tmp_path)
    with pytest.raises(SystemExit, match="khong co luot cua mo hinh"):
        afin.desc_models(acf.resolve_region(a))


def test_check_run_lech_bo_dac_trung_loi(tmp_path):
    d = base._write_run(tmp_path, "r", "cv", T, {"feature_set": FS})
    acf.check_run(str(d), "r", T, "cv", "dbscl", FS)
    acf.check_run(str(d), "r", T, "cv")  # mac dinh: khong kiem (analyze_hybrid tu kiem)
    with pytest.raises(SystemExit, match="feature_set"):
        acf.check_run(str(d), "r", T, "cv", "dbscl", None)
    d2 = base._write_run(tmp_path, "r2", "cv", T)
    with pytest.raises(SystemExit, match="feature_set"):
        acf.check_run(str(d2), "r2", T, "cv", "dbscl", FS)


def test_features_sha_khac_nhau_loi():
    infos = [{"meta": {"feature_set": FS, "features_sha256": "a"}}, {"meta": {"feature_set": FS, "features_sha256": "b"}}]
    with pytest.raises(SystemExit, match="features_sha256"):
        acf.feature_info(infos, FS)
    assert acf.feature_info([{"meta": {}}], None) == {}


def test_feature_set_chi_cho_dbscl_va_mac_dinh_thu_muc():
    a = acf.build_parser().parse_args(["--region", "nile", "--pairs-mode", "ratio", "--feature-set", FS])
    with pytest.raises(SystemExit, match="chi cho --region dbscl"):
        acf.resolve_region(a)
    a = acf.resolve_region(acf.build_parser().parse_args(["--feature-set", FS]))
    assert os.path.normpath(a.out_dir) == os.path.join(acf.RESULTS, "dbscl_fs_khong_diem")
    assert acf.out_name("salinity", "cv1", FS) == f"dot5_cv1_kiem_dinh_khoi__fs-{FS}.csv"
    assert acf.out_name("salinity", "cv1") == "dot5_cv1_kiem_dinh_khoi.csv"
    assert acf.gate_name("hist_gb", FS) == f"dot7_cong_kiem_dinh__fs-{FS}.csv"


def test_ledger_bo_dac_trung(world, tmp_path, monkeypatch):
    res = world["out"]
    monkeypatch.setattr(bcl, "TARGETS", [T])
    monkeypatch.setattr(sys, "argv", ["x", "--res-dir", str(res), "--feature-set", FS, "--area-table",
                                      str(world["b"] / "dien_tich.csv")])
    bcl.main()
    led = pd.read_csv(res / f"comparison_ledger__fs-{FS}.csv")
    assert len(led) == 12 and "ratio_gt_1_2" not in led.columns and not (res / "comparison_ledger.csv").exists()
    pv = _prov(res / f"comparison_ledger__fs-{FS}.csv")
    assert pv["feature_set"] == FS and pv["region"] == "dbscl" and len(pv["inputs"]) == 3
    # provenance khong ghi feature_set (luot (a)) -> tu choi khi doc voi --feature-set
    p = res / f"dot7_{T}_cv1_kiem_dinh_khoi__fs-{FS}.csv.provenance.json"
    saved = p.read_text(encoding="utf-8")
    p.write_text(json.dumps({k: v for k, v in json.loads(saved).items() if k != "feature_set"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="feature_set"):
        bcl.main()
    p.write_text(saved, encoding="utf-8")

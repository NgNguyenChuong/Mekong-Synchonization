"""analyze_cv_family / analyze_final / analyze_learnability / build_comparison_ledger voi --region (Dot 8).

Du lieu gia vung 'mississippi': 13 luoi (dien tich bang MS that), 27 don vi CV (G > 20 -> Monte Carlo) + 2 khoi giu
rieng x 31 diem x 2 mua; run_meta/config ghi region. Kiem: assert region, diem loai, pairs-mode, G, m Holm.
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
from training.block_stats import DEFAULT_MC_PERM  # noqa: E402

REG = "mississippi"
AREA_MS = {"h3_res_5": (5, 270.2553), "h3_res_6": (6, 38.6084), "h3_res_7": (7, 5.5161), "s2_level_9": (5, 305.4716),
           "s2_level_10": (6, 76.3071), "s2_level_11": (6, 19.0765), "s2_level_12": (7, 4.769),
           "square_utm_17087m": (5, 291.8682), "square_utm_6458m": (6, 41.6938), "square_utm_2441m": (7, 5.9568),
           "latlon_0.1552deg": (5, 257.5282), "latlon_0.0586deg": (6, 36.6963), "latlon_0.0222deg": (7, 5.2669)}
AREA_NL = {"h3_res_5": (5, 297.2772), "h3_res_6": (6, 42.47), "h3_res_7": (7, 6.0672), "s2_level_9": (5, 322.6967),
           "s2_level_10": (6, 80.6985), "s2_level_11": (6, 20.1776), "s2_level_12": (7, 5.0443),
           "square_utm_17087m": (5, 291.8667), "square_utm_6458m": (6, 41.6932), "square_utm_2441m": (7, 5.9567),
           "latlon_0.1552deg": (5, 255.5381), "latlon_0.0586deg": (6, 36.422), "latlon_0.0222deg": (7, 5.2274)}
CV_UNITS = [f"u{i}" for i in range(27)]
HO_UNITS = ["h0", "h1"]
NPT = 31  # bo 1 diem/khoi van con >= 30 diem (min_pts) -> G khong doi
EXCL = ["u0_p0", "u1_p3"]
T = "ndwi"


def _pts(units):
    return [(f"{u}_p{i}", u) for u in units for i in range(NPT)]


def _area(region, area):
    return pd.DataFrame([{"file": f"C:/x/{region}/grids/{g}.geojson", "tier_h3_res": t, "mean_area_km2": a}
                         for g, (t, a) in area.items()])


def _at(region, area):
    """Bang dien tich nhu acf.area_table (them cot grid)."""
    t = _area(region, area)
    t["grid"] = list(area)
    return t


def _oof(grid, model, scheme):
    pts = _pts(CV_UNITS)
    df = pd.DataFrame([(grid, p, s, u, "oof") for p, u in pts for s in base.SEASONS],
                      columns=["grid", "point_id", "season", "unit_id", "pred_source"])
    df["err"] = base._abs_err(T, grid, model, scheme, pts, base.SEASONS)
    return df


def _final(grid, model):
    sp = _pts(HO_UNITS)
    rows = [(p, s, u, "final", "khong_gian") for p, u in sp for s in base.SEASONS]
    rows += [(p, 2020, u, "final", "thoi_gian") for p, u in _pts(CV_UNITS)]
    rows += [(p, 2020, u, "final", "ca_hai") for p, u in sp]
    df = pd.DataFrame(rows, columns=["point_id", "season", "unit_id", "pred_source", "test_group"])
    df["err"] = 0.0
    k = df["test_group"] == "khong_gian"
    df.loc[k, "err"] = base._abs_err(T, grid, model, 42, sp, base.SEASONS, offset=1)
    return df


def build(exp, region=REG):
    meta = {"region": region}
    for g in AREA_MS:
        for m in base.MODELS:
            for s in (42, 43, 44):
                d = base._write_run(exp, run_name("cv1", g, m, s, "", None, T), "cv", T, meta, {"region": region})
                _oof(g, m, s).to_csv(d / "oof_points.csv", index=False)
            f = _final(g, m)
            d = base._write_run(exp, run_name("cv1", g, m, 42, "", None, T), "final", T,
                                {**meta, "empty_test_groups_declared": []}, {**base._final_cfg(f, ()), "region": region})
            f.to_csv(d / "final_points.csv", index=False)


SEEN_EVAL = []


def _fake_point_units(err, folds_csv, eval_dir=None):
    SEEN_EVAL.append(eval_dir)
    return pd.Series({p: u for p, u in _pts(CV_UNITS + HO_UNITS)}), CV_UNITS, HO_UNITS


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(acf, "point_units", _fake_point_units)
    b = tmp_path_factory.mktemp("vung")
    exp = b / "exp"
    build(exp)
    at = b / "dien_tich.csv"
    _area(REG, AREA_MS).to_csv(at, index=False)
    man = b / "repo" / "KE_HOACH" / "ket-qua" / "man.csv"
    man.parent.mkdir(parents=True)
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/x.csv"]}).to_csv(man, index=False)
    excl = b / "excl.txt"
    excl.write_text("# thu\n" + "\n".join(EXCL) + "\n", encoding="utf-8")
    empty = b / "rong.txt"
    empty.write_text("# rong\n", encoding="utf-8")
    yield {"b": b, "exp": exp, "at": at, "man": man, "excl": excl, "empty": empty}
    mp.undo()


def _args(w, out, mode, excl=None, extra=()):
    return ["--region", REG, "--pairs-mode", mode, "--exp-root", str(w["exp"]), "--area-table", str(w["at"]),
            "--out-dir", str(out), "--frozen-manifest", str(w["man"]), "--folds-csv", "x.csv",
            "--exclude-points", str(excl or w["excl"]), "--no-table", *extra]


def _run(mod, args):
    mod.main(mod.build_parser().parse_args(args))


def _prov(path):
    return json.loads(open(f"{path}.provenance.json", encoding="utf-8").read())


@pytest.fixture(scope="module")
def runs(world):
    """learnability -> cv_family -> final cho ratio va dbscl12 (thu muc rieng)."""
    outs = {}
    for mode in ("ratio", "dbscl12"):
        out = world["b"] / f"out_{mode}"
        _run(alr, _args(world, out, mode, extra=["--targets", T]))
        _run(acf, _args(world, out, mode, extra=["--target", T]))
        _run(afin, _args(world, out, mode, extra=["--target", T]))
        outs[mode] = out
    return outs


# ---------------------------------------------------------
# Tap cap
# ---------------------------------------------------------
def test_pairs_ratio_theo_vung():
    ms, _ = acf.select_pairs(_at("mississippi", AREA_MS), "mississippi", "ratio")
    nl, _ = acf.select_pairs(_at("nile", AREA_NL), "nile", "ratio")
    assert len(ms) == 14 and ms.groupby("tier").size().to_dict() == {5: 6, 6: 3, 7: 5}
    assert len(nl) == 13 and nl.groupby("tier").size().to_dict() == {5: 5, 6: 3, 7: 5}


def test_pairs_dbscl12_giu_cap_va_chieu_dbscl_canh_bao_ti_le():
    p, warn = acf.select_pairs(_at("mississippi", AREA_MS), "mississippi", "dbscl12")
    assert list(zip(p["grid_a"], p["grid_b"])) == list(acf.DBSCL_PAIRS)
    over = p[p["area_ratio"] > 1.2]
    assert len(over) == len(warn) > 0  # MS: s2_level_12/square_utm_2441m ~ 1,25
    assert all(f"{a}/{b}" in " ".join(warn) for a, b in zip(over["grid_a"], over["grid_b"]))
    dbscl = _at("dbscl", base.AREA)
    d, w = acf.select_pairs(dbscl, "dbscl", "dbscl12")
    assert set(zip(d["grid_a"], d["grid_b"])) == set(acf.DBSCL_PAIRS) and w == []


def test_pairs_bang_dien_tich_sai_vung_loi():
    with pytest.raises(SystemExit, match="khong phai cua vung"):
        acf.select_pairs(_at("nile", AREA_NL), "mississippi", "ratio")


# ---------------------------------------------------------
# Mac dinh theo vung, diem loai
# ---------------------------------------------------------
def test_mac_dinh_theo_vung():
    a = acf.build_parser().parse_args(["--region", REG, "--pairs-mode", "ratio"])
    acf.resolve_region(a)
    assert os.path.normpath(a.exp_root) == os.path.join(ROOT, "artifacts", "experiments", REG)
    assert os.path.normpath(a.out_dir) == os.path.join(ROOT, "KE_HOACH", "ket-qua", "doi_chung", REG)
    assert a.area_table.replace("\\", "/").endswith(f"/{REG}/reports/dot4_doi_chieu_dien_tich.csv")
    assert a.folds_csv.replace("\\", "/").endswith(f"/{REG}/eval/cv_folds.csv")
    assert a.exclude == ["pt_03727", "pt_15303"] and a.eval_kw["eval_dir"].endswith(f"/{REG}/eval")
    n = acf.build_parser().parse_args(["--region", "nile", "--pairs-mode", "dbscl12"])
    assert acf.resolve_region(n).exclude == []


def test_dbscl_mac_dinh_nhu_cu():
    for mod in (acf, afin, alr):
        a = mod.build_parser().parse_args([])
        mod_a = acf.resolve_region(a)
        assert mod_a.pairs_mode == "dbscl12" and mod_a.exclude == [] and mod_a.eval_kw == {}
        assert mod_a.exp_root == acf.EXP_ROOT and mod_a.out_dir == acf.RESULTS
        assert mod_a.area_table == os.path.join(acf.RESULTS, "dot4_doi_chieu_dien_tich.csv")
        assert mod_a.folds_csv == os.path.join(ROOT, "data", "eval", "cv_folds.csv")


def test_vung_bat_buoc_pairs_mode():
    for mod in (acf, afin, alr):
        with pytest.raises(SystemExit, match="bat buoc --pairs-mode"):
            acf.resolve_region(mod.build_parser().parse_args(["--region", "nile"]))


def test_diem_loai_doc_va_kiem(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("a  # ghi chu\n\n# chi ghi chu\nb\n", encoding="utf-8")
    assert acf.read_point_list(str(p)) == ["a", "b"]
    p.write_text("a\na\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="trung"):
        acf.read_point_list(str(p))
    with pytest.raises(SystemExit, match="khong co trong tap diem"):
        acf.check_excluded(["zz"], pd.Series({"a": "u0"}))
    df = pd.DataFrame({"point_id": ["a", "b", "c"]})
    assert list(acf.drop_points(df, ["b"])["point_id"]) == ["a", "c"]
    assert acf.drop_points(df, []) is df


# ---------------------------------------------------------
# Assert region tren run_meta / config
# ---------------------------------------------------------
@pytest.mark.parametrize("meta,cfg,want,ok", [
    ({}, {}, "dbscl", True),                                   # luot DBSCL cu khong co khoa region
    ({}, {}, REG, False),                                      # thieu region = dbscl != mississippi
    ({"region": REG}, {"region": REG}, REG, True),
    ({"region": REG}, {"region": REG}, "dbscl", False),        # quen --region khi doc luot vung
    ({"region": REG}, {"region": "nile"}, REG, False),         # config lech
])
def test_check_run_region(tmp_path, meta, cfg, want, ok):
    d = base._write_run(tmp_path, "r", "cv", T, meta, cfg)
    if ok:
        acf.check_run(str(d), "r", T, "cv", want)
    else:
        with pytest.raises(SystemExit, match="region"):
            acf.check_run(str(d), "r", T, "cv", want)


def test_doc_luot_vung_bang_region_khac_loi(world, runs, tmp_path):
    """Luot MS doc voi --region nile (bang dien tich/cong hop le) -> loi o check_run, khong ghi bang."""
    at = tmp_path / "nl.csv"
    _area("nile", AREA_NL).to_csv(at, index=False)
    a = _args(world, tmp_path / "o", "ratio", extra=["--target", T, "--gate", str(runs["ratio"] / "dot7_cong_kiem_dinh.csv")])
    a[a.index("--region") + 1] = "nile"
    a[a.index("--area-table") + 1] = str(at)
    with pytest.raises(SystemExit, match="region 'mississippi' khac --region 'nile'"):
        _run(acf, a)
    assert not (tmp_path / "o" / acf.out_name(T, "cv1")).exists()


# ---------------------------------------------------------
# Dau-cuoi: G, Monte Carlo, m Holm, diem loai
# ---------------------------------------------------------
def test_cv_family_vung_G_holm_diem_loai(world, runs):
    for mode, n_pairs, m_cong in (("ratio", 14, 9), ("dbscl12", 12, 6)):  # muc 7 truot cong (luoi BAD)
        out = runs[mode] / acf.out_name(T, "cv1")
        r = pd.read_csv(out)
        f1 = r[r["family"] == "F1_chinh"]
        assert len(f1) == n_pairs
        assert (f1["G"] == 27).all() and (~f1["p_exact"].astype(bool)).all() and (f1["n_perm"] == DEFAULT_MC_PERM).all()
        assert int(f1["kiem_dinh"].astype(bool).sum()) == m_cong
        assert (f1["n_common_point_seasons"] == (len(CV_UNITS) * NPT - len(EXCL)) * 2).all()
        pv = _prov(out)
        assert pv["region"] == REG and pv["pairs_mode"] == mode and pv["holm_m_f1"] == m_cong
        assert pv["G_don_vi_cv"] == 27 and pv["sign_flip"]["exact"] == [False]
        assert pv["sign_flip"]["n_perm"] == [DEFAULT_MC_PERM]
        assert pv["n_diem_loai"] == 2 and pv["diem_loai"] == EXCL and len(pv["cap_f1"]) == n_pairs
        assert (pv["canh_bao_ti_le"] != []) == (mode == "dbscl12")
        sens = pd.read_csv(runs[mode] / os.path.basename(out).replace(".csv", "__holm_muc_kiem_dinh.csv"))
        assert int(sens[sens["family"] == "F1_chinh"]["holm_m"].max()) == n_pairs
    assert set(SEEN_EVAL) == {acf.region_defaults(REG)["eval_dir"]}


def test_final_va_hoc_duoc_vung(runs):
    out = runs["ratio"]
    cap = pd.read_csv(out / f"dot7_{T}_cv1_final_cap.csv")
    assert len(cap) == 14 and cap["doi_chieu_cv"].all() and (cap["G"] == 27).all()
    pv = _prov(out / f"dot7_{T}_cv1_final_cap.csv")
    assert pv["region"] == REG and pv["holm_m_f1"] == 9 and pv["n_diem_loai"] == 2
    assert _prov(out / f"dot7_{T}_cv1_final_nhom.csv")["G_don_vi_cv"] == 27
    hoc = pd.read_csv(out / f"dot7_{T}_hoc_duoc.csv")
    assert int(hoc["holm_m"].max()) == 13 and (hoc["G"] == 27).all()
    ph = _prov(out / f"dot7_{T}_hoc_duoc.csv")
    assert ph["region"] == REG and ph["holm_m_cong"] == 13 and ph["G_don_vi_cv"] == 27
    assert _prov(out / "dot7_cong_kiem_dinh.csv")["region"] == REG


def test_khong_loai_diem_thi_du_diem(world, tmp_path):
    out = tmp_path / "o"
    _run(alr, _args(world, out, "ratio", excl=world["empty"], extra=["--targets", T]))
    _run(acf, _args(world, out, "ratio", excl=world["empty"], extra=["--target", T]))
    r = pd.read_csv(out / acf.out_name(T, "cv1"))
    assert (r["n_common_point_seasons"] == len(CV_UNITS) * NPT * 2).all()
    assert _prov(out / acf.out_name(T, "cv1"))["n_diem_loai"] == 0


# ---------------------------------------------------------
# Ledger
# ---------------------------------------------------------
def test_ledger_kiem_region(tmp_path):
    p = tmp_path / "kd.csv"
    pd.DataFrame({"x": [1]}).to_csv(p, index=False)
    with pytest.raises(SystemExit, match="thieu provenance"):
        bcl.read_checked(p, REG, set())
    assert len(bcl.read_checked(p, "dbscl", set())) == 1  # file DBSCL cu khong co provenance
    (tmp_path / "kd.csv.provenance.json").write_text(json.dumps({"region": "nile", "pairs_mode": "ratio"}))
    with pytest.raises(SystemExit, match="region"):
        bcl.read_checked(p, REG, set())
    seen = set()
    bcl.read_checked(p, "nile", seen)
    assert seen == {"ratio"}
    kd = pd.DataFrame([base_kd()])
    rows = bcl.ledger_for_target("ndwi", kd, pd.DataFrame(columns=["grid_a", "grid_b", "label"]), None,
                                 {"h3_res_5": 1.0, "latlon_0.1552deg": 1.0}, bcl.REGION_LABEL[REG])
    assert rows[0]["region"] == "Mississippi" and rows[0]["G"] == 27


def base_kd():
    return dict(family="F1_chinh", grid_a="h3_res_5", grid_b="latlon_0.1552deg", delta_ref_level=5, area_ratio=1.0,
                G=27, n_common_point_seasons=10, delta_hat=1.0, delta_ref=10.0, delta_min_thr=0.5, ci_tost_low=0,
                ci_tost_high=2, ci_low=-1, ci_high=3, p_value=0.01, p_holm=0.02, seeds_same_dir=True,
                label="tuong_duong", cong_hoc_duoc=True, muc_kiem_dinh=True)

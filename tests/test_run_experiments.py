"""scripts/run_experiments.py: ghi tag/commit + sha256 bang/fold/dap an, chay tiep duoc, bang doi -> chay lai.
Bang + dap an TONG HOP tren luoi that h3_res_5 (can data/grids, data/eval)."""
import json
import os
import shutil
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRID = os.path.join(ROOT, "data", "grids", "h3_res_5.geojson")
POINTS = os.path.join(ROOT, "data", "eval", "eval_points.geojson")
FOLDS = os.path.join(ROOT, "data", "eval", "cv_folds.csv")
RUNNER = os.path.join(ROOT, "scripts", "run_experiments.py")
need_real = pytest.mark.skipif(not all(os.path.exists(p) for p in (GRID, POINTS, FOLDS)), reason="thieu du lieu luoi")


@pytest.fixture()
def setup(tmp_path):
    rng = np.random.default_rng(5)
    g = gpd.read_file(GRID)
    cells = g["cell_id"].astype(str).tolist()
    t = pd.DataFrame([(c, s) for c in cells for s in range(2018, 2022)], columns=["cell_id", "season"])
    dem = pd.Series(rng.uniform(0, 3, len(cells)), index=cells)
    t["dem_mean"] = t["cell_id"].map(dem)
    t["salinity"] = 4 - t["dem_mean"] + rng.normal(0, 0.1, len(t))
    # du 23 dac trung mac dinh (runner coi thieu muc nao la LOI)
    from static_features import WORLDCOVER_NAMES
    from training.features import DEFAULT_ALLOWED_FEATURES
    for f in DEFAULT_ALLOWED_FEATURES:
        names = [f"landcover_class_{n}" for n in WORLDCOVER_NAMES.values()] if f.endswith("*") else [f]
        for n in names:
            if n not in t.columns:
                t[n] = rng.uniform(0, 1, len(t))
    t["train_ok"], t["scope_frac"] = True, 0.5
    t["train_ok_scope"] = True
    gu = g.to_crs(32648)
    t = t.merge(pd.DataFrame({"cell_id": cells, "scope_cx": gu.geometry.centroid.x, "scope_cy": gu.geometry.centroid.y}),
                on="cell_id")
    (tmp_path / "tables").mkdir()
    t.to_csv(tmp_path / "tables" / "h3_res_5_unified.csv", index=False)
    (tmp_path / "grids").mkdir()
    shutil.copy(GRID, tmp_path / "grids" / "h3_res_5.geojson")
    (tmp_path / "folds").mkdir()
    shutil.copy(FOLDS, tmp_path / "folds" / "cv_folds_s42.csv")
    pts = gpd.read_file(POINTS)
    ref = pd.DataFrame([(p, s) for p in pts["point_id"] for s in range(2018, 2022)], columns=["point_id", "season"])
    ref["n_valid_3x3"] = 9
    ref["ref_salinity"] = rng.uniform(0, 4, len(ref))
    ref.to_csv(tmp_path / "ref.csv", index=False)
    return tmp_path


def _run(tmp, *extra):  # noqa: D103
    cmd = [sys.executable, RUNNER, "--prefix", "t", "--grids", "h3_res_5", "--models", "linear", "idw",
           "--schemes", "42", "--tables-dir", str(tmp / "tables"), "--grids-dir", str(tmp / "grids"),
           "--folds-dir", str(tmp / "folds"), "--points-ref", str(tmp / "ref.csv"), "--cwd", str(tmp), *extra]
    return subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=900,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))


@need_real
def test_ghi_meta_chay_tiep_va_chay_lai_khi_bang_doi(setup):
    tmp = setup
    p = _run(tmp, "--allow-untagged", "--allow-dirty")
    assert p.returncode == 0, p.stdout + p.stderr
    man = pd.read_csv(tmp / "artifacts" / "experiments" / "t_manifest.csv")
    assert list(man["status"]) == ["xong", "xong"]
    exp = tmp / "artifacts" / "experiments"
    meta = json.loads((exp / "t__h3_res_5__linear__s42" / "cv" / "run_meta.json").read_text(encoding="utf-8"))
    for k in ("git_commit", "git_tag", "table_sha256", "cv_folds_sha256", "points_ref_sha256", "cmd"):
        assert k in meta
    assert meta["returncode"] == 0 and len(meta["table_sha256"]) == 64 and meta["allow_untagged"]
    assert (exp / "t__h3_res_5__idw__s42" / "cv" / "oof_points.csv").exists()
    # S2: bien the dap an mac dinh main ghi vao meta/khoa va truyen xuong train.py
    assert meta["points_ref_variant"] == "main" and meta["points_subset_of_ref"] is False
    i = meta["cmd"].index("--points-ref-variant")
    assert meta["cmd"][i + 1] == "main" and "--points-subset-of-ref" not in meta["cmd"]
    # chay lai: bo qua ca hai
    p = _run(tmp, "--allow-untagged", "--allow-dirty")
    assert p.returncode == 0 and list(pd.read_csv(exp / "t_manifest.csv")["status"]) == ["bo_qua_da_xong"] * 2
    # mat config cua mot lan (vd bi ngat) -> chi lan do chay lai
    (exp / "t__h3_res_5__idw__s42" / "cv" / "config.json").unlink()
    p = _run(tmp, "--allow-untagged", "--allow-dirty")
    assert list(pd.read_csv(exp / "t_manifest.csv")["status"]) == ["bo_qua_da_xong", "xong"]
    assert any(d.name.startswith("t__h3_res_5__idw__s42") for d in exp.iterdir())
    # bang doi -> chay lai ca hai, thu muc cu giu lai
    t = pd.read_csv(tmp / "tables" / "h3_res_5_unified.csv")
    t.loc[0, "salinity"] += 1
    t.to_csv(tmp / "tables" / "h3_res_5_unified.csv", index=False)
    p = _run(tmp, "--allow-untagged", "--allow-dirty")
    assert list(pd.read_csv(exp / "t_manifest.csv")["status"]) == ["xong", "xong"]
    assert (exp / "_cu").is_dir() and not any("__cu_" in d.name for d in (exp / "t__h3_res_5__linear__s42").iterdir())
    # seeds doi -> chay lai
    p = _run(tmp, "--allow-untagged", "--allow-dirty", "--seeds", "42", "43")
    assert list(pd.read_csv(exp / "t_manifest.csv")["status"]) == ["xong", "xong"]


def test_ket_qua_thu_nghiem_khong_thay_lan_chinh_thuc(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_experiments as rx

    want = {k: None for k in rx.KEY_FIELDS}
    want.update(git_commit="c1", allow_dirty=False, allow_untagged=False, git_dirty_src_scripts=False, seeds=[42],
                mode="cv")
    meta = {**want, "returncode": 0, "allow_dirty": True, "git_dirty_src_scripts": True}
    # CHG-22: config phai co khoa features moi duoc tinh la xong (truoc day "{}" du) - xem test ngay duoi
    (tmp_path / "config.json").write_text(json.dumps({"features": ["dem_mean"]}))
    (tmp_path / "run_meta.json").write_text(json.dumps(meta))
    assert not rx.is_done(str(tmp_path / "run_meta.json"), str(tmp_path / "config.json"), want)
    (tmp_path / "run_meta.json").write_text(json.dumps({**want, "returncode": 0}))
    assert rx.is_done(str(tmp_path / "run_meta.json"), str(tmp_path / "config.json"), want)
    (tmp_path / "run_meta.json").write_text(json.dumps({**want, "returncode": 0, "seeds": [42, 43]}))
    assert not rx.is_done(str(tmp_path / "run_meta.json"), str(tmp_path / "config.json"), want)


def test_is_done_thieu_features_hoac_file_diem_la_chua_xong(tmp_path):
    """CHG-22: config thieu khoa features; co cham theo diem (points_ref_sha256) ma thieu oof_points.csv (cv) /
    final_points.csv (final) -> CHUA xong (chay lai), khong bo qua. check_features: thieu khoa -> LOI."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_experiments as rx

    want = {k: None for k in rx.KEY_FIELDS}
    want.update(git_commit="c1", allow_dirty=False, allow_untagged=False, git_dirty_src_scripts=False, seeds=[42],
                mode="cv", points_ref_sha256="abc")
    meta, cfg = tmp_path / "run_meta.json", tmp_path / "config.json"
    meta.write_text(json.dumps({**want, "returncode": 0}))
    cfg.write_text("{}")
    (tmp_path / "oof_points.csv").write_text("point_id\n")
    assert not rx.is_done(str(meta), str(cfg), want)                       # thieu khoa features
    assert rx.check_features(str(tmp_path), "hist_gb", {}) == "config.json thieu khoa features"
    cfg.write_text(json.dumps({"features": ["dem_mean"]}))
    assert rx.is_done(str(meta), str(cfg), want)
    os.remove(tmp_path / "oof_points.csv")
    assert not rx.is_done(str(meta), str(cfg), want)                       # thieu file diem cua mode cv
    (tmp_path / "final_points.csv").write_text("point_id\n")
    assert not rx.is_done(str(meta), str(cfg), want)                       # file cua mode khac khong thay duoc
    want_f = {**want, "mode": "final"}
    meta.write_text(json.dumps({**want_f, "returncode": 0}))
    assert rx.is_done(str(meta), str(cfg), want_f)
    want_np = {**want, "points_ref_sha256": None}                          # --no-point-eval: khong can file diem
    meta.write_text(json.dumps({**want_np, "returncode": 0}))
    assert rx.is_done(str(meta), str(cfg), want_np)


@need_real
def test_khong_tag_thi_tu_choi(setup):
    r = subprocess.run(["git", "describe", "--tags", "--exact-match", "HEAD"], cwd=ROOT, capture_output=True)
    if r.returncode == 0:
        pytest.skip("HEAD dang co tag")
    p = _run(setup, "--allow-dirty")
    assert p.returncode != 0 and "tag" in (p.stdout + p.stderr)


@need_real
def test_thieu_points_ref_phai_ghi_ro(setup):
    cmd = [sys.executable, RUNNER, "--prefix", "t", "--grids", "h3_res_5", "--points-ref", "", "--allow-untagged",
           "--allow-dirty", "--tables-dir", str(setup / "tables"), "--grids-dir", str(setup / "grids"),
           "--folds-dir", str(setup / "folds"), "--cwd", str(setup)]
    p = subprocess.run(cmd, cwd=setup, capture_output=True, text=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert p.returncode != 0 and "--no-point-eval" in (p.stdout + p.stderr)


@need_real
def test_bo_dac_trung_dat_ten_tach_lan_chay_va_ghi_khoa(setup):
    """--feature-set: ten lan chay co hau to __fs-<ten>, config dung dung danh sach, khoa co sha file bo dac trung."""
    tmp = setup
    p = _run(tmp, "--allow-untagged", "--allow-dirty", "--feature-set", "khong_diem", "--models", "linear")
    assert p.returncode == 0, p.stdout + p.stderr
    exp = tmp / "artifacts" / "experiments"
    out = exp / "t__h3_res_5__linear__s42__fs-khong_diem" / "cv"
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    banned = {"tch_wl_p20c", "dist_mouth_river_km", "zos_mouth_p90", "sluice_frac"}
    assert not banned & set(cfg["features"]) and len(cfg["features"]) == 19
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["feature_set"] == "khong_diem" and len(meta["features_sha256"]) == 64
    p = _run(tmp, "--allow-untagged", "--allow-dirty", "--feature-set", "khong_diem", "--models", "linear")
    assert "bo_qua_da_xong" in pd.read_csv(exp / "t_manifest.csv")["status"].tolist()


# ------------------------------------------------------------------ S2: bien the dap an / cham phu (tien to rieng)
def _meta(exp, name, mode="cv", **kw):
    d = exp / name / mode
    d.mkdir(parents=True)
    (d / "run_meta.json").write_text(json.dumps(kw), encoding="utf-8")


def test_variant_conflicts_cung_tien_to_cung_bo(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_experiments as rx

    exp = tmp_path / "experiments"
    _meta(exp, "cv1__h3_res_5__hist_gb__s42__keep6090", label_set="keep6090")          # truoc S2: khong co khoa = main
    _meta(exp, "cv1__h3_res_5__hist_gb__s42__keepwater", label_set="keepwater", points_ref_variant="main")
    _meta(exp, "cv10__h3_res_5__hist_gb__s42__keep6090", label_set="keep6090", points_ref_variant="keep6090")
    got = rx.variant_conflicts(str(exp), "cv1", "keep6090", "keep6090")
    assert len(got) == 1 and "cv1__h3_res_5__hist_gb__s42__keep6090" in got[0][0]   # cv10 khong tinh la cv1
    assert rx.variant_conflicts(str(exp), "cv1", "keep6090", "main") == []
    assert rx.variant_conflicts(str(exp), "cv1", "keepwater", "keep6090")              # bo khac cung chan
    assert rx.variant_conflicts(str(exp), "phu6090", "keep6090", "keep6090") == []
    assert rx.variant_conflicts(str(exp), "cv10", "keep6090", "main")                  # main khong ghi de cham phu
    (exp / "cv1__x__hist_gb__s42__keep6090" / "cv").mkdir(parents=True)
    (exp / "cv1__x__hist_gb__s42__keep6090" / "cv" / "run_meta.json").write_text("{hong", encoding="utf-8")
    assert any("khong doc duoc" in r for _, r in rx.variant_conflicts(str(exp), "cv1", "keep6090", "main"))


def _runner_light(tmp, *extra):
    """Goi runner voi file gia (chi can ton tai/sha) - cac kiem S2 xay ra TRUOC khi doc bang."""
    for sub in ("grids", "tables", "folds"):
        (tmp / sub).mkdir(exist_ok=True)
    (tmp / "grids" / "h3_res_5.geojson").write_text("{}")
    for f in ("blocks.geojson", "points.geojson"):
        (tmp / f).write_text("{}")
    cmd = [sys.executable, RUNNER, "--grids", "h3_res_5", "--models", "hist_gb", "--schemes", "42",
           "--tables-dir", str(tmp / "tables"), "--grids-dir", str(tmp / "grids"), "--folds-dir", str(tmp / "folds"),
           "--blocks", str(tmp / "blocks.geojson"), "--points", str(tmp / "points.geojson"),
           "--points-ref", str(tmp / "ref.csv"), "--cwd", str(tmp), "--allow-untagged", "--allow-dirty", *extra]
    p = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return p.returncode, p.stdout + p.stderr


def test_runner_cham_phu_khong_ghi_de_lan_cham_chinh(tmp_path):
    (tmp_path / "ref.csv").write_text("point_id,season,ref_salinity,n_valid_3x3\n")
    (tmp_path / "ref.csv.provenance.json").write_text(json.dumps({"variant": "keep6090"}))
    exp = tmp_path / "artifacts" / "experiments"
    _meta(exp, "cv1__h3_res_5__hist_gb__s42__keep6090", label_set="keep6090", points_ref_variant="main")
    rc, out = _runner_light(tmp_path, "--prefix", "cv1", "--label-set", "keep6090", "--points-ref-variant", "keep6090")
    assert rc != 0 and "[LOI] Tien to 'cv1'" in out, out
    assert (exp / "cv1__h3_res_5__hist_gb__s42__keep6090" / "cv" / "run_meta.json").exists()  # khong bi chuyen _cu
    assert not (exp / "_cu").exists()
    # tien to rieng: qua kiem S2, dung o buoc sau (khong co bang) - chung to chan chi do trung tien to
    rc, out = _runner_light(tmp_path, "--prefix", "phu6090", "--label-set", "keep6090", "--points-ref-variant",
                            "keep6090", "--points-subset-of-ref")
    assert rc != 0 and "Khong co bang" in out and "[LOI] Tien to" not in out, out
    # dap an provenance keep6090 ma yeu cau main -> loi som
    rc, out = _runner_light(tmp_path, "--prefix", "phu6090", "--label-set", "keep6090")
    assert rc != 0 and "khac --points-ref-variant 'main'" in out, out


def test_dot7_target_khoa_meta_cu_va_ten_lan_chay(tmp_path):
    """--target: meta CU thieu khoa target = salinity (khong chay lai do man); bien khac -> chua xong; ten lan chay
    do man khong doi, bien khac them __t-<target>; provenance target lech -> chuoi loi."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_experiments as rx

    assert rx.run_name("cv1", "h3_res_5", "hist_gb", 42, "") == "cv1__h3_res_5__hist_gb__s42"
    assert rx.run_name("cv1", "h3_res_5", "hist_gb", 42, "", None, "salinity") == "cv1__h3_res_5__hist_gb__s42"
    assert rx.run_name("cv1", "h3_res_5", "hist_gb", 42, "", None, "ndwi") == "cv1__h3_res_5__hist_gb__s42__t-ndwi"
    want = {k: None for k in rx.KEY_FIELDS}
    want.update(git_commit="c1", allow_dirty=False, allow_untagged=False, git_dirty_src_scripts=False, seeds=[42],
                mode="cv", target="salinity")
    old_meta = {k: v for k, v in want.items() if k != "target"}             # lan chay truoc khi co khoa target
    (tmp_path / "config.json").write_text(json.dumps({"features": ["dem_mean"]}))
    (tmp_path / "run_meta.json").write_text(json.dumps({**old_meta, "returncode": 0}))
    assert rx.is_done(str(tmp_path / "run_meta.json"), str(tmp_path / "config.json"), want)
    assert not rx.is_done(str(tmp_path / "run_meta.json"), str(tmp_path / "config.json"), {**want, "target": "ndwi"})
    p = tmp_path / "ref.csv"
    p.write_text("point_id\n")
    assert rx.target_mismatch(str(p), "ndwi", default="salinity") is None    # khong provenance -> train.py kiem
    (tmp_path / "ref.csv.provenance.json").write_text(json.dumps({"variant": "main"}))
    assert "salinity" in rx.target_mismatch(str(p), "ndwi", default="salinity")
    assert rx.target_mismatch(str(p), "salinity", default="salinity") is None
    (tmp_path / "ref.csv.provenance.json").write_text(json.dumps({"target": "ndwi"}))
    assert rx.target_mismatch(str(p), "ndwi") is None


@need_real
def test_dot7_runner_dap_an_target_lech_dung_som(setup):
    tmp = setup
    with open(tmp / "ref.csv.provenance.json", "w", encoding="utf-8") as f:
        json.dump({"variant": "main"}, f)                                   # dap an do man (khong ghi target)
    p = _run(tmp, "--allow-untagged", "--allow-dirty", "--target", "rain_chirps")
    assert p.returncode != 0 and "target" in (p.stdout + p.stderr)
    assert not (tmp / "artifacts" / "experiments" / "t__h3_res_5__linear__s42__t-rain_chirps").exists()


def test_feature_set_tru_cot_cam_theo_bien():
    """khong_diem (b) = file tru TARGET_FORBIDDEN cua bien; do man / ndwi khong bo muc nao."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_experiments as rx

    ff = os.path.join(ROOT, "configs", "feature_sets", "khong_diem.txt")
    temps = ["temp_c", "temp_max_c", "temp_min_c", "rh_percent"]
    for t, drop in (("salinity", []), ("ndwi", []), ("rain_chirps", ["rain_mm"]), ("dsr_mcd18", ["solar"]),
                    ("t2m_era5", temps), ("rh_era5", temps)):
        req, keep, dropped = rx.feature_set_lists(ff, t)
        assert dropped == drop and keep == [e for e in req if e not in drop] and len(req) == 11


def test_b_mua_bang_khong_diem_cong_3():
    """(c) b_mua = (b) + tch_wl_p20c, zos_mua_mean, sluice_flag; dem cot that sau khi giai tien to landcover."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_experiments as rx
    from training.features import find_leak_columns, find_target_leak_columns, resolve_feature_list

    fs = os.path.join(ROOT, "configs", "feature_sets")
    lc = [f"landcover_class_{c}" for c in ("Trees", "Shrubland", "Grassland", "Cropland", "Built_up", "Bareland",
                                            "Water", "Wetland", "Mangroves")]
    new, old = ["tch_wl_p20c", "zos_mua_mean", "sluice_flag"], ["dist_mouth_river_km", "zos_mouth_p90", "sluice_frac"]
    cols = ["rain_mm", "solar", "temp_c", "temp_max_c", "temp_min_c", "rh_percent", "dem_mean", "dist_main_river_km",
            "dist_any_water_km", "dist_coast_km", *lc, *old, *new]
    for t, n in (("salinity", 22), ("ndwi", 22), ("rain_chirps", 21), ("dsr_mcd18", 21), ("t2m_era5", 18),
                 ("rh_era5", 18)):
        kb = resolve_feature_list(cols, rx.feature_set_lists(os.path.join(fs, "khong_diem.txt"), t)[1])[0]
        kc = resolve_feature_list(cols, rx.feature_set_lists(os.path.join(fs, "b_mua.txt"), t)[1])[0]
        assert len(kc) == n == len(kb) + 3 and set(kc) - set(kb) == set(new) and not set(old) & set(kc), t
        assert not find_leak_columns(new) and not find_target_leak_columns(new, t)


@need_real
def test_runner_feature_set_cung_target_tru_cot_cam(setup):
    """--feature-set khong_diem --target rain_chirps: truoc day train.py loi (rain_mm cam); nay truyen --features da
    tru, config khong co rain_mm (18 dac trung), run_meta ghi features_requested / features_dropped_forbidden."""
    tmp = setup
    rng = np.random.default_rng(7)
    t = pd.read_csv(tmp / "tables" / "h3_res_5_unified.csv")
    t["rain_chirps"] = t["rain_mm"] + rng.normal(0, 0.1, len(t))
    t.to_csv(tmp / "tables" / "h3_res_5_unified.csv", index=False)
    with open(tmp / "tables" / "h3_res_5_unified.csv.provenance.json", "w", encoding="utf-8") as f:
        json.dump({"target": "rain_chirps", "label_set": "chinh"}, f)
    ref = pd.read_csv(tmp / "ref.csv")[["point_id", "season"]]
    ref["ref_rain_chirps"], ref["src_px_valid"] = rng.uniform(0, 1, len(ref)), True
    ref.to_csv(tmp / "ref.csv", index=False)
    with open(tmp / "ref.csv.provenance.json", "w", encoding="utf-8") as f:
        json.dump({"target": "rain_chirps", "variant": "main", "ref_rule_kind": "pixel"}, f)
    p = _run(tmp, "--allow-untagged", "--allow-dirty", "--feature-set", "khong_diem", "--models", "linear",
             "--target", "rain_chirps")
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp / "artifacts" / "experiments" / "t__h3_res_5__linear__s42__fs-khong_diem__t-rain_chirps" / "cv"
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert "rain_mm" not in cfg["features"] and len(cfg["features"]) == 18
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["features_dropped_forbidden"] == ["rain_mm"] and "rain_mm" in meta["features_requested"]
    assert "--features-file" not in meta["cmd"] and "rain_mm" not in meta["cmd"]
    assert meta["feature_set"] == "khong_diem" and len(meta["features_sha256"]) == 64


# ------------------------------------------------------------------ CHG-28: giu rieng tung mua (--holdout-season)
def test_holdout_season_ten_va_nhom_rong_theo_mua(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_experiments as rx

    assert (rx.run_name("cv1", "h3_res_7", "hist_gb", 42, "", "khong_diem", "dsr_mcd18", 2016)
            == "cv1__h3_res_7__hist_gb__s42__fs-khong_diem__t-dsr_mcd18__ho-2016")
    assert rx.run_name("cv1", "h3_res_7", "hist_gb", 42, "", None, "salinity", 2020) == "cv1__h3_res_7__hist_gb__s42__ho-2020"
    tb = tmp_path / "t.csv"
    tb.write_text("x\n")
    (tmp_path / "t.csv.provenance.json").write_text(json.dumps({"target": "dsr_mcd18", "holdout_season_nan": {
        "target": "dsr_mcd18", "seasons": [2020], "reason": "gia"}}))
    assert rx.table_empty_groups(str(tb), "dsr_mcd18") == ["thoi_gian", "ca_hai"]
    assert rx.table_empty_groups(str(tb), "dsr_mcd18", (2016,)) == []    # truoc day van khai bao rong -> train.py ma 2


def test_holdout_season_chi_voi_mode_final(tmp_path):
    (tmp_path / "ref.csv").write_text("point_id,season,ref_salinity,n_valid_3x3\n")
    for mode in ([], ["--mode", "cv"]):
        rc, out = _runner_light(tmp_path, "--prefix", "cv1", "--holdout-season", "2016", *mode)
        assert rc != 0 and "chi dung voi --mode final" in out, out


@need_real
def test_holdout_season_bang_nan_2020_khong_vao_huan_luyen(setup):
    """Bang kieu buc xa: nhan 2020 NaN + train_ok_scope sai + khai bao holdout_season_nan [2020].
    Giu rieng 2019 -> co override, khong khai bao nhom rong, tap huan luyen chi mua 2018/2021 (2020 NaN bi bo);
    giu rieng 2020 -> khong co override, van khai bao nhom rong nhu cu."""
    tmp = setup
    rng = np.random.default_rng(9)
    t = pd.read_csv(tmp / "tables" / "h3_res_5_unified.csv")
    t["dsr_mcd18"] = t["solar"] + rng.normal(0, 0.1, len(t))
    t.loc[t["season"] == 2020, "dsr_mcd18"] = np.nan
    t.loc[t["season"] == 2020, "train_ok_scope"] = False
    t.to_csv(tmp / "tables" / "h3_res_5_unified.csv", index=False)
    decl = {"target": "dsr_mcd18", "seasons": [2020], "reason": "gia"}
    with open(tmp / "tables" / "h3_res_5_unified.csv.provenance.json", "w", encoding="utf-8") as f:
        json.dump({"target": "dsr_mcd18", "label_set": "chinh", "holdout_season_nan": decl}, f)
    ref = pd.read_csv(tmp / "ref.csv")[["point_id", "season"]]
    ref["ref_dsr_mcd18"] = np.where(ref["season"] == 2020, np.nan, rng.uniform(0, 1, len(ref)))
    ref["src_px_valid"] = ref["season"] != 2020
    ref.to_csv(tmp / "ref.csv", index=False)
    with open(tmp / "ref.csv.provenance.json", "w", encoding="utf-8") as f:
        json.dump({"target": "dsr_mcd18", "variant": "main", "ref_rule_kind": "pixel"}, f)
    exp = tmp / "artifacts" / "experiments"
    for s in (2019, 2020):
        p = _run(tmp, "--allow-untagged", "--allow-dirty", "--feature-set", "khong_diem", "--models", "linear",
                 "--target", "dsr_mcd18", "--mode", "final", "--holdout-season", str(s))
        assert p.returncode == 0, p.stdout + p.stderr
        out = exp / f"t__h3_res_5__linear__s42__fs-khong_diem__t-dsr_mcd18__ho-{s}" / "final"
        meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
        i = meta["cmd"].index("--holdout-seasons")
        assert meta["cmd"][i + 1] == str(s) and meta["holdout_season"] == s
        assert ("--allow-holdout-season-override" in meta["cmd"]) == (s != 2020)
        assert ("--allow-empty-test-groups" in meta["cmd"]) == (s == 2020)
        log = (out / "train_stdout.log").read_text(encoding="utf-8")
        assert ("mua [2018, 2021]" if s == 2019 else "mua [2018, 2019, 2021]") in log
        if s == 2019:
            fp = pd.read_csv(out / "final_points.csv")
            tg = fp[fp["test_group"] == "thoi_gian"]
            assert len(tg) and set(tg["season"]) == {2019} and np.isfinite(fp["err"]).all()

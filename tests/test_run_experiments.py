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
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "run_meta.json").write_text(json.dumps(meta))
    assert not rx.is_done(str(tmp_path / "run_meta.json"), str(tmp_path / "config.json"), want)
    (tmp_path / "run_meta.json").write_text(json.dumps({**want, "returncode": 0}))
    assert rx.is_done(str(tmp_path / "run_meta.json"), str(tmp_path / "config.json"), want)
    (tmp_path / "run_meta.json").write_text(json.dumps({**want, "returncode": 0, "seeds": [42, 43]}))
    assert not rx.is_done(str(tmp_path / "run_meta.json"), str(tmp_path / "config.json"), want)


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

"""Nhom test rong CO CHU DICH (CHG-25 muc 4): chi nhan khi provenance bang khai bao 'holdout_season_nan';
moi truong hop khac van LOI nhu cu (do man + bien khac khong doi)."""
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
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from training.split import (EMPTY_DECL_KEY, EMPTY_LABEL, assign_eval_split,  # noqa: E402
                            declared_empty_groups)

GRID = os.path.join(ROOT, "data", "grids", "h3_res_5.geojson")
POINTS = os.path.join(ROOT, "data", "eval", "eval_points.geojson")
FOLDS = os.path.join(ROOT, "data", "eval", "cv_folds.csv")
BLOCKS = os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson")
RUNNER = os.path.join(ROOT, "scripts", "run_experiments.py")
TRAIN = os.path.join(ROOT, "src", "training", "train.py")
need_real = pytest.mark.skipif(not all(os.path.exists(p) for p in (GRID, POINTS, FOLDS, BLOCKS)),
                               reason="thieu du lieu luoi")
DSR = "dsr_mcd18"
DECL = {"target": DSR, "seasons": [2020], "reason": "E4 (b): MCD18A1 thieu 03-31/12/2019"}
BOTH = ("thoi_gian", "ca_hai")


# ------------------------------------------------------------------ declared_empty_groups
def test_declared_empty_groups_hop_le_va_khong_khai_bao():
    assert declared_empty_groups(None, DSR) == ()
    assert declared_empty_groups({}, DSR) == ()
    assert declared_empty_groups({"target": DSR}, DSR) == ()
    assert declared_empty_groups({EMPTY_DECL_KEY: DECL}, DSR) == BOTH
    # khong phu het mua giu rieng -> nhom khong rong -> khong cho phep gi
    assert declared_empty_groups({EMPTY_DECL_KEY: {**DECL, "seasons": [2019]}}, DSR) == ()
    assert declared_empty_groups({EMPTY_DECL_KEY: DECL}, DSR, holdout_seasons=(2020, 2021)) == ()


@pytest.mark.parametrize("decl, msg", [
    ({**DECL, "target": "rain_chirps"}, "target"),
    ({k: v for k, v in DECL.items() if k != "target"}, "target"),
    ({**DECL, "seasons": []}, "seasons"),
    ({**DECL, "seasons": ["2020"]}, "seasons"),
    ({**DECL, "seasons": [True]}, "seasons"),
    ({**DECL, "seasons": 2020}, "seasons"),
    ({**DECL, "reason": "  "}, "reason"),
    ({k: v for k, v in DECL.items() if k != "reason"}, "reason"),
    ([2020], "dict"),
])
def test_declared_empty_groups_sai_dang_la_loi(decl, msg):
    with pytest.raises(ValueError, match=msg):
        declared_empty_groups({EMPTY_DECL_KEY: decl}, DSR)


# ------------------------------------------------------------------ assign_eval_split(allowed_empty)
@pytest.fixture()
def toy():
    cells = pd.DataFrame({"cell_id": ["a", "b", "h"], "touches_holdout": [False, False, True]})
    df = pd.DataFrame([(c, s) for c in cells["cell_id"] for s in (2019, 2020, 2021)], columns=["cell_id", "season"])
    df[DSR] = np.arange(len(df), dtype=float)
    return df, cells


def test_assign_eval_split_khong_khai_bao_van_loi(toy):
    df, cells = toy
    no2020 = df[df["season"] != 2020]
    with pytest.raises(ValueError, match=r"Nhom test rong: \['thoi_gian', 'ca_hai'\]"):
        assign_eval_split(no2020, cells)
    with pytest.raises(ValueError, match=r"Nhom test rong: \['ca_hai'\]"):      # khai bao thieu nhom
        assign_eval_split(no2020, cells, allowed_empty=("thoi_gian",))
    with pytest.raises(ValueError, match=r"Nhom test rong: \['khong_gian'\]"):  # nhom khong gian khong khai bao duoc
        assign_eval_split(no2020[no2020["cell_id"] != "h"], cells, allowed_empty=BOTH)


def test_assign_eval_split_khai_bao_dung(toy):
    df, cells = toy
    no2020 = df[df["season"] != 2020]
    out = assign_eval_split(no2020, cells, allowed_empty=BOTH)
    assert set(out["test_group"]) == {"", "khong_gian"}
    assert out.index.equals(no2020.index)
    pd.testing.assert_frame_equal(out, assign_eval_split(no2020, cells, allowed_empty=("ca_hai", "thoi_gian")))


@pytest.mark.parametrize("allowed", [("khong_gian",), ("xyz",), ("thoi_gian", "khong_gian")])
def test_assign_eval_split_nhom_ngoai_danh_sach_la_loi(toy, allowed):
    df, cells = toy
    with pytest.raises(ValueError, match="ngoai"):
        assign_eval_split(df[df["season"] != 2020], cells, allowed_empty=allowed)


def test_assign_eval_split_khai_bao_rong_ma_co_dong_la_loi(toy):
    df, cells = toy
    with pytest.raises(ValueError, match="khai bao rong nhung"):
        assign_eval_split(df, cells, allowed_empty=BOTH)


# ------------------------------------------------------------------ build_unified_dot7.holdout_nan_declaration
def test_build_unified_dot7_khai_bao_tu_provenance_nhan():
    from build_unified_dot7 import NAN_SEASON_REASON, holdout_nan_declaration
    from unified_table import TRAIN_COL

    t = pd.DataFrame({"cell_id": ["a", "a", "b", "b"], "season": [2019, 2020, 2019, 2020],
                      DSR: [1.0, np.nan, 2.0, np.nan], TRAIN_COL: [True, False, True, False]})
    assert holdout_nan_declaration({}, t, DSR) is None
    assert holdout_nan_declaration({"nan_seasons": []}, t, DSR) is None
    decl = holdout_nan_declaration({"nan_seasons": [2020]}, t, DSR)
    assert decl["seasons"] == [2020] and decl["reason"] == NAN_SEASON_REASON[DSR] and decl["target"] == DSR
    assert declared_empty_groups({EMPTY_DECL_KEY: decl}, DSR) == BOTH   # vong tron ghi -> doc
    bad = t.copy()
    bad.loc[1, DSR] = 3.0
    with pytest.raises(SystemExit, match="nhan huu han"):
        holdout_nan_declaration({"nan_seasons": [2020]}, bad, DSR)
    bad = t.copy()
    bad.loc[3, TRAIN_COL] = True
    with pytest.raises(SystemExit, match=TRAIN_COL):
        holdout_nan_declaration({"nan_seasons": [2020]}, bad, DSR)
    with pytest.raises(SystemExit, match="ly do"):
        holdout_nan_declaration({"nan_seasons": [2020]}, t.rename(columns={DSR: "rain_chirps"}), "rain_chirps")
    with pytest.raises(SystemExit, match="khong co dong"):
        holdout_nan_declaration({"nan_seasons": [2030]}, t, DSR)


# ------------------------------------------------------------------ runner
def test_runner_table_empty_groups(tmp_path):
    import run_experiments as rx

    tab = tmp_path / "h3_res_5_unified.csv"
    tab.write_text("cell_id\n")
    with pytest.raises(ValueError, match="thieu provenance"):
        rx.table_empty_groups(str(tab), DSR)
    assert rx.table_empty_groups(str(tab), "salinity") == []                 # do man: khong doi
    prov = tmp_path / "h3_res_5_unified.csv.provenance.json"
    prov.write_text(json.dumps({"target": DSR}))
    assert rx.table_empty_groups(str(tab), DSR) == []
    prov.write_text(json.dumps({"target": DSR, EMPTY_DECL_KEY: DECL}))
    assert rx.table_empty_groups(str(tab), DSR) == list(BOTH)
    with pytest.raises(ValueError, match="target"):
        rx.table_empty_groups(str(tab), "rain_chirps")
    assert "empty_test_groups_declared" in rx.KEY_FIELDS and rx.META_DEFAULTS["empty_test_groups_declared"] == []


def test_runner_bang_dot7_thieu_provenance_dung_som(tmp_path):
    for sub in ("grids", "tables", "folds"):
        (tmp_path / sub).mkdir()
    (tmp_path / "grids" / "h3_res_5.geojson").write_text("{}")
    (tmp_path / "tables" / "h3_res_5_unified.csv").write_text("cell_id\n")
    for f in ("blocks.geojson", "points.geojson"):
        (tmp_path / f).write_text("{}")
    (tmp_path / "ref.csv").write_text("point_id\n")
    (tmp_path / "ref.csv.provenance.json").write_text(json.dumps({"target": DSR, "variant": "main"}))
    cmd = [sys.executable, RUNNER, "--prefix", "t", "--grids", "h3_res_5", "--models", "hist_gb", "--schemes", "42",
           "--target", DSR, "--tables-dir", str(tmp_path / "tables"), "--grids-dir", str(tmp_path / "grids"),
           "--folds-dir", str(tmp_path / "folds"), "--blocks", str(tmp_path / "blocks.geojson"),
           "--points", str(tmp_path / "points.geojson"), "--points-ref", str(tmp_path / "ref.csv"),
           "--cwd", str(tmp_path), "--allow-untagged", "--allow-dirty"]
    p = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert p.returncode != 0 and "thieu provenance" in p.stdout + p.stderr, p.stdout + p.stderr
    assert not (tmp_path / "artifacts").exists()


# ------------------------------------------------------------------ dau-cuoi tren luoi that h3_res_5
@pytest.fixture()
def dsr_setup(tmp_path):
    from static_features import WORLDCOVER_NAMES
    from training.features import DEFAULT_ALLOWED_FEATURES

    rng = np.random.default_rng(7)
    g = gpd.read_file(GRID)
    cells = g["cell_id"].astype(str).tolist()
    t = pd.DataFrame([(c, s) for c in cells for s in range(2018, 2022)], columns=["cell_id", "season"])
    t["dem_mean"] = rng.uniform(0, 3, len(t))
    for f in DEFAULT_ALLOWED_FEATURES:
        names = [f"landcover_class_{n}" for n in WORLDCOVER_NAMES.values()] if f.endswith("*") else [f]
        for n in names:
            if n not in t.columns:
                t[n] = rng.uniform(0, 1, len(t))
    t[DSR] = 200 + 10 * t["dem_mean"] + rng.normal(0, 1, len(t))
    t.loc[t["season"] == 2020, DSR] = np.nan
    t["train_ok_scope"] = t[DSR].notna()
    gu = g.to_crs(32648)
    t = t.merge(pd.DataFrame({"cell_id": cells, "scope_cx": gu.geometry.centroid.x,
                              "scope_cy": gu.geometry.centroid.y}), on="cell_id")
    (tmp_path / "tables").mkdir()
    tab = tmp_path / "tables" / "h3_res_5_unified.csv"
    t.to_csv(tab, index=False)
    (tmp_path / "tables" / "h3_res_5_unified.csv.provenance.json").write_text(
        json.dumps({"target": DSR, "label_set": "chinh", EMPTY_DECL_KEY: DECL}), encoding="utf-8")
    (tmp_path / "grids").mkdir()
    shutil.copy(GRID, tmp_path / "grids" / "h3_res_5.geojson")
    (tmp_path / "folds").mkdir()
    shutil.copy(FOLDS, tmp_path / "folds" / "cv_folds_s42.csv")
    pts = gpd.read_file(POINTS)
    ref = pd.DataFrame([(p, s) for p in pts["point_id"] for s in range(2018, 2022)], columns=["point_id", "season"])
    ref["n_valid_3x3"] = np.where(ref["season"] == 2020, 0, 9)
    ref[f"ref_{DSR}"] = np.where(ref["season"] == 2020, np.nan, rng.uniform(180, 240, len(ref)))
    ref.to_csv(tmp_path / "ref.csv", index=False)
    (tmp_path / "ref.csv.provenance.json").write_text(json.dumps({"target": DSR, "variant": "main"}))
    return tmp_path


def _runner(tmp, *extra):
    cmd = [sys.executable, RUNNER, "--prefix", "t", "--grids", "h3_res_5", "--models", "linear", "--schemes", "42",
           "--target", DSR, "--tables-dir", str(tmp / "tables"), "--grids-dir", str(tmp / "grids"),
           "--folds-dir", str(tmp / "folds"), "--points-ref", str(tmp / "ref.csv"), "--cwd", str(tmp),
           "--allow-untagged", "--allow-dirty", *extra]
    return subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=900,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))


@need_real
def test_dau_cuoi_khai_bao_cv_final_va_bo_khai_bao(dsr_setup):
    tmp = dsr_setup
    out = tmp / "artifacts" / "experiments" / "t__h3_res_5__linear__s42__t-dsr_mcd18"
    for mode in ("cv", "final"):
        p = _runner(tmp, "--mode", mode)
        assert p.returncode == 0, p.stdout + p.stderr + (out / mode / "train_stdout.log").read_text(encoding="utf-8")
        meta = json.loads((out / mode / "run_meta.json").read_text(encoding="utf-8"))
        i = meta["cmd"].index("--allow-empty-test-groups")
        assert meta["cmd"][i + 1:i + 3] == list(BOTH) and meta["empty_test_groups_declared"] == list(BOTH)
        cfg = json.loads((out / mode / "config.json").read_text(encoding="utf-8"))
        assert cfg["eval_split"]["empty_test_groups_declared"] == list(BOTH)
        assert cfg["eval_split"]["empty_test_groups_source"]["reason"] == DECL["reason"]
    man = pd.read_csv(tmp / "artifacts" / "experiments" / "t_manifest.csv", dtype=str)
    assert set(man["empty_test_groups_declared"]) == {"thoi_gian ca_hai"}
    # final: nhom rong ghi ro, khong chia 0 / NaN; nhom khong gian van co chi so
    m = cfg["test_metrics_by_group_mean_over_seeds"]
    for g in BOTH:
        assert m[g]["status"] == EMPTY_LABEL and m[g]["n_seeds"] == 0 and m[g]["mae"] is None
        assert cfg["n_test_rows_by_group"][g] == 0 and cfg["test_metrics_by_group"][g]["42"] is None
        assert cfg["point_metrics_by_group"][g] == {} and cfg["n_point_rows_by_group"][g] == 0
    assert np.isfinite(m["khong_gian"]["mae"]) and "status" not in m["khong_gian"]
    assert EMPTY_LABEL in cfg["notes"]["nhom_test_rong"]
    fp = pd.read_csv(out / "final" / "final_predictions.csv")
    assert set(fp["test_group"]) == {"khong_gian"} and np.isfinite(fp["y_pred"]).all()
    # bo khai bao -> khoa doi -> chay lai -> LOI nhu cu
    prov = tmp / "tables" / "h3_res_5_unified.csv.provenance.json"
    prov.write_text(json.dumps({"target": DSR, "label_set": "chinh"}), encoding="utf-8")
    p = _runner(tmp, "--mode", "cv")
    assert p.returncode != 0
    log = (out / "cv" / "train_stdout.log").read_text(encoding="utf-8")
    assert "Nhom test rong: ['thoi_gian', 'ca_hai']" in log
    assert "--allow-empty-test-groups" not in json.loads((out / "cv" / "run_meta.json").read_text())["cmd"]


def _train(tmp, *extra):
    cmd = [sys.executable, TRAIN, "--table", str(tmp / "tables" / "h3_res_5_unified.csv"), "--mode", "cv",
           "--grid", GRID, "--blocks", BLOCKS, "--cv-folds", str(tmp / "folds" / "cv_folds_s42.csv"),
           "--model", "linear", "--target", DSR, "--no-point-eval", "--experiment-name", "x", *extra]
    return subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))


@need_real
def test_train_co_khong_khop_khai_bao_la_loi(dsr_setup):
    tmp = dsr_setup
    prov = tmp / "tables" / "h3_res_5_unified.csv.provenance.json"
    p = _train(tmp, "--allow-empty-test-groups", "thoi_gian")                      # thieu ca_hai
    assert p.returncode == 2 and "khac khai bao" in p.stderr, p.stderr
    p = _train(tmp)                                                                 # co khai bao ma khong truyen co
    assert p.returncode == 2 and "Nhom test rong" in p.stderr, p.stderr
    prov.write_text(json.dumps({"target": DSR}), encoding="utf-8")                  # truyen co ma khong khai bao
    p = _train(tmp, "--allow-empty-test-groups", *BOTH)
    assert p.returncode == 2 and "khac khai bao" in p.stderr, p.stderr
    prov.unlink()                                                                   # khong co provenance
    p = _train(tmp, "--allow-empty-test-groups", *BOTH)
    assert p.returncode == 2 and "khong co provenance" in p.stderr, p.stderr

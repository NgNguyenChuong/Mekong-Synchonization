"""Chay src/training/train.py dau-cuoi tren bo dac trung TONG HOP co gia tri thieu."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def feature_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("processed")
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="7D")
    rows = []
    for c in range(20):
        for t in dates:
            rain = rng.gamma(2.0, 5.0)
            rows.append({"cell_id": f"sq_{c:03d}", "date": t.strftime("%Y-%m-%d"),
                         "rain_mm": np.nan if rng.random() < 0.1 else rain,
                         "temp_c": 25 + 0.1 * c + rng.normal(0, 0.5),
                         "salinity": 0.3 * rain + 0.05 * c + rng.normal(0, 0.1)})
    pd.DataFrame(rows).to_csv(d / "DYNAMIC_MERGE.csv", index=False)
    return d


@pytest.mark.parametrize("model", ["linear", "hist_gb", "persistence"])
def test_train_runs_with_missing_values(feature_dir, tmp_path, model):
    env = dict(os.environ, OUTPUT_DIR=str(feature_dir), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, os.path.join(ROOT, "src", "training", "train.py"), "--model", model,
           "--target", "salinity", "--train-end", "2020-08-31", "--val-end", "2020-10-31",
           "--experiment-name", f"t_{model}"]
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    out = tmp_path / "artifacts" / "experiments" / f"t_{model}"
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert np.isfinite(cfg["test_metrics"]["mae"])
    has_handler = (out / "missing_handler.joblib").exists()
    assert has_handler == (model == "linear")
    assert cfg["features"] == ["rain_mm", "temp_c"]


@pytest.fixture(scope="module")
def season_data(tmp_path_factory):
    """Dac trung ngay du mua 2017-2021 cho 12 o + STATIC co cot CAM + nhan (o, mua)."""
    d = tmp_path_factory.mktemp("season_processed")
    raw = tmp_path_factory.mktemp("season_raw")  # RAW_DIR rong -> DATA_SPECS/PERIODIC mac dinh
    lab_dir = tmp_path_factory.mktemp("season_labels")
    rng = np.random.default_rng(1)
    dates = pd.date_range("2016-10-01", "2021-05-31", freq="D")
    cells = [f"sq_{c:03d}" for c in range(12)]
    dyn = pd.DataFrame([(c, t.strftime("%Y-%m-%d")) for c in cells for t in dates], columns=["cell_id", "date"])
    dyn["rain_mm"] = rng.gamma(1.0, 2.0, len(dyn))
    dyn["temp_c"] = 27 + rng.normal(0, 1, len(dyn))
    dyn.to_csv(d / "DYNAMIC_MERGE.csv", index=False)
    # Q5 soat 2026-10-03: cot CAM dat trong STATIC_MERGED (vao bang dac trung), khong chi trong CSV
    # nhan (bi bo truoc buoc chon) -> test that su kiem bo chon dac trung.
    pd.DataFrame({"cell_id": cells, "dem_mean": np.linspace(0.5, 2.0, len(cells)),
                  "landcover_class_Cropland": np.linspace(0.9, 0.2, len(cells)),
                  "landcover_class_Water": np.linspace(0.0, 0.3, len(cells)),
                  "n_land_px": np.arange(len(cells)) * 10 + 5,
                  "water_freq": np.linspace(0.0, 60.0, len(cells)),
                  "overlap_frac": np.linspace(0.3, 1.0, len(cells)),
                  "is_holdout": [i % 4 == 0 for i in range(len(cells))]}).to_csv(
        d / "STATIC_MERGED.csv", index=False)
    lab = pd.DataFrame([(c, s) for c in cells for s in range(2017, 2022)], columns=["cell_id", "season"])
    lab["salinity"] = rng.uniform(0.5, 4.0, len(lab))
    lab["NDWIchen"] = 0.1
    lab_csv = lab_dir / "labels.csv"
    lab.to_csv(lab_csv, index=False)
    env = dict(os.environ, OUTPUT_DIR=str(d), RAW_DIR=str(raw), PYTHONIOENCODING="utf-8")
    return env, lab_csv


def _run_season(season_data, cwd, name, *extra):
    env, lab_csv = season_data
    cmd = [sys.executable, os.path.join(ROOT, "src", "training", "train.py"), "--model", "linear",
           "--label-csv", str(lab_csv), "--train-end", "2019", "--val-end", "2020",
           "--experiment-name", name, *extra]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def test_train_label_csv_ghep_theo_mua(season_data, tmp_path):
    """Luong chinh co --label-csv: dac trung ngay -> gop mua -> ghep chinh xac (cell_id, season)."""
    proc = _run_season(season_data, tmp_path, "t_season", "--provisional-split")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # 12 o x 5 mua = 60 dong (khong nhan ban theo ngay); train 2017-2019, val 2020, test 2021
    assert "60 dong co nhan" in proc.stdout
    assert "'n_train': 36, 'n_val': 12, 'n_test': 12" in proc.stdout
    cfg = json.loads((tmp_path / "artifacts" / "experiments" / "t_season" / "config.json").read_text(encoding="utf-8"))
    # Cot cam trong STATIC (n_land_px, water_freq, overlap_frac, is_holdout) va cot chat luong
    # rain_mm_n_days/temp_c_n_days khong vao dac trung.
    # Tien to landcover_class_* da giai thanh ten day du (ghi vao config de so giua luoi).
    assert cfg["features"] == ["rain_mm", "temp_c", "dem_mean", "landcover_class_Cropland", "landcover_class_Water"]
    assert cfg["features_source"] == "default"
    assert "solar" in cfg["features_missing"] and "landcover_class_*" not in cfg["features_missing"]
    # Muc 1: chia tam -> danh dau ro, KHONG co khoa test_metrics
    assert cfg["split_kind"] == "season_sequential_PROVISIONAL"
    assert "test_metrics" not in cfg and np.isfinite(cfg["provisional_test_metrics"]["mae"])


def test_bang_mua_thieu_co_provisional_split_thoat_ma_2(season_data, tmp_path):
    # Muc 1 (CHAN) soat 2026-10-03: truoc day bang mua mac dinh chia tuan tu va ghi test_metrics.
    proc = _run_season(season_data, tmp_path, "t_noflag")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "assign_eval_split/block_cv_splits" in proc.stderr
    assert not (tmp_path / "artifacts" / "experiments" / "t_noflag" / "config.json").exists()


@pytest.mark.parametrize("features, msg", [
    (["dem_mean", "n_land_px"], "n_land_px"),        # cot cam yeu cau tuong minh
    (["dem_mean", "overlap_frac"], "overlap_frac"),  # cot chat luong (muc 5b)
    (["rain_mm", "solar"], "solar"),                 # yeu cau tuong minh ma thieu (muc 2)
    (["*"], "khong hop le"),                         # '*' don le (muc 5a)
])
def test_features_tuong_minh_sai_bao_loi(season_data, tmp_path, features, msg):
    proc = _run_season(season_data, tmp_path, "t_bad", "--provisional-split", "--features", *features)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert msg in proc.stderr


def test_features_tuong_minh_tien_to_ghi_ten_da_giai(season_data, tmp_path):
    proc = _run_season(season_data, tmp_path, "t_ok", "--provisional-split",
                       "--features", "dem_mean", "rain_mm", "landcover_class_*")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cfg = json.loads((tmp_path / "artifacts" / "experiments" / "t_ok" / "config.json").read_text(encoding="utf-8"))
    assert cfg["features"] == ["rain_mm", "dem_mean", "landcover_class_Cropland", "landcover_class_Water"]
    assert cfg["features_requested"] == ["dem_mean", "rain_mm", "landcover_class_*"]
    assert cfg["features_source"] == "cli"
    assert cfg["features_missing"] == []

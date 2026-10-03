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

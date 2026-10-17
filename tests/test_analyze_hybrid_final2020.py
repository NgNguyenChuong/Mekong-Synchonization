import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_hybrid_final2020 as m  # noqa: E402


def _write_run(root, grid, fs, shift, rng):
    d = m.run_dir(root, grid, fs, "ndwi")
    os.makedirs(d)
    n = 200
    pts = pd.DataFrame({"point_id": [f"p{i}" for i in range(n)], "season": 2020,
                        "block_id": [f"b{i % 16}" for i in range(n)], "test_group": "thoi_gian",  # 16 khoi: p min 2/65536 qua duoc Holm 13
                        "err": rng.normal(0, 1, n) + shift})
    pd.concat([pts, pts.assign(test_group="khong_gian", season=2019)]).to_csv(os.path.join(d, "final_points.csv"), index=False)
    json.dump({"returncode": 0, "mode": "final", "git_tag": "t1"}, open(os.path.join(d, "run_meta.json"), "w"))


def test_phan_ra_va_dau(tmp_path):
    rng = np.random.default_rng(0)
    for g in m.GRIDS:
        _write_run(tmp_path, g, None, 0.0, rng)          # (a) tot nhat
        _write_run(tmp_path, g, "khong_diem", 2.0, rng)  # (b) te hon
        _write_run(tmp_path, g, "b_mua", 0.5, rng)       # (c) gan (a)
    out, tags = m.analyze("ndwi", str(tmp_path), ["t1"])
    assert tags == ["t1"] and len(out) == 39 and out["G"].eq(16).all()
    i = out.set_index(["thanh_phan", "grid"])["I"]
    assert (i["I"] > 0).all() and (i["I_mua"] > 0).all()
    assert np.allclose(i["I"].values, (i["I_mua"] + i["I_kg"]).values)
    assert out[out["thanh_phan"] == "I"]["co_y_nghia"].all()


def test_tag_ngoai_danh_sach_loi(tmp_path):
    rng = np.random.default_rng(1)
    for g in m.GRIDS:
        for fs, s in ((None, 0), ("khong_diem", 1), ("b_mua", 0.5)):
            _write_run(tmp_path, g, fs, s, rng)
    with pytest.raises(SystemExit):
        m.analyze("ndwi", str(tmp_path), ["khac"])

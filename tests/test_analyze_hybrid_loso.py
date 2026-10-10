import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_hybrid_loso as m  # noqa: E402


def _write_run(root, fs, target, season, shift, tag="t1"):
    d = m.run_dir(root, fs, target, season)
    os.makedirs(d)
    n = 60
    base = 1.0 + 0.1 * np.arange(n) / n
    pts = pd.DataFrame({"point_id": [f"p{i}" for i in range(n)], "season": season,
                        "block_id": [f"b{i % 6}" for i in range(n)], "test_group": "thoi_gian", "err": base + shift})
    pd.concat([pts, pts.assign(test_group="khong_gian", season=season - 1)]).to_csv(
        os.path.join(d, "final_points.csv"), index=False)
    json.dump({"returncode": 0, "mode": "final", "git_tag": tag}, open(os.path.join(d, "run_meta.json"), "w"))


def _fake(root, target, signs):
    for s, sg in zip(m.seasons_of(target), signs):
        _write_run(root, None, target, s, 0.0)
        _write_run(root, "khong_diem", target, s, 0.3 * sg)   # sg > 0: (b) te hon -> I_s > 0


def test_I_theo_mua_va_bo_2020_buc_xa(tmp_path):
    _fake(tmp_path, "ndwi", [1] * 11 + [-1] * 2)
    out, tags = m.analyze("ndwi", str(tmp_path), ["t1"])
    assert tags == ["t1"] and list(out["season"]) == list(range(2014, 2027)) and out["G"].eq(6).all()
    assert (out["I"] > 0).sum() == 11 and np.allclose(out["I"].abs(), 0.3)
    assert np.allclose(out["I_tuong_doi"], out["I"] / out["mae_b"])
    assert ((out["ci90_low"] <= out["I"] + 1e-12) & (out["I"] - 1e-12 <= out["ci90_high"])).all()
    _fake(tmp_path, "dsr_mcd18", [-1] * 12)                   # khong co thu muc 2020 van chay
    out, _ = m.analyze("dsr_mcd18", str(tmp_path), None)
    assert len(out) == 12 and 2020 not in set(out["season"])
    with pytest.raises(SystemExit):
        m.analyze("ndwi", str(tmp_path), ["khac"])            # tag ngoai danh sach
    with pytest.raises(FileNotFoundError):
        m.analyze("rain_chirps", str(tmp_path), None)         # thieu luot -> loi, khong bo qua


def test_dem_dau_nhan_va_holm():
    per = {"ndwi": pd.DataFrame({"I": [1.0] * 11 + [-1.0] * 2}),        # 11/13 duong -> co_ich
           "dsr_mcd18": pd.DataFrame({"I": [-1.0] * 9 + [1.0] * 3}),    # 9/12 am -> co_hai
           "rain_chirps": pd.DataFrame({"I": [1.0] * 9 + [-1.0] * 4})}  # 9/13 -> khong_on_dinh
    s = m.summarize(per).set_index("target")
    assert s.loc["ndwi", ["n_mua", "n_I_duong", "n_I_am", "nhan"]].tolist() == [13, 11, 2, "co_ich"]
    assert s.loc["dsr_mcd18", ["n_mua", "n_I_am", "nguong", "huong", "nhan"]].tolist() == [12, 9, 9, "am", "co_hai"]
    assert s.loc["rain_chirps", "nhan"] == "khong_on_dinh"
    assert s.loc["ndwi", "p_mot_phia"] == pytest.approx(92 / 8192) and s.loc["dsr_mcd18", "p_mot_phia"] == pytest.approx(299 / 4096)
    assert (s["p_holm"] >= s["p_mot_phia"]).all() and s.loc["ndwi", "p_holm"] == pytest.approx(3 * 92 / 8192)
    with pytest.raises(SystemExit):
        m.summarize({"ndwi": pd.DataFrame({"I": [1.0] * 11})})


def test_file_dong_bang_ma_2(tmp_path, monkeypatch):
    res = tmp_path / "KE_HOACH" / "ket-qua"
    res.mkdir(parents=True)
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/dot7_ndwi_hybrid_loso.csv"]}).to_csv(res / "m.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["x", "--target", "ndwi", "--exp-root", str(tmp_path), "--out-dir", str(res),
                                      "--frozen-manifest", str(res / "m.csv")])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert e.value.code == 2

"""E7 (mo ta): analyze_e7_moran.py + analyze_e7_chart.py tren du lieu gia nho; variogram/Moran thay bang ham gia."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mo():
    return _load("analyze_e7_moran")


# ---------------------------------------------------------
# analyze_e7_moran
# ---------------------------------------------------------
def test_tong_hop_trung_vi(mo):
    res = pd.DataFrame({"variable": ["ndwi"] * 3 + ["rain_chirps"] * 3, "season": [2014, 2015, 2016] * 2,
                        "n": [100, 90, 110, 50, 50, 50], "moran_I": [0.1, 0.5, 0.3, 0.9, 0.8, 0.7],
                        "range_km": [10.0, np.inf, 30.0, np.inf, np.inf, 20.0]})
    s = mo.summarize(res, {"ndwi": 120, "rain_chirps": 50}).set_index("variable")
    assert s.loc["ndwi", "moran_median"] == pytest.approx(0.3)
    assert (s.loc["ndwi", "moran_min"], s.loc["ndwi", "moran_max"]) == (0.1, 0.5)
    assert s.loc["ndwi", "range_median_km"] == 30.0 and s.loc["ndwi", "n_range_inf"] == 1
    assert s.loc["ndwi", "n_points_median"] == 100 and s.loc["ndwi", "n_seasons"] == 3
    assert np.isposinf(s.loc["rain_chirps", "range_median_km"])  # 2/3 mua cham bien -> trung vi inf
    assert s.loc["rain_chirps", "do_phan_giai_nguon_km"] == 5.0 and s.loc["ndwi", "n_tap_chung"] == 120


def test_tam_nan_la_loi(mo):
    res = pd.DataFrame({"variable": ["ndwi"], "season": [2014], "n": [5], "moran_I": [0.2], "range_km": [np.nan]})
    with pytest.raises(SystemExit, match="NaN"):
        mo.summarize(res, {"ndwi": 5})


def test_read_done_khu_trung_giong_het_loi_khi_khac(mo, tmp_path):
    p = tmp_path / "r.csv"
    row = {"variable": "ndwi", "season": 2014, "n": 4, "moran_I": 0.5, "range_km": 20.0}
    pd.DataFrame([row, row]).to_csv(p, index=False)
    assert len(mo.read_done(str(p))) == 1
    pd.DataFrame([row, {**row, "moran_I": 0.6}]).to_csv(p, index=False)
    with pytest.raises(SystemExit, match="trung khoa"):
        mo.read_done(str(p))


PTS = ["p1", "p2", "p3", "p4", "p5"]
SEASONS = [2014, 2015]


def _labels(tmp_path):
    """p5 khong co NDWI -> ngoai tap chung (4 diem); buc xa 2015 NaN ca vung; t2m p4 NaN -> 3 diem."""
    paths = {}
    for t in ("salinity", "ndwi", "rain_chirps", "dsr_mcd18", "t2m_era5", "rh_era5"):
        rows = []
        for s in SEASONS:
            for i, p in enumerate(PTS):
                v = 1.0 + i + (s - 2014)
                if (t == "ndwi" and p == "p5") or (t == "dsr_mcd18" and s == 2015) or (t == "t2m_era5" and p == "p4"):
                    v = np.nan
                rows.append({"point_id": p, "season": s, f"ref_{t}": v})
        paths[t] = str(tmp_path / f"points_reference_{t}.csv")
        pd.DataFrame(rows).to_csv(paths[t], index=False)
    return paths


@pytest.fixture
def fake_run(mo, tmp_path, monkeypatch):
    paths = _labels(tmp_path)
    calls = []
    monkeypatch.setattr(mo, "label_path", lambda t: paths[t])
    monkeypatch.setattr(mo, "point_xy", lambda: pd.DataFrame(
        {"point_id": PTS, "x": [500_000.0 + 1000 * i for i in range(5)], "y": [1_100_000.0] * 5}))
    monkeypatch.setattr(mo, "git_state", lambda: {"git_commit": "gia"})
    eval_pts = tmp_path / "eval.geojson"
    eval_pts.write_text("{}")
    monkeypatch.setattr(mo, "EVAL_POINTS", str(eval_pts))

    def vg(xy, v):
        calls.append(len(v))
        return {"n": len(v), "range_m": 1000.0 * float(np.mean(v)), "ok": True, "reason": "", "nugget": 0.1,
                "sill": 1.0}

    monkeypatch.setattr(mo.sa, "variogram_range", vg)
    monkeypatch.setattr(mo.sa, "moran_band", lambda xy, v: {"I": float(np.mean(v)) / 10, "p_sim": 0.001,
                                                            "n_islands": 0})
    man = tmp_path / "KE_HOACH" / "ket-qua" / "man.csv"
    man.parent.mkdir(parents=True)
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/dong_bang.csv"]}).to_csv(man, index=False)
    out = tmp_path / "out"
    args = type("A", (), {"out_dir": str(out), "expected_common": 4, "frozen_manifest": [str(man)]})
    return args, out, calls, man


def test_chay_tiep_bo_qua_da_co(mo, fake_run):
    args, out, calls, _ = fake_run
    mo.main(args)
    res = pd.read_csv(out / mo.SEASON_FILE)
    assert len(res) == 11 and len(calls) == 11  # dsr 1 mua; 5 bien x 2 mua
    assert set(res.loc[res["variable"] == "t2m_era5", "n"]) == {3}
    assert set(res.loc[res["variable"] == "salinity", "n"]) == {4}
    s = pd.read_csv(out / mo.SUMMARY_FILE).set_index("variable")
    assert s.loc["dsr_mcd18", "n_seasons"] == 1 and s.loc["t2m_era5", "n_tap_chung"] == 3
    assert (out / f"{mo.SUMMARY_FILE}.provenance.json").exists()
    calls.clear()
    mo.main(args)  # chay lai: moi (bien, mua) da co -> khong tinh lai
    assert calls == [] and len(pd.read_csv(out / mo.SEASON_FILE)) == 11


def test_dong_cu_n_lech_la_loi(mo, fake_run):
    args, out, calls, _ = fake_run
    out.mkdir()
    pd.DataFrame([{"variable": "ndwi", "season": 2014, "n": 99, "moran_I": 0.1, "n_islands": 0, "range_km": 1.0,
                   "ok": True, "reason": "", "nugget": 0, "sill": 1}]).to_csv(out / mo.SEASON_FILE, index=False)
    with pytest.raises(SystemExit, match="n lech"):
        mo.main(args)
    assert calls == []


def test_tap_chung_khac_ky_vong_la_loi(mo, fake_run):
    args, _, calls, _ = fake_run
    args.expected_common = 10424
    with pytest.raises(SystemExit, match="tap diem chung 4"):
        mo.main(args)
    assert calls == []


def test_file_ra_trong_manifest_ma_2(mo, fake_run, tmp_path):
    args, _, calls, man = fake_run
    args.out_dir = str(tmp_path / "KE_HOACH" / "ket-qua")
    pd.DataFrame({"file": [f"KE_HOACH/ket-qua/{mo.SUMMARY_FILE}"]}).to_csv(man, index=False)
    with pytest.raises(SystemExit) as e:
        mo.main(args)
    assert e.value.code == 2 and calls == []

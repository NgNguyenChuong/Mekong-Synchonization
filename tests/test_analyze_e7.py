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


# ---------------------------------------------------------
# analyze_e7_chart
# ---------------------------------------------------------
@pytest.fixture(scope="module")
def ch():
    return _load("analyze_e7_chart")


TIER = {"h3_res_5": 5, "h3_res_6": 6, "h3_res_7": 7, "s2_level_9": 5, "s2_level_10": 6, "s2_level_11": 6,
        "s2_level_12": 7, "square_utm_17087m": 5, "square_utm_6458m": 6, "square_utm_2441m": 7,
        "latlon_0.1552deg": 5, "latlon_0.0586deg": 6, "latlon_0.0222deg": 7}
AREA = {5: 292.0, 6: 41.7, 7: 6.0}


def _area():
    return pd.DataFrame({"tier_h3_res": list(TIER.values()), "mean_area_km2": [AREA[t] for t in TIER.values()],
                         "file": [f"data/grids/{g}.geojson" for g in TIER]})


def _grids(ch, tmp_path):
    p = tmp_path / "area.csv"
    _area().to_csv(p, index=False)
    return ch.grid_table(str(p))


def _kd(target, gate=None, extra=()):
    """Cap F1: muc 5 tl 0,5 / 1,5; muc 6 tl 0,2; muc 7 tl 0,4 / -0,9 (|.|); + ho phu tl 9 (phai bi bo qua)."""
    pairs = [(5, "h3_res_5", "square_utm_17087m", 0.5), (5, "h3_res_5", "latlon_0.1552deg", 1.5),
             (6, "h3_res_6", "square_utm_6458m", 0.2), (7, "h3_res_7", "s2_level_12", 0.4),
             (7, "s2_level_12", "latlon_0.0222deg", -0.9)]
    rows = [{"family": "F1_chinh", "grid_a": a, "grid_b": b, "m": m, "delta_hat": r * 2.0, "delta_min_thr": 2.0}
            for m, a, b, r in pairs]
    rows.append({"family": "phu_idw", "grid_a": "h3_res_7", "grid_b": "s2_level_12", "m": 7, "delta_hat": 18.0,
                 "delta_min_thr": 2.0})
    rows += list(extra)
    d = pd.DataFrame(rows)
    if target == "salinity":
        return d.rename(columns={"m": "delta_ref_level"})
    d = d.rename(columns={"m": "muc"})
    gate = gate or {5: (True, True), 6: (True, True), 7: (True, True)}
    d["muc_kiem_dinh"] = [gate[m][0] for m in d["muc"]]
    d["cong_hoc_duoc"] = [gate[m][1] if gate[m][0] else np.nan for m in d["muc"]]
    d["kiem_dinh"] = d["muc_kiem_dinh"] & d["cong_hoc_duoc"].fillna(False).astype(bool)
    return d


def test_anh_huong_khung_max_tren_f1_dung_muc(ch, tmp_path):
    e = ch.framework_effect(_kd("ndwi"), "ndwi", _grids(ch, tmp_path)).set_index("muc")
    assert e.loc[5, "anh_huong_khung"] == pytest.approx(1.5) and e.loc[5, "n_cap"] == 2
    assert e.loc[6, "anh_huong_khung"] == pytest.approx(0.2)
    assert e.loc[7, "anh_huong_khung"] == pytest.approx(0.9)  # |delta| am; ho phu tl 9 khong tinh
    s = ch.framework_effect(_kd("salinity"), "salinity", _grids(ch, tmp_path))  # muc qua delta_ref_level
    assert s["anh_huong_khung"].round(6).tolist() == [1.5, 0.2, 0.9] and s["kiem_dinh"].all()


def test_kiem_dinh_theo_cong(ch, tmp_path):
    g = _grids(ch, tmp_path)
    e = ch.framework_effect(_kd("dsr_mcd18", {5: (True, False), 6: (True, True), 7: (False, None)}), "dsr_mcd18", g)
    assert e.set_index("muc")["kiem_dinh"].to_dict() == {5: False, 6: True, 7: False}
    bad = _kd("ndwi")
    bad.loc[0, "kiem_dinh"] = False  # cot kiem_dinh khac muc_kiem_dinh & cong -> sai file
    with pytest.raises(SystemExit, match="kiem_dinh"):
        ch.framework_effect(bad, "ndwi", g)


def test_cap_lech_muc_la_loi(ch, tmp_path):
    extra = [{"family": "F1_chinh", "grid_a": "h3_res_5", "grid_b": "h3_res_6", "m": 5, "delta_hat": 0.1,
              "delta_min_thr": 1.0}]
    with pytest.raises(SystemExit, match="lech muc"):
        ch.framework_effect(_kd("salinity", extra=extra), "salinity", _grids(ch, tmp_path))


def _chart_inputs(ch, tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    _area().to_csv(res / ch.AREA_FILE, index=False)
    for t in ch.VARIABLES:
        _kd(t).to_csv(ch.kd_path(str(res), t), index=False)
    out = tmp_path / "KE_HOACH" / "ket-qua"
    out.mkdir(parents=True)
    pd.DataFrame({"variable": ch.VARIABLES, "moran_median": [0.3, 0.5, 0.97, 0.9, 0.99, 0.98],
                  "range_median_km": [np.inf, 40.0, np.inf, 60.0, np.inf, np.inf],
                  "do_phan_giai_nguon_km": [0.03, 0.03, 5.0, 1.0, 9.0, 9.0]}).to_csv(
        out / ch.SUMMARY_FILE, index=False)
    man = out / "man.csv"
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/khac.csv"]}).to_csv(man, index=False)
    args = type("A", (), {"out_dir": str(out), "results_dir": str(res), "frozen_manifest": [str(man)]})
    return args, out, man


def test_chart_dau_cuoi(ch, tmp_path):
    args, out, _ = _chart_inputs(ch, tmp_path)
    ch.main(args)
    tab = pd.read_csv(out / ch.MAIN_FILE)
    assert len(tab) == 18 and set(tab["n_cap"]) == {1, 2}
    tam = pd.read_csv(out / ch.TAM_FILE)
    assert len(tam) == 78
    r = tam[(tam["variable"] == "ndwi") & (tam["grid"] == "h3_res_6")].iloc[0]
    assert r["ti_le_tam_tren_canh"] == pytest.approx(40.0 / np.sqrt(41.7))
    assert np.isposinf(tam.loc[tam["variable"] == "salinity", "ti_le_tam_tren_canh"]).all()
    assert r["nguon_tren_canh"] == pytest.approx(0.03 / np.sqrt(41.7))
    m = tab.set_index(["variable", "muc"])["nguon_tren_canh_median"]
    assert m[("t2m_era5", 6)] == pytest.approx(9.0 / np.sqrt(41.7))
    assert m[("rain_chirps", 7)] == pytest.approx(5.0 / np.sqrt(6.0))
    assert m[("dsr_mcd18", 5)] == pytest.approx(1.0 / np.sqrt(292.0)) and np.isfinite(m).all()
    for p in ch.outputs(str(out)):
        assert os.path.exists(p) and os.path.exists(f"{p}.provenance.json")
    assert (out / "hinh" / f"{ch.FIG}_nguon_canh_o.png").exists()
    assert not any(f"{ch.FIG}_tam_canh_o" in p for p in ch.outputs(str(out)))


@pytest.mark.parametrize("bad", [0.0, np.nan])
def test_chart_do_phan_giai_nguon_sai_la_loi(ch, tmp_path, bad):
    args, out, _ = _chart_inputs(ch, tmp_path)
    s = pd.read_csv(out / ch.SUMMARY_FILE)
    s.loc[s["variable"] == "rain_chirps", "do_phan_giai_nguon_km"] = bad
    s.to_csv(out / ch.SUMMARY_FILE, index=False)
    with pytest.raises(SystemExit, match="do_phan_giai_nguon_km"):
        ch.main(args)
    assert not (out / ch.MAIN_FILE).exists()


def test_chart_file_ra_trong_manifest_ma_2(ch, tmp_path):
    args, out, man = _chart_inputs(ch, tmp_path)
    pd.DataFrame({"file": [f"KE_HOACH/ket-qua/hinh/{ch.FIG}_bo_ndwi.svg"]}).to_csv(man, index=False)
    with pytest.raises(SystemExit) as e:
        ch.main(args)
    assert e.value.code == 2 and not (out / ch.MAIN_FILE).exists() and not (out / "hinh").exists()

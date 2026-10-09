"""plot_dot7_figures.py tren du lieu gia: doc/ghep dung so dong, file/cot thieu -> bo + ghi chu, guard manifest."""
import importlib.util
import json
import os

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIER = {"h3_res_5": 5, "h3_res_6": 6, "h3_res_7": 7, "s2_level_9": 5, "s2_level_10": 6, "s2_level_11": 6,
        "s2_level_12": 7, "square_utm_17087m": 5, "square_utm_6458m": 6, "square_utm_2441m": 7,
        "latlon_0.1552deg": 5, "latlon_0.0586deg": 6, "latlon_0.0222deg": 7}
F1_PAIRS = [(5, "h3_res_5", "square_utm_17087m"), (5, "h3_res_5", "latlon_0.1552deg"),
            (5, "square_utm_17087m", "latlon_0.1552deg"), (6, "h3_res_6", "square_utm_6458m"),
            (6, "h3_res_6", "latlon_0.0586deg"), (6, "square_utm_6458m", "latlon_0.0586deg"),
            (7, "h3_res_7", "s2_level_12"), (7, "h3_res_7", "square_utm_2441m"), (7, "h3_res_7", "latlon_0.0222deg"),
            (7, "s2_level_12", "square_utm_2441m"), (7, "s2_level_12", "latlon_0.0222deg"),
            (7, "square_utm_2441m", "latlon_0.0222deg")]
REF = {5: 2.0, 6: 1.0, 7: 0.5}


@pytest.fixture(scope="module")
def pf():
    spec = importlib.util.spec_from_file_location("plot_dot7_figures",
                                                  os.path.join(ROOT, "scripts", "plot_dot7_figures.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _st():
    return {"notes": [], "used": []}


def _levels():
    return pd.Series(TIER, name="muc")


def _hoc(pf, model="hist_gb", fail_level=None):
    rows = [{"grid_a": g, "grid_b": g, "model_a": pf.run_model(model), "model_b": "season_mean", "muc": m,
             "delta_hat": -0.5, "kiem_dinh": True, "hoc_duoc": m != fail_level} for g, m in TIER.items()]
    return pd.DataFrame(rows)


def _f1(pf, model="hist_gb"):
    rows = [{"family": pf.main_family(model), "grid_a": a, "grid_b": b, "delta_ref_level": m, "delta_ref": REF[m],
             "delta_hat": 0.1, "ci_low": -0.1, "ci_high": 0.3, "delta_min_thr": 0.2, "p_holm": 1.0,
             "label": "tuong_duong"} for m, a, b in F1_PAIRS]
    rows[1].update(label="cho_giu_rieng", p_holm=0.01)
    rows.append({**rows[0], "family": "phu_idw", "delta_ref": 99.0, "delta_hat": 9.0})  # ho phu: phai bi bo qua
    return pd.DataFrame(rows)


def _gate(levels_ok=(5, 6, 7), bien="ndwi"):
    return pd.DataFrame([{"bien": bien, "muc": m, "hop_le": m in levels_ok,
                          "ly_do": "moi_luoi_hoc_duoc" if m in levels_ok else "co_luoi_khong_hoc_duoc"}
                         for m in (5, 6, 7)])


def _area():
    return pd.DataFrame({"file": [f"data/grids/{g}.geojson" for g in TIER], "tier_h3_res": list(TIER.values()),
                         "mean_area_km2": [{5: 292.0, 6: 41.7, 7: 6.0}[m] for m in TIER.values()]})


def test_learn_table_ghep_dung_so_dong_va_ti_le(pf, tmp_path):
    _hoc(pf).to_csv(tmp_path / "dot7_ndwi_hoc_duoc.csv", index=False)
    _f1(pf).to_csv(tmp_path / "dot7_ndwi_cv1_kiem_dinh_khoi.csv", index=False)
    pd.DataFrame({"grid_a": list(TIER), "delta_hat": 0.0}).to_csv(
        tmp_path / "dot5_cv1_kiem_dinh_vs_baseline_rong.csv", index=False)  # khong co hoc_duoc -> bo
    st = _st()
    t = pf.learn_table(str(tmp_path), "hist_gb", _levels(), st)
    assert len(t) == 13 and set(t["target"]) == {"ndwi"} and (t["trang_thai"] == "hoc_duoc").all()
    r = t.set_index("grid")
    assert r.loc["h3_res_5", "y_pct"] == pytest.approx(-25.0) and r.loc["h3_res_7", "y_pct"] == pytest.approx(-100.0)
    assert any("Độ mặn" in n and "thiếu cột" in n for n in st["notes"])
    assert sum("không có" in n for n in st["notes"]) == 4  # mua, buc xa, nhiet, am


def test_learn_table_rf_va_cong_khop(pf, tmp_path):
    _hoc(pf, "rf", fail_level=5).to_csv(tmp_path / "dot7_ndwi_hoc_duoc__rf.csv", index=False)
    _f1(pf, "rf").to_csv(tmp_path / "dot7_ndwi_cv1_kiem_dinh_khoi__rf.csv", index=False)
    st = _st()
    t = pf.learn_table(str(tmp_path), "rf", _levels(), st)
    assert len(t) == 13 and (t.loc[t["muc"] == 5, "trang_thai"] == "khong_hoc_duoc").all()
    _gate(levels_ok=(6, 7)).assign(model="rf").to_csv(tmp_path / "dot7_cong_kiem_dinh__rf.csv", index=False)
    assert pf.gate_fails(str(tmp_path), "rf", t, st) == {5: ["ndwi"], 6: [], 7: []}
    _gate().assign(model="rf").to_csv(tmp_path / "dot7_cong_kiem_dinh__rf.csv", index=False)
    with pytest.raises(SystemExit, match="khong khop"):
        pf.gate_fails(str(tmp_path), "rf", t, st)


def test_learn_table_sai_mo_hinh_la_loi(pf, tmp_path):
    _hoc(pf, "mlp").to_csv(tmp_path / "dot7_ndwi_hoc_duoc__rf.csv", index=False)
    with pytest.raises(SystemExit, match="random_forest"):
        pf.learn_table(str(tmp_path), "rf", _levels(), _st())


def test_f1_table_12_cap_bo_ho_phu(pf, tmp_path):
    _f1(pf).to_csv(tmp_path / "dot5_cv1_kiem_dinh_khoi.csv", index=False)
    st = _st()
    t = pf.f1_table(str(tmp_path), _levels(), st)
    assert len(t) == 12 and set(t["target"]) == {"salinity"}
    assert t["r"].tolist() == pytest.approx([0.5] * 12) and t["r_hi"].iloc[0] == pytest.approx(1.5)
    assert len(st["notes"]) == 5
    _f1(pf).iloc[1:].to_csv(tmp_path / "dot5_cv1_kiem_dinh_khoi.csv", index=False)
    with pytest.raises(SystemExit, match="so cap F1"):
        pf.f1_table(str(tmp_path), _levels(), _st())


def test_k100_do_man_dinh_dang_dot6(pf, tmp_path):
    mae = pd.DataFrame({"grid": list(TIER), "tang_rel_tb": 0.06})
    mae.to_csv(tmp_path / "dot6_khoi100_mae.csv", index=False)  # do man: khong co cot target
    mae.assign(target="ndwi").to_csv(tmp_path / "dot7_ndwi_khoi100_mae.csv", index=False)
    cap = pd.DataFrame([{"grid_a": a, "grid_b": b, "abs_delta_k100_tb": 0.01, "delta_min": 0.02}
                        for m, a, b in F1_PAIRS if m == 7])
    cap.to_csv(tmp_path / "dot6_khoi100_cap_min.csv", index=False)
    cap.drop(columns="delta_min").to_csv(tmp_path / "dot7_ndwi_khoi100_cap_min.csv", index=False)
    st = _st()
    m, c = pf.k100_tables(str(tmp_path), _levels(), st)
    assert len(m) == 26 and m["y_pct"].tolist() == pytest.approx([6.0] * 26)
    assert len(c) == 6 and set(c["target"]) == {"salinity"} and c["ratio"].tolist() == pytest.approx([0.5] * 6)
    assert any("NDWI cặp mịn" in n and "thiếu cột delta_min" in n for n in st["notes"])
    mae.assign(target="rain_chirps").to_csv(tmp_path / "dot7_ndwi_khoi100_mae.csv", index=False)
    with pytest.raises(SystemExit, match="target"):
        pf.k100_tables(str(tmp_path), _levels(), _st())


def test_oracle_chi_cap_min_k50(pf, tmp_path):
    rows = [{"target": t, "blocks": b, "delta_ref_level": m, "grid_a": a, "grid_b": g, "ti_le_delta_tuong_doi": 0.01}
            for t in ("salinity", "ndwi") for b in ("k50", "k100") for m, a, g in F1_PAIRS]
    pd.DataFrame(rows).to_csv(tmp_path / "dot7_oracle_cap.csv", index=False)
    st = _st()
    o = pf.oracle_table(str(tmp_path), _levels(), st)
    assert len(o) == 12 and set(o["blocks"]) == {"k50"} and o["y_pct"].tolist() == pytest.approx([1.0] * 12)
    assert len(st["notes"]) == 4


def _full_inputs(pf, tmp_path):
    res = tmp_path / "KE_HOACH" / "ket-qua"
    res.mkdir(parents=True)
    _area().to_csv(res / pf.AREA_FILE, index=False)
    _hoc(pf).to_csv(res / "dot7_ndwi_hoc_duoc.csv", index=False)
    _f1(pf).to_csv(res / "dot7_ndwi_cv1_kiem_dinh_khoi.csv", index=False)
    _gate().to_csv(res / "dot7_cong_kiem_dinh.csv", index=False)
    pd.DataFrame({"grid": list(TIER), "tang_rel_tb": 0.06}).to_csv(res / "dot6_khoi100_mae.csv", index=False)
    pd.DataFrame([{"pair_index": i, "level": m, "k": k, "kind": kind, "ty_le_phat_hien": 0.5}
                  for i, (m, _, _) in enumerate(F1_PAIRS) for k in (1.0, 2.0) for kind in ("deu", "tau_mu")]).to_csv(
        res / "dot7_pc1_tong_hop.csv", index=False)
    pd.DataFrame([{"target": "ndwi", "blocks": "k50", "delta_ref_level": m, "grid_a": a, "grid_b": b,
                   "ti_le_delta_tuong_doi": 0.02} for m, a, b in F1_PAIRS]).to_csv(res / "dot7_oracle_cap.csv",
                                                                                    index=False)
    man = res / "man.csv"
    pd.DataFrame({"file": ["KE_HOACH/ket-qua/khac.csv"]}).to_csv(man, index=False)
    args = type("A", (), {"results_dir": str(res), "out_dir": str(res), "frozen_manifest": [str(man)]})
    return args, res, man


def test_main_dau_cuoi_ghi_hinh_va_provenance(pf, tmp_path):
    args, res, _ = _full_inputs(pf, tmp_path)
    pf.main(args)
    for p in pf.outputs(str(res)):
        assert os.path.getsize(p) > 0
        with open(f"{p}.provenance.json", encoding="utf-8") as f:
            info = json.load(f)
        assert pf.AREA_FILE in info["sha_dau_vao"] and info["vai_tro"] == "mo_ta"
    with open(os.path.join(res, "hinh", f"{pf.FIGS['h2']}.png.provenance.json"), encoding="utf-8") as f:
        info = json.load(f)
    assert {"dot7_ndwi_hoc_duoc.csv", "dot7_cong_kiem_dinh.csv"} <= set(info["sha_dau_vao"])
    assert any("Random Forest" in n for n in info["phan_bo"])
    with open(os.path.join(res, "hinh", f"{pf.FIGS['h5']}.svg.provenance.json"), encoding="utf-8") as f:
        assert any("PC-1 kết luận" in n for n in json.load(f)["phan_bo"])


def test_guard_chan_file_trong_manifest(pf, tmp_path):
    args, res, man = _full_inputs(pf, tmp_path)
    pd.DataFrame({"file": [f"KE_HOACH/ket-qua/hinh/{pf.FIGS['h3']}.svg"]}).to_csv(man, index=False)
    with pytest.raises(SystemExit) as e:
        pf.main(args)
    assert e.value.code == 2 and not (res / "hinh").exists()

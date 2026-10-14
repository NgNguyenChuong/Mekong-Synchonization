import os
import sys

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import box, mapping
from shapely.ops import transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_grid_phase as m  # noqa: E402
from preprocessing import generate_square_grid  # noqa: E402


def _boundary(tmp_path):
    to_ll = pyproj.Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True).transform
    poly = transform(to_ll, box(500000, 1000000, 560000, 1050000))
    import geopandas as gpd

    p = tmp_path / "b.geojson"
    gpd.GeoDataFrame(geometry=[poly], crs=4326).to_file(p, driver="GeoJSON")
    return str(p)


def test_luoi_dich_cung_so_o_dien_tich_tb(tmp_path):
    b, e = _boundary(tmp_path), 6458
    g0 = generate_square_grid(b, e)
    for kw in ({"shift_x_m": e / 2}, {"shift_y_m": e / 2}):
        g = generate_square_grid(b, e, **kw)
        assert abs(len(g) - len(g0)) <= 0.1 * len(g0) and set(g.cell_id).isdisjoint(g0.cell_id)
        a0, a1 = [x.to_crs(32648).area.mean() for x in (g0, g)]
        assert abs(a1 / a0 - 1) < 0.01
        assert abs(g.overlap_frac.sum() - g0.overlap_frac.sum()) < 1.0  # tong dien tich phu ranh gioi gan nhau
    gx = generate_square_grid(b, e, shift_x_m=e / 2)
    x0 = gx.to_crs(32648).total_bounds[0]
    assert (x0 - e / 2) % e < 1e-6 or abs(((x0 - e / 2) % e) - e) < 1e-6  # goc dich dung 1/2 canh


def _err(rng, shift_sy, n_units=20, per_unit=40):
    pid = [f"p{i}" for i in range(n_units * per_unit)]
    unit = pd.Series([f"u{i // per_unit}" for i in range(len(pid))], index=pid)
    rows = []
    for edge in m.EDGES:
        g = f"square_utm_{edge}m"
        base = {(p, s): rng.normal(0, 1) for p in pid for s in (2019, 2021)}
        for grid, delta in ((g, 0.0), (f"{g}_sx", 0.0), (f"{g}_sy", shift_sy)):
            for sd in (42, 43, 44):
                for (p, s), v in base.items():
                    rows.append((grid, "hist_gb", sd, p, s, v + np.sign(v) * delta, "oof"))
    err = pd.DataFrame(rows, columns=["grid", "model", "seed", "point_id", "season", "err", "pred_source"])
    return err, unit, sorted(unit.unique())


def test_compare_scheme_sx_tuong_duong_sy_khac():
    err, unit, cv = _err(np.random.default_rng(0), shift_sy=0.3)
    out = m.compare_scheme(err, unit, cv, [], "hist_gb", 0.05, 0.05)
    assert len(out) == 6 and out["holm_m"].eq(6).all()
    sx, sy = out[out.variant == "sx"], out[out.variant == "sy"]
    assert sx["label"].eq("tuong_duong").all()
    assert sy["label"].eq("cho_giu_rieng").all() and (sy["delta_hat"] < 0).all()  # sy sai so cao hon goc -> A thap hon -> am
    # delta_min = 5% MAE luoi goc
    assert np.allclose(out["delta_min_thr"], 0.05 * out["delta_ref"])


def test_quy_tac_d_khong_vung():
    err, unit, cv = _err(np.random.default_rng(1), shift_sy=0.3)
    main = m.compare_scheme(err, unit, cv, [], "hist_gb", 0.05, 0.05)
    per = {s: m.compare_scheme(err[err.seed == s], unit, cv, [], "hist_gb", 0.05, 0.05) for s in (42, 43, 44)}
    assert m.apply_rule_d(main, per)["ket_luan_D"].tolist().count("khong_vung_theo_cach_chia") == 0
    per[43] = per[43].assign(label="chua_phan_dinh")  # mot cach chia khong giu -> khong vung
    res = m.apply_rule_d(main, per)
    assert res.loc[res["label"] != "chua_phan_dinh", "ket_luan_D"].eq("khong_vung_theo_cach_chia").all()


def _schemes(seed=1, shift_sy=0.3):
    err, unit, cv = _err(np.random.default_rng(seed), shift_sy=shift_sy)
    main = m.compare_scheme(err, unit, cv, [], "hist_gb", 0.05, 0.05)
    per = {s: m.compare_scheme(err[err.seed == s], unit, cv, [], "hist_gb", 0.05, 0.05) for s in (42, 43, 44)}
    return err, main, per


def test_delta_ref_la_mae_goc_doc_lap():
    err, unit, cv = _err(np.random.default_rng(2), shift_sy=3.0)  # sy sai so lon: doi mau so sang sy/TB se lech
    out = m.compare_scheme(err, unit, cv, [], "hist_gb", 0.05, 0.05)
    for edge in m.EDGES:
        mae = err.loc[err.grid == f"square_utm_{edge}m", "err"].abs().mean()
        assert np.allclose(out.loc[out.edge_m == edge, "delta_ref"], mae)


def test_dau_delta_khac_giua_cach_chia_khong_vung():
    _, main, per = _schemes()
    per[43] = per[43].assign(delta_hat=-per[43]["delta_hat"])  # nhan giu nguyen, dau dao
    res = m.apply_rule_d(main, per)
    assert res.loc[main.variant == "sy", "ket_luan_D"].eq("khong_vung_theo_cach_chia").all()
    assert res.loc[main.variant == "sx", "ket_luan_D"].eq("tuong_duong").all()


def test_p_holm_ge_p_value():
    out = _schemes()[1]
    assert (out["p_holm"] >= out["p_value"]).all() and (out["p_holm"] > out["p_value"]).any()


def test_e_pha_e_khung():
    res = pd.DataFrame({"muc": [5, 5, 6, 6], "delta_hat": [-0.02, 0.04, 0.01, 0.01], "delta_min_thr": [0.1] * 4})
    f1 = pd.DataFrame({"family": ["F1_chinh"] * 3 + ["do_nhay_chi_s42"], "delta_ref_level": [5, 5, 6, 5],
                       "delta_hat": [0.03, -0.05, 0.02, 9.0], "delta_min_thr": [0.1] * 4})
    out = m.add_e_criteria(res, f1)
    e5, e6 = out[out.muc == 5].iloc[0], out[out.muc == 6].iloc[0]
    assert np.isclose(e5.E_pha, 0.4) and np.isclose(e5.E_khung, 0.5) and e5.pha_le_khung
    assert np.isclose(e6.E_pha, 0.1) and np.isclose(e6.E_khung, 0.2)
    f1.loc[2, "delta_hat"] = 0.005  # khung muc 6 nho hon pha
    assert not m.add_e_criteria(res, f1).query("muc == 6")["pha_le_khung"].any()


def test_muc_truot_cong_mo_ta():
    _, main, per = _schemes()
    res = m.apply_rule_d(main, per)
    g = m.apply_gate_pha(res, {5: True, 6: False, 7: True})
    bad, ok = g[g.muc == m.MUC[6458]], g[g.muc != m.MUC[6458]]
    assert bad[["label", "ket_luan_D", "nhan_s42"]].eq("mo_ta").all().all()
    assert bad[["p_value", "p_holm", "tost_tuong_duong"]].isna().all().all()
    assert ok["label"].eq(res.loc[ok.index, "label"]).all() and ok["p_holm"].notna().all()
    assert m.apply_gate_pha(res, None)["label"].eq(res["label"]).all()

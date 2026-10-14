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

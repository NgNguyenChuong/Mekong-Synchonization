#!/usr/bin/env python
"""Boc tach hybrid (tuan 5 viec 3) - quy tac chot TRUOC (NHAT_KY 2026-10-05).

Cau hinh HistGB: (a) day du = lan chay mac dinh; (b) --feature-set khong_diem; (c) --feature-set zos_vung.
Buoc 1 - du lieu diem co giup khong: cai thien I = MAE(b) - MAE(a) moi luoi (I > 0: du lieu diem giup), kiem
  dinh doi dau theo khoi tren don vi CV (trung binh 3 cach chia), Holm m = 13 luoi; I tuong doi = I / MAE(b).
Buoc 2 - cai thien co khac giua cac khung khong: D = I_A - I_B cho 12 cap ho chinh, doi dau theo khoi, Holm m = 12,
  TOST voi Delta_min_D = 25% x trung binh I cua MUC (luoi trong ho); muc co I TB <= 0 hoac buoc 1 khong co y nghia
  o MOI luoi cua muc -> "khong_ap_dung".
Theo nhom khoang cach toi song tinh THEO DIEM (graph_lateral_km cua pixel chua diem): <=2 / 2-10 / >10 km - MO TA.
Quy tac D: bao them tung cach chia rieng (ket luan phai giu ca 3).
(c) bao giong buoc 1 voi (c) thay (b).

Chay:  venv/Scripts/python.exe scripts/analyze_hybrid.py --prefix cv1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze_cv_family import GRIDS, area_levels, area_table, pairs_by_area  # noqa: E402
from settings import data_path  # noqa: E402
from training.block_stats import cluster_t_ci, comparison_seed, seed_mean_errors, signflip_test, unit_metrics  # noqa: E402
from training.evaluate import holm_adjust  # noqa: E402

ALPHA = 0.05
DMIN_FRAC = 0.25
STRATA = [(-0.001, 2.0, "<=2 km"), (2.0, 10.0, "2-10 km"), (10.0, 1e9, ">10 km")]


def load(exp, prefix, grid, schemes, feature_set=None):
    parts = []
    for s in schemes:
        name = f"{prefix}__{grid}__hist_gb__s{s}" + (f"__fs-{feature_set}" if feature_set else "")
        d = pd.read_csv(os.path.join(exp, name, "cv", "oof_points.csv"), dtype={"point_id": str},
                        usecols=["point_id", "season", "err"])
        d["grid"], d["seed"] = grid, s
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def unit_improvement(full, ablated, pu, min_pts=30):
    """I theo don vi: MAE(ablated) - MAE(full), trong so n_pts_mean. Hai bang cung tap (diem, mua)."""
    a = seed_mean_errors(full.assign(model="a"))
    b = seed_mean_errors(ablated.assign(model="b"))
    ua, _, _ = unit_metrics(a, pu, min_pts)
    ub, _, _ = unit_metrics(b, pu, min_pts)
    m = ua.merge(ub, on=["grid", "unit"], suffixes=("_a", "_b"), validate="one_to_one")
    m["I"] = m["mae_b"] - m["mae_a"]
    return m[["grid", "unit", "I", "mae_a", "mae_b", "n_pts_mean_a"]].rename(columns={"n_pts_mean_a": "w"})


def test_vec(d, w, cmp_id, thr=None):
    t = signflip_test(d, w, seed=comparison_seed(cmp_id))
    ci2 = cluster_t_ci(d, w, 2 * ALPHA)
    out = {"delta_hat": t["delta_hat"], "p_value": t["p_value"], "G": t["G"],
           "k_pos": int((np.asarray(d) > 0).sum()), "ci_tost_low": ci2["ci_low"], "ci_tost_high": ci2["ci_high"]}
    if thr is not None:
        out["thr"] = thr
        out["tuong_duong"] = bool(ci2["ci_low"] > -thr and ci2["ci_high"] < thr)
    return out


def step1(imp_by_grid, tag):
    rows = []
    for g, m in imp_by_grid.items():
        r = test_vec(m["I"].to_numpy(), (m["w"] / m["w"].sum()).to_numpy(), f"hybrid_I|{tag}|{g}")
        mae_b = float(np.average(m["mae_b"], weights=m["w"]))
        rows.append({"cau_hinh": tag, "grid": g, "I": r["delta_hat"], "I_tuong_doi": r["delta_hat"] / mae_b,
                     "p_value": r["p_value"], "don_vi_cai_thien": f"{r['k_pos']}/{r['G']}",
                     "ci_tost_low": r["ci_tost_low"], "ci_tost_high": r["ci_tost_high"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
    return out


def step2(imp_by_grid, pairs, levels, s1, tag):
    mean_I = {lv: float(s1[s1["grid"].isin(levels[levels == lv].index)]["I"].mean()) for lv in levels.unique()}
    sig = s1.set_index("grid")["p_holm"] < ALPHA
    rows = []
    for a, b in zip(pairs["grid_a"], pairs["grid_b"]):
        lv = levels[a]
        ma, mb = imp_by_grid[a].set_index("unit"), imp_by_grid[b].set_index("unit")
        units = ma.index.intersection(mb.index)
        d = (ma.loc[units, "I"] - mb.loc[units, "I"]).to_numpy()
        w = ma.loc[units, "w"].to_numpy()
        thr = DMIN_FRAC * mean_I[lv]
        r = test_vec(d, w / w.sum(), f"hybrid_D|{tag}|{a}|{b}", thr=thr if thr > 0 else None)
        ap_dung = thr > 0 and bool(sig[[g for g in levels[levels == lv].index]].any())
        rows.append({"cau_hinh": tag, "muc": lv, "grid_a": a, "grid_b": b, "D": r["delta_hat"], "p_value": r["p_value"],
                     "nguong_D": thr, "ci_tost_low": r["ci_tost_low"], "ci_tost_high": r["ci_tost_high"],
                     "tuong_duong": r.get("tuong_duong", False), "ap_dung": ap_dung})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
    out["nhan"] = np.where(~out["ap_dung"], "khong_ap_dung",
                           np.where(out["p_holm"] < ALPHA, "khac_co_y_nghia",
                                    np.where(out["tuong_duong"], "tuong_duong", "chua_phan_dinh")))
    return out


def strata_table(full, ablated, lat_by_point, tag):
    rows = []
    for lo, hi, ten in STRATA:
        ids = lat_by_point[(lat_by_point > lo) & (lat_by_point <= hi)].index
        for g in GRIDS:
            f = full[(full["grid"] == g) & full["point_id"].isin(ids)]
            b = ablated[(ablated["grid"] == g) & ablated["point_id"].isin(ids)]
            mae_f, mae_b = f["err"].abs().mean(), b["err"].abs().mean()
            rows.append({"cau_hinh": tag, "nhom": ten, "grid": g, "n_diem": int(f["point_id"].nunique()),
                         "MAE_day_du": mae_f, "MAE_bo": mae_b, "I": mae_b - mae_f})
    return pd.DataFrame(rows)


def point_lateral():
    import geopandas as gpd
    import rasterio

    p = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson")).to_crs(32648)
    with rasterio.open(data_path("raw", "river", "river_graph_90m.tif")) as s:
        b = list(s.descriptions).index("graph_lateral_km")
        v = np.array([x[b] for x in s.sample([(q.x, q.y) for q in p.geometry])])
    return pd.Series(v, index=p["point_id"].astype(str))


def main(a):
    exp = os.path.join(ROOT, "artifacts", "experiments")
    at = area_table(a.area_table)
    pairs = pairs_by_area(at, max_ratio=1.2, expected_m=12)
    levels = area_levels(at, pairs=pairs)
    lat = point_lateral()
    pu = _point_unit()
    fulls = {g: load(exp, a.prefix, g, a.schemes) for g in GRIDS}
    res1, res2, strata, sens = [], [], [], []
    for fs in ("khong_diem", "zos_vung"):
        abls = {g: load(exp, a.prefix, g, a.schemes, fs) for g in GRIDS}
        imp = {g: unit_improvement(fulls[g], abls[g], pu) for g in GRIDS}
        s1 = step1(imp, fs)
        res1.append(s1)
        res2.append(step2(imp, pairs, levels, s1, fs))
        strata.append(strata_table(pd.concat(fulls.values()), pd.concat(abls.values()), lat, fs))
        for s in a.schemes:  # quy tac D
            imp_s = {g: unit_improvement(fulls[g][fulls[g]["seed"] == s], abls[g][abls[g]["seed"] == s], pu)
                     for g in GRIDS}
            s1s = step1(imp_s, f"{fs}|s{s}")
            sens.append(s1s.assign(scheme=s))
            sens.append(step2(imp_s, pairs, levels, s1s, f"{fs}|s{s}").assign(scheme=s))
    os.makedirs(a.out_dir, exist_ok=True)
    pd.concat(res1).to_csv(os.path.join(a.out_dir, "dot6_hybrid_buoc1.csv"), index=False)
    pd.concat(res2).to_csv(os.path.join(a.out_dir, "dot6_hybrid_buoc2.csv"), index=False)
    pd.concat(strata).to_csv(os.path.join(a.out_dir, "dot6_hybrid_theo_nhom_song.csv"), index=False)
    pd.concat(sens).to_csv(os.path.join(a.out_dir, "dot6_hybrid_do_nhay_cach_chia.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.concat(res1).round(4).to_string(index=False))
        print(pd.concat(res2).round(4).to_string(index=False))


def _point_unit():
    import geopandas as gpd

    from training.point_eval import point_blocks

    folds = pd.read_csv(os.path.join(ROOT, "data", "eval", "cv_folds.csv"), dtype={"block_id": str, "unit_id": str})
    pts = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    blocks = gpd.read_file(os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    return point_blocks(pts, blocks, folds).set_index("point_id")["unit_id"].astype(str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--area-table", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_doi_chieu_dien_tich.csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    main(ap.parse_args())

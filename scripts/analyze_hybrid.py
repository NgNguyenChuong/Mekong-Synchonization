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
Cot ci90_low/ci90_high cua buoc 1 = KTC 90% (t theo cum) cua I - CHI MO TA, KHONG co nguong tuong duong dang ky
  cho buoc 1 (T7 N3, An 2026-10-05 21:22); buoc 2 giu ci_tost_* vi co Delta_min_D dang ky.
CHG-22 (3 trang thai): moi oof_points phai pred_source = "oof", err huu han, khong trung (diem, mua); (a)/(b)/(c) va
  moi luoi phai CUNG tap (point_id, season, unit_id) theo tung cach chia, unit_id khop don vi tinh tu vi tri diem;
  (a)/(b) va cac luoi trong cap phai cung tap don vi sau loc min_pts; mean_I khong huu han; diem thieu/NaN/ngoai
  khoang graph_lateral_km -> LOI (dung, khong ghi ket qua). Khong con giao/merge inner am tham.

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


KEYS = ["seed", "point_id", "season", "unit_id"]


def load(exp, prefix, grid, schemes, feature_set=None):
    parts = []
    for s in schemes:
        name = f"{prefix}__{grid}__hist_gb__s{s}" + (f"__fs-{feature_set}" if feature_set else "")
        path = os.path.join(exp, name, "cv", "oof_points.csv")
        d = pd.read_csv(path, dtype={"point_id": str, "unit_id": str},
                        usecols=["point_id", "season", "err", "pred_source", "unit_id"])
        if (d["pred_source"] != "oof").any():
            raise SystemExit(f"LOI: {name}: {int((d['pred_source'] != 'oof').sum())} dong pred_source khac 'oof'")
        if not np.isfinite(d["err"].to_numpy(float)).all():
            raise SystemExit(f"LOI: {name}: {int((~np.isfinite(d['err'].to_numpy(float))).sum())} err khong huu han")
        if d["unit_id"].isna().any() or d.duplicated(["point_id", "season"]).any():
            raise SystemExit(f"LOI: {name}: unit_id NaN hoac trung (point_id, season)")
        d["grid"], d["seed"] = grid, s
        parts.append(d.drop(columns="pred_source"))
    return pd.concat(parts, ignore_index=True)


def key_frame(d):
    """Tap khoa (seed, point_id, season, unit_id) da sap xep - de so bang nhau giua cac bang."""
    return d[KEYS].sort_values(KEYS).reset_index(drop=True)


def assert_same_keys(frames: dict, pu=None):
    """CHG-22: moi bang (ten -> DataFrame) phai CUNG tap (seed, point_id, season, unit_id); unit_id khop pu."""
    names = list(frames)
    ref = key_frame(frames[names[0]])
    if ref.empty:
        raise SystemExit(f"LOI: {names[0]} rong")
    for n in names[1:]:
        if not key_frame(frames[n]).equals(ref):
            raise SystemExit(f"LOI: {n} khac tap (seed, point_id, season, unit_id) voi {names[0]}")
    if pu is not None:
        u = ref.drop_duplicates("point_id").set_index("point_id")["unit_id"]
        exp = u.index.map(pu)
        bad = exp.isna() | (exp.to_numpy() != u.to_numpy())
        if bad.any():
            raise SystemExit(f"LOI: {int(bad.sum())} diem co unit_id trong oof_points khac don vi tinh tu vi tri diem")
    return ref


def _same_index(x, y, what):
    if not x.equals(y):
        raise SystemExit(f"LOI: {what}: khac tap don vi - chi o mot ben {sorted(set(x) ^ set(y))[:10]}")


def unit_improvement(full, ablated, pu, min_pts=30):
    """I theo don vi: MAE(ablated) - MAE(full), trong so n_pts_mean. Hai bang cung tap (diem, mua) (assert)."""
    assert_same_keys({"day_du": full, "bo": ablated})
    a = seed_mean_errors(full.assign(model="a"))
    b = seed_mean_errors(ablated.assign(model="b"))
    ua, _, _ = unit_metrics(a, pu, min_pts)
    ub, _, _ = unit_metrics(b, pu, min_pts)
    _same_index(pd.MultiIndex.from_frame(ua[["grid", "unit"]]).sort_values(),
                pd.MultiIndex.from_frame(ub[["grid", "unit"]]).sort_values(), "unit_improvement (a)/(b)")
    m = ua.merge(ub, on=["grid", "unit"], suffixes=("_a", "_b"), validate="one_to_one")
    m["I"] = m["mae_b"] - m["mae_a"]
    return m[["grid", "unit", "I", "mae_a", "mae_b", "n_pts_mean_a"]].rename(columns={"n_pts_mean_a": "w"})


def test_vec(d, w, cmp_id, thr=None):
    t = signflip_test(d, w, seed=comparison_seed(cmp_id))
    ci2 = cluster_t_ci(d, w, 2 * ALPHA)
    out = {"delta_hat": t["delta_hat"], "p_value": t["p_value"], "G": t["G"],
           "k_pos": int((np.asarray(d) > 0).sum()), "ci90_low": ci2["ci_low"], "ci90_high": ci2["ci_high"]}
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
                     # KTC 90%, mo ta, KHONG co nguong tuong duong dang ky cho buoc 1 (khong goi la TOST)
                     "ci90_low": r["ci90_low"], "ci90_high": r["ci90_high"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
    return out


def step2(imp_by_grid, pairs, levels, s1, tag):
    mean_I = {lv: float(s1[s1["grid"].isin(levels[levels == lv].index)]["I"].mean()) for lv in levels.unique()}
    bad = {lv: v for lv, v in mean_I.items() if not np.isfinite(v)}
    if bad:
        raise SystemExit(f"LOI: {tag}: I trung binh muc khong huu han {bad} (luoi cua muc thieu trong buoc 1?)")
    sig = s1.set_index("grid")["p_holm"] < ALPHA
    rows = []
    for a, b in zip(pairs["grid_a"], pairs["grid_b"]):
        lv = levels[a]
        ma, mb = imp_by_grid[a].set_index("unit"), imp_by_grid[b].set_index("unit")
        _same_index(ma.index.sort_values(), mb.index.sort_values(), f"buoc 2 {tag} {a}/{b}")
        units = ma.index
        d = (ma.loc[units, "I"] - mb.loc[units, "I"]).to_numpy()
        w = ma.loc[units, "w"].to_numpy()
        thr = DMIN_FRAC * mean_I[lv]
        r = test_vec(d, w / w.sum(), f"hybrid_D|{tag}|{a}|{b}", thr=thr if thr > 0 else None)
        ap_dung = thr > 0 and bool(sig[[g for g in levels[levels == lv].index]].any())
        rows.append({"cau_hinh": tag, "muc": lv, "grid_a": a, "grid_b": b, "D": r["delta_hat"], "p_value": r["p_value"],
                     # buoc 2 co Delta_min_D dang ky -> KTC 90% dung cho TOST, giu ten ci_tost_*
                     "nguong_D": thr, "ci_tost_low": r["ci90_low"], "ci_tost_high": r["ci90_high"],
                     "tuong_duong": r.get("tuong_duong", False), "ap_dung": ap_dung})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
    out["nhan"] = np.where(~out["ap_dung"], "khong_ap_dung",
                           np.where(out["p_holm"] < ALPHA, "khac_co_y_nghia",
                                    np.where(out["tuong_duong"], "tuong_duong", "chua_phan_dinh")))
    return out


def strata_table(full, ablated, lat_by_point, tag):
    pts = pd.Index(full["point_id"].unique()).union(ablated["point_id"].unique())
    lat = lat_by_point.reindex(pts)
    lo_min, hi_max = STRATA[0][0], STRATA[-1][1]
    bad = lat.isna() | ~((lat > lo_min) & (lat <= hi_max))
    if bad.any():  # CHG-22: diem roi khoi moi nhom -> LOI (truoc day roi lang)
        raise SystemExit(f"LOI: {tag}: {int(bad.sum())} diem graph_lateral_km NaN/thieu/ngoai ({lo_min}, {hi_max}]: "
                         f"{list(lat[bad].index[:5])}")
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
    ref = assert_same_keys({f"(a) {g}": fulls[g] for g in GRIDS}, pu)
    print(f"(diem, mua) moi cach chia: {len(ref) // len(a.schemes)}; cach chia {a.schemes}", flush=True)
    res1, res2, strata, sens = [], [], [], []
    for fs in ("khong_diem", "zos_vung"):
        abls = {g: load(exp, a.prefix, g, a.schemes, fs) for g in GRIDS}
        assert_same_keys({"(a)": fulls[GRIDS[0]], **{f"({fs}) {g}": abls[g] for g in GRIDS}})
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

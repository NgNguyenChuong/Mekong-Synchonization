#!/usr/bin/env python
"""Sai so ly tuong theo khung (CHG-24 (1), CHG-25 muc 5(iii)): du doan = TB nhan that cua o chua diem, cham tai diem;
MAE diem + MAE don vi (khoi 50/100 km) + cap F1 gan nhan "oracle" (mo ta, can hinh hoc o). Khong mo hinh.

Chay:  venv/Scripts/python.exe scripts/analyze_oracle.py [--targets salinity ndwi ...] [--grids ...]
"""
import argparse
import os
import sys
from types import SimpleNamespace

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze_cv_family import GRIDS, area_levels, area_table, pairs_by_area  # noqa: E402
from met_cell_labels import MET_TARGETS  # noqa: E402
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
from training.oracle import (cell_label_series, check_reference_coverage, common_set, mae_table,  # noqa: E402
                             oracle_errors, oracle_family, ratio_table, to_err_long, unit_mae)
from training.split import MAIN_HOLDOUT_SEASON  # noqa: E402

SEASONS = tuple(range(2014, 2027))
# So diem co dap an huu han (E5): buc xa mat 30 diem, nhiet/am 160 diem ven bien
EXPECT_POINTS = {"salinity": 10454, "ndwi": 10454, "rain_chirps": 10454, "dsr_mcd18": 10424,
                 "t2m_era5": 10294, "rh_era5": 10294}
TARGETS = tuple(EXPECT_POINTS)
COARSE_ONLY = ("rain_chirps",)  # CHG-25 muc 1: mua chi kiem o muc tho
NAN_SEASONS = {s["col"]: tuple(s["nan_seasons"]) for s in MET_TARGETS.values()}
OUT_KINDS = ("mae", "cap", "ti_le", "bo")


def loi(msg):
    raise SystemExit(f"LOI: {msg}")


def label_path(labels_dir, target, grid):
    if target == "salinity":
        return os.path.join(labels_dir, f"{grid}_labels_season.csv")
    return os.path.join(labels_dir, "dot7", f"{grid}_labels_season_{target}.csv")


def ref_path(labels_dir, target):
    if target == "salinity":
        return os.path.join(labels_dir, "points_reference.csv")
    return os.path.join(labels_dir, "dot7", f"points_reference_{target}.csv")


def ref_rule_of(ref_file, table_file, target):
    """Dung lai kiem bien the/target dap an + bang cua train.py; tra quy tac dap an (3x3 | pixel)."""
    from training.train import _check_ref_variant

    ns = SimpleNamespace(points_ref=ref_file, table=table_file, target=target, points_ref_variant="main")
    return _check_ref_variant(ns)[2]


def load_static(a):
    import geopandas as gpd

    from training.point_eval import point_blocks

    for p in (a.points, a.blocks, a.folds50, a.folds100, a.area_table):
        if not os.path.exists(p):
            loi(f"thieu file {p}")
    points = gpd.read_file(a.points)
    blocks = gpd.read_file(a.blocks)
    designs = {}
    for name, fp in (("k50", a.folds50), ("k100", a.folds100)):
        folds = pd.read_csv(fp, dtype={"block_id": str, "unit_id": str})
        if "unit_id" not in folds.columns:
            loi(f"{fp} thieu cot unit_id")
        pu = point_blocks(points, blocks, folds).set_index("point_id")["unit_id"].astype(str)
        designs[name] = {"folds": folds, "pu": pu,
                         "cv": sorted(folds.loc[folds["cv_fold"] >= 0, "unit_id"].unique()),
                         "ho": sorted(folds.loc[folds["cv_fold"] < 0, "unit_id"].unique())}
    grids = {}
    for g in a.grids:
        gp = os.path.join(a.grids_dir, f"{g}.geojson")
        if not os.path.exists(gp):
            loi(f"thieu file luoi {gp}")
        gdf = gpd.read_file(gp)
        gdf["cell_id"] = gdf["cell_id"].astype(str)
        grids[g] = gdf
    return points, blocks, designs, grids


def run_target(target, a, points, blocks, designs, grids, pairs, levels, tiers):
    from training.point_eval import point_frame

    rf = ref_path(a.labels_dir, target)
    if not os.path.exists(rf):
        loi(f"thieu file dap an {rf}")
    ref = pd.read_csv(rf, dtype={"point_id": str})
    nan_seasons = NAN_SEASONS.get(target, ())
    parts = []
    for g in a.grids:
        lf = label_path(a.labels_dir, target, g)
        if not os.path.exists(lf):
            loi(f"thieu file nhan o {lf}")
        rule = ref_rule_of(rf, lf, target)
        pf = point_frame(points, grids[g], blocks, designs["k50"]["folds"], ref, (MAIN_HOLDOUT_SEASON,),
                         ref_col=f"ref_{target}", ref_rule=rule)
        check_reference_coverage(pf, a.expect_points[target], a.seasons, nan_seasons)
        lab = pd.read_csv(lf, dtype={"cell_id": str})
        cl = cell_label_series(lab, target)
        flag_col = next((c for c in ("train_ok", "lbl_ok") if c in lab.columns), None)
        flag = None
        if flag_col:
            flag = pd.Series(lab[flag_col].astype(str).str.lower().eq("true").to_numpy(), index=cl.index)
        parts.append(oracle_errors(pf, cl, g, flag))
        print(f"  {target} {g}: {len(pf)} (diem, mua), o thieu nhan {int(parts[-1]['err'].isna().sum())}", flush=True)
    err_all = pd.concat(parts, ignore_index=True)
    common, dropped = common_set(err_all)
    if len(common) == 0:
        loi(f"{target}: tap (diem, mua) chung rong")
    mae = mae_table(err_all, common)
    el = to_err_long(err_all, common)
    pair_rows = []
    for dname, d in designs.items():
        um = unit_mae(el, d["pu"], a.min_pts).rename(columns=lambda c: c if c == "grid" else f"{c}_{dname}")
        mae = mae.merge(um, on="grid", how="left", validate="one_to_one")
        if pairs.empty:
            continue
        out = oracle_family(el, d["pu"], pairs, levels, cv_units=d["cv"], holdout_units=d["ho"],
                            holdout_seasons=(MAIN_HOLDOUT_SEASON,), family=f"oracle_{target}_{dname}",
                            alpha=a.alpha, delta_min=a.delta_min, min_pts=a.min_pts)
        out.insert(0, "blocks", dname)
        out.insert(0, "target", target)
        pair_rows.append(out)
    mae.insert(0, "target", target)
    mae.insert(2, "tier", mae["grid"].map(tiers))
    mae["mua_bo"] = ";".join(map(str, nan_seasons))
    dr = dropped.groupby(["grid", "season"], as_index=False)[["n_universe", "n_invalid_grid", "n_dropped_common",
                                                               "n_common"]].sum()
    dr.insert(0, "target", target)
    return mae, (pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()), dr


def target_pairs(target, main_pairs, grids):
    p = main_pairs[main_pairs["grid_a"].isin(grids) & main_pairs["grid_b"].isin(grids)]
    if target in COARSE_ONLY:
        p = p[p["tier"] == main_pairs["tier"].min()]  # tier h3 nho nhat = muc tho
    return p.reset_index(drop=True)


def run(a):
    outs = {k: f"{a.out}_{k}.csv" for k in OUT_KINDS}
    for p in outs.values():  # chay loi giua chung khong de lai bang cu
        if os.path.exists(p):
            os.remove(p)
    at = area_table(a.area_table)
    main_pairs = pairs_by_area(at, max_ratio=1.2, expected_m=a.expect_pairs)
    levels = area_levels(at, pairs=main_pairs)
    levels = levels[levels.index.isin(a.grids)]
    tiers = area_levels(at)
    miss = sorted(set(a.grids) - set(tiers.index))
    if miss:
        loi(f"luoi khong co trong bang dien tich: {miss}")
    points, blocks, designs, grids = load_static(a)
    res = {k: [] for k in ("mae", "cap", "bo")}
    for t in a.targets:
        print(f"[{t}]", flush=True)
        mae, cap, dr = run_target(t, a, points, blocks, designs, grids, target_pairs(t, main_pairs, a.grids),
                                  levels, tiers)
        res["mae"].append(mae)
        res["cap"].append(cap)
        res["bo"].append(dr)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
    tables["ti_le"] = ratio_table(tables["cap"]) if not tables["cap"].empty else pd.DataFrame()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for k, p in outs.items():
        tmp = p + ".tmp"
        tables[k].to_csv(tmp, index=False)
        os.replace(tmp, p)
    cols = ["target", "grid", "tier", "n_ref", "n_nan_cell", "mae_diem", "mae_don_vi_k50", "mae_don_vi_k100"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(tables["mae"][cols].to_string(index=False))
        if not tables["ti_le"].empty:
            print(tables["ti_le"].to_string(index=False))
    print("Ghi: " + ", ".join(outs.values()))
    return tables


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    ap.add_argument("--grids", nargs="+", default=GRIDS)
    ap.add_argument("--out", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot7_oracle"),
                    help="Tien to: <out>_{mae,cap,ti_le,bo}.csv")
    ap.add_argument("--labels-dir", default=data_path("labels"))
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--points", default=os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    ap.add_argument("--blocks", default=os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    ap.add_argument("--folds50", default=os.path.join(ROOT, "data", "eval", "cv_folds_s42.csv"))
    ap.add_argument("--folds100", default=os.path.join(ROOT, "data", "eval", "khoi100", "cv_folds_s42.csv"))
    ap.add_argument("--area-table", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_doi_chieu_dien_tich.csv"))
    ap.add_argument("--seasons", nargs="+", type=int, default=list(SEASONS))
    ap.add_argument("--expect-pairs", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta-min", type=float, default=0.05)
    ap.add_argument("--min-pts", type=int, default=30)
    a = ap.parse_args(argv)
    a.expect_points = dict(EXPECT_POINTS)
    return a


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError, AssertionError) as e:
        raise SystemExit(f"LOI: {e}")

#!/usr/bin/env python
"""Bang hop nhat (cell_id, season) cho 13 luoi x 4 bo nhan (tuan 3 viec 3). Quy tac ghep: src/unified_table.py.

Dau ra A:/Dataset_NCKH/features/unified/<luoi>_unified[_<bo>].csv (+ .provenance.json: sha256 moi nguon,
vai tro cot, danh sach dac trung mac dinh). Bao cao -> KE_HOACH/ket-qua/dot4_bang_hop_nhat.csv (so dong,
so o, dong train_ok, dong train_ok_scope = tap huan luyen, NaN tung dac trung mac dinh tren tap huan luyen).
Tap huan luyen = train_ok_scope (train_ok VA scope_frac > 0); dong nao trong tap co MOI dac trung tinh NaN -> loi.
ERA5 dung ban `_filled` (lap ven bien 1 pixel, CHG-11); co era5_fill_frac chi de phan tang.

Chay:  venv/Scripts/python.exe scripts/build_unified_table.py [--grids h3_res_7 ...] [--sets "" keepwater ...]
"""
import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402
from training.features import DEFAULT_ALLOWED_FEATURES, find_leak_columns, resolve_feature_list  # noqa: E402
from unified_table import (KEY_COLS, LABEL_SETS, QUALITY_COLS, TARGET_COL, TRAIN_COL,  # noqa: E402
                           finite_or_nan, label_suffix, merge_sources, nan_report)

STATIC_FEATURES = ("dem_mean", "dist_main_river_km", "dist_any_water_km", "dist_coast_km")
HYBRID_FEATURES = ("dist_mouth_river_km", "zos_mouth_p90", "sluice_frac")
LABEL_ONLY_COLS = ("salinity", "n_valid_px", "valid_frac", "train_ok", "train_ok_10pct", TRAIN_COL)

DATA = "A:/Dataset_NCKH"


def sources(a, grid, label_set):
    return {
        "labels": os.path.join(a.labels_dir, f"{grid}_labels_season{label_suffix(label_set)}.csv"),
        "era5": os.path.join(a.features_dir, "era5_season", f"{grid}_era5_season_filled.csv"),
        "static": os.path.join(a.features_dir, "static", f"{grid}_static.csv"),
        "hydro": os.path.join(a.features_dir, "hydro_season_2014_2026.csv"),
        "hybrid": os.path.join(a.features_dir, "hybrid", f"{grid}_hybrid.csv"),
    }


def scope_centroid_csv(a, grid):
    """Tam phan dat (pixel scope v3) moi o -> cache <static>/<luoi>_scope_centroid.csv (+ provenance)."""
    import geopandas as gpd

    from static_features import scope_centroids

    out = os.path.join(a.features_dir, "static", f"{grid}_scope_centroid.csv")
    grid_path = os.path.join(a.grids_dir, f"{grid}.geojson")
    if os.path.exists(out) and not a.force_centroids:
        return out
    g = gpd.read_file(grid_path)
    df = scope_centroids(a.scope, (g["cell_id"].astype(str).tolist(), None, g.geometry.tolist()))
    df.to_csv(out, index=False, float_format="%.3f")
    write_provenance(out, CANONICAL_BOUNDARY, grid=os.path.basename(grid_path), grid_sha256=file_sha256(grid_path),
                     scope=a.scope, scope_sha256=file_sha256(a.scope),
                     rule="tam TAM pixel scope == 1 trong o (pixel theo tam), EPSG:32648; cho IDW (An 2026-10-04)")
    return out


def build_one(a, grid, label_set):
    src = sources(a, grid, label_set)
    src["centroid"] = scope_centroid_csv(a, grid)
    missing = [p for p in src.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{grid}/{label_set or 'chinh'}: thieu {missing}")
    rd = {k: pd.read_csv(p, dtype={"cell_id": str}) for k, p in src.items()}
    static = rd["static"].merge(rd["centroid"], on="cell_id", how="left", validate="one_to_one")
    table = merge_sources(rd["labels"], rd["era5"], static, rd["hydro"], rd["hybrid"])
    no_xy = table[TRAIN_COL].astype(bool) & table["scope_cx"].isna()
    if no_xy.any():
        raise ValueError(f"{grid}: {int(no_xy.sum())} dong huan luyen khong co tam phan dat (scope_n_px = 0)")
    feats, absent = resolve_feature_list(table.columns, DEFAULT_ALLOWED_FEATURES)
    if absent:
        raise ValueError(f"{grid}: dac trung mac dinh vang mat trong bang: {absent}")
    leaks = find_leak_columns(feats)
    if leaks:
        raise ValueError(f"{grid}: dac trung mac dinh khop mau cam: {leaks}")
    bad_inf = [c for c in feats if not finite_or_nan(table[c])]
    if bad_inf:
        raise ValueError(f"{grid}: cot co +-inf: {bad_inf}")
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{grid}_unified{label_suffix(label_set)}.csv")
    table.to_csv(out, index=False, float_format="%.6g")
    write_provenance(out, CANONICAL_BOUNDARY, grid=grid, label_set=label_set or "chinh",
                     sources={k: {"path": p, "sha256": file_sha256(p)} for k, p in src.items()},
                     key_cols=list(KEY_COLS), target=TARGET_COL,
                     quality_cols=[c for c in QUALITY_COLS if c in table.columns],
                     default_features=feats, rule="tuan 3 viec 3; ERA5 _filled (CHG-11); hybrid CHG-16")
    ok = table[TRAIN_COL].astype(bool)
    for group, cols in (("tinh", STATIC_FEATURES), ("hybrid", HYBRID_FEATURES)):
        n_all_nan = int(table.loc[ok, list(cols)].isna().all(axis=1).sum())
        if n_all_nan:
            raise ValueError(f"{grid}/{label_set or 'chinh'}: {n_all_nan} dong {TRAIN_COL} co MOI dac trung {group} NaN")
    nr = nan_report(table, feats)
    row = {"grid": grid, "label_set": label_set or "chinh", "rows": len(table), "cells": table["cell_id"].nunique(),
           "rows_label": int(table[TARGET_COL].notna().sum()), "rows_train_ok": int(table["train_ok"].sum()),
           "rows_train_ok_scope": int(ok.sum()),
           "rows_dropped_scope0": int(table["train_ok"].astype(bool).sum() - ok.sum()),
           "n_features": len(feats),
           "train_rows_any_nan": int(table.loc[ok, feats].isna().any(axis=1).sum())}
    col = f"nan_{TRAIN_COL}"
    row.update({f"nan_{c}": int(v) for c, v in zip(nr["column"], nr[col]) if v})
    return row, table


def check_same_features(tables: dict, grid: str):
    """Bo phu chi khac nhan (CHG-15): moi cot ngoai LABEL_ONLY_COLS phai giong het bo chinh (NaN cung cho)."""
    if "" not in tables:
        return
    base = tables[""].drop(columns=list(LABEL_ONLY_COLS))
    for s, t in tables.items():
        other = t.drop(columns=list(LABEL_ONLY_COLS))
        if list(other.columns) != list(base.columns) or not other.equals(base):
            diff = [c for c in base.columns if c not in other.columns or not other[c].equals(base[c])]
            raise ValueError(f"{grid}: bo '{s}' khac bo chinh o cot dac trung {diff[:5]}")


def main(a):
    grids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(a.grids_dir, "*.geojson")))
    if a.grids:
        grids = [g for g in grids if g in a.grids]
    rows = []
    for g in grids:
        tables = {}
        for s in a.sets:
            row, tables[s] = build_one(a, g, s)
            rows.append(row)
            r = rows[-1]
            print(f"  {g:<18} {r['label_set']:<13} dong {r['rows']:>6}  train_ok {r['rows_train_ok']:>6}  "
                  f"{TRAIN_COL} {r['rows_train_ok_scope']:>6} (loai scope 0: {r['rows_dropped_scope0']})  "
                  f"dong huan luyen co NaN {r['train_rows_any_nan']}", flush=True)
        check_same_features(tables, g)
    rep = pd.DataFrame(rows).fillna(0)
    if a.report:
        rep.to_csv(a.report, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(rep.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels-dir", default=f"{DATA}/labels")
    ap.add_argument("--features-dir", default=f"{DATA}/features")
    ap.add_argument("--out-dir", default=f"{DATA}/features/unified")
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--grids", nargs="*")
    ap.add_argument("--sets", nargs="*", default=list(LABEL_SETS), help='"" = bo chinh')
    ap.add_argument("--scope", default=f"{DATA}/features/scope_mask_v3.tif")
    ap.add_argument("--force-centroids", action="store_true", help="Tinh lai tam phan dat da cache")
    ap.add_argument("--report", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_bang_hop_nhat.csv"))
    main(ap.parse_args())

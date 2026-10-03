#!/usr/bin/env python
"""Dung thiet ke danh gia chung (T3-V4): khoi giu rieng + diem danh gia + kiem tra loai tru.

Chay:  python scripts/build_eval_design.py [--block-km 50] [--holdout-frac 0.2] [--n-points 3000] [--seed 42]
"""
import argparse
import glob
import os
import sys

import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval_design import make_holdout_blocks, sample_eval_points, training_mask  # noqa: E402
from preprocessing import CANONICAL_BOUNDARY  # noqa: E402


def main(args):
    boundary = gpd.read_file(args.boundary).to_crs(4326)
    blocks = make_holdout_blocks(boundary, args.block_km, args.holdout_frac, args.seed)
    points = sample_eval_points(boundary, blocks, args.n_points, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    blocks.to_file(os.path.join(args.out_dir, "holdout_blocks.geojson"), driver="GeoJSON")
    points.to_file(os.path.join(args.out_dir, "eval_points.geojson"), driver="GeoJSON")

    total = blocks["land_km2"].sum()
    held = blocks.loc[blocks["is_holdout"], "land_km2"].sum()
    print(f"Khoi {args.block_km:g} km: {len(blocks)} khoi, giu rieng {int(blocks['is_holdout'].sum())} "
          f"({held:.0f}/{total:.0f} km2 dat = {100 * held / total:.1f}%) | {len(points)} diem danh gia")

    holdout_union = blocks[blocks["is_holdout"]].geometry.union_all()
    in_holdout = int(points.geometry.within(holdout_union).sum())
    print(f"Diem nam trong vung kiem tra: {in_holdout}/{len(points)}")

    rows = []
    for path in sorted(glob.glob(os.path.join(args.grids_dir, "*.geojson"))):
        grid = gpd.read_file(path).to_crs(4326)
        is_train = training_mask(grid, blocks)
        # Kiem tra doc lap bang spatial join: o huan luyen giao vung kiem tra phai = 0.
        train = grid[is_train]
        leaks = gpd.sjoin(train, blocks[blocks["is_holdout"]], predicate="intersects", how="inner")
        n_leak = leaks.index.nunique()
        rows.append({
            "grid": os.path.basename(path),
            "cells": len(grid),
            "train_cells": int(is_train.sum()),
            "excluded_cells": int((~is_train).sum()),
            "excluded_pct": round(100 * (~is_train).sum() / len(grid), 2),
            "train_cells_touching_holdout": int(n_leak),
        })
        if n_leak:
            raise SystemExit(f"LOI: {n_leak} o huan luyen giao vung kiem tra trong {path}")

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        report.to_csv(args.report, index=False)
        print(f"Bang: {args.report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--report", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "tuan3_thiet_ke_danh_gia.csv"))
    ap.add_argument("--block-km", type=float, default=50.0)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--n-points", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())

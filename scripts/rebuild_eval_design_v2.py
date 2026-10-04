#!/usr/bin/env python
"""Dung lai thiet ke danh gia tren ranh gioi v2 (An duyet 2026-10-03) - GIU NGUYEN khoi giu rieng.

- Khoi: giu block_id / is_holdout cua file hien co; chi tinh lai land_km2 = dien tich (khoi giao v2).
  Khoi 50 km cung luoi ma v2 moi cham toi -> them voi is_holdout = False.
- Diem danh gia tren MAT NA PHAM VI TINH (An duyet M 2026-10-03; raster scripts/build_scope_mask.py):
  v2 ∩ WorldCover khong 0/80/95 ∩ dau chan Zenodo (Salinity huu han >= 1 nam 2014-2023); diem nhan khi
  pixel 30 m chua diem co scope = 1. Hop le theo nam do buoc danh gia xu ly (quy tac 5/9).
  * n_holdout diem trong khoi giu rieng (seed 42)
  * diem trong khoi KHONG giu rieng cung mat do tren mat na (seed 43) -> cham bang du doan out-of-fold.
  Cot: point_id, block_id, is_holdout.
- Kiem: 0 o huan luyen giao vung kiem tra (13 luoi), moi diem co o o ca 13 luoi, moi diem scope = 1.
- Ban dau tien (v1) da sao luu o <out-dir>/deprecated_v1/ (nguon block_id/is_holdout). Ban dang co + cv_folds
  + cell_blocks sao luu sang <out-dir>/<--backup-name> (mac dinh deprecated_v2a) truoc khi ghi; khong ghi de
  ban sao luu da co. Bao so diem ban truoc nam ngoai mat na moi, theo ly do.

CHG-10 (2026-10-03): diem hien hanh = ban sinh boi script nay tren scope_mask_v2.tif (mac dinh GIU v2 co chu dich
de tai lap) roi LOC theo scope_mask_v3.tif bang scripts/filter_eval_points_scope.py (khong sinh lai). Chay script nay
se SINH LAI va ghi de diem -> chi dung khi that su muon gieo lai.

Chay:  venv/Scripts/python.exe scripts/rebuild_eval_design_v2.py [--n-holdout 3000] [--seed-holdout 42] [--seed-cv 43]
"""
import argparse
import glob
import os
import shutil
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval_design import (  # noqa: E402
    exact_block_frame,
    raster_values_at,
    refresh_blocks,
    sample_points_in_mask,
    training_mask,
)
from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402
from scope_mask import scope_area_by_block  # noqa: E402

EXCLUDE_LC = (0, 80, 95)


def log(msg):
    print(msg, flush=True)


def backup(paths, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for p in paths:
        dst = os.path.join(dst_dir, os.path.basename(p))
        if os.path.isdir(p):
            if not os.path.exists(dst):
                shutil.copytree(p, dst)
                log(f"  sao luu thu muc {p} -> {dst}")
        elif os.path.exists(p) and not os.path.exists(dst):  # khong ghi de ban sao luu da co
            shutil.copy2(p, dst)
            log(f"  sao luu {p} -> {dst}")


def dropped_old_points(old_points, scope_path, worldcover, exclude):
    """Diem ban truoc nam ngoai mat na pham vi moi, kem ly do: ngoai_dau_chan (zenodo_n_years = 0, ke ca
    ngoai khung raster), wc_tam_pixel (WorldCover tai TAM pixel 30 m chua diem thuoc exclude - diem cu duoc
    nhan theo pixel WorldCover 10 m chua diem), ngoai_ranh_gioi_raster (con lai: tam pixel ngoai v2)."""
    with rasterio.open(scope_path) as src:
        crs, tr = src.crs.to_string(), src.transform
    p = old_points.to_crs(crs)
    x, y = p.geometry.x.to_numpy(), p.geometry.y.to_numpy()
    sc = raster_values_at(scope_path, x, y, band=1, fill=0)
    ny = raster_values_at(scope_path, x, y, band=2, fill=0)
    inv = ~tr
    col, row = np.floor(inv.a * x + inv.c), np.floor(inv.e * y + inv.f)
    cx, cy = tr.c + tr.a * (col + 0.5), tr.f + tr.e * (row + 0.5)  # tam pixel 30 m chua diem
    with rasterio.open(worldcover) as src:
        to_wc = pyproj.Transformer.from_crs(crs, src.crs.to_string(), always_xy=True)
    wx, wy = to_wc.transform(cx, cy)
    wc = raster_values_at(worldcover, wx, wy, fill=0)
    reason = np.where(sc == 1, "", np.where(ny == 0, "ngoai_dau_chan",
                                            np.where(np.isin(wc, exclude), "wc_tam_pixel", "ngoai_ranh_gioi_raster")))
    ll = old_points.to_crs(4326).geometry
    out = old_points[["point_id", "block_id", "is_holdout"]].copy()
    out["reason"], out["wc_tam_pixel"], out["zenodo_n_years"] = reason, wc, ny
    out["lon"], out["lat"] = ll.x.to_numpy(), ll.y.to_numpy()
    return out[out["reason"] != ""]


def check_grids(grids_dir, blocks, points):
    """Bang kiem tung luoi: o huan luyen cham vung kiem tra (phai 0), diem khong co o (phai 0)."""
    rows = []
    hold = blocks[blocks["is_holdout"]]
    for path in sorted(glob.glob(os.path.join(grids_dir, "*.geojson"))):
        grid = gpd.read_file(path).to_crs(4326)
        is_train = training_mask(grid, blocks)
        leaks = gpd.sjoin(grid[is_train], hold, predicate="intersects", how="inner").index.nunique()
        hit = gpd.sjoin(points[["point_id", "geometry"]], grid[["cell_id", "geometry"]],
                        predicate="intersects", how="left")
        no_cell = int(hit.groupby("point_id")["cell_id"].apply(lambda s: s.notna().any()).eq(False).sum())
        rows.append({
            "grid": os.path.basename(path), "cells": len(grid), "train_cells": int(is_train.sum()),
            "excluded_cells": int((~is_train).sum()),
            "excluded_pct": round(100 * (~is_train).sum() / len(grid), 2),
            "train_cells_touching_holdout": int(leaks), "points_without_cell": no_cell,
        })
    return pd.DataFrame(rows)


def main(args):
    boundary = gpd.read_file(args.boundary).to_crs(4326)
    blocks_path = os.path.join(args.out_dir, "holdout_blocks.geojson")
    points_path = os.path.join(args.out_dir, "eval_points.geojson")
    dep_dir = os.path.join(args.out_dir, "deprecated_v1")
    backup([blocks_path, points_path], dep_dir)  # chi lan dau (ban v1); chay lai khong ghi de
    prev_dir = os.path.join(args.out_dir, args.backup_name)
    prev = [blocks_path, points_path, os.path.join(args.out_dir, "cv_folds.csv"),
            os.path.join(args.out_dir, "cell_blocks")]
    backup(prev + [p + ".provenance.json" for p in prev[:3]], prev_dir)
    blocks_sha_before = file_sha256(blocks_path) if os.path.exists(blocks_path) else None
    # Nguon block_id / is_holdout = ban v1 da sao luu -> chay lai cho cung ket qua (tai lap).
    old_blocks = gpd.read_file(os.path.join(dep_dir, "holdout_blocks.geojson"))

    blocks, added = refresh_blocks(old_blocks, boundary, args.block_km)
    same = blocks.set_index("block_id").loc[old_blocks["block_id"], "is_holdout"].to_numpy()
    assert (same == old_blocks["is_holdout"].astype(bool).to_numpy()).all(), "is_holdout bi doi"
    log(f"Khoi: {len(old_blocks)} cu, them {len(added)} {added} | giu rieng "
        f"{sorted(blocks.loc[blocks['is_holdout'], 'block_id'])}")

    log(f"Dien tich mat na pham vi theo khoi ({args.scope_mask}) ...")
    area = scope_area_by_block(exact_block_frame(blocks)[["block_id", "geometry"]], args.scope_mask)
    blocks = blocks.assign(mask_km2=blocks["block_id"].map(area.set_index("block_id")["scope_km2"]).to_numpy())
    hold_mask = blocks.loc[blocks["is_holdout"], "mask_km2"].sum()
    cv_mask = blocks.loc[~blocks["is_holdout"], "mask_km2"].sum()
    n_cv = int(round(args.n_holdout * cv_mask / hold_mask))
    log(f"  mat na dat: giu rieng {hold_mask:.1f} km2, khong giu rieng {cv_mask:.1f} km2 "
        f"-> {args.n_holdout} + {n_cv} diem (mat do {args.n_holdout / hold_mask:.4f} diem/km2)")

    utm = boundary.estimate_utm_crs()
    land = boundary.to_crs(utm).geometry.union_all()
    bl_utm = blocks.to_crs(utm)
    with rasterio.open(args.scope_mask) as src:
        sc_crs = src.crs.to_string()
    to_sc = pyproj.Transformer.from_crs(utm, sc_crs, always_xy=True)

    def accept(xs, ys):
        sx, sy = to_sc.transform(xs, ys)
        return raster_values_at(args.scope_mask, sx, sy, band=1, fill=0) == 1

    parts = []
    for is_hold, n, seed in ((True, args.n_holdout, args.seed_holdout), (False, n_cv, args.seed_cv)):
        region = land.intersection(bl_utm.loc[bl_utm["is_holdout"] == is_hold].geometry.union_all())
        pts = sample_points_in_mask(region, utm, n, seed, accept)
        pts["is_holdout"] = is_hold
        parts.append(pts)
    points = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    points.insert(0, "point_id", [f"pt_{i:05d}" for i in range(len(points))])
    # Gan khoi trong UTM (khoi dinh nghia la o vuong UTM; ban EPSG:4326 chi 4 dinh nen canh lech vai m).
    j = gpd.sjoin(points.to_crs(utm),
                  bl_utm[["block_id", "is_holdout", "geometry"]].rename(columns={"is_holdout": "blk_hold"}),
                  predicate="within", how="left")
    j = j[~j.index.duplicated()]
    points["block_id"] = j["block_id"].to_numpy()
    if points["block_id"].isna().any():
        raise SystemExit(f"LOI: {points['block_id'].isna().sum()} diem khong thuoc khoi nao")
    if (j["blk_hold"].astype(bool).to_numpy() != points["is_holdout"].to_numpy()).any():
        raise SystemExit("LOI: is_holdout cua diem khac cua khoi")
    points = points[["point_id", "block_id", "is_holdout", "geometry"]]

    # Kiem lai mat na tai diem (doc lap voi ham accept: dung lai to_crs cua geopandas).
    p_sc = points.to_crs(sc_crs)
    sc_at = raster_values_at(args.scope_mask, p_sc.geometry.x, p_sc.geometry.y, band=1, fill=0)
    ny_at = raster_values_at(args.scope_mask, p_sc.geometry.x, p_sc.geometry.y, band=2, fill=0)
    in_b = points.geometry.within(boundary.geometry.union_all())
    log(f"Kiem diem: ngoai v2 {int((~in_b).sum())} | scope != 1 {int((sc_at != 1).sum())} | dau chan 0 nam "
        f"{int((ny_at == 0).sum())} | point_id trung {int(points['point_id'].duplicated().sum())}")
    if (sc_at != 1).any() or (ny_at == 0).any():
        raise SystemExit("LOI: co diem ngoai mat na pham vi")

    old_pts_path = os.path.join(prev_dir, "eval_points.geojson")
    if os.path.exists(old_pts_path):
        old_pts = gpd.read_file(old_pts_path)
        drop = dropped_old_points(old_pts, args.scope_mask, args.worldcover, EXCLUDE_LC)
        log(f"Diem ban truoc ({old_pts_path}): {len(old_pts)}; ngoai mat na moi {len(drop)} "
            f"(giu rieng {int(drop['is_holdout'].sum())}, CV {int((~drop['is_holdout']).sum())})")
        log(drop.groupby(["reason", "is_holdout"]).size().to_string())
        log(drop.groupby("block_id").size().sort_values(ascending=False).to_string())
        if args.report:
            os.makedirs(os.path.dirname(args.report), exist_ok=True)
            drop.to_csv(args.report.replace(".csv", "_diem_cu_bi_loai.csv"), index=False)

    blocks_out = blocks[["block_id", "land_km2", "is_holdout", "geometry"]]
    blocks_out.to_file(blocks_path, driver="GeoJSON")
    points.to_file(points_path, driver="GeoJSON")
    common = dict(scope_mask=args.scope_mask, scope_mask_sha256=file_sha256(args.scope_mask),
                  rule="diem nhan khi pixel 30 m chua diem co scope = 1 (v2 ∩ WC khong 0/80/95 ∩ dau chan Zenodo)",
                  exclude_worldcover=list(EXCLUDE_LC), worldcover=args.worldcover)
    write_provenance(blocks_path, args.boundary, note="block_id/is_holdout giu tu ban v1; land_km2 tinh lai",
                     added_blocks=added)
    write_provenance(points_path, args.boundary, n_holdout=args.n_holdout, n_cv=n_cv,
                     seed_holdout=args.seed_holdout, seed_cv=args.seed_cv, **common)
    log(f"Ghi {blocks_path}, {points_path}")
    if blocks_sha_before is not None:
        log(f"File khoi {'KHONG doi' if file_sha256(blocks_path) == blocks_sha_before else 'DA DOI'} so voi ban truoc")

    per_block = (points.groupby(["block_id", "is_holdout"]).size().rename("n_points").reset_index()
                 .merge(blocks[["block_id", "land_km2", "mask_km2"]], on="block_id", how="right")
                 .fillna({"n_points": 0}))
    per_block["is_holdout"] = per_block["block_id"].map(blocks.set_index("block_id")["is_holdout"])
    per_block["points_per_km2"] = (per_block["n_points"] / per_block["mask_km2"]).round(4)
    per_block = per_block.sort_values(["is_holdout", "block_id"], ascending=[False, True])
    log(per_block.to_string(index=False))

    report = check_grids(args.grids_dir, blocks, points)
    log(report.to_string(index=False))
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        report.to_csv(args.report, index=False)
        per_block.to_csv(args.report.replace(".csv", "_theo_khoi.csv"), index=False)
        log(f"Bang: {args.report}")
    if report["train_cells_touching_holdout"].any() or report["points_without_cell"].any():
        raise SystemExit("LOI: co o huan luyen cham vung kiem tra hoac diem khong co o")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--worldcover", default="A:/Dataset_NCKH/gee/LandCover_DBSCL_2021.tif")
    ap.add_argument("--scope-mask", default="A:/Dataset_NCKH/features/scope_mask_v2.tif",
                    help="Mat na pham vi tinh (scripts/build_scope_mask.py)")
    ap.add_argument("--backup-name", default="deprecated_v2a",
                    help="Thu muc sao luu ban dang co (diem/khoi/cv_folds/cell_blocks) truoc khi ghi")
    ap.add_argument("--report", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_thiet_ke_danh_gia.csv"))
    ap.add_argument("--block-km", type=float, default=50.0)
    ap.add_argument("--n-holdout", type=int, default=3000)
    ap.add_argument("--seed-holdout", type=int, default=42)
    ap.add_argument("--seed-cv", type=int, default=43)
    main(ap.parse_args())

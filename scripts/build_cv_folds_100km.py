#!/usr/bin/env python
"""File fold CV khoi ~100 km (CHG-20, NHAT_KY 2026-10-05) - do nhay kich thuoc khoi.

Quy tac (chot truoc khi chay mo hinh):
  - Gop 2x2 co dinh: khoi 50 km `blk_i_j` -> sieu khoi `blk_{i//2}_{j//2}`; khoi giu rieng GIU NGUYEN (cv_fold -1,
    unit_id = chinh no). Don vi CV = phan CV cua sieu khoi, gop sieu khoi < 30 diem vao sieu khoi ke
    (eval_design.merge_small_blocks, quy tac J).
  - Gan fold: tap ung vien eval_design.unit_pool (5 fold, 3.000 diem xuat phat, pool_seed 0), xep hang rank_pool;
    hang 1 = phuong an chinh (ghi cv_folds_s42.csv), hang 2 = do nhay (cv_folds_s43.csv). Rang buoc ven bien /
    do trung cua fold 50 km KHONG ap (khong kha thi voi 9 don vi - ghi trong NHAT_KY).
Dau ra --out-dir/cv_folds_s{42,43}.csv: block_id (khoi 50 km), unit_id (sieu khoi), cv_fold, + cot dien tich; kem
.provenance.json. train.py ap quy tac "cham" tren khoi 50 km -> moi khoi cua cung sieu khoi cung fold = cham sieu khoi.

Chay:  venv/Scripts/python.exe scripts/build_cv_folds_100km.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval_design import merge_small_blocks, rank_pool, unit_pool  # noqa: E402
from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402

CRITERIA = ("scope_km2", "coast_scope_km2")


def super_id(block_id: pd.Series) -> pd.Series:
    ij = block_id.str.extract(r"^blk_(\d+)_(\d+)$")
    if ij.isna().any().any():
        raise ValueError("block_id khong theo mau blk_i_j")
    ij = ij.astype(int)
    return "blk_" + (ij[0] // 2).astype(str) + "_" + (ij[1] // 2).astype(str)


def build(folds50: pd.DataFrame, n_folds=5, n_starts=3000, pool_seed=0, min_points=30):
    f = folds50.copy()
    f["super"] = super_id(f["block_id"].astype(str))
    cv = f[f["cv_fold"] >= 0]
    t = (cv.groupby("super").agg(land_km2=("land_km2", "sum"), scope_km2=("scope_km2", "sum"),
                                 coast_scope_km2=("coast_scope_km2", "sum"), n_points=("n_cv_points", "sum"))
         .reset_index().rename(columns={"super": "block_id"}))
    t["is_holdout"] = False
    merged, log = merge_small_blocks(t, min_points=min_points)
    unit_of_super = merged.set_index("block_id")["unit_id"]
    units = merged.groupby("unit_id").agg(scope_km2=("scope_km2", "sum"),
                                          coast_scope_km2=("coast_scope_km2", "sum")).reset_index()
    pool = unit_pool(units, CRITERIA, n_folds, n_starts, pool_seed)
    keys, *_ = rank_pool(units, pool, CRITERIA, n_folds)
    schemes = {}
    for seed, key in zip((42, 43), keys[:2]):
        fold_of_unit = dict(zip(units["unit_id"], np.asarray(key, int)))
        out = f.drop(columns="super").copy()
        is_cv = out["cv_fold"] >= 0
        out.loc[is_cv, "unit_id"] = f.loc[is_cv, "super"].map(unit_of_super)
        out.loc[is_cv, "cv_fold"] = out.loc[is_cv, "unit_id"].map(fold_of_unit)
        out.loc[~is_cv, "unit_id"] = out.loc[~is_cv, "block_id"]
        if out.loc[is_cv, "cv_fold"].isna().any():
            raise ValueError("khoi CV khong co fold")
        out["cv_fold"] = out["cv_fold"].astype(int)
        schemes[seed] = (out, float(pool[key][0]))
    return schemes, log, units


def main(a):
    folds50 = pd.read_csv(a.folds50, dtype={"block_id": str, "unit_id": str})
    schemes, log, units = build(folds50)
    os.makedirs(a.out_dir, exist_ok=True)
    for seed, (out, obj) in schemes.items():
        p = os.path.join(a.out_dir, f"cv_folds_s{seed}.csv")
        out.to_csv(p, index=False)
        write_provenance(p, CANONICAL_BOUNDARY, rule="CHG-20 khoi 100 km: gop 2x2 + merge_small_blocks + unit_pool hang "
                         f"{1 if seed == 42 else 2}", folds50=a.folds50, folds50_sha256=file_sha256(a.folds50),
                         objective=obj, n_units=int(len(units)))
        print(f"{p}: muc tieu {obj:.3f}; fold theo don vi {out[out.cv_fold >= 0].groupby('unit_id').cv_fold.first().to_dict()}")
    print("Gop:", log.to_dict(orient="records"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds50", default=os.path.join(ROOT, "data", "eval", "cv_folds.csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "eval", "khoi100"))
    main(ap.parse_args())

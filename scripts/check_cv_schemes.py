#!/usr/bin/env python
"""Kiem doc lap cac phuong an fold CV (cv_folds_s<seed>.csv) tren 13 luoi bang check_split.py --membership.

Moi (phuong an, luoi):
- doc data/eval/cell_blocks/s<seed>/<luoi>.csv, doi chieu provenance (cv_folds/grid/blocks sha256);
- bang GIA (o x mua 2019-2021, nhan gia) -> assign_eval_split -> block_cv_splits (quy tac "cham") ->
  cv_membership (cell_id, fold, role); chay .claude/skills/method-review/scripts/check_split.py voi --cv-folds,
  --membership (kiem hinh hoc tren luoi that) -> phai thoat ma 0;
- doi chung am (luoi dau tien moi phuong an): hoan vi fold trong membership -> check_split phai thoat ma 1;
- phuong an C: moi diem CV p (khoi fold k) -> o chua p KHONG duoc co role "train" o fold k (theo membership);
  o chua p cham khoi giu rieng (khong co trong membership) -> van du doan duoc (mo hinh fold k, chi dac trung).
Bang ket qua: --report (mac dinh KE_HOACH/ket-qua/dot4_cv_kiem_membership.csv).

Chay:  venv/Scripts/python.exe scripts/check_cv_schemes.py [--seeds 42 43 44]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval_design import read_cell_block_table  # noqa: E402
from preprocessing import file_sha256, provenance_path  # noqa: E402
from training.split import assign_eval_split, block_cv_membership, block_cv_splits  # noqa: E402

CHECK_SPLIT = os.path.join(ROOT, ".claude", "skills", "method-review", "scripts", "check_split.py")
SEASONS = (2019, 2020, 2021)


def log(msg):
    print(msg, flush=True)


def run_check(table_csv, grid, blocks, folds, membership):
    cmd = [sys.executable, CHECK_SPLIT, table_csv, "--grid", grid, "--blocks", blocks, "--holdout-years", "2020",
           "--mask-col", "label_mask_id", "--fold-col", "cv_fold", "--block-col", "block_id",
           "--cv-folds", folds, "--membership", membership]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()


def main(a):
    if not os.path.exists(CHECK_SPLIT):
        raise SystemExit(f"[LOI] khong co {CHECK_SPLIT}")
    pts = gpd.read_file(a.points)
    pts_cv = pts[~pts["is_holdout"].astype(bool)]
    blocks_sha = file_sha256(a.blocks)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in a.seeds:
            folds_path = os.path.join(a.eval_dir, f"cv_folds_s{seed}.csv")
            folds = pd.read_csv(folds_path, dtype={"block_id": str})
            fold_of = folds.set_index("block_id")["cv_fold"].astype(int)
            fsha = file_sha256(folds_path)
            grids = sorted(glob.glob(os.path.join(a.grids_dir, "*.geojson")))
            for gi, gpath in enumerate(grids):
                name = os.path.splitext(os.path.basename(gpath))[0]
                cpath = os.path.join(a.eval_dir, "cell_blocks", f"s{seed}", f"{name}.csv")
                with open(provenance_path(cpath), encoding="utf-8") as f:
                    prov = json.load(f)
                prov_ok = (prov.get("cv_folds_sha256") == fsha and prov.get("grid_sha256") == file_sha256(gpath)
                           and prov.get("blocks_sha256") == blocks_sha)
                table = read_cell_block_table(cpath)
                df = pd.DataFrame([(c, s) for c in table["cell_id"] for s in SEASONS], columns=["cell_id", "season"])
                df["salinity"] = 1.0
                out = assign_eval_split(df, table, holdout_seasons=(2020,))
                out = out.merge(table[["cell_id", "block_id", "cv_fold"]], on="cell_id", how="left")
                out.loc[out["split"] == "test", "cv_fold"] = np.nan
                out["label_mask_id"] = "gia"
                train = out[out["split"] == "train"].reset_index(drop=True)
                splits = block_cv_splits(train.drop(columns=["block_id", "cv_fold"]), table, holdout_seasons=(2020,),
                                         rule="touch", min_val_cells=1)
                ks = sorted(int(k) for k in fold_of.unique() if k >= 0)
                mem = block_cv_membership(train, splits, ks)
                t_csv, m_csv = os.path.join(tmp, "t.csv"), os.path.join(tmp, "m.csv")
                out.to_csv(t_csv, index=False)
                mem.to_csv(m_csv, index=False)
                code, lines = run_check(t_csv, gpath, a.blocks, folds_path, m_csv)
                neg = None
                if gi == 0:  # doi chung am: hoan vi fold trong membership
                    bad = mem.copy()
                    bad["fold"] = (bad["fold"] + 1) % len(ks)
                    bad.to_csv(m_csv, index=False)
                    neg, _ = run_check(t_csv, gpath, a.blocks, folds_path, m_csv)

                # Phuong an C tren membership
                grid = gpd.read_file(gpath)
                grid["cell_id"] = grid["cell_id"].astype(str)
                hit = gpd.sjoin(pts_cv[["point_id", "block_id", "geometry"]], grid[["cell_id", "geometry"]]
                                .to_crs(pts_cv.crs), predicate="intersects", how="left").drop_duplicates("point_id")
                kp = hit["block_id"].map(fold_of).astype(int)
                role = mem.set_index(["cell_id", "fold"])["role"]
                key = list(zip(hit["cell_id"].fillna(""), kp))
                r = pd.Series([role.get(k, "khong_trong_membership") for k in key], index=hit.index)
                in_mem = r != "khong_trong_membership"
                train_per_fold = mem[mem["role"] == "train"].groupby("fold").size().reindex(ks, fill_value=0)
                row = {"seed": seed, "grid": name, "provenance_ok": prov_ok, "check_split_exit": code,
                       "check_split_neg_control_exit": neg,
                       "cv_points": len(pts_cv), "cv_points_no_cell": int(hit["cell_id"].isna().sum()),
                       "cv_points_cell_train_in_own_fold": int((r == "train").sum()),
                       "cv_points_cell_val_own_fold": int((r == "val").sum()),
                       "cv_points_cell_loai_own_fold": int((r == "loai").sum()),
                       "cv_points_cell_touch_holdout": int((~in_mem & hit["cell_id"].notna()).sum())}
                row["cv_points_scorable_C"] = row["cv_points"] - row["cv_points_no_cell"] - row[
                    "cv_points_cell_train_in_own_fold"]
                for k in ks:
                    row[f"train_cells_f{k}"] = int(train_per_fold[k])
                rows.append(row)
                log(f"s{seed} {name}: check_split {code} (doi chung am {neg}) prov {prov_ok} | train/fold "
                    f"{train_per_fold.tolist()} | diem C cham duoc {row['cv_points_scorable_C']}/{row['cv_points']} "
                    f"(o train fold minh {row['cv_points_cell_train_in_own_fold']}, o cham giu rieng "
                    f"{row['cv_points_cell_touch_holdout']})")
                if code != 0:
                    log("\n".join(lines[-15:]))
    res = pd.DataFrame(rows)
    if a.report:
        os.makedirs(os.path.dirname(a.report), exist_ok=True)
        res.to_csv(a.report, index=False)
        log(f"Bang -> {a.report}")
    bad = (res["check_split_exit"] != 0) | ~res["provenance_ok"] | (res["cv_points_cell_train_in_own_fold"] > 0) \
        | (res["cv_points_no_cell"] > 0) | res["check_split_neg_control_exit"].isin([0])
    if bad.any():
        raise SystemExit(f"[LOI] {int(bad.sum())} (phuong an, luoi) khong dat")
    log("TAT CA DAT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--blocks", default=os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    ap.add_argument("--points", default=os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--report", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_cv_kiem_membership.csv"))
    main(ap.parse_args())

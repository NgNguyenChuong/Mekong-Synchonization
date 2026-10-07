#!/usr/bin/env python
"""E5d Dot 7: bang hop nhat theo BIEN MUC TIEU (ndwi, rain_chirps, dsr_mcd18, t2m_era5, rh_era5) cho 13 luoi.

Moi bang = bang hop nhat bo chinh do man (<DATA_ROOT>/features/unified/<luoi>_unified.csv: dac trung CHUNG, ERA5
ban lap CHG-11, tinh, thuy van, hybrid, tam phan dat cho IDW) + nhan moi (<DATA_ROOT>/labels/dot7/
<luoi>_labels_season_<bien>.csv) - quy tac ghep: src/unified_table.target_table:
  - bo cot nhan do man (salinity, n_valid_px, valid_frac, train_ok, train_ok_10pct); train_ok_scope goc ->
    train_ok_scope_sal; bo cot dac trung CAM THEO BIEN (cung dai luong vat ly, training.features.TARGET_FORBIDDEN);
  - train_ok_scope = train_ok_scope_sal VA nhan bien moi huu han.
Kiem (LOI -> dung): khoa trung tuyet doi; NDWI: n_valid_px/train_ok trung bo do man (cung mat na); dac trung mac
dinh sau khi bo cam deu co trong bang, khong khop mau cam chung/theo bien, khong +-inf; NaN dac trung trong dong
huan luyen ngoai danh sach cho phep (build_unified_table.ALLOWED_TRAIN_NAN) -> LOI.
Dau ra <out-root>/<bien>/<luoi>_unified.csv (+ .provenance.json: target, label_set chinh, sha256 nguon, dac trung
mac dinh, cot da bo) - dung truc tiep: train.py --target <bien> --table ... / run_experiments.py --target <bien>
--tables-dir <out-root>/<bien>. Bao cao -> KE_HOACH/ket-qua/dot7_e5_bang_hop_nhat.csv (dong, dong huan luyen,
dong/o bi loai vi nhan NaN theo luoi).

Chay:  venv/Scripts/python.exe scripts/build_unified_dot7.py [--targets ndwi rain_chirps dsr_mcd18 t2m_era5 rh_era5]
           [--grids h3_res_5 ...] [--out-root <DATA_ROOT>/features/unified_dot7]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_unified_table import unexpected_train_nan  # noqa: E402
from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402
from training.features import (DEFAULT_ALLOWED_FEATURES, drop_target_forbidden, find_leak_columns,  # noqa: E402
                               find_target_leak_columns, resolve_feature_list)
from unified_table import SAL_TRAIN_COL, TRAIN_COL, finite_or_nan, target_table  # noqa: E402

DATA = data_path()
# bien -> cot chat luong cua nhan (KHONG phai dac trung)
TARGET_LABEL_COLS = {
    "ndwi": ("n_valid_px", "valid_frac", "train_ok", "train_ok_10pct"),
    "rain_chirps": ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"),
    "dsr_mcd18": ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"),
    "t2m_era5": ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"),
    "rh_era5": ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_one(a, grid, target):
    base_p = os.path.join(a.base_dir, f"{grid}_unified.csv")
    lab_p = os.path.join(a.labels_dir, f"{grid}_labels_season_{target}.csv")
    for p in (base_p, lab_p):
        if not os.path.exists(p):
            raise SystemExit(f"[LOI] thieu {p}")
    with open(base_p + ".provenance.json", encoding="utf-8") as f:
        bprov = json.load(f)
    if bprov.get("target") != "salinity" or bprov.get("label_set") != "chinh":
        raise SystemExit(f"[LOI] {base_p}: khong phai bang bo chinh do man (target {bprov.get('target')}, "
                         f"label_set {bprov.get('label_set')})")
    lprov_p = lab_p + ".provenance.json"
    if not os.path.exists(lprov_p):
        raise SystemExit(f"[LOI] {lab_p}: thieu provenance")
    with open(lprov_p, encoding="utf-8") as f:
        if json.load(f).get("target") != target:
            raise SystemExit(f"[LOI] {lab_p}: provenance target khac '{target}'")
    base = pd.read_csv(base_p, dtype={"cell_id": str})
    lab = pd.read_csv(lab_p, dtype={"cell_id": str})
    try:
        table, dropped = target_table(base, lab, target, TARGET_LABEL_COLS[target])
    except ValueError as exc:
        raise SystemExit(f"[LOI] {grid}/{target}: {exc}")
    if target == "ndwi":   # cung raster + cung mat na -> so pixel hop le va co huan luyen trung bo do man
        b = base.sort_values(["cell_id", "season"]).reset_index(drop=True)
        for c in ("n_valid_px", "train_ok"):
            if not np.array_equal(b[c].to_numpy(), table[c].to_numpy()):
                raise SystemExit(f"[LOI] {grid}/ndwi: cot {c} khac bo do man - mat na khong trung")
    allowed, _ = drop_target_forbidden(DEFAULT_ALLOWED_FEATURES, target)
    feats, absent = resolve_feature_list(table.columns, allowed)
    if absent:
        raise SystemExit(f"[LOI] {grid}/{target}: dac trung mac dinh vang mat {absent}")
    bad = find_leak_columns(feats) + find_target_leak_columns(feats, target)
    if bad:
        raise SystemExit(f"[LOI] {grid}/{target}: dac trung khop mau cam {bad}")
    inf = [c for c in feats if not finite_or_nan(table[c])]
    if inf:
        raise SystemExit(f"[LOI] {grid}/{target}: cot co +-inf {inf}")
    ok = table[TRAIN_COL].astype(bool)
    bad_nan = unexpected_train_nan(table, feats, ok)
    if bad_nan:
        raise SystemExit(f"[LOI] {grid}/{target}: NaN ngoai danh sach cho phep trong dong {TRAIN_COL}: {bad_nan}")
    out_dir = os.path.join(a.out_root, target)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{grid}_unified.csv")
    table.to_csv(out + ".part", index=False, float_format="%.6g")
    os.replace(out + ".part", out)
    write_provenance(out, CANONICAL_BOUNDARY, grid=grid, target=target, label_set="chinh",
                     sources={"base": {"path": base_p, "sha256": file_sha256(base_p)},
                              "labels": {"path": lab_p, "sha256": file_sha256(lab_p)}},
                     key_cols=["cell_id", "season"], quality_cols=list(TARGET_LABEL_COLS[target]) + [TRAIN_COL, SAL_TRAIN_COL],
                     default_features=feats, dropped_target_forbidden=dropped,
                     rule=f"bang bo chinh do man - cot nhan do man - cot cam theo bien + nhan {target}; "
                          f"{TRAIN_COL} = {SAL_TRAIN_COL} VA nhan {target} huu han")
    sal = table[SAL_TRAIN_COL].astype(bool)
    lost = sal & table[target].isna()
    row = {"target": target, "grid": grid, "rows": len(table), "cells": table["cell_id"].nunique(),
           "rows_label": int(table[target].notna().sum()), "rows_train_sal": int(sal.sum()),
           "rows_train": int(ok.sum()), "rows_lost_label_nan": int(lost.sum()),
           "cells_lost_any_season": int(table.loc[lost, "cell_id"].nunique()),
           "cells_label_nan_all_seasons": int((~table.groupby("cell_id")[target].apply(lambda s: s.notna().any())).sum()),
           "n_features": len(feats), "dropped": " ".join(dropped),
           "target_min_train": float(table.loc[ok, target].min()), "target_median_train": float(table.loc[ok, target].median()),
           "target_max_train": float(table.loc[ok, target].max())}
    log(f"{target}/{grid}: {len(table)} dong, huan luyen {row['rows_train']} (bo do man {row['rows_train_sal']}, "
        f"mat {row['rows_lost_label_nan']}), {len(feats)} dac trung, bo {dropped}")
    return row


def main(a):
    t0 = time.time()
    grids = sorted(os.path.basename(p)[:-len("_unified.csv")]
                   for p in glob.glob(os.path.join(a.base_dir, "*_unified.csv")))
    if a.grids:
        miss = sorted(set(a.grids) - set(grids))
        if miss:
            raise SystemExit(f"[LOI] khong co bang goc cho luoi {miss}")
        grids = [g for g in grids if g in a.grids]
    if len(grids) != 13 and not a.grids:
        raise SystemExit(f"[LOI] can 13 bang goc, thay {len(grids)}")
    os.makedirs(a.out_root, exist_ok=True)
    free = shutil.disk_usage(a.out_root).free / 1e9
    log(f"Cho trong {a.out_root}: {free:.1f} GB")
    if free < 1.0:
        raise SystemExit("[LOI] o dich con < 1 GB")
    rows = [build_one(a, g, t) for t in a.targets for g in grids]
    rep = pd.DataFrame(rows)
    os.makedirs(a.report_dir, exist_ok=True)
    p = os.path.join(a.report_dir, "dot7_e5_bang_hop_nhat.csv")
    if os.path.exists(p):  # gop: dong moi thay dong cu cung (target, grid)
        old = pd.read_csv(p)
        keep = ~old.set_index(["target", "grid"]).index.isin(rep.set_index(["target", "grid"]).index)
        rep = pd.concat([old[keep], rep], ignore_index=True)
    rep.to_csv(p, index=False, float_format="%.6g")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 100):
        print(rep[["target", "grid", "rows", "rows_train_sal", "rows_train", "rows_lost_label_nan",
                   "cells_lost_any_season", "n_features", "dropped"]].to_string(index=False), flush=True)
    log(f"Ghi {p}. Xong trong {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", nargs="+", choices=list(TARGET_LABEL_COLS), default=list(TARGET_LABEL_COLS))
    ap.add_argument("--grids", nargs="*")
    ap.add_argument("--base-dir", default=f"{DATA}/features/unified")
    ap.add_argument("--labels-dir", default=f"{DATA}/labels/dot7")
    ap.add_argument("--out-root", default=f"{DATA}/features/unified_dot7")
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    main(ap.parse_args())

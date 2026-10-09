#!/usr/bin/env python
"""Bang hop nhat theo bien muc tieu Dot 7 (13 luoi) = bang bo chinh do man - cot nhan do man - cot cam theo bien
+ nhan moi (src/unified_table.target_table); ghi <DATA_ROOT>/features/unified_dot7/<bien>/<luoi>_unified.csv.
Mua NaN theo quy tac nhan (provenance nhan 'nan_seasons') -> khai bao 'holdout_season_nan' trong provenance bang.
Chay:  venv/Scripts/python.exe scripts/build_unified_dot7.py [--targets ndwi rain_chirps ...] [--grids ...]
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
from training.split import EMPTY_DECL_KEY  # noqa: E402
from unified_table import SAL_TRAIN_COL, TRAIN_COL, finite_or_nan, target_table  # noqa: E402

DATA = data_path()
# cot chat luong cua nhan (khong phai dac trung)
TARGET_LABEL_COLS = {
    "ndwi": ("n_valid_px", "valid_frac", "train_ok", "train_ok_10pct"),
    "rain_chirps": ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"),
    "dsr_mcd18": ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"),
    "t2m_era5": ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"),
    "rh_era5": ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"),
}

# ly do bat buoc cho moi bien co mua NaN theo quy tac nhan
NAN_SEASON_REASON = {"dsr_mcd18": "E4 (b): MCD18A1 thieu 03-31/12/2019"}


def holdout_nan_declaration(lprov: dict, table: pd.DataFrame, target: str) -> dict | None:
    """Khai bao mua NaN co chu dich (None neu nhan khong co 'nan_seasons'). Thieu ly do, hoac bang con nhan huu han /
    dong huan luyen o mua do -> SystemExit."""
    seasons = sorted(int(x) for x in lprov.get("nan_seasons") or [])
    if not seasons:
        return None
    if target not in NAN_SEASON_REASON:
        raise SystemExit(f"[LOI] {target}: nhan co nan_seasons {seasons} nhung chua ghi ly do (NAN_SEASON_REASON)")
    m = table["season"].astype(int).isin(seasons)
    if not m.any():
        raise SystemExit(f"[LOI] {target}: bang khong co dong nao cua mua {seasons}")
    n_val, n_train = int(table.loc[m, target].notna().sum()), int(table.loc[m, TRAIN_COL].astype(bool).sum())
    if n_val or n_train:
        raise SystemExit(f"[LOI] {target}: mua {seasons} khai bao NaN nhung con {n_val} nhan huu han, "
                         f"{n_train} dong {TRAIN_COL}")
    return {"target": target, "seasons": seasons, "reason": NAN_SEASON_REASON[target],
            "source": "provenance nhan: nan_seasons"}


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
        lprov = json.load(f)
    if lprov.get("target") != target:
        raise SystemExit(f"[LOI] {lab_p}: provenance target khac '{target}'")
    base = pd.read_csv(base_p, dtype={"cell_id": str})
    lab = pd.read_csv(lab_p, dtype={"cell_id": str})
    try:
        table, dropped = target_table(base, lab, target, TARGET_LABEL_COLS[target])
    except ValueError as exc:
        raise SystemExit(f"[LOI] {grid}/{target}: {exc}")
    if target == "ndwi":   # cung raster + mat na voi do man -> n_valid_px, train_ok phai trung
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
    decl = holdout_nan_declaration(lprov, table, target)
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
                     **({EMPTY_DECL_KEY: decl} if decl else {}),
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

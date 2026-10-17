#!/usr/bin/env python
"""Bang hop nhat cho bo dac trung b_mua (CHG-26): ban sao bang do man / 5 bien Dot 7 them zos_mua_mean (TB khong gian
zos_mouth_p90 cua moi o luoi theo mua) va sluice_flag (season >= SLUICE_FIRST_SEASON); ghi <DATA_ROOT>/features/unified_bmua/<bien>/.
Chay:  venv/Scripts/python.exe scripts/build_bmua_tables.py [--targets salinity ndwi ...] [--grids h3_res_5 ...]
"""
import argparse
import datetime
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
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from river_graph import SLUICE_FIRST_SEASON, sluice_active  # noqa: E402
from training.dot7_rules import file_sha256, provenance_path  # noqa: E402

DATA = data_path()
TARGETS = ("salinity", "ndwi", "rain_chirps", "dsr_mcd18", "t2m_era5", "rh_era5")
NEW_COLS = ("zos_mua_mean", "sluice_flag")
FORMULA = {"zos_mua_mean": "mean(zos_mouth_p90) tren moi dong (o) cua bang cung season, bo NaN",
           "sluice_flag": f"1 neu season >= {SLUICE_FIRST_SEASON} (river_graph.SLUICE_FIRST_SEASON) else 0"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_bmua_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Them 2 cot b_mua; cot cu giu nguyen. Du lieu trai voi dinh nghia thoi ky cong / mua thieu zos -> ValueError."""
    have = [c for c in NEW_COLS if c in df.columns]
    if have:
        raise ValueError(f"bang da co cot {have}")
    season = df["season"].astype(int)
    on = df["sluice_frac"] > 0
    if on.any() and season[on].min() != SLUICE_FIRST_SEASON:
        raise ValueError(f"sluice_frac > 0 tu mua {season[on].min()}, khac SLUICE_FIRST_SEASON {SLUICE_FIRST_SEASON}")
    z = df.groupby(season)["zos_mouth_p90"].transform("mean")
    if z.isna().any():
        raise ValueError(f"mua khong co zos_mouth_p90 huu han: {sorted(season[z.isna()].unique())}")
    out = df.copy()
    out["zos_mua_mean"] = z.to_numpy()
    out["sluice_flag"] = sluice_active(season).astype("int8").to_numpy()
    check_bmua(out)
    return out


def check_bmua(df: pd.DataFrame):
    n = df.groupby("season")["zos_mua_mean"].nunique(dropna=False)
    if (n != 1).any() or df["zos_mua_mean"].isna().any():
        raise ValueError(f"zos_mua_mean khong dung 1 gia tri/mua: {n[n != 1].to_dict()}")
    if not df["sluice_flag"].isin([0, 1]).all():
        raise ValueError("sluice_flag ngoai {0, 1}")


def source_path(a, target, grid):
    if target == "salinity":
        return os.path.join(a.unified_dir, f"{grid}_unified.csv")
    return os.path.join(a.dot7_dir, target, f"{grid}_unified.csv")


def up_to_date(out, src_sha) -> bool:
    pp = provenance_path(out)
    if not (os.path.exists(out) and os.path.exists(pp)):
        return False
    with open(pp, encoding="utf-8") as f:
        b = json.load(f).get("bmua", {})
    return b.get("source", {}).get("sha256") == src_sha and b.get("output_sha256") == file_sha256(out)


def build_one(a, target, grid, zos_ref: dict):
    src = source_path(a, target, grid)
    if not (os.path.exists(src) and os.path.exists(provenance_path(src))):
        raise SystemExit(f"[LOI] thieu {src} hoac provenance")
    out_dir = os.path.join(a.out_root, target)
    out = os.path.join(out_dir, f"{grid}_unified.csv")
    src_sha = file_sha256(src)
    if up_to_date(out, src_sha):
        log(f"[bo qua] {target}/{grid}: da co, nguon khong doi")
        return None
    df = pd.read_csv(src, dtype={"cell_id": str})
    try:
        res = add_bmua_columns(df)
    except ValueError as exc:
        raise SystemExit(f"[LOI] {target}/{grid}: {exc}")
    zm = res.groupby("season")["zos_mua_mean"].first()
    ref = zos_ref.setdefault(grid, zm)  # moi bien cung tap o -> cung zos_mua_mean
    if not (ref.index.equals(zm.index) and np.allclose(ref.to_numpy(), zm.to_numpy(), rtol=0, atol=1e-9)):
        raise SystemExit(f"[LOI] {target}/{grid}: zos_mua_mean khac bien truoc cung luoi")
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(out + ".part", index=False)  # khong float_format: giu nguyen gia tri cot cu
    back = pd.read_csv(out + ".part", dtype={"cell_id": str})
    if not back[list(df.columns)].equals(df):
        raise SystemExit(f"[LOI] {target}/{grid}: cot cu doi gia tri sau khi ghi")
    os.replace(out + ".part", out)
    with open(provenance_path(src), encoding="utf-8") as f:
        prov = json.load(f)  # giu target / label_set / holdout_season_nan cho runner + train.py
    prov.update(output=os.path.basename(out), created=datetime.datetime.now().isoformat(timespec="seconds"),
                script=os.path.basename(__file__),
                bmua={"chg": "CHG-26", "source": {"path": src.replace("\\", "/"), "sha256": src_sha,
                                                   "provenance": provenance_path(src).replace("\\", "/")},
                      "output_sha256": file_sha256(out), "new_cols": list(NEW_COLS), "formula": FORMULA,
                      "zos_mua_mean_by_season": {str(k): float(v) for k, v in zm.items()}})
    with open(provenance_path(out), "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)
    on = res.loc[res["sluice_flag"] == 1, "season"]
    log(f"{target}/{grid}: {len(res)} dong, {res['season'].nunique()} mua, zos_mua_mean "
        f"{zm.min():.4f}..{zm.max():.4f}, sluice_flag=1 tu mua {on.min()} ({len(on)} dong)")
    return {"target": target, "grid": grid, "rows": len(res), "seasons": res["season"].nunique(),
            "zos_mua_min": zm.min(), "zos_mua_max": zm.max(), "sluice_first": int(on.min()), "rows_flag1": len(on)}


def main(a):
    grids = sorted(os.path.basename(p)[:-len("_unified.csv")] for p in glob.glob(os.path.join(a.unified_dir,
                                                                                              "*_unified.csv")))
    if a.grids:
        miss = sorted(set(a.grids) - set(grids))
        if miss:
            raise SystemExit(f"[LOI] khong co bang goc cho luoi {miss}")
        grids = [g for g in grids if g in a.grids]
    elif len(grids) != 13:
        raise SystemExit(f"[LOI] can 13 bang goc trong {a.unified_dir}, thay {len(grids)}")
    os.makedirs(a.out_root, exist_ok=True)
    free = shutil.disk_usage(a.out_root).free / 1e9
    log(f"Cho trong {a.out_root}: {free:.1f} GB")
    if free < 1.5:  # ~0,8 GB cho 6 bien x 13 luoi
        raise SystemExit("[LOI] o dich con < 1,5 GB")
    zos_ref = {}
    rows = [r for t in a.targets for g in grids if (r := build_one(a, t, g, zos_ref)) is not None]
    if rows:
        with pd.option_context("display.width", 200, "display.max_rows", 100):
            print(pd.DataFrame(rows).to_string(index=False), flush=True)
    log(f"Xong: {len(rows)} bang moi")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    ap.add_argument("--grids", nargs="*")
    ap.add_argument("--unified-dir", default=f"{DATA}/features/unified")
    ap.add_argument("--dot7-dir", default=f"{DATA}/features/unified_dot7")
    ap.add_argument("--out-root", default=f"{DATA}/features/unified_bmua")
    main(ap.parse_args())

#!/usr/bin/env python
"""Hybrid tren nam giu rieng 2020 (CHG-27): MAE nhom thoi_gian cua 3 cau hinh (a) day du, (b) khong_diem, (c) b_mua.
I = MAE(b)-MAE(a), I_mua = MAE(b)-MAE(c), I_kg = MAE(c)-MAE(a); doi dau theo khoi, Holm m = 13 luoi trong bien.
Chay: venv/Scripts/python.exe scripts/analyze_hybrid_final2020.py --target ndwi --allowed-tags nckh-dot7-e6a nckh-dot7-bmua
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from training.block_stats import signflip_test  # noqa: E402
from training.dot7_rules import guard_frozen, write_provenance, file_sha256  # noqa: E402
from training.evaluate import holm_adjust  # noqa: E402
import analyze_cv_family as acf  # noqa: E402

GRIDS = ["h3_res_5", "h3_res_6", "h3_res_7", "s2_level_9", "s2_level_10", "s2_level_11", "s2_level_12",
         "latlon_0.1552deg", "latlon_0.0586deg", "latlon_0.0222deg",
         "square_utm_17087m", "square_utm_6458m", "square_utm_2441m"]
CFG = {"a": None, "b": "khong_diem", "c": "b_mua"}
COMP = {"I": ("b", "a"), "I_mua": ("b", "c"), "I_kg": ("c", "a")}


def run_dir(exp_root, grid, fs, target):
    name = f"cv1__{grid}__hist_gb__s42" + (f"__fs-{fs}" if fs else "") + (f"__t-{target}" if target != "salinity" else "")
    return os.path.join(exp_root, name, "final")


def load_final(d, allowed, tags):
    meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
    if meta.get("returncode") != 0 or meta.get("mode") != "final":
        raise SystemExit(f"LOI: {d}: returncode/mode sai")
    if allowed and meta.get("git_tag") not in allowed:
        raise SystemExit(f"LOI: {d}: tag {meta.get('git_tag')} ngoai danh sach {allowed}")
    tags.add(meta.get("git_tag"))
    p = pd.read_csv(os.path.join(d, "final_points.csv"), usecols=["point_id", "season", "block_id", "err", "test_group"])
    p = p[p["test_group"] == "thoi_gian"]
    if not len(p) or not np.isfinite(p["err"]).all():
        raise SystemExit(f"LOI: {d}: nhom thoi_gian rong hoac err khong huu han")
    return p.set_index(["point_id", "season"])["err"].abs()


def block_mae(errs, blocks):
    """errs: dict cfg -> Series |err| cung index; tra ve MAE theo khoi (bang) + trong so n diem."""
    df = pd.DataFrame(errs)
    if df.isna().any().any():
        raise SystemExit("LOI: 3 cau hinh khong cung tap (diem, mua)")
    df["blk"] = blocks.reindex(df.index).values
    g = df.groupby("blk")
    return g[list(errs)].mean(), g.size().astype(float)


def analyze(target, exp_root, allowed):
    tags, rows = set(), []
    for grid in GRIDS:
        errs, blk = {}, None
        for cfg, fs in CFG.items():
            d = run_dir(exp_root, grid, fs, target)
            s = load_final(d, allowed, tags)
            errs[cfg] = s
            if blk is None:
                blk = pd.read_csv(os.path.join(d, "final_points.csv"), usecols=["point_id", "season", "block_id"]
                                  ).set_index(["point_id", "season"])["block_id"]
        mae_b, w = block_mae(errs, blk)
        n = float(w.sum())
        for comp, (x, y) in COMP.items():
            dvec = (mae_b[x] - mae_b[y]).values
            r = signflip_test(dvec, w.values)
            mae_all = {c: float((errs[c]).mean()) for c in errs}
            rows.append({"target": target, "thanh_phan": comp, "grid": grid, "G": r["G"], "n_diem_mua": int(n),
                         "mae_a": mae_all["a"], "mae_b": mae_all["b"], "mae_c": mae_all["c"],
                         "I": r["delta_hat"], "I_tuong_doi": r["delta_hat"] / mae_all["b"], "p_value": r["p_value"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for comp in COMP:
        m = out["thanh_phan"] == comp
        out.loc[m, "p_holm"] = holm_adjust(out.loc[m, "p_value"].values)
    out["co_y_nghia"] = (out["p_holm"] < 0.05) & (out["I"] > 0)
    if target == "salinity":  # mo ta, khong kiem dinh (ket qua do man dong bang)
        out[["p_value", "p_holm"]] = np.nan
        out["co_y_nghia"] = np.nan
    return out, sorted(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--exp-root", default=acf.EXP_ROOT)
    ap.add_argument("--out-dir", default=acf.RESULTS)
    ap.add_argument("--allowed-tags", nargs="+", default=None)
    ap.add_argument("--frozen-manifest", nargs="+", default=acf.FROZEN_MANIFESTS)
    ap.add_argument("--no-table", action="store_true")
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, f"dot7_{a.target}_hybrid_final2020.csv")
    guard_frozen([out_csv, out_csv + ".provenance.json"], a.frozen_manifest)
    out, tags = analyze(a.target, a.exp_root, a.allowed_tags)
    out.to_csv(out_csv, index=False)
    write_provenance(out_csv, target=a.target, git_tags_seen=tags, git_tags_allowed=a.allowed_tags, holm_m=13,
                     quy_tac="CHG-27 NHAT_KY 2026-10-10 09:54:33", mode="final", nhom="thoi_gian", scheme=42,
                     cv_folds_sha256=file_sha256(os.path.join(ROOT, "data", "eval", "cv_folds.csv")))
    print("Ghi:", out_csv, "| tags", tags)
    if not a.no_table:
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

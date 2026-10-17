#!/usr/bin/env python
"""Q5 (CHG-31): I = MAE(khong_quang_hoc) - MAE(day du) theo khoi, 13 luoi, Holm m = 13, quy tac D, CI90, TOST 5 % level_mean;
va ho F1 (12 cap cung co o) tren chinh feature set (d).

Chay: venv/Scripts/python.exe scripts/analyze_q5.py --allowed-tags nckh-cv1 nckh-dot7-e6a nckh-dot7-q5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import analyze_hybrid as ah  # noqa: E402
from analyze_cv_family import (EXP_ROOT, FROZEN_MANIFESTS, GRIDS, area_levels, area_table, check_consistent,  # noqa: E402
                                pairs_by_area, point_units, run_family, tag_info)
from training.dot7_rules import file_sha256, guard_frozen, provenance_path, write_csv_atomic, write_provenance  # noqa: E402

FS = "khong_quang_hoc"
REASON = "(a) luot day du tag cu (nckh-cv1/nckh-dot7-e6a), (d) khong_quang_hoc tag nckh-dot7-q5; so OOF cung tap (diem, mua)"


def thr_by_grid(imp, levels, frac):
    """Delta_min = frac x trung binh MAE(a) cua muc (trung binh cac luoi trong muc)."""
    mae = pd.Series({g: float(np.average(m["mae_a"], weights=m["w"])) for g, m in imp.items()})
    lv_mean = mae.groupby(levels.reindex(mae.index)).mean()
    return levels.reindex(mae.index).map(lv_mean) * frac


def table(imp, levels, tag, frac):
    s1 = ah.step1(imp, tag)
    thr = thr_by_grid(imp, levels, frac)
    s1["muc"] = s1["grid"].map(levels)
    s1["nguong_tost"] = s1["grid"].map(thr)
    s1["tuong_duong"] = ((s1["ci90_low"] > -s1["nguong_tost"]) & (s1["ci90_high"] < s1["nguong_tost"])).astype(object)
    s1.loc[s1["nguong_tost"].isna(), "tuong_duong"] = pd.NA   # luoi khong thuoc muc nao (s2_9/10/11): khong ap dung TOST
    return s1


def label(r):
    if r["p_holm"] < ah.ALPHA and r["cung_chieu_3"]:
        return "khac_vung_D"
    if r["p_holm"] < ah.ALPHA:
        return "khac_khong_vung_theo_cach_chia"
    if pd.isna(r["tuong_duong"]):
        return "khong_ap_dung_tost"
    return "tuong_duong" if r["tuong_duong"] else "chua_phan_dinh"


def main(a):
    out = os.path.join(a.out_dir, "dot7_q5_khong_quang_hoc.csv")
    out_f1 = os.path.join(a.out_dir, "dot7_q5_cap_khung_fs_khong_quang_hoc.csv")
    guard_frozen([out, provenance_path(out), out_f1, provenance_path(out_f1)], a.frozen_manifest)
    at = area_table(a.area_table)
    pairs = pairs_by_area(at, max_ratio=1.2, expected_m=12)
    levels = area_levels(at, pairs=pairs)
    pu, infos = ah._point_unit(), []
    full = {g: ah.load(a.exp_root, a.prefix, g, a.schemes, None, "salinity", infos) for g in GRIDS}
    abl = {g: ah.load(a.exp_root, a.prefix, g, a.schemes, FS, "salinity", infos) for g in GRIDS}
    ah.assert_same_keys({f"(a) {g}": full[g] for g in GRIDS} | {f"(d) {g}": abl[g] for g in GRIDS}, pu)
    imp = {g: ah.unit_improvement(full[g], abl[g], pu) for g in GRIDS}
    main_t = table(imp, levels, FS, a.delta_min)
    for s in a.schemes:
        imp_s = {g: ah.unit_improvement(full[g][full[g]["seed"] == s], abl[g][abl[g]["seed"] == s], pu) for g in GRIDS}
        ts = table(imp_s, levels, f"{FS}|s{s}", a.delta_min)
        main_t[f"I_s{s}"], main_t[f"p_holm_s{s}"] = ts["I"].to_numpy(), ts["p_holm"].to_numpy()
    sg = np.sign(main_t["I"].to_numpy())
    main_t["cung_chieu_3"] = np.logical_and.reduce([(main_t[f"p_holm_s{s}"] < ah.ALPHA).to_numpy()
                                                    & (np.sign(main_t[f"I_s{s}"]).to_numpy() == sg) for s in a.schemes])
    main_t["nhan"] = main_t.apply(label, axis=1)
    # ho F1 tren feature set (d): 12 cap cung co o
    err = pd.concat([abl[g].assign(model="hist_gb") for g in GRIDS], ignore_index=True)
    err["pred_source"] = "oof"
    folds = os.path.join(ROOT, "data", "eval", "cv_folds.csv")
    pu2, cv_units, ho_units = point_units(err, folds)
    f1 = run_family(err, pu2, pairs, levels, cv_units, ho_units, "hist_gb", "F1_fs_khong_quang_hoc", 0.05,
                    a.delta_min, expected_m=12)
    same = check_consistent(infos, a.allowed_tags)
    write_csv_atomic(main_t, out)
    write_csv_atomic(f1, out_f1)
    for p in (out, out_f1):
        write_provenance(p, target="salinity", chg="CHG-31", prefix=a.prefix, schemes=list(a.schemes), grids=list(GRIDS),
                         feature_set=FS, holm_m=len(GRIDS) if p == out else len(pairs), delta_min_frac=a.delta_min,
                         delta_ref="level_mean MAE(a)" if p == out else "level_mean cua (d)", git_tag=same["git_tag"],
                         points_ref_sha256=same["points_ref_sha256"], n_luot=len(infos),
                         area_table_sha256=file_sha256(a.area_table), **tag_info(same, a.allowed_tags, REASON))
    print(f"Ghi: {out}\n     {out_f1}", flush=True)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(main_t[["grid", "muc", "I", "I_tuong_doi", "ci90_low", "ci90_high", "nguong_tost", "p_holm",
                      "cung_chieu_3", "tuong_duong", "nhan"]].to_string(index=False))
        print(f1.drop(columns=[c for c in f1.columns if c.startswith("n_")]).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--delta-min", type=float, default=0.05)
    ap.add_argument("--area-table", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_doi_chieu_dien_tich.csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--exp-root", default=EXP_ROOT)
    ap.add_argument("--frozen-manifest", nargs="+", default=FROZEN_MANIFESTS)
    ap.add_argument("--allowed-tags", nargs="+", default=None)
    main(ap.parse_args())

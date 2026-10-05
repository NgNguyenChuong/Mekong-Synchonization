#!/usr/bin/env python
"""Kiem dinh so sanh khung luoi THEO KHOI tren ket qua CV (mode cv) - CHG-06, A2-A4, D.

Thiet ke da chot TRUOC (NHAT_KY 2026-10-03/04):
  - Ho chinh F1: HistGB, MAE, cap CUNG muc co ti le dien tich o TB <= 1,2 -> 12 cap; Holm m = 12; alpha 0,05.
  - Don vi kiem dinh = don vi CV (khoi 50 km, khoi < 30 diem gop vao khoi ke: cv_folds.csv unit_id) - 19 don vi.
  - "seed" = 3 cach gan khoi -> fold (42, 43, 44): sai so trung binh qua 3 cach; nhan can cung chieu ca 3.
  - Delta_min = 5% MAE tuong doi, MOT so moi muc (level_mean tren cac luoi trong ho).
  - Mode cv: chua co khoi giu rieng -> nhan cao nhat "cho_giu_rieng" (Holm dat + cung chieu moi cach chia).
Phu (khong vao Holm cua F1): linear, IDW (cung 12 cap); cap S2 L9 voi luoi tho (ti le 1,35); do nhay tung cach
chia rieng. Tap diem chung tinh THEO MO HINH (An 2026-10-04). Delta < 0: luoi A sai so THAP hon B.

Chay:  venv/Scripts/python.exe scripts/analyze_cv_family.py --prefix cv1 [--out-dir KE_HOACH/ket-qua]
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from training.block_stats import area_levels, common_point_set, compare_family, pairs_by_area  # noqa: E402

GRIDS = ["h3_res_5", "h3_res_6", "h3_res_7", "latlon_0.0222deg", "latlon_0.0586deg", "latlon_0.1552deg",
         "s2_level_9", "s2_level_10", "s2_level_11", "s2_level_12", "square_utm_17087m", "square_utm_2441m",
         "square_utm_6458m"]


def area_table(path):
    t = pd.read_csv(path)
    t["grid"] = t["file"].map(lambda f: os.path.splitext(os.path.basename(str(f)))[0])
    return t


def load_errors(exp_root, prefix, model, schemes, grids=GRIDS):
    """err_long (grid, model, seed=cach chia, point_id, season, err, pred_source, unit_id) tu oof_points.csv."""
    parts = []
    for g in grids:
        for s in schemes:
            p = os.path.join(exp_root, f"{prefix}__{g}__{model}__s{s}", "cv", "oof_points.csv")
            d = pd.read_csv(p, dtype={"point_id": str, "unit_id": str},
                            usecols=["grid", "point_id", "season", "unit_id", "err", "pred_source"])
            if (d["pred_source"] != "oof").any():
                raise ValueError(f"{p}: co dong khong phai oof")
            d["model"], d["seed"] = model, s
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def point_units(err, folds_csv):
    """point_id -> unit_id cho MOI diem danh gia (gom diem giu rieng - chi de doi chieu don vi, khong dung gia tri),
    tinh lai tu vi tri diem (point_eval.point_blocks); kiem khop unit_id trong oof_points."""
    import geopandas as gpd

    from training.point_eval import point_blocks

    folds = pd.read_csv(folds_csv, dtype={"block_id": str, "unit_id": str})
    pts = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    blocks = gpd.read_file(os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    pb = point_blocks(pts, blocks, folds).set_index("point_id")
    pu = pb["unit_id"].astype(str)
    got = err.drop_duplicates(["point_id", "unit_id"]).set_index("point_id")["unit_id"]
    if got.index.duplicated().any() or (got != pu.reindex(got.index)).any():
        raise ValueError("unit_id trong oof_points khac don vi tinh tu vi tri diem")
    cv_units = sorted(folds.loc[folds["cv_fold"] >= 0, "unit_id"].unique())
    ho_units = sorted(folds.loc[folds["cv_fold"] < 0, "unit_id"].unique())
    return pu, cv_units, ho_units


def run_family(err, pu, pairs, levels, cv_units, ho_units, model, family, alpha, delta_min):
    valid = err.assign(valid=err["err"].notna())[["grid", "point_id", "season", "valid"]].drop_duplicates(
        ["grid", "point_id", "season"])
    common, dropped = common_point_set(valid)
    key = pd.MultiIndex.from_frame(err[["point_id", "season"]])
    e = err[key.isin(common)]
    out = compare_family(e, pu, pairs, cv_units=cv_units, holdout_units=ho_units, holdout_seasons=(2020,),
                         mode="cv", delta_min=delta_min, delta_min_kind="rel", alpha=alpha,
                         delta_ref="level_mean", levels=levels, require_practical=True, model=model,
                         family=family, expected_m=len(pairs))
    out.insert(0, "n_common_point_seasons", len(common))
    return out


def main(a):
    exp_root = os.path.join(ROOT, "artifacts", "experiments")
    at = area_table(a.area_table)
    main_pairs = pairs_by_area(at, max_ratio=1.2, expected_m=12)
    levels = area_levels(at, pairs=main_pairs)
    wide = pairs_by_area(at, max_ratio=1.5)
    key_main = set(zip(main_pairs["grid_a"], main_pairs["grid_b"]))
    s2_pairs = wide[[(x, y) not in key_main for x, y in zip(wide["grid_a"], wide["grid_b"])]].reset_index(drop=True)
    folds_csv = os.path.join(ROOT, "data", "eval", "cv_folds.csv")
    results = []
    for model, fam in (("hist_gb", "F1_chinh"), ("linear", "phu_linear"), ("idw", "phu_idw")):
        err = load_errors(exp_root, a.prefix, model, a.schemes)
        pu, cv_units, ho_units = point_units(err, folds_csv)
        results.append(run_family(err, pu, main_pairs, levels, cv_units, ho_units, model, fam, a.alpha, a.delta_min))
        if model == "hist_gb":
            if len(s2_pairs):
                lv2 = area_levels(at, pairs=s2_pairs)
                results.append(run_family(err, pu, s2_pairs, lv2, cv_units, ho_units, model, "phu_S2_ti_le_1.2-1.5",
                                          a.alpha, a.delta_min))
            for s in a.schemes:  # do nhay: tung cach chia rieng (khong trung binh)
                es = err[err["seed"] == s]
                results.append(run_family(es, pu, main_pairs, levels, cv_units, ho_units, model,
                                          f"do_nhay_chi_s{s}", a.alpha, a.delta_min))
    res = pd.concat(results, ignore_index=True)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"dot5_{a.prefix}_kiem_dinh_khoi.csv")
    res.to_csv(out, index=False)
    cols = [c for c in ["family", "grid_a", "grid_b", "delta_cv", "rel_delta", "threshold", "ci_low", "ci_high",
                        "p_value", "p_holm", "seeds_same_dir", "k_over_G", "label"] if c in res.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(res[cols].to_string(index=False))
    print(f"\nGhi: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--area-table", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_doi_chieu_dien_tich.csv"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta-min", type=float, default=0.05)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    main(ap.parse_args())

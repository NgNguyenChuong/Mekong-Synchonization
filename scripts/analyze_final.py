#!/usr/bin/env python
"""Final (CHG-25 muc 4): xac nhan chieu cap F1 tren khoi giu rieng (chi nhom khong_gian) cho cap duoc kiem dinh cua
bien, con lai mo_ta; bang mo ta nhom test (khong_gian / thoi_gian / ca_hai) tu config final, nhom rong khai bao truoc
in "rong (khai bao truoc)". --holm-scope nhu analyze_cv_family; doi chieu voi file CV cung pham vi (tap cap kiem
dinh phai trung). --model rf|mlp: bang cap va bang nhom chi mo hinh do, file doc/ghi hau to __<model>.

Chay:  venv/Scripts/python.exe scripts/analyze_final.py --target ndwi [--prefix cv1] [--min-frac 0.8]
           [--model hist_gb|rf|mlp] [--holm-scope cong|muc_kiem_dinh] [--no-table]
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

import analyze_cv_family as acf  # noqa: E402
from run_experiments import run_name  # noqa: E402
from training.block_stats import area_levels, family_verdict, pairs_by_area  # noqa: E402
from training.dot7_rules import (HOLM_SCOPES, TESTED_TIERS, all_scoped_names, apply_gate,  # noqa: E402
                                 compare_tiered, expected_f1_m, file_sha256, gate_required, guard_frozen,
                                 holm_provenance, holm_tiers, load_gate, main_family, model_provenance,
                                 provenance_path, run_model, scoped_name, scopes_differ, write_csv_atomic,
                                 write_provenance)
from training.split import EMPTY_LABEL, TEST_GROUPS  # noqa: E402
NA_DESIGN_LABEL = "khong ap dung (thiet ke mo hinh)"

FINAL_SCHEME = 42
DESC_MODELS = ("hist_gb", "linear", "idw", "season_mean")
SPATIAL_GROUP = "khong_gian"


def out_names(target, prefix):
    return f"dot7_{target}_{prefix}_final_cap.csv", f"dot7_{target}_{prefix}_final_nhom.csv"


def final_dir(a, grid, model):
    name = run_name(a.prefix, grid, model, FINAL_SCHEME, "", None, a.target)
    return name, os.path.join(a.exp_root, name, "final")


def load_final_points(a, grid, infos):
    """final_points cua mo hinh --model s42, CHI nhom khong_gian (khoi giu rieng, mua != 2020); seed = cach chia."""
    model = run_model(a.model)
    name, run_dir = final_dir(a, grid, model)
    infos.append(acf.check_run(run_dir, name, a.target, "final"))
    p = os.path.join(run_dir, "final_points.csv")
    if not os.path.isfile(p):
        acf.loi(f"{name}: thieu {p}")
    d = pd.read_csv(p, dtype={"point_id": str, "unit_id": str},
                    usecols=["point_id", "season", "unit_id", "err", "pred_source", "test_group"])
    d = d[d["test_group"] == SPATIAL_GROUP].drop(columns="test_group")
    if d.empty:
        acf.loi(f"{name}: khong co dong nhom {SPATIAL_GROUP}")
    if (d["pred_source"] != "final").any():
        acf.loi(f"{name}: co dong khong phai final")
    d["grid"], d["model"], d["seed"] = grid, model, FINAL_SCHEME
    return d


def check_against_cv(res, cv_path, tol=1e-9, family="F1_chinh"):
    """Cap kiem dinh: delta_hat / p_holm (phan CV) phai khop ket qua analyze_cv_family (ho chinh cua mo hinh)."""
    if not os.path.isfile(cv_path):
        print(f"CANH BAO: khong co {cv_path} - khong doi chieu voi ket qua CV", flush=True)
        return False
    cv = pd.read_csv(cv_path)
    cv = cv[cv["family"] == family].set_index(["grid_a", "grid_b"])
    if "kiem_dinh" in cv.columns:
        k_cv = set(cv.index[cv["kiem_dinh"].astype(bool)])
        kf = res[res["kiem_dinh"].astype(bool)]
        k_fin = set(zip(kf["grid_a"], kf["grid_b"]))
        if k_cv != k_fin:
            acf.loi(f"tap cap kiem dinh final ({len(k_fin)}) khac CV ({len(k_cv)}) - khac pham vi Holm / cong?")
    for r in res[res["kiem_dinh"].astype(bool)].itertuples():
        c = cv.loc[(r.grid_a, r.grid_b)]
        if not bool(c["kiem_dinh"]):
            acf.loi(f"{r.grid_a}/{r.grid_b}: kiem dinh o final nhung mo_ta o CV")
        for k in ("delta_hat", "p_holm"):
            if not np.isclose(getattr(r, k), c[k], rtol=0, atol=tol):
                acf.loi(f"{r.grid_a}/{r.grid_b}: {k} final (phan CV) {getattr(r, k)} khac CV {c[k]}")
    return True


def _declared(meta):
    v = meta.get("empty_test_groups_declared") or []
    return set(v.split()) if isinstance(v, str) else set(v)


def group_rows(a, grid, model, level, infos):
    """Mot luot final -> dong mo ta moi nhom; nhom rong chi hop le khi khai bao truoc."""
    name, run_dir = final_dir(a, grid, model)
    info = acf.check_run(run_dir, name, a.target, "final")
    infos.append(info)
    cfg, declared = info["config"], _declared(info["meta"])
    if "point_metrics_by_group" not in cfg:
        acf.loi(f"{name}: config thieu point_metrics_by_group")
    pm_all = cfg["point_metrics_by_group"]
    n_all = cfg.get("n_point_rows_by_group", {})
    status_all = cfg.get("test_metrics_by_group_mean_over_seeds", {})
    note = cfg.get("notes", {}).get("nhom_test_rong", "")
    rows = []
    for g in TEST_GROUPS:
        pm = pm_all.get(g) or {}
        vals = [m for m in pm.values() if m]
        n = int(n_all.get(g, 0) or 0)
        status = (status_all.get(g) or {}).get("status")
        empty = n == 0 or not vals
        if g in declared or status == EMPTY_LABEL:
            if not empty:
                acf.loi(f"{name}: nhom {g} khai bao rong nhung co {n} dong")
            rows.append({"nhom": g, "n_diem": 0, "MAE_diem": np.nan, "R2_diem": np.nan, "trang_thai": EMPTY_LABEL,
                         "ghi_chu": note})
            continue
        design_note = cfg.get("notes", {}).get(model, "")
        if empty and g in ("thoi_gian", "ca_hai") and n > 0 and g in design_note:
            # IDW/season_mean: thiet ke khong du doan mua giu rieng (ghi chu trong config cua chinh luot do)
            rows.append({"nhom": g, "n_diem": n, "MAE_diem": np.nan, "R2_diem": np.nan,
                         "trang_thai": NA_DESIGN_LABEL, "ghi_chu": design_note})
            continue
        if empty:
            acf.loi(f"{name}: nhom {g} rong ma khong khai bao truoc")
        r2 = [m["r2"] for m in vals if m.get("r2") is not None]
        rows.append({"nhom": g, "n_diem": n, "MAE_diem": float(np.mean([m["mae"] for m in vals])),
                     "R2_diem": float(np.mean(r2)) if r2 else np.nan, "trang_thai": "co_du_lieu", "ghi_chu": ""})
    for r in rows:
        r.update({"target": a.target, "grid": grid, "muc": level, "model": model})
    return rows


def main(a):
    if a.target not in TESTED_TIERS:
        acf.loi(f"--target '{a.target}' khong co trong TESTED_TIERS")
    models = desc_models(a)
    os.makedirs(a.out_dir, exist_ok=True)
    cand = [p for n in out_names(a.target, a.prefix) for p in all_scoped_names(os.path.join(a.out_dir, n), a.model)]
    guard_frozen(cand + [provenance_path(p) for p in cand], a.frozen_manifest)
    gate_path = a.gate or os.path.join(a.out_dir, acf.gate_name(a.model))
    try:
        gate = load_gate(gate_path, a.target, required=gate_required(a.target, a.model), model=a.model)
    except ValueError as exc:
        acf.loi(str(exc))
    differ = scopes_differ(a.target, gate)
    out_cap, out_nhom = (os.path.join(a.out_dir, scoped_name(n, a.holm_scope, differ, a.model))
                         for n in out_names(a.target, a.prefix))
    model, fam = run_model(a.model), main_family(a.model)
    ht = holm_tiers(a.target, gate, a.holm_scope)
    at = acf.area_table(a.area_table)
    main_pairs = pairs_by_area(at, max_ratio=1.2, expected_m=12)
    levels = area_levels(at, pairs=main_pairs)
    all_levels = area_levels(at)

    infos = []
    err_cv = acf.load_errors(a.exp_root, a.prefix, model, a.schemes, target=a.target, infos=infos)
    e_cv, n_cv = acf.common_subset(err_cv)
    fin = pd.concat([load_final_points(a, g, infos) for g in acf.GRIDS], ignore_index=True)
    e_fin, n_fin = acf.common_subset(fin)
    e = pd.concat([e_cv, e_fin], ignore_index=True)
    pu, cv_units, ho_units = acf.point_units(e, a.folds_csv)
    res = compare_tiered(e, pu, main_pairs, a.target, levels=levels, expected_m=expected_f1_m(ht), holm_tiers=ht,
                         cv_units=cv_units, holdout_units=ho_units, holdout_seasons=(2020,), mode="final",
                         delta_min=a.delta_min, delta_min_kind="rel", alpha=a.alpha, delta_ref="level_mean",
                         require_practical=True, model=model, family=fam)
    res = apply_gate(res, gate)
    res, summ = family_verdict(res, min_frac=a.min_frac)
    res.insert(1, "n_cv_point_seasons", n_cv)
    res.insert(2, "n_final_point_seasons", n_fin)
    res["final_nhom"] = SPATIAL_GROUP
    cv_path = os.path.join(a.cv_dir or a.out_dir,
                           scoped_name(acf.out_name(a.target, a.prefix), a.holm_scope, differ, a.model))
    matched = check_against_cv(res, cv_path, family=fam)
    res["doi_chieu_cv"] = matched

    rows = []
    for m in models:
        for g in acf.GRIDS:
            rows += group_rows(a, g, m, all_levels.get(g), infos)
    nhom = pd.DataFrame(rows)[["target", "grid", "muc", "model", "nhom", "n_diem", "MAE_diem", "R2_diem",
                               "trang_thai", "ghi_chu"]]
    same = acf.check_consistent(infos, a.allowed_tags)
    for t in (res, nhom):
        t["git_tag"], t["points_ref_sha256"] = same["git_tag"], same["points_ref_sha256"]
    res["cong_file"] = gate_path if gate is not None else ""
    hp = holm_provenance(a.target, gate, gate_path, a.holm_scope, res)
    cv_info = {"cv_ref": os.path.abspath(cv_path).replace("\\", "/") if matched else None,
               "cv_ref_sha256": file_sha256(cv_path) if matched else None}
    for t, p in ((res, out_cap), (nhom, out_nhom)):
        write_csv_atomic(t, p)
        write_provenance(p, **hp, **cv_info, git_tag=same["git_tag"], points_ref_sha256=same["points_ref_sha256"],
                         n_luot=len(infos), **model_provenance(a.model),
                         **acf.tag_info(same, a.allowed_tags, a.tags_reason))

    if not a.no_table:
        cols = ["grid_a", "grid_b", "kiem_dinh", "delta_hat", "p_holm", "delta_holdout", "holdout_same_dir", "label"]
        with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
            print(res[cols].to_string(index=False))
        print(f"Ho F1 final: {summ}")
    empty = nhom[nhom["trang_thai"] == EMPTY_LABEL].groupby(["model", "nhom"]).size()
    for (m, g), k in empty.items():
        print(f"  {m} nhom {g}: {EMPTY_LABEL} ({k} luoi)")
    print(f"Ghi: {out_cap}, {out_nhom} | git_tag {same['git_tag']} | holm {a.holm_scope} m = {expected_f1_m(ht)}")


def desc_models(a) -> list:
    """Mo hinh cho bang nhom: hist_gb -> --models (mac dinh DESC_MODELS); rf/mlp -> chi mo hinh do."""
    if a.model == "hist_gb":
        return list(a.models or DESC_MODELS)
    own = [run_model(a.model)]
    if a.models and list(a.models) != own:
        acf.loi(f"--model {a.model}: bang nhom chi cho {own}, khong nhan --models {a.models}")
    return own


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default="salinity")
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"mo hinh cho bang mo ta nhom (hist_gb: mac dinh {' '.join(DESC_MODELS)})")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta-min", type=float, default=0.05)
    ap.add_argument("--min-frac", type=float, default=0.8, help="family_verdict: ti le cap giu chieu toi thieu")
    ap.add_argument("--area-table", default=os.path.join(acf.RESULTS, "dot4_doi_chieu_dien_tich.csv"))
    ap.add_argument("--out-dir", default=acf.RESULTS)
    ap.add_argument("--cv-dir", default=None, help="thu muc ket qua analyze_cv_family (mac dinh = --out-dir)")
    ap.add_argument("--gate", default=None, help=f"bang cong (mac dinh <out-dir>/{acf.GATE_FILE}; rf/mlp: dot7_cong_kiem_dinh__<model>.csv)")
    ap.add_argument("--exp-root", default=acf.EXP_ROOT)
    ap.add_argument("--folds-csv", default=os.path.join(ROOT, "data", "eval", "cv_folds.csv"))
    acf.add_common_args(ap)
    ap.add_argument("--holm-scope", choices=HOLM_SCOPES, default="cong")
    ap.add_argument("--no-table", action="store_true", help="khong in bang so ra stdout")
    return ap


if __name__ == "__main__":
    main(build_parser().parse_args())

#!/usr/bin/env python
"""Cong "mo hinh hoc duoc" (CHG-25 muc 3): moi (bien, luoi) so MAE(HistGB) voi MAE(season_mean), CV 3 cach chia,
kiem dinh doi dau theo khoi; Holm theo bien tren luoi cua cac muc kiem dinh. Xuat dot7_<bien>_hoc_duoc.csv va
gop dot7_cong_kiem_dinh.csv (bien, muc, hop_le) - analyze_cv_family doc bang nay de ep mo_ta.
--model rf|mlp: mo hinh do vs season_mean, ghi dot7_<bien>_hoc_duoc__<model>.csv + cong rieng
dot7_cong_kiem_dinh__<model>.csv (them cot model), khong cham cong HistGB.
--region/--pairs-mode/--exclude-points nhu analyze_cv_family; vung: m Holm cong = so luoi thuc (khong dung EXPECTED_GATE_M).

Chay:  venv/Scripts/python.exe scripts/analyze_learnability.py --targets ndwi rain_chirps dsr_mcd18 t2m_era5 rh_era5
           [--prefix cv1] [--model hist_gb|rf|mlp] [--allowed-tags TAG ...] [--scope muc_kiem_dinh|cap_f1]
           [--out-dir KE_HOACH/ket-qua] [--no-table]
       Vung:  ... --region mississippi --pairs-mode ratio --frozen-manifest <manifest...>
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import analyze_cv_family as acf  # noqa: E402
from region import DEFAULT_REGION  # noqa: E402
from training.block_stats import area_levels  # noqa: E402
from training.dot7_rules import (EXPECTED_GATE_M, GATE_SCOPES, TESTED_TIERS, gate_table, guard_frozen,  # noqa: E402
                                 learnability, merge_gate, model_name, model_provenance, provenance_path, run_model,
                                 write_csv_atomic, write_provenance)

NULL_MODEL = "season_mean"
DOT7_TARGETS = ("ndwi", "rain_chirps", "dsr_mcd18", "t2m_era5", "rh_era5")


def out_name(target, model="hist_gb"):
    return model_name(f"dot7_{target}_hoc_duoc.csv", model)


def run_target(a, target, levels, pair_grids):
    infos = []
    model = run_model(a.model)
    err = pd.concat([acf.load_errors(a.exp_root, a.prefix, m, a.schemes, target=target, infos=infos,
                                     region=a.region, exclude=a.exclude)
                     for m in (model, NULL_MODEL)], ignore_index=True)
    e, n_common = acf.common_subset(err)
    pu, cv_units, ho_units = acf.point_units(e, a.folds_csv, **a.eval_kw)
    acf.check_excluded(a.exclude, pu)
    acf.warn_g(a.region, cv_units)
    hoc = learnability(e, pu, acf.GRIDS, levels, target, cv_units=cv_units, holdout_units=ho_units,
                       holdout_seasons=(2020,), alpha=a.alpha, scope=a.scope, pair_grids=pair_grids,
                       model=model, null_model=NULL_MODEL, n_schemes=len(a.schemes),
                       expected_m=EXPECTED_GATE_M[a.scope][target] if a.region == DEFAULT_REGION else None)
    same = acf.check_consistent(infos, a.allowed_tags)
    hoc.insert(2, "n_common_point_seasons", n_common)
    gate = gate_table(hoc, target)
    if a.model != "hist_gb":
        gate.insert(1, "model", a.model)
    for t in (hoc, gate):
        t["git_tag"], t["points_ref_sha256"] = same["git_tag"], same["points_ref_sha256"]
    info = {"git_tag": same["git_tag"], "points_ref_sha256": same["points_ref_sha256"], "n_luot": len(infos),
            **acf.tag_info(same, a.allowed_tags, a.tags_reason)}
    if a.region != DEFAULT_REGION:
        info.update(acf.region_info(a, cv_units, hoc), holm_m_cong=int(hoc["holm_m"].max()))
    return hoc, gate, info


def main(a):
    bad = [t for t in a.targets if t not in TESTED_TIERS]
    if bad:
        acf.loi(f"target khong hop le {bad}")
    acf.resolve_region(a)
    os.makedirs(a.out_dir, exist_ok=True)
    outs = {t: os.path.join(a.out_dir, out_name(t, a.model)) for t in a.targets}
    gate_path = os.path.join(a.out_dir, acf.gate_name(a.model))
    files = [*outs.values(), gate_path]
    guard_frozen(files + [provenance_path(p) for p in files], a.frozen_manifest)
    sidecar = a.model != "hist_gb" or bool(a.allowed_tags) or a.region != DEFAULT_REGION  # dbscl hist_gb: nhu cu
    at = acf.area_table(a.area_table)
    levels = area_levels(at)  # moi luoi cua muc (gom s2_level_9/10/11)
    main_pairs, _ = acf.select_pairs(at, a.region, a.pairs_mode)
    pair_grids = set(main_pairs["grid_a"]) | set(main_pairs["grid_b"])
    gates, infos = [], {}
    for t in a.targets:  # tuan tu: mot bien mot luc (RAM)
        hoc, gate, infos[t] = run_target(a, t, levels, pair_grids)
        write_csv_atomic(hoc, outs[t])
        if sidecar:
            write_provenance(outs[t], target=t, prefix=a.prefix, scope=a.scope, null_model=NULL_MODEL,
                             **model_provenance(a.model), **infos[t])
        gates.append(gate)
        if a.no_table:
            print(f"--- {t}: ghi {outs[t]}", flush=True)
            continue
        cols = ["grid_a", "muc", "kiem_dinh", "delta_hat", "ci_low", "ci_high", "p_holm", "vung_D", "label"]
        with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
            print(f"--- {t} (Holm m = {int(hoc['holm_m'].max())}) ---")
            print(hoc[cols].to_string(index=False))
            print(gate[["bien", "muc", "n_luoi", "n_hoc_duoc", "hop_le", "ly_do"]].to_string(index=False), flush=True)
    try:
        merged = merge_gate(gate_path, pd.concat(gates, ignore_index=True))
    except ValueError as exc:
        acf.loi(str(exc))
    write_csv_atomic(merged, gate_path)
    if sidecar:
        write_provenance(gate_path, prefix=a.prefix, scope=a.scope, null_model=NULL_MODEL,
                         bien_luot_nay=list(a.targets), **model_provenance(a.model), theo_bien=infos,
                         **({} if a.region == DEFAULT_REGION else {"region": a.region, "pairs_mode": a.pairs_mode}))
    print(f"Ghi: {list(outs.values())} + {gate_path}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", nargs="+", default=list(DOT7_TARGETS))
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--scope", choices=GATE_SCOPES, default="muc_kiem_dinh",
                    help="muc_kiem_dinh: moi luoi cua muc (m 13/4); cap_f1: chi luoi trong cap F1 (cho An Q1)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--area-table", default=None, help="mac dinh theo vung (nhu analyze_cv_family)")
    ap.add_argument("--out-dir", default=None, help="mac dinh theo vung (dbscl: KE_HOACH/ket-qua)")
    ap.add_argument("--exp-root", default=None, help="mac dinh theo vung (dbscl: artifacts/experiments)")
    ap.add_argument("--folds-csv", default=None, help="mac dinh theo vung (dbscl: data/eval/cv_folds.csv)")
    acf.add_common_args(ap)
    acf.add_region_args(ap)
    ap.add_argument("--no-table", action="store_true", help="khong in bang so ra stdout")
    return ap


if __name__ == "__main__":
    main(build_parser().parse_args())

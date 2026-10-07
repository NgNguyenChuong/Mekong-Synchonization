#!/usr/bin/env python
"""Cong "mo hinh hoc duoc" (CHG-25 muc 3): moi (bien, luoi) so MAE(HistGB) voi MAE(season_mean), CV 3 cach chia,
kiem dinh doi dau theo khoi; Holm theo bien tren luoi cua cac muc kiem dinh. Xuat dot7_<bien>_hoc_duoc.csv va
gop dot7_cong_kiem_dinh.csv (bien, muc, hop_le) - analyze_cv_family doc bang nay de ep mo_ta.

Chay:  venv/Scripts/python.exe scripts/analyze_learnability.py --targets ndwi rain_chirps dsr_mcd18 t2m_era5 rh_era5
           [--prefix cv1] [--scope muc_kiem_dinh|cap_f1] [--out-dir KE_HOACH/ket-qua]
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
from training.block_stats import area_levels, pairs_by_area  # noqa: E402
from training.dot7_rules import (EXPECTED_GATE_M, GATE_SCOPES, TESTED_TIERS, gate_table, guard_frozen,  # noqa: E402
                                 learnability, merge_gate, write_csv_atomic)

MODEL, NULL_MODEL = "hist_gb", "season_mean"
DOT7_TARGETS = ("ndwi", "rain_chirps", "dsr_mcd18", "t2m_era5", "rh_era5")


def out_name(target):
    return f"dot7_{target}_hoc_duoc.csv"


def run_target(a, target, levels, pair_grids):
    infos = []
    err = pd.concat([acf.load_errors(a.exp_root, a.prefix, m, a.schemes, target=target, infos=infos)
                     for m in (MODEL, NULL_MODEL)], ignore_index=True)
    e, n_common = acf.common_subset(err)
    pu, cv_units, ho_units = acf.point_units(e, a.folds_csv)
    hoc = learnability(e, pu, acf.GRIDS, levels, target, cv_units=cv_units, holdout_units=ho_units,
                       holdout_seasons=(2020,), alpha=a.alpha, scope=a.scope, pair_grids=pair_grids,
                       model=MODEL, null_model=NULL_MODEL, n_schemes=len(a.schemes),
                       expected_m=EXPECTED_GATE_M[a.scope][target])
    same = acf.check_consistent(infos)
    hoc.insert(2, "n_common_point_seasons", n_common)
    gate = gate_table(hoc, target)
    for t in (hoc, gate):
        t["git_tag"], t["points_ref_sha256"] = same["git_tag"], same["points_ref_sha256"]
    return hoc, gate


def main(a):
    bad = [t for t in a.targets if t not in TESTED_TIERS]
    if bad:
        acf.loi(f"target khong hop le {bad}")
    os.makedirs(a.out_dir, exist_ok=True)
    outs = {t: os.path.join(a.out_dir, out_name(t)) for t in a.targets}
    gate_path = os.path.join(a.out_dir, acf.GATE_FILE)
    guard_frozen([*outs.values(), gate_path], a.frozen_manifest)
    at = acf.area_table(a.area_table)
    levels = area_levels(at)  # moi luoi cua muc (gom s2_level_9/10/11)
    main_pairs = pairs_by_area(at, max_ratio=1.2, expected_m=12)
    pair_grids = set(main_pairs["grid_a"]) | set(main_pairs["grid_b"])
    gates = []
    for t in a.targets:  # tuan tu: mot bien mot luc (RAM)
        hoc, gate = run_target(a, t, levels, pair_grids)
        write_csv_atomic(hoc, outs[t])
        gates.append(gate)
        cols = ["grid_a", "muc", "kiem_dinh", "delta_hat", "ci_low", "ci_high", "p_holm", "vung_D", "label"]
        with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
            print(f"--- {t} (Holm m = {int(hoc['holm_m'].max())}) ---")
            print(hoc[cols].to_string(index=False))
            print(gate[["bien", "muc", "n_luoi", "n_hoc_duoc", "hop_le", "ly_do"]].to_string(index=False), flush=True)
    merged = merge_gate(gate_path, pd.concat(gates, ignore_index=True))
    write_csv_atomic(merged, gate_path)
    print(f"Ghi: {list(outs.values())} + {gate_path}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", nargs="+", default=list(DOT7_TARGETS))
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--scope", choices=GATE_SCOPES, default="muc_kiem_dinh",
                    help="muc_kiem_dinh: moi luoi cua muc (m 13/4); cap_f1: chi luoi trong cap F1 (cho An Q1)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--area-table", default=os.path.join(acf.RESULTS, "dot4_doi_chieu_dien_tich.csv"))
    ap.add_argument("--out-dir", default=acf.RESULTS)
    ap.add_argument("--exp-root", default=acf.EXP_ROOT)
    ap.add_argument("--folds-csv", default=os.path.join(ROOT, "data", "eval", "cv_folds.csv"))
    ap.add_argument("--frozen-manifest", default=acf.FROZEN_MANIFEST)
    return ap


if __name__ == "__main__":
    main(build_parser().parse_args())

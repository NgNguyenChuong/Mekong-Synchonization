#!/usr/bin/env python
"""Doi chung duong PC-1: tiem chenh k*Delta_min vao sai so khung B cua tung cap F1 (HistGB do man, cv1), chay lai F1.
Ket luan DAT / KHONG_DAT / LOI theo tieu chi chot (6 cap min, k = 2, deu + lognormal).

Chay:  venv/Scripts/python.exe scripts/analyze_pc1.py [--k 1 2] [--kinds deu lognormal tau_mu] [--n-rep 200]
       [--seed 0] [--n-perm N] [--out KE_HOACH/ket-qua]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import analyze_cv_family as acf  # noqa: E402
from training import pc1  # noqa: E402
from training.block_stats import (_arm_level, area_levels, common_point_set, pairs_by_area,  # noqa: E402
                                  seed_mean_errors, unit_metrics)

FAMILY = "F1_chinh"
MODEL = "hist_gb"
FINE_LEVEL = 7  # muc min (tier h3_res 7): 6 cap
N_FINE = 6


def log(msg):
    print(msg, flush=True)


def load_frozen(path, alpha_expected=0.05):
    if not os.path.exists(path):
        raise FileNotFoundError(f"LOI: khong co ban dong bang {path}")
    fz = pd.read_csv(path)
    fz = fz[fz["family"] == FAMILY].reset_index(drop=True)
    if len(fz) != 12:
        raise ValueError(f"LOI: ban dong bang co {len(fz)} cap {FAMILY}, ky vong 12")
    for c, v in (("mode", "cv"), ("model_a", MODEL), ("model_b", MODEL), ("metric", "mae"), ("n_seeds", 3)):
        if (fz[c] != v).any():
            raise ValueError(f"LOI: ban dong bang cot {c} khac {v}")
    if not np.allclose(fz["alpha"], alpha_expected):
        raise ValueError("LOI: alpha dong bang khac 0,05")
    return fz


def prepare(a):
    """err CV da loc tap diem chung (nhu run_family F1), don vi, cap, muc."""
    at = acf.area_table(a.area_table)
    pairs = pairs_by_area(at, max_ratio=1.2, expected_m=12)
    levels = area_levels(at, pairs=pairs)
    exp_root = os.path.join(ROOT, "artifacts", "experiments")
    err = acf.load_errors(exp_root, a.prefix, MODEL, a.schemes)  # thieu file -> FileNotFoundError -> LOI
    pu, cv_units, ho_units = acf.point_units(err, os.path.join(ROOT, "data", "eval", "cv_folds.csv"))
    valid = err.assign(valid=err["err"].notna())[["grid", "point_id", "season", "valid"]].drop_duplicates(
        ["grid", "point_id", "season"])
    common, _ = common_point_set(valid)
    e = err[pd.MultiIndex.from_frame(err[["point_id", "season"]]).isin(common)]
    if e["point_id"].map(pu).isin(ho_units).any():
        raise ValueError("LOI: OOF co diem thuoc khoi giu rieng")
    if e["season"].eq(2020).any():
        raise ValueError("LOI: OOF CV co mua giu rieng 2020")
    return e, pu, pairs, levels, len(common)


def unit_inputs(e, pu, pairs, levels, frozen, delta_min, alpha):
    """d, w, d theo cach chia cho tung cap; thr kiem lai voi ban dong bang; lan chay dong nhat so ban dong bang."""
    arm = lambda g: (g, MODEL)  # noqa: E731
    inputs = []
    for i, r in pairs.iterrows():
        d, w, ds = pc1.pair_unit_deltas(e, pu, arm(r["grid_a"]), arm(r["grid_b"]))
        if len(d) != int(frozen.loc[i, "G"]):
            raise ValueError(f"LOI: cap {i} co {len(d)} don vi, dong bang {frozen.loc[i, 'G']}")
        inputs.append((d, w, ds))
    thr_re = {}
    for lv in sorted(set(levels)):
        arms = [arm(g) for g in sorted(levels.index[levels == lv])]
        sub = e[e.set_index(["grid", "model"]).index.isin(arms)]
        units = unit_metrics(seed_mean_errors(sub), pu)[0]
        thr_re[lv] = delta_min * float(np.mean([_arm_level(units, x, "mae") for x in arms]))
    for i, r in pairs.iterrows():
        t_fz, t_re = float(frozen.loc[i, "delta_min_thr"]), thr_re[r["tier"]]
        if not np.isclose(t_fz, t_re, rtol=1e-9, atol=0):
            raise ValueError(f"LOI: thr cap {i} tinh lai {t_re:.9g} khac dong bang {t_fz:.9g}")
    p_fz = frozen["p_value"].to_numpy(float)
    ident = []
    for i, (d, w, ds) in enumerate(inputs):
        G = len(d)
        r = pc1.simulate_pair(d, ds, w, float(frozen.loc[i, "delta_min_thr"]), p_fz, i, alpha,
                              np.ones((1, G)), np.zeros((1, G)))
        ident.append(r.iloc[0].rename({"delta_star": "delta_hat"}))
    ident = pd.DataFrame(ident).reset_index(drop=True)
    bad = pc1.compare_identity(ident, frozen)
    if bad:
        raise ValueError("LOI: lan chay dong nhat lech ban dong bang - " + "; ".join(bad))
    return inputs


def main(a):
    t0 = time.time()
    frozen = load_frozen(a.frozen)
    e, pu, pairs, levels, n_common = prepare(a)
    if n_common != int(frozen["n_common_point_seasons"].iloc[0]):
        raise ValueError(f"LOI: tap diem chung {n_common} khac dong bang {frozen['n_common_point_seasons'].iloc[0]}")
    frozen = pc1.match_frozen(list(zip(pairs["grid_a"], pairs["grid_b"])), frozen)
    if int((pairs["tier"] == FINE_LEVEL).sum()) != N_FINE:
        raise ValueError(f"LOI: muc {FINE_LEVEL} co {int((pairs['tier'] == FINE_LEVEL).sum())} cap, ky vong {N_FINE}")
    alpha = float(frozen["alpha"].iloc[0])
    delta_min = float(frozen["delta_min"].iloc[0])
    log(f"Nap xong {len(e):,} hang OOF, {n_common:,} (diem, mua) chung ({time.time() - t0:.0f} s)")
    inputs = unit_inputs(e, pu, pairs, levels, frozen, delta_min, alpha)
    del e
    log("Lan chay dong nhat khop ban dong bang (12/12 cap)")

    rep_seeds = list(range(a.seed, a.seed + a.n_rep))
    p_fz = frozen["p_value"].to_numpy(float)
    parts = []
    for i, r in pairs.iterrows():
        t1 = time.time()
        d, w, ds = inputs[i]
        cmp_id = frozen.loc[i, "cmp_id"]
        parts.append(pc1.run_pair(i, r["grid_a"], r["grid_b"], r["tier"], r["tier"] == FINE_LEVEL, d, ds, w,
                                  float(frozen.loc[i, "delta_min_thr"]), p_fz, alpha, a.k, a.kinds, rep_seeds,
                                  cmp_id, n_perm=a.n_perm))
        log(f"  cap {i + 1}/12 {r['grid_a']} vs {r['grid_b']}: {time.time() - t1:.1f} s")
    reps = pd.concat(parts, ignore_index=True)
    if (reps["inj_err"] > 1e-9).any():
        raise ValueError("LOI: MAE muc tiem lech k*thr > 1e-9")
    summ = pc1.summarize(reps)
    ver = pc1.verdict(summ, n_fine=N_FINE, k_crit=2, kinds_crit=("deu", "lognormal"), n_rep=a.n_rep)

    os.makedirs(a.out, exist_ok=True)
    f_rep = os.path.join(a.out, "dot7_pc1_lan_lap.csv")
    f_sum = os.path.join(a.out, "dot7_pc1_tong_hop.csv")
    f_ver = os.path.join(a.out, "dot7_pc1_ket_luan.csv")
    reps.to_csv(f_rep, index=False)
    summ.to_csv(f_sum, index=False)
    pd.DataFrame([{**ver, "n_rep": a.n_rep, "seed0": a.seed, "n_perm": a.n_perm or "exact",
                   "k_chay": ";".join(map(str, a.k)), "kieu_chay": ";".join(a.kinds)}]).to_csv(f_ver, index=False)
    cols = ["grid_a", "grid_b", "level", "k", "kind", "ty_le_phat_hien", "ty_le_phat_hien_tren_nguong",
            "ty_le_tost", "trung_vi_delta_star_tren_thr"]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        log(summ[cols].to_string(index=False))
    log(f"\nKET LUAN PC-1: {ver['trang_thai']} - {ver['ly_do']}; cap yeu nhat {ver['cap_yeu_nhat']}")
    log(f"Ghi: {f_rep}\n     {f_sum}\n     {f_ver}\nTong thoi gian {time.time() - t0:.0f} s")
    return ver


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--kinds", nargs="+", choices=pc1.KINDS, default=list(pc1.KINDS))
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="seed lan lap dau; lan lap r dung seed + r")
    ap.add_argument("--n-perm", type=int, default=None,
                    help="Monte Carlo doi dau; mac dinh None = liet ke du 2^19 nhu F1")
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--frozen", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot5_cv1_kiem_dinh_khoi.csv"))
    ap.add_argument("--area-table", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_doi_chieu_dien_tich.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    a = ap.parse_args(argv)
    if a.n_rep < 1 or any(k <= 0 for k in a.k):
        ap.error("--n-rep >= 1 va k > 0")
    try:
        ver = main(a)
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as ex:  # MemoryError khong bat
        log(f"KET LUAN PC-1: LOI - {ex}")
        return 2
    return 0 if ver["trang_thai"] in ("DAT", "KHONG_DAT") else 2


if __name__ == "__main__":
    sys.exit(cli())

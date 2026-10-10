#!/usr/bin/env python
"""Hybrid giu rieng tung mua (CHG-28, h3_res_7): I_s = MAE_s(b) - MAE_s(a) nhom thoi_gian, CI90 theo khoi trong mua;
dem dau qua mua, nhi thuc mot phia; --target all them bang tong ket + Holm m = 6.
Chay: venv/Scripts/python.exe scripts/analyze_hybrid_loso.py --target all --allowed-tags nckh-dot7-loso
"""
import argparse
import os
import sys

import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from training.block_stats import cluster_t_ci  # noqa: E402
from training.dot7_rules import guard_frozen, write_provenance, file_sha256  # noqa: E402
from training.evaluate import holm_adjust  # noqa: E402
import analyze_cv_family as acf  # noqa: E402
from analyze_hybrid_final2020 import block_mae, load_final  # noqa: E402

GRID = "h3_res_7"
TARGETS = ("salinity", "ndwi", "dsr_mcd18", "rain_chirps", "t2m_era5", "rh_era5")
SEASONS = range(2014, 2027)
NAN_SEASON = {"dsr_mcd18": 2020}  # nhan buc xa 2020 NaN (E4)
NGUONG = {13: 10, 12: 9}
CFG = {"a": None, "b": "khong_diem"}
QUY_TAC = "CHG-28 NHAT_KY 2026-10-10 10:23:20"


def run_dir(exp_root, fs, target, season):
    name = (f"cv1__{GRID}__hist_gb__s42" + (f"__fs-{fs}" if fs else "")
            + (f"__t-{target}" if target != "salinity" else "") + f"__ho-{season}")
    return os.path.join(exp_root, name, "final")


def seasons_of(target):
    return [s for s in SEASONS if s != NAN_SEASON.get(target)]


def analyze(target, exp_root, allowed):
    tags, rows = set(), []
    for s in seasons_of(target):
        errs = {c: load_final(run_dir(exp_root, fs, target, s), allowed, tags) for c, fs in CFG.items()}
        if any(set(e.index.get_level_values("season")) != {s} for e in errs.values()):
            raise SystemExit(f"LOI: {target} mua {s}: nhom thoi_gian co mua khac {s}")
        d = run_dir(exp_root, None, target, s)
        blk = pd.read_csv(os.path.join(d, "final_points.csv"), usecols=["point_id", "season", "block_id"]
                          ).set_index(["point_id", "season"])["block_id"]
        mae_b, w = block_mae(errs, blk)
        ci = cluster_t_ci((mae_b["b"] - mae_b["a"]).values, w.values, 0.10)
        mae = {c: float(e.mean()) for c, e in errs.items()}
        i_s = mae["b"] - mae["a"]
        rows.append({"target": target, "season": s, "G": len(w), "n_diem": int(w.sum()), "mae_a": mae["a"],
                     "mae_b": mae["b"], "I": i_s, "I_tuong_doi": i_s / mae["b"],
                     "ci90_low": ci["ci_low"], "ci90_high": ci["ci_high"]})
    return pd.DataFrame(rows), sorted(tags)


def summarize(per_target: dict) -> pd.DataFrame:
    """Moi bien: dem dau I_s, p nhi thuc mot phia theo dau da so, nhan theo NGUONG; Holm tren cac bien."""
    rows = []
    for t, d in per_target.items():
        n, pos, neg = len(d), int((d["I"] > 0).sum()), int((d["I"] < 0).sum())
        if n not in NGUONG:
            raise SystemExit(f"LOI: {t}: {n} mua, can {sorted(NGUONG)}")
        k = max(pos, neg)
        nhan = "khong_on_dinh" if k < NGUONG[n] else ("co_ich" if pos >= neg else "co_hai")
        rows.append({"target": t, "n_mua": n, "n_I_duong": pos, "n_I_am": neg, "nguong": NGUONG[n],
                     "huong": "duong" if pos >= neg else "am",
                     "p_mot_phia": binomtest(k, n, 0.5, alternative="greater").pvalue, "nhan": nhan})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_mot_phia"].values)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True, choices=[*TARGETS, "all"])
    ap.add_argument("--exp-root", default=acf.EXP_ROOT)
    ap.add_argument("--out-dir", default=acf.RESULTS)
    ap.add_argument("--allowed-tags", nargs="+", default=None)
    ap.add_argument("--frozen-manifest", nargs="+", default=acf.FROZEN_MANIFESTS)
    a = ap.parse_args()
    targets = TARGETS if a.target == "all" else (a.target,)
    outs = {t: os.path.join(a.out_dir, f"dot7_{t}_hybrid_loso.csv") for t in targets}
    sum_csv = os.path.join(a.out_dir, "dot7_hybrid_loso_tong_ket.csv")
    paths = [*outs.values(), *([sum_csv] if a.target == "all" else [])]
    guard_frozen(paths + [p + ".provenance.json" for p in paths], a.frozen_manifest)
    common = dict(git_tags_allowed=a.allowed_tags, quy_tac=QUY_TAC, mode="final", nhom="thoi_gian", scheme=42,
                  grid=GRID, cv_folds_sha256=file_sha256(os.path.join(ROOT, "data", "eval", "cv_folds.csv")))
    per, all_tags = {}, set()
    for t in targets:
        per[t], tags = analyze(t, a.exp_root, a.allowed_tags)
        all_tags.update(tags)
        per[t].to_csv(outs[t], index=False)
        write_provenance(outs[t], target=t, git_tags_seen=tags, mua=seasons_of(t), **common)
        print(f"Ghi: {outs[t]} | tags {tags}")
        print(per[t].round(4).to_string(index=False))
    if a.target == "all":
        s = summarize(per)
        s.to_csv(sum_csv, index=False)
        write_provenance(sum_csv, git_tags_seen=sorted(all_tags), holm_m=len(targets), nguong=NGUONG, **common)
        print(f"Ghi: {sum_csv}")
        print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

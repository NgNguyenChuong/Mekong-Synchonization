#!/usr/bin/env python
"""Do nhay pha goc luoi o vuong UTM (CHG-30): moi luoi dich 1/2 canh (x, y) so voi luoi goc nhu mot cap khung;
HistGB cv1, delta_min = 5% MAE TB cua luoi goc, Holm m = 6 trong bien, quy tac D qua 3 cach chia.
Ghi <out-dir>/dot7_<bien>_pha_luoi.csv (+ .provenance.json).
Chay:  venv/Scripts/python.exe scripts/analyze_grid_phase.py --target ndwi [--prefix cv1] [--out-dir KE_HOACH/ket-qua]
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
from training.block_stats import (_arm_level, compare_family, seed_mean_errors,  # noqa: E402
                                  unit_metrics)
from training.dot7_rules import (MO_TA, apply_gate, file_sha256, gate_required, guard_frozen,  # noqa: E402
                                 load_gate, provenance_path, write_csv_atomic, write_provenance)
from training.evaluate import holm_adjust  # noqa: E402

EDGES = (17087, 6458, 2441)
MUC = dict(zip(EDGES, (5, 6, 7)))  # canh -> muc H3 cung dien tich (khop cot muc cua F1)
VARIANTS = ("sx", "sy")
TARGETS = ("salinity", "ndwi", "dsr_mcd18")


def grid_names():
    base = [f"square_utm_{e}m" for e in EDGES]
    return base + [f"{b}_{v}" for b in base for v in VARIANTS]


def label_of(p_holm, delta, tost, same_seeds, alpha):
    ok = p_holm < alpha and delta != 0
    if ok and same_seeds:
        return "cho_giu_rieng"
    return "tuong_duong" if (not ok and tost) else "chua_phan_dinh"


def compare_scheme(e, pu, cv_units, ho_units, model, alpha, delta_min):
    """6 cap (goc vs _sx/_sy theo 3 muc) tren err e; Holm m = 6 gom ca ba muc."""
    units, _, _ = unit_metrics(seed_mean_errors(e), pu, min_pts=30)
    parts = []
    for edge in EDGES:
        g = f"square_utm_{edge}m"
        ref = _arm_level(units, (g, model), "mae")  # delta_min theo MAE TB luoi goc, khong gom luoi dich
        r = compare_family(e, pu, [(g, f"{g}_{v}") for v in VARIANTS], cv_units=cv_units, holdout_units=ho_units,
                           holdout_seasons=(2020,), mode="cv", delta_min=delta_min, delta_min_kind="rel", alpha=alpha,
                           delta_ref=ref, require_practical=True, model=model, family=f"pha_{edge}", expected_m=2)
        r.insert(0, "edge_m", edge)
        r.insert(1, "variant", list(VARIANTS))
        parts.append(r)
    out = pd.concat(parts, ignore_index=True)
    out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
    out["holm_m"] = len(out)
    out["label"] = [label_of(r.p_holm, r.delta_hat, r.tost_tuong_duong, r.seeds_same_dir, alpha)
                    for r in out.itertuples()]
    out["delta_rel"] = out["delta_hat"] / out["delta_ref"]
    return out


def apply_rule_d(main, per):
    """per: {cach chia: bang compare_scheme}. cho_giu_rieng: moi cach cung nhan + dau; tuong_duong: moi cach tuong_duong."""
    res = main.copy()
    keep, final = [], []
    for i, r in enumerate(main.itertuples()):
        labs = [per[s]["label"].iloc[i] for s in per]
        dels = [per[s]["delta_hat"].iloc[i] for s in per]
        if r.label == "cho_giu_rieng":
            ok = all(lb == r.label and np.sign(x) == np.sign(r.delta_hat) for lb, x in zip(labs, dels))
        elif r.label == "tuong_duong":
            ok = all(lb == r.label for lb in labs)
        else:
            ok = None
        keep.append(ok)
        final.append(r.label if ok is None or ok else "khong_vung_theo_cach_chia")
    for s, t in per.items():
        res[f"nhan_s{s}"] = t["label"].to_numpy()
        res[f"delta_s{s}"] = t["delta_hat"].to_numpy()
    res["giu_ca_3_cach_chia"] = pd.array(keep, dtype="boolean")
    res["ket_luan_D"] = final
    return res


def apply_gate_pha(res, gate):
    """Muc truot cong -> mo_ta, bo p/TOST (nhu F1). gate None (do man) = qua."""
    res = apply_gate(res.assign(muc=res["edge_m"].map(MUC), muc_kiem_dinh=True), gate)
    fail = res["cong_hoc_duoc"].eq(False)
    for c in ["ket_luan_D"] + [c for c in res.columns if c.startswith("nhan_s")]:
        res[c] = res[c].astype(object)
        res.loc[fail, c] = MO_TA
    res["giu_ca_3_cach_chia"] = res["giu_ca_3_cach_chia"].mask(fail)
    return res.drop(columns="muc_kiem_dinh")


def add_e_criteria(res, f1):
    """E_pha = max|delta_hat sx, sy| / delta_min_thr (goc); E_khung = max|delta_hat| / delta_min_thr tren cap F1_chinh
    cung muc. Them cot E_pha, E_khung, pha_le_khung."""
    f = f1[f1["family"] == "F1_chinh"]
    lv = (f["muc"] if "muc" in f.columns else f["delta_ref_level"]).astype(int)
    ek = (f["delta_hat"].abs() / f["delta_min_thr"]).groupby(lv).max()
    res = res.copy()
    ep = {}
    for muc, d in res.groupby("muc"):
        if muc not in ek.index:
            acf.loi(f"F1_chinh khong co muc {muc}")
        ep[muc] = (d["delta_hat"].abs() / d["delta_min_thr"]).max()
    res["E_pha"] = res["muc"].map(ep)
    res["E_khung"] = res["muc"].map(ek)
    res["pha_le_khung"] = res["E_pha"] <= res["E_khung"]
    return res


def analyze_target(a, target):
    infos = []
    err = acf.load_errors(a.exp_root, a.prefix, "hist_gb", a.schemes, grids=grid_names(), target=target, infos=infos)
    pu, cv_units, ho_units = acf.point_units(err, a.folds_csv)
    e, n_common = acf.common_subset(err)
    same = acf.check_consistent(infos, a.allowed_tags)
    main = compare_scheme(e, pu, cv_units, ho_units, "hist_gb", a.alpha, a.delta_min)
    per = {s: compare_scheme(e[e["seed"] == s], pu, cv_units, ho_units, "hist_gb", a.alpha, a.delta_min)
           for s in a.schemes}
    res = apply_rule_d(main, per)
    gate_path = os.path.join(a.out_dir, acf.GATE_FILE)
    try:
        gate = load_gate(gate_path, target, required=gate_required(target, "hist_gb"))
    except ValueError as exc:
        acf.loi(str(exc))
    f1_path = os.path.join(a.out_dir, acf.out_name(target, a.prefix))
    if not os.path.isfile(f1_path):
        acf.loi(f"thieu file F1 {f1_path}")
    res = add_e_criteria(apply_gate_pha(res, gate), pd.read_csv(f1_path))
    res.insert(0, "target", target)
    res.insert(1, "n_common_point_seasons", n_common)
    res["git_tag"], res["points_ref_sha256"], res["n_luot"] = same["git_tag"], same["points_ref_sha256"], len(infos)
    return res, same, len(infos), {"f1_file": os.path.basename(f1_path), "f1_sha256": file_sha256(f1_path),
                                   "cong_ap_dung": True, "cong_file": gate_path if gate is not None else ""}


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"dot7_{a.target}_pha_luoi.csv")
    guard_frozen([out, provenance_path(out)], a.frozen_manifest)
    res, same, n, extra = analyze_target(a, a.target)
    write_csv_atomic(res, out)
    lv = res.drop_duplicates("muc").set_index("muc")
    tong_ket = {"n_muc_pha_le_khung": f"{int(lv['pha_le_khung'].sum())}/{len(lv)}",
                "tost_2_cap_pha": {int(m): d["ket_luan_D"].tolist() for m, d in res.groupby("muc")}}
    write_provenance(out, prefix=a.prefix, target=a.target, git_tag=same["git_tag"],
                     points_ref_sha256=same["points_ref_sha256"], n_luot=n, holm_m=6, delta_min=a.delta_min,
                     delta_min_kind="rel", delta_ref={int(m): float(d["delta_ref"].iloc[0]) for m, d in res.groupby("muc")},
                     delta_ref_desc="MAE TB luoi goc (CV, theo muc)", rule="CHG-30", tong_ket=tong_ket, **extra,
                     **acf.tag_info(same, a.allowed_tags))
    if not a.no_table:
        cols = ["edge_m", "variant", "delta_hat", "delta_rel", "ci_tost_low", "ci_tost_high", "p_holm",
                "tost_tuong_duong", "label", "ket_luan_D"]
        with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
            print(res[cols].to_string(index=False))
    print(f"\ntong ket {a.target}: {tong_ket}")
    print(f"target {a.target} | {n} luot | git_tag {same['git_tag']} | Ghi: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=TARGETS, required=True)
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta-min", type=float, default=0.05)
    ap.add_argument("--out-dir", default=acf.RESULTS)
    ap.add_argument("--exp-root", default=acf.EXP_ROOT)
    ap.add_argument("--folds-csv", default=os.path.join(ROOT, "data", "eval", "cv_folds.csv"))
    ap.add_argument("--frozen-manifest", nargs="+", default=acf.FROZEN_MANIFESTS)
    ap.add_argument("--allowed-tags", nargs="+", default=None)
    ap.add_argument("--no-table", action="store_true")
    main(ap.parse_args())

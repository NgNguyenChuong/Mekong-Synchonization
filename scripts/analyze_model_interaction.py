#!/usr/bin/env python
"""Kiem tuong tac mo hinh x khung luoi theo khoi: I_b = d_b(rf|mlp) - d_b(HistGB) tren cap kiem dinh qua cong CA HAI
mo hinh; doi dau theo khoi, Holm theo (bien, mo hinh), quy tac D (3 cach chia cung dau I_hat), TOST chi bao cao.

Chay:  venv/Scripts/python.exe scripts/analyze_model_interaction.py --model rf [--targets salinity ndwi ...]
           [--allowed-tags TAG ...] [--tags-reason "..."] [--prefix cv1] [--out-dir KE_HOACH/ket-qua] [--no-table]
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
import analyze_model_sensitivity as ams  # noqa: E402
from training.dot7_rules import (MODELS, MULTI_TAG_REASON, TESTED_TIERS, file_sha256, gate_required,  # noqa: E402
                                 guard_frozen, load_gate, model_name, model_provenance, provenance_path, run_model,
                                 write_csv_atomic, write_provenance)
from training.model_interaction import SIG, check_same_points, interaction_family, verdict  # noqa: E402

HOLDOUT_SEASONS = (2020,)  # nhu run_family cua analyze_cv_family
F1_EXTRA = ("delta_min_thr", "delta_ref", "n_common_point_seasons")
TOL = {"rtol": 1e-6, "atol": 1e-9}  # Delta_hat tinh lai vs file F1 (to_csv giu du chu so)
REPORT = {"rf": "kiem_dinh", "mlp": "mo_ta_phu_luc"}
SIDES = ("hist_gb", "mo_hinh")
OUT_COLS = ["target", "model", "grid_a", "grid_b", "muc", "G", "I_hat", "ci_low", "ci_high", "p_value", "p_holm",
            "holm_m", "seed_deltas", "seeds_same_dir", "delta_min_thr_histgb", "tost_tuong_tac", "label", "chua_ro",
            "delta_hat_histgb", "delta_hat_model", "n_common_point_seasons", "git_tags", "points_ref_sha256",
            "ci_tost_low", "ci_tost_high", "se", "p_exact", "n_perm", "k_same", "k_over_G", "n_seeds",
            "delta_ref_histgb", "alpha", "min_pts", "family", "cmp_id", "run_model", "run_model_ref"]


def pair_name(target, model):
    return model_name(f"dot7_{target}_tuong_tac_mo_hinh.csv", model)


def summary_name(model):
    return model_name("dot7_tuong_tac_mo_hinh_tong_ket.csv", model)


def _abs(p):
    return os.path.abspath(p).replace("\\", "/")


def _split_tags(tags) -> set:
    return {x for s in tags for x in str(s).split(";") if x and x != "nan"}


def f1_points_ref(path):
    """Tap points_ref_sha256 trong file F1 (None neu file cu khong co cot)."""
    if "points_ref_sha256" not in pd.read_csv(path, nrows=0).columns:
        return None
    return set(pd.read_csv(path, usecols=["points_ref_sha256"])["points_ref_sha256"].dropna().astype(str))


def tag_str(tags: dict, model: str) -> str:
    return f"hist_gb:{','.join(tags['hist_gb'])};{model}:{','.join(tags['mo_hinh'])}"


# ---------------------------------------------------------
# Doc cap kiem dinh hai ben
# ---------------------------------------------------------
def load_pairs(a, t):
    """Cap muc kiem dinh hai ben (ghep theo grid_a, grid_b, muc) + thong tin dau vao tung ben."""
    cv_dir = a.cv_dir or a.out_dir
    specs = (("hist_gb", "hist_gb", a.gate_histgb or os.path.join(cv_dir, acf.gate_name("hist_gb"))),
             ("mo_hinh", a.model, a.gate_model or os.path.join(cv_dir, acf.gate_name(a.model))))
    got, info = {}, {}
    for side, model, gp in specs:
        try:
            gate = load_gate(gp, t, required=gate_required(t, model), model=model)
        except ValueError as exc:
            acf.loi(str(exc))
        path = os.path.join(cv_dir, model_name(acf.out_name(t, a.prefix), model))
        got[side], tags = ams.load_side(path, t, model, gate, F1_EXTRA)
        info[side] = {"model": model, "run_model": run_model(model), "cv_file": _abs(path),
                      "cv_sha256": file_sha256(path), "f1_git_tag": sorted(_split_tags(tags)),
                      "f1_points_ref_sha256": sorted(f1_points_ref(path) or []),
                      "gate_file": _abs(gp) if gate is not None else None,
                      "gate_sha256": file_sha256(gp) if gate is not None else None,
                      "gate_hop_le": ({str(k): bool(v) for k, v in sorted(gate.items())} if gate is not None else None)}
    return ams.compare_sides(got["hist_gb"], got["mo_hinh"], t, a.model), info


# ---------------------------------------------------------
# Sai so OOF hai mo hinh -> kiem tuong tac
# ---------------------------------------------------------
def check_f1_inputs(t, side, inf, run_tags, ref, n_common, k):
    """Luot doc vao phai la luot da tao file F1 cua ben do (tag, points_ref, so (diem, mua) chung)."""
    f1_tags = set(inf["f1_git_tag"])
    if f1_tags and not set(run_tags) <= f1_tags:
        acf.loi(f"{t} {side}: git_tag luot {sorted(set(run_tags) - f1_tags)} khong co trong file F1 {sorted(f1_tags)}")
    if inf["f1_points_ref_sha256"] and inf["f1_points_ref_sha256"] != [ref]:
        acf.loi(f"{t} {side}: points_ref_sha256 file F1 khac luot dang doc")
    n_f1 = set(k[f"n_common_point_seasons_{side}"].astype(int))
    if n_f1 != {n_common}:
        acf.loi(f"{t} {side}: so (diem, mua) chung {n_common} khac file F1 {sorted(n_f1)}")


def check_delta(t, res, k):
    """Delta_hat tinh lai tu d_b moi ben phai trung Delta_hat file F1 cung cap."""
    for col, side in (("delta_hat_histgb", "hist_gb"), ("delta_hat_model", "mo_hinh")):
        f1 = k[f"delta_hat_{side}"].to_numpy(float)
        got = res[col].to_numpy(float)
        bad = ~np.isclose(got, f1, **TOL)
        if bad.any():
            keys = k.loc[bad, ["grid_a", "grid_b", "muc"]].to_dict("records")
            acf.loi(f"{t} {side}: Delta_hat tinh lai khac file F1 o {int(bad.sum())} cap {keys[:3]} "
                    f"(max |chenh| {np.abs(got - f1)[bad].max():.3g})")


def run_target(a, t, k, info):
    """Doc sai so OOF hai ben, kiem nhat quan, chay interaction_family tren cap k (thu tu giu nguyen)."""
    infos, err, units, n_common = {s: [] for s in SIDES}, {}, {}, {}
    for side, model in zip(SIDES, ("hist_gb", a.model)):
        e = acf.load_errors(a.exp_root, a.prefix, run_model(model), a.schemes, target=t, infos=infos[side])
        units[side] = acf.point_units(e, a.folds_csv)
        err[side], n_common[side] = acf.common_subset(e)
        del e
    same = acf.check_consistent(infos["hist_gb"] + infos["mo_hinh"], a.allowed_tags)
    (pu, cv_u, ho_u), (pu_m, cv_m, ho_m) = units["hist_gb"], units["mo_hinh"]
    if list(cv_u) != list(cv_m) or list(ho_u) != list(ho_m) or not pu.sort_index().equals(pu_m.sort_index()):
        acf.loi(f"{t}: don vi (khoi) cua hai mo hinh khac nhau")
    try:
        n = check_same_points(err["hist_gb"], err["mo_hinh"])
    except ValueError as exc:
        acf.loi(f"{t}: {exc}")
    tags = {s: sorted({i["git_tag"] for i in infos[s]}) for s in SIDES}
    for side in SIDES:
        check_f1_inputs(t, side, info[side], tags[side], same["points_ref_sha256"], n_common[side], k)
        info[side].update(git_tags=tags[side], n_luot=len(infos[side]), n_common_point_seasons=n_common[side])
    try:
        res = interaction_family(err["hist_gb"], err["mo_hinh"], pu, list(zip(k["grid_a"], k["grid_b"])),
                                 k["delta_min_thr_hist_gb"], ref_model="hist_gb", model=run_model(a.model),
                                 cv_units=cv_u, holdout_units=ho_u, holdout_seasons=HOLDOUT_SEASONS, alpha=a.alpha,
                                 n_schemes=len(a.schemes), family=f"tuong_tac_{a.model}")
    except (ValueError, AssertionError) as exc:
        acf.loi(f"{t}: {exc}")
    check_delta(t, res, k)
    res = res.rename(columns={"model": "run_model", "model_ref": "run_model_ref"})
    res["muc"] = k["muc"].to_numpy()
    res["delta_ref_histgb"] = k["delta_ref_hist_gb"].to_numpy()
    res["n_common_point_seasons"] = n
    return res, same, tags, n


def main(a):
    bad = [t for t in a.targets if t not in TESTED_TIERS]
    if bad:
        acf.loi(f"target khong hop le {bad}")
    os.makedirs(a.out_dir, exist_ok=True)
    outs = {t: os.path.join(a.out_dir, pair_name(t, a.model)) for t in a.targets}
    summ_path = os.path.join(a.out_dir, summary_name(a.model))
    files = [*outs.values(), summ_path]
    guard_frozen(files + [provenance_path(p) for p in files], a.frozen_manifest)
    rows, inputs, done = [], {}, []
    for t in a.targets:
        x, info = load_pairs(a, t)
        k = x[x["qua_cong_ca_hai"]].reset_index(drop=True)
        prov = {"target": t, "prefix": a.prefix, "schemes": list(a.schemes), "alpha": a.alpha,
                "holm_m": len(k), "holm_ho": "cap muc kiem dinh qua cong ca hai mo hinh",
                "holdout_seasons": list(HOLDOUT_SEASONS), "nhan_y_nghia": SIG, "bao_cao": REPORT[a.model],
                "exp_root": _abs(a.exp_root), "folds_csv": _abs(a.folds_csv),
                "folds_sha256": file_sha256(a.folds_csv) if os.path.isfile(a.folds_csv) else None}
        if k.empty:  # khong cap nao qua cong ca hai: khong doc sai so
            res, gt, ref, n = pd.DataFrame(columns=OUT_COLS), "", "", 0
        else:
            res, same, tags, n = run_target(a, t, k, info)
            gt, ref = tag_str(tags, a.model), same["points_ref_sha256"]
            prov.update(git_tags=tags, points_ref_sha256=ref, n_common_point_seasons=n,
                        **acf.tag_info(same, a.allowed_tags, a.tags_reason))
        res["target"], res["model"], res["git_tags"], res["points_ref_sha256"] = t, a.model, gt, ref
        res = res[OUT_COLS]
        done.append((t, res, prov, info))
        n_sig = int((res["label"] == SIG).sum())
        n_unclear = int(res["chua_ro"].astype(bool).sum())
        rows.append({"target": t, "model": a.model, "n_cap_kiem_dinh": len(x), "n_cap": len(k),
                     "n_tuong_tac_y_nghia": n_sig, "n_chua_ro": n_unclear,
                     "ket_luan": verdict(len(k), n_sig, n_unclear),
                     "holm_m": len(k), "alpha": a.alpha, "bao_cao": REPORT[a.model], "n_common_point_seasons": n,
                     "git_tags": gt, "points_ref_sha256": ref})
        inputs[t] = info
    for t, res, prov, info in done:  # ghi sau khi moi bien chay xong: LOI giua chung -> khong ghi file nao
        write_csv_atomic(res, outs[t])
        write_provenance(outs[t], **prov, **model_provenance(a.model), dau_vao=info)
    summ = pd.DataFrame(rows)
    holm_m = {r["target"]: r["holm_m"] for r in rows}
    write_csv_atomic(summ, summ_path)
    write_provenance(summ_path, prefix=a.prefix, alpha=a.alpha, nhan_y_nghia=SIG, **model_provenance(a.model),
                     holm_m=holm_m,
                     bang_cap={t: _abs(p) for t, p in outs.items()},
                     bang_cap_sha256={t: file_sha256(p) for t, p in outs.items()}, dau_vao=inputs)
    if not a.no_table:
        with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
            print(summ.to_string(index=False))
    print(f"Ghi: {list(outs.values())} + {summ_path} | model {a.model} | holm_m {holm_m}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=[m for m in MODELS if m != "hist_gb"], required=True)
    ap.add_argument("--targets", nargs="+", default=list(ams.TARGETS))
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-dir", default=acf.RESULTS)
    ap.add_argument("--cv-dir", default=None, help="thu muc file F1 + bang cong (mac dinh --out-dir)")
    ap.add_argument("--gate-histgb", default=None, help=f"mac dinh <cv-dir>/{acf.gate_name('hist_gb')}")
    ap.add_argument("--gate-model", default=None, help="mac dinh <cv-dir>/dot7_cong_kiem_dinh__<model>.csv")
    ap.add_argument("--exp-root", default=acf.EXP_ROOT)
    ap.add_argument("--folds-csv", default=os.path.join(ROOT, "data", "eval", "cv_folds.csv"))
    ap.add_argument("--frozen-manifest", nargs="+", default=acf.FROZEN_MANIFESTS)
    ap.add_argument("--allowed-tags", nargs="+", default=None,
                    help="cho phep nhieu git_tag (hop luot HistGB + mo hinh phai la tap con)")
    ap.add_argument("--tags-reason", default=MULTI_TAG_REASON, help="ly do nhieu tag ghi vao provenance")
    ap.add_argument("--no-table", action="store_true", help="khong in bang tong ket ra stdout")
    return ap


if __name__ == "__main__":
    main(build_parser().parse_args())

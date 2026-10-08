#!/usr/bin/env python
"""Do nhay theo mo hinh: cap kiem dinh ho chinh HistGB vs cung cap cua rf/mlp (ket qua analyze_cv_family), chi tinh cap
co muc qua cong cua CA HAI mo hinh; ket_luan_giu = khong cap nao co y nghia them va ti le cung dau >= --min-frac.

Chay:  venv/Scripts/python.exe scripts/analyze_model_sensitivity.py --model rf [--targets salinity ndwi ...]
           [--prefix cv1] [--holm-scope cong|muc_kiem_dinh] [--min-frac 0.8] [--out-dir KE_HOACH/ket-qua] [--no-table]
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
from analyze_learnability import DOT7_TARGETS  # noqa: E402
from training.dot7_rules import (HOLM_SCOPES, MODELS, TESTED_TIERS, file_sha256, gate_required,  # noqa: E402
                                 guard_frozen, holm_tiers, load_gate, main_family, model_provenance, provenance_path,
                                 scoped_name, scopes_differ, tested_tiers, write_csv_atomic, write_provenance)

SIG_LABELS = frozenset({"cho_giu_rieng"})  # nhan CV co y nghia: Holm dat + cung chieu moi cach chia
KEY = ["grid_a", "grid_b", "muc"]
SIDE_COLS = ["delta_hat", "label", "p_holm", "qua_cong", "co_y_nghia"]
TARGETS = ("salinity", *DOT7_TARGETS)


def pair_name(target, model, scope):
    return scoped_name(f"dot7_{target}_do_nhay_mo_hinh.csv", scope, scope == "muc_kiem_dinh", model)


def summary_name(model, scope):
    return scoped_name("dot7_do_nhay_mo_hinh_tong_ket.csv", scope, scope == "muc_kiem_dinh", model)


def _abs(p):
    return os.path.abspath(p).replace("\\", "/")


def load_side(path, target, model, gate):
    """Cap ho chinh thuoc muc kiem dinh cua mot mo hinh: KEY + SIDE_COLS; cot cong_hoc_duoc phai khop bang cong."""
    if not os.path.isfile(path):
        acf.loi(f"thieu ket qua analyze_cv_family {path}")
    d = pd.read_csv(path)
    if "target" in d.columns and (d["target"] != target).any():
        acf.loi(f"{path}: co dong target khac '{target}'")
    d = d[d["family"] == main_family(model)].copy()
    if d.empty:
        acf.loi(f"{path}: khong co ho {main_family(model)}")
    if (d["mode"] != "cv").any():
        acf.loi(f"{path}: co dong khong phai mode cv")
    if "muc" not in d.columns:  # file do man truoc CHG-25 khong co cot muc
        d["muc"] = d["delta_ref_level"]
    elif not np.array_equal(d["muc"].to_numpy(float), d["delta_ref_level"].to_numpy(float)):
        acf.loi(f"{path}: muc khac delta_ref_level")
    d["muc"] = d["muc"].astype(int)
    in_tiers = d["muc"].isin(tested_tiers(target))
    if "muc_kiem_dinh" in d.columns and (d["muc_kiem_dinh"].astype(bool) != in_tiers).any():
        acf.loi(f"{path}: muc_kiem_dinh khac TESTED_TIERS['{target}']")
    d = d[in_tiers.to_numpy()].copy()
    if d.duplicated(KEY).any():
        acf.loi(f"{path}: trung cap {KEY}")
    if d["delta_hat"].isna().any():
        acf.loi(f"{path}: delta_hat NaN o cap kiem dinh")
    d["qua_cong"] = d["muc"].isin(holm_tiers(target, gate, "cong"))
    if "cong_hoc_duoc" in d.columns:
        c = d["cong_hoc_duoc"]
        ok = c.isna().all() if gate is None else bool(c.notna().all() and (c.astype(bool) == d["qua_cong"]).all())
        if not ok:
            acf.loi(f"{path}: cot cong_hoc_duoc khong khop bang cong dang doc")
    d["co_y_nghia"] = d["label"].isin(SIG_LABELS)
    tags = sorted(set(d["git_tag"].astype(str))) if "git_tag" in d.columns else []
    return d[KEY + SIDE_COLS].reset_index(drop=True), tags


def compare_sides(h, m, target, model):
    """Ghep cap HistGB / mo hinh theo KEY (tap cap phai trung) -> cung_dau, co_y_nghia_them, qua_cong_ca_hai."""
    x = h.merge(m, on=KEY, how="outer", suffixes=("_hist_gb", "_mo_hinh"), indicator=True)
    if (x["_merge"] != "both").any():
        lech = x.loc[x["_merge"] != "both", KEY].to_dict("records")
        acf.loi(f"{target}: tap cap kiem dinh HistGB va {model} khac nhau: {lech[:5]}")
    x = x.drop(columns="_merge").sort_values(["muc", "grid_a", "grid_b"]).reset_index(drop=True)
    for c in ("qua_cong", "co_y_nghia"):
        for s in ("hist_gb", "mo_hinh"):
            x[f"{c}_{s}"] = x[f"{c}_{s}"].astype(bool)
    x["qua_cong_ca_hai"] = x["qua_cong_hist_gb"] & x["qua_cong_mo_hinh"]
    sh, sm = np.sign(x["delta_hat_hist_gb"]), np.sign(x["delta_hat_mo_hinh"])
    x["cung_dau"] = (sh == sm) & (sh != 0)
    x["co_y_nghia_them"] = x["co_y_nghia_mo_hinh"] & ~x["co_y_nghia_hist_gb"]
    x["mat_y_nghia"] = x["co_y_nghia_hist_gb"] & ~x["co_y_nghia_mo_hinh"]  # mo ta, khong vao quy tac
    x.insert(0, "target", target)
    x.insert(1, "model", model)
    return x


def summarize(x, target, model, min_frac):
    k = x[x["qua_cong_ca_hai"]]
    n = len(k)
    n_same, n_extra = int(k["cung_dau"].sum()), int(k["co_y_nghia_them"].sum())
    frac = n_same / n if n else np.nan
    return {"target": target, "model": model, "n_cap_kiem_dinh": len(x), "n_cap": n, "n_cung_dau": n_same,
            "ti_le_cung_dau": frac, "n_co_y_nghia_them": n_extra, "n_mat_y_nghia": int(k["mat_y_nghia"].sum()),
            "min_frac": min_frac, "ket_luan_giu": (n_extra == 0 and frac >= min_frac) if n else np.nan,
            "trang_thai": "co_cap" if n else "khong_co_cap_qua_cong_ca_hai"}


def main(a):
    bad = [t for t in a.targets if t not in TESTED_TIERS]
    if bad:
        acf.loi(f"target khong hop le {bad}")
    os.makedirs(a.out_dir, exist_ok=True)
    outs = {t: os.path.join(a.out_dir, pair_name(t, a.model, a.holm_scope)) for t in a.targets}
    summ_path = os.path.join(a.out_dir, summary_name(a.model, a.holm_scope))
    files = [*outs.values(), summ_path]
    guard_frozen(files + [provenance_path(p) for p in files], a.frozen_manifest)
    cv_dir = a.cv_dir or a.out_dir
    sides = (("hist_gb", "hist_gb", a.gate_histgb or os.path.join(cv_dir, acf.gate_name("hist_gb"))),
             ("mo_hinh", a.model, a.gate_model or os.path.join(cv_dir, acf.gate_name(a.model))))
    rows, inputs = [], {}
    for t in a.targets:
        got, info = {}, {}
        for side, model, gp in sides:
            try:
                gate = load_gate(gp, t, required=gate_required(t, model), model=model)
            except ValueError as exc:
                acf.loi(str(exc))
            path = os.path.join(cv_dir, scoped_name(acf.out_name(t, a.prefix), a.holm_scope,
                                                    scopes_differ(t, gate), model))
            got[side], tags = load_side(path, t, model, gate)
            info[side] = {"model": model, "cv_file": _abs(path), "cv_sha256": file_sha256(path), "git_tag": tags,
                          "gate_file": _abs(gp) if gate is not None else None,
                          "gate_sha256": file_sha256(gp) if gate is not None else None,
                          "gate_hop_le": ({str(k): bool(v) for k, v in sorted(gate.items())}
                                          if gate is not None else None)}
        x = compare_sides(got["hist_gb"], got["mo_hinh"], t, a.model)
        write_csv_atomic(x, outs[t])
        write_provenance(outs[t], target=t, holm_scope=a.holm_scope, prefix=a.prefix, sig_labels=sorted(SIG_LABELS),
                         **model_provenance(a.model), dau_vao=info)
        rows.append(summarize(x, t, a.model, a.min_frac))
        inputs[t] = info
    summ = pd.DataFrame(rows)
    write_csv_atomic(summ, summ_path)
    write_provenance(summ_path, holm_scope=a.holm_scope, prefix=a.prefix, min_frac=a.min_frac,
                     sig_labels=sorted(SIG_LABELS), **model_provenance(a.model),
                     bang_cap={t: _abs(p) for t, p in outs.items()},
                     bang_cap_sha256={t: file_sha256(p) for t, p in outs.items()}, dau_vao=inputs)
    if not a.no_table:
        with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
            print(summ.to_string(index=False))
    print(f"Ghi: {list(outs.values())} + {summ_path} | model {a.model} | holm {a.holm_scope}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=[m for m in MODELS if m != "hist_gb"], required=True)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--holm-scope", choices=HOLM_SCOPES, default="cong")
    ap.add_argument("--min-frac", type=float, default=0.8, help="ti le cap cung dau toi thieu de giu ket luan")
    ap.add_argument("--out-dir", default=acf.RESULTS)
    ap.add_argument("--cv-dir", default=None, help="thu muc ket qua analyze_cv_family + bang cong (mac dinh --out-dir)")
    ap.add_argument("--gate-histgb", default=None, help=f"mac dinh <cv-dir>/{acf.gate_name('hist_gb')}")
    ap.add_argument("--gate-model", default=None, help="mac dinh <cv-dir>/dot7_cong_kiem_dinh__<model>.csv")
    ap.add_argument("--frozen-manifest", nargs="+", default=acf.FROZEN_MANIFESTS)
    ap.add_argument("--no-table", action="store_true", help="khong in bang so ra stdout")
    return ap


if __name__ == "__main__":
    main(build_parser().parse_args())

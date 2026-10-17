#!/usr/bin/env python
"""CHG-20 tieu chi 2: so CV khoi 50 km (cv1, s42/s43/s44) voi khoi ~100 km (k100, P1=s42 / P2=s43), HistGB.

Nguong + he qua ghi TRUOC khi chay (NHAT_KY 2026-10-05, CHG-20):
  1. MAE theo diem tung luoi: TB(P1, P2) khoi 100 km vs TB(s42, s43, s44) khoi 50 km; bao them P1 vs s42.
     Tang > 5% o BAT KY luoi nao -> bai dung MAE khoi 100 km lam sai so tuyet doi chinh.
  2. 6 cap ho chinh muc min: |Delta| (TB P1, P2) < Delta_min muc min o MOI cap -> giu ket luan
     "4 khung tuong duong o muc min"; co cap vuot -> ha thanh "khong vung theo kich thuoc khoi".
Tap so sanh: (diem, mua) co sai so hop le o MOI luoi, MOI cach chia, CA HAI thiet ke (cung diem cho moi phep so).
Delta = MAE(a) - MAE(b) (cung chieu delta_hat cua analyze_cv_family.py); script doi chieu lai delta_hat khoi 50 km.
Dot 7 (--target khac salinity): Delta_min muc min lay tu ket qua analyze_cv_family cua bien (bat buoc co); nguong 2
chi la ket luan khi muc min duoc kiem dinh (CHG-25 + cong), con lai mo_ta. Kiem luot + chan file dong bang nhu
analyze_cv_family. --holm-scope doc file analyze_cv_family cung pham vi (muc_kiem_dinh khac cong -> hau to
__holm_muc_kiem_dinh cho ca file doc va file ghi). --model rf|mlp: luot cua mo hinh do, Delta_min muc min tu file
analyze_cv_family cung mo hinh, file doc/ghi hau to __<model>.

Chay:  venv/Scripts/python.exe scripts/analyze_block_size.py [--target ndwi] [--model hist_gb|rf|mlp]
           [--holm-scope cong|muc_kiem_dinh] [--no-table]
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
from analyze_cv_family import GRIDS  # noqa: E402
from run_experiments import run_name  # noqa: E402
from training.dot7_rules import (HOLM_SCOPES, TESTED_TIERS, all_scoped_names, file_sha256, gate_required,  # noqa: E402
                                 guard_frozen, holm_provenance, load_gate, main_family, model_provenance,
                                 provenance_path, run_model, scoped_name, scopes_differ, write_csv_atomic,
                                 write_provenance)

MAE_RISE_MAX = 0.05
DELTA_MIN_FINE = 0.0309  # do man (dS/m), tu ban dong bang
FINE_LEVEL = 7
FINE_PAIRS = [("h3_res_7", "s2_level_12"), ("h3_res_7", "square_utm_2441m"), ("h3_res_7", "latlon_0.0222deg"),
              ("s2_level_12", "square_utm_2441m"), ("s2_level_12", "latlon_0.0222deg"),
              ("square_utm_2441m", "latlon_0.0222deg")]
DESIGNS = (("k50", [42, 43, 44]), ("k100", [42, 43]))


def out_names(target):
    if target == "salinity":
        return "dot6_khoi100_mae.csv", "dot6_khoi100_cap_min.csv"
    return f"dot7_{target}_khoi100_mae.csv", f"dot7_{target}_khoi100_cap_min.csv"


def read_err(exp, prefix, grid, scheme, target="salinity", infos=None, model="hist_gb") -> pd.Series:
    """Sai so OOF theo (point_id, season) cua mot lan chay; doc tung lan (it bo nho)."""
    name = run_name(prefix, grid, run_model(model), scheme, "", None, target)
    run_dir = os.path.join(exp, name, "cv")
    info = acf.check_run(run_dir, name, target, "cv")
    if infos is not None and name not in {i["run"] for i in infos}:
        infos.append(info)
    p = os.path.join(run_dir, "oof_points.csv")
    if not os.path.isfile(p):
        acf.loi(f"{name}: thieu {p}")
    d = pd.read_csv(p, dtype={"point_id": str}, usecols=["point_id", "season", "err", "pred_source"])
    if (d["pred_source"] != "oof").any():
        raise ValueError(f"{p}: co dong khong phai oof")
    s = d.set_index(["point_id", "season"])["err"]
    if s.index.duplicated().any():
        raise ValueError(f"{p}: trung (diem, mua)")
    return s


def runs(a, infos=None):
    prefix = {"k50": a.prefix50, "k100": a.prefix100}
    for design, seeds in DESIGNS:
        for g in GRIDS:
            for s in seeds:
                yield design, g, s, (lambda d=design, g=g, s=s: read_err(a.exp_root, prefix[d], g, s, a.target, infos,
                                                                         a.model))


def common_keys(a, infos=None) -> pd.MultiIndex:
    """(point_id, season) co sai so hop le o MOI (thiet ke, luoi, cach chia)."""
    keys, n_all = None, set()
    for *_, load in runs(a, infos):
        e = load()
        n_all.update(e.index)
        ok = e.index[e.notna().to_numpy()]
        keys = ok if keys is None else keys.intersection(ok)
    return keys, len(n_all)


def mae_table(a, keys) -> pd.DataFrame:
    rows = [{"design": d, "grid": g, "seed": s, "MAE": float(load().reindex(keys).abs().mean())}
            for d, g, s, load in runs(a)]
    return pd.DataFrame(rows)


def load_target_gate(a):
    if a.target == "salinity" and a.model == "hist_gb":
        return None
    return load_gate(a.gate, a.target, required=gate_required(a.target, a.model), model=a.model)


def fine_rule(a, ref_path, gate):
    """(delta_min muc min, muc min co kiem dinh, delta_hat F1 theo cap) cho target."""
    tested = FINE_LEVEL in TESTED_TIERS[a.target] and (gate is None or gate.get(FINE_LEVEL, False))
    k = None
    if os.path.exists(ref_path):
        k = pd.read_csv(ref_path)
        if "target" in k.columns and (k["target"] != a.target).any():
            acf.loi(f"{ref_path}: co dong target khac '{a.target}'")
        k = k[k["family"] == main_family(a.model)]
        if "kiem_dinh" in k.columns and "muc" in k.columns:  # file CV phai cung quy tac muc min
            kd = set(k.loc[k["muc"] == FINE_LEVEL, "kiem_dinh"].astype(bool))
            if kd and kd != {tested}:
                acf.loi(f"{ref_path}: kiem_dinh muc min {sorted(kd)} khac quy tac hien tai ({tested})")
    if a.target == "salinity" and a.model == "hist_gb":
        return DELTA_MIN_FINE, tested, k
    if k is None:
        acf.loi(f"thieu ket qua analyze_cv_family {ref_path} (can Delta_min muc min cua {a.target})")
    thr = k.loc[k["delta_ref_level"] == FINE_LEVEL, "delta_min_thr"].dropna().unique()
    if len(thr) != 1:
        acf.loi(f"{ref_path}: Delta_min muc min khong duy nhat ({thr})")
    return float(thr[0]), tested, k


def main(a):
    if a.target not in TESTED_TIERS:
        acf.loi(f"--target '{a.target}' khong co trong TESTED_TIERS")
    a.gate = a.gate or os.path.join(a.out_dir, acf.gate_name(a.model))
    cand = [p for n in out_names(a.target) for p in all_scoped_names(os.path.join(a.out_dir, n), a.model)]
    guard_frozen(cand + [provenance_path(p) for p in cand], a.frozen_manifest)
    try:
        gate = load_target_gate(a)
    except ValueError as exc:
        acf.loi(str(exc))
    differ = scopes_differ(a.target, gate)
    out_mae, out_cap = (os.path.join(a.out_dir, scoped_name(n, a.holm_scope, differ, a.model))
                        for n in out_names(a.target))
    ref = os.path.join(a.ref_dir or a.out_dir,
                       scoped_name(acf.out_name(a.target, a.prefix50), a.holm_scope, differ, a.model))
    try:
        dmin, fine_tested, k = fine_rule(a, ref, gate)
    except ValueError as exc:
        acf.loi(str(exc))
    infos = []
    keys, n_all = common_keys(a, infos)
    same = acf.check_consistent(infos, a.allowed_tags)
    print(f"(diem, mua) chung: {len(keys)} / {n_all}")
    m = mae_table(a, keys)
    piv = m.pivot_table(index="grid", columns=["design", "seed"], values="MAE")

    # Nguong 1
    t1 = pd.DataFrame({"MAE_k50_tb3": piv["k50"].mean(axis=1), "MAE_k100_tbP1P2": piv["k100"].mean(axis=1),
                       "MAE_k50_s42": piv[("k50", 42)], "MAE_k100_P1": piv[("k100", 42)],
                       "MAE_k100_P2": piv[("k100", 43)]}).reindex(GRIDS)
    t1["tang_rel_tb"] = t1["MAE_k100_tbP1P2"] / t1["MAE_k50_tb3"] - 1
    t1["tang_rel_P1_vs_s42"] = t1["MAE_k100_P1"] / t1["MAE_k50_s42"] - 1
    t1["vuot_5pct"] = t1["tang_rel_tb"] > MAE_RISE_MAX
    t1 = t1.reset_index().rename(columns={"index": "grid"})
    t1.insert(0, "target", a.target)
    t1.insert(2, "n_diem_mua_chung", len(keys))

    # Nguong 2
    rows = []
    for ga, gb in FINE_PAIRS:
        r = {"target": a.target, "grid_a": ga, "grid_b": gb}
        for design, seeds in DESIGNS:
            d = [piv.loc[ga, (design, s)] - piv.loc[gb, (design, s)] for s in seeds]
            r.update({f"delta_{design}_s{s}": v for s, v in zip(seeds, d)})
            r[f"delta_{design}_tb"] = sum(d) / len(d)
        r["abs_delta_k100_tb"] = abs(r["delta_k100_tb"])
        r["delta_min"] = dmin
        r["vuot_delta_min"] = r["abs_delta_k100_tb"] >= dmin
        r["cung_chieu_k50"] = (r["delta_k100_tb"] > 0) == (r["delta_k50_tb"] > 0)
        r["kiem_dinh"] = fine_tested
        rows.append(r)
    t2 = pd.DataFrame(rows)
    if k is not None:  # doi chieu: Delta khoi 50 km tinh lai vs delta_hat cua ho chinh
        kk = k.set_index(["grid_a", "grid_b"])["delta_hat"]
        t2["delta_hat_cv1"] = [kk.get((ga, gb)) for ga, gb in FINE_PAIRS]
    for t in (t1, t2):
        t["git_tag"], t["points_ref_sha256"] = same["git_tag"], same["points_ref_sha256"]
    hp = holm_provenance(a.target, gate, a.gate, a.holm_scope)
    ref_info = {"cv_ref": os.path.abspath(ref).replace("\\", "/") if k is not None else None,
                "cv_ref_sha256": file_sha256(ref) if k is not None else None}
    for t, p in ((t1, out_mae), (t2, out_cap)):
        write_csv_atomic(t, p)
        write_provenance(p, **hp, **ref_info, git_tag=same["git_tag"], points_ref_sha256=same["points_ref_sha256"],
                         n_luot=len(infos), **model_provenance(a.model),
                         **acf.tag_info(same, a.allowed_tags, a.tags_reason))

    if not a.no_table:
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(t1.round(4).to_string(index=False))
            print(t2.round(4).to_string(index=False))
        rise = bool(t1["vuot_5pct"].any())
        over = bool(t2["vuot_delta_min"].any())
        print("NGUONG 1 - MAE tang > 5% o bat ky luoi:", rise,
              "-> dung MAE khoi 100 km lam sai so tuyet doi chinh" if rise else "-> giu MAE 50 km, 100 km la do nhay")
        if fine_tested:
            print(f"NGUONG 2 - co cap muc min |Delta| >= {dmin:.4g}:", over,
                  "-> ha ket luan: khong vung theo kich thuoc khoi" if over
                  else "-> giu '4 khung tuong duong o muc min'")
    if not fine_tested:
        print(f"NGUONG 2 - muc min KHONG kiem dinh cho {a.target} (CHG-25 / cong) -> chi mo_ta")
    print(f"Ghi: {out_mae}, {out_cap} | model {a.model} | git_tag {same['git_tag']} | holm {a.holm_scope}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix50", default="cv1")
    ap.add_argument("--prefix100", default="k100")
    ap.add_argument("--target", default="salinity")
    ap.add_argument("--out-dir", default=acf.RESULTS)
    ap.add_argument("--ref-dir", default=None, help="thu muc ket qua analyze_cv_family (mac dinh = --out-dir)")
    ap.add_argument("--gate", default=None, help=f"bang cong (mac dinh <out-dir>/{acf.GATE_FILE}; rf/mlp: dot7_cong_kiem_dinh__<model>.csv)")
    ap.add_argument("--exp-root", default=acf.EXP_ROOT)
    acf.add_common_args(ap)
    ap.add_argument("--holm-scope", choices=HOLM_SCOPES, default="cong")
    ap.add_argument("--no-table", action="store_true", help="khong in bang so ra stdout")
    return ap


if __name__ == "__main__":
    main(build_parser().parse_args())

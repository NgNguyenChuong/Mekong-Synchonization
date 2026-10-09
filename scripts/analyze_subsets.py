#!/usr/bin/env python
"""Phan tich BO NHAN PHU (S4, Dot 6) - thiet ke An CHOT TRUOC khi mo ket qua S1/S3 (NHAT_KY 2026-10-06 14:19).

Cau hoi: THEM nhan nam ngoai pham vi (nuoc / rung ngap man / dat 60-90) vao huan luyen co lam THAY DOI du doan
tren dat nong nghiep khong (KHONG phai "nhan o cac vung do co dung khong").

Cham chinh (kiem dinh): moi bo s in {keepwater, keepmangrove, keep6090} x 13 luoi:
  Delta = MAE(huan luyen bo s) - MAE(bo chinh), cung (diem, mua) CV, dap an bo chinh (points_reference chinh).
  Don vi = khoi CV (unit_id, G = 19), MAE theo don vi = TB deu theo mua (unit_metrics, min_pts 30), sai so TB qua
  3 cach chia (seed_mean_errors), trong so n_pts_mean - nhu ho chinh / analyze_hybrid buoc 1.
  Kiem dinh doi dau theo khoi (signflip_test), Holm m = 13 TRONG MOI BO, TOST: KTC 90% (t theo cum) nam tron trong
  +-Delta_min, Delta_min = 5% x MAE TB cua MUC (level mean nhu ho F1: TB MAE bo chinh cua cac luoi cung muc co mat
  trong 12 cap F1; luoi khong vao cap F1 - s2_level_9/10/11 - dung muc cua tier cua no).
  Nhan: khac_co_y_nghia (Holm-p < 0,05, Delta != 0) / tuong_duong (TOST) / chua_phan_dinh.
  Delta > 0 co y nghia = them lop X lam du doan dat nong nghiep KEM di.
  Quy tac D: chay rieng tung cach chia (Holm m = 13 moi bo, nguong tinh lai tren cach chia do nhu do_nhay F1);
  ket luan chi giu neu ca 3 cach chia cung nhan (+ cung chieu voi khac_co_y_nghia) -> cot giu_ca_3_cach_chia.
Phu (MO TA, khong kiem dinh them): Delta khung cua 12 cap F1 tinh lai duoi moi bo phu, so chieu + do lon voi bo chinh.
Cham phu 349 diem 60/90 (MO TA kem KTC): 312 diem CV (OOF, TB 3 cach chia) + 37 diem giu rieng (final s42), dap an
  keep6090; KTC 95% bootstrap theo don vi (unit_id) 2.000 lan, seed goc 42 (SeedSequence khoa theo ma dong);
  ngu canh = MAE mo hinh keep6090 (luot cv1, dap an bo chinh) tren diem dat nong nghiep CUNG don vi.
  So tuyet doi CV 50 km "co the lac quan" (bo phu khong chay khoi 100 km).

CHG-22 (3 trang thai): thieu file / run_meta; err hoac y_ref khong huu han; pred_source sai; trung (diem, mua);
  run_meta sai label_set / bien the dap an / mode / returncode; tap (seed, diem, mua, don vi) KHAC giua bo chinh va
  bo phu hoac giua cac luoi; y_ref bo phu != y_ref bo chinh; sha cv_folds / points_ref / points khac; G != ky vong
  -> LOI (dung, khong ghi ket qua). Xoa ket qua cu khi bat dau; ghi file tam roi os.replace.

Chay:  venv/Scripts/python.exe scripts/analyze_subsets.py --prefix cv1 --aux-prefix phu6090
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze_cv_family import GRIDS, area_levels, area_table, pairs_by_area  # noqa: E402
from analyze_hybrid import KEYS, _point_unit, assert_same_keys  # noqa: E402
from run_experiments import run_name  # noqa: E402
from training.block_stats import (_arm_level, _pair_d, _wmean, cluster_t_ci, comparison_seed,  # noqa: E402
                                  seed_mean_errors, signflip_test, unit_metrics)
from training.evaluate import holm_adjust  # noqa: E402

CHINH = "chinh"
SUBSETS = ("keepwater", "keepmangrove", "keep6090")
AUX_SET = "keep6090"
MODEL = "hist_gb"
ALPHA = 0.05
DMIN_FRAC = 0.05
OUT_FILES = {
    "cham_chinh": "dot6_bo_phu_cham_chinh.csv",
    "theo_cach_chia": "dot6_bo_phu_cham_chinh_theo_cach_chia.csv",
    "f1": "dot6_bo_phu_f1.csv",
    "phu_6090": "dot6_bo_phu_cham_phu_6090.csv",
}
NOTE_CV = "so tuyet doi CV khoi 50 km - co the lac quan (bo phu khong chay khoi 100 km)"
NOTE_FINAL = ("giu rieng = nhom khong_gian (khoi giu rieng, mua != 2020; muc 10), final 1 cach chia; "
              "khong co luot final keep6090 tren diem nong nghiep -> khong co ngu canh")


def loi(msg):
    raise SystemExit(f"LOI: {msg}")


# ---------------------------------------------------------
# Doc luot chay
# ---------------------------------------------------------
def read_meta(run_dir, name):
    p = os.path.join(run_dir, "run_meta.json")
    if not os.path.isfile(p):
        loi(f"{name}: thieu {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check_meta(meta, name, label_set, variant, mode):
    """run_meta phai dung bo nhan, bien the dap an (khong co khoa = 'main', lan chay truoc S2), mode, returncode 0."""
    got = {"label_set": meta.get("label_set") or CHINH, "points_ref_variant": meta.get("points_ref_variant") or "main",
           "mode": meta.get("mode"), "returncode": meta.get("returncode")}
    want = {"label_set": label_set, "points_ref_variant": variant, "mode": mode, "returncode": 0}
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        loi(f"{name}: run_meta khac ky vong (co, can) {bad}")


def load_points(path, name, pred_source):
    """Mot file oof_points / final_points: kiem pred_source, err + y_ref huu han, unit_id, khong trung (diem, mua).

    final_points chua ca nhom khong_gian / thoi_gian / ca_hai -> CHI giu nhom KHONG GIAN (khoi giu rieng, mua != 2020;
    muc 10 NHAT_KY 2026-10-05: nhom cho ket luan chinh). Thieu cot test_group -> LOI.
    """
    if not os.path.isfile(path):
        loi(f"{name}: thieu file {path}")
    cols = ["point_id", "season", "unit_id", "y_ref", "err", "pred_source"]
    if pred_source == "final":
        cols.append("test_group")
    have = pd.read_csv(path, nrows=0).columns
    if missing := [c for c in cols if c not in have]:
        loi(f"{name}: thieu cot {missing} trong {path}")
    d = pd.read_csv(path, dtype={"point_id": str, "unit_id": str}, usecols=cols)
    if pred_source == "final":
        d = d[d["test_group"] == "khong_gian"].drop(columns="test_group")
    if d.empty:
        loi(f"{name}: {path} rong")
    if (d["pred_source"] != pred_source).any():
        loi(f"{name}: {int((d['pred_source'] != pred_source).sum())} dong pred_source khac '{pred_source}'")
    for c in ("err", "y_ref"):
        bad = ~np.isfinite(d[c].to_numpy(float))
        if bad.any():
            loi(f"{name}: {int(bad.sum())} {c} khong huu han")
    if d["unit_id"].isna().any():
        loi(f"{name}: unit_id NaN")
    if d.duplicated(["point_id", "season"]).any():
        loi(f"{name}: trung (point_id, season)")
    return d.drop(columns="pred_source")


def load_arm(exp, prefix, grid, schemes, label_set, variant="main", mode="cv"):
    """Cac luot (grid, label_set) qua cac cach chia -> (bang grid/seed/point_id/season/unit_id/y_ref/err, meta theo seed).

    label_set = CHINH -> ten luot khong co hau to bo.
    """
    parts, metas = [], {}
    fname = "oof_points.csv" if mode == "cv" else "final_points.csv"
    src = "oof" if mode == "cv" else "final"
    for s in schemes:
        name = run_name(prefix, grid, MODEL, s, "" if label_set == CHINH else label_set)
        run_dir = os.path.join(exp, name, mode)
        meta = read_meta(run_dir, name)
        check_meta(meta, name, label_set, variant, mode)
        d = load_points(os.path.join(run_dir, fname), name, src)
        d["grid"], d["seed"] = grid, s
        parts.append(d)
        metas[s] = meta
    return pd.concat(parts, ignore_index=True), metas


def check_same_answer(chinh, sub, metas_c, metas_s, what):
    """Bo phu va bo chinh: CUNG tap (seed, diem, mua, don vi), CUNG y_ref (dap an bo chinh), cung sha fold/diem/dap an."""
    assert_same_keys({f"bo chinh {what}": chinh, f"{what}": sub})
    a = chinh.sort_values(KEYS)["y_ref"].to_numpy(float)
    b = sub.sort_values(KEYS)["y_ref"].to_numpy(float)
    if not np.allclose(a, b, rtol=0, atol=1e-9):
        loi(f"{what}: {int((~np.isclose(a, b, rtol=0, atol=1e-9)).sum())} y_ref khac bo chinh (khong phai dap an bo chinh)")
    for s, mc in metas_c.items():
        for k in ("cv_folds_sha256", "points_ref_sha256", "points_sha256"):
            if metas_s[s].get(k) != mc.get(k):
                loi(f"{what} s{s}: {k} khac bo chinh")


# ---------------------------------------------------------
# Chi so theo don vi
# ---------------------------------------------------------
def units_of(df, label_set, pu, min_pts=30):
    """MAE theo (grid, don vi) cua mot bo: sai so TB qua cach chia -> unit_metrics (TB deu theo mua, bo < min_pts)."""
    me = seed_mean_errors(df[["grid", "seed", "point_id", "season", "err"]].assign(model=label_set))
    return unit_metrics(me, pu, min_pts)[0]


def level_refs(units, ref_levels):
    """Muc tham chieu moi tier = TB MAE bo chinh (trong so n_pts_mean) cua cac luoi trong muc (nhu level_mean F1)."""
    out = {}
    for lv in sorted(ref_levels.unique()):
        vals = [_arm_level(units, (g, CHINH), "mae") for g in sorted(ref_levels.index[ref_levels == lv])]
        v = float(np.mean(vals)) if vals else np.nan
        if not (np.isfinite(v) and v > 0):
            loi(f"muc tham chieu tier {lv} khong huu han / <= 0 ({v})")
        out[lv] = v
    return out


def label_of(p_holm, delta, tost, alpha=ALPHA):
    if p_holm < alpha and delta != 0:
        return "khac_co_y_nghia"
    if tost:
        return "tuong_duong"
    return "chua_phan_dinh"


def subset_test(units, label_set, grids, tier_of, refs, tag, alpha=ALPHA, dmin=DMIN_FRAC, expected_g=None):
    """Cham chinh mot bo: Delta = MAE(bo) - MAE(chinh) moi luoi; Holm m = len(grids) trong bo."""
    rows = []
    for g in grids:
        d, w = _pair_d(units, (g, label_set), (g, CHINH), "mae")  # assert cung tap don vi + cung trong so
        G = len(d)
        if G < 2:
            loi(f"{label_set} {g} {tag}: chi con {G} don vi sau loc min_pts")
        if expected_g is not None and G != expected_g:
            loi(f"{label_set} {g} {tag}: G = {G} don vi, ky vong {expected_g}")
        wn = (w / w.sum()).to_numpy()
        sf = signflip_test(d.to_numpy(), wn, seed=comparison_seed(f"S4|{tag}|{label_set}|{g}"))
        ci2 = cluster_t_ci(d.to_numpy(), wn, 2 * alpha)
        lv = tier_of[g]
        if lv not in refs:
            loi(f"{g}: tier {lv} khong co muc tham chieu")
        thr = dmin * refs[lv]
        mae_c = _arm_level(units, (g, CHINH), "mae")
        rows.append({
            "bo": label_set, "cach_chia": tag, "grid": g, "muc": lv, "G": G,
            "mae_chinh": mae_c, "mae_bo": _arm_level(units, (g, label_set), "mae"),
            "delta": sf["delta_hat"], "delta_tuong_doi": sf["delta_hat"] / mae_c,
            "p_value": sf["p_value"], "p_exact": sf["exact"],
            "don_vi_tang": f"{int((d > 0).sum())}/{G}",
            "ci90_low": ci2["ci_low"], "ci90_high": ci2["ci_high"],
            "muc_tham_chieu": refs[lv], "nguong": thr,
            "tost_tuong_duong": bool(-thr < ci2["ci_low"] and ci2["ci_high"] < thr),
            "vuot_nguong": bool(abs(sf["delta_hat"]) >= thr),
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"].to_numpy())  # m = so luoi trong bo
    out["holm_m"] = len(out)
    out["nhan"] = [label_of(p, dl, t, alpha) for p, dl, t in zip(out["p_holm"], out["delta"], out["tost_tuong_duong"])]
    return out


def apply_rule_d(tb, per):
    """Quy tac D: them nhan + Delta tung cach chia; giu_ca_3_cach_chia (NaN voi chua_phan_dinh); ket_luan_D."""
    res = tb.copy()
    schemes = sorted(per["cach_chia"].unique())
    p = per.set_index(["bo", "grid", "cach_chia"])
    keep, final = [], []
    for r in res.itertuples():
        labs = [p.loc[(r.bo, r.grid, s), "nhan"] for s in schemes]
        dels = [p.loc[(r.bo, r.grid, s), "delta"] for s in schemes]
        if r.nhan == "khac_co_y_nghia":
            ok = all(lb == r.nhan and np.sign(x) == np.sign(r.delta) for lb, x in zip(labs, dels))
        elif r.nhan == "tuong_duong":
            ok = all(lb == r.nhan for lb in labs)
        else:
            ok = None
        keep.append(ok)
        final.append(r.nhan if ok is None or ok else "khong_vung_theo_cach_chia")
    for s in schemes:
        res[f"nhan_{s}"] = [p.loc[(b, g, s), "nhan"] for b, g in zip(res["bo"], res["grid"])]
        res[f"delta_{s}"] = [p.loc[(b, g, s), "delta"] for b, g in zip(res["bo"], res["grid"])]
    res["giu_ca_3_cach_chia"] = pd.array(keep, dtype="boolean")
    res["ket_luan_D"] = final
    return res


# ---------------------------------------------------------
# Phu: ho F1 duoi moi bo (mo ta)
# ---------------------------------------------------------
def f1_table(units, pairs, tier_of, subsets):
    rows = []
    for a, b in zip(pairs["grid_a"], pairs["grid_b"]):
        dc = _wmean(*_pair_d(units, (a, CHINH), (b, CHINH), "mae"))
        for s in subsets:
            ds = _wmean(*_pair_d(units, (a, s), (b, s), "mae"))
            rows.append({"bo": s, "muc": tier_of[a], "grid_a": a, "grid_b": b, "delta_chinh": dc, "delta_bo": ds,
                         "cung_chieu": bool(np.sign(ds) == np.sign(dc) and dc != 0),
                         "ti_le_do_lon": ds / dc if dc != 0 else np.nan, "chenh": ds - dc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Cham phu 60/90 (mo ta + KTC bootstrap theo don vi)
# ---------------------------------------------------------
def bootstrap_mae(abs_err, unit, n_boot, seed):
    """KTC 95% percentile cho MAE gop (diem, mua): lay lai DON VI co hoan lai n_boot lan."""
    t = pd.DataFrame({"a": np.asarray(abs_err, float), "u": np.asarray(unit)}).groupby("u")["a"].agg(["sum", "size"])
    S, N = t["sum"].to_numpy(), t["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(n_boot, len(t)))
    m = S[idx].sum(axis=1) / N[idx].sum(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def _mae_point_season(df):
    """|e| TB qua cach chia theo (diem, mua) + don vi (seed_mean_errors kiem du seed moi diem)."""
    me = seed_mean_errors(df[["grid", "seed", "point_id", "season", "err"]].assign(model="x"))
    unit = df.drop_duplicates("point_id").set_index("point_id")["unit_id"]
    return me.assign(unit=me["point_id"].map(unit))


def _check_one_unit(df, what):
    if df.groupby("point_id")["unit_id"].nunique().gt(1).any():
        loi(f"{what}: mot diem co nhieu unit_id")


def aux_6090_rows(grid, cv, fin, ctx, n_boot, boot_seed):
    """2 dong (cv_oof, giu_rieng_final) cho mot luoi. ctx = luot cv1 keep6090 (dap an bo chinh) tren diem nong nghiep."""
    for d, w in ((cv, "6090 cv"), (fin, "6090 final"), (ctx, "ngu canh")):
        _check_one_unit(d, f"{grid} {w}")
    overlap = set(cv["point_id"]) & set(fin["point_id"])
    if overlap:
        loi(f"{grid}: {len(overlap)} diem 60/90 vua o CV vua o giu rieng")
    overlap = (set(cv["point_id"]) | set(fin["point_id"])) & set(ctx["point_id"])
    if overlap:
        loi(f"{grid}: {len(overlap)} diem 60/90 trung point_id voi diem nong nghiep")
    rows = []
    for phan, d, note in (("cv_oof", cv, NOTE_CV), ("giu_rieng_final", fin, NOTE_FINAL)):
        me = _mae_point_season(d)
        lo, hi = bootstrap_mae(me["abs_err"], me["unit"], n_boot, comparison_seed(f"S4|6090|{phan}|{grid}", boot_seed))
        units = sorted(me["unit"].unique())
        r = {"grid": grid, "phan": phan, "cach_chia": ";".join(str(s) for s in sorted(d["seed"].unique())),
             "n_diem": int(me["point_id"].nunique()), "n_diem_mua": len(me), "n_don_vi": len(units),
             "mae": float(me["abs_err"].mean()), "ci95_low": lo, "ci95_high": hi, "n_boot": n_boot,
             "mae_nong_nghiep_cung_don_vi": np.nan, "n_diem_nong_nghiep_cung_don_vi": 0, "ghi_chu": note}
        if phan == "cv_oof":
            c = ctx[ctx["unit_id"].isin(units)]
            miss = set(units) - set(c["unit_id"])
            if miss:
                loi(f"{grid}: don vi {sorted(miss)} co diem 60/90 CV nhung khong co diem nong nghiep (ngu canh)")
            mc = _mae_point_season(c)
            r["mae_nong_nghiep_cung_don_vi"] = float(mc["abs_err"].mean())
            r["n_diem_nong_nghiep_cung_don_vi"] = int(mc["point_id"].nunique())
        rows.append(r)
    return rows


# ---------------------------------------------------------
# Luong chinh
# ---------------------------------------------------------
def analyze(exp, prefix, aux_prefix, grids, schemes, subsets, pairs, tier_of, ref_levels, pu, *,
            final_scheme=42, min_pts=30, alpha=ALPHA, dmin=DMIN_FRAC, expected_g=None, n_boot=2000, boot_seed=42):
    """Tra ve dict ten -> DataFrame (khoa nhu OUT_FILES). Doc tung luoi mot (giu RAM thap)."""
    missing = [g for g in grids if g not in tier_of.index]
    if missing:
        loi(f"luoi khong co tier trong bang dien tich: {missing}")
    U = {"tb": [], **{f"s{s}": [] for s in schemes}}
    ref_keys, aux_keys, aux_rows = None, {}, []
    for g in grids:
        chinh, mc = load_arm(exp, prefix, g, schemes, CHINH)
        keys = assert_same_keys({f"bo chinh {g}": chinh}, pu)  # unit_id khop don vi tinh tu vi tri diem
        if ref_keys is None:
            ref_keys = keys
        elif not keys.equals(ref_keys):
            loi(f"bo chinh {g} khac tap (seed, point_id, season, unit_id) voi {grids[0]}")
        arms = {CHINH: chinh}
        for s in subsets:
            sub, ms = load_arm(exp, prefix, g, schemes, s)
            check_same_answer(chinh, sub, mc, ms, f"{s} {g}")
            arms[s] = sub
        for name, df in arms.items():
            U["tb"].append(units_of(df, name, pu, min_pts))
            for s in schemes:
                U[f"s{s}"].append(units_of(df[df["seed"] == s], name, pu, min_pts))
        if aux_prefix and AUX_SET in subsets:
            cv, mcv = load_arm(exp, aux_prefix, g, schemes, AUX_SET, variant=AUX_SET, mode="cv")
            fin, _ = load_arm(exp, aux_prefix, g, [final_scheme], AUX_SET, variant=AUX_SET, mode="final")
            for s in schemes:
                if mcv[s].get("cv_folds_sha256") != mc[s].get("cv_folds_sha256"):
                    loi(f"{aux_prefix} {g} s{s}: cv_folds_sha256 khac bo chinh")
            for part, d in (("cv", cv), ("final", fin)):
                k = d[KEYS].sort_values(KEYS).reset_index(drop=True)
                if part not in aux_keys:
                    aux_keys[part] = k
                elif not k.equals(aux_keys[part]):
                    loi(f"{aux_prefix} {g} ({part}) khac tap (seed, point_id, season, unit_id) voi {grids[0]}")
            aux_rows += aux_6090_rows(g, cv, fin, arms[AUX_SET], n_boot, boot_seed)
        print(f"  {g}: {len(chinh) // len(schemes)} (diem, mua)/cach chia, bo {list(arms)}", flush=True)
        del arms
    U = {k: pd.concat(v, ignore_index=True) for k, v in U.items()}

    tb_parts, per_parts = [], []
    for s in subsets:
        tb_parts.append(subset_test(U["tb"], s, grids, tier_of, level_refs(U["tb"], ref_levels), "tb",
                                    alpha, dmin, expected_g))
        for sc in schemes:
            per_parts.append(subset_test(U[f"s{sc}"], s, grids, tier_of, level_refs(U[f"s{sc}"], ref_levels),
                                         f"s{sc}", alpha, dmin, expected_g))
    per = pd.concat(per_parts, ignore_index=True)
    out = {"cham_chinh": apply_rule_d(pd.concat(tb_parts, ignore_index=True), per), "theo_cach_chia": per,
           "f1": f1_table(U["tb"], pairs, tier_of, subsets)}
    if aux_rows:
        out["phu_6090"] = pd.DataFrame(aux_rows)
    return out


def clear_outputs(out_dir):
    for f in OUT_FILES.values():
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)


def write_outputs(out_dir, tables):
    os.makedirs(out_dir, exist_ok=True)
    for k, df in tables.items():
        p = os.path.join(out_dir, OUT_FILES[k])
        tmp = p + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, p)


def main(a):
    exp = a.exp_root
    at = area_table(a.area_table)
    pairs = pairs_by_area(at, max_ratio=1.2, expected_m=12)
    ref_levels = area_levels(at, pairs=pairs)  # luoi cua muc tham chieu = luoi trong 12 cap F1 (nhu ho F1)
    tier_of = area_levels(at)                  # tier cua MOI luoi (gom s2_level_9/10/11)
    if len(GRIDS) != 13:
        loi(f"GRIDS co {len(GRIDS)} luoi, ky vong 13 (Holm m = 13)")
    clear_outputs(a.out_dir)
    pu = _point_unit()
    tables = analyze(exp, a.prefix, a.aux_prefix, GRIDS, a.schemes, a.subsets, pairs, tier_of, ref_levels, pu,
                     final_scheme=a.final_scheme, min_pts=a.min_pts, alpha=a.alpha, dmin=a.delta_min,
                     expected_g=a.expected_g or None, n_boot=a.n_boot, boot_seed=a.boot_seed)
    write_outputs(a.out_dir, tables)
    cols = ["bo", "grid", "G", "delta", "delta_tuong_doi", "ci90_low", "ci90_high", "nguong", "p_holm", "nhan",
            "giu_ca_3_cach_chia", "ket_luan_D"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(tables["cham_chinh"][cols].to_string(index=False))
        if "phu_6090" in tables:
            print(tables["phu_6090"].drop(columns="ghi_chu").to_string(index=False))
    print(f"Ghi: {[os.path.join(a.out_dir, OUT_FILES[k]) for k in tables]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--aux-prefix", default="phu6090", help='"" = bo qua cham phu 60/90')
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--final-scheme", type=int, default=42)
    ap.add_argument("--exp-root", default=os.path.join(ROOT, "artifacts", "experiments"))
    ap.add_argument("--area-table", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_doi_chieu_dien_tich.csv"))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--delta-min", type=float, default=DMIN_FRAC)
    ap.add_argument("--min-pts", type=int, default=30)
    ap.add_argument("--expected-g", type=int, default=19, help="0 = khong kiem so don vi")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    main(ap.parse_args())

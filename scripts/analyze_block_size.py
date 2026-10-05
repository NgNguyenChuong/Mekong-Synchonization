#!/usr/bin/env python
"""CHG-20 tieu chi 2: so CV khoi 50 km (cv1, s42/s43/s44) voi khoi ~100 km (k100, P1=s42 / P2=s43), HistGB.

Nguong + he qua ghi TRUOC khi chay (NHAT_KY 2026-10-05, CHG-20):
  1. MAE theo diem tung luoi: TB(P1, P2) khoi 100 km vs TB(s42, s43, s44) khoi 50 km; bao them P1 vs s42.
     Tang > 5% o BAT KY luoi nao -> bai dung MAE khoi 100 km lam sai so tuyet doi chinh.
  2. 6 cap ho chinh muc min: |Delta| (TB P1, P2) < Delta_min muc min 0,0309 dS/m o MOI cap -> giu ket luan
     "4 khung tuong duong o muc min"; co cap vuot -> ha thanh "khong vung theo kich thuoc khoi".
Tap so sanh: (diem, mua) co sai so hop le o MOI luoi, MOI cach chia, CA HAI thiet ke (cung diem cho moi phep so).
Delta = MAE(a) - MAE(b) (cung chieu delta_hat cua analyze_cv_family.py); script doi chieu lai delta_hat khoi 50 km.

Chay:  venv/Scripts/python.exe scripts/analyze_block_size.py
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze_cv_family import GRIDS  # noqa: E402

MAE_RISE_MAX = 0.05
DELTA_MIN_FINE = 0.0309
FINE_PAIRS = [("h3_res_7", "s2_level_12"), ("h3_res_7", "square_utm_2441m"), ("h3_res_7", "latlon_0.0222deg"),
              ("s2_level_12", "square_utm_2441m"), ("s2_level_12", "latlon_0.0222deg"),
              ("square_utm_2441m", "latlon_0.0222deg")]
DESIGNS = (("k50", [42, 43, 44]), ("k100", [42, 43]))


def read_err(exp, prefix, grid, scheme) -> pd.Series:
    """Sai so OOF theo (point_id, season) cua mot lan chay; doc tung lan (it bo nho)."""
    p = os.path.join(exp, f"{prefix}__{grid}__hist_gb__s{scheme}", "cv", "oof_points.csv")
    d = pd.read_csv(p, dtype={"point_id": str}, usecols=["point_id", "season", "err", "pred_source"])
    if (d["pred_source"] != "oof").any():
        raise ValueError(f"{p}: co dong khong phai oof")
    s = d.set_index(["point_id", "season"])["err"]
    if s.index.duplicated().any():
        raise ValueError(f"{p}: trung (diem, mua)")
    return s


def runs(a):
    exp = os.path.join(ROOT, "artifacts", "experiments")
    prefix = {"k50": a.prefix50, "k100": a.prefix100}
    for design, seeds in DESIGNS:
        for g in GRIDS:
            for s in seeds:
                yield design, g, s, (lambda d=design, g=g, s=s: read_err(exp, prefix[d], g, s))


def common_keys(a) -> pd.MultiIndex:
    """(point_id, season) co sai so hop le o MOI (thiet ke, luoi, cach chia)."""
    keys, n_all = None, set()
    for *_, load in runs(a):
        e = load()
        n_all.update(e.index)
        ok = e.index[e.notna().to_numpy()]
        keys = ok if keys is None else keys.intersection(ok)
    return keys, len(n_all)


def mae_table(a, keys) -> pd.DataFrame:
    rows = [{"design": d, "grid": g, "seed": s, "MAE": float(load().reindex(keys).abs().mean())}
            for d, g, s, load in runs(a)]
    return pd.DataFrame(rows)


def main(a):
    keys, n_all = common_keys(a)
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
    t1.insert(1, "n_diem_mua_chung", len(keys))
    t1.to_csv(os.path.join(a.out_dir, "dot6_khoi100_mae.csv"), index=False)

    # Nguong 2
    rows = []
    for ga, gb in FINE_PAIRS:
        r = {"grid_a": ga, "grid_b": gb}
        for design, seeds in DESIGNS:
            d = [piv.loc[ga, (design, s)] - piv.loc[gb, (design, s)] for s in seeds]
            r.update({f"delta_{design}_s{s}": v for s, v in zip(seeds, d)})
            r[f"delta_{design}_tb"] = sum(d) / len(d)
        r["abs_delta_k100_tb"] = abs(r["delta_k100_tb"])
        r["delta_min"] = DELTA_MIN_FINE
        r["vuot_delta_min"] = r["abs_delta_k100_tb"] >= DELTA_MIN_FINE
        r["cung_chieu_k50"] = (r["delta_k100_tb"] > 0) == (r["delta_k50_tb"] > 0)
        rows.append(r)
    t2 = pd.DataFrame(rows)
    ref = os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot5_cv1_kiem_dinh_khoi.csv")
    if os.path.exists(ref):  # doi chieu: Delta khoi 50 km tinh lai vs delta_hat cua ho chinh (cung tap chung?)
        k = pd.read_csv(ref)
        k = k[k["family"] == "F1_chinh"].set_index(["grid_a", "grid_b"])["delta_hat"]
        t2["delta_hat_cv1"] = [k.get((ga, gb)) for ga, gb in FINE_PAIRS]
    t2.to_csv(os.path.join(a.out_dir, "dot6_khoi100_cap_min.csv"), index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(t1.round(4).to_string(index=False))
        print(t2.round(4).to_string(index=False))
    rise = bool(t1["vuot_5pct"].any())
    over = bool(t2["vuot_delta_min"].any())
    print("NGUONG 1 - MAE tang > 5% o bat ky luoi:", rise,
          "-> dung MAE khoi 100 km lam sai so tuyet doi chinh" if rise else "-> giu MAE 50 km, 100 km la do nhay")
    print("NGUONG 2 - co cap muc min |Delta| >= 0,0309:", over,
          "-> ha ket luan: khong vung theo kich thuoc khoi" if over else "-> giu '4 khung tuong duong o muc min'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix50", default="cv1")
    ap.add_argument("--prefix100", default="k100")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    main(ap.parse_args())

#!/usr/bin/env python
"""Do nhay kich thuoc khoi cua boc tach hybrid - CHI MO TA (CHG-21, An chot 2026-10-05 21:22, TRUOC khi mo bat ky
ket qua khoi 100 km nao cua boc tach hybrid).

I = MAE(b) - MAE(a), MAE gop theo (diem, mua); (b) = bo dac trung `khong_diem`, (a) = day du; HistGB.
Moi luoi:
  I50_s   (s = s42/s43/s44, khoi 50 km, prefix cv1); dai nhieu cach chia 50 km = [min, max] I50_s; I_50 = TB I50_s
  I100_s  (P1 = s42, P2 = s43, khoi ~100 km, prefix k100); I_100 = TB I100_s
  trong_dai_nhieu_50km = min I50_s <= I_100 <= max I50_s
Tap (diem, mua): CUNG MOT tap o MOI lan chay cua ca 4 nhom (a/b x 50/100 km) va = --expected-n (86.073).
Khong co nhan "vung"/"phu thuoc" (quy tac 21:06 thay bang CHG-21 - o khoi 50 km khong co hieu ung de lam moc).
Tom tat in ra (mo ta): khoang I_100, so luoi I_100 > 0, so luoi trong dai nhieu 50 km. Luoi trong dai -> "khong co
bang chung hieu ung thay doi theo kich thuoc khoi"; ngoai dai -> chi bao so, khong gan nhan.
Kiem dinh chinh thuc cua boc tach van o khoi 50 km (analyze_hybrid.py).
CHG-22 (3 trang thai): thieu file / pred_source khac oof / err NaN / trung (diem, mua) / tap (diem, mua) khac nhau
giua cac lan chay / tap chung rong hoac != --expected-n -> LOI, dung, khong ghi ket qua (file ket qua cu bi xoa
khi bat dau; ghi file tam roi os.replace).

Chay:  venv/Scripts/python.exe scripts/analyze_hybrid_block_size.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze_cv_family import GRIDS  # noqa: E402

FEATURE_SET = "khong_diem"
EXPECTED_N = 86073
SEEDS = {"k50": (42, 43, 44), "k100": (42, 43)}  # k100: P1 = s42, P2 = s43
OUT_NAME = "dot6_hybrid_khoi100.csv"
CAU_TRONG_DAI = "khong co bang chung hieu ung thay doi theo kich thuoc khoi"


def run_dir(exp_root, prefix, grid, scheme, feature_set=None):
    name = f"{prefix}__{grid}__hist_gb__s{scheme}" + (f"__fs-{feature_set}" if feature_set else "")
    return os.path.join(exp_root, name, "cv", "oof_points.csv")


def read_err(path) -> pd.Series:
    """err theo (point_id, season), da sap xep. Thieu file / khong phai oof / NaN / trung -> LOI."""
    if not os.path.exists(path):
        raise SystemExit(f"LOI: thieu {path}")
    d = pd.read_csv(path, dtype={"point_id": str}, usecols=["point_id", "season", "err", "pred_source"])
    if (d["pred_source"] != "oof").any():
        raise SystemExit(f"LOI: {path} co dong khong phai oof")
    if not np.isfinite(d["err"].to_numpy(float)).all():
        raise SystemExit(f"LOI: {path} co {int((~np.isfinite(d['err'].to_numpy(float))).sum())} err NaN/khong huu han")
    s = d.set_index(["point_id", "season"])["err"].sort_index()
    if s.index.duplicated().any():
        raise SystemExit(f"LOI: {path} trung (diem, mua)")
    return s


def runs(a):
    for design, prefix in (("k50", a.prefix50), ("k100", a.prefix100)):
        for arm, fs in (("a", None), ("b", a.feature_set)):
            for g in GRIDS:
                for s in SEEDS[design]:
                    yield design, arm, g, s, run_dir(a.exp_root, prefix, g, s, fs)


def mae_table(a):
    """MAE moi lan chay tren tap (diem, mua) chung - moi lan chay phai CUNG tap (khong cat giao am tham)."""
    keys, ref_path, rows = None, None, []
    for design, arm, g, s, p in runs(a):
        e = read_err(p)
        if keys is None:
            keys, ref_path = e.index, p
            if len(keys) == 0:
                raise SystemExit("LOI: tap (diem, mua) chung rong")
            if a.expected_n is not None and len(keys) != a.expected_n:
                raise SystemExit(f"LOI: tap (diem, mua) {len(keys)} != ky vong {a.expected_n} ({p})")
        elif not e.index.equals(keys):
            raise SystemExit(f"LOI: {p} khac tap (diem, mua) voi {ref_path} "
                             f"({len(e.index.difference(keys))} chi o file nay, {len(keys.difference(e.index))} thieu)")
        rows.append({"design": design, "arm": arm, "grid": g, "seed": s, "MAE": float(e.abs().mean())})
    return pd.DataFrame(rows), len(keys)


def block_size_table(m: pd.DataFrame, n_keys: int) -> pd.DataFrame:
    piv = m.pivot_table(index=["design", "grid", "seed"], columns="arm", values="MAE", aggfunc="first")
    i = (piv["b"] - piv["a"]).unstack(["design", "seed"])
    want = [("k50", s) for s in SEEDS["k50"]] + [("k100", s) for s in SEEDS["k100"]]
    i = i.reindex(index=GRIDS, columns=pd.MultiIndex.from_tuples(want, names=["design", "seed"]))
    if not np.isfinite(i.to_numpy(float)).all():
        raise SystemExit("LOI: thieu / khong huu han I o mot luoi / lan chay")
    t = pd.DataFrame({"grid": GRIDS, "n_diem_mua_chung": n_keys})
    for s in SEEDS["k50"]:
        t[f"I50_s{s}"] = i[("k50", s)].to_numpy()
    i50 = t[[f"I50_s{s}" for s in SEEDS["k50"]]]
    t["dai50_min"], t["dai50_max"], t["I_50"] = i50.min(axis=1), i50.max(axis=1), i50.mean(axis=1)
    for s, ten in zip(SEEDS["k100"], ("P1", "P2")):
        t[f"I100_{ten}_s{s}"] = i[("k100", s)].to_numpy()
    t["I_100"] = t[[f"I100_{ten}_s{s}" for s, ten in zip(SEEDS["k100"], ("P1", "P2"))]].mean(axis=1)
    t["trong_dai_nhieu_50km"] = (t["dai50_min"] <= t["I_100"]) & (t["I_100"] <= t["dai50_max"])
    return t


def summary_lines(t: pd.DataFrame) -> list:
    n = len(t)
    trong = t.loc[t["trong_dai_nhieu_50km"], "grid"].tolist()
    ngoai = t.loc[~t["trong_dai_nhieu_50km"]]
    lines = [f"I_100 (khoi ~100 km, TB P1/P2): tu {t['I_100'].min():+.4f} den {t['I_100'].max():+.4f} dS/m",
             f"So luoi I_100 > 0: {int((t['I_100'] > 0).sum())}/{n}",
             f"So luoi I_100 trong dai nhieu cach chia 50 km [min, max] I50_s: {len(trong)}/{n}"]
    if trong:
        lines.append(f"  Trong dai ({', '.join(trong)}): {CAU_TRONG_DAI}.")
    for r in ngoai.itertuples():  # ngoai dai: chi bao so, khong gan nhan
        lines.append(f"  Ngoai dai: {r.grid}: I_100 = {r.I_100:+.4f}, dai 50 km [{r.dai50_min:+.4f}; {r.dai50_max:+.4f}]")
    return lines


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, OUT_NAME)
    if os.path.exists(out):  # dung giua chung (LOI) khong de lai ket qua cu
        os.remove(out)
    m, n_keys = mae_table(a)
    print(f"(diem, mua) chung: {n_keys}", flush=True)
    t = block_size_table(m, n_keys)
    tmp = out + ".tmp"
    t.to_csv(tmp, index=False)
    os.replace(tmp, out)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(t.round(4).to_string(index=False))
    print("\n".join(summary_lines(t)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix50", default="cv1")
    ap.add_argument("--prefix100", default="k100")
    ap.add_argument("--feature-set", default=FEATURE_SET)
    ap.add_argument("--expected-n", type=int, default=EXPECTED_N,
                    help="so (diem, mua) chung ky vong (CV khoi, khong mua 2020); khac -> LOI")
    ap.add_argument("--exp-root", default=os.path.join(ROOT, "artifacts", "experiments"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    main(ap.parse_args())

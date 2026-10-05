#!/usr/bin/env python
"""Do nhay kich thuoc khoi cua boc tach hybrid (An chot 2026-10-05 21:06, TRUOC khi mo ket qua hybrid).

I = MAE(b) - MAE(a) theo diem, (b) = bo dac trung `khong_diem`, (a) = day du; HistGB.
  I_50  = TB qua 3 cach chia s42/s43/s44 khoi 50 km (prefix cv1)
  I_100 = TB qua P1/P2 (s42/s43) khoi ~100 km (prefix k100)
Tap (diem, mua): hop le o MOI lan chay cua ca 4 nhom (a/b x 50/100 km) - cung diem cho moi phep so.
Luoi "vung": sign(I_100) == sign(I_50) VA |I_100| >= 0,5 |I_50|.
Dong gop cua du lieu diem VUNG theo kich thuoc khoi neu >= 7/13 luoi vung; khong thi bai ghi "phu thuoc kich thuoc
khoi". Chi mo ta (kiem dinh chinh thuc cua boc tach van o khoi 50 km). Thieu file / err NaN ngoai du kien -> LOI, dung.

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
MIN_RATIO = 0.5
MIN_GRIDS = 7


def run_dir(prefix, grid, scheme, feature_set=None):
    name = f"{prefix}__{grid}__hist_gb__s{scheme}" + (f"__fs-{feature_set}" if feature_set else "")
    return os.path.join(ROOT, "artifacts", "experiments", name, "cv", "oof_points.csv")


def read_err(path) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(f"LOI: thieu {path}")
    d = pd.read_csv(path, dtype={"point_id": str}, usecols=["point_id", "season", "err", "pred_source"])
    if (d["pred_source"] != "oof").any():
        raise ValueError(f"LOI: {path} co dong khong phai oof")
    s = d.set_index(["point_id", "season"])["err"]
    if s.index.duplicated().any():
        raise ValueError(f"LOI: {path} trung (diem, mua)")
    return s


def runs(a):
    for design, prefix, seeds in (("k50", a.prefix50, (42, 43, 44)), ("k100", a.prefix100, (42, 43))):
        for arm, fs in (("a", None), ("b", a.feature_set)):
            for g in GRIDS:
                for s in seeds:
                    yield design, arm, g, s, run_dir(prefix, g, s, fs)


def main(a):
    keys, n_all = None, set()
    for *_, p in runs(a):
        e = read_err(p)
        n_all.update(e.index)
        ok = e.index[e.notna().to_numpy()]
        keys = ok if keys is None else keys.intersection(ok)
    if keys is None or len(keys) == 0:
        raise ValueError("LOI: tap (diem, mua) chung rong")
    print(f"(diem, mua) chung: {len(keys)} / {len(n_all)}")
    m = pd.DataFrame([{"design": d, "arm": arm, "grid": g, "seed": s, "MAE": float(read_err(p).reindex(keys).abs().mean())}
                      for d, arm, g, s, p in runs(a)])
    piv = m.pivot_table(index=["design", "grid", "seed"], columns="arm", values="MAE")
    piv["I"] = piv["b"] - piv["a"]
    i = piv["I"].groupby(level=["design", "grid"]).mean().unstack("design").reindex(GRIDS)
    if i.isna().any().any():
        raise ValueError("LOI: thieu I o mot luoi/thiet ke")
    t = pd.DataFrame({"grid": GRIDS, "I_50": i["k50"].to_numpy(), "I_100": i["k100"].to_numpy()})
    per = piv["I"].unstack(["design", "seed"])
    for (d, s) in per.columns:
        t[f"I_{d}_s{s}"] = per[(d, s)].reindex(GRIDS).to_numpy()
    t["ti_le"] = t["I_100"] / t["I_50"]
    t["cung_dau"] = np.sign(t["I_100"]) == np.sign(t["I_50"])
    t["vung"] = t["cung_dau"] & (t["I_100"].abs() >= MIN_RATIO * t["I_50"].abs())
    t.insert(1, "n_diem_mua_chung", len(keys))
    t.to_csv(os.path.join(a.out_dir, "dot6_hybrid_khoi100.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(t.round(4).to_string(index=False))
    n = int(t["vung"].sum())
    print(f"Luoi vung: {n}/13 (can >= {MIN_GRIDS}) ->",
          "DONG GOP DU LIEU DIEM VUNG THEO KICH THUOC KHOI" if n >= MIN_GRIDS
          else "dong gop cua hybrid PHU THUOC kich thuoc khoi")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix50", default="cv1")
    ap.add_argument("--prefix100", default="k100")
    ap.add_argument("--feature-set", default=FEATURE_SET)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    main(ap.parse_args())

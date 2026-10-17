"""Ti le phat hien PC-1 theo (muc, k, kind) kem CI Wilson 95%, tong hop tu dot7_pc1_lan_lap.csv (khong chay lai mo phong).

Chay: venv/Scripts/python.exe scripts/pc1_wilson_ci.py --res-dir KE_HOACH/ket-qua
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from md_util import md_table

LEVEL = {5: "coarse", 6: "medium", 7: "fine"}


def wilson(x, n, z=1.959964):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default="KE_HOACH/ket-qua")
    res = Path(ap.parse_args().res_dir)
    d = pd.read_csv(res / "dot7_pc1_lan_lap.csv")
    d["hit"] = d.phat_hien_vung_D.astype(bool)
    d["pair"] = d.grid_a + " vs " + d.grid_b
    pair = d.groupby(["level", "k", "kind", "pair"]).hit.agg(["sum", "count"]).reset_index()
    rows = []
    for (lv, k, kind), g in pair.groupby(["level", "k", "kind"]):
        x, n = int(g["sum"].sum()), int(g["count"].sum())
        lo, hi = wilson(x, n)
        w = g.loc[(g["sum"] / g["count"]).idxmin()]
        wl, wh = wilson(w["sum"], w["count"])
        rows.append(dict(level=LEVEL[lv], k=k, kind=kind, n_pairs=len(g), n_rep_total=n, rate_pct=100 * x / n,
                         ci95_low=100 * lo, ci95_high=100 * hi, weakest_pair=w["pair"], weakest_n=int(w["count"]),
                         weakest_pct=100 * w["sum"] / w["count"], weakest_ci_low=100 * wl, weakest_ci_high=100 * wh))
    out = pd.DataFrame(rows)
    th = pd.read_csv(res / "dot7_pc1_tong_hop.csv")
    chk = (th.ty_le_phat_hien * th.n_rep).round().sum() - d.hit.sum()
    assert abs(chk) < 1e-6, f"lech tong hop vs lan_lap: {chk}"
    md = ["# PC-1: ti le phat hien kem CI Wilson 95 %", "",
          "Phat hien = vung D (ket qua `phat_hien_vung_D` trong dot7_pc1_lan_lap.csv). Ty le gop qua cac cap cung muc "
          "(n_rep_total = so cap x n_rep = 200); cac cap cung muc KHONG doc lap (cung du lieu, khac khung) nen CI gop "
          "chi la tham khao, cot `weakest_*` la cap yeu nhat voi n = 200 (cung la cap quyet dinh ket luan PC-1). "
          "k = muc cay (boi cua delta_min), kind = kieu cay.", "", md_table(out, "{:.1f}"), ""]
    (res / "pc1_ci.md").write_text("\n".join(md), encoding="utf-8")
    out.to_csv(res / "pc1_ci.csv", index=False)
    print(md_table(out, "{:.1f}"))


if __name__ == "__main__":
    main()

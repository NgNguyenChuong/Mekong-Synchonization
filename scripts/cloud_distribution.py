"""Phan bo diem mat nhan NDWI (NaN, chu yeu do may) theo khoi (don vi kiem dinh) x mua + heatmap.

Chay: venv/Scripts/python.exe scripts/cloud_distribution.py --res-dir KE_HOACH/ket-qua
"""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from md_util import md_table  # noqa: E402

DATA = os.environ["DATA_ROOT"]  # bat buoc dat bien moi truong (xem .env)


def nan_table(ref, folds):
    """Ty le NaN (%) theo don vi x mua; khoi giu rieng (khong co trong cv_folds unit) gom thanh 'holdout'."""
    unit = folds.set_index("block_id").unit_id
    r = ref.copy()
    r["unit"] = r.block_id.map(unit)
    r.loc[r.is_holdout, "unit"] = "holdout_blocks"
    r["nan"] = r.ref_ndwi.isna()
    return r.groupby(["unit", "season"]).nan.mean().mul(100).unstack("season"), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default="KE_HOACH/ket-qua")
    ap.add_argument("--ref", default=f"{DATA}/labels/dot7/points_reference_ndwi.csv")
    ap.add_argument("--folds", default="data/eval/cv_folds.csv")
    a = ap.parse_args()
    res = Path(a.res_dir)
    (res / "hinh").mkdir(exist_ok=True)
    ref = pd.read_csv(a.ref)
    folds = pd.read_csv(a.folds)
    tab, r = nan_table(ref, folds)
    cv = tab.drop(index="holdout_blocks")
    assert cv.shape[0] == 19, cv.shape
    cv = cv.loc[sorted(cv.index, key=lambda s: tuple(int(x) for x in s.split("_")[1:]))]
    tot_season = r[~r.is_holdout].groupby("season").nan.mean().mul(100)
    tot_unit = r[~r.is_holdout].groupby("unit").nan.mean().mul(100).reindex(cv.index)
    overall = 100 * r[~r.is_holdout].nan.mean()
    over15 = (cv > 15).sum(axis=1)
    flagged = over15[over15 >= 3]

    show = cv.copy()
    show["all"] = tot_unit
    show.loc["all"] = list(tot_season.reindex(cv.columns)) + [overall]
    show.columns = [str(c) for c in show.columns]
    npts = r[~r.is_holdout].groupby("unit").point_id.nunique().reindex(cv.index)
    md = ["# Diem mat do may (NaN nhan NDWI) theo khoi x mua", "",
          "Nguon: points_reference_ndwi.csv (ref_ndwi NaN = pixel tham chieu khong hop le/mat do may+quy tac NaN), "
          "khoi = unit_id trong data/eval/cv_folds.csv (19 don vi kiem dinh; khoi nho da gop). "
          f"Chi diem CV ({int((~r.is_holdout).sum())} dong diem x mua; mua 2020 la mua giu rieng, van hien trong bang). "
          f"Khoi giu rieng (6) dua rieng o cuoi.", "", "Ty le NaN (%), hang = don vi, cot = mua:", "",
          md_table(show.round(1), "{:.1f}", index=True), "",
          f"Tong ty le NaN diem CV: {overall:.1f} % (toan bo gom ca khoi giu rieng: {100 * r.nan.mean():.1f} %).", "",
          "Khoi co > 15 % o >= 3 mua: " + (", ".join(f"{u} ({int(n)} mua)" for u, n in flagged.items()) or "khong co"), "",
          "Khoi giu rieng (6 khoi gom lai), % NaN theo mua: " + ", ".join(
              f"{c}: {v:.1f}" for c, v in tab.loc["holdout_blocks"].items()), ""]
    pp = r.groupby("point_id").nan.agg(["sum", "count"])
    always = pp["sum"] == pp["count"]
    md += [f"Cau truc hay ngau nhien: {int(always.sum())} / {len(pp)} diem NaN o ca {int(pp['count'].max())} mua (khong doi theo mua, "
           f"khong phai may); chiem {100 * pp.loc[always, 'sum'].sum() / r.nan.sum():.1f} % tong so o NaN. "
           f"Phan con lai ({100 * (1 - pp.loc[always, 'sum'].sum() / r.nan.sum()):.1f} %) doi theo mua (may, quy tac 3x3 >= 5/9 pixel hop le).", "",
           "Khoi co > 15 % o >= 3 mua, ty le diem NaN ca moi mua: " + ", ".join(
               f"{u}: {100 * always[r[r.unit == u].point_id.unique()].mean():.0f} %" for u in flagged.index), ""]
    (res / "may_phan_bo.md").write_text("\n".join(md), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    im = ax.imshow(cv.values, cmap="viridis", aspect="auto", vmin=0, vmax=max(30, float(cv.max().max())))
    ax.set_xticks(range(cv.shape[1]), [str(c) for c in cv.columns], rotation=45, ha="right")
    ax.set_yticks(range(cv.shape[0]), [f"{u.replace('blk_', '')} (n={int(npts[u])})" for u in cv.index])
    for i in range(cv.shape[0]):
        for j in range(cv.shape[1]):
            v = cv.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if v < 15 else "black")
    ax.set_xlabel("Season (year in which the dry season ends)")
    ax.set_ylabel("Spatial block (row_col, number of points)")
    fig.colorbar(im, ax=ax, label="Reference points with missing NDWI label (%)")
    fig.tight_layout()
    fig.savefig(res / "hinh" / "may_phan_bo.png", dpi=200)
    print("\n".join(md))


if __name__ == "__main__":
    main()

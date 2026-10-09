"""CHG-23 ban do phan du (MO TA): phan du OOF HistGB (err = du doan - dap an) theo diem va theo khoi 50 km.

Chay:  venv/Scripts/python.exe scripts/analyze_residual_map.py [--grid h3_res_7 --scheme 42]
"""
import argparse
import os
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#bdbcb7"
MIN_DIEM = 30
CMAP = LinearSegmentedColormap.from_list("phan_ky", ["#2a78d6", "#f0efec", "#e34948"])  # xanh <- 0 -> do


def load(grid, scheme):
    d = pd.read_csv(os.path.join(ROOT, "artifacts", "experiments", f"cv1__{grid}__hist_gb__s{scheme}", "cv",
                                 "oof_points.csv"), dtype={"point_id": str})
    if (d["pred_source"] != "oof").any() or d["err"].isna().any():
        raise SystemExit("LOI: oof khong hop le")
    p = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson"))[["point_id", "geometry"]]
    return d, p


def main(a):
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    d, pts = load(a.grid, a.scheme)
    blocks = gpd.read_file(os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    bnd = gpd.read_file(os.path.join(ROOT, "webapp", "backend", "data", "mekong_delta_boundary_v2.geojson"))
    # theo diem: TB moi mua + TB cac mua
    per_pt = d.groupby("point_id")["err"].agg(err_tb="mean", n_mua="size").reset_index()
    per_pt_s = d.pivot_table(index="point_id", columns="season", values="err")
    g = pts.merge(per_pt, on="point_id", how="inner")
    if len(g) != per_pt.shape[0]:
        raise SystemExit("LOI: diem khong co toa do")
    # theo khoi 50 km
    per_blk = d.groupby("block_id").agg(err_tb=("err", "mean"), n=("err", "size"),
                                         n_diem=("point_id", "nunique")).reset_index()
    gb = blocks.merge(per_blk, on="block_id", how="left")
    per_blk.to_csv(os.path.join(out, f"phan_du_theo_khoi_{a.grid}_s{a.scheme}.csv"), index=False)
    g.drop(columns="geometry").to_csv(os.path.join(out, f"phan_du_theo_diem_{a.grid}_s{a.scheme}.csv"), index=False)

    lim = float(np.nanpercentile(np.abs(per_pt_s.to_numpy()), 98))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    seasons = [s for s in a.seasons if s in per_pt_s.columns]
    panels = [("TB 12 mua", g.set_index("point_id")["err_tb"])] + [(str(s), per_pt_s[s]) for s in seasons]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 4.6), facecolor=SURFACE)
    gi = g.set_index("point_id")
    for ax, (title, ser) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        bnd.boundary.plot(ax=ax, color="#b9b8b3", linewidth=0.6)
        blocks.boundary.plot(ax=ax, color="#d8d7d2", linewidth=0.4)
        blocks[blocks["is_holdout"]].boundary.plot(ax=ax, color=INK2, linewidth=0.9, linestyle="--")
        v = ser.dropna()
        gg = gi.loc[v.index]
        ax.scatter(gg.geometry.x, gg.geometry.y, c=v.to_numpy(), cmap=CMAP, norm=norm, s=3, linewidths=0)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = axes[-1]  # khoi 50 km
    ax.set_facecolor(SURFACE)
    bnd.boundary.plot(ax=ax, color="#b9b8b3", linewidth=0.6)
    it = gb["n_diem"] < MIN_DIEM  # khoi < 30 diem: to xam, khong to theo phan du (An 2026-10-06)
    gb_ok = gb[~it.fillna(False)]
    gb_ok.plot(ax=ax, column="err_tb", cmap=CMAP, norm=norm, edgecolor=SURFACE, linewidth=1.0,
               missing_kwds={"color": "#ffffff", "edgecolor": "#d8d7d2", "hatch": "///"})
    gb[it.fillna(False)].plot(ax=ax, color=GRAY, edgecolor=SURFACE, linewidth=1.0)
    for r in gb.dropna(subset=["err_tb"]).itertuples():
        c = r.geometry.representative_point()
        nho = r.n_diem < MIN_DIEM
        ax.annotate(f"{r.err_tb:+.2f}" + (f"\n({int(r.n_diem)} diem)" if nho else ""), (c.x, c.y), ha="center",
                    va="center", fontsize=6, color=INK2 if nho else INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GRAY, label=f"< {MIN_DIEM} diem (it tin cay)"),
                       Patch(facecolor="#ffffff", edgecolor="#d8d7d2", hatch="///", label="khong co phan du CV")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=7, labelcolor=INK2, ncol=1)
    ax.set_title("TB theo khoi 50 km (CV)", color=INK, fontsize=10, loc="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("phan du = du doan - dap an (dS/m); do = du doan cao", color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.suptitle(f"Phan du OOF HistGB {a.grid} cach chia s{a.scheme} (CHG-23, mo ta); net dut = khoi giu rieng "
                 f"(khong co phan du CV)", color=INK, fontsize=10, x=0.01, ha="left")
    p = os.path.join(out, f"ban_do_phan_du_{a.grid}_s{a.scheme}.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(p)
    print(per_blk.sort_values("err_tb").round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="h3_res_7")
    ap.add_argument("--scheme", type=int, default=42)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2016, 2019, 2024])
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "chg23"))
    main(ap.parse_args())

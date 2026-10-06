"""CHG-23 B (MO TA): MAE theo khoang cach diem -> o huan luyen gan nhat cua fold, theo 3 vung, gop CV 50 + 100 km.

Khoang cach: tu toa do diem (EPSG:32648) toi tam phan dat (scope_cx/cy) gan nhat trong cac o role "train" cua
fold cham diem do (cv_membership.csv cua chinh lan chay). Vung (nhu bang sai so dot5): lat < 9,5 = Ca Mau-Bac Lieu;
con lai dist_coast_km cua diem <= 20 = ven bien khac; > 20 = noi dong.
Chay:  venv/Scripts/python.exe scripts/analyze_mae_distance.py [--grid h3_res_7]
"""
import argparse
import os
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from settings import data_path  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
VUNG = [("Noi dong (>20 km)", "#2a78d6"), ("Ven bien khac (<=20 km)", "#eb6834"), ("Ca Mau-Bac Lieu (lat<9,5)", "#1baf7a")]
BINS = [0, 5, 10, 20, 30, 40, 60, 200]
N_MIN = 300  # bin it diem: ve rong (khong to), van ghi trong bang


def point_info():
    p = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson"))[["point_id", "geometry"]]
    lat = p.geometry.y.to_numpy()
    q = p.to_crs(32648)
    with rasterio.open(data_path("raw", "river", "river_distance_90m.tif")) as src:
        b = list(src.descriptions).index("dist_coast_km")
        dc = np.array([v[b] for v in src.sample([(g.x, g.y) for g in q.geometry])])
    if not np.isfinite(dc).all():
        raise SystemExit("LOI: dist_coast NaN tai diem")
    vung = np.where(lat < 9.5, VUNG[2][0], np.where(dc <= 20, VUNG[1][0], VUNG[0][0]))
    return pd.DataFrame({"point_id": p["point_id"].astype(str), "x": q.geometry.x, "y": q.geometry.y, "vung": vung})


def run_distances(run_dir, cells_xy, pinfo):
    d = pd.read_csv(os.path.join(run_dir, "oof_points.csv"), dtype={"point_id": str},
                    usecols=["point_id", "season", "err", "cv_fold", "pred_source"])
    if (d["pred_source"] != "oof").any() or d["err"].isna().any():
        raise SystemExit(f"LOI: {run_dir} oof khong hop le")
    m = pd.read_csv(os.path.join(run_dir, "cv_membership.csv"), dtype={"cell_id": str})
    d = d.merge(pinfo, on="point_id", how="left", validate="many_to_one")
    if d[["x", "y"]].isna().any().any():
        raise SystemExit("LOI: diem khong co toa do")
    d["dist_km"] = np.nan
    for k, idx in d.groupby("cv_fold").groups.items():
        tr = m.loc[(m["fold"] == k) & (m["role"] == "train"), "cell_id"]
        xy = cells_xy.reindex(tr).dropna().to_numpy()
        if len(xy) == 0 or len(xy) < 0.99 * len(tr):
            raise SystemExit(f"LOI: fold {k} thieu tam o huan luyen")
        dist, _ = cKDTree(xy).query(d.loc[idx, ["x", "y"]].to_numpy())
        d.loc[idx, "dist_km"] = dist / 1000.0
    return d


def main(a):
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    pinfo = point_info()
    t = pd.read_csv(data_path("features", "unified", f"{a.grid}_unified.csv"), usecols=["cell_id", "scope_cx", "scope_cy"],
                    dtype={"cell_id": str}).drop_duplicates("cell_id").set_index("cell_id")
    cells_xy = t[["scope_cx", "scope_cy"]]
    parts = []
    for design, seeds in (("k50", (42, 43, 44)), ("k100", (42, 43))):
        prefix = "cv1" if design == "k50" else "k100"
        for s in seeds:
            d = run_distances(os.path.join(ROOT, "artifacts", "experiments", f"{prefix}__{a.grid}__hist_gb__s{s}", "cv"),
                              cells_xy, pinfo)
            parts.append(d.assign(design=design, seed=s))
    d = pd.concat(parts, ignore_index=True)
    d["bin"] = pd.cut(d["dist_km"], BINS, right=False)
    # kiem vung khop bang sai so dot5 (so (diem, mua) moi vung, 1 lan chay)
    one = d[(d.design == "k50") & (d.seed == 42)]
    print("so (diem, mua) theo vung (k50 s42):", one["vung"].value_counts().to_dict())
    tab = (d.assign(ae=d["err"].abs()).groupby(["vung", "bin"], observed=True)
           .agg(MAE=("ae", "mean"), n=("ae", "size"), n_k50=("design", lambda x: int((x == "k50").sum())),
                n_k100=("design", lambda x: int((x == "k100").sum())), dist_tb=("dist_km", "mean")).reset_index())
    tab["bin"] = tab["bin"].astype(str)
    tab.to_csv(os.path.join(out, f"mae_theo_khoang_cach_{a.grid}.csv"), index=False)
    by_design = (d.assign(ae=d["err"].abs()).groupby(["design", "vung", "bin"], observed=True)
                 .agg(MAE=("ae", "mean"), n=("ae", "size")).reset_index())
    by_design["bin"] = by_design["bin"].astype(str)
    by_design.to_csv(os.path.join(out, f"mae_theo_khoang_cach_{a.grid}_theo_thiet_ke.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for name, col in VUNG:
        x = tab[tab["vung"] == name]
        ok = x["n"] >= N_MIN
        ax.plot(x["dist_tb"], x["MAE"], color=col, linewidth=2, label=name, zorder=2)
        ax.scatter(x.loc[ok, "dist_tb"], x.loc[ok, "MAE"], s=36, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter(x.loc[~ok, "dist_tb"], x.loc[~ok, "MAE"], s=36, facecolor=SURFACE, edgecolor=col, linewidth=2, zorder=3)
        last = x.iloc[-1]
        ax.annotate(name.split(" (")[0], (last["dist_tb"], last["MAE"]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.set_xlabel("khoang cach diem -> o huan luyen gan nhat cua fold (km, TB trong bin)", color=INK2, fontsize=9)
    ax.set_ylabel("MAE (dS/m)", color=INK2, fontsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title(f"MAE theo khoang cach toi du lieu huan luyen - {a.grid}, HistGB, gop CV 50 km (3 cach chia) + 100 km "
                 f"(P1/P2)\nCHG-23 B, mo ta; diem rong = bin < {N_MIN} (diem, mua)", color=INK, fontsize=9, loc="left")
    p = os.path.join(out, f"mae_theo_khoang_cach_{a.grid}.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(p)
    with pd.option_context("display.width", 200):
        print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="h3_res_7")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "chg23"))
    main(ap.parse_args())

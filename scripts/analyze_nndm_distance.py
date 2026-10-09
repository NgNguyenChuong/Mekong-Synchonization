"""CHG-23 A (MO TA, kieu NNDM): so phan bo khoang cach diem -> o huan luyen gan nhat giua CV 50 km, CV 100 km va vung giu
rieng (dai dien kich ban (iii) "ap sang vung moi"). Thiet ke ghi truoc trong NHAT_KY 2026-10-06 05:51.

Chay:  venv/Scripts/python.exe scripts/analyze_nndm_distance.py [--grids h3_res_7 square_utm_17087m]
"""
import argparse
import os
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from settings import data_path  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {"CV 50 km": ("#2a78d6", "-"), "CV 100 km": ("#eb6834", "-"), "Vung giu rieng (kich ban iii)": ("#1baf7a", "-")}
EXP = os.path.join(ROOT, "artifacts", "experiments")


def cells_xy(grid):
    t = pd.read_csv(data_path("features", "unified", f"{grid}_unified.csv"), usecols=["cell_id", "scope_cx", "scope_cy"],
                    dtype={"cell_id": str}).drop_duplicates("cell_id").set_index("cell_id")
    return t[["scope_cx", "scope_cy"]]


def tree_of(ids, cxy):
    xy = cxy.reindex(ids).dropna().to_numpy()
    if len(xy) == 0 or len(xy) < 0.99 * len(ids):
        raise SystemExit("LOI: thieu tam o huan luyen")
    return cKDTree(xy)


def cv_distances(prefix, grid, scheme, cxy, pxy):
    run = os.path.join(EXP, f"{prefix}__{grid}__hist_gb__s{scheme}", "cv")
    d = pd.read_csv(os.path.join(run, "oof_points.csv"), dtype={"point_id": str}, usecols=["point_id", "cv_fold"])
    d = d.drop_duplicates("point_id")
    if d["point_id"].duplicated().any():
        raise SystemExit("LOI: diem co hon 1 fold")
    m = pd.read_csv(os.path.join(run, "cv_membership.csv"), dtype={"cell_id": str})
    d = d.merge(pxy, on="point_id", how="left", validate="one_to_one")
    out = []
    for k, x in d.groupby("cv_fold"):
        tr = m.loc[(m["fold"] == k) & (m["role"] == "train"), "cell_id"]
        dist, _ = tree_of(tr, cxy).query(x[["x", "y"]].to_numpy())
        out.append(dist / 1000.0)
    return np.concatenate(out), m


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    pts = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    q = pts.to_crs(32648)
    pxy = pd.DataFrame({"point_id": pts["point_id"].astype(str), "x": q.geometry.x, "y": q.geometry.y,
                        "is_holdout": pts["is_holdout"].astype(bool)})
    rows, summ = [], []
    for grid in a.grids:
        cxy = cells_xy(grid)
        dists = {"CV 50 km": [], "CV 100 km": []}
        for s in (42, 43, 44):
            dd, m50 = cv_distances("cv1", grid, s, cxy, pxy)
            dists["CV 50 km"].append(dd)
        for s in (42, 43):
            dists["CV 100 km"].append(cv_distances("k100", grid, s, cxy, pxy)[0])
        # final = moi o CV (moi o la 'val' dung 1 lan trong cv_membership)
        val = m50.loc[m50["role"] == "val", "cell_id"]
        if val.duplicated().any():
            raise SystemExit("LOI: o la val nhieu lan")
        cfg = pd.read_json(os.path.join(EXP, f"cv1__{grid}__hist_gb__s42", "final", "config.json"), typ="series")
        if int(cfg["n_train_cells"]) != len(val):
            raise SystemExit(f"LOI: so o final {cfg['n_train_cells']} != {len(val)}")
        ho = pxy[pxy["is_holdout"]]
        dh, _ = tree_of(val, cxy).query(ho[["x", "y"]].to_numpy())
        dists["Vung giu rieng (kich ban iii)"] = [dh / 1000.0]
        dists = {k: np.concatenate(v) for k, v in dists.items()}
        ref = dists["Vung giu rieng (kich ban iii)"]
        for k, v in dists.items():
            r = {"grid": grid, "thiet_ke": k, "n": len(v), "trung_vi_km": np.median(v), "p90_km": np.percentile(v, 90),
                 "ty_le_duoi_5km": float((v < 5).mean()), "ty_le_tren_20km": float((v > 20).mean())}
            if k != "Vung giu rieng (kich ban iii)":
                r["wasserstein_vs_giu_rieng_km"] = wasserstein_distance(v, ref)
                r["ks_vs_giu_rieng"] = ks_2samp(v, ref).statistic
            summ.append(r)
        # ECDF
        fig, ax = plt.subplots(figsize=(6.6, 4.0), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        for k, v in dists.items():
            col, ls = STYLE[k]
            xs = np.sort(v)
            ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, linestyle=ls, linewidth=2, label=k)
        ax.set_xlabel("khoang cach diem -> o huan luyen gan nhat (km)", color=INK2, fontsize=9)
        ax.set_ylabel("ty le diem tich luy", color=INK2, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
        ax.set_title(f"Phan bo khoang cach toi du lieu huan luyen - {grid} (CHG-23 A, mo ta)", color=INK, fontsize=9,
                     loc="left")
        p = os.path.join(a.out_dir, f"nndm_khoang_cach_{grid}.png")
        fig.savefig(p, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(p)
    t = pd.DataFrame(summ)
    t.to_csv(os.path.join(a.out_dir, "nndm_khoang_cach_tom_tat.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=["h3_res_7", "square_utm_17087m"])
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "chg23"))
    main(ap.parse_args())

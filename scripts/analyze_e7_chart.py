#!/usr/bin/env python
"""E7 (MO TA): bang tam/canh o + nguon/canh o 13 luoi x 6 bien, bang + hinh bieu do chinh (Moran trung vi x
max|Delta_hat|/Delta_min cua cap F1 HistGB theo muc; truc phu do phan giai nguon / canh o).
Khong tinh lai Moran; chi doc dot7_e7_moran_bien.csv va ket qua HistGB da mo.

Chay:  venv/Scripts/python.exe scripts/analyze_e7_chart.py [--out-dir KE_HOACH/ket-qua]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze_cv_family import FROZEN_MANIFESTS, GRIDS, RESULTS, area_table, out_name  # noqa: E402
from analyze_e7_moran import SUMMARY_FILE, VARIABLES  # noqa: E402
from training.dot7_rules import file_sha256, guard_frozen, provenance_path, write_csv_atomic, write_provenance  # noqa: E402

LEVELS = (5, 6, 7)
AREA_FILE = "dot4_doi_chieu_dien_tich.csv"
TAM_FILE = "dot7_e7_tam_canh_o.csv"
MAIN_FILE = "dot7_e7_bieu_do_chinh.csv"
FIG = "dot7_e7_bieu_do_chinh"
FIGS = {"": dict(x="moran_median"), "_bo_ndwi": dict(x="moran_median", drop=("ndwi",)),
        "_nguon_canh_o": dict(x="nguon_tren_canh_median", logx=True)}
XLABEL = {"moran_median": "Moran's I trung vị theo mùa (dải 10 km)",
          "nguon_tren_canh_median": "độ phân giải nguồn / cạnh ô (trung vị các lưới cùng mức, log)"}
NAMES = {"salinity": "Độ mặn", "ndwi": "NDWI", "rain_chirps": "Mưa (CHIRPS)", "dsr_mcd18": "Bức xạ (MCD18)",
         "t2m_era5": "Nhiệt độ (ERA5-Land)", "rh_era5": "Độ ẩm (ERA5-Land)"}
COLORS = {"salinity": "#2a78d6", "ndwi": "#2a78d6", "rain_chirps": "#1baf7a", "dsr_mcd18": "#eda100",
          "t2m_era5": "#e34948", "rh_era5": "#4a3aa7"}  # do man - NDWI cung mau, khac ky hieu
MARKERS = {"ndwi": "D"}
SIZES = {5: 160, 6: 95, 7: 45}
LEVEL_DY = {5: -3, 6: 1, 7: -9}  # so muc lech doc: muc 6 va 7 hay trung y
SURFACE, INK, INK2, GRID_INK = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PEARSON_NOTE = "NDWI (cùng màu độ mặn; Pearson với độ mặn, trung vị −0,40)"


def loi(msg):
    raise SystemExit(f"LOI: {msg}")


def kd_path(results_dir, t):
    return os.path.join(results_dir, out_name(t, "cv1"))


def grid_table(path) -> pd.DataFrame:
    t = area_table(path)
    if sorted(t["grid"]) != sorted(GRIDS):
        loi(f"{path}: luoi {sorted(t['grid'])} khac 13 luoi {sorted(GRIDS)}")
    t = t.rename(columns={"tier_h3_res": "muc"})[["grid", "muc", "mean_area_km2"]]
    t["canh_o_km"] = np.sqrt(t["mean_area_km2"])
    return t


def tam_canh_o(grids: pd.DataFrame, moran: pd.DataFrame) -> pd.DataFrame:
    out = moran[["variable", "range_median_km", "do_phan_giai_nguon_km"]].merge(grids, how="cross")
    out["ti_le_tam_tren_canh"] = out["range_median_km"] / out["canh_o_km"]  # inf / canh = inf
    out["nguon_tren_canh"] = out["do_phan_giai_nguon_km"] / out["canh_o_km"]
    return out[["variable", "grid", "muc", "mean_area_km2", "canh_o_km", "range_median_km", "ti_le_tam_tren_canh",
                "do_phan_giai_nguon_km", "nguon_tren_canh"]]


def _bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() == "true")  # NaN (muc ngoai kiem dinh) -> False


def framework_effect(kd: pd.DataFrame, t: str, grids: pd.DataFrame) -> pd.DataFrame:
    """(bien, muc): max |delta_hat|/delta_min_thr tren cap F1_chinh; kiem_dinh = muc_kiem_dinh & cong (do man: True)."""
    lv = grids.set_index("grid")["muc"]
    f = kd[kd["family"] == "F1_chinh"].copy()
    f["muc"] = f["muc" if "muc" in f.columns else "delta_ref_level"].astype(int)
    if (f["grid_a"].map(lv) != f["muc"]).any() or (f["grid_b"].map(lv) != f["muc"]).any():
        loi(f"{t}: cap F1 co luoi lech muc hoac khong co trong bang dien tich")
    if t == "salinity":
        f["kd"] = True
    else:
        f["kd"] = _bool(f["muc_kiem_dinh"]) & _bool(f["cong_hoc_duoc"])
        if (f["kd"] != _bool(f["kiem_dinh"])).any():
            loi(f"{t}: kiem_dinh trong file khac muc_kiem_dinh & cong_hoc_duoc (sai file / sai pham vi Holm?)")
    f["ratio"] = f["delta_hat"].abs() / f["delta_min_thr"]
    if not np.isfinite(f["ratio"]).all():
        loi(f"{t}: |delta_hat|/delta_min_thr khong huu han")
    rows = []
    for m in LEVELS:
        x = f[f["muc"] == m]
        if x.empty or x["kd"].nunique() != 1:
            loi(f"{t} muc {m}: khong co cap F1 hoac kiem_dinh khong dong nhat")
        rows.append({"variable": t, "muc": m, "anh_huong_khung": float(x["ratio"].max()),
                     "kiem_dinh": bool(x["kd"].iloc[0]), "n_cap": len(x)})
    return pd.DataFrame(rows)


def main_table(moran: pd.DataFrame, eff: pd.DataFrame, tam: pd.DataFrame) -> pd.DataFrame:
    tm = (tam.groupby(["variable", "muc"], as_index=False)[["ti_le_tam_tren_canh", "nguon_tren_canh"]].median()
          .rename(columns={"ti_le_tam_tren_canh": "tam_tren_canh_median",
                           "nguon_tren_canh": "nguon_tren_canh_median"}))
    out = (eff.merge(moran[["variable", "moran_median"]], on="variable", how="left", validate="many_to_one")
           .merge(tm, on=["variable", "muc"], how="left", validate="one_to_one"))
    if out[["moran_median", "tam_tren_canh_median", "nguon_tren_canh_median"]].isna().any().any():
        loi("thieu Moran / tam / nguon cho mot (bien, muc)")
    return out[["variable", "muc", "moran_median", "anh_huong_khung", "kiem_dinh", "tam_tren_canh_median",
                "nguon_tren_canh_median", "n_cap"]]


def plot(tab: pd.DataFrame, path_noext, x="moran_median", drop=(), logx=False) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    d = tab[~tab["variable"].isin(drop)].copy()
    if not np.isfinite(d[x].to_numpy(float)).all():
        loi(f"{x} khong huu han - khong ve")
    fig, ax = plt.subplots(figsize=(8, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for t in [v for v in VARIABLES if v in set(d["variable"])]:
        g = d[d["variable"] == t].sort_values("muc")
        c, mk = COLORS[t], MARKERS.get(t, "o")
        ax.plot(g[x], g["anh_huong_khung"], color=c, lw=0.8, alpha=0.45, zorder=2)
        for _, r in g.iterrows():
            ax.scatter(r[x], r["anh_huong_khung"], s=SIZES[int(r["muc"])], marker=mk, linewidths=1.6, zorder=3,
                       facecolors=c if r["kiem_dinh"] else SURFACE, edgecolors=c)
            ax.annotate(str(int(r["muc"])), (r[x], r["anh_huong_khung"]), xytext=(7, LEVEL_DY[int(r["muc"])]),
                        textcoords="offset points", fontsize=7, color=INK2)
    ax.axhline(1.0, color=INK2, ls="--", lw=1, zorder=1)
    ax.annotate("ngưỡng Δ_min", (0.01, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", fontsize=7, color=INK2)
    if x == "nguon_tren_canh_median":
        ax.axvline(1.0, color=INK2, ls=":", lw=1, zorder=1)
        ax.annotate("x = 1: pixel nguồn = cạnh ô\nx > 1: pixel nguồn lớn hơn ô", (1.0, 0.98),
                    xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=7,
                    color=INK2, va="top")
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(XLABEL[x], color=INK2, fontsize=9)
    ax.set_ylabel(r"mức ảnh hưởng khung = max $|\hat{\Delta}|/\Delta_{\min}$ (cặp F1, HistGB)", color=INK2, fontsize=9)
    ax.set_title("E7 (mô tả): tự tương quan của biến và mức ảnh hưởng của khung lưới", color=INK, fontsize=10,
                 loc="left")
    ax.grid(color=GRID_INK, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(which="both", colors=INK2, labelsize=8)
    leg = [Line2D([], [], ls="", marker=MARKERS.get(t, "o"), ms=7, mfc=COLORS[t], mec=COLORS[t],
                  label=PEARSON_NOTE if t == "ndwi" else NAMES[t])
           for t in VARIABLES if t in set(d["variable"])]
    leg += [Line2D([], [], ls="", marker="o", ms=7, mfc=INK2, mec=INK2, label="đặc: kiểm định"),
            Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=INK2, label="rỗng: mô tả / trượt cổng")]
    leg += [Line2D([], [], ls="", marker="o", ms=np.sqrt(SIZES[m]) * 0.8, mfc=SURFACE, mec=INK2,
                   label=f"mức {m} (số cạnh điểm)") for m in LEVELS]
    ax.legend(handles=leg, frameon=False, fontsize=7, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3)
    paths = []
    for ext in ("png", "svg"):
        p = f"{path_noext}.{ext}"
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def outputs(out_dir) -> list:
    figs = [os.path.join(out_dir, "hinh", f"{FIG}{suf}.{ext}") for suf in FIGS for ext in ("png", "svg")]
    return [os.path.join(out_dir, TAM_FILE), os.path.join(out_dir, MAIN_FILE)] + figs


def main(a):
    outs = outputs(a.out_dir)
    guard_frozen(outs + [provenance_path(p) for p in outs], a.frozen_manifest)
    moran_path = os.path.join(a.out_dir, SUMMARY_FILE)
    area_path = os.path.join(a.results_dir, AREA_FILE)
    moran = pd.read_csv(moran_path)
    if sorted(moran["variable"]) != sorted(VARIABLES):
        loi(f"{moran_path}: bien {sorted(moran['variable'])} khac {sorted(VARIABLES)}")
    r = moran["do_phan_giai_nguon_km"].to_numpy(float)
    if not (np.isfinite(r) & (r > 0)).all():
        loi(f"{moran_path}: do_phan_giai_nguon_km phai huu han > 0")
    grids = grid_table(area_path)
    tam = tam_canh_o(grids, moran)
    eff = pd.concat([framework_effect(pd.read_csv(kd_path(a.results_dir, t)), t, grids) for t in VARIABLES],
                    ignore_index=True)
    tab = main_table(moran, eff, tam)
    os.makedirs(os.path.join(a.out_dir, "hinh"), exist_ok=True)
    src = {os.path.basename(p): file_sha256(p)
           for p in [moran_path, area_path] + [kd_path(a.results_dir, t) for t in VARIABLES]}
    info = dict(quy_tac_nhat_ky="2026-10-09 10:56:38 E7; 2026-10-09 12:01:14 truc phu nguon/canh o", vai_tro="mo_ta",
                sha_dau_vao=src)
    write_csv_atomic(tam, outs[0])
    write_csv_atomic(tab, outs[1])
    figs = []
    for suf, kw in FIGS.items():
        figs += plot(tab, os.path.join(a.out_dir, "hinh", f"{FIG}{suf}"), **kw)
    for p in outs[:2] + figs:
        write_provenance(p, **info)
    with pd.option_context("display.width", 200):
        print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=RESULTS)
    ap.add_argument("--results-dir", default=RESULTS, help="noi co dot4_doi_chieu_dien_tich.csv + file kiem dinh HistGB")
    ap.add_argument("--frozen-manifest", nargs="+", default=FROZEN_MANIFESTS)
    main(ap.parse_args())

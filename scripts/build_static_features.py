#!/usr/bin/env python
"""Dac trung TINH theo o cho 13 luoi, tinh tren PIXEL DAT HOP LE (scope == 1) - An chot 2026-10-03.

Dau ra A:/Dataset_NCKH/features/static/<luoi>_static.csv (+ .provenance.json):
  cell_id, dem_mean, dist_main_river_km, dist_any_water_km, dist_coast_km  (chi pixel scope; o 0 scope -> NaN)
  scope_frac                                                               (cot CHAT LUONG, khong phai dac trung)
  landcover_class_<Lop>                                                    (WorldCover 10 m, TOAN O)
Cach lay mau: xem src/static_features.py (DEM 30 m dung pixel luoi scope; khoang cach 90 m lay pixel chua
tam pixel scope 30 m; trung binh co trong so dien tich bang exactextract).
Trung gian: <out-dir>/static_scope_30m.tif (4 band, NaN ngoai scope) - dung lai neu sha256 dau vao khop.
Bao cao (so voi ban ca o: raw/river/features/*_river.csv va DEM ca o; diem quen) -> KE_HOACH/ket-qua/.
Bao cao doi phien ban (tuy chon): --prev-dir (CSV ban truoc, vd static/deprecated_v2: dem_mean, ma 0 WorldCover,
ty le Water/Mangroves o ven ria) va --control-stack (raster doi chung KHONG ap CHG-09 tren cung scope -> tach
rieng tac dong cua quy tac DEM).
Da co CSV -> bo qua luoi do (tru --force); bao cao van tinh lai tu CSV.

CHG-09 (DEM): DEM thieu khi DEM == 0 VA co WBM > 0 (--dem-flags, band WBM). CHG-10: scope_mask_v3.
WorldCover: LandCover_DBSCL_2021_v2.tif (dem 25 km, bien xa da dien 80).

Chay:  venv/Scripts/python.exe scripts/build_static_features.py [--grids h3_res_7 ...] [--force]
       [--prev-dir A:/Dataset_NCKH/features/static/deprecated_v2 --control-stack <scratch>/static_scope_30m_doi_chung.tif]
"""
import argparse
import glob
import json
import os
import shutil
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import CANONICAL_BOUNDARY, file_sha256, provenance_path, write_provenance  # noqa: E402
from processing import area_weighted_means  # noqa: E402
from static_features import (DEM_RULE, LANDCOVER_PREFIX, RIVER_BANDS, SCOPE_FEATURES,  # noqa: E402
                             SCOPE_FRAC_COL, build_scope_stack, cell_static_features)

SAMPLING = {
    "scope": "mat na pham vi (--scope; mac dinh scope_mask_v3.tif: v2 ∩ WorldCover {10,20,30,40} tai tam pixel 30 m "
             "∩ dau chan Zenodo, CHG-10; ban truoc scope_mask_v2.tif: WorldCover khong {0,80,95}) band 'scope' == 1",
    "dem": "DEM GLO30 30 m cung luoi pixel scope (lech so nguyen pixel) -> lay dung pixel; chi pixel scope; "
           "co WBM (DEM_DBSCL_flags_v2.tif, lech nguyen pixel) lay dung pixel, khong noi suy; " + DEM_RULE,
    "river": "raster khoang cach 90 m: moi pixel scope 30 m nhan gia tri pixel 90 m chua TAM no (nearest) = trong "
             "so pixel 90 m theo so tam pixel scope 30 m ben trong; chi pixel scope",
    "aggregation": "trung binh co trong so dien tich phu (exactextract mean) tren raster 30 m NaN ngoai scope; "
                   "o khong co pixel scope -> NaN",
    "scope_frac": "exactextract sum(scope) * 900 m2 / dien tich o EPSG:32648 - cot chat luong",
    "landcover": "WorldCover 2021 10 m (ban v2: dem 25 km quanh ranh gioi v2, bien xa unmask 80), ty le dien "
                 "tich tren TOAN O (exactextract frac); ma 0 khong sinh cot nhung van o mau so",
}
CHANGES = ["CHG-09 (DEM == 0 & WBM > 0 -> NaN)", "CHG-10 (scope_mask_v3)", "WorldCover v2 dem 25 km (An chot c)"]
KNOWN_POINTS = {  # skill data-audit; dist_coast ky vong: Can Tho ~76, Chau Doc ~66, My Tho ~46, Rach Gia ~0,3
    "Can Tho": (105.78, 10.03), "Chau Doc": (105.12, 10.70), "My Tho": (106.36, 10.36),
    "Rach Gia": (105.08, 10.01), "Ca Mau": (105.15, 9.18),
}


def log(msg):
    print(msg, flush=True)


def read_grid(path):
    g = gpd.read_file(path)
    g["cell_id"] = g["cell_id"].astype(str)
    return g


def ensure_stack(a, shas):
    prov = provenance_path(a.stack)
    want = {k: shas[k] for k in ("scope", "dem", "dem_flags", "river")}
    want["dem_rule"] = DEM_RULE
    if os.path.exists(a.stack) and os.path.exists(prov) and not a.force_stack:
        with open(prov, encoding="utf-8") as f:
            have = json.load(f).get("input_sha256", {})
        if have == want:
            log(f"Dung lai {a.stack} (sha256 dau vao khop).")
            return
        log(f"{a.stack}: sha256 dau vao khac -> dung lai.")
    log(f"Dung raster trung gian 30 m tren pixel scope -> {a.stack}")
    tmp = a.stack + ".tmp.tif"
    stats = build_scope_stack(a.scope, a.dem, a.river, tmp, a.dem_flags, log=log)
    os.replace(tmp, a.stack)
    log(f"  thong ke pixel scope: {stats}")
    write_provenance(a.stack, a.boundary, function="static_features.build_scope_stack", bands=list(SCOPE_FEATURES),
                     inputs={"scope": a.scope, "dem": a.dem, "dem_flags": a.dem_flags, "river": a.river},
                     input_sha256=want, dem_rule=DEM_RULE, changes=CHANGES, sampling=SAMPLING, stats=stats)


def ensure_control_stack(a, shas):
    """Raster DOI CHUNG (cung scope, KHONG ap CHG-09) chi de do tac dong quy tac DEM; khong dung cho mo hinh."""
    prov = provenance_path(a.control_stack)
    want = {k: shas[k] for k in ("scope", "dem", "river")}
    want["dem_rule"] = "KHONG ap (ban doi chung)"
    if os.path.exists(a.control_stack) and os.path.exists(prov):
        with open(prov, encoding="utf-8") as f:
            if json.load(f).get("input_sha256", {}) == want:
                log(f"Dung lai doi chung {a.control_stack}.")
                return
    log(f"Dung raster DOI CHUNG (khong ap CHG-09) -> {a.control_stack}")
    os.makedirs(os.path.dirname(os.path.abspath(a.control_stack)), exist_ok=True)
    tmp = a.control_stack + ".tmp.tif"
    stats = build_scope_stack(a.scope, a.dem, a.river, tmp, None, log=log)
    os.replace(tmp, a.control_stack)
    write_provenance(a.control_stack, a.boundary, function="static_features.build_scope_stack (doi chung)",
                     inputs={"scope": a.scope, "dem": a.dem, "river": a.river}, input_sha256=want, stats=stats)


def build_grid(a, path, shas):
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(a.out_dir, f"{name}_static.csv")
    if os.path.exists(out) and not a.force:
        pv = provenance_path(out)
        have = json.load(open(pv, encoding="utf-8")).get("input_sha256") if os.path.exists(pv) else None
        if have is not None and have != shas:
            raise SystemExit(f"[LOI] {out} dung dau vao khac (sha256 scope/dem/river/worldcover lech, vd doi mat na "
                             f"pham vi v2 -> v3) - chay lai voi --force de ghi lai, khong de CSV cu lan voi stack moi.")
        log(f"[bo qua] {name}: da co {out}")
        return out
    grid = read_grid(path)
    cells = (grid["cell_id"].tolist(), None, grid.geometry.tolist())
    log(f"[{name}] {len(grid)} o ...")
    df = cell_static_features(a.stack, a.scope, a.worldcover, cells)
    if len(df) != len(grid) or set(df["cell_id"]) != set(grid["cell_id"]):
        raise RuntimeError(f"{name}: so o dau ra khac luoi.")
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    write_provenance(out, a.boundary, grid=os.path.basename(path), grid_sha256=file_sha256(path),
                     stack=os.path.basename(a.stack), sampling=SAMPLING,
                     inputs={"scope": a.scope, "dem": a.dem, "dem_flags": a.dem_flags, "river": a.river,
                             "worldcover": a.worldcover},
                     input_sha256=shas, dem_rule=DEM_RULE, changes=CHANGES, columns=list(df.columns))
    log(f"[{name}] ghi {out}")
    return out


def compare_grid(a, path, out_csv):
    """Bang so sanh voi ban ca o (song cu + DEM ca o) cho mot luoi."""
    name = os.path.splitext(os.path.basename(path))[0]
    grid = read_grid(path)
    new = pd.read_csv(out_csv, dtype={"cell_id": str})
    old = pd.read_csv(os.path.join(a.old_river_dir, f"{name}_river.csv"), dtype={"cell_id": str})
    dem_full = area_weighted_means(a.dem, (grid["cell_id"].tolist(), None, grid.geometry.tolist()))
    dem_full.columns = ["cell_id", "dem_mean"]
    full = old.merge(dem_full, on="cell_id", how="outer", validate="1:1")
    m = new.merge(full, on="cell_id", suffixes=("", "_full"), validate="1:1")
    cen = grid.to_crs(32648).geometry.centroid.to_crs(4326)
    m = m.merge(pd.DataFrame({"cell_id": grid["cell_id"], "lon": cen.x.round(4), "lat": cen.y.round(4),
                              "overlap_frac": grid.get("overlap_frac", np.nan)}), on="cell_id")
    for c in SCOPE_FEATURES:
        m[f"d_{c}"] = m[c] - m[f"{c}_full"]
    m["nhom"] = pd.cut(m[SCOPE_FRAC_COL], [-1, 0, 0.1, 0.5, 0.9, 2],
                       labels=["0", "(0,0.1)", "[0.1,0.5)", "[0.5,0.9)", ">=0.9"], right=False)
    m.loc[m[SCOPE_FRAC_COL] <= 0, "nhom"] = "0"
    return name, grid, new, m


def summarize(name, new, m):
    lc = [c for c in new.columns if c.startswith("landcover_class_")]
    common = m.dropna(subset=["dem_mean", "dem_mean_full"])
    row = {"grid": name, "cells": len(new), "nan_scope_feat": int(new["dem_mean"].isna().sum()),
           **{f"nan_{c}": int(new[c].isna().sum()) for c in SCOPE_FEATURES},
           "scope_frac_0": int((new[SCOPE_FRAC_COL] <= 0).sum()),
           "scope_frac_lt_0.1": int(((new[SCOPE_FRAC_COL] > 0) & (new[SCOPE_FRAC_COL] < 0.1)).sum()),
           "nan_landcover": int(new[lc].isna().any(axis=1).sum()),
           "lc_sum_min": round(float(new[lc].sum(axis=1).min()), 4),
           "dem_median_scope_common": round(float(common["dem_mean"].median()), 3),
           "dem_median_full_common": round(float(common["dem_mean_full"].median()), 3),
           "dem_min": round(float(new["dem_mean"].min()), 3), "dem_max": round(float(new["dem_mean"].max()), 3)}
    for c in SCOPE_FEATURES:
        d = m[f"d_{c}"].abs()
        row[f"absd_{c}_median"] = round(float(d.median()), 4)
        row[f"absd_{c}_max"] = round(float(d.max()), 3)
        row[f"n_absd_{c}_gt_1"] = int((d > 1).sum())
    for c in RIVER_BANDS:
        row[f"{c}_median_scope_common"] = round(float(m.dropna(subset=[c, f"{c}_full"])[c].median()), 3)
        row[f"{c}_median_full_common"] = round(float(m.dropna(subset=[c, f"{c}_full"])[f"{c}_full"].median()), 3)
    return row


def by_group(name, m):
    rows = []
    for g, sub in m.groupby("nhom", observed=True):
        r = {"grid": name, "nhom_scope_frac": g, "cells": len(sub)}
        for c in SCOPE_FEATURES:
            d = sub[f"d_{c}"]
            r[f"{c}_median_d"] = round(float(d.median()), 4) if d.notna().any() else np.nan
            r[f"{c}_median_absd"] = round(float(d.abs().median()), 4) if d.notna().any() else np.nan
        rows.append(r)
    return rows


def top_cells(name, m, k=5):
    cols = ["cell_id", "lon", "lat", SCOPE_FRAC_COL, "overlap_frac", "landcover_class_Water"]
    out = []
    for c in ("dist_coast_km", "dem_mean"):
        t = m.reindex(m[f"d_{c}"].abs().sort_values(ascending=False).index).head(k)
        for _, r in t.iterrows():
            out.append({"grid": name, "bien": c, **{x: r[x] for x in cols}, "scope": r[c], "ca_o": r[f"{c}_full"],
                        "chenh": r[f"d_{c}"]})
    return out


def known_points(name, grid, new):
    pts = gpd.GeoDataFrame({"diem": list(KNOWN_POINTS)},
                           geometry=gpd.points_from_xy(*zip(*KNOWN_POINTS.values())), crs=4326)
    j = gpd.sjoin(pts, grid[["cell_id", "geometry"]], how="left", predicate="within")
    j = j.merge(new, on="cell_id", how="left")
    keep = ["diem", "cell_id", *SCOPE_FEATURES, SCOPE_FRAC_COL, "landcover_class_Water", "landcover_class_Cropland"]
    return [{"grid": name, **{c: r[c] for c in keep}} for _, r in j.iterrows()]


def wc_frame_cover(grid, wc_path):
    """Ty le dien tich o nam trong KHUNG raster WorldCover (phan ngoai khung khong vao mau so cua exactextract)."""
    import rasterio
    from shapely.geometry import box

    with rasterio.open(wc_path) as ds:
        frame = box(*ds.bounds)
    g = grid.to_crs(32648).geometry
    return pd.Series((g.intersection(frame).area / g.area).to_numpy(), index=grid["cell_id"].to_numpy())


def lc_code0(df):
    """Ty le ma 0 ("ngoai vung") = 1 - tong cac lop co ten (frac exactextract cong lai = 1 tren phan trong khung;
    ma khong ten da bao loi khi trich)."""
    lc = [c for c in df.columns if c.startswith(LANDCOVER_PREFIX)]
    return (1.0 - df[lc].sum(axis=1)).clip(lower=0)


def compare_versions(a, name, grid, new, ctrl_dem):
    """So voi doi chung CHG-09 (cung scope) va ban truoc (prev-dir). Tra ve (dong tong hop, dong o ven ria WC)."""
    nw = new.set_index("cell_id")
    row = {"grid": name, "cells": len(nw), "new_nan_dem": int(nw["dem_mean"].isna().sum()),
           "new_scope_frac_0": int((nw[SCOPE_FRAC_COL] <= 0).sum())}
    c0_new = lc_code0(nw)
    cov_new = wc_frame_cover(grid, a.worldcover).reindex(nw.index)
    row["new_code0_gt0"] = int((c0_new > 1e-9).sum())
    row["new_code0_gt0.5"] = int((c0_new > 0.5).sum())
    row["new_code0_max"] = round(float(c0_new.max()), 6)
    row["new_frame_cover_lt1"] = int((cov_new < 1 - 1e-9).sum())
    if ctrl_dem is not None:
        c = ctrl_dem.reindex(nw.index)
        d = (nw["dem_mean"] - c).abs()
        row["chg09_dem_absd_gt0.1"] = int((d > 0.1).sum())
        row["chg09_dem_absd_gt0.5"] = int((d > 0.5).sum())
        row["chg09_dem_absd_max"] = round(float(d.max()), 3)
        row["chg09_dem_d_median_changed"] = round(float((nw["dem_mean"] - c)[d > 1e-9].median()), 4) \
            if (d > 1e-9).any() else 0.0
        row["chg09_nan_moi"] = int((nw["dem_mean"].isna() & c.notna()).sum())
    edge = []
    prev_csv = os.path.join(a.prev_dir, f"{name}_static.csv") if a.prev_dir else None
    if not (prev_csv and os.path.exists(prev_csv)):
        return row, edge
    pv = pd.read_csv(prev_csv, dtype={"cell_id": str}).set_index("cell_id").reindex(nw.index)
    d = (nw["dem_mean"] - pv["dem_mean"]).abs()
    row["prev_nan_dem"] = int(pv["dem_mean"].isna().sum())
    row["prev_dem_absd_gt0.1"] = int((d > 0.1).sum())
    row["prev_dem_absd_gt0.5"] = int((d > 0.5).sum())
    c0 = lc_code0(pv)
    cov_old = wc_frame_cover(grid, a.worldcover_prev).reindex(nw.index)
    row["prev_code0_gt0"] = int((c0 > 1e-9).sum())
    row["prev_code0_gt0.5"] = int((c0 > 0.5).sum())
    row["prev_frame_cover_lt1"] = int((cov_old < 1 - 1e-9).sum())
    rim = (c0 > 1e-9) | (cov_old < 1 - 1e-9)
    row["rim_cells"] = int(rim.sum())
    for cls in ("Water", "Mangroves"):
        col = f"{LANDCOVER_PREFIX}{cls}"
        o, n_ = pv.loc[rim, col], nw.loc[rim, col]
        row[f"rim_{cls}_median_prev"] = round(float(o.median()), 4) if rim.any() else np.nan
        row[f"rim_{cls}_median_new"] = round(float(n_.median()), 4) if rim.any() else np.nan
        row[f"rim_{cls}_d_max"] = round(float((n_ - o).max()), 4) if rim.any() else np.nan
        row[f"nonrim_{cls}_absd_max"] = round(float((nw.loc[~rim, col] - pv.loc[~rim, col]).abs().max()), 6)
    for cid in nw.index[rim]:
        edge.append({"grid": name, "cell_id": cid, "code0_prev": round(float(c0[cid]), 4),
                     "frame_cover_prev": round(float(cov_old[cid]), 4), "code0_new": round(float(c0_new[cid]), 6),
                     **{f"{cls}_{v}": round(float(src.loc[cid, f"{LANDCOVER_PREFIX}{cls}"]), 4)
                        for cls in ("Water", "Mangroves", "Cropland") for v, src in (("prev", pv), ("new", nw))}})
    return row, edge


def control_dem(a, grid, out_csv):
    """dem_mean cua raster doi chung (khong ap CHG-09) theo o; o scope 0 -> NaN nhu ban chinh."""
    ctrl = area_weighted_means(a.control_stack, (grid["cell_id"].tolist(), None, grid.geometry.tolist()))
    ctrl = ctrl.set_index(ctrl.columns[0]).iloc[:, 0]  # band 1 = dem_mean
    ctrl.index = ctrl.index.astype(str)
    sf = pd.read_csv(out_csv, dtype={"cell_id": str}).set_index("cell_id")[SCOPE_FRAC_COL].reindex(ctrl.index)
    ctrl[~(sf > 0)] = np.nan
    return ctrl


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    free_gb = shutil.disk_usage(a.out_dir).free / 1e9
    log(f"Cho trong o dich: {free_gb:.2f} GB")
    if free_gb < a.min_free_gb:
        raise SystemExit(f"[LOI] o dich con < {a.min_free_gb} GB.")
    for p in (a.scope, a.dem, a.dem_flags, a.river, a.worldcover, a.boundary):
        if not os.path.exists(p):
            raise SystemExit(f"[LOI] thieu {p}")
    log("Tinh sha256 dau vao ...")
    shas = {"scope": file_sha256(a.scope), "dem": file_sha256(a.dem), "dem_flags": file_sha256(a.dem_flags),
            "river": file_sha256(a.river), "worldcover": file_sha256(a.worldcover)}
    for k, v in shas.items():
        log(f"  {k}: {v[:12]}…")
    ensure_stack(a, shas)
    if a.control_stack and not a.no_report:
        ensure_control_stack(a, shas)

    paths = sorted(glob.glob(os.path.join(a.grids_dir, "*.geojson")))
    if a.grids:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in set(a.grids)]
    summ, grp, top, pts, ver, rim = [], [], [], [], [], []
    for path in paths:
        out = build_grid(a, path, shas)
        if a.no_report:
            continue
        name, grid, new, m = compare_grid(a, path, out)
        summ.append(summarize(name, new, m))
        grp += by_group(name, m)
        top += top_cells(name, m)
        pts += known_points(name, grid, new)
        v, e = compare_versions(a, name, grid, new, control_dem(a, grid, out) if a.control_stack else None)
        ver.append(v)
        rim += e
        log(f"  {name}: doi phien ban {v}")
        s = summ[-1]
        log(f"  {name}: NaN dac trung scope {s['nan_scope_feat']}/{s['cells']}; |Δ| trung vi dem "
            f"{s['absd_dem_mean_median']}, coast {s['absd_dist_coast_km_median']} km; DEM trung vi chung "
            f"scope {s['dem_median_scope_common']} vs ca o {s['dem_median_full_common']}")
    if summ and a.report_dir:
        os.makedirs(a.report_dir, exist_ok=True)
        for rows, fname in ((summ, "tong_hop"), (grp, "chenh_theo_nhom"), (top, "o_chenh_lon"), (pts, "diem_quen"),
                            (ver, "doi_phien_ban"), (rim, "o_ven_ria_worldcover")):
            if not rows:
                continue
            p = os.path.join(a.report_dir, f"{a.report_prefix}_{fname}.csv")
            pd.DataFrame(rows).to_csv(p, index=False)
            log(f"Bao cao: {p}")
        log(pd.DataFrame(summ)[["grid", "cells", "nan_scope_feat", "scope_frac_0", "dem_median_scope_common",
                                "dem_median_full_common", "absd_dist_coast_km_median",
                                "absd_dist_coast_km_max"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scope", default="A:/Dataset_NCKH/features/scope_mask_v3.tif",
                    help="Mat na pham vi (v3 CHG-10; v2: A:/Dataset_NCKH/features/scope_mask_v2.tif)")
    ap.add_argument("--dem", default="A:/Dataset_NCKH/gee/DEM_DBSCL.tif")
    ap.add_argument("--river", default="A:/Dataset_NCKH/raw/river/river_distance_90m.tif")
    ap.add_argument("--dem-flags", default="A:/Dataset_NCKH/gee/DEM_DBSCL_flags_v2.tif",
                    help="co Copernicus DEM (band WBM) cho CHG-09")
    ap.add_argument("--worldcover", default="A:/Dataset_NCKH/gee/LandCover_DBSCL_2021_v2.tif")
    ap.add_argument("--worldcover-prev", default="A:/Dataset_NCKH/gee/LandCover_DBSCL_2021.tif",
                    help="WorldCover cua ban truoc (chi de bao cao do phu khung o --prev-dir)")
    ap.add_argument("--prev-dir", default=None, help="thu muc CSV ban truoc de so (vd <out-dir>/deprecated_v2)")
    ap.add_argument("--control-stack", default=None,
                    help="raster doi chung KHONG ap CHG-09 (dat o o C/scratch) - tach tac dong quy tac DEM")
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--grids", nargs="*", help="chi chay cac luoi nay (ten file khong duoi)")
    ap.add_argument("--out-dir", default="A:/Dataset_NCKH/features/static")
    ap.add_argument("--stack", default=None, help="raster trung gian (mac dinh <out-dir>/static_scope_30m.tif)")
    ap.add_argument("--old-river-dir", default="A:/Dataset_NCKH/raw/river/features")
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--report-prefix", default="dot4_dac_trung_tinh", help="tien to file bao cao")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--force", action="store_true", help="ghi lai CSV da co")
    ap.add_argument("--force-stack", action="store_true", help="dung lai raster trung gian")
    args = ap.parse_args()
    args.stack = args.stack or os.path.join(args.out_dir, "static_scope_30m.tif")
    main(args)

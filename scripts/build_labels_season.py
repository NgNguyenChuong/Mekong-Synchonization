#!/usr/bin/env python
"""Nhan theo (o, mua kho) 2014-2026 cho 13 luoi + gia tri tham chieu tai diem danh gia (P5, ban v3).

Quy tac pixel (src/label_season.py; quy tac NaN, mask nuoc buffer 0, N, M, A5 - 2026-10-03; CHG-08, CHG-10, CHG-15):
  nhan 2014-2023 = Zenodo; 2024-2026 = GEE `_l8_v2` (can theo transform, cat dau chan Zenodo qua pham vi).
  bo CHINH: quy tac NaN ∩ pham vi v3 (v2 ∩ WC {10,20,30,40} ∩ dau chan) ∩ khong nuoc dong (water_freq >= 50 ∪ WC 80)
            ∩ mask du tin cay (CHG-08: n_clear >= 2 va water_freq != 255).
  bo PHU (moi bo khac bo chinh dung 1 dieu kien): keepwater (bo mask nuoc dong, them WC 80), keepmangrove (them
  WC 95), keep6090 (them WC 60, 90). CHG-08 ap moi bo.
Don vi: EC1:5 (dS/m), do man DAT.

Buoc:
  1. Tinh tinh (mot lan): v2 (tam pixel) + band wc_class/zenodo_n_years/scope cua scope_mask_v3.tif;
     doi chieu: scope == v2 ∩ wc_class {10,20,30,40} ∩ zenodo_n_years >= 1, wc_class == WorldCover tai tam pixel
     trong v2 (lech -> loi).
  2. Moi mua: raster 4 band (main, keepwater, keepmangrove, keep6090; NaN = khong hop le) o --work-dir (mac dinh
     o C) + thong ke pixel theo bo va cac tap tach tac dong + tham chieu diem (median 3x3, >= 5/9). Da co va cung
     RULES_VERSION -> bo qua.
  3. Moi luoi: exactextract mean + count -> <luoi>_labels_season{,_keepwater,_keepmangrove,_keep6090}.csv:
     cell_id, season, salinity, n_valid_px, valid_frac, train_ok (>= 100 px, quy tac N), train_ok_10pct.
  4. Diem: points_reference.csv (bo chinh, --points) va points_reference_<bo phu>.csv (diem chinh + diem goc
     --points-source nam tren lop duoc them; giu point_id, block_id, is_holdout). Cung cot:
     point_id, block_id, is_holdout, wc_class, added, season, ref_salinity, n_valid_3x3.
  5. Bao cao KE_HOACH/ket-qua/dot4_phan_bo_nhan_v3.csv (nam x tap), dot4_phan_bo_nhan_v3_tach_tac_dong.csv (so P5 cu:
     (a) loai WC 50/60/90, (b) CHG-08, ca hai thu tu), dot4_phan_bo_nhan_v3_luoi.csv (luoi x bo x mua: o train_ok,
     ven bien = tam o <= 20 km toi bo; kem so P5 cu cua bo chinh neu co bao cao cu).
  Moi dau ra kem .provenance.json.

Chay:  venv/Scripts/python.exe scripts/build_labels_season.py [--years 2014 2026] [--grids h3_res_5 ...]
           [--out-dir A:/Dataset_NCKH/labels] [--work-dir <tam>] [--force] [--report-dir KE_HOACH/ket-qua]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from label_season import (MAIN_LC, MIN_CLEAR, MIN_TRAIN_FRAC, MIN_TRAIN_PX, REF_MIN_VALID,  # noqa: E402
                          RULES_VERSION, VARIANT_EXTRA_LC, VARIANTS, area_weighted_mean_count, cell_label_rows,
                          clear_ok, comparison_masks, median_3x3, nan_rule, points_to_pixels, read_aligned,
                          value_summary, variant_masks, variant_point_sets, water_mask)
from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402
from scope_mask import SCOPE_BANDS_V3, SCOPE_INCLUDE_LC, band_index, wc_at_centers  # noqa: E402

DATA = "A:/Dataset_NCKH"
COMPARE = ("v2_old", "v2_clear", "v3_noclear", "drop_lc", "drop_lc50", "drop_lc6090", "drop_clear",
           "drop_clear_n0", "drop_clear_n1")
SETS = ("before",) + VARIANTS + COMPARE
SUFFIX = {"main": "", "keepwater": "_keepwater", "keepmangrove": "_keepmangrove", "keep6090": "_keep6090"}
COAST_KM = 20.0
UNIT_NOTE = "salinity = EC1:5 (dS/m), do man DAT (Wang 2025 / Nguyen 2020: S = 28,013*exp(-13,39*B5))"
RULES = {
    "rules_version": RULES_VERSION,
    "nan_rule": "NDWIchen NaN | Salinity NaN | S > 28,013 | B7 suy nguoc <= 0 (hoac khong huu han); "
                "B5 = -ln(S/28,013)/13,39, B7 = B5*(1-NDWI)/(1+NDWI)",
    "water": "water_freq >= 50 (255 = khong quan sat, khong tinh) ∪ WorldCover 80 (tam pixel 30 m); buffer 0 (CHG-05)",
    "mask_reliability": f"CHG-08: n_clear < {MIN_CLEAR} hoac water_freq = 255 -> NaN, moi bo (MIN_CLEAR = {MIN_CLEAR})",
    "scope_main": "v3 (CHG-10): v2 (tam pixel) ∩ WorldCover thuoc {10,20,30,40} ∩ dau chan Zenodo "
                  "(Salinity huu han >= 1 nam 2014-2023)",
    "keepwater": "CHG-15: bo mask nuoc dong, them WC 80; van WC {10,20,30,40} + CHG-08",
    "keepmangrove": "CHG-15: them WC 95; van mask nuoc dong + CHG-08",
    "keep6090": "CHG-15: them WC 60, 90; van mask nuoc dong + CHG-08",
    "label_sources": "2014-2023 Zenodo 15653696; 2024-2026 GEE _l8_v2 (can luoi theo transform, khong noi suy)",
    "cell_value": "trung binh co trong so dien tich (exactextract mean) tren pixel hop le",
    "n_valid_px": "tong ty le phu cua pixel hop le trong o (exactextract count) = so pixel 30 m tuong duong",
    "train_ok": f"n_valid_px >= {MIN_TRAIN_PX} (quy tac N)",
    "train_ok_10pct": f"valid_frac >= {MIN_TRAIN_FRAC} (do nhay)",
}
SET_NOTE = {
    "before": "chi quy tac NaN, trong v2 ∩ dau chan (khong WC, khong CHG-08)",
    "main": "bo chinh v3 + CHG-08", "keepwater": "bo phu giu nuoc", "keepmangrove": "bo phu giu WC 95",
    "keep6090": "bo phu giu WC 60/90",
    "v2_old": "bo chinh P5 cu (WC khong {0,80,95}, khong CHG-08) - dung lai de doi chieu",
    "v2_clear": "P5 cu + CHG-08", "v3_noclear": "v3 khong CHG-08",
    "drop_lc": "pixel bi loai boi (a) = v2_old - v3_noclear", "drop_lc50": "phan WC 50 cua drop_lc",
    "drop_lc6090": "phan WC 60/90 cua drop_lc", "drop_clear": "pixel bi loai boi (b) = v3_noclear - main",
    "drop_clear_n0": "drop_clear co water_freq 255 hoac n_clear 0", "drop_clear_n1": "drop_clear co n_clear 1",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def label_path(a, s):
    if s <= 2023:
        return os.path.join(a.zenodo_dir, f"{s}_MD_dry_NDWIchen_Salinity.tif")
    return os.path.join(a.gee_dir, f"{s}_MD_dry_NDWIchen_Salinity_l8_v2.tif")


def mask_path(a, s):
    return os.path.join(a.gee_dir, f"{s}_MD_dry_watermask_l8_v2.tif")


class Hasher:
    """sha256 co cache theo (duong dan, kich thuoc, mtime) trong work_dir - file nhan lon, bam mot lan."""

    def __init__(self, path):
        self.path = path
        self.cache = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.cache = json.load(f)

    def __call__(self, p):
        st = os.stat(p)
        key = f"{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}"
        if key not in self.cache:
            self.cache[key] = file_sha256(p)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=1)
        return self.cache[key]


def rel(p):
    """Duong dan tuong doi ROOT; khac o dia (Windows) -> duong dan tuyet doi."""
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:
        return os.path.abspath(p).replace("\\", "/")


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- buoc 1
def load_static(a):
    import rasterio
    from rasterio import features

    with rasterio.open(a.scope) as sc:
        if tuple(sc.descriptions) != SCOPE_BANDS_V3:
            raise SystemExit(f"[LOI] {a.scope}: band {sc.descriptions} != {SCOPE_BANDS_V3} (can scope v3 co wc_class).")
        tr, crs, shape = sc.transform, sc.crs, (sc.height, sc.width)
        scope = sc.read(band_index(sc, "scope")).astype(bool)
        fp = sc.read(band_index(sc, "zenodo_n_years")) >= 1
        wc = sc.read(band_index(sc, "wc_class"))
    prov = a.scope + ".provenance.json"
    if os.path.exists(prov):
        info = read_json(prov)
        if info.get("boundary_sha256") and info["boundary_sha256"] != file_sha256(a.boundary):
            raise SystemExit(f"[LOI] {a.scope} dung ranh gioi khac {a.boundary} (sha256 lech).")
        inc = info.get("include_worldcover")
        if inc is not None and tuple(inc) != tuple(SCOPE_INCLUDE_LC):
            raise SystemExit(f"[LOI] {a.scope}: include_worldcover {inc} != {SCOPE_INCLUDE_LC} (khong phai v3).")
    boundary = gpd.read_file(a.boundary).to_crs(crs).geometry.union_all()
    in_v2 = features.rasterize([(boundary, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8",
                               all_touched=False).astype(bool)
    if (wc[~in_v2] != 0).any():
        raise SystemExit(f"[LOI] wc_class khac 0 ngoai ranh gioi o {int((wc[~in_v2] != 0).sum())} pixel.")
    wc_file = wc_at_centers(a.worldcover, tr, crs, shape)
    n_wc_diff = int((wc_file[in_v2] != wc[in_v2]).sum())
    del wc_file
    if n_wc_diff:
        raise SystemExit(f"[LOI] wc_class lech WorldCover {a.worldcover} o {n_wc_diff} pixel trong v2.")
    recon = in_v2 & np.isin(wc, MAIN_LC) & fp
    n_diff = int((recon != scope).sum())
    if n_diff:
        raise SystemExit(f"[LOI] scope v3 dung lai lech file {a.scope} o {n_diff} pixel - kiem lai ranh gioi/WC.")
    classes, counts = np.unique(wc[in_v2 & fp], return_counts=True)
    log(f"Tinh tinh: luoi {shape[1]}x{shape[0]}, scope v3 {scope.sum():,} px (khop dung lai), dau chan "
        f"{fp.sum():,} px, wc_class khop WorldCover; lop trong v2 ∩ dau chan: "
        + ", ".join(f"{c}:{n:,}" for c, n in zip(classes, counts)))
    return {"transform": tr, "crs": crs, "shape": shape, "in_v2": in_v2, "fp": fp, "wc": wc}


def load_points(a, st):
    """Tap diem hop (bo chinh + diem them cua bo phu) voi wc_class tai pixel chua diem."""
    def read(p):
        g = gpd.read_file(p)
        need = {"point_id", "block_id", "is_holdout"}
        if not need <= set(g.columns):
            raise SystemExit(f"[LOI] {p}: thieu cot {need - set(g.columns)}.")
        g["point_id"] = g["point_id"].astype(str)
        gu = g.to_crs(st["crs"])
        r, c = points_to_pixels(gu.geometry.x.to_numpy(), gu.geometry.y.to_numpy(), st["transform"])
        h, w = st["shape"]
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        rc = np.where(ok, r, 0), np.where(ok, c, 0)
        df = pd.DataFrame({"point_id": g["point_id"], "block_id": g["block_id"].astype(str),
                           "is_holdout": g["is_holdout"].astype(bool), "row": r, "col": c,
                           "wc_class": np.where(ok, st["wc"][rc], 0).astype(int),
                           "in_v2": np.where(ok, st["in_v2"][rc], False),
                           "in_fp": np.where(ok, st["fp"][rc], False),
                           "x": g.geometry.x.to_numpy(), "y": g.geometry.y.to_numpy()})
        return df

    main = read(a.points)
    src = read(a.points_source)
    bad = ~(main["in_v2"] & main["in_fp"] & main["wc_class"].isin(MAIN_LC))
    if bad.any():
        raise SystemExit(f"[LOI] {int(bad.sum())} diem bo chinh nam ngoai pham vi v3 - kiem lai --points.")
    # diem chinh phai la tap con cua nguon voi thuoc tinh/toa do giu nguyen
    j = main.merge(src, on="point_id", how="left", suffixes=("", "_src"), indicator=True)
    n_missing = int((j["_merge"] != "both").sum())
    jb = j[j["_merge"] == "both"]
    n_attr = int(((jb["block_id"] != jb["block_id_src"]) | (jb["is_holdout"] != jb["is_holdout_src"])
                  | ((jb["x"] - jb["x_src"]).abs() > 1e-9) | ((jb["y"] - jb["y_src"]).abs() > 1e-9)).sum())
    if n_missing or n_attr:
        raise SystemExit(f"[LOI] diem chinh khong khop nguon {a.points_source}: thieu {n_missing}, lech thuoc tinh "
                         f"{n_attr}.")
    allp = variant_point_sets(main, src)
    added = allp[~allp["in_main"]]
    bad = ~(added["in_v2"] & added["in_fp"])
    if bad.any():
        raise SystemExit(f"[LOI] {int(bad.sum())} diem them nam ngoai v2 ∩ dau chan.")
    msg = ", ".join(f"{v}: +{int((allp[f'in_{v}'] & ~allp['in_main']).sum())}" for v in VARIANTS if v != "main")
    log(f"Diem: bo chinh {len(main):,} (nguon {len(src):,}); diem them theo bo: {msg}")
    return allp


# ---------------------------------------------------------------- buoc 2
def season_raster(a, s, st, pts, chunk_rows=1024):
    """Ghi raster 4 band cua mua s; tra ve (thong ke theo tap, bang tham chieu diem cho tap hop diem)."""
    import rasterio
    from rasterio.windows import Window

    tif = os.path.join(a.work_dir, f"labels_{s}.tif")
    stats_p = os.path.join(a.work_dir, f"labels_{s}.stats.json")
    pts_p = os.path.join(a.work_dir, f"points_{s}.csv")
    if not a.force and all(os.path.exists(p) for p in (tif, stats_p, pts_p)):
        stats = read_json(stats_p)
        ptab = pd.read_csv(pts_p, dtype={"point_id": str})
        if stats.get("rules_version") == RULES_VERSION and len(ptab) == len(pts) \
                and ptab["point_id"].tolist() == pts["point_id"].tolist():
            log(f"  mua {s}: da co raster/thong ke ({RULES_VERSION}) o work-dir - bo qua")
            return stats, ptab
        log(f"  mua {s}: raster cu khac quy tac/tap diem - tinh lai")
    tr, (h, w) = st["transform"], st["shape"]
    S_full = np.full((h, w), np.nan, np.float32)
    flags = np.zeros((h, w), np.uint16)
    n_fake0 = 0
    prof = dict(driver="GTiff", width=w, height=h, count=len(VARIANTS), dtype="float32", crs=st["crs"],
                transform=tr, nodata=np.nan, compress="deflate", predictor=3, tiled=True, blockxsize=512,
                blockysize=512, BIGTIFF="IF_SAFER")
    with rasterio.open(label_path(a, s)) as lab, rasterio.open(mask_path(a, s)) as wm, \
            rasterio.open(tif + ".part", "w", **prof) as dst:
        for i, v in enumerate(VARIANTS, start=1):
            dst.set_band_description(i, v)
        dst.update_tags(unit="EC1:5 dS/m", season=str(s), source=os.path.basename(label_path(a, s)),
                        watermask=os.path.basename(mask_path(a, s)), rules_version=RULES_VERSION)
        for r0 in range(0, h, chunk_rows):
            n = min(chunk_rows, h - r0)
            sl = slice(r0, r0 + n)
            ndwi = read_aligned(lab, "NDWIchen", tr, (n, w), r0, fill=np.nan, dtype="float32")
            sal = read_aligned(lab, "Salinity", tr, (n, w), r0, fill=np.nan, dtype="float32")
            # ngoai khung mask: water_freq 0, n_clear 0 -> khong du tin cay (CHG-08) -> NaN
            wf = read_aligned(wm, "water_freq", tr, (n, w), r0, fill=0, dtype="uint8")
            ncl = read_aligned(wm, "n_clear", tr, (n, w), r0, fill=0, dtype="uint8")
            wc = st["wc"][sl]
            S = nan_rule(ndwi, sal)
            fin = np.isfinite(S)
            water = water_mask(wf, wc)
            m = variant_masks(fin, st["in_v2"][sl], st["fp"][sl], wc, water, clear_ok(wf, ncl))
            m.update(comparison_masks(fin, st["in_v2"][sl], st["fp"][sl], wc, water, wf, ncl))
            fl = np.zeros((n, w), np.uint16)
            for bit, k in enumerate(SETS):
                fl |= m[k].astype(np.uint16) << bit
            flags[sl] = fl
            S_full[sl] = S
            n_fake0 += int((m["before"] & (ncl == 0) & (wf != 255)).sum())   # "0 gia"/ngoai khung (pitfall 4q)
            out = np.full((len(VARIANTS), n, w), np.nan, np.float32)
            for i, v in enumerate(VARIANTS):
                out[i][m[v]] = S[m[v]]
            dst.write(out, window=Window(0, r0, w, n))
    os.replace(tif + ".part", tif)
    px_km2 = abs(tr.a * tr.e) / 1e6
    stats = {"rules_version": RULES_VERSION}
    for bit, k in enumerate(SETS):
        v = S_full[((flags >> bit) & 1).astype(bool)]
        d = value_summary(v)
        del v
        d["area_km2"] = d["n_px"] * px_km2
        stats[k] = {kk: (float(x) if isinstance(x, (float, np.floating)) else int(x)) for kk, x in d.items()}
    del S_full, flags
    stats["before"]["n_px_mask_fake0"] = n_fake0
    # tham chieu diem: median 3x3 tren tung bo, tai moi diem cua tap hop
    rows, cols = pts["row"].to_numpy(), pts["col"].to_numpy()
    ptab = pd.DataFrame({"point_id": pts["point_id"].to_numpy(), "season": s})
    with rasterio.open(tif) as ds:
        for i, v in enumerate(VARIANTS, start=1):
            med, nv = median_3x3(ds.read(i), rows, cols, REF_MIN_VALID)
            ptab[f"ref_salinity_{v}"] = med
            ptab[f"n_valid_3x3_{v}"] = nv
    for v in VARIANTS:
        mem = pts[f"in_{v}"].to_numpy()
        stats[v]["n_points"] = int(mem.sum())
        stats[v]["n_points_ref"] = int(ptab.loc[mem, f"ref_salinity_{v}"].notna().sum())
    with open(stats_p, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    ptab.to_csv(pts_p, index=False, float_format="%.6g")
    m_, o_ = stats["main"], stats["v2_old"]
    log(f"  mua {s}: truoc {stats['before']['n_px']:,} px (>4: {stats['before']['pct_gt4']:.2f}%); P5 cu "
        f"{o_['n_px']:,} (>4: {o_['pct_gt4']:.2f}%) -> chinh {m_['n_px']:,} px (>4: {m_['pct_gt4']:.2f}%), "
        f"dai {m_['min']:.3f}..{m_['max']:.3f}; CHG-08 loai {stats['drop_clear']['n_px']:,} "
        f"(n0 {stats['drop_clear_n0']['n_px']:,}, n1 {stats['drop_clear_n1']['n_px']:,}); "
        f"diem ref {m_['n_points_ref']}/{m_['n_points']}, 0 gia {n_fake0}")
    return stats, ptab


# ---------------------------------------------------------------- buoc 3
def coastal_flags(cells_utm_centroids, dist_path):
    """dist_coast_km tai pixel 90 m chua tam o (NaN neu ngoai raster/NaN)."""
    import rasterio

    with rasterio.open(dist_path) as ds:
        b = band_index(ds, "dist_coast_km")
        arr = ds.read(b)
        r, c = points_to_pixels(cells_utm_centroids.x.to_numpy(), cells_utm_centroids.y.to_numpy(), ds.transform)
        ok = (r >= 0) & (r < ds.height) & (c >= 0) & (c < ds.width)
        d = np.full(len(r), np.nan)
        d[ok] = arr[r[ok], c[ok]]
    return d


def _outputs_current(paths):
    for p in paths:
        pp = p + ".provenance.json"
        if not (os.path.exists(p) and os.path.exists(pp)):
            return False
        if read_json(pp).get("rules", {}).get("rules_version") != RULES_VERSION:
            return False
    return True


def grid_tables(a, grid_path, years, st, hasher, src_prov):
    name = os.path.splitext(os.path.basename(grid_path))[0]
    outs = {v: os.path.join(a.out_dir, f"{name}_labels_season{SUFFIX[v]}.csv") for v in VARIANTS}
    grid = gpd.read_file(grid_path)
    grid["cell_id"] = grid["cell_id"].astype(str)
    if grid["cell_id"].duplicated().any():
        raise SystemExit(f"[LOI] {name}: cell_id trung.")
    if not a.force and _outputs_current(outs.values()):
        log(f"{name}: da co {len(outs)} file ({RULES_VERSION}) - bo qua tinh lai")
        tabs = {v: pd.read_csv(p, dtype={"cell_id": str}) for v, p in outs.items()}
    else:
        t = time.time()
        cells = (grid["cell_id"].tolist(), None, grid.geometry.tolist())
        px_m2 = abs(st["transform"].a * st["transform"].e)
        parts = {v: [] for v in VARIANTS}
        for s in years:
            df = area_weighted_mean_count(os.path.join(a.work_dir, f"labels_{s}.tif"), cells)
            for i, v in enumerate(VARIANTS, start=1):
                lab = cell_label_rows(df[f"b{i}_mean"], df[f"b{i}_count"], df["cell_area_m2"], px_m2)
                lab.insert(0, "season", s)
                lab.insert(0, "cell_id", df["cell_id"].to_numpy())
                parts[v].append(lab)
        tabs = {}
        for v in VARIANTS:
            tab = pd.concat(parts[v], ignore_index=True)
            if tab.duplicated(["cell_id", "season"]).any():
                raise SystemExit(f"[LOI] {name}/{v}: khoa (cell_id, season) trung.")
            if len(tab) != len(grid) * len(years):
                raise SystemExit(f"[LOI] {name}/{v}: {len(tab)} dong != {len(grid)} o x {len(years)} mua.")
            tab.to_csv(outs[v] + ".part", index=False, float_format="%.6g")
            os.replace(outs[v] + ".part", outs[v])
            write_provenance(outs[v], a.boundary, variant=v, unit=UNIT_NOTE, rules=RULES, seasons=list(years),
                             grid=os.path.basename(grid_path), grid_sha256=hasher(grid_path), **src_prov)
            tabs[v] = tab
        log(f"{name}: {len(grid)} o x {len(years)} mua ({time.time() - t:.0f}s)")
    cent = grid.to_crs(st["crs"]).geometry.centroid
    dist = coastal_flags(cent, a.dist_raster)
    coast = pd.Series(dist <= COAST_KM, index=grid["cell_id"])
    rows = []
    for v, tab in tabs.items():
        tab = tab.assign(coastal=tab["cell_id"].map(coast).astype(bool))
        for s, g in tab.groupby("season"):
            rows.append({"grid": name, "variant": v, "season": int(s), "n_cells": len(g),
                         "n_cells_coastal": int(g["coastal"].sum()),
                         "n_cells_coast_nan": int(np.isnan(dist).sum()),
                         "n_cells_label": int((g["n_valid_px"] > 0).sum()),
                         "n_train_ok": int(g["train_ok"].sum()),
                         "n_train_ok_coastal": int((g["train_ok"] & g["coastal"]).sum()),
                         "n_train_ok_10pct": int(g["train_ok_10pct"].sum()),
                         "n_train_ok_10pct_coastal": int((g["train_ok_10pct"] & g["coastal"]).sum()),
                         "salinity_median_train_ok": float(g.loc[g["train_ok"], "salinity"].median())})
    return rows


# ---------------------------------------------------------------- bao cao
def decomposition_table(rep_year, old_year=None):
    """Bang tach tac dong theo mua: P5 cu (v2_old) -(a)-> v3_noclear -(b)-> main; va thu tu nguoc qua v2_clear."""
    piv = rep_year.set_index(["season", "set"])
    metrics = ("n_px", "pct_gt2", "pct_gt4", "n_gt4", "n_gt21", "p50", "p99", "max")
    rows = []
    for s in sorted(rep_year["season"].unique()):
        r = {"season": int(s)}
        for k, tag in (("before", "truoc"), ("v2_old", "p5cu"), ("v3_noclear", "sau_a"), ("main", "chinh"),
                       ("v2_clear", "p5cu_chg08")):
            for mtr in metrics:
                r[f"{tag}_{mtr}"] = piv.loc[(s, k), mtr]
        for mtr in ("n_px", "pct_gt2", "pct_gt4", "n_gt4", "n_gt21", "p99", "max"):
            r[f"delta_a_{mtr}"] = r[f"sau_a_{mtr}"] - r[f"p5cu_{mtr}"]          # (a) truoc, (b) sau
            r[f"delta_b_{mtr}"] = r[f"chinh_{mtr}"] - r[f"sau_a_{mtr}"]
            r[f"delta_b_truoc_{mtr}"] = r[f"p5cu_chg08_{mtr}"] - r[f"p5cu_{mtr}"]   # thu tu nguoc
            r[f"delta_a_sau_{mtr}"] = r[f"chinh_{mtr}"] - r[f"p5cu_chg08_{mtr}"]
        for k in ("drop_lc", "drop_lc50", "drop_lc6090", "drop_clear", "drop_clear_n0", "drop_clear_n1"):
            for mtr in ("n_px", "n_gt4", "pct_gt4", "n_gt21", "p50"):
                r[f"{k}_{mtr}"] = piv.loc[(s, k), mtr]
        if old_year is not None:
            o = old_year[(old_year["season"] == s) & (old_year["set"] == "main")]
            if len(o):
                o = o.iloc[0]
                r["p5cu_bao_cao_n_px"] = int(o["n_px"])
                r["p5cu_bao_cao_n_gt4"] = int(o["n_gt4"])
                r["v2_old_khop_bao_cao_cu"] = bool(int(o["n_px"]) == r["p5cu_n_px"]
                                                   and int(o["n_gt4"]) == r["p5cu_n_gt4"]
                                                   and int(o["n_gt2"]) == piv.loc[(s, "v2_old"), "n_gt2"])
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main(a):
    t0 = time.time()
    years = list(range(a.years[0], a.years[1] + 1))
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.work_dir, exist_ok=True)
    os.makedirs(a.report_dir, exist_ok=True)
    free_out = shutil.disk_usage(a.out_dir).free / 1e9
    free_work = shutil.disk_usage(a.work_dir).free / 1e9
    log(f"Cho trong: dich {a.out_dir} {free_out:.1f} GB; work {a.work_dir} {free_work:.1f} GB")
    if free_out < 1.0:
        raise SystemExit("[LOI] o dich con < 1 GB.")
    if free_work < 0.5 * len(years):
        raise SystemExit(f"[LOI] work-dir con {free_work:.1f} GB < {0.5 * len(years):.1f} GB uoc tinh.")
    srcs = {s: (label_path(a, s), mask_path(a, s)) for s in years}
    miss = [p for pair in srcs.values() for p in pair if not os.path.exists(p)]
    if miss:
        raise SystemExit(f"[LOI] thieu file: {miss}")
    grids = sorted(glob.glob(os.path.join(a.grids_dir, "*.geojson")))
    if a.grids:
        grids = [os.path.join(a.grids_dir, f"{g}.geojson") for g in a.grids]
    if not grids or any(not os.path.exists(g) for g in grids):
        raise SystemExit(f"[LOI] luoi khong ton tai: {grids}")

    hasher = Hasher(os.path.join(a.work_dir, "sha256_cache.json"))
    log("Bam sha256 nguon (co cache) ...")
    src_prov = {
        "labels_sha256": {os.path.basename(srcs[s][0]): hasher(srcs[s][0]) for s in years},
        "watermask_sha256": {os.path.basename(srcs[s][1]): hasher(srcs[s][1]) for s in years},
        "scope": os.path.basename(a.scope), "scope_sha256": hasher(a.scope),
        "worldcover": os.path.basename(a.worldcover), "worldcover_sha256": hasher(a.worldcover),
        "min_clear": MIN_CLEAR, "main_worldcover": list(MAIN_LC),
        "variant_extra_worldcover": {k: list(v) for k, v in VARIANT_EXTRA_LC.items()},
    }

    st = load_static(a)
    pts = load_points(a, st)

    log(f"Buoc 2: raster nhan theo mua -> {a.work_dir}")
    year_rows, pt_tabs = [], []
    for s in years:
        stats, ptab = season_raster(a, s, st, pts)
        pt_tabs.append(ptab)
        for k in SETS:
            year_rows.append({"season": s, "set": k, "note": SET_NOTE[k], **stats[k]})
    st = {k: st[k] for k in ("transform", "crs", "shape")}   # giai phong mang tinh

    ptab = pd.concat(pt_tabs, ignore_index=True)
    if ptab.duplicated(["point_id", "season"]).any():
        raise SystemExit("[LOI] diem: khoa (point_id, season) trung.")
    attrs = pts[["point_id", "block_id", "is_holdout", "wc_class", "in_main"] + [f"in_{v}" for v in VARIANTS[1:]]]
    ptab = ptab.merge(attrs, on="point_id", how="left", validate="many_to_one")
    pt_prov = dict(unit=UNIT_NOTE, rules=RULES, seasons=years,
                   ref_rule=f"median cua so 3x3 pixel hop le (cua bo) quanh pixel chua diem; < {REF_MIN_VALID}/9 -> NaN",
                   points=os.path.basename(a.points), points_sha256=hasher(a.points),
                   points_source=rel(a.points_source),
                   points_source_sha256=hasher(a.points_source), **src_prov)
    for v in VARIANTS:
        sub = ptab[ptab[f"in_{v}"]]
        cols = {"point_id": sub["point_id"], "block_id": sub["block_id"], "is_holdout": sub["is_holdout"],
                "wc_class": sub["wc_class"], "added": ~sub["in_main"], "season": sub["season"],
                "ref_salinity": sub[f"ref_salinity_{v}"], "n_valid_3x3": sub[f"n_valid_3x3_{v}"]}
        out_df = pd.DataFrame(cols).sort_values(["season", "point_id"], kind="stable")
        p = os.path.join(a.out_dir, f"points_reference{SUFFIX[v]}.csv")
        out_df.to_csv(p + ".part", index=False, float_format="%.6g")
        os.replace(p + ".part", p)
        note = ("diem bo chinh (--points)" if v == "main" else
                f"diem bo chinh + diem nguon tren WC {list(VARIANT_EXTRA_LC[v])} (added = True)")
        write_provenance(p, a.boundary, variant=v, point_set=note, n_points=int(pts[f"in_{v}"].sum()),
                         n_points_added=int((pts[f"in_{v}"] & ~pts["in_main"]).sum()), **pt_prov)
        log(f"Ghi {p}: {len(out_df)} dong ({int(pts[f'in_{v}'].sum())} diem)")

    log("Buoc 3: bang nhan theo luoi")
    grid_rows = []
    for g in grids:
        grid_rows += grid_tables(a, g, years, st, hasher, src_prov)

    rep_year = pd.DataFrame(year_rows)
    old_year_p = os.path.join(a.report_dir, "dot4_phan_bo_nhan.csv")
    old_grid_p = os.path.join(a.report_dir, "dot4_phan_bo_nhan_luoi.csv")
    old_year = pd.read_csv(old_year_p) if os.path.exists(old_year_p) else None
    rep_dec = decomposition_table(rep_year, old_year)
    rep_grid = pd.DataFrame(grid_rows)
    if os.path.exists(old_grid_p):
        og = pd.read_csv(old_grid_p)
        og = og[og["variant"] == "main"][["grid", "season", "n_cells_label", "n_train_ok", "n_train_ok_coastal"]]
        og = og.rename(columns={c: f"{c}_p5cu" for c in ("n_cells_label", "n_train_ok", "n_train_ok_coastal")})
        og["variant"] = "main"
        rep_grid = rep_grid.merge(og, on=["grid", "season", "variant"], how="left")
    p_year = os.path.join(a.report_dir, "dot4_phan_bo_nhan_v3.csv")
    p_dec = os.path.join(a.report_dir, "dot4_phan_bo_nhan_v3_tach_tac_dong.csv")
    p_grid = os.path.join(a.report_dir, "dot4_phan_bo_nhan_v3_luoi.csv")
    rep_year.to_csv(p_year, index=False, float_format="%.6g")
    rep_dec.to_csv(p_dec, index=False, float_format="%.6g")
    rep_grid.to_csv(p_grid, index=False, float_format="%.6g")
    common = dict(unit=UNIT_NOTE, rules=RULES, seasons=years, sets=SET_NOTE, dist_coast=os.path.basename(a.dist_raster),
                  dist_coast_sha256=hasher(a.dist_raster), coast_km=COAST_KM,
                  coastal_rule="tam o (EPSG:32648) <= 20 km, band dist_coast_km pixel 90 m chua tam", **src_prov)
    write_provenance(p_year, a.boundary, **common)
    write_provenance(p_dec, a.boundary, compared_with=os.path.basename(old_year_p) if old_year is not None else None,
                     **common)
    write_provenance(p_grid, a.boundary, grids={os.path.basename(g): hasher(g) for g in grids}, **common)

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        cols = ["season", "set", "n_px", "pct_gt2", "pct_gt4", "n_gt21", "p50", "p99", "max"]
        print(rep_year[rep_year["set"].isin(["before", "v2_old", "v3_noclear", "main"] + list(VARIANTS[1:]))][cols]
              .to_string(index=False), flush=True)
        main_g = rep_grid[rep_grid["variant"] == "main"]
        piv = main_g.groupby("grid").agg(n_cells=("n_cells", "first"), coastal=("n_cells_coastal", "first"),
                                         train_ok_min=("n_train_ok", "min"), train_ok_max=("n_train_ok", "max"),
                                         coast_train_min=("n_train_ok_coastal", "min"),
                                         coast_train_max=("n_train_ok_coastal", "max"))
        print(piv.to_string(), flush=True)
    log(f"Ghi {p_year}, {p_dec}, {p_grid}. Xong trong {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zenodo-dir", default=os.environ.get("ZENODO_DIR", f"{DATA}/zenodo_15653696"))
    ap.add_argument("--gee-dir", default=f"{DATA}/gee")
    ap.add_argument("--scope", default=f"{DATA}/features/scope_mask_v3.tif")
    ap.add_argument("--worldcover", default=f"{DATA}/gee/LandCover_DBSCL_2021_v2.tif")
    ap.add_argument("--dist-raster", default=os.path.join(os.environ.get("RAW_DIR", f"{DATA}/raw"),
                                                         "river", "river_distance_90m.tif"))
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--points", default=os.path.join(ROOT, "data", "eval", "eval_points.geojson"),
                    help="Diem danh gia bo chinh (da loc v3)")
    ap.add_argument("--points-source", default=os.path.join(ROOT, "data", "eval", "deprecated_v2b",
                                                            "eval_points.geojson"),
                    help="Diem goc truoc khi loc v3 - nguon diem them cho bo phu (CHG-15)")
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--grids", nargs="*", help="Ten luoi (khong .geojson); mac dinh moi luoi trong --grids-dir")
    ap.add_argument("--years", nargs=2, type=int, default=[2014, 2026], metavar=("TU", "DEN"))
    ap.add_argument("--out-dir", default=os.environ.get("LABELS_DIR", f"{DATA}/labels"))
    ap.add_argument("--work-dir", default=os.environ.get("LABEL_WORK_DIR",
                                                         os.path.join(tempfile.gettempdir(), "mekong_label_season_v3")))
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--force", action="store_true", help="Tinh lai raster mua va bang luoi da co")
    main(ap.parse_args())

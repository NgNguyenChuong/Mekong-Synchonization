#!/usr/bin/env python
"""E5b + E5c Dot 7: nhan theo (o, mua) cho 13 luoi + dap an tai diem cho mua, buc xa, nhiet do, do am.

Quy tac (src/met_cell_labels.py; An chot 2026-10-07): TB CO TRONG SO PHAM VI (trong so pixel nguon = ty le dien
tich thuoc scope_mask_v3 30 m), exactextract weighted_mean, da giac o chieu sang CRS raster nguon (MCD18 =
sinusoidal hinh cau), khong lay mau lai raster. Dap an tai diem = pixel nguon chua diem.
  rain : CHIRPS v3 <DATA_ROOT>/raw/chirps3/chirps3_rain_<s>.tif band rain_sum           -> cot rain_chirps
  dsr  : MCD18A1 <DATA_ROOT>/raw/mcd18a1/mcd18a1_dsr_<s>.tif band dsr_mean (mua 2020 NaN) -> cot dsr_mcd18
  temp : ERA5 GOC <DATA_ROOT>/features/era5_season/temp_avg_<s>.tif band temp_c_mean     -> cot t2m_era5
  rh   : ERA5 GOC <DATA_ROOT>/features/era5_season/humid_<s>.tif band rh_percent_mean    -> cot rh_era5
Diem cham = eval_points ∩ co ref_salinity hop le >= 1 mua (10.454; --n-points-expected).

Dau ra (--out-dir, mac dinh <DATA_ROOT>/labels/dot7), moi file kem .provenance.json:
  scope_weight_<nguon>.tif                : trong so pham vi tren luoi nguon (chirps3 / mcd18a1 / era5)
  <luoi>_labels_season_<cot>.csv          : cell_id, season, <cot>, lbl_cover_frac, lbl_valid_px, lbl_ok
  points_reference_<cot>.csv              : point_id, block_id, is_holdout, season, ref_<cot>, src_px_valid,
                                            src_row, src_col (variant main, ref_rule_kind pixel)
Bao cao (--report-dir): dot7_e5_khi_tuong_luoi.csv (bien x luoi x mua: so o, o co nhan, o NaN, o phu mot phan),
  dot7_e5_khi_tuong_diem.csv (bien x mua: diem, diem NaN), dot7_e5_trong_so.csv.
Chay lai: bo qua bang/diem da co cung MET_RULES_VERSION + sha256 nguon/trong so/luoi (--force de tinh lai).

Chay:  venv/Scripts/python.exe scripts/build_met_labels.py [--targets rain dsr temp rh] [--grids h3_res_5 ...]
           [--years 2014 2026] [--out-dir <DATA_ROOT>/labels/dot7] [--work-dir <tam>] [--force]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from met_cell_labels import (MET_RULES_VERSION, MET_TARGETS, cell_weighted_labels, check_source,  # noqa: E402
                             grid_signature, nan_pattern_changes, point_pixel_values, scope_weights, stack_seasons,
                             write_raster)
from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402

DATA = data_path()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_points(a):
    """Diem cham duoc: eval_points ∩ co ref_salinity huu han o >= 1 mua."""
    import geopandas as gpd

    pts = gpd.read_file(a.points)
    pts["point_id"] = pts["point_id"].astype(str)
    if pts["point_id"].duplicated().any():
        raise SystemExit("[LOI] point_id trung trong file diem")
    ref = pd.read_csv(a.points_ref_sal, dtype={"point_id": str}, usecols=["point_id", "ref_salinity"])
    ok = set(ref.loc[ref["ref_salinity"].notna(), "point_id"])
    pts = pts[pts["point_id"].isin(ok)].reset_index(drop=True)
    if a.n_points_expected and len(pts) != a.n_points_expected:
        raise SystemExit(f"[LOI] so diem cham duoc {len(pts)} != {a.n_points_expected}")
    need = {"block_id", "is_holdout"}
    if not need <= set(pts.columns):
        raise SystemExit(f"[LOI] file diem thieu cot {need - set(pts.columns)}")
    return pts


def read_values(path, band_idx):
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(band_idx).astype(np.float64), ds.crs, ds.transform


def weights_for(a, src_key, sig, crs, transform):
    """Raster trong so cua luoi nguon (cache trong out-dir, kiem sha scope + luoi)."""
    import rasterio

    path = os.path.join(a.out_dir, f"scope_weight_{src_key}.tif")
    scope_sha = file_sha256(a.scope)
    if os.path.exists(path) and os.path.exists(path + ".provenance.json") and not a.force:
        info = read_json(path + ".provenance.json")
        with rasterio.open(path) as ds:
            same = grid_signature(ds) == sig
        if same and info.get("scope_sha256") == scope_sha and info.get("rules_version") == MET_RULES_VERSION:
            log(f"  trong so {src_key}: da co - dung lai")
            return path, info["weights_info"]
    t = time.time()
    w, info = scope_weights(a.scope, crs, transform, (sig[3], sig[2]))
    if info["n_scope_outside"]:
        raise SystemExit(f"[LOI] {info['n_scope_outside']} pixel scope nam ngoai luoi nguon {src_key}")
    write_raster(path, [w], crs, transform, ["scope_weight"], nodata=None)
    write_provenance(path, a.boundary, rules_version=MET_RULES_VERSION, scope=os.path.basename(a.scope),
                     scope_sha256=scope_sha, weights_info=info,
                     rule="w_p = (so tam pixel scope 30 m trong p) * 900 m2 / dien tich pixel p (kinh-vi: geodesic "
                          "WGS84; sinusoidal hinh cau: |a*e|), cat <= 1")
    log(f"  trong so {src_key}: {info['n_px_w_gt0']} pixel w > 0, w max (truoc cat) {info['w_max_raw']:.4f}, "
        f"{info['n_scope_px']:,} pixel scope ({time.time() - t:.0f}s)")
    return path, info


def outputs_current(path, want):
    pp = path + ".provenance.json"
    if not (os.path.exists(path) and os.path.exists(pp)):
        return False
    info = read_json(pp)
    return all(info.get(k) == v for k, v in want.items())


def run_target(a, key, years, grids, pts):
    spec = MET_TARGETS[key]
    col = spec["col"]
    log(f"== {key} -> cot {col} ({spec['unit']})")
    paths = {s: os.path.join(DATA if a.data_root is None else a.data_root, *spec["path"][:-1],
                             spec["path"][-1].format(s=s)) for s in years}
    sigs, band_idx = {}, None
    for s, p in paths.items():
        try:
            sigs[s], band_idx = check_source(p, spec)
        except ValueError as exc:
            raise SystemExit(f"[LOI] {exc}")
    if len(set(sigs.values())) != 1:
        raise SystemExit(f"[LOI] {key}: luoi pixel khac nhau giua cac mua")
    sig = sigs[years[0]]
    values, crs, transform = {}, None, None
    for s, p in paths.items():
        values[s], crs, transform = read_values(p, band_idx)
    src_sha = {os.path.basename(p): file_sha256(p) for p in paths.values()}
    wpath, winfo = weights_for(a, spec["source"], sig, crs, transform)
    import rasterio
    with rasterio.open(wpath) as ds:
        w = ds.read(1).astype(np.float64)
    n_change, n_always = nan_pattern_changes(values, w, spec["nan_seasons"])
    log(f"  pixel w > 0: NaN o MOI mua {n_always}, NaN chi mot so mua {n_change}")
    if spec["same_nan_all_seasons"] and n_change:
        raise SystemExit(f"[LOI] {key}: {n_change} pixel co trong so thieu chi o mot so mua (quy tac E4)")
    for s in spec["nan_seasons"]:
        if s in values:
            log(f"  mua {s}: NaN toan bo theo quy tac chot truoc (E4 phuong an (b))")

    bands, names, seasons = stack_seasons(values, spec["nan_seasons"])
    stack = os.path.join(a.work_dir, f"stack_{key}.tif")
    write_raster(stack, bands, crs, transform, names)
    del bands
    want = {"rules_version": MET_RULES_VERSION, "target": col, "sources_sha256": src_sha,
            "weights_sha256": file_sha256(wpath), "seasons": years}
    common = dict(unit=spec["unit"], variable=key, value_band=spec["band"], weights=os.path.basename(wpath),
                  weights_info=winfo, nan_seasons=list(spec["nan_seasons"]), scope_sha256=file_sha256(a.scope),
                  rule="TB co trong so pham vi (exactextract weighted_mean, trong so = ty le dien tich scope v3 cua "
                       "pixel nguon); da giac o chieu sang CRS nguon, khong lay mau lai raster; o khong co dien tich "
                       "pham vi tren pixel hop le -> NaN", **want)

    # ---- diem
    pp = os.path.join(a.out_dir, f"points_reference_{col}.csv")
    pt_rows = []
    pt_want = {**want, "points_sha256": file_sha256(a.points), "points_ref_sal_sha256": file_sha256(a.points_ref_sal)}
    xy = pts.to_crs(crs)
    pv = point_pixel_values(values, transform, xy.geometry.x.to_numpy(), xy.geometry.y.to_numpy(),
                            spec["nan_seasons"])
    pref = pd.DataFrame({"point_id": pts["point_id"].to_numpy()[pv["idx"]],
                         "block_id": pts["block_id"].astype(str).to_numpy()[pv["idx"]],
                         "is_holdout": pts["is_holdout"].astype(bool).to_numpy()[pv["idx"]],
                         "season": pv["season"].to_numpy(), f"ref_{col}": pv["value"].to_numpy(),
                         "src_px_valid": np.isfinite(pv["value"].to_numpy()),
                         "src_row": pv["src_row"].to_numpy(), "src_col": pv["src_col"].to_numpy()})
    pref = pref.sort_values(["season", "point_id"], kind="stable").reset_index(drop=True)
    if pref.duplicated(["point_id", "season"]).any():
        raise SystemExit("[LOI] diem: khoa (point_id, season) trung")
    for s, g in pref.groupby("season"):
        nan_ids = g.loc[~g["src_px_valid"], "point_id"]
        pt_rows.append({"variable": key, "column": col, "season": int(s), "n_points": len(g),
                        "n_points_nan": int(len(nan_ids)), "n_points_ok": int(g["src_px_valid"].sum())})
    if a.force or not outputs_current(pp, pt_want):
        pref.to_csv(pp + ".part", index=False, float_format="%.6g")
        os.replace(pp + ".part", pp)
        nan_all = pref.groupby("point_id")["src_px_valid"].any()
        write_provenance(pp, a.boundary, variant="main", ref_rule_kind="pixel",
                         ref_rule="gia tri pixel nguon CHUA diem (khong 3x3, khong noi suy); pixel khong hop le -> NaN",
                         points=os.path.basename(a.points), n_points=int(len(pts)),
                         n_points_nan_all_seasons=int((~nan_all).sum()),
                         points_nan_all_seasons=sorted(nan_all[~nan_all].index.tolist()),
                         **{**common, **pt_want})
        log(f"  diem: ghi {pp} ({len(pref)} dong; NaN moi mua: {int((~nan_all).sum())} diem)")
    else:
        log(f"  diem: {pp} da co - bo qua")

    # ---- o theo luoi
    import geopandas as gpd

    grid_rows = []
    for gpath in grids:
        name = os.path.splitext(os.path.basename(gpath))[0]
        out = os.path.join(a.out_dir, f"{name}_labels_season_{col}.csv")
        g_want = {**want, "grid": os.path.basename(gpath), "grid_sha256": file_sha256(gpath)}
        if not a.force and outputs_current(out, g_want):
            log(f"  {name}: da co - bo qua")
            tab = pd.read_csv(out, dtype={"cell_id": str})
        else:
            t = time.time()
            grid = gpd.read_file(gpath)
            grid["cell_id"] = grid["cell_id"].astype(str)
            if grid["cell_id"].duplicated().any():
                raise SystemExit(f"[LOI] {name}: cell_id trung")
            tab = cell_weighted_labels(stack, wpath, (grid["cell_id"].tolist(), None, grid.geometry.tolist()),
                                       seasons, col)
            if len(tab) != len(grid) * len(seasons):
                raise SystemExit(f"[LOI] {name}: {len(tab)} dong != {len(grid)} o x {len(seasons)} mua")
            tab.to_csv(out + ".part", index=False, float_format="%.6g")
            os.replace(out + ".part", out)
            write_provenance(out, a.boundary, **{**common, **g_want})
            log(f"  {name}: {len(grid)} o x {len(seasons)} mua ({time.time() - t:.0f}s)")
        for s, g in tab.groupby("season"):
            fin = g[col].notna()
            grid_rows.append({"variable": key, "column": col, "grid": name, "season": int(s), "n_cells": len(g),
                              "n_cells_label": int(fin.sum()), "n_cells_nan": int((~fin).sum()),
                              "n_cells_partial": int((fin & (g["lbl_cover_frac"] < 1 - 1e-9)).sum()),
                              "value_min": float(g[col].min()), "value_median": float(g[col].median()),
                              "value_max": float(g[col].max())})
    os.remove(stack)
    w_row = {"variable": key, "source": spec["source"], "n_px_w_gt0_nan_all": n_always,
             "n_px_w_gt0_nan_some": n_change, **winfo}
    return grid_rows, pt_rows, w_row


def main(a):
    t0 = time.time()
    years = list(range(a.years[0], a.years[1] + 1))
    for d in (a.out_dir, a.work_dir, a.report_dir):
        os.makedirs(d, exist_ok=True)
    free = shutil.disk_usage(a.out_dir).free / 1e9
    log(f"Cho trong o dich {a.out_dir}: {free:.1f} GB")
    if free < 1.0:
        raise SystemExit("[LOI] o dich con < 1 GB")
    grids = sorted(glob.glob(os.path.join(a.grids_dir, "*.geojson")))
    if a.grids:
        grids = [os.path.join(a.grids_dir, f"{g}.geojson") for g in a.grids]
    if not grids or any(not os.path.exists(g) for g in grids):
        raise SystemExit(f"[LOI] luoi khong ton tai: {grids}")
    pts = load_points(a)
    log(f"Diem cham duoc: {len(pts):,}")
    grid_rows, pt_rows, w_rows = [], [], []
    for key in a.targets:
        g, p, w = run_target(a, key, years, grids, pts)
        grid_rows += g
        pt_rows += p
        w_rows.append(w)
    rg, rp, rw = pd.DataFrame(grid_rows), pd.DataFrame(pt_rows), pd.DataFrame(w_rows)
    tag = "" if a.targets == list(MET_TARGETS) else "_" + "_".join(a.targets)
    for df, nm in ((rg, "luoi"), (rp, "diem"), (rw, "trong_so")):
        p = os.path.join(a.report_dir, f"dot7_e5_khi_tuong_{nm}{tag}.csv")
        df.to_csv(p, index=False, float_format="%.6g")
        write_provenance(p, a.boundary, rules_version=MET_RULES_VERSION, targets=a.targets, seasons=years)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        piv = rg.groupby(["variable", "grid"]).agg(n_cells=("n_cells", "first"), nan_min=("n_cells_nan", "min"),
                                                   nan_max=("n_cells_nan", "max"),
                                                   partial_max=("n_cells_partial", "max"),
                                                   vmin=("value_min", "min"), vmax=("value_max", "max"))
        print(piv.to_string(), flush=True)
        print(rp.groupby("variable").agg(n_points=("n_points", "first"), nan_min=("n_points_nan", "min"),
                                         nan_max=("n_points_nan", "max")).to_string(), flush=True)
    log(f"Xong trong {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", nargs="+", choices=list(MET_TARGETS), default=list(MET_TARGETS))
    ap.add_argument("--data-root", default=None, help="Goc tim raster nguon (mac dinh DATA_ROOT trong .env)")
    ap.add_argument("--scope", default=f"{DATA}/features/scope_mask_v3.tif")
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--points", default=os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    ap.add_argument("--points-ref-sal", default=f"{DATA}/labels/points_reference.csv",
                    help="Dap an do man bo chinh - xac dinh diem cham duoc (ref_salinity huu han >= 1 mua)")
    ap.add_argument("--n-points-expected", type=int, default=10454, help="0 = khong kiem")
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--grids", nargs="*")
    ap.add_argument("--years", nargs=2, type=int, default=[2014, 2026], metavar=("TU", "DEN"))
    ap.add_argument("--out-dir", default=f"{DATA}/labels/dot7")
    ap.add_argument("--work-dir", default=os.path.join(tempfile.gettempdir(), "mekong_met_labels"))
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())

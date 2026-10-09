#!/usr/bin/env python
"""Nhan (o, mua) cho 13 luoi + dap an tai diem cua 4 bien khi tuong Dot 7 (mua CHIRPS, buc xa MCD18, T2m va RH
ERA5 goc) - quy tac o src/met_cell_labels.py. Ghi <DATA_ROOT>/labels/dot7 + bao cao KE_HOACH/ket-qua/dot7_e5_khi_tuong_*.
Chay:  venv/Scripts/python.exe scripts/build_met_labels.py [--targets rain dsr temp rh] [--grids ...] [--force]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from met_cell_labels import (MET_RULES_VERSION, MET_TARGETS, cell_count_labels, check_source,  # noqa: E402
                             nan_pattern_changes, point_pixel_values, scope_source_counts)
from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402

DATA = data_path()
RULE = ("giong dac trung tinh: moi pixel pham vi 30 m (scope v3) gan o theo TAM pixel, nhan gia tri pixel nguon "
        "CHUA tam; nhan o = TB tren pixel pham vi co gia tri hop le (ma tran dem o x pixel nguon); o khong pham vi "
        "-> NaN; o co pham vi chi tren pixel nguon NaN -> NaN (khong muon lang gieng); khong lay mau lai raster")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_points(a):
    """Diem cham duoc = eval_points co ref_salinity huu han o >= 1 mua."""
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
    """Gia tri band (float64); NoData khac NaN -> NaN."""
    import rasterio

    with rasterio.open(path) as ds:
        v = ds.read(band_idx).astype(np.float64)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            v[v == ds.nodata] = np.nan
        return v, ds.crs, ds.transform


def outputs_current(path, want):
    pp = path + ".provenance.json"
    if not (os.path.exists(path) and os.path.exists(pp)):
        return False
    info = read_json(pp)
    return all(info.get(k) == v for k, v in want.items())


def load_target(a, key, years):
    """Kiem + doc raster nguon moi mua (cung luoi pixel)."""
    spec = MET_TARGETS[key]
    root = DATA if a.data_root is None else a.data_root
    paths = {s: os.path.join(root, *spec["path"][:-1], spec["path"][-1].format(s=s)) for s in years}
    sigs, band_idx = {}, None
    for s, p in paths.items():
        try:
            sigs[s], band_idx = check_source(p, spec)
        except ValueError as exc:
            raise SystemExit(f"[LOI] {exc}")
    if len(set(sigs.values())) != 1:
        raise SystemExit(f"[LOI] {key}: luoi pixel khac nhau giua cac mua")
    values, crs, transform = {}, None, None
    for s, p in paths.items():
        values[s], crs, transform = read_values(p, band_idx)
    sig = sigs[years[0]]
    return {"key": key, "spec": spec, "values": values, "crs": crs, "transform": transform,
            "shape": (sig[3], sig[2]), "sig": sig,
            "src_sha": {os.path.basename(p): file_sha256(p) for p in paths.values()}}


def write_points(a, t, pts, years, scope_sha):
    key, spec = t["key"], t["spec"]
    col = spec["col"]
    pp = os.path.join(a.out_dir, f"points_reference_{col}.csv")
    want = {"rules_version": MET_RULES_VERSION, "target": col, "sources_sha256": t["src_sha"], "seasons": years,
            "points_sha256": file_sha256(a.points), "points_ref_sal_sha256": file_sha256(a.points_ref_sal)}
    xy = pts.to_crs(t["crs"])
    pv = point_pixel_values(t["values"], t["transform"], xy.geometry.x.to_numpy(), xy.geometry.y.to_numpy(),
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
    rows = []
    for s, g in pref.groupby("season"):
        rows.append({"variable": key, "column": col, "season": int(s), "n_points": len(g),
                     "n_points_nan": int((~g["src_px_valid"]).sum()), "n_points_ok": int(g["src_px_valid"].sum())})
    if a.force or not outputs_current(pp, want):
        pref.to_csv(pp + ".part", index=False, float_format="%.6g")
        os.replace(pp + ".part", pp)
        nan_all = pref.groupby("point_id")["src_px_valid"].any()
        write_provenance(pp, a.boundary, variant="main", ref_rule_kind="pixel",
                         ref_rule="gia tri pixel nguon CHUA diem (khong 3x3, khong noi suy); pixel khong hop le -> NaN",
                         points=os.path.basename(a.points), n_points=int(len(pts)),
                         n_points_nan_all_seasons=int((~nan_all).sum()),
                         points_nan_all_seasons=sorted(nan_all[~nan_all].index.tolist()),
                         unit=spec["unit"], variable=key, value_band=spec["band"],
                         nan_seasons=list(spec["nan_seasons"]), scope_sha256=scope_sha, **want)
        log(f"  {key} diem: ghi {pp} ({len(pref)} dong; NaN moi mua: {int((~nan_all).sum())} diem)")
    else:
        log(f"  {key} diem: {pp} da co - bo qua")
    return rows


def grid_report_rows(t, name, tab):
    spec = t["spec"]
    col = spec["col"]
    rows = []
    for s, g in tab.groupby("season"):
        fin = g[col].notna()
        no_scope = g["lbl_cover_frac"].isna()
        rows.append({"variable": t["key"], "column": col, "grid": name, "season": int(s), "n_cells": len(g),
                     "n_cells_label": int(fin.sum()), "n_cells_nan": int((~fin).sum()),
                     "n_cells_no_scope": int(no_scope.sum()),
                     "n_cells_scope_all_invalid": int((~fin & ~no_scope).sum()),
                     "n_cells_partial": int((fin & (g["lbl_cover_frac"] < 1 - 1e-12)).sum()),
                     "value_min": float(g[col].min()), "value_median": float(g[col].median()),
                     "value_max": float(g[col].max())})
    return rows


def main(a):
    t0 = time.time()
    years = list(range(a.years[0], a.years[1] + 1))
    for d in (a.out_dir, a.report_dir):
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
    scope_sha = file_sha256(a.scope)

    targets = {}
    for key in a.targets:
        targets[key] = load_target(a, key, years)
        log(f"{key}: {len(years)} mua, luoi nguon {targets[key]['shape']} ({targets[key]['spec']['source']})")
    sources = {}
    for t in targets.values():
        src = t["spec"]["source"]
        if src in sources and sources[src]["sig"] != t["sig"]:
            raise SystemExit(f"[LOI] nguon {src}: luoi pixel khac nhau giua cac bien")
        sources.setdefault(src, t)

    pt_rows = []
    for t in targets.values():
        pt_rows += write_points(a, t, pts, years, scope_sha)

    import geopandas as gpd

    grid_rows = []
    used_px = {src: set() for src in sources}   # pixel nguon co pham vi, hop cac luoi da tinh
    for gpath in grids:
        name = os.path.splitext(os.path.basename(gpath))[0]
        g_sha = file_sha256(gpath)
        todo = {}
        for key, t in targets.items():
            col = t["spec"]["col"]
            out = os.path.join(a.out_dir, f"{name}_labels_season_{col}.csv")
            want = {"rules_version": MET_RULES_VERSION, "target": col, "sources_sha256": t["src_sha"],
                    "seasons": years, "scope_sha256": scope_sha, "grid": os.path.basename(gpath), "grid_sha256": g_sha}
            if not a.force and outputs_current(out, want):
                log(f"  {name}/{key}: da co - bo qua")
                grid_rows += grid_report_rows(t, name, pd.read_csv(out, dtype={"cell_id": str}))
            else:
                todo[key] = (out, want)
        if not todo:
            continue
        tg = time.time()
        grid = gpd.read_file(gpath)
        grid["cell_id"] = grid["cell_id"].astype(str)
        if grid["cell_id"].duplicated().any():
            raise SystemExit(f"[LOI] {name}: cell_id trung")
        need_src = {targets[k]["spec"]["source"] for k in todo}
        src_args = {s: (sources[s]["crs"], sources[s]["transform"], sources[s]["shape"]) for s in need_src}
        cell_ids, counts, cinfo = scope_source_counts(a.scope, (grid["cell_id"].tolist(), None,
                                                                 grid.geometry.tolist()), src_args)
        for s in need_src:
            if cinfo[s]["n_scope_outside"]:
                raise SystemExit(f"[LOI] {name}: {cinfo[s]['n_scope_outside']} pixel pham vi trong o nam ngoai "
                                 f"luoi nguon {s}")
            used_px[s].update(counts[s]["px"].tolist())
        n_scope = next(iter(cinfo.values()))["n_scope_in_cells"]
        log(f"  {name}: ma tran dem {len(grid)} o, {n_scope:,} pixel pham vi trong o, "
            + ", ".join(f"{s} {len(counts[s])} cap (o, pixel)" for s in sorted(need_src))
            + f" ({time.time() - tg:.0f}s)")
        for key, (out, want) in todo.items():
            t = targets[key]
            spec, col = t["spec"], t["spec"]["col"]
            h, w = t["shape"]
            cnt = counts[spec["source"]]
            wpx = np.bincount(cnt["px"].to_numpy(), weights=cnt["n"].to_numpy(), minlength=h * w).reshape(h, w)
            n_change, n_always = nan_pattern_changes(t["values"], wpx, spec["nan_seasons"])
            if spec["same_nan_all_seasons"] and n_change:
                raise SystemExit(f"[LOI] {name}/{key}: {n_change} pixel nguon co pham vi thieu chi o mot so mua "
                                 f"(quy tac E4)")
            tab = cell_count_labels(cnt, cell_ids, t["values"], col, spec["nan_seasons"])
            if len(tab) != len(grid) * len(years):
                raise SystemExit(f"[LOI] {name}: {len(tab)} dong != {len(grid)} o x {len(years)} mua")
            tab.to_csv(out + ".part", index=False, float_format="%.6g")
            os.replace(out + ".part", out)
            write_provenance(out, a.boundary, unit=spec["unit"], variable=key, value_band=spec["band"],
                             nan_seasons=list(spec["nan_seasons"]), scope=os.path.basename(a.scope), rule=RULE,
                             n_scope_px_in_cells=n_scope, n_src_px_with_scope=int((wpx > 0).sum()),
                             n_src_px_with_scope_nan_all=n_always, n_src_px_with_scope_nan_some=n_change, **want)
            rows = grid_report_rows(t, name, tab)
            grid_rows += rows
            r0 = rows[0]
            log(f"  {name}/{key}: {len(grid)} o x {len(years)} mua; mua {r0['season']}: NaN {r0['n_cells_nan']} "
                f"(khong pham vi {r0['n_cells_no_scope']}, pham vi chi tren pixel NaN "
                f"{r0['n_cells_scope_all_invalid']})")
        del counts

    src_rows = []
    for key, t in targets.items():
        h, w = t["shape"]
        px = np.fromiter(used_px[t["spec"]["source"]], dtype=np.int64)
        wpx = np.zeros(h * w)
        wpx[px] = 1
        n_change, n_always = nan_pattern_changes(t["values"], wpx.reshape(h, w), t["spec"]["nan_seasons"])
        src_rows.append({"variable": key, "source": t["spec"]["source"], "n_src_px_with_scope": int(px.size),
                         "n_src_px_nan_all": n_always, "n_src_px_nan_some": n_change,
                         "note": "" if px.size else "khong tinh (moi luoi da co - bo qua)"})
    rg, rp, rs = pd.DataFrame(grid_rows), pd.DataFrame(pt_rows), pd.DataFrame(src_rows)
    tag = "" if a.targets == list(MET_TARGETS) else "_" + "_".join(a.targets)
    for df, nm in ((rg, "luoi"), (rp, "diem"), (rs, "nguon")):
        p = os.path.join(a.report_dir, f"dot7_e5_khi_tuong_{nm}{tag}.csv")
        df.to_csv(p, index=False, float_format="%.6g")
        write_provenance(p, a.boundary, rules_version=MET_RULES_VERSION, targets=a.targets, seasons=years)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        piv = rg.groupby(["variable", "grid"]).agg(n_cells=("n_cells", "first"), nan_min=("n_cells_nan", "min"),
                                                   nan_max=("n_cells_nan", "max"),
                                                   no_scope=("n_cells_no_scope", "max"),
                                                   all_invalid_max=("n_cells_scope_all_invalid", "max"),
                                                   partial_max=("n_cells_partial", "max"),
                                                   vmin=("value_min", "min"), vmax=("value_max", "max"))
        print(piv.to_string(), flush=True)
        print(rp.groupby("variable").agg(n_points=("n_points", "first"), nan_min=("n_points_nan", "min"),
                                         nan_max=("n_points_nan", "max")).to_string(), flush=True)
        print(rs.to_string(index=False), flush=True)
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
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())

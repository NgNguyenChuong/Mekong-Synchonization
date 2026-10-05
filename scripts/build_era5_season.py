#!/usr/bin/env python
"""Dac trung ERA5-Land theo MUA KHO 2014-2026 cho 13 luoi (P3).

Mua kho Y = 01/11/(Y-1) .. 29/04/Y (src/seasons.py; thang 4 chi lay band 1..29). Mua 2014 can 11-12/2013.
Gop (src/era5_season.py): mua = TONG (cat am ve 0 truoc), bien khac = TRUNG BINH; pixel thieu >= 1 ngay
-> NaN (khong lap); pixel ngoai mat na dat temp_avg -> NaN moi bien.

Dau ra (trong --out-dir, mac dinh <DATA_ROOT>/features/era5_season):
  <bien>_<mua>.tif           : band 1 = gia tri mua, band 2 = n_days (so ngay co gia tri)
  era5_valid_<mua>.tif       : 1 = pixel dat va du ngay o MOI bien, 0 = khong
  <luoi>_era5_season.csv     : cell_id, season, rain_mm, solar, temp_c, temp_max_c, temp_min_c, rh_percent,
                               era5_cover_frac (ty le dien tich o co pixel ERA5 hop le o moi bien; 0 = toan bien).
                               Trung binh co trong so dien tich (processing.area_weighted_means, exactextract).
                               O khong co pixel hop le -> NaN.
  -- Ban LAP (cau K, An duyet 2026-10-03), song song ban goc (khong lap) o tren --
  <bien>_<mua>_filled.tif    : nhu <bien>_<mua>.tif nhung pixel KHONG hop le GIAO ranh gioi (--boundary, v2)
                               duoc lap bang pixel hop le trong 1 pixel (8 lang gieng): co lang gieng canh ->
                               trung binh cac canh hop le; khong co -> trung binh cac cheo hop le; khong co ->
                               NaN (era5_season.fill_nearest_1px). Band 2 = n_days goc (pixel lap = 0).
  era5_filled_<mua>.tif      : band 1 = 1 pixel duoc lap / 0; band 2 = so pixel nguon (1..4).
  <luoi>_era5_season_filled.csv : cot nhu ban goc + era5_fill_frac (ty le dien tich o tren pixel LAP);
                               era5_cover_frac giu nghia goc (pixel hop le that) -> cover + fill = ty le co gia tri.
  moi file kem .provenance.json (sha256 ranh gioi + luoi / file ERA5 nguon).

Chay:  python scripts/build_era5_season.py [--raw-dir <DATA_ROOT>/raw] [--grids-dir data/grids]
           [--out-dir <DATA_ROOT>/features/era5_season] [--years 2014 2026] [--grids h3_res_5 ...]
           [--skip-existing] [--report KE_HOACH/ket-qua/dot4_era5_season.csv]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from era5_season import (COVER_COL, ERA5_SEASON_SPECS, FILL_COL, boundary_touch_mask,  # noqa: E402
                         fill_nearest_1px, index_monthly_files, season_layers, season_months, season_table,
                         write_season_raster)
from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402

VALUE_COLS = [s[1] for s in ERA5_SEASON_SPECS.values()]
FILLED = "@filled"   # hau to cot tam cho ban lap trong by_season
FILL_RULE = ("lap pixel khong hop le giao ranh gioi bang pixel hop le trong 1 pixel (8 lang gieng): "
             "lang gieng canh hop le -> trung binh canh; khong co -> trung binh cheo; khong co -> NaN; "
             "nguon = pixel hop le goc (khong lan truyen)")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_rasters(a, years, file_maps):
    """Tinh raster mua cho moi nam; ghi ra dia; tra ve {mua: {cot: mang}} va profile."""
    by_season, profile, diag_rows, target = {}, None, [], None
    for y in years:
        layers, counts, valid_all, prof, diag = season_layers(a.raw_dir, y, file_maps=file_maps)
        if profile is None:
            profile = prof
            target = boundary_touch_mask(a.boundary, prof)
            log(f"Pixel ERA5 giao ranh gioi: {int(target.sum())}; transform {tuple(prof['transform'])[:6]}")
        filled_layers, filled, n_src = fill_nearest_1px(layers, valid_all, target)
        by_season[y] = {ERA5_SEASON_SPECS[v][1]: layers[v] for v in ERA5_SEASON_SPECS}
        by_season[y][COVER_COL] = valid_all.astype("float32")
        by_season[y].update({ERA5_SEASON_SPECS[v][1] + FILLED: filled_layers[v] for v in ERA5_SEASON_SPECS})
        by_season[y][FILL_COL] = filled.astype("float32")
        for var, (_, col, how) in ERA5_SEASON_SPECS.items():
            fout = os.path.join(a.out_dir, f"{var}_{y}_filled.tif")
            write_season_raster(fout, filled_layers[var], counts[var], prof, f"{col}_{how}_filled")
            write_provenance(fout, a.boundary, season=y, variable=var, source_raster=f"{var}_{y}.tif",
                             fill_rule=FILL_RULE)
        fpath = os.path.join(a.out_dir, f"era5_filled_{y}.tif")
        write_season_raster(fpath, filled.astype("float32"), n_src, prof, "era5_filled", band2="n_source_px")
        write_provenance(fpath, a.boundary, season=y, fill_rule=FILL_RULE, px_filled=int(filled.sum()),
                         px_target_invalid=int((target & ~valid_all).sum()))
        for var, (folder, col, how) in ERA5_SEASON_SPECS.items():
            out = os.path.join(a.out_dir, f"{var}_{y}.tif")
            write_season_raster(out, layers[var], counts[var], prof, f"{col}_{how}")
            srcs = [file_maps[var][(p.year, p.month)] for p in season_months(y) if (p.year, p.month) in file_maps[var]]
            write_provenance(out, a.boundary, season=y, variable=var, column=col, aggregation=how,
                             clip_negative=var == "rain", land_mask="temp_avg",
                             sources={os.path.basename(s): file_sha256(s) for s in srcs}, **diag[var])
            d = diag[var]
            diag_rows.append({"season": y, "var": var, "missing_months": ",".join(d["missing_months"]),
                              "px_full": d["px_full"], "px_partial": d["px_partial"],
                              "px_outside_land": d["px_outside_land"],
                              "min": float(np.nanmin(layers[var])), "max": float(np.nanmax(layers[var]))})
        vpath = os.path.join(a.out_dir, f"era5_valid_{y}.tif")
        write_season_raster(vpath, valid_all.astype("float32"), counts["temp_avg"], prof, "valid_all_vars")
        write_provenance(vpath, a.boundary, season=y, rule="pixel dat (temp_avg) va du ngay o moi bien")
        log(f"Mua {y}: pixel hop le moi bien {int(valid_all.sum())}; lap {int(filled.sum())} "
            f"/ {int((target & ~valid_all).sum())} pixel khong hop le giao ranh gioi")
    return by_season, profile, pd.DataFrame(diag_rows)


def main(a):
    import geopandas as gpd

    t0 = time.time()
    years = list(range(a.years[0], a.years[1] + 1))
    os.makedirs(a.out_dir, exist_ok=True)
    file_maps = {v: index_monthly_files(os.path.join(a.raw_dir, s[0])) for v, s in ERA5_SEASON_SPECS.items()}
    by_season, profile, diag = build_rasters(a, years, file_maps)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        bad = diag[(diag["px_partial"] > 0) | (diag["px_outside_land"] > 0) | (diag["missing_months"] != "")]
        log(f"Raster mua: {len(diag)} (bien x mua); bat thuong (thieu thang / pixel thieu ngay / ngoai dat): {len(bad)}")
        if len(bad):
            print(bad.to_string(index=False), flush=True)
        print(diag.groupby("var")[["px_full", "min", "max"]].agg(["min", "max"]).to_string(), flush=True)

    grids = sorted(glob.glob(os.path.join(a.grids_dir, "*.geojson")))
    if a.grids:
        grids = [g for g in grids if os.path.splitext(os.path.basename(g))[0] in set(a.grids)]
    rows = []
    for path in grids:
        name = os.path.splitext(os.path.basename(path))[0]
        csv = os.path.join(a.out_dir, f"{name}_era5_season.csv")
        csv_f = os.path.join(a.out_dir, f"{name}_era5_season_filled.csv")
        done = all(os.path.exists(p) and os.path.exists(p + ".provenance.json") for p in (csv, csv_f))
        if a.skip_existing and done:
            log(f"{name}: da co, bo qua")
            df = pd.read_csv(csv, dtype={"cell_id": str})
            dff = pd.read_csv(csv_f, dtype={"cell_id": str})
        else:
            t = time.time()
            grid = gpd.read_file(path)
            cells = (grid["cell_id"].astype(str).tolist(), None, grid.geometry.tolist())
            wide = season_table(by_season, profile, cells)
            df = wide[["cell_id", "season", *VALUE_COLS, COVER_COL]]
            dff = wide[["cell_id", "season", *[c + FILLED for c in VALUE_COLS], COVER_COL, FILL_COL]]
            dff = dff.rename(columns={c + FILLED: c for c in VALUE_COLS})
            common = dict(grid=os.path.basename(path), grid_sha256=file_sha256(path), seasons=years,
                          raster_dir=os.path.abspath(a.out_dir).replace("\\", "/"),
                          method="exactextract mean (processing.area_weighted_means), raster dem 5 px",
                          aggregation={c: h for _, c, h in ERA5_SEASON_SPECS.values()})
            filled_extra = {"fill_rule": FILL_RULE, "rasters": "<bien>_<mua>_filled.tif",
                            "fill_col": f"{FILL_COL} = ty le dien tich o tren pixel lap"}
            for out_df, out_csv, extra in ((df, csv, {}), (dff, csv_f, filled_extra)):
                out_df.to_csv(out_csv + ".part", index=False, float_format="%.6g")
                os.replace(out_csv + ".part", out_csv)
                write_provenance(out_csv, a.boundary, **common, **extra)
            log(f"{name}: {len(grid)} o -> {len(df)} dong ({time.time() - t:.1f}s)")
        n_cells = df["cell_id"].nunique()
        assert len(df) == n_cells * len(years), f"{name}: so dong {len(df)} != {n_cells} x {len(years)}"
        assert len(dff) == len(df), f"{name}: ban lap {len(dff)} dong != {len(df)}"
        nan_cells = df.loc[df[VALUE_COLS].isna().any(axis=1), "cell_id"].unique()
        nan_cells_f = dff.loc[dff[VALUE_COLS].isna().any(axis=1), "cell_id"].unique()
        rows.append({"grid": name, "cells": n_cells, "rows": len(df),
                     "nan_rows": int(df[VALUE_COLS].isna().any(axis=1).sum()), "nan_cells": len(nan_cells),
                     "nan_cells_filled": len(nan_cells_f),
                     "fill_gt0_cells": int(dff.loc[dff[FILL_COL] > 0, "cell_id"].nunique()),
                     "cover_lt1_cells": int(df.loc[df[COVER_COL] < 1, "cell_id"].nunique()),
                     **{f"{c}_median": round(float(df[c].median()), 3) for c in VALUE_COLS}})
    report = pd.DataFrame(rows)
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(report.to_string(index=False), flush=True)
    if a.report:
        report.to_csv(a.report, index=False)
    log(f"Xong trong {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default=os.environ.get("RAW_DIR", data_path("raw")))
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--out-dir", default=data_path("features/era5_season"))
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--years", nargs=2, type=int, default=[2014, 2026], metavar=("TU", "DEN"))
    ap.add_argument("--grids", nargs="*", help="Ten luoi (khong .geojson); mac dinh moi luoi trong --grids-dir")
    ap.add_argument("--skip-existing", action="store_true", help="Bo qua luoi da co CSV + provenance")
    ap.add_argument("--report", help="Ghi bang tom tat theo luoi ra CSV")
    main(ap.parse_args())

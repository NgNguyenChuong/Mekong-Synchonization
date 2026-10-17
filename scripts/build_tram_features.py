#!/usr/bin/env python
"""Dac trung tram MRC theo o x mua (CHG-29): p20 muc nuoc giua, bien do trieu TB, p90 dinh trieu cua 6 tram, noi suy doc
do thi song (river_graph.station_weights) roi TB co trong so dien tich tren pixel scope; ban sao unified_bmua/<bien> + 4 cot
-> <DATA_ROOT>/features/unified_tram/<bien>/; trung gian (thong ke tram, stack 30 m, bang luoi) o <DATA_ROOT>/features/tram/.
Chay:  venv/Scripts/python.exe scripts/build_tram_features.py [--targets salinity ndwi] [--grids h3_res_7 ...]
"""
import argparse
import datetime
import glob
import hashlib
import inspect
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fetch_mrc_timeseries import STATIONS  # noqa: E402
from hydro_season import season_stats  # noqa: E402
from training.dot7_rules import file_sha256, provenance_path, write_provenance  # noqa: E402

TARGETS = ("salinity", "ndwi")
YEARS = list(range(2014, 2027))
NEW_COLS = ("tram_wl_p20", "tram_tide_amp", "tram_wl_p90", "tram_dist_km")
STAT_OF = {"tram_wl_p20": "mid_p20", "tram_tide_amp": "rng_mean", "tram_wl_p90": "max_p90"}
MIN_READS = 4    # ngay < 4 lan doc: max - min khong phai bien do trieu (1 lan doc -> 0)
OK_FRAC = 0.7    # mua can >= 70% ngay hop le
SNAP_MAX_M = 3000.0
KIND = {0: "noi_suy", 1: "gan_nhat_doc_song", 2: "gan_nhat_duong_thang"}
FORMULA = {"tram_wl_p20": "p20 cua (max + min)/2 theo ngay", "tram_tide_amp": "TB cua (max - min) theo ngay",
           "tram_wl_p90": "p90 cua max ngay", "tram_dist_km": "khoang cach doc song (km) dinh -> tram gan nhat dung "
           "(gan_nhat_duong_thang: duong thang), TB dien tich",
           "o": "sum_s w_s * X_s; w = TB dien tich pixel scope cua trong so dinh song gan nhat (river_graph.station_weights)"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def station_season_stats(df, min_reads=MIN_READS, ok_frac=OK_FRAC, years=YEARS) -> pd.DataFrame:
    """df (datetime +07:00 chuoi, value m) -> moi mua: mid_p20, rng_mean, max_p90, n_days (ngay >= min_reads lan doc)."""
    day = pd.to_datetime(df["datetime"].str[:10])
    g = df.assign(day=day).groupby("day")["value"].agg(["max", "min", "count"])
    g = g[g["count"] >= min_reads]
    dates = g.index
    parts = [season_stats(dates, (g["max"] + g["min"]) / 2, years, "mid", ok_frac, {"p20": 0.20}, nan_if_not_ok=True),
             season_stats(dates, g["max"] - g["min"], years, "rng", ok_frac, nan_if_not_ok=True),
             season_stats(dates, g["max"], years, "max", ok_frac, {"p90": 0.90}, nan_if_not_ok=True)]
    out = parts[0][["season", "mid_p20", "mid_n_days", "mid_ok"]]
    out = out.merge(parts[1][["season", "rng_mean"]], on="season").merge(parts[2][["season", "max_p90"]], on="season")
    return out.rename(columns={"mid_n_days": "n_days", "mid_ok": "ok"})


def load_station_stats(a) -> pd.DataFrame:
    rows = []
    for key in STATIONS:
        csv = os.path.join(a.mrc_dir, f"{key}.csv")
        if not (os.path.exists(csv) and os.path.exists(provenance_path(csv))):
            raise SystemExit(f"[LOI] thieu {csv} hoac provenance (chay fetch_mrc_timeseries.py)")
        st = station_season_stats(pd.read_csv(csv), a.min_reads, a.ok_frac)
        rows.append(st.assign(station=key, csv_sha256=file_sha256(csv)))
    st = pd.concat(rows, ignore_index=True)
    bad = st[~st["ok"] | st[list(STAT_OF.values())].isna().any(axis=1)]
    if len(bad):
        raise SystemExit(f"[LOI] tram x mua khong du ngay hop le:\n{bad[['station', 'season', 'n_days']].to_string()}")
    return st


def station_matrix(st, col) -> pd.DataFrame:
    """mua x tram (thu tu STATIONS)."""
    return st.pivot(index="season", columns="station", values=col)[list(STATIONS)]


def add_tram_columns(df, wcell, st) -> pd.DataFrame:
    """Ban sao df + 4 cot. wcell: cell_id, w_<tram>..., tram_dist_km. O scope_frac <= 0 -> NaN (nhu dist_mouth_river_km)."""
    have = [c for c in NEW_COLS if c in df.columns]
    if have:
        raise ValueError(f"bang da co cot {have}")
    wcols = [f"w_{k}" for k in STATIONS]
    m = df[["cell_id", "season", "scope_frac"]].merge(wcell, on="cell_id", how="left", validate="many_to_one")
    w = m[wcols].to_numpy(float)
    empty = ~(m["scope_frac"].to_numpy(float) > 0)
    out = df.copy()
    for c, s in STAT_OF.items():
        x = station_matrix(st, s).reindex(m["season"].astype(int)).to_numpy(float)  # (dong, tram)
        v = (w * x).sum(axis=1)
        v[empty | np.isnan(w).any(axis=1)] = np.nan
        out[c] = v
    out["tram_dist_km"] = np.where(empty, np.nan, m["tram_dist_km"].to_numpy(float))
    check_tram(out, st)
    return out


def check_tram(df, st):
    if df.duplicated(["cell_id", "season"]).any():
        raise ValueError("trung (cell_id, season)")
    scope = df["scope_frac"] > 0
    nan = df.loc[scope, list(NEW_COLS)].isna().sum()
    if nan.any():
        raise ValueError(f"NaN o o co pham vi: {nan[nan > 0].to_dict()}")
    for c, s in STAT_OF.items():
        mat = station_matrix(st, s)
        lo = df["season"].map(mat.min(axis=1)) - 1e-9
        hi = df["season"].map(mat.max(axis=1)) + 1e-9
        out = scope & ~df[c].between(lo, hi)
        if out.any():
            raise ValueError(f"{c}: {int(out.sum())} dong ngoai dai tram cung mua")


# ------------------------------------------------------------------ do thi + stack 30 m
def graph_weights(a):
    """Dung lai do thi CHG-16 (build_river_graph.build) -> dinh, trong so tram, khoang cach, loai, dinh 90 m."""
    import rasterio

    import build_river_graph as brg
    import river_graph as rg
    from scipy.spatial import cKDTree

    _, _, g, mouths, _, _, _ = brg.build(argparse.Namespace(osm=a.osm, coast=a.coast))
    dist, lab = rg.mouth_distances(g, mouths)
    ok = np.nonzero(lab >= 0)[0]
    st_xy = brg.to_xy([(v[2], v[3]) for v in STATIONS.values()])
    snap_m, k = cKDTree(g.xy[ok]).query(st_xy)
    st_v = ok[k]
    snaps = {key: {"vertex": int(v), "snap_m": round(float(d), 1), "river": str(g.river[v]),
                   "dist_mouth_km": round(float(dist[v]) / 1000, 2)} for key, v, d in zip(STATIONS, st_v, snap_m)}
    log(f"Tram -> dinh song: {snaps}")
    if (snap_m > SNAP_MAX_M).any():
        raise SystemExit(f"[LOI] tram cach song > {SNAP_MAX_M} m")
    W, tdist, kind = rg.station_weights(g, dist, st_v, st_xy)
    with rasterio.open(a.graph_tif) as gr:
        tr, shape, crs = gr.transform, gr.shape, gr.crs
        ref = gr.read(list(gr.descriptions).index("dist_mouth_km") + 1)
    kk, _ = rg.snap_index(g.xy[ok], tr, shape)
    v90 = ok[kk]
    diff = np.abs((dist[v90] / 1000).astype("float32") - ref)
    if not np.nanmax(diff) < 1e-3:
        raise SystemExit(f"[LOI] do thi dung lai khac {a.graph_tif} (lech dist_mouth_km toi {np.nanmax(diff):.3f} km)")
    log(f"Do thi khop river_graph_90m.tif (lech max {np.nanmax(diff):.2e} km); dinh theo loai "
        f"{ {KIND[i]: int((kind[ok] == i).sum()) for i in KIND} }")
    return dict(W=W, tdist_km=tdist / 1000, kind=kind, v90=v90.astype("int32"), tr90=tr, crs=crs, snaps=snaps)


def write_stack(path, scope_path, gw, chunk_rows=512):
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.windows import Window

    from scope_mask import band_index, nearest_on_grid

    names = [f"w_{k}" for k in STATIONS] + ["tram_dist_km", "noi_suy", "duong_thang"]
    lut = np.column_stack([gw["W"], gw["tdist_km"], gw["kind"] == 0, gw["kind"] == 2]).astype("float32")
    v90 = gw["v90"]
    with MemoryFile() as mf, mf.open(driver="GTiff", width=v90.shape[1], height=v90.shape[0], count=1, dtype="int32",
                                     crs=gw["crs"], transform=gw["tr90"]) as vr, rasterio.open(scope_path) as sc:
        vr.write(v90, 1)
        sb, (h, w) = band_index(sc, "scope"), sc.shape
        prof = dict(driver="GTiff", width=w, height=h, count=len(names), dtype="float32", crs=sc.crs,
                    transform=sc.transform, nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                    blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        n_px = 0
        with rasterio.open(path + ".part", "w", **prof) as out:
            for i, nm in enumerate(names, start=1):
                out.set_band_description(i, nm)
            for r0 in range(0, h, chunk_rows):
                n = min(chunk_rows, h - r0)
                win = Window(0, r0, w, n)
                m = sc.read(sb, window=win) == 1
                v = nearest_on_grid(vr, 1, sc.transform, (n, w), r0, fill=-1)
                if (m & (v < 0)).any():
                    raise SystemExit("[LOI] pixel scope ngoai luoi 90 m")
                n_px += int(m.sum())
                for i in range(len(names)):
                    arr = np.full((n, w), np.nan, "float32")
                    arr[m] = lut[v[m], i]
                    out.write(arr, i + 1, window=win)
    os.replace(path + ".part", path)
    return names, n_px


def stack_key(a):
    import river_graph as rg

    return {"osm_sha256": file_sha256(a.osm), "coast_sha256": file_sha256(a.coast),
            "graph_tif_sha256": file_sha256(a.graph_tif), "scope_sha256": file_sha256(a.scope),
            "stations": {k: [v[2], v[3]] for k, v in STATIONS.items()}, "snap_max_m": SNAP_MAX_M,
            "station_weights_src_sha256": hashlib.sha256(inspect.getsource(rg.station_weights).encode()).hexdigest()}


def ensure_stack(a, path):
    key = stack_key(a)
    pp = provenance_path(path)
    if os.path.exists(path) and os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            p = json.load(f)
        if p.get("key") == key and p.get("csv_sha256") == file_sha256(path):
            log(f"[bo qua] {path}: da co, dau vao khong doi")
            return p
    gw = graph_weights(a)
    names, n_px = write_stack(path, a.scope, gw)
    write_provenance(path, key=key, bands=names, scope_px=n_px, station_vertices=gw["snaps"],
                     osm=a.osm, coast=a.coast, graph_tif=a.graph_tif, scope=a.scope, kind=KIND)
    log(f"Ghi {path}: {n_px} pixel scope")
    with open(pp, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ bang
def out_path(a, target, grid):
    return os.path.join(a.out_root, target, f"{grid}_unified.csv")


def up_to_date(out, src_sha, deps) -> bool:
    pp = provenance_path(out)
    if not (os.path.exists(out) and os.path.exists(pp)):
        return False
    with open(pp, encoding="utf-8") as f:
        t = json.load(f).get("tram", {})
    return (t.get("source", {}).get("sha256") == src_sha and t.get("deps") == deps
            and t.get("output_sha256") == file_sha256(out))


def cell_weights(stack, grid_path):
    import geopandas as gpd

    from processing import area_weighted_means

    grid = gpd.read_file(grid_path)
    w = area_weighted_means(stack, (grid["cell_id"].astype(str).tolist(), None, grid.geometry.tolist()))
    w.columns = ["cell_id"] + [f"w_{k}" for k in STATIONS] + ["tram_dist_km", "noi_suy", "duong_thang"]
    s = w[[f"w_{k}" for k in STATIONS]].sum(axis=1, min_count=1)  # o khong co pixel scope -> NaN, khong 0
    if (np.abs(s[s.notna()] - 1) > 1e-4).any():
        raise SystemExit(f"[LOI] {grid_path}: tong trong so tram != 1 (lech max {np.abs(s - 1).max():.2e})")
    return w


def build_one(a, target, grid, wcell, st, deps):
    src = os.path.join(a.src_root, target, f"{grid}_unified.csv")
    out = out_path(a, target, grid)
    df = pd.read_csv(src, dtype={"cell_id": str})
    try:
        res = add_tram_columns(df, wcell, st)
    except ValueError as exc:
        raise SystemExit(f"[LOI] {target}/{grid}: {exc}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res.to_csv(out + ".part", index=False)  # khong float_format: giu nguyen gia tri cot cu
    back = pd.read_csv(out + ".part", dtype={"cell_id": str})
    if not back[list(df.columns)].equals(df):
        raise SystemExit(f"[LOI] {target}/{grid}: cot cu doi gia tri sau khi ghi")
    os.replace(out + ".part", out)
    with open(provenance_path(src), encoding="utf-8") as f:
        prov = json.load(f)  # giu target / label_set / holdout_season_nan cho runner + train.py
    prov.update(output=os.path.basename(out), created=datetime.datetime.now().isoformat(timespec="seconds"),
                script=os.path.basename(__file__),
                tram={"chg": "CHG-29", "source": {"path": src.replace("\\", "/"), "sha256": file_sha256(src),
                                                   "provenance": provenance_path(src).replace("\\", "/")},
                      "deps": deps, "output_sha256": file_sha256(out), "new_cols": list(NEW_COLS),
                      "formula": FORMULA, "min_reads_per_day": a.min_reads, "ok_frac": a.ok_frac})
    with open(provenance_path(out), "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)


def summary_row(grid, wcell, scope_cells):
    w = wcell[wcell["cell_id"].isin(scope_cells)]
    noi, thang = w["noi_suy"] >= 0.5, w["duong_thang"] >= 0.5
    return {"grid": grid, "cells_scope": len(w), "o_noi_suy": int(noi.sum()), "o_gan_nhat_doc_song": int((~noi & ~thang).sum()),
            "o_gan_nhat_duong_thang": int(thang.sum()), "tram_dist_km_p50": round(float(w["tram_dist_km"].median()), 1),
            "tram_dist_km_max": round(float(w["tram_dist_km"].max()), 1)}


def main(a):
    grid_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(a.grids_dir, "*.geojson"))}
    grids = sorted(a.grids or grid_paths)
    miss = [g for g in grids if g not in grid_paths or not all(
        os.path.exists(os.path.join(a.src_root, t, f"{g}_unified.csv")) for t in a.targets)]
    if miss or (not a.grids and len(grids) != 13):
        raise SystemExit(f"[LOI] can 13 luoi co bang nguon; thieu {miss}, thay {len(grids)}")
    os.makedirs(a.work_dir, exist_ok=True)
    free = shutil.disk_usage(a.work_dir).free / 1e9
    log(f"Cho trong {a.work_dir}: {free:.1f} GB")
    if free < 1.5:
        raise SystemExit("[LOI] o dich con < 1,5 GB")
    st = load_station_stats(a)
    st_csv = os.path.join(a.work_dir, "tram_station_season.csv")
    st.drop(columns="csv_sha256").to_csv(st_csv, index=False)
    write_provenance(st_csv, mrc_dir=a.mrc_dir, csv_sha256_by_station=dict(zip(st["station"], st["csv_sha256"])),
                     min_reads_per_day=a.min_reads, ok_frac=a.ok_frac, formula=FORMULA)
    with pd.option_context("display.width", 200):
        for c in STAT_OF.values():
            log(f"{c} (m):\n{station_matrix(st, c).round(3).to_string()}")
    stack = os.path.join(a.work_dir, "tram_scope_30m.tif")
    sp = ensure_stack(a, stack)
    deps = {"stack_sha256": sp["csv_sha256"], "station_stats_sha256": file_sha256(st_csv)}
    rows = []
    for grid in grids:
        srcs = {t: file_sha256(os.path.join(a.src_root, t, f"{grid}_unified.csv")) for t in a.targets}
        todo = [t for t in a.targets if not up_to_date(out_path(a, t, grid), srcs[t], deps)]
        if not todo:
            log(f"[bo qua] {grid}: da co, nguon khong doi")
            continue
        wcell = cell_weights(stack, grid_paths[grid])
        for t in todo:
            build_one(a, t, grid, wcell, st, deps)
        df0 = pd.read_csv(os.path.join(a.src_root, todo[0], f"{grid}_unified.csv"), usecols=["cell_id", "scope_frac"],
                          dtype={"cell_id": str})
        rows.append(summary_row(grid, wcell, set(df0.loc[df0["scope_frac"] > 0, "cell_id"])))
        log(f"{grid} ({', '.join(todo)}): {rows[-1]}")
    if rows:
        rep = pd.DataFrame(rows)
        rep.to_csv(os.path.join(a.work_dir, "tram_luoi.csv"), index=False)
        print(rep.to_string(index=False), flush=True)
    log(f"Xong: {len(rows)} luoi moi")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    ap.add_argument("--grids", nargs="*")
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--mrc-dir", default=data_path("mrc_ts"))
    ap.add_argument("--src-root", default=data_path("features", "unified_bmua"))
    ap.add_argument("--out-root", default=data_path("features", "unified_tram"))
    ap.add_argument("--work-dir", default=data_path("features", "tram"))
    ap.add_argument("--osm", default=data_path("rivers/osm_waterways_mekong_delta.gpkg"))
    ap.add_argument("--coast", default=data_path("raw/river/coastline_sayre2019.gpkg"))
    ap.add_argument("--graph-tif", default=data_path("raw/river/river_graph_90m.tif"))
    ap.add_argument("--scope", default=data_path("features/scope_mask_v3.tif"))
    ap.add_argument("--min-reads", type=int, default=MIN_READS)
    ap.add_argument("--ok-frac", type=float, default=OK_FRAC)
    main(ap.parse_args())

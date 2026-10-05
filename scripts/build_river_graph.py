#!/usr/bin/env python
"""Do thi song chinh + dac trung hybrid theo o (tuan 3 viec 2, CHG-16). Quy tac: xem src/river_graph.py.

Buoc 1 (luon chay): dung do thi, tim cua song, noi cho ho (tu dong + thu cong), cat cong, kiem bat bien
  -> in + ghi danh sach de An duyet: KE_HOACH/ket-qua/dot4_do_thi_cua_song.csv, dot4_do_thi_cho_noi.csv,
  dot4_do_thi_manh.csv, dot4_do_thi_diem_quen.csv. Bat bien hoac thu tu song Hau sai -> dung (exit 1).
Buoc 2 (bo qua neu --lists-only):
  <DATA_ROOT>/raw/river/river_graph_90m.tif (cung luoi river_distance_90m.tif): dist_mouth_km, mouth_id
      (-1 = khong), behind_sluice (0/1), graph_lateral_km
  <DATA_ROOT>/features/hybrid/zos_mouth_season.csv  (mouth_id x mua: zos p90; pixel CMEMS; cmems_far > 10 km)
  <DATA_ROOT>/features/hybrid/hybrid_scope_30m.tif  (luoi scope 30 m, NaN ngoai scope)
  <DATA_ROOT>/features/hybrid/<luoi>_hybrid.csv     (cell_id, season,
      dac trung: dist_mouth_river_km, zos_mouth_p90, sluice_frac;
      cot kiem/do nhay (KHONG vao bo chinh): sluice_frac_from2021, graph_lateral_km, zos_cmems_far_frac)
Moi dau ra kem .provenance.json.

Chay:  venv/Scripts/python.exe scripts/build_river_graph.py [--lists-only] [--grids h3_res_7 ...]
"""
import argparse
import glob
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_river_features import MAIN_RIVERS  # noqa: E402
from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402
import river_graph as rg  # noqa: E402

YEARS = list(range(2014, 2027))
CMEMS_FAR_KM = 10.0
PLACES = {  # (lon, lat) - diem quen
    "Can Tho": (105.7850, 10.0340), "Long Xuyen": (105.4350, 10.3860), "Chau Doc": (105.1170, 10.7050),
    "Tan Chau": (105.2400, 10.8000), "Tra On": (105.9250, 9.9670), "My Tho": (106.3600, 10.3600),
    "Ben Tre": (106.3750, 10.2410), "Tra Vinh": (106.3420, 9.9350), "Soc Trang": (105.9720, 9.6030),
    "Bac Lieu": (105.7240, 9.2850), "Ca Mau": (105.1500, 9.1770), "Nam Can": (105.0000, 8.7600),
    "Rach Gia": (105.0800, 10.0120), "Vi Thanh": (105.4700, 9.7840), "Tan An": (106.4130, 10.5350),
}
# Khoang kiem hop ly (NHAT_KY 2026-10-04): Can Tho/Long Xuyen/Chau Doc SUA SAU khi thay ket qua (lan dau
# 85-135 / 140-200 / 180-260 khong dat); Tra On dat TRUOC khi do (Znews 25/02/2026: man 1 phan nghin toi Tra On
# "cach cua song gan 67 km").
EXPECTED_KM = {"Can Tho": (70, 90), "Long Xuyen": (120, 160), "Chau Doc": (170, 230), "My Tho": (35, 75),
               "Tra On": (57, 77)}
OUT_LISTS = os.path.join(ROOT, "KE_HOACH", "ket-qua")


def to_lonlat(xy):
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(np.asarray(xy)[:, 0], np.asarray(xy)[:, 1])
    return np.round(lon, 5), np.round(lat, 5)


def to_xy(lonlat):
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    return np.column_stack(tf.transform(*np.asarray(lonlat, float).T))


def build(args):
    osm = gpd.read_file(args.osm, layer="lines").to_crs(32648)
    lines = rg.select_graph_lines(osm, rg.graph_pattern(MAIN_RIVERS.pattern))
    km = (lines.length.groupby(lines["river"]).sum() / 1000).round(1)
    print(f"Song trong do thi ({len(km)}):", km.to_dict(), flush=True)
    missing = [e for e in rg.GRAPH_EXTRA if e not in km.index]
    if missing:
        raise ValueError(f"Khong tim thay song GRAPH_EXTRA trong OSM: {missing}")
    coast = gpd.read_file(args.coast).to_crs(32648).union_all()
    g0 = rg.build_graph(lines)
    mouths, excl = rg.find_mouths(g0, coast)
    g1, auto = rg.bridge_gaps(g0, mouths["vertex"].tolist())
    g, manual = rg.manual_joins(g1, rg.MANUAL_JOINS, to_xy)
    joins = pd.concat([t for t in (auto, manual) if len(t)] or [auto], ignore_index=True)
    joins = joins.astype({"from_vertex": "int64", "to_vertex": "int64"})
    cuts = rg.sluice_vertices(g, dict(zip(rg.SLUICES, to_xy(list(rg.SLUICES.values())))))
    return lines, coast, g, mouths, excl, joins, cuts


def report_lists(g, mouths, excl, joins, cuts):
    lon, lat = to_lonlat(g.xy)
    m = mouths.copy()
    m["lon"], m["lat"] = lon[m["vertex"]], lat[m["vertex"]]
    m["dist_coast_km"] = (m.pop("dist_coast_m") / 1000).round(2)
    j = joins.copy()
    j["lon"], j["lat"] = lon[j["from_vertex"]], lat[j["from_vertex"]]
    j["gap_m"] = j["gap_m"].round(0)
    comp = g.components()
    mcomp = set(comp[mouths["vertex"]])
    seg_len = np.bincount(comp[g.ei], weights=g.ew, minlength=comp.max() + 1)
    rows = []
    for c in np.unique(comp):
        idx = np.nonzero(comp == c)[0]
        rows.append({"comp": int(c), "km": round(seg_len[c] / 1000, 1),
                     "n_mouths": int(np.isin(mouths["vertex"], idx).sum()), "reaches_sea": bool(c in mcomp),
                     "rivers": ";".join(sorted(set(g.river[idx])))})
    comps = pd.DataFrame(rows).sort_values("km", ascending=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print("\n== Cua song (dau mut cua mot song <= 5 km tu bo; manual = ngoai le):")
        print(m.to_string(index=False))
        if len(excl):
            print("== Dau mut <= 5 km bi LOAI:")
            print(excl.assign(dist_coast_km=(excl["dist_coast_m"] / 1000).round(2)).drop(columns="dist_coast_m")
                  .to_string(index=False))
        print("\n== Cho noi (tu dong: dau mut -> manh khac <= 1 km; manual = MANUAL_JOINS):")
        print(j.to_string(index=False) if len(j) else "(khong co)")
        print("\n== Manh sau noi:")
        print(comps.to_string(index=False))
    print("\n== Cong (dinh cat, m tu toa do OSM):", {k: (v, round(d, 1)) for k, (v, d) in cuts.items()})
    os.makedirs(OUT_LISTS, exist_ok=True)
    m.to_csv(os.path.join(OUT_LISTS, "dot4_do_thi_cua_song.csv"), index=False)
    j.to_csv(os.path.join(OUT_LISTS, "dot4_do_thi_cho_noi.csv"), index=False)
    comps.to_csv(os.path.join(OUT_LISTS, "dot4_do_thi_manh.csv"), index=False)
    return comps


def place_table(g, dist, lab, behind):
    from scipy.spatial import cKDTree

    idx = np.nonzero(lab >= 0)[0]
    d, k = cKDTree(g.xy[idx]).query(to_xy(list(PLACES.values())))
    v = idx[k]
    t = pd.DataFrame({"place": list(PLACES), "along_km": np.round(dist[v] / 1000, 1),
                      "lateral_km": np.round(d / 1000, 1), "mouth_id": lab[v], "river": g.river[v],
                      "behind_sluice": behind[v],
                      "expected_km": [f"{EXPECTED_KM[p][0]}-{EXPECTED_KM[p][1]}" if p in EXPECTED_KM else ""
                                      for p in PLACES]})
    print("\n== Diem quen:")
    print(t.to_string(index=False))
    t.to_csv(os.path.join(OUT_LISTS, "dot4_do_thi_diem_quen.csv"), index=False)
    a = t.set_index("place")["along_km"]
    bad = [p for p, (lo, hi) in EXPECTED_KM.items() if not lo <= a[p] <= hi]
    order = bool(a["Chau Doc"] > a["Long Xuyen"] > a["Can Tho"] > a["Tra On"])
    print(f"Kiem hop ly: thu tu Chau Doc > Long Xuyen > Can Tho > Tra On: {order}; ngoai khoang: {bad or 'khong'}")
    return t, order, bad


def write_graph_raster(path, g, dist, lab, behind, template):
    import rasterio

    with rasterio.open(template) as t:
        transform, shape, crs = t.transform, t.shape, t.crs
    ok = np.nonzero(lab >= 0)[0]
    k, lat = rg.snap_index(g.xy[ok], transform, shape)
    v = ok[k]
    bands = {"dist_mouth_km": (dist[v] / 1000).astype("float32"), "mouth_id": lab[v].astype("float32"),
             "behind_sluice": behind[v].astype("float32"), "graph_lateral_km": (lat / 1000).astype("float32")}
    with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1], count=len(bands),
                       dtype="float32", crs=crs, transform=transform, compress="deflate", predictor=3,
                       tiled=True, nodata=np.nan) as dst:
        for i, (nm, b) in enumerate(bands.items(), start=1):
            dst.write(b, i)
            dst.set_band_description(i, nm)
    return list(bands)


def zos_mouth_table(cmems, mouths_lonlat, years):
    import xarray as xr

    from hydro_season import season_stats

    with xr.open_dataset(cmems) as ds:
        z = ds["zos"]
        lat, lon = ds[z.dims[1]].values, ds[z.dims[2]].values
        arr = z.values.astype("float64")
        times = ds["time"].values
    valid = np.isfinite(arr).all(axis=0)
    ii, jj, dkm = rg.nearest_valid_sea_pixel(lat, lon, valid, mouths_lonlat)
    rows = []
    for mid, (i, j, d) in enumerate(zip(ii, jj, dkm)):
        st = season_stats(times, arr[:, i, j], years, "zos", quantiles={"p90": 0.90}, nan_if_not_ok=True)
        st.insert(0, "mouth_id", mid)
        st["cmems_lon"], st["cmems_lat"] = float(lon[j]), float(lat[i])
        st["cmems_dist_km"], st["cmems_far"] = round(float(d), 2), bool(d > CMEMS_FAR_KM)
        rows.append(st)
    return pd.concat(rows, ignore_index=True), int(valid.sum())


def write_scope_stack(path, scope_path, graph_tif, zos_tab, years, chunk_rows=1024):
    import rasterio
    from rasterio.windows import Window

    from scope_mask import band_index, nearest_on_grid

    piv = zos_tab.pivot(index="mouth_id", columns="season", values="zos_p90")
    far = zos_tab.drop_duplicates("mouth_id").set_index("mouth_id")["cmems_far"]
    n_m = int(piv.index.max()) + 1
    lut = {y: np.full(n_m + 1, np.nan) for y in years}  # chi so -1 (khong co cua) -> phan tu cuoi = NaN
    for y in years:
        lut[y][piv.index.to_numpy()] = piv[y].to_numpy()
    far_lut = np.full(n_m + 1, np.nan)
    far_lut[far.index.to_numpy()] = far.to_numpy(float)
    names = ["dist_mouth_km", "graph_lateral_km", "behind_sluice", "zos_cmems_far"] + \
        [f"zos_mouth_p90_{y}" for y in years]
    with rasterio.open(scope_path) as sc, rasterio.open(graph_tif) as gr:
        sb = band_index(sc, "scope")
        h, w = sc.height, sc.width
        prof = dict(driver="GTiff", width=w, height=h, count=len(names), dtype="float32", crs=sc.crs,
                    transform=sc.transform, nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                    blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        stats = {"scope_px": 0, "no_mouth_px": 0, "behind_sluice_px": 0}
        with rasterio.open(path, "w", **prof) as out:
            for i, nm in enumerate(names, start=1):
                out.set_band_description(i, nm)
            for r0 in range(0, h, chunk_rows):
                n = min(chunk_rows, h - r0)
                win = Window(0, r0, w, n)
                m = sc.read(sb, window=win) == 1
                get = {b: nearest_on_grid(gr, b, sc.transform, (n, w), r0, fill=np.nan).astype("float32")
                       for b in ("dist_mouth_km", "graph_lateral_km", "behind_sluice", "mouth_id")}
                lab = get["mouth_id"]
                li = np.where(np.isfinite(lab) & (lab >= 0), lab, -1).astype(np.int64)
                stats["scope_px"] += int(m.sum())
                stats["no_mouth_px"] += int((m & (li < 0)).sum())
                stats["behind_sluice_px"] += int((m & (get["behind_sluice"] == 1)).sum())
                arrs = [get["dist_mouth_km"], get["graph_lateral_km"], get["behind_sluice"],
                        far_lut[li].astype("float32")] + [lut[y][li].astype("float32") for y in years]
                for i, a in enumerate(arrs, start=1):
                    a[~m] = np.nan
                    out.write(a, i, window=win)
    return names, stats


def cell_tables(stack, grids, static_dir, out_dir, years, prov):
    from processing import area_weighted_means

    rows = []
    for path in grids:
        name = os.path.splitext(os.path.basename(path))[0]
        grid = gpd.read_file(path)
        means = area_weighted_means(stack, (grid["cell_id"].tolist(), None, grid.geometry.tolist()))
        means.columns = ["cell_id", "dist", "lateral", "behind", "far"] + [f"z{y}" for y in years]
        st = pd.read_csv(os.path.join(static_dir, f"{name}_static.csv"), usecols=["cell_id", "scope_frac"])
        means = means.merge(st, on="cell_id", how="left", validate="1:1")
        empty = ~(means["scope_frac"] > 0)
        means.loc[empty, means.columns.drop(["cell_id", "scope_frac"])] = np.nan
        parts = []
        for y in years:
            def sl(first):
                return means["behind"] if rg.sluice_active(y, first) else means["behind"] * 0.0
            parts.append(pd.DataFrame({
                "cell_id": means["cell_id"], "season": y, "dist_mouth_river_km": means["dist"],
                "zos_mouth_p90": means[f"z{y}"], "sluice_frac": sl(rg.SLUICE_FIRST_SEASON),
                "sluice_frac_from2021": sl(rg.SLUICE_ALT_SEASON), "graph_lateral_km": means["lateral"],
                "zos_cmems_far_frac": means["far"]}))
        long = pd.concat(parts, ignore_index=True)
        csv = os.path.join(out_dir, f"{name}_hybrid.csv")
        long.to_csv(csv, index=False, float_format="%.5f")
        write_provenance(csv, CANONICAL_BOUNDARY, grid=os.path.basename(path), grid_sha256=file_sha256(path),
                         stack=os.path.basename(stack), **prov)
        ok = means[~empty]
        rows.append({"grid": name, "cells": len(means), "cells_scope": int((~empty).sum()),
                     "nan_dist": int(ok["dist"].isna().sum()), "dist_p50": round(ok["dist"].median(), 1),
                     "lateral_p50": round(ok["lateral"].median(), 1),
                     "lateral_gt10_cells": round(float((ok["lateral"] > 10).mean()), 3),
                     "cells_sluice_gt0": int((ok["behind"] > 0).sum()),
                     "zos_2020_p50": round(ok["z2020"].median(), 4)})
        print(f"  {name}: {rows[-1]}", flush=True)
    return pd.DataFrame(rows)


def main(a):
    lines, coast, g, mouths, excl, joins, cuts = build(a)
    report_lists(g, mouths, excl, joins, cuts)
    dist, lab = rg.mouth_distances(g, mouths)
    _, lab_cut = rg.mouth_distances(g, mouths, drop_vertices=[v for v, _ in cuts.values()])
    behind = rg.behind_sluice(lab, lab_cut)
    n_bad, worst = rg.check_invariant(g, mouths, dist, lab)
    print(f"\nDinh toi duoc bien {np.mean(lab >= 0):.1%}, max {np.nanmax(dist[lab >= 0]) / 1000:.0f} km; dinh sau cong "
          f"{int(behind.sum())}; bat bien vi pham {n_bad} dinh (lon nhat {worst:.2f} m)")
    _, order, bad = place_table(g, dist, lab, behind)
    if n_bad or not order:
        sys.exit("DUNG: bat bien hoac thu tu song Hau sai - khong tinh dac trung.")
    if a.lists_only:
        return
    prov = dict(osm=a.osm, osm_sha256=file_sha256(a.osm), coast=a.coast, coast_sha256=file_sha256(a.coast),
                main_rivers=MAIN_RIVERS.pattern, graph_extra=list(rg.GRAPH_EXTRA), spacing_m=rg.SPACING_M,
                gap_tol_m=rg.GAP_TOL_M, mouth_tol_m=rg.MOUTH_TOL_M, mouth_tol_exceptions=rg.MOUTH_TOL_EXCEPTIONS,
                manual_joins=[list(j) for j in rg.MANUAL_JOINS], excluded_mouths=list(rg.EXCLUDED_MOUTH_RIVERS),
                sluices=rg.SLUICES, sluice_first_season=rg.SLUICE_FIRST_SEASON,
                sluice_alt_season=rg.SLUICE_ALT_SEASON, n_mouths=len(mouths), n_joins=len(joins), rule="CHG-16")
    os.makedirs(a.out_dir, exist_ok=True)
    gtif = os.path.join(a.river_dir, "river_graph_90m.tif")
    write_graph_raster(gtif, g, dist, lab, behind, os.path.join(a.river_dir, "river_distance_90m.tif"))
    write_provenance(gtif, CANONICAL_BOUNDARY, **prov)
    lon, lat = to_lonlat(g.xy)
    zt, n_valid = zos_mouth_table(a.cmems, np.column_stack([lon[mouths["vertex"]], lat[mouths["vertex"]]]), YEARS)
    zt = zt.merge(mouths[["mouth_id", "river"]], on="mouth_id")
    zcsv = os.path.join(a.out_dir, "zos_mouth_season.csv")
    zt.to_csv(zcsv, index=False, float_format="%.5f")
    write_provenance(zcsv, CANONICAL_BOUNDARY, cmems=a.cmems, cmems_sha256=file_sha256(a.cmems),
                     cmems_far_km=CMEMS_FAR_KM, **prov)
    print("\n== Pixel CMEMS cho tung cua (pixel hop le moi ngay: %d):" % n_valid)
    print(zt.drop_duplicates("mouth_id")[["mouth_id", "river", "cmems_lon", "cmems_lat", "cmems_dist_km", "cmems_far"]]
          .to_string(index=False))
    print(zt.pivot(index="season", columns="mouth_id", values="zos_p90").round(3).to_string())
    stack = os.path.join(a.out_dir, "hybrid_scope_30m.tif")
    names, stats = write_scope_stack(stack, a.scope, gtif, zt, YEARS)
    write_provenance(stack, CANONICAL_BOUNDARY, scope=a.scope, scope_sha256=file_sha256(a.scope),
                     graph=os.path.basename(gtif), bands=names, **stats, **prov)
    print("Scope stack:", stats, flush=True)
    grids = sorted(glob.glob(os.path.join(a.grids_dir, "*.geojson")))
    if a.grids:
        grids = [p for p in grids if os.path.splitext(os.path.basename(p))[0] in a.grids]
    rep = cell_tables(stack, grids, a.static_dir, a.out_dir, YEARS, prov)
    rep.to_csv(os.path.join(OUT_LISTS, "dot4_hybrid_luoi.csv"), index=False)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--osm", default=data_path("rivers/osm_waterways_mekong_delta.gpkg"))
    ap.add_argument("--coast", default=data_path("raw/river/coastline_sayre2019.gpkg"))
    ap.add_argument("--cmems", default=data_path("cmems/cmems_zos_daily_mekong_1999_2026.nc"))
    ap.add_argument("--scope", default=data_path("features/scope_mask_v3.tif"))
    ap.add_argument("--river-dir", default=data_path("raw/river"))
    ap.add_argument("--static-dir", default=data_path("features/static"))
    ap.add_argument("--out-dir", default=data_path("features/hybrid"))
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--grids", nargs="*")
    ap.add_argument("--lists-only", action="store_true")
    main(ap.parse_args())

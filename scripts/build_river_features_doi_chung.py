#!/usr/bin/env python
"""Raster khoang cach song/bien 90 m, 3 band, cho vung doi chung (mississippi, nile) - cung ten band voi DBSCL.

  dist_main_river_km : OSM fclass river co ten khop MAIN_NAMES[vung] (--source osm, mac dinh)
  dist_any_water_km  : OSM river+canal+stream (+drain o Nile) HOP mat nuoc JRC dang dai (occurrence >= 50, nhu DBSCL)
  dist_coast_km      : bo bien Sayre 2019 (GEE), open 4 km / tol 1 km nhu DBSCL; --drop-small-islands bo dao < 1 km2
--source hydrorivers: ban cu (HydroRIVERS, chi de doi chieu; mac dinh ghi ra *_hydrorivers.tif).

Chay:  python scripts/build_river_features_doi_chung.py --region mississippi [--project <id>] [--osm-zip <egypt-latest-free.shp.zip>]
       (nile: --osm-zip de lay fclass drain; mississippi: them --drop-small-islands khi chua co cache bo bien)
"""
import argparse
import os
import re
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_river_features import BANDS, _LazyEE, distance_km  # noqa: E402
from check_osm_vs_jrc import label_linear_water  # noqa: E402
from preprocessing import file_sha256, write_provenance  # noqa: E402
from region import get_region, region_paths  # noqa: E402

# Ten song chinh (OSM name, so khop sau strip) - An chot CHG-39.
MAIN_NAMES = {
    "mississippi": ["Mississippi River", "Lower Atchafalaya River", "Atchafalaya River", "Wax Lake Outlet",
                    "Southwest Pass", "South Pass", "Pass a Loutre", "Main Pass", "Bayou Lafourche", "Bayou Teche",
                    "Bayou Terrebonne", "Vermilion River", "Mermentau River", "Calcasieu River", "Sabine River",
                    "Pearl River", "West Pearl River", "Red River"],
    "nile": ["فرع رشيد", "فرع دمياط", "نهر النيل", "Rosetta Branch", "Damietta Branch", "Nile"],
}
ANY_FCLASS = {"mississippi": ["river", "canal", "stream"], "nile": ["river", "canal", "stream", "drain"]}
OSM_URL = {"mississippi": "https://download.geofabrik.de/north-america/us/louisiana-latest-free.shp.zip",
           "nile": "https://download.geofabrik.de/africa/egypt-latest-free.shp.zip"}
OSM_LICENCE = "OpenStreetMap contributors, ODbL 1.0 (Geofabrik extract, tai 2026-10-16); JRC GSW v1.4 CC-BY 4.0; Sayre et al. 2019"
HYDRO_LICENCE = "HydroRIVERS v1.0 (Lehner & Grill 2013), HydroSHEDS licence - mien phi, ghi nguon; JRC GSW v1.4 CC-BY 4.0"


def main_regex(reg):
    return re.compile("^(" + "|".join(MAIN_NAMES[reg]) + ")$")


def select_main_osm(osm, reg):
    """Doan fclass river co name (da strip) khop ten song chinh; tra ve (gdf, ten khop da strip)."""
    name = osm["name"].fillna("").str.strip()
    sel = (osm["fclass"] == "river") & name.str.match(main_regex(reg))
    return osm[sel], name[sel]


def load_osm(gpkg, reg, epsg, osm_zip=None):
    """Doc OSM lines (CRS vung); fclass thieu trong gpkg (drain o Nile) lay them tu zip Geofabrik trong bbox gpkg."""
    g = gpd.read_file(gpkg, layer="lines")
    missing = [c for c in ANY_FCLASS[reg] if c not in set(g["fclass"])]
    if missing:
        if not osm_zip:
            raise SystemExit(f"Thieu fclass {missing} trong {gpkg}: truyen --osm-zip")
        extra = gpd.read_file(f"zip://{osm_zip}!gis_osm_waterways_free_1.shp", bbox=tuple(g.total_bounds),
                              where="fclass IN (" + ",".join(f"'{c}'" for c in missing) + ")")
        print(f"Bo sung tu zip: {extra['fclass'].value_counts().to_dict()}", flush=True)
        g = pd.concat([g, extra[g.columns]], ignore_index=True)
    return g.to_crs(epsg)


def select_main(g, min_dis_cms):
    """Doan song chinh = luu luong TB nam (m3/s) >= nguong."""
    return g[g["DIS_AV_CMS"] >= min_dis_cms]


def wide_linear_water(water, px_km2, min_km2, near=None):
    """Mat nuoc dang dai (song/kenh) dien tich >= min_km2 -> bool; loai kenh nho, ao/ho tron.

    near: mask bool; neu co, chi giu vung cham near (loai dam pha/ho ven bien dang dai khong phai song).
    """
    lab, n, _, area, linear = label_linear_water(water, px_km2)
    keep = linear & (area * px_km2 >= min_km2)
    if near is not None:
        keep &= np.bincount(lab[near], minlength=n + 1) > 0
    return keep[lab]


def list_big_components(mask, transform, epsg):
    """In vung nuoc JRC rong da chon (km2, tam lon/lat) de kiem tra bang mat."""
    from pyproj import Transformer
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not n:
        return
    area = np.bincount(lab.ravel())[1:] * abs(transform.a * transform.e) / 1e6
    com = ndimage.center_of_mass(mask, lab, range(1, n + 1))
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    print(f"JRC rong chon: {n} vung")
    for i in np.argsort(-area)[:15]:
        lon, lat = tr.transform(*(transform * (com[i][1] + 0.5, com[i][0] + 0.5)))
        print(f"  {area[i]:8.1f} km2  lon {lon:.2f} lat {lat:.2f}")
    sys.stdout.flush()


def clip_hydrorivers(path, poly, epsg):
    """Doc doan song trong da giac poly (CRS metric), loc theo bbox cho nhanh."""
    bb = tuple(gpd.GeoSeries([poly], crs=epsg).to_crs(4326).total_bounds)
    g = gpd.read_file(path, bbox=bb).to_crs(epsg)
    return g[g.intersects(poly)].copy()


def sea_and_coast(ee, boundary, epsg, cache, pad_deg=0.4, open_m=4000.0, tol_m=1000.0, tile_deg=1.0,
                  drop_small_islands=False):
    """(bien mo, bo bien) tu Sayre 2019 - giong DBSCL (open 4 km, tol 1 km); chi giu bien mo noi voi mep khung."""
    if os.path.exists(cache):
        return gpd.read_file(cache, layer="sea").union_all(), gpd.read_file(cache, layer="coast")
    x0, y0, x1, y1 = boundary.to_crs(4326).total_bounds
    x0, y0, x1, y1 = x0 - pad_deg, y0 - pad_deg, x1 + pad_deg, y1 + pad_deg
    base = "projects/sat-io/open-datasets/shoreline/"
    fc = ee.FeatureCollection(base + "mainlands").merge(ee.FeatureCollection(base + "big_islands"))
    if not drop_small_islands:  # small_islands = dao < 1 km2 (kiem 2026-10-16); hop 4.900 dao ngon 6-8 GB RAM o Louisiana
        fc = fc.merge(ee.FeatureCollection(base + "small_islands"))
    parts = []
    for xa in np.arange(x0, x1, tile_deg):          # chia o 1 deg: getInfo gioi han kich thuoc
        for ya in np.arange(y0, y1, tile_deg):
            r = ee.Geometry.Rectangle([xa, ya, min(xa + tile_deg, x1), min(ya + tile_deg, y1)])
            feats = fc.filterBounds(r).map(lambda f: ee.Feature(f.geometry().intersection(r, 10))).getInfo()["features"]
            if feats:
                parts.append(gpd.GeoDataFrame.from_features(feats, crs=4326).to_crs(epsg))
            print(f"  bo bien o ({xa:.1f},{ya:.1f}): {len(feats)} doi tuong", flush=True)
    land = gpd.GeoSeries(pd.concat(parts).geometry.buffer(15)).union_all()  # +15 m: khep ke ho giua cac o
    rect = gpd.GeoSeries([box(x0, y0, x1, y1)], crs=4326).to_crs(epsg).iloc[0]
    sea = rect.difference(land).buffer(-open_m).buffer(open_m)
    sea = gpd.GeoSeries([sea], crs=epsg).explode(index_parts=False)
    sea = sea[sea.intersects(rect.exterior.buffer(open_m))].union_all()  # bo ho/dam phi khong thong bien mo
    edge = land.boundary.intersection(sea.buffer(tol_m)).difference(rect.exterior.buffer(200))
    coast = gpd.GeoDataFrame(geometry=[edge], crs=epsg)
    gpd.GeoDataFrame(geometry=[sea], crs=epsg).to_file(cache, layer="sea", driver="GPKG")
    coast.to_file(cache, layer="coast", driver="GPKG")
    return sea, coast


def jrc_masks(args, epsg, paths, boundary, sea, shape, transform, main_h=None):
    """JRC 30 m -> (any[, big]) tren luoi 90 m. osm: occurrence >= --jrc-occ; hydrorivers: max_extent == 1.
    Chi tinh nuoc trong ranh gioi, tru bien mo (bien la vung nuoc khong lo -> bi tinh la "kenh", known-pitfalls 10)."""
    with rasterio.open(paths["jrc"]) as src:
        water = (src.read(1) >= args.jrc_occ) if args.source == "osm" else (src.read(2) == 1)
        inside = features.rasterize([boundary.union_all()], out_shape=water.shape, transform=src.transform, fill=0,
                                    default_value=1) > 0
        in_sea = features.rasterize([sea], out_shape=water.shape, transform=src.transform, fill=0, default_value=1) > 0
        water &= inside & ~in_sea
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
        arrs = [wide_linear_water(water, px_km2, 0.0).astype("uint8")]
        if main_h is not None:
            near = features.rasterize(main_h.geometry.buffer(args.main_jrc_link_km * 1000), out_shape=water.shape,
                                      transform=src.transform, fill=0, default_value=1) > 0
            big = wide_linear_water(water, px_km2, args.main_jrc_km2, near).astype("uint8")
            list_big_components(big, src.transform, epsg)
            arrs.append(big)
        del water, inside, in_sea
        out = []
        for a in arrs:
            d = np.zeros(shape, "uint8")
            reproject(a, d, src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                      dst_crs=f"EPSG:{epsg}", resampling=Resampling.max)
            out.append(d > 0)
    return out


def hydrorivers_cut(args, reg, paths, frame, epsg):
    cache = args.hydrorivers_cache or os.path.join(os.path.dirname(paths["jrc"]), f"HydroRIVERS_{get_region(reg)['label']}.gpkg")
    if os.path.exists(cache):
        return gpd.read_file(cache)
    g = clip_hydrorivers(args.hydrorivers, frame, epsg)
    g.to_file(cache, driver="GPKG")
    return g


def main(args):
    t0 = time.time()
    reg, epsg = args.region, get_region(args.region)["epsg"]
    label = get_region(reg)["label"]
    paths = region_paths(reg)
    boundary = gpd.read_file(paths["boundary"]).to_crs(epsg)
    frame = boundary.buffer(args.pad_km * 1000).union_all()
    x0, y0, x1, y1 = frame.bounds
    res = args.res_m
    shape = (int(np.ceil((y1 - y0) / res)), int(np.ceil((x1 - x0) / res)))
    transform = from_origin(x0, y1, res, res)
    out_dir = os.path.dirname(paths["river"])
    os.makedirs(out_dir, exist_ok=True)
    tif = args.out or (paths["river"] if args.source == "osm" else paths["river"].replace(".tif", "_hydrorivers.tif"))
    if os.path.exists(tif) and not args.overwrite:
        raise SystemExit(f"{tif} da co: doi ten truoc hoac them --overwrite")

    def burn(geoms):
        return features.rasterize(geoms, out_shape=shape, transform=transform, fill=0, default_value=1,
                                  all_touched=True) > 0

    coast_gpkg = os.path.join(out_dir, "coastline_sayre2019.gpkg")
    sea, coast = sea_and_coast(_LazyEE(args.project), boundary, epsg, coast_gpkg,
                               drop_small_islands=args.drop_small_islands)
    m_coast = burn(coast.geometry)
    prov = dict(region=reg, epsg=epsg, res_m=res, pad_km=args.pad_km, source=args.source,
                jrc=os.path.basename(paths["jrc"]), jrc_sha256=file_sha256(paths["jrc"]),
                coastline="coastline_sayre2019.gpkg", coastline_sha256=file_sha256(coast_gpkg), coast_open_m=4000,
                coast_tol_m=1000, coast_km=round(coast.length.sum() / 1000, 1),
                coast_method=("Sayre 2019 mainlands+big_islands; BO small_islands (< 1 km2) truoc khi hop nhat/dem"
                              if args.drop_small_islands else "Sayre 2019 mainlands+big_islands+small_islands"))
    if args.source == "osm":
        osm_gpkg = args.osm or os.path.join(out_dir, f"osm_waterways_{label}.gpkg")
        osm = load_osm(osm_gpkg, reg, epsg, args.osm_zip)
        main_o, main_name = select_main_osm(osm, reg)
        km = (main_o.length / 1000).groupby(main_name).sum()
        cnt = main_name.value_counts()
        by_name = {n: {"segments": int(cnt.get(n, 0)), "km": round(float(km.get(n, 0.0)), 1)} for n in MAIN_NAMES[reg]}
        print("Song chinh theo ten (so doan / km):", flush=True)
        for n, v in by_name.items():
            print(f"  {n}: {v['segments']} doan, {v['km']} km" + ("   CANH BAO: 0 doan" if not v["segments"] else ""))
        any_o = osm[osm["fclass"].isin(ANY_FCLASS[reg])]
        by_fclass = {c: {"segments": int((any_o.fclass == c).sum()),
                         "km": round(float(any_o[any_o.fclass == c].length.sum() / 1000), 1)} for c in ANY_FCLASS[reg]}
        print("Any water OSM:", by_fclass, flush=True)
        (jrc_any,) = jrc_masks(args, epsg, paths, boundary, sea, shape, transform)
        m_main, m_any_vec = burn(main_o.geometry), burn(any_o.geometry)
        m_any = m_any_vec | jrc_any
        prov.update(osm_gpkg=os.path.basename(osm_gpkg), osm_gpkg_sha256=file_sha256(osm_gpkg), osm_url=OSM_URL[reg],
                    osm_date="2026-10-16", licence=OSM_LICENCE, main_regex=main_regex(reg).pattern,
                    main_rule="OSM fclass == river AND name (strip) khop main_regex", main_by_name=by_name,
                    any_rule=f"OSM fclass in {ANY_FCLASS[reg]} UNION JRC occurrence >= {args.jrc_occ} (30 m, trong ranh gioi, "
                             "tru bien mo, chi vung dang dai label_linear_water, resample max ve 90 m) - nhu DBSCL",
                    any_by_fclass=by_fclass, jrc_occ=args.jrc_occ, jrc_pixels_added=int((jrc_any & ~m_any_vec).sum()))
        if args.osm_zip:
            prov["osm_zip"] = os.path.basename(args.osm_zip)
        print(f"Pixel {res} m: song chinh {m_main.sum()}, any {m_any.sum()} (JRC them {prov['jrc_pixels_added']}), "
              f"bo bien {m_coast.sum()} ({prov['coast_km']} km)", flush=True)
    else:
        g = hydrorivers_cut(args, reg, paths, frame, epsg)
        main_h = select_main(g, args.main_min_dis_cms)
        print(f"HydroRIVERS trong khung: {len(g)} doan; song chinh (DIS_AV_CMS >= {args.main_min_dis_cms}): {len(main_h)}",
              flush=True)
        jrc_any, jrc_big = jrc_masks(args, epsg, paths, boundary, sea, shape, transform, main_h)
        m_main, m_any = burn(main_h.geometry) | jrc_big, burn(g.geometry) | jrc_any
        prov.update(licence=HYDRO_LICENCE, main_min_dis_cms=args.main_min_dis_cms, main_jrc_km2=args.main_jrc_km2,
                    main_jrc_link_km=args.main_jrc_link_km, jrc_rule="max_extent == 1")

    stack = np.stack([distance_km(m, res) for m in (m_main, m_any, m_coast)])
    stack[:, features.rasterize([frame], out_shape=shape, transform=transform, fill=1, default_value=0) > 0] = np.nan
    with rasterio.open(tif, "w", driver="GTiff", height=shape[0], width=shape[1], count=3, dtype="float32",
                       crs=f"EPSG:{epsg}", transform=transform, compress="deflate", nodata=np.nan) as dst:
        dst.write(stack)
        dst.descriptions = tuple(BANDS)
    import psutil
    peak_gb = round(psutil.Process().memory_info().peak_wset / 1e9, 2)
    write_provenance(tif, paths["boundary"], elapsed_s=round(time.time() - t0), peak_ram_gb=peak_gb, **prov)
    print(f"Raster: {tif}  ({time.time() - t0:.0f} s, dinh RAM {peak_gb} GB)")
    if args.figure:
        draw_figure(args.figure, reg, boundary, (m_any, m_main, m_coast), (x0, y1, res, shape))
    for i, b in enumerate(BANDS):
        v = stack[i]
        print(f"{b}: min {np.nanmin(v):.3f} max {np.nanmax(v):.1f} NaN {np.isnan(v).mean() * 100:.1f}%")


def draw_figure(path, reg, boundary, masks, geo, k=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    x0, y1, res, shape = geo
    h, w = shape[0] // k * k, shape[1] // k * k
    ext = (x0, x0 + w * res, y1 - h * res, y1)
    fig, ax = plt.subplots(figsize=(11, 7))
    cols = ("#9ecae1", "#d62728", "#111111")
    for m, c in zip(masks, cols):
        pooled = m[:h, :w].reshape(h // k, k, w // k, k).any(axis=(1, 3))
        ax.imshow(np.ma.masked_where(~pooled, np.ones(pooled.shape)), extent=ext,
                  cmap=matplotlib.colors.ListedColormap([c]), interpolation="nearest")
    boundary.boundary.plot(ax=ax, color="#777777", lw=0.8, ls="--")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    names = ("Other water (OSM + JRC)", "Main rivers (OSM, named)", "Coastline (Sayre 2019)", "Study-area boundary")
    ax.legend(handles=[Line2D([], [], color=c, lw=3, ls="-" if i < 3 else "--", label=t) for i, (c, t) in enumerate(zip(cols + ("#777777",), names))],
              loc="lower left", frameon=True)
    ax.set_title(f"{'Mississippi' if reg == 'mississippi' else 'Nile'} Delta: distance-feature sources "
                 f"(EPSG:{get_region(reg)['epsg']}, 90 m grid)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", required=True, choices=sorted(MAIN_NAMES))
    ap.add_argument("--source", choices=("osm", "hydrorivers"), default="osm")
    ap.add_argument("--osm", help="gpkg OSM (lop lines); mac dinh <doi_chung_out>/<vung>/features/river/osm_waterways_<MS|NL>.gpkg")
    ap.add_argument("--osm-zip", help="zip Geofabrik <nuoc>-latest-free.shp.zip (de bo sung fclass thieu, vd drain)")
    ap.add_argument("--jrc-occ", type=int, default=50, help="JRC occurrence (%%) toi thieu - nhu DBSCL")
    ap.add_argument("--drop-small-islands", action="store_true",
                    help="bo dao Sayre < 1 km2 khi dung bo bien (chi tac dong khi chua co cache)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", help="tif dau ra (mac dinh river_distance_90m.tif; hydrorivers: *_hydrorivers.tif)")
    ap.add_argument("--hydrorivers", help="HydroRIVERS_v10_<na|af>.shp (chi --source hydrorivers, chua co cache)")
    ap.add_argument("--hydrorivers-cache", help="gpkg ban cat HydroRIVERS theo vung")
    ap.add_argument("--project", default=os.getenv("EE_PROJECT"))
    ap.add_argument("--main-min-dis-cms", type=float, default=100.0)
    ap.add_argument("--main-jrc-km2", type=float, default=20.0)
    ap.add_argument("--main-jrc-link-km", type=float, default=5.0)
    ap.add_argument("--figure", help="png kiem nhanh (nhan tieng Anh)")
    ap.add_argument("--res-m", type=float, default=90.0)
    ap.add_argument("--pad-km", type=float, default=20.0)
    main(ap.parse_args())

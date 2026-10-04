#!/usr/bin/env python
"""Dac trung khoang cach toi song/bien cho 13 luoi (T3-V1) - 3 bien the de chon bang thi nghiem.

  dist_main_river_km : toi cac song lon cua song (nhanh Mekong + Vam Co + Cai Lon/Cai Be + Ganh Hao/Ong Doc/Cua Lon), OSM
  dist_any_water_km  : toi song/kenh gan nhat = OSM (river, canal) HOP mat nuoc dang dai JRC
  dist_coast_km      : toi bo bien (Global Shoreline Vector, Sayre et al. 2019)

Cach tinh: raster khoang cach (bien doi khoang cach Euclid, 90 m, EPSG:32648) roi trung binh
co trong so dien tich trong tung o (khong dung khoang cach tu tam o: o lon se bi sai lech).

Ranh gioi v2 (noi 3 km ra bien, 2026-10-03) chi dung cho KHUNG raster va luoi. Hai thu KHONG theo v2:
  - bo bien: tu duong bo that (Sayre + phep mo 4 km, cache coastline_sayre2019.gpkg), khong tu ranh gioi;
  - mat nuoc JRC cat theo --jrc-clip-boundary (mac dinh v1): cat theo v2 thi dai bien 3 km noi thanh mot
    vung nuoc dai -> 2,14 trieu pixel bien bi tinh la "kenh" (do 2026-10-03, loi kieu known-pitfalls 10).
Dau ra cu duoc chuyen sang <out-dir>/deprecated_v1/ truoc khi ghi; moi dau ra kem .provenance.json.

Chay:  python scripts/build_river_features.py [--project <id>] [--grids-dir data/grids] [--out-dir A:/Dataset_NCKH/raw/river]
       (--project chi can khi chua co cache bo bien)
"""
import argparse
import glob
import os
import re
import shutil
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from check_osm_vs_jrc import label_linear_water  # noqa: E402
from preprocessing import BOUNDARY_V1, CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402
from processing import area_weighted_means  # noqa: E402

# Song lon cua song: duong nuoc man vao noi dong (An duyet 2026-10-01, them Ganh Hao, Ong Doc;
# 2026-10-03 them Cua Lon - OSM "Sông Cửa Lớn", fclass river, 50,9 km, Nam Can - Ca Mau).
MAIN_RIVERS = re.compile(r"^Sông (Tiền|Hậu|Cổ Chiên|Hàm Luông|Mỹ Tho|Ba Lai|Cửa Đại|Cửa Tiểu|Cung Hầu|"
                         r"Định An|Trần Đề|Vàm Cỏ Đông|Vàm Cỏ Tây|Vàm Cỏ|Cái Lớn|Cái Bé|Gành Hào|Ông Đốc|"
                         r"Cửa Lớn)\b")
BANDS = ["dist_main_river_km", "dist_any_water_km", "dist_coast_km"]


def distance_km(mask, res_m):
    """Khoang cach (km) tu moi pixel toi pixel True gan nhat."""
    if not mask.any():
        raise ValueError("Mat na rong: khong co doi tuong de tinh khoang cach")
    return (ndimage.distance_transform_edt(~mask) * res_m / 1000).astype("float32")


def coastline(ee, boundary, cache, pad_deg=0.4, open_m=4000.0, tol_m=1000.0):
    """Bo bien tu Global Shoreline Vector (Sayre et al. 2019, tu Landsat 30 m), GEE community catalog.

    Khong suy tu ranh gioi tinh: canh ranh gioi con lan bien gioi Long An - TP.HCM/Tay Ninh va mep
    cu lao giua song (kiem tra 2026-10-01).
    """
    if os.path.exists(cache):
        return gpd.read_file(cache)
    x0, y0, x1, y1 = boundary.to_crs(4326).total_bounds
    rect = ee.Geometry.Rectangle([x0 - pad_deg, y0 - pad_deg, x1 + pad_deg, y1 + pad_deg])
    base = "projects/sat-io/open-datasets/shoreline/"
    land = (ee.FeatureCollection(base + "mainlands").merge(ee.FeatureCollection(base + "big_islands"))
            .merge(ee.FeatureCollection(base + "small_islands")).filterBounds(rect)
            .map(lambda f: ee.Feature(f.geometry().intersection(rect, 10))))
    gdf = gpd.GeoDataFrame.from_features(land.getInfo()["features"], crs=4326).to_crs(32648)
    rect_m = gpd.GeoSeries.from_wkt([f"POLYGON(({x0 - pad_deg} {y0 - pad_deg},{x1 + pad_deg} {y0 - pad_deg},"
                                     f"{x1 + pad_deg} {y1 + pad_deg},{x0 - pad_deg} {y1 + pad_deg},"
                                     f"{x0 - pad_deg} {y0 - pad_deg}))"], crs=4326).to_crs(32648).iloc[0]
    land_u = gdf.union_all()
    # Bo du lieu nay coi song rong (Tien, Hau: 1-3 km) la bien -> bo bien chay sau vao noi dong
    # (1.952 km nam sau > 10 km, kiem tra 2026-10-01). Chi giu BIEN MO bang phep mo hinh thai hoc:
    # co vung nuoc vao open_m roi no lai -> kenh/song hep hon ~2*open_m bien mat.
    water = rect_m.difference(land_u)
    sea = water.buffer(-open_m).buffer(open_m)
    edge = land_u.boundary.intersection(sea.buffer(tol_m)).difference(rect_m.exterior.buffer(200))
    coast = gpd.GeoDataFrame(geometry=[edge], crs=32648)
    coast.to_file(cache, driver="GPKG")
    return coast


class _LazyEE:
    """Chi khoi tao Earth Engine khi that su can (cache bo bien chua co)."""

    def __init__(self, project):
        self.project, self._ee = project, None

    def __getattr__(self, name):
        if self._ee is None:
            import ee
            ee.Initialize(project=self.project)
            self._ee = ee
        return getattr(self._ee, name)


def backup_outputs(out_dir, dst_name="deprecated_v1"):
    """Chep raster + bang dac trung cu sang <out_dir>/<dst_name>/ (khong ghi de ban sao luu da co)."""
    dst = os.path.join(out_dir, dst_name)
    olds = glob.glob(os.path.join(out_dir, "river_distance_*m.tif")) +         glob.glob(os.path.join(out_dir, "features", "*_river.csv"))
    for p in olds:
        target = os.path.join(dst, os.path.relpath(p, out_dir))
        if os.path.exists(target):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(p, target)
    if olds:
        print(f"Sao luu {len(olds)} file cu -> {dst}", flush=True)


def main(args):
    ee = _LazyEE(args.project)
    boundary = gpd.read_file(args.boundary).to_crs(32648)
    jrc_clip = gpd.read_file(args.jrc_clip_boundary).to_crs(32648)
    x0, y0, x1, y1 = boundary.buffer(args.pad_km * 1000).total_bounds
    res = args.res_m
    width, height = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    transform = from_origin(x0, y1, res, res)
    shape = (height, width)

    osm = gpd.read_file(args.osm, layer="lines").to_crs(32648)
    main_r = osm[(osm["fclass"] == "river") & osm["name"].fillna("").str.match(MAIN_RIVERS)]
    print("Song chinh theo ten:", main_r["name"].str.extract(MAIN_RIVERS)[0].value_counts().to_dict())
    print("Chieu dai (km):", (main_r.length.groupby(main_r["name"].str.extract(MAIN_RIVERS)[0]).sum() / 1000)
          .round(1).to_dict(), flush=True)
    any_r = osm[osm["fclass"].isin(["river", "canal"])]

    def burn(geoms):
        return features.rasterize(geoms, out_shape=shape, transform=transform, fill=0, default_value=1,
                                  all_touched=True) > 0

    m_main = burn(main_r.geometry)
    m_any = burn(any_r.geometry)
    # JRC 30 m -> vung nuoc dang dai -> gop vao luoi 90 m (pixel 90 m co nuoc neu co bat ky pixel 30 m nao).
    with rasterio.open(args.jrc) as src:
        water = src.read(1) > 0
        # Chi xet mat nuoc TRONG ranh gioi cat JRC (v1): bien la mot vung nuoc khong lo "dang dai" -> se bi
        # tinh la kenh. KHONG dung v2: dai bien 3 km cua v2 noi thanh vung nuoc dai (xem docstring).
        water &= features.rasterize(jrc_clip.geometry, out_shape=water.shape, transform=src.transform,
                                    fill=0, default_value=1) > 0
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
        lab, n, _, _, linear = label_linear_water(water, px_km2)
        lin = linear[lab].astype("uint8")
        jrc = np.zeros(shape, "uint8")
        reproject(lin, jrc, src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                  dst_crs="EPSG:32648", resampling=Resampling.max)
    m_any_jrc = m_any | (jrc > 0)
    print(f"Pixel {res} m: song chinh {m_main.sum()}, OSM song+kenh {m_any.sum()}, "
          f"+JRC {m_any_jrc.sum()} (JRC them {(m_any_jrc & ~m_any).sum()})")

    os.makedirs(args.out_dir, exist_ok=True)
    backup_outputs(args.out_dir)
    coast_cache = os.path.join(args.out_dir, "coastline_sayre2019.gpkg")
    coast = coastline(ee, boundary, coast_cache)
    m_coast = burn(coast.geometry)
    print(f"Bo bien: {coast.length.sum() / 1000:.0f} km")

    stack = np.stack([distance_km(m, res) for m in (m_main, m_any_jrc, m_coast)])
    tif = os.path.join(args.out_dir, f"river_distance_{int(res)}m.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=height, width=width, count=3, dtype="float32",
                       crs="EPSG:32648", transform=transform, compress="deflate", nodata=np.nan) as dst:
        dst.write(stack)
        dst.descriptions = tuple(BANDS)
    prov = dict(main_rivers=MAIN_RIVERS.pattern, jrc_clip_boundary=args.jrc_clip_boundary,
                jrc_clip_boundary_sha256=file_sha256(args.jrc_clip_boundary),
                coastline=coast_cache, coastline_sha256=file_sha256(coast_cache), res_m=res, pad_km=args.pad_km)
    write_provenance(tif, args.boundary, **prov)
    print(f"Raster: {tif}")

    feat_dir = os.path.join(args.out_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.grids_dir, "*.geojson"))):
        grid = gpd.read_file(path)
        df = area_weighted_means(tif, (grid["cell_id"].tolist(), None, grid.geometry.tolist()))
        df.columns = ["cell_id"] + BANDS
        name = os.path.splitext(os.path.basename(path))[0]
        csv = os.path.join(feat_dir, f"{name}_river.csv")
        df.to_csv(csv, index=False)
        write_provenance(csv, args.boundary, grid=os.path.basename(path), grid_sha256=file_sha256(path),
                         raster=os.path.basename(tif), **prov)
        rows.append({"grid": name, "cells": len(df), "nan": int(df[BANDS].isna().sum().sum()),
                     **{f"{b}_median": round(df[b].median(), 2) for b in BANDS}})
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    if args.report:
        report.to_csv(args.report, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=os.getenv("EE_PROJECT"))
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--jrc-clip-boundary", default=BOUNDARY_V1,
                    help="Ranh gioi cat mat nuoc JRC (mac dinh v1; KHONG dung v2 - dai bien thanh 'kenh')")
    ap.add_argument("--osm", default="A:/Dataset_NCKH/rivers/osm_waterways_mekong_delta.gpkg")
    ap.add_argument("--jrc", default="A:/Dataset_NCKH/rivers/jrc_gsw_occ50_30m.tif")
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--out-dir", default="A:/Dataset_NCKH/raw/river")
    ap.add_argument("--res-m", type=float, default=90.0)
    ap.add_argument("--pad-km", type=float, default=10.0)
    ap.add_argument("--report", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "tuan3_dac_trung_song.csv"))
    main(ap.parse_args())

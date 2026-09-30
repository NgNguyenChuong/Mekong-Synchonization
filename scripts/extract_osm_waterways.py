#!/usr/bin/env python
"""Loc song/kenh OSM (Geofabrik Viet Nam) cho DBSCL (T3-V1) - thay HydroRIVERS.

Ly do: HydroRIVERS sinh tu DEM 15" chi ve MOT nhanh dong chinh (song Hau), thieu song Tien
(kiem tra 2026-09-30 tren tuyen cat kinh do 105.95). OSM co ca song lon dang polygon va kenh dao.

Dau ra (GeoPackage, EPSG:4326):
  lines   : waterways fclass in (river, canal, stream) - cot fclass, name, width
  polygons: vung nuoc (gis_osm_water_a fclass in (river, water))

Chay:  python scripts/extract_osm_waterways.py --zip <vietnam-YYMMDD-free.shp.zip> [--out ...gpkg]
"""
import argparse
import os
import sys
import zipfile

import geopandas as gpd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import CANONICAL_BOUNDARY  # noqa: E402

LINE_CLASSES = ["river", "canal", "stream"]
POLY_CLASSES = ["river", "water"]


def _layer(zip_path, stem, region):
    with zipfile.ZipFile(zip_path) as z:
        shp = next(n for n in z.namelist() if n.endswith(f"{stem}.shp"))
    gdf = gpd.read_file(f"zip://{zip_path}!{shp}", bbox=tuple(region.bounds))
    return gdf[gdf.intersects(region)].reset_index(drop=True)


def main(args):
    boundary = gpd.read_file(args.boundary).to_crs(4326)
    region = boundary.to_crs(32648).buffer(args.buffer_km * 1000).to_crs(4326).union_all()

    lines = _layer(args.zip, "gis_osm_waterways_free_1", region)
    lines = lines[lines["fclass"].isin(LINE_CLASSES)][["osm_id", "fclass", "name", "width", "geometry"]]
    polys = _layer(args.zip, "gis_osm_water_a_free_1", region)
    polys = polys[polys["fclass"].isin(POLY_CLASSES)][["osm_id", "fclass", "name", "geometry"]]

    lines.to_file(args.out, layer="lines", driver="GPKG")
    polys.to_file(args.out, layer="polygons", driver="GPKG")

    km = lines.to_crs(32648).length / 1000
    print(lines.assign(km=km).groupby("fclass").agg(doan=("osm_id", "size"), km=("km", "sum")).round(0).to_string())
    area = polys.to_crs(32648).area / 1e6
    print(polys.assign(km2=area).groupby("fclass").agg(vung=("osm_id", "size"), km2=("km2", "sum")).round(1).to_string())
    for key in ["Tiền", "Hậu", "Cổ Chiên", "Hàm Luông", "Vàm Cỏ"]:
        hit_l = lines["name"].fillna("").str.contains(key)
        hit_p = polys["name"].fillna("").str.contains(key)
        print(f"  ten chua '{key}': {int(hit_l.sum())} doan, {int(hit_p.sum())} vung")
    print(f"Da ghi: {args.out} (layer lines, polygons)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--buffer-km", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw", "rivers", "osm_waterways_mekong_delta.gpkg"))
    main(ap.parse_args())

#!/usr/bin/env python
"""Tai HydroRIVERS v1.0 (chau A) va cat theo ranh gioi DBSCL (T3-V1).

Giu cac truong can cho dac trung: HYRIV_ID, ORD_STRA, ORD_FLOW, DIS_AV_CMS, UPLAND_SKM.
Luu y: HydroRIVERS sinh tu DEM 15" nen o vung chau tho phang, kenh dao nhan tao
khong co; can doi chieu voi song Tien/Hau truoc khi dung khoang cach toi song.

Chay:  python scripts/fetch_hydrorivers.py [--out data/raw/rivers/hydrorivers_mekong_delta.gpkg]
"""
import argparse
import os
import sys
import urllib.request
import zipfile

import geopandas as gpd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import CANONICAL_BOUNDARY  # noqa: E402

URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip"
KEEP = ["HYRIV_ID", "NEXT_DOWN", "ORD_STRA", "ORD_FLOW", "DIS_AV_CMS", "UPLAND_SKM", "LENGTH_KM", "geometry"]


def download(zip_path):
    if os.path.exists(zip_path):
        print(f"[co san] {zip_path}")
        return
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    print(f"[tai] {URL}")
    # Server tra 403 voi User-Agent mac dinh cua urllib.
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(zip_path + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(zip_path + ".part", zip_path)


def main(args):
    zip_path = os.path.join(os.path.dirname(args.out), os.path.basename(URL))
    download(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp"))
    boundary = gpd.read_file(args.boundary).to_crs(4326)
    # Dem 5 km quanh ranh gioi de khoang cach toi song o bien khong bi cat cut.
    region = boundary.to_crs(32648).buffer(args.buffer_km * 1000).to_crs(4326).union_all()
    rivers = gpd.read_file(f"zip://{zip_path}!{shp}", bbox=tuple(region.bounds))
    rivers = rivers[rivers.intersects(region)][KEEP].reset_index(drop=True)
    rivers.to_file(args.out, driver="GPKG")

    total_km = rivers["LENGTH_KM"].sum()
    print(f"{len(rivers)} doan song, {total_km:.0f} km trong vung dem {args.buffer_km:g} km")
    print(rivers.groupby("ORD_STRA").agg(doan=("HYRIV_ID", "size"), km=("LENGTH_KM", "sum"),
                                         Q_max=("DIS_AV_CMS", "max")).round(1).to_string())
    print(f"Da ghi: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--buffer-km", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw", "rivers", "hydrorivers_mekong_delta.gpkg"))
    main(ap.parse_args())

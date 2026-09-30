#!/usr/bin/env python
"""Lay du lieu tu Google Earth Engine cho DBSCL (T2-V2, mo rong den 2026).

Lenh con:
  era5    ERA5-Land daily -> 1 file/thang/bien, band k = ngay k, ten `<bien>_YYYY_MM.tif`
          (dung quy uoc `utils_h3.parse_year_month` + `processing.extract_generic`).
          Tai truc tiep (anh nho, ~0.1 do).
  static  Copernicus DEM GLO-30 (ban 2024_1) + ESA WorldCover v200 (2021) -> Export len Google Drive.
  labels  Tai tao nhan man mua kho theo DUNG script README Zenodo 15653696:
          Landsat 8 C2 L2, 01/11/(N-1)..30/04/N, mat na QA_PIXEL bit 0-4 + QA_RADSAT,
          median, Salinity = 28.013*exp(-13.39*SR_B5), NDWIchen = ND(SR_B5, SR_B7).
          --sensors l8l9 them Landsat 9 (khac tac gia; chi dung sau khi da doi chieu).
          -> Export len Google Drive (moi nam ~380 MB).

Can: pip install earthengine-api ; earthengine authenticate ; mot Cloud project da bat EE API.
Chay:  python scripts/gee_fetch.py era5 --project <id> --start 2000-01 --end 2026-08 --out <RAW_DIR>
       python scripts/gee_fetch.py static --project <id>
       python scripts/gee_fetch.py labels --project <id> --years 2020 2021 2022 2023
"""
import argparse
import json
import os
import sys
import urllib.request

import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import CANONICAL_BOUNDARY  # noqa: E402

# bien -> (thu muc theo DEFAULT_DATA_SPECS, ham tao anh ngay tu 1 anh ERA5-Land daily)
ERA5_VARS = {
    "rain": ("daily_rain", lambda im: im.select("total_precipitation_sum").multiply(1000)),          # m -> mm
    "temp_avg": ("daily_temp_avg", lambda im: im.select("temperature_2m").subtract(273.15)),
    "temp_max": ("daily_temp_max", lambda im: im.select("temperature_2m_max").subtract(273.15)),
    "temp_min": ("daily_temp_min", lambda im: im.select("temperature_2m_min").subtract(273.15)),
    "solar": ("daily_solar", lambda im: im.select("surface_solar_radiation_downwards_sum").divide(1e6)),  # J -> MJ/m2
    "humidity": ("daily_humid", None),  # tinh tu nhiet do + diem suong (Magnus), xem _rh
}
ERA5_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_SCALE_M = 11132  # 0.1 do


def _init(project):
    import ee
    ee.Initialize(project=project or os.getenv("EE_PROJECT"))
    return ee


def _region(ee, boundary_path, buffer_km):
    b = gpd.read_file(boundary_path).to_crs(32648).buffer(buffer_km * 1000).to_crs(4326)
    return ee.Geometry(json.loads(gpd.GeoSeries([b.union_all()], crs=4326).to_json())["features"][0]["geometry"])


def _rh(ee, im):
    # RH trung binh ngay xap xi tu T va Td trung binh ngay (Magnus, Alduchov & Eskridge 1996).
    t = im.select("temperature_2m").subtract(273.15)
    td = im.select("dewpoint_temperature_2m").subtract(273.15)
    es = t.expression("exp(17.625*T/(243.04+T))", {"T": t})
    ed = td.expression("exp(17.625*T/(243.04+T))", {"T": td})
    return ed.divide(es).multiply(100).min(100)


def _download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(path + ".part", "wb") as f:
        f.write(r.read())
    os.replace(path + ".part", path)


def cmd_era5(args):
    ee = _init(args.project)
    region = _region(ee, args.boundary, args.buffer_km)
    months = pd.period_range(args.start, args.end, freq="M")
    variables = args.vars or list(ERA5_VARS)
    for var in variables:
        folder, fn = ERA5_VARS[var]
        os.makedirs(os.path.join(args.out, folder), exist_ok=True)
    for p in months:
        start, end = p.start_time.strftime("%Y-%m-%d"), (p.end_time + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        coll = ee.ImageCollection(ERA5_ID).filterDate(start, end).sort("system:time_start")
        n = coll.size().getInfo()
        if n != p.days_in_month:
            print(f"[bo qua] {p}: {n}/{p.days_in_month} ngay (du lieu chua du)")
            continue
        for var in variables:
            folder, fn = ERA5_VARS[var]
            path = os.path.join(args.out, folder, f"{var}_{p.year}_{p.month:02d}.tif")
            if os.path.exists(path):
                continue
            # Bien: ERA5-Land ghi solar = 0 (khong phai NoData) o pixel bien -> keo trung binh o ven bien.
            # Dung mat na dat cua temperature_2m cho moi bien.
            daily = coll.map(lambda im: (_rh(ee, im) if fn is None else fn(im))
                             .updateMask(im.select("temperature_2m").mask()).float())
            img = daily.toBands().rename([f"b{i + 1}" for i in range(n)])
            url = img.getDownloadURL({"region": region, "scale": ERA5_SCALE_M, "crs": "EPSG:4326",
                                      "format": "GEO_TIFF"})
            _download(url, path)
        print(f"[xong] {p}")


def _export(ee, image, name, region, scale, folder):
    task = ee.batch.Export.image.toDrive(image=image, description=name, folder=folder, fileNamePrefix=name,
                                         region=region, scale=scale, crs="EPSG:32648", maxPixels=1e13)
    task.start()
    print(f"[export] {name} -> Drive/{folder} (task {task.id})")


def cmd_static(args):
    ee = _init(args.project)
    region = _region(ee, args.boundary, args.buffer_km)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(region).select("DEM").mosaic().float()
    _export(ee, dem.clip(region), "DEM_DBSCL", region, 30, args.drive_folder)
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").uint8()
    _export(ee, wc.clip(region), "LandCover_DBSCL_2021", region, 10, args.drive_folder)


def _mask_oli(image):
    qa = image.select("QA_PIXEL").bitwiseAnd(int("11111", 2)).eq(0)
    sat = image.select("QA_RADSAT").eq(0)
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True).updateMask(qa).updateMask(sat)


def cmd_labels(args):
    ee = _init(args.project)
    region = _region(ee, args.boundary, 0)
    ids = {"l8": ["LANDSAT/LC08/C02/T1_L2"], "l8l9": ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"]}[args.sensors]
    for year in args.years:
        coll = ee.ImageCollection(ids[0])
        for extra in ids[1:]:
            coll = coll.merge(ee.ImageCollection(extra))
        coll = coll.filterDate(f"{year - 1}-11-01", f"{year}-04-30").filterBounds(region).map(_mask_oli)
        print(f"{year}: {coll.size().getInfo()} canh ({args.sensors})")
        comp = coll.median().clip(region)
        ndwi = comp.normalizedDifference(["SR_B5", "SR_B7"]).rename("NDWIchen")
        sal = comp.expression("28.013 * exp(-13.39 * NIR)", {"NIR": comp.select("SR_B5")}).rename("Salinity").float()
        _export(ee, ndwi.addBands(sal), f"{year}_MD_dry_NDWIchen_Salinity_{args.sensors}", region, 30,
                args.drive_folder)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=None, help="Cloud project co bat Earth Engine (hoac bien EE_PROJECT)")
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("era5")
    e.add_argument("--start", default="1999-11")
    e.add_argument("--end", default="2026-08")
    e.add_argument("--vars", nargs="+", choices=list(ERA5_VARS))
    e.add_argument("--buffer-km", type=float, default=15.0)
    e.add_argument("--out", default=os.getenv("RAW_DIR") or os.path.join(ROOT, "data", "raw"))
    s = sub.add_parser("static")
    s.add_argument("--buffer-km", type=float, default=5.0)
    s.add_argument("--drive-folder", default="NCKH_GEE")
    lb = sub.add_parser("labels")
    lb.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023])
    lb.add_argument("--sensors", choices=["l8", "l8l9"], default="l8")
    lb.add_argument("--drive-folder", default="NCKH_GEE")
    a = ap.parse_args()
    {"era5": cmd_era5, "static": cmd_static, "labels": cmd_labels}[a.cmd](a)

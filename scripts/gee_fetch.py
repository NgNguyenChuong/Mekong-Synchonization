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
  chirps3 Nhan MUA Dot 7: tong mua kho 01/11/(s-1)..30/04/s tu CHIRPS v3 PENTAD (GEE khong co ban thang;
          ban thang CHC = tong 6 pentad) -> 1 file/mua `chirps3_rain_<s>.tif`, luoi goc 0,05 do, tai truc tiep
          vao <DATA_ROOT>/raw/chirps3/ + bang mo ta chirps3_season_summary.csv.
  mcd18   Nhan BUC XA Dot 7: TB mua DSR ngay (TB 8 band 3 gio, W/m2) tu MODIS/062/MCD18A1, 01/11..30/04
          -> 1 file/mua `mcd18a1_dsr_<s>.tif` (dsr_mean, n_days_valid, frac_quality2, dsr_mean_no_dec),
          luoi goc sinusoidal ~926 m, tai truc tiep vao <DATA_ROOT>/raw/mcd18a1/ + mcd18a1_season_summary.csv.

Can: pip install earthengine-api ; earthengine authenticate ; mot Cloud project da bat EE API.
Chay:  python scripts/gee_fetch.py era5 --project <id> --start 2000-01 --end 2026-08 --out <RAW_DIR>
       python scripts/gee_fetch.py static --project <id>
       python scripts/gee_fetch.py labels --project <id> --years 2020 2021 2022 2023
       python scripts/gee_fetch.py chirps3 --years 2014 2015 ... 2026      (mac dinh 2014-2026; EE_PROJECT tu .env)
       python scripts/gee_fetch.py mcd18 --years 2014 2015 ... 2026
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request

import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import settings  # noqa: E402,F401  nap .env (EE_PROJECT, GEE_DRIVE_FOLDER)
import met_labels as ml  # noqa: E402
from preprocessing import CANONICAL_BOUNDARY, write_provenance  # noqa: E402

# Export.toDrive chi nhan TEN thu muc: neu co thu muc trung ten o bat ky cap nao thi ghi vao do
# (trung nhieu -> thu muc sua gan nhat), khong co thi tao moi o goc Drive. "GEE" = NCKH_Source/GEE
# tren Drive cua tai khoan GEE (da kiem 2026-10-03). Khong tao them thu muc nao ten "GEE" khac tren Drive.
DRIVE_FOLDER = os.getenv("GEE_DRIVE_FOLDER", "GEE")  # .env

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
# Luoi goc ERA5-Land (tam pixel o boi so 0,1 do). KHONG dung `scale=11132`: luoi xuat khi do co tam pixel
# roi sat mep pixel goc -> nearest chon pixel Dong-Bac, file lech nua pixel (+0,05 E / +0,05 N) so voi
# toa do ghi (known-pitfalls 1a; file cu da sua header bang scripts/fix_era5_transform.py).
ERA5_CRS_TRANSFORM = [0.1, 0, -180.05, 0, -0.1, 90.05]


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


def _retry(fn, what, tries=6, wait=10):
    """Thu lai khi loi mang/server tam thoi (GEE 503, mat ket noi); cho tang dan 10, 20, 40... giay."""
    for k in range(tries):
        try:
            return fn()
        except Exception as exc:  # EEException, HTTPError, URLError, ConnectionError, TimeoutError
            if k == tries - 1:
                raise
            print(f"[thu lai {k + 1}/{tries - 1}] {what}: {str(exc)[:120]} -> cho {wait * 2 ** k}s", flush=True)
            time.sleep(wait * 2 ** k)


def _download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(path + ".part", "wb") as f:
        f.write(r.read())
    os.replace(path + ".part", path)


def _era5_region(ee, args):
    """Vung tai ERA5: neu da co file (luoi goc) trong --out thi dung DUNG khung cua no (thu nho 0,001 do de
    khong an them pixel ria) -> moi thang cung luoi pixel (era5_season.aggregate_season bat buoc); chua co
    file thi dung ranh gioi + buffer."""
    import glob
    import rasterio
    from era5_georef import is_native_aligned
    for path in sorted(glob.glob(os.path.join(args.out, "daily_temp_avg", "*.tif")))[:1]:
        with rasterio.open(path) as ds:
            if not is_native_aligned(ds.transform):
                raise SystemExit(f"{path} chua tren luoi goc - chay scripts/fix_era5_transform.py truoc")
            b, e = ds.bounds, 1e-3
        print(f"[vung] theo khung file co san {os.path.basename(path)}: {tuple(round(v, 3) for v in b)}", flush=True)
        return ee.Geometry.Rectangle([b.left + e, b.bottom + e, b.right - e, b.top - e], "EPSG:4326", False)
    return _region(ee, args.boundary, args.buffer_km)


def cmd_era5(args):
    ee = _init(args.project)
    region = _era5_region(ee, args)
    months = pd.period_range(args.start, args.end, freq="M")
    variables = args.vars or list(ERA5_VARS)
    for var in variables:
        folder, fn = ERA5_VARS[var]
        os.makedirs(os.path.join(args.out, folder), exist_ok=True)
    for p in months:
        start, end = p.start_time.strftime("%Y-%m-%d"), (p.end_time + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        coll = ee.ImageCollection(ERA5_ID).filterDate(start, end).sort("system:time_start")
        n = _retry(lambda: coll.size().getInfo(), f"dem so ngay {p}")
        if n != p.days_in_month:
            print(f"[bo qua] {p}: {n}/{p.days_in_month} ngay (du lieu chua du)", flush=True)
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
            params = {"region": region, "crs": "EPSG:4326", "crs_transform": ERA5_CRS_TRANSFORM, "format": "GEO_TIFF"}
            _retry(lambda: _download(img.getDownloadURL(params), path), f"{var} {p}")
        print(f"[xong] {p}", flush=True)


def _export(ee, image, name, region, scale, folder):
    task = ee.batch.Export.image.toDrive(image=image, description=name, folder=folder, fileNamePrefix=name,
                                         region=region, scale=scale, crs="EPSG:32648", maxPixels=1e13)
    task.start()
    print(f"[export] {name} -> Drive/{folder} (task {task.id})")


def _export_grid(ee, image, name, region, crs_transform, folder):
    """Xuat theo LUOI CO DINH (crsTransform) thay vi scale - tranh lech nua pixel (known-pitfalls 1a)."""
    task = ee.batch.Export.image.toDrive(image=image, description=name, folder=folder, fileNamePrefix=name,
                                         region=region, crs="EPSG:32648", crsTransform=crs_transform,
                                         maxPixels=1e13)
    task.start()
    print(f"[export] {name} -> Drive/{folder} (task {task.id})")


# Luoi cua file da tai (goc trai tren, EPSG:32648) - xuat lai PHAI trung luoi nay
WC_GRID = [10, 0, 367790, 0, -10, 1225100]
DEM_GRID = [30, 0, 367770, 0, -30, 1225110]


def cmd_static_v2(args):
    """WorldCover dem 25 km quanh ranh gioi (phu het o luoi tho ven ria); NoData ngoai khoi -> 80 (nuoc):
    WorldCover khong co tile o bien xa (da kiem 2026-10-03: 106,3E 8,9N -> None; gan bo -> 80).
    Them lop co WBM (Water Body Mask) cua Copernicus DEM: DEM chi coi la thieu khi DEM == 0 VA WBM > 0 (CHG-09);
    chi WBM > 0 thi xoa nham 62% dat thap 0,05-1 m (co ho)."""
    ee = _init(args.project)
    wc_region = _region(ee, args.boundary, args.wc_buffer_km)
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").unmask(80).uint8()
    _export_grid(ee, wc.clip(wc_region), "LandCover_DBSCL_2021_v2", wc_region, WC_GRID, args.drive_folder)
    dem_region = _region(ee, args.boundary, args.dem_buffer_km)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(dem_region)
    flags = dem.select(["WBM", "EDM"]).mosaic().uint8()
    _export_grid(ee, flags.clip(dem_region), "DEM_DBSCL_flags_v2", dem_region, DEM_GRID, args.drive_folder)


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
        _export(ee, ndwi.addBands(sal), f"{year}_MD_dry_NDWIchen_Salinity_{args.sensors}{args.name_suffix}", region, 30,
                args.drive_folder)


def cmd_watermask(args):
    """Mask nuoc dong theo mua kho: cung bo canh, cung mat na may, cung khoang ngay voi nhan.
    Moi canh: MNDWI = ND(SR_B3, SR_B6) > 0 la nuoc. Xuat 2 band uint8:
      water_freq = % so lan quan sat khong may la nuoc (255 = khong co quan sat),
      n_clear    = so lan quan sat khong may.
    Chi dung de loai pixel nuoc khoi nhan, KHONG dung lam dac trung."""
    ee = _init(args.project)
    region = _region(ee, args.boundary, 0)
    for year in args.years:
        coll = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterDate(f"{year - 1}-11-01", f"{year}-04-30").filterBounds(region).map(_mask_oli))
        print(f"{year}: {coll.size().getInfo()} canh (l8)")
        water = coll.map(lambda im: im.normalizedDifference(["SR_B3", "SR_B6"]).gt(0).rename("water"))
        freq = water.mean().multiply(100).round().unmask(255).uint8().rename("water_freq")
        n = water.count().unmask(0).min(255).uint8().rename("n_clear")
        _export(ee, freq.addBands(n).clip(region), f"{year}_MD_dry_watermask_l8{args.name_suffix}", region, 30, args.drive_folder)


# ---------------------------------------------------------------- nhan khi tuong Dot 7 (mua, buc xa)
CHIRPS3_ID = "UCSB-CHC/CHIRPS/V3/PENTAD"
MCD18_ID = "MODIS/062/MCD18A1"


def _rect(boundary_path, buffer_km):
    """KHUNG chu nhat (EPSG:4326) cua ranh gioi + dem `buffer_km` (nhu cmd_era5) -> [xmin, ymin, xmax, ymax].
    Xuat ca khung (khong clip theo da giac) de pixel ria van co gia tri; pham vi ap sau khi trich."""
    b = gpd.read_file(boundary_path).to_crs(32648).buffer(buffer_km * 1000).to_crs(4326)
    return [round(float(v), 6) for v in b.total_bounds]


def _native_grid(ee, image, band):
    """(crs, crsTransform) cua LUOI GOC san pham -> xuat khong noi suy, khong lech nua pixel (pitfall 1a)."""
    p = _retry(lambda: image.select(band).projection().getInfo(), f"luoi goc {band}")
    return p["crs"], p["transform"]


def _finish_download(tmp, path, band_names):
    """File GEE: pixel bi mask ghi -inf/khong co tag nodata (pitfall 4d) -> doi moi gia tri khong huu han thanh NaN,
    gan ten band, nodata = NaN. Giu nguyen crs/transform/kich thuoc (khong noi suy)."""
    import numpy as np
    import rasterio
    with rasterio.open(tmp) as src:
        if src.count != len(band_names):
            raise RuntimeError(f"{tmp}: {src.count} band, can {len(band_names)} {band_names}")
        data = src.read().astype("float32")
        prof = src.profile.copy()
    data[~np.isfinite(data)] = np.nan
    prof.update(dtype="float32", nodata=np.nan, compress="deflate", driver="GTiff")
    part = path + ".part.tif"
    with rasterio.open(part, "w", **prof) as dst:
        dst.write(data)
        dst.descriptions = tuple(band_names)
    os.replace(part, path)
    os.remove(tmp)


def _get_tif(ee, image, rect, crs, transform, path, band_names, what):
    tmp = path + ".dl.tif"
    params = {"region": ee.Geometry.Rectangle(rect, "EPSG:4326", False), "crs": crs, "crs_transform": transform,
              "format": "GEO_TIFF"}
    _retry(lambda: _download(image.getDownloadURL(params), tmp), what)
    _finish_download(tmp, path, band_names)


def _iso_us(us):
    return (pd.Timestamp(int(us), unit="us")).isoformat(timespec="seconds")


def _write_summary(rows, path, boundary):
    df = pd.DataFrame(rows).sort_values("season")
    df.to_csv(path, index=False, float_format="%.6g")
    write_provenance(path, boundary, n_seasons=len(df), note="so mo ta trong vung: tam pixel trong ranh gioi")
    print(f"[tom tat] {path}", flush=True)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df.to_string(index=False), flush=True)


def cmd_chirps3(args):
    """Tong mua kho tu 36 pentad CHIRPS v3. Pentad NoData (mask hoac < 0, vd -9999) -> thang/mua do = NaN
    (khong cong thieu thanh 0). Band chan doan: rain_m11..m04, rain_jfm (mat na kho), n_pentad_valid, n_pentad_neg."""
    ee = _init(args.project)
    out_dir = args.out or settings.data_path("raw", "chirps3")
    os.makedirs(out_dir, exist_ok=True)
    rect = _rect(args.boundary, args.buffer_km)
    coll = ee.ImageCollection(CHIRPS3_ID)
    rows = []
    for year in args.years:
        path = os.path.join(out_dir, ml.output_name("chirps3", year))
        idx = ml.chirps_pentad_indices(year)
        if not (os.path.exists(path) and os.path.exists(path + ".provenance.json")):
            sub = coll.filter(ee.Filter.inList("system:index", idx))
            meta = _retry(lambda: ee.Dictionary({"i": sub.aggregate_array("system:index"),
                                                 "v": sub.aggregate_array("system:version")}).getInfo(),
                          f"meta chirps3 {year}")
            missing = ml.missing_indices(idx, meta["i"])
            if missing:
                print(f"[bo qua] mua {year}: thieu {len(missing)}/36 pentad {missing[:6]}", flush=True)
                continue
            version = dict(zip(meta["i"], meta["v"]))
            status = {i: ml.chirps_status_from_version(i, version[i]) for i in idx}
            crs, transform = _native_grid(ee, sub.first(), "precipitation")

            def valid(im):
                p = im.select("precipitation")
                return p.updateMask(p.gte(0))

            months = {}
            for (y, m) in ml.season_year_months(year):
                mi = [ml.chirps_pentad_index(y, m, d) for d in ml.PENTAD_START_DAYS]
                s = sub.filter(ee.Filter.inList("system:index", mi)).map(valid)
                months[m] = s.sum().updateMask(s.count().eq(len(mi)))   # thieu pentad -> thang NaN
            vcoll = sub.map(valid)
            n_valid = vcoll.count().unmask(0)
            n_neg = sub.map(lambda im: im.select("precipitation").lt(0)).sum().unmask(0)
            total = months[11]
            for m in (12, 1, 2, 3, 4):
                total = total.add(months[m])
            total = total.updateMask(n_valid.eq(len(idx)))
            img = ee.Image.cat([total, months[11], months[12], months[1], months[2], months[3], months[4],
                                months[1].add(months[2]).add(months[3]), n_valid, n_neg]).toFloat()
            img = img.rename(list(ml.CHIRPS_BANDS))
            _get_tif(ee, img, rect, crs, transform, path, ml.CHIRPS_BANDS, f"chirps3 {year}")
            start, end = ml.met_season_window(year)
            n_final = sum(v == "final" for v in status.values())
            write_provenance(
                path, args.boundary, asset=CHIRPS3_ID, product="CHIRPS v3.0 PENTAD (GEE; khong co ban THANG tren GEE,"
                " ban thang CHC = tong 6 pentad)", variable="precipitation (mm/pentad) -> tong mm",
                query_date=pd.Timestamp.now().date().isoformat(), season=year,
                window=[start.isoformat(), end.isoformat()], window_rule="01/11/(s-1)..30/04/s gom 2 dau (An chot"
                " 2026-10-06 16:40)", n_images=len(idx), pentads=[idx[0], idx[-1]],
                ingest_version_min=_iso_us(min(version.values())), ingest_version_max=_iso_us(max(version.values())),
                final_prelim=("final" if n_final == len(idx) else f"CO PRELIM: {len(idx) - n_final}/36"),
                final_prelim_rule=f"suy luan tu system:version >= 01/(M+1) + {ml.FINAL_MIN_LAG_DAYS} ngay; GEE"
                " khong ghi thuoc tinh final/prelim", prelim_pentads=[i for i, s in status.items() if s != "final"],
                crs=crs, crs_transform=transform, region_rect_4326=rect, buffer_km=args.buffer_km,
                bands=list(ml.CHIRPS_BANDS), nodata="NaN", aggregation="tong; pentad mask hoac < 0 -> thang/mua NaN")
            print(f"[xong] chirps3 {year}: {n_final}/36 pentad final (suy luan)", flush=True)
        prov = json.load(open(path + ".provenance.json", encoding="utf-8"))
        rows.append({"season": year, "final_prelim": prov.get("final_prelim"),
                     **ml.chirps_season_summary(path, args.boundary)})
    if rows:
        _write_summary(rows, os.path.join(out_dir, "chirps3_season_summary.csv"), args.boundary)


def _mcd18_day(ee, im):
    """DSR ngay = TB 8 band 3 gio (W/m2); ngay hop le tai pixel khi DU 8 band (dem = 0, khong bi mask).
    q2 = 1 neu DSR_Quality bit 0-1 = 2 (phan xa be mat lay tu khi hau); thieu band chat luong -> 0."""
    g = im.select(list(ml.MCD18_GMT_BANDS))
    ok = g.mask().reduce(ee.Reducer.min()).gt(0)
    dsr = g.reduce(ee.Reducer.sum()).divide(len(ml.MCD18_GMT_BANDS)).updateMask(ok).rename("dsr")
    q2 = (im.select("DSR_Quality").bitwiseAnd(ml.DSR_QUALITY_MASK).eq(ml.DSR_QUALITY_CLIM)
          .unmask(0).updateMask(ok).rename("q2"))
    return dsr.addBands(q2).set("system:index", im.get("system:index"))


def cmd_mcd18(args):
    ee = _init(args.project)
    out_dir = args.out or settings.data_path("raw", "mcd18a1")
    os.makedirs(out_dir, exist_ok=True)
    rect = _rect(args.boundary, args.buffer_km)
    region = ee.Geometry.Rectangle(rect, "EPSG:4326", False)
    coll = ee.ImageCollection(MCD18_ID)
    rows = []
    for year in args.years:
        path = os.path.join(out_dir, ml.output_name("mcd18a1", year))
        idx = ml.mcd18_indices(year)
        idx_nd = ml.mcd18_indices(year, exclude_months=(12,))
        if not (os.path.exists(path) and os.path.exists(path + ".provenance.json")):
            sub = coll.filter(ee.Filter.inList("system:index", idx))
            crs, transform = _native_grid(ee, sub.first(), "GMT_0000_DSR")
            daily = sub.map(lambda im: _mcd18_day(ee, im))
            # So pixel hop le tung ngay trong khung (anh co ma pixel khong co gia tri cung bi phat hien)
            per_day = daily.map(lambda d: ee.Feature(None, {
                "i": d.get("system:index"),
                "n": ee.Dictionary(d.select("dsr").reduceRegion(ee.Reducer.count(), region, crs=crs,
                                                                crsTransform=transform, maxPixels=1e9)).get("dsr", 0)}))
            meta = _retry(lambda: ee.Dictionary({"i": per_day.aggregate_array("i"),
                                                 "n": per_day.aggregate_array("n")}).getInfo(), f"meta mcd18 {year}")
            avail = meta["i"]
            missing = ml.missing_indices(idx, avail)
            empty = [i for i, n in zip(meta["i"], meta["n"]) if not n]
            nd = daily.filter(ee.Filter.inList("system:index", idx_nd)).select("dsr")
            d = daily.select("dsr")
            n = d.count().unmask(0)
            img = ee.Image.cat([d.mean(), n, daily.select("q2").sum().divide(n).updateMask(n.gt(0)),
                                nd.mean()]).toFloat().rename(list(ml.MCD18_BANDS))
            _get_tif(ee, img, rect, crs, transform, path, ml.MCD18_BANDS, f"mcd18 {year}")
            start, end = ml.met_season_window(year)
            to_dates = [ml.parse_mcd18_index(i) for i in missing]
            write_provenance(
                path, args.boundary, asset=MCD18_ID, product="MCD18A1 Collection 6.2 (062), Terra+Aqua, ngay, ~1 km",
                variable="DSR ngay = TB(GMT_0000..GMT_2100_DSR) W/m2", units="W/m2",
                query_date=pd.Timestamp.now().date().isoformat(), season=year,
                window=[start.isoformat(), end.isoformat()], window_rule="01/11/(s-1)..30/04/s gom 2 dau (An chot"
                " 2026-10-06 16:40)", n_days_expected=len(idx), n_images=len(avail),
                n_days_expected_no_dec=len(idx_nd), n_images_no_dec=len(set(idx_nd) & set(avail)),
                missing_dates=ml.date_runs(to_dates),
                images_without_valid_px=[ml.parse_mcd18_index(i).isoformat() for i in empty],
                valid_px_per_image_min=min(int(v or 0) for v in meta["n"]) if meta["n"] else None,
                valid_px_per_image_max=max(int(v or 0) for v in meta["n"]) if meta["n"] else None,
                final_prelim="khong ap dung (MODIS 062 la ban xu ly chinh thuc, khong co prelim)",
                crs=crs, crs_transform=transform, region_rect_4326=rect, buffer_km=args.buffer_km,
                bands=list(ml.MCD18_BANDS), nodata="NaN",
                aggregation="dsr_mean = TB tren ngay hop le (du 8 band); n_days_valid = so ngay hop le tai pixel;"
                " frac_quality2 = ty le ngay hop le co DSR_Quality&3 == 2; dsr_mean_no_dec = TB tren ngay hop le"
                " khong thuoc thang 12")
            print(f"[xong] mcd18 {year}: {len(avail)}/{len(idx)} ngay co anh; thieu {ml.date_runs(to_dates)};"
                  f" anh rong {len(empty)}", flush=True)
        prov = json.load(open(path + ".provenance.json", encoding="utf-8"))
        rows.append({"season": year, "n_days_expected": prov["n_days_expected"], "n_images": prov["n_images"],
                     "missing_dates": ";".join(prov["missing_dates"]),
                     "n_images_without_valid_px": len(prov["images_without_valid_px"]),
                     "images_without_valid_px": ";".join(ml.date_runs(
                         [dt.date.fromisoformat(d) for d in prov["images_without_valid_px"]])),
                     "n_days_with_data": prov["n_images"] - len(prov["images_without_valid_px"]),
                     **ml.mcd18_season_summary(path, args.boundary)})
    if rows:
        _write_summary(rows, os.path.join(out_dir, "mcd18a1_season_summary.csv"), args.boundary)


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
    s.add_argument("--drive-folder", default=DRIVE_FOLDER)
    s2 = sub.add_parser("static_v2")
    s2.add_argument("--wc-buffer-km", type=float, default=25.0)
    s2.add_argument("--dem-buffer-km", type=float, default=5.0)
    s2.add_argument("--drive-folder", default=DRIVE_FOLDER)
    lb = sub.add_parser("labels")
    lb.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023])
    lb.add_argument("--sensors", choices=["l8", "l8l9"], default="l8")
    lb.add_argument("--drive-folder", default=DRIVE_FOLDER)
    lb.add_argument("--name-suffix", default="_v2",
                    help="hau to ten file; mac dinh _v2 vi --boundary mac dinh la v2 (khong hau to = ban cu theo v1)")
    wm = sub.add_parser("watermask")
    wm.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2027)))
    wm.add_argument("--drive-folder", default=DRIVE_FOLDER)
    wm.add_argument("--name-suffix", default="_v2",
                    help="hau to ten file; mac dinh _v2 vi --boundary mac dinh la v2 (khong hau to = ban cu theo v1)")
    for name, sub_dir in (("chirps3", "chirps3"), ("mcd18", "mcd18a1")):
        sp = sub.add_parser(name)
        sp.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2027)), help="mua kho s (01/11/s-1..30/04/s)")
        sp.add_argument("--buffer-km", type=float, default=15.0, help="dem quanh ranh gioi truoc khi lay khung (nhu era5)")
        sp.add_argument("--out", default=None, help=f"mac dinh <DATA_ROOT>/raw/{sub_dir}")
    a = ap.parse_args()
    {"era5": cmd_era5, "static": cmd_static, "static_v2": cmd_static_v2, "labels": cmd_labels, "watermask": cmd_watermask,
     "chirps3": cmd_chirps3, "mcd18": cmd_mcd18}[a.cmd](a)

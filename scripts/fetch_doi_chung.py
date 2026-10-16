#!/usr/bin/env python
"""Tai tho du lieu vung doi chung (CHG-32/33): ranh gioi, ERA5-Land, CHIRPS v3, MCD18A1, Landsat 8 (NDWI + mask nuoc),
DEM, WorldCover, JRC GSW, tram NOAA/USGS. Chi tai + kiem, KHONG dung nhan/dac trung. Chay lai bo qua file da co.

Chay: python scripts/fetch_doi_chung.py --region mississippi [--sources boundary era5 chirps mcd18 landsat static stations]
      python scripts/fetch_doi_chung.py --region mississippi check
Goc ra: <DATA_ROOT>/doi_chung/<region> (doi bang DOI_CHUNG_ROOT neu o A day).
"""
import argparse
import concurrent.futures as cf
import glob
import io
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import settings  # noqa: E402
import gee_fetch as gf  # noqa: E402
import met_labels as ml  # noqa: E402
from preprocessing import file_sha256, write_provenance  # noqa: E402

UA = {"User-Agent": "NCKH-Mekong-research (academic)"}
SEASONS = list(range(2014, 2027))
TIGER = "TIGER/2018/States"
GAUL1 = "FAO/GAUL/2015/level1"
NILE_GOV = ["Alexandria", "Behera", "Kafr El-Shikh", "Gharbia", "Dakahlia", "Damietta", "Shrkia", "Port Said",
            "Menoufia", "Kalyoubia"]

REGIONS = {
    "mississippi": dict(utm=32615, label="MS", stations=True, crms=True,
                        src=f"{TIGER} (US Census, STUSPS=LA)", cut="giu phan vi do <= 31N (box lon -95..-87)"),
    "nile": dict(utm=32636, label="NL", stations=False,
                 src=f"{GAUL1} (ADM0=Egypt, ADM1 in {NILE_GOV})", cut="giu phan vi do >= 30.1N, kinh do 29-33E (dinh chau tho Cairo)"),
}
NOAA = {"8761724": "Grand Isle", "8762075": "Port Fourchon", "8764311": "Eugene Island", "8764227": "LAWMA Amerada Pass",
        "8768094": "Calcasieu Pass", "8761305": "Shell Beach", "8761927": "New Canal", "8761955": "Carrollton",
        "8760922": "Pilots Station East", "8766072": "Freshwater Canal Locks"}
USGS = {"07374000": "Mississippi Baton Rouge", "07374525": "Mississippi Belle Chasse",
        "07381490": "Atchafalaya Simmesport", "07381600": "Atchafalaya Morgan City"}
STATION_START, STATION_END = "2013-11-01", "2026-04-30"


class Ctx:
    def __init__(self, a):
        self.name = a.region
        self.cfg = REGIONS[a.region]
        self.utm = self.cfg["utm"]
        self.root = (a.root or os.getenv("DOI_CHUNG_ROOT") or settings.data_path("doi_chung")).rstrip("/") + "/" + self.name
        self.root = self.root.replace("\\", "/")
        self.boundary = self.root + "/boundary.geojson"
        self.project = a.project
        self.threads = a.threads
        self.static_buffer_km = getattr(a, "static_buffer_km", 0.0)
        os.makedirs(self.root, exist_ok=True)

    def p(self, *parts):
        d = "/".join([self.root, *parts[:-1]])
        os.makedirs(d, exist_ok=True)
        return d + "/" + parts[-1]


def log(msg):
    print(f"{pd.Timestamp.now():%H:%M:%S} {msg}", flush=True)


def free_gb(path):
    import shutil
    d = path
    while not os.path.exists(d):
        d = os.path.dirname(d)
    return shutil.disk_usage(d).free / 2 ** 30


# ------------------------------------------------------------------ ranh gioi
def src_boundary(ee, name):
    from shapely.geometry import box, shape
    from shapely.ops import unary_union
    if name == "mississippi":
        f = ee.FeatureCollection(TIGER).filter(ee.Filter.eq("STUSPS", "LA")).first()
        clip = box(-95, 27, -87, 31)
    else:
        fc = ee.FeatureCollection(GAUL1).filter(ee.Filter.eq("ADM0_NAME", "Egypt")).filter(
            ee.Filter.inList("ADM1_NAME", NILE_GOV))
        names = sorted(gf._retry(lambda: fc.aggregate_array("ADM1_NAME").getInfo(), "gaul"))
        if len(names) != len(NILE_GOV):
            raise RuntimeError(f"GAUL thieu/du tinh: {names}")
        f = ee.Feature(fc.geometry().dissolve(100))
        clip = box(29, 30.1, 33, 32.5)
    geom = gf._retry(lambda: f.geometry().getInfo(), "geometry")
    g = shape(geom).intersection(clip)
    polys = [p for p in getattr(g, "geoms", [g]) if p.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(polys)


def cmd_boundary(c):
    import geopandas as gpd
    if os.path.exists(c.boundary + ".provenance.json"):
        log("[boundary] da co")
        return
    ee = gf._init(c.project)
    g = src_boundary(ee, c.name)
    gdf = gpd.GeoDataFrame({"region": [c.name]}, geometry=[g], crs=4326)
    gdf.to_file(c.boundary, driver="GeoJSON")
    u = gdf.to_crs(c.utm)
    write_provenance(c.boundary, c.boundary, source=c.cfg["src"], cut=c.cfg["cut"], query_date=str(pd.Timestamp.now().date()),
                     crs_out="EPSG:4326", utm=f"EPSG:{c.utm}", area_km2=round(float(u.area.sum()) / 1e6, 1),
                     bounds_4326=[round(v, 5) for v in gdf.total_bounds], n_vertices=len(__import__("shapely").get_coordinates(g)))
    log(f"[boundary] {c.boundary} {float(u.area.sum()) / 1e6:.0f} km2")


# ------------------------------------------------------------------ ERA5 / CHIRPS / MCD18 (dung lai gee_fetch)
def _ns(c, **kw):
    return argparse.Namespace(project=c.project, boundary=c.boundary, utm=c.utm, buffer_km=15.0, **kw)


def cmd_era5(c):
    gf.cmd_era5(_ns(c, start="2013-09", end="2026-04", vars=None, out=c.p("raw", "x")[:-2]))


def cmd_chirps(c):
    gf.cmd_chirps3(_ns(c, years=SEASONS, out=c.p("raw", "chirps3", "x")[:-2]))


def cmd_mcd18(c):
    gf.cmd_mcd18(_ns(c, years=SEASONS, out=c.p("raw", "mcd18a1", "x")[:-2]))


# ------------------------------------------------------------------ tai theo o (tile) tu GEE
def grid_of(c, res, buf_m=1000):
    import geopandas as gpd
    b = gpd.read_file(c.boundary).to_crs(c.utm).buffer(buf_m).union_all()
    x0, y0, x1, y1 = b.bounds
    X0, Y1 = np.floor(x0 / res) * res, np.ceil(y1 / res) * res
    return b, X0, Y1, int(np.ceil((x1 - X0) / res)), int(np.ceil((Y1 - y0) / res))


def clip_geom(ee, c, b):
    from shapely.geometry import mapping
    s = b.simplify(100)
    return ee.Geometry(mapping(__import__("geopandas").GeoSeries([s], crs=c.utm).to_crs(4326).iloc[0]), geodesic=False)


def export_tiles(ee, c, make_img, outs, res, tile_px, what, meta, buf_m=1000):
    """make_img(tile_geom) -> anh nhieu band (float32). outs = [(path, [chi so band], dtype, ten band)].
    Ghi tung o vao file .part (khung 512 khop o), nho o da xong trong .done de chay lai; xong -> doi ten + provenance."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    from shapely.geometry import box
    if all(os.path.exists(o[0]) and os.path.exists(o[0] + ".provenance.json") for o in outs):
        return log(f"[bo qua] {what}")
    b, X0, Y1, W, H = grid_of(c, res, buf_m)
    if free_gb(outs[0][0]) < 1.0:
        raise RuntimeError(f"o dia con {free_gb(outs[0][0]):.1f} GB < 1 GB")
    aff = Affine(res, 0, X0, 0, -res, Y1)
    tiles = []
    for r in range(0, H, tile_px):
        for q in range(0, W, tile_px):
            w, h = min(tile_px, W - q), min(tile_px, H - r)
            if b.intersects(box(X0 + q * res, Y1 - (r + h) * res, X0 + (q + w) * res, Y1 - r * res)):
                tiles.append((q, r, w, h))
    done_path = outs[0][0] + ".done"
    done = set(map(tuple, json.load(open(done_path)))) if os.path.exists(done_path) else set()
    prof = dict(driver="GTiff", crs=f"EPSG:{c.utm}", transform=aff, width=W, height=H, tiled=True, blockxsize=512,
                blockysize=512, compress="deflate", BIGTIFF="IF_SAFER")
    for path, idx, dt, names in outs:
        if not os.path.exists(path + ".part"):
            done = set()
            kw = dict(prof, count=len(idx), dtype=dt, predictor=3 if dt == "float32" else 2)
            if dt == "float32":
                kw["nodata"] = float("nan")
            with rasterio.open(path + ".part", "w", **kw) as d:
                d.descriptions = tuple(names)
    todo = [t for t in tiles if t not in done]
    log(f"[{what}] {len(tiles)} o, con {len(todo)}")

    def get(t):
        q, r, w, h = t
        rect = [X0 + q * res, Y1 - (r + h) * res, X0 + (q + w) * res, Y1 - r * res]
        geom = ee.Geometry.Rectangle(rect, f"EPSG:{c.utm}", False)
        img = make_img(geom).unmask(-9999)
        tmp = outs[0][0] + f".t{q}_{r}.tif"
        params = {"region": geom, "crs": f"EPSG:{c.utm}", "crs_transform": [res, 0, rect[0], 0, -res, rect[3]],
                  "format": "GEO_TIFF"}
        gf._retry(lambda: gf._download(img.getDownloadURL(params), tmp), f"{what} o {t}", tries=7, wait=15)
        with rasterio.open(tmp) as s:   # clip() thu anh ve phan co du lieu -> dat lai vao o day du theo goc
            sub = s.read()
            dc, dr = (s.transform.c - rect[0]) / res, (rect[3] - s.transform.f) / res
            ok = dc == int(dc) and dr == int(dr) and s.transform.a == res
        os.remove(tmp)
        if not ok:
            raise RuntimeError(f"{what} o {t}: goc anh tai ve lech luoi")
        dc, dr = int(dc), int(dr)
        arr = np.full((sub.shape[0], h, w), -9999, dtype="float32")
        r0, c0 = max(dr, 0), max(dc, 0)
        r1, c1 = min(dr + sub.shape[1], h), min(dc + sub.shape[2], w)   # clip co the lan 1 pixel ra ngoai o
        arr[:, r0:r1, c0:c1] = sub[:, r0 - dr:r1 - dr, c0 - dc:c1 - dc]
        return t, arr

    t0 = time.time()
    with cf.ThreadPoolExecutor(c.threads) as ex:
        futs = [ex.submit(get, t) for t in todo]
        for i, fu in enumerate(cf.as_completed(futs), 1):
            try:
                t, arr = fu.result()   # loi 1 o -> dung nguon nay (cac o da xong van con trong .done)
            except Exception:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            q, r, w, h = t
            for path, idx, dt, names in outs:
                a = arr[idx]
                if dt == "float32":
                    a = np.where(np.isfinite(a) & (a > -9000), a, np.nan).astype("float32")
                else:
                    nd = {"water_freq": 255}
                    a = np.stack([np.where(np.isfinite(x) & (x >= 0), np.clip(x, 0, 255).round(), nd.get(nm, 0)) for x, nm in
                                  zip(a, names)]).astype("uint8")
                with rasterio.open(path + ".part", "r+") as d:
                    d.write(a, window=Window(q, r, w, h))
            done.add(t)
            json.dump(sorted(done), open(done_path, "w"))
            if i % 5 == 0 or i == len(todo):
                log(f"  {what}: {i}/{len(todo)} o, {time.time() - t0:.0f}s")
    for path, idx, dt, names in outs:
        os.replace(path + ".part", path)
        write_provenance(path, c.boundary, bands=names, crs=f"EPSG:{c.utm}", crs_transform=[res, 0, X0, 0, -res, Y1],
                         shape=[H, W], n_tiles=len(tiles), tile_px=tile_px, nodata="NaN" if dt == "float32" else "255=khong quan sat",
                         query_date=str(pd.Timestamp.now().date()), sha256=file_sha256(path), **meta)
    os.remove(done_path)
    log(f"[xong] {what} {sum(os.path.getsize(o[0]) for o in outs) / 2 ** 20:.0f} MB")


def _oli_coll(ee, year, geom, sensors="l8", snow=False):
    coll = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    if sensors == "l8l9":
        coll = coll.merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
    keep = ["SR_B2", "SR_B3", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL", "QA_RADSAT"]
    return (coll.filterDate(f"{year - 1}-11-01", f"{year}-04-30").filterBounds(geom).select(keep)
            .map(lambda im: gf._mask_oli(im, snow=snow)))


MIN_LAND_PX = 1000   # o 300 m; it hon -> khong du de ket luan ve canh


def scene_drop(stats, max_b3, min_b2b3):
    """stats: [{id, b3, b2}] (TB SR_B3/SR_B2 tren pixel sach TREN DAT; None = qua it pixel).
    Loai canh khi b3 > max_b3 VA b2/b3 > min_b2b3 (may/khoi: sang va xanh; sa mac: sang nhung do)."""
    out = []
    for s in stats:
        b3, b2 = s.get("b3"), s.get("b2")
        if not b3 or b2 is None:
            out.append(dict(id=s["id"], insufficient=True, dropped=False))
            continue
        r = b2 / b3
        if b3 > max_b3 and r > min_b2b3:
            out.append(dict(id=s["id"], b3=b3, b2=b2, b2b3=r, dropped=True,
                            reason=f"b3={b3:.3f}>{max_b3}; b2/b3={r:.2f}>{min_b2b3}"))
    return out


def scene_stats(ee, year, cg, sensors, scale=300):
    """Thong ke tung canh tren toan vung de moi o dung chung tap canh; chi pixel sach tren dat (WorldCover != 80)."""
    land = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").neq(80)
    red = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    def one(im):
        x = im.select(["SR_B2", "SR_B3"]).updateMask(land)
        r = x.reduceRegion(red, cg, scale, maxPixels=int(1e9), bestEffort=True)
        return ee.Feature(None, {"id": im.get("system:index"), "b3": r.get("SR_B3_mean"), "b2": r.get("SR_B2_mean"),
                                 "n": r.get("SR_B3_count")})
    fc = _oli_coll(ee, year, cg, sensors, snow=False).map(one)   # thong ke khong che bit 5: tuyet khong lam lech tieu chi loai canh
    rows = gf._retry(lambda: [f["properties"] for f in fc.getInfo()["features"]], f"scene_stats {year}")
    for r in rows:
        if (r.get("n") or 0) < MIN_LAND_PX:
            r["b3"] = r["b2"] = None
    return rows


def landsat_img(ee, year, geom, sensors="l8", drop_ids=None, snow=False):
    coll = _oli_coll(ee, year, geom, sensors, snow=snow)
    if drop_ids:
        coll = coll.filter(ee.Filter.inList("system:index", drop_ids).Not())
    ndwi = coll.median().normalizedDifference(["SR_B5", "SR_B7"]).rename("NDWIchen").unmask(-9999)
    water = coll.map(lambda im: im.normalizedDifference(["SR_B3", "SR_B6"]).gt(0).rename("water"))
    freq = water.mean().multiply(100).round().unmask(255).rename("water_freq")
    n = water.count().unmask(0).min(255).rename("n_clear")
    return ee.Image.cat([ndwi, freq, n]).toFloat()


def landsat_suffix(scene_filter, mask_snow, suffix=None):
    if suffix is not None:
        return suffix
    return ("_sf3" if scene_filter else "") + ("_snow" if mask_snow else "")


def cmd_landsat(c, sensors="l8", seasons=None, scene_filter=None, mask_snow=False, suffix=None):
    ee = gf._init(c.project)
    b = grid_of(c, 30)[0]
    cg = clip_geom(ee, c, b)
    sfx = sensors + landsat_suffix(scene_filter, mask_snow, suffix)
    snow = bool(mask_snow)
    for y in seasons or SEASONS:
        tag = f"{y}_{c.cfg['label']}_dry"
        outs = [(c.p("landsat", f"{tag}_NDWIchen_{sfx}.tif"), [0], "float32", ["NDWIchen"]),
                (c.p("landsat", f"{tag}_watermask_{sfx}.tif"), [1, 2], "uint8", ["water_freq", "n_clear"])]
        try:
            n = gf._retry(lambda: ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(f"{y - 1}-11-01", f"{y}-04-30")
                          .filterBounds(cg).size().getInfo(), f"dem canh {y}")
            drop, extra = None, dict(mask_snow=snow)
            if scene_filter:
                rows = scene_stats(ee, y, cg, sensors)
                res = scene_drop(rows, **scene_filter)
                drop = [d["id"] for d in res if d["dropped"]]
                extra.update(scene_filter=dict(**scene_filter, land_only=True, min_land_px=MIN_LAND_PX),
                             scene_stats_mask="bit0-4", scenes_dropped=[d for d in res if d["dropped"]],
                             scenes_stats=[dict(id=x["id"], b3=x.get("b3"), b2=x.get("b2"), n_px=x.get("n")) for x in rows],
                             scenes_insufficient=[d["id"] for d in res if d.get("insufficient")])
                log(f"[landsat {y}] scene-filter loai {len(drop)}/{n} canh, thieu pixel {len(extra['scenes_insufficient'])}")
            export_tiles(ee, c, lambda g: landsat_img(ee, y, g, sensors, drop, snow).clip(cg), outs, 30, 1536, f"landsat {y}",
                         dict(asset="LANDSAT/LC08/C02/T1_L2" + ("+LC09" if sensors == "l8l9" else ""), season=y,
                              window=[f"{y - 1}-11-01", f"{y}-04-30 (GEE bo ngay cuoi)"], n_scenes_l8=n,
                              product="SR C2 L2, mat na QA_PIXEL bit0-4 (+bit5 tuyet neu mask_snow) + QA_RADSAT, median; NDWIchen = ND(B5,B7);"
                              " water_freq = % MNDWI(B3,B6)>0 theo canh khong may, 255 = khong quan sat", **extra))
        except Exception:
            log(f"[LOI] landsat {y}: {traceback.format_exc(limit=2)}")


def export_wc(ee, c, lab, cg):
    """WorldCover trong ranh gioi (buffer 0, file cu) hoac + dem static_buffer_km (file _padNN, ngoai vung phu -> 80 nhu DBSCL v2)."""
    k = c.static_buffer_km
    pad = f"_pad{k:g}" if k > 0 else ""
    buf_m = 1000 + k * 1000
    g = clip_geom(ee, c, grid_of(c, 30, buf_m)[0]) if k > 0 else cg
    export_tiles(ee, c, lambda _: ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").unmask(80).toFloat().clip(g),
                 [(c.p("static", f"LandCover_{lab}_2021{pad}.tif"), [0], "uint8", ["Map"])], 10, 2304, "worldcover" + pad,
                 dict(asset="ESA/WorldCover/v200", buffer_km=k,
                      product="WorldCover 2021 trong ranh gioi" + (f" + dem {k:g} km" if k > 0 else "") + ", ngoai vung phu WorldCover -> 80 (nuoc)"),
                 buf_m=buf_m)


def cmd_static(c):
    ee = gf._init(c.project)
    b30 = grid_of(c, 30)[0]
    cg = clip_geom(ee, c, b30)
    lab = c.cfg["label"]
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1")

    def dem_img(g):
        m = dem.filterBounds(g)
        return ee.Image.cat([m.select("DEM").mosaic().unmask(-9999), m.select("WBM").mosaic().unmask(0),
                             m.select("EDM").mosaic().unmask(0)]).toFloat().clip(cg)

    steps = [("dem", lambda: export_tiles(ee, c, dem_img, [
        (c.p("static", f"DEM_{lab}.tif"), [0], "float32", ["DEM"]),
        (c.p("static", f"DEM_{lab}_flags.tif"), [1, 2], "uint8", ["WBM", "EDM"])], 30, 1536, "dem",
        dict(asset="COPERNICUS/DEM/GLO30_2024_1", product="DEM (m) + WBM/EDM; tai lai tu EPSG:4326 goc ve luoi UTM 30 m (nearest)"))),
        ("worldcover", lambda: export_wc(ee, c, lab, cg)),
        ("jrc", lambda: export_tiles(ee, c, lambda g: ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select(["occurrence", "max_extent"])
                                     .unmask(0).toFloat().clip(cg), [
            (c.p("static", f"JRC_GSW_{lab}.tif"), [0, 1], "uint8", ["occurrence", "max_extent"])], 30, 1536, "jrc",
            dict(asset="JRC/GSW1_4/GlobalSurfaceWater", product="occurrence (%) va max_extent (0/1), 1984-2021")))]
    for name, fn in steps:
        try:
            fn()
        except Exception:
            log(f"[LOI] static {name}: {traceback.format_exc(limit=2)}")


# ------------------------------------------------------------------ tram NOAA / USGS
def _get(url, tries=6):
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 200:
                return r.text
            if r.status_code == 400:   # NOAA: khong co du lieu trong khoang nay
                return ""
            err = f"HTTP {r.status_code}"
        except Exception as e:
            err = str(e)[:100]
        log(f"[thu lai {k + 1}] {url[:110]} {err}")
        time.sleep(10 * 2 ** k)
    raise RuntimeError(f"khong tai duoc {url}")


def _noaa_part(txt):
    """CSV NOAA -> DataFrame bo dong 'Error: ...'; None neu khong con dong du lieu (nam khong co du lieu)."""
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.startswith("Error")]
    if not lines or not lines[0].startswith("Date Time") or len(lines) < 2:
        return None
    return pd.read_csv(io.StringIO(chr(10).join(lines)))


def cmd_stations(c):
    if not c.cfg["stations"]:
        return log("[stations] vung khong co tram")
    out = c.p("stations", "x")[:-1]
    nodata_file = f"{out}NOAA_khong_co_du_lieu.json"
    nodata = json.load(open(nodata_file, encoding="utf-8")) if os.path.exists(nodata_file) else {}
    for sid, nm in NOAA.items():
        path = f"{out}noaa_{sid}_hourly_msl.csv"
        if os.path.exists(path + ".provenance.json"):
            continue
        try:
            meta = json.loads(_get(f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{sid}.json"))["stations"][0]
            urls, parts, no_years = [], [], []
            for y in range(2013, 2027):
                b, e = max(f"{y}0101", STATION_START.replace("-", "")), min(f"{y}1231", STATION_END.replace("-", ""))
                if b > e:
                    continue
                u = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=hourly_height&application=NCKH"
                     f"&begin_date={b}&end_date={e}&datum=MSL&station={sid}&time_zone=gmt&units=metric&format=csv")
                txt = _get(u)
                urls.append(u)
                part = _noaa_part(txt)
                if part is None:
                    no_years.append(y)
                else:
                    parts.append(part)
            df = pd.concat(parts) if parts else pd.DataFrame()
            if df.empty:
                log(f"[stations] NOAA {sid} {nm}: khong co du lieu")
                nodata[sid] = dict(name=nm, years_checked=[2013, 2026], note="API tra 'No data was found' moi nam; khong co file",
                                   query_date=str(pd.Timestamp.now().date()))
                json.dump(nodata, open(nodata_file, "w", encoding="utf-8"), indent=1)
                continue
            df.columns = [x.strip() for x in df.columns]
            df.to_csv(path, index=False)
            write_provenance(path, c.boundary, source="NOAA CO-OPS hourly_height", urls=urls, station=meta["name"], station_id=sid,
                             lat=meta["lat"], lon=meta["lng"], datum="MSL (epoch hien hanh)", time_zone="GMT", units="m",
                             rows=len(df), first=str(df.iloc[0, 0]), last=str(df.iloc[-1, 0]), years_no_data=no_years,
                             sha256=file_sha256(path), query_date=str(pd.Timestamp.now().date()))
            log(f"[stations] NOAA {sid} {nm}: {len(df)} dong {df.iloc[0, 0]} -> {df.iloc[-1, 0]}")
        except Exception:
            log(f"[LOI] NOAA {sid}: {traceback.format_exc(limit=1)}")
    for sid, nm in USGS.items():
        path = f"{out}usgs_{sid}_daily.csv"
        if os.path.exists(path + ".provenance.json"):
            continue
        try:
            u = (f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={sid}&parameterCd=00065,00060&statCd=00003"
                 f"&startDT={STATION_START}&endDT={STATION_END}")
            lines = [ln for ln in _get(u).splitlines() if not ln.startswith("#")]
            df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
            if df.empty:
                log(f"[stations] USGS {sid} {nm}: khong co du lieu")
                continue
            df.to_csv(path, index=False)
            write_provenance(path, c.boundary, source="USGS NWIS dv", url=u, site=sid, name=nm, columns=list(df.columns),
                             note="00065 = gage height (ft), 00060 = discharge (ft3/s) - don vi gui nguyen", rows=len(df),
                             first=df.iloc[0]["datetime"], last=df.iloc[-1]["datetime"], sha256=file_sha256(path),
                             query_date=str(pd.Timestamp.now().date()))
            log(f"[stations] USGS {sid} {nm}: {len(df)} dong, cot {list(df.columns)[2:]}")
        except Exception:
            log(f"[LOI] USGS {sid}: {traceback.format_exc(limit=1)}")


# ------------------------------------------------------------------ CRMS (Louisiana) do man nuoc lo rong
CRMS_URL = "https://cims.coastal.louisiana.gov/RequestedDownloads/ZippedFiles/CRMS_Discrete_Hydrographic.zip"
CRMS_TERMS = ("CPRA/USGS CIMS, cong khai; trich dan: CPRA Coastwide Reference Monitoring System - Wetlands Monitoring Data, "
              "Coastal Information Management System (CIMS) database; kem RESTORE disclaimer (CIMS About.aspx, RESTORE_disclaimer.txt)")
PORE = "Soil Porewater Salinity (ppt)"


def crms_table(csv_path, boundary_geom):
    """CSV roi rac CIMS -> bang chuan: chi dong co do man lo rong, trong cua so 2013-11-01..2026-04-30 va trong ranh gioi.
    flag: 'ok' | 'gt60ppt' (>60, ngoai khoang 0-60) | 'no_depth'. Gio dia phuong ghi theo cot Time Zone (CST)."""
    from shapely import contains_xy
    d = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
    d = d[d[PORE].notna() & d["Latitude"].notna() & d["Longitude"].notna()].copy()
    d["datetime"] = pd.to_datetime(d["Date (mm/dd/yyyy)"] + " " + d["Time (hh:mm)"].fillna("00:00"), format="%m/%d/%Y %H:%M", errors="coerce")
    d = d[d["datetime"].between(STATION_START, pd.Timestamp(STATION_END) + pd.Timedelta(days=1), inclusive="left")]
    d = d[contains_xy(boundary_geom, d["Longitude"].to_numpy(), d["Latitude"].to_numpy())]
    depth = (d["Measurement Depth (ft)"] * 30.48).round(1)
    flag = np.where(d[PORE] > 60, "gt60ppt", np.where(depth.isna(), "no_depth", "ok"))
    out = pd.DataFrame({"station_id": d["CPRA Station ID"], "lon": d["Longitude"], "lat": d["Latitude"], "datetime": d["datetime"],
                        "depth_cm": depth, "salinity_ppt": d[PORE], "flag": flag, "time_zone": d["Time Zone"]})
    return out.sort_values(["station_id", "datetime"]).reset_index(drop=True)


def cmd_crms(c, local=None):
    import tempfile
    import zipfile
    import geopandas as gpd
    if not c.cfg.get("crms"):
        return log("[crms] vung khong co CRMS")
    out = c.p("raw", "crms", "crms_porewater_salinity.csv")
    if os.path.exists(out + ".provenance.json"):
        return log("[crms] da co")
    d = tempfile.mkdtemp(prefix="crms_")
    src, last_mod = local, None
    if not local:
        import requests
        r = None
        for k in range(5):
            try:
                r = requests.get(CRMS_URL, headers=UA, timeout=300)
                r.raise_for_status()
                break
            except Exception as e:
                log(f"[thu lai {k + 1}] CRMS {str(e)[:80]}")
                time.sleep(10 * 2 ** k)
        else:
            raise SystemExit(f"khong tai duoc {CRMS_URL}; tai tay tu https://cims.coastal.louisiana.gov (Data > Hydrographic) roi chay voi --crms-file <zip|csv>")
        src, last_mod = os.path.join(d, "CRMS_Discrete_Hydrographic.zip"), r.headers.get("Last-Modified")
        open(src, "wb").write(r.content)
    if src.lower().endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            z.extract("CRMS_Discrete_Hydrographic.csv", d)
        csv = os.path.join(d, "CRMS_Discrete_Hydrographic.csv")
    else:
        csv = src
    geom = gpd.read_file(c.boundary).to_crs(4326).union_all()
    t = crms_table(csv, geom)
    t.to_csv(out, index=False)
    st = t.groupby("station_id").agg(lon=("lon", "first"), lat=("lat", "first"), n_obs=("salinity_ppt", "size"), first=("datetime", "min"),
                                     last=("datetime", "max")).reset_index()
    st_path = c.p("raw", "crms", "crms_stations.csv")
    st.to_csv(st_path, index=False)
    write_provenance(out, c.boundary, source="CRMS Discrete Hydrographic (cot Soil Porewater Salinity, ppt)", url=CRMS_URL,
                     manual_file=local, last_modified=last_mod, query_date=str(pd.Timestamp.now().date()), terms=CRMS_TERMS,
                     window=[STATION_START, STATION_END], rows=len(t), n_stations=len(st), flags=t["flag"].value_counts().to_dict(),
                     sal_range=[float(t.salinity_ppt.min()), float(t.salinity_ppt.max())], time_zone=sorted(t["time_zone"].unique().tolist()),
                     note="datetime la gio dia phuong (CST) nhu CIMS ghi; depth_cm = ft*30.48; giu gia tri >60 ppt va gan flag; chua gop mua/o",
                     sha256=file_sha256(out))
    log(f"[crms] {len(t)} dong, {len(st)} tram, {t.salinity_ppt.min()}-{t.salinity_ppt.max()} ppt, flag {t['flag'].value_counts().to_dict()}")


# ------------------------------------------------------------------ kiem sau tai
def _pts(path, boundary, n=10, seed=0):
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    import geopandas as gpd
    with rasterio.open(path) as ds:
        b = gpd.read_file(boundary).to_crs(ds.crs)
        m = rasterize([(g, 1) for g in b.geometry], out_shape=(ds.height, ds.width), transform=ds.transform, dtype="uint8")
        rng, rr, cc = np.random.default_rng(seed), [], []
        while len(rr) < n:   # lay mau loai bo: np.nonzero tren 800 trieu pixel WorldCover can 12 GB
            r, q = int(rng.integers(ds.height)), int(rng.integers(ds.width))
            if m[r, q] and (r, q) not in zip(rr, cc):
                rr.append(r); cc.append(q)
        rr, cc = np.array(rr), np.array(cc)
        xs, ys = rasterio.transform.xy(ds.transform, rr, cc)
        vals = np.stack([ds.read(i + 1)[rr, cc] for i in range(ds.count)], 1).astype("float64")
        lon, lat = Transformer.from_crs(ds.crs, 4326, always_xy=True).transform(xs, ys)
        return list(zip(lon, lat)), vals, ds.descriptions, ds.crs, ds.transform, m


def _match(g, f):
    """GEE (sentinel unmask -9999 -> NaN) vs file; NaN hai ben = khop."""
    g = np.where(np.asarray(g, dtype="float64") <= -9000, np.nan, g)
    f = np.asarray(f, dtype="float64")
    return bool(np.allclose(np.nan_to_num(g, nan=-1e9), np.nan_to_num(f, nan=-1e9), rtol=1e-4, atol=1e-4))


def _cmp(ee, img, path, c, bands, seed=0, grid_src=None):
    pts, vals, desc, crs, tr, _ = _pts(path, c.boundary, 10, seed)
    crs_s = f"EPSG:{crs.to_epsg()}"
    if "sinu" in str(crs.to_proj4()):
        crs_s, trn = gf._native_grid(ee, grid_src, "GMT_0000_DSR")   # luoi goc MODIS (anh tinh TB mat phep chieu)
    else:
        trn = [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f]
    ok = 0
    for (lo, la), v in zip(pts, vals):
        r = gf._retry(lambda: img.reduceRegion(ee.Reducer.first(), ee.Geometry.Point([lo, la]), crs=crs_s, crsTransform=trn)
                      .getInfo(), "cmp")
        g = np.array([np.nan if r.get(bn) is None else r[bn] for bn in bands], dtype="float64")
        f = v[[list(desc).index(bn) for bn in bands]]
        ok += _match(g, f)
    return ok


def cmd_check(c, sensors="l8"):
    import rasterio
    ee = gf._init(c.project)
    lab = c.cfg["label"]
    rows = []
    prods = {
        "era5_rain": lambda s: None,
        "chirps3": lambda s: f"raw/chirps3/chirps3_rain_{s}.tif",
        "mcd18a1": lambda s: f"raw/mcd18a1/mcd18a1_dsr_{s}.tif",
        "ndwi": lambda s: f"landsat/{s}_{lab}_dry_NDWIchen_{sensors}.tif",
        "watermask": lambda s: f"landsat/{s}_{lab}_dry_watermask_{sensors}.tif",
    }
    for pn, fn in prods.items():
        for s in SEASONS:
            if pn == "era5_rain":
                fs = [f"{c.root}/raw/daily_rain/rain_{y}_{m:02d}.tif" for (y, m) in ml.season_year_months(s)]
                have = sum(os.path.exists(f) for f in fs)
                jan = f"{c.root}/raw/daily_rain/rain_{s}_01.tif"
                pn_ = None
                if os.path.exists(jan):
                    with rasterio.open(jan) as ds:
                        m = ml.region_masks(c.boundary, {"crs": ds.crs, "transform": ds.transform, "height": ds.height, "width": ds.width})[0]
                        pn_ = round(100 * float((~np.isfinite(ds.read(1)) & m).sum()) / max(int(m.sum()), 1), 2)
                        crs_e = ds.crs.to_epsg()
                rows.append(dict(source=pn, season=s, files=f"{have}/{len(fs)}", crs=crs_e if pn_ is not None else None,
                                 pct_nan_in_boundary=pn_, MB=round(sum(os.path.getsize(f) for f in fs if os.path.exists(f)) / 2 ** 20, 2)))
                continue
            f = f"{c.root}/{fn(s)}"
            r = dict(source=pn, season=s, files="1/1" if os.path.exists(f) else "0/1")
            if os.path.exists(f):
                with rasterio.open(f) as ds:
                    r.update(MB=round(os.path.getsize(f) / 2 ** 20, 1), crs=ds.crs.to_epsg() or "sinu" + ("-sphere" if ml.is_modis_sphere(ds.crs) else "-ELLIPS!"),
                             shape=f"{ds.height}x{ds.width}")
                try:
                    from met_labels import region_masks, read_bands
                    with rasterio.open(f) as ds:
                        prof = {"crs": ds.crs, "transform": ds.transform, "height": ds.height, "width": ds.width}
                        m = region_masks(c.boundary, prof)[0]
                        a = ds.read(1).astype("float64")
                    nanv = ~np.isfinite(a) if pn != "watermask" else (a == 255)
                    r["pct_nan_in_boundary"] = round(100 * float((nanv & m).sum()) / max(int(m.sum()), 1), 2)
                    del a, m, nanv
                except Exception as e:
                    r["pct_nan_in_boundary"] = f"loi {str(e)[:40]}"
            rows.append(r)
        log(f"[kiem] {pn} xong")
    # doi chieu 10 diem GEE (mua 2020 va 2026 cho Landsat; 2020 cho cac nguon con lai)
    cmpres = {}
    for s in (2020, 2026):
        f = f"{c.root}/landsat/{s}_{lab}_dry_NDWIchen_{sensors}.tif"
        if os.path.exists(f):
            b = grid_of(c, 30)[0]
            cg = clip_geom(ee, c, b)
            img = landsat_img(ee, s, cg, sensors)
            cmpres[f"ndwi_{s}"] = _cmp(ee, img.select(["NDWIchen"]), f, c, ["NDWIchen"], s)
            f2 = f.replace("NDWIchen", "watermask")
            if os.path.exists(f2):
                cmpres[f"watermask_{s}"] = _cmp(ee, img.select(["water_freq", "n_clear"]), f2, c, ["water_freq", "n_clear"], s)
    s = 2020
    f = f"{c.root}/raw/mcd18a1/mcd18a1_dsr_{s}.tif"
    if os.path.exists(f):
        idx = ml.mcd18_indices(s)
        daily = ee.ImageCollection(gf.MCD18_ID).filter(ee.Filter.inList("system:index", idx)).map(lambda im: gf._mcd18_day(ee, im))
        cmpres["mcd18_dsr_mean_2020"] = _cmp(ee, daily.select("dsr").mean().rename("dsr_mean"), f, c, ["dsr_mean"], s,
                                             grid_src=ee.ImageCollection(gf.MCD18_ID).filter(ee.Filter.inList("system:index", idx)).first())
    f = f"{c.root}/raw/chirps3/chirps3_rain_{s}.tif"
    if os.path.exists(f):
        sub = ee.ImageCollection(gf.CHIRPS3_ID).filter(ee.Filter.inList("system:index", ml.chirps_pentad_indices(s)))
        tot = sub.map(lambda im: im.select("precipitation").updateMask(im.select("precipitation").gte(0))).sum().rename("rain_sum")
        cmpres["chirps_rain_sum_2020"] = _cmp(ee, tot, f, c, ["rain_sum"], s)
    for name, bn in (("static/DEM_%s.tif" % lab, "DEM"), ("static/LandCover_%s_2021.tif" % lab, "Map")):
        f = f"{c.root}/{name}"
        if os.path.exists(f):
            img = (ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic() if bn == "DEM" else
                   ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").unmask(80))
            cmpres[bn] = _cmp(ee, img.rename(bn), f, c, [bn], 3)
    df = pd.DataFrame(rows)
    df.to_csv(f"{c.root}/check_report.csv", index=False)
    json.dump({k: f"{v}/10" for k, v in cmpres.items()}, open(f"{c.root}/check_gee_10pts.json", "w"), indent=1)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df.to_string(index=False), flush=True)
    print("doi chieu GEE 10 diem (khop/10):", {k: f"{v}/10" for k, v in cmpres.items()}, flush=True)


STEPS = {"boundary": cmd_boundary, "stations": cmd_stations, "era5": cmd_era5, "chirps": cmd_chirps, "mcd18": cmd_mcd18,
         "static": cmd_static, "landsat": cmd_landsat, "crms": cmd_crms}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", required=True, choices=list(REGIONS))
    ap.add_argument("--root", default=None, help="mac dinh <DATA_ROOT>/doi_chung hoac bien DOI_CHUNG_ROOT")
    ap.add_argument("--project", default=None)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--static-buffer-km", type=float, default=0.0,
                    help="static/worldcover: dem quanh ranh gioi (km), ghi LandCover_<L>_2021_pad<km>.tif; 0 = nhu cu")
    ap.add_argument("--sources", nargs="+", choices=list(STEPS), default=["boundary", "stations", "chirps", "mcd18", "era5", "static", "landsat"])
    ap.add_argument("--seasons", nargs="+", type=int, default=None, help="chi cho landsat")
    ap.add_argument("--crms-file", default=None, help="zip/csv CRMS tai tay (neu khong tai tu dong duoc)")
    ap.add_argument("--scene-filter", action="store_true", help="landsat: loai canh may/khoi, hau to _sf3 (mac dinh TAT)")
    ap.add_argument("--mask-snow", action="store_true", help="landsat: che tuyet/bang (bit 5) o buoc ghep, them hau to _snow (mac dinh TAT)")
    ap.add_argument("--suffix", default=None, help="landsat: hau to file (gom dau _), ghi de mac dinh")
    ap.add_argument("--max-b3", type=float, default=0.25, help="--scene-filter: nguong TB SR_B3 (phan xa) tren pixel sach")
    ap.add_argument("--min-b2b3", type=float, default=0.85, help="--scene-filter: loai canh khi b3>max-b3 VA b2/b3 > nguong (may/khoi)")
    ap.add_argument("action", nargs="?", choices=["fetch", "check"], default="fetch")
    a = ap.parse_args()
    ctx = Ctx(a)
    if a.action == "check":
        cmd_check(ctx)
        sys.exit()
    for name in a.sources:
        log(f"=== {name} ({ctx.name}) ===")
        try:
            if name == "landsat":
                cmd_landsat(ctx, seasons=a.seasons, scene_filter=a.scene_filter and dict(max_b3=a.max_b3, min_b2b3=a.min_b2b3),
                             mask_snow=a.mask_snow, suffix=a.suffix)
            elif name == "crms":
                cmd_crms(ctx, a.crms_file)
            else:
                STEPS[name](ctx)
        except SystemExit as e:
            log(f"[LOI] {name}: {e}")
        except Exception:
            log(f"[LOI] {name}: {traceback.format_exc(limit=3)}")
    log("=== HET ===")

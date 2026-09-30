#!/usr/bin/env python
"""Kiem do day du song/kenh OSM bang nguon doc lap JRC Global Surface Water (T3-V1).

Cach do: lay mat nuoc thuong xuyen JRC (occurrence >= nguong) tren luoi 30 m, tach thanh
cac vung nuoc lien thong (8-lan can). Mot vung duoc coi la "OSM da ve" neu co it nhat mot
duong song/kenh OSM di qua (sau khi dem --tol-m). Bao cao theo tinh (FAO GAUL cap 1):
ty le dien tich mat nuoc thuoc vung da ve, va cac vung lon nhat chua ve.
Chi xet mat nuoc DANG DAI (bo ao/ho: ty le dai/rong thap) de khong phat oan OSM vi ho nuoi tom.

Chay:  python scripts/check_osm_vs_jrc.py --project <id> --osm <osm_waterways.gpkg> [--out bang.csv]
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import CANONICAL_BOUNDARY  # noqa: E402

PROVINCES = ["An Giang", "Bac Lieu", "Ben Tre", "Ca Mau", "Can Tho city", "Dong Thap", "Hau Giang",
             "Kien Giang", "Long An", "Soc Trang", "Tien Giang", "Tra Vinh", "Vinh Long"]


def _download_tif(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if data[:2] == b"PK":  # doi khi tra ve zip
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(next(n for n in z.namelist() if n.endswith(".tif")))
    return data


def fetch_jrc(ee, bounds, scale, threshold, path, tiles=4):
    """Tai theo tiles x tiles o (getDownloadURL gioi han 50 MB/yeu cau) roi ghep."""
    if os.path.exists(path):
        return
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
    img = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(threshold).unmask(0).uint8()
    x0, y0, x1, y1 = bounds
    xs, ys = np.linspace(x0, x1, tiles + 1), np.linspace(y0, y1, tiles + 1)
    mems = []
    for i in range(tiles):
        for j in range(tiles):
            region = ee.Geometry.Rectangle([xs[i], ys[j], xs[i + 1], ys[j + 1]])
            url = img.getDownloadURL({"region": region, "scale": scale, "crs": "EPSG:32648", "format": "GEO_TIFF"})
            mems.append(MemoryFile(_download_tif(url)))
    srcs = [m.open() for m in mems]
    mosaic, transform = merge(srcs)
    profile = srcs[0].profile | {"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform,
                                 "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mosaic)
    for s, m in zip(srcs, mems):
        s.close()
        m.close()


def fetch_provinces(ee):
    fc = (ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq("ADM0_NAME", "Viet Nam"))
          .filter(ee.Filter.inList("ADM1_NAME", PROVINCES)).map(lambda f: f.simplify(200)))
    gdf = gpd.GeoDataFrame.from_features(fc.getInfo()["features"], crs=4326)
    return gdf[["ADM1_NAME", "geometry"]].rename(columns={"ADM1_NAME": "province"})


def label_linear_water(water, px_km2, min_elong=8.0, min_km2=0.05):
    """Tach mat nuoc thanh vung lien thong (8-lan can); `linear[k]` = vung k co dang dai (song/kenh).

    Do dai = (duong cheo khung bao)^2 / dien tich: ~1-2 cho ao vuong, lon cho song/kenh.
    """
    lab, n = ndimage.label(water, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    diag_px = np.array([0] + [np.hypot(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in objs])
    elong = np.where(area > 0, diag_px ** 2 / np.maximum(area, 1), 0)
    linear = (elong >= min_elong) & (area * px_km2 >= min_km2)
    linear[0] = False
    return lab, n, objs, area, linear


def main(args):
    import ee
    ee.Initialize(project=args.project)
    boundary = gpd.read_file(args.boundary).to_crs(32648)
    bounds4326 = boundary.to_crs(4326).total_bounds
    jrc_path = os.path.join(os.path.dirname(args.osm), f"jrc_gsw_occ{args.threshold}_{args.scale}m.tif")
    fetch_jrc(ee, bounds4326, args.scale, args.threshold, jrc_path)
    prov = fetch_provinces(ee).to_crs(32648)
    print(f"Tinh GAUL tim thay: {len(prov)}/13")

    with rasterio.open(jrc_path) as src:
        water = src.read(1) > 0
        transform, shape = src.transform, src.shape
    inside = features.rasterize(boundary.geometry, out_shape=shape, transform=transform, fill=0, default_value=1) > 0
    water &= inside
    px_km2 = abs(transform.a * transform.e) / 1e6
    lab, n, objs, area, linear = label_linear_water(water, px_km2, args.min_elong, args.min_km2)

    osm = gpd.read_file(args.osm, layer="lines").to_crs(32648)
    osm = osm[osm["fclass"].isin(["river", "canal"])]
    osm_mask = features.rasterize(osm.geometry.buffer(args.tol_m), out_shape=shape, transform=transform,
                                  fill=0, default_value=1) > 0
    touched = np.zeros(n + 1, bool)
    touched[np.unique(lab[osm_mask & (lab > 0)])] = True

    prov_r = features.rasterize(((g, i + 1) for i, g in enumerate(prov.geometry)), out_shape=shape,
                                transform=transform, fill=0)
    rows = []
    sel = linear[lab]
    for i, name in enumerate(prov["province"], start=1):
        m = (prov_r == i) & sel
        tot = m.sum() * px_km2
        cov = (m & touched[lab]).sum() * px_km2
        rows.append({"province": name, "water_linear_km2": round(tot, 1), "covered_km2": round(cov, 1),
                     "covered_pct": round(100 * cov / tot, 1) if tot else np.nan})
    allm = sel
    tot = allm.sum() * px_km2
    cov = (allm & touched[lab]).sum() * px_km2
    rows.append({"province": "TOAN VUNG", "water_linear_km2": round(tot, 1), "covered_km2": round(cov, 1),
                 "covered_pct": round(100 * cov / tot, 1)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    miss = [(k, area[k] * px_km2) for k in range(1, n + 1) if linear[k] and not touched[k]]
    miss.sort(key=lambda t: -t[1])
    print(f"\nVung nuoc dang dai CHUA co OSM: {len(miss)}; 10 vung lon nhat (km2, tam lon/lat):")
    for k, a in miss[:10]:
        rr, cc = np.nonzero(lab[objs[k - 1]] == k)
        r0, c0 = objs[k - 1][0].start + rr.mean(), objs[k - 1][1].start + cc.mean()
        x, y = rasterio.transform.xy(transform, r0, c0)
        p = gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=32648).to_crs(4326).iloc[0]
        print(f"  {a:6.2f} km2  ({p.x:.3f}, {p.y:.3f})")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Da ghi: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=os.getenv("EE_PROJECT"))
    ap.add_argument("--osm", required=True)
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--scale", type=int, default=30)
    ap.add_argument("--threshold", type=int, default=50, help="JRC occurrence (%%) toi thieu de coi la nuoc thuong xuyen")
    ap.add_argument("--tol-m", type=float, default=60.0, help="Dem quanh duong OSM (sai lech vi tri)")
    ap.add_argument("--min-elong", type=float, default=8.0)
    ap.add_argument("--min-km2", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())

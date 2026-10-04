#!/usr/bin/env python
"""Ranh gioi v2 cho DBSCL: v1 noi 3 km phia bien, tru lanh tho lang gieng (An duyet 2026-10-03).

  v2 = v1 HOP (buffer(v1, 3 km) TRU buffer(lang gieng, 3 km)), tinh trong EPSG:32648.
  Lang gieng = TP. Ho Chi Minh, Tay Ninh (geoBoundaries gbOpen VNM ADM1) va Campuchia (gbOpen KHM ADM0),
  cung commit 9469f09 voi v1. Khong bao gio tru vao phan v1 (phan them duoc hop voi hinh v1 GOC).

Ly do: v1 chi co 533 dinh (~2,6 km/dinh) -> duong bo ve bang doan thang dai, bo sot dat ven bien
(tam TP Rach Gia nam ngoai v1 1,86 km). Noi 3 km phu ~98% dat ven bien thuoc 13 tinh (do 2026-10-03).
Bien trong dai moi them do mask nuoc dong theo nam loai; khoang cach toi bien van tinh tu duong bo that.

Kiem tra (--check, can raster o A:): dien tich v1/v2/phan them; % dat ven bien (WorldCover 10-60, 90,
95; cach bien <= 2 km theo dist_coast_km; ngoai v1) nay nam trong v2; dien tich lan sang lang gieng;
bang mask nuoc theo nam tren dai moi them (v2 - v1) -> --mask-csv.

Chay:  python scripts/build_boundary_v2.py [--check] [--mask-csv KE_HOACH/ket-qua/dot4_kiem_mask_dai_bien_moi.csv]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio import features
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import BOUNDARY_V1, CANONICAL_BOUNDARY, file_sha256  # noqa: E402

GB_COMMIT = "9469f09"
GB_URL = "https://github.com/wmgeolab/geoBoundaries/raw/{commit}/releaseData/gbOpen/{iso}/{adm}/geoBoundaries-{iso}-{adm}.geojson"
NEIGHBOR_PROVINCES = ["Ho Chi Minh", "Tây Ninh"]  # shapeName trong VNM ADM1 tai commit tren
NEIGHBOR_COUNTRIES = ["KHM"]
MEKONG_DELTA_PROVINCES = {
    "An Giang", "Bạc Liêu", "Bến Tre", "Cà Mau", "Cần Thơ", "Hậu Giang", "Kiên Giang", "Long An",
    "Sóc Trăng", "Tiền Giang", "Trà Vinh", "Vĩnh Long", "Đồng Tháp",
}
CRS_M = "EPSG:32648"
WC_LAND = [10, 20, 30, 40, 50, 60, 90, 95]  # cay, bui, co, nong nghiep, xay dung, dat trong, dat ngap, ngap man


def log(msg):
    print(msg, flush=True)


def fetch_geoboundaries(iso, adm, cache_dir, tries=5):
    """Tai (co cache) file geoBoundaries gbOpen tai commit GB_COMMIT; thu lai khi loi mang."""
    os.makedirs(cache_dir, exist_ok=True)
    url = GB_URL.format(commit=GB_COMMIT, iso=iso, adm=adm)
    path = os.path.join(cache_dir, os.path.basename(url))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, url
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NCKH-Mekong-research-script"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return path, url
        except Exception as exc:  # noqa: BLE001 - loi mang tam thoi
            wait = 10 * 2 ** k
            log(f"  [thu lai {k + 1}/{tries}] {url}: {exc} - cho {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Khong tai duoc {url}")


def build_v2(v1_geom_wgs, neighbors_m, buffer_m):
    """v2 (EPSG:4326) = v1 goc HOP phan them; phan them tinh trong EPSG:32648.

    v1_geom_wgs: hinh v1 EPSG:4326; neighbors_m: hinh lang gieng EPSG:32648. Tra ve (v2_wgs, added_m).
    """
    to_m = gpd.GeoSeries([v1_geom_wgs], crs=4326).to_crs(CRS_M).iloc[0]
    added_m = to_m.buffer(buffer_m).difference(neighbors_m.buffer(buffer_m)).difference(to_m)
    added_wgs = gpd.GeoSeries([added_m], crs=CRS_M).to_crs(4326).iloc[0]
    v2 = shapely.union(v1_geom_wgs, added_wgs)
    v2 = shapely.make_valid(v2) if not v2.is_valid else v2
    return v2, added_m


def _drop_z(g):
    return shp_transform(lambda x, y, z=None: (x, y), g)


# -----------------------------------------------------------------------------------------------
# Kiem tra tren raster
# -----------------------------------------------------------------------------------------------
def _nearest_index(src, xs, ys):
    """Chi so (hang, cot) pixel gan nhat cua src cho toa do tam pixel; -1 neu ngoai khung."""
    inv = ~src.transform
    cols = np.floor(inv.a * xs + inv.c).astype(np.int64)
    rows = np.floor(inv.e * ys + inv.f).astype(np.int64)
    cols[(cols < 0) | (cols >= src.width)] = -1
    rows[(rows < 0) | (rows >= src.height)] = -1
    return rows, cols


def sample_on_grid(path, band, transform, shape, fill, chunk_rows=1024):
    """Lay gia tri pixel GAN NHAT (tam pixel luoi dich) cua `band` (ten band) tren luoi dich (cung CRS).

    Doc theo khoi hang de khong nap ca raster 10 m vao bo nho. Pixel ngoai khung = fill.
    """
    h, w = shape
    xs = transform.c + transform.a * (np.arange(w) + 0.5)
    ys = transform.f + transform.e * (np.arange(h) + 0.5)
    with rasterio.open(path) as src:
        bidx = list(src.descriptions).index(band) + 1 if isinstance(band, str) else band
        dtype = src.dtypes[bidx - 1]
        rows, _ = _nearest_index(src, np.zeros(h), ys)
        _, cols = _nearest_index(src, xs, np.zeros(w))
        out = np.full(shape, fill, dtype=dtype)
        okc = cols >= 0
        if not okc.any():
            return out
        c0, c1 = int(cols[okc].min()), int(cols[okc].max())
        for r0 in range(0, h, chunk_rows):
            rr = rows[r0:r0 + chunk_rows]
            okr = rr >= 0
            if not okr.any():
                continue
            a0, a1 = int(rr[okr].min()), int(rr[okr].max())
            data = src.read(bidx, window=Window(c0, a0, c1 - c0 + 1, a1 - a0 + 1))
            block = data[np.ix_(rr[okr] - a0, cols[okc] - c0)]
            sub = out[r0:r0 + chunk_rows]
            sub[np.ix_(np.flatnonzero(okr), np.flatnonzero(okc))] = block
    return out


def coastal_land_check(v1_m, v2_m, nb_m, wc_path, dist_path, coast_km=2.0, chunk_rows=1024):
    """Dien tich (km2) dat ven bien ngoai v1 va phan nam trong v2, tren luoi WorldCover 10 m."""
    with rasterio.open(dist_path) as d:
        bidx = list(d.descriptions).index("dist_coast_km") + 1
        dist = d.read(bidx)
        dist_tr, dist_w, dist_h = d.transform, d.width, d.height
    inv = ~dist_tr
    tot = {k: 0 for k in ("coast_land_out_v1", "coast_land_out_v1_in_v2", "coast_land_out_v1_13",
                          "coast_land_out_v1_13_in_v2", "coast_land_out_v1_nb", "coast_land_out_v1_nb_in_v2")}
    with rasterio.open(wc_path) as src:
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
        xs = src.transform.c + src.transform.a * (np.arange(src.width) + 0.5)
        dcols = np.floor(inv.a * xs + inv.c).astype(np.int64)
        okc = (dcols >= 0) & (dcols < dist_w)
        for r0 in range(0, src.height, chunk_rows):
            n = min(chunk_rows, src.height - r0)
            win = Window(0, r0, src.width, n)
            wtr = src.window_transform(win)
            wc = src.read(1, window=win)
            ys = wtr.f + wtr.e * (np.arange(n) + 0.5)
            drows = np.floor(inv.e * ys + inv.f).astype(np.int64)
            okr = (drows >= 0) & (drows < dist_h)
            dd = np.full((n, src.width), np.nan, "float32")
            dd[np.ix_(np.flatnonzero(okr), np.flatnonzero(okc))] = dist[np.ix_(drows[okr], dcols[okc])]
            cand = np.isin(wc, WC_LAND) & (dd <= coast_km)
            if not cand.any():
                continue

            def ras(g):
                return features.rasterize([g], out_shape=(n, src.width), transform=wtr, fill=0,
                                          default_value=1, dtype="uint8") > 0
            in1, in2, innb = ras(v1_m), ras(v2_m), ras(nb_m)
            out1 = cand & ~in1
            tot["coast_land_out_v1"] += int(out1.sum())
            tot["coast_land_out_v1_in_v2"] += int((out1 & in2).sum())
            tot["coast_land_out_v1_13"] += int((out1 & ~innb).sum())
            tot["coast_land_out_v1_13_in_v2"] += int((out1 & ~innb & in2).sum())
            tot["coast_land_out_v1_nb"] += int((out1 & innb).sum())
            tot["coast_land_out_v1_nb_in_v2"] += int((out1 & innb & in2).sum())
    return {k: v * px_km2 for k, v in tot.items()}


def strip_grid(strip_m, ref_path, pad_m=60.0):
    """Luoi 30 m CUNG goc neo voi raster mask (ref_path), mo rong cho phu het dai moi them."""
    with rasterio.open(ref_path) as ref:
        rt = ref.transform
    res = rt.a
    minx, miny, maxx, maxy = strip_m.bounds
    x0 = rt.c + np.floor((minx - pad_m - rt.c) / res) * res
    y1 = rt.f - np.floor((rt.f - (maxy + pad_m)) / res) * res
    w = int(np.ceil((maxx + pad_m - x0) / res))
    h = int(np.ceil((y1 - (miny - pad_m)) / res))
    return Affine(res, 0, x0, 0, -res, y1), (h, w)


def gee_label_path(gee_dir, year):
    """Nhan GEE tu tao: uu tien ban `_v2` (2024-2026, xuat theo ranh gioi v2); 2020-2023 chi co ban
    khong hau to (xuat theo v1, dung de so Zenodo). Khong co file nao -> tra ve duong dan `_v2` (khong ton tai)."""
    v2 = os.path.join(gee_dir, f"{year}_MD_dry_NDWIchen_Salinity_l8_v2.tif")
    v1 = os.path.join(gee_dir, f"{year}_MD_dry_NDWIchen_Salinity_l8.tif")
    return v2 if os.path.exists(v2) or not os.path.exists(v1) else v1


def mask_table(strip_m, years, gee_dir, zenodo_dir, wc_path, mask_suffix="_v2"):
    """Bang mask nuoc theo nam tren dai moi them (v2 - v1), luoi 30 m cua file mask.

    Mask doc tu `{nam}_MD_dry_watermask_l8{mask_suffix}.tif` (mac dinh `_v2`: ban cu xuat theo v1 bi
    clip o ranh gioi v1 -> dai moi them toan "ngoai vung mask", vo nghia; 16 file cu se bi xoa, cau L).

    Nhom LOAI TRU nhau (tong 100%): ngoai_vung_mask (wf = 0 va n_clear = 0: file mask khong xuat o
    day), nuoc (wf >= 50 bo 255, HOP WC80), khong_quan_sat (255, khong WC80), ngap_man (WC95),
    dat (WC khong 0/80/95), wc_trong (WC = 0). Them: ty le WC tinh va do phu nhan theo nam.
    """
    ref = os.path.join(gee_dir, f"{years[0]}_MD_dry_watermask_l8{mask_suffix}.tif")
    tr, shape = strip_grid(strip_m, ref)
    strip = features.rasterize([strip_m], out_shape=shape, transform=tr, fill=0, default_value=1,
                               dtype="uint8") > 0
    n = int(strip.sum())
    log(f"  Dai moi them tren luoi 30 m: {n} pixel ({n * 9e-4:.1f} km2), khung {shape}")
    wc = sample_on_grid(wc_path, "Map", tr, shape, fill=0)[strip]
    wc80, wc95 = wc == 80, wc == 95
    wc_land = ~np.isin(wc, [0, 80, 95])
    rows = []
    for y in years:
        p = os.path.join(gee_dir, f"{y}_MD_dry_watermask_l8{mask_suffix}.tif")
        wf = sample_on_grid(p, "water_freq", tr, shape, fill=0)[strip]
        nc = sample_on_grid(p, "n_clear", tr, shape, fill=0)[strip]
        outside = (wf == 0) & (nc == 0)
        cov = ~outside
        water = cov & (((wf >= 50) & (wf != 255)) | wc80)
        unobs = cov & ~water & (wf == 255)
        rest = cov & ~water & ~unobs
        row = {
            "year": y, "n_px_30m": n,
            "ngoai_vung_mask_pct": 100 * outside.mean(),
            "nuoc_mask_pct": 100 * water.mean(),
            "khong_quan_sat_255_pct": 100 * unobs.mean(),
            "ngap_man_wc95_pct": 100 * (rest & wc95).mean(),
            "dat_pct": 100 * (rest & wc_land).mean(),
            "wc_trong_pct": 100 * (rest & (wc == 0)).mean(),
            "n_clear_tb_trong_vung_mask": float(nc[cov & (wf != 255)].mean()) if (cov & (wf != 255)).any() else np.nan,
            "wc80_tinh_pct": 100 * wc80.mean(), "wc95_tinh_pct": 100 * wc95.mean(),
            "wc_dat_tinh_pct": 100 * wc_land.mean(),
        }
        # Do phu nhan: Zenodo 2014-2023; ban tu tao (gee) 2020-2026.
        for tag, lp in (("zenodo", os.path.join(zenodo_dir, f"{y}_MD_dry_NDWIchen_Salinity.tif")),
                        ("gee", gee_label_path(gee_dir, y))):
            if os.path.exists(lp):
                s = sample_on_grid(lp, "Salinity", tr, shape, fill=np.nan)[strip]
                row[f"nhan_{tag}_huu_han_pct"] = 100 * np.isfinite(s).mean()
                row[f"nhan_{tag}_huu_han_tren_dat_pct"] = (100 * np.isfinite(s[wc_land]).mean()
                                                          if wc_land.any() else np.nan)
            else:
                row[f"nhan_{tag}_huu_han_pct"] = np.nan
                row[f"nhan_{tag}_huu_han_tren_dat_pct"] = np.nan
        rows.append(row)
        log(f"  {y}: ngoai vung mask {row['ngoai_vung_mask_pct']:.2f}% | nuoc {row['nuoc_mask_pct']:.2f}% | "
            f"dat {row['dat_pct']:.2f}% | ngap man {row['ngap_man_wc95_pct']:.2f}% | 255 "
            f"{row['khong_quan_sat_255_pct']:.2f}% | nhan zenodo {row['nhan_zenodo_huu_han_pct']:.1f}% "
            f"gee {row['nhan_gee_huu_han_pct']:.1f}%")
    df = pd.DataFrame(rows)
    return df.round(4)


def main(args):
    v1_path = args.v1
    v1 = gpd.read_file(v1_path).to_crs(4326)
    v1_geom = _drop_z(v1.geometry.union_all())
    v1_sha = file_sha256(v1_path)

    vnm_path, vnm_url = fetch_geoboundaries("VNM", "ADM1", args.cache_dir)
    vnm = gpd.read_file(vnm_path).to_crs(4326)
    vnm["shapeName"] = vnm["shapeName"].str.strip()
    nb_parts, nb_src = [], []
    missing = [n for n in NEIGHBOR_PROVINCES if n not in set(vnm["shapeName"])]
    if missing:
        raise RuntimeError(f"Thieu lang gieng trong VNM ADM1: {missing}")
    nb_parts += list(vnm.loc[vnm["shapeName"].isin(NEIGHBOR_PROVINCES)].geometry)
    nb_src.append({"url": vnm_url, "sha256": file_sha256(vnm_path), "features": NEIGHBOR_PROVINCES})
    for iso in NEIGHBOR_COUNTRIES:
        pth, url = fetch_geoboundaries(iso, "ADM0", args.cache_dir)
        nb_parts += list(gpd.read_file(pth).to_crs(4326).geometry)
        nb_src.append({"url": url, "sha256": file_sha256(pth), "features": [iso]})
    nb_m = gpd.GeoSeries([shapely.union_all(nb_parts)], crs=4326).to_crs(CRS_M).iloc[0]

    # Kiem nguon: hop 13 tinh tu file VNM tai ve phai trung v1.
    rebuilt = vnm.loc[vnm["shapeName"].isin(MEKONG_DELTA_PROVINCES)]
    v1_m = gpd.GeoSeries([v1_geom], crs=4326).to_crs(CRS_M).iloc[0]
    rb_m = gpd.GeoSeries([rebuilt.geometry.union_all()], crs=4326).to_crs(CRS_M).iloc[0]
    log(f"Kiem nguon: {len(rebuilt)} tinh, lech doi xung v1 vs hop 13 tinh tai ve = "
        f"{v1_m.symmetric_difference(rb_m).area / 1e6:.4f} km2")

    v2_geom, added_m = build_v2(v1_geom, nb_m, args.buffer_km * 1000)
    feature = {
        "type": "Feature",
        "properties": {
            "name": "Mekong Delta (13 provinces) v2 - noi 3 km phia bien",
            "version": "v2",
            "method": "v2 = v1 UNION (buffer(v1, d) - buffer(neighbors, d)); EPSG:32648; khong tru vao v1",
            "buffer_km": args.buffer_km,
            "neighbors": NEIGHBOR_PROVINCES + NEIGHBOR_COUNTRIES,
            "neighbor_sources": nb_src,
            "v1_file": os.path.basename(v1_path),
            "v1_sha256": v1_sha,
            "v1_source": GB_URL.format(commit=GB_COMMIT, iso="VNM", adm="ADM1"),
            "approved": "An duyet 2026-10-03 (KE_HOACH/NHAT_KY.md, AN DUYET 8 de xuat, muc 1)",
            "script": "scripts/build_boundary_v2.py",
        },
        "geometry": mapping(v2_geom),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": [feature]}, f, ensure_ascii=False)
    v2_sha = file_sha256(args.out)
    log(f"Ghi {args.out}")
    log(f"sha256 v1 = {v1_sha}")
    log(f"sha256 v2 = {v2_sha}")

    # Doc lai file da ghi de do (dung dung hinh se duoc dung o cac buoc sau).
    v2_m = gpd.read_file(args.out).to_crs(CRS_M).geometry.union_all()
    a1, a2 = v1_m.area / 1e6, v2_m.area / 1e6
    log(f"Dien tich (EPSG:32648): v1 {a1:.1f} km2 | v2 {a2:.1f} km2 | them {a2 - a1:.1f} km2 "
        f"(+{100 * (a2 / a1 - 1):.2f}%) | so dinh v2 {shapely.get_num_coordinates(v2_m)}")
    log(f"v1 nam tron trong v2: phan v1 ngoai v2 = {v1_m.difference(v2_m).area:.3f} m2")
    log(f"Lan sang lang gieng: v2 {v2_m.intersection(nb_m).area / 1e6:.4f} km2 "
        f"(v1 san co {v1_m.intersection(nb_m).area / 1e6:.4f} km2; phan them "
        f"{v2_m.difference(v1_m).intersection(nb_m).area / 1e6:.6f} km2)")

    if not args.check:
        return
    log("Kiem dat ven bien (WorldCover 10 m, dist_coast_km <= 2) ...")
    t0 = time.time()
    c = coastal_land_check(v1_m, v2_m, nb_m, args.worldcover, args.dist)
    log(f"  ({time.time() - t0:.0f}s) dat ven bien ngoai v1: {c['coast_land_out_v1']:.1f} km2 "
        f"(13 tinh {c['coast_land_out_v1_13']:.1f}, lanh tho lang gieng {c['coast_land_out_v1_nb']:.1f})")
    log(f"  nay trong v2: 13 tinh {c['coast_land_out_v1_13_in_v2']:.1f} km2 = "
        f"{100 * c['coast_land_out_v1_13_in_v2'] / c['coast_land_out_v1_13']:.1f}% | tat ca "
        f"{100 * c['coast_land_out_v1_in_v2'] / c['coast_land_out_v1']:.1f}% | phan cua lang gieng vao v2 "
        f"{c['coast_land_out_v1_nb_in_v2']:.3f} km2")

    if args.mask_csv:
        log("Bang mask nuoc tren dai moi them (v2 - v1) ...")
        strip_m = v2_m.difference(v1_m)
        df = mask_table(strip_m, list(range(args.year_start, args.year_end + 1)), args.gee_dir,
                        args.zenodo_dir, args.worldcover, mask_suffix=args.mask_suffix)
        df.insert(0, "boundary_v2_sha256", v2_sha)
        os.makedirs(os.path.dirname(args.mask_csv), exist_ok=True)
        df.to_csv(args.mask_csv, index=False)
        log(f"Bang: {args.mask_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v1", default=BOUNDARY_V1)
    ap.add_argument("--out", default=CANONICAL_BOUNDARY)
    ap.add_argument("--buffer-km", type=float, default=3.0)
    ap.add_argument("--cache-dir", default="A:/Dataset_NCKH/raw/boundary")
    ap.add_argument("--check", action="store_true", help="Kiem dat ven bien + bang mask (can raster o A:)")
    ap.add_argument("--worldcover", default="A:/Dataset_NCKH/gee/LandCover_DBSCL_2021.tif")
    ap.add_argument("--dist", default="A:/Dataset_NCKH/raw/river/river_distance_90m.tif")
    ap.add_argument("--gee-dir", default="A:/Dataset_NCKH/gee")
    ap.add_argument("--mask-suffix", default="_v2", help="hau to file mask nuoc (mac dinh _v2 theo ranh gioi v2)")
    ap.add_argument("--zenodo-dir", default="A:/Dataset_NCKH/zenodo_15653696")
    ap.add_argument("--mask-csv", default=os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_kiem_mask_dai_bien_moi.csv"))
    ap.add_argument("--year-start", type=int, default=2014)
    ap.add_argument("--year-end", type=int, default=2026)
    main(ap.parse_args())

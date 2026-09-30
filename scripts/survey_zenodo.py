#!/usr/bin/env python
"""Khao sat bo nhan Zenodo 15653696 (T1-01).

Buoc 1 (tuy chon --download): tai tung file qua API Zenodo, kiem tra MD5.
Buoc 2: voi moi nam, doc theo khoi cua so, chi tinh pixel trong ranh gioi 13 tinh:
        so pixel NaN / NoData / bang 0 / > 28.013, min, max, mean, p1, p50, p99,
        che do cam bien suy ra tu gia tri nho nhat.
Buoc 3: footprint = pixel hop le o >= 50% so nam; ty le thieu cua moi nam
        tinh tren footprint (khong tinh tren toan polygon).

Chay:  python scripts/survey_zenodo.py --data-dir <thu muc chua .tif> [--download]
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDARY = os.path.join(ROOT, "webapp", "backend", "data", "mekong_delta_boundary.geojson")
RECORD_URL = "https://zenodo.org/api/records/15653696"
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "raw", "zenodo_salinity")
DEFAULT_OUT = os.path.join(ROOT, "KE_HOACH", "ket-qua", "tuan1_khao_sat_zenodo.csv")
OLI_CAP = 28.013   # tran cong thuc OLI khi NIR >= 0
TM_FLOOR = 0.237   # san cong thuc hieu chinh TM
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def md5sum(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_manifest():
    with urllib.request.urlopen(RECORD_URL, timeout=60) as r:
        rec = json.load(r)
    return [
        {"key": f["key"], "md5": f["checksum"].split(":", 1)[1], "size": f["size"], "url": f["links"]["self"]}
        for f in rec["files"] if f["key"].endswith(".tif")
    ]


def download(manifest, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for f in sorted(manifest, key=lambda x: x["key"]):
        path = os.path.join(data_dir, f["key"])
        if os.path.exists(path) and os.path.getsize(path) == f["size"] and md5sum(path) == f["md5"]:
            print(f"  [co san, MD5 dung] {f['key']}", flush=True)
            continue
        print(f"  [tai] {f['key']} ({f['size'] / 1e6:.0f} MB)", flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(f["url"], tmp)
        got = md5sum(tmp)
        if got != f["md5"]:
            os.remove(tmp)
            raise RuntimeError(f"MD5 sai cho {f['key']}: {got} != {f['md5']}")
        os.replace(tmp, path)


def band_index(ds, name, fallback):
    return list(ds.descriptions).index(name) + 1 if name in ds.descriptions else fallback


def missing_mask(arr, nodata):
    bad = np.isnan(arr)
    if nodata is not None and not np.isnan(nodata):
        bad |= arr == nodata
    return bad


def survey_year(path, boundary_geom, valid_count):
    """Thong ke mot nam; cong don so nam hop le cua tung pixel vao valid_count."""
    with rasterio.open(path) as ds:
        s_idx = band_index(ds, "Salinity", 2)
        n_idx = band_index(ds, "NDWIchen", 1)
        nodata = ds.nodata
        info = {
            "crs": str(ds.crs), "res_m": f"{ds.res[0]:g}x{ds.res[1]:g}", "width": ds.width, "height": ds.height,
            "bands": "|".join(str(b) for b in ds.descriptions), "dtype": ds.dtypes[s_idx - 1],
            "nodata_val": "None" if nodata is None else repr(nodata),
        }
        n_in = n_nan = n_nodata = n_zero = n_above = 0
        values, ndwi = [], []
        for _, win in ds.block_windows(s_idx):
            inside = ~geometry_mask([boundary_geom], out_shape=(win.height, win.width),
                                    transform=ds.window_transform(win))
            if not inside.any():
                continue
            s_block = ds.read(s_idx, window=win).astype("float64")
            w_block = ds.read(n_idx, window=win).astype("float64")
            bad_block = missing_mask(s_block, nodata)

            s, w, bad = s_block[inside], w_block[inside], bad_block[inside]
            n_in += s.size
            n_nan += int(np.isnan(s).sum())
            n_nodata += int((bad & ~np.isnan(s)).sum())
            good = s[~bad]
            n_zero += int((good == 0).sum())
            n_above += int((good > OLI_CAP).sum())
            values.append(good.astype("float32"))
            ndwi.append(w[~bad & np.isfinite(w)].astype("float32"))

            r0, c0 = win.row_off, win.col_off
            valid_count[r0:r0 + win.height, c0:c0 + win.width] += (inside & ~bad_block).astype(np.uint8)

    v = np.concatenate(values) if values else np.array([], dtype="float32")
    w = np.concatenate(ndwi) if ndwi else np.array([], dtype="float32")
    p1, p50, p99 = np.percentile(v, [1, 50, 99]) if v.size else (np.nan,) * 3
    vmin = float(v.min()) if v.size else np.nan
    if not v.size:
        regime = "khong xac dinh"
    elif vmin >= TM_FLOOR - 0.005:
        regime = "TM (san ~0.237)"
    elif vmin < 0.05:
        regime = "OLI (min ~0)"
    else:
        regime = "khong khop che do nao"
    info.update({
        "boundary_pixels": n_in, "nan_pixels": n_nan, "nodata_pixels": n_nodata,
        "missing_pct_polygon": round(100.0 * (n_nan + n_nodata) / n_in, 3) if n_in else np.nan,
        "zero_pixels": n_zero, "above_28_013_pixels": n_above,
        "min": vmin, "p1": float(p1), "p50": float(p50), "mean": float(v.mean()) if v.size else np.nan,
        "p99": float(p99), "max": float(v.max()) if v.size else np.nan,
        "sensor_regime_inferred": regime,
        "ndwi_min": float(w.min()) if w.size else np.nan,
        "ndwi_p50": float(np.percentile(w, 50)) if w.size else np.nan,
        "ndwi_max": float(w.max()) if w.size else np.nan,
    })
    return info


def main():
    ap = argparse.ArgumentParser(description="Khao sat bo nhan Zenodo 15653696")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--download", action="store_true", help="Tai file con thieu va kiem tra MD5")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--footprint-share", type=float, default=0.5, help="Pixel hop le o >= ty le nay so nam")
    args = ap.parse_args()

    manifest = fetch_manifest()
    md5 = {f["key"]: f["md5"] for f in manifest}
    if args.download:
        download(manifest, args.data_dir)

    files = sorted(k for k in md5 if os.path.exists(os.path.join(args.data_dir, k)))
    if not files:
        sys.exit(f"Khong co file .tif nao trong {args.data_dir}. Chay lai voi --download.")
    print(f"{len(files)}/{len(md5)} file co san trong {args.data_dir}", flush=True)

    with rasterio.open(os.path.join(args.data_dir, files[0])) as ds0:
        shape, transform, crs = (ds0.height, ds0.width), ds0.transform, ds0.crs
    boundary = gpd.read_file(BOUNDARY).to_crs(crs).geometry.union_all()
    valid_count = np.zeros(shape, dtype=np.uint8)

    rows = []
    for key in files:
        path = os.path.join(args.data_dir, key)
        with rasterio.open(path) as ds:
            if (ds.height, ds.width) != shape or ds.transform != transform:
                sys.exit(f"{key}: luoi pixel khac nam dau tien - can can chinh truoc khi tinh footprint.")
        print(f"  khao sat {key} ...", flush=True)
        info = survey_year(path, boundary, valid_count)
        info.update({"year": int(key[:4]), "file": key, "md5_ok": md5sum(path) == md5[key]})
        rows.append(info)

    inside = ~geometry_mask([boundary], out_shape=shape, transform=transform)
    footprint = inside & (valid_count >= np.ceil(args.footprint_share * len(files)))
    n_fp = int(footprint.sum())
    print(f"footprint: {n_fp} pixel ({100.0 * n_fp / inside.sum():.2f}% polygon), nguong >= {args.footprint_share:.0%} so nam")

    for row in rows:
        with rasterio.open(os.path.join(args.data_dir, row["file"])) as ds:
            s_idx = band_index(ds, "Salinity", 2)
            missing = 0
            for _, win in ds.block_windows(s_idx):
                fp = footprint[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width]
                if fp.any():
                    missing += int((missing_mask(ds.read(s_idx, window=win), ds.nodata) & fp).sum())
        row["footprint_pixels"] = n_fp
        row["missing_pct_footprint"] = round(100.0 * missing / n_fp, 3) if n_fp else np.nan

    cols = ["year", "file", "md5_ok", "sensor_regime_inferred", "crs", "res_m", "width", "height", "dtype", "bands",
            "nodata_val", "boundary_pixels", "nan_pixels", "nodata_pixels", "missing_pct_polygon",
            "footprint_pixels", "missing_pct_footprint", "zero_pixels", "above_28_013_pixels",
            "min", "p1", "p50", "mean", "p99", "max", "ndwi_min", "ndwi_p50", "ndwi_max"]
    df = pd.DataFrame(rows)[cols].sort_values("year")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df[["year", "sensor_regime_inferred", "missing_pct_footprint", "zero_pixels",
              "above_28_013_pixels", "min", "p50", "max"]].to_string(index=False))
    print(f"Da ghi: {args.out}")


if __name__ == "__main__":
    main()

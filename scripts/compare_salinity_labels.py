#!/usr/bin/env python
"""Doi chieu nhan tu tai tao (GEE) voi nhan Zenodo cung nam, tung pixel (dieu kien truoc T-labels 2024-2026).

Anh tu tao duoc chieu lai ve DUNG luoi pixel cua Zenodo (nearest, cung EPSG:32648, 30 m),
doc theo khoi cua so de khong tran bo nho. Chi so tren pixel hop le o ca hai anh.

Chay:  python scripts/compare_salinity_labels.py --zenodo <2020_...tif> --ours <2020_..._l8.tif> [--out bang.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BANDS = {1: "NDWIchen", 2: "Salinity"}


def _valid(a, nodata):
    ok = np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):
        ok &= a != nodata
    return ok


def compare(zenodo_path, ours_path, tol=(0.01, 0.1)):
    """Tra ve dict chi so theo band. tol = nguong |chenh lech| cho (NDWI, Salinity)."""
    acc = {b: {"n": 0, "n_zen_only": 0, "n_ours_only": 0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0,
               "sxy": 0.0, "sad": 0.0, "sd": 0.0, "within": 0} for b in BANDS}
    with rasterio.open(zenodo_path) as z, rasterio.open(ours_path) as o:
        with WarpedVRT(o, crs=z.crs, transform=z.transform, width=z.width, height=z.height,
                       resampling=Resampling.nearest) as ov:
            for _, win in z.block_windows(1):
                for b, name in BANDS.items():
                    zx = z.read(b, window=win).astype(np.float64)
                    oy = ov.read(b, window=win).astype(np.float64)
                    zv, ovld = _valid(zx, z.nodata), _valid(oy, o.nodata)
                    both = zv & ovld
                    a = acc[b]
                    a["n_zen_only"] += int((zv & ~ovld).sum())
                    a["n_ours_only"] += int((ovld & ~zv).sum())
                    x, y = zx[both], oy[both]
                    d = y - x
                    a["n"] += x.size
                    a["sx"] += x.sum(); a["sy"] += y.sum()
                    a["sxx"] += (x * x).sum(); a["syy"] += (y * y).sum(); a["sxy"] += (x * y).sum()
                    a["sad"] += np.abs(d).sum(); a["sd"] += d.sum()
                    a["within"] += int((np.abs(d) <= tol[b - 1]).sum())
    rows = []
    for b, name in BANDS.items():
        a = acc[b]
        n = a["n"]
        if n == 0:
            rows.append({"band": name, "n_both": 0})
            continue
        cov = a["sxy"] / n - (a["sx"] / n) * (a["sy"] / n)
        vx = a["sxx"] / n - (a["sx"] / n) ** 2
        vy = a["syy"] / n - (a["sy"] / n) ** 2
        rows.append({
            "band": name, "n_both": n, "n_zenodo_only": a["n_zen_only"], "n_ours_only": a["n_ours_only"],
            "pearson_r": cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan,
            "mae": a["sad"] / n, "bias_ours_minus_zenodo": a["sd"] / n,
            "pct_within_tol": 100 * a["within"] / n, "tol": tol[b - 1],
        })
    return pd.DataFrame(rows)


def main(args):
    df = compare(args.zenodo, args.ours)
    df.insert(0, "ours", os.path.basename(args.ours))
    df.insert(0, "zenodo", os.path.basename(args.zenodo))
    print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"Da ghi them vao: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zenodo", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())

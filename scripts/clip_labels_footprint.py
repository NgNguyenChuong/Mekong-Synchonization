#!/usr/bin/env python
"""Cat nhan GEE tu tao (2024-2026, luoi _v2) theo dau chan nhan Zenodo (src/labels.py).

Dau chan = pixel Salinity huu han o >= 1 nam 2014-2023 (Zenodo). Can luoi theo transform (lech nguyen
pixel), khong noi suy. Mac dinh CHI DEM so pixel bi cat (toan raster + trong ranh gioi v1, v2);
`--out-dir` -> ghi them file `<ten>_fp.tif` (ca hai band NaN ngoai dau chan; file da co -> bo qua).

Chay:  python scripts/clip_labels_footprint.py [--labels A:/Dataset_NCKH/gee/2024_MD_dry_NDWIchen_Salinity_l8_v2.tif ...]
           [--zenodo-dir A:/Dataset_NCKH/zenodo_15653696] [--out-dir A:/Dataset_NCKH/labels_fp]
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from labels import FOOTPRINT_YEARS, ZENODO_DIR, build_footprint, clip_label_file, zenodo_label_paths  # noqa: E402
from preprocessing import BOUNDARY_V1, CANONICAL_BOUNDARY  # noqa: E402

GEE_DIR = os.environ.get("GEE_DIR", "A:/Dataset_NCKH/gee")
DEFAULT_LABELS = [os.path.join(GEE_DIR, f"{y}_MD_dry_NDWIchen_Salinity_l8_v2.tif") for y in (2024, 2025, 2026)]
PX_KM2 = 900 / 1e6


def _rasterize_boundary(path, ref_path):
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(ref_path) as r:
        crs, transform, shape = r.crs, r.transform, (r.height, r.width)
    g = gpd.read_file(path).to_crs(crs)
    return rasterize([(geom, 1) for geom in g.geometry], out_shape=shape, transform=transform, fill=0,
                     dtype="uint8", all_touched=False).astype(bool)


def main(a):
    zpaths = zenodo_label_paths(a.zenodo_dir, FOOTPRINT_YEARS)
    print(f"Dau chan tu {len(zpaths)} file Zenodo {FOOTPRINT_YEARS[0]}-{FOOTPRINT_YEARS[-1]}", flush=True)
    fp, fp_tr, _ = build_footprint(zpaths)
    print(f"  pixel dau chan {int(fp.sum())} ({fp.sum() * PX_KM2:.1f} km²)", flush=True)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    import rasterio

    cache = {}   # (transform, shape) -> ranh gioi rasterize (moi luoi mot lan)
    for lp in a.labels:
        name = os.path.basename(lp)
        with rasterio.open(lp) as r:
            key = (tuple(r.transform), r.height, r.width)
        if key not in cache:
            cache[key] = {"v1": _rasterize_boundary(a.boundary_v1, lp), "v2": _rasterize_boundary(a.boundary_v2, lp)}
        regions = cache[key]
        dst = None
        if a.out_dir:
            dst = os.path.join(a.out_dir, name.replace(".tif", "_fp.tif"))
            if os.path.exists(dst):
                print(f"{name}: da co {dst} - bo qua", flush=True)
                continue
            free = shutil.disk_usage(os.path.dirname(os.path.abspath(dst))).free
            need = 1.2 * os.path.getsize(lp)
            if free < need:
                raise SystemExit(f"Khong du cho trong ({free / 1e9:.2f} GB < {need / 1e9:.2f} GB) cho {dst}")
        st = clip_label_file(lp, fp, fp_tr, dst_path=dst, regions=regions)
        line = [f"{name}:"]
        for tag in regions:
            line.append(f"{tag}: cat {st[f'n_cut_{tag}']} px ({st[f'n_cut_{tag}'] * PX_KM2:.1f} km²), "
                        f"con {st[f'n_finite_after_{tag}']}")
        line.append(f"toan raster: cat {st['n_cut']} / huu han truoc {st['n_finite_before']}")
        print(" | ".join(line), flush=True)
        if dst:
            print(f"  da ghi {dst}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--zenodo-dir", default=ZENODO_DIR)
    ap.add_argument("--boundary-v1", default=BOUNDARY_V1)
    ap.add_argument("--boundary-v2", default=CANONICAL_BOUNDARY)
    ap.add_argument("--out-dir", default=None, help="Ghi file da cat; bo trong -> chi dem")
    main(ap.parse_args())

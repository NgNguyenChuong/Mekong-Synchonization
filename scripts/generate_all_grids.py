#!/usr/bin/env python
"""Sinh 13 khung luoi cho DBSCL va bang doi chieu dien tich (T1-03).

Kich thuoc o vuong va buoc kinh-vi do duoc chon de khop dien tich trung binh
CUC BO cua luoi H3 tren ranh gioi chuan (khong phai trung binh toan cau).
S2 chon level gan nhat theo ty le logarit; chenh lech dien tich con lai la
dac tinh cua he S2 va duoc xu ly bang bien kiem soat dien tich khi phan tich.

Quy tac chon o (An duyet 2026-10-03): o thuoc luoi neu GIAO ranh gioi (overlap_frac > 0), giu
nguyen hinh o; moi file luoi kem <luoi>.geojson.provenance.json (sha256 ranh gioi). Bang ghi them
% ranh gioi khong duoc phu (phai = 0) va phan bo overlap_frac; --previous-dir de so so o voi luoi cu.

Chay:  python scripts/generate_all_grids.py [--previous-dir data/grids/deprecated_centroid_rule]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyproj

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import (  # noqa: E402
    CANONICAL_BOUNDARY,
    file_sha256,
    write_provenance,
    generate_h3_grid,
    generate_latlon_grid,
    generate_s2_grid,
    generate_square_grid,
)

DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "data", "grids")
DEFAULT_REPORT = os.path.join(ROOT, "KE_HOACH", "ket-qua", "dot4_doi_chieu_dien_tich.csv")
GEOD = pyproj.Geod(ellps="WGS84")

# (khung, nhan, tier, ham sinh luoi). tier = muc do phan giai tuong ung H3 res.
GRID_SPECS = [
    ("H3", "res_5", 5, lambda b: generate_h3_grid(b, 5)),
    ("H3", "res_6", 6, lambda b: generate_h3_grid(b, 6)),
    ("H3", "res_7", 7, lambda b: generate_h3_grid(b, 7)),
    ("S2", "level_9", 5, lambda b: generate_s2_grid(b, 9)),
    ("S2", "level_10", 6, lambda b: generate_s2_grid(b, 10)),  # luoi kep o res 6
    ("S2", "level_11", 6, lambda b: generate_s2_grid(b, 11)),  # luoi chinh o res 6
    ("S2", "level_12", 7, lambda b: generate_s2_grid(b, 12)),
    ("Square_UTM", "17087m", 5, lambda b: generate_square_grid(b, 17087)),
    ("Square_UTM", "6458m", 6, lambda b: generate_square_grid(b, 6458)),
    ("Square_UTM", "2441m", 7, lambda b: generate_square_grid(b, 2441)),
    ("LatLon", "0.1552deg", 5, lambda b: generate_latlon_grid(b, 0.1552)),
    ("LatLon", "0.0586deg", 6, lambda b: generate_latlon_grid(b, 0.0586)),
    ("LatLon", "0.0222deg", 7, lambda b: generate_latlon_grid(b, 0.0222)),
]


def geodesic_km2(geom) -> float:
    return abs(GEOD.geometry_area_perimeter(geom)[0]) / 1e6


def main(boundary, output_dir, report_csv, previous_dir=None):
    import geopandas as gpd

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_csv), exist_ok=True)
    bnd = gpd.read_file(boundary).to_crs(4326)
    boundary_km2 = geodesic_km2(bnd.geometry.union_all())
    boundary_m = bnd.to_crs("EPSG:32648").geometry.union_all()
    sha = file_sha256(boundary)
    print(f"Ranh gioi: {boundary} ({boundary_km2:.2f} km2, sha256 {sha[:12]})")

    rows, h3_mean = [], {}
    for framework, label, tier, build in GRID_SPECS:
        t0 = time.time()
        gdf = build(boundary)
        areas = np.array([geodesic_km2(g) for g in gdf.geometry])
        planar = gdf.to_crs("EPSG:32648").geometry.area.to_numpy() / 1e6
        gdf["area_km2"] = np.round(areas, 6)
        gdf = gdf[["cell_id", "area_km2", "overlap_frac", "geometry"]]

        out = os.path.join(output_dir, f"{framework.lower()}_{label}.geojson")
        gdf.to_file(out, driver="GeoJSON")
        write_provenance(out, boundary, selection_rule="intersects (overlap_frac > 0), o giu nguyen hinh",
                         framework=framework, resolution=label, cells=len(gdf))

        # % ranh gioi khong duoc o nao phu (quy tac giao -> phai = 0), do trong EPSG:32648.
        cells_m = gdf.to_crs("EPSG:32648").geometry.union_all()
        uncovered_pct = boundary_m.difference(cells_m).area / boundary_m.area * 100.0
        frac = gdf["overlap_frac"].to_numpy()
        prev = None
        if previous_dir:
            prev_path = os.path.join(previous_dir, os.path.basename(out))
            if os.path.exists(prev_path):
                prev = len(gpd.read_file(prev_path))

        mean = float(areas.mean())
        if framework == "H3":
            h3_mean[tier] = mean
        diff = (mean / h3_mean[tier] - 1.0) * 100.0
        rows.append({
            "framework": framework,
            "resolution": label,
            "tier_h3_res": tier,
            "cell_count": len(gdf),
            "mean_area_km2": round(mean, 4),
            "std_area_km2": round(float(areas.std()), 4),
            "min_area_km2": round(float(areas.min()), 4),
            "max_area_km2": round(float(areas.max()), 4),
            "area_cv_pct": round(float(areas.std() / mean * 100.0), 4),
            "diff_from_h3_pct": round(diff, 2),
            "sum_area_km2": round(float(areas.sum()), 2),
            "boundary_area_km2": round(boundary_km2, 2),
            "area_coverage_ratio": round(float(areas.sum()) / boundary_km2, 4),
            "mean_overlap_frac": round(float(gdf["overlap_frac"].mean()), 4),
            "cells_overlap_lt_1": int((gdf["overlap_frac"] < 1.0).sum()),
            "cells_overlap_lt_0.5": int((frac < 0.5).sum()),
            "cells_overlap_lt_0.1": int((frac < 0.1).sum()),
            "cells_overlap_lt_0.01": int((frac < 0.01).sum()),
            "overlap_frac_min": float(frac.min()),
            "overlap_frac_p10": round(float(np.quantile(frac, 0.10)), 4),
            "overlap_frac_p50": round(float(np.quantile(frac, 0.50)), 4),
            "uncovered_boundary_pct": round(float(uncovered_pct), 8),
            "cell_count_prev": prev,
            "boundary_sha256": sha,
            "planar_sum_area_km2": round(float(planar.sum()), 2),
            "geo_planar_diff_pct": round(abs(areas.sum() - planar.sum()) / areas.sum() * 100.0, 4),
            "file": os.path.relpath(out, ROOT).replace("\\", "/"),
        })
        print(f"  {framework:10s} {label:10s} {len(gdf):6d} o (truoc {prev}) | TB {mean:9.4f} km2 | lech H3 {diff:+7.2f}% "
              f"| khong phu {uncovered_pct:.2e}% | frac<0.1: {(frac < 0.1).sum()} | {time.time() - t0:5.1f}s", flush=True)

    pd.DataFrame(rows).to_csv(report_csv, index=False)
    print(f"Bang doi chieu: {report_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--report-csv", default=DEFAULT_REPORT)
    ap.add_argument("--previous-dir", default=None, help="Thu muc luoi cu de so so o truoc/sau")
    args = ap.parse_args()
    main(args.boundary, args.output_dir, args.report_csv, args.previous_dir)

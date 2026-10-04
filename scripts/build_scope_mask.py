#!/usr/bin/env python
"""Dung mat na pham vi TINH v3 (CHG-10, An chot 2026-10-03) -> A:/Dataset_NCKH/features/scope_mask_v3.tif.

pham vi v3 = ranh gioi v2 ∩ WorldCover 2021 THUOC {10, 20, 30, 40} (tam pixel 30 m)
           ∩ dau chan Zenodo (Salinity huu han o >= 1 nam 2014-2023).
WorldCover mac dinh = LandCover_DBSCL_2021_v2.tif (10 m, dem 25 km quanh v2).
Luoi = luoi 30 m cua file nhan Zenodo (src/scope_mask.py). Band: scope (0/1), zenodo_n_years (0-10),
wc_class (lop WC tai tam pixel 30 m trong v2; 0 ngoai) -> bo phu dung lai duoc.
Kiem kem (bang o --report-dir):
  - WorldCover moi vs cu (--compare-worldcover) tren pixel 10 m trong v2 va tai tam pixel 30 m: phai trung;
    bao so pixel khac (dot4_scope_v3_worldcover_moi_vs_cu.csv).
  - Dien tich pham vi cu (--compare-scope, mac dinh v2) -> moi, phan mat/them theo lop (dot4_scope_v3_dien_tich.csv).
Da co file -> khong ghi lai (tru --force). Kem .provenance.json (sha256 ranh gioi, WorldCover, nhan).

Chay:  venv/Scripts/python.exe scripts/build_scope_mask.py [--force]
Chay lai v2 (quy tac cu, 2 band):
       venv/Scripts/python.exe scripts/build_scope_mask.py --worldcover A:/Dataset_NCKH/gee/LandCover_DBSCL_2021.tif
           --exclude-lc 0 80 95 --no-wc-class --compare-scope none --compare-worldcover none
           --out <thu_muc_khac>/scope_mask_v2.tif
"""
import argparse
import os
import shutil
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import CANONICAL_BOUNDARY, file_sha256, write_provenance  # noqa: E402
from scope_mask import (SCOPE_INCLUDE_LC, band_index, build_scope_mask, compare_worldcover,  # noqa: E402
                        scope_change_by_class, wc_at_centers, write_scope_mask)

WC_NAMES = {0: "ngoai_vung", 10: "cay", 20: "bui", 30: "co", 40: "canh_tac", 50: "xay_dung", 60: "dat_trong",
            70: "bang_tuyet", 80: "nuoc", 90: "dat_ngap", 95: "ngap_man", 100: "reu"}


def log(msg):
    print(msg, flush=True)


def _none(p):
    return None if p is None or str(p).lower() in ("", "none") else p


def main(a):
    if os.path.exists(a.out) and not a.force:
        log(f"Da co {a.out} - bo qua (dung --force de dung lai).")
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    free_gb = shutil.disk_usage(os.path.dirname(a.out)).free / 1e9
    log(f"Cho trong o dich: {free_gb:.1f} GB")
    if free_gb < 1.0:
        raise SystemExit("[LOI] o dich con < 1 GB.")
    labels = [os.path.join(a.zenodo_dir, f"{y}_MD_dry_NDWIchen_Salinity.tif") for y in range(a.y0, a.y1 + 1)]
    miss = [p for p in labels if not os.path.exists(p)]
    if miss:
        raise SystemExit(f"[LOI] thieu file nhan: {miss}")
    import rasterio

    with rasterio.open(labels[0]) as ref:
        crs = ref.crs
    boundary = gpd.read_file(a.boundary).to_crs(crs).geometry.union_all()
    include = None if a.exclude_lc else a.include_lc
    log(f"Dung mat na tren luoi {labels[0]} ({a.y0}-{a.y1}, {len(labels)} nam); WorldCover {a.worldcover}; "
        f"{'giu ' + str(include) if include else 'bo ' + str(a.exclude_lc)} ...")
    res = build_scope_mask(boundary, a.worldcover, labels, include=include, exclude=a.exclude_lc or None, log=log)
    px_km2 = abs(res["transform"].a * res["transform"].e) / 1e6
    inb, wc, ny, sc = res["in_boundary"], res["wc_ok"], res["zenodo_n_years"], res["scope"].astype(bool)
    cls = res["wc_class"]
    stats = {
        "pixel_km2": px_km2,
        "in_boundary_km2": round(float(inb.sum() * px_km2), 2),
        "in_boundary_wc_ok_km2": round(float((inb & wc).sum() * px_km2), 2),
        "in_boundary_wc_ok_no_footprint_km2": round(float((inb & wc & (ny == 0)).sum() * px_km2), 2),
        "footprint_km2": round(float((ny > 0).sum() * px_km2), 2),
        "footprint_outside_boundary_km2": round(float(((ny > 0) & ~inb).sum() * px_km2), 2),
        "scope_km2": round(float(sc.sum() * px_km2), 2),
        "scope_all_years_km2": round(float((sc & (ny == len(labels))).sum() * px_km2), 2),
        "scope_n_years_hist": {int(k): int(v) for k, v in zip(*np.unique(ny[sc], return_counts=True))},
        "scope_km2_by_wc_class": {int(k): round(float(v * px_km2), 2) for k, v in zip(*np.unique(cls[sc],
                                                                                            return_counts=True))},
        "in_boundary_footprint_km2_by_wc_class": {
            int(k): round(float(v * px_km2), 2) for k, v in zip(*np.unique(cls[inb & (ny > 0)], return_counts=True))},
    }
    for k, v in stats.items():
        log(f"  {k}: {v}")

    compare = {}
    cmp_wc = _none(a.compare_worldcover)
    if cmp_wc:
        log(f"So WorldCover moi vs cu ({cmp_wc}) trong ranh gioi v2 ...")
        c10 = compare_worldcover(a.worldcover, cmp_wc, boundary)
        old_c = wc_at_centers(cmp_wc, res["transform"], res["crs"], res["shape"])
        old_c[~inb] = 0
        d30 = inb & (old_c != cls)
        compare["worldcover_10m"] = {k: (v if k != "diff_pairs" else {f"{p[0]}->{p[1]}": n for p, n in v.items()})
                                     for k, v in c10.items()}
        compare["worldcover_30m_center"] = {"n_px_in_boundary": int(inb.sum()), "n_diff": int(d30.sum()),
                                            "n_diff_in_footprint": int((d30 & (ny > 0)).sum())}
        log(f"  10 m: {c10['n_diff']:,} / {c10['n_px_in_boundary']:,} pixel khac (ngoai khung moi "
            f"{c10['n_outside_new']:,}); offset {c10['offset']}; cap khac {c10['diff_pairs']}")
        log(f"  tam 30 m: {int(d30.sum()):,} / {int(inb.sum()):,} pixel khac")
        if a.report_dir:
            rows = [{"muc": "pixel_10m_trong_v2", "n_px": c10["n_px_in_boundary"], "n_khac": c10["n_diff"],
                     "n_ngoai_khung_moi": c10["n_outside_new"], "cap_khac": str(c10["diff_pairs"]),
                     "offset_cot_hang": str(c10["offset"])},
                    {"muc": "tam_pixel_30m_trong_v2", "n_px": int(inb.sum()), "n_khac": int(d30.sum()),
                     "n_ngoai_khung_moi": "", "cap_khac": "", "offset_cot_hang": ""},
                    {"muc": "tam_pixel_30m_trong_v2_va_dau_chan", "n_px": int((inb & (ny > 0)).sum()),
                     "n_khac": int((d30 & (ny > 0)).sum()), "n_ngoai_khung_moi": "", "cap_khac": "",
                     "offset_cot_hang": ""}]
            os.makedirs(a.report_dir, exist_ok=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.report_dir, "dot4_scope_v3_worldcover_moi_vs_cu.csv"),
                                      index=False)
    cmp_sc = _none(a.compare_scope)
    if cmp_sc:
        with rasterio.open(cmp_sc) as s:
            if (s.transform, (s.height, s.width)) != (res["transform"], res["shape"]):
                raise SystemExit(f"[LOI] {cmp_sc} khac luoi.")
            old_scope = s.read(band_index(s, "scope")) == 1
        ch = scope_change_by_class(old_scope, res)
        for c in ("old_px", "new_px", "lost_px", "gained_px"):
            ch[c.replace("_px", "_km2")] = (ch[c] * px_km2).round(2)
        ch.insert(1, "ten_lop", ch["wc_class"].map(WC_NAMES))
        tot = ch[[c for c in ch.columns if c.endswith(("_px", "_km2"))]].sum()
        tot_row = {"wc_class": "tong", "ten_lop": "", **tot.to_dict()}
        ch = pd.concat([ch, pd.DataFrame([tot_row])], ignore_index=True)
        for c in ("old_px", "new_px", "lost_px", "gained_px"):
            ch[c] = ch[c].astype("int64")
        ch["lost_pct_of_old_total"] = (100 * ch["lost_px"] / old_scope.sum()).round(3)
        compare["scope_change_vs"] = {"path": cmp_sc, "sha256": file_sha256(cmp_sc),
                                      "old_km2": round(float(old_scope.sum() * px_km2), 2),
                                      "new_km2": stats["scope_km2"]}
        log(f"Pham vi {os.path.basename(cmp_sc)} -> moi theo lop WC (moi):\n{ch.to_string(index=False)}")
        if a.report_dir:
            os.makedirs(a.report_dir, exist_ok=True)
            ch.to_csv(os.path.join(a.report_dir, "dot4_scope_v3_dien_tich.csv"), index=False)

    write_scope_mask(a.out, res, with_wc_class=not a.no_wc_class)
    write_provenance(a.out, a.boundary, function="scope_mask.build_scope_mask", rule=res["rule"],
                     include_worldcover=list(res["include"]) if res["include"] else None,
                     exclude_worldcover=list(res["exclude"]) if res["exclude"] else None,
                     change="CHG-10 (2026-10-03): bo chinh chi WorldCover 10, 20, 30, 40" if res["include"] else None,
                     worldcover=a.worldcover, worldcover_sha256=file_sha256(a.worldcover), label_years=[a.y0, a.y1],
                     labels={os.path.basename(p): file_sha256(p) for p in labels},
                     grid_ref=os.path.basename(labels[0]),
                     bands=["scope", "zenodo_n_years"] + ([] if a.no_wc_class else ["wc_class"]),
                     stats=stats, compare=compare)
    log(f"Ghi {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--worldcover", default="A:/Dataset_NCKH/gee/LandCover_DBSCL_2021_v2.tif")
    ap.add_argument("--include-lc", type=int, nargs="+", default=list(SCOPE_INCLUDE_LC),
                    help="Lop WorldCover GIU (mac dinh v3: 10 20 30 40)")
    ap.add_argument("--exclude-lc", type=int, nargs="+", default=None,
                    help="Thay --include-lc: lop WorldCover BO (v2: 0 80 95)")
    ap.add_argument("--no-wc-class", action="store_true", help="Khong ghi band wc_class (dinh dang v2)")
    ap.add_argument("--zenodo-dir", default="A:/Dataset_NCKH/zenodo_15653696")
    ap.add_argument("--y0", type=int, default=2014)
    ap.add_argument("--y1", type=int, default=2023)
    ap.add_argument("--out", default="A:/Dataset_NCKH/features/scope_mask_v3.tif")
    ap.add_argument("--compare-scope", default="A:/Dataset_NCKH/features/scope_mask_v2.tif",
                    help="Mat na cu de bao dien tich mat/them theo lop ('none' = bo qua)")
    ap.add_argument("--compare-worldcover", default="A:/Dataset_NCKH/gee/LandCover_DBSCL_2021.tif",
                    help="WorldCover cu de kiem trung gia tri trong v2 ('none' = bo qua)")
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())

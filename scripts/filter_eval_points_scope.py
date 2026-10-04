#!/usr/bin/env python
"""LOC diem danh gia theo mat na pham vi v3 (CHG-10, An chot 2026-10-03) - KHONG sinh lai diem.

- Nguon = ban diem TRUOC khi loc (data/eval/deprecated_v2b/eval_points.geojson; lan dau: sao luu ban dang co
  vao do roi dung lam nguon) -> chay lai cho cung ket qua (tai lap), khong loc chong.
- Giu diem khi pixel 30 m CHUA diem co scope_v3 = 1 (src/scope_mask.scope_values_at_points). point_id, block_id,
  is_holdout, thu tu va toa do GIU NGUYEN (kiem lai sau khi ghi).
- Bao cao (--report-dir): diem bi loai theo lop WC (tam pixel 30 m) x giu rieng/CV, theo khoi, theo dai khoang
  cach bien (<= 5, 5-20, > 20 km; dist_coast_km tai pixel 90 m chua diem).
- Provenance moi: sha256 scope v3, sha256 nguon, so diem truoc/sau; giu provenance cu trong muc "parent".

Chay:  venv/Scripts/python.exe scripts/filter_eval_points_scope.py [--dry-run]
"""
import argparse
import json
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

from eval_design import raster_values_at  # noqa: E402
from preprocessing import CANONICAL_BOUNDARY, file_sha256, provenance_path, write_provenance  # noqa: E402
from scope_mask import band_index, scope_values_at_points  # noqa: E402

WC_NAMES = {0: "ngoai_vung", 10: "cay", 20: "bui", 30: "co", 40: "canh_tac", 50: "xay_dung", 60: "dat_trong",
            70: "bang_tuyet", 80: "nuoc", 90: "dat_ngap", 95: "ngap_man", 100: "reu"}


def log(msg):
    print(msg, flush=True)


def rel(p):
    """Duong dan tuong doi ROOT; khac o dia (Windows) -> duong dan tuyet doi."""
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:
        return os.path.abspath(p).replace("\\", "/")


def coast_band(d, edges=(5.0, 20.0)):
    """Dai khoang cach bien: '<=5', '5-20', '>20' (canh tren TINH vao dai duoi); NaN -> 'nan'."""
    d = np.asarray(d, float)
    lo, hi = edges
    out = np.where(d <= lo, f"<={lo:g}", np.where(d <= hi, f"{lo:g}-{hi:g}", f">{hi:g}"))
    return np.where(np.isfinite(d), out, "nan")


def point_attrs(points, scope_path, dist_raster):
    """scope/zenodo_n_years/wc_class (pixel 30 m chua diem) + dist_coast_km (pixel 90 m chua diem)."""
    import rasterio

    a = scope_values_at_points(points, scope_path)
    with rasterio.open(dist_raster) as src:
        crs, b = src.crs, band_index(src, "dist_coast_km")
    p = points.to_crs(crs)
    a["dist_coast_km"] = raster_values_at(dist_raster, p.geometry.x.to_numpy(), p.geometry.y.to_numpy(), band=b,
                                          fill=np.nan).astype(float)
    a["coast_band"] = coast_band(a["dist_coast_km"])
    return a


def summarize(pts, keep):
    """Bang theo lop x giu rieng/CV, theo khoi, theo dai ven bien (truoc / bi loai / sau)."""
    d = pts.assign(dropped=~keep, kept=keep)
    d["nhom"] = np.where(d["is_holdout"], "giu_rieng", "cv")
    by_cls = (d[d["dropped"]].groupby(["wc_class", "nhom"]).size().unstack(fill_value=0)
              .reindex(columns=["giu_rieng", "cv"], fill_value=0))
    by_cls["tong"] = by_cls.sum(axis=1)
    by_cls = by_cls.reset_index()
    by_cls.insert(1, "ten_lop", by_cls["wc_class"].map(WC_NAMES))
    by_cls = pd.concat([by_cls, pd.DataFrame([{"wc_class": "tong", "ten_lop": "",
                                               **by_cls[["giu_rieng", "cv", "tong"]].sum().to_dict()}])],
                       ignore_index=True)

    def table(keys):
        g = d.groupby(keys)
        t = pd.DataFrame({"n_truoc": g.size(), "n_loai": g["dropped"].sum(), "n_sau": g["kept"].sum()})
        for c in sorted(d.loc[d["dropped"], "wc_class"].unique()):
            t[f"loai_wc{int(c)}"] = d[d["dropped"] & (d["wc_class"] == c)].groupby(keys).size().reindex(
                t.index, fill_value=0)
        t["pct_loai"] = (100 * t["n_loai"] / t["n_truoc"]).round(2)
        return t.reset_index()

    by_block = table(["block_id", "nhom"]).sort_values(["nhom", "block_id"])
    by_coast = table(["coast_band", "nhom"])
    tot = table(["coast_band"]).assign(nhom="tat_ca")
    by_coast = pd.concat([by_coast, tot], ignore_index=True)
    order = {"<=5": 0, "5-20": 1, ">20": 2, "nan": 3}
    by_coast = by_coast.sort_values(["nhom", "coast_band"], key=lambda s: s.map(order) if s.name == "coast_band"
                                    else s).reset_index(drop=True)
    return by_cls, by_block, by_coast


def main(a):
    points_path = a.points
    bdir = a.backup_dir
    src_path = os.path.join(bdir, os.path.basename(points_path))
    if not os.path.exists(src_path):
        if a.dry_run:
            src_path = points_path
        else:
            os.makedirs(bdir, exist_ok=True)
            for p in (points_path, provenance_path(points_path)):
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(bdir, os.path.basename(p)))
                    log(f"Sao luu {p} -> {bdir}")
    src_sha = file_sha256(src_path)
    log(f"Nguon (truoc khi loc): {src_path} sha {src_sha[:12]}")
    src = gpd.read_file(src_path)
    src["is_holdout"] = src["is_holdout"].astype(bool)
    if src["point_id"].duplicated().any():
        raise SystemExit("[LOI] point_id trung trong nguon.")
    attrs = point_attrs(src, a.scope, a.dist_raster)
    keep = (attrs["scope"] == 1).to_numpy()
    pts = pd.concat([src.drop(columns="geometry"), attrs], axis=1)
    if a.compare_scope and str(a.compare_scope).lower() != "none":
        old = scope_values_at_points(src, a.compare_scope)
        n_old_bad = int((old["scope"] != 1).sum())
        log(f"Kiem: diem nguon co scope cu ({os.path.basename(a.compare_scope)}) != 1: {n_old_bad} (phai 0)")
        if n_old_bad:
            raise SystemExit("[LOI] nguon co diem ngoai mat na cu - nguon sai?")
    n_h, n_c = int(src["is_holdout"].sum()), int((~src["is_holdout"]).sum())
    kh, kc = int((keep & src["is_holdout"]).sum()), int((keep & ~src["is_holdout"]).sum())
    log(f"Diem: {len(src)} (giu rieng {n_h}, CV {n_c}) -> giu {int(keep.sum())} (giu rieng {kh}, CV {kc}); "
        f"loai {int((~keep).sum())} (giu rieng {n_h - kh}, CV {n_c - kc})")
    by_cls, by_block, by_coast = summarize(pts, keep)
    log(f"--- Loai theo lop WC (tam pixel 30 m) ---\n{by_cls.to_string(index=False)}")
    log(f"--- Theo dai khoang cach bien ---\n{by_coast.to_string(index=False)}")
    log(f"--- Theo khoi (chi khoi co diem bi loai) ---\n"
        f"{by_block[by_block['n_loai'] > 0].to_string(index=False)}")
    if a.report_dir:
        os.makedirs(a.report_dir, exist_ok=True)
        ll = src.to_crs(4326).geometry
        drop = pts.loc[~keep, ["point_id", "block_id", "is_holdout", "wc_class", "zenodo_n_years", "dist_coast_km",
                               "coast_band"]].copy()
        drop.insert(4, "ten_lop", drop["wc_class"].map(WC_NAMES))
        drop["lon"], drop["lat"] = ll.x[~keep].round(6).to_numpy(), ll.y[~keep].round(6).to_numpy()
        drop.to_csv(os.path.join(a.report_dir, "dot4_scope_v3_diem_bi_loai.csv"), index=False)
        by_cls.to_csv(os.path.join(a.report_dir, "dot4_scope_v3_diem_loai_theo_lop.csv"), index=False)
        by_block.to_csv(os.path.join(a.report_dir, "dot4_scope_v3_diem_theo_khoi.csv"), index=False)
        by_coast.to_csv(os.path.join(a.report_dir, "dot4_scope_v3_diem_theo_ven_bien.csv"), index=False)
        log(f"Bang -> {a.report_dir}/dot4_scope_v3_diem_*.csv")
    if a.dry_run:
        log("--dry-run: khong ghi diem.")
        return

    out = src.loc[keep].copy()
    out.to_file(points_path, driver="GeoJSON")
    back = gpd.read_file(points_path)
    same = (list(back["point_id"]) == list(out["point_id"])
            and np.array_equal(back.geometry.x.to_numpy(), out.geometry.x.to_numpy())
            and np.array_equal(back.geometry.y.to_numpy(), out.geometry.y.to_numpy())
            and (back["block_id"].to_numpy() == out["block_id"].to_numpy()).all()
            and (back["is_holdout"].astype(bool).to_numpy() == out["is_holdout"].to_numpy()).all())
    if not same:
        raise SystemExit("[LOI] diem ghi lai khac nguon (toa do/point_id/block_id) - kiem lai.")
    chk = scope_values_at_points(back, a.scope)
    if (chk["scope"] != 1).any():
        raise SystemExit("[LOI] con diem scope_v3 != 1 sau khi loc.")
    parent = {}
    pp = provenance_path(src_path)
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            parent = json.load(f)
    write_provenance(points_path, a.boundary,
                     rule="LOC (khong sinh lai): giu diem khi pixel 30 m chua diem co scope = 1 tren mat na v3 "
                          "(v2 ∩ WorldCover {10,20,30,40} ∩ dau chan Zenodo; CHG-10)",
                     scope_mask=a.scope, scope_mask_sha256=file_sha256(a.scope),
                     filtered_from=rel(src_path), filtered_from_sha256=src_sha,
                     n_before=len(src), n_after=int(keep.sum()), n_holdout=kh, n_cv=kc,
                     n_dropped_holdout=n_h - kh, n_dropped_cv=n_c - kc,
                     dropped_by_wc_class={str(int(k)): int(v) for k, v in
                                          pts.loc[~keep, "wc_class"].value_counts().sort_index().items()},
                     note="point_id/block_id/is_holdout/toa do giu nguyen tu nguon; seed_holdout/seed_cv o parent",
                     parent=parent)
    log(f"Ghi {points_path} ({len(out)} diem) + provenance")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", default=os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    ap.add_argument("--backup-dir", default=os.path.join(ROOT, "data", "eval", "deprecated_v2b"),
                    help="Ban truoc khi loc (nguon) - tao lan dau, khong ghi de")
    ap.add_argument("--scope", default="A:/Dataset_NCKH/features/scope_mask_v3.tif")
    ap.add_argument("--compare-scope", default="A:/Dataset_NCKH/features/scope_mask_v2.tif",
                    help="Mat na cu: moi diem nguon phai co scope = 1 ('none' = bo qua)")
    ap.add_argument("--dist-raster", default="A:/Dataset_NCKH/raw/river/river_distance_90m.tif")
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())

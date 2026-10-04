#!/usr/bin/env python
"""Sua georeference (transform trong header) cua file ERA5-Land ngay bi lech nua pixel (known-pitfalls 1a).

File `raw/daily_*/<bien>_YYYY_MM.tif` tai bang `scale=11132` + EPSG:4326 co goc 103,700474 / 11,200051 nhung
gia tri la cua pixel luoi goc GEE (crs_transform [0.1,0,-180.05,0,-0.1,90.05]) lech +0,05 do Dong, +0,05 do Bac.
Script dat transform = luoi goc (src/era5_georef.native_transform), KHONG doi du lieu (rasterio "r+").

  - Truoc khi sua: ghi <raw-dir>/era5_transform_fix.json (transform cu/moi, danh sach file, sha256 file
    va sha256 du lieu truoc sua); sau khi sua: bo sung sha256 sau sua + kiem du lieu khong doi.
  - Idempotent: file co transform da nam tren luoi goc -> bo qua (khong dich them). Chay lai sau khi
    dut giua chung chi sua phan con lai, giu ban ghi cu trong JSON.
  - --revert: tra transform cu cho file co trong JSON (chi file dang mang transform moi).
  - --verify-gee: tai vai ngay tu GEE theo luoi goc va so tung pixel voi file (theo transform hien tai
    va theo transform sau sua neu khac) - khong ghi gi vao file ERA5.

Chay:  python scripts/fix_era5_transform.py --dry-run
       python scripts/fix_era5_transform.py --verify-gee --project <EE_PROJECT>
       python scripts/fix_era5_transform.py            # sua that
       python scripts/fix_era5_transform.py --revert   # dao nguoc
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from era5_georef import (NATIVE_CRS_TRANSFORM, data_digest, is_native_aligned,  # noqa: E402
                         native_transform, transform_list)
from preprocessing import file_sha256  # noqa: E402

PATTERN = "daily_*/*.tif"
# ngay kiem voi GEE: (bien trong gee_fetch.ERA5_VARS, thu muc, ngay) - trung bo ngay data-auditor da dung
VERIFY_DAYS = [("temp_max", "daily_temp_max", "2016-02-10"), ("rain", "daily_rain", "2016-03-05"),
               ("solar", "daily_solar", "2021-01-15"), ("temp_min", "daily_temp_min", "2019-12-20"),
               ("humidity", "daily_humid", "2024-03-15"), ("temp_avg", "daily_temp_avg", "2014-01-05")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path, obj):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def list_files(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, PATTERN)))
    return [f.replace("\\", "/") for f in files]


def _rel(path, raw_dir):
    return os.path.relpath(path, raw_dir).replace("\\", "/")


def plan(files):
    """[(file, transform hien tai, transform moi | None neu da dung)]."""
    out = []
    for f in files:
        with rasterio.open(f) as ds:
            t = ds.transform
            new = None if is_native_aligned(t) else native_transform(t, ds.width, ds.height)
        out.append((f, t, new))
    return out


def fix_files(raw_dir, log_path, dry_run=False):
    files = list_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"Khong co file {PATTERN} trong {raw_dir}")
    items = plan(files)
    todo = [(f, t, n) for f, t, n in items if n is not None]
    log(f"{len(files)} file; da tren luoi goc: {len(files) - len(todo)}; can sua: {len(todo)}")
    for t_old, t_new in sorted({(tuple(transform_list(t)), tuple(transform_list(n))) for _, t, n in todo}):
        log(f"  cu {t_old} -> moi {t_new}")
    if dry_run or not todo:
        return items
    rec = {}
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as fh:
            rec = json.load(fh)
    rec.setdefault("purpose", "Sua georeference ERA5 lech nua pixel (scale=11132 -> luoi goc 0,1 do). "
                              "Chi doi transform trong header; du lieu khong doi.")
    rec.setdefault("native_crs_transform", NATIVE_CRS_TRANSFORM)
    rec.setdefault("raw_dir", raw_dir.replace("\\", "/"))
    rec.setdefault("files", {})
    # ghi ke hoach (transform cu/moi + sha256 truoc) TRUOC khi sua file nao
    for f, t, n in todo:
        key = _rel(f, raw_dir)
        if key in rec["files"] and rec["files"][key].get("status") == "planned":
            continue   # lan chay truoc dung giua chung: giu sha256 truoc ban dau
        with rasterio.open(f) as ds:
            digest = data_digest(ds)
        rec["files"][key] = {"transform_old": transform_list(t), "transform_new": transform_list(n),
                             "sha256_before": file_sha256(f), "data_sha256": digest, "status": "planned"}
    rec["planned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(log_path, rec)
    log(f"Da ghi ke hoach: {log_path}")
    for i, (f, t, n) in enumerate(todo, 1):
        key = _rel(f, raw_dir)
        with rasterio.open(f, "r+") as ds:
            ds.transform = n
        with rasterio.open(f) as ds:
            if not ds.transform.almost_equals(n, precision=1e-12):
                raise RuntimeError(f"{key}: transform sau sua {ds.transform} != {n}")
            if data_digest(ds) != rec["files"][key]["data_sha256"]:
                raise RuntimeError(f"{key}: DU LIEU DOI sau khi sua header")
        rec["files"][key].update(sha256_after=file_sha256(f), status="fixed")
        if i % 200 == 0 or i == len(todo):
            _write_json(log_path, rec)
            log(f"  da sua {i}/{len(todo)}")
    rec["fixed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(log_path, rec)
    return items


def revert_files(raw_dir, log_path):
    with open(log_path, encoding="utf-8") as fh:
        rec = json.load(fh)
    n_rev = 0
    for key, r in rec["files"].items():
        f = os.path.join(raw_dir, key)
        old, new = Affine(*r["transform_old"]), Affine(*r["transform_new"])
        with rasterio.open(f) as ds:
            cur = ds.transform
        if cur.almost_equals(old, precision=1e-12):
            continue
        if not cur.almost_equals(new, precision=1e-12):
            raise RuntimeError(f"{key}: transform hien tai {cur} khong phai cu/moi trong JSON")
        with rasterio.open(f, "r+") as ds:
            ds.transform = old
        with rasterio.open(f) as ds:
            if data_digest(ds) != r["data_sha256"]:
                raise RuntimeError(f"{key}: du lieu doi khi dao nguoc")
        r["status"] = "reverted"
        n_rev += 1
    rec["reverted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(log_path, rec)
    log(f"Da tra transform cu cho {n_rev} file")


# ---------- kiem voi GEE ----------

def _gee_native_day(ee, gee_fetch, var, day, bbox):
    """Tai 1 ngay ERA5 tu GEE theo luoi goc (cung bieu thuc + mat na dat nhu gee_fetch.cmd_era5)."""
    import pandas as pd
    _, fn = gee_fetch.ERA5_VARS[var]
    d0 = pd.Timestamp(day)
    im = ee.ImageCollection(gee_fetch.ERA5_ID).filterDate(d0.strftime("%Y-%m-%d"),
                                                         (d0 + pd.Timedelta(days=1)).strftime("%Y-%m-%d")).first()
    img = (gee_fetch._rh(ee, im) if fn is None else fn(im)).updateMask(im.select("temperature_2m").mask()).float()
    region = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)
    params = {"region": region, "crs": "EPSG:4326", "crs_transform": NATIVE_CRS_TRANSFORM, "format": "GEO_TIFF"}
    path = os.path.join(tempfile.mkdtemp(), f"{var}_{day}.tif")
    gee_fetch._retry(lambda: gee_fetch._download(img.getDownloadURL(params), path), f"GEE {var} {day}")
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float64")
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
        return arr, ds.transform


def compare_on_grid(local, t_local, ref, t_ref, shift=(0, 0)):
    """So tung pixel `local` (transform t_local) voi pixel `ref` chua tam pixel (dich them `shift` = (dr, dc)
    pixel ref). Tra ve dict dem + `footprint_overlap`: ty le dien tich trung binh cua o pixel `local` (theo
    t_local) nam trong pixel ref ma no khop gia tri - 1,0 = georeference dung; ~0,25 = lech nua pixel ca 2 truc.

    Chu y: so theo TAM pixel khong phan biet duoc transform cu/moi (tam pixel cu roi dung vao pixel goc
    giu gia tri) - bang chung la `footprint_overlap` va phep dich `shift` tren luoi da can.
    """
    h, w = local.shape
    rows, cols = np.mgrid[0:h, 0:w]
    xs, ys = t_local @ (cols + 0.5, rows + 0.5)
    rc, rr = ~t_ref @ (xs, ys)
    rr, rc = np.floor(rr).astype(int) + shift[0], np.floor(rc).astype(int) + shift[1]
    inside = (rr >= 0) & (rr < ref.shape[0]) & (rc >= 0) & (rc < ref.shape[1])
    refv = np.full(local.shape, np.nan)
    refv[inside] = ref[rr[inside], rc[inside]]
    lf, rf = np.isfinite(local), np.isfinite(refv)
    both = lf & rf
    match = both & np.isclose(local, refv, rtol=1e-5, atol=1e-6)
    # dien tich o pixel local nam trong pixel ref (rr, rc)
    lx0, ly0 = t_local @ (cols, rows)
    lx1, ly1 = t_local @ (cols + 1, rows + 1)
    rx0, ry0 = t_ref @ (rc, rr)
    rx1, ry1 = t_ref @ (rc + 1, rr + 1)
    ox = np.clip(np.minimum(np.maximum(lx0, lx1), np.maximum(rx0, rx1))
                 - np.maximum(np.minimum(lx0, lx1), np.minimum(rx0, rx1)), 0, None)
    oy = np.clip(np.minimum(np.maximum(ly0, ly1), np.maximum(ry0, ry1))
                 - np.maximum(np.minimum(ly0, ly1), np.minimum(ry0, ry1)), 0, None)
    frac = ox * oy / abs(t_local.a * t_local.e)
    return {"shift": f"{shift[0]},{shift[1]}", "local_land": int(lf.sum()), "both_land": int(both.sum()),
            "match": int(match.sum()), "mask_mismatch": int((lf != rf).sum()),
            "match_frac_local_land": float(match.sum() / max(lf.sum(), 1)),
            "footprint_overlap": float(frac[match].mean()) if match.any() else float("nan")}


def verify_gee(raw_dir, project, days=VERIFY_DAYS):
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import gee_fetch
    ee = gee_fetch._init(project)
    rows = []
    for var, folder, day in days:
        d = pd.Timestamp(day)
        f = os.path.join(raw_dir, folder, f"{var}_{d.year}_{d.month:02d}.tif")
        with rasterio.open(f) as ds:
            local = ds.read(d.day).astype("float64")
            if ds.nodata is not None:
                local[local == ds.nodata] = np.nan
            t_cur = ds.transform
            t_fix = native_transform(t_cur, ds.width, ds.height)
            b = ds.bounds
        bbox = (min(b.left, t_fix.c) - 0.2, min(b.bottom, t_fix.f - 0.1 * ds.height) - 0.2,
                max(b.right, t_fix.c + 0.1 * ds.width) + 0.2, max(b.top, t_fix.f) + 0.2)
        ref, t_ref = _gee_native_day(ee, gee_fetch, var, day, bbox)
        # transform sau sua phai trung luoi GEE (lech goc = so nguyen pixel)
        off = ((t_fix.c - t_ref.c) / 0.1, (t_ref.f - t_fix.f) / 0.1)
        aligned = all(abs(o - round(o)) < 1e-6 for o in off) and abs(t_ref.a - 0.1) < 1e-12
        log(f"{var} {day}: luoi GEE {transform_list(t_ref)}; transform sau sua can luoi GEE: {aligned}")
        labels = [("hien_tai", t_cur)] + ([] if t_fix.almost_equals(t_cur, precision=1e-12) else [("sau_sua", t_fix)])
        for label, t in labels:
            shifts = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            for s in shifts:
                r = compare_on_grid(local, t, ref, t_ref, shift=s)
                rows.append({"var": var, "day": day, "transform": label, "grid_aligned": aligned, **r})
                if s == (0, 0):
                    log(f"  [{label}] dich 0,0: khop {r['match']}/{r['local_land']} pixel dat "
                        f"({100 * r['match_frac_local_land']:.1f}%), lech mat na {r['mask_mismatch']}, "
                        f"o pixel nam trong pixel goc khop gia tri {100 * r['footprint_overlap']:.1f}% dien tich")
            other = [x["match_frac_local_land"] for x in rows[-len(shifts):] if x["shift"] != "0,0"]
            log(f"  [{label}] 8 phep dich khac: khop {100 * min(other):.1f}-{100 * max(other):.1f}%")
    return rows


def main(a):
    log_path = a.log or os.path.join(a.raw_dir, "era5_transform_fix.json")
    if a.verify_gee:
        rows = verify_gee(a.raw_dir, a.project)
        if a.verify_out:
            import pandas as pd
            pd.DataFrame(rows).to_csv(a.verify_out, index=False)
        return
    if a.revert:
        revert_files(a.raw_dir, log_path)
        return
    fix_files(a.raw_dir, log_path, dry_run=a.dry_run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default=os.environ.get("RAW_DIR", "A:/Dataset_NCKH/raw"))
    ap.add_argument("--log", help="JSON ghi transform cu/moi + sha256 (mac dinh <raw-dir>/era5_transform_fix.json)")
    ap.add_argument("--dry-run", action="store_true", help="Chi in ke hoach, khong ghi")
    ap.add_argument("--revert", action="store_true", help="Tra transform cu theo JSON")
    ap.add_argument("--verify-gee", action="store_true", help="So vai ngay voi GEE theo luoi goc")
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--verify-out", help="Ghi bang so sanh GEE ra CSV")
    main(ap.parse_args())

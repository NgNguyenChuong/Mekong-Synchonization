"""Cat nhan tu tao (GEE) theo DAU CHAN nhan Zenodo (known-pitfalls 4b; An duyet M, 2026-10-03).

Dau chan = pixel co Salinity HUU HAN o >= 1 nam trong cac file Zenodo dua vao (bo chinh: 2014-2023).
Nhan GEE 2024-2026 phu rong hon Zenodo (~1.210 km² trong ranh gioi v1 + dai bien moi cua v2) -> neu
khong cat, mot so o chi co nhan o 3 nam cuoi. Ham o day dat NaN moi pixel nhan nam ngoai dau chan.

Can luoi THEO TRANSFORM, KHONG noi suy: hai luoi phai cung CRS, cung kich thuoc pixel, khong xoay, va
goc lech nhau mot so NGUYEN pixel (vd GEE v2 goc 369780/1220100 vs Zenodo 438540/1219680 -> lech 2292
cot, 14 hang). Lech khong nguyen -> loi (khong tu lam tron). Pixel nhan nam ngoai khung Zenodo = ngoai
dau chan.

Doc band theo TEN (band 1 = NDWIchen, band 2 = Salinity, ca Zenodo va GEE). File khong co tag nodata
-> dung np.isfinite, khong dua vao masked=True.
"""
import glob
import os

import numpy as np

from scope_mask import band_index

ZENODO_DIR = os.environ.get("ZENODO_DIR", "A:/Dataset_NCKH/zenodo_15653696")
FOOTPRINT_YEARS = tuple(range(2014, 2024))   # bo chinh OLI; TM 2000-2010 khong vao dau chan


def zenodo_label_paths(zenodo_dir=ZENODO_DIR, years=FOOTPRINT_YEARS) -> list[str]:
    """Duong dan file Zenodo cho tung nam; thieu nam nao -> loi (dau chan khong duoc am tham hep lai)."""
    paths, missing = [], []
    for y in years:
        hit = sorted(glob.glob(os.path.join(zenodo_dir, f"{y}_MD_dry_NDWIchen_Salinity.tif")))
        (paths if hit else missing).append(hit[0] if hit else y)
    if missing:
        raise FileNotFoundError(f"Thieu file Zenodo cho nam {missing} trong {zenodo_dir}")
    return paths


def build_footprint(label_paths, band="Salinity", chunk_rows=1024):
    """Dau chan tren luoi cua file dau tien: (bool array, transform, crs).

    True = band `band` huu han o >= 1 file. Moi file phai cung luoi (khac -> loi).
    """
    import rasterio
    from rasterio.windows import Window

    label_paths = list(label_paths)
    if not label_paths:
        raise ValueError("Khong co file nao de dung dau chan.")
    with rasterio.open(label_paths[0]) as ref:
        transform, crs, (h, w) = ref.transform, ref.crs, (ref.height, ref.width)
    fp = np.zeros((h, w), dtype=bool)
    for p in label_paths:
        with rasterio.open(p) as s:
            if (s.transform, s.crs, (s.height, s.width)) != (transform, crs, (h, w)):
                raise ValueError(f"{p} khac luoi voi {label_paths[0]}.")
            b = band_index(s, band)
            for r0 in range(0, h, chunk_rows):
                n = min(chunk_rows, h - r0)
                fp[r0:r0 + n] |= np.isfinite(s.read(b, window=Window(0, r0, w, n)))
    return fp, transform, crs


def pixel_offset(src_transform, dst_transform, tol=1e-6) -> tuple[int, int]:
    """(hang, cot) cua goc `src` tinh bang pixel cua `dst`; luoi phai trung nhau (cung kich thuoc pixel,
    khong xoay, lech nguyen pixel) -> neu khong: ValueError (khong noi suy, khong lam tron)."""
    s, d = src_transform, dst_transform
    if s.b != 0 or s.d != 0 or d.b != 0 or d.d != 0:
        raise ValueError("Luoi co xoay - khong ho tro.")
    if abs(s.a - d.a) > tol * abs(d.a) or abs(s.e - d.e) > tol * abs(d.e):
        raise ValueError(f"Kich thuoc pixel khac nhau ({s.a}, {s.e}) vs ({d.a}, {d.e}) - can noi suy, khong ho tro.")
    col = (s.c - d.c) / d.a
    row = (s.f - d.f) / d.e
    if abs(col - round(col)) > tol or abs(row - round(row)) > tol:
        raise ValueError(f"Goc luoi lech khong nguyen pixel (cot {col:.6f}, hang {row:.6f}) - khong can duoc.")
    return int(round(row)), int(round(col))


def footprint_on_grid(fp, fp_transform, dst_transform, dst_shape) -> np.ndarray:
    """Chieu dau chan `fp` (bool, luoi fp_transform) sang luoi dich (dst_transform, dst_shape) bang dich
    nguyen pixel. Pixel dich nam ngoai khung `fp` -> False."""
    fp = np.asarray(fp, dtype=bool)
    r_off, c_off = pixel_offset(fp_transform, dst_transform)   # goc fp trong toa do pixel dich
    h, w = dst_shape
    out = np.zeros((h, w), dtype=bool)
    r0, r1 = max(0, r_off), min(h, r_off + fp.shape[0])
    c0, c1 = max(0, c_off), min(w, c_off + fp.shape[1])
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = fp[r0 - r_off:r1 - r_off, c0 - c_off:c1 - c_off]
    return out


def clip_to_footprint(arr, arr_transform, fp, fp_transform):
    """Ban sao `arr` (2D hoac (band, h, w), so thuc) voi NaN ngoai dau chan; tra ve (mang, so pixel bi cat).

    `arr_transform` la transform cua chinh mang (cua so doc -> transform cua cua so). "So pixel bi cat"
    = so vi tri 2D ngoai dau chan co gia tri huu han o IT NHAT mot band (truoc khi cat). NaN san co
    giu nguyen NaN, khong dien gia tri nao.
    """
    a = np.array(arr, dtype=np.result_type(np.asarray(arr).dtype, np.float32), copy=True)  # so nguyen -> so thuc
    shape2d = a.shape[-2:]
    inside = footprint_on_grid(fp, fp_transform, arr_transform, shape2d)
    fin = np.isfinite(a) if a.ndim == 2 else np.isfinite(a).any(axis=0)
    n_cut = int((fin & ~inside).sum())
    if a.ndim == 2:
        a[~inside] = np.nan
    else:
        a[:, ~inside] = np.nan
    return a, n_cut


def clip_label_file(src_path, fp, fp_transform, dst_path=None, band="Salinity", regions=None, chunk_rows=1024):
    """Cat file nhan GEE theo dau chan, doc theo khoi hang (khong nap ca raster).

    Tra ve dict: n_cut (pixel `band` huu han ngoai dau chan, toan raster), n_finite_before, n_finite_after;
    voi moi `regions` {ten: bool array cung luoi file} (vd ranh gioi rasterize): n_cut_<ten>,
    n_finite_after_<ten>. dst_path -> ghi GeoTIFF cung luoi, cung band/ten band, MOI band bi cat (NDWIchen cung NaN
    ngoai dau chan de hai band nhat quan); None -> chi dem.
    """
    import rasterio
    from rasterio.windows import Window, transform as win_transform

    stats = {"n_cut": 0, "n_finite_before": 0, "n_finite_after": 0}
    regions = dict(regions or {})
    for k in regions:
        stats.update({f"n_cut_{k}": 0, f"n_finite_after_{k}": 0})
    with rasterio.open(src_path) as s:
        b = band_index(s, band)
        h, w = s.height, s.width
        for k, reg in regions.items():
            if np.shape(reg) != (h, w):
                raise ValueError(f"region {k} {np.shape(reg)} khac kich thuoc file {(h, w)}")
        dst = None
        if dst_path:
            prof = s.profile.copy()
            prof.update(compress="lzw", tiled=True, blockxsize=512, blockysize=512)
            dst = rasterio.open(dst_path, "w", **prof)
            for i, name in enumerate(s.descriptions, start=1):
                if name:
                    dst.set_band_description(i, name)
            dst.update_tags(footprint="NaN ngoai dau chan Zenodo (Salinity huu han >= 1 nam 2014-2023)",
                            source=os.path.basename(src_path))
        try:
            for r0 in range(0, h, chunk_rows):
                n = min(chunk_rows, h - r0)
                win = Window(0, r0, w, n)
                data = s.read(window=win)
                lab = data[b - 1]
                inside = footprint_on_grid(fp, fp_transform, win_transform(win, s.transform), (n, w))
                fin = np.isfinite(lab)
                cut = fin & ~inside
                stats["n_cut"] += int(cut.sum())
                stats["n_finite_before"] += int(fin.sum())
                stats["n_finite_after"] += int((fin & inside).sum())
                for k, region in regions.items():
                    reg = np.asarray(region[r0:r0 + n], dtype=bool)
                    stats[f"n_cut_{k}"] += int((cut & reg).sum())
                    stats[f"n_finite_after_{k}"] += int((fin & inside & reg).sum())
                if dst is not None:
                    data = data.astype(np.float32, copy=False)
                    data[:, ~inside] = np.nan
                    dst.write(data, window=win)
        finally:
            if dst is not None:
                dst.close()
    return stats

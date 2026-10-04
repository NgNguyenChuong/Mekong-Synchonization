"""Nhan theo (o, mua kho) va gia tri tham chieu tai diem (P5; An duyet A5/N/M 2026-10-03; CHG-08, CHG-10, CHG-15).

Quy tac pixel nhan (bo CHINH), mua s:
  1. Quy tac NaN (theo nguyen nhan): NDWIchen NaN | Salinity NaN | S > 28,013 | B7 suy nguoc <= 0
       B5 = -ln(S / 28,013) / 13,39 ;  B7 = B5 * (1 - NDWI) / (1 + NDWI)
     (B7 khong huu han, vd NDWI = -1 -> chia 0, cung coi la khong hop le: khong suy nguoc duoc.)
  2. Mask nuoc dong buffer 0 (CHG-05): water_freq >= 50 (255 = khong quan sat -> KHONG phai nuoc) ∪ WorldCover 80.
  3. Do tin cay mask (CHG-08): mask nuoc cua mua do co n_clear < MIN_CLEAR (= 2) HOAC water_freq = 255 -> NaN,
     ap MOI bo (ke ca keepwater). 1 lan chup chi cho water_freq 0% / 100% -> khong xac nhan duoc la dat.
  4. Pham vi v3 (CHG-10): v2 (tam pixel) ∩ WorldCover THUOC {10,20,30,40} (tam pixel 30 m) ∩ dau chan Zenodo.
Bo PHU (CHG-15) - moi bo khac bo chinh DUNG MOT dieu kien:
  keepwater    = bo mask nuoc dong (buoc 2) va cho them WC 80;
  keepmangrove = cho them WC 95 (van mask nuoc dong);
  keep6090     = cho them WC 60 va 90 (van mask nuoc dong). Khong co bo do thi 50.
"Truoc" (bao cao) = chi quy tac NaN, trong v2 ∩ dau chan (khong loc WorldCover, khong CHG-08).

Don vi: EC1:5 (dS/m) - do man DAT, khong phai ECe hay do man nuoc.
Luoi lam viec = luoi 30 m Zenodo (scope_mask_v3.tif); nhan/mask GEE can theo transform (lech nguyen pixel,
labels.pixel_offset) - khong noi suy.
"""
import numpy as np
import pandas as pd

from labels import pixel_offset
from scope_mask import SCOPE_EXCLUDE_LC, SCOPE_INCLUDE_LC, band_index

S_MAX = 28.013          # S = 28,013 * exp(-13,39 * NIR) -> S > 28,013 <=> NIR < 0
K_NIR = 13.39
WATER_FREQ_MIN = 50     # % so canh MNDWI > 0
WATER_FREQ_NOOBS = 255
WC_OUT, WC_WATER, WC_MANGROVE = 0, 80, 95
WC_BUILT, WC_BARE, WC_WETLAND = 50, 60, 90
MAIN_LC = tuple(SCOPE_INCLUDE_LC)          # CHG-10: bo chinh chi WC 10, 20, 30, 40
MIN_CLEAR = 2           # CHG-08: can >= 2 lan chup sach o mask nuoc cua mua (N chot TRUOC khi do nhom)
MIN_TRAIN_PX = 100      # quy tac N: o vao tap huan luyen khi >= 100 pixel dat hop le
MIN_TRAIN_FRAC = 0.10   # do nhay: >= 10% dien tich o
REF_MIN_VALID = 5       # A5: median cua so 3x3 can >= 5/9 pixel hop le
VARIANTS = ("main", "keepwater", "keepmangrove", "keep6090")
# lop WorldCover moi bo phu duoc THEM so voi bo chinh (CHG-15); keepwater con bo mask nuoc dong
VARIANT_EXTRA_LC = {"main": (), "keepwater": (WC_WATER,), "keepmangrove": (WC_MANGROVE,),
                    "keep6090": (WC_BARE, WC_WETLAND)}
RULES_VERSION = "v3_chg08_chg15_2026-10-04"   # doi quy tac -> doi chuoi nay (raster/bang cu khong duoc dung lai)


def nan_rule(ndwi, sal):
    """Salinity (float32) sau quy tac NaN; khong dien gia tri nao, chi dat NaN."""
    ndwi = np.asarray(ndwi, dtype=np.float64)
    s = np.asarray(sal, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        b5 = -np.log(s / S_MAX) / K_NIR
        b7 = b5 * (1.0 - ndwi) / (1.0 + ndwi)
    bad = ~np.isfinite(ndwi) | ~np.isfinite(s) | (s > S_MAX) | ~np.isfinite(b7) | (b7 <= 0)
    out = np.asarray(sal, dtype=np.float32).copy()
    out[bad] = np.nan
    return out


def water_mask(water_freq, wc):
    """Mat nuoc bo chinh: water_freq >= 50 (255 khong tinh) HOAC WorldCover 80."""
    wf = np.asarray(water_freq)
    return ((wf >= WATER_FREQ_MIN) & (wf != WATER_FREQ_NOOBS)) | (np.asarray(wc) == WC_WATER)


def clear_ok(water_freq, n_clear, min_clear=MIN_CLEAR):
    """CHG-08: mask nuoc du tin cay = n_clear >= min_clear VA water_freq != 255 (255 = 0 lan chup sach)."""
    return (np.asarray(n_clear) >= min_clear) & (np.asarray(water_freq) != WATER_FREQ_NOOBS)


def variant_masks(finite, in_v2, footprint, wc, water, clear):
    """Mat na pixel hop le cho 'before' va 4 bo nhan (CHG-10, CHG-15, CHG-08).

    finite: nhan huu han sau quy tac NaN; water: water_mask(...); clear: clear_ok(...). Moi doi so cung kich thuoc.
    'before' = finite ∩ v2 ∩ dau chan (khong WorldCover, khong CHG-08) - chi de bao cao.
    """
    wc = np.asarray(wc)
    base = finite & in_v2 & footprint
    ok = base & clear
    out = {"before": base}
    for v in VARIANTS:
        lc = np.isin(wc, MAIN_LC + VARIANT_EXTRA_LC[v])
        out[v] = ok & lc if v == "keepwater" else ok & lc & ~water
    return out


def comparison_masks(finite, in_v2, footprint, wc, water, water_freq, n_clear, min_clear=MIN_CLEAR):
    """Cac tap pixel de TACH tac dong so voi P5 cu (bao cao, khong ghi raster).

    v2_old     = bo chinh P5 cu: WC khong {0,80,95} ∩ khong nuoc, KHONG CHG-08;
    v2_clear   = v2_old ∩ CHG-08 (thu tu nguoc: CHG-08 truoc);
    v3_noclear = WC {10,20,30,40} ∩ khong nuoc, KHONG CHG-08 (chi tac dong (a));
    drop_lc    = v2_old TRU v3_noclear (tac dong (a)); drop_lc50 / drop_lc6090 = phan cua WC 50 / {60, 90};
    drop_clear = v3_noclear TRU main (tac dong (b)); _n0 = water_freq 255 hoac n_clear 0;
                 _n1 = phan con lai (1 <= n_clear < min_clear).
    """
    wc = np.asarray(wc)
    wf, ncl = np.asarray(water_freq), np.asarray(n_clear)
    base = finite & in_v2 & footprint
    clear = clear_ok(wf, ncl, min_clear)
    v2_old = base & ~np.isin(wc, SCOPE_EXCLUDE_LC) & ~water
    v3_noclear = base & np.isin(wc, MAIN_LC) & ~water
    drop_lc = v2_old & ~v3_noclear
    drop_clear = v3_noclear & ~clear
    n0 = (wf == WATER_FREQ_NOOBS) | (ncl == 0)
    return {"v2_old": v2_old, "v2_clear": v2_old & clear, "v3_noclear": v3_noclear,
            "drop_lc": drop_lc, "drop_lc50": drop_lc & (wc == WC_BUILT),
            "drop_lc6090": drop_lc & np.isin(wc, (WC_BARE, WC_WETLAND)),
            "drop_clear": drop_clear, "drop_clear_n0": drop_clear & n0, "drop_clear_n1": drop_clear & ~n0}


def variant_point_sets(main_pts, source_pts, variants=VARIANTS):
    """Tap diem danh gia theo bo (CHG-15): bo phu v = diem bo chinh + diem NGUON (ban truoc khi loc v3) khong thuoc
    bo chinh, co wc_class (pixel 30 m chua diem) thuoc VARIANT_EXTRA_LC[v]. Thuoc tinh diem giu nguyen.

    main_pts, source_pts: DataFrame co point_id (+ wc_class o source_pts va main_pts; cot khac giu nguyen).
    Tra ve DataFrame: diem chinh (thu tu cu) roi diem them (thu tu nguon), cot in_<v> (bool) cho moi bo.
    point_id trung trong mot bang -> ValueError.
    """
    for name, df in (("main", main_pts), ("source", source_pts)):
        if df["point_id"].astype(str).duplicated().any():
            raise ValueError(f"point_id trung trong tap {name}.")
    main_ids = set(main_pts["point_id"].astype(str))
    extra = source_pts[~source_pts["point_id"].astype(str).isin(main_ids)]
    extra_lc = sorted({c for v in variants for c in VARIANT_EXTRA_LC[v]})
    extra = extra[extra["wc_class"].isin(extra_lc)]
    out = pd.concat([main_pts.assign(_main=True), extra.assign(_main=False)], ignore_index=True)
    for v in variants:
        out[f"in_{v}"] = out["_main"] | (~out["_main"] & out["wc_class"].isin(VARIANT_EXTRA_LC[v]))
    return out.drop(columns="_main")


def read_aligned(src, band, dst_transform, dst_shape, row0=0, fill=np.nan, dtype=None):
    """Doc khoi hang [row0, row0 + h) cua luoi dich tu `src` (cung kich thuoc pixel, lech NGUYEN pixel).

    Pixel dich nam ngoai khung nguon = fill. Khong noi suy; lech khong nguyen -> ValueError.
    """
    from rasterio.windows import Window

    h, w = dst_shape
    r_off, c_off = pixel_offset(dst_transform, src.transform)   # goc luoi dich trong pixel nguon
    b = band_index(src, band)
    dtype = dtype or src.dtypes[b - 1]
    out = np.full((h, w), fill, dtype=dtype)
    sr0, sc0 = r_off + row0, c_off
    r0, r1 = max(0, sr0), min(src.height, sr0 + h)
    c0, c1 = max(0, sc0), min(src.width, sc0 + w)
    if r1 > r0 and c1 > c0:
        out[r0 - sr0:r1 - sr0, c0 - sc0:c1 - sc0] = src.read(b, window=Window(c0, r0, c1 - c0, r1 - r0))
    return out


def median_3x3(arr, rows, cols, min_valid=REF_MIN_VALID):
    """Median cac pixel huu han trong cua so 3x3 quanh (rows, cols); < min_valid hop le -> NaN.

    Pixel cua so ngoai khung = khong hop le. Tra ve (median float64, n_valid int).
    """
    arr = np.asarray(arr)
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    h, w = arr.shape
    vals = np.full((len(rows), 9), np.nan)
    k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = rows + dr, cols + dc
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            vals[ok, k] = arr[r[ok], c[ok]]
            k += 1
    n = np.isfinite(vals).sum(axis=1)
    med = np.full(len(rows), np.nan)
    good = n >= min_valid
    if good.any():
        med[good] = np.nanmedian(vals[good], axis=1)
    return med, n


def points_to_pixels(xs, ys, transform):
    """(rows, cols) cua pixel CHUA moi diem (toa do cung CRS voi transform)."""
    inv = ~transform
    cols = np.floor(inv.a * np.asarray(xs) + inv.b * np.asarray(ys) + inv.c).astype(np.int64)
    rows = np.floor(inv.d * np.asarray(xs) + inv.e * np.asarray(ys) + inv.f).astype(np.int64)
    return rows, cols


def area_weighted_mean_count(tif_path, cell_data):
    """Nhu processing.area_weighted_means nhung them 'count' (tong ty le phu cua pixel huu han) moi band.

    Tra ve DataFrame: cell_id, cell_area_m2 (dien tich o trong CRS raster), b<i>_mean, b<i>_count.
    O khong co pixel huu han -> mean NaN, count 0 (khong dien 0 vao mean).
    """
    import rasterio
    from exactextract import exact_extract

    from processing import _cells_gdf

    with rasterio.open(tif_path) as ds:
        crs, n_bands = ds.crs, ds.count
    gdf = _cells_gdf(cell_data, crs)
    res = exact_extract(tif_path, gdf, ["mean", "count"], include_cols=["cell_id"], output="pandas")
    res = res.set_index("cell_id").reindex(gdf["cell_id"])
    out = pd.DataFrame({"cell_id": gdf["cell_id"].to_numpy(), "cell_area_m2": gdf.geometry.area.to_numpy()})
    for i in range(1, n_bands + 1):
        m, c = ("mean", "count") if n_bands == 1 else (f"band_{i}_mean", f"band_{i}_count")
        out[f"b{i}_mean"] = res[m].to_numpy(dtype=float)
        out[f"b{i}_count"] = res[c].fillna(0).to_numpy(dtype=float)
    return out


def cell_label_rows(mean, count, cell_area_m2, px_area_m2, min_px=MIN_TRAIN_PX, min_frac=MIN_TRAIN_FRAC):
    """Cot nhan cua o: salinity, n_valid_px (pixel tuong duong, co trong so phu), valid_frac, train_ok, train_ok_10pct."""
    count = np.asarray(count, dtype=float)
    mean = np.where(count > 0, np.asarray(mean, dtype=float), np.nan)
    frac = count * px_area_m2 / np.asarray(cell_area_m2, dtype=float)
    return pd.DataFrame({"salinity": mean, "n_valid_px": count, "valid_frac": frac,
                         "train_ok": count >= min_px, "train_ok_10pct": frac >= min_frac})


def value_summary(vals, thresholds=(2.0, 4.0, 21.0)):
    """So pixel, so/ty le > nguong, va min/p1/p50/p99/max cua mang gia tri hop le (1D)."""
    v = np.asarray(vals, dtype=np.float64)
    n = int(v.size)
    out = {"n_px": n}
    for t in thresholds:
        k = int((v > t).sum())
        tag = f"{t:g}"
        out[f"n_gt{tag}"] = k
        out[f"pct_gt{tag}"] = 100.0 * k / n if n else np.nan
    q = np.percentile(v, [0, 1, 50, 99, 100]) if n else [np.nan] * 5
    out.update({"min": q[0], "p1": q[1], "p50": q[2], "p99": q[3], "max": q[4]})
    return out

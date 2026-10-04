"""Dac trung TINH theo o tren PIXEL DAT HOP LE (An chot 2026-10-03, bo sung (1) cau N, cau G = co).

Quy tac (giong nhau o moi luoi, ghi truoc khi co ket qua mo hinh):
  - dem_mean, dist_main_river_km, dist_any_water_km, dist_coast_km: trung binh co trong so dien tich CHI
    tren pixel scope == 1 (mat na pham vi tinh; mac dinh scope_mask_v3.tif: v2 ∩ WorldCover {10,20,30,40} ∩ dau
    chan Zenodo - CHG-10; truoc do scope_mask_v2.tif: WorldCover khong {0,80,95}). O khong co pixel scope -> NaN
    (khong dien 0).
  - scope_frac: dien tich pixel scope trong o / dien tich ca o (EPSG:32648). Cot CHAT LUONG, khong phai
    dac trung (khong nam trong DEFAULT_ALLOWED_FEATURES).
  - landcover_class_<Lop>: ty le dien tich tung lop WorldCover 2021 (10 m) tren TOAN O (gom nuoc 80, ngap
    man 95 - muc 8 da chot). Mau so = phan o nam trong khung WorldCover (ma 0 "ngoai vung" van o mau so
    nhung khong sinh cot).

Lay mau ve luoi scope 30 m (luoi pixel Zenodo, EPSG:32648):
  - DEM GLO30 30 m: luoi DEM trung luoi scope (goc lech so nguyen pixel; khac -> loi) -> lay dung pixel.
    Quy tac DEM thieu (CHG-09, An duyet 2026-10-03): DEM == 0 VA co Copernicus WBM > 0 (file co
    DEM_DBSCL_flags_v2.tif, band WBM; luoi 30 m lech nguyen pixel so voi DEM - cung lay dung pixel, khong noi
    suy). Ly do: 100% pixel DEM == 0 co WBM > 0 (gia tri 0 do lam phang mat nuoc), con 62% dat thap 0,05-1 m
    cung co co ho -> KHONG dung rieng WBM, KHONG dung gia tri 0 don thuan (dat that cao 0-1 m). DEM NoData goc
    -> NaN (khong dien 0). Han che: GLO30 la DSM (gom cay, nha).
  - Raster khoang cach 90 m (goc khong can boi so 30 m): moi pixel scope 30 m nhan gia tri pixel 90 m
    CHUA TAM no (nearest). Trung binh o tren cac pixel 30 m nay = trung binh pixel 90 m co trong so bang
    phan scope (dem theo tam pixel 30 m) nam trong pixel 90 m do. Pixel bien (raster khoang cach co gia tri
    ca tren bien, known-pitfalls 4m) bi loai.
Trung binh co trong so dien tich phu dung exactextract (processing.area_weighted_means), khong lay tam o.
"""
import numpy as np
import pandas as pd

from scope_mask import band_index, nearest_on_grid

SCOPE_FEATURES = ("dem_mean", "dist_main_river_km", "dist_any_water_km", "dist_coast_km")
RIVER_BANDS = ("dist_main_river_km", "dist_any_water_km", "dist_coast_km")
SCOPE_FRAC_COL = "scope_frac"
LANDCOVER_PREFIX = "landcover_class_"

# Ten lop WorldCover - TRUNG src/config.py DEFAULT_STATIC_SPECS["landcover"]["class_names"] (test kiem).
# Bo cot co dinh 9 lop giong nhau moi luoi; lop vang mat trong o = 0. Ma 70/100 (tuyet, reu) khong co o
# DBSCL -> neu xuat hien la loi du lieu.
WORLDCOVER_NAMES = {
    10: "Trees", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built_up", 60: "Bareland",
    80: "Water", 90: "Wetland", 95: "Mangroves",
}


def _offset_pixels(a, b, res):
    """So pixel lech giua hai goc (phai la so nguyen neu cung luoi)."""
    return (a - b) / res


def check_aligned(src_transform, dst_transform, tol=1e-6):
    """Hai luoi cung kich thuoc pixel va goc lech mot so NGUYEN pixel; khac -> ValueError."""
    if not (np.isclose(src_transform.a, dst_transform.a) and np.isclose(src_transform.e, dst_transform.e)):
        raise ValueError(f"Kich thuoc pixel khac nhau: {src_transform.a} vs {dst_transform.a}.")
    for off in (_offset_pixels(src_transform.c, dst_transform.c, dst_transform.a),
                _offset_pixels(src_transform.f, dst_transform.f, dst_transform.e)):
        if abs(off - round(off)) > tol:
            raise ValueError(f"Luoi lech {off:.4f} pixel (khong nguyen) - can can luoi truoc khi lay dung pixel.")


DEM_RULE = ("CHG-09: DEM thieu (NaN) khi DEM == 0 VA co WBM > 0 (Copernicus DEM water body mask); "
            "DEM NoData goc -> NaN; dat thap > 0 giu nguyen ke ca khi WBM > 0; WBM = 0 voi DEM == 0 giu nguyen")
WBM_VALID = (0, 1, 2, 3)  # 0 khong nuoc, 1 bien, 2 ho, 3 song (Copernicus GLO30)
LOW_LAND_M = 1.0  # chi de thong ke "dat thap co co nuoc duoc giu" (0 < DEM < 1 m)


def dem_missing_mask(dem, wbm):
    """CHG-09: True o pixel DEM == 0 (dung bang) VA WBM > 0. Khong dung gia tri 0 don thuan, khong dung rieng WBM."""
    dem = np.asarray(dem)
    wbm = np.asarray(wbm)
    return (dem == 0) & (wbm > 0)


def build_scope_stack(scope_path, dem_path, river_path, out_path, dem_flags_path, chunk_rows=1024, dem_band="DEM",
                      wbm_band="WBM", log=None) -> dict:
    """Ghi GeoTIFF float32 4 band (SCOPE_FEATURES) tren luoi scope 30 m; NaN ngoai scope.

    dem_flags_path: raster co Copernicus DEM (band wbm_band) - BAT BUOC truyen. Luoi phai cung kich thuoc pixel
    va lech so nguyen pixel so voi scope (khong nguyen -> ValueError). Pixel scope ngoai khung co hoac co ma WBM
    ngoai {0,1,2,3} -> ValueError (khong doan). Truyen None CO Y chi de dung ban DOI CHUNG khong ap quy tac
    (do tac dong CHG-09), khong dung cho bo chinh.
    Tra ve thong ke tren pixel scope: so pixel, DEM NaN goc, DEM == 0, DEM == 0 & WBM > 0 (-> NaN), DEM == 0 &
    WBM == 0 (giu), 0 < DEM < 1 m & WBM > 0 (giu), DEM < -5, khoang cach NaN, so pixel scope theo ma WBM.
    """
    import rasterio
    from rasterio.windows import Window

    use_rule = dem_flags_path is not None
    stats = {"scope_px": 0, "dem_nan_px": 0, "dem_zero_px": 0, "dem_zero_wbm_to_nan_px": 0,
             "dem_zero_wbm0_kept_px": 0, "dem_low_wbm_kept_px": 0, "dem_lt_m5_px": 0,
             "dem_rule": DEM_RULE if use_rule else "KHONG ap (ban doi chung)",
             "wbm_px": {str(v): 0 for v in WBM_VALID},
             **{f"{b}_nan_px": 0 for b in RIVER_BANDS}}
    with rasterio.open(scope_path) as sc, rasterio.open(dem_path) as dem, rasterio.open(river_path) as rv:
        fl = rasterio.open(dem_flags_path) if use_rule else None
        try:
            others = [(dem, dem_path), (rv, river_path)] + ([(fl, dem_flags_path)] if fl else [])
            for other, p in others:
                if other.crs != sc.crs:
                    raise ValueError(f"{p} CRS {other.crs} khac {scope_path} {sc.crs}.")
            check_aligned(dem.transform, sc.transform)
            if fl:
                check_aligned(fl.transform, sc.transform)
                band_index(fl, wbm_band)
            for b in RIVER_BANDS:
                band_index(rv, b)  # bao loi som neu thieu band theo ten
            sb = band_index(sc, "scope")
            h, w = sc.height, sc.width
            profile = dict(driver="GTiff", width=w, height=h, count=len(SCOPE_FEATURES), dtype="float32",
                           crs=sc.crs, transform=sc.transform, nodata=np.nan, compress="deflate", predictor=3,
                           tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
            with rasterio.open(out_path, "w", **profile) as out:
                for i, name in enumerate(SCOPE_FEATURES, start=1):
                    out.set_band_description(i, name)
                for r0 in range(0, h, chunk_rows):
                    n = min(chunk_rows, h - r0)
                    win = Window(0, r0, w, n)
                    m = sc.read(sb, window=win) == 1
                    dem_arr = nearest_on_grid(dem, dem_band, sc.transform, (n, w), r0, fill=np.nan).astype("float32")
                    bands = [dem_arr]
                    bands += [nearest_on_grid(rv, b, sc.transform, (n, w), r0, fill=np.nan).astype("float32")
                              for b in RIVER_BANDS]
                    d = dem_arr[m]
                    stats["scope_px"] += int(m.sum())
                    stats["dem_nan_px"] += int(np.isnan(d).sum())
                    stats["dem_zero_px"] += int((d == 0).sum())
                    stats["dem_lt_m5_px"] += int((d < -5).sum())
                    if fl:
                        # 255 = ngoai khung co (WBM thuc chi 0-3) -> loi o buoc kiem ma ben duoi.
                        wbm = nearest_on_grid(fl, wbm_band, sc.transform, (n, w), r0, fill=255).astype("int16")
                        wm = wbm[m]
                        bad = ~np.isin(wm, WBM_VALID)
                        if bad.any():
                            raise ValueError(f"{int(bad.sum())} pixel scope (hang {r0}-{r0 + n}) co WBM ngoai "
                                             f"{WBM_VALID} hoac ngoai khung {dem_flags_path} - khong doan.")
                        for v in WBM_VALID:
                            stats["wbm_px"][str(v)] += int((wm == v).sum())
                        miss = dem_missing_mask(d, wm)
                        stats["dem_zero_wbm_to_nan_px"] += int(miss.sum())
                        stats["dem_zero_wbm0_kept_px"] += int(((d == 0) & (wm == 0)).sum())
                        stats["dem_low_wbm_kept_px"] += int(((d > 0) & (d < LOW_LAND_M) & (wm > 0)).sum())
                        dem_arr[dem_missing_mask(dem_arr, wbm)] = np.nan
                    for i, arr in enumerate(bands, start=1):
                        arr[~m] = np.nan
                        out.write(arr, i, window=win)
                    for b, arr in zip(RIVER_BANDS, bands[1:]):
                        stats[f"{b}_nan_px"] += int(np.isnan(arr[m]).sum())
                    if log and (r0 // chunk_rows) % 3 == 0:
                        log(f"  hang {r0 + n}/{h}")
        finally:
            if fl:
                fl.close()
    return stats


def scope_fraction(scope_path, cell_data):
    """(cell_id, scope_frac): dien tich pixel scope (theo do phu exactextract) / dien tich o trong CRS scope."""
    import rasterio
    from exactextract import exact_extract

    from processing import _cells_gdf

    with rasterio.open(scope_path) as ds:
        crs, n_bands = ds.crs, ds.count
        sb = band_index(ds, "scope") if "scope" in ds.descriptions else 1
        px_area = abs(ds.transform.a * ds.transform.e)
    gdf = _cells_gdf(cell_data, crs)
    res = exact_extract(scope_path, gdf, ["sum"], include_cols=["cell_id"], output="pandas")
    col = "sum" if n_bands == 1 else f"band_{sb}_sum"
    frac = res[col].to_numpy(dtype=float) * px_area / gdf.geometry.area.to_numpy()
    return pd.DataFrame({"cell_id": res["cell_id"].to_numpy(), SCOPE_FRAC_COL: frac})


def landcover_fractions(wc_path, cell_data, names=None):
    """landcover_class_<Ten> tren TOAN O; bo ma 0; ma ngoai bang ten -> loi. O ngoai khung WC -> NaN."""
    from processing import area_weighted_class_fractions

    names = dict(names or WORLDCOVER_NAMES)
    ids, fractions = area_weighted_class_fractions(wc_path, cell_data)
    unknown = sorted({c for f in fractions for c in f} - set(names) - {0})
    if unknown:
        raise ValueError(f"Ma WorldCover khong co ten: {unknown}")
    codes = sorted(c for c in names if c != 0)
    rows = [{c: (f.get(c, 0.0) if f else np.nan) for c in codes} for f in fractions]
    df = pd.DataFrame(rows, columns=codes)
    df.columns = [f"{LANDCOVER_PREFIX}{names[c]}" for c in codes]
    df.insert(0, "cell_id", ids)
    return df


def cell_static_features(stack_path, scope_path, wc_path, cell_data, names=None):
    """Bang dac trung tinh cua mot luoi: cell_id, SCOPE_FEATURES, scope_frac, landcover_class_*."""
    from processing import area_weighted_means

    means = area_weighted_means(stack_path, cell_data)
    means.columns = ["cell_id", *SCOPE_FEATURES]
    out = means.merge(scope_fraction(scope_path, cell_data), on="cell_id", how="left", validate="1:1")
    out = out.merge(landcover_fractions(wc_path, cell_data, names), on="cell_id", how="left", validate="1:1")
    if out["cell_id"].duplicated().any():
        raise ValueError("cell_id trung trong bang dac trung tinh.")
    # O khong co pixel scope: exactextract da tra NaN; ep lai cho chac (khong de gia tri tu pixel do phu ~0).
    out.loc[~(out[SCOPE_FRAC_COL] > 0), list(SCOPE_FEATURES)] = np.nan
    return out

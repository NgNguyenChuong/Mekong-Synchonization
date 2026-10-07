"""Ham thuan cho nhan khi tuong GEE Dot 7: mua CHIRPS v3 PENTAD va buc xa MCD18A1.062, cua so mua kho theo
seasons.season_window (CHIRPS giu 36 pentad -> gom ca 30/04). Khong goi GEE (test khong can mang).
"""
import datetime as dt

import numpy as np

from seasons import season_window

SEASON_MONTHS = (11, 12, 1, 2, 3, 4)

# Pentad CHIRPS v3 bat dau ngay 1, 6, 11, 16, 21, 26 (pentad 6 den het thang) -> 36 pentad = 6 thang.
PENTAD_START_DAYS = (1, 6, 11, 16, 21, 26)

# Anh nap truoc 01/(M+1) + so ngay nay chua the la ban final cua thang M (suy luan theo lich phat hanh CHC).
FINAL_MIN_LAG_DAYS = 9

# TB 8 band DSR 3 gio (W/m2, dem = 0) = TB ngay; band `DSR` cua catalog la tuc thoi, khong dung.
MCD18_GMT_BANDS = tuple(f"GMT_{h:02d}00_DSR" for h in range(0, 24, 3))
# Sinusoidal MODIS la HINH CAU; GeoTIFF GEE (SR-ORG:6974) ghi ellipsoid WGS84 -> phai ghi de CRS (pitfall 45).
MODIS_SINU_PROJ4 = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"

# DSR_Quality bit 0-1: 2 = phan xa be mat lay tu khi hau (climatology).
DSR_QUALITY_CLIM = 2
DSR_QUALITY_MASK = 0b11


def met_season_window(year: int) -> tuple[dt.date, dt.date]:
    """(ngay dau, ngay cuoi) GOM ca hai dau cua mua kho nam `year` - lay tu seasons.season_window."""
    start, end = season_window(year)
    return start.date(), end.date()


def met_season_filter_dates(year: int) -> tuple[str, str]:
    """Cap ngay cho GEE filterDate (ngay cuoi BI LOAI) -> ('Y-1-11-01', 'Y-04-30')."""
    start, end = met_season_window(year)
    return start.isoformat(), (end + dt.timedelta(days=1)).isoformat()


def met_season_dates(year: int, exclude_months=()) -> list[dt.date]:
    """Moi ngay trong mua (gom ca hai dau), bo cac thang trong `exclude_months` (vd (12,) cho ban khong thang 12)."""
    start, end = met_season_window(year)
    n = (end - start).days + 1
    days = [start + dt.timedelta(days=k) for k in range(n)]
    excl = set(int(m) for m in exclude_months)
    return [d for d in days if d.month not in excl]


def met_season_days(year: int) -> int:
    return len(met_season_dates(year))


def season_year_months(year: int) -> list[tuple[int, int]]:
    """6 thang (nam, thang) cua mua: 11/(s-1), 12/(s-1), 1..4/s."""
    return [(year - 1 if m >= 11 else year, m) for m in SEASON_MONTHS]


def chirps_pentad_index(y: int, m: int, start_day: int) -> str:
    return f"{y:04d}{m:02d}{start_day:02d}"


def chirps_pentad_indices(year: int) -> list[str]:
    """36 `system:index` cua UCSB-CHC/CHIRPS/V3/PENTAD trong mua `year` (dang YYYYMMDD, ngay dau pentad)."""
    return [chirps_pentad_index(y, m, d) for (y, m) in season_year_months(year) for d in PENTAD_START_DAYS]


def mcd18_index(d: dt.date) -> str:
    """`system:index` cua MODIS/062/MCD18A1 (dang YYYY_MM_DD)."""
    return f"{d.year:04d}_{d.month:02d}_{d.day:02d}"


def mcd18_indices(year: int, exclude_months=()) -> list[str]:
    return [mcd18_index(d) for d in met_season_dates(year, exclude_months)]


def missing_indices(expected, available) -> list[str]:
    """Phan tu `expected` khong co trong `available` (giu thu tu expected)."""
    have = set(available)
    return [e for e in expected if e not in have]


def date_runs(dates) -> list[str]:
    """Gop danh sach ngay thanh doan lien tiep 'YYYY-MM-DD..YYYY-MM-DD' (de ghi provenance gon)."""
    ds = sorted(dates)
    runs, i = [], 0
    while i < len(ds):
        j = i
        while j + 1 < len(ds) and (ds[j + 1] - ds[j]).days == 1:
            j += 1
        runs.append(ds[i].isoformat() if i == j else f"{ds[i].isoformat()}..{ds[j].isoformat()}")
        i = j + 1
    return runs


def parse_mcd18_index(s: str) -> dt.date:
    y, m, d = (int(v) for v in s.split("_"))
    return dt.date(y, m, d)


def chirps_status_from_version(pentad_index: str, version_us: int) -> str:
    """'final' neu anh nap (system:version, micro giay UTC) tu 01/(M+1) + FINAL_MIN_LAG_DAYS, nguoc lai 'prelim'."""
    y, m = int(pentad_index[:4]), int(pentad_index[4:6])
    first_next = dt.date(y + (m == 12), m % 12 + 1, 1)
    earliest_final = dt.datetime.combine(first_next, dt.time()) + dt.timedelta(days=FINAL_MIN_LAG_DAYS)
    ingest = dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=int(version_us))
    return "final" if ingest >= earliest_final else "prelim"


def output_name(product: str, year: int) -> str:
    """Ten file mua theo quy uoc ERA5 season (`<bien>_<mua>.tif`)."""
    names = {"chirps3": "chirps3_rain", "mcd18a1": "mcd18a1_dsr"}
    return f"{names[product]}_{int(year)}.tif"


CHIRPS_BANDS = ("rain_sum", "rain_m11", "rain_m12", "rain_m01", "rain_m02", "rain_m03", "rain_m04",
                "rain_jfm", "n_pentad_valid", "n_pentad_neg")
MCD18_BANDS = ("dsr_mean", "n_days_valid", "frac_quality2", "dsr_mean_no_dec")


def daily_dsr(gmt_values: np.ndarray) -> np.ndarray:
    """TB ngay tu 8 gia tri 3 gio (truc 0 = 8 band). Thieu BAT KY band nao (NaN) -> NaN (khong dien 0)."""
    v = np.asarray(gmt_values, dtype="float64")
    if v.shape[0] != len(MCD18_GMT_BANDS):
        raise ValueError(f"can {len(MCD18_GMT_BANDS)} band 3 gio, nhan {v.shape[0]}")
    return v.mean(axis=0)  # NaN lan truyen: ngay thieu 1 moc gio la ngay khong hop le


def season_dsr_stats(daily: np.ndarray, quality: np.ndarray, dates) -> dict:
    """Ban numpy cua phep gop mua MCD18 tren GEE (kiem cheo): dsr_mean, n_days_valid, frac_quality2,
    dsr_mean_no_dec tren ngay hop le (truc 0 = ngay); pixel khong co ngay hop le -> NaN."""
    daily = np.asarray(daily, dtype="float64")
    q = np.asarray(quality)
    if daily.shape != q.shape or daily.shape[0] != len(dates):
        raise ValueError("daily, quality, dates khong cung kich thuoc")
    valid = np.isfinite(daily)
    n = valid.sum(axis=0)
    s = np.where(valid, daily, 0.0).sum(axis=0)
    q2 = (valid & ((q.astype("int64") & DSR_QUALITY_MASK) == DSR_QUALITY_CLIM)).sum(axis=0)
    not_dec = np.array([d.month != 12 for d in dates]).reshape((-1,) + (1,) * (daily.ndim - 1))
    v_nd = valid & not_dec
    n_nd = v_nd.sum(axis=0)
    s_nd = np.where(v_nd, daily, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "dsr_mean": np.where(n > 0, s / np.maximum(n, 1), np.nan),
            "n_days_valid": n,
            "frac_quality2": np.where(n > 0, q2 / np.maximum(n, 1), np.nan),
            "dsr_mean_no_dec": np.where(n_nd > 0, s_nd / np.maximum(n_nd, 1), np.nan),
        }


def season_rain_sum(pentads: np.ndarray) -> np.ndarray:
    """Tong mua tu 36 pentad (truc 0); co pentad NoData (NaN hoac < 0) -> NaN, khong coi la 0 mm."""
    p = np.asarray(pentads, dtype="float64")
    if p.shape[0] != len(SEASON_MONTHS) * len(PENTAD_START_DAYS):
        raise ValueError(f"can 36 pentad, nhan {p.shape[0]}")
    bad = ~np.isfinite(p) | (p < 0)
    return np.where(bad.any(axis=0), np.nan, np.where(bad, 0.0, p).sum(axis=0))


def region_masks(boundary_path, profile):
    """(tam pixel trong ranh gioi, pixel cham ranh gioi) tren luoi `profile` (crs, transform, height, width)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    b = gpd.read_file(boundary_path).to_crs(profile["crs"])
    shapes = [(g, 1) for g in b.geometry if g is not None and not g.is_empty]
    out = []
    for touched in (False, True):
        out.append(rasterize(shapes, out_shape=(profile["height"], profile["width"]), transform=profile["transform"],
                             fill=0, all_touched=touched, dtype="uint8").astype(bool))
    return out[0], out[1]


def read_bands(path) -> tuple[dict, dict]:
    """Doc raster nhieu band theo TEN band (description) -> ({ten: mang float64}, profile). Khong theo so thu tu."""
    import rasterio
    with rasterio.open(path) as ds:
        names = list(ds.descriptions)
        if any(n is None for n in names) or len(set(names)) != len(names):
            raise ValueError(f"{path}: band thieu ten/trung ten {names}")
        arrs = {n: ds.read(i + 1).astype("float64") for i, n in enumerate(names)}
        prof = {"crs": ds.crs, "transform": ds.transform, "height": ds.height, "width": ds.width}
    return arrs, prof


def _desc(v: np.ndarray, prefix: str) -> dict:
    """TB/SD/min/max tren gia tri HUU HAN (NaN khong tinh, khong dien 0)."""
    f = v[np.isfinite(v)]
    if f.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_sd": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan}
    return {f"{prefix}_mean": float(f.mean()), f"{prefix}_sd": float(f.std(ddof=1)) if f.size > 1 else np.nan,
            f"{prefix}_min": float(f.min()), f"{prefix}_max": float(f.max())}


def chirps_season_summary(path, boundary_path, small_mm: float = 1.0) -> dict:
    """So mo ta 1 file mua CHIRPS trong vung (tam pixel trong ranh gioi); NoData dem ca theo pixel cham."""
    b, prof = read_bands(path)
    center, touched = region_masks(boundary_path, prof)
    rs, jfm = b["rain_sum"], b["rain_jfm"]
    out = {"n_px_center": int(center.sum()), "n_px_touched": int(touched.sum()),
           "n_nodata_center": int((center & ~np.isfinite(rs)).sum()),
           "n_nodata_touched": int((touched & ~np.isfinite(rs)).sum()),
           "n_px_pentad_neg_touched": int((touched & (b["n_pentad_neg"] > 0)).sum()),
           "n_px_pentad_missing_touched": int((touched & ~(b["n_pentad_valid"] >= 36)).sum())}
    for m in ("m01", "m02", "m03"):
        v = b[f"rain_{m}"]
        out[f"n_zero_{m}"] = int((center & np.isfinite(v) & (v == 0)).sum())
    out["n_jfm_zero"] = int((center & np.isfinite(jfm) & (jfm == 0)).sum())
    out[f"n_jfm_lt{small_mm:g}mm"] = int((center & np.isfinite(jfm) & (jfm < small_mm)).sum())
    out.update(_desc(np.where(center, jfm, np.nan), "jfm"))
    out.update(_desc(np.where(center, rs, np.nan), "rain_sum"))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = (b["rain_m11"] + b["rain_m04"]) / rs
    out["share_m11_m04_mean"] = _desc(np.where(center & (rs > 0), share, np.nan), "s")["s_mean"]
    return out


def mcd18_season_summary(path, boundary_path) -> dict:
    """So mo ta 1 file mua MCD18 trong vung (tam pixel trong ranh gioi)."""
    b, prof = read_bands(path)
    center, touched = region_masks(boundary_path, prof)
    n = b["n_days_valid"]
    nc = n[center]
    out = {"n_px_center": int(center.sum()), "n_px_touched": int(touched.sum()),
           "n_px_no_valid_day_center": int((center & ~(n > 0)).sum()),
           "n_days_valid_min": float(np.nanmin(nc)) if nc.size else np.nan,
           "n_days_valid_mean": float(np.nanmean(nc)) if nc.size else np.nan}
    pos = nc[nc > 0]   # bo pixel khong co ngay nao (mat nuoc/bien theo mat na MODIS)
    out.update({"n_days_valid_min_pos": float(pos.min()) if pos.size else np.nan,
                "n_days_valid_mean_pos": float(pos.mean()) if pos.size else np.nan})
    q = _desc(np.where(center, b["frac_quality2"], np.nan), "frac_quality2")
    out.update({k: q[k] for k in ("frac_quality2_mean", "frac_quality2_min", "frac_quality2_max")})
    out.update(_desc(np.where(center, b["dsr_mean"], np.nan), "dsr_mean"))
    out.update(_desc(np.where(center, b["dsr_mean_no_dec"], np.nan), "dsr_mean_no_dec"))
    out.update(_desc(np.where(center, b["dsr_mean"] - b["dsr_mean_no_dec"], np.nan), "diff_dec"))
    return out


def is_modis_sphere(crs) -> bool:
    """True neu `crs` (rasterio/pyproj) la sinusoidal tren hinh cau R = 6371007.181 m."""
    from pyproj import CRS
    c = CRS.from_user_input(crs)
    if c.to_dict().get("proj") != "sinu":
        return False
    e = c.ellipsoid
    return e is not None and abs(e.semi_major_metre - 6371007.181) < 1e-3 and abs(e.semi_minor_metre - 6371007.181) < 1e-3


def set_modis_sphere_crs(path) -> bool:
    """Ghi de CRS file MODIS sang sinusoidal hinh cau (khong doi pixel/transform). Tra True neu da phai sua.
    File khong phai sinusoidal -> ValueError (khong doan)."""
    import rasterio
    from rasterio.crs import CRS as RCRS
    with rasterio.open(path, "r+") as ds:
        if is_modis_sphere(ds.crs):
            return False
        if ds.crs is None or "sinu" not in ds.crs.to_proj4():
            raise ValueError(f"{path}: CRS {ds.crs} khong phai sinusoidal")
        ds.crs = RCRS.from_proj4(MODIS_SINU_PROJ4)
    return True

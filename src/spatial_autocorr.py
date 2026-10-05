"""Tu tuong quan khong gian cua phan du / nhan tai diem danh gia (tuan 5 viec 1-2; quy tac An chot 2026-10-05).

Quy tac (ghi TRUOC khi tinh, NHAT_KY 2026-10-05):
  - Semivariogram: scikit-gstat, mo hinh MU, CO nugget, 30 bin deu tren 0-150 km, tam = tam hieu dung (effective
    range) cua skgstat; moi diem cua mua. Fit that bai / khong huu han / tam >= maxlag -> coi la VUOT (tra inf).
  - Vi pham cho mot bo (luoi, cach chia): trung vi tam qua cac mua > nguong HOAC so mua vuot nguong >= 3.
  - Moran's I: esda/libpysal, dai khoang cach 10 km nhi phan, chuan hoa hang; diem khong co lan can (dao) bi loai
    va dem; p hoan vi 999 lan, seed 42.
  - Bat dang huong (chi bao cao): DirectionalVariogram phuong vi 45 / 135 do, dung sai 22,5 do.
Toa do phai la MET (EPSG:32648).
"""
import warnings

import numpy as np
import pandas as pd

MAXLAG_M = 150_000.0
N_LAGS = 30
BLOCK_KM = 50.0
MAX_SEASONS_OVER = 2  # >= 3 mua vuot -> vi pham
MORAN_BAND_M = 10_000.0
MORAN_PERM = 999
MORAN_SEED = 42


def _check_xy(xy, values):
    xy = np.asarray(xy, dtype=float)
    v = np.asarray(values, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(v):
        raise ValueError("xy phai (n, 2) cung do dai voi values")
    if np.nanmax(np.abs(xy)) < 1000:
        raise ValueError("toa do phai la MET (UTM), khong phai do")
    ok = np.isfinite(v) & np.isfinite(xy).all(axis=1)
    return xy[ok], v[ok]


def variogram_range(xy, values, maxlag=MAXLAG_M, n_lags=N_LAGS, azimuth=None, tolerance=22.5) -> dict:
    """Tam hieu dung (m) cua semivariogram mu co nugget. Loi/khong huu han/tam >= maxlag -> range_m = inf, ok False."""
    import skgstat

    xy, v = _check_xy(xy, values)
    out = {"n": int(len(v)), "range_m": np.inf, "sill": np.nan, "nugget": np.nan, "ok": False, "reason": ""}
    if len(v) < 30 or np.nanstd(v) == 0:
        out["reason"] = "it diem hoac phuong sai 0"
        return out
    kw = dict(model="exponential", use_nugget=True, maxlag=maxlag, n_lags=n_lags, bin_func="even")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if azimuth is None:
                vg = skgstat.Variogram(xy, v, **kw)
            else:
                vg = skgstat.DirectionalVariogram(xy, v, azimuth=azimuth, tolerance=tolerance, **kw)
            r, sill, nug = (float(x) for x in vg.parameters[:3])
    except Exception as exc:  # fit that bai -> tinh la vuot (quy tac chot truoc)
        out["reason"] = f"fit loi: {type(exc).__name__}"
        return out
    out.update(sill=sill, nugget=nug)
    if not np.isfinite(r) or r <= 0:
        out["reason"] = "tam khong huu han"
        return out
    if r >= maxlag * (1 - 1e-3):  # skgstat chan tam o maxlag: fit cham bien (vd 149,9999 km) = "tam >= maxlag"
        out["reason"] = "tam cham bien maxlag"
        out["range_m"] = np.inf
        return out
    out.update(range_m=r, ok=True)
    return out


def violates(ranges_km, threshold_km=BLOCK_KM, max_over=MAX_SEASONS_OVER) -> dict:
    """Quy tac dung lai khoi cho mot bo (luoi, cach chia): range NaN/inf duoc tinh la VUOT."""
    r = np.asarray(ranges_km, dtype=float)
    r = np.where(np.isfinite(r), r, np.inf)
    over = int((r > threshold_km).sum())
    med = float(np.median(r)) if len(r) else np.inf
    return {"median_km": med, "max_km": float(np.max(r)) if len(r) else np.inf, "n_over": over,
            "violates": bool(med > threshold_km or over > max_over)}


def moran_band(xy, values, band=MORAN_BAND_M, permutations=MORAN_PERM, seed=MORAN_SEED) -> dict:
    """Moran's I dai khoang cach nhi phan, chuan hoa hang; dao (khong lan can) bi loai va dem."""
    import esda
    from libpysal.weights import DistanceBand

    xy, v = _check_xy(xy, values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = DistanceBand(xy, threshold=band, binary=True, silence_warnings=True)
        islands = list(w.islands)
        if islands:
            keep = np.setdiff1d(np.arange(len(v)), islands)
            xy, v = xy[keep], v[keep]
            w = DistanceBand(xy, threshold=band, binary=True, silence_warnings=True)
        w.transform = "r"
        np.random.seed(seed)
        m = esda.Moran(v, w, permutations=permutations)
    return {"n": int(len(v)), "n_islands": len(islands), "I": float(m.I), "EI": float(m.EI),
            "p_sim": float(m.p_sim), "z_sim": float(m.z_sim)}


def season_frame(points_xy: pd.DataFrame, values: pd.DataFrame, value_col: str) -> dict:
    """{season: (xy, v)}: points_xy (point_id, x, y) + values (point_id, season, value_col)."""
    d = values.merge(points_xy, on="point_id", how="left", validate="many_to_one")
    if d[["x", "y"]].isna().any().any():
        raise ValueError("diem khong co toa do")
    return {int(s): (g[["x", "y"]].to_numpy(float), g[value_col].to_numpy(float)) for s, g in d.groupby("season")}


def detrend_poly(values, x, degree=2):
    """Phan du cua hoi quy OLS values ~ 1 + x + ... + x^degree (CHG-20: x = khoang cach toi bo cua CHINH DIEM, km).

    Bo xu huong co he thong theo truc bien -> noi dong truoc khi tinh semivariogram (cung cau hinh cau 1).
    """
    v = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    if v.shape != x.shape:
        raise ValueError("values va x phai cung do dai")
    ok = np.isfinite(v) & np.isfinite(x)
    out = np.full(v.shape, np.nan)
    X = np.vander(x[ok], degree + 1, increasing=True)
    beta, *_ = np.linalg.lstsq(X, v[ok], rcond=None)
    out[ok] = v[ok] - X @ beta
    return out

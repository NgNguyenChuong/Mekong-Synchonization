"""Tu tuong quan khong gian cua phan du / nhan tai diem danh gia (tuan 5 viec 1-2; quy tac An chot 2026-10-05).

Quy tac (ghi TRUOC khi tinh, NHAT_KY 2026-10-05):
  - Semivariogram: scikit-gstat, mo hinh MU, CO nugget, 30 bin deu tren 0-150 km, tam = tam hieu dung (effective
    range) cua skgstat; moi diem cua mua. Fit that bai / khong huu han / tam >= maxlag -> coi la VUOT (tra inf).
  - CHG-22 (An 2026-10-05 21:22, 3 trang thai DAT / KHONG DAT / LOI): chi that bai THUC CHAT moi la VUOT -
    curve_fit khong hoi tu (RuntimeError "Optimal parameters not found", xem _KHONG_HOI_TU), tam cham bien
    >= 0,999*maxlag, tam = +inf. That bai PHAN MEM/SO HOC (het RAM, TypeError/AttributeError/LinAlgError/ValueError...,
    tham so NaN, tam <= 0), < 30 diem, phuong sai 0, gia tri khong huu han trong dau vao -> raise (LOI, dung).
  - Vi pham cho mot bo (luoi, cach chia): trung vi tam qua cac mua > nguong HOAC so mua vuot nguong >= 3.
  - Moran's I: esda/libpysal, dai khoang cach 10 km nhi phan, chuan hoa hang; diem khong co lan can (dao) bi loai
    va dem; p hoan vi 999 lan, seed 42.
  - Bat dang huong (chi bao cao): DirectionalVariogram phuong vi 45 / 135 do, dung sai 22,5 do.
Toa do phai la MET (EPSG:32648).
"""
import gc
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
MIN_POINTS = 30
# Thong diep scipy.optimize.curve_fit khi KHONG HOI TU (scipy 1.16 _minpack_py.py: trf -> `not res.success` = het
# max_nfev; lm -> ier not in 1..4). skgstat 1.0.24 Variogram.__init__ goi fit() -> curve_fit(method='trf', bounds
# (0, [maxlag, max gamma, 0,99 max gamma])) va KHONG bat loi -> RuntimeError noi thang ra. Day la that bai THUC CHAT.
_KHONG_HOI_TU = "Optimal parameters not found"


def _check_xy(xy, values):
    """Kiem dau vao; CHG-22: bat ky gia tri khong huu han nao (toa do hoac gia tri) -> LOI (ValueError).

    Moi loi goi hien tai (phan du OOF, y_ref, phan du bo xu huong) deu truyen gia tri huu han (kiem 2026-10-05:
    117 oof_points cv1 hist_gb, 0 err/y_ref khong huu han) -> khong co truong hop hop le nao can bo diem.
    """
    xy = np.asarray(xy, dtype=float)
    v = np.asarray(values, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(v):
        raise ValueError("xy phai (n, 2) cung do dai voi values")
    ok = np.isfinite(v) & np.isfinite(xy).all(axis=1)
    n_bo = int((~ok).sum())
    if n_bo:
        raise ValueError(f"LOI: {n_bo}/{len(v)} diem co toa do/gia tri khong huu han (CHG-22: khong bo lang)")
    if len(v) and np.max(np.abs(xy)) < 1000:
        raise ValueError("toa do phai la MET (UTM), khong phai do")
    return xy, v


def variogram_range(xy, values, maxlag=MAXLAG_M, n_lags=N_LAGS, azimuth=None, tolerance=22.5) -> dict:
    """Tam hieu dung (m) cua semivariogram mu co nugget.

    VUOT (range_m = inf, ok False): khong hoi tu (reason "fit khong hoi tu"), tam cham bien maxlag
    (reason "tam cham bien maxlag"), tam = +inf. LOI (raise): moi ngoai le khac, tham so NaN, tam <= 0,
    < 30 diem, phuong sai 0, dau vao khong huu han (CHG-22).
    """
    import skgstat

    xy, v = _check_xy(xy, values)
    out = {"n": int(len(v)), "range_m": np.inf, "sill": np.nan, "nugget": np.nan, "ok": False, "reason": ""}
    if len(v) < MIN_POINTS:
        raise ValueError(f"LOI: chi {len(v)} diem (< {MIN_POINTS}) - khong tinh semivariogram (CHG-22)")
    if np.std(v) == 0:
        raise ValueError("LOI: phuong sai 0 (gia tri hang so) - khong tinh semivariogram (CHG-22)")
    kw = dict(model="exponential", use_nugget=True, maxlag=maxlag, n_lags=n_lags, bin_func="even")
    vg = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if azimuth is None:
                vg = skgstat.Variogram(xy, v, **kw)
            else:
                vg = skgstat.DirectionalVariogram(xy, v, azimuth=azimuth, tolerance=tolerance, **kw)
            r, sill, nug = (float(x) for x in vg.parameters[:3])
    except RuntimeError as exc:
        # Chi "khong hoi tu" cua curve_fit la that bai thuc chat (tam qua lon) -> VUOT; RuntimeError khac -> LOI.
        if not str(exc).startswith(_KHONG_HOI_TU):
            raise
        out["reason"] = "fit khong hoi tu"
        return out
    finally:
        # MemoryError / moi loi khac noi len (LOI, dung). Variogram skgstat co tham chieu vong -> ma tran cap diem
        # (~0,85 GB / mua 7.000 diem) khong duoc giai phong ngay; chay lien tiep nhieu mua da lam tran RAM 16 GB
        # (2026-10-05). Don ngay sau moi lan fit.
        vg = None
        gc.collect()
    out.update(sill=sill, nugget=nug)
    if np.isnan(r) or not np.isfinite(sill) or not np.isfinite(nug):
        raise ValueError(f"LOI: tham so fit khong huu han (range={r}, sill={sill}, nugget={nug}) (CHG-22)")
    if r <= 0:
        raise ValueError(f"LOI: tam fit <= 0 (range={r}) (CHG-22)")
    if np.isposinf(r):
        out["reason"] = "tam khong huu han"
        return out
    if r >= maxlag * (1 - 1e-3):  # skgstat chan tam o maxlag: fit cham bien (vd 149,9999 km) = "tam >= maxlag"
        out["reason"] = "tam cham bien maxlag"
        out["range_m"] = np.inf
        return out
    out.update(range_m=r, ok=True)
    return out


def violates(ranges_km, threshold_km=BLOCK_KM, max_over=MAX_SEASONS_OVER) -> dict:
    """Quy tac dung lai khoi cho mot bo (luoi, cach chia): range +inf (vuot that) duoc tinh la VUOT.

    CHG-22: mang rong, NaN, -inf hoac tam <= 0 -> LOI (raise), khong bao gio quy ve vi pham/khong vi pham.
    """
    r = np.asarray(ranges_km, dtype=float)
    if r.ndim != 1 or len(r) == 0:
        raise ValueError("LOI: khong co mua nao de xet vi pham (CHG-22)")
    if np.isnan(r).any():
        raise ValueError(f"LOI: {int(np.isnan(r).sum())} tam NaN (CHG-22: NaN khong phai 'vuot')")
    if (r <= 0).any():
        raise ValueError("LOI: tam <= 0 hoac -inf (CHG-22)")
    over = int((r > threshold_km).sum())
    med = float(np.median(r))
    return {"median_km": med, "max_km": float(np.max(r)), "n_over": over,
            "violates": bool(med > threshold_km or over > max_over)}


def moran_band(xy, values, band=MORAN_BAND_M, permutations=MORAN_PERM, seed=MORAN_SEED) -> dict:
    """Moran's I dai khoang cach nhi phan, chuan hoa hang; dao (khong lan can) bi loai va dem."""
    import esda
    from libpysal.weights import DistanceBand

    xy, v = _check_xy(xy, values)
    if len(v) == 0:
        raise ValueError("LOI: khong co diem nao de tinh Moran (CHG-22)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = DistanceBand(xy, threshold=band, binary=True, silence_warnings=True)
        islands = list(w.islands)
        if islands:
            keep = np.setdiff1d(np.arange(len(v)), islands)
            if len(keep) == 0:  # CHG-22: khong con diem nao -> LOI, khong tra Moran rong
                raise ValueError(f"LOI: ca {len(v)} diem deu la dao (khong lan can trong {band:.0f} m)")
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

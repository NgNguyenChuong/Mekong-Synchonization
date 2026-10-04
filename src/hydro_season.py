"""Gop chuoi thuy van ngay (MRC, CMEMS zos) theo mua kho - bien cap vung, dung chung cho moi o.

Mua kho lay tu `seasons.py` (01/11/(Y-1) .. 29/04/Y), khong tu dinh nghia lai.
Nguyen tac: KHONG lap so lieu thieu; trung binh chi tren ngay co gia tri, mau so cua
`frac_days` la so ngay lich cua mua (`season_days`), nen ngay vang mat trong file cung tinh la thieu.
"""
import numpy as np
import pandas as pd

from seasons import season_days, season_of, season_window

OK_FRAC = 0.8  # mua co >= 80% ngay co so lieu -> *_ok = True (khong loai dong; nan_if_not_ok -> gia tri NaN)


def season_stats(dates, values, years, prefix, ok_frac=OK_FRAC, quantiles=None, roll_min=None,
                 nan_if_not_ok=False) -> pd.DataFrame:
    """Thong ke mot chuoi ngay theo mua kho.

    Tra ve mot dong moi mua trong `years`: `{prefix}_mean` (trung binh tren ngay co gia tri,
    NaN neu khong co ngay nao), `{prefix}_n_days`, `{prefix}_frac_days` (= n_days / season_days),
    `{prefix}_ok` (frac_days >= ok_frac). NaN/inf = thieu (khong coi la 0). Ngay trung -> loi.

    Cuc tri (tuy chon):
      quantiles : dict {hau_to: q}, vd {"p10": 0.10} -> `{prefix}_p10` = phan vi q cua cac ngay CO
                  so lieu trong mua (noi suy tuyen tinh, mac dinh numpy/pandas).
      roll_min  : so ngay w, vd 7 -> `{prefix}_min7` = trung binh truot w ngay LICH lien tiep nho nhat
                  trong mua. Chi tinh cua so nam tron trong mua va du w ngay co so lieu (ngay thieu
                  hoac vang mat trong file lam hong moi cua so chua no); khong cua so nao du -> NaN.
    nan_if_not_ok : True -> mua co `{prefix}_ok = False` thi `_mean` va moi cot cuc tri = NaN
                  (n_days/frac_days/ok giu nguyen de con biet vi sao).
    """
    idx = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if idx.duplicated().any():
        dup = idx[idx.duplicated()].dt.strftime("%Y-%m-%d").unique()[:5].tolist()
        raise ValueError(f"{prefix}: ngay bi trung {dup}")
    val = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    val = val.where(np.isfinite(val))
    d = pd.DataFrame({"date": idx, "season": season_of(idx), "value": val}).dropna(subset=["season"])
    quantiles = dict(quantiles or {})
    rows = []
    for y in years:
        v = d.loc[d["season"] == y, "value"].dropna()
        n = int(len(v))
        frac = n / season_days(y)
        ok = bool(frac >= ok_frac)
        row = {
            "season": int(y),
            f"{prefix}_mean": float(v.mean()) if n else np.nan,
            f"{prefix}_n_days": n,
            f"{prefix}_frac_days": frac,
            f"{prefix}_ok": ok,
        }
        for suf, q in quantiles.items():
            row[f"{prefix}_{suf}"] = float(v.quantile(q)) if n else np.nan
        if roll_min:
            row[f"{prefix}_min{roll_min}"] = _min_rolling_mean(d.loc[d["season"] == y], y, roll_min)
        if nan_if_not_ok and not ok:
            for k in row:
                if k not in ("season", f"{prefix}_n_days", f"{prefix}_frac_days", f"{prefix}_ok"):
                    row[k] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _min_rolling_mean(season_df, year, w) -> float:
    """Trung binh truot w ngay nho nhat; truc ngay lich day du cua mua, cua so thieu ngay -> bo."""
    start, end = season_window(year)
    s = season_df.set_index("date")["value"].reindex(pd.date_range(start, end, freq="D"))
    r = s.rolling(w, min_periods=w).mean()
    return float(r.min()) if r.notna().any() else np.nan


def censored_missing_days(dates, values, max_gap=8, edge_below=0.3, window=None) -> pd.Series:
    """Ngay thieu "bi chan" (bool, chi muc = truc ngay lich lien tuc).

    Ngay thieu = NaN/inf hoac vang mat trong file. Mot DOAN ngay thieu lien tiep dai <= `max_gap`
    ngay, co ngay CO so lieu ngay truoc va ngay sau doan, ca hai gia tri < `edge_below` -> moi ngay
    trong doan bi chan (so lieu thap bi ghi 0 roi thanh NaN: MRC, known-pitfalls 14/14b).
    Doan cham dau/cuoi truc (khong co mot dau) -> khong bi chan.

    window : (ngay dau, ngay cuoi) -> truc chi gom khoang nay (doan cham mep khoang coi nhu thieu
             mot dau). Mac dinh None = truc tu ngay dau den ngay cuoi cua chuoi: doan cat ngang
             ranh gioi mua van xet dau thuc cua no (vd doan 23/04-29/04 co dau phai la 30/04).
    """
    idx = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if idx.duplicated().any():
        raise ValueError("censored_missing_days: ngay bi trung")
    val = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    s = pd.Series(val.where(np.isfinite(val)).to_numpy(), index=idx).sort_index()
    if window is not None:
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    else:
        start, end = s.index.min(), s.index.max()
    s = s.reindex(pd.date_range(start, end, freq="D"))
    v = s.to_numpy()
    miss = np.isnan(v)
    out = np.zeros(len(v), dtype=bool)
    i, n = 0, len(v)
    while i < n:
        if not miss[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and miss[j + 1]:
            j += 1
        if (j - i + 1) <= max_gap and i > 0 and j < n - 1 and v[i - 1] < edge_below and v[j + 1] < edge_below:
            out[i:j + 1] = True
        i = j + 1
    return pd.Series(out, index=s.index)


def censored_season_stats(dates, values, years, prefix, q=0.20, suffix="p20c", floor=0.0, max_gap=8,
                          edge_below=0.3, low_thresh=0.3, warn_frac=0.20, ok_frac=OK_FRAC,
                          nan_if_not_ok=True, runs_within_season=False) -> pd.DataFrame:
    """Phan vi duoi co xu ly chan (An duyet cau B, 2026-10-03) cho chuoi muc nuoc ngay.

    - Chan CA chuoi o `floor` (gia tri < floor -> floor, moi nam) de truoc/sau 2021 nhat quan
      (truoc 2021 khong co ngay am nao vi so lieu thap bi ghi 0 -> NaN).
    - Ngay thieu bi chan (`censored_missing_days`) coi la DUOI moi gia tri do (-inf) khi tinh phan vi,
      method "lower" (lay mot gia tri that cua mau, khong noi suy voi -inf). Ket qua < floor
      (phan vi roi vao vung chan) -> floor.
    - Ngay thieu KHONG bi chan: bo qua (khong lap, khong coi la 0).

    Cot (mot dong moi mua trong `years`):
      `{prefix}_{suffix}`          : phan vi q tren mau = ngay co so lieu (da chan) + ngay thieu bi chan.
      `{prefix}_n_censored`        : so ngay bi chan (ngay thieu bi chan) + ngay co so lieu <= floor.
      `{prefix}_days_le_<low>`     : so ngay <= low_thresh (ngay co so lieu sau chan + ngay thieu bi chan)
                                     - do nhay, khong phu thuoc gia tri ben duoi nguong.
    `_ok` tinh nhu season_stats (ngay CO so lieu / season_days >= ok_frac); `nan_if_not_ok` -> hai cot
    gia tri = NaN (n_censored giu nguyen nhu n_days). Ty le chan = n_censored / co mau >= warn_frac
    -> canh bao (phan vi co the bi anh huong).

    runs_within_season : True -> xet doan thieu chi trong cua so mua (doan cham 01/11 hoac 29/04 coi nhu
                         thieu mot dau -> khong chan). Mac dinh False: dung dau thuc ngoai mua.
    """
    import warnings

    idx = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    val = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    full = pd.Series(val.where(np.isfinite(val)).to_numpy(), index=idx)
    if full.index.duplicated().any():
        raise ValueError(f"{prefix}: ngay bi trung")
    full = full.sort_index()
    cens_all = None if runs_within_season else censored_missing_days(full.index, full.values, max_gap, edge_below)
    low_col = f"{prefix}_days_le_{str(low_thresh).replace('.', '_')}"
    rows = []
    for y in years:
        start, end = season_window(y)
        obs = full.loc[start:end].dropna()
        obs = obs.clip(lower=floor)
        if runs_within_season:
            cens = censored_missing_days(full.index, full.values, max_gap, edge_below, window=(start, end))
        else:
            cens = cens_all.reindex(pd.date_range(start, end, freq="D"), fill_value=False)
        n_cens_missing = int(cens.loc[start:end].sum())
        n_obs = int(len(obs))
        n_sample = n_obs + n_cens_missing
        n_censored = n_cens_missing + int((obs <= floor).sum())
        ok = bool(n_obs / season_days(y) >= ok_frac)
        if n_sample:
            x = np.r_[obs.to_numpy(dtype="float64"), np.full(n_cens_missing, -np.inf)]
            p = float(np.quantile(x, q, method="lower"))
            p = max(p, floor)
            frac = n_censored / n_sample
            if frac >= warn_frac:
                warnings.warn(f"{prefix} mua {y}: ty le ngay bi chan {frac:.1%} >= {warn_frac:.0%} "
                              f"-> {suffix} co the bi anh huong boi chan", stacklevel=2)
        else:
            p = np.nan
        row = {
            "season": int(y),
            f"{prefix}_{suffix}": p,
            f"{prefix}_n_censored": n_censored,
            low_col: float(int((obs <= low_thresh).sum()) + n_cens_missing) if n_sample else np.nan,
        }
        if nan_if_not_ok and not ok:
            row[f"{prefix}_{suffix}"] = np.nan
            row[low_col] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def regional_daily_mean(field, lat, mask) -> np.ndarray:
    """Trung binh vung moi ngay cua `field` (time, lat, lon) tren cac pixel `mask` (lat, lon).

    Trong so cos(vi do) (luoi deu theo do). Ngay nao co bat ky pixel trong mask bi thieu
    -> NaN ca ngay (tranh trung binh tren tap pixel thay doi theo ngay; khong lap gia tri).
    """
    field = np.asarray(field, dtype="float64")
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Mask pixel rong")
    w = np.broadcast_to(np.cos(np.deg2rad(np.asarray(lat, dtype="float64")))[:, None], mask.shape)[mask]
    sel = field[:, mask]                                   # (time, n_pixel)
    complete = np.isfinite(sel).all(axis=1)
    out = np.full(field.shape[0], np.nan)
    out[complete] = (sel[complete] * w).sum(axis=1) / w.sum()
    return out


def coastal_pixel_mask(lat, lon, boundary_geom, max_km, crs_metric="EPSG:32648") -> np.ndarray:
    """True o pixel co TAM cach `boundary_geom` (EPSG:4326) <= max_km (do trong crs_metric).

    Khoang cach toi DA GIAC (bang 0 neu tam nam trong ranh gioi), khong phai toi bo bien.
    """
    import geopandas as gpd

    lon2, lat2 = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    pts = gpd.GeoSeries(gpd.points_from_xy(lon2.ravel(), lat2.ravel()), crs="EPSG:4326").to_crs(crs_metric)
    geom = gpd.GeoSeries([boundary_geom], crs="EPSG:4326").to_crs(crs_metric).iloc[0]
    dist_km = pts.distance(geom).to_numpy() / 1000.0
    return (dist_km <= max_km).reshape(lat2.shape)

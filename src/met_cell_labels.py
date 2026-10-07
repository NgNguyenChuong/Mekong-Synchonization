"""Nhan khi tuong Dot 7 theo o (TB gia tri pixel nguon chua tam moi pixel pham vi 30 m, giong dac trung tinh)
va dap an tai diem (pixel nguon chua diem). O khong pham vi, hoac pham vi chi tren pixel nguon NaN -> NaN.
"""
import json
import os

import numpy as np
import pandas as pd

from label_season import points_to_pixels

MET_RULES_VERSION = "dot7_e5_scope_pixel_count_2026-10-07"   # doi quy tac -> doi chuoi de bang cu bi tinh lai

# path = (thu muc con trong DATA_ROOT..., mau ten file); bien cung `source` dung chung ma tran dem
MET_TARGETS = {
    "rain": {"col": "rain_chirps", "band": "rain_sum", "path": ("raw", "chirps3", "chirps3_rain_{s}.tif"),
             "source": "chirps3", "crs_kind": "geographic", "nan_seasons": (),
             "unit": "mm/mua kho (tong 36 pentad CHIRPS v3, 01/11..30/04)", "same_nan_all_seasons": True},
    "dsr": {"col": "dsr_mcd18", "band": "dsr_mean", "path": ("raw", "mcd18a1", "mcd18a1_dsr_{s}.tif"),
            "source": "mcd18a1", "crs_kind": "modis_sphere", "nan_seasons": (2020,),
            "unit": "W/m2 (TB ngay buc xa toi be mat MCD18A1.062, 01/11..29/04)", "same_nan_all_seasons": False},
    "temp": {"col": "t2m_era5", "band": "temp_c_mean", "path": ("features", "era5_season", "temp_avg_{s}.tif"),
             "source": "era5", "crs_kind": "geographic", "nan_seasons": (),
             "unit": "do C (TB ngay T2m ERA5-Land, raster mua GOC chua lap)", "same_nan_all_seasons": False},
    "rh": {"col": "rh_era5", "band": "rh_percent_mean", "path": ("features", "era5_season", "humid_{s}.tif"),
           "source": "era5", "crs_kind": "geographic", "nan_seasons": (),
           "unit": "% (TB ngay RH ERA5-Land, raster mua GOC chua lap)", "same_nan_all_seasons": False},
}
QUALITY_COLS = ("lbl_cover_frac", "lbl_valid_px", "lbl_ok")


def target_spec(key):
    if key not in MET_TARGETS:
        raise ValueError(f"Bien '{key}' khong thuoc {sorted(MET_TARGETS)}")
    return MET_TARGETS[key]


def check_source(path, spec):
    """Kiem file nguon: co band theo ten, dung loai CRS, khong phai ban lap ERA5 (`_filled`). Sai -> ValueError."""
    import rasterio

    from met_labels import is_modis_sphere

    if not os.path.exists(path):
        raise ValueError(f"Thieu file nguon {path}")
    base = os.path.basename(path)
    if "_filled" in base:
        raise ValueError(f"{base}: ban LAP (CHG-11) - nhan khong duoc dung ban lap")
    prov = f"{path}.provenance.json"
    if os.path.exists(prov):
        with open(prov, encoding="utf-8") as f:
            if "fill_rule" in json.load(f):
                raise ValueError(f"{base}: provenance co fill_rule - la ban lap, khong dung lam nhan")
    with rasterio.open(path) as ds:
        names = list(ds.descriptions)
        if spec["band"] not in names:
            raise ValueError(f"{base}: khong co band '{spec['band']}' (co {names})")
        if spec["crs_kind"] == "modis_sphere":
            if not is_modis_sphere(ds.crs):
                raise ValueError(f"{base}: CRS khong phai sinusoidal HINH CAU R = 6371007,181 (pitfall 45)")
        elif not (ds.crs is not None and ds.crs.is_geographic):
            raise ValueError(f"{base}: can CRS kinh-vi, gap {ds.crs}")
        return grid_signature(ds), names.index(spec["band"]) + 1


def grid_signature(ds):
    """(crs wkt, transform 6 so, rong, cao) - de kiem cung luoi giua cac mua / giua cac bien cung nguon."""
    return (ds.crs.to_wkt(), tuple(round(float(v), 9) for v in tuple(ds.transform)[:6]), ds.width, ds.height)


def scope_source_counts(scope_path, cell_data, sources, chunk_rows=1024):
    """Ma tran dem (o, pixel nguon, so tam pixel pham vi 30 m) cho moi nguon {khoa: (crs, transform, (cao, rong))}.

    Gan o theo tam pixel nhu scope_centroids; tra (cell_ids, {khoa: DataFrame cell, px, n}, info) - info[khoa]
    co n_scope_outside (pixel pham vi ngoai luoi nguon, nguoi goi bao LOI).
    """
    import rasterio
    from pyproj import CRS, Transformer

    from processing import _cells_gdf
    from static_features import iter_scope_cell_pixels

    acc = {k: [] for k in sources}
    n_out = dict.fromkeys(sources, 0)
    n_in = 0
    with rasterio.open(scope_path) as sc:
        gdf = _cells_gdf(cell_data, sc.crs).reset_index(drop=True)
        groups = {}
        for k, (crs, _, _) in sources.items():
            groups.setdefault(CRS.from_user_input(crs).to_wkt(), []).append(k)
        tfs = {wkt: Transformer.from_crs(sc.crs, CRS.from_wkt(wkt), always_xy=True) for wkt in groups}
        for idx, x, y in iter_scope_cell_pixels(sc, gdf, chunk_rows):
            n_in += int(idx.size)
            for wkt, keys in groups.items():
                X, Y = tfs[wkt].transform(x, y)
                for k in keys:
                    _, st, (h, w) = sources[k]
                    r, c = points_to_pixels(X, Y, st)
                    ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
                    n_out[k] += int((~ok).sum())
                    key = idx[ok].astype(np.int64) * (h * w) + r[ok] * w + c[ok]
                    acc[k].append(np.unique(key, return_counts=True))
                del X, Y
    out = {}
    for k, (_, _, (h, w)) in sources.items():
        if acc[k]:
            u = np.concatenate([a for a, _ in acc[k]])
            c = np.concatenate([b for _, b in acc[k]])
            uu, inv = np.unique(u, return_inverse=True)
            cnt = np.bincount(inv, weights=c).astype(np.int64)
        else:
            uu, cnt = np.zeros(0, np.int64), np.zeros(0, np.int64)
        out[k] = pd.DataFrame({"cell": uu // (h * w), "px": uu % (h * w), "n": cnt})
    info = {k: {"n_scope_in_cells": n_in, "n_scope_outside": n_out[k]} for k in sources}
    return gdf["cell_id"].astype(str).to_numpy(), out, info


def cell_count_labels(counts, cell_ids, values: dict, value_col, nan_seasons=()):
    """Nhan (o, mua) = TB tren pixel pham vi co gia tri nguon huu han; mua thuoc nan_seasons -> NaN.

    Tra cell_id, season, <value_col>, lbl_cover_frac (NaN neu o khong pham vi), lbl_valid_px, lbl_ok.
    """
    n_cells = len(cell_ids)
    cell = counts["cell"].to_numpy(dtype=np.int64)
    px = counts["px"].to_numpy(dtype=np.int64)
    n = counts["n"].to_numpy(dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("ma tran dem co n <= 0")
    if counts.duplicated(["cell", "px"]).any():
        raise ValueError("ma tran dem trung cap (o, pixel nguon)")
    if cell.size and (cell.min() < 0 or cell.max() >= n_cells):
        raise ValueError("chi so o ngoai 0..n-1")
    n_all = np.bincount(cell, weights=n, minlength=n_cells)
    parts = []
    for s in sorted(values):
        arr = np.asarray(values[s], dtype=np.float64).ravel()
        if px.size and px.max() >= arr.size:
            raise ValueError(f"mua {s}: chi so pixel nguon vuot kich thuoc raster")
        v = np.full(px.size, np.nan) if s in nan_seasons else arr[px]
        ok = np.isfinite(v)
        n_ok = np.bincount(cell[ok], weights=n[ok], minlength=n_cells)
        s_v = np.bincount(cell[ok], weights=n[ok] * v[ok], minlength=n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(n_ok > 0, s_v / n_ok, np.nan)
            cover = np.where(n_all > 0, n_ok / n_all, np.nan)
        parts.append(pd.DataFrame({"cell_id": cell_ids, "season": int(s), value_col: val, "lbl_cover_frac": cover,
                                   "lbl_valid_px": n_ok.astype(np.int64), "lbl_ok": np.isfinite(val)}))
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(["cell_id", "season"]).any():
        raise ValueError("khoa (cell_id, season) trung")
    return out


def point_pixel_values(values: dict, transform, xs, ys, nan_seasons=()):
    """Gia tri pixel nguon CHUA diem (xs, ys cung CRS raster) cho moi mua. Diem ngoai luoi -> ValueError.
    Tra (DataFrame idx, season, value, src_row, src_col)."""
    rows, cols = points_to_pixels(xs, ys, transform)
    parts = []
    for s in sorted(values):
        arr = np.asarray(values[s], dtype=np.float64)
        h, w = arr.shape
        out = (rows < 0) | (rows >= h) | (cols < 0) | (cols >= w)
        if out.any():
            raise ValueError(f"{int(out.sum())} diem nam ngoai luoi nguon")
        v = np.full(len(rows), np.nan) if s in nan_seasons else arr[rows, cols]
        parts.append(pd.DataFrame({"idx": np.arange(len(rows)), "season": int(s), "value": v,
                                   "src_row": rows, "src_col": cols}))
    return pd.concat(parts, ignore_index=True)


def nan_pattern_changes(values: dict, weights, nan_seasons=()):
    """(so pixel co weights > 0 ma NaN doi giua cac mua, so pixel NaN moi mua); bo mua nan_seasons."""
    seasons = [s for s in sorted(values) if s not in nan_seasons]
    m = np.stack([~np.isfinite(np.asarray(values[s], dtype=np.float64)) for s in seasons])
    changes = m.any(axis=0) & ~m.all(axis=0) & (np.asarray(weights) > 0)
    always = m.all(axis=0) & (np.asarray(weights) > 0)
    return int(changes.sum()), int(always.sum())

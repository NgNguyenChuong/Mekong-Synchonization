"""Nhan theo o + dap an tai diem cho 4 bien khi tuong Dot 7 (E5b mua + buc xa, E5c nhiet do + do am).

Quy tac (An chot 2026-10-07, NHAT_KY "An chot E5 + X + doi chung duong"):
  - Nhan theo o = TB CO TRONG SO PHAM VI: trong so moi pixel nguon w_p = ty le dien tich pixel p thuoc pham vi
    (scope_mask_v3.tif 30 m, band scope == 1; cung mat na dac trung tinh). exactextract `weighted_mean`:
        nhan_o = sum_p(cov_op * w_p * v_p) / sum_p(cov_op * w_p)   (chi pixel v_p huu han)
    cov_op = ty le dien tich pixel p nam trong o (exactextract). Da giac o CHIEU sang CRS raster nguon (MCD18 =
    sinusoidal HINH CAU), KHONG lay mau lai raster nguon. O khong co dien tich pham vi tren pixel hop le -> NaN.
  - w_p tinh bang dem TAM pixel scope 30 m roi vao pixel nguon: w_p = n_scope_p * 900 m2 / dien tich pixel p
    (pixel kinh-vi: dien tich geodesic WGS84; sinusoidal hinh cau: |a*e| - phep chieu dong dien tich). Lech ty le
    chung (UTM k^2 ~ 0,9993; hinh cau vs ellipsoid) gan nhu deu tren vung -> khong doi TB co trong so; w > 1 do sai so
    nay bi cat ve 1 (ghi so lieu max truoc khi cat).
  - Cot chat luong (KHONG phai dac trung; mau cam trong check_features bat):
      lbl_cover_frac = sum_valid(cov*w) / sum_all(cov*w) - ty le pham vi cua o nam tren pixel nguon co gia tri;
      lbl_valid_px   = sum_valid(cov*w) - dien tich pham vi tren pixel hop le, don vi "pixel nguon tuong duong";
      lbl_ok         = nhan huu han.
  - Dap an tai diem = gia tri pixel nguon CHUA diem (khong 3x3, khong noi suy); pixel khong hop le -> NaN, dem.
  - Mua trong `nan_seasons` (buc xa 2020, E4 phuong an (b)) -> NaN toan bo (o + diem).
  - Nhiet do / do am: raster ERA5 mua GOC chua lap (<bien>_<mua>.tif) - KHONG dung ban `_filled` (CHG-11).
"""
import json
import os

import numpy as np
import pandas as pd

from label_season import points_to_pixels

MET_RULES_VERSION = "dot7_e5_scope_weighted_2026-10-07"   # doi quy tac -> doi chuoi (bang cu khong dung lai)

# bien -> cau hinh. path = (thu muc con trong DATA_ROOT..., mau ten file); source = khoa luoi nguon (trong so chung)
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
    """Kiem file nguon: ton tai, co band theo TEN, CRS dung loai, khong phai ban lap ERA5. Sai -> ValueError."""
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
    """(crs wkt, transform 6 so, rong, cao) - de kiem cung luoi giua cac mua / voi raster trong so."""
    return (ds.crs.to_wkt(), tuple(round(float(v), 9) for v in tuple(ds.transform)[:6]), ds.width, ds.height)


def pixel_areas_m2(crs, transform, shape):
    """Dien tich (m2) moi pixel cua luoi nguon: kinh-vi -> geodesic WGS84 theo hang; chieu -> |a*e|."""
    from pyproj import CRS, Geod

    h, w = shape
    c = CRS.from_user_input(crs)
    if not c.is_geographic:
        return np.full((h, w), abs(transform.a * transform.e), dtype=np.float64)
    if transform.b != 0 or transform.d != 0:
        raise ValueError("luoi kinh-vi xoay - khong ho tro")
    geod = Geod(ellps="WGS84")
    x0, x1 = transform.c, transform.c + transform.a
    rows = np.empty(h)
    for r in range(h):
        y0, y1 = transform.f + r * transform.e, transform.f + (r + 1) * transform.e
        area, _ = geod.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
        rows[r] = abs(area)
    return np.repeat(rows[:, None], w, axis=1)


def scope_weights(scope_path, src_crs, src_transform, src_shape, chunk_rows=1024):
    """Ty le dien tich pham vi (scope == 1) cua moi pixel nguon, tu tam pixel scope 30 m.

    Tra (w float64 [0, 1], thong tin: n_scope_px, n_scope_outside, w_max_raw). Doc scope theo khoi hang.
    Pixel scope roi ngoai luoi nguon -> n_scope_outside > 0 (nguoi goi quyet LOI).
    """
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window

    from scope_mask import band_index

    h, w = src_shape
    counts = np.zeros(h * w, dtype=np.int64)
    n_scope = n_out = 0
    with rasterio.open(scope_path) as sc:
        b = band_index(sc, "scope")
        tr = sc.transform
        px_m2 = abs(tr.a * tr.e)
        to_src = Transformer.from_crs(sc.crs, src_crs, always_xy=True)
        for r0 in range(0, sc.height, chunk_rows):
            n = min(chunk_rows, sc.height - r0)
            arr = sc.read(b, window=Window(0, r0, sc.width, n))
            rr, cc = np.nonzero(arr == 1)
            del arr
            if not rr.size:
                continue
            x = tr.c + (cc + 0.5) * tr.a
            y = tr.f + (r0 + rr + 0.5) * tr.e
            X, Y = to_src.transform(x, y)
            r, c = points_to_pixels(X, Y, src_transform)
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            n_scope += int(rr.size)
            n_out += int((~ok).sum())
            counts += np.bincount(r[ok] * w + c[ok], minlength=h * w)
    area = pixel_areas_m2(src_crs, src_transform, src_shape)
    frac = counts.reshape(h, w) * px_m2 / area
    info = {"n_scope_px": n_scope, "n_scope_outside": n_out, "w_max_raw": float(frac.max()) if frac.size else 0.0,
            "n_px_w_gt0": int((frac > 0).sum()), "n_px_w_gt1": int((frac > 1).sum()), "scope_px_m2": px_m2}
    return np.minimum(frac, 1.0), info


def write_raster(path, bands, crs, transform, names, nodata=np.nan):
    """Ghi GeoTIFF float32 nhieu band (dat ten band)."""
    import rasterio

    arr = np.stack([np.asarray(b, dtype=np.float32) for b in bands])
    prof = dict(driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0], dtype="float32",
                crs=crs, transform=transform, nodata=nodata, compress="deflate")
    tmp = path + ".part.tif"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)
    os.replace(tmp, path)


def stack_seasons(values: dict, nan_seasons=()):
    """{mua: mang gia tri} -> (danh sach band [gia tri mua..., co hop le mua...], ten band, mua).
    Mua thuoc nan_seasons -> gia tri NaN, co hop le 0. Co hop le = gia tri huu han (0/1, khong NaN)."""
    seasons = sorted(values)
    vals, valid = [], []
    for s in seasons:
        v = np.asarray(values[s], dtype=np.float64)
        if s in nan_seasons:
            v = np.full_like(v, np.nan)
        vals.append(v)
        valid.append(np.isfinite(v).astype(np.float64))
    names = [f"v_{s}" for s in seasons] + [f"ok_{s}" for s in seasons]
    return vals + valid, names, seasons


def cell_weighted_labels(stack_path, weight_path, cell_data, seasons, value_col):
    """exactextract weighted_mean / weighted_sum tren raster chong mua (band v_<s>, ok_<s>) voi trong so pham vi.

    Tra DataFrame dai: cell_id, season, <value_col>, lbl_cover_frac, lbl_valid_px, lbl_ok.
    Kiem: nhan huu han <=> lbl_valid_px > 0 (lech -> ValueError, khong doan).
    """
    import rasterio
    from exactextract import exact_extract

    from processing import _cells_gdf

    with rasterio.open(stack_path) as ds:
        crs = ds.crs
        names = list(ds.descriptions)
    gdf = _cells_gdf(cell_data, crs)
    res = exact_extract(stack_path, gdf, ["weighted_mean", "weighted_sum"], weights=weight_path,
                        include_cols=["cell_id"], output="pandas")
    res = res.set_index("cell_id").reindex(gdf["cell_id"])
    parts = []
    for s in seasons:
        iv, io = names.index(f"v_{s}") + 1, names.index(f"ok_{s}") + 1
        val = res[f"band_{iv}_weight_weighted_mean"].to_numpy(dtype=float)
        cover = res[f"band_{io}_weight_weighted_mean"].to_numpy(dtype=float)
        vpx = res[f"band_{io}_weight_weighted_sum"].fillna(0).to_numpy(dtype=float)
        fin = np.isfinite(val)
        bad = fin != (vpx > 0)
        if bad.any():
            raise ValueError(f"mua {s}: {int(bad.sum())} o nhan huu han khong khop dien tich pham vi hop le > 0")
        parts.append(pd.DataFrame({"cell_id": gdf["cell_id"].to_numpy(), "season": int(s), value_col: val,
                                   "lbl_cover_frac": cover, "lbl_valid_px": vpx, "lbl_ok": fin}))
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
    """So pixel CO TRONG SO > 0 ma trang thai NaN doi giua cac mua (bo mua nan_seasons)."""
    seasons = [s for s in sorted(values) if s not in nan_seasons]
    m = np.stack([~np.isfinite(np.asarray(values[s], dtype=np.float64)) for s in seasons])
    changes = m.any(axis=0) & ~m.all(axis=0) & (np.asarray(weights) > 0)
    always = m.all(axis=0) & (np.asarray(weights) > 0)
    return int(changes.sum()), int(always.sum())

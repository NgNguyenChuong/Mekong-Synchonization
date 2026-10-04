"""Gop ERA5-Land ngay theo mua kho va trich theo o (trung binh co trong so dien tich).

Mua kho lay tu `seasons.py` (01/11/(Y-1) .. 29/04/Y) - khong tu dinh nghia lai. Moi file thang
`<bien>_YYYY_MM.tif` co band k = ngay k cua thang; band thuoc mua nao do `season_of` quyet dinh
(thang 4 chi lay band 1..29, thang 10 va 5 khong lay band nao).

Quy tac gop (NHAT_KY, quy uoc du lieu):
  - mua (rain) = TONG, cat gia tri am ve 0 TRUOC khi cong (ERA5 tru tich luy -> ~ -4e-5);
  - bien khac  = TRUNG BINH;
  - pixel phai co gia tri o DU moi ngay cua mua (`season_days`), thieu >= 1 ngay -> NaN (khong lap);
  - pixel ngoai mat na dat cua temp_avg -> NaN cho MOI bien (loi solar = 0 o bien, known-pitfalls 1).
Gia tri thieu = NaN/inf hoac bang `nodata` cua file (ERA5 tai ve dung -inf).
"""
import os
import tempfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from processing import area_weighted_means
from seasons import season_days, season_of, season_window
from utils_h3 import parse_year_month

# bien -> (thu muc trong RAW_DIR, cot dau ra = ten trong DEFAULT_ALLOWED_FEATURES, cach gop)
ERA5_SEASON_SPECS = {
    "rain": ("daily_rain", "rain_mm", "sum"),
    "solar": ("daily_solar", "solar", "mean"),
    "temp_avg": ("daily_temp_avg", "temp_c", "mean"),
    "temp_max": ("daily_temp_max", "temp_max_c", "mean"),
    "temp_min": ("daily_temp_min", "temp_min_c", "mean"),
    "humid": ("daily_humid", "rh_percent", "mean"),
}
CLIP_NEGATIVE = {"rain"}   # bien cat am ve 0 truoc khi gop
LAND_VAR = "temp_avg"      # mat na dat dung chung cho moi bien
COVER_COL = "era5_cover_frac"
FILL_COL = "era5_fill_frac"          # ty le dien tich o nam tren pixel LAP (cau K, chi o ban _filled)
ZERO_PAD_COLS = {COVER_COL, FILL_COL}  # cot mat na 0/1: dem bang 0 (khong phai NaN) khi trich


def index_monthly_files(folder) -> dict:
    """{(nam, thang): duong dan} cho moi .tif trong `folder`. Hai file cung (nam, thang) -> loi."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Khong co thu muc ERA5: {folder}")
    out = {}
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".tif", ".tiff")):
            continue
        ym = parse_year_month(f)
        if ym is None:
            continue
        if ym in out:
            raise ValueError(f"Hai file cung thang {ym}: {os.path.basename(out[ym])}, {f}")
        out[ym] = os.path.join(folder, f)
    return out


def season_months(year: int) -> list:
    start, end = season_window(year)
    return list(pd.period_range(start, end, freq="M"))


def _grid_key(ds):
    return (str(ds.crs), tuple(round(v, 9) for v in ds.transform[:6]), ds.width, ds.height)


def aggregate_season(file_map: dict, year: int, how: str, clip_negative: bool = False):
    """Gop mot bien theo mua kho `year`.

    Tra ve (value float32 [NaN neu pixel thieu ngay], n_days int16, profile, thang_thieu).
    `how`: "sum" | "mean". Band vuot qua so ngay cua thang -> loi; luoi pixel khac nhau giua thang -> loi.
    """
    if how not in ("sum", "mean"):
        raise ValueError(f"how = {how!r}")
    acc = cnt = ref = profile = None
    missing = []
    for p in season_months(year):
        path = file_map.get((p.year, p.month))
        if path is None:
            missing.append(f"{p.year}-{p.month:02d}")
            continue
        with rasterio.open(path) as ds:
            key = _grid_key(ds)
            if ref is None:
                ref, profile = key, {"crs": ds.crs, "transform": ds.transform, "width": ds.width, "height": ds.height}
                acc = np.zeros((ds.height, ds.width), "float64")
                cnt = np.zeros((ds.height, ds.width), "int32")
            elif key != ref:
                raise ValueError(f"Luoi pixel khac nhau: {os.path.basename(path)} {key} vs {ref}")
            if ds.count > p.days_in_month:
                raise ValueError(f"{os.path.basename(path)}: {ds.count} band > {p.days_in_month} ngay cua thang")
            dates = p.start_time + pd.to_timedelta(np.arange(ds.count), unit="D")
            in_season = (season_of(dates) == year).fillna(False).to_numpy(dtype=bool)
            nodata = ds.nodata
            for b in np.flatnonzero(in_season):
                arr = ds.read(int(b) + 1).astype("float64")
                valid = np.isfinite(arr)
                if nodata is not None and np.isfinite(nodata):
                    valid &= arr != nodata
                if clip_negative:
                    arr = np.where(valid, np.maximum(arr, 0.0), arr)
                acc[valid] += arr[valid]
                cnt += valid
    if ref is None:
        raise FileNotFoundError(f"Mua {year}: khong co file thang nao ({missing})")
    full = cnt == season_days(year)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = acc if how == "sum" else acc / cnt
    value = np.where(full, value, np.nan).astype("float32")
    return value, cnt.astype("int16"), profile, missing


def season_layers(raw_dir: str, year: int, specs=ERA5_SEASON_SPECS, file_maps=None, land_var=LAND_VAR):
    """Raster mua cho moi bien + mat na hop le chung.

    Tra ve (layers {bien: value}, n_days {bien: cnt}, valid_all bool, profile, diag {bien: dict}).
    Mat na dat = pixel `land_var` co gia tri it nhat 1 ngay trong mua; ngoai mat na -> NaN moi bien.
    `valid_all` = pixel dat va MOI bien deu du ngay (dung cho era5_cover_frac).
    """
    if land_var not in specs:
        raise ValueError(f"Thieu bien mat na dat {land_var}")
    file_maps = file_maps or {v: index_monthly_files(os.path.join(raw_dir, s[0])) for v, s in specs.items()}
    layers, counts, diag, profile = {}, {}, {}, None
    for var, (_, _, how) in specs.items():
        val, cnt, prof, missing = aggregate_season(file_maps[var], year, how, clip_negative=var in CLIP_NEGATIVE)
        if profile is None:
            profile = prof
        elif _grid_key_profile(prof) != _grid_key_profile(profile):
            raise ValueError(f"Bien {var} khac luoi pixel voi bien truoc")
        layers[var], counts[var] = val, cnt
        diag[var] = {"missing_months": missing}
    land = counts[land_var] > 0
    valid_all = land.copy()
    expected = season_days(year)
    for var in specs:
        cnt = counts[var]
        diag[var].update(
            px_full=int((cnt == expected).sum()),
            px_partial=int(((cnt > 0) & (cnt < expected)).sum()),
            px_outside_land=int(((cnt > 0) & ~land).sum()),   # phai = 0 (mat na giong nhau giua bien)
        )
        layers[var] = np.where(land, layers[var], np.nan).astype("float32")
        valid_all &= np.isfinite(layers[var])
    return layers, counts, valid_all, profile, diag


def _grid_key_profile(p):
    return (str(p["crs"]), tuple(round(v, 9) for v in p["transform"][:6]), p["width"], p["height"])


def write_season_raster(path, value, n_days, profile, description, band2="n_days"):
    """Raster mua 2 band: 1 = gia tri (NaN = thieu), 2 = `band2` (mac dinh so ngay co gia tri). Ghi tam roi doi ten."""
    tmp = f"{path}.part.tif"
    with rasterio.open(tmp, "w", driver="GTiff", height=profile["height"], width=profile["width"], count=2,
                       dtype="float32", crs=profile["crs"], transform=profile["transform"], nodata=np.nan,
                       compress="deflate") as dst:
        dst.write(value.astype("float32"), 1)
        dst.write(n_days.astype("float32"), 2)
        dst.descriptions = (description, band2)
    os.replace(tmp, path)


def season_table(layers_by_season: dict, profile: dict, cells, pad_px: int = 5) -> pd.DataFrame:
    """Trich trung binh theo dien tich cho moi (o, mua).

    layers_by_season: {mua: {cot: mang 2D (NaN = thieu)}} - cot trong `ZERO_PAD_COLS` la mat na 0/1.
    Raster duoc dem them `pad_px` pixel moi canh (NaN cho gia tri, 0 cho mat na) de o tran ra ngoai
    vung ERA5 tai ve van tinh dung ty le phu tren TOAN dien tich o. O vuot ca vung dem -> loi.
    Mot lan exact_extract cho moi luoi (raster nhieu band tam). O khong co pixel hop le -> NaN.
    """
    seasons = sorted(layers_by_season)
    cols = list(layers_by_season[seasons[0]])
    bands, names = [], []
    for s in seasons:
        if list(layers_by_season[s]) != cols:
            raise ValueError(f"Mua {s} khac bo cot")
        for c in cols:
            arr = np.asarray(layers_by_season[s][c], dtype="float32")
            fill = 0.0 if c in ZERO_PAD_COLS else np.nan
            bands.append(np.pad(arr, pad_px, constant_values=fill))
            names.append((s, c))
    t = profile["transform"]
    transform = t @ Affine.translation(-pad_px, -pad_px)
    h, w = bands[0].shape
    _check_cells_inside(cells, profile["crs"], transform, w, h)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "stack.tif")
        with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=len(bands), dtype="float32",
                           crs=profile["crs"], transform=transform, nodata=np.nan) as dst:
            for i, b in enumerate(bands, start=1):
                dst.write(b, i)
        wide = area_weighted_means(path, cells)
    wide["cell_id"] = wide["cell_id"].astype(str)
    parts = []
    for s in seasons:
        sub = pd.DataFrame({"cell_id": wide["cell_id"], "season": int(s)})
        for c in cols:
            sub[c] = wide[f"b{names.index((s, c)) + 1}"].to_numpy()
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(["cell_id", "season"]).any():
        raise ValueError("Khoa (cell_id, season) bi trung")
    return out


def boundary_touch_mask(boundary_path, profile) -> np.ndarray:
    """Pixel raster ERA5 GIAO ranh gioi (all_touched: moi pixel co phan dien tich chung voi da giac)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    b = gpd.read_file(boundary_path).to_crs(profile["crs"])
    shapes = [(g, 1) for g in b.geometry if g is not None and not g.is_empty]
    return rasterize(shapes, out_shape=(profile["height"], profile["width"]), transform=profile["transform"],
                     fill=0, all_touched=True, dtype="uint8").astype(bool)


# 8 lang gieng: (dr, dc); canh truoc (tam cach 1 px), cheo sau (tam cach sqrt(2) px)
_ROOK = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _shifted(a, dr, dc, fill):
    """out[r, c] = a[r + dr, c + dc]; ngoai bien raster -> `fill`."""
    out = np.full_like(a, fill)
    h, w = a.shape
    rs, re_ = max(0, -dr), min(h, h - dr)
    cs, ce = max(0, -dc), min(w, w - dc)
    out[rs:re_, cs:ce] = a[rs + dr:re_ + dr, cs + dc:ce + dc]
    return out


def fill_nearest_1px(layers: dict, valid: np.ndarray, target: np.ndarray):
    """Cau K: lap pixel KHONG hop le thuoc `target` bang pixel hop le gan nhat TRONG 1 PIXEL (8 lang gieng).

    Quy tac (gan nhat theo tam pixel, hoa thi trung binh):
      - co >= 1 lang gieng CANH hop le (tam cach 1 px) -> trung binh cac lang gieng canh hop le;
      - khong co canh, co >= 1 lang gieng CHEO hop le (sqrt(2) px) -> trung binh cac lang gieng cheo hop le;
      - khong co lang gieng hop le nao -> giu NaN (pixel cach >= 2 pixel KHONG duoc lap).
    Nguon chi la pixel hop le GOC (`valid`) - pixel vua lap khong lam nguon (khong lan truyen).
    Moi bien dung CUNG tap pixel nguon (`valid` = hop le o moi bien) -> nhat quan giua bien.

    Tra ve (layers_filled {bien: mang}, filled bool, n_src int8 [so pixel nguon, 0 neu khong lap]).
    """
    valid = np.asarray(valid, bool)
    cand = np.asarray(target, bool) & ~valid
    n_rook = sum(_shifted(valid, dr, dc, False).astype("int8") for dr, dc in _ROOK)
    n_diag = sum(_shifted(valid, dr, dc, False).astype("int8") for dr, dc in _DIAG)
    use_rook = cand & (n_rook > 0)
    use_diag = cand & (n_rook == 0) & (n_diag > 0)
    filled = use_rook | use_diag
    n_src = np.where(use_rook, n_rook, np.where(use_diag, n_diag, 0)).astype("int8")
    out = {}
    for name, arr in layers.items():
        arr = np.asarray(arr, "float64")
        bad = valid & ~np.isfinite(arr)
        if bad.any():
            raise ValueError(f"{name}: {int(bad.sum())} pixel 'hop le' nhung gia tri khong huu han")
        src = np.where(valid, arr, 0.0)
        s_rook = sum(_shifted(src, dr, dc, 0.0) for dr, dc in _ROOK)
        s_diag = sum(_shifted(src, dr, dc, 0.0) for dr, dc in _DIAG)
        res = np.where(valid, arr, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            res = np.where(use_rook, s_rook / n_rook, res)
            res = np.where(use_diag, s_diag / n_diag, res)
        out[name] = res.astype("float32")
    return out, filled, n_src


def _check_cells_inside(cells, crs, transform, width, height):
    from shapely.geometry import box
    from processing import _cells_gdf
    x0, y1 = transform @ (0, 0)
    x1, y0 = transform @ (width, height)
    frame = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    gdf = _cells_gdf(cells, crs)
    bad = gdf.loc[~gdf.within(frame), "cell_id"].tolist()
    if bad:
        raise ValueError(f"{len(bad)} o vuot ra ngoai raster da dem (vd {bad[:3]}) - tang pad_px")

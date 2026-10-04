"""Dac trung tinh tren pixel scope (src/static_features.py) - raster TONG HOP gia tri biet truoc.

Luoi scope 30 m, 6 x 6 pixel, EPSG:32648, goc (500000, 1100180). O "nua" = ca khung 180 x 180 m:
cot 0-2 scope = 1 (dat), cot 3-5 scope = 0 (nuoc). DEM: dat 2,0 / nuoc 0,0 ("0 gia").
Khoang cach 90 m: 2 x 2 pixel [[1, 2], [3, 4]] (band k nhan 10^(k-1)).
WorldCover 10 m 18 x 18: nua trai 40 (Cropland), nua phai 80 (Water).
Co DEM (WBM) 30 m: rong hon, goc lech -3 cot (nhu file that lech -100 cot): dat 0, nuoc 1.
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import static_features as sf  # noqa: E402

CRS = "EPSG:32648"
X0, Y0 = 500000.0, 1100180.0


def _write(path, bands, res, x0=X0, y0=Y0, dtype="float32", names=None, nodata=None):
    bands = [np.asarray(b, dtype=dtype) for b in bands]
    h, w = bands[0].shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=len(bands), dtype=dtype, crs=CRS,
                       transform=from_origin(x0, y0, res, res), nodata=nodata) as ds:
        for i, b in enumerate(bands, start=1):
            ds.write(b, i)
            if names:
                ds.set_band_description(i, names[i - 1])
    return str(path)


def _cells(boxes_utm, ids):
    """cell_data (ids, None, geoms EPSG:4326) tu hinh chu nhat UTM - nhu luoi that (geojson 4326)."""
    g = gpd.GeoSeries(boxes_utm, crs=CRS).to_crs(4326)
    return (list(ids), None, list(g))


@pytest.fixture
def rasters(tmp_path):
    scope = np.zeros((6, 6), "uint8")
    scope[:, :3] = 1
    sc = _write(tmp_path / "scope.tif", [scope, scope * 10], 30, dtype="uint8", names=["scope", "zenodo_n_years"])
    # DEM rong hon 2 pixel moi phia (goc lech so nguyen pixel) - van "cung luoi".
    dem = np.zeros((10, 10), "float32")
    dem[:, :5] = 2.0  # cot 2-4 cua DEM = cot 0-2 cua scope -> dat; con lai 0 (nuoc)
    dm = _write(tmp_path / "dem.tif", [dem], 30, x0=X0 - 60, y0=Y0 + 60, names=["DEM"])
    fl = _flags(tmp_path / "flags.tif", np.where(np.arange(6) < 3, 0, 1)[None, :].repeat(6, 0))
    base = np.array([[1, 2], [3, 4]], "float32")
    rv = _write(tmp_path / "river.tif", [base, base * 10, base * 100], 90, nodata=np.nan,
                names=list(sf.RIVER_BANDS))
    wc = np.full((18, 18), 40, "uint8")
    wc[:, 9:] = 80
    lc = _write(tmp_path / "wc.tif", [wc], 10, dtype="uint8", names=["Map"])
    return {"scope": sc, "dem": dm, "river": rv, "wc": lc, "flags": fl, "dir": tmp_path}


def _flags(path, wbm_scope, pad=3, x_shift=0.0):
    """Ghi co DEM (band WBM, EDM) phu khung scope 6 x 6, rong them `pad` pixel moi phia (goc lech -pad cot/hang)."""
    wbm = np.zeros((6 + 2 * pad, 6 + 2 * pad), "uint8")
    wbm[pad:pad + 6, pad:pad + 6] = wbm_scope
    return _write(path, [wbm, np.ones_like(wbm)], 30, x0=X0 - 30 * pad + x_shift, y0=Y0 + 30 * pad,
                  dtype="uint8", names=["WBM", "EDM"])


def _run(r, cells, **kw):
    stack = str(r["dir"] / "stack.tif")
    stats = sf.build_scope_stack(r["scope"], r["dem"], r["river"], stack, r["flags"], **kw)
    return sf.cell_static_features(stack, r["scope"], r["wc"], cells).set_index("cell_id"), stats


def test_half_scope_cell_means_only_scope_part(rasters):
    cells = _cells([box(X0, Y0 - 180, X0 + 180, Y0)], ["half"])
    df, stats = _run(rasters, cells)
    row = df.loc["half"]
    # DEM: chi phan dat (2,0), khong bi "0 gia" tren nuoc keo ve 1,0.
    assert row["dem_mean"] == pytest.approx(2.0)
    # Khoang cach: pixel scope o cot 90 m dau -> trung binh pixel 1 va 3 = 2,0 (ca o: 2,5).
    assert row["dist_main_river_km"] == pytest.approx(2.0)
    assert row["dist_any_water_km"] == pytest.approx(20.0)
    assert row["dist_coast_km"] == pytest.approx(200.0)
    assert row["scope_frac"] == pytest.approx(0.5, rel=1e-6)
    # Lop phu tren TOAN O: nua Cropland, nua Water (khong phai 100% Cropland nhu neu chi tinh tren scope).
    assert row["landcover_class_Cropland"] == pytest.approx(0.5, rel=1e-6)
    assert row["landcover_class_Water"] == pytest.approx(0.5, rel=1e-6)
    assert row["landcover_class_Mangroves"] == 0.0
    assert stats["scope_px"] == 18 and stats["dem_zero_px"] == 0 and stats["dem_nan_px"] == 0
    assert stats["wbm_px"] == {"0": 18, "1": 0, "2": 0, "3": 0}


def test_cell_without_scope_is_nan_not_zero(rasters):
    # Canh 4326 -> UTM lech ~1e-9 m: dat canh o giua pixel de khong cham manh pixel dat ben canh.
    cells = _cells([box(X0 + 100, Y0 - 180, X0 + 180, Y0), box(X0, Y0 - 180, X0 + 80, Y0)], ["water", "land"])
    df, _ = _run(rasters, cells)
    w = df.loc["water"]
    for c in sf.SCOPE_FEATURES:
        assert np.isnan(w[c]), c
    assert w["scope_frac"] == pytest.approx(0.0, abs=1e-12)
    assert w["landcover_class_Water"] == pytest.approx(1.0)  # lop phu van co tren toan o
    assert df.loc["land", "dem_mean"] == pytest.approx(2.0)


def test_dem_nan_in_scope_is_skipped_not_filled(rasters, tmp_path):
    dem = np.full((6, 6), 4.0, "float32")
    dem[:, 0] = np.nan  # cot dat dau tien khong co DEM
    dem[:, 1] = 1.0
    dem[:, 3:] = 0.0
    _write(tmp_path / "dem2.tif", [dem], 30, names=["DEM"])
    rasters["dem"] = str(tmp_path / "dem2.tif")
    df, stats = _run(rasters, _cells([box(X0, Y0 - 180, X0 + 180, Y0)], ["c"]))
    assert df.loc["c", "dem_mean"] == pytest.approx(2.5)  # (1 + 4) / 2; NaN dien 0 se ra 5/3
    assert stats["dem_nan_px"] == 6


def test_river_sampled_at_scope_pixel_centre_when_grids_not_aligned(rasters, tmp_path):
    # Goc raster 90 m lech -30 m: pixel 90 m cot 0 phu x in [X0-30, X0+60) -> tam pixel 30 m cot 0, 1;
    # cot 1 phu [X0+60, X0+150) -> cot 2 (cot 3, 4 ngoai scope).
    base = np.array([[1, 2, 5], [3, 4, 6]], "float32")
    rv = _write(tmp_path / "river2.tif", [base, base, base], 90, x0=X0 - 30, nodata=np.nan,
                names=list(sf.RIVER_BANDS))
    rasters["river"] = rv
    df, _ = _run(rasters, _cells([box(X0, Y0 - 180, X0 + 180, Y0)], ["c"]))
    # Hang 0-2 (y > Y0-90): cot 0,1 -> 1; cot 2 -> 2. Hang 3-5: cot 0,1 -> 3; cot 2 -> 4.
    expected = (3 * (1 + 1 + 2) + 3 * (3 + 3 + 4)) / 18
    assert df.loc["c", "dist_main_river_km"] == pytest.approx(expected)


def test_dem_grid_not_aligned_raises(rasters, tmp_path):
    _write(tmp_path / "dem3.tif", [np.ones((6, 6), "float32")], 30, x0=X0 + 15, names=["DEM"])
    rasters["dem"] = str(tmp_path / "dem3.tif")
    with pytest.raises(ValueError, match="khong nguyen"):
        _run(rasters, _cells([box(X0, Y0 - 180, X0 + 180, Y0)], ["c"]))


def test_landcover_code_zero_has_no_column_but_stays_in_denominator(rasters, tmp_path):
    wc = np.full((18, 18), 40, "uint8")
    wc[:9, :] = 0  # nua tren "ngoai vung"
    _write(tmp_path / "wc0.tif", [wc], 10, dtype="uint8", names=["Map"])
    rasters["wc"] = str(tmp_path / "wc0.tif")
    df, _ = _run(rasters, _cells([box(X0, Y0 - 180, X0 + 180, Y0)], ["c"]))
    lc_cols = [c for c in df.columns if c.startswith(sf.LANDCOVER_PREFIX)]
    assert not any(c.endswith("_0") for c in lc_cols)
    assert len(lc_cols) == len(sf.WORLDCOVER_NAMES)
    assert df.loc["c", "landcover_class_Cropland"] == pytest.approx(0.5, rel=1e-6)


def test_unknown_worldcover_code_raises(rasters, tmp_path):
    wc = np.full((18, 18), 40, "uint8")
    wc[0, 0] = 70
    _write(tmp_path / "wc70.tif", [wc], 10, dtype="uint8", names=["Map"])
    rasters["wc"] = str(tmp_path / "wc70.tif")
    with pytest.raises(ValueError, match="khong co ten"):
        _run(rasters, _cells([box(X0, Y0 - 180, X0 + 180, Y0)], ["c"]))


def test_names_match_config_and_allowlist():
    from config import DEFAULT_STATIC_SPECS
    from training.features import resolve_feature_list, find_leak_columns, DEFAULT_ALLOWED_FEATURES

    assert sf.WORLDCOVER_NAMES == {int(k): v for k, v in DEFAULT_STATIC_SPECS["landcover"]["class_names"].items()}
    cols = ["cell_id", *sf.SCOPE_FEATURES, sf.SCOPE_FRAC_COL,
            *[f"{sf.LANDCOVER_PREFIX}{n}" for n in sf.WORLDCOVER_NAMES.values()]]
    chosen, _ = resolve_feature_list(cols, DEFAULT_ALLOWED_FEATURES)
    assert sf.SCOPE_FRAC_COL not in chosen and "cell_id" not in chosen
    assert set(chosen) == set(cols) - {"cell_id", sf.SCOPE_FRAC_COL}
    assert find_leak_columns(chosen) == []


def _chg09_fixture(rasters, tmp_path):
    """3 cot dat (CHG-09): cot 0 DEM 0 & WBM 2 -> NaN; cot 1 DEM 0 & WBM 0 -> giu 0; cot 2 DEM 0,3 & WBM 2 -> giu."""
    dem = np.zeros((6, 6), "float32")
    dem[:, 2] = 0.3
    _write(tmp_path / "dem_chg09.tif", [dem], 30, names=["DEM"])
    rasters["dem"] = str(tmp_path / "dem_chg09.tif")
    wbm = np.zeros((6, 6), "uint8")
    wbm[:, 0] = 2
    wbm[:, 2] = 2
    wbm[:, 3:] = 3  # nuoc ngoai scope
    rasters["flags"] = _flags(tmp_path / "flags_chg09.tif", wbm, pad=100 // 30)


def test_chg09_dem_zero_and_wbm_is_nan_others_kept(rasters, tmp_path):
    _chg09_fixture(rasters, tmp_path)
    cells = _cells([box(X0, Y0 - 180, X0 + 180, Y0), box(X0, Y0 - 180, X0 + 25, Y0), box(X0 + 35, Y0 - 180, X0 + 55, Y0)],
                   ["c", "zero_wbm", "zero_dry"])
    df, stats = _run(rasters, cells)
    # Cot 0 bi loai -> (0 + 0,3) / 2. Quy tac theo gia tri 0 se ra 0,3; quy tac chi WBM se ra 0.
    assert df.loc["c", "dem_mean"] == pytest.approx(0.15)
    # O chi gom pixel (0, WBM 2) -> NaN (khong phai 0); dac trung khoang cach van co.
    assert np.isnan(df.loc["zero_wbm", "dem_mean"])
    assert df.loc["zero_wbm", "dist_main_river_km"] == pytest.approx(2.0)
    # O chi gom pixel (0, WBM 0) -> 0 that, giu.
    assert df.loc["zero_dry", "dem_mean"] == pytest.approx(0.0, abs=1e-9)
    assert stats["dem_zero_px"] == 12 and stats["dem_zero_wbm_to_nan_px"] == 6
    assert stats["dem_zero_wbm0_kept_px"] == 6 and stats["dem_low_wbm_kept_px"] == 6
    assert stats["wbm_px"] == {"0": 6, "1": 0, "2": 12, "3": 0}


def test_dem_missing_mask_truth_table():
    dem = np.array([0.0, 0.0, 0.3, 0.0, np.nan, 2.0], "float32")
    wbm = np.array([2, 0, 2, 1, 2, 3], "uint8")
    assert sf.dem_missing_mask(dem, wbm).tolist() == [True, False, False, True, False, False]


def test_flags_grid_not_aligned_raises(rasters, tmp_path):
    rasters["flags"] = _flags(tmp_path / "flags_shift.tif", np.zeros((6, 6), "uint8"), x_shift=10.0)
    with pytest.raises(ValueError, match="khong nguyen"):
        _run(rasters, _cells([box(X0, Y0 - 180, X0 + 180, Y0)], ["c"]))


def test_scope_pixel_outside_flags_frame_raises(rasters, tmp_path):
    # Co chi phu 5 cot dau (khung lech nguyen pixel nhung hep hon scope) -> cot scope ngoai khung khong duoc doan.
    wbm = np.zeros((6, 2), "uint8")
    rasters["flags"] = _write(tmp_path / "flags_small.tif", [wbm, wbm], 30, dtype="uint8", names=["WBM", "EDM"])
    with pytest.raises(ValueError, match="ngoai khung"):
        _run(rasters, _cells([box(X0, Y0 - 180, X0 + 180, Y0)], ["c"]))


def test_without_flags_is_explicit_control_only(rasters, tmp_path):
    # Ban doi chung (None co y): DEM giu nguyen ca pixel (0, WBM 2); ghi ro trong thong ke.
    _chg09_fixture(rasters, tmp_path)
    rasters["flags"] = None
    df, stats = _run(rasters, _cells([box(X0, Y0 - 180, X0 + 180, Y0)], ["c"]))
    assert df.loc["c", "dem_mean"] == pytest.approx(0.1)
    assert stats["dem_zero_wbm_to_nan_px"] == 0 and stats["dem_rule"].startswith("KHONG")

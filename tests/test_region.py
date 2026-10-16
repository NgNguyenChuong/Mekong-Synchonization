"""Tham so vung (src/region.py): dbscl giu duong cu, vung doi chung dung UTM/ten file rieng; o vuong UTM theo vung."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import box, shape
from shapely.ops import transform

import region
from label_season import label_values
from preprocessing import CANONICAL_BOUNDARY, generate_square_grid
from settings import data_path
from unified_table import merge_sources

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_ba_vung_epsg_nhan_hau_to():
    assert set(region.REGIONS) == {"dbscl", "mississippi", "nile"}
    assert {k: v["epsg"] for k, v in region.REGIONS.items()} == {"dbscl": 32648, "mississippi": 32615, "nile": 32636}
    assert {k: v["label"] for k, v in region.REGIONS.items()} == {"dbscl": "MD", "mississippi": "MS", "nile": "NL"}
    assert region.REGIONS["mississippi"]["landsat_suffix"] == region.REGIONS["nile"]["landsat_suffix"] == "_l8_sf3"
    assert region.crs_metric() == "EPSG:32648" and region.crs_metric("nile") == "EPSG:32636"
    assert region.SEASON_WINDOW == ((11, 1), (4, 29))
    with pytest.raises(ValueError):
        region.get_region("mekong")


def test_dbscl_mac_dinh_tra_dung_duong_cu():
    p = region.region_paths()
    assert p["boundary"] == CANONICAL_BOUNDARY
    assert p["grids_dir"] == os.path.join(ROOT, "data", "grids")
    assert p["eval_dir"] == os.path.join(ROOT, "data", "eval")
    assert p["report_dir"] == os.path.join(ROOT, "KE_HOACH", "ket-qua")
    assert p["scope"] == data_path("features/scope_mask_v3.tif")
    assert p["worldcover"] == data_path("gee/LandCover_DBSCL_2021_v2.tif")
    assert p["dem"] == data_path("gee/DEM_DBSCL.tif") and p["dem_flags"] == data_path("gee/DEM_DBSCL_flags_v2.tif")
    assert p["river"] == data_path("raw/river/river_distance_90m.tif")
    assert f"{p['features_dir']}/static" == data_path("features/static")
    assert f"{p['labels_dir']}/dot7" == f"{data_path()}/labels/dot7"
    from config import CRS_METRIC, crs_metric
    assert crs_metric() == CRS_METRIC == "EPSG:32648"


def test_vung_doi_chung_dau_vao_dau_ra(monkeypatch):
    monkeypatch.setenv("DOI_CHUNG_OUT", "X:/ra/")
    p = region.region_paths("mississippi")
    assert p["grids_dir"] == "X:/ra/mississippi/grids" and p["scope"] == "X:/ra/mississippi/features/scope_mask_v3.tif"
    assert p["boundary"] == data_path("doi_chung", "mississippi", "boundary.geojson")
    assert p["worldcover"].endswith("doi_chung/mississippi/static/LandCover_MS_2021.tif")
    assert p["dem_flags"].endswith("static/DEM_MS_flags.tif") and p["jrc"].endswith("static/JRC_GSW_MS.tif")
    assert p["raw_dir"] == data_path("doi_chung", "mississippi", "raw")
    assert region.landsat_file("nile", "ndwi", 2020).endswith("doi_chung/nile/landsat/2020_NL_dry_NDWIchen_l8_sf3.tif")
    assert region.landsat_file("mississippi", "watermask", 2014).endswith("landsat/2014_MS_dry_watermask_l8_sf3.tif")
    monkeypatch.setenv("DOI_CHUNG_OUT", "")  # bien trong .env.example de trong -> mac dinh
    assert region.region_paths("nile")["eval_dir"] == "C:/NCKH_Mekong/doi_chung_out/nile/eval"
    with pytest.raises(ValueError):
        region.landsat_file("dbscl", "ndwi", 2020)


def _parser(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="cu.tif")
    ap.add_argument("--compare", default="v2.tif")
    region.add_region_arg(ap, {"scope": "scope"}, extra={"compare": "none"}, argv=argv)
    return ap.parse_args(argv)


def test_add_region_arg_mac_dinh_va_ghi_de():
    a = _parser([])
    assert a.region == "dbscl" and a.scope == region.region_paths()["scope"] and a.compare == "v2.tif"
    b = _parser(["--region", "nile"])
    assert b.scope == region.region_paths("nile")["scope"] and b.compare == "none"
    c = _parser(["--region=nile", "--scope", "rieng.tif"])  # nguoi dung truyen -> thang
    assert c.scope == "rieng.tif" and c.compare == "none"


def _boundary(tmp_path, lon, lat, d=0.15):
    p = tmp_path / "b.geojson"
    geom = box(lon, lat, lon + d, lat + d).__geo_interface__
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": geom}]}), encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("name,lon,lat", [("mississippi", -91.2, 29.9), ("nile", 31.0, 30.8), ("dbscl", 105.8, 10.0)])
def test_o_vuong_utm_theo_vung(tmp_path, name, lon, lat):
    size = 6458
    crs = region.crs_metric(name)
    g = generate_square_grid(_boundary(tmp_path, lon, lat), size, crs_metric=crs)
    to_m = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    for cid, geom in zip(g["cell_id"], g.geometry):
        x, y = (int(v) for v in cid.split("_")[-2:])
        assert x % size == 0 and y % size == 0  # goc neo boi so canh trong UTM cua vung
        gm = transform(to_m, geom)
        assert np.allclose(gm.bounds, (x, y, x + size, y + size), atol=1e-3)
    bnd = transform(to_m, shape(json.loads(open(_boundary(tmp_path, lon, lat)).read())["features"][0]["geometry"]))
    assert bnd.difference(transform(to_m, g.geometry.union_all())).area < 1.0  # phu het ranh gioi


def test_o_vuong_mac_dinh_van_32648(tmp_path):
    b = _boundary(tmp_path, 105.8, 10.0)
    assert generate_square_grid(b, 2441)["cell_id"].tolist() == \
        generate_square_grid(b, 2441, crs_metric="EPSG:32648")["cell_id"].tolist()


def test_nhan_ndwi_khong_co_band_salinity():
    nd = np.array([0.2, np.nan, -0.5], np.float32)
    out = label_values(nd, None, "ndwi")
    assert out[0] == np.float32(0.2) and np.isnan(out[1]) and out[2] == np.float32(-0.5)
    with pytest.raises(ValueError):
        label_values(nd, None, "salinity")


def test_ghep_khong_thuy_van_hybrid_cot_vang_mat():
    keys = pd.DataFrame([(c, s) for c in ("a", "b") for s in (2019, 2020)], columns=["cell_id", "season"])
    labels = keys.assign(ndwi=[0.1, np.nan, 0.2, 0.3], n_valid_px=100.0, valid_frac=0.5, train_ok=True,
                         train_ok_10pct=True)
    era5 = keys.assign(rain_mm=[1.0, 2.0, 3.0, np.nan], era5_cover_frac=1.0, era5_fill_frac=0.0)
    static = pd.DataFrame({"cell_id": ["a", "b"], "dem_mean": [1.0, np.nan], "scope_frac": [0.5, 0.2]})
    t = merge_sources(labels, era5, static, None, None, target_col="ndwi")
    assert list(t.columns[:3]) == ["cell_id", "season", "ndwi"] and len(t) == 4
    assert not {"tch_wl_p20c", "zos_coast_p90", "dist_mouth_river_km", "sluice_frac"} & set(t.columns)
    assert t["rain_mm"].isna().sum() == 1 and t["dem_mean"].isna().sum() == 2  # NaN giu, khong dien 0
    with pytest.raises(ValueError, match="muc tieu"):
        merge_sources(labels, era5, static, None, None)  # mac dinh van doi cot salinity


def test_eval_dir_vung_doi_chung(monkeypatch):
    monkeypatch.setenv("DOI_CHUNG_OUT", "X:/ra")
    for name in ("mississippi", "nile"):
        p = region.region_paths(name)
        assert p["eval_dir"] == f"X:/ra/{name}/eval" and p["report_dir"] == f"X:/ra/{name}/reports"
        assert p["river"] == f"X:/ra/{name}/features/river/river_distance_90m.tif"


@pytest.mark.parametrize("name,lon", [("mississippi", -89.9), ("nile", 29.2)])
def test_khoi_50km_neo_crs_vung(tmp_path, name, lon):
    import geopandas as gpd

    from eval_design import make_holdout_blocks, refresh_blocks

    # ranh gioi nam o mui UTM ke ben: estimate_utm_crs chon mui khac -> phai dung CRS truyen vao
    bnd = gpd.read_file(_boundary(tmp_path, lon, 30.4, d=0.6))
    crs = region.crs_metric(name)
    assert bnd.estimate_utm_crs().to_epsg() != region.get_region(name)["epsg"]
    size = 50_000
    blk = make_holdout_blocks(bnd, 50.0, 0.2, 42, utm=crs)
    assert blk["is_holdout"].any() and not blk["is_holdout"].all()
    ref, added = refresh_blocks(blk, bnd, 50.0, utm=crs)
    assert added == [] and ref["block_id"].tolist() == blk["block_id"].tolist()
    assert ref["is_holdout"].tolist() == blk["is_holdout"].tolist()
    for bid, g in zip(ref["block_id"], ref.to_crs(crs).geometry):
        i, j = (int(v) for v in bid.split("_")[1:])
        assert np.allclose(g.bounds, (i * size, j * size, (i + 1) * size, (j + 1) * size), atol=1.0)
    land = bnd.to_crs(crs).geometry.union_all()
    assert np.isclose(ref["land_km2"].sum(), land.area / 1e6, rtol=1e-6)


def test_unified_dot7_bang_goc_ndwi_khong_thuy_van():
    from unified_table import NDWI_LABEL_COLS, SAL_TRAIN_COL, TRAIN_COL, target_table

    keys = pd.DataFrame([(c, s) for c in ("a", "b") for s in (2019, 2020)], columns=["cell_id", "season"])
    base = keys.assign(ndwi=[0.1, np.nan, 0.2, 0.3], n_valid_px=100.0, valid_frac=0.5, train_ok=[True, False, True, True],
                       train_ok_10pct=True, train_ok_scope=[True, False, True, True], rain_mm=[1.0, 2.0, 3.0, 4.0],
                       dem_mean=[1.0, 1.0, np.nan, np.nan], dist_coast_km=5.0)
    lab = keys.assign(rain_chirps=[5.0, 6.0, np.nan, 8.0], lbl_cover_frac=1.0, lbl_valid_px=3, lbl_ok=True)
    t, dropped = target_table(base, lab, "rain_chirps", ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"),
                              base_label_cols=NDWI_LABEL_COLS)
    assert dropped == ["rain_mm"] and "ndwi" not in t.columns and "salinity" not in t.columns
    assert not {"tch_wl_p20c", "zos_coast_p90", "dist_mouth_river_km", "sluice_frac", "zos_mouth_p90"} & set(t.columns)
    assert t[SAL_TRAIN_COL].tolist() == [True, False, True, True]
    assert t[TRAIN_COL].tolist() == [True, False, False, True]  # nhan moi NaN -> khong huan luyen
    assert t["dem_mean"].isna().sum() == 2  # NaN giu, khong dien 0
    with pytest.raises(ValueError, match="thieu cot"):
        target_table(base, lab, "rain_chirps")  # mac dinh doi bang goc do man

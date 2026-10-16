"""Chon song chinh HydroRIVERS + mat nuoc JRC rong cho vung doi chung."""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_river_features import distance_km  # noqa: E402
from build_river_features_doi_chung import select_main, wide_linear_water  # noqa: E402


def test_select_main_threshold_inclusive():
    g = pd.DataFrame({"HYRIV_ID": [1, 2, 3], "DIS_AV_CMS": [99.9, 100.0, 21894.0]})
    assert select_main(g, 100.0)["HYRIV_ID"].tolist() == [2, 3]


def test_wide_linear_water_drops_small_canal_and_pond():
    w = np.zeros((80, 80), bool)
    w[5:8, 2:78] = True        # song rong 3 x 76 px
    w[20, 2:78] = True         # kenh 1 x 76 px (nho hon min_km2)
    w[40:60, 40:60] = True     # ao vuong
    keep = wide_linear_water(w, px_km2=0.01, min_km2=1.5)
    assert keep[6, 40] and not keep[20, 40] and not keep[50, 50]


def test_wide_linear_water_near_keeps_only_touching_component():
    w = np.zeros((60, 100), bool)
    w[5:8, 2:98] = True        # nhanh gan song HydroRIVERS
    w[40:43, 2:98] = True      # dam pha dang dai o xa
    near = np.zeros_like(w)
    near[5:8, 10] = True
    keep = wide_linear_water(w, px_km2=0.01, min_km2=1.5, near=near)
    assert keep[6, 50] and not keep[41, 50]


def test_distance_to_selected_line_on_small_grid():
    g = gpd.GeoDataFrame({"DIS_AV_CMS": [500.0]}, geometry=gpd.GeoSeries.from_wkt(["LINESTRING(0 0, 0 900)"]))
    from rasterio import features
    from rasterio.transform import from_origin
    m = features.rasterize(select_main(g, 100).geometry, out_shape=(10, 10), transform=from_origin(0, 900, 90, 90),
                           all_touched=True) > 0
    d = distance_km(m, 90)
    assert d[5, 0] == 0 and d[5, 3] == pytest.approx(0.27)


def _osm(rows):
    return gpd.GeoDataFrame(rows, columns=["fclass", "name"], geometry=gpd.GeoSeries.from_wkt(["LINESTRING(0 0, 0 1000)"] * len(rows)))


def test_select_main_osm_matches_exact_names_only_and_strips():
    from build_river_features_doi_chung import select_main_osm
    g = _osm([("river", "Mississippi River"), ("river", "Mississippi River "), ("river", "Mississippi River Gulf Outlet"),
              ("canal", "Mississippi River"), ("river", "Old Pearl River"), ("river", "Pearl River"),
              ("river", None), ("stream", "Pearl River")])
    sel, names = select_main_osm(g, "mississippi")
    assert sel.index.tolist() == [0, 1, 5] and names.tolist() == ["Mississippi River"] * 2 + ["Pearl River"]


def test_select_main_osm_nile_arabic_names():
    from build_river_features_doi_chung import select_main_osm
    g = _osm([("river", "فرع رشيد"), ("river", "فرع دمياط"), ("river", "Bahr el Libeini"), ("canal", "نهر النيل"),
              ("river", "نهر النيل")])
    assert select_main_osm(g, "nile")[0].index.tolist() == [0, 1, 4]


def test_main_regex_equals_pattern_agreed_with_an():
    from build_river_features_doi_chung import main_regex
    assert main_regex("mississippi").pattern == (
        "^(Mississippi River|Lower Atchafalaya River|Atchafalaya River|Wax Lake Outlet|Southwest Pass|South Pass|"
        "Pass a Loutre|Main Pass|Bayou Lafourche|Bayou Teche|Bayou Terrebonne|Vermilion River|Mermentau River|"
        "Calcasieu River|Sabine River|Pearl River|West Pearl River|Red River)$")
    assert main_regex("nile").pattern == "^(فرع رشيد|فرع دمياط|نهر النيل|Rosetta Branch|Damietta Branch|Nile)$"

"""Luoi tai doi chung: goc khop boi so do phan giai, phu het ranh gioi."""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_doi_chung as f  # noqa: E402


def test_grid_of_snap_and_cover(tmp_path):
    gpd.GeoDataFrame(geometry=[box(-91.0, 29.0, -90.5, 29.4)], crs=4326).to_file(tmp_path / "boundary.geojson", driver="GeoJSON")
    c = f.Ctx(argparse.Namespace(region="mississippi", root=str(tmp_path.parent), project=None, threads=1))
    c.boundary = str(tmp_path / "boundary.geojson")
    for res in (30, 10):
        b, x0, y1, w, h = f.grid_of(c, res)
        assert x0 % res == 0 and y1 % res == 0
        bx0, by0, bx1, by1 = b.bounds
        assert x0 <= bx0 and x0 + w * res >= bx1 and y1 >= by1 and y1 - h * res <= by0


def test_grid_of_buffer_grows_by_pad(tmp_path):
    gpd.GeoDataFrame(geometry=[box(-91.0, 29.0, -90.5, 29.4)], crs=4326).to_file(tmp_path / "boundary.geojson", driver="GeoJSON")
    c = f.Ctx(argparse.Namespace(region="mississippi", root=str(tmp_path.parent), project=None, threads=1))
    c.boundary = str(tmp_path / "boundary.geojson")
    assert c.static_buffer_km == 0.0
    b0, b1 = f.grid_of(c, 10)[0], f.grid_of(c, 10, 26000)[0]
    assert b1.bounds[0] == __import__("pytest").approx(b0.bounds[0] - 25000, abs=10)
    assert b1.bounds[3] == __import__("pytest").approx(b0.bounds[3] + 25000, abs=10)


def test_regions_utm():
    assert f.REGIONS["mississippi"]["utm"] == 32615 and f.REGIONS["nile"]["utm"] == 32636


def test_match_sentinel_and_nan():
    import numpy as np
    assert f._match([-9999.0, 1.0], [np.nan, 1.0])        # sentinel GEE = NaN file
    assert f._match([np.nan], [np.nan])
    assert not f._match([-9999.0], [0.0])                 # khong bien sentinel thanh khop gia
    assert not f._match([1.0], [1.1])


def test_noaa_part_drops_error_line():
    ok = "Date Time, Water Level, Sigma, I, L\n2013-11-15 16:00,0.1,0.0,0,0\nError: No data was found. x,,,,\n"
    assert len(f._noaa_part(ok)) == 1
    assert f._noaa_part("Date Time, Water Level, Sigma, I, L\nError: No data was found.\n") is None
    assert f._noaa_part("") is None


def test_crms_table_window_boundary_flag(tmp_path):
    import pandas as pd
    from shapely.geometry import box
    cols = ["CPRA Station ID", "Date (mm/dd/yyyy)", "Time (hh:mm)", "Time Zone", "Measurement Depth (ft)", f.PORE, "Latitude", "Longitude"]
    rows = [["A", "10/31/2013", "10:00", "CST", 1.0, 5.0, 29.5, -90.5],    # truoc cua so
            ["A", "11/1/2013", "10:00", "CST", 1.0, 5.0, 29.5, -90.5],     # ok
            ["A", "4/30/2026", "23:00", "CST", 1.0, 70.0, 29.5, -90.5],    # trong cua so, >60
            ["A", "5/1/2026", "10:00", "CST", 1.0, 5.0, 29.5, -90.5],      # sau cua so
            ["B", "1/1/2020", "10:00", "CST", None, 5.0, 29.5, -90.5],     # thieu do sau
            ["C", "1/1/2020", "10:00", "CST", 1.0, 5.0, 40.0, -90.5],      # ngoai ranh gioi
            ["D", "1/1/2020", "10:00", "CST", 1.0, None, 29.5, -90.5]]     # khong co do man
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "x.csv", index=False)
    t = f.crms_table(tmp_path / "x.csv", box(-91, 29, -90, 30))
    assert list(t.station_id) == ["A", "A", "B"] and list(t["flag"]) == ["ok", "gt60ppt", "no_depth"]
    assert t.depth_cm.iloc[0] == 30.5


def test_scene_drop_thresholds():
    st = [dict(id="may", b3=0.30, b2=0.285), dict(id="samac", b3=0.30, b2=0.18), dict(id="toi", b3=0.20, b2=0.19),
          dict(id="thieu", b3=None, b2=None)]
    d = {x["id"]: x for x in f.scene_drop(st, 0.25, 0.85)}
    assert d["may"]["dropped"] and "b2/b3" in d["may"]["reason"]
    assert "samac" not in d and "toi" not in d
    assert d["thieu"]["insufficient"] and not d["thieu"]["dropped"]


def test_landsat_suffix_4_cases():
    assert f.landsat_suffix(None, False) == ""
    assert f.landsat_suffix({"a": 1}, False) == "_sf3"
    assert f.landsat_suffix(None, True) == "_snow"
    assert f.landsat_suffix({"a": 1}, True) == "_sf3_snow"
    assert f.landsat_suffix({"a": 1}, True, "_x") == "_x"


def test_scene_stats_khong_che_tuyet_du_mask_snow(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    calls, metas = [], []
    monkeypatch.setattr(f, "_oli_coll", lambda ee, y, g, s="l8", snow=False: calls.append(snow) or MagicMock())
    monkeypatch.setattr(f.gf, "_init", lambda p: MagicMock())
    monkeypatch.setattr(f.gf, "_retry", lambda fn, what: [{"id": "A", "b3": 0.3, "b2": 0.3, "n": 5000},
                                                           {"id": "B", "b3": 0.1, "b2": 0.08, "n": 5000}])
    monkeypatch.setattr(f, "grid_of", lambda c, r: (None,))
    monkeypatch.setattr(f, "clip_geom", lambda ee, c, b: None)
    monkeypatch.setattr(f, "landsat_img", lambda *a, **k: MagicMock())
    monkeypatch.setattr(f, "export_tiles", lambda ee, c, mk, outs, r, t, what, meta: (metas.append(meta), mk(None), calls.append(("outs", [o[0] for o in outs]))))
    c = MagicMock(project=None)
    c.p = lambda *a: "/".join(a)
    f.cmd_landsat(c, seasons=[2020], scene_filter=dict(max_b3=0.25, min_b2b3=0.85), mask_snow=True)
    assert calls[0] is False                     # scene_stats khong che bit 5
    m = metas[0]
    assert m["mask_snow"] is True and m["scene_stats_mask"] == "bit0-4"
    assert [x["id"] for x in m["scenes_dropped"]] == ["A"]
    assert [x["id"] for x in m["scenes_stats"]] == ["A", "B"] and m["scenes_stats"][0]["n_px"] == 5000
    assert all(o.endswith("_l8_sf3_snow.tif") for o in calls[-1][1])

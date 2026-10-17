"""CHG-29: thong ke mua tram MRC (ranh gioi mua, ngay it lan doc), trong so tram doc song (noi suy giua thuong/ha luu,
thieu mot phia -> gan nhat doc song, manh khong tram -> duong thang), gan cot vao bang (NaN ngoai scope, trong dai tram)."""
import importlib.util
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

import river_graph as rg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture(scope="module")
def bt():
    spec = importlib.util.spec_from_file_location("build_tram_features",
                                                  os.path.join(ROOT, "scripts", "build_tram_features.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------- thong ke mua ----------------
def _readings(day, values):
    return pd.DataFrame({"datetime": [f"{day}T{h:02d}:00:00.0000000+07:00" for h in range(len(values))],
                         "value": values})


def test_thong_ke_mua_ranh_gioi_va_ngay_it_lan_doc(bt):
    df = pd.concat([
        _readings("2019-10-31", [9, 9, 9, 9]),            # mua 2019 -> khong thuoc 2020
        _readings("2019-11-01", [0.0, 1.0, 2.0, 1.0]),    # mid 1, rng 2, max 2
        _readings("2020-02-10", [1.0, 3.0, 2.0, 2.0]),    # mid 2, rng 2, max 3
        _readings("2020-03-01", [5.0]),                    # 1 lan doc -> bo (rng 0 gia)
        _readings("2020-04-29", [-1.0, 3.0, 1.0, 0.0]),   # mid 1, rng 4, max 3 (29/04 thuoc mua)
        _readings("2020-04-30", [9, 9, 9, 9]),            # 30/04 khong thuoc mua
    ])
    st = bt.station_season_stats(df, min_reads=4, ok_frac=0.01, years=[2020]).iloc[0]
    assert st["n_days"] == 3 and bool(st["ok"])
    assert st["rng_mean"] == pytest.approx(8 / 3)
    assert st["mid_p20"] == pytest.approx(np.quantile([1, 2, 1], 0.2))
    assert st["max_p90"] == pytest.approx(np.quantile([2, 3, 3], 0.9))
    one = bt.station_season_stats(df, min_reads=1, ok_frac=0.01, years=[2020]).iloc[0]
    assert one["n_days"] == 4 and one["rng_mean"] == pytest.approx(8 / 4)   # ngay 1 lan doc keo bien do xuong
    few = bt.station_season_stats(df, min_reads=4, ok_frac=0.5, years=[2020]).iloc[0]
    assert not bool(few["ok"]) and np.isnan(few["rng_mean"])                # thieu ngay -> NaN, khong dien 0


# ---------------- trong so tram doc song ----------------
def _graph():
    # A: song chinh x = 0 tu y = 100 km xuong y = 1 km (cua, bo y = 0); B: nhanh tu (0, 50 km) toi cua (20 km, 1 km)
    # (dai hon -> phan lon B chay ve cua rieng); C: song rieng khong noi (x = 60 km)
    lines = gpd.GeoDataFrame({"river": ["A", "B", "C"], "geometry": [
        LineString([(0, 100_000), (0, 50_000), (0, 1000)]), LineString([(0, 50_000), (20_000, 1000)]),
        LineString([(60_000, 40_000), (60_000, 1000)])]}, crs=32648)
    g = rg.build_graph(lines, spacing=1000)
    mouths, _ = rg.find_mouths(g, LineString([(-50_000, 0), (100_000, 0)]), tol=5000, excluded=())
    dist, lab = rg.mouth_distances(g, mouths)
    return g, dist


def _v(g, x, y):
    return int(np.argmin(np.hypot(g.xy[:, 0] - x, g.xy[:, 1] - y)))


def test_trong_so_noi_suy_gan_nhat_va_duong_thang():
    g, dist = _graph()
    st_xy = np.array([[0, 85_000], [0, 20_000], [500, 99_000]])      # U, D tren A; T3 gan dau nguon A
    st_v = np.array([_v(g, *p) for p in st_xy[:2]] + [_v(g, 0, 99_000)])
    W, td, kind = rg.station_weights(g, dist, st_v, st_xy)
    assert np.allclose(W.sum(1), 1.0)
    v = _v(g, 0, 65_000)                                              # giua U (85) va D (20): du 20, dd 45
    assert kind[v] == 0 and W[v, 0] == pytest.approx(45 / 65) and W[v, 1] == pytest.approx(20 / 65)
    assert td[v] == pytest.approx(20_000)
    v = _v(g, 0, 50_000)                                              # nga ba: du 35, dd 30
    assert kind[v] == 0 and W[v, 0] == pytest.approx(30 / 65)
    v = st_v[1]                                                       # dinh trung tram D -> chi D
    assert kind[v] == 0 and W[v, 1] == 1.0 and td[v] == 0.0
    v = _v(g, 0, 10_000)                                              # duoi D, khong co tram ha luu
    assert kind[v] == 1 and W[v, 1] == 1.0 and td[v] == pytest.approx(10_000)
    v = _v(g, 15_000, 13_250)                                         # tren B qua duong phan thuy: chi gan nhat
    assert kind[v] == 1 and W[v, 1] == 1.0                            # D (30 + ~41 km) gan hon U (35 + ~41)
    d_b = np.hypot(15_000, 50_000 - 13_250)
    assert td[v] == pytest.approx(30_000 + d_b, rel=0.01)
    v = _v(g, 60_000, 20_000)                                         # song C khong noi -> duong thang
    assert kind[v] == 2 and W[v, 1] == 1.0 and td[v] == pytest.approx(60_000, rel=1e-3)


def test_tram_sau_duong_phan_thuy_van_noi_suy():
    """Nhu Tien duoi Vam Nao: tu U duong ngan nhat ra bien di nhanh tat E-H, nen tren A co duong phan thuy giua U va D.
    Dinh tren A giua hai tram van phai noi suy U-D (ban dinh huong canh theo dist tung roi ve 'gan nhat')."""
    lines = gpd.GeoDataFrame({"river": ["A", "E", "H"], "geometry": [
        LineString([(0, 100_000), (0, 80_000), (0, 60_000), (40_000, 60_000), (40_000, 1000)]),
        LineString([(0, 80_000), (-10_000, 80_000)]), LineString([(-10_000, 80_000), (-10_000, 1000)])]}, crs=32648)
    g = rg.build_graph(lines, spacing=1000)
    mouths, _ = rg.find_mouths(g, LineString([(-50_000, 0), (100_000, 0)]), tol=5000, excluded=())
    dist, _ = rg.mouth_distances(g, mouths)
    st_xy = np.array([[0, 90_000], [40_000, 20_000]])
    W, td, kind = rg.station_weights(g, dist, np.array([_v(g, *p) for p in st_xy]), st_xy)
    v = _v(g, 20_000, 60_000)                                         # cach U 50 km, cach D 60 km doc A
    assert kind[v] == 0 and W[v, 0] == pytest.approx(60 / 110) and W[v, 1] == pytest.approx(50 / 110)


# ---------------- gan cot vao bang ----------------
KEYS = ["Tan_Chau", "Chau_Doc", "Vam_Nao", "My_Thuan", "Can_Tho", "Vam_Kenh"]


def _stats():
    rows = []
    for s in (2020, 2021):
        for i, k in enumerate(KEYS):
            rows.append({"station": k, "season": s, "mid_p20": i + 0.1 * (s - 2020), "rng_mean": 1.0 + i,
                         "max_p90": 2.0 * i, "n_days": 170, "ok": True})
    return pd.DataFrame(rows)


def _wcell():
    w = np.zeros((3, 6))
    w[0, 0], w[0, 1] = 0.25, 0.75     # c1 noi suy Tan Chau / Chau Doc
    w[1, 5] = 1.0                     # c2 gan nhat Vam Kenh
    w[2] = np.nan                     # c3 khong co pixel scope
    d = pd.DataFrame(w, columns=[f"w_{k}" for k in KEYS])
    d.insert(0, "cell_id", ["c1", "c2", "c3"])
    d["tram_dist_km"] = [12.0, 80.0, np.nan]
    return d


def _table():
    return pd.DataFrame([{"cell_id": c, "season": s, "scope_frac": f, "dem_mean": 1.234567891}
                         for s in (2020, 2021) for c, f in (("c1", 0.9), ("c2", 0.2), ("c3", 0.0))])


def test_gan_cot_gia_tri_biet_truoc(bt):
    out = bt.add_tram_columns(_table(), _wcell(), _stats()).set_index(["cell_id", "season"])
    assert out.loc[("c1", 2020), "tram_wl_p20"] == pytest.approx(0.75)        # 0.25*0 + 0.75*1
    assert out.loc[("c1", 2021), "tram_wl_p20"] == pytest.approx(0.85)        # moi mua mot gia tri
    assert out.loc[("c2", 2021), "tram_tide_amp"] == pytest.approx(6.0)
    assert out.loc[("c2", 2020), "tram_wl_p90"] == pytest.approx(10.0)
    assert out.loc[("c1", 2020), "tram_dist_km"] == 12.0
    assert out.loc["c3", list(bt.NEW_COLS)].isna().all().all()                # ngoai scope -> NaN, khong dien 0
    pd.testing.assert_frame_equal(out.reset_index()[list(_table().columns)], _table())


@pytest.mark.parametrize("hong, khop", [
    (lambda t, w: (t.assign(scope_frac=0.5), w), "NaN o o co pham vi"),                 # c3 co scope ma khong trong so
    (lambda t, w: (pd.concat([t, t.iloc[[0]]]), w), "trung"),
    (lambda t, w: (t, w.assign(w_Vam_Kenh=np.where(w.cell_id == "c1", 2.0, w.w_Vam_Kenh))), "ngoai dai tram"),
    (lambda t, w: (t.assign(tram_wl_p20=0.0), w), "da co cot"),
])
def test_du_lieu_sai_la_loi(bt, hong, khop):
    t, w = hong(_table(), _wcell())
    with pytest.raises(ValueError, match=khop):
        bt.add_tram_columns(t, w, _stats())


def test_bo_tram_bang_b_mua_cong_4():
    """(d) tram = (c) b_mua + 4 cot tram; salinity / ndwi 26 dac trung; khong khop mau cam."""
    import run_experiments as rx
    from training.features import find_leak_columns, find_target_leak_columns, resolve_feature_list

    fs = os.path.join(ROOT, "configs", "feature_sets")
    lc = [f"landcover_class_{c}" for c in ("Trees", "Shrubland", "Grassland", "Cropland", "Built_up", "Bareland",
                                            "Water", "Wetland", "Mangroves")]
    new = ["tram_wl_p20", "tram_tide_amp", "tram_wl_p90", "tram_dist_km"]
    cols = ["rain_mm", "solar", "temp_c", "temp_max_c", "temp_min_c", "rh_percent", "dem_mean", "dist_main_river_km",
            "dist_any_water_km", "dist_coast_km", *lc, "tch_wl_p20c", "zos_mua_mean", "sluice_flag",
            "dist_mouth_river_km", "zos_mouth_p90", "sluice_frac", *new]
    for t in ("salinity", "ndwi"):
        kc = resolve_feature_list(cols, rx.feature_set_lists(os.path.join(fs, "b_mua.txt"), t)[1])[0]
        kd = resolve_feature_list(cols, rx.feature_set_lists(os.path.join(fs, "tram.txt"), t)[1])[0]
        assert len(kd) == 26 == len(kc) + 4 and set(kd) - set(kc) == set(new), t
        assert not find_leak_columns(new) and not find_target_leak_columns(new, t)

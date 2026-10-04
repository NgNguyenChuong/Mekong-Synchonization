"""Do thi song CHG-16 (src/river_graph.py): cua = dau mut, noi cho ho, cat cong, NFC, khoang cach doc song."""
import os
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

import river_graph as rg


def _lines(geoms, rivers):
    return gpd.GeoDataFrame({"river": rivers, "geometry": geoms}, crs=32648)


def _coast_y0():
    return LineString([(-50_000, 0), (50_000, 0)])  # bo bien la truc y = 0, dat o y > 0


def test_nfc_match_decomposed_name():
    nfd = unicodedata.normalize("NFD", "Sông Vàm Cỏ Tây")
    osm = gpd.GeoDataFrame({"fclass": ["river", "river", "canal"], "name": [nfd, None, "Sông Vàm Cỏ Tây"],
                            "geometry": [LineString([(0, 0), (1, 1)])] * 3}, crs=32648)
    sel = rg.select_graph_lines(osm, rg.graph_pattern(r"^Sông (Vàm Cỏ Tây)\b"))
    assert len(sel) == 1 and sel["river"].iloc[0] == "Vàm Cỏ Tây"  # NFD van khop; canal + None bi loai


def test_mouth_is_endpoint_not_parallel_river():
    # Song A vuong goc bo, dau mut cach bo 1 km. Song B chay song song bo o y = 2 km dai 20 km, hai dau mut
    # cach bo 2 km -> cung la dau mut <= 5 km (dung quy tac). Diem giua B KHONG duoc thanh cua.
    a = LineString([(0, 1000), (0, 30_000)])
    b = LineString([(10_000, 2000), (30_000, 2000)])
    g = rg.build_graph(_lines([a, b], ["A", "B"]), spacing=500)
    mouths, _ = rg.find_mouths(g, _coast_y0(), tol=5000, excluded=())
    assert len(mouths) == 3  # 1 cua A + 2 dau mut B, khong phai ~40 dinh doc B
    assert sorted(mouths["river"]) == ["A", "B", "B"]


def test_excluded_river_not_mouth():
    g = rg.build_graph(_lines([LineString([(0, 500), (0, 10_000)])], ["Ba Lai"]), spacing=500)
    mouths, excl = rg.find_mouths(g, _coast_y0(), tol=5000)
    assert mouths.empty and excl["river"].tolist() == ["Ba Lai"]


def test_along_distance_adds_coast_leg_and_picks_nearest_mouth():
    # Song chinh doc x = 0 tu y = 1 km (cua, cach bo 1 km) toi y = 50 km; nhanh ngang tai y = 40 km ra cua
    # thu hai cach bo 3 km qua duong y=40 -> x=10 km -> xuong y = 3 km.
    main = LineString([(0, 1000), (0, 50_000)])
    br = LineString([(0, 40_000), (10_000, 40_000), (10_000, 3000)])
    g = rg.build_graph(_lines([main, br], ["M", "N"]), spacing=100)
    mouths, _ = rg.find_mouths(g, _coast_y0(), tol=5000)
    dist, lab = rg.mouth_distances(g, mouths)
    top = int(np.argmin(np.hypot(g.xy[:, 0] - 0, g.xy[:, 1] - 50_000)))
    assert dist[top] == pytest.approx(1000 + 49_000, abs=1)  # qua cua M (1 km + 49 km) < qua N (3+37+10+10)
    mid_n = int(np.argmin(np.hypot(g.xy[:, 0] - 10_000, g.xy[:, 1] - 10_000)))
    assert dist[mid_n] == pytest.approx(3000 + 7000, abs=1)
    assert mouths.set_index("mouth_id").loc[lab[mid_n], "river"] == "N"


def test_bridge_only_from_endpoint_to_other_component():
    a = LineString([(0, 1000), (0, 20_000)])
    b = LineString([(400, 10_000), (5000, 15_000)])  # dau mut (400, 10000) cach A 400 m
    c = LineString([(3000, 1000), (3000, 9000)])     # song song A cach 3 km: KHONG noi
    g0 = rg.build_graph(_lines([a, b, c], ["A", "B", "C"]), spacing=100)
    mouths, _ = rg.find_mouths(g0, _coast_y0(), tol=1500)
    g, joins = rg.bridge_gaps(g0, mouths["vertex"].tolist(), tol=1000)
    assert len(joins) == 1 and {joins["river_from"].iloc[0], joins["river_to"].iloc[0]} == {"A", "B"}
    assert joins["gap_m"].iloc[0] == pytest.approx(400, abs=60)
    comp = g.components()
    va = int(np.argmin(np.hypot(*(g.xy - [0, 15_000]).T)))
    vb = int(np.argmin(np.hypot(*(g.xy - [5000, 15_000]).T)))
    vc = int(np.argmin(np.hypot(*(g.xy - [3000, 5000]).T)))
    assert comp[va] == comp[vb] and comp[vc] != comp[va]


def test_sluice_cut_makes_upstream_unreachable():
    a = LineString([(0, 1000), (0, 30_000)])
    g = rg.build_graph(_lines([a], ["Cái Lớn"]), spacing=100)
    mouths, _ = rg.find_mouths(g, _coast_y0(), tol=5000)
    cut = rg.sluice_vertices(g, {"Cái Lớn": (30, 10_000)})
    v, d = cut["Cái Lớn"]
    assert d == pytest.approx(30, abs=1)
    dist_o, lab_o = rg.mouth_distances(g, mouths)
    dist_s, lab_s = rg.mouth_distances(g, mouths, drop_vertices=[v])
    up = g.xy[:, 1] > 10_100
    assert (lab_o[up] >= 0).all() and (lab_s[up] == -1).all() and np.isinf(dist_s[up]).all()
    down = g.xy[:, 1] < 9_900
    np.testing.assert_allclose(dist_s[down], dist_o[down])


def test_sluice_active_and_behind():
    assert not rg.sluice_active(2021) and rg.sluice_active(rg.SLUICE_FIRST_SEASON)
    assert rg.sluice_active(2021, first=rg.SLUICE_ALT_SEASON)
    lo, lc = np.array([0, 0, -1, 2]), np.array([0, -1, -1, -1])
    assert rg.behind_sluice(lo, lc).tolist() == [False, True, False, True]


def test_shared_endpoint_of_two_rivers_is_mouth():
    # Cua Lon va Dam Doi cung ket thuc tai (0, 2000) - bac 2 nhung la dau mut cua ca hai song -> mot cua.
    a = LineString([(0, 2000), (-20_000, 30_000)])
    b = LineString([(0, 2000), (20_000, 30_000)])
    g = rg.build_graph(_lines([a, b], ["Cửa Lớn", "Đầm Dơi"]), spacing=500)
    mouths, _ = rg.find_mouths(g, _coast_y0(), tol=5000, excluded=())
    assert mouths["river"].tolist() == ["Cửa Lớn/Đầm Dơi"]


def test_mouth_tolerance_exception_is_manual():
    a = LineString([(0, 10_000), (0, 40_000)])
    g = rg.build_graph(_lines([a], ["Bảy Háp"]), spacing=500)
    assert rg.find_mouths(g, _coast_y0(), tol=5000, exceptions={})[0].empty
    m, _ = rg.find_mouths(g, _coast_y0(), tol=5000, exceptions={"Bảy Háp": 12_000})
    assert len(m) == 1 and bool(m["manual"].iloc[0]) and m["dist_coast_m"].iloc[0] == pytest.approx(10_000)


def test_manual_join_same_river_only():
    a = LineString([(0, 1000), (0, 20_000)])
    b = LineString([(0, 25_000), (0, 40_000)])      # cung song A, ho 5 km
    c = LineString([(3000, 26_000), (3000, 30_000)])  # song khac gan hon - KHONG duoc chon
    g0 = rg.build_graph(_lines([a, b, c], ["A", "A", "C"]), spacing=100)
    g, tab = rg.manual_joins(g0, (("A", (0, 25_100)),), to_xy=lambda ll: np.asarray(ll, float))
    assert len(tab) == 1 and tab["river_to"].iloc[0] == "A"
    assert tab["gap_m"].iloc[0] == pytest.approx(5000, abs=60)
    mouths, _ = rg.find_mouths(g0, _coast_y0(), tol=5000, excluded=())
    dist, lab = rg.mouth_distances(g, mouths)
    top = int(np.argmin(np.hypot(*(g.xy - [0, 40_000]).T)))
    assert dist[top] == pytest.approx(1000 + 19_000 + 5000 + 15_000, abs=5)
    with pytest.raises(ValueError):
        rg.manual_joins(g0, (("A", (9000, 9000)),), to_xy=lambda ll: np.asarray(ll, float))


def test_invariant_holds_and_detects_violation():
    main = LineString([(0, 1000), (0, 50_000)])
    br = LineString([(0, 40_000), (10_000, 40_000), (10_000, 3000)])
    g = rg.build_graph(_lines([main, br], ["M", "N"]), spacing=100)
    mouths, _ = rg.find_mouths(g, _coast_y0(), tol=5000)
    dist, lab = rg.mouth_distances(g, mouths)
    assert rg.check_invariant(g, mouths, dist, lab) == (0, 0.0)
    bad = dist.copy()
    bad[lab >= 0] *= 0.5
    assert rg.check_invariant(g, mouths, bad, lab)[0] > 0


PLACES_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "KE_HOACH", "ket-qua",
                          "dot4_do_thi_diem_quen.csv")


COMPS_CSV = PLACES_CSV.replace("diem_quen", "manh")
# Khoang kiem (NHAT_KY 2026-10-04): Can Tho/Long Xuyen/Chau Doc sua SAU khi thay ket qua; Tra On dat TRUOC.
EXPECTED_KM = {"Can Tho": (70, 90), "Long Xuyen": (120, 160), "Chau Doc": (170, 230), "My Tho": (35, 75),
               "Tra On": (57, 77)}


@pytest.mark.skipif(not os.path.exists(PLACES_CSV), reason="chua chay build_river_graph.py")
def test_real_graph_places():
    t = pd.read_csv(PLACES_CSV).set_index("place")["along_km"]
    assert t["Chau Doc"] > t["Long Xuyen"] > t["Can Tho"] > t["Tra On"]
    for p, (lo, hi) in EXPECTED_KM.items():
        assert lo <= t[p] <= hi, p


@pytest.mark.skipif(not os.path.exists(COMPS_CSV), reason="chua chay build_river_graph.py")
def test_real_graph_components_all_reach_sea():
    # Sau noi tu dong + MANUAL_JOINS + ngoai le Bay Hap: moi manh deu co cua (An duyet 2026-10-04).
    # Neu OSM/quy tac doi lam xuat hien manh cut -> test do, phai liet ke va duyet lai.
    c = pd.read_csv(COMPS_CSV)
    assert c["reaches_sea"].all(), c.loc[~c["reaches_sea"], "rivers"].tolist()
    assert len(c) == 8

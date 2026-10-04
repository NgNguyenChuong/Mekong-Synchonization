"""Khoang cach raster + nhan dien mat nuoc dang dai (dac trung song T3-V1)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_river_features import MAIN_RIVERS, distance_km  # noqa: E402
from check_osm_vs_jrc import label_linear_water  # noqa: E402


def test_distance_km_from_vertical_line():
    m = np.zeros((5, 10), bool)
    m[:, 0] = True
    d = distance_km(m, res_m=90)
    assert d[2, 0] == 0 and d[2, 4] == pytest.approx(0.36)


def test_distance_km_empty_mask_raises():
    with pytest.raises(ValueError):
        distance_km(np.zeros((3, 3), bool), 90)


def test_linear_water_keeps_canal_drops_pond():
    w = np.zeros((60, 60), bool)
    w[5, 2:58] = True          # kenh: 1 x 56 pixel
    w[30:40, 30:40] = True     # ao vuong 10 x 10
    lab, n, _, _, linear = label_linear_water(w, px_km2=0.0009, min_elong=8, min_km2=0.0)
    assert n == 2
    assert linear[lab[5, 10]] and not linear[lab[35, 35]]


@pytest.mark.parametrize("name,ok", [("Sông Tiền", True), ("Sông Hậu", True), ("Sông Cổ Chiên", True),
                                     ("Rạch Tiền", False), ("Sông Vàm Cỏ Đông", True), ("Sông Cái Tàu", False),
                                     ("Sông Gành Hào", True), ("Sông Ông Đốc", True), ("Sông Cửa Lớn", True),
                                     ("Sông Bảy Háp", False),
                                     ("Kênh Gành Hào - Hộ Phòng", False)])
def test_main_river_names(name, ok):
    assert bool(MAIN_RIVERS.match(name)) == ok

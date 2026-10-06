"""Kiem thu ham thuan E4 Dot 7 (quy tac buc xa 2020, ty le phuong sai trong o cua mua) - khong can du lieu that."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import analyze_dot7_label_rules as e4  # noqa: E402


def test_ty_le_trong_o_bien():
    # moi o hang so -> 0; mot o duy nhat -> 1
    assert e4.within_ss_ratio([1, 1, 5, 5], ["a", "a", "b", "b"]) == pytest.approx(0.0)
    assert e4.within_ss_ratio([1, 2, 3, 4], ["a"] * 4) == pytest.approx(1.0)
    # tay: TB tong 2,5; SS tong = 5; trong o a (1,2) 0,5 + o b (3,4) 0,5 = 1 -> 0,2
    assert e4.within_ss_ratio([1, 2, 3, 4], ["a", "a", "b", "b"]) == pytest.approx(0.2)


def test_ty_le_trong_o_loi():
    with pytest.raises(ValueError):
        e4.within_ss_ratio([1, np.nan, 3], ["a", "a", "b"])
    with pytest.raises(ValueError):
        e4.within_ss_ratio([2, 2, 2], ["a", "b", "b"])     # tong SS = 0
    with pytest.raises(ValueError):
        e4.within_ss_ratio([1, 2], ["a"])


def test_pearson_bo_nan_va_it_cap():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    b = a + rng.normal(scale=0.01, size=100)
    a2 = a.copy()
    a2[:5] = np.nan
    r, n = e4.pearson(a2, b)
    assert n == 95 and r > 0.99
    with pytest.raises(ValueError):
        e4.pearson(a[:10], b[:10])
    with pytest.raises(ValueError):
        e4.pearson(np.ones(50), b[:50])


def test_quyet_dinh_buc_xa():
    full = {s: 0.99 for s in e4.SEASONS if s != 2020}
    assert e4.radiation_decision(full) == "a"
    assert e4.radiation_decision({**full, 2017: 0.949}) == "b"        # mot mua duoi nguong -> (b)
    assert e4.radiation_decision({**full, 2017: 0.95}) == "a"         # bang nguong -> dat
    with pytest.raises(ValueError):
        e4.radiation_decision({k: v for k, v in full.items() if k != 2021})
    with pytest.raises(ValueError):
        e4.radiation_decision({**full, 2021: np.nan})


def test_quyet_dinh_mua():
    assert e4.rain_decision([0.01, 0.02, 0.06]) == (pytest.approx(0.02), "diem_tham_chieu")
    assert e4.rain_decision([0.05, 0.05, 0.01])[1] == "giu_muc_tho"   # trung vi 5% -> khong < 5%
    with pytest.raises(ValueError):
        e4.rain_decision([])
    with pytest.raises(ValueError):
        e4.rain_decision([0.1, np.nan])

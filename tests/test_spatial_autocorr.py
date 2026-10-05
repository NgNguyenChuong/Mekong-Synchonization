"""Tu tuong quan khong gian (src/spatial_autocorr.py): truong tong hop co tam biet truoc, quy tac vi pham, Moran."""
import numpy as np
import pytest

from spatial_autocorr import moran_band, variogram_range, violates


def _field(n=700, corr_len_m=10_000.0, seed=1, nugget=0.05):
    """Truong Gauss covariance mu exp(-d/a) -> tam hieu dung ~ 3a."""
    rng = np.random.default_rng(seed)
    xy = 500_000 + rng.uniform(0, 120_000, (n, 2))
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    cov = np.exp(-d / corr_len_m) + 1e-9 * np.eye(n)
    z = np.linalg.cholesky(cov) @ rng.normal(size=n)
    return xy, z + rng.normal(0, np.sqrt(nugget), n)


def test_tam_uoc_luong_dung_bac_do_lon():
    xy, z = _field(corr_len_m=8_000.0)
    r = variogram_range(xy, z)
    assert r["ok"] and 10_000 < r["range_m"] < 60_000  # ~24 km (3a), dung sai rong do mau huu han


def test_nhieu_trang_tam_nho_va_hang_so_bi_tinh_la_vuot():
    rng = np.random.default_rng(0)
    xy = 500_000 + rng.uniform(0, 120_000, (600, 2))
    r = variogram_range(xy, rng.normal(size=600))
    assert (not r["ok"]) or r["range_m"] < 50_000
    c = variogram_range(xy, np.ones(600))
    assert not c["ok"] and np.isinf(c["range_m"])  # khong fit duoc -> inf -> vuot


def test_toa_do_do_bi_tu_choi():
    with pytest.raises(ValueError, match="MET"):
        variogram_range(np.array([[105.0, 10.0]] * 40), np.arange(40.0))


def test_quy_tac_vi_pham():
    assert not violates([20] * 13)["violates"]
    assert violates([60] * 7 + [20] * 6)["violates"]                 # trung vi > 50
    assert violates([20] * 10 + [60, 70, 80])["violates"]           # 3 mua vuot
    assert not violates([20] * 11 + [60, 70])["violates"]           # 2 mua vuot: chap nhan
    v = violates([20] * 10 + [np.nan, np.inf, 30])
    assert v["n_over"] == 2 and not v["violates"]                   # NaN/inf tinh la vuot
    assert violates([20] * 10 + [np.nan, np.inf, np.inf])["violates"]


def test_moran_duong_va_dao_bi_loai():
    xy, z = _field(n=500, corr_len_m=15_000.0)
    xy = np.vstack([xy, [[900_000, 900_000]]])                     # mot diem dao cach xa
    z = np.append(z, 0.0)
    m = moran_band(xy, z, band=10_000.0, permutations=99)
    # diem dao co y + co the vai diem ngau nhien o mep khong co lan can trong 10 km
    assert m["n_islands"] >= 1 and m["n"] == 501 - m["n_islands"] and m["I"] > 0.2 and m["p_sim"] <= 0.05


def test_tam_cham_bien_maxlag_tinh_la_vuot():
    """Truong co xu huong tuyen tinh (khong dung) -> fit cham bien 150 km -> inf (vuot), khong 'ok'."""
    rng = np.random.default_rng(3)
    xy = 500_000 + rng.uniform(0, 200_000, (800, 2))
    v = (xy[:, 0] - 500_000) / 50_000 + rng.normal(0, 0.3, 800)
    r = variogram_range(xy, v)
    assert not r["ok"] and np.isinf(r["range_m"])


def test_bo_xu_huong_bac_2_theo_khoang_cach_bo():
    """Xu huong bac 2 theo khoang cach bo + truong cuc bo -> sau bo xu huong: tam huu han, nho; truoc: cham bien."""
    from spatial_autocorr import detrend_poly

    xy, z = _field(n=700, corr_len_m=4_000.0, seed=5)
    dist = (xy[:, 0] - 500_000) / 1000.0                       # "khoang cach toi bo" theo truc x (km)
    v = 0.002 * dist ** 2 - 0.1 * dist + 0.5 * z
    before = variogram_range(xy, v)
    after = variogram_range(xy, detrend_poly(v, dist, 2))
    assert (not before["ok"]) or before["range_m"] > 50_000
    assert after["ok"] and after["range_m"] < 50_000
    r = detrend_poly(v, dist, 2)
    assert abs(np.nanmean(r)) < 1e-9 and abs(np.corrcoef(r, dist)[0, 1]) < 1e-9

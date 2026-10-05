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


def test_nhieu_trang_tam_nho_va_hang_so_la_loi():
    """CHG-22 (An 2026-10-05 21:22): truoc day hang so -> inf -> 'vuot'; nay phuong sai 0 = LOI (raise), khong vuot."""
    rng = np.random.default_rng(0)
    xy = 500_000 + rng.uniform(0, 120_000, (600, 2))
    r = variogram_range(xy, rng.normal(size=600))
    assert (not r["ok"]) or r["range_m"] < 50_000
    with pytest.raises(ValueError, match="phuong sai 0"):
        variogram_range(xy, np.ones(600))


def test_it_diem_va_gia_tri_khong_huu_han_la_loi():
    """CHG-22: < 30 diem -> LOI; NaN/inf trong gia tri hoac toa do -> LOI (khong bo lang roi tinh tiep)."""
    rng = np.random.default_rng(0)
    xy = 500_000 + rng.uniform(0, 100_000, (100, 2))
    v = rng.normal(size=100)
    with pytest.raises(ValueError, match="< 30"):
        variogram_range(xy[:29], v[:29])
    for bad in (np.nan, np.inf, -np.inf):
        vb = v.copy()
        vb[5] = bad
        with pytest.raises(ValueError, match="1/100 diem"):
            variogram_range(xy, vb)
        with pytest.raises(ValueError, match="1/100 diem"):
            moran_band(xy, vb, permutations=9)
    xb = xy.copy()
    xb[7, 1] = np.nan
    with pytest.raises(ValueError, match="khong huu han"):
        variogram_range(xb, v)


def test_toa_do_do_bi_tu_choi():
    with pytest.raises(ValueError, match="MET"):
        variogram_range(np.array([[105.0, 10.0]] * 40), np.arange(40.0))


def test_quy_tac_vi_pham():
    assert not violates([20] * 13)["violates"]
    assert violates([60] * 7 + [20] * 6)["violates"]                 # trung vi > 50
    assert violates([20] * 10 + [60, 70, 80])["violates"]           # 3 mua vuot
    assert not violates([20] * 11 + [60, 70])["violates"]           # 2 mua vuot: chap nhan
    v = violates([20] * 10 + [60, np.inf, 30])
    assert v["n_over"] == 2 and not v["violates"]                   # +inf (vuot that) tinh la vuot
    assert violates([20] * 10 + [60, np.inf, np.inf])["violates"]


def test_quy_tac_vi_pham_loi_khong_quy_ve_dat():
    """CHG-22: mang rong (truoc: trung vi inf -> 'vi pham'), NaN (truoc: tinh la vuot), -inf, tam <= 0 -> LOI."""
    with pytest.raises(ValueError, match="khong co mua"):
        violates([])
    with pytest.raises(ValueError, match="NaN"):
        violates([20] * 12 + [np.nan])
    with pytest.raises(ValueError):
        violates([20] * 12 + [-np.inf])
    with pytest.raises(ValueError):
        violates([20] * 12 + [0.0])


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


def test_het_ram_khong_bi_tinh_la_vuot(monkeypatch):
    """3 trang thai (An 2026-10-05): MemoryError = LOI may -> noi len (dung), khong ghi 'vuot'.

    CHG-22: chi RuntimeError "khong hoi tu" cua curve_fit moi la vuot; RuntimeError khac -> LOI.
    """
    import skgstat

    rng = np.random.default_rng(0)
    xy = 500_000 + rng.uniform(0, 100_000, (100, 2))
    v = rng.normal(size=100)

    def het_ram(*a, **k):
        raise MemoryError("gia lap het RAM")

    def fit_loi(*a, **k):
        raise RuntimeError("Optimal parameters not found: gia lap fit khong hoi tu")

    monkeypatch.setattr(skgstat, "Variogram", het_ram)
    monkeypatch.setattr(skgstat, "DirectionalVariogram", het_ram)
    with pytest.raises(MemoryError):
        variogram_range(xy, v)
    with pytest.raises(MemoryError):
        variogram_range(xy, v, azimuth=45.0)
    monkeypatch.setattr(skgstat, "Variogram", fit_loi)
    r = variogram_range(xy, v)
    assert not r["ok"] and np.isinf(r["range_m"]) and r["reason"] == "fit khong hoi tu"

    def runtime_khac(*a, **k):
        raise RuntimeError("loi phan mem bat ky")

    monkeypatch.setattr(skgstat, "Variogram", runtime_khac)
    with pytest.raises(RuntimeError, match="phan mem"):
        variogram_range(xy, v)


def test_khong_hoi_tu_that_cua_scipy_la_vuot(monkeypatch):
    """Xac minh thong diep THAT cua scipy (khong gia lap): curve_fit het max_nfev -> RuntimeError
    'Optimal parameters not found' noi ra tu skgstat.Variogram -> tinh la VUOT (that bai thuc chat, CHG-22)."""
    import sys

    import skgstat  # noqa: F401

    mod = sys.modules["skgstat.Variogram"]
    goc = mod.curve_fit
    monkeypatch.setattr(mod, "curve_fit", lambda *a, **k: goc(*a, max_nfev=1, **k))
    rng = np.random.default_rng(0)
    xy = 500_000 + rng.uniform(0, 100_000, (100, 2))
    r = variogram_range(xy, rng.normal(size=100))
    assert not r["ok"] and np.isposinf(r["range_m"]) and r["reason"] == "fit khong hoi tu"


@pytest.mark.parametrize("exc", [TypeError("t"), AttributeError("a"), ValueError("v"),
                                 np.linalg.LinAlgError("suy bien"), FloatingPointError("f"), KeyError("k")])
def test_loi_phan_mem_so_hoc_la_loi_khong_phai_vuot(monkeypatch, exc):
    """CHG-22: truoc day 'except Exception' -> 'fit loi' -> vuot; nay moi ngoai le khac khong hoi tu -> noi len."""
    import skgstat

    def nem(*a, **k):
        raise exc

    monkeypatch.setattr(skgstat, "Variogram", nem)
    monkeypatch.setattr(skgstat, "DirectionalVariogram", nem)
    rng = np.random.default_rng(0)
    xy = 500_000 + rng.uniform(0, 100_000, (100, 2))
    v = rng.normal(size=100)
    with pytest.raises(type(exc)):
        variogram_range(xy, v)
    with pytest.raises(type(exc)):
        variogram_range(xy, v, azimuth=135.0)


@pytest.mark.parametrize("params, ket_qua", [
    ([np.nan, 1.0, 0.1], "loi"),            # tham so NaN -> LOI
    ([20_000.0, np.nan, 0.1], "loi"),
    ([20_000.0, 1.0, np.nan], "loi"),
    ([0.0, 1.0, 0.1], "loi"),               # tam <= 0 -> LOI
    ([-5.0, 1.0, 0.1], "loi"),
    ([np.inf, 1.0, 0.1], "vuot"),           # tam = +inf -> vuot
    ([149_900.0, 1.0, 0.1], "vuot"),        # cham bien 0,999*maxlag -> vuot (giu reason cu)
    ([149_800.0, 1.0, 0.1], "ok"),
    ([24_000.0, 1.0, 0.1], "ok"),
])
def test_phan_loai_tham_so_fit(monkeypatch, params, ket_qua):
    import skgstat

    class GiaVg:
        def __init__(self, *a, **k):
            self.parameters = params

    monkeypatch.setattr(skgstat, "Variogram", GiaVg)
    rng = np.random.default_rng(0)
    xy = 500_000 + rng.uniform(0, 100_000, (100, 2))
    v = rng.normal(size=100)
    if ket_qua == "loi":
        with pytest.raises(ValueError, match="LOI"):
            variogram_range(xy, v)
        return
    r = variogram_range(xy, v)
    if ket_qua == "vuot":
        assert not r["ok"] and np.isposinf(r["range_m"])
        if params[0] == 149_900.0:
            assert r["reason"] == "tam cham bien maxlag"
    else:
        assert r["ok"] and r["range_m"] == params[0] and r["reason"] == ""


def test_moran_khong_con_diem_nao_la_loi():
    """CHG-22: moi diem deu la dao (khong lan can trong dai) -> LOI, khong tra Moran rong."""
    xy = np.array([[500_000.0 + 50_000 * i, 1_000_000.0] for i in range(40)])
    with pytest.raises(ValueError, match="dao"):
        moran_band(xy, np.arange(40.0), band=10_000.0, permutations=9)

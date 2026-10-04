"""Gop thuy van theo mua kho (P4): ranh gioi mua, thieu != 0, frac_days nam nhuan, pixel ven bo,
cuc tri (p10, min7, p90), phan vi co xu ly chan (p20c) va quy tac _ok = False -> NaN."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_hydro_season import mrc_season_table, zos_season_table  # noqa: E402
from hydro_season import (  # noqa: E402
    censored_missing_days, censored_season_stats, regional_daily_mean, season_stats)


def _row(df, year):
    return df.set_index("season").loc[year]


def test_ranh_gioi_mua_chi_tinh_01_11_den_29_04():
    # 31/10 va 30/04 mang gia tri 100 -> neu bi tinh, trung binh se lech manh
    dates = ["2019-10-31", "2019-11-01", "2020-02-29", "2020-04-29", "2020-04-30"]
    vals = [100.0, 1.0, 2.0, 3.0, 100.0]
    r = _row(season_stats(dates, vals, [2020], "x"), 2020)
    assert r["x_mean"] == pytest.approx(2.0)
    assert r["x_n_days"] == 3
    assert r["x_frac_days"] == pytest.approx(3 / 181)


def test_thieu_khong_bi_coi_la_0():
    dates = pd.date_range("2020-11-01", periods=4)
    r = _row(season_stats(dates, [4.0, np.nan, np.inf, 2.0], [2021], "x"), 2021)
    assert r["x_mean"] == pytest.approx(3.0)       # khong phai (4+0+0+2)/4 = 1.5
    assert r["x_n_days"] == 2
    # Mua khong co ngay nao -> NaN, khong phai 0
    r0 = _row(season_stats(dates, [np.nan] * 4, [2021], "x"), 2021)
    assert np.isnan(r0["x_mean"]) and r0["x_n_days"] == 0 and not r0["x_ok"]


def test_frac_days_nam_nhuan_va_co_ok():
    d20 = pd.date_range("2019-11-01", "2020-04-29")   # co 29/02/2020
    d21 = pd.date_range("2020-11-01", "2021-04-29")
    assert len(d20) == 181 and len(d21) == 180
    t = season_stats(d20.append(d21), np.ones(len(d20) + len(d21)), [2020, 2021], "x")
    assert t["x_frac_days"].tolist() == [1.0, 1.0]
    assert t["x_n_days"].tolist() == [181, 180]
    # 144/181 = 0.796 -> co False; 145/181 = 0.801 -> True (mau so 181, khong phai 180)
    for n, ok in [(144, False), (145, True)]:
        v = np.r_[np.ones(n), np.full(181 - n, np.nan)]
        r = _row(season_stats(d20, v, [2020], "x"), 2020)
        assert r["x_frac_days"] == pytest.approx(n / 181) and bool(r["x_ok"]) is ok


def test_ngay_vang_mat_trong_file_tinh_la_thieu():
    d = pd.date_range("2020-11-01", periods=90)        # chi nua mua co dong
    r = _row(season_stats(d, np.ones(90), [2021], "x"), 2021)
    assert r["x_frac_days"] == pytest.approx(90 / 180) and not r["x_ok"]


def test_ngay_trung_bao_loi():
    with pytest.raises(ValueError, match="trung"):
        season_stats(["2020-11-01", "2020-11-01"], [1.0, 2.0], [2021], "x")


def test_mrc_season_table_tach_tram():
    df = pd.DataFrame({
        "station": ["TCH", "TCH", "CDO", "CDO"],
        "date": pd.to_datetime(["2020-11-01", "2020-04-30", "2020-11-01", "2021-04-29"]),
        "value": [2.0, 9.0, 1.0, np.nan],
    })
    # nan_if_not_ok=False: test nay chi kiem tach tram + ranh gioi mua (1 ngay -> _ok False)
    t = _row(mrc_season_table(df, [2021], nan_if_not_ok=False), 2021)
    assert t["tch_wl_mean"] == 2.0 and t["tch_wl_n_days"] == 1   # 30/04/2020 khong thuoc mua 2021
    assert t["cdo_wl_mean"] == 1.0 and t["cdo_wl_n_days"] == 1


def test_regional_daily_mean_ngay_thieu_pixel_la_nan():
    lat = np.array([0.0, 60.0])                         # cos = 1 va 0.5
    f = np.array([[[1.0, 1.0], [4.0, np.nan]],
                  [[1.0, 1.0], [np.nan, np.nan]]])       # ngay 2: pixel (1,0) thieu
    mask = np.array([[True, True], [True, False]])      # (1,1) = dat
    out = regional_daily_mean(f, lat, mask)
    assert out[0] == pytest.approx((1 + 1 + 0.5 * 4) / 2.5)
    assert np.isnan(out[1])


def test_zos_season_table_ven_bo_chi_lay_pixel_gan_ranh_gioi():
    lat = np.array([9.5])
    lon = np.array([105.6, 106.5])                      # ~11 km va ~110 km tu canh 105.5E
    times = pd.date_range("2019-10-31", "2020-04-30")
    z = np.empty((len(times), 1, 2))
    z[:, 0, 0], z[:, 0, 1] = 1.0, 3.0
    z[0], z[-1] = 50.0, 50.0                            # 31/10 va 30/04 khong duoc tinh
    tab, n_sea, n_coast = zos_season_table(times, z, lat, lon, box(105.0, 9.4, 105.5, 9.6), [2020], 30.0)
    r = _row(tab, 2020)
    assert (n_sea, n_coast) == (2, 1)
    assert r["zos_coast_mean"] == pytest.approx(1.0)
    assert r["zos_mean"] == pytest.approx(2.0)
    assert r["zos_n_days"] == 181 and r["zos_frac_days"] == 1.0 and r["zos_ok"]
    assert r["zos_coast_p90"] == pytest.approx(1.0)     # hang so 1.0; 50 o 31/10, 30/04 khong lot vao


# ---------------- Cuc tri ----------------

def _full_season(year):
    return pd.date_range(pd.Timestamp(year - 1, 11, 1), pd.Timestamp(year, 4, 29))


def test_p10_p90_gia_tri_biet_truoc_va_ngoai_mua_khong_tinh():
    d = _full_season(2021)                              # 180 ngay
    v = np.arange(1.0, 181.0)                           # 1..180
    # 31/10 va 30/04 mang -999 -> neu lot vao mua, p10 lech han
    dates = d.append(pd.DatetimeIndex(["2020-10-31", "2021-04-30"]))
    vals = np.r_[v, -999.0, -999.0]
    r = _row(season_stats(dates, vals, [2021], "x", quantiles={"p10": 0.10, "p90": 0.90}), 2021)
    assert r["x_p10"] == pytest.approx(18.9)            # 1 + 0.1 * 179 (noi suy tuyen tinh)
    assert r["x_p90"] == pytest.approx(162.1)


def test_p10_bo_qua_nan_khong_coi_la_0():
    d = _full_season(2021)
    v = np.full(len(d), 5.0)
    v[:30] = np.nan                                     # neu NaN -> 0 thi p10 = 0
    v[30:40] = 1.0
    r = _row(season_stats(d, v, [2021], "x", quantiles={"p10": 0.10}), 2021)
    assert r["x_p10"] == pytest.approx(np.quantile(v[30:], 0.10))
    assert r["x_p10"] > 0.9


def test_min7_cua_so_lien_tiep_gia_tri_biet_truoc():
    d = _full_season(2021)
    v = np.full(len(d), 10.0)
    v[50:57] = 1.0                                      # 7 ngay lien tiep = 1 -> min7 = 1
    v[100:103] = -50.0                                  # dot 3 ngay rat thap: (3*-50 + 4*10)/7 = -15.71
    r = _row(season_stats(d, v, [2021], "x", roll_min=7), 2021)
    assert r["x_min7"] == pytest.approx((3 * -50 + 4 * 10) / 7)
    v[100:103] = 10.0
    r = _row(season_stats(d, v, [2021], "x", roll_min=7), 2021)
    assert r["x_min7"] == pytest.approx(1.0)


def test_min7_cua_so_co_nan_bi_bo_khong_lap():
    d = _full_season(2021)
    v = np.full(len(d), 10.0)
    v[50:57] = 1.0
    v[53] = np.nan                                      # thung giua dot thap
    r = _row(season_stats(d, v, [2021], "x", roll_min=7), 2021)
    # Moi cua so chua ngay 53 bi bo; cua so du ngay thap nhat: [46..52] hoac [54..60] = (3*1 + 4*10)/7.
    # Neu lap NaN (noi suy/ffill/0) thi se ra <= 1.0
    assert r["x_min7"] == pytest.approx((3 * 1 + 4 * 10) / 7)


def test_min7_ngay_vang_mat_trong_file_va_khong_cua_so_nao_du():
    d = _full_season(2021)
    keep = d[::2]                                       # ngay xen ke -> khong co 7 ngay lien tiep
    r = _row(season_stats(keep, np.ones(len(keep)), [2021], "x", roll_min=7), 2021)
    assert np.isnan(r["x_min7"])
    # Dong vang mat trong file (khong phai NaN) cung lam hong cua so: bo dong ngay 53
    drop = d.delete(53)
    v = np.full(len(drop), 10.0)
    v[50:56] = 1.0                                      # theo lich: ngay 50..52 va 54..56 = 1
    r = _row(season_stats(drop, v, [2021], "x", roll_min=7), 2021)
    assert r["x_min7"] == pytest.approx((3 * 1 + 4 * 10) / 7)


def test_min7_khong_vuot_ranh_gioi_mua():
    d = pd.date_range("2020-10-25", "2021-05-05")
    v = np.full(len(d), 10.0)
    v[d < "2020-11-03"] = 0.0                           # 25/10..02/11 = 0
    v[d > "2021-04-27"] = 0.0                           # 28/04..05/05 = 0
    r = _row(season_stats(d, v, [2021], "x", roll_min=7), 2021)
    # Cua so phai nam tron trong mua: 01/11..07/11 hoac 23/04..29/04 = (2*0 + 5*10)/7
    assert r["x_min7"] == pytest.approx(50 / 7)


def test_ok_false_thi_mean_va_cuc_tri_nan_nhung_giu_n_days():
    d = _full_season(2021)[:40]                         # 40/180 ngay -> ok False
    kw = dict(quantiles={"p10": 0.1}, roll_min=7)
    r = _row(season_stats(d, np.arange(40.0), [2021], "x", nan_if_not_ok=True, **kw), 2021)
    assert not r["x_ok"] and r["x_n_days"] == 40 and r["x_frac_days"] == pytest.approx(40 / 180)
    assert np.isnan(r["x_mean"]) and np.isnan(r["x_p10"]) and np.isnan(r["x_min7"])
    r2 = _row(season_stats(d, np.arange(40.0), [2021], "x", **kw), 2021)   # mac dinh van tinh
    assert r2["x_mean"] == pytest.approx(19.5) and r2["x_min7"] == pytest.approx(3.0)


def test_mrc_season_table_mac_dinh_ok_false_thanh_nan_va_co_cot_cuc_tri():
    d21 = _full_season(2021)
    d22 = _full_season(2022)[:37]                       # giong MRC 2015: 37 ngay -> ok False
    v21 = np.linspace(2.0, 0.2, len(d21))
    tch = pd.DataFrame({"station": "TCH", "date": d21.append(d22), "value": np.r_[v21, np.full(37, 0.1)]})
    t = mrc_season_table(pd.concat([tch, tch.assign(station="CDO")]), [2021, 2022]).set_index("season")
    assert {"tch_wl_p20c", "tch_wl_n_censored", "tch_wl_days_le_0_3"} <= set(t.columns)
    # Cau B: bo p10, min7; tram CDO khong co cot chan
    assert not {"tch_wl_p10", "tch_wl_min7"} & set(t.columns)
    assert not [c for c in t.columns if c.startswith("cdo_wl_") and ("p20" in c or "censor" in c)]
    assert t.loc[2021, "tch_wl_p20c"] == pytest.approx(np.quantile(v21, 0.2, method="lower"))
    assert t.loc[2021, "tch_wl_n_censored"] == 0
    assert t.loc[2021, "tch_wl_days_le_0_3"] == (v21 <= 0.3).sum()
    for c in ["tch_wl_mean", "tch_wl_p20c", "tch_wl_days_le_0_3", "cdo_wl_mean"]:
        assert np.isnan(t.loc[2022, c]), c
    assert t.loc[2022, "tch_wl_n_days"] == 37 and not t.loc[2022, "tch_wl_ok"]


# ---------------- Phan vi co xu ly chan (cau B) ----------------

def _season_series(year, base=1.0):
    d = _full_season(year)
    return d, np.full(len(d), base)


def test_chan_ca_chuoi_o_0_moi_nam():
    d, v = _season_series(2016)                         # mua truoc 2021 cung bi chan
    v[:60] = -0.5                                       # 60/181 ngay am -> neu khong chan, p20 = -0.5
    v[60:] = np.linspace(0.4, 2.0, len(v) - 60)
    with pytest.warns(UserWarning, match="ty le ngay bi chan"):   # 60/181 = 33% >= 20%
        r = _row(censored_season_stats(d, v, [2016], "x"), 2016)
    assert r["x_p20c"] == 0.0
    assert r["x_n_censored"] == 60
    assert r["x_days_le_0_3"] == 60


def test_p20_ngay_thieu_bi_chan_la_duoi_moi_gia_tri():
    d, v = _season_series(2021)                         # 180 ngay
    v[:] = np.linspace(0.31, 2.0, 180)
    v[[9, 18, 30]] = 0.1                                # hai dau doan [10..17] < 0,3; doan [31..34] chi dau trai
    v[10:18] = np.nan                                   # 8 ngay -> bi chan
    v[31:35] = np.nan                                   # dau phai v[35] >= 0,3 -> KHONG chan
    r = _row(censored_season_stats(d, v, [2021], "x"), 2021)
    obs = v[np.isfinite(v)]
    x = np.r_[obs, np.full(8, -np.inf)]                 # 168 ngay do + 8 ngay chan = 176
    assert len(x) == 176
    expect = np.quantile(x, 0.2, method="lower")
    assert r["x_p20c"] == pytest.approx(expect)
    # Neu bo qua ngay chan (chi ngay do) -> phan vi cao hon
    assert expect < np.quantile(obs, 0.2, method="lower")
    assert r["x_n_censored"] == 8
    assert r["x_days_le_0_3"] == 8 + 3                  # 8 ngay chan + 3 ngay do 0,1


def test_doan_nan_dai_hon_8_ngay_khong_bi_chan():
    v = np.r_[0.1, np.full(9, np.nan), 0.1, np.ones(20)]
    cens = censored_missing_days(pd.date_range("2020-12-01", periods=len(v)), v)
    assert not cens.any()
    v8 = np.r_[0.1, np.full(8, np.nan), 0.1, np.ones(20)]
    cens8 = censored_missing_days(pd.date_range("2020-12-01", periods=len(v8)), v8)
    assert cens8.sum() == 8 and cens8.iloc[1:9].all()


def test_hai_dau_tu_0_3_tro_len_khong_bi_chan():
    dates = pd.date_range("2020-12-01", periods=12)
    for left, right, expect in [(0.3, 0.1, 0), (0.1, 0.3, 0), (0.29, 0.29, 3), (0.5, 0.5, 0)]:
        v = np.r_[np.ones(4), left, np.nan, np.nan, np.nan, right, np.ones(3)]
        assert censored_missing_days(dates, v).sum() == expect, (left, right)


def test_ngay_vang_mat_trong_file_cung_la_doan_thieu_va_doan_cham_mep_khong_chan():
    dates = pd.date_range("2020-12-01", periods=20)
    v = np.full(20, 0.1)
    keep = np.ones(20, bool)
    keep[5:8] = False                                   # 3 dong vang mat (khong phai NaN)
    cens = censored_missing_days(dates[keep], v[keep])
    assert cens.sum() == 3 and cens.loc["2020-12-06":"2020-12-08"].all()
    # Doan o dau chuoi (khong co dau trai) -> khong chan
    v2 = np.r_[np.nan, np.nan, np.full(18, 0.1)]
    assert not censored_missing_days(dates, v2).any()


def test_doan_cat_ranh_gioi_mua_dung_dau_thuc_ngoai_mua():
    # Doan 24/04..29/04 (6 ngay); dau phai la 30/04 (ngoai mua) = 0,1
    d = pd.date_range("2020-11-01", "2021-05-05")
    v = np.linspace(0.31, 2.0, len(d))
    v[(d >= "2021-04-24") & (d <= "2021-04-29")] = np.nan
    v[d == "2021-04-23"] = 0.1
    v[d == "2021-04-30"] = 0.1
    r = _row(censored_season_stats(d, v, [2021], "x"), 2021)
    assert r["x_n_censored"] == 6
    r_in = _row(censored_season_stats(d, v, [2021], "x", runs_within_season=True), 2021)
    assert r_in["x_n_censored"] == 0                    # trong cua so mua: thieu dau phai -> khong chan
    assert r["x_p20c"] < r_in["x_p20c"]


def test_ngoai_mua_khong_vao_p20c_va_ok_false_thanh_nan():
    d = pd.date_range("2020-10-31", "2021-04-30")
    v = np.linspace(0.5, 2.0, len(d))
    v[0] = v[-1] = -9.0                                 # 31/10, 30/04 -> neu lot vao, n_censored > 0
    r = _row(censored_season_stats(d, v, [2021], "x"), 2021)
    assert r["x_n_censored"] == 0
    assert r["x_p20c"] == pytest.approx(np.quantile(v[1:-1], 0.2, method="lower"))
    short = _full_season(2021)[:37]
    r2 = _row(censored_season_stats(short, np.full(37, 0.1), [2021], "x"), 2021)
    assert np.isnan(r2["x_p20c"]) and np.isnan(r2["x_days_le_0_3"]) and r2["x_n_censored"] == 0   # 0,1 > 0: khong bi chan


def test_zos_ok_false_thi_ca_cot_ven_bo_nan():
    lat, lon = np.array([9.5]), np.array([105.6])
    times = pd.date_range("2020-11-01", periods=60)     # 60/180 ngay
    z = np.ones((len(times), 1, 1))
    tab, _, _ = zos_season_table(times, z, lat, lon, box(105.0, 9.4, 105.5, 9.6), [2021], 30.0)
    r = _row(tab, 2021)
    assert not r["zos_ok"] and r["zos_n_days"] == 60
    assert np.isnan(r["zos_mean"]) and np.isnan(r["zos_coast_mean"]) and np.isnan(r["zos_coast_p90"])

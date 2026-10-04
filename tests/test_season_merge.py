"""Chong tai phat: gop dac trung theo mua kho va ghep nhan chinh xac theo (cell_id, season).

Du lieu TONG HOP voi gia tri biet truoc, khong phai du lieu that.
Loi da gap (soat 2026-10-03): merge_asof +-3 ngay nhan mot nhan mua thanh toi 7 dong (R3);
PERIODIC_MERGED.csv cu bi doc du spec rong (R2).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import dataset
from dataset import aggregate_dry_season, merge_season_labels, merge_with_salinity_labels
from seasons import season_days, season_of, season_window


def _daily():
    # c1: ranh gioi mua 2020 = 01/11/2019 .. 29/04/2020
    return pd.DataFrame({
        "cell_id": ["c1"] * 5 + ["c2"] * 2,
        "date": ["2019-10-31", "2019-11-01", "2020-01-10", "2020-04-29", "2020-04-30",
                 "2019-12-01", "2020-02-01"],
        "rain_mm": [100.0, 2.0, -4e-5, 3.0, 50.0, np.nan, np.nan],
        "temp_c": [10.0, 20.0, np.nan, 30.0, 99.0, 26.0, 28.0],
    })


def test_gop_mua_thieu_ngay_tong_nan_trung_binh_giu():
    # DOI DAC TA (soat 2026-10-03 muc 3): truoc day c1 rain_mm = 5,0 (tong 3 ngay co so lieu) va
    # n_days chung = 3. Nay: tong mua chi co khi du 181 ngay, thieu -> NaN + canh bao; dem ngay
    # theo TUNG cot (`<cot>_n_days`), bo n_days chung.
    with pytest.warns(UserWarning, match=r"rain_mm.*\(c1, 2020: 3/181 ngay\)"):
        out = aggregate_dry_season(_daily(), sum_cols=["rain_mm"], mean_cols=["temp_c"])
    assert list(out.columns) == ["cell_id", "season", "rain_mm", "temp_c", "rain_mm_n_days", "temp_c_n_days"]
    assert len(out) == 2 and out["season"].tolist() == [2020, 2020]
    c1 = out[out["cell_id"] == "c1"].iloc[0]
    assert np.isnan(c1["rain_mm"])
    # 31/10 va 30/04 khong tinh: rain co 3 ngay (2, -4e-5, 3), temp co 2 ngay (20, 30; 10/01 NaN)
    assert c1["rain_mm_n_days"] == 3 and c1["temp_c_n_days"] == 2
    # trung binh bo qua NaN, khong tinh 10 (31/10) va 99 (30/04) - giu nhu cu
    assert c1["temp_c"] == pytest.approx(25.0)
    c2 = out[out["cell_id"] == "c2"].iloc[0]
    # mua toan NaN -> NaN, KHONG duoc thanh 0 mm
    assert np.isnan(c2["rain_mm"]) and c2["rain_mm_n_days"] == 0
    assert c2["temp_c"] == pytest.approx(27.0) and c2["temp_c_n_days"] == 2


def test_gop_mua_du_ngay_ranh_gioi_va_cat_am():
    # Mua 2020 du 181 ngay, moi ngay 1 mm; 31/10 (100) va 30/04 (50) nam ngoai mua; 1 ngay -4e-5 -> 0.
    df = pd.DataFrame({"cell_id": "c1", "date": pd.date_range("2019-10-31", "2020-04-30", freq="D")})
    df["rain_mm"] = 1.0
    df.loc[df["date"] == "2019-10-31", "rain_mm"] = 100.0
    df.loc[df["date"] == "2020-04-30", "rain_mm"] = 50.0
    df.loc[df["date"] == "2020-01-10", "rain_mm"] = -4e-5
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # du ngay -> khong canh bao
        out = aggregate_dry_season(df, sum_cols=["rain_mm"], mean_cols=[])
    r = out.iloc[0]
    assert r["season"] == 2020 and r["rain_mm_n_days"] == 181
    assert r["rain_mm"] == pytest.approx(180.0, abs=0)


def test_gop_mua_60_ngay_mua_nan_thanh_nan_va_canh_bao():
    # Probe soat 2026-10-03: 60 ngay mua NaN -> truoc day rain = 121 (thay vi 181), n_days = 181.
    daily = _season_daily()
    in_season = (daily["date"] >= "2019-12-01") & (daily["date"] <= "2020-01-29")
    assert in_season.sum() == 60
    daily.loc[in_season, "rain_mm"] = np.nan
    with pytest.warns(UserWarning, match=r"\(c1, 2020: 121/181 ngay\)"):
        out = aggregate_dry_season(daily, sum_cols=["rain_mm"], mean_cols=["temp_c"])
    r = out.iloc[0]
    assert np.isnan(r["rain_mm"]) and r["rain_mm_n_days"] == 121
    assert r["temp_c"] == pytest.approx(27.0) and r["temp_c_n_days"] == 181


def test_gop_mua_du_lieu_bat_dau_01_01_mua_dau_nan():
    # DYNAMIC bat dau 01/01/2014: mua 2014 (01/11/2013..29/04/2014) chi co 119 ngay -> tong NaN;
    # mua 2015 du 180 ngay -> co tong. Truoc day mua 2014 rain = 119 khong canh bao.
    daily = _season_daily(start="2014-01-01", end="2015-04-29")
    with pytest.warns(UserWarning, match=r"\(c1, 2014: 119/180 ngay\)"):
        out = aggregate_dry_season(daily, sum_cols=["rain_mm"], mean_cols=["temp_c"]).set_index("season")
    assert np.isnan(out.loc[2014, "rain_mm"]) and out.loc[2014, "rain_mm_n_days"] == 119
    assert out.loc[2015, "rain_mm"] == pytest.approx(180.0) and out.loc[2015, "rain_mm_n_days"] == 180
    assert out.loc[2014, "temp_c"] == pytest.approx(27.0)


def test_gop_mua_cot_chua_khai_bao_bao_loi():
    df = _daily().assign(overlap_frac=1.0)
    with pytest.raises(ValueError, match="overlap_frac"):
        aggregate_dry_season(df, sum_cols=["rain_mm"], mean_cols=["temp_c"])


def test_gop_mua_trung_khoa_ngay_bao_loi():
    df = pd.concat([_daily(), _daily().iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="Trung khoa"):
        aggregate_dry_season(df, sum_cols=["rain_mm"], mean_cols=["temp_c"])


def test_gop_mua_cot_khai_bao_ca_hai_kieu_bao_loi():
    with pytest.raises(ValueError, match="vua khai bao"):
        aggregate_dry_season(_daily(), sum_cols=["rain_mm"], mean_cols=["rain_mm", "temp_c"])


def _season_daily(cell="c1", start="2019-10-20", end="2020-05-10"):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"cell_id": cell, "date": dates.strftime("%Y-%m-%d"),
                         "rain_mm": 1.0, "temp_c": 27.0})


def test_mot_nhan_mua_dung_mot_dong_sau_ghep():
    daily = _season_daily()
    # Cach cu: nhan mua gan ngay 15/01 -> merge_asof +-3 ngay gan nhan cho 7 dong ngay.
    old_lab = pd.DataFrame({"cell_id": ["c1"], "date": ["2020-01-15"], "salinity": [2.5]})
    with pytest.deprecated_call():
        old = merge_with_salinity_labels(daily, old_lab, tolerance_days=3)
    assert int(old["salinity"].notna().sum()) == 7

    feats = aggregate_dry_season(daily, sum_cols=["rain_mm"], mean_cols=["temp_c"])
    labels = pd.DataFrame({"cell_id": ["c1"], "season": [2020], "salinity": [2.5]})
    new = merge_season_labels(feats, labels, target_col="salinity")
    assert len(new) == 1 and new["salinity"].tolist() == [2.5]
    # 01/11/2019 .. 29/04/2020 = 181 ngay (2020 nhuan)
    assert new["rain_mm_n_days"].tolist() == [181]
    assert new["rain_mm"].tolist() == [181.0]


def test_ghep_nhan_trung_khoa_bao_loi():
    feats = pd.DataFrame({"cell_id": ["c1"], "season": [2020], "rain_mm": [5.0]})
    labels = pd.DataFrame({"cell_id": ["c1", "c1"], "season": [2020, 2020], "salinity": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Nhan trung khoa"):
        merge_season_labels(feats, labels, target_col="salinity")


def test_ghep_dac_trung_chua_gop_bao_loi():
    feats = pd.DataFrame({"cell_id": ["c1", "c1"], "season": [2020, 2020], "rain_mm": [1.0, 2.0]})
    labels = pd.DataFrame({"cell_id": ["c1"], "season": [2020], "salinity": [1.0]})
    with pytest.raises(ValueError, match="Dac trung trung khoa"):
        merge_season_labels(feats, labels, target_col="salinity")


def test_ghep_giu_o_khong_nhan_la_nan_va_khop_kieu_khoa():
    feats = pd.DataFrame({"cell_id": [1, 2], "season": pd.array([2020, 2020], dtype="Int64"), "rain_mm": [1.0, 2.0]})
    labels = pd.DataFrame({"cell_id": ["1", "1"], "season": [2020, 2021], "salinity": [3.0, 9.0]})
    out = merge_season_labels(feats, labels, target_col="salinity")
    assert len(out) == 2
    assert out.set_index("cell_id")["salinity"].to_dict()["1"] == 3.0
    assert np.isnan(out.set_index("cell_id")["salinity"].to_dict()["2"])


def test_build_feature_dataset_tu_choi_periodic_khi_spec_rong(tmp_path, monkeypatch, capsys):
    pd.DataFrame({"cell_id": ["c1"], "date": ["2020-01-01"], "rain_mm": [1.0]}).to_csv(
        tmp_path / dataset.DYNAMIC_MERGE_FILE, index=False)
    pd.DataFrame({"cell_id": ["c1"], "date": ["2020-01-01"], "ndvi": [0.8]}).to_csv(
        tmp_path / dataset.PERIODIC_MERGED_FILE, index=False)
    monkeypatch.setattr(dataset, "DATA_PROCESSED", str(tmp_path))

    monkeypatch.setattr(dataset, "PERIODIC_SPECS", {})
    out = dataset.build_feature_dataset()
    assert "ndvi" not in out.columns
    assert "Bo qua" in capsys.readouterr().err

    # Doi chung: spec khai bao thi moi doc (quyet dinh theo spec, khong theo file con sot)
    monkeypatch.setattr(dataset, "PERIODIC_SPECS", {"x": {}})
    assert "ndvi" in dataset.build_feature_dataset().columns


def test_build_season_feature_dataset_gop_va_ghep_static(tmp_path, monkeypatch):
    _season_daily("c1").to_csv(tmp_path / dataset.DYNAMIC_MERGE_FILE, index=False)
    pd.DataFrame({"cell_id": ["c1"], "dem_mean": [1.2]}).to_csv(tmp_path / dataset.STATIC_MERGED_FILE, index=False)
    monkeypatch.setattr(dataset, "DATA_PROCESSED", str(tmp_path))
    monkeypatch.setattr(dataset, "PERIODIC_SPECS", {})
    out = dataset.build_season_feature_dataset()
    assert len(out) == 1
    assert out.iloc[0][["season", "rain_mm", "temp_c", "dem_mean"]].tolist() == [2020, 181.0, 27.0, 1.2]


def test_season_agg_spec_mua_la_tong():
    sums, means = dataset.season_agg_spec({"rain": {"col_name": "rain_mm"}, "t": {"col_name": "temp_c"},
                                           "wl": {"col_name": "wl_m", "season_agg": "mean"}})
    assert sums == ["rain_mm"] and means == ["temp_c", "wl_m"]


# ---------- ranh gioi mua kho (cu: test_seasons.py)

def test_ranh_gioi_mua_kho():
    s = season_of(["2019-10-31", "2019-11-01", "2019-12-31", "2020-01-01", "2020-04-29", "2020-04-30", "2020-07-15"])
    assert s.tolist() == [pd.NA, 2020, 2020, 2020, 2020, pd.NA, pd.NA]


def test_nam_nhuan_va_so_ngay():
    assert season_of(["2020-02-29"]).tolist() == [2020]
    # 01/11/2019 .. 29/04/2020: 30 + 31 + 31 + 29 + 31 + 29 = 181 ngay (2020 nhuan)
    assert season_days(2020) == 181
    assert season_days(2021) == 180
    assert season_window(2020) == (pd.Timestamp("2019-11-01"), pd.Timestamp("2020-04-29"))

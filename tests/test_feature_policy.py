"""Chong tai phat R1 (soat 2026-10-03): dac trung theo danh sach CHO PHEP + kiem mau cot ro ri.

Truoc day select_feature_columns la danh sach loai tru -> salinity_median, NDWIchen, is_holdout,
fold, water_freq, n_land_px, season deu thanh dac trung.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from training.features import (
    FORBIDDEN_PATTERNS,
    assert_no_leak_columns,
    find_leak_columns,
    read_feature_list,
    resolve_feature_list,
    select_feature_columns,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECK_FEATURES = os.path.join(ROOT, ".claude", "skills", "method-review", "scripts", "check_features.py")

LEAKY = ["salinity_median", "NDWIchen", "is_holdout", "fold", "water_freq", "n_land_px", "season"]


def _table():
    n = 4
    return pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(n)],
        "salinity_median": np.arange(n, dtype=float),
        "NDWIchen": np.linspace(0, 0.3, n),
        "is_holdout": [True, False, False, True],
        "fold": [0, 1, 2, 3],
        "water_freq": [0.0, 10.0, 60.0, 5.0],
        "n_land_px": [100, 90, 20, 80],
        "season": [2020] * n,
        "dem_mean": [1.0, 1.5, 0.8, 2.0],
        "rain_mm": [10.0, 20.0, 0.0, 5.0],
        "landcover_class_Cropland": [0.5, 0.2, 0.1, 0.9],
    })


def test_assert_no_leak_columns_bao_du_7_cot():
    df = _table()
    bad = find_leak_columns(df.columns)
    assert sorted(c for c, _ in bad) == sorted(LEAKY)
    with pytest.raises(ValueError) as exc:
        assert_no_leak_columns(df.columns)
    for c in LEAKY:
        assert c in str(exc.value)
    assert_no_leak_columns(["dem_mean", "rain_mm", "dist_coast_km", "landcover_class_Water"])


def test_select_feature_columns_chi_lay_danh_sach_cho_phep():
    assert select_feature_columns(_table()) == ["dem_mean", "rain_mm", "landcover_class_Cropland"]
    assert select_feature_columns(_table(), allowed=["rain_mm"]) == ["rain_mm"]


def test_select_feature_columns_cho_phep_cot_cam_van_bi_chan():
    # Nguoi dung tu dua NDWIchen vao --features -> van loi (tru khi allow_leak co y)
    with pytest.raises(ValueError, match="NDWIchen"):
        select_feature_columns(_table(), allowed=["rain_mm", "NDWIchen"])
    assert select_feature_columns(_table(), allowed=["rain_mm", "NDWIchen"], allow_leak=["NDWIchen"]) == [
        "NDWIchen", "rain_mm"]


def test_cot_chat_luong_khong_vao_dac_trung():
    df = pd.DataFrame({"zos_coast_p90": [1.0, 1.1], "zos_n_days": [180, 170], "zos_ok": [True, False],
                       "tch_wl_frac_days": [1.0, 0.7], "n_days": [181, 181], "rain_mm": [1.0, 2.0],
                       "tch_wl_p20c": [0.2, 0.3], "tch_wl_n_censored": [21, 0], "tch_wl_days_le_0_3": [42.0, 20.0],
                       "zos_mean": [0.9, 0.95], "zos_coast_mean": [0.9, 0.95], "tch_wl_mean": [1.0, 1.2],
                       "cdo_wl_mean": [1.1, 1.2], "zos_mouth_p90": [1.0, 1.1], "dist_mouth_river_km": [5.0, 9.0],
                       "sluice_frac": [0.0, 0.3], "sluice_frac_from2021": [0.0, 0.3], "graph_lateral_km": [1.0, 2.0],
                       "zos_cmems_far_frac": [0.0, 0.0]})
    # Bo chinh: tch_wl_p20c (cau B) + hybrid CHG-16; zos_coast_p90 = ban doi chung (khong mac dinh);
    # mean/cdo/n_censored/days_le va cot kiem hybrid khong vao mac dinh.
    assert select_feature_columns(df) == ["rain_mm", "tch_wl_p20c", "zos_mouth_p90", "dist_mouth_river_km",
                                          "sluice_frac"]


def test_mau_cam_cot_chat_luong():
    # Muc 5b soat 2026-10-03: truoc day --features n_days / overlap_frac lot qua kiem mau cam.
    bad = [c for c, _ in find_leak_columns(
        ["n_days", "rain_mm_n_days", "tch_wl_frac_days", "zos_ok", "overlap_frac",
         "zos_mean", "zos_coast_mean", "tch_wl_mean", "rain_mm", "landcover_class_Water"])]
    assert bad == ["n_days", "rain_mm_n_days", "tch_wl_frac_days", "zos_ok", "overlap_frac"]
    df = pd.DataFrame({"rain_mm": [1.0, 2.0], "n_days": [181, 120], "overlap_frac": [1.0, 0.3]})
    for col in ("n_days", "overlap_frac"):
        with pytest.raises(ValueError, match=col):
            select_feature_columns(df, allowed=["rain_mm", col])


def test_giai_tien_to_thanh_ten_day_du():
    cols = ["cell_id", "rain_mm", "landcover_class_Cropland", "dem_mean", "landcover_class_Water"]
    chosen, missing = resolve_feature_list(cols, ["dem_mean", "landcover_class_*", "rain_mm", "solar"])
    assert chosen == ["rain_mm", "landcover_class_Cropland", "dem_mean", "landcover_class_Water"]
    assert missing == ["solar"]
    _, missing = resolve_feature_list(["rain_mm"], ["rain_mm", "landcover_class_*"])
    assert missing == ["landcover_class_*"]


@pytest.mark.parametrize("entry", ["*", "rain*", "dist_*", "landcover_*", "landcover_class_*_x", "*_mean"])
def test_tien_to_chua_khai_bao_bi_tu_choi(entry):
    # Muc 5a: '*' don le truoc day nhan moi cot (ke ca cot chat luong); tien to la chi qua FEATURE_PREFIXES.
    with pytest.raises(ValueError, match="khong hop le"):
        select_feature_columns(_table(), allowed=["rain_mm", entry])


def test_cot_yeu_cau_tuong_minh_ma_thieu_bao_loi():
    # Muc 2: truoc day chi in "bo qua" -> bo dac trung khac nhau giua luoi ma khong biet.
    with pytest.raises(ValueError, match="solar"):
        select_feature_columns(_table(), allowed=["rain_mm", "solar"], require_all=True)
    with pytest.raises(ValueError, match="landcover_class_"):
        select_feature_columns(_table().drop(columns=["landcover_class_Cropland"]),
                               allowed=["rain_mm", "landcover_class_*"], require_all=True)
    # Danh sach mac dinh (require_all=False): muc thieu khong loi (train.py ghi features_missing)
    assert select_feature_columns(_table(), allowed=["rain_mm", "solar"]) == ["rain_mm"]


def test_select_feature_columns_khong_co_cot_nao_bao_loi():
    with pytest.raises(ValueError, match="Khong co cot"):
        select_feature_columns(_table(), allowed=["temp_c"])


def test_read_feature_list(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("# dac trung\nrain_mm\n\ndem_mean  # DEM\nlandcover_class_*\n", encoding="utf-8")
    assert read_feature_list(str(p)) == ["rain_mm", "dem_mean", "landcover_class_*"]


@pytest.mark.skipif(not os.path.exists(CHECK_FEATURES), reason="skill method-review khong co trong ban sao nay")
def test_mau_cam_dong_bo_voi_check_features():
    spec = importlib.util.spec_from_file_location("check_features", CHECK_FEATURES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert FORBIDDEN_PATTERNS == mod.FORBIDDEN


@pytest.mark.skipif(not os.path.exists(CHECK_FEATURES), reason="skill method-review khong co trong ban sao nay")
def test_mau_cam_theo_bien_dong_bo_voi_check_features():
    from training.features import TARGET_FORBIDDEN

    spec = importlib.util.spec_from_file_location("check_features", CHECK_FEATURES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert TARGET_FORBIDDEN == mod.TARGET_FORBIDDEN


# ---------------- Dot 7 (d): cam theo bien muc tieu (cung dai luong vat ly) ----------------
UNIFIED_FEATURES = ["rain_mm", "solar", "temp_c", "temp_max_c", "temp_min_c", "rh_percent", "dem_mean",
                    "dist_main_river_km", "dist_any_water_km", "dist_coast_km", "landcover_class_Trees",
                    "landcover_class_Water", "tch_wl_p20c", "dist_mouth_river_km", "zos_mouth_p90", "sluice_frac"]


@pytest.mark.parametrize("target, removed", [
    ("salinity", []),
    ("ndwi", []),
    ("rain_chirps", ["rain_mm"]),
    ("dsr_mcd18", ["solar"]),
    ("t2m_era5", ["temp_c", "temp_max_c", "temp_min_c", "rh_percent"]),   # RH = f(T, Td) cung pixel (An quyet)
    ("rh_era5", ["temp_c", "temp_max_c", "temp_min_c", "rh_percent"]),
])
def test_cam_theo_bien_tren_cot_bang_hop_nhat(target, removed):
    from training.features import drop_target_forbidden, find_target_leak_columns

    assert [c for c, _ in find_target_leak_columns(UNIFIED_FEATURES, target)] == removed
    keep, dropped = drop_target_forbidden(list(UNIFIED_FEATURES) + ["landcover_class_*"], target)
    assert dropped == removed and "landcover_class_*" in keep


def test_cam_theo_bien_bat_ten_ngoai_bang_hien_co():
    from training.features import find_target_leak_columns

    def hit(cols, t):
        return [c for c, _ in find_target_leak_columns(cols, t)]
    assert hit(["NDVI_dry", "sr_b5", "swir1", "landsat_b7", "salinity_lag1", "ec_dsm"], "ndwi") == \
        ["NDVI_dry", "sr_b5", "swir1", "landsat_b7", "salinity_lag1", "ec_dsm"]
    assert hit(["precip_era5", "n_rain_days", "rain7_max"], "rain_chirps") == ["precip_era5", "n_rain_days", "rain7_max"]
    assert hit(["ssrd_mean", "cloud_frac", "dist_coast_km"], "dsr_mcd18") == ["ssrd_mean", "cloud_frac"]
    assert hit(["skin_temperature", "td_mean", "dewpoint_c", "rh_percent"], "t2m_era5") == \
        ["skin_temperature", "td_mean", "dewpoint_c", "rh_percent"]
    assert hit(["d2m", "vpd_kpa", "specific_humidity", "dem_mean"], "rh_era5") == ["d2m", "vpd_kpa", "specific_humidity"]


def test_select_feature_columns_tu_choi_cot_cam_theo_bien():
    df = pd.DataFrame({"rain_mm": [1.0, 2.0], "dem_mean": [1.0, 2.0]})
    assert select_feature_columns(df, allowed=["rain_mm", "dem_mean"]) == ["rain_mm", "dem_mean"]   # do man: nhu cu
    with pytest.raises(ValueError, match="rain_chirps"):
        select_feature_columns(df, allowed=["rain_mm", "dem_mean"], target="rain_chirps")
    assert select_feature_columns(df, allowed=["dem_mean"], target="rain_chirps") == ["dem_mean"]


def test_cam_theo_bien_khong_bat_nham_cot_chat_luong():
    """'rain' khong duoc bat 'train_ok*' (chong tai phat: mau 'rain' tran tung khop train_ok_scope)."""
    from training.features import find_target_leak_columns

    cols = ["train_ok", "train_ok_scope", "train_ok_10pct", "era5_cover_frac", "scope_frac", "dist_coast_km"]
    for t in ("rain_chirps", "dsr_mcd18", "t2m_era5", "rh_era5", "ndwi"):
        assert find_target_leak_columns(cols, t) == []


# ---------------- soat method-reviewer 2026-10-07 muc 2-3: mau moi + chong khop nham ----------------
@pytest.mark.parametrize("target, cols", [
    ("ndwi", ["ndmi_dry", "s2_ndii", "msi", "gvmi_mean", "nmdi"]),
    ("rain_chirps", ["chirps_sum", "imerg_rain", "gsmap_mm"]),
    ("dsr_mcd18", ["mcd18_dsr", "ghi", "era5_ghi_mean", "sw_down", "swdown_mean", "insolation", "sunshine_h"]),
    ("t2m_era5", ["lst_day", "modis_lst", "tas", "tasmax", "chelsa_tas", "tmean", "tmax_c", "tmin",
                  "temperature", "era5_temp_mean", "rh_percent", "rh", "humid_mean", "hurs"]),
    ("rh_era5", ["hurs", "chelsa_hurs", "lst_night", "tas", "tmax", "temp_c", "rh_percent"]),
])
def test_mau_cam_theo_bien_moi_bat_ten(target, cols):
    from training.features import find_target_leak_columns

    assert [c for c, _ in find_target_leak_columns(cols, target)] == cols


def test_mau_cam_chung_bat_chi_so_am_chua_nir():
    """NDMI/NDII (= NDWI Gao), MSI, GVMI, NMDI chua NIR -> cam ca voi do man (mau chung)."""
    bad = [c for c, _ in find_leak_columns(["ndmi_dry", "ndii", "msi", "gvmi", "nmdi_mean", "dem_mean"])]
    assert bad == ["ndmi_dry", "ndii", "msi", "gvmi", "nmdi_mean"]


@pytest.mark.parametrize("target", ["ndwi", "rain_chirps", "dsr_mcd18", "t2m_era5", "rh_era5"])
def test_mau_cam_theo_bien_khong_khop_nham(target):
    """Ranh gioi (^|_): 'temp' khong bat attempt/temporal_*; 'tas'/'lst'/'ghi'/'msi'/'tmax' khong bat cot khac."""
    from training.features import find_target_leak_columns

    innocent = ["attempt", "n_attempts", "temporal_lag", "temporary_flag", "task_id", "fantasy", "last_obs",
                "list_id", "high_tide", "neighbor_km", "emsi", "dist_coast_km", "dem_mean", "landcover_class_Trees",
                "tch_wl_p20c", "zos_mouth_p90", "sluice_frac", "dist_mouth_river_km", "train_ok_scope",
                "scope_frac", "era5_cover_frac", "lbl_cover_frac", "thurs", "atmax"]
    assert find_target_leak_columns(innocent, target) == []


def test_mau_cam_tren_moi_cot_bang_hien_co_chi_khop_dung_y():
    """Toan bo ten cot hien co (13 bang goc + 65 bang Dot 7 + configs/feature_sets, thu 2026-10-07): moi bien chi
    khop dung tap du kien; mau chung quang hoc chi bat 'ndwi'."""
    from training.features import find_target_leak_columns

    cols = ["cell_id", "dem_mean", "dist_any_water_km", "dist_coast_km", "dist_main_river_km", "dist_mouth_river_km",
            "dsr_mcd18", "era5_cover_frac", "era5_fill_frac", "graph_lateral_km", "landcover_class_Bareland",
            "landcover_class_Built_up", "landcover_class_Cropland", "landcover_class_Grassland",
            "landcover_class_Mangroves", "landcover_class_Shrubland", "landcover_class_Trees", "landcover_class_Water",
            "landcover_class_Wetland", "lbl_cover_frac", "lbl_ok", "lbl_valid_px", "n_valid_px", "ndwi",
            "rain_chirps", "rain_mm", "rh_era5", "rh_percent", "salinity", "scope_cx", "scope_cy", "scope_frac",
            "scope_n_px", "season", "sluice_frac", "sluice_frac_from2021", "solar", "t2m_era5", "tch_wl_ok",
            "tch_wl_p20c", "temp_c", "temp_max_c", "temp_min_c", "train_ok", "train_ok_10pct", "train_ok_scope",
            "train_ok_scope_sal", "valid_frac", "zos_cmems_far_frac", "zos_coast_p90", "zos_mouth_p90", "zos_ok"]
    temp_rh = ["rh_era5", "rh_percent", "t2m_era5", "temp_c", "temp_max_c", "temp_min_c"]
    want = {"ndwi": ["ndwi", "salinity"], "rain_chirps": ["rain_chirps", "rain_mm"],
            "dsr_mcd18": ["dsr_mcd18", "solar"], "t2m_era5": temp_rh, "rh_era5": temp_rh}
    for t, w in want.items():
        assert [c for c, _ in find_target_leak_columns(cols, t)] == w
    optical = [c for c, why in find_leak_columns(cols) if why == FORBIDDEN_PATTERNS[0][1]]
    assert optical == ["ndwi"]

"""Bang hop nhat (src/unified_table.py): khoa phai trung tuyet doi, khong dien, khong trung ten cot."""
import os

import numpy as np
import pandas as pd
import pytest

from unified_table import QUALITY_COLS, TRAIN_COL, finite_or_nan, label_suffix, merge_sources, nan_report
from settings import data_path  # (.env: DATA_ROOT)


def _src(cells=("a", "b"), seasons=(2019, 2020)):
    keys = pd.DataFrame([(c, s) for c in cells for s in seasons], columns=["cell_id", "season"])
    labels = keys.assign(salinity=[0.5, np.nan, 1.0, 2.0][: len(keys)], n_valid_px=100.0, valid_frac=0.5,
                         train_ok=[True, False, True, True][: len(keys)], train_ok_10pct=True)
    era5 = keys.assign(rain_mm=[0.0, 10.0, 5.0, np.nan][: len(keys)], era5_cover_frac=1.0, era5_fill_frac=0.0)
    static = pd.DataFrame({"cell_id": list(cells), "dem_mean": [1.0, np.nan], "scope_frac": [0.5, 0.0]})
    hydro = pd.DataFrame({"season": list(seasons), "tch_wl_p20c": [0.2, np.nan], "zos_coast_p90": [1.0, 1.1],
                          "tch_wl_ok": [True, False], "zos_ok": True, "tch_wl_mean": [9.0, 9.0]})
    hybrid = keys.assign(dist_mouth_river_km=[10.0, 10.0, 20.0, 20.0][: len(keys)], zos_mouth_p90=1.0,
                         sluice_frac=0.0, graph_lateral_km=1.0)
    return labels, era5, static, hydro, hybrid


def test_ghep_dung_gia_tri_khong_dien_va_thu_tu_cot():
    t = merge_sources(*_src())
    assert len(t) == 4 and list(t.columns[:3]) == ["cell_id", "season", "salinity"]
    r = t.set_index(["cell_id", "season"])
    assert r.loc[("a", 2020), "tch_wl_p20c"] != r.loc[("a", 2020), "tch_wl_p20c"]  # NaN giu nguyen
    assert r.loc[("a", 2019), "rain_mm"] == 0.0 and np.isnan(r.loc[("b", 2020), "rain_mm"])
    assert np.isnan(r.loc[("b", 2019), "dem_mean"]) and r.loc[("b", 2019), "dist_mouth_river_km"] == 20.0
    assert "tch_wl_mean" not in t.columns  # chi lay cot thuy van khai bao
    q = [c for c in t.columns if c in QUALITY_COLS]
    assert list(t.columns[3:3 + len(q)]) == q  # cot chat luong ngay sau nhan


def test_tap_khoa_lech_bao_loi():
    labels, era5, static, hydro, hybrid = _src()
    with pytest.raises(ValueError, match="Tap khoa"):
        merge_sources(labels, era5.iloc[:3], static, hydro, hybrid)
    with pytest.raises(ValueError, match="Tap khoa"):
        merge_sources(labels, era5, static, hydro, hybrid.assign(season=hybrid["season"] + 1))
    with pytest.raises(ValueError, match="tinh: thieu"):
        merge_sources(labels, era5, static.iloc[:1], hydro, hybrid)
    with pytest.raises(ValueError, match="thuy van: thieu mua"):
        merge_sources(labels, era5, static, hydro.iloc[:1], hybrid)


def test_trung_khoa_va_trung_ten_cot_bao_loi():
    labels, era5, static, hydro, hybrid = _src()
    with pytest.raises(ValueError, match="trung khoa"):
        merge_sources(pd.concat([labels, labels.iloc[:1]]), era5, static, hydro, hybrid)
    with pytest.raises(ValueError, match="Cot trung ten"):
        merge_sources(labels, era5, static, hydro, hybrid.assign(rain_mm=1.0))


def test_cell_id_so_va_chuoi_khop():
    labels, era5, static, hydro, hybrid = _src(cells=("101", "102"))
    era5["cell_id"] = era5["cell_id"].astype(int)  # doc CSV khong dtype -> so
    assert len(merge_sources(labels, era5, static, hydro, hybrid)) == 4


def test_phu_tro():
    assert label_suffix("") == "" and label_suffix("keepwater") == "_keepwater"
    with pytest.raises(ValueError):
        label_suffix("keepurban")
    assert finite_or_nan(pd.Series([1.0, np.nan])) and not finite_or_nan(pd.Series([1.0, -np.inf]))
    t = merge_sources(*_src())
    nr = nan_report(t, ["rain_mm", "dem_mean"], mask_col="train_ok").set_index("column")
    assert nr.loc["rain_mm", "nan_all"] == 1 and nr.loc["rain_mm", "nan_train_ok"] == 1
    assert nr.loc["dem_mean", "nan_all"] == 2 and nr.loc["dem_mean", "nan_train_ok"] == 2


def test_tap_huan_luyen_loai_o_scope_0():
    # o "b" scope_frac = 0 (toan nuoc/rung ngap man): train_ok True nhung KHONG vao tap huan luyen.
    t = merge_sources(*_src()).set_index(["cell_id", "season"])
    assert bool(t.loc[("b", 2019), "train_ok"]) and not bool(t.loc[("b", 2019), TRAIN_COL])
    assert bool(t.loc[("a", 2019), TRAIN_COL]) and not bool(t.loc[("a", 2020), TRAIN_COL])  # train_ok False
    assert TRAIN_COL in QUALITY_COLS


def test_cot_tap_huan_luyen_bi_chan_khoi_dac_trung():
    from training.features import find_leak_columns
    assert find_leak_columns([TRAIN_COL])


UNIFIED_DIR = data_path("features/unified")
REPORT = os.path.join(os.path.dirname(__file__), "..", "KE_HOACH", "ket-qua", "dot4_bang_hop_nhat.csv")


@pytest.mark.skipif(not (os.path.isdir(UNIFIED_DIR) and os.path.exists(REPORT)),
                    reason="khong co bang hop nhat that")
def test_bang_that_tap_huan_luyen_khop_bao_cao_va_khong_toan_nan():
    rep = pd.read_csv(REPORT, keep_default_na=False)
    main = rep[rep["label_set"] == "chinh"]
    assert len(main) == 13
    for r in main.itertuples():
        t = pd.read_csv(os.path.join(UNIFIED_DIR, f"{r.grid}_unified.csv"),
                        usecols=["salinity", "train_ok", "scope_frac", "scope_n_px", "scope_cx", TRAIN_COL, "dem_mean",
                                 "dist_main_river_km", "dist_any_water_km", "dist_coast_km"])
        m = t[TRAIN_COL].astype(bool)
        assert int(m.sum()) == r.rows_train_ok_scope, r.grid
        # CHG-17: train_ok VA scope_frac > 0 VA co >= 1 tam pixel dat trong o
        assert (m == (t["train_ok"].astype(bool) & (t["scope_frac"] > 0) & (t["scope_n_px"] > 0))).all(), r.grid
        assert t.loc[m, "scope_cx"].notna().all() and (t.loc[m, "scope_cx"] > 1000).all(), r.grid  # met UTM
        assert t.loc[m, "salinity"].notna().all(), r.grid
        static = ["dem_mean", "dist_main_river_km", "dist_any_water_km", "dist_coast_km"]
        assert not t.loc[m, static].isna().all(axis=1).any(), r.grid



def test_o_manh_dat_duoi_1_pixel_khong_vao_tap_huan_luyen():
    labels, era5, static, hydro, hybrid = _src()
    static = static.assign(scope_frac=[0.5, 1e-4], scope_n_px=[100, 0], dem_mean=[1.0, 2.0])
    t = merge_sources(labels, era5, static, hydro, hybrid).set_index(["cell_id", "season"])
    assert not bool(t.loc[("b", 2019), TRAIN_COL]) and bool(t.loc[("a", 2019), TRAIN_COL])


def test_bay_train_ok_nan_va_chuoi_false():
    """pandas astype(bool): NaN -> True, "False" -> True. strict_bool phai bao loi / doc dung (An 2026-10-04)."""
    from unified_table import strict_bool

    assert strict_bool(pd.Series(["False", "True", "false", "1", "0"]), "x").tolist() == [False, True, False, True, False]
    with pytest.raises(ValueError, match="NaN"):
        strict_bool(pd.Series([True, np.nan], dtype=object), "train_ok")
    with pytest.raises(ValueError, match="khong phai True/False"):
        strict_bool(pd.Series(["yes"]), "train_ok")
    labels, era5, static, hydro, hybrid = _src()
    labels = labels.assign(train_ok=["False", "False", "True", "False"])  # CSV doc thanh chuoi
    t = merge_sources(labels, era5, static.assign(scope_frac=[0.5, 0.5]), hydro, hybrid)
    assert t[TRAIN_COL].tolist() == [False, False, True, False]


def test_scope_frac_nan_la_loi_khong_loai_lang():
    """CHG-22: scope_frac NaN -> (NaN > 0) = False -> truoc day dong bi loai lang khoi tap huan luyen."""
    labels, era5, static, hydro, hybrid = _src()
    with pytest.raises(ValueError, match="scope_frac NaN"):
        merge_sources(labels, era5, static.assign(scope_frac=[0.5, np.nan]), hydro, hybrid)


def test_build_nan_dong_huan_luyen_ngoai_danh_sach_cho_phep():
    """CHG-22: NaN trong dong huan luyen chi duoc o tch_wl_p20c mua 2015 va dem_mean; con lai (vd ERA5 thieu ca
    mua) -> LOI khi dung bang."""
    import importlib.util

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    spec = importlib.util.spec_from_file_location("build_unified_table",
                                                  os.path.join(root, "scripts", "build_unified_table.py"))
    bu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bu)
    t = pd.DataFrame({"season": [2015, 2016, 2016, 2017], "tch_wl_p20c": [np.nan, 1.0, 1.0, 1.0],
                      "dem_mean": [1.0, np.nan, 2.0, 3.0], "rain_mm": [1.0, 2.0, 3.0, 4.0]})
    ok = pd.Series([True, True, True, False])
    feats = ["tch_wl_p20c", "dem_mean", "rain_mm"]
    assert bu.unexpected_train_nan(t, feats, ok) == {}
    assert bu.unexpected_train_nan(t.assign(tch_wl_p20c=[1.0, np.nan, 1.0, 1.0]), feats, ok) == {"tch_wl_p20c": 1}
    assert bu.unexpected_train_nan(t.assign(rain_mm=[1.0, np.nan, np.nan, np.nan]), feats, ok) == {"rain_mm": 2}


# ---------------------------------------------------------------- Dot 7: bang theo bien muc tieu
def test_target_table_bo_nhan_do_man_cot_cam_va_co_huan_luyen():
    from unified_table import SAL_TRAIN_COL, target_table

    base = merge_sources(*_src()).assign(solar=1.0, temp_c=27.0, rh_percent=80.0)
    # a/2019: sal train True, nhan moi co; a/2020: sal False; b/2019: sal False (scope 0); b/2020: nhan moi NaN
    lab = base[["cell_id", "season"]].assign(rain_chirps=[300.0, 310.0, 250.0, np.nan], lbl_cover_frac=1.0,
                                             lbl_valid_px=2.0, lbl_ok=[True, True, True, False])
    t, dropped = target_table(base, lab.iloc[::-1], "rain_chirps", ("lbl_cover_frac", "lbl_valid_px", "lbl_ok"))
    assert dropped == ["rain_mm"]
    for c in ("salinity", "n_valid_px", "valid_frac", "train_ok", "train_ok_10pct", "rain_mm"):
        assert c not in t.columns
    assert list(t.columns[:3]) == ["cell_id", "season", "rain_chirps"]
    r = t.set_index(["cell_id", "season"])
    assert r[SAL_TRAIN_COL].tolist() == base.set_index(["cell_id", "season"])[TRAIN_COL].tolist()
    assert r[TRAIN_COL].tolist() == [True, False, False, False]
    assert {"solar", "temp_c", "dem_mean", "lbl_cover_frac"} <= set(t.columns)
    assert np.isnan(r.loc[("b", 2020), "rain_chirps"])                    # NaN khong bi dien
    t2, d2 = target_table(base, lab.rename(columns={"rain_chirps": "rh_era5"}), "rh_era5", ("lbl_ok",))
    assert sorted(d2) == ["rh_percent", "temp_c"] and "solar" in t2.columns and "rain_mm" in t2.columns


def test_target_table_khoa_lech_hoac_trung_bao_loi():
    from unified_table import target_table

    base = merge_sources(*_src())
    lab = base[["cell_id", "season"]].assign(ndwi=0.1)
    with pytest.raises(ValueError, match="Tap khoa"):
        target_table(base, lab.iloc[:3], "ndwi")
    with pytest.raises(ValueError, match="trung"):
        target_table(base, pd.concat([lab, lab.iloc[:1]]), "ndwi")
    with pytest.raises(ValueError, match="muc tieu"):
        target_table(base, lab.rename(columns={"ndwi": "x"}), "ndwi")

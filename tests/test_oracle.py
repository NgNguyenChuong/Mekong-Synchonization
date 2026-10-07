"""Sai so ly tuong theo khung (src/training/oracle.py, scripts/analyze_oracle.py) tren du lieu TONG HOP."""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import analyze_oracle as ao  # noqa: E402
from training.block_stats import compare_family  # noqa: E402
from training.oracle import cell_label_series, check_reference_coverage, oracle_errors  # noqa: E402

UTM = "EPSG:32648"
SEASONS = [2019, 2020, 2021]
OFFS = [0.1, -0.1, 0.3, -0.3]  # dap an = nhan o + lech -> |e| TB = 0,2


def _cells():
    # 8 o 25 x 50 km, x 500..700 km: o 0-1 trong khoi giu rieng blk_10_20, con lai 3 khoi CV
    return [box(500000 + i * 25000, 1000000, 525000 + i * 25000, 1050000) for i in range(8)]


def _label(i, s):
    return 1.0 + 0.5 * i + 0.01 * (s - 2019)


def _write_fixture(d, nan_cell_b=None):
    gd, ld = d / "grids", d / "labels"
    gd.mkdir()
    ld.mkdir()
    for g in ("gA", "gB"):
        gpd.GeoDataFrame({"cell_id": [f"c{i}" for i in range(8)]}, geometry=_cells(), crs=UTM).to_file(
            gd / f"{g}.geojson", driver="GeoJSON")
    pts, ref = [], []
    for i in range(8):
        for k, off in enumerate(OFFS):
            pid = f"p{i}_{k}"
            pts.append((pid, Point(505000 + i * 25000 + k * 4000, 1010000 + k * 8000)))
            for s in SEASONS:
                ref.append((pid, s, _label(i, s) + off, 9))
    gpd.GeoDataFrame({"point_id": [p for p, _ in pts]}, geometry=[g for _, g in pts], crs=UTM).to_crs(4326).to_file(
        d / "points.geojson", driver="GeoJSON")
    pd.DataFrame(ref, columns=["point_id", "season", "ref_salinity", "n_valid_3x3"]).to_csv(
        ld / "points_reference.csv", index=False)
    for g in ("gA", "gB"):
        rows = [(f"c{i}", s, _label(i, s), True) for i in range(8) for s in SEASONS]
        lab = pd.DataFrame(rows, columns=["cell_id", "season", "salinity", "train_ok"])
        if g == "gB" and nan_cell_b is not None:
            lab.loc[(lab["cell_id"] == nan_cell_b[0]) & (lab["season"] == nan_cell_b[1]), "salinity"] = np.nan
        lab.to_csv(ld / f"{g}_labels_season.csv", index=False)
    blk = ["blk_10_20", "blk_11_20", "blk_12_20", "blk_13_20"]
    gpd.GeoDataFrame({"block_id": blk, "is_holdout": [True, False, False, False]},
                     geometry=[box(500000 + j * 50000, 1000000, 550000 + j * 50000, 1050000) for j in range(4)],
                     crs=UTM).to_crs(4326).to_file(d / "blocks.geojson", driver="GeoJSON")
    pd.DataFrame({"block_id": blk, "unit_id": blk, "cv_fold": [-1, 0, 1, 2]}).to_csv(d / "f50.csv", index=False)
    pd.DataFrame({"block_id": blk, "unit_id": ["blk_10_20", "s_a", "s_a", "s_b"], "cv_fold": [-1, 0, 0, 1]}).to_csv(
        d / "f100.csv", index=False)
    pd.DataFrame({"file": ["data/grids/gA.geojson", "data/grids/gB.geojson"], "mean_area_km2": [1250.0, 1250.0],
                  "tier_h3_res": [7, 7]}).to_csv(d / "area.csv", index=False)


def _args(d, **kw):
    a = ao.parse_args(["--targets", "salinity", "--grids", "gA", "gB", "--out", str(d / "out" / "oracle"),
                       "--labels-dir", str(d / "labels"), "--grids-dir", str(d / "grids"),
                       "--points", str(d / "points.geojson"), "--blocks", str(d / "blocks.geojson"),
                       "--folds50", str(d / "f50.csv"), "--folds100", str(d / "f100.csv"),
                       "--area-table", str(d / "area.csv"), "--seasons", *map(str, SEASONS),
                       "--expect-pairs", "1", "--min-pts", "1"])
    a.expect_points = {"salinity": 32}
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_mae_dung_va_hai_luoi_giong_nhau_delta_0(tmp_path):
    _write_fixture(tmp_path)
    t = ao.run(_args(tmp_path))
    m = t["mae"].set_index("grid")
    assert (m["n_ref"] == 32 * 3).all() and (m["n_nan_cell"] == 0).all()
    for c in ("mae_diem", "mae_diem_chung", "mae_diem_chung_cv", "mae_don_vi_k50", "mae_don_vi_k100"):
        assert m[c].to_numpy() == pytest.approx([0.2, 0.2])
    # cv: 24 diem khoi CV x 2 mua (2020 giu rieng)
    assert (m["n_chung_cv"] == 24 * 2).all() and (m["G_k50"] == 3).all() and (m["G_k100"] == 2).all()
    cap = t["cap"]
    assert len(cap) == 2 and set(cap["blocks"]) == {"k50", "k100"}
    assert cap["delta_hat"].abs().max() == 0.0 and (cap["ti_le_delta_tren_nguong"] == 0).all()
    assert cap["label"].str.startswith("oracle_").all() and (cap["pred_source"] == "oracle").all()
    assert cap["delta_ref"].to_numpy() == pytest.approx([0.2, 0.2])  # Delta_min = 5% MAE oracle cua muc
    assert len(t["ti_le"]) == 2
    for k in ao.OUT_KINDS:
        assert os.path.exists(tmp_path / "out" / f"oracle_{k}.csv")


def test_o_nan_bi_bo_dem_va_khong_dien_0(tmp_path):
    _write_fixture(tmp_path, nan_cell_b=("c3", 2019))  # o c3 cua gB khong co nhan mua 2019
    t = ao.run(_args(tmp_path))
    m = t["mae"].set_index("grid")
    assert m.loc["gB", "n_nan_cell"] == 4 and m.loc["gB", "n_points_nan_cell"] == 4
    assert m.loc["gA", "n_nan_cell"] == 0
    assert (m["n_chung"] == 96 - 4).all()  # bo khoi tap chung cua MOI luoi
    assert m.loc["gB", "n_valid"] == 92 and m.loc["gB", "mae_diem"] == pytest.approx(0.2)
    b = t["bo"]
    assert int(b.loc[(b["grid"] == "gA") & (b["season"] == 2019), "n_dropped_common"].iloc[0]) == 4
    assert t["cap"]["delta_hat"].abs().max() == 0.0


def test_oracle_errors_tinh_tay_va_nan():
    pf = pd.DataFrame({"point_id": ["a", "a", "b"], "season": [2019, 2020, 2019], "cell_id": ["x", "x", "y"],
                       "y_ref": [1.0, 2.0, 5.0], "role": "cv"})
    lab = pd.DataFrame({"cell_id": ["x", "x", "y"], "season": [2019, 2020, 2019], "v": [1.5, np.nan, 4.0]})
    e = oracle_errors(pf, cell_label_series(lab, "v"), "g")
    assert e["err"].iloc[0] == pytest.approx(0.5) and np.isnan(e["err"].iloc[1]) and e["err"].iloc[2] == pytest.approx(-1)
    with pytest.raises(ValueError, match="trung"):
        cell_label_series(pd.concat([lab, lab.iloc[:1]]), "v")
    with pytest.raises(ValueError, match="thieu cot"):
        cell_label_series(lab, "salinity")


def test_so_diem_va_mua_sai_bao_loi():
    pf = pd.DataFrame({"point_id": ["a", "b", "a"], "season": [2019, 2019, 2021]})
    assert check_reference_coverage(pf, 2, [2019, 2020, 2021], nan_seasons=(2020,)) == [2020]
    with pytest.raises(ValueError, match="So diem"):
        check_reference_coverage(pf, 3, [2019, 2020, 2021], nan_seasons=(2020,))
    with pytest.raises(ValueError, match="Mua co dap an"):
        check_reference_coverage(pf, 2, [2019, 2020, 2021])  # thieu 2020 ma khong khai bao


def test_thieu_file_va_sai_so_diem_la_loi(tmp_path):
    _write_fixture(tmp_path)
    os.remove(tmp_path / "labels" / "gB_labels_season.csv")
    with pytest.raises(SystemExit, match="LOI: thieu file nhan o"):
        ao.run(_args(tmp_path))
    assert not os.path.exists(tmp_path / "out" / "oracle_mae.csv")
    d2 = tmp_path / "d2"
    d2.mkdir()
    _write_fixture(d2)
    with pytest.raises(ValueError, match="So diem"):
        ao.run(_args(d2, expect_points={"salinity": 31}))


def test_mua_chi_cap_muc_tho():
    mp = pd.DataFrame({"grid_a": ["a5", "a7"], "grid_b": ["b5", "b7"], "tier": [5, 7]})
    assert ao.target_pairs("rain_chirps", mp, ["a5", "b5", "a7", "b7"])["tier"].tolist() == [5]
    assert len(ao.target_pairs("ndwi", mp, ["a5", "b5", "a7", "b7"])) == 2
    assert ao.NAN_SEASONS["dsr_mcd18"] == (2020,)


def test_compare_family_khong_tron_oracle_voi_mo_hinh():
    rows = [(g, "m", 0, f"u{u}_{i}", 2019, 0.1 if g == "A" else -0.1, "oracle")
            for u in range(3) for i in range(2) for g in ("A", "B")]
    df = pd.DataFrame(rows, columns=["grid", "model", "seed", "point_id", "season", "err", "pred_source"])
    pu = pd.Series({f"u{u}_{i}": f"u{u}" for u in range(3) for i in range(2)})
    kw = dict(cv_units=["u0", "u1", "u2"], holdout_units=[], holdout_seasons=(2020,), mode="cv", delta_min=0.05,
              delta_min_kind="abs", alpha=0.05, model="m", min_pts=1)
    assert compare_family(df, pu, [("A", "B")], **kw)["delta_hat"].iloc[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="khong tron"):
        compare_family(df.assign(pred_source=np.where(df.index == 0, "oof", "oracle")), pu, [("A", "B")], **kw)

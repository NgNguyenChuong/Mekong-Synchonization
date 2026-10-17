"""Thiet ke danh gia vung doi chung: khoi file la o vuong UTM chinh xac (khong ro ri hinh hoc), quy tac R-H chon
giu rieng theo scope, build_cv_folds --objective-mode rel. Bang khoi: tests/fixtures/blocks_doi_chung_{ms,nl}.csv."""
import glob
import importlib.util
import os
from argparse import Namespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from eval_design import (dense_blocks_4326, exact_block_frame, fold_schemes, merge_small_blocks, pick_holdout_blocks,
                         select_holdout_by_scope, training_mask, unit_pool)
from region import crs_metric, region_paths

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIX = os.path.join(ROOT, "tests", "fixtures")
RH_MS = ["blk_8_66", "blk_9_66", "blk_11_66", "blk_12_66", "blk_14_66", "blk_14_67", "blk_16_67"]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pick_holdout_blocks_bo_khoi_khong_dat_va_khoi_nho():
    w = np.array([0.0, 10.0, 200.0, 300.0, 400.0])
    h = pick_holdout_blocks(w, 0.5, 1, min_weight=100)
    assert not h[0] and not h[1]  # weight 0 va < min_weight khong bao gio vao giu rieng
    assert h.any() and w[h].sum() >= 0.5 * w.sum()


def test_dense_blocks_khong_ro_ri_o_canh_khoi():
    # MS xa kinh tuyen truc UTM 15N: canh 4 dinh 4326 lech ~30 m so voi canh UTM -> o chi cham canh bac bi coi la huan luyen
    utm = crs_metric("mississippi")
    b = gpd.GeoDataFrame({"block_id": ["blk_16_67"], "is_holdout": [True]}, geometry=[box(0, 0, 1, 1)], crs=utm)
    ex = exact_block_frame(b, utm=utm)
    x0, y0, x1, y1 = ex.geometry.iloc[0].bounds
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    cells = [box(x0 + 1, my - 5, x0 + 9, my + 5), box(x1 - 9, my - 5, x1 - 1, my + 5),
             box(mx - 5, y0 + 1, mx + 5, y0 + 9), box(mx - 5, y1 - 9, mx + 5, y1 - 1)]  # moi o nam 1-9 m trong canh
    grid = gpd.GeoDataFrame({"cell_id": list("WESN")}, geometry=cells, crs=utm).to_crs(4326)
    assert training_mask(grid, ex.to_crs(4326)).any()  # hinh 4 dinh cu: ro ri
    assert not training_mask(grid, dense_blocks_4326(ex.to_crs(4326), utm)).any()


@pytest.mark.skipif(not os.path.exists(region_paths("mississippi")["eval_dir"] + "/holdout_blocks.geojson"),
                    reason="chua co du lieu MS")
def test_ms_khong_o_huan_luyen_giao_khoi_giu_rieng_chinh_xac():
    # CHI DOC file khoi hien co: dung lai khoi giu rieng bang dense_blocks_4326 -> 0 o train giao hinh UTM chinh xac
    p, utm = region_paths("mississippi"), crs_metric("mississippi")
    b = gpd.read_file(p["eval_dir"] + "/holdout_blocks.geojson")
    hold = exact_block_frame(b, utm=utm).query("is_holdout").geometry.union_all()
    dense = dense_blocks_4326(b, utm)
    for path in sorted(glob.glob(p["grids_dir"] + "/*.geojson")):
        grid = gpd.read_file(path)
        train = training_mask(grid, dense)
        leak = grid[train].to_crs(utm).geometry.intersects(hold).sum()
        assert leak == 0, f"{os.path.basename(path)}: {leak} o huan luyen giao khoi giu rieng"


def _tab(name):
    return pd.read_csv(os.path.join(FIX, f"blocks_doi_chung_{name}.csv"))


def test_rh_ms_seed_43_bay_khoi():
    t = _tab("ms")
    r = select_holdout_by_scope(t, 0.2, 42, 150.0, (1 / 3, 3.0))
    assert (r["seed"], r["tried"]) == (43, 2)
    assert sorted(t.block_id[r["is_holdout"]]) == sorted(RH_MS)
    assert r["scope_share"] == pytest.approx(0.236, abs=0.001)
    assert r["coast_ratio"] == pytest.approx(1.46, abs=0.01)
    assert (t.loc[r["is_holdout"], "scope_km2"] >= 150).all()


def test_rh_khong_rang_buoc_nhan_seed_dau_va_nl_giu_seed_42():
    t = _tab("ms")
    r = select_holdout_by_scope(t, 0.2, 42, 150.0, None)
    assert (r["seed"], r["tried"]) == (42, 1)
    assert r["coast_ratio"] < 0.2  # seed 42 la duoi rui ro (it ven bien) -> day la ly do can rang buoc
    n = select_holdout_by_scope(_tab("nl"), 0.2, 42, 150.0, (1 / 3, 3.0))
    assert (n["seed"], n["tried"]) == (42, 1)


def test_rh_het_seed_bao_loi():
    with pytest.raises(ValueError):
        select_holdout_by_scope(_tab("ms"), 0.2, 42, 150.0, (50.0, 60.0), max_tries=5)


def _units(t, dens):
    bt = t[["block_id", "land_km2", "scope_km2", "coast_scope_km2", "is_holdout"]].copy()
    bt["n_points"] = np.where(bt.is_holdout, 0, np.round(dens * bt.scope_km2)).astype(int)
    m, _ = merge_small_blocks(bt, min_points=30)
    el = (~m.is_holdout) & (m.land_km2 > 0)
    return m[el].groupby("unit_id").agg(scope_km2=("scope_km2", "sum"), coast_scope_km2=("coast_scope_km2", "sum"),
                                        n_points=("n_points", "sum")).reset_index()


def test_resolve_rule_abs_mac_dinh_giu_nguyen_va_rel():
    cv = _script("build_cv_folds")
    base = dict(tols=None, max_objective=None, objective_mode="abs")
    a = cv.resolve_rule(Namespace(**base))
    assert (a.max_objective, a.tols) == (0.183, None)
    a = cv.resolve_rule(Namespace(**{**base, "objective_mode": "rel"}))
    assert (a.max_objective, a.tols) == (None, [0.10, 0.15])
    with pytest.raises(SystemExit):
        cv.resolve_rule(Namespace(**{**base, "objective_mode": "rel", "max_objective": 0.183}))


def test_nl_cv_dat_voi_coast_tol_030_va_rel_nhung_loi_voi_abs():
    u = _units(_tab("nl"), 0.542)
    assert len(u) == 15
    pool = unit_pool(u, n_folds=5, n_starts=1000, pool_seed=0)
    with pytest.raises(ValueError):  # abs 0,183: muc tieu tot nhat NL 0,499
        fold_schemes(u, [42, 43, 44], n_starts=1000, coast_tol=0.30, max_objective=0.183, pool=pool)
    out, _, info = fold_schemes(u, [42, 43, 44], n_starts=1000, coast_tol=0.30, max_objective=None,
                                tols=(0.10, 0.15), pool=pool)
    objs = [out[s].attrs["objective"] for s in (42, 43, 44)]
    assert objs == pytest.approx([0.499, 0.506, 0.512], abs=0.001)
    assert info["mode"] == "tuong_doi"
    assert max(out[s].attrs["coast_max_dev"] for s in out) == pytest.approx(0.253, abs=0.001)

"""Chia tap theo thiet ke danh gia chinh (An duyet 2026-10-03) va CV khoi 5 fold dung chung 13 luoi.

- Hinh hoc TONG HOP voi gia tri biet truoc (khoi 50 km tren luoi UTM 32648).
- Tich hop: luoi THAT data/grids/h3_res_5.geojson + khoi THAT data/eval/holdout_blocks.geojson, nhan
  TONG HOP; bang chia do code sinh ra phai DAT .claude/skills/method-review/scripts/check_split.py.
  (data/ va .claude/ khong track git -> skip neu thieu.)
"""
import json
import os
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval_design import (  # noqa: E402
    assign_cv_folds,
    cell_block_table,
    fold_balance_objective,
    fold_max_rel_dev,
    fold_overlap,
    merge_small_blocks,
    parse_touched_folds,
    read_cell_block_table,
    scheme_code,
    scheme_code_str,
    training_mask,
)
from training.split import (  # noqa: E402
    assign_eval_split,
    block_cv_cell_summary,
    block_cv_membership,
    block_cv_splits,
    block_cv_summary,
)

UTM = "EPSG:32648"
HS = (2020,)
S = 50_000.0
GRID_H3_5 = os.path.join(ROOT, "data", "grids", "h3_res_5.geojson")
GRID_S2_9 = os.path.join(ROOT, "data", "grids", "s2_level_9.geojson")
BLOCKS = os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson")
CV_FOLDS = os.path.join(ROOT, "data", "eval", "cv_folds.csv")
CHECK_SPLIT = os.path.join(ROOT, ".claude", "skills", "method-review", "scripts", "check_split.py")
need_real = pytest.mark.skipif(not (os.path.exists(GRID_H3_5) and os.path.exists(BLOCKS)),
                               reason="thieu data/grids hoac data/eval (khong track git)")


# ------------------------------------------------------------------ tong hop
def _blocks(spec):
    """spec: [(i, j, land_km2, is_holdout)] -> GeoDataFrame EPSG:4326 (nhu file that)."""
    rows = [{"block_id": f"blk_{i}_{j}", "land_km2": land, "is_holdout": hold,
             "geometry": box(i * S, j * S, (i + 1) * S, (j + 1) * S)} for i, j, land, hold in spec]
    return gpd.GeoDataFrame(rows, crs=UTM).to_crs(4326)


def _grid(cells):
    """cells: [(cell_id, xmin, ymin, xmax, ymax)] trong UTM -> EPSG:4326."""
    return gpd.GeoDataFrame({"cell_id": [c[0] for c in cells]},
                            geometry=[box(*c[1:]) for c in cells], crs=UTM).to_crs(4326)


# 3 khoi ngang hang: blk_10_20 (fold A), blk_11_20 (fold B), blk_12_20 (GIU RIENG)
X0, Y0 = 10 * S, 20 * S


@pytest.fixture()
def toy():
    blocks = _blocks([(10, 20, 2500.0, False), (11, 20, 2500.0, False), (12, 20, 2500.0, True)])
    folds = pd.DataFrame({"block_id": ["blk_10_20", "blk_11_20", "blk_12_20"],
                          "cv_fold": [0, 1, -1], "land_km2": [2500.0] * 3})
    m = 1000.0
    cells = [
        ("a_in", X0 + 10 * m, Y0 + 10 * m, X0 + 20 * m, Y0 + 20 * m),           # trong A
        ("a_in2", X0 + 25 * m, Y0 + 10 * m, X0 + 35 * m, Y0 + 20 * m),          # trong A
        ("ab_cut", X0 + 45 * m, Y0 + 10 * m, X0 + 52 * m, Y0 + 20 * m),         # tam o A, cham B
        ("b_in", X0 + 60 * m, Y0 + 10 * m, X0 + 70 * m, Y0 + 20 * m),           # trong B
        ("b_in2", X0 + 75 * m, Y0 + 10 * m, X0 + 85 * m, Y0 + 20 * m),          # trong B
        ("bh_cut", X0 + 95 * m, Y0 + 30 * m, X0 + 104 * m, Y0 + 40 * m),        # tam o B, cham GIU RIENG
        ("h_in", X0 + 110 * m, Y0 + 10 * m, X0 + 120 * m, Y0 + 20 * m),         # trong GIU RIENG
        ("out_a", X0 + 10 * m, Y0 + 46 * m, X0 + 20 * m, Y0 + 56 * m),          # tam ngoai moi khoi -> A
    ]
    return _grid(cells), blocks, folds


def test_cell_block_table_tam_cham_va_ngoai_khoi(toy):
    grid, blocks, folds = toy
    t = cell_block_table(grid, blocks, folds).set_index("cell_id")
    assert t.loc["a_in", "block_id"] == "blk_10_20" and t.loc["a_in", "touched_folds"] == "0"
    assert t.loc["ab_cut", "block_id"] == "blk_10_20" and t.loc["ab_cut", "cv_fold"] == 0
    assert t.loc["ab_cut", "touched_folds"] == "0|1"
    assert t.loc["bh_cut", "cv_fold"] == 1 and bool(t.loc["bh_cut", "touches_holdout"])
    assert t.loc["bh_cut", "touched_folds"] == "1"           # fold giu rieng (-1) khong ghi vao
    assert t.loc["h_in", "cv_fold"] == -1 and bool(t.loc["h_in", "touches_holdout"])
    assert t.loc["h_in", "touched_folds"] == ""
    # tam (y = 51 km) nam ngoai moi khoi trong file -> khoi giao lon nhat
    assert t.loc["out_a", "block_id"] == "blk_10_20" and t.loc["out_a", "block_assign"] == "max_overlap"
    assert (t.drop("out_a")["block_assign"] == "centroid").all()
    # touches_holdout = dung training_mask (cung ham voi test khong gian / check_split)
    assert (t.loc[grid["cell_id"], "touches_holdout"].to_numpy() == ~training_mask(grid, blocks)).all()


def test_cell_block_table_ghi_doc_lai_giu_chuoi(toy, tmp_path):
    grid, blocks, folds = toy
    t = cell_block_table(grid, blocks, folds)
    p = tmp_path / "t.csv"
    t.to_csv(p, index=False)
    back = read_cell_block_table(p)
    assert back["touched_folds"].tolist() == t["touched_folds"].tolist()  # "" va "0" khong thanh NaN/so
    assert back["touches_holdout"].tolist() == t["touches_holdout"].tolist()
    assert parse_touched_folds(["0|1", "", 3, np.nan]) == [{0, 1}, set(), {3}, set()]


def test_cell_block_table_fold_thieu_khoi_bao_loi(toy):
    grid, blocks, folds = toy
    with pytest.raises(ValueError, match="thieu khoi"):
        cell_block_table(grid, blocks, folds.iloc[:2])


def test_assign_cv_folds_can_bang_va_tat_dinh():
    # 8 khoi dat 10, 10, 10, 10, 20, 20, 30, 40 (tong 150) + 1 khoi giu rieng + 1 khoi khong dat
    spec = [(10 + k, 20, land, False) for k, land in enumerate([10, 10, 10, 10, 20, 20, 30, 40])]
    spec += [(20, 20, 50.0, True), (21, 20, 0.0, False)]
    blocks = _blocks(spec)
    f1 = assign_cv_folds(blocks, n_folds=3, seed=7, n_trials=500)
    f2 = assign_cv_folds(blocks, n_folds=3, seed=7, n_trials=500)
    pd.testing.assert_frame_equal(f1, f2)
    fo = f1.set_index("block_id")["cv_fold"]
    assert fo["blk_20_20"] == -1 and fo["blk_21_20"] == -1  # giu rieng / khong dat
    land = f1[f1["cv_fold"] >= 0].groupby("cv_fold")["land_km2"].sum()
    assert sorted(land.index) == [0, 1, 2]
    assert land.tolist() == [50.0, 50.0, 50.0]  # tong 150 chia deu duoc
    with pytest.raises(ValueError, match="n_folds"):
        assign_cv_folds(blocks, n_folds=9)


def _season_table(cells, seasons):
    df = pd.DataFrame([(c, s) for c in cells for s in seasons], columns=["cell_id", "season"])
    df["salinity"] = np.arange(len(df), dtype=float)
    return df


def test_assign_eval_split_ba_nhom(toy):
    grid, blocks, folds = toy
    t = cell_block_table(grid, blocks, folds)
    df = _season_table(grid["cell_id"], [2019, 2020, 2021])
    out = assign_eval_split(df, t, holdout_seasons=(2020,))
    g = out.set_index(["cell_id", "season"])["test_group"]
    assert g[("a_in", 2019)] == "" and out.loc[(out.cell_id == "a_in") & (out.season == 2019), "split"].item() == "train"
    assert g[("a_in", 2020)] == "thoi_gian"
    assert g[("bh_cut", 2019)] == "khong_gian"     # cham khoi giu rieng du tam o khoi CV
    assert g[("h_in", 2020)] == "ca_hai"
    assert set(out.loc[out["split"] == "train", "season"]) == {2019, 2021}
    assert out.index.equals(df.index)
    with pytest.raises(ValueError, match="Nhom test rong"):
        assign_eval_split(df[df["season"] != 2020], t)
    with pytest.raises(ValueError, match="khong co trong cell_table"):
        assign_eval_split(pd.concat([df, pd.DataFrame({"cell_id": ["la"], "season": [2019], "salinity": [1.0]})]), t)


def test_block_cv_splits_touch_va_centroid(toy):
    grid, blocks, folds = toy
    t = cell_block_table(grid, blocks, folds)
    df = assign_eval_split(_season_table(grid["cell_id"], [2019, 2020, 2021]), t)
    train = df[df["split"] == "train"].reset_index(drop=True)
    sp = block_cv_splits(train, t, holdout_seasons=HS, rule="touch", min_val_cells=1)
    cells = train["cell_id"].to_numpy()
    # fold 0 (A): val = a_in, a_in2, ab_cut, out_a; train = b_in, b_in2 (ab_cut cham 0 -> khong train fold 1)
    assert set(cells[sp[0][1]]) == {"a_in", "a_in2", "ab_cut", "out_a"}
    assert set(cells[sp[0][0]]) == {"b_in", "b_in2"}
    assert set(cells[sp[1][1]]) == {"b_in", "b_in2"}
    assert set(cells[sp[1][0]]) == {"a_in", "a_in2", "out_a"}       # ab_cut bi loai do "cham"
    cen = block_cv_splits(train, t, holdout_seasons=HS, rule="centroid", min_val_cells=1)
    assert set(cells[cen[1][0]]) == {"a_in", "a_in2", "ab_cut", "out_a"}
    summ = block_cv_summary(train, t, holdout_seasons=HS).set_index("fold")
    assert summ.loc[1, "n_excluded_touch_cells"] == 1 and summ.loc[0, "n_excluded_touch_cells"] == 0
    # moi dong train la val dung mot lan
    allval = np.concatenate([v for _, v in sp])
    assert sorted(allval.tolist()) == list(range(len(train)))
    with pytest.raises(ValueError, match="min_val_cells"):
        block_cv_splits(train, t, holdout_seasons=HS, min_val_cells=3)
    with pytest.raises(ValueError, match="khong phai split"):
        block_cv_splits(df, t, holdout_seasons=HS, min_val_cells=1)
    with pytest.raises(ValueError, match="CHAM khoi giu rieng"):  # khong cot split, khong 2020, co o cham
        block_cv_splits(df[df["season"] != 2020].drop(columns=["split"]), t, holdout_seasons=HS, min_val_cells=1)


# ------------------------------------------------------------------ tich hop luoi/khoi that
@pytest.fixture(scope="module")
def real():
    if not (os.path.exists(GRID_H3_5) and os.path.exists(BLOCKS)):
        pytest.skip("thieu data/grids hoac data/eval")
    blocks = gpd.read_file(BLOCKS)
    # Fold THAT (don vi kiem dinh, can bang mat na pham vi + ven bien) neu da sinh; khong thi phuong an cu theo dat.
    if os.path.exists(CV_FOLDS):
        folds = pd.read_csv(CV_FOLDS, dtype={"block_id": str, "unit_id": str})
    else:
        folds = assign_cv_folds(blocks, n_folds=5, seed=42)
    grid = gpd.read_file(GRID_H3_5)
    grid["cell_id"] = grid["cell_id"].astype(str)
    table = cell_block_table(grid, blocks, folds)
    return blocks, folds, grid, table


@need_real
def test_cv_folds_file_that_khop_ham(real):
    """cv_folds*.csv (J gop khoi < 30 diem, I can bang 2 tieu chi, D 3 phuong an) tai lap tu cot trong file."""
    blocks, _, _, _ = real
    if not os.path.exists(CV_FOLDS):
        pytest.skip("chua co data/eval/cv_folds.csv")
    disk = pd.read_csv(CV_FOLDS, dtype={"block_id": str, "unit_id": str})
    with open(CV_FOLDS + ".provenance.json", encoding="utf-8") as fh:
        prov = json.load(fh)
    seeds = prov["seeds"]
    s0 = pd.read_csv(os.path.join(ROOT, "data", "eval", f"cv_folds_s{seeds[0]}.csv"), dtype={"block_id": str, "unit_id": str})
    pd.testing.assert_frame_equal(disk, s0)                              # cv_folds.csv = phuong an seed dau
    hold = set(blocks.loc[blocks["is_holdout"], "block_id"])
    assert set(disk.loc[disk["cv_fold"] == -1, "block_id"]) == hold
    assert set(disk["block_id"]) == set(blocks["block_id"])
    # so diem CV moi khoi = dem tu eval_points.geojson
    pts = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    cnt = pts[~pts["is_holdout"].astype(bool)].groupby("block_id").size()
    assert (disk.set_index("block_id")["n_cv_points"] == cnt.reindex(disk["block_id"]).fillna(0).astype(int).to_numpy()).all()
    if "points_sha256" in prov:                                            # fold dung dung ban diem hien tai
        from preprocessing import file_sha256
        assert prov["points_sha256"] == file_sha256(os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    # so diem CV ven bien (dist_coast_km <= coast_km tai pixel chua diem) - chi khi co cot + raster
    if "n_cv_coast_points" in disk.columns and os.path.exists(prov.get("dist_coast", "")):
        from eval_design import dist_coast_at_points
        cvp = pts[~pts["is_holdout"].astype(bool)].copy()
        cvp["d"] = dist_coast_at_points(cvp, prov["dist_coast"])
        cc = cvp[cvp["d"] <= prov["coast_km"]].groupby("block_id").size()
        assert (disk.set_index("block_id")["n_cv_coast_points"]
                == cc.reindex(disk["block_id"]).fillna(0).astype(int).to_numpy()).all()
    # J: tai lap don vi; moi don vi CV >= min_points diem; khoi cung don vi cung fold
    tab = disk.merge(blocks[["block_id", "is_holdout"]], on="block_id").rename(columns={"n_cv_points": "n_points"})
    merged, _ = merge_small_blocks(tab.drop(columns=["unit_id"]), min_points=prov["min_points"])
    assert (merged.set_index("block_id")["unit_id"] == tab.set_index("block_id")["unit_id"]).all()
    cv = tab[tab["cv_fold"] >= 0]
    assert (cv.groupby("unit_id")["n_points"].sum() >= prov["min_points"]).all()
    assert (cv.groupby("unit_id")["cv_fold"].nunique() == 1).all()
    # I + D (An chot; CHG-14 2026-10-04): s42 = phuong an chinh (muc tieu thap nhat, ghi trong provenance); moi
    # phuong an: ven bien +-coast_tol quanh TB fold; muc tieu <= max_objective (TUYET DOI, CHG-14) - hoac quy tac
    # cu <= (1 + tol_used) x tot nhat; do trung Hungarian <= max_overlap tung cap; ma phan chia khop provenance.
    # (Tai lap day du tap ung vien 3000 diem xuat phat mat ~3 phut -> kiem bang build_cv_folds.py khong --force.)
    units = cv.groupby("unit_id")[["scope_km2", "coast_scope_km2"]].sum().reset_index().sort_values("unit_id")
    labs = {}
    for sd in seeds:
        f = pd.read_csv(os.path.join(ROOT, "data", "eval", f"cv_folds_s{sd}.csv"), dtype={"block_id": str, "unit_id": str})
        with open(os.path.join(ROOT, "data", "eval", f"cv_folds_s{sd}.csv.provenance.json"), encoding="utf-8") as fh:
            p = json.load(fh)
        fo = f[f["cv_fold"] >= 0].drop_duplicates("unit_id").set_index("unit_id")["cv_fold"]
        assert sorted(fo.unique()) == [0, 1, 2, 3, 4]
        lab = fo.reindex(units["unit_id"]).to_numpy()
        labs[sd] = lab
        tot = np.zeros((5, 2))
        np.add.at(tot, lab, units[["scope_km2", "coast_scope_km2"]].to_numpy())
        assert fold_balance_objective(tot) == pytest.approx(p["objective"], abs=1e-5)
        assert fold_max_rel_dev(lab, units["coast_scope_km2"], 5) <= p["coast_tol"] + 1e-9
        best = prov["objective"]                                           # s42 = tot nhat
        assert p["objective"] >= best - 1e-9
        if p.get("rule_mode") == "tuyet_doi":                              # CHG-14
            assert p["objective"] <= p["max_objective"] + 1e-9
            assert p["tol_used"] is None
            assert scheme_code_str(scheme_code(lab, units["unit_id"].to_numpy())) == p["scheme_code"]
        else:                                                              # quy tac cu CHG-06
            assert p["objective"] <= best * (1 + p["tol_used"]) + 1e-9
            assert p["tol_used"] in p["tols"]
    for i, x in enumerate(seeds):
        for y in seeds[i + 1:]:
            assert fold_overlap(labs[x], labs[y]) <= prov["max_overlap"] + 1e-9, (x, y)


@need_real
def test_cung_cv_folds_cho_moi_luoi(real):
    blocks, folds, _, t5 = real
    fold_of = folds.set_index("block_id")["cv_fold"]
    g9 = gpd.read_file(GRID_S2_9)
    t9 = cell_block_table(g9, blocks, folds)
    for t in (t5, t9):
        assert (t["block_id"].map(fold_of).to_numpy() == t["cv_fold"].to_numpy()).all()
    # bang o -> fold da ghi cho 13 luoi deu sinh tu CUNG mot file cv_folds.csv
    cell_dir = os.path.join(ROOT, "data", "eval", "cell_blocks")
    if os.path.exists(CV_FOLDS) and os.path.isdir(cell_dir):
        from preprocessing import file_sha256
        sha = file_sha256(CV_FOLDS)
        provs = [f for f in os.listdir(cell_dir) if f.endswith(".provenance.json")]
        assert len(provs) == 13
        for f in provs:
            with open(os.path.join(cell_dir, f), encoding="utf-8") as fh:
                assert json.load(fh)["cv_folds_sha256"] == sha, f


@need_real
def test_check_split_dat_tren_bang_sinh_ra(real, tmp_path):
    if not os.path.exists(CHECK_SPLIT):
        pytest.skip("khong co .claude/skills/method-review/scripts/check_split.py")
    _, _, grid, table = real
    df = _season_table(grid["cell_id"], range(2014, 2027))
    out = assign_eval_split(df, table, holdout_seasons=(2020,))
    out = out.merge(table[["cell_id", "block_id", "cv_fold"]], on="cell_id", how="left")
    out.loc[out["split"] == "test", "cv_fold"] = np.nan
    out["label_mask_id"] = "mask_v2_buffer0"
    p = tmp_path / "split.csv"
    out.to_csv(p, index=False)
    cmd = [sys.executable, CHECK_SPLIT, str(p), "--grid", GRID_H3_5, "--blocks", BLOCKS, "--holdout-years", "2020",
           "--mask-col", "label_mask_id", "--fold-col", "cv_fold", "--block-col", "block_id"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
    # doi chung: dua mot dong 2020 vao train -> check_split phai bao vi pham
    bad = out.copy()
    i = bad.index[(bad["test_group"] == "thoi_gian")][0]
    bad.loc[i, "split"] = "train"
    bad.to_csv(p, index=False)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert proc.returncode == 1, proc.stdout


@need_real
def test_v4_check_split_cv_folds_hoan_vi_rieng_h3_res_5(real, tmp_path):
    """V4 tren luoi THAT: hoan vi nhan fold rieng h3_res_5 (0->1->...->4->0) -> truoc OK, sau (--cv-folds) ma 1."""
    if not os.path.exists(CHECK_SPLIT):
        pytest.skip("khong co check_split.py")
    blocks, folds, grid, table = real
    fp = tmp_path / "cv_folds.csv"
    folds.to_csv(fp, index=False)
    perm = folds.copy()
    perm.loc[perm["cv_fold"] >= 0, "cv_fold"] = (perm.loc[perm["cv_fold"] >= 0, "cv_fold"] + 1) % 5
    tperm = cell_block_table(grid, blocks, perm)
    for t, name, want in ((table, "dung", 0), (tperm, "hoan_vi", 1)):
        out = assign_eval_split(_season_table(grid["cell_id"], [2019, 2020, 2021]), t)
        out = out.merge(t[["cell_id", "block_id", "cv_fold"]], on="cell_id", how="left")
        out.loc[out["split"] == "test", "cv_fold"] = np.nan
        p = tmp_path / f"split_{name}.csv"
        out.to_csv(p, index=False)
        base = [sys.executable, CHECK_SPLIT, str(p), "--grid", GRID_H3_5, "--blocks", BLOCKS, "--holdout-years", "2020",
                "--fold-col", "cv_fold", "--block-col", "block_id"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        old = subprocess.run(base, capture_output=True, text=True, timeout=300, env=env)
        assert old.returncode == 0, old.stdout                      # khong --cv-folds: khong phat hien
        new = subprocess.run(base + ["--cv-folds", str(fp)], capture_output=True, text=True, timeout=300, env=env)
        assert new.returncode == want, new.stdout


@need_real
def test_cv_tren_luoi_that_val_khong_cham_train(real):
    _, _, grid, table = real
    out = assign_eval_split(_season_table(grid["cell_id"], range(2014, 2027)), table)
    train = out[out["split"] == "train"].reset_index(drop=True)
    assert 2020 not in set(train["season"])
    assert not train["cell_id"].isin(table.loc[table["touches_holdout"], "cell_id"]).any()
    touched = dict(zip(table["cell_id"], parse_touched_folds(table["touched_folds"])))
    fold_of = table.set_index("cell_id")["cv_fold"]
    sp = block_cv_splits(train, table, holdout_seasons=HS, rule="touch")
    assert len(sp) == 5
    for k, (tr, va) in enumerate(sp):
        tr_cells, va_cells = set(train["cell_id"].iloc[tr]), set(train["cell_id"].iloc[va])
        assert not tr_cells & va_cells
        assert all(fold_of[c] == k for c in va_cells)
        assert all(k not in touched[c] for c in tr_cells)      # train fold k khong cham fold k


# ------------------------------------------------------------------ CLI train.py --mode cv|final
@pytest.fixture(scope="module")
def cli_data(tmp_path_factory, real):
    blocks, folds, grid, table = real
    d = tmp_path_factory.mktemp("proc")
    raw = tmp_path_factory.mktemp("raw")
    lab_dir = tmp_path_factory.mktemp("lab")
    rng = np.random.default_rng(3)
    cells = grid["cell_id"].tolist()
    dates = pd.date_range("2017-11-01", "2021-04-29", freq="15D")
    dyn = pd.DataFrame([(c, t.strftime("%Y-%m-%d")) for c in cells for t in dates], columns=["cell_id", "date"])
    dyn["temp_c"] = 27 + rng.normal(0, 1, len(dyn))
    dyn.to_csv(d / "DYNAMIC_MERGE.csv", index=False)
    dem = pd.Series(rng.uniform(0, 3, len(cells)), index=cells)
    pd.DataFrame({"cell_id": cells, "dem_mean": dem.to_numpy()}).to_csv(d / "STATIC_MERGED.csv", index=False)
    lab = pd.DataFrame([(c, s) for c in cells for s in range(2018, 2022)], columns=["cell_id", "season"])
    lab["salinity"] = 4 - lab["cell_id"].map(dem).to_numpy() + 0.1 * (lab["season"] - 2018) + rng.normal(0, 0.1, len(lab))
    lab.loc[lab.index[::37], "salinity"] = np.nan  # vai nhan thieu
    lab_csv = lab_dir / "labels.csv"
    lab.to_csv(lab_csv, index=False)
    folds_csv = lab_dir / "cv_folds.csv"
    folds.to_csv(folds_csv, index=False)
    env = dict(os.environ, OUTPUT_DIR=str(d), RAW_DIR=str(raw), PYTHONIOENCODING="utf-8")
    test_keys = set(map(tuple, lab.merge(table[["cell_id", "touches_holdout"]], on="cell_id")
                        .query("touches_holdout or season == 2020")[["cell_id", "season"]].astype(str).to_numpy()))
    return env, lab_csv, folds_csv, table, test_keys


def _run_mode(cli_data, cwd, name, mode, *extra, no_filter=True):
    # Nhan tong hop khong co train_ok_scope -> phai ghi ro --allow-no-train-filter (mac dinh bat loc).
    env, lab_csv, folds_csv, _, _ = cli_data
    cmd = [sys.executable, os.path.join(ROOT, "src", "training", "train.py"), "--label-csv", str(lab_csv),
           "--mode", mode, "--grid", GRID_H3_5, "--blocks", BLOCKS, "--cv-folds", str(folds_csv),
           "--experiment-name", name, *extra] + (["--allow-no-train-filter"] if no_filter else [])
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


@need_real
def test_cli_mode_khong_co_cot_loc_bao_loi(cli_data, tmp_path):
    proc = _run_mode(cli_data, tmp_path, "e0", "cv", "--model", "linear", no_filter=False)
    assert proc.returncode == 2 and "train_ok_scope" in proc.stderr


@need_real
def test_cli_table_chi_huan_luyen_dong_train_ok_scope(cli_data, tmp_path):
    """--table: moi dong vao CV co train_ok_scope = True; so dong huan luyen khop bang (An 2026-10-04)."""
    env, lab_csv, folds_csv, table, _ = cli_data
    lab = pd.read_csv(lab_csv, dtype={"cell_id": str})
    rng = np.random.default_rng(11)
    t = lab.copy()
    t["dem_mean"] = rng.uniform(0, 3, len(t))
    t["train_ok"] = t["salinity"].notna()
    t["scope_frac"] = np.where(rng.uniform(size=len(t)) < 0.2, 0.0, 0.5)
    t["train_ok_scope"] = t["train_ok"] & (t["scope_frac"] > 0)
    tab = tmp_path / "unified.csv"
    t.to_csv(tab, index=False)
    cmd = [sys.executable, os.path.join(ROOT, "src", "training", "train.py"), "--table", str(tab), "--mode", "cv",
           "--grid", GRID_H3_5, "--blocks", BLOCKS, "--cv-folds", str(folds_csv), "--model", "linear",
           "--experiment-name", "et", "--no-point-eval"]
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "artifacts" / "experiments" / "et" / "cv"
    oof = pd.read_csv(out / "oof_predictions.csv", dtype={"cell_id": str})
    ok = t.set_index(["cell_id", "season"])["train_ok_scope"]
    assert ok.reindex(pd.MultiIndex.from_frame(oof[["cell_id", "season"]])).all()  # khong dong loai nao lot vao
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    sp = assign_eval_split(t[t["train_ok_scope"]], table)
    assert cfg["n_train_rows"] == int((sp["split"] == "train").sum())
    assert cfg["train_filter"]["n_rows_train_col_true"] == int(t["train_ok_scope"].sum())
    assert cfg["train_filter"]["n_rows_label_excluded"] == int((t["salinity"].notna() & ~t["train_ok_scope"]).sum())
    assert cfg["features"] == ["dem_mean"] and cfg["table_sha256"]
    # --table khong kem --mode -> tu choi
    bad = subprocess.run(cmd[:4] + ["--experiment-name", "x"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert bad.returncode == 2


@need_real
def test_cli_mode_cv_chi_dung_train_va_ghi_oof(cli_data, tmp_path):
    _, _, _, table, test_keys = cli_data
    proc = _run_mode(cli_data, tmp_path, "e1", "cv", "--model", "linear", "--seeds", "1", "2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "artifacts" / "experiments" / "e1" / "cv"
    oof = pd.read_csv(out / "oof_predictions.csv", dtype={"cell_id": str})
    assert list(oof.columns) == ["cell_id", "season", "y_true", "y_pred", "fold", "seed", "model", "pred_source"]
    assert 2020 not in set(oof["season"])
    assert not oof["cell_id"].isin(table.loc[table["touches_holdout"], "cell_id"]).any()
    assert (oof["pred_source"] == "oof").all() and set(oof["seed"]) == {1, 2}
    # moi (o, mua) train xuat hien dung 1 lan / seed, fold = fold cua o (theo tam)
    assert not oof.duplicated(["cell_id", "season", "seed"]).any()
    assert (oof["fold"].to_numpy() == oof["cell_id"].map(table.set_index("cell_id")["cv_fold"]).to_numpy()).all()
    assert oof["y_true"].notna().all() and np.isfinite(oof["y_pred"]).all()
    # KHONG co file nao trong thu muc thi nghiem chua (o, mua) cua tap test, khong co final_predictions
    exp = tmp_path / "artifacts" / "experiments" / "e1"
    files = [p for p in exp.rglob("*") if p.is_file()]
    assert not any(p.name.startswith("final") for p in files)
    for p in files:
        if p.suffix == ".csv":
            f = pd.read_csv(p, dtype={"cell_id": str})
            if {"cell_id", "season"} <= set(f.columns):
                keys = set(map(tuple, f[["cell_id", "season"]].astype(str).to_numpy()))
                assert not keys & test_keys, p
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert not [k for k in cfg if "test" in k.lower()]
    assert cfg["split_kind"] == "eval_design_cv" and cfg["eval_split"]["holdout_seasons"] == [2020]
    assert cfg["seeds"] == [1, 2] and len(cfg["cv_summary"]) == 5
    assert "khong_gian" not in proc.stdout and "thoi_gian" not in proc.stdout
    assert cfg["eval_split"]["analysis_kind"] == "thiet_ke_chinh" and cfg["eval_split"]["grid_sha256"]
    # V4: cv_membership.csv dat kiem tra hinh hoc doc lap cua check_split
    mem = pd.read_csv(out / "cv_membership.csv", dtype={"cell_id": str})
    assert list(mem.columns) == ["cell_id", "fold", "role"] and set(mem["role"]) <= {"train", "val", "loai"}
    assert (mem.groupby("cell_id")["role"].apply(lambda r: (r == "val").sum()) == 1).all()
    if os.path.exists(CHECK_SPLIT):
        env, lab_csv, folds_csv, table, _ = cli_data
        lab = pd.read_csv(lab_csv, dtype={"cell_id": str}).dropna(subset=["salinity"])
        sp = assign_eval_split(lab, table)
        sp_csv = tmp_path / "split_cli.csv"
        sp.to_csv(sp_csv, index=False)
        chk = subprocess.run([sys.executable, CHECK_SPLIT, str(sp_csv), "--grid", GRID_H3_5, "--blocks", BLOCKS,
                              "--holdout-years", "2020", "--cv-folds", str(folds_csv), "--membership",
                              str(out / "cv_membership.csv")], capture_output=True, text=True, timeout=300, env=env)
        assert chk.returncode == 0, chk.stdout + chk.stderr


@need_real
def test_cli_mode_final_cham_theo_nhom(cli_data, tmp_path):
    proc = _run_mode(cli_data, tmp_path, "e2", "final", "--model", "hist_gb", "--seeds", "5")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "artifacts" / "experiments" / "e2" / "final"
    fp = pd.read_csv(out / "final_predictions.csv", dtype={"cell_id": str})
    assert set(fp["test_group"]) == {"khong_gian", "thoi_gian", "ca_hai"}
    assert (fp["pred_source"] == "final").all()
    _, _, _, _, test_keys = cli_data
    assert set(map(tuple, fp[["cell_id", "season"]].astype(str).to_numpy())) <= test_keys
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert set(cfg["test_metrics_by_group"]) == {"khong_gian", "thoi_gian", "ca_hai"}
    # V7: HistGB mac dinh tat early stopping, tham so ghi vao config
    assert cfg["model_params"]["early_stopping"] is False and cfg["model_params"]["max_iter"] == 300
    assert cfg["model_params_source"].startswith("default")


@need_real
def test_cli_persistence_final_chi_lich_su_train(cli_data, tmp_path):
    proc = _run_mode(cli_data, tmp_path, "e3", "final", "--model", "persistence")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "artifacts" / "experiments" / "e3" / "final"
    fp = pd.read_csv(out / "final_predictions.csv", dtype={"cell_id": str})
    assert fp.loc[fp["test_group"] != "thoi_gian", "y_pred"].isna().all()
    tg = fp[fp["test_group"] == "thoi_gian"]
    assert tg["y_pred"].notna().mean() > 0.9
    # du doan 2020 = nhan train mua 2019 cua chinh o (khong dung nhan test/2021)
    env, lab_csv, *_ = cli_data
    lab = pd.read_csv(lab_csv, dtype={"cell_id": str}).dropna(subset=["salinity"])
    prev = lab[lab["season"] == 2019].set_index("cell_id")["salinity"]
    t = tg.dropna(subset=["y_pred"])
    has19 = t["cell_id"].isin(prev.index)
    np.testing.assert_allclose(t.loc[has19, "y_pred"], t.loc[has19, "cell_id"].map(prev))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert "khong_gian" in cfg["notes"]["persistence"]
    assert cfg["test_metrics_by_group"]["khong_gian"]["42"] is None


@need_real
def test_cli_persistence_cv_va_toa_do_bi_tu_choi(cli_data, tmp_path):
    proc = _run_mode(cli_data, tmp_path, "e4", "cv", "--model", "persistence")
    assert proc.returncode == 2 and "persistence" in proc.stderr.lower()
    proc = _run_mode(cli_data, tmp_path, "e5", "cv", "--model", "linear", "--include-coords")
    assert proc.returncode == 2 and "toa do" in proc.stderr


@need_real
def test_cli_model_params_file(cli_data, tmp_path):
    pfile = tmp_path / "params.json"
    pfile.write_text(json.dumps({"hist_gb": {"max_iter": 20, "learning_rate": 0.1, "early_stopping": False}}))
    proc = _run_mode(cli_data, tmp_path, "e6", "cv", "--model", "hist_gb", "--model-params", str(pfile))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cfg = json.loads((tmp_path / "artifacts" / "experiments" / "e6" / "cv" / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_params"] == {"max_iter": 20, "learning_rate": 0.1, "early_stopping": False}
    assert cfg["model_params_source"] == f"file:{pfile}"


def test_build_model_mac_dinh_tat_early_stopping():
    from training.train import build_model
    assert build_model("hist_gb", 0).early_stopping is False
    assert build_model("mlp", 0).steps[-1][1].early_stopping is False
    assert build_model("mlp", 0).steps[-1][1].hidden_layer_sizes == (64, 32)
    assert build_model("mlp", 0).steps[-1][1].max_iter == 300
    # RF chot 2026-10-06: tai lap theo seed, n_jobs gioi han RAM
    rf = build_model("random_forest", 7)
    assert rf.random_state == 7 and rf.n_estimators == 200 and rf.min_samples_leaf == 5
    assert rf.max_features == "sqrt" and rf.n_jobs == 4


# ------------------------------------------------------------------ V3: mua giu rieng bat buoc trong CV
def test_v3_bang_khong_cot_split_chua_2020_bi_tu_choi(toy):
    """Truoc: khong cot split -> khong kiem mua -> 2020 vao ca train lan val. Sau: ValueError."""
    grid, blocks, folds = toy
    t = cell_block_table(grid, blocks, folds)
    ok_cells = t.loc[~t["touches_holdout"], "cell_id"]
    df = _season_table(ok_cells, [2019, 2020, 2021])          # khong co cot split
    assert "split" not in df.columns
    with pytest.raises(ValueError, match=r"mua giu rieng \[2020\]"):
        block_cv_splits(df, t, holdout_seasons=HS, min_val_cells=1)
    with pytest.raises(ValueError, match=r"mua giu rieng \[2020\]"):
        block_cv_summary(df, t, holdout_seasons=HS)
    # bo 2020 -> chay duoc
    sp = block_cv_splits(df[df["season"] != 2020].reset_index(drop=True), t, holdout_seasons=HS, min_val_cells=1)
    assert len(sp) == 2


def test_v3_holdout_seasons_bat_buoc(toy):
    grid, blocks, folds = toy
    t = cell_block_table(grid, blocks, folds)
    df = _season_table(t.loc[~t["touches_holdout"], "cell_id"], [2019, 2021])
    with pytest.raises(TypeError):
        block_cv_splits(df, t, min_val_cells=1)                 # thieu tham so bat buoc
    with pytest.raises(TypeError):
        block_cv_summary(df, t)
    with pytest.raises(ValueError, match="holdout_seasons rong"):
        block_cv_splits(df, t, holdout_seasons=(), min_val_cells=1)
    with pytest.raises(ValueError, match="thieu cot 'season'"):
        block_cv_splits(df.drop(columns=["season"]), t, holdout_seasons=HS, min_val_cells=1)
    # mua giu rieng khac (vd 2019) cung bi chan o cap ham
    with pytest.raises(ValueError, match=r"\[2019\]"):
        block_cv_splits(df, t, holdout_seasons=(2019,), min_val_cells=1)


def test_block_cv_cell_summary_khop_summary_theo_dong(toy):
    grid, blocks, folds = toy
    t = cell_block_table(grid, blocks, folds)
    one = _season_table(t.loc[~t["touches_holdout"], "cell_id"], [2019])
    pd.testing.assert_frame_equal(block_cv_cell_summary(t), block_cv_summary(one, t, holdout_seasons=HS))


def test_holdout_season_policy_train_cli():
    from training.train import holdout_season_policy
    assert holdout_season_policy([2020], False) is False
    assert holdout_season_policy([2019, 2020], False) is False
    with pytest.raises(ValueError, match="allow-holdout-season-override"):
        holdout_season_policy([2019], False)
    assert holdout_season_policy([2019], True) is True       # phan tich do nhay, co tuong minh
    with pytest.raises(ValueError, match="rong"):
        holdout_season_policy([], True)


# ------------------------------------------------------------------ V5: provenance cua --cell-table
def _prov_files(tmp_path):
    from preprocessing import file_sha256
    files = {}
    for name in ("blocks.geojson", "grid.geojson", "cv_folds.csv"):
        p = tmp_path / name
        p.write_text(f"noi dung {name}", encoding="utf-8")
        files[name] = p
    table = tmp_path / "cells.csv"
    table.write_text("cell_id\n", encoding="utf-8")
    prov = {"blocks_sha256": file_sha256(files["blocks.geojson"]), "grid_sha256": file_sha256(files["grid.geojson"]),
            "cv_folds_sha256": file_sha256(files["cv_folds.csv"])}
    return files, table, prov


def test_v5_cell_table_provenance(tmp_path):
    from training.train import check_cell_table_provenance
    files, table, prov = _prov_files(tmp_path)
    args = (str(table), str(files["blocks.geojson"]), str(files["grid.geojson"]), str(files["cv_folds.csv"]))
    with pytest.raises(ValueError, match="provenance"):           # chua co provenance
        check_cell_table_provenance(*args)
    pp = tmp_path / "cells.csv.provenance.json"
    pp.write_text(json.dumps(prov), encoding="utf-8")
    assert check_cell_table_provenance(*args)["grid_sha256"] == prov["grid_sha256"]
    with pytest.raises(ValueError, match="can --grid"):
        check_cell_table_provenance(args[0], args[1], None, args[3])
    for key, fname in (("blocks_sha256", "blocks.geojson"), ("grid_sha256", "grid.geojson"),
                       ("cv_folds_sha256", "cv_folds.csv")):
        files[fname].write_text("da sua", encoding="utf-8")          # file thuc dung khac file luc dung bang
        with pytest.raises(ValueError, match=key):
            check_cell_table_provenance(*args)
        files[fname].write_text(f"noi dung {fname}", encoding="utf-8")
    pp.write_text(json.dumps({k: v for k, v in prov.items() if k != "grid_sha256"}), encoding="utf-8")
    with pytest.raises(ValueError, match="thieu 'grid_sha256'"):
        check_cell_table_provenance(*args)


# ------------------------------------------------------------------ V4: check_split --cv-folds / --membership
need_check = pytest.mark.skipif(not os.path.exists(CHECK_SPLIT), reason="khong co check_split.py (.claude/ khong track)")


def _run_check(table, grid, blocks, *extra):
    cmd = [sys.executable, CHECK_SPLIT, str(table), "--grid", str(grid), "--blocks", str(blocks),
           "--holdout-years", "2020", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))


@pytest.fixture()
def toy_files(toy, tmp_path):
    """Ghi luoi/khoi/fold toy + bang chia + membership do code sinh (tat ca hop le)."""
    grid, blocks, folds = toy
    gp, bp, fp = tmp_path / "grid.geojson", tmp_path / "blocks.geojson", tmp_path / "cv_folds.csv"
    grid.to_file(gp, driver="GeoJSON")
    blocks.to_file(bp, driver="GeoJSON")
    folds.to_csv(fp, index=False)

    def build(fold_table):
        t = cell_block_table(grid, blocks, fold_table)
        df = assign_eval_split(_season_table(grid["cell_id"], [2019, 2020, 2021]), t)
        df = df.merge(t[["cell_id", "cv_fold"]], on="cell_id", how="left")
        df.loc[df["split"] == "test", "cv_fold"] = np.nan
        train = df[df["split"] == "train"].reset_index(drop=True)
        sp = block_cv_splits(train, t, holdout_seasons=HS, min_val_cells=1)
        mem = block_cv_membership(train, sp, sorted(set(t.loc[t["cv_fold"] >= 0, "cv_fold"])))
        return df, mem

    df, mem = build(folds)
    tp, mp = tmp_path / "split.csv", tmp_path / "mem.csv"
    df.to_csv(tp, index=False)
    mem.to_csv(mp, index=False)
    return {"grid": gp, "blocks": bp, "folds": fp, "table": tp, "mem": mp, "mem_df": mem,
            "build": build, "tmp": tmp_path}


@need_check
def test_v4_check_split_membership_hop_le(toy_files):
    f = toy_files
    m = f["mem_df"].set_index(["cell_id", "fold"])["role"]
    assert m[("ab_cut", 1)] == "loai" and m[("ab_cut", 0)] == "val" and m[("b_in", 0)] == "train"
    proc = _run_check(f["table"], f["grid"], f["blocks"], "--fold-col", "cv_fold", "--cv-folds", str(f["folds"]),
                      "--membership", str(f["mem"]))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout and "lech 0" in proc.stdout


@need_check
def test_v4_doi_chung_am_o_cham_giu_rieng_trong_train(toy_files):
    """(i) o bh_cut (cham khoi giu rieng, khong cham fold 0) dua vao train fold 0 -> ma 1."""
    f = toy_files
    bad = pd.concat([f["mem_df"], pd.DataFrame({"cell_id": ["bh_cut"], "fold": [0], "role": ["train"]})])
    p = f["tmp"] / "mem_bad1.csv"
    bad.to_csv(p, index=False)
    proc = _run_check(f["table"], f["grid"], f["blocks"], "--cv-folds", str(f["folds"]), "--membership", str(p))
    assert proc.returncode == 1, proc.stdout
    assert "TRAIN giao/cham khoi giu rieng" in proc.stdout


@need_check
def test_v4_doi_chung_am_fold_hoan_vi_rieng_mot_luoi(toy_files):
    """(ii) luoi nay dung fold hoan vi (0<->1) trong khi file dung chung giu nguyen -> ma 1.

    Khong co --cv-folds (hanh vi cu) hoan vi van OK - day la lo hong V4."""
    f = toy_files
    perm = pd.read_csv(f["folds"], dtype={"block_id": str})
    perm["cv_fold"] = perm["cv_fold"].map({0: 1, 1: 0, -1: -1})
    df_p, mem_p = f["build"](perm)
    tp, mp = f["tmp"] / "split_perm.csv", f["tmp"] / "mem_perm.csv"
    df_p.to_csv(tp, index=False)
    mem_p.to_csv(mp, index=False)
    proc = _run_check(tp, f["grid"], f["blocks"], "--fold-col", "cv_fold")
    assert proc.returncode == 0, proc.stdout
    proc = _run_check(tp, f["grid"], f["blocks"], "--fold-col", "cv_fold", "--cv-folds", str(f["folds"]))
    assert proc.returncode == 1 and "KHAC" in proc.stdout, proc.stdout
    proc = _run_check(f["table"], f["grid"], f["blocks"], "--cv-folds", str(f["folds"]), "--membership", str(mp))
    assert proc.returncode == 1, proc.stdout


@need_check
def test_v4_doi_chung_am_o_cat_ngang_khoi_trong_train_fold_no_cham(toy_files):
    """(iii) ab_cut (tam fold 0, cham fold 1) dua vao train fold 1 (nhu quy tac 'centroid') -> ma 1."""
    f = toy_files
    bad = f["mem_df"].copy()
    bad.loc[(bad["cell_id"] == "ab_cut") & (bad["fold"] == 1), "role"] = "train"
    p = f["tmp"] / "mem_bad3.csv"
    bad.to_csv(p, index=False)
    proc = _run_check(f["table"], f["grid"], f["blocks"], "--cv-folds", str(f["folds"]), "--membership", str(p))
    assert proc.returncode == 1, proc.stdout
    assert "fold 1: 1 o TRAIN giao/cham khoi cua chinh fold 1" in proc.stdout


@need_real
def test_cli_v3_holdout_seasons_khong_2020_bi_chan(cli_data, tmp_path):
    """Truoc: --holdout-seasons 2019 cho 2020 vao CV khong bi chan. Sau: ma 2, tru khi co co tuong minh."""
    proc = _run_mode(cli_data, tmp_path, "e7", "cv", "--model", "linear", "--holdout-seasons", "2019")
    assert proc.returncode == 2 and "allow-holdout-season-override" in proc.stderr, proc.stdout + proc.stderr
    proc = _run_mode(cli_data, tmp_path, "e7f", "final", "--model", "linear", "--holdout-seasons", "2019")
    assert proc.returncode == 2
    proc = _run_mode(cli_data, tmp_path, "e8", "cv", "--model", "linear", "--holdout-seasons", "2019",
                     "--allow-holdout-season-override")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cfg = json.loads((tmp_path / "artifacts" / "experiments" / "e8" / "cv" / "config.json").read_text(encoding="utf-8"))
    assert cfg["eval_split"]["analysis_kind"] == "phan_tich_do_nhay"
    assert cfg["eval_split"]["holdout_season_override"] is True and cfg["eval_split"]["holdout_seasons"] == [2019]


@need_real
def test_cli_v5_cell_table_doi_chieu_provenance(cli_data, tmp_path):
    from preprocessing import file_sha256
    env, lab_csv, folds_csv, table, _ = cli_data
    ct = tmp_path / "h3_res_5.csv"
    table.to_csv(ct, index=False)
    prov = {"grid_sha256": file_sha256(GRID_H3_5), "blocks_sha256": file_sha256(BLOCKS),
            "cv_folds_sha256": file_sha256(folds_csv)}
    pp = tmp_path / "h3_res_5.csv.provenance.json"
    pp.write_text(json.dumps(prov), encoding="utf-8")
    proc = _run_mode(cli_data, tmp_path, "e9", "cv", "--model", "linear", "--cell-table", str(ct))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # bang o dung tu file khoi khac (sha khac) -> loi
    pp.write_text(json.dumps(dict(prov, blocks_sha256="0" * 64)), encoding="utf-8")
    proc = _run_mode(cli_data, tmp_path, "e10", "cv", "--model", "linear", "--cell-table", str(ct))
    assert proc.returncode == 2 and "blocks_sha256" in proc.stderr, proc.stdout + proc.stderr
    pp.write_text(json.dumps(dict(prov, grid_sha256=file_sha256(GRID_S2_9))), encoding="utf-8")
    proc = _run_mode(cli_data, tmp_path, "e11", "cv", "--model", "linear", "--cell-table", str(ct))
    assert proc.returncode == 2 and "grid_sha256" in proc.stderr, proc.stdout + proc.stderr


# ------------------------------------------------------------------ phuong an C (cham theo diem)
POINTS = os.path.join(ROOT, "data", "eval", "eval_points.geojson")


@pytest.fixture(scope="module")
def c_data(cli_data, tmp_path_factory):
    """Bang hop nhat TONG HOP tren luoi that h3_res_5 + dap an TONG HOP cho 10.801 diem that."""
    import geopandas as gpd

    env, lab_csv, folds_csv, table, _ = cli_data
    d = tmp_path_factory.mktemp("c")
    rng = np.random.default_rng(21)
    t = pd.read_csv(lab_csv, dtype={"cell_id": str})
    t["dem_mean"] = rng.uniform(0, 3, len(t))
    t["train_ok"] = t["salinity"].notna()
    t["scope_frac"] = np.where(rng.uniform(size=len(t)) < 0.15, 0.0, 0.5)  # ~15% (o, mua) khong huan luyen
    t["train_ok_scope"] = t["train_ok"] & (t["scope_frac"] > 0)
    # tam phan dat: tam hinh hoc + lech ngau nhien theo o (de chung minh IDW dung cot bang, khong dung tam hinh hoc)
    g = gpd.read_file(GRID_H3_5).to_crs(32648)
    off = pd.DataFrame({"cell_id": g["cell_id"].astype(str), "scope_cx": g.geometry.centroid.x + rng.uniform(-3e3, 3e3, len(g)),
                        "scope_cy": g.geometry.centroid.y + rng.uniform(-3e3, 3e3, len(g))})
    t = t.merge(off, on="cell_id", how="left")
    tab = d / "unified.csv"
    t.to_csv(tab, index=False)
    pts = gpd.read_file(POINTS)
    ref = pd.DataFrame([(p, s) for p in pts["point_id"] for s in range(2018, 2022)], columns=["point_id", "season"])
    ref["n_valid_3x3"] = rng.choice([9, 7, 5, 3], size=len(ref), p=[0.6, 0.2, 0.1, 0.1])
    ref["ref_salinity"] = np.where(ref["n_valid_3x3"] >= 5, rng.uniform(0, 4, len(ref)), np.nan)
    ref_csv = d / "points_reference.csv"
    ref.to_csv(ref_csv, index=False)
    return env, tab, ref_csv, folds_csv, table, t


def _run_c(c_data, cwd, name, mode, *extra):
    env, tab, ref_csv, folds_csv, _, _ = c_data
    cmd = [sys.executable, os.path.join(ROOT, "src", "training", "train.py"), "--table", str(tab), "--mode", mode,
           "--grid", GRID_H3_5, "--blocks", BLOCKS, "--cv-folds", str(folds_csv), "--points", POINTS,
           "--points-ref", str(ref_csv), "--experiment-name", name, *extra]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=900)


@need_real
def test_phuong_an_c_cv_va_final_moi_diem_mot_lan(c_data, tmp_path):
    import geopandas as gpd

    from training.evaluate import assign_points_to_cells
    from training.point_eval import point_blocks

    env, tab, ref_csv, folds_csv, table, t = c_data
    proc = _run_c(c_data, tmp_path, "pc", "cv", "--model", "hist_gb", "--seeds", "1", "2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "artifacts" / "experiments" / "pc" / "cv"
    op = pd.read_csv(out / "oof_points.csv", dtype={"point_id": str, "point_cell_id": str, "block_id": str})
    pts = gpd.read_file(POINTS)
    folds = pd.read_csv(folds_csv, dtype={"block_id": str})
    blocks = gpd.read_file(BLOCKS)
    pb = point_blocks(pts, blocks, folds).set_index("point_id")
    ref = pd.read_csv(ref_csv, dtype={"point_id": str})
    valid = ref[ref["n_valid_3x3"] >= 5]
    # (1) moi (diem, mua) cv duoc du doan DUNG MOT LAN moi seed; tap = diem khoi khong giu rieng x mua != 2020
    cv_keys = valid[~valid["point_id"].map(pb["is_holdout"]) & (valid["season"] != 2020)]
    for sd in (1, 2):
        got = op[op["seed"] == sd]
        assert not got.duplicated(["point_id", "season"]).any()
        assert set(zip(got["point_id"], got["season"])) == set(zip(cv_keys["point_id"], cv_keys["season"]))
    # (2) bang mo hinh cua fold chua KHOI CUA DIEM (theo vi tri) - khoi lay lai doc lap tu hinh hoc
    assert (op["fold"] == op["point_id"].map(pb["cv_fold"])).all()
    assert (op["block_id"] == op["point_id"].map(pb["block_id"])).all()
    # (3) khong diem nao du doan bang mo hinh da hoc o chua no: o chua diem khong la "train" cua fold do
    mem = pd.read_csv(out / "cv_membership.csv", dtype={"cell_id": str})
    trained = set(zip(mem.loc[mem["role"] == "train", "cell_id"], mem.loc[mem["role"] == "train", "fold"]))
    assert not any((c, f) in trained for c, f in zip(op["point_cell_id"], op["fold"]))
    # o chua diem theo luat co dinh (canh chung -> cell_id nho nhat)
    cell = assign_points_to_cells(pts, gpd.read_file(GRID_H3_5).astype({"cell_id": str}))
    assert (op["point_cell_id"] == op["point_id"].map(cell)).all()
    # (4) diem o o KHONG huan luyen (train_ok_scope False) van duoc cham
    ok = t.set_index(["cell_id", "season"])["train_ok_scope"]
    in_excluded = ~ok.reindex(pd.MultiIndex.from_arrays([op["point_cell_id"], op["season"]])).to_numpy(dtype=bool)
    assert in_excluded.any()
    # (5) dap an = ref_salinity cua dong >= 5/9
    r = valid.set_index(["point_id", "season"])["ref_salinity"]
    assert np.allclose(op["y_ref"], r.reindex(pd.MultiIndex.from_frame(op[["point_id", "season"]])).to_numpy())
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["point_eval"]["enabled"] and cfg["point_metrics_mean_over_seeds"]["n_seeds"] == 2
    assert cfg["cv_metrics_mean_over_seeds"]["mae"] is not None  # sai so theo o van giu (ket qua phu)
    assert not [k for k in cfg if "test" in k.lower()]

    # final: phan con lai; hop cv + final = moi (diem, mua) co dap an hop le
    proc = _run_c(c_data, tmp_path, "pc", "final", "--model", "hist_gb", "--seeds", "1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    fp = pd.read_csv(tmp_path / "artifacts" / "experiments" / "pc" / "final" / "final_points.csv",
                     dtype={"point_id": str})
    assert not fp.duplicated(["point_id", "season"]).any()
    all_keys = set(zip(valid["point_id"], valid["season"]))
    cv1 = set(zip(op.loc[op["seed"] == 1, "point_id"], op.loc[op["seed"] == 1, "season"]))
    fin = set(zip(fp["point_id"], fp["season"]))
    assert not cv1 & fin and cv1 | fin == all_keys
    hb = fp["point_id"].map(pb["is_holdout"])
    exp_group = np.where(hb & (fp["season"] == 2020), "ca_hai", np.where(hb, "khong_gian", "thoi_gian"))
    assert (fp["test_group"].to_numpy() == exp_group).all()


@need_real
def test_phuong_an_c_table_thieu_points_ref_bao_loi(c_data, tmp_path):
    env, tab, ref_csv, folds_csv, _, _ = c_data
    cmd = [sys.executable, os.path.join(ROOT, "src", "training", "train.py"), "--table", str(tab), "--mode", "cv",
           "--grid", GRID_H3_5, "--blocks", BLOCKS, "--cv-folds", str(folds_csv), "--experiment-name", "nr"]
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 2 and "--points-ref" in proc.stderr



@need_real
def test_phuong_an_c_centroid_bi_tu_choi_som_va_cot_cho_block_stats(c_data, tmp_path):
    proc = _run_c(c_data, tmp_path, "ce", "cv", "--model", "linear", "--cv-rule", "centroid")
    assert proc.returncode == 2 and "centroid" in proc.stderr and "seed" not in proc.stdout  # truoc khi fit
    proc = _run_c(c_data, tmp_path, "cs", "cv", "--model", "linear")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    op = pd.read_csv(tmp_path / "artifacts" / "experiments" / "cs" / "cv" / "oof_points.csv", dtype={"point_id": str})
    assert {"grid", "model", "seed", "point_id", "season", "err", "pred_source", "unit_id"} <= set(op.columns)
    assert set(op["pred_source"]) == {"oof"} and set(op["grid"]) == {"h3_res_5"} and op["unit_id"].notna().all()
    assert np.allclose(op["err"], op["y_pred"] - op["y_ref"])
    folds = pd.read_csv(c_data[3], dtype={"block_id": str}).set_index("block_id")["unit_id"]
    assert (op["unit_id"] == op["block_id"].map(folds)).all()


@need_real
def test_phuong_an_c_bien_the_dap_an_phai_khop(c_data, tmp_path):
    env, tab, ref_csv, folds_csv, _, _ = c_data
    prov = str(ref_csv) + ".provenance.json"
    with open(prov, "w", encoding="utf-8") as f:
        json.dump({"variant": "keepwater"}, f)
    try:
        proc = _run_c(c_data, tmp_path, "vr", "cv", "--model", "linear")
        assert proc.returncode == 2 and "variant" in proc.stderr
        proc = _run_c(c_data, tmp_path, "vr2", "cv", "--model", "linear", "--points-ref-variant", "keepwater")
        assert proc.returncode == 0, proc.stdout + proc.stderr
    finally:
        os.remove(prov)


AUX_6090 = os.path.join(ROOT, "data", "eval", "aux_6090", "eval_points_6090.geojson")


@need_real
@pytest.mark.skipif(not os.path.exists(AUX_6090), reason="thieu data/eval/aux_6090 (scripts/build_aux_points_6090.py)")
def test_cham_phu_tap_diem_con_cua_dap_an(c_data, tmp_path):
    """S2 (2026-10-06): file diem = 349 diem 60/90 THAT, dap an (tong hop) = 10.801 diem chinh + 349 diem them (nhu
    points_reference_keep6090.csv). Khong co --points-subset-of-ref -> LOI (dap an thua diem); co co -> chay duoc ca
    cv lan final, chi cham dung 349 diem (moi (diem, mua) mot lan), config ghi so diem dap an bi bo. Khong co gia
    dinh cung so diem 10.801 trong train.py."""
    env, tab, ref_csv, folds_csv, _, _ = c_data
    aux = gpd.read_file(AUX_6090)
    rng = np.random.default_rng(7)
    extra = pd.DataFrame([(p, s) for p in aux["point_id"] for s in range(2018, 2022)], columns=["point_id", "season"])
    extra["n_valid_3x3"] = rng.choice([9, 4], size=len(extra), p=[0.8, 0.2])
    extra["ref_salinity"] = np.where(extra["n_valid_3x3"] >= 5, rng.uniform(0, 6, len(extra)), np.nan)
    ref_main = pd.read_csv(ref_csv, dtype={"point_id": str})
    ref6090 = tmp_path / "ref6090.csv"
    pd.concat([ref_main, extra], ignore_index=True).to_csv(ref6090, index=False)

    def run(name, mode, *more):
        cmd = [sys.executable, os.path.join(ROOT, "src", "training", "train.py"), "--table", str(tab), "--mode", mode,
               "--grid", GRID_H3_5, "--blocks", BLOCKS, "--cv-folds", str(folds_csv), "--points", AUX_6090,
               "--points-ref", str(ref6090), "--experiment-name", name, "--model", "hist_gb", *more]
        return subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900)

    proc = run("aux0", "cv")
    assert proc.returncode == 2 and "khong co trong file diem" in proc.stderr, proc.stdout + proc.stderr
    valid = extra[extra["n_valid_3x3"] >= 5]
    hb = valid["point_id"].map(aux.set_index("point_id")["is_holdout"].astype(bool))
    proc = run("aux", "cv", "--points-subset-of-ref")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "artifacts" / "experiments" / "aux"
    op = pd.read_csv(out / "cv" / "oof_points.csv", dtype={"point_id": str})
    cv_keys = valid[~hb & (valid["season"] != 2020)]
    assert not op.duplicated(["point_id", "season"]).any()
    assert set(zip(op["point_id"], op["season"])) == set(zip(cv_keys["point_id"], cv_keys["season"]))
    cfg = json.loads((out / "cv" / "config.json").read_text(encoding="utf-8"))
    assert cfg["point_eval"]["points_subset_of_ref"] is True
    assert cfg["point_eval"]["n_ref_points_not_in_points_file"] == ref_main["point_id"].nunique()
    proc = run("aux", "final", "--points-subset-of-ref")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    fp = pd.read_csv(out / "final" / "final_points.csv", dtype={"point_id": str})
    fin_keys = valid[hb | (valid["season"] == 2020)]
    assert set(zip(fp["point_id"], fp["season"])) == set(zip(fin_keys["point_id"], fin_keys["season"]))
    assert fp.loc[fp["point_id"].map(aux.set_index("point_id")["is_holdout"].astype(bool)), "point_id"].nunique() <= 37



@need_real
def test_idw_cv_chi_noi_suy_tu_o_train_cua_fold_cung_mua(c_data, tmp_path):
    """IDW (p=2, k=8 co dinh): du doan diem = IDW tinh tay CHI tu o vai tro 'train' cua fold, CUNG mua, nhan tai
    tam o UTM -> o thuoc fold dang cham va o chua diem khong bao gio duoc dung (chung khong phai 'train')."""
    import geopandas as gpd

    from training.baselines import IDWBaseline
    from training.split import assign_eval_split

    env, tab, ref_csv, folds_csv, table, t = c_data
    proc = _run_c(c_data, tmp_path, "idw", "cv", "--model", "idw")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "artifacts" / "experiments" / "idw" / "cv"
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_params"] == {"p": 2.0, "k": 8}
    op = pd.read_csv(out / "oof_points.csv", dtype={"point_id": str, "point_cell_id": str})
    mem = pd.read_csv(out / "cv_membership.csv", dtype={"cell_id": str})
    cxy = t.drop_duplicates("cell_id").set_index("cell_id")[["scope_cx", "scope_cy"]].rename(
        columns={"scope_cx": "x", "scope_cy": "y"})
    tr_all = assign_eval_split(t[t["train_ok_scope"]], table)
    tr_all = tr_all[tr_all["split"] == "train"]
    rng = np.random.default_rng(0)
    sample = op.iloc[rng.choice(len(op), 60, replace=False)]
    for row in sample.itertuples():
        cells = set(mem.loc[(mem["fold"] == row.fold) & (mem["role"] == "train"), "cell_id"])
        assert row.point_cell_id not in cells
        d = tr_all[tr_all["cell_id"].isin(cells) & (tr_all["season"] == row.season)]
        m = IDWBaseline(2.0, 8).fit(cxy.loc[d["cell_id"], ["x", "y"]].to_numpy(), d["salinity"].to_numpy())
        exp = m.predict(cxy.loc[[row.point_cell_id], ["x", "y"]].to_numpy())[0]
        assert np.isclose(row.y_pred, exp), (row.point_id, row.season)
    # O val (ket qua phu theo o) cung chi tu o train cua fold
    oof = pd.read_csv(out / "oof_predictions.csv", dtype={"cell_id": str})
    assert np.isfinite(oof["y_pred"]).all()


@need_real
def test_idw_final_mua_giu_rieng_nan_va_ngoai_mode_bi_tu_choi(c_data, tmp_path):
    proc = _run_c(c_data, tmp_path, "idwf", "final", "--model", "idw")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    fp = pd.read_csv(tmp_path / "artifacts" / "experiments" / "idwf" / "final" / "final_points.csv")
    assert fp.loc[fp["test_group"].isin(["thoi_gian", "ca_hai"]), "y_pred"].isna().all()
    assert np.isfinite(fp.loc[fp["test_group"] == "khong_gian", "y_pred"]).all()
    env, lab_csv, *_ = c_data
    bad = subprocess.run([sys.executable, os.path.join(ROOT, "src", "training", "train.py"), "--model", "idw",
                          "--label-csv", str(lab_csv), "--provisional-split", "--train-end", "2019", "--val-end",
                          "2020", "--experiment-name", "x"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert bad.returncode == 2 and "--mode" in bad.stderr


def test_touched_folds_kieu_chuoi_bi_tu_choi():
    """Bay kieu (An 2026-10-04): touched chua chuoi "0" thi `0 not in {"0"}` luon dung -> moi o vao train."""
    from training.split import _membership_arrays

    fold = np.array([0, 1])
    with pytest.raises(TypeError):
        _membership_arrays(fold, [{0}, {1}], "touch", "0")
    tr, val = _membership_arrays(fold, [{0}, {0, 1}], "touch", 0)
    assert tr.tolist() == [False, False] and val.tolist() == [True, False]  # o 2 cham fold 0 -> khong train


def test_season_mean_baseline_don_vi():
    from training.baselines import SeasonMeanBaseline

    m = SeasonMeanBaseline().fit([2019, 2019, 2021, 2021], [1.0, 3.0, 10.0, np.nan])
    assert m.predict([2019, 2021, 2020]).tolist()[:2] == [2.0, 10.0] and np.isnan(m.predict([2020])[0])


@need_real
def test_season_mean_cv_chi_tu_o_train_cua_fold_cung_mua(c_data, tmp_path):
    """Baseline rong CHG-18: du doan diem = trung binh nhan o vai tro 'train' cua fold, cung mua."""
    from training.split import assign_eval_split

    env, tab, ref_csv, folds_csv, table, t = c_data
    proc = _run_c(c_data, tmp_path, "sm", "cv", "--model", "season_mean")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "artifacts" / "experiments" / "sm" / "cv"
    op = pd.read_csv(out / "oof_points.csv", dtype={"point_id": str, "point_cell_id": str})
    mem = pd.read_csv(out / "cv_membership.csv", dtype={"cell_id": str})
    tr_all = assign_eval_split(t[t["train_ok_scope"]], table)
    tr_all = tr_all[tr_all["split"] == "train"]
    for (fold, season), g in op.groupby(["fold", "season"]):
        cells = set(mem.loc[(mem["fold"] == fold) & (mem["role"] == "train"), "cell_id"])
        exp = tr_all.loc[tr_all["cell_id"].isin(cells) & (tr_all["season"] == season), "salinity"].mean()
        assert np.allclose(g["y_pred"], exp), (fold, season)

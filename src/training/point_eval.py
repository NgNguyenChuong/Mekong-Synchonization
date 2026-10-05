"""Phuong an C (CHG-06): cham theo DIEM danh gia bang du doan out-of-fold cua o chua diem.

Quy tac (An duyet; test o tests/test_point_eval.py):
  - Khoi cua diem xac dinh theo VI TRI DIEM (hinh khoi UTM chinh xac, eval_design.exact_block_frame), khong
    theo o. Diem tren canh chung nhieu khoi -> block_id nho nhat. Co cot block_id trong file diem -> phai trung.
  - Fold cua diem = cv_fold cua khoi (file cv_folds dung chung 13 luoi); khoi giu rieng -> -1.
  - CV: diem o khoi KHONG giu rieng, mua KHONG giu rieng -> du doan DUNG MOT LAN moi seed boi mo hinh fold
    cua khoi chua diem, ap len dac trung cua o chua diem (o do co the KHONG thuoc tap huan luyen,
    vd train_ok_scope sai - van du doan duoc vi chi can dac trung).
  - Final: diem con lai (khoi giu rieng va/hoac mua giu rieng) -> mo hinh fit toan bo train; nhom
    khong_gian / thoi_gian / ca_hai theo diem.
  - Khong diem nao duoc du doan boi mo hinh da huan luyen tren o chua no (kiem khi chay: loi neu vi pham).
  - Dap an = ref_salinity cua points_reference*.csv: median cua so 3x3 voi >= REF_MIN_VALID/9 pixel hop le
    (CHG-13). Dong n_valid_3x3 < REF_MIN_VALID phai NaN va bi bo; dong >= nguong ma NaN -> loi;
    n_valid_3x3 NaN / ngoai [0, 9] -> loi (CHG-22).
  - O chua diem: evaluate.assign_points_to_cells (canh chung -> cell_id nho nhat). Diem ngoai luoi -> loi.
"""
import geopandas as gpd
import numpy as np
import pandas as pd

REF_MIN_VALID = 5  # = label_season.REF_MIN_VALID (test kiem dong bo)


def point_blocks(points: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, cv_folds: pd.DataFrame,
                 block_km: float = 50.0) -> pd.DataFrame:
    """(point_id, block_id, is_holdout, cv_fold) theo VI TRI diem."""
    from eval_design import exact_block_frame

    if points["point_id"].duplicated().any():
        raise ValueError("point_id trung trong file diem")
    exact = exact_block_frame(blocks, block_km=block_km)
    pts = points[["point_id", "geometry"]].to_crs(exact.crs)
    j = gpd.sjoin(pts, exact[["block_id", "is_holdout", "geometry"]], predicate="intersects", how="left")
    j = j.sort_values(["point_id", "block_id"]).drop_duplicates("point_id")
    if j["block_id"].isna().any():
        raise ValueError(f"{int(j['block_id'].isna().sum())} diem nam ngoai moi khoi")
    out = j[["point_id", "block_id", "is_holdout"]].reset_index(drop=True)
    if len(out) != len(points):  # join "left" + bo trung: khong diem nao duoc mat/nhan doi
        raise ValueError(f"point_blocks: {len(points)} diem vao nhung {len(out)} dong ra")
    out["is_holdout"] = out["is_holdout"].astype(bool)
    if "block_id" in points.columns:
        given = points.set_index("point_id")["block_id"].astype(str)
        bad = out[out["block_id"].astype(str).to_numpy() != given.reindex(out["point_id"]).to_numpy()]
        if len(bad):
            raise ValueError(f"{len(bad)} diem co block_id trong file khac khoi theo vi tri (vd {bad['point_id'].head(3).tolist()})")
    fold_of = cv_folds.set_index(cv_folds["block_id"].astype(str))["cv_fold"].astype(int)
    out["cv_fold"] = out["block_id"].astype(str).map(fold_of)
    if out["cv_fold"].isna().any():
        miss = out.loc[out["cv_fold"].isna(), "block_id"].unique()[:5].tolist()
        raise ValueError(f"Khoi cua diem khong co trong file fold: {miss}")
    out["cv_fold"] = out["cv_fold"].astype(int)
    if "unit_id" in cv_folds.columns:  # don vi kiem dinh (khoi it diem gop vao khoi ke) - block_stats
        out["unit_id"] = out["block_id"].astype(str).map(cv_folds.set_index(cv_folds["block_id"].astype(str))["unit_id"])
        if out["unit_id"].isna().any():  # CHG-22: NaN -> astype(str) thanh "nan" o buoc sau = mot don vi gia
            miss = out.loc[out["unit_id"].isna(), "block_id"].unique()[:5].tolist()
            raise ValueError(f"{int(out['unit_id'].isna().sum())} diem: khoi khong co unit_id trong file fold: {miss}")
    bad = out[(out["is_holdout"] & (out["cv_fold"] != -1)) | (~out["is_holdout"] & (out["cv_fold"] < 0))]
    if len(bad):
        raise ValueError(f"{len(bad)} diem: is_holdout va cv_fold khong nhat quan (khoi giu rieng phai co fold -1)")
    return out


def load_reference(ref: pd.DataFrame, min_valid: int = REF_MIN_VALID) -> pd.DataFrame:
    """(point_id, season, y_ref) chi gom dong hop le theo quy tac median 3x3 >= min_valid/9."""
    need = {"point_id", "season", "ref_salinity", "n_valid_3x3"}
    if not need <= set(ref.columns):
        raise ValueError(f"File dap an thieu cot {sorted(need - set(ref.columns))}")
    if ref.duplicated(["point_id", "season"]).any():
        raise ValueError("File dap an trung (point_id, season)")
    nv = pd.to_numeric(ref["n_valid_3x3"], errors="coerce")
    bad_nv = nv.isna() | (nv < 0) | (nv > 9)
    if bad_nv.any():  # CHG-22: NaN < 5 la False -> truoc day dong NaN bi coi la "du 5/9"
        raise ValueError(f"{int(bad_nv.sum())} dong n_valid_3x3 NaN hoac ngoai [0, 9] (so 3x3)")
    few = ref["n_valid_3x3"] < min_valid
    if ref.loc[few, "ref_salinity"].notna().any():
        raise ValueError(f"{int(ref.loc[few, 'ref_salinity'].notna().sum())} dap an co < {min_valid}/9 pixel hop le "
                         "nhung khong NaN (sai quy tac CHG-13)")
    if ref.loc[~few, "ref_salinity"].isna().any():
        raise ValueError(f"{int(ref.loc[~few, 'ref_salinity'].isna().sum())} dap an du {min_valid}/9 pixel nhung NaN")
    ok = ref[~few]
    return pd.DataFrame({"point_id": ok["point_id"].astype(str).to_numpy(), "season": ok["season"].astype(int).to_numpy(),
                         "y_ref": ok["ref_salinity"].to_numpy(dtype=float)})


def point_frame(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, cv_folds: pd.DataFrame,
                ref: pd.DataFrame, holdout_seasons) -> pd.DataFrame:
    """Moi (diem, mua) co dap an hop le: point_id, season, block_id, cv_fold, is_holdout, cell_id, y_ref, role.

    role: "cv" (khoi khong giu rieng & mua khong giu rieng) hoac nhom test "khong_gian"/"thoi_gian"/"ca_hai".
    """
    from training.evaluate import assign_points_to_cells

    pb = point_blocks(points, blocks, cv_folds)
    cell = assign_points_to_cells(points, grid)
    if cell.isna().any():
        raise ValueError(f"{int(cell.isna().sum())} diem nam ngoai luoi (vd {cell[cell.isna()].index[:3].tolist()})")
    pb["cell_id"] = pb["point_id"].map(cell.astype(str))
    r = load_reference(ref)
    unknown = set(r["point_id"]) - set(pb["point_id"])
    if unknown:
        raise ValueError(f"{len(unknown)} point_id trong dap an khong co trong file diem")
    out = r.merge(pb, on="point_id", how="left", validate="many_to_one")
    held_season = out["season"].isin([int(s) for s in holdout_seasons])
    hb = out["is_holdout"].astype(bool)
    out["role"] = np.select([~hb & ~held_season, hb & ~held_season, ~hb & held_season],
                            ["cv", "khong_gian", "thoi_gian"], default="ca_hai")
    cols = ["point_id", "season", "block_id", *(["unit_id"] if "unit_id" in out else []), "cv_fold", "is_holdout",
            "cell_id", "y_ref", "role"]
    return out[cols].sort_values(["point_id", "season"]).reset_index(drop=True)


def attach_features(pf: pd.DataFrame, cell_features: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Gan dac trung (cell_id, season) cua o chua diem. Thieu dong (o, mua) trong bang -> loi."""
    feats = cell_features[["cell_id", "season", *feature_cols]].copy()
    feats["cell_id"] = feats["cell_id"].astype(str)
    feats["season"] = feats["season"].astype(int)
    if feats.duplicated(["cell_id", "season"]).any():
        raise ValueError("Bang dac trung trung (cell_id, season)")
    out = pf.merge(feats, on=["cell_id", "season"], how="left", validate="many_to_one", indicator=True)
    miss = out["_merge"] != "both"
    if miss.any():
        raise ValueError(f"{int(miss.sum())} (diem, mua) khong co dong (o, mua) trong bang dac trung")
    return out.drop(columns="_merge")


def check_point_not_in_train(pts_fold: pd.DataFrame, train_cells, fold) -> None:
    """Loi neu mot diem cua fold duoc du doan boi mo hinh co o chua diem trong tap huan luyen."""
    leak = set(pts_fold["cell_id"].astype(str)) & set(map(str, train_cells))
    if leak:
        raise ValueError(f"Fold {fold}: {len(leak)} o chua diem nam trong tap huan luyen cua chinh fold do "
                         f"(vd {sorted(leak)[:3]}) - phuong an C vi pham; kiem quy tac cv-rule.")

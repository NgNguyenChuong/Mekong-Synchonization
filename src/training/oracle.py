"""Sai so ly tuong theo khung (CHG-24 (1)): du doan = TB nhan that cua o chua diem, cham tai diem bang dap an.

Khong mo hinh, khong huan luyen. Gan o/khoi/vai tro theo point_eval.point_frame; MAE don vi + cap F1 qua
block_stats (pred_source "oracle", mode cv). Ket qua la CAN THUC NGHIEM (hinh hoc o), chi mo ta.
"""
import numpy as np
import pandas as pd

from training.block_stats import (_arm_level, common_point_set, compare_family, seed_mean_errors,
                                  unit_metrics)
from training.evaluate import point_errors

MODEL = "oracle"
ORACLE_SEED = 0  # mot "cach chia" gia: oracle khong phu thuoc fold


def cell_label_series(labels: pd.DataFrame, value_col: str) -> pd.Series:
    """(cell_id, season) -> nhan o; NaN giu nguyen. Thieu cot / khoa trung -> ValueError."""
    need = {"cell_id", "season", value_col}
    if not need <= set(labels.columns):
        raise ValueError(f"Bang nhan o thieu cot {sorted(need - set(labels.columns))}")
    if labels.duplicated(["cell_id", "season"]).any():
        raise ValueError(f"Bang nhan o trung (cell_id, season): {int(labels.duplicated(['cell_id', 'season']).sum())} dong")
    idx = pd.MultiIndex.from_arrays([labels["cell_id"].astype(str).to_numpy(), labels["season"].astype(int).to_numpy()],
                                    names=["cell_id", "season"])
    return pd.Series(pd.to_numeric(labels[value_col], errors="coerce").to_numpy(dtype=float), index=idx, name=value_col)


def check_reference_coverage(pf: pd.DataFrame, expected_points: int, seasons, nan_seasons=()) -> list:
    """So diem co dap an huu han phai = expected_points; mua co dap an = seasons tru nan_seasons. Lech -> ValueError.

    Tra danh sach mua bi bo (nan_seasons)."""
    n = int(pf["point_id"].nunique())
    if n != int(expected_points):
        raise ValueError(f"So diem co dap an {n} khac ky vong {expected_points}")
    want = sorted(set(int(s) for s in seasons) - set(int(s) for s in nan_seasons))
    got = sorted(pf["season"].astype(int).unique())
    if got != want:
        raise ValueError(f"Mua co dap an {got} khac ky vong {want} (bo khai bao truoc: {sorted(nan_seasons)})")
    return sorted(int(s) for s in nan_seasons)


def oracle_errors(pf: pd.DataFrame, cell_labels: pd.Series, grid_name: str, cell_flag: pd.Series = None) -> pd.DataFrame:
    """Sai so co dau (TB nhan o - dap an) moi (diem, mua) cua point_frame; o khong co nhan mua do -> err NaN.

    cell_flag (tuy chon): (cell_id, season) -> co train_ok/lbl_ok, chi de dem (khong loc)."""
    if pf.duplicated(["point_id", "season"]).any():
        raise ValueError("point_frame trung (point_id, season)")
    pc = pf.drop_duplicates("point_id").set_index("point_id")["cell_id"].astype(str)
    if pf.groupby("point_id")["cell_id"].nunique().gt(1).any():
        raise ValueError("mot diem gan nhieu o")
    truth = pd.Series(pf["y_ref"].to_numpy(dtype=float),
                      index=pd.MultiIndex.from_arrays([pf["point_id"].astype(str), pf["season"].astype(int)]))
    err = point_errors(pc, cell_labels, truth)
    keys = pd.MultiIndex.from_arrays([pf["cell_id"].astype(str).to_numpy(), pf["season"].astype(int).to_numpy()])
    out = pf[["point_id", "season", "role", "cell_id", "y_ref"]].copy()  # cot toi thieu (RAM: 13 luoi x ~130k dong)
    out.insert(0, "grid", grid_name)
    out["y_pred"] = cell_labels.reindex(keys).to_numpy(dtype=float)
    out["err"] = err.to_numpy()
    if cell_flag is not None:
        f = cell_flag.reindex(keys)
        out["cell_flag"] = f.to_numpy()
    return out


def mae_table(err_all: pd.DataFrame, common: pd.MultiIndex) -> pd.DataFrame:
    """Theo luoi: so (diem, mua), so o thieu nhan, MAE diem tren tap rieng / tap chung / tap chung vai tro cv."""
    rows = []
    e = err_all.assign(_chung=pd.MultiIndex.from_frame(err_all[["point_id", "season"]]).isin(common))
    for g, d in e.groupby("grid", sort=True):
        ok = d["err"].notna()
        c = d["_chung"]
        cv = c & (d["role"] == "cv")
        r = {"grid": g, "n_ref": len(d), "n_nan_cell": int((~ok).sum()),
             "n_points_nan_cell": int(d.loc[~ok, "point_id"].nunique()),
             "n_valid": int(ok.sum()), "mae_diem": float(d.loc[ok, "err"].abs().mean()) if ok.any() else np.nan,
             "n_chung": int(c.sum()), "mae_diem_chung": float(d.loc[c, "err"].abs().mean()) if c.any() else np.nan,
             "n_chung_cv": int(cv.sum()),
             "mae_diem_chung_cv": float(d.loc[cv, "err"].abs().mean()) if cv.any() else np.nan}
        if "cell_flag" in d:
            r["n_pred_cell_flag_false"] = int((ok & ~d["cell_flag"].fillna(False).astype(bool)).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def to_err_long(err_all: pd.DataFrame, common: pd.MultiIndex) -> pd.DataFrame:
    """Dong vao block_stats: chi (diem, mua) chung, vai tro cv; model/seed/pred_source = oracle."""
    key = pd.MultiIndex.from_frame(err_all[["point_id", "season"]])
    e = err_all[key.isin(common) & (err_all["role"] == "cv").to_numpy()]
    if e["err"].isna().any():
        raise ValueError("tap chung con sai so NaN")
    return pd.DataFrame({"grid": e["grid"].to_numpy(), "model": MODEL, "seed": ORACLE_SEED,
                         "point_id": e["point_id"].astype(str).to_numpy(), "season": e["season"].astype(int).to_numpy(),
                         "err": e["err"].to_numpy(dtype=float), "pred_source": MODEL})


def common_set(err_all: pd.DataFrame):
    """common_point_set tren MOI luoi co trong err_all (hop le = err huu han)."""
    valid = err_all.assign(valid=np.isfinite(err_all["err"].to_numpy(dtype=float)))[["grid", "point_id", "season", "valid"]]
    return common_point_set(valid)


def unit_mae(err_long: pd.DataFrame, point_unit: pd.Series, min_pts: int = 30) -> pd.DataFrame:
    """MAE muc don vi nhu F1: unit_metrics (TB deu theo mua, bo < min_pts) -> TB co trong so n_pts_mean theo luoi."""
    units, _, dropped = unit_metrics(seed_mean_errors(err_long), point_unit, min_pts=min_pts)
    rows = []
    for g, u in units.groupby("grid", sort=True):
        rows.append({"grid": g, "mae_don_vi": _arm_level(units, (g, MODEL), "mae"),
                     "mae_don_vi_tb_deu": float(u["mae"].mean()), "G": int(len(u)),
                     "n_unit_season_bo": int(len(dropped))})
    return pd.DataFrame(rows)


def oracle_family(err_long: pd.DataFrame, point_unit: pd.Series, pairs: pd.DataFrame, levels: pd.Series, *,
                  cv_units, holdout_units, holdout_seasons, family: str, alpha: float = 0.05,
                  delta_min: float = 0.05, min_pts: int = 30) -> pd.DataFrame:
    """Cap F1 (mode cv, Delta_min tuong doi theo muc) tren sai so oracle; nhan gan tien to 'oracle_' (mo ta)."""
    out = compare_family(err_long, point_unit, pairs, cv_units=cv_units, holdout_units=holdout_units,
                         holdout_seasons=holdout_seasons, mode="cv", delta_min=delta_min, delta_min_kind="rel",
                         alpha=alpha, delta_ref="level_mean", levels=levels, require_practical=True, model=MODEL,
                         family=family, expected_m=len(pairs), min_pts=min_pts)
    out["label"] = "oracle_" + out["label"].astype(str)
    out["pred_source"] = MODEL
    out["ti_le_delta_tren_nguong"] = out["delta_hat"].abs() / out["delta_min_thr"]
    out["ti_le_delta_tuong_doi"] = out["delta_hat"].abs() / out["delta_ref"]
    return out


def ratio_table(pairs_out: pd.DataFrame, group_cols=("target", "blocks", "delta_ref_level")) -> pd.DataFrame:
    """Moi (bien, thiet ke khoi, muc): |Delta_oracle| / Delta_min va |Delta_oracle| / muc tham chieu (max, trung vi)."""
    g = pairs_out.groupby(list(group_cols), sort=True)
    return g.agg(n_cap=("delta_hat", "size"), delta_ref=("delta_ref", "first"), delta_min_thr=("delta_min_thr", "first"),
                 max_abs_delta=("delta_hat", lambda x: float(np.abs(x).max())),
                 max_ti_le_tren_nguong=("ti_le_delta_tren_nguong", "max"),
                 tv_ti_le_tren_nguong=("ti_le_delta_tren_nguong", "median"),
                 max_ti_le_tuong_doi=("ti_le_delta_tuong_doi", "max"),
                 tv_ti_le_tuong_doi=("ti_le_delta_tuong_doi", "median")).reset_index()

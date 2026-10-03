"""
Danh gia model hoi quy do man: MAE, RMSE, R^2 (va MAPE neu target khong
co nhieu gia tri gan 0). Tach rieng danh gia theo train/val/test, va theo
thang/tram/vung khi dataset cho phep.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }

    # MAPE chi co y nghia khi target khong co nhieu gia tri gan 0 (do man
    # co the = 0 ppt o vung ngot hoan toan -> MAPE se bi vo cuc/bien dang).
    near_zero_ratio = float((np.abs(y_true) < 0.5).mean())
    if near_zero_ratio < 0.2:
        nonzero = y_true != 0
        metrics["mape"] = float(
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        )
    else:
        metrics["mape"] = None
        metrics["mape_skipped_reason"] = (
            f"{near_zero_ratio:.0%} gia tri target gan 0 - MAPE khong on dinh, bo qua."
        )

    return metrics


def evaluate_by_group(df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str) -> dict:
    """Tinh metrics rieng cho tung nhom (vd: theo thang, theo tram, theo vung)."""
    result = {}
    for group_value, group_df in df.groupby(group_col):
        result[str(group_value)] = compute_metrics(group_df[y_true_col], group_df[y_pred_col])
    return result


# ---------------------------------------------------------
# DANH GIA THEO DIEM CHUNG (T3-V6): moi khung luoi cham tren cung bo diem
# ---------------------------------------------------------
def assign_points_to_cells(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> pd.Series:
    """Tra ve cell_id chua moi diem (index theo point_id); diem ngoai luoi -> NaN.

    Diem nam dung tren canh chung cua nhieu o: chon cell_id nho nhat theo thu tu
    chuoi - quy tac co dinh, khong phu thuoc thu tu du lieu.
    """
    pts = points[["point_id", "geometry"]].to_crs(grid.crs)
    joined = gpd.sjoin(pts, grid[["cell_id", "geometry"]], predicate="intersects", how="left")
    chosen = joined.groupby("point_id")["cell_id"].min()
    return chosen.reindex(points["point_id"].to_numpy())


def point_errors(point_cell: pd.Series, cell_pred: pd.Series, truth: pd.Series) -> pd.Series:
    """Sai so tuyet doi tai tung diem.

    point_cell: point_id -> cell_id; cell_pred: cell_id -> du doan; truth: point_id -> gia tri tham chieu.
    """
    pred = point_cell.map(cell_pred)
    return (pred - truth.reindex(point_cell.index)).abs()


def paired_wilcoxon(err_a, err_b) -> dict:
    """Wilcoxon theo cap tren sai so tuyet doi cua 2 khung tai cung bo diem."""
    a = pd.Series(err_a, dtype=float).reset_index(drop=True)
    b = pd.Series(err_b, dtype=float).reset_index(drop=True)
    ok = a.notna() & b.notna()
    diff = (a[ok] - b[ok]).to_numpy()
    if len(diff) == 0 or np.all(diff == 0):
        return {"n": int(len(diff)), "statistic": np.nan, "p_value": 1.0}
    res = wilcoxon(a[ok], b[ok])
    return {"n": int(ok.sum()), "statistic": float(res.statistic), "p_value": float(res.pvalue)}


def holm_adjust(p_values) -> np.ndarray:
    """Hieu chinh Holm-Bonferroni tren TOAN BO tap p-value (giu thu tu dau vao)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_median_diff_ci(err_a, err_b, n_boot=2000, alpha=0.05, seed=42) -> dict:
    """Do lon hieu ung: trung vi cua (sai so A - sai so B) tren cac cap, kem khoang tin cay bootstrap."""
    a = pd.Series(err_a, dtype=float).reset_index(drop=True)
    b = pd.Series(err_b, dtype=float).reset_index(drop=True)
    ok = a.notna() & b.notna()
    diff = (a[ok] - b[ok]).to_numpy()
    if len(diff) == 0:
        return {"n": 0, "median_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(diff, size=(n_boot, len(diff)), replace=True), axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"n": int(len(diff)), "median_diff": float(np.median(diff)), "ci_low": float(lo), "ci_high": float(hi)}

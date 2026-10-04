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
    """Sai so CO DAU (du doan - tham chieu) tai tung (point_id, season).

    point_cell: point_id -> cell_id (luoi tinh, khong doi theo mua).
    cell_pred: chi muc (cell_id, season) -> du doan cua o trong mua do.
    truth: chi muc (point_id, season) -> gia tri tham chieu tai diem (pixel nhan 30 m).

    Tra ve Series chi muc (point_id, season) theo dung chi muc cua truth. Diem ngoai luoi,
    o khong co du doan mua do, hoac truth NaN -> NaN (khong dien 0). Lay tri tuyet doi/binh
    phuong o buoc sau (block_stats.seed_mean_errors) - giu dau de con tinh duoc do lech.
    """
    for name, s in (("cell_pred", cell_pred), ("truth", truth)):
        if s.index.nlevels != 2:
            raise ValueError(f"{name} phai co chi muc 2 cap (id, season), nhan {s.index.nlevels} cap")
        if s.index.has_duplicates:
            raise ValueError(f"{name} co chi muc trung lap")
    if point_cell.index.has_duplicates:
        raise ValueError("point_cell co point_id trung lap")
    pids = truth.index.get_level_values(0)
    seasons = truth.index.get_level_values(1)
    cells = point_cell.reindex(pids).to_numpy()
    keys = pd.MultiIndex.from_arrays([cells, seasons])
    pred = cell_pred.reindex(keys).to_numpy(dtype=float)
    out = pd.Series(pred - truth.to_numpy(dtype=float), index=truth.index, name="err")
    out.index = out.index.set_names(["point_id", "season"])
    return out


def _align_by_index(err_a, err_b):
    """Ghep cap THEO CHI MUC (vd (point_id, season)), khong theo vi tri dong.

    Hai chuoi phai co cung tap chi muc, khong trung lap; khac nhau -> ValueError.
    """
    if not isinstance(err_a, pd.Series) or not isinstance(err_b, pd.Series):
        raise TypeError("err_a/err_b phai la pd.Series co chi muc (vd point_id hoac (point_id, season)) "
                        "de ghep cap theo chi muc; mang khong co chi muc bi tu choi")
    if err_a.index.has_duplicates or err_b.index.has_duplicates:
        raise ValueError("chi muc trung lap - khong ghep cap duoc")
    if len(err_a) != len(err_b) or not err_a.index.isin(err_b.index).all():
        only_a = err_a.index.difference(err_b.index)
        only_b = err_b.index.difference(err_a.index)
        raise ValueError(f"hai chuoi khac tap chi muc: {len(only_a)} chi co o A, {len(only_b)} chi co o B "
                         f"(vd {list(only_a[:3])} / {list(only_b[:3])})")
    a = err_a.astype(float)
    b = err_b.astype(float).reindex(a.index)
    return a, b


def paired_wilcoxon(err_a, err_b) -> dict:
    """Wilcoxon theo cap tren sai so tuyet doi cua 2 khung tai cung bo diem (ghep theo chi muc).

    CANH BAO: KHONG dung o cap DIEM de suy dien (p-value/"co y nghia") khi so khung luoi.
    Diem cung khoi/cung o tu tuong quan -> mo phong (thiet ke kiem dinh khoi 2026-10-03):
    sai lam loai I ~0,47 voi danh nghia 0,05. Kiem dinh chinh dung block_stats
    (sign-flip theo khoi + CI t robust theo cum). Ham nay chi con cho mo ta/khao sat.
    """
    a, b = _align_by_index(err_a, err_b)
    ok = a.notna() & b.notna()
    diff = (a[ok] - b[ok]).to_numpy()
    if len(diff) == 0 or np.all(diff == 0):
        return {"n": int(len(diff)), "statistic": np.nan, "p_value": 1.0}
    res = wilcoxon(a[ok], b[ok])
    return {"n": int(ok.sum()), "statistic": float(res.statistic), "p_value": float(res.pvalue)}


def holm_adjust(p_values) -> np.ndarray:
    """Hieu chinh Holm-Bonferroni tren TOAN BO tap p-value (giu thu tu dau vao).

    p NaN -> ValueError: truoc day NaN bi argsort dua xuong cuoi va van tinh nhu p hop le
    ([0.01, nan, 0.04] -> [0.03, 0.08, 0.08]), lam m va thu hang sai. Phep so sanh khong
    tinh duoc p phai duoc xu ly tuong minh (bo khoi ho kem ghi chu) truoc khi goi.
    """
    p = np.asarray(p_values, dtype=float)
    if np.isnan(p).any():
        raise ValueError(f"holm_adjust: {int(np.isnan(p).sum())} p-value NaN - xu ly tuong minh truoc khi hieu chinh")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("holm_adjust: p-value ngoai [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_median_diff_ci(err_a, err_b, n_boot=2000, alpha=0.05, seed=42) -> dict:
    """Trung vi cua (sai so A - sai so B) tren cac cap (ghep theo chi muc), kem CI bootstrap diem.

    CANH BAO: KHONG dung cho suy dien khi so khung luoi. (1) Bootstrap diem coi diem doc lap
    trong khi diem tu tuong quan theo khoi -> CI qua hep (cung ly do Wilcoxon diem co loai I
    ~0,47 trong mo phong). (2) Trung vi chenh lech != chenh lech MAE. Dung
    block_stats.cluster_t_ci tren chenh lech MAE cap khoi. Ham nay chi con cho mo ta.
    """
    a, b = _align_by_index(err_a, err_b)
    ok = a.notna() & b.notna()
    diff = (a[ok] - b[ok]).to_numpy()
    if len(diff) == 0:
        return {"n": 0, "median_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(diff, size=(n_boot, len(diff)), replace=True), axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"n": int(len(diff)), "median_diff": float(np.median(diff)), "ci_low": float(lo), "ci_high": float(hi)}

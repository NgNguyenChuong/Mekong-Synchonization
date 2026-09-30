"""
Danh gia model hoi quy do man: MAE, RMSE, R^2 (va MAPE neu target khong
co nhieu gia tri gan 0). Tach rieng danh gia theo train/val/test, va theo
thang/tram/vung khi dataset cho phep.
"""
import numpy as np
import pandas as pd
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

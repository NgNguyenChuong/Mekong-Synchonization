"""
Hop nhat feature dataset (tu pipeline src/) voi nhan do man quan trac
(tu src/salinity/) thanh 1 dataset training duy nhat.

KHONG fabricate du lieu. Neu file feature/label can thiet khong ton tai,
raise loi ro rang liet ke chinh xac file con thieu (BLOCKED BY REAL DATA).
"""
import os

import pandas as pd

from config import DATA_PROCESSED

DYNAMIC_MERGE_FILE = "DYNAMIC_MERGE.csv"
STATIC_MERGED_FILE = "STATIC_MERGED.csv"
PERIODIC_MERGED_FILE = "PERIODIC_MERGED.csv"


def _read_csv_if_exists(filename: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_PROCESSED, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Tương thích ngược: đổi h3_index sang cell_id ngay lúc nạp
    if "h3_index" in df.columns and "cell_id" not in df.columns:
        df.rename(columns={"h3_index": "cell_id"}, inplace=True)
    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype(str)
    return df


def build_feature_dataset() -> pd.DataFrame:
    """Hop nhat DYNAMIC_MERGE + STATIC_MERGED + PERIODIC_MERGED thanh 1 bang
    feature, key = (cell_id, date) cho phan dynamic/periodic, join them
    static theo cell_id (static khong co date, ap dung cho moi ngay).

    Raises:
        FileNotFoundError: neu khong tim thay file nao trong 3 file tren
            (nghia la pipeline src/ chua duoc chay voi du lieu that).
    """
    dynamic = _read_csv_if_exists(DYNAMIC_MERGE_FILE)
    static = _read_csv_if_exists(STATIC_MERGED_FILE)
    periodic = _read_csv_if_exists(PERIODIC_MERGED_FILE)

    missing = [
        name
        for name, df in [
            (DYNAMIC_MERGE_FILE, dynamic),
            (STATIC_MERGED_FILE, static),
            (PERIODIC_MERGED_FILE, periodic),
        ]
        if df is None
    ]
    if dynamic is None and static is None and periodic is None:
        raise FileNotFoundError(
            "BLOCKED BY REAL DATA: khong tim thay file feature nao trong "
            f"{DATA_PROCESSED}. Can chay main.py (pipeline) voi du lieu raw "
            f"that truoc. File con thieu: {missing}"
        )

    base = dynamic if dynamic is not None else periodic
    if dynamic is not None and periodic is not None:
        base = dynamic.merge(periodic, on=["cell_id", "date"], how="outer")

    if base is None:
        # Chi co static, khong co chieu thoi gian - tra ve nguyen static.
        return static

    if static is not None:
        base = base.merge(static, on="cell_id", how="left")

    return base


def merge_with_salinity_labels(
    feature_df: pd.DataFrame,
    salinity_df: pd.DataFrame,
    tolerance_days: int = 3,
    target_col: str = "salinity",
) -> pd.DataFrame:
    """Ghep nhan do man / bien muc tieu (da map sang cell_id)
    vao feature dataset theo (cell_id, ngay gan nhat trong nguong tolerance_days).

    Dung pd.merge_asof (direction='nearest') theo tung cell_id.

    Args:
        tolerance_days: khoang cach ngay toi da giua ngay feature va ngay
            quan trac de con duoc coi la 'gan nhat hop le'.
        target_col: ten cot muc tieu (default: 'salinity').
    """
    feature_df = feature_df.copy()
    if "h3_index" in feature_df.columns and "cell_id" not in feature_df.columns:
        feature_df.rename(columns={"h3_index": "cell_id"}, inplace=True)
    feature_df["cell_id"] = feature_df["cell_id"].astype(str)
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    salinity_df = salinity_df.copy()
    if "h3_index" in salinity_df.columns and "cell_id" not in salinity_df.columns:
        salinity_df.rename(columns={"h3_index": "cell_id"}, inplace=True)
    salinity_df["cell_id"] = salinity_df["cell_id"].astype(str)

    if "datetime" in salinity_df.columns:
        salinity_df["date"] = pd.to_datetime(salinity_df["datetime"]).dt.normalize()
    elif "date" in salinity_df.columns:
        salinity_df["date"] = pd.to_datetime(salinity_df["date"]).dt.normalize()

    salinity_agg = (
        salinity_df.groupby(["cell_id", "date"])[target_col].mean().reset_index()
    )

    merged_parts = []
    for cell_id, group in feature_df.groupby("cell_id"):
        group = group.sort_values("date")
        labels = salinity_agg[salinity_agg["cell_id"] == cell_id].sort_values("date")

        if labels.empty:
            group = group.copy()
            group[target_col] = pd.NA
            merged_parts.append(group)
            continue

        joined = pd.merge_asof(
            group,
            labels[["date", target_col]],
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days),
        )
        merged_parts.append(joined)

    return pd.concat(merged_parts, ignore_index=True)

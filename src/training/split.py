"""
Chia train/validation/test theo THOI GIAN (khong random split thuan tuy,
vi day la bai toan spatio-temporal - random split se gay data leakage khi
cac ngay gan nhau lot vao ca train lan test).

Ho tro them spatial holdout (giu rieng 1 so tram/vung lam test) de danh
gia kha nang generalization sang vi tri chua thay.
"""
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class SplitConfig:
    """Cau hinh split - luu lai cung experiment de reproducible."""
    date_col: str
    train_end: str   # ngay cuoi cung (inclusive) cua tap train, "YYYY-MM-DD"
    val_end: str      # ngay cuoi cung (inclusive) cua tap validation
    random_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def time_based_split(df: pd.DataFrame, config: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia df thanh (train, val, test) theo moc thoi gian.

        train:      date <= train_end
        validation: train_end < date <= val_end
        test:       date > val_end

    Split boundary duoc luu trong SplitConfig de ghi lai cung experiment
    config (xem src/training/train.py), dam bao reproducible.
    """
    dates = pd.to_datetime(df[config.date_col])
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)

    if train_end >= val_end:
        raise ValueError("train_end phai truoc val_end.")

    train_df = df[dates <= train_end]
    val_df = df[(dates > train_end) & (dates <= val_end)]
    test_df = df[dates > val_end]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def spatial_holdout_split(
    df: pd.DataFrame,
    holdout_cell_ids: list[str] | None = None,
    holdout_h3_indices: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tach rieng cac o trong holdout ra lam spatial test set."""
    ids = holdout_cell_ids if holdout_cell_ids is not None else holdout_h3_indices
    if ids is None:
        raise ValueError("Can truyen holdout_cell_ids hoac holdout_h3_indices.")
    cell_col = "cell_id" if "cell_id" in df.columns else ("h3_index" if "h3_index" in df.columns else None)
    if not cell_col:
        raise KeyError("DataFrame khong co cot cell_id hoac h3_index")
    is_holdout = df[cell_col].isin(ids)
    return df[~is_holdout].reset_index(drop=True), df[is_holdout].reset_index(drop=True)


def split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_total": len(train_df) + len(val_df) + len(test_df),
    }

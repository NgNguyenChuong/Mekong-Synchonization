"""
Test src/training/split.py - time-based split va spatial holdout.

Du lieu trong file nay la SYNTHETIC TEST DATA, KHONG phai du lieu that.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.split import SplitConfig, spatial_holdout_split, split_summary, time_based_split  # noqa: E402


def _synthetic_dataset():
    return pd.DataFrame({
        "cell_id": ["h1", "h1", "h1", "h2", "h2"],
        "date": ["2023-06-01", "2024-02-01", "2024-08-01", "2023-12-31", "2024-05-01"],
        "salinity": [1, 2, 3, 4, 5],
    })


def test_time_based_split_boundaries():
    df = _synthetic_dataset()
    config = SplitConfig(date_col="date", train_end="2023-12-31", val_end="2024-06-30")

    train_df, val_df, test_df = time_based_split(df, config)

    assert set(train_df["date"]) == {"2023-06-01", "2023-12-31"}
    assert set(val_df["date"]) == {"2024-02-01", "2024-05-01"}
    assert set(test_df["date"]) == {"2024-08-01"}

    summary = split_summary(train_df, val_df, test_df)
    assert summary["n_total"] == len(df)


def test_time_based_split_rejects_inverted_boundaries():
    df = _synthetic_dataset()
    config = SplitConfig(date_col="date", train_end="2024-12-31", val_end="2024-01-01")

    with pytest.raises(ValueError):
        time_based_split(df, config)


def test_spatial_holdout_split():
    df = _synthetic_dataset()
    remaining, holdout = spatial_holdout_split(df, holdout_cell_ids=["h2"])

    assert set(remaining["cell_id"]) == {"h1"}
    assert set(holdout["cell_id"]) == {"h2"}
    assert len(remaining) + len(holdout) == len(df)

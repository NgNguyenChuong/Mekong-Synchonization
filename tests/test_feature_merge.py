"""
Test src/dataset.py - hop nhat feature dataset voi nhan do man.

Du lieu trong file nay la SYNTHETIC TEST DATA, KHONG phai du lieu that.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import merge_with_salinity_labels  # noqa: E402


def test_merge_with_salinity_labels_matches_nearest_within_tolerance():
    feature_df = pd.DataFrame({
        "cell_id": ["h1", "h1", "h1", "h2"],
        "date": ["2024-01-01", "2024-01-05", "2024-01-20", "2024-01-01"],
        "rain_mm": [10, 5, 0, 20],
    })

    salinity_df = pd.DataFrame({
        "cell_id": ["h1"],
        "datetime": ["2024-01-06"],  # gan ngay 2024-01-05 (h1), cach 2024-01-01 5 ngay
        "salinity": [3.5],
    })

    merged = merge_with_salinity_labels(feature_df, salinity_df, tolerance_days=3)

    # h1/2024-01-05 cach quan trac (01-06) 1 ngay -> trong tolerance -> co nhan.
    row = merged[(merged["cell_id"] == "h1") & (merged["date"] == pd.Timestamp("2024-01-05"))]
    assert row.iloc[0]["salinity"] == 3.5

    # h1/2024-01-01 cach quan trac 5 ngay -> ngoai tolerance=3 -> khong co nhan.
    row_far = merged[(merged["cell_id"] == "h1") & (merged["date"] == pd.Timestamp("2024-01-01"))]
    assert pd.isna(row_far.iloc[0]["salinity"])

    # h1/2024-01-20 qua xa -> khong co nhan.
    row_very_far = merged[(merged["cell_id"] == "h1") & (merged["date"] == pd.Timestamp("2024-01-20"))]
    assert pd.isna(row_very_far.iloc[0]["salinity"])

    # h2 khong co tram quan trac nao -> toan bo la NaN, KHONG duoc fabricate gia tri.
    row_h2 = merged[merged["cell_id"] == "h2"]
    assert pd.isna(row_h2.iloc[0]["salinity"])


def test_merge_preserves_all_feature_rows():
    feature_df = pd.DataFrame({
        "cell_id": ["h1", "h2", "h3"],
        "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "rain_mm": [1, 2, 3],
    })
    salinity_df = pd.DataFrame({"cell_id": [], "datetime": [], "salinity": []})

    merged = merge_with_salinity_labels(feature_df, salinity_df)

    # Khong co quan trac nao -> tat ca feature row van giu nguyen, chi salinity la NaN.
    assert len(merged) == len(feature_df)
    assert merged["salinity"].isna().all()

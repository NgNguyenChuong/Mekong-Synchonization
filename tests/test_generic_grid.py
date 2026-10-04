"""
Unit test kiem chung pipeline hoat dong voi cell_id bat ky (khong phai H3).
Bao gom ID o vuong ('sq_00012'), token S2 ('s2_48b1'), kinh vi do ('ll_1005_1058').
"""
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_validation import validate_dataset
from dataset import merge_with_salinity_labels
from training.split import SplitConfig, spatial_holdout_split, time_based_split


def test_validation_with_generic_ids():
    df = pd.DataFrame({
        "cell_id": ["sq_00012", "s2_48b1", "ll_1005_1058", ""],
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "salinity": [1.5, 2.0, 3.5, 4.0],
    })
    # Khi grid_type='generic', chuoi rong se bi danh dau invalid_cell_id
    rep = validate_dataset(df, target_col="salinity", grid_type="generic")
    assert rep["invalid_cell_id"] == 1
    assert rep["duplicate_cell_date"] == 0

    # Test phan biet h3 vs generic
    rep_h3 = validate_dataset(df, target_col="salinity", grid_type="h3")
    # Vi cac ID deu khong phai H3 hop le -> bi danh dau loi
    assert rep_h3["invalid_cell_id"] == 4


def test_merge_and_split_with_generic_ids():
    feature_df = pd.DataFrame({
        "cell_id": ["sq_0001", "sq_0001", "s2_48b1", "s2_48b1"],
        "date": ["2024-01-01", "2024-01-05", "2024-01-01", "2024-01-10"],
        "temperature": [28.5, 29.0, 27.8, 28.1],
    })
    label_df = pd.DataFrame({
        "cell_id": ["sq_0001", "s2_48b1"],
        "date": ["2024-01-05", "2024-01-09"],
        "salinity": [3.2, 5.1],
    })

    with pytest.deprecated_call():  # ham cu, chi giu cho test - luong chinh dung merge_season_labels
        merged = merge_with_salinity_labels(feature_df, label_df, tolerance_days=2, target_col="salinity")
    assert "cell_id" in merged.columns
    assert len(merged) == 4
    
    # Kiem tra gia tri ghep dung tolerance
    sq_row = merged[(merged["cell_id"] == "sq_0001") & (merged["date"] == "2024-01-05")]
    assert sq_row["salinity"].iloc[0] == 3.2

    # Spatial holdout split tren cell_id generic
    rem, hold = spatial_holdout_split(merged, holdout_cell_ids=["s2_48b1"])
    assert set(rem["cell_id"]) == {"sq_0001"}
    assert set(hold["cell_id"]) == {"s2_48b1"}

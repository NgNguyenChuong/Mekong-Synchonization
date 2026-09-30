"""
Test src/salinity/loader.py va preprocessing.py - validate schema va lam
sach du lieu quan trac do man.

Du lieu trong file nay la SYNTHETIC TEST DATA, KHONG phai du lieu that.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from salinity.loader import load_salinity_observations  # noqa: E402
from salinity.preprocessing import clean_salinity_observations  # noqa: E402
from salinity.schema import REQUIRED_COLUMNS  # noqa: E402


def test_load_salinity_observations_missing_file_raises(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError, match="BLOCKED BY REAL DATA"):
        load_salinity_observations(missing_path)


def test_load_salinity_observations_missing_columns_raises(tmp_path):
    csv_path = tmp_path / "bad_schema.csv"
    pd.DataFrame({"station_id": ["A"], "salinity": [1.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="thieu cot bat buoc"):
        load_salinity_observations(str(csv_path))


def test_load_salinity_observations_valid_schema(tmp_path):
    csv_path = tmp_path / "good_schema.csv"
    df = pd.DataFrame({col: ["x"] for col in REQUIRED_COLUMNS})
    df.to_csv(csv_path, index=False)

    result = load_salinity_observations(str(csv_path))
    assert list(result.columns) == REQUIRED_COLUMNS


def _synthetic_raw_observations():
    return pd.DataFrame({
        "station_id": ["A", "A", "B", "C", "D"],
        "station_name": ["A", "A", "B", "C", "D"],
        "latitude": [10.0, 10.0, 9.5, 999.0, 9.0],   # C co lat khong hop le
        "longitude": [105.5, 105.5, 105.0, 105.0, 105.0],
        "datetime": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01", "not-a-date"],  # D co date loi
        "salinity": [1.0, 1.0, -5.0, 2.0, 3.0],       # B co salinity am
        "source": ["test", "test", "test", "test", "test"],
    })


def test_clean_salinity_observations_reports_and_drops_invalid_rows():
    raw = _synthetic_raw_observations()
    clean, report = clean_salinity_observations(raw)

    assert report["input_rows"] == 5
    assert report["invalid_datetime"] == 1     # row D
    assert report["invalid_coordinates"] == 1  # row C
    assert report["invalid_salinity"] == 1     # row B
    assert report["duplicate_station_datetime"] == 1  # row A trung voi A
    assert report["output_rows"] == len(clean)
    assert report["total_dropped"] == report["input_rows"] - report["output_rows"]

    # Chi con lai dung 1 record hop le (A, khong tinh ban trung).
    assert len(clean) == 1
    assert clean.iloc[0]["station_id"] == "A"

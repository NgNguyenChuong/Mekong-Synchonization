"""Chuan hoa phan hoi MRC: gan dung nam cho mua kho vat qua 2 nam, 0.0 -> thieu, bo ngay tuong lai."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from fetch_mrc_water_level import coverage, dry_to_long, normalize, wet_to_long  # noqa: E402


def test_dry_season_column_spans_two_calendar_years():
    rows = [{"date_gmt": "2025-11-01", "Min": 1.0, "2022.23": 2.5},
            {"date_gmt": "2026-03-15", "Min": 1.0, "2022.23": 1.2}]
    df = normalize(dry_to_long(rows), "TCH", today="2026-09-30")
    got = dict(zip(df["date"].dt.strftime("%Y-%m-%d"), df["value"]))
    assert got == {"2022-11-01": 2.5, "2023-03-15": 1.2}  # thang 3 thuoc 2023, khong phai 2022


def test_wet_zero_is_missing_and_future_dropped():
    rows = [{"date_gmt": "2026-06-01", "AV_1980": 0.7, "2024": 0.0, "2026": 1.1},
            {"date_gmt": "2026-10-31", "AV_1980": 3.3, "2024": 3.0, "2026": 3.5}]
    df = normalize(wet_to_long(rows), "TCH", today="2026-09-30")
    assert "2026-10-31" not in set(df["date"].dt.strftime("%Y-%m-%d"))
    assert np.isnan(df.loc[df["date"] == pd.Timestamp("2024-06-01"), "value"].iloc[0])
    assert set(df["source_col"]) == {"2024", "2026"}  # bo cot trung binh AV_1980


def test_coverage_uses_nov_to_apr_label_year():
    df = pd.DataFrame({"station": "TCH", "value": [1.0, np.nan, 2.0],
                       "date": pd.to_datetime(["2022-11-01", "2023-04-30", "2023-05-01"])})
    cov = coverage(df)
    assert list(cov["label_year"]) == [2023] and cov["days"].iloc[0] == 2 and cov["valid"].iloc[0] == 1


def test_feb29_template_kept_only_in_leap_years():
    rows = [{"date_gmt": "2026-02-29", "2022.23": 1.0, "2023.24": 2.0}]
    df = normalize(dry_to_long(rows), "TCH", today="2026-09-30")
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2024-02-29"]  # 2023 khong nhuan -> bo

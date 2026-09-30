"""
Test src/salinity/spatial_mapping.py - map toa do quan trac vao o H3.

Du lieu trong file nay la SYNTHETIC TEST DATA (bia ra de test logic),
KHONG phai du lieu do man that.
"""
import os
import sys

import h3
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from salinity.spatial_mapping import map_observations_to_h3, observation_station_coverage  # noqa: E402


def make_synthetic_observations():
    # 2 tram gan Can Tho, 1 tram gan Ca Mau - toa do bia, chi de test logic.
    return pd.DataFrame({
        "station_id": ["ST1", "ST2", "ST3"],
        "latitude": [10.03, 10.04, 9.18],
        "longitude": [105.78, 105.79, 105.15],
        "salinity": [1.2, 1.5, 6.8],
    })


def test_map_observations_to_h3_adds_column():
    df = make_synthetic_observations()
    result = map_observations_to_h3(df, resolution=7)

    assert "cell_id" in result.columns
    assert len(result) == len(df)
    assert all(h3.is_valid_cell(v) for v in result["cell_id"])


def test_nearby_stations_map_to_same_or_neighboring_cell():
    df = make_synthetic_observations()
    result = map_observations_to_h3(df, resolution=6)

    # ST1 va ST2 rat gan nhau (0.01 do ~ 1km) -> nen cung 1 o o resolution 6.
    assert result.loc[0, "cell_id"] == result.loc[1, "cell_id"]
    # ST3 o xa (Ca Mau) -> khac o voi ST1.
    assert result.loc[2, "cell_id"] != result.loc[0, "cell_id"]


def test_observation_station_coverage():
    df = make_synthetic_observations()
    mapped = map_observations_to_h3(df, resolution=6)
    coverage = observation_station_coverage(mapped)

    assert set(coverage.columns) == {"cell_id", "n_records", "n_stations", "salinity_mean"}
    # Tong so ban ghi trong coverage phai bang tong so quan trac dau vao.
    assert coverage["n_records"].sum() == len(df)

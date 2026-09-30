"""
Map cac diem quan trac do man (lat/lon) vao o luoi (cell_id).
"""
import h3
import pandas as pd


def map_observations_to_h3(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Them cot cell_id vao df quan trac, dua tren (latitude, longitude)."""
    df = df.copy()
    df["cell_id"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    return df


def observation_station_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Tong hop so tram quan trac va so ban ghi theo tung o."""
    cell_col = "cell_id" if "cell_id" in df.columns else "h3_index"
    return (
        df.groupby(cell_col)
        .agg(
            n_records=("salinity", "count"),
            n_stations=("station_id", "nunique"),
            salinity_mean=("salinity", "mean"),
        )
        .reset_index()
    )

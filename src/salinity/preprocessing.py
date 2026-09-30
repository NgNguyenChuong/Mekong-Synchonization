"""
Lam sach du lieu do man quan trac. Moi buoc loai bo record deu duoc BAO
CAO SO LUONG cu the - khong am tham drop toan bo (dung nguyen tac o
section "Data validation").
"""
import pandas as pd


def clean_salinity_observations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Lam sach du lieu do man, tra ve (df_sach, report).

    report la dict dem so record bi loai o tung buoc, de log/audit.
    """
    report: dict[str, int] = {}
    n0 = len(df)

    df = df.copy()

    # 1. Parse datetime, loai record khong parse duoc.
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    n_invalid_datetime = df["datetime"].isna().sum()
    df = df.dropna(subset=["datetime"])
    report["invalid_datetime"] = int(n_invalid_datetime)

    # 2. Toa do hop le (WGS84).
    invalid_coords = (
        df["latitude"].isna()
        | df["longitude"].isna()
        | ~df["latitude"].between(-90, 90)
        | ~df["longitude"].between(-180, 180)
    )
    report["invalid_coordinates"] = int(invalid_coords.sum())
    df = df[~invalid_coords]

    # 3. Salinity phai la so va >= 0.
    df["salinity"] = pd.to_numeric(df["salinity"], errors="coerce")
    invalid_salinity = df["salinity"].isna() | (df["salinity"] < 0)
    report["invalid_salinity"] = int(invalid_salinity.sum())
    df = df[~invalid_salinity]

    # 4. Trung lap (cung tram, cung thoi diem).
    dup_mask = df.duplicated(subset=["station_id", "datetime"], keep="first")
    report["duplicate_station_datetime"] = int(dup_mask.sum())
    df = df[~dup_mask]

    report["input_rows"] = n0
    report["output_rows"] = len(df)
    report["total_dropped"] = n0 - len(df)

    return df.reset_index(drop=True), report

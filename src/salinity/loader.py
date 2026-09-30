"""
Doc va validate du lieu do man quan trac tu file CSV theo schema.py.

KHONG fabricate du lieu: neu file khong ton tai hoac thieu cot bat buoc,
raise loi ro rang (FileNotFoundError / ValueError) thay vi tra ve DataFrame
rong hoac gia lap.
"""
import os

import pandas as pd

from .schema import REQUIRED_COLUMNS


def load_salinity_observations(csv_path: str) -> pd.DataFrame:
    """Doc file CSV quan trac do man, validate schema toi thieu.

    Raises:
        FileNotFoundError: neu csv_path khong ton tai.
        ValueError: neu thieu cot bat buoc trong REQUIRED_COLUMNS.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"BLOCKED BY REAL DATA: khong tim thay file du lieu do man tai '{csv_path}'. "
            f"Can file CSV voi cac cot: {REQUIRED_COLUMNS}"
        )

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"File '{csv_path}' thieu cot bat buoc: {missing}. "
            f"Can day du: {REQUIRED_COLUMNS}"
        )

    return df

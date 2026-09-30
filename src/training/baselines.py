"""
Baseline models - PHAI implement va so sanh truoc khi dung model phuc tap
hon (Deep Learning...). Neu model phuc tap khong tot hon baseline thi
khong nen dung.
"""
import numpy as np
import pandas as pd


class ClimatologyBaseline:
    """Du doan = trung binh salinity theo thang (tinh tu tap train).

    Neu thang khong xuat hien trong train (hiem), fallback ve trung binh
    toan bo tap train.
    """

    def __init__(self):
        self.monthly_mean_: dict[int, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, dates: pd.Series, y: pd.Series):
        months = pd.to_datetime(dates).dt.month
        self.global_mean_ = float(y.mean())
        self.monthly_mean_ = y.groupby(months).mean().to_dict()
        return self

    def predict(self, dates: pd.Series) -> np.ndarray:
        months = pd.to_datetime(dates).dt.month
        return months.map(lambda m: self.monthly_mean_.get(m, self.global_mean_)).to_numpy(dtype=float)


class PersistenceBaseline:
    """Du doan ngay hien tai = gia tri quan trac gan nhat truoc do CUNG 1 o H3.

    Chi dung duoc khi dataset co time series lien tuc theo h3_index. Neu 1
    o H3 khong co quan trac truoc do trong tap train, fallback ve trung
    binh toan cuc cua train.
    """

    def __init__(self):
        self.last_known_: dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, cell_id: pd.Series, dates: pd.Series, y: pd.Series):
        self.global_mean_ = float(y.mean())
        df = pd.DataFrame({"cell_id": cell_id, "date": pd.to_datetime(dates), "y": y})
        df = df.sort_values("date")
        self.last_known_ = df.groupby("cell_id")["y"].last().to_dict()
        return self

    def predict(self, cell_id: pd.Series) -> np.ndarray:
        return cell_id.map(lambda h: self.last_known_.get(h, self.global_mean_)).to_numpy(dtype=float)

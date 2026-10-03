"""Chuan bi ma tran dac trung cho mo hinh: xu ly gia tri thieu KHONG ro ri.

- HistGradientBoosting nhan NaN truc tiep -> giu nguyen.
- Mo hinh khac: dien trung vi tinh CHI tren tap huan luyen, them cot co
  "<cot>__missing" cho moi cot co gia tri thieu trong tap huan luyen.
Khong dien 0: voi luong mua, 0 la gia tri co nghia ("khong mua").
"""
import numpy as np
import pandas as pd

NATIVE_NAN_MODELS = {"hist_gb"}


class MissingValueHandler:
    def fit(self, X: pd.DataFrame):
        medians = X.median(numeric_only=True)
        self.dropped_ = [c for c in X.columns if pd.isna(medians.get(c, np.nan))]
        self.columns_ = [c for c in X.columns if c not in self.dropped_]
        self.medians_ = medians[self.columns_]
        self.flag_columns_ = [c for c in self.columns_ if X[c].isna().any()]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.columns_].copy()
        for c in self.flag_columns_:
            out[f"{c}__missing"] = X[c].isna().astype(np.int8)
        return out.fillna(self.medians_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def prepare_matrices(model_name, train_X, *others):
    """Tra ve (train, others..., handler). handler = None neu mo hinh nhan NaN."""
    if model_name in NATIVE_NAN_MODELS:
        return (train_X, *others, None)
    handler = MissingValueHandler().fit(train_X)
    return (handler.transform(train_X), *(handler.transform(o) for o in others), handler)

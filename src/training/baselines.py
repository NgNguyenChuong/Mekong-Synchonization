"""Duong co so - phai so sanh truoc khi ket luan ve mo hinh phuc tap hon.

Bai toan uoc luong khong gian (T3-V5): trung binh toan cuc, climatology, IDW.
Bai toan du bao (T6-V5): persistence.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold, KFold


class GlobalMeanBaseline:
    """Du doan = trung binh tap huan luyen. Muc san tuyet doi."""

    def fit(self, y):
        self.mean_ = float(np.nanmean(np.asarray(y, dtype=float)))
        return self

    def predict(self, n):
        return np.full(int(n), self.mean_)


class ClimatologyBaseline:
    """Du doan = trung binh theo thang cua tap huan luyen (thang khong co -> trung binh chung)."""

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


class IDWBaseline:
    """Noi suy nghich dao khoang cach tu k diem huan luyen gan nhat.

    Toa do phai tinh bang MET (vd. tam o chieu sang UTM). Trong so = 1 / d^p;
    diem trung khop (d = 0) lay dung gia tri do.
    """

    def __init__(self, p=2.0, k=8):
        self.p = p
        self.k = k

    def fit(self, xy, y):
        xy = np.asarray(xy, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(y)
        self.tree_ = cKDTree(xy[keep])
        self.y_ = y[keep]
        return self

    def predict(self, xy):
        xy = np.asarray(xy, dtype=float)
        k = min(self.k, len(self.y_))
        dist, idx = self.tree_.query(xy, k=k)
        dist = dist.reshape(len(xy), k)
        idx = idx.reshape(len(xy), k)
        vals = self.y_[idx]
        exact = dist[:, 0] == 0
        w = 1.0 / np.power(np.where(dist == 0, 1.0, dist), self.p)
        weighted = np.sum(w * vals, axis=1) / np.sum(w, axis=1)
        return np.where(exact, vals[:, 0], weighted)

    @classmethod
    def tune(cls, xy, y, groups=None, p_grid=(1.0, 2.0, 3.0), k_grid=(4, 8, 16), n_splits=5, seed=42):
        """Chon p, k bang kiem dinh cheo CHI tren tap huan luyen.

        Nen truyen `groups` (vd. ma khoi khong gian) de cac fold tach nhau theo khong gian;
        chia ngau nhien se danh gia qua lac quan do tu tuong quan khong gian.
        """
        xy = np.asarray(xy, dtype=float)
        y = np.asarray(y, dtype=float)
        if groups is not None and len(np.unique(groups)) >= n_splits:
            splits = list(GroupKFold(n_splits=n_splits).split(xy, y, groups))
        else:
            splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(xy))
        best, scores = None, {}
        for p in p_grid:
            for k in k_grid:
                errs = []
                for tr, te in splits:
                    m = cls(p, k).fit(xy[tr], y[tr])
                    errs.append(np.nanmean(np.abs(m.predict(xy[te]) - y[te])))
                scores[(p, k)] = float(np.mean(errs))
                if best is None or scores[(p, k)] < scores[best]:
                    best = (p, k)
        model = cls(*best).fit(xy, y)
        model.cv_scores_ = scores
        return model


class PersistenceBaseline:
    """Du doan tai ngay t = gia tri quan trac GAN NHAT TRUOC t cua cung o (bai toan du bao).

    `history` la chuoi da quan trac (co the gom ca giai doan kiem tra, vi chi dung
    gia tri co ngay < t nen khong ro ri). O khong co lich su -> trung binh tap huan luyen.
    """

    def fit(self, cell_id, dates, y):
        self.global_mean_ = float(pd.Series(y).mean())
        self.history_ = self._frame(cell_id, dates, y)
        return self

    @staticmethod
    def _frame(cell_id, dates, y):
        return pd.DataFrame({
            "cell_id": pd.Series(cell_id).astype(str).to_numpy(),
            "date": pd.to_datetime(pd.Series(dates)).to_numpy(),
            "y": pd.Series(y, dtype=float).to_numpy(),
        }).dropna(subset=["y"])

    def predict(self, cell_id, dates, history=None):
        hist = self.history_ if history is None else history
        hist = hist.sort_values("date")
        query = pd.DataFrame({
            "cell_id": pd.Series(cell_id).astype(str).to_numpy(),
            "date": pd.to_datetime(pd.Series(dates)).to_numpy(),
            "_order": np.arange(len(cell_id)),
        }).sort_values("date")
        joined = pd.merge_asof(query, hist, on="date", by="cell_id", direction="backward",
                               allow_exact_matches=False)
        joined = joined.sort_values("_order")
        return joined["y"].fillna(self.global_mean_).to_numpy(dtype=float)

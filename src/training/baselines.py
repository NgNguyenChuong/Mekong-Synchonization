"""Duong co so - phai so sanh truoc khi ket luan ve mo hinh phuc tap hon.

Bai toan uoc luong khong gian (T3-V5): trung binh toan cuc, climatology, IDW.
Bai toan du bao (T6-V5): persistence.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold


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
        """Fit tren MOT lat cat thoi gian (vd mot mua kho).

        Toa do trung lap -> loi: thuong la (o, mua) cua nhieu mua ghep chung, khi do d = 0 va
        "noi suy" thanh lay nhan nam khac cua chinh o (loi V4b). Dung `predict_by_season`.
        """
        xy = np.asarray(xy, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(y)
        pts = xy[keep]
        if len(pts) == 0:
            raise ValueError("IDW: khong co diem huan luyen co nhan.")
        n_dup = len(pts) - len(np.unique(pts, axis=0))
        if n_dup:
            raise ValueError(
                f"IDW: {n_dup} toa do trung lap trong tap fit - neu la nhieu mua cua cung o, "
                "fit theo tung mua (predict_by_season / tune(seasons=...))."
            )
        self.tree_ = cKDTree(pts)
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

    def predict_by_season(self, xy_train, y_train, season_train, xy_query, season_query):
        """Fit rieng tung mua tren diem huan luyen CUA MUA DO roi du doan diem cung mua (V4c).

        Mua truy van khong co diem huan luyen nao -> NaN (khong muon nhan mua khac).
        """
        xy_train = np.asarray(xy_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        xy_query = np.asarray(xy_query, dtype=float)
        s_tr = np.asarray(season_train)
        s_q = np.asarray(season_query)
        out = np.full(len(xy_query), np.nan)
        for s in np.unique(s_q):
            q = s_q == s
            tr = (s_tr == s) & np.isfinite(y_train)
            if tr.any():
                out[q] = type(self)(self.p, self.k).fit(xy_train[tr], y_train[tr]).predict(xy_query[q])
        return out

    @classmethod
    def tune(cls, xy, y, cv_splits=None, p_grid=(1.0, 2.0, 3.0), k_grid=(4, 8, 16), seasons=None,
             groups=None, n_splits=5, allow_group_kfold=False):
        """Chon p, k bang CV khoi khong gian CHI tren tap huan luyen.

        - `cv_splits` (BAT BUOC khi chay that): list (train_idx, val_idx) dung san - fold cap khoi
          50 km dung chung 13 luoi, de moi luoi chon tham so tren cung cach chia.
        - `groups` + `allow_group_kfold=True`: CHI DE TEST. GroupKFold chia nhom theo so mau nen fold
          khac nhau giua cac luoi (soat 2026-10-03 muc 6). Thieu nhom -> LOI.
        - Moi fold phai cho MAE huu han (co it nhat mot diem kiem co nhan va du doan). Fold toan NaN
          (vd mua cua fold kiem khong co trong fold huan luyen) -> LOI, khong lay nanmean am tham.
        - `seasons`: neu co, moi fold fit theo tung mua (predict_by_season); mo hinh tra ve
          CHUA fit (chi mang p, k, cv_scores_) - du doan bang predict_by_season.
          Khong co `seasons`: tra ve mo hinh da fit tren toan bo (xy, y) (mot lat cat thoi gian).
        Thuoc tinh ket qua: cv_scores_ {(p, k): MAE trung binh fold}, cv_n_folds_,
        cv_n_pred_nan_ {(p, k): so diem kiem co nhan nhung khong du doan duoc}.
        """
        xy = np.asarray(xy, dtype=float)
        y = np.asarray(y, dtype=float)
        if cv_splits is not None:
            splits = [(np.asarray(tr), np.asarray(va)) for tr, va in cv_splits]
            if not splits:
                raise ValueError("cv_splits rong.")
        elif groups is not None and allow_group_kfold:
            n_groups = len(np.unique(np.asarray(groups)))
            if n_groups < n_splits:
                raise ValueError(f"IDW.tune: chi {n_groups} nhom < n_splits={n_splits}; giam n_splits hoac truyen cv_splits.")
            splits = list(GroupKFold(n_splits=n_splits).split(xy, y, groups))
        else:
            raise ValueError(
                "IDW.tune: can `cv_splits` (fold cap khoi dung chung 13 luoi); khong chia ngau nhien. "
                "`groups` chi dung trong test (allow_group_kfold=True)."
            )
        s_arr = None if seasons is None else np.asarray(seasons)
        best, scores, n_pred_nan = None, {}, {}
        for p in p_grid:
            for k in k_grid:
                errs, n_nan = [], 0
                for i, (tr, te) in enumerate(splits):
                    m = cls(p, k)
                    if s_arr is None:
                        pred = m.fit(xy[tr], y[tr]).predict(xy[te])
                    else:
                        pred = m.predict_by_season(xy[tr], y[tr], s_arr[tr], xy[te], s_arr[te])
                    has_y = np.isfinite(y[te])
                    ok = has_y & np.isfinite(pred)
                    n_nan += int((has_y & ~ok).sum())
                    if not ok.any():
                        raise ValueError(
                            f"IDW.tune: fold {i} (p={p}, k={k}) khong co diem kiem nao vua co nhan vua du doan "
                            f"duoc - chi {i}/{len(splits)} fold hop le truoc do. Kiem cv_splits/seasons."
                        )
                    errs.append(float(np.mean(np.abs(pred[ok] - y[te][ok]))))
                scores[(p, k)] = float(np.mean(errs))
                n_pred_nan[(p, k)] = n_nan
                if best is None or scores[(p, k)] < scores[best]:
                    best = (p, k)
        model = cls(*best)
        if s_arr is None:
            model.fit(xy, y)
        model.cv_scores_ = scores
        model.cv_n_folds_ = len(splits)
        model.cv_n_pred_nan_ = n_pred_nan
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

"""
Chia train/validation/test theo THOI GIAN (khong random split thuan tuy,
vi day la bai toan spatio-temporal - random split se gay data leakage khi
cac ngay gan nhau lot vao ca train lan test).

Ho tro them spatial holdout (giu rieng 1 so tram/vung lam test) de danh
gia kha nang generalization sang vi tri chua thay.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    """Cau hinh split - luu lai cung experiment de reproducible."""
    date_col: str
    train_end: str   # ngay cuoi cung (inclusive) cua tap train, "YYYY-MM-DD"
    val_end: str      # ngay cuoi cung (inclusive) cua tap validation
    random_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def time_based_split(df: pd.DataFrame, config: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia df thanh (train, val, test) theo moc thoi gian.

        train:      date <= train_end
        validation: train_end < date <= val_end
        test:       date > val_end

    Split boundary duoc luu trong SplitConfig de ghi lai cung experiment
    config (xem src/training/train.py), dam bao reproducible.
    """
    dates = pd.to_datetime(df[config.date_col])
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)

    if train_end >= val_end:
        raise ValueError("train_end phai truoc val_end.")

    train_df = df[dates <= train_end]
    val_df = df[(dates > train_end) & (dates <= val_end)]
    test_df = df[dates > val_end]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def season_sequential_split(
    df: pd.DataFrame, train_end_season: int, val_end_season: int, season_col: str = "season"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia tuan tu theo MUA KHO (don vi (o, mua)) - cau noi tam cho train.py.

        train: season <= train_end_season; val: (train_end, val_end]; test: season > val_end_season

    Khong cat ngang mua kho nhu time_based_split theo ngay. CHUA phai thiet ke danh gia chinh
    (khoi khong gian + nam giu rieng + CV khoi) - phan do cho An chot.
    """
    train_end_season, val_end_season = int(train_end_season), int(val_end_season)
    if train_end_season >= val_end_season:
        raise ValueError("train_end_season phai nho hon val_end_season.")
    s = df[season_col].astype("int64")
    train_df = df[s <= train_end_season]
    val_df = df[(s > train_end_season) & (s <= val_end_season)]
    test_df = df[s > val_end_season]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def spatial_holdout_split(
    df: pd.DataFrame,
    holdout_cell_ids: list[str] | None = None,
    holdout_h3_indices: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tach rieng cac o trong holdout ra lam spatial test set."""
    ids = holdout_cell_ids if holdout_cell_ids is not None else holdout_h3_indices
    if ids is None:
        raise ValueError("Can truyen holdout_cell_ids hoac holdout_h3_indices.")
    cell_col = "cell_id" if "cell_id" in df.columns else ("h3_index" if "h3_index" in df.columns else None)
    if not cell_col:
        raise KeyError("DataFrame khong co cot cell_id hoac h3_index")
    is_holdout = df[cell_col].isin(ids)
    return df[~is_holdout].reset_index(drop=True), df[is_holdout].reset_index(drop=True)


def split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_total": len(train_df) + len(val_df) + len(test_df),
    }


# -----------------------------------------------------------------------------------------------
# Thiet ke danh gia chinh (An duyet 2026-10-03): test = o CHAM khoi giu rieng va/hoac mua 2020;
# CV 5 fold cap khoi 50 km (cv_folds.csv dung chung 13 luoi), o cat ngang khoi theo quy tac "cham".
# -----------------------------------------------------------------------------------------------
TEST_GROUPS = ("khong_gian", "thoi_gian", "ca_hai")


@dataclass
class EvalSplitConfig:
    """Cau hinh chia tap theo thiet ke danh gia - ghi vao config.json moi lan chay."""
    holdout_seasons: tuple = (2020,)
    season_col: str = "season"
    cv_rule: str = "touch"
    min_val_cells: int = 5
    n_folds: int | None = None
    grid: str | None = None
    blocks: str | None = None
    blocks_sha256: str | None = None
    cv_folds: str | None = None
    cv_folds_sha256: str | None = None
    cell_table: str | None = None
    grid_sha256: str | None = None
    # Mua giu rieng khac thiet ke chinh (2020 khong nam trong holdout_seasons) chi duoc phep voi co
    # --allow-holdout-season-override -> analysis_kind = "phan_tich_do_nhay" (khong dung de so khung chinh).
    holdout_season_override: bool = False
    analysis_kind: str = "thiet_ke_chinh"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["holdout_seasons"] = [int(s) for s in self.holdout_seasons]
        return d


def _cell_lookup(df: pd.DataFrame, cell_table: pd.DataFrame, cols) -> pd.DataFrame:
    """Gia tri `cols` cua cell_table theo cell_id cua tung dong df (giu thu tu/chi muc df)."""
    ct = cell_table.copy()
    ct["cell_id"] = ct["cell_id"].astype(str)
    if ct["cell_id"].duplicated().any():
        raise ValueError("cell_table co cell_id trung.")
    ids = df["cell_id"].astype(str)
    unknown = sorted(set(ids) - set(ct["cell_id"]))
    if unknown:
        raise ValueError(f"{len(unknown)} cell_id khong co trong cell_table (vd {unknown[:5]}) - sai luoi?")
    return ct.set_index("cell_id").loc[ids.to_numpy(), list(cols)].set_axis(df.index)


def assign_eval_split(df: pd.DataFrame, cell_table: pd.DataFrame, holdout_seasons=(2020,),
                      season_col: str = "season") -> pd.DataFrame:
    """Them cot `split` (train/test) va `test_group` (khong_gian/thoi_gian/ca_hai; train -> "").

    test = o CHAM khoi giu rieng (cell_table.touches_holdout) HOAC mua thuoc holdout_seasons:
      khong_gian = cham & mua khong giu; thoi_gian = khong cham & mua giu; ca_hai = ca hai.
    Nhom test nao rong -> ValueError (thiet ke khong thuc hien duoc tren bang nay).
    Khong doi thu tu/chi muc dong cua df.
    """
    if not holdout_seasons:
        raise ValueError("holdout_seasons rong - thiet ke chinh giu rieng mua 2020.")
    if df[season_col].isna().any():
        raise ValueError(f"{int(df[season_col].isna().sum())} dong thieu {season_col}.")
    touch = _cell_lookup(df, cell_table, ["touches_holdout"])["touches_holdout"].astype(bool)
    held = df[season_col].astype("int64").isin([int(s) for s in holdout_seasons])
    out = df.copy()
    out["split"] = "train"
    out.loc[touch | held, "split"] = "test"
    out["test_group"] = ""
    out.loc[touch & ~held, "test_group"] = "khong_gian"
    out.loc[~touch & held, "test_group"] = "thoi_gian"
    out.loc[touch & held, "test_group"] = "ca_hai"
    empty = [g for g in TEST_GROUPS if not (out["test_group"] == g).any()]
    if empty:
        raise ValueError(f"Nhom test rong: {empty} (mua giu rieng {list(holdout_seasons)} co trong bang? "
                         "co o cham khoi giu rieng?)")
    if not (out["split"] == "train").any():
        raise ValueError("Tap huan luyen rong.")
    return out


MAIN_HOLDOUT_SEASON = 2020  # mua giu rieng cua thiet ke chinh (An chot 2026-10-03)


def _check_holdout_seasons(df: pd.DataFrame, holdout_seasons, season_col: str) -> None:
    """Loi neu holdout_seasons rong, bang thieu cot mua, hoac co dong thuoc mua giu rieng.

    Kiem TRUC TIEP tren cot mua - khong phu thuoc cot `split` (V3 soat 2026-10-03: bang khong co cot
    split chua 2020 tung lot vao ca train lan val).
    """
    if holdout_seasons is None or len(tuple(holdout_seasons)) == 0:
        raise ValueError("holdout_seasons rong - CV khoi phai biet mua giu rieng (thiet ke chinh: 2020).")
    if season_col not in df.columns:
        raise ValueError(f"Bang train thieu cot '{season_col}' - khong kiem duoc mua giu rieng.")
    s = df[season_col]
    if s.isna().any():
        raise ValueError(f"{int(s.isna().sum())} dong train thieu {season_col}.")
    held = s.astype("int64").isin([int(x) for x in holdout_seasons])
    if held.any():
        bad = sorted(int(x) for x in pd.unique(s[held]))
        raise ValueError(f"{int(held.sum())} dong train thuoc mua giu rieng {bad} - mua giu rieng "
                         "khong duoc vao CV (ca train lan val).")


def _membership_arrays(fold: np.ndarray, touched: list, rule: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(mask train, mask val) cua fold k theo quy tac."""
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError(f"fold k phai la so nguyen, nhan {type(k).__name__}")
    val = fold == k
    tr = np.array([k not in t for t in touched], dtype=bool) if rule == "touch" else ~val
    return tr, val


def _fold_membership(train_df: pd.DataFrame, cell_table: pd.DataFrame, rule: str, *, holdout_seasons,
                     season_col: str = "season"):
    """(cv_fold theo dong, danh sach tap fold cham theo dong) sau khi kiem train_df hop le."""
    from eval_design import parse_touched_folds  # src/ tren sys.path (train.py, test)

    if rule not in ("touch", "centroid"):
        raise ValueError(f"rule phai la 'touch' hoac 'centroid', nhan '{rule}'.")
    if "split" in train_df.columns and (train_df["split"] != "train").any():
        raise ValueError("train_df co dong khong phai split == 'train' (test lot vao CV).")
    _check_holdout_seasons(train_df, holdout_seasons, season_col)
    info = _cell_lookup(train_df, cell_table, ["touches_holdout", "cv_fold", "touched_folds"])
    n_touch = int(info["touches_holdout"].astype(bool).sum())
    if n_touch:
        raise ValueError(f"{n_touch} dong train thuoc o CHAM khoi giu rieng - phai qua assign_eval_split truoc.")
    fold = info["cv_fold"].astype(int).to_numpy()
    if (fold < 0).any():
        raise ValueError(f"{int((fold < 0).sum())} dong train co cv_fold < 0.")
    touched = parse_touched_folds(info["touched_folds"].tolist())
    # Bay kieu (An 2026-10-04): neu phan tu la chuoi "3" con k la so 3 thi `k not in t` LUON dung -> moi o vao
    # train moi fold (ro ri am tham). Ep moi phan tu la so nguyen Python/numpy.
    bad = [t for t in touched if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) for x in t)]
    if bad:
        raise TypeError(f"touched_folds phai la tap so nguyen, gap {bad[:3]}")
    for f, t in zip(fold, touched):  # fold theo tam phai nam trong cac fold cham
        if f not in t:
            raise ValueError("cell_table khong nhat quan: cv_fold khong nam trong touched_folds.")
    return fold, touched


def _summary_rows(fold: np.ndarray, touched: list, cells: np.ndarray, rule: str) -> pd.DataFrame:
    rows = []
    for k in sorted(set(int(x) for x in fold)):
        tr, val = _membership_arrays(fold, touched, rule, k)
        excluded = ~val & ~tr
        rows.append({
            "fold": int(k),
            "n_val_cells": int(pd.unique(cells[val]).size), "n_val_rows": int(val.sum()),
            "n_train_cells": int(pd.unique(cells[tr]).size), "n_train_rows": int(tr.sum()),
            "n_excluded_touch_cells": int(pd.unique(cells[excluded]).size),
            "n_excluded_touch_rows": int(excluded.sum()),
        })
    return pd.DataFrame(rows)


def block_cv_summary(train_df: pd.DataFrame, cell_table: pd.DataFrame, *, holdout_seasons, rule: str = "touch",
                     season_col: str = "season") -> pd.DataFrame:
    """Moi fold: so o/dong val, so o/dong train, so o bi loai khoi train do quy tac 'cham'.

    holdout_seasons BAT BUOC (nhu block_cv_splits): dong thuoc mua giu rieng -> ValueError.
    """
    fold, touched = _fold_membership(train_df, cell_table, rule, holdout_seasons=holdout_seasons,
                                     season_col=season_col)
    return _summary_rows(fold, touched, train_df["cell_id"].astype(str).to_numpy(), rule)


def block_cv_cell_summary(cell_table: pd.DataFrame, rule: str = "touch") -> pd.DataFrame:
    """Nhu block_cv_summary nhung o CAP O (khong co mua): moi o KHONG cham khoi giu rieng tinh mot lan.

    Chi dung cho thong ke luoi (scripts/build_cv_folds.py) - khong dung de chia dong (o, mua).
    """
    from eval_design import parse_touched_folds

    if rule not in ("touch", "centroid"):
        raise ValueError(f"rule phai la 'touch' hoac 'centroid', nhan '{rule}'.")
    t = cell_table[~cell_table["touches_holdout"].astype(bool)]
    fold = t["cv_fold"].astype(int).to_numpy()
    if (fold < 0).any():
        raise ValueError(f"{int((fold < 0).sum())} o huan luyen co cv_fold < 0.")
    return _summary_rows(fold, parse_touched_folds(t["touched_folds"].tolist()),
                         t["cell_id"].astype(str).to_numpy(), rule)


def block_cv_splits(train_df: pd.DataFrame, cell_table: pd.DataFrame, *, holdout_seasons, rule: str = "touch",
                    min_val_cells: int = 5, season_col: str = "season") -> list[tuple[np.ndarray, np.ndarray]]:
    """CV khoi khong gian: [(train_idx, val_idx)] - chi so VI TRI trong train_df, moi fold mot cap.

    - holdout_seasons BAT BUOC: bang co dong thuoc mua giu rieng -> ValueError, KE CA khi bang khong
      co cot `split` (kiem tren cot mua, V3 soat 2026-10-03).
    - val fold k = dong cua o co cv_fold == k (khoi chua tam o) -> moi o huan luyen la val o dung
      MOT fold; du doan out-of-fold cua o lay tu fold cua no.
    - train fold k: rule "touch" = dong cua o KHONG cham fold k (o cat ngang khoi thuoc fold k bi
      loai khoi train fold k); rule "centroid" = cv_fold != k.
    Fold co < min_val_cells o val hoac train rong -> ValueError (khong am tham bo fold).
    train_df chi duoc chua dong split == train (o cham khoi giu rieng -> loi).
    """
    fold, touched = _fold_membership(train_df, cell_table, rule, holdout_seasons=holdout_seasons,
                                     season_col=season_col)
    cells = train_df["cell_id"].astype(str).to_numpy()
    expected = sorted(int(k) for k in pd.unique(cell_table["cv_fold"].astype(int)) if k >= 0)
    present = sorted(set(int(k) for k in fold))
    if present != expected:
        raise ValueError(f"Fold co trong bang train {present} khac fold trong cell_table {expected} "
                         "(fold rong tren bang nay).")
    splits = []
    for k in expected:
        tr, val = _membership_arrays(fold, touched, rule, k)
        n_val_cells = pd.unique(cells[val]).size
        if n_val_cells < min_val_cells:
            raise ValueError(f"Fold {k}: chi {n_val_cells} o val < min_val_cells = {min_val_cells}.")
        if not tr.any():
            raise ValueError(f"Fold {k}: tap train rong sau quy tac '{rule}'.")
        assert not (tr & val).any()
        splits.append((np.flatnonzero(tr), np.flatnonzero(val)))
    return splits


MEMBERSHIP_ROLES = ("train", "val", "loai")


def block_cv_membership(train_df: pd.DataFrame, splits, folds) -> pd.DataFrame:
    """Bang (cell_id, fold, role) tu ket qua block_cv_splits - dau vao cho check_split --membership.

    Moi o cua train_df x moi fold dung mot dong: role train / val / loai (bi loai khoi train fold do
    quy tac 'cham'). O vua co dong train vua co dong val trong cung fold -> ValueError.
    """
    cells = train_df["cell_id"].astype(str).to_numpy()
    all_cells = pd.unique(cells)
    parts = []
    for (tr_idx, va_idx), k in zip(splits, folds):
        tr_c, va_c = set(cells[tr_idx]), set(cells[va_idx])
        both = tr_c & va_c
        if both:
            raise ValueError(f"Fold {k}: {len(both)} o vua train vua val (vd {sorted(both)[:3]}).")
        is_val = pd.Series(all_cells).isin(va_c).to_numpy()
        is_tr = pd.Series(all_cells).isin(tr_c).to_numpy()
        role = np.where(is_val, "val", np.where(is_tr, "train", "loai"))
        parts.append(pd.DataFrame({"cell_id": all_cells, "fold": int(k), "role": role}))
    return pd.concat(parts, ignore_index=True)

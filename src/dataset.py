"""
Hop nhat feature dataset (tu pipeline src/) voi nhan do man quan trac
(tu src/salinity/) thanh 1 dataset training duy nhat.

KHONG fabricate du lieu. Neu file feature/label can thiet khong ton tai,
raise loi ro rang liet ke chinh xac file con thieu (BLOCKED BY REAL DATA).
"""
import os
import sys
import warnings

import pandas as pd

from config import DATA_PROCESSED, DATA_SPECS, PERIODIC_SPECS
from seasons import season_days, season_of

DYNAMIC_MERGE_FILE = "DYNAMIC_MERGE.csv"
STATIC_MERGED_FILE = "STATIC_MERGED.csv"
PERIODIC_MERGED_FILE = "PERIODIC_MERGED.csv"


def _read_csv_if_exists(filename: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_PROCESSED, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Tương thích ngược: đổi h3_index sang cell_id ngay lúc nạp
    if "h3_index" in df.columns and "cell_id" not in df.columns:
        df.rename(columns={"h3_index": "cell_id"}, inplace=True)
    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype(str)
    return df


def _read_periodic_if_enabled() -> pd.DataFrame | None:
    """Doc PERIODIC_MERGED.csv CHI KHI PERIODIC_SPECS khong rong.

    Ro ri R2 (soat 2026-10-03): PERIODIC_SPECS da tat co y (NDVI/NDWI do cung NIR voi nhan)
    nhung file PERIODIC_MERGED.csv cua lan chay cu van co the nam trong OUTPUT_DIR va truoc
    day bi doc bat ke spec. Spec rong -> tu choi doc, canh bao de nguoi dung xoa file cu.
    """
    if PERIODIC_SPECS:
        return _read_csv_if_exists(PERIODIC_MERGED_FILE)
    path = os.path.join(DATA_PROCESSED, PERIODIC_MERGED_FILE)
    if os.path.exists(path):
        print(
            f"[CANH BAO] Bo qua {path}: PERIODIC_SPECS rong (dac trung quang hoc da tat vi ro ri "
            "nhan). File nay la cua lan chay cu - nen xoa.",
            file=sys.stderr, flush=True,
        )
    return None


def build_feature_dataset() -> pd.DataFrame:
    """Hop nhat DYNAMIC_MERGE + STATIC_MERGED + PERIODIC_MERGED thanh 1 bang
    feature, key = (cell_id, date) cho phan dynamic/periodic, join them
    static theo cell_id (static khong co date, ap dung cho moi ngay).

    Raises:
        FileNotFoundError: neu khong tim thay file nao trong 3 file tren
            (nghia la pipeline src/ chua duoc chay voi du lieu that).
    """
    dynamic = _read_csv_if_exists(DYNAMIC_MERGE_FILE)
    static = _read_csv_if_exists(STATIC_MERGED_FILE)
    periodic = _read_periodic_if_enabled()

    missing = [
        name
        for name, df in [
            (DYNAMIC_MERGE_FILE, dynamic),
            (STATIC_MERGED_FILE, static),
            *([(PERIODIC_MERGED_FILE, periodic)] if PERIODIC_SPECS else []),
        ]
        if df is None
    ]
    if dynamic is None and static is None and periodic is None:
        raise FileNotFoundError(
            "BLOCKED BY REAL DATA: khong tim thay file feature nao trong "
            f"{DATA_PROCESSED}. Can chay main.py (pipeline) voi du lieu raw "
            f"that truoc. File con thieu: {missing}"
        )

    base = dynamic if dynamic is not None else periodic
    if dynamic is not None and periodic is not None:
        base = dynamic.merge(periodic, on=["cell_id", "date"], how="outer")

    if base is None:
        # Chi co static, khong co chieu thoi gian - tra ve nguyen static.
        return static

    if static is not None:
        base = base.merge(static, on="cell_id", how="left")

    return base


def season_agg_spec(data_specs: dict | None = None) -> tuple[list[str], list[str]]:
    """(cot tong, cot trung binh) cua bang dac trung ngay, suy tu DATA_SPECS.

    Tong neu spec co `season_agg: "sum"` hoac khoa spec la "rain"; con lai trung binh
    (nhiet, am, buc xa; muc nuoc khi them vao).
    """
    specs = DATA_SPECS if data_specs is None else data_specs
    sum_cols, mean_cols = [], []
    for key, spec in specs.items():
        how = spec.get("season_agg", "sum" if key == "rain" else "mean")
        if how not in ("sum", "mean"):
            raise ValueError(f"DATA_SPECS['{key}'].season_agg phai la 'sum' hoac 'mean' (nhan '{how}')")
        (sum_cols if how == "sum" else mean_cols).append(spec["col_name"])
    return sum_cols, mean_cols


def aggregate_dry_season(feature_df: pd.DataFrame, sum_cols, mean_cols, date_col: str = "date") -> pd.DataFrame:
    """Gop bang dac trung NGAY thanh mot dong moi (cell_id, season) mua kho.

    - Mua lay bang `seasons.season_of` (01/11/(Y-1) .. 29/04/Y); ngay ngoai mua bi bo.
    - `<cot>_n_days` (moi cot khai bao): so ngay trong mua cot DO co gia tri. Ngay vang mat trong
      bang (vd du lieu bat dau 01/01) cung tinh la thieu. Cot chat luong - KHONG phai dac trung.
    - Cot tong (mua): cat gia tri am ve 0 roi cong. Neu `<cot>_n_days` < seasons.season_days(mua)
      -> tong = NaN + canh bao (khong lap so lieu thieu; tong thieu ngay se bi lech thap ma khong
      ai biet - soat 2026-10-03 muc 3). Doi dac ta: truoc day tong cua cac ngay co gia tri.
    - Cot trung binh: bo qua NaN; toan NaN -> NaN (giu nhu cu, mua thieu ngay van co gia tri -
      xem `<cot>_n_days`).
    - Doi dac ta: bo cot `n_days` chung ("ngay co it nhat mot cot") vi che thieu ngay cua tung cot.

    Raises:
        ValueError: cot so chua khai bao cach gop; cot khai bao ca hai kieu hoac khong ton tai;
            trung khoa (cell_id, ngay) - cong trung lam sai tong mua.
    """
    sum_cols, mean_cols = list(sum_cols), list(mean_cols)
    both = set(sum_cols) & set(mean_cols)
    if both:
        raise ValueError(f"Cot vua khai bao tong vua trung binh: {sorted(both)}")
    declared = sum_cols + mean_cols
    absent = [c for c in declared if c not in feature_df.columns]
    if absent:
        raise ValueError(f"Cot khai bao nhung khong co trong bang dac trung: {absent}")
    keys = {"cell_id", "h3_index", date_col, "season"}
    undeclared = [
        c for c in feature_df.columns
        if c not in keys and c not in declared
        and (pd.api.types.is_numeric_dtype(feature_df[c]) or pd.api.types.is_bool_dtype(feature_df[c]))
    ]
    if undeclared:
        raise ValueError(
            f"Cot so chua khai bao cach gop mua (tong hay trung binh): {undeclared}. "
            "Them vao sum_cols/mean_cols hoac bo cot truoc khi gop."
        )

    df = feature_df.rename(columns={"h3_index": "cell_id"}) if "cell_id" not in feature_df.columns else feature_df
    df = df[["cell_id", date_col, *declared]].copy()
    df["cell_id"] = df["cell_id"].astype(str)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    dup = df.duplicated(subset=["cell_id", date_col])
    if dup.any():
        raise ValueError(f"Trung khoa (cell_id, {date_col}) o {int(dup.sum())} dong - khong gop duoc.")

    df["season"] = season_of(df[date_col]).to_numpy()
    df = df[df["season"].notna()].copy()
    df["season"] = df["season"].astype("int64")
    if sum_cols:
        # Mua ERA5 co gia tri am rat nho (~-4e-5) do tru tich luy; clip giu nguyen NaN.
        df[sum_cols] = df[sum_cols].clip(lower=0)

    g = df.groupby(["cell_id", "season"], sort=True)
    parts = []
    if sum_cols:
        parts.append(g[sum_cols].sum(min_count=1))
    if mean_cols:
        parts.append(g[mean_cols].mean())
    n_cols = [f"{c}_n_days" for c in declared]
    parts.append(g[declared].count().astype("int64").set_axis(n_cols, axis=1))
    out = pd.concat(parts, axis=1).reset_index()

    if sum_cols and len(out):
        expected = out["season"].map(lambda y: season_days(int(y)))
        for c in sum_cols:
            short = out[f"{c}_n_days"] < expected
            if short.any():
                ex = out.loc[short, ["cell_id", "season", f"{c}_n_days"]].head(5).to_numpy().tolist()
                ex_txt = ", ".join(f"({cid}, {s}: {n}/{season_days(int(s))} ngay)" for cid, s, n in ex)
                warnings.warn(
                    f"Cot tong '{c}': {int(short.sum())} (o, mua) thieu ngay -> NaN (khong lap so lieu "
                    f"thieu). Vd {ex_txt}",
                    UserWarning, stacklevel=2,
                )
                out.loc[short, c] = float("nan")
    return out[["cell_id", "season", *declared, *n_cols]]


def merge_season_labels(features: pd.DataFrame, labels: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Ghep nhan mua vao bang dac trung mua, CHINH XAC theo (cell_id, season).

    Giu moi dong dac trung (left join); dong khong co nhan -> target NaN (khong dien).
    Moi khoa chi co toi da mot nhan va mot dong dac trung -> mot nhan mua cho dung mot
    dong (thay merge_asof +-3 ngay tung nhan mot nhan thanh toi 7 dong - loi R3).

    Raises:
        KeyError: thieu cot cell_id/season/target_col.
        ValueError: trung khoa (cell_id, season) o nhan hoac dac trung; season thieu;
            bang dac trung da co cot muc tieu.
    """
    for name, df, cols in (("features", features, ["cell_id", "season"]),
                           ("labels", labels, ["cell_id", "season", target_col])):
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise KeyError(f"{name} thieu cot {miss}")
    if target_col in features.columns:
        raise ValueError(f"Bang dac trung da co cot muc tieu '{target_col}' - khong ghep de len.")

    feat = features.copy()
    lab = labels[["cell_id", "season", target_col]].copy()
    for name, df in (("features", feat), ("labels", lab)):
        df["cell_id"] = df["cell_id"].astype(str)
        if df["season"].isna().any():
            raise ValueError(f"{name}: cot season co gia tri thieu - bo ngay ngoai mua kho truoc khi ghep.")
        df["season"] = df["season"].astype("int64")

    dup_lab = lab.duplicated(subset=["cell_id", "season"])
    if dup_lab.any():
        ex = lab.loc[dup_lab, ["cell_id", "season"]].head(3).to_dict("records")
        raise ValueError(f"Nhan trung khoa (cell_id, season) o {int(dup_lab.sum())} dong, vd {ex}")
    dup_feat = feat.duplicated(subset=["cell_id", "season"])
    if dup_feat.any():
        raise ValueError(
            f"Dac trung trung khoa (cell_id, season) o {int(dup_feat.sum())} dong - "
            "gop theo mua (aggregate_dry_season) truoc khi ghep."
        )
    return feat.merge(lab, on=["cell_id", "season"], how="left", validate="one_to_one")


def build_season_feature_dataset(sum_cols=None, mean_cols=None) -> pd.DataFrame:
    """Bang dac trung don vi (cell_id, season): gop DYNAMIC_MERGE theo mua kho roi ghep STATIC.

    Cach gop mac dinh lay tu DATA_SPECS (`season_agg_spec`); chi gop cot thuc co trong file.
    Khong ho tro dac trung dinh ky (PERIODIC): bo chinh khong dung dac trung quang hoc.
    """
    if PERIODIC_SPECS:
        raise ValueError(
            "PERIODIC_SPECS khong rong: bo dac trung mua chinh khong dung dac trung quang hoc "
            "(ro ri nhan). Tat periodic_specs.json hoac viet duong rieng cho thi nghiem kiem ro ri."
        )
    if sum_cols is None and mean_cols is None:
        sum_cols, mean_cols = season_agg_spec()
    sum_cols, mean_cols = list(sum_cols or []), list(mean_cols or [])

    dynamic = _read_csv_if_exists(DYNAMIC_MERGE_FILE)
    if dynamic is None:
        raise FileNotFoundError(
            f"BLOCKED BY REAL DATA: khong co {DYNAMIC_MERGE_FILE} trong {DATA_PROCESSED}. "
            "Can chay main.py voi du lieu ERA5 truoc."
        )
    _read_periodic_if_enabled()  # chi de canh bao neu con file PERIODIC cu
    present = set(dynamic.columns)
    out = aggregate_dry_season(
        dynamic, [c for c in sum_cols if c in present], [c for c in mean_cols if c in present]
    )
    static = _read_csv_if_exists(STATIC_MERGED_FILE)
    if static is not None:
        out = out.merge(static, on="cell_id", how="left", validate="many_to_one")
    return out


def merge_with_salinity_labels(
    feature_df: pd.DataFrame,
    salinity_df: pd.DataFrame,
    tolerance_days: int = 3,
    target_col: str = "salinity",
) -> pd.DataFrame:
    """[DEPRECATED] Dung aggregate_dry_season + merge_season_labels.

    Ghep nhan theo (cell_id, ngay gan nhat trong nguong tolerance_days) bang merge_asof.
    Voi dac trung NGAY va nhan MUA, mot nhan bi nhan thanh toi 2*tolerance_days+1 dong
    (nhan ban gia mau - loi R3 soat 2026-10-03). Chi con giu cho test cu, khong dung
    trong luong chinh.

    Args:
        tolerance_days: khoang cach ngay toi da giua ngay feature va ngay
            quan trac de con duoc coi la 'gan nhat hop le'.
        target_col: ten cot muc tieu (default: 'salinity').
    """
    warnings.warn(
        "merge_with_salinity_labels (merge_asof +-ngay) nhan ban nhan mua; "
        "dung aggregate_dry_season + merge_season_labels.",
        DeprecationWarning,
        stacklevel=2,
    )
    feature_df = feature_df.copy()
    if "h3_index" in feature_df.columns and "cell_id" not in feature_df.columns:
        feature_df.rename(columns={"h3_index": "cell_id"}, inplace=True)
    feature_df["cell_id"] = feature_df["cell_id"].astype(str)
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    salinity_df = salinity_df.copy()
    if "h3_index" in salinity_df.columns and "cell_id" not in salinity_df.columns:
        salinity_df.rename(columns={"h3_index": "cell_id"}, inplace=True)
    salinity_df["cell_id"] = salinity_df["cell_id"].astype(str)

    if "datetime" in salinity_df.columns:
        salinity_df["date"] = pd.to_datetime(salinity_df["datetime"]).dt.normalize()
    elif "date" in salinity_df.columns:
        salinity_df["date"] = pd.to_datetime(salinity_df["date"]).dt.normalize()

    salinity_agg = (
        salinity_df.groupby(["cell_id", "date"])[target_col].mean().reset_index()
    )

    merged_parts = []
    for cell_id, group in feature_df.groupby("cell_id"):
        group = group.sort_values("date")
        labels = salinity_agg[salinity_agg["cell_id"] == cell_id].sort_values("date")

        if labels.empty:
            group = group.copy()
            group[target_col] = pd.NA
            merged_parts.append(group)
            continue

        joined = pd.merge_asof(
            group,
            labels[["date", target_col]],
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days),
        )
        merged_parts.append(joined)

    return pd.concat(merged_parts, ignore_index=True)

"""Bang hop nhat don vi (cell_id, season) cho mot luoi x mot bo nhan (tuan 3 viec 3).

Nguon (moi nguon da kiem doc lap rieng):
  nhan     labels/<luoi>_labels_season[_<bo>].csv      cell_id, season, salinity, n_valid_px, valid_frac,
                                                        train_ok, train_ok_10pct
  ERA5     era5_season/<luoi>_era5_season_filled.csv   cell_id, season, 6 bien, era5_cover_frac, era5_fill_frac
  tinh     static/<luoi>_static.csv                    cell_id, dem_mean, dist_*, scope_frac, landcover_class_*
  thuy van hydro_season_2014_2026.csv                  season, tch_wl_p20c, zos_coast_p90, ... (cap vung)
  hybrid   hybrid/<luoi>_hybrid.csv                    cell_id, season, dist_mouth_river_km, zos_mouth_p90,
                                                        sluice_frac, cot kiem
Quy tac ghep (KHONG doan, KHONG dien):
  - Tap khoa (cell_id, season) cua nhan, ERA5, hybrid phai TRUNG NHAU tuyet doi; tinh phu dung moi cell_id
    (mot dong/o); thuy van phu dung moi season. Thieu/thua/trung khoa -> ValueError.
  - Ten cot trung giua cac nguon (ngoai khoa) -> ValueError (khong de pandas tu them hau to).
  - Gia tri NaN giu nguyen (HistGB nhan NaN; mo hinh khac dien trung vi tren tap huan luyen - features.py).
Dac trung tinh tinh tren pixel scope v3 cho MOI bo nhan (bo phu chi khac nhan - CHG-15).
Tap huan luyen: TRAIN_COL = train_ok VA scope_frac > 0 VA (neu co cot) scope_n_px > 0 (An duyet 2026-10-04): o khong co pixel dat scope v3
co NaN o MOI dac trung tinh + hybrid -> mau NaN tiet lo "o toan nuoc/rung ngap man" (ro ri kieu scope_frac).
Ap cho ca 4 bo (bo chinh khong co dong nao bi loai them).
Vai tro cot: KEY_COLS, TARGET_COL, QUALITY_COLS (chat luong/co - khong phai dac trung, chi loc/phan tang),
con lai la ung vien dac trung; dac trung thuc dung do DEFAULT_ALLOWED_FEATURES (training/features.py) quyet.
"""
import numpy as np
import pandas as pd

KEY_COLS = ("cell_id", "season")
TARGET_COL = "salinity"
TRAIN_COL = "train_ok_scope"
LABEL_SETS = ("", "keepwater", "keepmangrove", "keep6090")  # "" = bo chinh
HYDRO_COLS = ("tch_wl_p20c", "zos_coast_p90", "tch_wl_ok", "zos_ok")
# Cot chat luong / co / kiem: KHONG phai dac trung (mau cam trong features.py bat phan lon; phan con lai
# khong nam trong danh sach cho phep).
QUALITY_COLS = ("n_valid_px", "valid_frac", "train_ok", "train_ok_10pct", TRAIN_COL, "era5_cover_frac", "era5_fill_frac",
                "scope_frac", "tch_wl_ok", "zos_ok", "sluice_frac_from2021", "graph_lateral_km",
                "zos_cmems_far_frac", "scope_cx", "scope_cy", "scope_n_px")
# scope_cx/scope_cy: tam pixel dat scope cua o (EPSG:32648) - CHI de dat nhan o cho IDW, KHONG phai dac trung
# (toa do bi cam trong bo chinh).


def strict_bool(series: pd.Series, name: str) -> pd.Series:
    """Bool chat: chi nhan True/False (bool, 0/1, "True"/"False" bat ke hoa thuong). NaN hoac gia tri khac ->
    ValueError. Tranh bay pandas: astype(bool) bien NaN va chuoi "False" thanh True (An 2026-10-04)."""
    mapping = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    if series.isna().any():
        raise ValueError(f"Cot '{name}' co {int(series.isna().sum())} gia tri NaN")
    out = series.map(lambda v: v if isinstance(v, (bool,)) or type(v).__name__ == "bool_" else
                     mapping.get(str(v).strip().lower()))
    if out.isna().any():
        raise ValueError(f"Cot '{name}' co gia tri khong phai True/False: {sorted(series[out.isna()].astype(str).unique())[:5]}")
    return out.astype(bool)


def label_suffix(label_set: str) -> str:
    if label_set not in LABEL_SETS:
        raise ValueError(f"Bo nhan '{label_set}' khong thuoc {LABEL_SETS}")
    return f"_{label_set}" if label_set else ""


def _check_keys(df, keys, name):
    miss = [k for k in keys if k not in df.columns]
    if miss:
        raise ValueError(f"{name}: thieu cot khoa {miss}")
    if df[list(keys)].isna().any().any():
        raise ValueError(f"{name}: khoa {list(keys)} co NaN")
    dup = df.duplicated(list(keys))
    if dup.any():
        raise ValueError(f"{name}: {int(dup.sum())} dong trung khoa {list(keys)}")


def _normalize(df):
    out = df.copy()
    if "cell_id" in out.columns:
        out["cell_id"] = out["cell_id"].astype(str)
    if "season" in out.columns:
        out["season"] = out["season"].astype("int64")
    return out


def _same_keyset(a, b, name_a, name_b):
    ka = pd.MultiIndex.from_frame(a[list(KEY_COLS)])
    kb = pd.MultiIndex.from_frame(b[list(KEY_COLS)])
    only_a, only_b = ka.difference(kb), kb.difference(ka)
    if len(only_a) or len(only_b):
        raise ValueError(f"Tap khoa (cell_id, season) khac nhau: {len(only_a)} chi co o {name_a} "
                         f"(vd {list(only_a[:3])}), {len(only_b)} chi co o {name_b} (vd {list(only_b[:3])})")


def merge_sources(labels, era5, static, hydro, hybrid, hydro_cols=HYDRO_COLS) -> pd.DataFrame:
    """Ghep 5 nguon thanh bang (cell_id, season). Xem quy tac o docstring module."""
    labels, era5, static, hydro, hybrid = (_normalize(d) for d in (labels, era5, static, hydro, hybrid))
    for df, keys, name in ((labels, KEY_COLS, "nhan"), (era5, KEY_COLS, "ERA5"), (hybrid, KEY_COLS, "hybrid"),
                           (static, ("cell_id",), "tinh"), (hydro, ("season",), "thuy van")):
        _check_keys(df, keys, name)
    if TARGET_COL not in labels.columns:
        raise ValueError(f"nhan: thieu cot muc tieu '{TARGET_COL}'")
    _same_keyset(labels, era5, "nhan", "ERA5")
    _same_keyset(labels, hybrid, "nhan", "hybrid")
    miss_cells = sorted(set(labels["cell_id"]) - set(static["cell_id"]))
    if miss_cells:
        raise ValueError(f"tinh: thieu {len(miss_cells)} o (vd {miss_cells[:3]})")
    miss_seasons = sorted(set(labels["season"]) - set(hydro["season"]))
    if miss_seasons:
        raise ValueError(f"thuy van: thieu mua {miss_seasons}")
    absent = [c for c in hydro_cols if c not in hydro.columns]
    if absent:
        raise ValueError(f"thuy van: thieu cot {absent}")
    hydro = hydro[["season", *hydro_cols]]

    seen = {c: "nhan" for c in labels.columns}
    for df, name in ((era5, "ERA5"), (static, "tinh"), (hydro, "thuy van"), (hybrid, "hybrid")):
        clash = [c for c in df.columns if c not in KEY_COLS and c in seen]
        if clash:
            raise ValueError(f"Cot trung ten giua {name} va {[seen[c] for c in clash]}: {clash}")
        seen.update({c: name for c in df.columns})

    out = labels.merge(era5, on=list(KEY_COLS), how="left", validate="one_to_one")
    out = out.merge(static, on="cell_id", how="left", validate="many_to_one")
    out = out.merge(hydro, on="season", how="left", validate="many_to_one")
    out = out.merge(hybrid, on=list(KEY_COLS), how="left", validate="one_to_one")
    if len(out) != len(labels):
        raise ValueError(f"So dong doi sau ghep: {len(labels)} -> {len(out)}")
    if "train_ok" not in out.columns or "scope_frac" not in out.columns:
        raise ValueError("Can cot train_ok (nhan) va scope_frac (tinh) de tao " + TRAIN_COL)
    if out["scope_frac"].isna().any():  # CHG-22: NaN > 0 la False -> truoc day loai lang khoi tap huan luyen
        raise ValueError(f"{int(out['scope_frac'].isna().sum())} dong scope_frac NaN (o khong co trong bang tinh?)")
    out[TRAIN_COL] = strict_bool(out["train_ok"], "train_ok") & (out["scope_frac"] > 0)
    if "scope_n_px" in out.columns:
        # O chi co manh dat < 1 pixel (khong tam pixel dat nao trong o, scope_frac ~1e-4): dac trung tinh tren manh
        # vun = cung loai "o khong co pixel dat hop le" (2026-10-04; chi xay ra o bo phu).
        out[TRAIN_COL] &= out["scope_n_px"].fillna(0) > 0
    quality = [c for c in QUALITY_COLS if c in out.columns]
    rest = [c for c in out.columns if c not in (*KEY_COLS, TARGET_COL, *quality)]
    return out[[*KEY_COLS, TARGET_COL, *quality, *rest]].sort_values(list(KEY_COLS)).reset_index(drop=True)


def nan_report(table, feature_cols, mask_col=TRAIN_COL) -> pd.DataFrame:
    """So NaN theo cot dac trung tren toan bang va tren dong `mask_col` == True."""
    m = table[mask_col].astype(bool) if mask_col in table.columns else pd.Series(True, index=table.index)
    rows = [{"column": c, "nan_all": int(table[c].isna().sum()), f"nan_{mask_col}": int(table.loc[m, c].isna().sum())}
            for c in feature_cols]
    return pd.DataFrame(rows)


def finite_or_nan(series) -> bool:
    """True neu cot chi gom so huu han hoac NaN (khong +-inf)."""
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return not np.isinf(v).any()


# ---------------------------------------------------------------- Dot 7: bang hop nhat theo bien muc tieu
SAL_LABEL_COLS = ("salinity", "n_valid_px", "valid_frac", "train_ok", "train_ok_10pct")
SAL_TRAIN_COL = "train_ok_scope_sal"   # co huan luyen cua bo do man (giu de doi chieu; mau 'train_ok' cam lam dac trung)


def target_table(base: pd.DataFrame, labels: pd.DataFrame, target_col: str, label_cols=()) -> tuple[pd.DataFrame, list]:
    """Bang hop nhat cho bien muc tieu khac do man (E5d Dot 7) tu bang hop nhat bo chinh do man.

    - Tap khoa (cell_id, season) cua nhan moi phai TRUNG bang goc (thieu/thua/trung -> ValueError).
    - Bo cot nhan do man (SAL_LABEL_COLS); TRAIN_COL goc doi ten SAL_TRAIN_COL.
    - Bo moi cot khop mau cam THEO BIEN (training.features.TARGET_FORBIDDEN - cung dai luong vat ly).
    - TRAIN_COL moi = SAL_TRAIN_COL VA nhan bien moi huu han (An: "train_ok_scope nhu cu ∧ nhan bien hop le").
    Tra (bang, danh sach cot da bo vi cam theo bien).
    """
    from training.features import find_target_leak_columns

    base, labels = _normalize(base), _normalize(labels)
    _check_keys(base, KEY_COLS, "bang goc")
    _check_keys(labels, KEY_COLS, "nhan moi")
    _same_keyset(base, labels, "bang goc", "nhan moi")
    if target_col not in labels.columns:
        raise ValueError(f"nhan moi thieu cot muc tieu '{target_col}'")
    miss = [c for c in (*SAL_LABEL_COLS, TRAIN_COL) if c not in base.columns]
    if miss:
        raise ValueError(f"bang goc thieu cot {miss} (can bang hop nhat bo chinh do man)")
    keep_lab = [*KEY_COLS, target_col, *label_cols]
    clash = [c for c in keep_lab[2:] if c in base.columns and c not in SAL_LABEL_COLS]
    if clash:
        raise ValueError(f"Cot nhan moi trung ten cot bang goc: {clash}")
    b = base.drop(columns=list(SAL_LABEL_COLS)).rename(columns={TRAIN_COL: SAL_TRAIN_COL})
    cand = [c for c in b.columns if c not in (*KEY_COLS, SAL_TRAIN_COL)]
    dropped = [c for c, _ in find_target_leak_columns(cand, target_col)]
    b = b.drop(columns=dropped)
    out = b.merge(labels[keep_lab], on=list(KEY_COLS), how="left", validate="one_to_one")
    if len(out) != len(base):
        raise ValueError(f"So dong doi sau ghep: {len(base)} -> {len(out)}")
    out[TRAIN_COL] = strict_bool(out[SAL_TRAIN_COL], SAL_TRAIN_COL) & out[target_col].notna()
    quality = [*label_cols, TRAIN_COL, SAL_TRAIN_COL] + [c for c in QUALITY_COLS if c in out.columns
                                                         and c not in (*label_cols, TRAIN_COL)]
    rest = [c for c in out.columns if c not in (*KEY_COLS, target_col, *quality)]
    out = out[[*KEY_COLS, target_col, *quality, *rest]].sort_values(list(KEY_COLS)).reset_index(drop=True)
    return out, dropped

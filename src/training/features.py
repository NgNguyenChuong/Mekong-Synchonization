"""Chuan bi ma tran dac trung cho mo hinh: xu ly gia tri thieu KHONG ro ri.

- HistGradientBoosting nhan NaN truc tiep -> giu nguyen.
- Mo hinh khac: dien trung vi tinh CHI tren tap huan luyen, them cot co
  "<cot>__missing" cho moi cot co gia tri thieu trong tap huan luyen.
Khong dien 0: voi luong mua, 0 la gia tri co nghia ("khong mua").
"""
import re

import numpy as np
import pandas as pd

NATIVE_NAN_MODELS = {"hist_gb"}

# ---------------- Chinh sach cot dac trung (soat 2026-10-03, R1) ----------------
# Mau ten cot CAM - PHAI GIONG HET FORBIDDEN trong
# .claude/skills/method-review/scripts/check_features.py (test_feature_policy.py kiem dong bo).
FORBIDDEN_PATTERNS = [
    (r"ndvi|ndwi|savi|vssi|(^|_)nir($|_)|(^|_)b5($|_)|(^|_)b7($|_)|sr_b\d", "quang hoc cung dai luong band nhan"),
    (r"^salinity|(^|_)salinity|^ec_|_ec$", "chinh nhan hoac suy tu nhan"),
    (r"water_freq|n_clear|mndwi|(^|_)mask", "suy tu mask nuoc (chi de loai pixel)"),
    (r"n_land|land_px|valid_px|n_valid|frac_valid|coverage|n_px|(^|_)(scope|valid|cover|fill)_frac|train_ok", "do phu nhan sau mask / cot chat luong (thong tin cua nhan)"),
    (r"fold|block|(^|_)split($|_)|holdout|test_group", "lo thiet ke chia tap"),
    (r"^(season|year|date|month|doy)$", "thoi gian (ma dinh danh / ngoai suy xu huong)"),
    (r"^(lat|lon|latitude|longitude|x_utm|y_utm)$", "toa do (chi dung nhu bien the chon bang CV)"),
    (r"(^|_)n_days$|_frac_days$|_ok$|overlap_frac", "cot chat luong / do phu (khong phai dac trung)"),
]

# Mau CAM THEM theo BIEN MUC TIEU (Dot 7, de xuat (d) - An chot): dac trung do CUNG DAI LUONG VAT LY voi nhan
# (dua vao thi bai toan thanh ha thang/hieu chinh nguon khac, khong phai du doan bien). Khoa = ten cot muc tieu.
# PHAI GIONG HET TARGET_FORBIDDEN trong check_features.py (test_feature_policy.py kiem dong bo).
# Do man (salinity): khong co muc rieng - chi FORBIDDEN_PATTERNS chung (hanh vi cu).
TARGET_FORBIDDEN = {
    "ndwi": [
        (r"ndvi|ndwi|mndwi|savi|(^|_)evi|lswi|ndbi|(^|_)nbr|(^|_)nir($|_)|(^|_)swir|(^|_)b\d+($|_)|sr_b\d|landsat|reflect|albedo",
         "cung dai luong vat ly voi nhan NDWI (quang hoc / band Landsat)"),
        (r"salinity|(^|_)ec($|_)", "cung dai luong vat ly voi nhan NDWI (do man dung chung band NIR B5)"),
    ],
    "rain_chirps": [(r"(^|_)rain|precip|(^|_)tp($|_)", "cung dai luong vat ly voi nhan mua (mua ERA5)")],
    "dsr_mcd18": [(r"solar|ssrd|(^|_)ssr($|_)|(^|_)dsr|radiation|irradiance|cloud",
                   "cung dai luong vat ly voi nhan buc xa (buc xa ERA5)")],
    "t2m_era5": [(r"temp|t2m|skin|dewpoint|d2m|(^|_)td($|_)",
                  "cung dai luong vat ly voi nhan nhiet do (T2m / nhiet do be mat / diem suong)")],
    "rh_era5": [(r"dewpoint|d2m|(^|_)td($|_)|temp|t2m|(^|_)rh($|_)|humid|vapou?r|(^|_)q2m|specific_hum|vpd",
                 "cung dai luong vat ly voi nhan do am (RH = f(T, Td); moi dac trung am)")],
}


def target_forbidden(target) -> list[tuple[str, str]]:
    """Mau cam them cho bien muc tieu (rong voi do man / bien khong khai bao)."""
    return list(TARGET_FORBIDDEN.get(str(target), []))


def find_target_leak_columns(columns, target) -> list[tuple[str, str]]:
    """[(cot, ly do)] cho cot khop mau cam THEO BIEN MUC TIEU (khong phan biet hoa thuong)."""
    bad = []
    for c in columns:
        lc = str(c).lower()
        for pat, why in target_forbidden(target):
            if re.search(pat, lc):
                bad.append((c, why))
                break
    return bad


def drop_target_forbidden(allowed, target) -> tuple[list[str], list[str]]:
    """Bo khoi danh sach cho phep cac muc khop mau cam theo bien muc tieu. Tra (con lai, da bo).
    Muc tien to 'x_*' xet theo phan tien to."""
    keep, dropped = [], []
    for e in allowed:
        name = str(e)[:-1] if str(e).endswith("*") else str(e)
        (dropped if find_target_leak_columns([name], target) else keep).append(e)
    return keep, dropped


# Tien to DUOC PHEP dung dang 'x_*' trong danh sach cho phep (soat 2026-10-03, muc 5a): chi nhom
# cot co so lop thay doi (WorldCover). Muc '*' don le hoac tien to khac -> loi.
FEATURE_PREFIXES = ("landcover_class_",)

# Danh sach CHO PHEP mac dinh (skill mekong-data-conventions: ERA5, DEM, khoang cach song/bien,
# WorldCover, MRC, CMEMS). Muc ket thuc bang '*' la tien to (phai thuoc FEATURE_PREFIXES).
# - ERA5: col_name trong DEFAULT_DATA_SPECS (src/config.py).
# - DEM: dem_mean. Khoang cach: cot cua scripts/build_river_features.py.
# - WorldCover: landcover_class_<Lop> (processing.py, method all_classes).
# - MRC/CMEMS: cot cua scripts/build_hydro_season.py (bien cap vung theo mua). Bo chinh (An duyet cau B,
#   2026-10-03): tch_wl_p20c (phan vi 20% Tan Chau co xu ly chan) + zos_coast_p90. tch_wl_mean,
#   cdo_wl_mean, zos_mean, zos_coast_mean KHONG vao mac dinh (zos_mean ~ zos_coast_mean r 0,999; CDO
#   trung thong tin TCH) - muon dung phai truyen --features tuong minh. Ten chinh xac, KHONG dung tien
#   to: cot chat luong *_n_days, *_frac_days, *_ok, tch_wl_n_censored khong phai dac trung;
#   tch_wl_days_le_0_3 chi cho do nhay.
# - Hybrid (CHG-16, An chot 2026-10-04; scripts/build_river_graph.py): dist_mouth_river_km, zos_mouth_p90,
#   sluice_frac. zos_coast_p90 (cap vung) RA KHOI mac dinh -> ban DOI CHUNG khi boc tach hybrid (truyen
#   --features). Cot kiem cua bang hybrid (graph_lateral_km, sluice_frac_from2021, zos_cmems_far_frac)
#   khong phai dac trung.
DEFAULT_ALLOWED_FEATURES = (
    "rain_mm", "solar", "temp_c", "temp_max_c", "temp_min_c", "rh_percent",
    "dem_mean",
    "dist_main_river_km", "dist_any_water_km", "dist_coast_km",
    "landcover_class_*",
    "tch_wl_p20c",
    "dist_mouth_river_km", "zos_mouth_p90", "sluice_frac",
)


def find_leak_columns(columns, allow=()) -> list[tuple[str, str]]:
    """[(cot, ly do)] cho moi cot khop mau cam (khong phan biet hoa thuong), tru cot trong `allow`."""
    allow = {c.lower() for c in allow}
    bad = []
    for c in columns:
        lc = str(c).lower()
        if lc in allow:
            continue
        for pat, why in FORBIDDEN_PATTERNS:
            if re.search(pat, lc):
                bad.append((c, why))
                break
    return bad


def assert_no_leak_columns(columns, allow=()) -> None:
    """Bao loi neu co cot dac trung khop mau cam. `allow`: cot cam nhung duoc phep CO Y."""
    bad = find_leak_columns(columns, allow)
    if bad:
        detail = "; ".join(f"{c} ({why})" for c, why in bad)
        raise ValueError(f"Cot ro ri trong dac trung ({len(bad)}): {detail}")


def _check_entry(entry: str) -> None:
    """Muc danh sach cho phep: ten day du, hoac '<tien to>*' voi tien to khai bao trong FEATURE_PREFIXES."""
    if "*" not in entry:
        return
    if entry.count("*") != 1 or not entry.endswith("*") or entry[:-1] not in FEATURE_PREFIXES:
        raise ValueError(
            f"Muc dac trung '{entry}' khong hop le: chi cho phep ten day du hoac tien to "
            f"{[p + '*' for p in FEATURE_PREFIXES]} ('*' don le se nhan moi cot)."
        )


def resolve_feature_list(columns, allowed) -> tuple[list[str], list[str]]:
    """Giai danh sach cho phep thanh (ten cot day du co trong bang, muc yeu cau ma bang khong co).

    - Ten day du: co trong `columns` -> chon; khong co -> vao danh sach thieu.
    - Tien to 'x_*' (chi FEATURE_PREFIXES): giai thanh moi cot bat dau bang 'x_'; khong khop cot
      nao -> ca muc vao danh sach thieu.
    Thu tu ket qua theo thu tu cot cua bang. Khong kiem mau cam (xem select_feature_columns).
    """
    cols = [str(c) for c in columns]
    chosen: set[str] = set()
    missing: list[str] = []
    for entry in allowed:
        entry = str(entry)
        _check_entry(entry)
        if entry.endswith("*"):
            hit = [c for c in cols if c.startswith(entry[:-1])]
        else:
            hit = [entry] if entry in cols else []
        if hit:
            chosen.update(hit)
        else:
            missing.append(entry)
    return [c for c in cols if c in chosen], missing


def select_feature_columns(df: pd.DataFrame, allowed=DEFAULT_ALLOWED_FEATURES, allow_leak=(),
                           require_all: bool = False, target=None) -> list[str]:
    """Cot dac trung = cot cua df nam trong danh sach CHO PHEP (theo thu tu cot df), da giai tien to.

    Khong suy tu "moi cot so tru danh sach loai" (cach cu de lot salinity_median, is_holdout,
    fold, season...). Cot cho phep nhung khong phai so -> loi; sau khi chon van kiem mau cam.

    `require_all=True` (nguoi dung yeu cau TUONG MINH qua --features/--features-file): muc nao
    khong co trong bang -> loi (truoc day chi in roi bo qua -> bo dac trung khac nhau giua cac
    luoi ma khong ai biet). Danh sach mac dinh: muc thieu chi canh bao (xem resolve_feature_list
    de ghi vao config).
    """
    allowed = list(allowed)
    chosen, missing = resolve_feature_list(df.columns, allowed)
    if missing and require_all:
        raise ValueError(f"Cot dac trung duoc yeu cau nhung khong co trong bang: {missing}")
    if not chosen:
        raise ValueError(f"Khong co cot nao thuoc danh sach cho phep {allowed} trong bang ({list(df.columns)}).")
    non_numeric = [c for c in chosen if not (pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]))]
    if non_numeric:
        raise ValueError(f"Cot cho phep nhung khong phai so: {non_numeric}")
    assert_no_leak_columns(chosen, allow=allow_leak)
    tbad = find_target_leak_columns(chosen, target) if target is not None else []
    if tbad:  # khong co ngoai le --allow cho mau theo bien (cung dai luong vat ly)
        raise ValueError(f"Cot cam theo bien muc tieu '{target}' ({len(tbad)}): "
                         + "; ".join(f"{c} ({why})" for c, why in tbad))
    return chosen


def read_feature_list(path: str) -> list[str]:
    """Doc file danh sach dac trung: moi dong mot ten (hoac tien to 'x_*'); '#' la chu thich."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            name = line.split("#", 1)[0].strip()
            if name:
                out.append(name)
    if not out:
        raise ValueError(f"File danh sach dac trung rong: {path}")
    return out


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

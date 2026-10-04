"""
Kiem dinh so sanh khung luoi THEO KHOI (thiet ke kiem dinh khoi, phuong an B,
method-reviewer 2026-10-03; An duyet huong; sua theo bao cao soat V1-V5).

Luong:
  1. common_point_set: tap (point_id, season) hop le o MOI luoi -> dung chung cho moi cap.
  2. seed_mean_errors: trung binh |e| va e^2 qua seed theo (luoi, mo hinh, diem, mua);
     seed KHONG phai don vi suy dien.
  3. unit_metrics: MAE/MSE theo (luoi, don vi, mua) -> trung binh deu theo mua;
     bo (don vi, mua) < min_pts cho moi luoi nhu nhau, BAO danh sach bi bo.
  4. pairs_by_area / area_levels: cap cung muc (ti le dien tich <= max_ratio) tu bang
     doi chieu dien tich.
  5. compare_family: MOI suy dien (d_b, w_b, p sign-flip, CI t theo cum, TOST, Holm,
     seed, k/G, nguong) CHI tren khoi CV (du doan out-of-fold). Khoi giu rieng (du doan
     final) chi dung de XAC NHAN CHIEU: Delta_holdout tinh rieng, khong vao phep kiem dinh
     (V1: gop vao thi "giu rieng cung chieu" bi tu thoa vi Delta da chua chinh no).
  6. family_verdict: ti le cap giu chieu tren vung giu rieng / cap co y nghia (V2).

Khong dung o day: kiem dinh cap diem (Wilcoxon/bootstrap diem) - diem tu tuong quan,
mo phong cho sai lam loai I ~0,47. Cac tham so chua chot (delta_min, alpha, min_frac,
require_practical, mua giu rieng) la doi so, khong gan cung.

Khoi < min_pts diem/mua can gop vao khoi ke: lam o buoc tao point_unit (ngoai module nay).
"""
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from training.evaluate import holm_adjust

KEY = ["point_id", "season"]
EXACT_MAX_G = 20          # G <= 20: liet ke du 2^G to hop dau
DEFAULT_MC_PERM = 100_000  # G > 20: Monte Carlo
LABELS = ("xac_nhan", "co_y_nghia_duoi_nguong", "khong_tai_lap", "khong_tai_lap_cap_ho",
          "tuong_duong", "cho_giu_rieng", "chua_phan_dinh")
PRED_SOURCES = ("oof", "final")


# ---------------------------------------------------------
# 1. Tap diem chung
# ---------------------------------------------------------
def common_point_set(valid_long: pd.DataFrame, unit_col: str = "block"):
    """Tap (point_id, season) hop le o MOI luoi + bang so diem bi loai.

    valid_long: cot grid, point_id, season, valid (bool) [, unit_col]. Moi (grid, point_id,
    season) toi da 1 dong; (diem, mua) co o luoi nay nhung THIEU dong o luoi khac = khong
    hop le o luoi do. valid NaN -> khong hop le.

    Tra ve (common, dropped):
      common: MultiIndex (point_id, season) da sap xep.
      dropped: DataFrame theo (grid, season, unit): n_universe (diem-mua xuat hien o it
        nhat mot luoi), n_invalid_grid (khong hop le/thieu o CHINH luoi nay),
        n_dropped_common (bi loai khoi tap chung vi mot luoi bat ky), n_common.
    """
    need = {"grid", "point_id", "season", "valid"}
    miss = need - set(valid_long.columns)
    if miss:
        raise ValueError(f"valid_long thieu cot {sorted(miss)}")
    df = valid_long.copy()
    if df.duplicated(["grid"] + KEY).any():
        raise ValueError("valid_long co khoa (grid, point_id, season) trung lap")
    df["valid"] = df["valid"].fillna(False).astype(bool)
    grids = sorted(df["grid"].unique())
    wide = df.pivot(index=KEY, columns="grid", values="valid").reindex(columns=grids)
    wide = wide.fillna(False).astype(bool)  # thieu dong o mot luoi = khong hop le o luoi do
    in_common = wide.all(axis=1)
    common = wide.index[in_common.to_numpy()].sort_values()

    has_unit = unit_col in df.columns
    if has_unit:
        unit = df.drop_duplicates(KEY).set_index(KEY)[unit_col]
        if df.groupby(KEY)[unit_col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"mot (point_id, season) co nhieu gia tri {unit_col} giua cac luoi")
        unit = unit.reindex(wide.index)
    rows = []
    for g in grids:
        t = pd.DataFrame({"season": wide.index.get_level_values("season"),
                          "invalid": ~wide[g].to_numpy(),
                          "dropped": ~in_common.to_numpy()})
        t["unit"] = unit.to_numpy() if has_unit else "ALL"
        agg = t.groupby(["season", "unit"], dropna=False).agg(
            n_universe=("invalid", "size"), n_invalid_grid=("invalid", "sum"),
            n_dropped_common=("dropped", "sum")).reset_index()
        agg.insert(0, "grid", g)
        rows.append(agg)
    dropped = pd.concat(rows, ignore_index=True)
    dropped["n_common"] = dropped["n_universe"] - dropped["n_dropped_common"]
    return common, dropped


# ---------------------------------------------------------
# 2. Trung binh qua seed
# ---------------------------------------------------------
def seed_mean_errors(err_long: pd.DataFrame) -> pd.DataFrame:
    """Trung binh |e| va e^2 qua seed theo (grid, model, point_id, season).

    err_long: cot grid, model, seed, point_id, season, err (sai so CO DAU, point_errors).
    err NaN -> ValueError (loc bang common_point_set truoc; khong dien 0, khong bo am tham).
    Moi (grid, model) phai co du MOI seed cua no o moi (diem, mua).
    """
    need = {"grid", "model", "seed", "point_id", "season", "err"}
    miss = need - set(err_long.columns)
    if miss:
        raise ValueError(f"err_long thieu cot {sorted(miss)}")
    keys = ["grid", "model", "seed"] + KEY
    if err_long.duplicated(keys).any():
        raise ValueError("err_long co khoa (grid, model, seed, point_id, season) trung lap")
    if err_long["err"].isna().any():
        raise ValueError(f"err_long co {int(err_long['err'].isna().sum())} sai so NaN - "
                         "loc theo common_point_set truoc")
    df = err_long.assign(abs_err=err_long["err"].abs(), sq_err=err_long["err"] ** 2)
    n_seed_arm = df.groupby(["grid", "model"])["seed"].nunique()
    out = df.groupby(["grid", "model"] + KEY).agg(
        abs_err=("abs_err", "mean"), sq_err=("sq_err", "mean"), mean_err=("err", "mean"),
        n_seed=("seed", "size")).reset_index()
    expect = out.set_index(["grid", "model"]).index.map(n_seed_arm)
    bad = out["n_seed"].to_numpy() != np.asarray(expect)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} (diem, mua) thieu seed so voi cac diem khac cung (grid, model)")
    return out


# ---------------------------------------------------------
# 3. Chi so theo don vi
# ---------------------------------------------------------
def _arm_key_sets(err: pd.DataFrame) -> dict:
    return {arm: pd.MultiIndex.from_frame(g[KEY]).sort_values()
            for arm, g in err.groupby(["grid", "model"])}


def assert_same_points(err: pd.DataFrame, arms=None) -> None:
    """Assert: cac (grid, model) co DUNG cung tap chi muc (point_id, season)."""
    sets = _arm_key_sets(err)
    if arms is not None:
        missing = [a for a in arms if a not in sets]
        if missing:
            raise ValueError(f"khong co du lieu cho {missing}")
        sets = {a: sets[a] for a in arms}
    items = list(sets.items())
    ref_arm, ref = items[0]
    for arm, idx in items[1:]:
        if not idx.equals(ref):
            n_a = len(idx.difference(ref))
            n_b = len(ref.difference(idx))
            raise AssertionError(f"{arm} va {ref_arm} khac tap (point_id, season): "
                                 f"{n_a} chi co o {arm}, {n_b} chi co o {ref_arm}")


def unit_metrics(err: pd.DataFrame, point_unit: pd.Series, min_pts: int = 30):
    """MAE/MSE theo (grid, model, unit, season) -> trung binh DEU theo mua.

    err: bang tu seed_mean_errors (cot grid, model, point_id, season, abs_err, sq_err).
    point_unit: point_id -> don vi (khoi). Diem khong co don vi -> ValueError.
    (unit, season) co < min_pts diem chung bi bo cho MOI luoi nhu nhau (dem tren tap
    chung, giong nhau o moi luoi vi assert cung tap diem).

    Tra ve (units, seasons, dropped):
      units: grid, model, unit, mae, mse, rmse, n_seasons, n_pts_mean (so diem chung
        trung binh/mua - dung lam trong so w_b).
      seasons: grid, model, unit, season, n_pts, mae, mse, kept.
      dropped: unit, season, n_pts, unit_fully_dropped - cac (unit, season) bi bo vi
        < min_pts (giong nhau o moi luoi). unit_fully_dropped = khoi mat MOI mua -> khong
        con trong units (vd khoi giu rieng nho 36 diem). Rong neu khong bo gi.
    """
    assert_same_points(err)
    if point_unit.index.has_duplicates:
        raise ValueError("point_unit co point_id trung lap")
    df = err.copy()
    df["unit"] = df["point_id"].map(point_unit)
    if df["unit"].isna().any():
        raise ValueError(f"{df.loc[df['unit'].isna(), 'point_id'].nunique()} diem khong co don vi trong point_unit")
    seasons = df.groupby(["grid", "model", "unit", "season"]).agg(
        n_pts=("abs_err", "size"), mae=("abs_err", "mean"), mse=("sq_err", "mean")).reset_index()
    seasons["kept"] = seasons["n_pts"] >= min_pts
    kept = seasons[seasons["kept"]]
    units = kept.groupby(["grid", "model", "unit"]).agg(
        mae=("mae", "mean"), mse=("mse", "mean"), n_seasons=("season", "size"),
        n_pts_mean=("n_pts", "mean")).reset_index()
    units["rmse"] = np.sqrt(units["mse"])
    # cung tap diem o moi nhanh -> (unit, season) bi bo giong nhau; lay theo khoa duy nhat
    us = seasons.drop_duplicates(["unit", "season"])[["unit", "season", "n_pts", "kept"]]
    dropped = us[~us["kept"]].drop(columns="kept").reset_index(drop=True)
    fully = set(us["unit"]) - set(us.loc[us["kept"], "unit"])
    dropped["unit_fully_dropped"] = dropped["unit"].isin(fully)
    return units, seasons, dropped


# ---------------------------------------------------------
# 4. Kiem dinh
# ---------------------------------------------------------
def _check_dw(d, w):
    d = np.asarray(d, dtype=float)
    w = np.asarray(w, dtype=float)
    if d.ndim != 1 or d.shape != w.shape:
        raise ValueError("d va w phai la mang 1 chieu cung do dai")
    if len(d) == 0:
        raise ValueError("khong co don vi nao (G = 0)")
    if not (np.isfinite(d).all() and np.isfinite(w).all()):
        raise ValueError("d/w co NaN hoac vo cung")
    if (w <= 0).any():
        raise ValueError("trong so w phai > 0")
    return d, w / w.sum()


def comparison_seed(cmp_id: str, base_seed: int = 42) -> np.random.SeedSequence:
    """SeedSequence(base_seed) con, khoa theo MA phep so sanh (khong theo thu tu).

    Tuong duong SeedSequence(base_seed).spawn nhung spawn_key = bam sha256 cua cmp_id
    -> them/bot cap khong doi seed cua cap khac.
    """
    key = int(hashlib.sha256(cmp_id.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.SeedSequence(base_seed, spawn_key=(key,))


def signflip_test(d, w, seed=None, n_perm=None) -> dict:
    """Kiem dinh hoan vi doi dau theo khoi (hai phia), Delta = sum w d (w chuan hoa tong 1).

    H0: phan bo d_b doi xung quanh 0. Thong ke |sum w s d|.
    G <= 20: liet ke du 2^G to hop, p = #(|null| >= |obs|) / 2^G (gom to hop dong nhat).
    G > 20: Monte Carlo n_perm (mac dinh 100.000) lan, p = (1 + #) / (n_perm + 1);
    seed (int hoac SeedSequence) bat buoc de tai lap.
    """
    d, w = _check_dw(d, w)
    wd = w * d
    obs = abs(wd.sum())
    tol = 1e-12 * max(1.0, float(np.abs(wd).sum()))
    G = len(d)
    if G <= EXACT_MAX_G:
        total = 1 << G
        cnt = 0
        bits = np.arange(G, dtype=np.int64)
        chunk = 1 << 16
        for start in range(0, total, chunk):
            masks = np.arange(start, min(total, start + chunk), dtype=np.int64)
            signs = ((masks[:, None] >> bits) & 1) * 2 - 1
            cnt += int((np.abs(signs @ wd) >= obs - tol).sum())
        return {"delta_hat": float(wd.sum()), "p_value": cnt / total, "G": G,
                "n_perm": total, "exact": True}
    if seed is None:
        raise ValueError("G > 20 (Monte Carlo) can seed")
    n_perm = DEFAULT_MC_PERM if n_perm is None else int(n_perm)
    rng = np.random.default_rng(seed)
    cnt = 0
    chunk = 10_000
    for start in range(0, n_perm, chunk):
        b = min(chunk, n_perm - start)
        signs = rng.integers(0, 2, size=(b, G)) * 2 - 1
        cnt += int((np.abs(signs @ wd) >= obs - tol).sum())
    return {"delta_hat": float(wd.sum()), "p_value": (1 + cnt) / (n_perm + 1), "G": G,
            "n_perm": n_perm, "exact": False}


def cluster_t_ci(d, w, alpha: float) -> dict:
    """CI (1 - alpha) cho Delta = sum w d bang t robust theo cum co trong so, df = G - 1.

    w chuan hoa tong 1; se^2 = G/(G-1) * sum (w_b (d_b - Delta))^2 (CR1); q = t_{1-alpha/2, G-1}.
    """
    d, w = _check_dw(d, w)
    G = len(d)
    if G < 2:
        raise ValueError("cluster_t_ci can G >= 2")
    m = float((w * d).sum())
    se = float(np.sqrt(G / (G - 1) * ((w * (d - m)) ** 2).sum()))
    q = float(t_dist.ppf(1 - alpha / 2, G - 1))
    return {"delta_hat": m, "se": se, "df": G - 1, "q": q, "ci_low": m - q * se, "ci_high": m + q * se}


# ---------------------------------------------------------
# 5. Cap cung muc tu bang dien tich
# ---------------------------------------------------------
def _read_area_table(area_table, tier_col=None, area_col="mean_area_km2"):
    df = pd.read_csv(area_table) if isinstance(area_table, (str, Path)) else area_table.copy()
    if tier_col is None:
        tier_col = "tier" if "tier" in df.columns else "tier_h3_res"
    for c in (tier_col, area_col):
        if c not in df.columns:
            raise ValueError(f"bang dien tich thieu cot '{c}'")
    if "grid" in df.columns:
        grid = df["grid"].astype(str)
    elif "file" in df.columns:
        grid = df["file"].map(lambda f: Path(str(f)).stem)  # data/grids/h3_res_5.geojson -> h3_res_5
    else:
        raise ValueError("bang dien tich can cot 'grid' hoac 'file' de lay ten luoi")
    out = pd.DataFrame({"grid": grid, "tier": df[tier_col], "area": df[area_col].astype(float)})
    if out["grid"].duplicated().any():
        raise ValueError("bang dien tich co ten luoi trung lap")
    if out[["tier", "area"]].isna().any().any() or (out["area"] <= 0).any():
        raise ValueError("bang dien tich co tier/dien tich NaN hoac <= 0")
    return out


def area_levels(area_table, pairs=None, tier_col=None) -> pd.Series:
    """grid -> muc phan giai (tier) tu bang doi chieu dien tich (dung cho delta_ref='level_mean').

    pairs: None -> MOI luoi trong bang (muc 6 gom ca s2_level_10/11 du khong vao cap nao);
    DataFrame tu pairs_by_area -> chi cac luoi co mat trong cap. Chon cach nao la quyet dinh
    nghiep vu (dinh nghia "moi luoi trong muc") - ghi ro khi dung.
    """
    t = _read_area_table(area_table, tier_col=tier_col)
    if pairs is not None:
        keep = set(pairs["grid_a"]) | set(pairs["grid_b"])
        t = t[t["grid"].isin(keep)]
    return pd.Series(t["tier"].to_numpy(), index=t["grid"].to_numpy(), name="tier")


def pairs_by_area(area_table, max_ratio: float = 1.5, expected_m=None, tier_col=None,
                  area_col: str = "mean_area_km2") -> pd.DataFrame:
    """Cap (A, B) CUNG muc (tier) co dien tich o trung binh lech <= max_ratio lan.

    area_table: duong dan CSV hoac DataFrame (KE_HOACH/ket-qua/tuan1_doi_chieu_dien_tich.csv:
      cot mean_area_km2, tier_h3_res, file). Ten luoi = cot 'grid' hoac ten file bo duoi.
    ti le = max(dien tich) / min(dien tich). A la luoi dung truoc trong bang (chi doi dau Delta).
    expected_m: neu truyen -> assert so cap == expected_m (bang 13 luoi, 1,5 -> 15 cap).

    Tra ve DataFrame grid_a, grid_b, tier, area_a, area_b, area_ratio (theo thu tu bang).
    Thay cho co "lan kich thuoc o" cu: cap lech dien tich > max_ratio khong duoc ghep.
    """
    if not max_ratio >= 1:
        raise ValueError("max_ratio phai >= 1")
    t = _read_area_table(area_table, tier_col=tier_col, area_col=area_col).reset_index(drop=True)
    rows = []
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            a, b = t.iloc[i], t.iloc[j]
            if a["tier"] != b["tier"]:
                continue
            ratio = max(a["area"], b["area"]) / min(a["area"], b["area"])
            if ratio <= max_ratio * (1 + 1e-12):
                rows.append({"grid_a": a["grid"], "grid_b": b["grid"], "tier": a["tier"],
                             "area_a": a["area"], "area_b": b["area"], "area_ratio": ratio})
    out = pd.DataFrame(rows, columns=["grid_a", "grid_b", "tier", "area_a", "area_b", "area_ratio"])
    if expected_m is not None and len(out) != expected_m:
        raise AssertionError(f"pairs_by_area: {len(out)} cap, ky vong {expected_m} (max_ratio={max_ratio})")
    return out


# ---------------------------------------------------------
# 6. Ho so sanh
# ---------------------------------------------------------
def _arm(a, model):
    if isinstance(a, tuple):
        return a
    if model is None:
        raise ValueError(f"cap dung ten luoi '{a}' thi phai truyen model")
    return (a, model)


def _pair_d(units: pd.DataFrame, arm_a, arm_b, metric: str):
    ua = units[(units["grid"] == arm_a[0]) & (units["model"] == arm_a[1])].set_index("unit")
    ub = units[(units["grid"] == arm_b[0]) & (units["model"] == arm_b[1])].set_index("unit")
    if not ua.index.sort_values().equals(ub.index.sort_values()):
        raise AssertionError(f"{arm_a} va {arm_b} khac tap don vi sau loc min_pts")
    ub = ub.reindex(ua.index)
    d = (ua[metric] - ub[metric]).sort_index()
    w = ua["n_pts_mean"].reindex(d.index)
    if not np.allclose(w.to_numpy(), ub["n_pts_mean"].reindex(d.index).to_numpy()):
        raise AssertionError("so diem chung khac nhau giua hai luoi")
    return d, w


def _arm_level(units: pd.DataFrame, arm, metric: str) -> float:
    """Chi so cua mot nhanh tren cac khoi CV: trung binh co trong so n_pts_mean."""
    u = units[(units["grid"] == arm[0]) & (units["model"] == arm[1])]
    return float((u[metric] * u["n_pts_mean"]).sum() / u["n_pts_mean"].sum())


def _wmean(d: pd.Series, w: pd.Series) -> float:
    return float(((w / w.sum()) * d).sum())


def _normalize_pairs(pairs):
    if isinstance(pairs, pd.DataFrame):
        if not {"grid_a", "grid_b"} <= set(pairs.columns):
            raise ValueError("pairs DataFrame can cot grid_a, grid_b")
        ratio = pairs["area_ratio"].tolist() if "area_ratio" in pairs.columns else [np.nan] * len(pairs)
        return list(zip(pairs["grid_a"], pairs["grid_b"])), ratio
    pairs = list(pairs)
    return pairs, [np.nan] * len(pairs)


def compare_family(err_long: pd.DataFrame, point_unit: pd.Series, pairs, *,
                   cv_units, holdout_units, holdout_seasons, mode: str,
                   delta_min: float, delta_min_kind: str, alpha: float,
                   delta_ref="level_mean", levels=None, require_practical: bool = True,
                   confirm_min_frac=None, expected_m=None, model=None, metric: str = "mae",
                   min_pts: int = 30, family: str = "F1", base_seed: int = 42,
                   n_perm=None) -> pd.DataFrame:
    """So sanh tung cap (A, B) trong mot ho Holm; Delta < 0 nghia la A sai so THAP hon B.

    err_long: cot grid, model, seed, point_id, season, err (sai so co dau), pred_source
      ("oof" = du doan out-of-fold cua CV khoi; "final" = mo hinh cuoi tren vung giu rieng);
      da loc theo common_point_set. Hai nhanh trong cap phai cung tap (point_id, season) va
      cung tap seed (assert).
    point_unit: point_id -> khoi. Moi khoi phai thuoc DUNG MOT trong cv_units / holdout_units.
    cv_units: khoi CV ngoai - MOI suy dien chi tren cac khoi nay (hang phai la "oof", khong
      chua mua trong holdout_seasons).
    holdout_units: khoi giu rieng - chi tinh Delta_holdout de xac nhan chieu (hang phai la
      "final"). Them/bot khoi giu rieng KHONG doi p, CI, Holm, nguong.
    holdout_seasons: mua giu rieng (vd (2020,)), bat buoc, khong rong.
    mode: "cv" -> tu choi MOI hang thuoc holdout_units trong err_long (khong ai nhin so vung
      giu rieng truoc khi chot tieu chi); "final" -> can co hang giu rieng.
    pairs: list (A, B) (ten luoi dung `model`, hoac tuple (grid, model)) hoac DataFrame tu
      pairs_by_area. expected_m: assert so cap cua ho.
    delta_min / delta_min_kind: "abs" (dS/m) hoac "rel" (= delta_min x delta_ref).
    delta_ref (chi dung khi rel): "level_mean" -> MOT muc tham chieu cho moi muc phan giai =
      trung binh chi so (tren cv_units) cua MOI luoi trong muc theo `levels` (grid -> muc) x
      MOI mo hinh co trong ho; hoac mot so duong truyen thang. Tinh mot lan, ghi vao output.
    alpha: muc y nghia ho (Holm) va CI (1 - alpha); TOST dung CI (1 - 2 alpha).
    require_practical: True -> cap dat moi dieu kien xac nhan nhung |Delta_cv| < nguong ghi
      nhan "co_y_nghia_duoi_nguong" (Q3 chua chot).
    confirm_min_frac: None = khong ap; neu truyen, giu chieu con can ti le khoi giu rieng
      cung dau voi Delta_cv >= gia tri nay.

    Nhan (mode final):
      khong_tai_lap: Holm-p < alpha (CV) nhung sign(Delta_holdout) != sign(Delta_cv).
      xac_nhan: Holm-p < alpha VA giu chieu VA moi seed (CV) cung chieu
        [VA |Delta_cv| >= nguong neu require_practical, nguoc lai co_y_nghia_duoi_nguong].
      tuong_duong: khong dat Holm, CI (1 - 2 alpha) trong (-thr, +thr) (TOST).
      chua_phan_dinh: con lai (Holm dat + giu chieu nhung seed khong cung chieu; khong con
        khoi giu rieng sau loc min_pts; ...).
    mode cv: thay xac_nhan/khong_tai_lap bang "cho_giu_rieng" (Holm dat + seed cung chieu).
    Nhan cap ho (khong_tai_lap_cap_ho) do family_verdict ghi.
    """
    if mode not in ("cv", "final"):
        raise ValueError("mode phai la 'cv' hoac 'final'")
    if delta_min_kind not in ("abs", "rel"):
        raise ValueError("delta_min_kind phai la 'abs' hoac 'rel'")
    if metric not in ("mae", "rmse", "mse"):
        raise ValueError("metric phai la mae/rmse/mse")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha phai trong (0; 0,5)")
    if confirm_min_frac is not None and not 0 <= confirm_min_frac <= 1:
        raise ValueError("confirm_min_frac phai trong [0, 1] hoac None")
    if not (isinstance(delta_ref, str) and delta_ref == "level_mean"):
        if isinstance(delta_ref, str) or not np.isfinite(delta_ref) or delta_ref <= 0:
            raise ValueError("delta_ref phai la 'level_mean' hoac so duong")
    holdout_seasons = tuple(sorted(set(holdout_seasons)))
    if not holdout_seasons:
        raise ValueError("holdout_seasons rong - phai khai bao mua giu rieng")
    pairs, area_ratio = _normalize_pairs(pairs)
    if len(pairs) == 0:
        raise ValueError("pairs rong")
    if expected_m is not None and len(pairs) != expected_m:
        raise AssertionError(f"ho {family}: {len(pairs)} cap, ky vong expected_m={expected_m}")

    # --- don vi: roi nhau, phu kin ---
    cv_set, ho_set = set(cv_units), set(holdout_units)
    both = cv_set & ho_set
    if both:
        raise ValueError(f"cv_units va holdout_units giao nhau: {sorted(both)}")
    all_units = set(point_unit.dropna().unique())
    orphan = all_units - cv_set - ho_set
    if orphan:
        raise ValueError(f"don vi khong thuoc cv_units hay holdout_units: {sorted(orphan)}")
    unknown = (cv_set | ho_set) - all_units
    if unknown:
        raise ValueError(f"cv_units/holdout_units co don vi khong co trong point_unit: {sorted(unknown)}")

    # --- nguon du doan, mua, che do ---
    need = {"grid", "model", "seed", "point_id", "season", "err", "pred_source"}
    miss = need - set(err_long.columns)
    if miss:
        raise ValueError(f"err_long thieu cot {sorted(miss)}")
    unit_all = err_long["point_id"].map(point_unit)
    if unit_all.isna().any():
        raise ValueError(f"{err_long.loc[unit_all.isna(), 'point_id'].nunique()} diem khong co don vi trong point_unit")
    is_ho_all = unit_all.isin(ho_set)
    if mode == "cv" and is_ho_all.any():
        raise ValueError(f"mode='cv': err_long co {int(is_ho_all.sum())} hang thuoc holdout_units - "
                         "khong xem vung giu rieng truoc khi chot tieu chi (dung mode='final')")
    bad_src = ~err_long["pred_source"].isin(PRED_SOURCES)
    if bad_src.any():
        raise ValueError(f"pred_source phai thuoc {PRED_SOURCES}; co {int(bad_src.sum())} hang khac")
    cv_rows = ~is_ho_all
    if (err_long.loc[cv_rows, "pred_source"] != "oof").any():
        raise ValueError("hang cua cv_units phai co pred_source='oof'")
    if (err_long.loc[is_ho_all, "pred_source"] != "final").any():
        raise ValueError("hang cua holdout_units phai co pred_source='final'")
    leak = cv_rows & err_long["season"].isin(holdout_seasons)
    if leak.any():
        raise ValueError(f"cv_units co {int(leak.sum())} hang thuoc mua giu rieng {holdout_seasons}")

    # --- nhanh can dung: cap + (neu rel/level_mean) moi luoi cung muc ---
    arm_pairs = [(_arm(a, model), _arm(b, model)) for a, b in pairs]
    pair_arms = sorted({x for p in arm_pairs for x in p})
    for arm_a, arm_b in arm_pairs:
        if arm_a == arm_b:
            raise ValueError(f"cap trung nhanh {arm_a}")
    use_level = delta_min_kind == "rel" and isinstance(delta_ref, str)
    ref_arms_by_level = {}
    pair_level = [None] * len(arm_pairs)
    if use_level:
        if levels is None:
            raise ValueError("delta_ref='level_mean' can levels (grid -> muc), vd area_levels(...)")
        levels = levels if isinstance(levels, pd.Series) else pd.Series(levels)
        fam_models = sorted({a[1] for a in pair_arms})
        for i, (arm_a, arm_b) in enumerate(arm_pairs):
            if arm_a[0] not in levels.index or arm_b[0] not in levels.index:
                raise ValueError(f"levels thieu luoi cua cap {arm_a}/{arm_b}")
            la, lb = levels[arm_a[0]], levels[arm_b[0]]
            if la != lb:
                raise ValueError(f"cap {arm_a}/{arm_b} khac muc ({la} vs {lb}) - delta_ref theo muc khong dinh nghia")
            pair_level[i] = la
        for lv in sorted(set(pair_level)):
            ref_arms_by_level[lv] = [(g, m) for g in sorted(levels.index[levels == lv]) for m in fam_models]
    ref_arms = sorted({a for v in ref_arms_by_level.values() for a in v})
    arms = sorted(set(pair_arms) | set(ref_arms))
    present = set(map(tuple, err_long[["grid", "model"]].drop_duplicates().to_numpy()))
    missing = [a for a in arms if a not in present]
    if missing:
        raise ValueError(f"err_long khong co du lieu cho {missing}")
    in_arms = err_long.set_index(["grid", "model"]).index.isin(arms)
    sub_cv = err_long[in_arms & cv_rows.to_numpy()]
    sub_ho = err_long[in_arms & is_ho_all.to_numpy()]
    if mode == "final" and sub_ho.empty:
        raise ValueError("mode='final' nhung khong co hang nao thuoc holdout_units")

    seed_sets = err_long[in_arms].groupby(["grid", "model"])["seed"].agg(lambda s: tuple(sorted(set(s))))
    for arm_a, arm_b in arm_pairs:
        if seed_sets[arm_a] != seed_sets[arm_b]:
            raise ValueError(f"{arm_a} va {arm_b} khac tap seed: {seed_sets[arm_a]} vs {seed_sets[arm_b]}")

    # --- CV: moi suy dien ---
    me_cv = seed_mean_errors(sub_cv)
    for arm_a, arm_b in arm_pairs:
        assert_same_points(me_cv, [arm_a, arm_b])
    if ref_arms:
        assert_same_points(me_cv, ref_arms)
    units_cv, _, drop_cv = unit_metrics(me_cv, point_unit, min_pts=min_pts)
    seeds = sorted(sub_cv["seed"].unique())
    seed_units_cv = {s: unit_metrics(seed_mean_errors(sub_cv[sub_cv["seed"] == s]), point_unit,
                                     min_pts=min_pts)[0] for s in seeds}
    cv_dropped = sorted(drop_cv.loc[drop_cv["unit_fully_dropped"], "unit"].unique())
    if cv_dropped:
        warnings.warn(f"{family}: khoi CV bi loai (< {min_pts} diem/mua o moi mua): {cv_dropped}", stacklevel=2)

    # --- giu rieng: chi Delta_holdout (khong vao kiem dinh) ---
    units_ho = None
    ho_dropped = []
    seed_units_ho = {}
    if mode == "final":
        me_ho = seed_mean_errors(sub_ho)
        for arm_a, arm_b in arm_pairs:
            assert_same_points(me_ho, [arm_a, arm_b])
        units_ho, _, drop_ho = unit_metrics(me_ho, point_unit, min_pts=min_pts)
        ho_dropped = set(drop_ho.loc[drop_ho["unit_fully_dropped"], "unit"])
        # khoi giu rieng khong co diem chung nao cung tinh la bi loai
        ho_dropped = sorted(ho_dropped | (ho_set - set(me_ho["point_id"].map(point_unit))))
        if ho_dropped:
            warnings.warn(f"{family}: khoi giu rieng bi loai (< {min_pts} diem/mua o moi mua hoac "
                          f"khong co diem): {ho_dropped}", stacklevel=2)
        for s in sorted(sub_ho["seed"].unique()):
            seed_units_ho[s] = unit_metrics(seed_mean_errors(sub_ho[sub_ho["seed"] == s]), point_unit,
                                            min_pts=min_pts)[0]

    # --- muc tham chieu cho nguong tuong doi: tinh MOT lan / muc, chi tren CV ---
    level_ref = {lv: float(np.mean([_arm_level(units_cv, a, metric) for a in arms_lv]))
                 for lv, arms_lv in ref_arms_by_level.items()}

    seasons_cv = ";".join(str(x) for x in sorted(sub_cv["season"].unique()))
    seasons_ho = ";".join(str(x) for x in sorted(sub_ho["season"].unique())) if mode == "final" else ""
    rows = []
    for i, (arm_a, arm_b) in enumerate(arm_pairs):
        cmp_id = f"{family}:{arm_a[0]}|{arm_a[1]}-vs-{arm_b[0]}|{arm_b[1]}"
        d, w = _pair_d(units_cv, arm_a, arm_b, metric)
        G = len(d)
        if G < 2:
            raise ValueError(f"{cmp_id}: chi con {G} khoi CV sau loc min_pts - khong kiem dinh duoc")
        sf = signflip_test(d.to_numpy(), w.to_numpy(), seed=comparison_seed(cmp_id, base_seed), n_perm=n_perm)
        ci = cluster_t_ci(d.to_numpy(), w.to_numpy(), alpha)
        ci2 = cluster_t_ci(d.to_numpy(), w.to_numpy(), 2 * alpha)
        delta = sf["delta_hat"]
        sgn = np.sign(delta)
        if delta_min_kind == "abs":
            ref, ref_level, ref_grids = np.nan, None, ""
            thr = delta_min
        elif use_level:
            ref_level = pair_level[i]
            ref = level_ref[ref_level]
            ref_grids = ";".join(f"{g}|{m}" for g, m in ref_arms_by_level[ref_level])
            thr = delta_min * ref
        else:
            ref, ref_level, ref_grids = float(delta_ref), None, ""
            thr = delta_min * ref
        seed_deltas = [_wmean(*_pair_d(seed_units_cv[s], arm_a, arm_b, metric)) for s in seeds]
        seed_same = bool(sgn != 0 and all(np.sign(x) == sgn for x in seed_deltas))
        k_same = int((np.sign(d) == sgn).sum()) if sgn != 0 else 0

        delta_ho, frac_ho, n_ho, ho_same, seed_ho = np.nan, np.nan, 0, False, []
        if mode == "final" and not units_ho.empty:
            d_ho, w_ho = _pair_d(units_ho, arm_a, arm_b, metric)
            n_ho = len(d_ho)
            if n_ho > 0:
                delta_ho = _wmean(d_ho, w_ho)
                frac_ho = float((np.sign(d_ho) == sgn).mean()) if sgn != 0 else 0.0
                ho_same = bool(sgn != 0 and np.sign(delta_ho) == sgn)
                if confirm_min_frac is not None:
                    ho_same = ho_same and frac_ho >= confirm_min_frac
                seed_ho = [_wmean(*_pair_d(us, arm_a, arm_b, metric)) for us in seed_units_ho.values()]
        rows.append({
            "family": family, "cmp_id": cmp_id, "mode": mode,
            "grid_a": arm_a[0], "model_a": arm_a[1], "grid_b": arm_b[0], "model_b": arm_b[1],
            "area_ratio": area_ratio[i], "metric": metric, "G": G,
            "delta_hat": delta, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "ci_tost_low": ci2["ci_low"], "ci_tost_high": ci2["ci_high"], "se": ci["se"],
            "p_value": sf["p_value"], "p_exact": sf["exact"], "n_perm": sf["n_perm"],
            "k_same": k_same, "k_over_G": f"{k_same}/{G}",
            "seed_deltas": ";".join(f"{x:.6g}" for x in seed_deltas), "n_seeds": len(seeds),
            "seeds_same_dir": seed_same,
            "delta_ref": ref, "delta_ref_level": ref_level, "delta_ref_grids": ref_grids,
            "delta_min_thr": thr,
            "tost_tuong_duong": bool(-thr < ci2["ci_low"] and ci2["ci_high"] < thr),
            "practical_ok": bool(abs(delta) >= thr),
            "n_holdout_units": n_ho, "delta_holdout": delta_ho, "holdout_frac_same": frac_ho,
            "holdout_same_dir": ho_same,
            "seed_deltas_holdout": ";".join(f"{x:.6g}" for x in seed_ho),
            "better": "A" if delta < 0 else ("B" if delta > 0 else ""),
            "seasons_cv": seasons_cv, "seasons_holdout": seasons_ho,
            "holdout_seasons": ";".join(str(x) for x in holdout_seasons),
            "cv_units_dropped": ";".join(map(str, cv_dropped)),
            "holdout_units_dropped": ";".join(map(str, ho_dropped)),
        })

    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
    labels = []
    for r in out.itertuples():
        holm_ok = r.p_holm < alpha and r.delta_hat != 0
        if mode == "cv":
            if holm_ok and r.seeds_same_dir:
                labels.append("cho_giu_rieng")
            elif not holm_ok and r.tost_tuong_duong:
                labels.append("tuong_duong")
            else:
                labels.append("chua_phan_dinh")
            continue
        has_ho = r.n_holdout_units > 0
        if holm_ok and has_ho and not r.holdout_same_dir:
            labels.append("khong_tai_lap")
        elif holm_ok and r.holdout_same_dir and r.seeds_same_dir:
            if require_practical and not r.practical_ok:
                labels.append("co_y_nghia_duoi_nguong")
            else:
                labels.append("xac_nhan")
        elif not holm_ok and r.tost_tuong_duong:
            labels.append("tuong_duong")
        else:
            labels.append("chua_phan_dinh")
    out["label"] = labels
    out["alpha"] = alpha
    out["confirm_min_frac"] = confirm_min_frac
    out["require_practical"] = require_practical
    out["delta_min"] = delta_min
    out["delta_min_kind"] = delta_min_kind
    out["min_pts"] = min_pts
    return out


def family_verdict(out: pd.DataFrame, min_frac: float = 0.8,
                   seed_inconsistent_counts_as_fail: bool = True,
                   empty_label: str = "khong_co_khac_biet_xac_nhan"):
    """Ket luan cap HO (V2; Q1 chua chot -> moi nguong la tham so).

    Mau so = cap co Holm-p (CV) < alpha va Delta_cv != 0.
    Tu so = cap trong mau so giu chieu tren vung giu rieng (holdout_same_dir) [va, neu
      seed_inconsistent_counts_as_fail, cung chieu o moi seed].
    tu/mau < min_frac -> nhan ho "khong_tai_lap", cac cap "xac_nhan" ghi de thanh
      "khong_tai_lap_cap_ho" (cot label_cap giu nhan cap goc). Mau so = 0 -> empty_label.
      tu/mau >= min_frac -> nhan ho "tai_lap", nhan cap giu nguyen.
    Chi dung cho output mode='final' cua MOT ho.

    Tra ve (out_moi, tom_tat dict: family, label, numerator, denominator, frac, min_frac).
    """
    if out.empty:
        raise ValueError("out rong")
    if not 0 <= min_frac <= 1:
        raise ValueError("min_frac phai trong [0, 1]")
    if (out["mode"] != "final").any():
        raise ValueError("family_verdict chi dung cho mode='final'")
    if out["family"].nunique() != 1 or out["alpha"].nunique() != 1:
        raise ValueError("out phai la mot ho voi mot alpha")
    alpha = float(out["alpha"].iloc[0])
    res = out.copy()
    sig = (res["p_holm"] < alpha) & (res["delta_hat"] != 0)
    ok = sig & res["holdout_same_dir"].astype(bool)
    if seed_inconsistent_counts_as_fail:
        ok = ok & res["seeds_same_dir"].astype(bool)
    den, num = int(sig.sum()), int(ok.sum())
    res["label_cap"] = res["label"]
    if den == 0:
        label, frac = empty_label, np.nan
    else:
        frac = num / den
        label = "tai_lap" if frac >= min_frac else "khong_tai_lap"
        if label == "khong_tai_lap":
            res.loc[res["label"] == "xac_nhan", "label"] = "khong_tai_lap_cap_ho"
    res["family_label"] = label
    res["family_num"] = num
    res["family_den"] = den
    res["family_min_frac"] = min_frac
    summary = {"family": res["family"].iloc[0], "label": label, "numerator": num,
               "denominator": den, "frac": frac, "min_frac": min_frac,
               "seed_inconsistent_counts_as_fail": seed_inconsistent_counts_as_fail}
    return res, summary

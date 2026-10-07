"""
Quy tac phan tich Dot 7 (CHG-25 muc 1-4): muc kiem dinh theo bien, tach cap kiem dinh / mo ta, cong "hoc duoc"
(HistGB vs season_mean), chan ghi de file dong bang.
"""
import os

import numpy as np
import pandas as pd

from training.block_stats import compare_family

TESTED_TIERS = {"salinity": frozenset({5, 6, 7}), "ndwi": frozenset({5, 6, 7}), "dsr_mcd18": frozenset({5, 6, 7}),
                "rain_chirps": frozenset({5}), "t2m_era5": frozenset(), "rh_era5": frozenset()}
EXPECTED_F1_M = {"salinity": 12, "ndwi": 12, "dsr_mcd18": 12, "rain_chirps": 3, "t2m_era5": 0, "rh_era5": 0}
# so luoi trong ho Holm cua cong hoc duoc theo pham vi (muc_kiem_dinh: moi luoi cua muc; cap_f1: luoi trong cap F1)
EXPECTED_GATE_M = {
    "muc_kiem_dinh": {"salinity": 13, "ndwi": 13, "dsr_mcd18": 13, "rain_chirps": 4, "t2m_era5": 0, "rh_era5": 0},
    "cap_f1": {"salinity": 10, "ndwi": 10, "dsr_mcd18": 10, "rain_chirps": 3, "t2m_era5": 0, "rh_era5": 0},
}
GATE_SCOPES = tuple(EXPECTED_GATE_M)
MO_TA = "mo_ta"
DESC_NAN = ("p_value", "p_holm", "tost_tuong_duong")
GATE_COLS = ("bien", "muc", "hop_le")
FROZEN_EXIT = 2


def tested_tiers(target: str) -> frozenset:
    if target not in TESTED_TIERS:
        raise ValueError(f"target '{target}' khong co trong TESTED_TIERS {sorted(TESTED_TIERS)}")
    return TESTED_TIERS[target]


# ---------------------------------------------------------
# Chan ghi de file dong bang
# ---------------------------------------------------------
def frozen_paths(manifest_csv) -> set:
    """Duong dan tuyet doi (normcase) trong manifest dong bang; cot 'file' tuong doi goc repo (2 cap tren manifest)."""
    if not os.path.isfile(manifest_csv):
        raise FileNotFoundError(f"khong co manifest dong bang {manifest_csv}")
    m = pd.read_csv(manifest_csv, dtype=str)
    if "file" not in m.columns:
        raise ValueError(f"{manifest_csv}: thieu cot 'file'")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(manifest_csv))))
    return {os.path.normcase(os.path.abspath(os.path.join(root, f))) for f in m["file"].dropna()}


def frozen_hits(paths, manifest_csv) -> list:
    frozen = frozen_paths(manifest_csv)
    return [p for p in paths if os.path.normcase(os.path.abspath(p)) in frozen]


def guard_frozen(paths, manifest_csv):
    """Duong dan nao co trong manifest dong bang (hoac thieu manifest) -> SystemExit ma 2, truoc khi doc/ghi gi."""
    try:
        hits = frozen_hits(paths, manifest_csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LOI: {exc} - khong kiem duoc file dong bang, khong ghi", flush=True)
        raise SystemExit(FROZEN_EXIT)
    if hits:
        print(f"LOI: tu choi ghi de file dong bang: {hits}", flush=True)
        raise SystemExit(FROZEN_EXIT)


def write_csv_atomic(df: pd.DataFrame, path):
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------
# Tach cap kiem dinh / mo ta
# ---------------------------------------------------------
def _tier_of_pairs(pairs: pd.DataFrame, levels) -> pd.Series:
    if "tier" in pairs.columns:
        return pairs["tier"]
    lv = levels if isinstance(levels, pd.Series) else pd.Series(levels)
    return pairs["grid_a"].map(lv)


def mark_descriptive(out: pd.DataFrame) -> pd.DataFrame:
    """Dong mo ta: giu Delta_hat + CI, bo p / Holm / TOST (khong bao gio tuong_duong / cho_giu_rieng)."""
    res = out.copy()
    for c in DESC_NAN:
        res[c] = np.nan
    res["label"] = MO_TA
    res["kiem_dinh"] = False
    return res


def compare_tiered(err, point_unit, pairs: pd.DataFrame, target: str, *, levels, expected_m=None, **kw):
    """compare_family rieng cho cap o muc kiem dinh (Holm tren cac cap do) va cap con lai (mo ta).

    Cot them: muc, muc_kiem_dinh, kiem_dinh, holm_m. Thu tu dong = thu tu `pairs`.
    """
    tiers = tested_tiers(target)
    p = pairs.reset_index(drop=True)
    tier = _tier_of_pairs(p, levels)
    if tier.isna().any():
        raise ValueError("cap khong xac dinh duoc muc (tier)")
    is_test = tier.isin(tiers).to_numpy()
    if expected_m is not None and int(is_test.sum()) != expected_m:
        raise AssertionError(f"{target}: {int(is_test.sum())} cap o muc kiem dinh, ky vong {expected_m}")
    parts = []
    for flag in (True, False):
        sub = p[is_test == flag]
        if sub.empty:
            continue
        o = compare_family(err, point_unit, sub, levels=levels, expected_m=len(sub), **kw)
        o["_ord"] = sub.index.to_numpy()
        o["muc"] = tier[sub.index].to_numpy()
        if flag:
            o["kiem_dinh"] = True
            o["holm_m"] = len(sub)
        else:
            o = mark_descriptive(o)
            o["holm_m"] = 0
        o["muc_kiem_dinh"] = flag
        parts.append(o)
    out = pd.concat(parts, ignore_index=True).sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    out.insert(0, "target", target)
    return out


# ---------------------------------------------------------
# Cong kiem dinh (doc / ap)
# ---------------------------------------------------------
def load_gate(path, target: str, required: bool):
    """Bang cong (bien, muc, hop_le) cua target. Thieu file / thieu dong cho muc kiem dinh -> ValueError khi required.

    Tra ve dict muc -> bool, hoac None khi khong bat buoc va khong co.
    """
    tiers = tested_tiers(target)
    if not os.path.isfile(path):
        if required:
            raise ValueError(f"thieu bang cong {path} (bat buoc cho target '{target}')")
        return None
    g = pd.read_csv(path)
    miss = set(GATE_COLS) - set(g.columns)
    if miss:
        raise ValueError(f"{path}: thieu cot {sorted(miss)}")
    g = g[g["bien"] == target]
    if g.empty:
        if required:
            raise ValueError(f"{path}: khong co dong nao cho bien '{target}'")
        return None
    if g.duplicated("muc").any():
        raise ValueError(f"{path}: trung (bien, muc) cho '{target}'")
    hop = {}
    for m, v in zip(g["muc"].astype(int), g["hop_le"]):
        s = str(v).strip().lower()
        if s not in ("true", "false"):
            raise ValueError(f"{path}: hop_le '{v}' khong phai True/False ({target}, muc {m})")
        hop[m] = s == "true"
    lack = sorted(t for t in tiers if t not in hop)
    if lack:
        raise ValueError(f"{path}: thieu dong cong cho muc kiem dinh {lack} cua '{target}'")
    return hop


def apply_gate(out: pd.DataFrame, gate, level_col: str = "muc") -> pd.DataFrame:
    """Muc kiem dinh khong qua cong -> mo_ta (truoc khi ghi). gate None -> chi them cot cong_hoc_duoc = NaN."""
    res = out.copy()
    if gate is None:
        res["cong_hoc_duoc"] = np.nan
        return res
    lv = res[level_col] if level_col in res.columns else pd.Series(np.nan, index=res.index)
    if res["muc_kiem_dinh"].astype(bool).any() and lv[res["muc_kiem_dinh"].astype(bool)].isna().any():
        raise ValueError(f"dong kiem dinh thieu cot muc '{level_col}'")
    ok = [bool(gate.get(int(v), False)) if t else np.nan
          for v, t in zip(lv, res["muc_kiem_dinh"].astype(bool))]
    res["cong_hoc_duoc"] = ok
    fail = res["muc_kiem_dinh"].astype(bool) & ~res["cong_hoc_duoc"].fillna(True).astype(bool)
    if fail.any():
        for c in DESC_NAN:
            res[c] = res[c].astype(object)  # cot bool (tost) khong nhan NaN
            res.loc[fail, c] = np.nan
        res.loc[fail, "label"] = MO_TA
        res.loc[fail, "kiem_dinh"] = False
    return res


# ---------------------------------------------------------
# Cong "mo hinh hoc duoc" (CHG-25 muc 3)
# ---------------------------------------------------------
def gate_grids(target: str, levels: pd.Series, scope: str = "muc_kiem_dinh", pair_grids=None) -> list:
    """Luoi vao ho Holm cua cong: moi luoi cua muc kiem dinh (mac dinh) hoac chi luoi trong cap F1."""
    if scope not in GATE_SCOPES:
        raise ValueError(f"scope phai thuoc {GATE_SCOPES}")
    tiers = tested_tiers(target)
    g = [x for x in sorted(levels.index) if levels[x] in tiers]
    if scope == "cap_f1":
        if pair_grids is None:
            raise ValueError("scope cap_f1 can pair_grids")
        g = [x for x in g if x in set(pair_grids)]
    return g


def _seed_signs(s: str) -> list:
    return [float(x) for x in str(s).split(";") if x != ""]


def learnability(err, point_unit, grids, levels: pd.Series, target: str, *, cv_units, holdout_units,
                 holdout_seasons, alpha: float = 0.05, scope: str = "muc_kiem_dinh", pair_grids=None,
                 model: str = "hist_gb", null_model: str = "season_mean", n_schemes: int = 3, expected_m=None,
                 n_perm=None) -> pd.DataFrame:
    """Moi luoi g: cap ((g, model), (g, null_model)), mode cv; Holm tren luoi cua cong (gate_grids).

    hoc_duoc = p_holm < alpha & Delta_hat < 0 & seeds_same_dir & vung_D (moi cach chia Delta < 0, du n_schemes).
    Nguong Delta_min khong dung (abs gia, require_practical False) -> cot nguong/TOST de NaN.
    """
    gg = gate_grids(target, levels, scope, pair_grids)
    if expected_m is not None and len(gg) != expected_m:
        raise AssertionError(f"{target}: cong co {len(gg)} luoi, ky vong {expected_m} (scope {scope})")
    kw = dict(cv_units=cv_units, holdout_units=holdout_units, holdout_seasons=holdout_seasons, mode="cv",
              delta_min=1.0, delta_min_kind="abs", alpha=alpha, require_practical=False,
              family=f"hoc_duoc_{target}", n_perm=n_perm)
    parts = []
    for flag, gs in ((True, gg), (False, [g for g in grids if g not in gg])):
        if not gs:
            continue
        pairs = [((g, model), (g, null_model)) for g in gs]
        o = compare_family(err, point_unit, pairs, expected_m=len(pairs), **kw)
        o["kiem_dinh"] = flag
        o["holm_m"] = len(pairs) if flag else 0
        parts.append(o)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "target", target)
    out.insert(1, "muc", out["grid_a"].map(levels).to_numpy())
    out["pham_vi_cong"] = scope
    seeds = [_seed_signs(s) for s in out["seed_deltas"]]
    out["vung_D"] = [len(s) == n_schemes and all(x < 0 for x in s) for s in seeds]
    hoc = ((out["p_holm"] < alpha) & (out["delta_hat"] < 0) & out["seeds_same_dir"].astype(bool) & out["vung_D"])
    out["hoc_duoc"] = hoc.astype(object)
    out["label"] = np.where(hoc, "hoc_duoc", "khong_hoc_duoc")
    desc = ~out["kiem_dinh"].astype(bool)
    out.loc[desc, ["p_value", "p_holm"]] = np.nan
    out.loc[desc, "hoc_duoc"] = np.nan
    out.loc[desc, "label"] = MO_TA
    for c in ("delta_min_thr", "delta_min", "tost_tuong_duong", "practical_ok", "ci_tost_low", "ci_tost_high"):
        out[c] = np.nan
    return out


def gate_table(hoc: pd.DataFrame, target: str, all_tiers=(5, 6, 7)) -> pd.DataFrame:
    """(bien, muc, hop_le): muc kiem dinh hop le <=> MOI luoi cua cong o muc do hoc_duoc."""
    tiers = tested_tiers(target)
    rows = []
    k = hoc[hoc["kiem_dinh"].astype(bool)]
    for m in all_tiers:
        sub = k[k["muc"] == m]
        if m not in tiers:
            rows.append({"bien": target, "muc": m, "n_luoi": 0, "n_hoc_duoc": 0, "hop_le": False,
                         "ly_do": "khong_kiem_dinh (CHG-25 muc 1)"})
            continue
        if sub.empty:
            raise ValueError(f"{target} muc {m}: khong co luoi nao trong cong")
        n_ok = int(sub["hoc_duoc"].astype(bool).sum())
        ok = n_ok == len(sub)
        rows.append({"bien": target, "muc": m, "n_luoi": len(sub), "n_hoc_duoc": n_ok, "hop_le": ok,
                     "ly_do": "moi_luoi_hoc_duoc" if ok else "co_luoi_khong_hoc_duoc"})
    out = pd.DataFrame(rows)
    out["holm_m"] = int(k["holm_m"].iloc[0]) if len(k) else 0
    out["pham_vi_cong"] = hoc["pham_vi_cong"].iloc[0] if len(hoc) else ""
    return out


def merge_gate(path, new: pd.DataFrame) -> pd.DataFrame:
    """Thay dong cua cac bien trong `new`, giu dong bien khac."""
    if os.path.isfile(path):
        old = pd.read_csv(path)
        old = old[~old["bien"].isin(set(new["bien"]))]
        new = pd.concat([old, new], ignore_index=True)
    return new.sort_values(["bien", "muc"]).reset_index(drop=True)

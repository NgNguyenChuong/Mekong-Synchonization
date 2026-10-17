"""
Quy tac phan tich Dot 7 (CHG-25 muc 1-4): muc kiem dinh theo bien, tach cap kiem dinh / mo ta, cong "hoc duoc"
(HistGB vs season_mean), chan ghi de file dong bang.
Ho Holm F1: mac dinh "cong" = muc kiem dinh hop le trong bang cong; "muc_kiem_dinh" = do nhay.
--model rf/mlp: do nhay theo mo hinh, file ra hau to __<model>, cong rieng tung mo hinh.
"""
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from training.block_stats import compare_family

TESTED_TIERS = {"salinity": frozenset({5, 6, 7}), "ndwi": frozenset({5, 6, 7}), "dsr_mcd18": frozenset({5, 6, 7}),
                "rain_chirps": frozenset({5}), "t2m_era5": frozenset(), "rh_era5": frozenset()}
# m F1 pham vi muc_kiem_dinh; pham vi cong suy tu bang cong (expected_f1_m)
EXPECTED_F1_M = {"salinity": 12, "ndwi": 12, "dsr_mcd18": 12, "rain_chirps": 3, "t2m_era5": 0, "rh_era5": 0}
# so luoi trong ho Holm cua cong hoc duoc theo pham vi (muc_kiem_dinh: moi luoi cua muc; cap_f1: luoi trong cap F1)
EXPECTED_GATE_M = {
    "muc_kiem_dinh": {"salinity": 13, "ndwi": 13, "dsr_mcd18": 13, "rain_chirps": 4, "t2m_era5": 0, "rh_era5": 0},
    "cap_f1": {"salinity": 10, "ndwi": 10, "dsr_mcd18": 10, "rain_chirps": 3, "t2m_era5": 0, "rh_era5": 0},
}
GATE_SCOPES = tuple(EXPECTED_GATE_M)
F1_PAIRS_PER_TIER = {5: 3, 6: 3, 7: 6}  # ho F1 12 cap (ti le dien tich <= 1,2)
HOLM_SCOPES = ("cong", "muc_kiem_dinh")
SENS_SUFFIX = "__holm_muc_kiem_dinh"
MO_TA = "mo_ta"
DESC_NAN = ("p_value", "p_holm", "tost_tuong_duong")
GATE_COLS = ("bien", "muc", "hop_le")
FROZEN_EXIT = 2
MODELS = ("hist_gb", "rf", "mlp")
RUN_MODEL = {"rf": "random_forest"}  # ten mo hinh trong ten luot cua runner
SENS_ROLE = "do_nhay_mo_hinh"
MULTI_TAG_REASON = "duong huan luyen e6a->e6e diff 0 dong (method-reviewer 09/10)"


def tested_tiers(target: str) -> frozenset:
    if target not in TESTED_TIERS:
        raise ValueError(f"target '{target}' khong co trong TESTED_TIERS {sorted(TESTED_TIERS)}")
    return TESTED_TIERS[target]


# ---------------------------------------------------------
# Mo hinh: ten luot, ten file, ho chinh
# ---------------------------------------------------------
def check_model(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"model '{model}' phai thuoc {MODELS}")
    return model


def run_model(model: str) -> str:
    return RUN_MODEL.get(model, model)


def model_name(name, model: str) -> str:
    """hist_gb -> giu nguyen; rf/mlp -> hau to __<model> truoc phan mo rong (ten file hoac duong dan)."""
    if check_model(model) == "hist_gb":
        return name
    root, ext = os.path.splitext(name)
    return f"{root}__{model}{ext}"


def main_family(model: str) -> str:
    return "F1_chinh" if check_model(model) == "hist_gb" else f"do_nhay_{model}"


def gate_required(target: str, model: str) -> bool:
    """Cong bat buoc: HistGB moi bien tru do man (giu quy tac cu); rf/mlp moi bien."""
    return check_model(model) != "hist_gb" or target != "salinity"


def model_provenance(model: str) -> dict:
    """Khoa provenance cho rf/mlp; hist_gb -> {} (sidecar giu nguyen)."""
    if check_model(model) == "hist_gb":
        return {}
    return {"model": model, "run_model": run_model(model), "vai_tro": SENS_ROLE}


# ---------------------------------------------------------
# Chan ghi de file dong bang
# ---------------------------------------------------------
def _manifest_list(manifests) -> list:
    ms = [manifests] if isinstance(manifests, (str, os.PathLike)) else list(manifests or [])
    if not ms:
        raise FileNotFoundError("khong co manifest dong bang nao")
    return ms


def frozen_paths(manifests) -> set:
    """Hop duong dan tuyet doi (normcase) cua mot hoac nhieu manifest; cot 'file' tuong doi goc repo (2 cap tren
    manifest). Thieu bat ky manifest nao -> FileNotFoundError."""
    out = set()
    for manifest_csv in _manifest_list(manifests):
        if not os.path.isfile(manifest_csv):
            raise FileNotFoundError(f"khong co manifest dong bang {manifest_csv}")
        m = pd.read_csv(manifest_csv, dtype=str)
        if "file" not in m.columns:
            raise ValueError(f"{manifest_csv}: thieu cot 'file'")
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(manifest_csv))))
        out |= {os.path.normcase(os.path.abspath(os.path.join(root, f))) for f in m["file"].dropna()}
    return out


def frozen_hits(paths, manifests) -> list:
    frozen = frozen_paths(manifests)
    return [p for p in paths if os.path.normcase(os.path.abspath(p)) in frozen]


def guard_frozen(paths, manifests):
    """Duong dan nao co trong mot manifest dong bang (hoac thieu manifest) -> SystemExit ma 2, truoc khi doc/ghi."""
    try:
        hits = frozen_hits(paths, manifests)
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


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_path(path) -> str:
    return f"{path}.provenance.json"


def write_provenance(csv_path, **info) -> str:
    """<csv>.provenance.json (khong them cot vao CSV)."""
    rec = {"output": os.path.basename(csv_path), "csv_sha256": file_sha256(csv_path),
           "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "script": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else None, **info}
    path = provenance_path(csv_path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------
# Pham vi ho Holm F1
# ---------------------------------------------------------
def holm_tiers(target: str, gate, scope: str = "cong") -> frozenset:
    """Muc vao Holm F1. cong: muc kiem dinh co hop_le trong bang cong (gate None -> moi muc kiem dinh);
    muc_kiem_dinh: moi muc kiem dinh (cong van ep mo_ta sau do)."""
    if scope not in HOLM_SCOPES:
        raise ValueError(f"holm_scope phai thuoc {HOLM_SCOPES}")
    tiers = tested_tiers(target)
    if scope == "muc_kiem_dinh" or gate is None:
        return tiers
    return frozenset(t for t in tiers if gate.get(t, False))


def expected_f1_m(tiers) -> int:
    return sum(F1_PAIRS_PER_TIER[t] for t in tiers)


def scopes_differ(target: str, gate) -> bool:
    return holm_tiers(target, gate, "cong") != holm_tiers(target, gate, "muc_kiem_dinh")


def scoped_name(name: str, scope: str, differ: bool, model: str = "hist_gb") -> str:
    """Pham vi do nhay khac pham vi chinh -> hau to __holm_muc_kiem_dinh; sau do hau to mo hinh (model_name)."""
    if scope not in HOLM_SCOPES:
        raise ValueError(f"holm_scope phai thuoc {HOLM_SCOPES}")
    if scope == "muc_kiem_dinh" and differ:
        root, ext = os.path.splitext(name)
        name = f"{root}{SENS_SUFFIX}{ext}"
    return model_name(name, model)


def all_scoped_names(name: str, model: str = "hist_gb") -> list:
    return [scoped_name(name, "cong", False, model), scoped_name(name, "muc_kiem_dinh", True, model)]


def holm_provenance(target: str, gate, gate_path, scope: str, res: pd.DataFrame = None) -> dict:
    """Thong tin pham vi Holm cho sidecar: muc, m theo ho, bang cong (duong dan + sha256)."""
    tiers = tested_tiers(target)
    ht = holm_tiers(target, gate, scope)
    info = {"target": target, "holm_scope": scope, "holm_scopes_khac_nhau": scopes_differ(target, gate),
            "muc_kiem_dinh": sorted(int(t) for t in tiers), "muc_holm": sorted(int(t) for t in ht),
            "muc_truot_cong": sorted(int(t) for t in tiers if gate is not None and not gate.get(t, False)),
            "holm_m_f1": expected_f1_m(ht),
            "gate_file": os.path.abspath(gate_path).replace("\\", "/") if gate is not None else None,
            "gate_sha256": file_sha256(gate_path) if gate is not None else None,
            "gate_hop_le": {str(k): bool(v) for k, v in sorted(gate.items())} if gate is not None else None}
    if res is not None and "holm_m" in res.columns:
        info["holm_m_theo_ho"] = {str(f): int(m) for f, m in res.groupby("family", sort=False)["holm_m"].max().items()}
    return info


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


def compare_tiered(err, point_unit, pairs: pd.DataFrame, target: str, *, levels, expected_m=None, holm_tiers=None,
                   **kw):
    """compare_family rieng cho cap o muc Holm (holm_tiers, mac dinh moi muc kiem dinh) va cap con lai (mo ta).

    Cot them: muc, muc_kiem_dinh (muc thuoc TESTED_TIERS), kiem_dinh, holm_m. Thu tu dong = thu tu `pairs`.
    """
    tiers = tested_tiers(target)
    holm = tiers if holm_tiers is None else frozenset(holm_tiers)
    if not holm <= tiers:
        raise ValueError(f"{target}: holm_tiers {sorted(holm)} ngoai muc kiem dinh {sorted(tiers)}")
    p = pairs.reset_index(drop=True)
    tier = _tier_of_pairs(p, levels)
    if tier.isna().any():
        raise ValueError("cap khong xac dinh duoc muc (tier)")
    is_test = tier.isin(holm).to_numpy()
    if expected_m is not None and int(is_test.sum()) != expected_m:
        raise AssertionError(f"{target}: {int(is_test.sum())} cap trong ho Holm, ky vong {expected_m}")
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
        o["muc_kiem_dinh"] = tier[sub.index].isin(tiers).to_numpy()
        parts.append(o)
    out = pd.concat(parts, ignore_index=True).sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    out.insert(0, "target", target)
    return out


# ---------------------------------------------------------
# Cong kiem dinh (doc / ap)
# ---------------------------------------------------------
def _gate_models(g: pd.DataFrame) -> set:
    return set(g["model"].astype(str)) if "model" in g.columns else {"hist_gb"}  # bang cong khong cot model = HistGB


def load_gate(path, target: str, required: bool, model: str = None):
    """Bang cong (bien, muc, hop_le) cua target. Thieu file / thieu dong cho muc kiem dinh -> ValueError khi required.
    model khac None: dong cua target phai thuoc dung mo hinh do (cot model; khong co cot = hist_gb).

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
    if model is not None and _gate_models(g) != {check_model(model)}:
        raise ValueError(f"{path}: bang cong cua mo hinh {sorted(_gate_models(g))}, can '{model}'")
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
    """Thay dong cua cac bien trong `new`, giu dong bien khac; khong tron cong cua hai mo hinh."""
    if os.path.isfile(path):
        old = pd.read_csv(path)
        if len(old) and _gate_models(old) != _gate_models(new):
            raise ValueError(f"{path}: cong mo hinh {sorted(_gate_models(old))} khac {sorted(_gate_models(new))}")
        old = old[~old["bien"].isin(set(new["bien"]))]
        new = pd.concat([old, new], ignore_index=True)
    return new.sort_values(["bien", "muc"]).reset_index(drop=True)

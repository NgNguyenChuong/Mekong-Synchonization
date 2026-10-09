"""
Tuong tac mo hinh x khung luoi theo khoi: I_b = d_b(mo hinh) - d_b(tham chieu), d_b = MAE_a - MAE_b trong khoi b
(d_b, w_b cung duong tinh voi compare_family). Doi dau theo khoi hai phia, CI t theo cum, TOST, Holm tren ho cap.
"""
import numpy as np
import pandas as pd

from training.block_stats import KEY, _wmean, cluster_t_ci, comparison_seed, signflip_test
from training.evaluate import holm_adjust
from training.pc1 import pair_unit_deltas

SIG = "tuong_tac_co_y_nghia"
NONE = "khong_tuong_tac"
COLS = ["family", "cmp_id", "grid_a", "grid_b", "model", "model_ref", "G", "I_hat", "ci_low", "ci_high",
        "ci_tost_low", "ci_tost_high", "se", "p_value", "p_exact", "n_perm", "k_same", "k_over_G", "seed_deltas",
        "n_seeds", "seeds_same_dir", "delta_min_thr_histgb", "tost_tuong_tac", "delta_hat_histgb", "delta_hat_model",
        "p_holm", "holm_m", "label", "chua_ro", "alpha", "min_pts"]


def point_season_set(err: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(err[KEY].drop_duplicates()).sort_values()


def check_same_points(err_ref: pd.DataFrame, err_mod: pd.DataFrame) -> int:
    """Hai mo hinh phai cung tap (point_id, season); lech -> ValueError. Tra ve so (diem, mua)."""
    a, b = point_season_set(err_ref), point_season_set(err_mod)
    if not a.equals(b):
        raise ValueError(f"tap (diem, mua) lech giua hai mo hinh: {len(a.difference(b))} chi o mo hinh tham chieu, "
                         f"{len(b.difference(a))} chi o mo hinh so sanh")
    return len(a)


def _check_rows(err, model, point_unit, cv_units, holdout_units, holdout_seasons):
    """Chi hang oof cua khoi CV (mode cv), dung mot mo hinh, khong co mua giu rieng."""
    if set(err["model"]) != {model}:
        raise ValueError(f"err phai chi co mo hinh '{model}', co {sorted(set(err['model']))}")
    if (err["pred_source"] != "oof").any():
        raise ValueError(f"{model}: co hang pred_source khac 'oof'")
    unit = err["point_id"].map(point_unit)
    if unit.isna().any():
        raise ValueError(f"{model}: {err.loc[unit.isna(), 'point_id'].nunique()} diem khong co don vi")
    if unit.isin(set(holdout_units)).any():
        raise ValueError(f"{model}: co hang thuoc khoi giu rieng - kiem tuong tac chi tren khoi CV")
    if not unit.isin(set(cv_units)).all():
        raise ValueError(f"{model}: co don vi ngoai cv_units")
    if err["season"].isin(holdout_seasons).any():
        raise ValueError(f"{model}: co hang thuoc mua giu rieng {tuple(holdout_seasons)}")


def interaction_family(err_ref: pd.DataFrame, err_mod: pd.DataFrame, point_unit: pd.Series, pairs, delta_min_thr, *,
                       ref_model: str, model: str, cv_units, holdout_units, holdout_seasons, alpha: float = 0.05,
                       min_pts: int = 30, n_schemes=None, family: str = "tuong_tac", base_seed: int = 42,
                       n_perm=None) -> pd.DataFrame:
    """Moi cap (grid_a, grid_b): I_b theo khoi, I_hat = sum w I_b, p doi dau (G <= 20 chinh xac), Holm m = len(pairs).

    err_ref / err_mod: err_long (grid, model, seed, point_id, season, err, pred_source) cua hai mo hinh, da loc tap
    chung theo tung mo hinh; hai tap (diem, mua) va tap cach chia phai trung. delta_min_thr: nguong TOST moi cap.
    label = SIG khi p_holm < alpha, I_hat != 0 va moi cach chia cung dau I_hat; chua_ro = p_holm < alpha nhung khong.
    """
    if not 0 < alpha < 0.5:
        raise ValueError("alpha phai trong (0; 0,5)")
    holdout_seasons = tuple(sorted(set(holdout_seasons)))
    if not holdout_seasons:
        raise ValueError("holdout_seasons rong")
    if set(cv_units) & set(holdout_units):
        raise ValueError("cv_units va holdout_units giao nhau")
    pairs = [tuple(p) for p in pairs]
    thrs = [float(t) for t in delta_min_thr]
    if len(thrs) != len(pairs):
        raise ValueError("delta_min_thr phai cung do dai pairs")
    if not pairs:
        return pd.DataFrame(columns=COLS)
    for e, m in ((err_ref, ref_model), (err_mod, model)):
        _check_rows(e, m, point_unit, cv_units, holdout_units, holdout_seasons)
    check_same_points(err_ref, err_mod)
    seeds = sorted(set(err_ref["seed"]))
    if seeds != sorted(set(err_mod["seed"])):
        raise ValueError(f"tap cach chia khac nhau: {seeds} vs {sorted(set(err_mod['seed']))}")
    if n_schemes is not None and len(seeds) != n_schemes:
        raise ValueError(f"can {n_schemes} cach chia, co {seeds}")
    rows = []
    for (ga, gb), thr in zip(pairs, thrs):
        if not (np.isfinite(thr) and thr > 0):
            raise ValueError(f"{ga}/{gb}: delta_min_thr khong huu han / <= 0 ({thr})")
        cmp_id = f"{family}:{ga}-vs-{gb}|{model}-minus-{ref_model}"
        d_r, w, ds_r = pair_unit_deltas(err_ref, point_unit, (ga, ref_model), (gb, ref_model), min_pts)
        d_m, w_m, ds_m = pair_unit_deltas(err_mod, point_unit, (ga, model), (gb, model), min_pts)
        if not d_r.index.equals(d_m.index):
            raise AssertionError(f"{cmp_id}: hai mo hinh khac tap khoi sau loc min_pts")
        if not np.allclose(w.to_numpy(), w_m.to_numpy()):
            raise AssertionError(f"{cmp_id}: trong so khoi khac nhau giua hai mo hinh")
        inter = d_m - d_r
        G = len(inter)
        if G < 2:
            raise ValueError(f"{cmp_id}: chi con {G} khoi")
        sf = signflip_test(inter.to_numpy(), w.to_numpy(), seed=comparison_seed(cmp_id, base_seed), n_perm=n_perm)
        ci = cluster_t_ci(inter.to_numpy(), w.to_numpy(), alpha)
        ci2 = cluster_t_ci(inter.to_numpy(), w.to_numpy(), 2 * alpha)
        i_hat = sf["delta_hat"]
        sgn = np.sign(i_hat)
        seed_i = [_wmean(ds_m[s] - ds_r[s], w) for s in seeds]
        k_same = int((np.sign(inter) == sgn).sum()) if sgn != 0 else 0
        rows.append({
            "family": family, "cmp_id": cmp_id, "grid_a": ga, "grid_b": gb, "model": model, "model_ref": ref_model,
            "G": G, "I_hat": i_hat, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "ci_tost_low": ci2["ci_low"], "ci_tost_high": ci2["ci_high"], "se": ci["se"],
            "p_value": sf["p_value"], "p_exact": sf["exact"], "n_perm": sf["n_perm"],
            "k_same": k_same, "k_over_G": f"{k_same}/{G}",
            "seed_deltas": ";".join(f"{x:.6g}" for x in seed_i), "n_seeds": len(seeds),
            "seeds_same_dir": bool(sgn != 0 and all(np.sign(x) == sgn for x in seed_i)),
            "delta_min_thr_histgb": thr,
            "tost_tuong_tac": bool(-thr < ci2["ci_low"] and ci2["ci_high"] < thr),
            "delta_hat_histgb": _wmean(d_r, w), "delta_hat_model": _wmean(d_m, w),
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
    out["holm_m"] = len(out)
    sig = (out["p_holm"] < alpha) & (out["I_hat"] != 0)
    out["label"] = np.where(sig & out["seeds_same_dir"], SIG, NONE)
    out["chua_ro"] = sig & ~out["seeds_same_dir"]
    out["alpha"] = alpha
    out["min_pts"] = min_pts
    return out[COLS]


def verdict(n_cap: int, n_sig: int, n_unclear: int) -> str:
    if n_cap == 0:
        return "khong_co_cap"
    if n_sig > 0:
        return "co_tuong_tac"
    return "chua_ro" if n_unclear > 0 else "khong_phu_thuoc_mo_hinh"

"""
Doi chung duong thong ke PC-1: tiem chenh biet truoc vao sai so cua khung B trong mot cap F1, chay lai quy trinh F1
(mode cv) o muc don vi. d_b* = Delta_hat + s_b (d_b - Delta_hat) - delta_b; s_b = +-1 chung 3 cach chia; 11 cap con lai
giu p dong bang; Holm tren ca ho; nguong thr co dinh tu ban dong bang.
"""
import hashlib

import numpy as np
import pandas as pd

from training.block_stats import (EXACT_MAX_G, _pair_d, cluster_t_ci, seed_mean_errors,
                                  unit_metrics)
from training.evaluate import holm_adjust

KINDS = ("deu", "lognormal", "tau_mu")
LOGN_CV = 0.76  # CV cua |N(0,1)| ~ sqrt(pi/2 - 1): hang so, khong lay tu du lieu


# ---------------------------------------------------------
# d_b theo don vi (cung duong tinh voi compare_family)
# ---------------------------------------------------------
def pair_unit_deltas(err_cv: pd.DataFrame, point_unit: pd.Series, arm_a, arm_b, min_pts: int = 30):
    """d_b = MAE_A - MAE_B theo don vi: TB qua cach chia (Series) + theo tung cach chia (DataFrame don vi x seed) + w_b.

    err_cv: cot grid, model, seed, point_id, season, err - chi hang oof cua khoi CV, da loc tap diem chung.
    """
    keys = err_cv.set_index(["grid", "model"]).index
    sub = err_cv[keys.isin([arm_a, arm_b])]
    if sub.empty:
        raise ValueError(f"khong co du lieu cho {arm_a} / {arm_b}")
    units, _, _ = unit_metrics(seed_mean_errors(sub), point_unit, min_pts=min_pts)
    d, w = _pair_d(units, arm_a, arm_b, "mae")
    cols = {}
    for s in sorted(sub["seed"].unique()):
        us = unit_metrics(seed_mean_errors(sub[sub["seed"] == s]), point_unit, min_pts=min_pts)[0]
        ds, ws = _pair_d(us, arm_a, arm_b, "mae")
        if not ds.index.equals(d.index) or not np.allclose(ws.to_numpy(), w.to_numpy()):
            raise AssertionError(f"cach chia {s}: tap don vi / trong so khac TB qua cach chia")
        cols[s] = ds
    d_seed = pd.DataFrame(cols)
    # MAE tuyen tinh theo |e| -> d TB qua cach chia = TB cua d tung cach chia
    if not np.allclose(d_seed.mean(axis=1).to_numpy(), d.to_numpy(), rtol=0, atol=1e-12):
        raise AssertionError("d TB qua cach chia khac TB cua d tung cach chia")
    return d, w, d_seed


# ---------------------------------------------------------
# Tiem va lay mau lai
# ---------------------------------------------------------
def rep_rng(rep_seed: int, cmp_id: str, stream: int) -> np.random.Generator:
    """RNG cua mot lan lap, khoa theo (seed lap, ma cap, luong); luong 0 = dau wild, 1+ = he so tiem."""
    key = int(hashlib.sha256(cmp_id.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(np.random.SeedSequence(int(rep_seed), spawn_key=(key, int(stream))))


def wild_signs(rep_seeds, G: int, cmp_id: str) -> np.ndarray:
    """(n_rep, G) dau +-1 theo don vi; dung chung cho moi k, moi kieu tiem va 3 cach chia."""
    return np.stack([rep_rng(r, cmp_id, 0).integers(0, 2, G) * 2 - 1 for r in rep_seeds]).astype(float)


def injection_pattern(kind: str, wn: np.ndarray, rep_seeds, cmp_id: str) -> np.ndarray:
    """(n_rep, G) he so x_b voi sum wn_b x_b = 1 dung; delta_b = k * thr * x_b.

    deu: x = 1. lognormal: CV LOGN_CV, x >= 0. tau_mu: N(1, 1) dich ve TB 1 (cho phep am - chi bao).
    """
    wn = np.asarray(wn, dtype=float)
    G = len(wn)
    if kind not in KINDS:
        raise ValueError(f"kieu tiem '{kind}' khong thuoc {KINDS}")
    if not np.isclose(wn.sum(), 1.0) or (wn <= 0).any():
        raise ValueError("wn phai > 0 va tong 1")
    if kind == "deu":
        return np.ones((len(rep_seeds), G))
    stream = 1 + KINDS.index(kind)
    rows = []
    for r in rep_seeds:
        rng = rep_rng(r, cmp_id, stream)
        if kind == "lognormal":
            s2 = np.log1p(LOGN_CV ** 2)
            x = rng.lognormal(-s2 / 2, np.sqrt(s2), G)
            x = x / (wn @ x)
        else:
            x = rng.normal(1.0, 1.0, G)
            x = x - wn @ x + 1.0
        rows.append(x)
    return np.stack(rows)


def inject_abs_err(err_long: pd.DataFrame, arm, point_unit: pd.Series, delta_by_unit) -> pd.DataFrame:
    """Tiem o muc diem: |e'| = |e| + delta_{don vi cua diem} cho moi diem/mua/cach chia cua nhanh `arm`.

    Hang so trong don vi -> MAE (don vi, mua) va MAE don vi (TB deu theo mua) dich dung delta_b.
    Giu dau e (e = 0 tinh la +). delta < 0 khong cai duoc o muc diem (|e'| co the < 0) -> loi.
    """
    delta = pd.Series(delta_by_unit, dtype=float)
    if (delta < 0).any() or not np.isfinite(delta).all():
        raise ValueError("delta_by_unit phai huu han va >= 0 khi tiem o muc diem")
    out = err_long.copy()
    m = (out["grid"] == arm[0]) & (out["model"] == arm[1])
    du = out.loc[m, "point_id"].map(point_unit).map(delta)
    if du.isna().any():
        raise ValueError(f"{int(du.isna().sum())} hang cua {arm} khong co delta theo don vi")
    e = out.loc[m, "err"].to_numpy(dtype=float)
    out.loc[m, "err"] = np.where(e < 0, -1.0, 1.0) * (np.abs(e) + du.to_numpy())
    return out


# ---------------------------------------------------------
# Kiem dinh theo lo (giong signflip_test)
# ---------------------------------------------------------
def signflip_p_batch(D, wn, n_perm=None, rng=None) -> np.ndarray:
    """p doi dau hai phia cho nhieu vector d (hang cua D) cung trong so.

    n_perm None va G <= EXACT_MAX_G: liet ke du 2^G, cung dung sai/quy tac >= nhu signflip_test.
    n_perm so nguyen: Monte Carlo, n_perm to hop dau dung chung moi hang, p = (1 + #) / (n_perm + 1).
    """
    D = np.atleast_2d(np.asarray(D, dtype=float))
    wn = np.asarray(wn, dtype=float)
    G = D.shape[1]
    if len(wn) != G or not np.isfinite(D).all():
        raise ValueError("D/wn sai kich thuoc hoac co NaN")
    WD = D * wn
    obs = np.abs(WD.sum(axis=1))
    tol = 1e-12 * np.maximum(1.0, np.abs(WD).sum(axis=1))
    cnt = np.zeros(len(D), dtype=np.int64)
    if n_perm is None:
        if G > EXACT_MAX_G:
            raise ValueError(f"G = {G} > {EXACT_MAX_G}: can n_perm (Monte Carlo)")
        total = 1 << G
        bits = np.arange(G, dtype=np.int64)
        chunk = 1 << 16
        for start in range(0, total, chunk):
            masks = np.arange(start, min(total, start + chunk), dtype=np.int64)
            signs = (((masks[:, None] >> bits) & 1) * 2 - 1).astype(float)
            cnt += (np.abs(signs @ WD.T) >= obs - tol).sum(axis=0)
        return cnt / total
    if rng is None:
        raise ValueError("Monte Carlo can rng")
    n_perm = int(n_perm)
    chunk = 10_000
    for start in range(0, n_perm, chunk):
        b = min(chunk, n_perm - start)
        signs = (rng.integers(0, 2, size=(b, G)) * 2 - 1).astype(float)
        cnt += (np.abs(signs @ WD.T) >= obs - tol).sum(axis=0)
    return (1 + cnt) / (n_perm + 1)


def cv_label(holm_ok: bool, seeds_same: bool, tost: bool) -> str:
    """Nhan mode cv cua compare_family."""
    if holm_ok and seeds_same:
        return "cho_giu_rieng"
    if not holm_ok and tost:
        return "tuong_duong"
    return "chua_phan_dinh"


def simulate_pair(d, d_seed: pd.DataFrame, w, thr: float, p_family, pair_index: int, alpha: float,
                  flips, inj, n_perm=None, rng_perm=None) -> pd.DataFrame:
    """Chay lai F1 cho MOT cap tren n lan lap (hang cua flips/inj).

    d, d_seed, w: tu pair_unit_deltas. thr: nguong co dinh (khong tinh lai). p_family: p cua ca ho (dong bang);
    chi phan tu pair_index duoc thay. flips (n, G) +-1; inj (n, G) = delta_b (B sai so tang -> d giam).
    """
    d = np.asarray(d, dtype=float)
    Ds = np.asarray(d_seed, dtype=float).T  # (S, G)
    w = np.asarray(w, dtype=float)
    wn = w / w.sum()
    flips = np.atleast_2d(np.asarray(flips, dtype=float))
    inj = np.atleast_2d(np.asarray(inj, dtype=float))
    G = len(d)
    if flips.shape != inj.shape or flips.shape[1] != G or Ds.shape[1] != G:
        raise ValueError("flips/inj/d_seed sai kich thuoc")
    if not np.isin(flips, (-1.0, 1.0)).all():
        raise ValueError("flips phai la +-1")
    if not (np.isfinite(thr) and thr > 0):
        raise ValueError(f"thr phai huu han va > 0 (nhan {thr})")
    p_family = np.asarray(p_family, dtype=float)
    if not 0 <= pair_index < len(p_family):
        raise ValueError("pair_index ngoai ho")
    dhat = float(wn @ d)
    dhat_s = Ds @ wn
    dstar = dhat + flips * (d - dhat) - inj
    dstar_s = dhat_s[None, :, None] + flips[:, None, :] * (Ds - dhat_s[:, None])[None] - inj[:, None, :]
    delta = dstar @ wn
    delta_s = dstar_s @ wn  # (n, S)
    p = signflip_p_batch(dstar, wn, n_perm=n_perm, rng=rng_perm)
    rows = []
    for i in range(len(dstar)):
        pf = p_family.copy()
        pf[pair_index] = p[i]
        p_holm = float(holm_adjust(pf)[pair_index])
        ci2 = cluster_t_ci(dstar[i], w, 2 * alpha)
        sgn = np.sign(delta[i])
        seeds_same = bool(sgn != 0 and (np.sign(delta_s[i]) == sgn).all())
        holm_ok = bool(p_holm < alpha and delta[i] != 0)
        tost = bool(-thr < ci2["ci_low"] and ci2["ci_high"] < thr)
        dung_chieu = bool(delta[i] < 0)
        rows.append({
            "delta_star": float(delta[i]), "inj_mean": float(wn @ inj[i]), "inj_min": float(inj[i].min()),
            "n_inj_am": int((inj[i] < 0).sum()),
            "seed_deltas": ";".join(f"{x:.6g}" for x in delta_s[i]),
            "p_value": float(p[i]), "p_holm": p_holm,
            "ci_tost_low": ci2["ci_low"], "ci_tost_high": ci2["ci_high"], "se": ci2["se"],
            "seeds_same_dir": seeds_same, "co_y_nghia": holm_ok, "dung_chieu": dung_chieu,
            "tost_tuong_duong": tost, "practical_ok": bool(abs(delta[i]) >= thr),
            "phat_hien_vung_D": bool(holm_ok and dung_chieu and seeds_same),
            "phat_hien_tren_nguong": bool(holm_ok and dung_chieu and seeds_same and abs(delta[i]) >= thr),
            "label": cv_label(holm_ok, seeds_same, tost),
            "delta_min_thr": float(thr),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Doi chieu ban dong bang, tong hop, ket luan
# ---------------------------------------------------------
def match_frozen(pairs, frozen: pd.DataFrame) -> pd.DataFrame:
    """Hang dong bang theo dung thu tu `pairs` [(grid_a, grid_b)]; thieu/trung/thua cap -> ValueError LOI."""
    fz = frozen.copy()
    key = list(zip(fz["grid_a"], fz["grid_b"]))
    if len(set(key)) != len(key):
        raise ValueError("LOI: ban dong bang co cap trung")
    want = [tuple(p) for p in pairs]
    miss = [p for p in want if p not in key]
    extra = [p for p in key if p not in want]
    if miss or extra:
        raise ValueError(f"LOI: ban dong bang lech ho - thieu {miss}, thua {extra}")
    fz.index = pd.MultiIndex.from_tuples(key)
    return fz.loc[want].reset_index(drop=True)


def compare_identity(ident: pd.DataFrame, frozen: pd.DataFrame, tol: float = 1e-9) -> list:
    """So lan chay dong nhat (s = +1, delta = 0) voi ban dong bang; tra ve danh sach lech (rong = khop)."""
    bad = []
    for c in ("delta_hat", "p_value", "p_holm", "ci_tost_low", "ci_tost_high", "delta_min_thr"):
        x, y = ident[c].to_numpy(float), frozen[c].to_numpy(float)
        if not np.allclose(x, y, rtol=tol, atol=tol):
            bad.append(f"{c}: lech toi da {np.max(np.abs(x - y)):.3g}")
    for c in ("seeds_same_dir", "tost_tuong_duong", "label"):
        a = ident[c].astype(str).str.lower().to_numpy()
        b = frozen[c].astype(str).str.lower().to_numpy()
        if (a != b).any():
            bad.append(f"{c}: {int((a != b).sum())} cap khac")
    return bad


def run_pair(pair_index: int, grid_a: str, grid_b: str, level, is_fine: bool, d, d_seed, w, thr: float,
             p_family, alpha: float, ks, kinds, rep_seeds, cmp_id: str, n_perm=None) -> pd.DataFrame:
    """Moi (k, kieu) x lan lap cho mot cap. Dau wild va he so tiem dung chung giua cac k (so ngau nhien chung)."""
    w = np.asarray(w, dtype=float)
    wn = w / w.sum()
    G = len(wn)
    flips = wild_signs(rep_seeds, G, cmp_id)
    parts = []
    for kind in kinds:
        x = injection_pattern(kind, wn, rep_seeds, cmp_id)
        for k in ks:
            rng_perm = None if n_perm is None else rep_rng(10_000 + int(round(100 * k)), cmp_id, 99)
            r = simulate_pair(d, d_seed, w, thr, p_family, pair_index, alpha, flips, k * thr * x,
                              n_perm=n_perm, rng_perm=rng_perm)
            r.insert(0, "rep", list(rep_seeds))
            r.insert(0, "kind", kind)
            r.insert(0, "k", k)
            for c, v in (("is_fine", bool(is_fine)), ("level", level), ("grid_b", grid_b), ("grid_a", grid_a),
                         ("pair_index", pair_index)):
                r.insert(0, c, v)
            r["delta_obs"] = float(wn @ np.asarray(d, dtype=float))
            r["inj_err"] = (r["inj_mean"] - k * thr).abs()
            parts.append(r)
    return pd.concat(parts, ignore_index=True)


GROUP = ["pair_index", "grid_a", "grid_b", "level", "is_fine", "k", "kind"]


def summarize(reps: pd.DataFrame) -> pd.DataFrame:
    """Theo (cap, k, kieu): ti le phat hien vung D, ti le TOST tuong duong, trung vi Delta*."""
    g = reps.groupby(GROUP, sort=False)
    out = g.agg(n_rep=("rep", "size"), delta_obs=("delta_obs", "first"), thr=("delta_min_thr", "first"),
                ty_le_phat_hien=("phat_hien_vung_D", "mean"), ty_le_co_y_nghia=("co_y_nghia", "mean"),
                ty_le_phat_hien_tren_nguong=("phat_hien_tren_nguong", "mean"),
                ty_le_tost=("tost_tuong_duong", "mean"), trung_vi_delta_star=("delta_star", "median"),
                inj_mean_lech_max=("inj_err", "max")).reset_index()
    out["trung_vi_delta_star_tren_thr"] = out["trung_vi_delta_star"] / out["thr"]
    return out


def verdict(summary: pd.DataFrame, n_fine: int = 6, k_crit: float = 2, kinds_crit=("deu", "lognormal"),
            min_detect: float = 0.80, max_tost: float = 0.05, n_rep: int = None) -> dict:
    """Tieu chi chot: o k_crit, moi cap min x kieu kinds_crit: phat hien vung D >= min_detect VA TOST <= max_tost.

    trang_thai: DAT / KHONG_DAT / LOI (thieu cap min, thieu to hop, so lan lap lech).
    """
    fine = summary[summary["is_fine"].astype(bool)]
    fine_pairs = sorted(set(zip(fine["grid_a"], fine["grid_b"])))
    res = {"trang_thai": "LOI", "ly_do": "", "n_cap_min": len(fine_pairs), "k": k_crit,
           "kieu": ";".join(kinds_crit), "min_phat_hien": np.nan, "max_tost": np.nan,
           "cap_yeu_nhat": "", "nguong_phat_hien": min_detect, "nguong_tost": max_tost}
    if len(fine_pairs) != n_fine:
        res["ly_do"] = f"so cap min {len(fine_pairs)} != {n_fine}"
        return res
    crit = fine[np.isclose(fine["k"].astype(float), k_crit) & fine["kind"].isin(kinds_crit)]
    have = set(zip(crit["grid_a"], crit["grid_b"], crit["kind"]))
    need = {(a, b, kd) for a, b in fine_pairs for kd in kinds_crit}
    if need - have:
        res["ly_do"] = f"thieu to hop tieu chi: {sorted(need - have)}"
        return res
    if crit.duplicated(["grid_a", "grid_b", "kind"]).any():
        res["ly_do"] = "to hop tieu chi trung lap"
        return res
    if n_rep is not None and (crit["n_rep"] != n_rep).any():
        res["ly_do"] = f"so lan lap khac {n_rep}"
        return res
    worst = crit.loc[crit["ty_le_phat_hien"].idxmin()]
    res["min_phat_hien"] = float(crit["ty_le_phat_hien"].min())
    res["max_tost"] = float(crit["ty_le_tost"].max())
    res["cap_yeu_nhat"] = f"{worst['grid_a']} vs {worst['grid_b']} ({worst['kind']})"
    ok = res["min_phat_hien"] >= min_detect and res["max_tost"] <= max_tost
    res["trang_thai"] = "DAT" if ok else "KHONG_DAT"
    res["ly_do"] = (f"min phat hien vung D {res['min_phat_hien']:.3f} (>= {min_detect}), "
                    f"max TOST {res['max_tost']:.3f} (<= {max_tost})")
    return res

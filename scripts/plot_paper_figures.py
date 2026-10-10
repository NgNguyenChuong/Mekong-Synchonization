#!/usr/bin/env python
"""Hinh bai bao (Fig 1-6, S1-S2) theo KE_HOACH/quy_uoc_hinh.md: tieng Anh, Okabe-Ito, PDF vector + PNG 600 dpi.
Chi doc ket qua da mo; ngoai le duy nhat: dai null Fig 6 mo phong tu se cua cap F1 HistGB.

Chay:  venv/Scripts/python.exe scripts/plot_paper_figures.py [--only fig3 fig6] [--out-dir KE_HOACH/ket-qua/hinh/paper]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import plot_dot7_figures as d7  # noqa: E402
from analyze_cv_family import FROZEN_MANIFESTS, RESULTS, gate_name, out_name  # noqa: E402
from analyze_e7_chart import AREA_FILE, LEVELS, MAIN_FILE, _bool, grid_table  # noqa: E402
from analyze_e7_moran import SUMMARY_FILE  # noqa: E402
from training.dot7_rules import (_seed_signs, file_sha256, guard_frozen, main_family, model_name,  # noqa: E402
                                 provenance_path, write_provenance)
from training.model_interaction import NONE as I_NONE, SIG as I_SIG  # noqa: E402

MM = 1 / 25.4
OUT_DIR = os.path.join(RESULTS, "hinh", "paper")
ORDER = ("salinity", "ndwi", "dsr_mcd18", "rain_chirps", "t2m_era5", "rh_era5")  # nguon <= 1 km truoc, nguon tho sau
VAR = {"salinity": "Soil salinity (Landsat)", "ndwi": "NDWI (Landsat)", "rain_chirps": "Rainfall (CHIRPS)",
       "dsr_mcd18": "Solar radiation (MCD18A1)", "t2m_era5": "Air temperature (ERA5-Land, reference)",
       "rh_era5": "Humidity (ERA5-Land, reference)"}
COLOR = {"salinity": "#0072B2", "ndwi": "#56B4E9", "rain_chirps": "#009E73", "dsr_mcd18": "#E69F00",
         "t2m_era5": "#CC79A7", "rh_era5": "#000000"}
VMARK = {"salinity": "o", "ndwi": "D", "dsr_mcd18": "^", "rain_chirps": "s", "t2m_era5": "v", "rh_era5": "P"}
VERM, GREY, BAND_C, INK = "#D55E00", "#8C8C8C", "#E3E3E3", "#000000"
LEVEL = {5: "Coarse (~17 km)", 6: "Medium (~6.5 km)", 7: "Fine (~2.4 km)"}
LEVEL_SHORT = {5: "Coarse", 6: "Medium", 7: "Fine"}
LEVEL_FILL = {5: "#000000", 6: "#8C8C8C", 7: "#FFFFFF"}  # phan biet duoc khi in xam
LEVEL_LS = {5: "-", 6: "--", 7: ":"}
FRAMES = (("h3_res_", "H3", "h"), ("s2_level_", "S2", "D"), ("square_utm_", "Square", "s"), ("latlon_", "Lat–lon", "o"))
FRAME_LONG = {"H3": "H3", "S2": "S2", "Square": "Square (UTM)", "Lat–lon": "Lat–lon"}
MODELS = ("hist_gb", "rf", "mlp")
MODEL = {"hist_gb": "HistGB", "rf": "RF", "mlp": "MLP"}
FINAL = {"tuong_duong": "eq", "khong_tai_lap": "nrep", "khong_tai_lap_cap_ho": "nrep", "chua_phan_dinh": "und",
         "mo_ta": "desc"}
CLS = {"eq": ("Equivalent (TOST)", "o", INK, INK, 4.0),
       "nrep": ("Significant in CV, not replicated on held-out blocks", "s", "white", VERM, 4.0),
       "sig": ("Significant interaction", "s", VERM, VERM, 4.0),
       "und": ("Undetermined", "o", "white", INK, 4.0),
       "desc": ("Not tested (descriptive)", "o", GREY, GREY, 2.6)}
GATE = {"pass": ("Gate passed", "#9ECAE1", None), "fail": ("Gate failed", "white", "////"),
        "ne": ("Not evaluated", "#F0F0F0", None)}
FIG1_GRIDS = {5: ("h3_res_5", "s2_level_9", "square_utm_17087m", "latlon_0.1552deg"),
              6: ("h3_res_6", "s2_level_11", "square_utm_6458m", "latlon_0.0586deg"),
              7: ("h3_res_7", "s2_level_12", "square_utm_2441m", "latlon_0.0222deg")}
FIG1_CENTER = (105.78, 10.03)  # giua dong bang (Can Tho)
FIG1_HALF_M = 15_000
XCLIP = 4.0
N_NULL, NULL_SEED = 200_000, 42
DMIN = r"Δ$_\mathregular{min}$"
FIGS = {"fig1": "fig1_study_design", "fig2": "fig2_skill_gate", "fig3": "fig3_same_level_pairs",
        "fig4": "fig4_blocks_100km", "fig5": "fig5_positive_control_oracle", "fig6": "fig6_main",
        "figS1": "figS1_model_frame", "figS2": "figS2_model_frame_mlp"}
SIZE_MM = {"fig1": (190, 120), "fig2": (140, 72), "fig3": (190, 170), "fig4": (190, 90), "fig5": (190, 80),
           "fig6": (140, 100), "figS1": (190, 130), "figS2": (190, 120)}
NULL_METHOD = (f"For each variable x level, Delta_i ~ N(0, se_i) independently for the {{3 or 6}} same-level HistGB "
               f"pairs (se from the block sign-flip analysis), {N_NULL} draws, seed {NULL_SEED}; band = 2.5-97.5 % "
               f"quantiles of max_i |Delta_i| / Delta_min. Ignores correlation between pairs sharing a grid.")
CAPTIONS = {
    "fig1": "Study design. (a) Mekong Delta with the 19 spatial cross-validation units built from 50 km blocks "
            "(outlined) and the held-out blocks (hatched); dots are evaluation points; the dashed square marks the "
            "window in (b). (b) The four grid frames at three resolution levels in the same 30 km window; numbers "
            "are mean cell edges; † grids not used in same-level pairs (area ratio > 1.2). (c) Analysis workflow.",
    "fig2": "Skill gate (model vs seasonal-mean baseline, spatial block cross-validation). Cell = gate outcome per "
            "variable, model and level; number = median over the grids of the level of ΔMAE vs. baseline as % of "
            "model MAE (ΔMAE = MAE(model) − MAE(seasonal mean); denominator = level-mean CV MAE of the model; "
            "negative = model better). Not evaluated = level not pre-registered for testing (values descriptive). "
            "† Soil salinity HistGB had no pre-registered gate; the same rule (Holm p < 0.05, ΔMAE < 0, same sign in "
            "all 3 splits) was applied to the frozen model-vs-baseline comparison.",
    "fig3": "Same-level grid pairs (area ratio ≤ 1.2), HistGB, spatial block cross-validation (50 km blocks). "
            "ΔMAE = MAE(A) − MAE(B) for pair 'A vs B', in units of Δmin = 5 % of the level-mean MAE (smallest effect "
            "size of interest). Thick bar = 90 % CI (TOST), thin bar = 95 % CI; grey band = equivalence margin "
            "(±Δmin); arrows = CI beyond ±4. Classes are final: block sign-flip permutation test (2^19), "
            "Holm-adjusted p, decision rule D, then held-out blocks and season (2020).",
    "fig4": "Sensitivity to block size (HistGB). (a) Change in MAE when cross-validation blocks grow from 50 to 100 km, "
            "per grid (shape = frame, fill = level); vertical bars = median over the 13 grids. (b) Fine-level pairs "
            "with 100 km blocks: ΔMAE / Δmin (mean over the two 100 km partitions; thin line = range of the two).",
    "fig5": "Positive control and oracle. (a) Detection rate when an effect of k × Δmin is injected into a "
            "same-level pair (soil salinity, HistGB); small dots = pairs, lines = mean over the pairs of a level; "
            "the τ = μ indicator run is omitted. (b) |ΔMAE| of the oracle (cell mean of true labels, 50 km blocks) "
            "for fine-level pairs, in units of the HistGB Δmin of the variable.",
    "fig6": "Frame effect vs. source resolution. (a) Max |ΔMAE| / Δmin across the same-level pairs (HistGB) against "
            "source pixel size / mean cell edge; filled = tested level, open = descriptive; points of variables "
            "sharing a source are dodged horizontally. Grey bars = expected under no frame effect (95 % range): "
            + NULL_METHOD + " (b) Moran's I (10 km distance band) of the point labels: median and min–max over seasons.",
    "figS1": "Model × frame (RF vs HistGB). (a) ΔMAE / Δmin of each pair under both models (whiskers = 90 % CI; "
             "pairs passing the skill gate under both models); sign agreement n/N per variable, criterion ≥ 80 %. "
             "(b) Interaction I = ΔMAE_RF − ΔMAE_HistGB in units of the HistGB Δmin; thick bar = 90 % CI, thin bar "
             "= 95 % CI; grey band = ±Δmin.",
}
CAPTIONS["figS2"] = CAPTIONS["figS1"].replace("RF vs HistGB", "MLP vs HistGB").replace("ΔMAE_RF", "ΔMAE_MLP")
RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titlesize": 7, "pdf.fonttype": 42,
      "ps.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
      "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
      "legend.frameon": False, "savefig.dpi": 600, "hatch.linewidth": 0.5}


loi, read_cols = d7.loi, d7.read_cols


def frame(g):
    return next(lab for pre, lab, _ in FRAMES if g.startswith(pre))


def fmark(g):
    return next(mk for pre, _, mk in FRAMES if g.startswith(pre))


def pair_label(a, b):
    return f"{frame(a)} vs {frame(b)}"


# ---------------------------------------------------------
# Doc / ghep (chi doc ket qua da mo)
# ---------------------------------------------------------
def final_labels(res, t, st):
    """Nhan CUOI theo cap F1 HistGB -> lop eq/nrep/und/desc; nhan la -> LOI (khong bia ky hieu)."""
    if t == "salinity":  # do man: khong cap nao Holm dat -> nhan final = nhan CV (file doi chieu final)
        path, fam = os.path.join(res, "dot5_final_so_cap_voi_cv.csv"), None
    else:
        path, fam = os.path.join(res, f"dot7_{t}_cv1_final_cap.csv"), "F1_chinh"
    d = read_cols(path, ["grid_a", "grid_b", "delta_hat", "label"], st, f"{VAR[t]} final labels")
    if d is None:
        return None
    if fam:
        d7._check_target(d, t, path)
        d = d[d["family"] == fam]
    bad = set(d["label"]) - set(FINAL)
    if bad:
        loi(f"{path}: nhan cuoi {sorted(bad)} chua co ky hieu tren hinh")
    return d.assign(final=d["label"].map(FINAL))[["grid_a", "grid_b", "delta_hat", "final"]]


def f1_pairs(res, levels, st) -> pd.DataFrame:
    """Cap F1 HistGB (kiem o d7.f1_table) + CI 90 %, se, Delta_min, nhan cuoi; bien thieu nhan cuoi -> bo."""
    f1 = d7.f1_table(res, levels, st)
    parts = []
    for t in ORDER:
        if t not in set(f1["target"]):
            continue
        d = pd.read_csv(os.path.join(res, out_name(t, "cv1")))
        miss = [c for c in ("ci_tost_low", "ci_tost_high", "se") if c not in d.columns]
        if miss:
            st["notes"].append(f"{VAR[t]}: F1 thiếu cột {', '.join(miss)} — bỏ")
            continue
        lab = final_labels(res, t, st)
        if lab is None:
            continue
        d = d[d["family"] == "F1_chinh"].merge(lab, on=["grid_a", "grid_b"], how="left", suffixes=("", "_fin"),
                                               validate="one_to_one")
        if d["final"].isna().any() or not np.allclose(d["delta_hat"], d["delta_hat_fin"]):
            loi(f"{t}: nhan cuoi khong khop 12 cap F1 (thieu cap / delta_hat lech)")
        parts.append(d.assign(target=t)[["target", "grid_a", "grid_b", "ci_tost_low", "ci_tost_high", "se",
                                         "delta_min_thr", "delta_hat", "final"]])
    cols = list(f1.columns) + ["thr", "se", "delta_hat", "r90_lo", "r90_hi", "final"]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = f1.merge(pd.concat(parts), on=["target", "grid_a", "grid_b"], validate="one_to_one")
    out = out.rename(columns={"delta_min_thr": "thr"})
    out["r90_lo"], out["r90_hi"] = out["ci_tost_low"] / out["thr"], out["ci_tost_high"] / out["thr"]
    return out[cols]


def null_band(se, thr, n=N_NULL, seed=NULL_SEED):
    """Phan vi 2,5-97,5 % cua max_i |Delta_i| / thr, Delta_i ~ N(0, se_i) doc lap (khong co anh huong khung)."""
    se = np.asarray(se, float)
    if not (se.size and np.isfinite(se).all() and (se > 0).all() and np.isfinite(thr) and thr > 0):
        loi(f"dai null: se {se} / Delta_min {thr} khong hop le")
    m = np.abs(np.random.default_rng(seed).normal(0.0, se, size=(n, se.size))).max(axis=1) / thr
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def gate_status(r) -> str:
    if r.hop_le:
        return "pass"
    if str(r.ly_do).startswith("khong_kiem_dinh"):
        return "ne"
    if r.ly_do == d7.FAIL_REASON:
        return "fail"
    loi(f"cong: ly_do la '{r.ly_do}'")


def salinity_histgb_gate(res, levels, st) -> dict:
    """Do man HistGB khong co file cong: so sanh baseline rong da dong bang + cung quy tac hoc_duoc (dot7_rules)."""
    path = os.path.join(res, "dot5_cv1_kiem_dinh_vs_baseline_rong.csv")
    d = read_cols(path, ["grid_a", "grid_b", "model_a", "model_b", "delta_hat", "p_holm", "seeds_same_dir",
                         "seed_deltas"], st, "Soil salinity (HistGB) vs baseline")
    if d is None:
        return {}
    if (sorted(d["grid_a"]) != sorted(levels.index) or (d["grid_a"] != d["grid_b"]).any()
            or set(d["model_a"]) != {"hist_gb"} or set(d["model_b"]) != {"season_mean"}):
        loi(f"{path}: khong phai 13 luoi hist_gb vs season_mean")
    ref = d7.level_ref(res, "salinity", "hist_gb", st)
    if ref is None:
        return {}
    m = d["grid_a"].map(levels)
    pct = 100 * d["delta_hat"] / m.map(ref)
    seeds_neg = d["seed_deltas"].map(lambda s: len(_seed_signs(s)) == 3 and all(x < 0 for x in _seed_signs(s)))
    ok = (d["p_holm"] < d7.ALPHA) & (d["delta_hat"] < 0) & _bool(d["seeds_same_dir"]) & seeds_neg
    st["notes"].append("Soil salinity HistGB: cổng áp lại quy tắc hoc_duoc lên dot5_cv1_kiem_dinh_vs_baseline_rong.csv "
                       "(Holm 13 lưới = phạm vi muc_kiem_dinh) — đánh dấu †")
    return {lv: ("pass" if ok[m == lv].all() else "fail", float(pct[m == lv].median())) for lv in LEVELS}


def gate_cells(res, levels, st) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        hoc = d7.learn_table(res, model, levels, st)
        d7.gate_fails(res, model, hoc, st)  # hop_le phai khop hoc_duoc
        g = read_cols(os.path.join(res, gate_name(model)), ["bien", "muc", "hop_le", "ly_do"], st,
                      f"gate {MODEL[model]}")
        status = {} if g is None else {(r.bien, int(r.muc)): gate_status(r)
                                       for r in g.assign(hop_le=_bool(g["hop_le"])).itertuples()}
        med = hoc.groupby(["target", "muc"])["y_pct"].median()
        for t in ORDER:
            for m in LEVELS:
                rows.append({"target": t, "model": model, "muc": m, "status": status.get((t, m)),
                             "pct": med.get((t, m), np.nan), "dagger": False})
    out = pd.DataFrame(rows)
    for m, (s, v) in salinity_histgb_gate(res, levels, st).items():
        out.loc[(out["target"] == "salinity") & (out["model"] == "hist_gb") & (out["muc"] == m),
                ["status", "pct", "dagger"]] = [s, v, True]
    miss = out["status"].isna()
    if miss.any():
        st["notes"].append(f"gate: {int(miss.sum())} ô không có trạng thái cổng → Not evaluated")
        out.loc[miss, "status"] = "ne"
    return out


def k100_signed(res, levels, st) -> pd.DataFrame:
    """Cap min o khoi 100 km: Delta co dau (TB 2 cach chia) va hai cach chia, chia Delta_min."""
    cols = ["grid_a", "grid_b", "delta_k100_tb", "delta_k100_s42", "delta_k100_s43", "delta_min"]
    parts = []
    for t in ORDER:
        pc = d7.k100_paths(res, t)[1]
        c = read_cols(pc, cols, st, f"{VAR[t]} fine pairs 100 km")
        if c is None:
            continue
        d7._check_target(c, t, pc)
        if not ((c["grid_a"].map(levels) == 7) & (c["grid_b"].map(levels) == 7)).all():
            loi(f"{pc}: co cap khong phai muc min")
        parts.append(c.assign(target=t, r=c["delta_k100_tb"] / c["delta_min"],
                              r_a=c["delta_k100_s42"] / c["delta_min"], r_b=c["delta_k100_s43"] / c["delta_min"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["target", "grid_a", "grid_b", "r"])


def e7_tables(res, f1, st):
    """Bang E7 (max|Delta|/Delta_min, nguon/canh o, kiem_dinh) + dai null theo (bien, muc); kiem lai max tu F1."""
    tab = read_cols(os.path.join(res, MAIN_FILE), ["variable", "muc", "anh_huong_khung", "kiem_dinh", "n_cap",
                                                   "nguon_tren_canh_median"], st, "E7 main table")
    mor = read_cols(os.path.join(res, SUMMARY_FILE), ["variable", "moran_median", "moran_min", "moran_max"], st,
                    "E7 Moran")
    if tab is None:
        return None, mor
    rows = []
    for r in tab.itertuples():
        g = f1[(f1["target"] == r.variable) & (f1["muc"] == r.muc)]
        if g.empty:
            st["notes"].append(f"{r.variable} mức {r.muc}: không có cặp F1 — bỏ dải null")
            rows.append((np.nan, np.nan))
            continue
        if (g["thr"].nunique() != 1 or len(g) != r.n_cap
                or not np.isclose(g["r"].abs().max(), r.anh_huong_khung)):
            loi(f"E7 {r.variable} muc {r.muc}: anh_huong_khung/n_cap khac F1 (max|delta_hat|/delta_min_thr)")
        rows.append(null_band(g["se"], g["thr"].iloc[0]))
    tab[["null_lo", "null_hi"]] = pd.DataFrame(rows, index=tab.index)
    return tab.assign(kiem_dinh=_bool(tab["kiem_dinh"])), mor


def sens_pairs(res, model, f1, st):
    """Cap qua cong ca hai mo hinh: Delta/Delta_min HistGB va mo hinh + CI 90 %; dong dau n/N tu file tong ket."""
    parts = []
    for t in ORDER:
        p = os.path.join(res, model_name(f"dot7_{t}_do_nhay_mo_hinh.csv", model))
        dn = read_cols(p, ["grid_a", "grid_b", "delta_hat_hist_gb", "delta_hat_mo_hinh", "qua_cong_ca_hai",
                           "cung_dau"], st, f"{VAR[t]} {MODEL[model]} sensitivity")
        pm = os.path.join(res, model_name(out_name(t, "cv1"), model))
        mf = read_cols(pm, ["family", "grid_a", "grid_b", "delta_hat", "ci_tost_low", "ci_tost_high",
                            "delta_min_thr"], st, f"{VAR[t]} {MODEL[model]} pairs")
        if dn is None or mf is None or dn.empty:
            continue
        dn = dn.loc[_bool(dn["qua_cong_ca_hai"]), ["grid_a", "grid_b", "delta_hat_hist_gb", "delta_hat_mo_hinh",
                                                    "cung_dau"]]
        mf = mf.loc[mf["family"] == main_family(model), ["grid_a", "grid_b", "delta_hat", "ci_tost_low",
                                                          "ci_tost_high", "delta_min_thr"]]
        mf.columns = ["grid_a", "grid_b", "d_m", "lo_m", "hi_m", "thr_m"]
        x = (dn.merge(f1[f1["target"] == t], on=["grid_a", "grid_b"], validate="one_to_one")
             .merge(mf, on=["grid_a", "grid_b"], validate="one_to_one"))
        if len(x) != len(dn) or not (np.allclose(x["delta_hat_hist_gb"], x["delta_hat"])
                                     and np.allclose(x["delta_hat_mo_hinh"], x["d_m"])):
            loi(f"{t}/{model}: cap do nhay khong khop file F1 HistGB / mo hinh")
        parts.append(x.assign(y=x["d_m"] / x["thr_m"], y_lo=x["lo_m"] / x["thr_m"], y_hi=x["hi_m"] / x["thr_m"],
                              same=_bool(x["cung_dau"])))
    pairs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    tk = read_cols(os.path.join(res, model_name("dot7_do_nhay_mo_hinh_tong_ket.csv", model)),
                   ["target", "n_cung_dau", "n_cap"], st, f"{MODEL[model]} sign agreement")
    agree = {}
    if tk is not None and not pairs.empty:
        for t, g in pairs.groupby("target"):
            r = tk[tk["target"] == t]
            if len(r) != 1 or int(r["n_cap"].iloc[0]) != len(g) or int(r["n_cung_dau"].iloc[0]) != int(g["same"].sum()):
                loi(f"{t}/{model}: n_cung_dau/n_cap khac so cap ve")
            agree[t] = (int(g["same"].sum()), len(g))
    return pairs, agree


def interaction_pairs(res, model, st) -> pd.DataFrame:
    parts = []
    for t in ORDER:
        p = os.path.join(res, model_name(f"dot7_{t}_tuong_tac_mo_hinh.csv", model))
        d = read_cols(p, ["grid_a", "grid_b", "muc", "I_hat", "ci_low", "ci_high", "ci_tost_low", "ci_tost_high",
                          "delta_min_thr_histgb", "tost_tuong_tac", "label", "chua_ro"], st,
                      f"{VAR[t]} {MODEL[model]} interaction")
        if d is None or d.empty:
            continue
        d7._check_target(d, t, p)
        if set(d["label"]) - {I_NONE, I_SIG}:
            loi(f"{p}: nhan tuong tac la {sorted(set(d['label']) - {I_NONE, I_SIG})}")
        thr = d["delta_min_thr_histgb"]
        cls = np.where(d["label"] == I_SIG, "sig", np.where(_bool(d["tost_tuong_tac"]) & ~_bool(d["chua_ro"]),
                                                            "eq", "und"))
        parts.append(d.assign(target=t, cls=cls, r=d["I_hat"] / thr, r_lo=d["ci_low"] / thr, r_hi=d["ci_high"] / thr,
                              r90_lo=d["ci_tost_low"] / thr, r90_hi=d["ci_tost_high"] / thr))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------
# Ve: tien ich
# ---------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC)
    return plt


def new_fig(key, **kw):
    w, h = SIZE_MM[key]
    return _plt().figure(figsize=(w * MM, h * MM), **kw)


def letter(ax, s, x=-0.01, y=1.01):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def mk(label, marker="o", face=INK, edge=INK, ms=4.0, ls="", lw=0.8, color=None):
    from matplotlib.lines import Line2D

    return Line2D([], [], ls=ls, lw=lw, color=color or edge, marker=marker, ms=ms, mfc=face, mec=edge, mew=0.7,
                  label=label)


def patch(label, face, hatch=None, edge="#555555"):
    from matplotlib.patches import Patch

    return Patch(facecolor=face, edgecolor=edge, hatch=hatch, lw=0.5, label=label)


def seg(ax, lo, hi, at, lim, vertical=False, **kw):
    """Doan [lo, hi] cat trong [-lim, lim]; dau bi cat -> mui ten."""
    a, b = max(lo, -lim), min(hi, lim)
    ax.plot(*(([at, at], [a, b]) if vertical else ([a, b], [at, at])), solid_capstyle="butt", **kw)
    for end, cut, m in ((a, lo < -lim, "v" if vertical else "<"), (b, hi > lim, "^" if vertical else ">")):
        if cut:
            ax.plot(*((at, end) if vertical else (end, at)), marker=m, ms=3.5, color=kw.get("color", INK), mew=0,
                    clip_on=False, zorder=5)


def point(ax, x, y, cls, lim=XCLIP, vertical=False):
    lab, m, face, edge, ms = CLS[cls]
    v = float(np.clip(y if vertical else x, -lim, lim))
    ax.plot(*((x, v) if vertical else (v, y)), ls="", marker=m, ms=ms, mfc=face, mec=edge, mew=0.8, zorder=4,
            clip_on=False)


def forest_rows(groups, gap=0.6):
    """groups = [(tieu de, [(khoa, nhan)])] -> ({khoa: y}, [(y, nhan, la_tieu_de)])."""
    ys, ticks, y = {}, [], 0.0
    for name, keys in groups:
        if ticks:
            y += gap
        ticks.append((y, name, True))
        y += 1
        for k, lab in keys:
            ys[k] = y
            ticks.append((y, lab, False))
            y += 1
    return ys, ticks


def apply_ticks(ax, ticks):
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks])  # sharey: truc trong tu an nhan
    for lab, t in zip(ax.get_yticklabels(), ticks):
        lab.set_fontweight("bold" if t[2] else "normal")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(max(t[0] for t in ticks) + 0.7, min(t[0] for t in ticks) - 0.7)
    ax.spines["left"].set_visible(False)


def margin(ax, lim=XCLIP, vertical=False, band=1.0):
    (ax.axhspan if vertical else ax.axvspan)(-band, band, color=BAND_C, lw=0, zorder=0)
    (ax.axhline if vertical else ax.axvline)(0, color=INK, lw=0.5, zorder=1)
    (ax.set_ylim if vertical else ax.set_xlim)(-lim * 1.06, lim * 1.06)


def empty(ax, text="Not available"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=GREY)


def var_handles(ts, line=False):
    return [mk(VAR[t], VMARK[t], COLOR[t], COLOR[t], ls="-" if line else "", color=COLOR[t]) for t in ORDER if t in ts]


def ci_handles():
    return [mk("90 % CI (TOST)", "", ls="-", lw=1.8), mk("95 % CI", "", ls="-", lw=0.6),
            patch(f"Equivalence margin (±{DMIN})", BAND_C, edge="none")]


def save(fig, path_noext) -> list:
    paths = []
    for ext in ("pdf", "png"):
        p = f"{path_noext}.{ext}"
        fig.savefig(p, dpi=600, facecolor="white")
        paths.append(p)
    _plt().close(fig)
    return paths


# ---------------------------------------------------------
# Hinh
# ---------------------------------------------------------
def fig1(res, levels, st, a):
    import geopandas as gpd
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    from matplotlib.ticker import FuncFormatter
    from eval_design import exact_block_frame

    fig = new_fig("fig1", layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[0.67, 0.33])
    sa, sb = top.subfigures(1, 2, width_ratios=[0.46, 0.54])
    paths = {k: os.path.join(a.eval_dir, f) for k, f in (("blocks", "holdout_blocks.geojson"),
                                                          ("folds", "cv_folds.csv"), ("pts", "eval_points.geojson"))}
    ax = sa.subplots()
    missing = [p for p in [a.boundary, *paths.values()] if not os.path.isfile(p)]
    if missing:
        st["notes"].append(f"Fig 1a: thiếu {missing} — bỏ")
        empty(ax)
    else:
        st["used"] += [a.boundary, *paths.values()]
        bnd = gpd.read_file(a.boundary).to_crs(4326)
        blocks = gpd.read_file(paths["blocks"])
        cv = pd.read_csv(paths["folds"], dtype={"block_id": str, "unit_id": str})
        ex = exact_block_frame(blocks).merge(cv[["block_id", "unit_id", "cv_fold"]], on="block_id",
                                             validate="one_to_one")
        units = ex[ex["cv_fold"] >= 0].dissolve("unit_id").to_crs(4326)
        hold = ex[_bool(ex["is_holdout"])].to_crs(4326)
        if len(units) != 19 or (hold["cv_fold"] != -1).any():
            loi(f"Fig 1a: {len(units)} don vi CV (ky vong 19) hoac khoi giu rieng co fold")
        pts = gpd.read_file(paths["pts"]).to_crs(4326)
        bnd.plot(ax=ax, color="#E6E6E6", lw=0)
        hold.plot(ax=ax, facecolor="none", edgecolor="#555555", hatch="//////", lw=0.4)
        ax.scatter(pts.geometry.x, pts.geometry.y, s=0.08, color=INK, alpha=0.35, lw=0, rasterized=True)
        units.boundary.plot(ax=ax, color=INK, lw=0.6)
        win = gpd.GeoSeries(gpd.points_from_xy([FIG1_CENTER[0]], [FIG1_CENTER[1]]), crs=4326).to_crs(32648)
        win = win.buffer(FIG1_HALF_M, cap_style="square").to_crs(4326)
        win.boundary.plot(ax=ax, color=INK, lw=0.8, ls="--")
        ax.set_aspect(1 / np.cos(np.deg2rad(FIG1_CENTER[1])))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}°E"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}°N"))
        ax.legend(handles=[patch("CV unit (50 km blocks)", "none", edge=INK),
                           patch("Held-out block", "none", "//////"), mk("Evaluation point", "o", INK, INK, 1.5)],
                  loc="upper left", fontsize=7, handlelength=1.4, frameon=True, facecolor="white", edgecolor="none",
                  framealpha=1)
        ax.set_xlabel("")
        ax.set_ylabel("")
    letter(ax, "a")

    axs = sb.subplots(3, 4)
    kd = os.path.join(res, out_name("salinity", "cv1"))
    st["used"].append(kd)
    f = pd.read_csv(kd).query("family == 'F1_chinh'")
    in_pairs = set(f["grid_a"]) | set(f["grid_b"])
    edge_km = grid_table(os.path.join(res, AREA_FILE)).set_index("grid")["canh_o_km"]
    c = gpd.GeoSeries(gpd.points_from_xy([FIG1_CENTER[0]], [FIG1_CENTER[1]]), crs=4326).to_crs(32648).iloc[0]
    cx, cy = c.x, c.y
    bb = (FIG1_CENTER[0] - 0.25, FIG1_CENTER[1] - 0.25, FIG1_CENTER[0] + 0.25, FIG1_CENTER[1] + 0.25)
    for i, m in enumerate(LEVELS):
        for j, g in enumerate(FIG1_GRIDS[m]):
            x = axs[i, j]
            if levels[g] != m:
                loi(f"Fig 1b: {g} khong o muc {m}")
            p = os.path.join(a.grids_dir, f"{g}.geojson")
            if os.path.isfile(p):
                st["used"].append(p)
                gpd.read_file(p, bbox=bb).to_crs(32648).boundary.plot(ax=x, color=INK, lw=0.35)
                x.set_xlabel("")
                x.set_ylabel("")
            else:
                st["notes"].append(f"Fig 1b: thiếu {p}")
                empty(x)
            x.set_xlim(cx - FIG1_HALF_M, cx + FIG1_HALF_M)
            x.set_ylim(cy - FIG1_HALF_M, cy + FIG1_HALF_M)
            x.set_aspect("equal")
            x.set_xticks([])
            x.set_yticks([])
            for s in x.spines.values():
                s.set_visible(True)
                s.set_lw(0.5)
            x.text(0.03, 0.04, f"{edge_km[g]:.1f} km" + ("" if g in in_pairs else " †"), transform=x.transAxes,
                   fontsize=7, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
            if i == 0:
                x.set_title(FRAME_LONG[frame(g)], fontsize=7)
            if j == 0:
                x.set_ylabel(LEVEL_SHORT[m], fontsize=7)
    letter(axs[0, 0], "b", x=-0.12)

    ax = bot.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 40)
    ax.axis("off")
    main = ["Six variables\n(Landsat, MODIS,\nCHIRPS,\nERA5-Land)", "Cell labels\n13 grids\n(4 frames ×\n3 levels)",
            "Features\n(area-weighted\ncell means)", "Models\nHistGB (primary)\nRF, MLP\n(sensitivity);\nbaselines",
            "Spatial block CV\n(50 km blocks,\n3 splits)", "Paired ΔMAE\n(12 pairs /\nvariable)",
            "Sign-flip (2$^{19}$)\n+ Holm + TOST", "Decision\n(held-out blocks\nand season 2020)"]
    w, gap, x0, yb, hb = 21.0, 2.4, 2.0, 17.0, 17.5
    xs = [x0 + k * (w + gap) for k in range(len(main))]
    for k, (x, txt) in enumerate(zip(xs, main)):
        ax.add_patch(FancyBboxPatch((x, yb), w, hb, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=INK,
                                    lw=0.6))
        ax.text(x + w / 2, yb + hb / 2, txt, ha="center", va="center", fontsize=7, linespacing=1.1)
        if k:
            ax.add_patch(FancyArrowPatch((x - gap, yb + hb / 2), (x, yb + hb / 2), arrowstyle="-|>",
                                         mutation_scale=6, lw=0.6, color=INK, shrinkA=0, shrinkB=0))
    pre = xs[4] - 1.2
    ax.add_patch(Rectangle((pre, yb - 1.2), xs[-1] + w + 1.2 - pre, hb + 2.4, fc="none", ec=GREY, lw=0.6, ls="--"))
    ax.text(xs[-1] + w + 1.2, yb + hb + 1.6, "pre-registered", ha="right", va="bottom", fontsize=7, style="italic",
            color="#555555")
    for k, txt in ((3, "Skill gate\n(model vs seasonal-\nmean baseline)"), (5, "Oracle (cell mean\nof true labels) &\n"
                   "geometric bound"), (6, "Positive control\n(injected effect)")):
        bw, bx = 22.4, xs[k] + w / 2 - 11.2
        ax.add_patch(FancyBboxPatch((bx, 1.0), bw, 11.5, boxstyle="round,pad=0,rounding_size=1.2", fc="#F0F0F0",
                                    ec=GREY, lw=0.6))
        ax.text(bx + bw / 2, 6.75, txt, ha="center", va="center", fontsize=7, linespacing=1.1)
        ax.plot([xs[k] + w / 2] * 2, [12.5, yb - (1.2 if k >= 4 else 0)], color=GREY, lw=0.6, ls=":")
    ax.text(0.5, 39.5, "(c)", fontsize=9, fontweight="bold", ha="left", va="top")
    return fig


def fig2(res, levels, st):
    from matplotlib.patches import Rectangle

    cells = gate_cells(res, levels, st)
    fig = new_fig("fig2")
    ax = fig.add_axes([0.335, 0.2, 0.655, 0.68])
    ax.axis("off")
    rows = {t: i + (0.35 if i >= 3 else 0) for i, t in enumerate(ORDER)}
    col = {(mo, m): k * 3.35 + LEVELS.index(m) for k, mo in enumerate(MODELS) for m in LEVELS}
    for r in cells.itertuples():
        x, y = col[(r.model, r.muc)], rows[r.target]
        _, face, hatch = GATE[r.status]
        ax.add_patch(Rectangle((x + 0.04, y + 0.05), 0.92, 0.9, fc=face, ec="#555555", hatch=hatch, lw=0.5))
        if np.isfinite(r.pct):
            txt = f"{r.pct:.0f}".replace("-", "−") + ("†" if r.dagger else "")
            ax.text(x + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=7,
                    color="#555555" if r.status == "ne" else INK, style="italic" if r.status == "ne" else "normal",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.4) if r.status == "fail" else None)
    for k, mo in enumerate(MODELS):
        ax.text(k * 3.35 + 1.5, -0.75, MODEL[mo], ha="center", va="bottom", fontsize=7, fontweight="bold")
        for m in LEVELS:
            ax.text(col[(mo, m)] + 0.5, -0.08, LEVEL_SHORT[m], ha="center", va="bottom", fontsize=7)
    for t, y in rows.items():
        ax.text(-0.15, y + 0.5, VAR[t], ha="right", va="center", fontsize=7)
    ax.set_xlim(0, max(col.values()) + 1)
    ax.set_ylim(max(rows.values()) + 1, -0.8)
    fig.legend(handles=[patch(lab, face, hatch) for lab, face, hatch in GATE.values()], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0), title="Number: ΔMAE vs. baseline (% of model MAE; negative = better)",
               title_fontsize=7)
    return fig


def fig3(res, levels, st):
    plt = _plt()
    f = f1_pairs(res, levels, st)
    pairs = f.drop_duplicates(["grid_a", "grid_b"]).sort_values("muc", kind="stable")
    groups = [(LEVEL[m], [((r.grid_a, r.grid_b), pair_label(r.grid_a, r.grid_b))
                          for r in pairs[pairs["muc"] == m].itertuples()]) for m in LEVELS]
    ys, ticks = forest_rows(groups)
    fig, axes = plt.subplots(2, 3, figsize=(190 * MM, 170 * MM), sharex=True, sharey=True, layout="constrained")
    for i, (ax, t) in enumerate(zip(axes.ravel(), ORDER)):
        margin(ax)
        ax.set_title(VAR[t], loc="left")
        letter(ax, "abcdef"[i])
        g = f[f["target"] == t]
        if g.empty:
            empty(ax)
        for r in g.itertuples():
            y = ys[(r.grid_a, r.grid_b)]
            c = CLS[r.final][3]
            seg(ax, r.r_lo, r.r_hi, y, XCLIP, color=c, lw=0.6, zorder=2)
            seg(ax, r.r90_lo, r.r90_hi, y, XCLIP, color=c, lw=1.8, zorder=3)
            point(ax, r.r, y, r.final)
        if ticks:
            apply_ticks(ax, ticks)
        if i >= 3:
            ax.set_xlabel(f"ΔMAE / {DMIN}")
    h = [mk(*CLS[k]) for k in ("eq", "nrep", "und", "desc")] + ci_handles()
    fig.legend(handles=h, loc="outside lower center", ncol=4)
    return fig


def fig4(res, levels, st):
    plt = _plt()
    mae, _ = d7.k100_tables(res, levels, st)
    cap = k100_signed(res, levels, st)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(190 * MM, 90 * MM), width_ratios=[1.45, 1], layout="constrained")
    foff = dict(zip([lab for _, lab, _ in FRAMES], np.linspace(-0.27, 0.27, len(FRAMES))))
    for i, t in enumerate(ORDER):
        g = mae[mae["target"] == t]
        for r in g.itertuples():
            a1.plot(r.y_pct, i + foff[frame(r.grid)], ls="", marker=fmark(r.grid), ms=3.6, mfc=LEVEL_FILL[r.muc],
                    mec=INK, mew=0.5, zorder=3)
        if len(g):
            med = g["y_pct"].median()
            a1.plot([med, med], [i - 0.4, i + 0.4], color=INK, lw=1.3, zorder=4)
    if mae.empty:
        empty(a1)
    a1.axvline(0, color=INK, lw=0.5)
    a1.axvline(5, color=GREY, lw=0.8, ls="--")
    a1.text(5, -0.75, " 5 % reference", ha="left", va="bottom", fontsize=7, color="#555555")
    a1.set_yticks(range(len(ORDER)))
    a1.set_yticklabels([VAR[t] for t in ORDER])
    a1.set_ylim(len(ORDER) - 0.5, -0.9)
    a1.tick_params(axis="y", length=0)
    a1.set_xlabel("Change in MAE, 100 km vs 50 km blocks (%)")
    fig.legend(handles=[mk(FRAME_LONG[lab], m, "white", INK, 3.6) for _, lab, m in FRAMES]
               + [mk(LEVEL[m], "o", LEVEL_FILL[m], INK, 3.6) for m in LEVELS] + [mk("Median", "|", INK, INK, 6)],
               loc="outside lower left", ncol=3, handletextpad=0.3, columnspacing=1.0)
    letter(a1, "a")
    pairs = list(dict.fromkeys(zip(cap["grid_a"], cap["grid_b"])))
    voff = dict(zip(ORDER, np.linspace(-0.3, 0.3, len(ORDER))))
    for r in cap.itertuples():
        y = pairs.index((r.grid_a, r.grid_b)) + voff[r.target]
        a2.plot([min(r.r_a, r.r_b), max(r.r_a, r.r_b)], [y, y], color=COLOR[r.target], lw=0.6, zorder=2)
        a2.plot(r.r, y, ls="", marker=VMARK[r.target], ms=3.6, mfc=COLOR[r.target], mec=COLOR[r.target], zorder=3)
    lim = max(1.2, float(np.nanmax(np.abs(cap[["r_a", "r_b"]].to_numpy(float)))) * 1.1) if len(cap) else 1.2
    margin(a2, lim)
    if cap.empty:
        empty(a2)
    else:
        a2.set_yticks(range(len(pairs)))
        a2.set_yticklabels([pair_label(*p) for p in pairs])
        a2.set_ylim(len(pairs) - 0.5, -0.5)
        a2.tick_params(axis="y", length=0)
    a2.set_xlabel(f"ΔMAE / {DMIN}, fine-level pairs, 100 km blocks")
    fig.legend(handles=var_handles(set(cap["target"])), loc="outside lower right", ncol=2, handletextpad=0.3)
    letter(a2, "b")
    return fig


def fig5(res, levels, st):
    plt = _plt()
    p, _ = d7.pc1_tables(res, st)
    f1 = f1_pairs(res, levels, st)
    orc = d7.oracle_table(res, levels, st)
    fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 80 * MM), width_ratios=[1, 1, 1.35], layout="constrained")
    kinds = (("deu", "Uniform injection"), ("lognormal", "Log-normal injection"))
    if p is not None and "tau_mu" in set(p["kind"]):
        st["notes"].append("PC-1: bỏ kiểu τ = μ (chỉ báo)")
    for ax, (kind, title) in zip(axes[:2], kinds):
        ax.set_title("   " + title, loc="left")
        ax.axhline(80, color=GREY, lw=0.8, ls="--")
        ax.text(0.72, 81, "80 % target power", fontsize=7, color="#555555", va="bottom")
        g = p[p["kind"] == kind] if p is not None else pd.DataFrame()
        if g.empty:
            empty(ax)
        for m in LEVELS:
            x = g[g["level"] == m].sort_values(["k", "pair_index"])
            if x.empty:
                continue
            for k, xk in x.groupby("k"):
                jit = np.linspace(-0.07, 0.07, len(xk)) if len(xk) > 1 else np.zeros(1)
                ax.plot(k + jit, 100 * xk["ty_le_phat_hien"], ls="", marker="o", ms=2.4, mfc=LEVEL_FILL[m], mec=INK,
                        mew=0.4)
            mean = x.groupby("k")["ty_le_phat_hien"].mean() * 100
            ax.plot(mean.index, mean.values, ls=LEVEL_LS[m], color=INK, lw=0.9, marker="o", ms=4,
                    mfc=LEVEL_FILL[m], mec=INK, mew=0.6, label=LEVEL[m])
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"1 × {DMIN}", f"2 × {DMIN}"])
        ax.set_xlim(0.65, 2.35)
        ax.set_ylim(0, 103)
        ax.set_xlabel(f"Injected effect (k × {DMIN})")
    axes[0].set_ylabel("Detection rate (%)")
    axes[1].tick_params(labelleft=False)
    fig.legend(handles=axes[0].get_legend_handles_labels()[0], loc="outside lower left", ncol=3, handlelength=2.2)
    letter(axes[0], "a")
    a2 = axes[2]
    thr7 = f1[f1["muc"] == 7].groupby("target")["thr"].agg(["first", "nunique"])
    if (thr7["nunique"] > 1).any():
        loi("Delta_min muc min khong duy nhat theo bien")
    if orc is None or orc.empty:
        empty(a2)
    else:
        orc = orc[orc["target"].isin(thr7.index)]
        orc = orc.assign(y=orc["delta_hat"].abs() / orc["target"].map(thr7["first"]))
        pairs = list(dict.fromkeys(zip(orc["grid_a"], orc["grid_b"])))
        voff = dict(zip(ORDER, np.linspace(-0.3, 0.3, len(ORDER))))
        for r in orc.itertuples():
            a2.plot(pairs.index((r.grid_a, r.grid_b)) + voff[r.target], r.y, ls="", marker=VMARK[r.target], ms=3.6,
                    mfc=COLOR[r.target], mec=COLOR[r.target])
        a2.axhline(1, color=INK, lw=0.7, ls="--")
        a2.text(-0.45, 1.02, DMIN, fontsize=7, va="bottom")
        a2.set_xticks(range(len(pairs)))
        a2.set_xticklabels([pair_label(*q) for q in pairs], rotation=35, ha="right", rotation_mode="anchor")
        a2.set_ylim(0, max(1.15, orc["y"].max() * 1.1))
        a2.legend(handles=var_handles(set(orc["target"])), loc="center right", fontsize=7, handletextpad=0.3)
        na = [VAR[t] for t in ORDER if t not in set(orc["target"])]
        if na:
            a2.text(0.98, 0.2, "Not available: " + ", ".join(na), transform=a2.transAxes, ha="right", fontsize=7,
                    color="#555555")
    a2.set_ylabel(f"|ΔMAE$_\\mathregular{{oracle}}$| / {DMIN}")
    letter(a2, "b")
    return fig


def fig6(res, levels, st, info):
    plt = _plt()
    from matplotlib.ticker import FuncFormatter

    f1 = f1_pairs(res, levels, st)
    tab, mor = e7_tables(res, f1, st)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(140 * MM, 100 * MM), width_ratios=[3, 1], layout="constrained")
    dodge = {"salinity": 0.93, "ndwi": 1.07, "t2m_era5": 0.93, "rh_era5": 1.07}
    if tab is None:
        empty(a1)
    else:
        info["dai_null"] = {"phuong_phap": NULL_METHOD,
                            "gia_tri": tab[["variable", "muc", "anh_huong_khung", "null_lo", "null_hi", "n_cap"]]
                            .round(4).to_dict("records")}
        for t in ORDER:
            g = tab[tab["variable"] == t].sort_values("nguon_tren_canh_median")
            if g.empty:
                continue
            x = g["nguon_tren_canh_median"] * dodge.get(t, 1.0)
            a1.vlines(x, g["null_lo"], g["null_hi"], color="#D0D0D0", lw=5, zorder=1)
            a1.plot(x, g["anh_huong_khung"], color=COLOR[t], lw=0.9, zorder=2)
            for xx, r in zip(x, g.itertuples()):
                face = COLOR[t] if r.kiem_dinh else "white"
                a1.plot(xx, r.anh_huong_khung, ls="", marker=VMARK[t], ms=4.5, mfc=face, mec=COLOR[t], mew=0.9,
                        zorder=3)
        a1.set_xscale("log")
        a1.set_yscale("log")
        fmt = FuncFormatter(lambda v, _: f"{v:g}")
        a1.xaxis.set_major_formatter(fmt)
        a1.yaxis.set_major_formatter(fmt)
        a1.axhline(1, color=INK, lw=0.7, ls="--")
        a1.axvline(1, color=INK, lw=0.7, ls=":")
        a1.text(1.1, 0.995, "pixel = cell", transform=a1.get_xaxis_transform(), fontsize=7, va="top")
        a1.text(1.01, 1, DMIN, transform=a1.get_yaxis_transform(), fontsize=7, va="center")
        a1.set_xlim(0.001, 6)
        a1.set_xticks([0.001, 0.01, 0.1, 1])
        a1.set_yticks([0.1, 0.2, 0.5, 1, 2])
        fig.legend(handles=var_handles(set(tab["variable"]), line=True)
                   + [mk("Tested level", "o", INK, INK), mk("Descriptive level", "o", "white", INK),
                      mk("Expected under no frame effect (95 % range)", "", ls="-", lw=5, color="#D0D0D0")],
                   loc="outside lower center", ncol=2, handletextpad=0.4, columnspacing=1.0)
    a1.set_xlabel("Source pixel size / cell edge")
    a1.set_ylabel(f"Max |ΔMAE| / {DMIN} across same-level pairs")
    letter(a1, "a")
    if mor is None:
        empty(a2)
    else:
        m = mor.set_index("variable")
        ts = [t for t in ORDER if t in m.index]
        for i, t in enumerate(ts):
            a2.plot([m.loc[t, "moran_min"], m.loc[t, "moran_max"]], [i, i], color=INK, lw=1.0)
            a2.plot(m.loc[t, "moran_median"], i, ls="", marker="o", ms=4, mfc=INK, mec=INK, clip_on=False)
        a2.set_yticks(range(len(ts)))
        a2.set_yticklabels([VAR[t].replace(" (", "\n(") for t in ts])
        a2.set_ylim(len(ts) - 0.5, -0.5)
        a2.tick_params(axis="y", length=0)
        a2.set_xlim(0, 1)
        a2.set_xticks([0, 0.5, 1])
    a2.set_xlabel("Moran's I")
    letter(a2, "b")
    return fig


def fig_sens(res, levels, st, model, key):
    plt = _plt()
    from matplotlib.patches import Rectangle

    f1 = f1_pairs(res, levels, st)
    pairs, agree = sens_pairs(res, model, f1, st)
    inter = interaction_pairs(res, model, st)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(SIZE_MM[key][0] * MM, SIZE_MM[key][1] * MM), width_ratios=[1, 1.1],
                                 layout="constrained")
    lim = XCLIP
    for sx, sy in ((1, 1), (-1, -1)):
        a1.add_patch(Rectangle((0, 0), sx * lim * 1.06, sy * lim * 1.06, fc="#F4F4F4", lw=0, zorder=0))
    a1.add_patch(Rectangle((-1, -1), 2, 2, fc=BAND_C, lw=0, zorder=0))
    a1.axhline(0, color=INK, lw=0.4)
    a1.axvline(0, color=INK, lw=0.4)
    a1.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.7, ls="--", zorder=1)
    if pairs.empty:
        empty(a1)
    for r in pairs.itertuples():
        c = COLOR[r.target]
        seg(a1, r.r90_lo, r.r90_hi, r.y, lim, color=c, lw=0.6, alpha=0.8, zorder=2)
        seg(a1, r.y_lo, r.y_hi, r.r, lim, vertical=True, color=c, lw=0.6, alpha=0.8, zorder=2)
        a1.plot(np.clip(r.r, -lim, lim), np.clip(r.y, -lim, lim), ls="", marker=VMARK[r.target], ms=4, mfc=c, mec=c,
                zorder=3)
    a1.set_xlim(-lim * 1.06, lim * 1.06)
    a1.set_ylim(-lim * 1.06, lim * 1.06)
    a1.set_aspect("equal")
    a1.set_xlabel(f"ΔMAE$_\\mathregular{{HistGB}}$ / {DMIN}")
    a1.set_ylabel(f"ΔMAE$_\\mathregular{{{MODEL[model]}}}$ / {DMIN}")
    h = [mk(f"{VAR[t]}: {agree[t][0]}/{agree[t][1]}", VMARK[t], COLOR[t], COLOR[t]) for t in ORDER if t in agree]
    h += [mk("y = x", "", ls="--", color=GREY), patch(f"±{DMIN}", BAND_C, edge="none"),
          patch("Same sign", "#F4F4F4", edge="none")]
    a1.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
              title="Sign agreement n/N (criterion ≥ 80 %)", title_fontsize=7, handletextpad=0.4)
    letter(a1, "a")
    if inter.empty:
        empty(a2)
    else:
        groups = []
        for t in ORDER:
            g = inter[inter["target"] == t].sort_values("muc", kind="stable")
            if len(g):
                groups.append((VAR[t], [((t, r.grid_a, r.grid_b), f"{LEVEL_SHORT[r.muc]} · {pair_label(r.grid_a, r.grid_b)}")
                                        for r in g.itertuples()]))
        ys, ticks = forest_rows(groups, gap=0.3)
        lim2 = 3.0
        margin(a2, lim2)
        for r in inter.itertuples():
            y = ys[(r.target, r.grid_a, r.grid_b)]
            c = CLS[r.cls][3]
            seg(a2, r.r_lo, r.r_hi, y, lim2, color=c, lw=0.6, zorder=2)
            seg(a2, r.r90_lo, r.r90_hi, y, lim2, color=c, lw=1.8, zorder=3)
            point(a2, r.r, y, r.cls, lim=lim2)
        apply_ticks(a2, ticks)
        a2.legend(handles=[mk(*CLS[k]) for k in ("eq", "und", "sig")
                           if k != "sig" or "sig" in set(inter["cls"])] + ci_handles(),
                  loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    a2.set_xlabel(f"I / {DMIN}  (I = ΔMAE$_\\mathregular{{{MODEL[model]}}}$ − ΔMAE$_\\mathregular{{HistGB}}$)")
    letter(a2, "b")
    return fig


def build(key, res, levels, st, a, info):
    if key == "fig1":
        return fig1(res, levels, st, a)
    if key == "fig6":
        return fig6(res, levels, st, info)
    if key in ("figS1", "figS2"):
        return fig_sens(res, levels, st, "rf" if key == "figS1" else "mlp", key)
    return {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}[key](res, levels, st)


def outputs(out_dir, keys=tuple(FIGS)) -> list:
    return [os.path.join(out_dir, f"{FIGS[k]}.{ext}") for k in keys for ext in ("pdf", "png")]


def main(a):
    keys = a.only or list(FIGS)
    outs = outputs(a.out_dir, keys)
    guard_frozen(outs + [provenance_path(p) for p in outs], a.frozen_manifest)
    area = os.path.join(a.results_dir, AREA_FILE)
    levels = grid_table(area).set_index("grid")["muc"]
    os.makedirs(a.out_dir, exist_ok=True)
    for key in keys:
        st, info = {"notes": [], "used": [area]}, {}
        fig = build(key, a.results_dir, levels, st, a, info)
        paths = save(fig, os.path.join(a.out_dir, FIGS[key]))
        notes = list(dict.fromkeys(st["notes"]))
        src = {os.path.relpath(p, ROOT): file_sha256(p) for p in dict.fromkeys(st["used"])}
        for p in paths:
            write_provenance(p, vai_tro="hinh_bai_bao", kich_thuoc_mm=SIZE_MM[key], sha_dau_vao=src, phan_bo=notes,
                             chu_thich_en=CAPTIONS[key], **info)
        print(f"{FIGS[key]}: {len(src)} file vao, {len(notes)} ghi chu", flush=True)
        for n in notes:
            print(f"  - {n}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=RESULTS, help="noi co file ket qua da mo")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--boundary", default=None, help="mac dinh CANONICAL_BOUNDARY (ranh gioi v2)")
    ap.add_argument("--only", nargs="+", choices=list(FIGS))
    ap.add_argument("--frozen-manifest", nargs="+", default=FROZEN_MANIFESTS)
    args = ap.parse_args()
    if args.boundary is None:
        from preprocessing import CANONICAL_BOUNDARY

        args.boundary = CANONICAL_BOUNDARY
    main(args)

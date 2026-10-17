#!/usr/bin/env python
"""Hinh Dot 7 cho GVHD (mo ta): H2 cong hoc duoc, H3 F1 theo muc, H4 khoi 100 km, H5 PC-1 + sai so ly tuong.
Chi doc file ket qua da mo, khong tinh chi so moi; file/cot thieu -> bo phan do, ghi chu tren hinh + provenance.

Chay:  venv/Scripts/python.exe scripts/plot_dot7_figures.py [--results-dir KE_HOACH/ket-qua] [--out-dir KE_HOACH/ket-qua]
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

from analyze_cv_family import FROZEN_MANIFESTS, RESULTS, gate_name, out_name  # noqa: E402
from analyze_e7_chart import (AREA_FILE, COLORS, GRID_INK, INK, INK2, LEVELS, MARKERS, NAMES, SURFACE,  # noqa: E402
                              _bool, grid_table)
from analyze_e7_moran import VARIABLES  # noqa: E402
from training.dot7_rules import (F1_PAIRS_PER_TIER, file_sha256, guard_frozen, main_family, model_name,  # noqa: E402
                                 provenance_path, run_model, write_provenance)

FIGS = {"h2": "dot7_hinh2_cong_hoc_duoc", "h3": "dot7_hinh3_f1_theo_muc", "h4": "dot7_hinh4_khoi100",
        "h5": "dot7_hinh5_pc1_oracle"}
MODELS = ("hist_gb", "rf", "mlp")
MODEL_LABEL = {"hist_gb": "HistGB", "rf": "Random Forest", "mlp": "MLP"}
LEVEL_NAME = {5: "thô", 6: "giữa", 7: "mịn"}
FAMILY = {"h3_res_": "H3 r", "s2_level_": "S2 L", "latlon_": "LL ", "square_utm_": "UTM "}
F1_LABELS = ("tuong_duong", "chua_phan_dinh", "cho_giu_rieng", "mo_ta")
PC1_KINDS = {"deu": ("o", "đều"), "lognormal": ("s", "lognormal"), "tau_mu": ("^", "τ = μ (chỉ báo)")}
FAIL_REASON = "co_luoi_khong_hoc_duoc"  # khac "khong_kiem_dinh": muc khong vao cong
ALPHA = 0.05
RED, GREY, SHADE, BAND = "#b3261e", "#9a9893", "#f3ebe3", "#ecebe7"
LEARN_COLS = ["grid_a", "grid_b", "model_a", "model_b", "muc", "delta_hat", "kiem_dinh", "hoc_duoc"]
F1_COLS = ["family", "grid_a", "grid_b", "delta_ref_level", "delta_hat", "ci_low", "ci_high", "delta_min_thr",
           "p_holm", "label"]


def loi(msg):
    raise SystemExit(f"LOI: {msg}")


def read_cols(path, cols, st, what):
    """CSV can cac cot `cols`; thieu file/cot -> None + ghi chu (bo phan do, khong bia)."""
    name = os.path.basename(path)
    if not os.path.isfile(path):
        st["notes"].append(f"{what}: không có {name} — bỏ")
        return None
    d = pd.read_csv(path)
    miss = [c for c in cols if c not in d.columns]
    if miss:
        st["notes"].append(f"{what}: {name} thiếu cột {', '.join(miss)} — bỏ")
        return None
    st["used"].append(path)
    return d


def _check_target(d, t, path):
    if "target" in d.columns and set(d["target"]) != {t}:
        loi(f"{path}: target {sorted(set(d['target']))} khac {t}")


def short(g):
    for pre, lab in FAMILY.items():
        if g.startswith(pre):
            return lab + g[len(pre):].replace("deg", "°").replace("m", " m")
    return g


def grid_order(levels: pd.Series) -> list:
    fam = list(FAMILY)
    return sorted(levels.index, key=lambda g: (levels[g], next(i for i, p in enumerate(fam) if g.startswith(p)), g))


def x_positions(order, levels, gap=0.8) -> dict:
    xs, x = {}, 0.0
    for i, g in enumerate(order):
        if i:
            x += 1 + (gap if levels[g] != levels[order[i - 1]] else 0)
        xs[g] = x
    return xs


# ---------------------------------------------------------
# Doc / ghep
# ---------------------------------------------------------
def learn_path(res, t, model):
    if t == "salinity" and model == "hist_gb":  # HistGB do man khong co cong (gate_required False)
        return os.path.join(res, "dot5_cv1_kiem_dinh_vs_baseline_rong.csv")
    return os.path.join(res, model_name(f"dot7_{t}_hoc_duoc.csv", model))


def level_ref(res, t, model, st):
    """delta_ref (MAE CV cua mo hinh o muc, TB luoi tham chieu) cua ho chinh cung mo hinh, theo muc."""
    path = os.path.join(res, model_name(out_name(t, "cv1"), model))
    kd = read_cols(path, ["family", "delta_ref_level", "delta_ref"], st, f"{NAMES[t]} ({MODEL_LABEL[model]}) Δ_ref")
    if kd is None:
        return None
    g = kd[kd["family"] == main_family(model)].groupby("delta_ref_level")["delta_ref"]
    if g.ngroups == 0 or (g.nunique() != 1).any():
        loi(f"{path}: delta_ref ho {main_family(model)} khong duy nhat theo muc")
    return g.first()


def learn_table(res, model, levels: pd.Series, st) -> pd.DataFrame:
    """(bien, luoi): Delta_hat cong hoc duoc (mo hinh - season_mean) / delta_ref; trang_thai hoc_duoc/khong/mo_ta."""
    cols = ["target", "model", "grid", "muc", "delta_hat", "delta_ref", "y_pct", "trang_thai"]
    parts = []
    for t in VARIABLES:
        path = learn_path(res, t, model)
        d = read_cols(path, LEARN_COLS, st, f"{NAMES[t]} ({MODEL_LABEL[model]}) cổng")
        if d is None:
            continue
        _check_target(d, t, path)
        if sorted(d["grid_a"]) != sorted(levels.index) or (d["grid_a"] != d["grid_b"]).any():
            loi(f"{path}: khong dung 13 luoi, moi luoi mot dong (grid_a = grid_b)")
        if set(d["model_a"]) != {run_model(model)} or set(d["model_b"]) != {"season_mean"}:
            loi(f"{path}: khong phai {run_model(model)} vs season_mean")
        if (d["grid_a"].map(levels) != d["muc"]).any():
            loi(f"{path}: muc lech bang dien tich")
        ref = level_ref(res, t, model, st)
        if ref is None:
            continue
        d = d.assign(target=t, model=model, grid=d["grid_a"], delta_ref=d["muc"].map(ref))
        if d["delta_ref"].isna().any():
            loi(f"{t}/{model}: thieu delta_ref cho mot muc")
        kd, hd = _bool(d["kiem_dinh"]), _bool(d["hoc_duoc"])
        d["trang_thai"] = np.where(~kd, "mo_ta", np.where(hd, "hoc_duoc", "khong_hoc_duoc"))
        d["y_pct"] = 100 * d["delta_hat"] / d["delta_ref"]
        parts.append(d[cols])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def gate_fails(res, model, hoc: pd.DataFrame, st) -> dict:
    """{muc: [bien truot cong]}; hop_le phai khop hoc_duoc cua moi luoi kiem dinh."""
    path = os.path.join(res, gate_name(model))
    g = read_cols(path, ["bien", "muc", "hop_le", "ly_do"], st, f"cổng {MODEL_LABEL[model]}")
    if g is None:
        return {}
    if "model" in g.columns and set(g["model"]) != {model}:
        loi(f"{path}: model {sorted(set(g['model']))} khac {model}")
    for (t, m), x in hoc[hoc["trang_thai"] != "mo_ta"].groupby(["target", "muc"]):
        r = g[(g["bien"] == t) & (g["muc"] == m)]
        if len(r) != 1 or bool(_bool(r["hop_le"]).iloc[0]) != bool((x["trang_thai"] == "hoc_duoc").all()):
            loi(f"{path}: hop_le ({t}, muc {m}) khong khop hoc_duoc trong file cong hoc duoc")
    f = g[g["ly_do"] == FAIL_REASON]
    return {m: [t for t in VARIABLES if t in set(f.loc[f["muc"] == m, "bien"])] for m in LEVELS}


def f1_table(res, levels: pd.Series, st) -> pd.DataFrame:
    """Cap F1_chinh HistGB 6 bien: Delta_hat va CI chia delta_min_thr."""
    cols = ["target", "muc", "grid_a", "grid_b", "r", "r_lo", "r_hi", "p_holm", "label"]
    parts = []
    for t in VARIABLES:
        path = os.path.join(res, out_name(t, "cv1"))
        d = read_cols(path, F1_COLS, st, f"{NAMES[t]} F1")
        if d is None:
            continue
        f = d[d["family"] == "F1_chinh"]
        n = {int(k): int(v) for k, v in f["delta_ref_level"].value_counts().items()}
        if n != F1_PAIRS_PER_TIER:
            loi(f"{path}: so cap F1 theo muc {n} khac {F1_PAIRS_PER_TIER}")
        lv = f["delta_ref_level"]
        if (f["grid_a"].map(levels) != lv).any() or (f["grid_b"].map(levels) != lv).any():
            loi(f"{path}: cap F1 lech muc")
        bad = set(f["label"]) - set(F1_LABELS)
        if bad:
            loi(f"{path}: nhan la {sorted(bad)}")
        thr = f["delta_min_thr"]
        f = f.assign(target=t, muc=lv.astype(int), r=f["delta_hat"] / thr, r_lo=f["ci_low"] / thr,
                     r_hi=f["ci_high"] / thr)
        if not np.isfinite(f[["r", "r_lo", "r_hi"]].to_numpy(float)).all():
            loi(f"{path}: Delta_hat/CI chia delta_min_thr khong huu han")
        parts.append(f[cols])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def k100_paths(res, t):
    pre = "dot6" if t == "salinity" else f"dot7_{t}"
    return os.path.join(res, f"{pre}_khoi100_mae.csv"), os.path.join(res, f"{pre}_khoi100_cap_min.csv")


def k100_tables(res, levels: pd.Series, st):
    """HistGB: % tang MAE 100 km so 50 km theo luoi; cap min |Delta| 100 km / Delta_min."""
    mae, cap = [], []
    for t in VARIABLES:
        pm, pc = k100_paths(res, t)
        d = read_cols(pm, ["grid", "tang_rel_tb"], st, f"{NAMES[t]} khối 100 km")
        if d is not None:
            _check_target(d, t, pm)
            if sorted(d["grid"]) != sorted(levels.index):
                loi(f"{pm}: khong dung 13 luoi")
            mae.append(d.assign(target=t, muc=d["grid"].map(levels), y_pct=100 * d["tang_rel_tb"])[
                ["target", "grid", "muc", "y_pct"]])
        c = read_cols(pc, ["grid_a", "grid_b", "abs_delta_k100_tb", "delta_min"], st, f"{NAMES[t]} cặp mịn 100 km")
        if c is not None:
            _check_target(c, t, pc)
            if not ((c["grid_a"].map(levels) == 7) & (c["grid_b"].map(levels) == 7)).all():
                loi(f"{pc}: co cap khong phai muc min")
            cap.append(c.assign(target=t, ratio=c["abs_delta_k100_tb"] / c["delta_min"])[
                ["target", "grid_a", "grid_b", "ratio"]])
    mae = pd.concat(mae, ignore_index=True) if mae else pd.DataFrame(columns=["target", "grid", "muc", "y_pct"])
    cap = pd.concat(cap, ignore_index=True) if cap else pd.DataFrame(columns=["target", "grid_a", "grid_b", "ratio"])
    return mae, cap


def pc1_tables(res, st):
    p = read_cols(os.path.join(res, "dot7_pc1_tong_hop.csv"), ["pair_index", "level", "k", "kind", "ty_le_phat_hien"],
                  st, "PC-1")
    if p is not None and set(p["kind"]) - set(PC1_KINDS):
        loi(f"PC-1: kieu tiem la {sorted(set(p['kind']) - set(PC1_KINDS))}")
    v = read_cols(os.path.join(res, "dot7_pc1_ket_luan.csv"),
                  ["trang_thai", "k", "kieu", "n_cap_min", "min_phat_hien", "max_tost"], st, "PC-1 kết luận")
    return p, v


def oracle_table(res, levels: pd.Series, st):
    """Cap min (muc 7), khoi 50 km: |Delta| oracle / MAE oracle muc (%)."""
    path = os.path.join(res, "dot7_oracle_cap.csv")
    d = read_cols(path, ["target", "blocks", "delta_ref_level", "grid_a", "grid_b", "ti_le_delta_tuong_doi"], st,
                  "sai số lý tưởng")
    if d is None:
        return None
    d = d[(d["blocks"] == "k50") & (d["delta_ref_level"] == 7)]
    d = d.assign(y_pct=100 * d["ti_le_delta_tuong_doi"])
    if not ((d["grid_a"].map(levels) == 7) & (d["grid_b"].map(levels) == 7)).all():
        loi(f"{path}: cap muc 7 co luoi khong o muc min")
    for t in VARIABLES:
        if t not in set(d["target"]):
            st["notes"].append(f"sai số lý tưởng: {NAMES[t]} không có cặp mịn (k50) trong {os.path.basename(path)}")
    return d


# ---------------------------------------------------------
# Ve
# ---------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID_INK, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(which="both", colors=INK2, labelsize=7)


def _offsets(keys, half=0.3):
    return dict(zip(keys, np.linspace(-half, half, len(keys)))) if len(keys) > 1 else {k: 0.0 for k in keys}


def _dot(ax, x, y, t, kind="dac", s=26):
    c = COLORS[t]
    ax.scatter(x, y, s=s, marker=MARKERS.get(t, "o"), facecolors=c if kind == "dac" else SURFACE, edgecolors=c,
               linewidths=1.1, alpha=0.35 if kind == "nhat" else 1.0, zorder=3)


def _span(xs, levels, m):
    gx = [xs[g] for g in xs if levels[g] == m]
    return min(gx) - 0.5, max(gx) + 0.5


def _grid_axis(ax, xs, levels):
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([short(g) for g in xs], rotation=60, ha="right", fontsize=6.5)
    for m in LEVELS:
        lo, hi = _span(xs, levels, m)
        ax.annotate(f"mức {m} ({LEVEL_NAME[m]})", ((lo + hi) / 2, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=INK2)
        if m != LEVELS[-1]:
            ax.axvline(hi + 0.4, color=GRID_INK, lw=0.8, zorder=0)
    ax.set_xlim(min(xs.values()) - 0.7, max(xs.values()) + 0.7)


def _mk(label, mfc, mec, marker="o", alpha=1.0):
    from matplotlib.lines import Line2D

    return Line2D([], [], ls="", marker=marker, ms=6, mfc=mfc, mec=mec, alpha=alpha, label=label)


def _var_handles(ts):
    return [_mk(NAMES[t], COLORS[t], COLORS[t], MARKERS.get(t, "o")) for t in VARIABLES if t in ts]


def _finish(fig, handles, notes, ncol=5):
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False,
                   fontsize=7, labelcolor=INK2)
    if notes:
        fig.text(0.01, -0.09, "Ghi chú (phần bị bỏ / không có):\n" + "\n".join(f"• {n}" for n in notes),
                 fontsize=6.5, color=INK2, va="top", ha="left")


def save(fig, path_noext) -> list:
    plt = _plt()
    paths = []
    for ext in ("png", "svg"):
        p = f"{path_noext}.{ext}"
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _empty(ax):
    ax.text(0.5, 0.5, "không có dữ liệu", transform=ax.transAxes, ha="center", color=INK2, fontsize=8)


def _ref_line(ax, y, text):
    ax.axhline(y, color=INK2, ls="--", lw=1, zorder=1)
    ax.annotate(text, (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                fontsize=7, color=INK2)


def fig_h2(res, levels, st, out):
    from matplotlib.patches import Patch

    plt = _plt()
    xs, off = x_positions(grid_order(levels), levels), _offsets(VARIABLES)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13.5, 4.6), sharey=True, facecolor=SURFACE)
    seen, any_fail = set(), False
    kind = {"hoc_duoc": "dac", "khong_hoc_duoc": "rong", "mo_ta": "nhat"}
    for ax, model in zip(axes, MODELS):
        _style(ax)
        hoc = learn_table(res, model, levels, st)
        for m, ts in gate_fails(res, model, hoc, st).items():
            if ts:
                any_fail = True
                lo, hi = _span(xs, levels, m)
                ax.axvspan(lo, hi, color=SHADE, lw=0, zorder=0)
                ax.annotate("trượt cổng:\n" + "\n".join(NAMES[t] for t in ts), ((lo + hi) / 2, 0.03),
                            xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=5.8, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        for r in hoc.itertuples():
            _dot(ax, xs[r.grid] + off[r.target], r.y_pct, r.target, kind[r.trang_thai])
        seen |= set(hoc["target"])
        if hoc.empty:
            _empty(ax)
        _grid_axis(ax, xs, levels)
        ax.set_title(MODEL_LABEL[model], loc="left", fontsize=9, color=INK, pad=14)
    axes[0].set_ylabel("$\\hat{\\Delta}$ / $\\Delta_{ref}$ (%)", color=INK2, fontsize=8)
    fig.suptitle("Cổng “mô hình học được” (CV khối 50 km): mô hình so với season_mean theo 13 lưới\n"
                 "$\\hat{\\Delta}$ = MAE(mô hình) − MAE(season_mean), < 0: mô hình tốt hơn;  $\\Delta_{ref}$ = MAE CV "
                 "của chính mô hình ở mức (TB 3 lưới tham chiếu, cột delta_ref của họ chính)", x=0.01, ha="left",
                 fontsize=9, color=INK)
    h = _var_handles(seen) + [_mk("đặc: học được", INK2, INK2), _mk("rỗng: không học được", SURFACE, INK2),
                              _mk("nhạt: mức không kiểm định (mô tả)", SURFACE, INK2, alpha=0.35)]
    if any_fail:
        h.append(Patch(facecolor=SHADE, edgecolor="none", label="nền: mức trượt cổng (≥ 1 biến)"))
    fig.tight_layout()
    _finish(fig, h, st["notes"], ncol=5)
    return save(fig, out)


def fig_h3(res, levels, st, out):
    from matplotlib.patches import Patch

    plt = _plt()
    f1 = f1_table(res, levels, st)
    pairs = f1.drop_duplicates(["grid_a", "grid_b"]).sort_values("muc", kind="stable")
    ys, y, prev = {}, 0.0, None
    for r in pairs.itertuples():
        if prev is not None:
            y += 1.6 if r.muc != prev else 1.0
        ys[(r.grid_a, r.grid_b)], prev = y, r.muc
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5), sharex=True, sharey=True, facecolor=SURFACE)
    axes = axes.ravel()
    for ax, t in zip(axes, VARIABLES):
        _style(ax)
        ax.grid(False)
        ax.axvspan(-1, 1, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        ax.set_title(NAMES[t], loc="left", fontsize=9, color=INK)
        g = f1[f1["target"] == t]
        if g.empty:
            _empty(ax)
        c, mk = COLORS[t], MARKERS.get(t, "o")
        sty = {"tuong_duong": (c, c, mk), "chua_phan_dinh": (c, SURFACE, mk), "cho_giu_rieng": (RED, RED, "s"),
               "mo_ta": (GREY, GREY, mk)}
        for r in g.itertuples():
            yy = ys[(r.grid_a, r.grid_b)]
            line, face, m = sty[r.label]
            ax.plot([r.r_lo, r.r_hi], [yy, yy], color=line, lw=1.4, solid_capstyle="round", zorder=2)
            ax.scatter(r.r, yy, s=30, marker=m, facecolors=face, edgecolors=line, linewidths=1.1, zorder=3)
            if pd.notna(r.p_holm) and r.p_holm < ALPHA:
                ax.annotate(f"p_Holm = {r.p_holm:.3f}".replace(".", ","), (r.r_hi, yy), xytext=(4, -2.5),
                            textcoords="offset points", fontsize=6.5, color=INK)
    if ys:
        axes[0].set_yticks(list(ys.values()))
        axes[0].set_yticklabels([f"[{levels[a]}] {short(a)} – {short(b)}" for a, b in ys], fontsize=6.5)
        axes[0].invert_yaxis()
    for ax in axes[-2:]:
        ax.set_xlabel("$\\hat{\\Delta}$ / $\\Delta_{min}$ (CI 95 %)", color=INK2, fontsize=8)
    fig.suptitle("F1 HistGB (họ F1_chính, CV khối 50 km): chênh MAE giữa khung cùng mức, đơn vị Δ_min", x=0.01,
                 ha="left", fontsize=10, color=INK)
    h = _var_handles(set(f1["target"])) + [
        _mk("đậm: tương đương (TOST)", INK2, INK2), _mk("rỗng: chưa phân định", SURFACE, INK2),
        _mk("đỏ: chờ giữ riêng (khác có ý nghĩa)", RED, RED, "s"), _mk("xám: mô tả (không kiểm định)", GREY, GREY),
        Patch(facecolor=BAND, edgecolor="none", label="vùng $|\\hat{\\Delta}| < \\Delta_{min}$")]
    fig.tight_layout()
    _finish(fig, h, st["notes"], ncol=6)
    return save(fig, out)


def _pair_axis(ax, pairs):
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{short(a)} – {short(b)}" for a, b in pairs], rotation=40, ha="right", fontsize=6.5,
                       rotation_mode="anchor")


def _log_if_positive(ax, v):
    from matplotlib.ticker import FuncFormatter

    if len(v) and (np.asarray(v, float) > 0).all():
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))


def fig_h4(res, levels, st, out):
    plt = _plt()
    mae, cap = k100_tables(res, levels, st)
    xs, off = x_positions(grid_order(levels), levels), _offsets(VARIABLES)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.1, 1]}, facecolor=SURFACE)
    for ax in (a1, a2):
        _style(ax)
    for r in mae.itertuples():
        _dot(a1, xs[r.grid] + off[r.target], r.y_pct, r.target)
    if mae.empty:
        _empty(a1)
    a1.axhline(0, color=INK2, lw=0.6, zorder=1)
    _ref_line(a1, 5, "ngưỡng 5 %")
    _grid_axis(a1, xs, levels)
    a1.set_ylabel("% tăng MAE: khối 100 km so với 50 km", color=INK2, fontsize=8)
    a1.set_title("MAE HistGB tăng khi khối CV 50 → 100 km (TB cách chia)", loc="left", fontsize=9, color=INK, pad=14)
    pairs = list(dict.fromkeys(zip(cap["grid_a"], cap["grid_b"])))
    poff = _offsets(VARIABLES, 0.28)
    for r in cap.itertuples():
        _dot(a2, pairs.index((r.grid_a, r.grid_b)) + poff[r.target], r.ratio, r.target)
    if cap.empty:
        _empty(a2)
    else:
        _pair_axis(a2, pairs)
        _log_if_positive(a2, cap["ratio"])
    _ref_line(a2, 1, "Δ_min")
    a2.set_ylabel("|Δ| khối 100 km / Δ_min (log)", color=INK2, fontsize=8)
    a2.set_title("Cặp mịn (mức 7) ở khối 100 km", loc="left", fontsize=9, color=INK, pad=14)
    fig.suptitle("Khối 100 km (HistGB): độ nhạy của sai số và của chênh giữa khung theo kích thước khối", x=0.01,
                 ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    _finish(fig, _var_handles(set(mae["target"]) | set(cap["target"])), st["notes"], ncol=6)
    return save(fig, out)


def fig_h5(res, levels, st, out):
    plt = _plt()
    p, v = pc1_tables(res, st)
    orc = oracle_table(res, levels, st)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.15]}, facecolor=SURFACE)
    for ax in (a1, a2):
        _style(ax)
    c = COLORS["salinity"]  # PC-1 cay tren sai so HistGB do man
    if p is None:
        _empty(a1)
    else:
        cats = [(m, k) for m in LEVELS for k in (1.0, 2.0)]
        cx = {ck: i + 0.6 * LEVELS.index(ck[0]) for i, ck in enumerate(cats)}
        koff = _offsets(list(PC1_KINDS), 0.24)
        for (m, k, kind), g in p.groupby(["level", "k", "kind"]):
            if (m, k) not in cx:
                loi(f"PC-1: (muc {m}, k {k}) ngoai {cats}")
            g = g.sort_values("pair_index")
            jit = np.linspace(-0.05, 0.05, len(g)) if len(g) > 1 else np.zeros(1)
            a1.scatter(cx[(m, k)] + koff[kind] + jit, 100 * g["ty_le_phat_hien"], s=22, marker=PC1_KINDS[kind][0],
                       facecolors=SURFACE if kind == "tau_mu" else c, edgecolors=c, linewidths=1.0, zorder=3)
        a1.set_xticks(list(cx.values()))
        a1.set_xticklabels([f"{LEVEL_NAME[m]}\nk = {k:g}" for m, k in cats], fontsize=7)
        _ref_line(a1, 80, "80 %")
        a1.set_ylim(-3, 105)
        a1.legend(handles=[_mk(lab, SURFACE if kd == "tau_mu" else c, c, mk) for kd, (mk, lab) in PC1_KINDS.items()],
                  frameon=False, fontsize=7, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0.0, 0.75),
                  title="kiểu cấy (mỗi điểm = 1 cặp)",
                  title_fontsize=7)
    a1.set_ylabel("tỉ lệ phát hiện (%) qua các lần lặp", color=INK2, fontsize=8)
    title = "PC-1 (HistGB, độ mặn): cấy chênh k·Δ_min vào một cặp F1"
    if v is not None and len(v) == 1:
        r = v.iloc[0]
        title += (f"\nkết luận file: {r['trang_thai']} (k = {r['k']:g}, {r['kieu']}, {int(r['n_cap_min'])} cặp mịn: "
                  f"min phát hiện {r['min_phat_hien']:.3f}, max TOST {r['max_tost']:.3f})").replace(".", ",")
    a1.set_title(title, loc="left", fontsize=8.5, color=INK)
    if orc is None or orc.empty:
        _empty(a2)
    else:
        pairs = list(dict.fromkeys(zip(orc["grid_a"], orc["grid_b"])))
        poff = _offsets(VARIABLES, 0.28)
        for r in orc.itertuples():
            _dot(a2, pairs.index((r.grid_a, r.grid_b)) + poff[r.target], r.y_pct, r.target)
        _pair_axis(a2, pairs)
        _log_if_positive(a2, orc["y_pct"])
        for yv in (1, 5):
            _ref_line(a2, yv, f"{yv} %")
    a2.set_ylabel("|Δ| oracle / MAE oracle mức (%, log)", color=INK2, fontsize=8)
    a2.set_title("Sai số lý tưởng (oracle = TB nhãn thật của ô, khối 50 km): cặp mịn", loc="left", fontsize=8.5,
                 color=INK)
    fig.tight_layout()
    ts = ({"salinity"} if p is not None else set()) | (set(orc["target"]) if orc is not None else set())
    _finish(fig, _var_handles(ts), st["notes"], ncol=6)
    return save(fig, out)


FIG_FUNCS = {"h2": fig_h2, "h3": fig_h3, "h4": fig_h4, "h5": fig_h5}


def outputs(out_dir) -> list:
    return [os.path.join(out_dir, "hinh", f"{FIGS[k]}.{ext}") for k in FIGS for ext in ("png", "svg")]


def main(a):
    outs = outputs(a.out_dir)
    guard_frozen(outs + [provenance_path(p) for p in outs], a.frozen_manifest)
    area = os.path.join(a.results_dir, AREA_FILE)
    levels = grid_table(area).set_index("grid")["muc"]
    os.makedirs(os.path.join(a.out_dir, "hinh"), exist_ok=True)
    for key, fn in FIG_FUNCS.items():
        st = {"notes": [], "used": [area]}
        paths = fn(a.results_dir, levels, st, os.path.join(a.out_dir, "hinh", FIGS[key]))
        src = {os.path.basename(p): file_sha256(p) for p in dict.fromkeys(st["used"])}
        for p in paths:
            write_provenance(p, vai_tro="mo_ta", sha_dau_vao=src, phan_bo=st["notes"])
        print(f"{FIGS[key]}: {len(src)} file vao, bo {len(st['notes'])} phan", flush=True)
        for n in st["notes"]:
            print(f"  - {n}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=RESULTS, help="noi co file ket qua da mo")
    ap.add_argument("--out-dir", default=RESULTS, help="hinh ghi vao <out-dir>/hinh")
    ap.add_argument("--frozen-manifest", nargs="+", default=FROZEN_MANIFESTS)
    main(ap.parse_args())

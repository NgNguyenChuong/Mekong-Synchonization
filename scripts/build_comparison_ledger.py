"""Comparison Ledger: mot dong cho moi cap khung cung muc x 6 bien (ho F1_chinh), chi tong hop file da mo.

Chay: venv/Scripts/python.exe scripts/build_comparison_ledger.py --res-dir KE_HOACH/ket-qua
"""
import argparse
from pathlib import Path

import pandas as pd

from md_util import md_table

TARGETS = ["salinity", "dsr_mcd18", "ndwi", "rain_chirps", "rh_era5", "t2m_era5"]
LEVEL = {5: "coarse", 6: "medium", 7: "fine"}


def grid_area_key(g):
    if g.startswith("h3_res_"):
        return "H3", "res_" + g[7:]
    if g.startswith("s2_level_"):
        return "S2", "level_" + g[9:]
    if g.startswith("square_utm_"):
        return "Square_UTM", g[11:]
    if g.startswith("latlon_"):
        return "LatLon", g[7:]
    raise ValueError(g)


def decide(label, label_cap):
    """label = nhan CV; label_cap = nhan sau giu rieng."""
    if label == "tuong_duong":
        return "Equivalent"
    if label == "cho_giu_rieng":
        return "Different" if label_cap == "cho_giu_rieng" else "Inconclusive"
    if label == "mo_ta":
        return "Descriptive"
    return "Inconclusive"


def ledger_for_target(var, kd, cap, inter, area):
    """kd: kiem_dinh_khoi; cap: bang final (cmp_id|grid_a,grid_b -> label); inter: tuong tac RF; area: grid->km2."""
    kd = kd[kd.family == "F1_chinh"]
    cap_by_pair = {frozenset((r.grid_a, r.grid_b)): r for r in cap.itertuples()}
    int_by_pair = {frozenset((r.grid_a, r.grid_b)): r for r in inter.itertuples()} if inter is not None else {}
    rows = []
    for r in kd.itertuples():
        pair = frozenset((r.grid_a, r.grid_b))
        c = cap_by_pair.get(pair)
        label_cap = None if c is None else (c.label_cap if hasattr(c, "label_cap") else c.label)
        gate = getattr(r, "cong_hoc_duoc", True)
        muc_kd = getattr(r, "muc_kiem_dinh", True)
        eligible = bool(muc_kd) and bool(gate is True or gate == 1)
        reason = "" if eligible else ("khong_kiem_dinh" if not muc_kd else "truot_cong")
        ia = int_by_pair.get(pair)
        a, b = area[r.grid_a], area[r.grid_b]
        rows.append(dict(
            region="MekongDelta", variable=var, grid_a=r.grid_a, grid_b=r.grid_b,
            level=LEVEL[int(r.delta_ref_level)], area_a_km2=a, area_b_km2=b, area_ratio=r.area_ratio,
            eligible=eligible, ineligible_reason=reason, G=r.G, n_point_seasons=r.n_common_point_seasons,
            delta_hat=r.delta_hat, delta_ref=r.delta_ref, delta_rel=r.delta_hat / r.delta_ref,
            delta_min_thr=r.delta_min_thr,
            ci90_low=r.ci_tost_low, ci90_high=r.ci_tost_high, ci95_low=r.ci_low, ci95_high=r.ci_high,
            p_raw=r.p_value, p_holm=r.p_holm, seeds_same_dir=r.seeds_same_dir,
            label_cv=r.label, decision=decide(r.label, label_cap), holdout_confirmed=label_cap,
            rf_interaction_p_holm=None if ia is None else ia.p_holm,
            rf_interaction_tost=None if ia is None else ia.tost_tuong_tac))
    return rows


def summary_md(df):
    out = ["", "## Tong ket (tu sinh)", "", "Theo (level, decision):", "",
           md_table(pd.crosstab(df.level, df.decision), index=True), "", "Theo bien x decision:", "",
           md_table(pd.crosstab(df.variable, df.decision), index=True), ""]
    mf = df[df.level != "coarse"]
    elig = mf[mf.eligible]
    x = int((elig.decision == "Equivalent").sum())
    out.append(f"Equivalent o medium+fine: {x}/{len(elig)} (cap du dieu kien); "
               f"tren {len(mf)} cap medium+fine ke ca khong du dieu kien/mo ta.")
    out.append("")
    rain = df[(df.variable == "rain_chirps") & (df.level == "coarse")]
    out.append("Cap mua coarse: quyet dinh cuoi = " + "; ".join(
        f"{r.grid_a}|{r.grid_b}: {r.decision} (cv={r.label_cv}, giu rieng={r.holdout_confirmed})"
        for r in rain.itertuples()))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default="KE_HOACH/ket-qua")
    res = Path(ap.parse_args().res_dir)
    ar = pd.read_csv(res / "dot4_doi_chieu_dien_tich.csv")
    area = {}
    for g in ["h3_res_5", "h3_res_6", "h3_res_7", "s2_level_12", "s2_level_9", "square_utm_17087m", "square_utm_6458m",
              "square_utm_2441m", "latlon_0.1552deg", "latlon_0.0586deg", "latlon_0.0222deg"]:
        f, rs = grid_area_key(g)
        area[g] = float(ar[(ar.framework == f) & (ar.resolution == rs)].mean_area_km2.iloc[0])
    rows = []
    for t in TARGETS:
        if t == "salinity":
            kd = pd.read_csv(res / "dot5_cv1_kiem_dinh_khoi.csv")
            cap = pd.read_csv(res / "dot5_final_so_cap_voi_cv.csv")
        else:
            kd = pd.read_csv(res / f"dot7_{t}_cv1_kiem_dinh_khoi.csv")
            cap = pd.read_csv(res / f"dot7_{t}_cv1_final_cap.csv")
        p = res / f"dot7_{t}_tuong_tac_mo_hinh__rf.csv"
        rows += ledger_for_target(t, kd, cap, pd.read_csv(p) if p.exists() else None, area)
    df = pd.DataFrame(rows)
    df.to_csv(res / "comparison_ledger.csv", index=False)
    md = ["# Comparison Ledger", "", f"{len(df)} cap (ho F1_chinh, 6 bien x 12 cap, HistGB, cv1). "
          "decision: Equivalent=tuong_duong; Different=cho_giu_rieng va tai lap o final; "
          "Inconclusive=chua_phan_dinh hoac co y nghia CV nhung khong tai lap; Descriptive=mo_ta.", "",
          md_table(df[["variable", "grid_a", "grid_b", "level", "area_ratio", "eligible", "ineligible_reason", "G", "delta_rel",
              "p_holm", "decision", "holdout_confirmed", "rf_interaction_p_holm"]]),
          summary_md(df)]
    (res / "comparison_ledger.md").write_text("\n".join(md), encoding="utf-8")
    print(summary_md(df))


if __name__ == "__main__":
    main()

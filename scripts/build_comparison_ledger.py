"""Comparison Ledger: mot dong cho moi cap khung cung muc x 6 bien (ho F1_chinh), chi tong hop file da mo.
--region mississippi|nile: doc KE_HOACH/ket-qua/doi_chung/<vung>/, bo salinity; provenance dau vao phai cung region.

--feature-set khong_diem (chi dbscl): doc/ghi file hau to __fs-<bo> trong KE_HOACH/ket-qua/dbscl_fs_<bo>/, co provenance.
--no-final <bien...>: bien khong co luot final -> holdout_confirmed = khong_co_luot_final, cap cho_giu_rieng = PendingFinal.
Vung: them cot ratio_gt_1_2 (area_ratio > 1,2) va ghi comparison_ledger.csv.provenance.json (sha dau vao, region, git).

Chay: venv/Scripts/python.exe scripts/build_comparison_ledger.py --res-dir KE_HOACH/ket-qua [--region mississippi]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from md_util import md_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from analyze_cv_family import FEATURE_SETS, MAX_RATIO, fs_name  # noqa: E402
from region import DEFAULT_REGION, REGIONS, region_paths  # noqa: E402
from training.dot7_rules import file_sha256, write_provenance  # noqa: E402

TARGETS = ["salinity", "dsr_mcd18", "ndwi", "rain_chirps", "rh_era5", "t2m_era5"]
REGION_LABEL = {"dbscl": "MekongDelta", "mississippi": "Mississippi", "nile": "Nile"}
GRIDS = ["h3_res_5", "h3_res_6", "h3_res_7", "s2_level_12", "s2_level_9", "square_utm_17087m", "square_utm_6458m",
         "square_utm_2441m", "latlon_0.1552deg", "latlon_0.0586deg", "latlon_0.0222deg", "s2_level_10", "s2_level_11"]
LEVEL = {5: "coarse", 6: "medium", 7: "fine"}
NO_FINAL = "khong_co_luot_final"


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
        if label_cap == NO_FINAL:
            return "PendingFinal"
        return "Different" if label_cap == "cho_giu_rieng" else "Inconclusive"
    if label == "mo_ta":
        return "Descriptive"
    return "Inconclusive"


def ledger_for_target(var, kd, cap, inter, area, region_label="MekongDelta", flag_ratio=False):
    """kd: kiem_dinh_khoi; cap: bang final (cmp_id|grid_a,grid_b -> label); inter: tuong tac RF; area: grid->km2.
    flag_ratio: them cot ratio_gt_1_2 (chi cho vung; DBSCL giu cot cu)."""
    kd = kd[kd.family == "F1_chinh"]
    cap_by_pair = {} if cap is None else {frozenset((r.grid_a, r.grid_b)): r for r in cap.itertuples()}
    int_by_pair = {frozenset((r.grid_a, r.grid_b)): r for r in inter.itertuples()} if inter is not None else {}
    rows = []
    for r in kd.itertuples():
        pair = frozenset((r.grid_a, r.grid_b))
        c = cap_by_pair.get(pair)
        label_cap = NO_FINAL if cap is None else None if c is None else (c.label_cap if hasattr(c, "label_cap") else c.label)
        gate = True if var == "salinity" else getattr(r, "cong_hoc_duoc", True)  # do man khong ap cong hoc duoc
        muc_kd = getattr(r, "muc_kiem_dinh", True)
        eligible = bool(muc_kd) and bool(gate is True or gate == 1)
        reason = "" if eligible else ("khong_kiem_dinh" if not muc_kd else "truot_cong")
        ia = int_by_pair.get(pair)
        a, b = area[r.grid_a], area[r.grid_b]
        rows.append(dict(
            region=region_label, variable=var, grid_a=r.grid_a, grid_b=r.grid_b,
            level=LEVEL[int(r.delta_ref_level)], area_a_km2=a, area_b_km2=b, area_ratio=r.area_ratio,
            **({"ratio_gt_1_2": bool(r.area_ratio > MAX_RATIO * (1 + 1e-12))} if flag_ratio else {}),
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


def git_state():
    """Commit HEAD + cay lam viec co sua file da theo doi hay khong."""
    def git(*args):
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def read_checked(path, region, seen, inputs=None, feature_set=None):
    """CSV + kiem provenance (neu co): region (thieu = dbscl) == --region, feature_set (thieu = None) == --feature-set;
    gom pairs_mode vao `seen`, sha vao `inputs`."""
    if inputs is not None:
        inputs[str(path).replace("\\", "/")] = file_sha256(path)
    pv = Path(f"{path}.provenance.json")
    if pv.exists():
        info = json.loads(pv.read_text(encoding="utf-8"))
        got = info.get("region", DEFAULT_REGION)
        if got != region:
            raise SystemExit(f"LOI: {path}: provenance region '{got}' khac --region '{region}'")
        if info.get("feature_set") != feature_set:
            raise SystemExit(f"LOI: {path}: provenance feature_set '{info.get('feature_set')}' khac '{feature_set}'")
        if "pairs_mode" in info:
            seen.add(info["pairs_mode"])
    elif region != DEFAULT_REGION or feature_set:
        raise SystemExit(f"LOI: {path}: thieu provenance - khong kiem duoc vung/bo dac trung")
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default=None, help="mac dinh KE_HOACH/ket-qua (vung: KE_HOACH/ket-qua/doi_chung/<vung>)")
    ap.add_argument("--region", choices=list(REGIONS), default=DEFAULT_REGION)
    ap.add_argument("--area-table", default=None, help="mac dinh <res-dir>/dot4_... (vung: bang dien tich cua vung)")
    ap.add_argument("--feature-set", choices=FEATURE_SETS, default=None, help="chi dbscl; xem docstring")
    ap.add_argument("--no-final", nargs="*", default=[], help="bien khong co luot final (xem docstring)")
    a = ap.parse_args()
    dbscl = a.region == DEFAULT_REGION
    fs = a.feature_set
    if fs and not dbscl:
        raise SystemExit(f"LOI: --feature-set chi cho --region {DEFAULT_REGION}")
    res = Path(a.res_dir or ("KE_HOACH/ket-qua" if dbscl and not fs else f"KE_HOACH/ket-qua/dbscl_fs_{fs}" if fs
                             else f"KE_HOACH/ket-qua/doi_chung/{a.region}"))
    area_path = a.area_table or ((Path("KE_HOACH/ket-qua") if fs else res) / "dot4_doi_chieu_dien_tich.csv" if dbscl else
                                 f"{region_paths(a.region)['report_dir']}/dot4_doi_chieu_dien_tich.csv")
    ar = pd.read_csv(area_path)
    area = {}
    for g in GRIDS:
        f, rs = grid_area_key(g)
        area[g] = float(ar[(ar.framework == f) & (ar.resolution == rs)].mean_area_km2.iloc[0])
    targets = TARGETS if dbscl else [t for t in TARGETS if t != "salinity"]  # vung doi chung khong co nhan do man
    rows, seen, inputs = [], set(), {}
    for t in targets:
        kd_p = res / fs_name("dot5_cv1_kiem_dinh_khoi.csv" if t == "salinity" else f"dot7_{t}_cv1_kiem_dinh_khoi.csv", fs)
        # DBSCL cu: salinity doc bang final dot5; bo dac trung: moi bien doc dot7_<bien>_cv1_final_cap
        cap_p = res / (fs_name(f"dot7_{t}_cv1_final_cap.csv", fs) if fs or t != "salinity" else "dot5_final_so_cap_voi_cv.csv")
        kd = read_checked(kd_p, a.region, seen, inputs, fs)
        cap = None if t in a.no_final else read_checked(cap_p, a.region, seen, inputs, fs)
        p = res / f"dot7_{t}_tuong_tac_mo_hinh__rf.csv"
        inter = read_checked(p, a.region, seen, inputs) if p.exists() and not fs else None
        rows += ledger_for_target(t, kd, cap, inter, area, REGION_LABEL[a.region], flag_ratio=not dbscl)
    if len(seen) > 1:
        raise SystemExit(f"LOI: cac bang dau vao khac pairs_mode {sorted(seen)}")
    df = pd.DataFrame(rows)
    out_csv, out_md = res / fs_name("comparison_ledger.csv", fs), res / fs_name("comparison_ledger.md", fs)
    df.to_csv(out_csv, index=False)
    if not dbscl or fs:  # DBSCL bo day du: giu nguyen (khong sidecar)
        write_provenance(out_csv, region=a.region, pairs_mode=sorted(seen)[0] if seen else None,
                         **({"feature_set": fs} if fs else {}), no_final=list(a.no_final),
                         inputs={**inputs, str(area_path).replace("\\", "/"): file_sha256(area_path)},
                         git=git_state(), n_rows=len(df))
    n_cap = df.groupby("variable").size().max()
    md = ["# Comparison Ledger", "", f"{len(df)} cap (ho F1_chinh, {len(targets)} bien x {n_cap} cap, HistGB, cv1). "
          "decision: Equivalent=tuong_duong; Different=cho_giu_rieng va tai lap o final; "
          "Inconclusive=chua_phan_dinh hoac co y nghia CV nhung khong tai lap; Descriptive=mo_ta." + (" PendingFinal=cho_giu_rieng o CV, bien khong co luot final." if a.no_final else ""), "",
          md_table(df[["variable", "grid_a", "grid_b", "level", "area_ratio", *(["ratio_gt_1_2"] if not dbscl else []),
                       "eligible", "ineligible_reason", "G", "delta_rel",
                       "p_holm", "decision", "holdout_confirmed", "rf_interaction_p_holm"]]),
          summary_md(df)]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(summary_md(df))


if __name__ == "__main__":
    main()

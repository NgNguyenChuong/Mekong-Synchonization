#!/usr/bin/env python
"""Gan fold CV cap DON VI kiem dinh (khoi 50 km, khoi it diem gop vao khoi ke) - dung chung 13 luoi.

Quyet dinh An duyet 2026-10-03 (chot TRUOC lan CV dau tien), sua theo CHG-14 (2026-10-04):
- J: khoi CV co < --min-points diem CV gop vao khoi KE CHUNG CANH co nhieu diem nhat (quy tac day du:
  src/eval_design.merge_small_blocks) TRUOC khi gan fold -> cot unit_id. Khoi giu rieng khong gop.
- I: 5 fold gan o cap don vi, can bang DONG THOI dien tich mat na pham vi (mac dinh scope_mask_v3.tif, CHG-10;
  phan chia da chot 2026-10-03 dung scope_mask_v2.tif - chay lai bang --scope-mask) va dien
  tich mat na pham vi ven bien (dist_coast_km <= --coast-km) - toi thieu hoa tong chenh tuong doi
  (src/eval_design.assign_unit_folds).
- D (An chot 2026-10-03, phuong an (1); CHG-14 2026-10-04): seed = NHAN phuong an, dung chung 13 luoi:
  data/eval/cv_folds_s<seed>.csv; cv_folds.csv = s42 = phuong an CHINH (hang 1) -> ket qua chinh;
  s43, s44 = phuong an PHU (phan tich do nhay). Quy tac (eval_design.fold_schemes), ap cho ca ba:
  * tap ung vien = toi uu dia phuong phan biet tu --n-starts diem xuat phat (rng(--pool-seed));
  * xep hang + PHA HOA (CHG-14): muc tieu lam tron 1e-9 -> lech ven bien max |lech| moi fold nho hon ->
    ma phan chia (tuple fold theo unit_id sap tang) nho hon;
  * ven bien: dien tich pham vi ven bien moi fold trong +-(--coast-tol) quanh trung binh fold;
  * phuong an phu (CHG-14, MAC DINH): muc tieu (diem can bang) <= --max-objective (TUYET DOI, 0,183);
    do trung (ty le don vi GIU fold sau ghep nhan Hungarian) <= --max-overlap voi s42 va giua s43-s44;
    trong cac cap hop le chon cap co TONG muc tieu nho nhat (hoa -> lech ven bien -> ma);
  * quy tac CU (CHG-06, chi de tai lap): --tols 0.10 0.15 (muc tieu <= (1 + tol) x tot nhat, thu tu tu dien);
  * Rand/ARI chi bao tham khao.
- O cat ngang khoi theo quy tac "cham"; khoi giu rieng -> cv_fold = -1.
- Kiem rieng ven bien moi fold (An yeu cau 2026-10-04): dien tich ven bien (dieu kien cung +-coast_tol) VA so
  diem CV ven bien (dist_coast_km tai pixel 90 m chua diem <= --coast-km; chi bao cao).

Dau ra:
- data/eval/cv_folds_s<seed>.csv, cv_folds.csv (block_id, unit_id, cv_fold, land_km2, scope_km2,
  coast_scope_km2, n_cv_points, n_cv_coast_points) + .provenance.json. Da co file va KHAC -> loi (ghi de chi
  khi --force; --force TU SAO LUU cv_folds*.csv, cell_blocks/ va cac bang bao cao se bi ghi de sang
  --backup-dir (mac dinh data/eval/deprecated_v2b/) TRUOC khi ghi; dich da co file khac -> thu muc con
  sao_luu_<thoi gian>, khong bao gio ghi de ban sao luu).
- data/eval/cell_blocks/s<seed>/<luoi>.csv cho moi phuong an; data/eval/cell_blocks/<luoi>.csv = phuong an
  dau (provenance tro cv_folds.csv) - dung boi train.py mac dinh.
- Bang tong hop (--report-dir): dot4_cv_tap_phuong_an.csv (moi phuong an: muc tieu, chenh, lech ven bien
  tung fold (dien tich + so diem CV), diem CV / diem CV ven bien tung fold, do trung/Rand/ARI tung cap, o train
  moi fold theo luoi, tham so tim kiem), dot4_cv_tap_ung_vien.csv (toan tap ung vien), dot4_cv_khoi.csv
  (khoi + don vi), dot4_cv_gop_khoi.csv (nhat ky gop), dot4_cv_folds_theo_fold.csv (moi phuong an x fold),
  dot4_cv_folds_theo_luoi.csv (moi phuong an x luoi: o train/val moi fold, diem CV cham duoc theo phuong an C).

Chay:  venv/Scripts/python.exe scripts/build_cv_folds.py [--seeds 42 43 44] [--max-objective 0.183] [--force]
Tai lap quy tac cu (CHG-06): venv/Scripts/python.exe scripts/build_cv_folds.py --tols 0.10 0.15 ...
Kiem lai dieu kien cua phuong an DA LUU (sau khi doi mat na/diem; khong ghi fold):
       venv/Scripts/python.exe scripts/build_cv_folds.py --check-only
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval_design import (  # noqa: E402
    TIE_BREAK_RULE,
    cell_block_table,
    dist_coast_at_points,
    exact_block_frame,
    fold_overlap,
    fold_pair_stats,
    fold_schemes,
    merge_small_blocks,
    parse_touched_folds,
    rank_pool,
    scheme_code,
    scheme_code_str,
    scheme_conditions,
    unit_folds_to_blocks,
    unit_pool,
)
from preprocessing import CANONICAL_BOUNDARY, file_sha256, provenance_path, write_provenance  # noqa: E402
from scope_mask import scope_area_by_block  # noqa: E402
from training.split import block_cv_cell_summary  # noqa: E402

CRITERIA = ("scope_km2", "coast_scope_km2")
FOLD_COLS = ["block_id", "unit_id", "cv_fold", "land_km2", "scope_km2", "coast_scope_km2", "n_cv_points",
             "n_cv_coast_points"]
DEFAULT_MAX_OBJECTIVE = 0.183  # CHG-14 (An chot 2026-10-04): = diem can bang s42 tren pham vi v2 (0,1829)
REPORT_NAMES = ("dot4_cv_tap_phuong_an.csv", "dot4_cv_tap_ung_vien.csv", "dot4_cv_khoi.csv", "dot4_cv_gop_khoi.csv",
                "dot4_cv_folds_theo_fold.csv", "dot4_cv_folds_theo_luoi.csv")


def log(msg):
    print(msg, flush=True)


def rel(p):
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:  # khac o dia
        return os.path.abspath(p).replace("\\", "/")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backup_plan(out_dir, report_dir):
    """Cac file SE BI GHI DE boi lan chay --force: [(nguon, duong dan tuong doi trong thu muc sao luu)]."""
    items = []
    for p in sorted(glob.glob(os.path.join(out_dir, "cv_folds*.csv*"))):
        items.append((p, os.path.basename(p)))
    cell_dir = os.path.join(out_dir, "cell_blocks")
    for dirpath, _, files in os.walk(cell_dir):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            items.append((p, os.path.join("cell_blocks", os.path.relpath(p, cell_dir))))
    if report_dir:
        for name in REPORT_NAMES:
            p = os.path.join(report_dir, name)
            if os.path.exists(p):
                items.append((p, os.path.join("ket-qua", name)))
    return items


def backup_before_force(out_dir, report_dir, backup_dir, now=None):
    """Sao luu file se bi ghi de TRUOC khi ghi (--force). Khong bao gio ghi de ban sao luu da co:
    - moi dich da co va TRUNG sha -> bo qua (da sao luu);
    - co dich da co nhung KHAC -> sao luu vao thu muc con sao_luu_<YYYYmmdd_HHMMSS>.
    Kiem sha sau khi chep; ghi SAO_LUU_cv_folds.json (nguon, dich, sha256). Tra ve thu muc sao luu (None neu
    khong co gi de sao luu)."""
    items = backup_plan(out_dir, report_dir)
    if not items:
        return None
    sha = {src: _sha(src) for src, _ in items}

    def status(base):
        st = []
        for src, r in items:
            d = os.path.join(base, r)
            st.append("moi" if not os.path.exists(d) else ("trung" if _sha(d) == sha[src] else "khac"))
        return st

    target = backup_dir
    st = status(target)
    if all(s == "trung" for s in st):
        log(f"Sao luu: {len(items)} file da co (trung sha) trong {target} - bo qua.")
        return target
    if "khac" in st:
        stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
        target = os.path.join(backup_dir, f"sao_luu_{stamp}")
        if os.path.exists(target):
            raise SystemExit(f"[LOI] thu muc sao luu {target} da ton tai.")
        st = status(target)
    rows = []
    for (src, r), s in zip(items, st):
        d = os.path.join(target, r)
        if s == "moi":
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(src, d)
            if _sha(d) != sha[src]:
                raise SystemExit(f"[LOI] sao luu {src} -> {d}: sha khac sau khi chep.")
        rows.append({"nguon": rel(src), "dich": rel(d), "sha256": sha[src]})
    manifest = os.path.join(target, "SAO_LUU_cv_folds.json")
    old = []
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            old = json.load(f).get("files", [])
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"created": datetime.now().isoformat(timespec="seconds"), "script": "build_cv_folds.py --force",
                   "files": old + [r for r in rows if r not in old]}, f, ensure_ascii=False, indent=2)
    log(f"Sao luu {len(items)} file (cv_folds*, cell_blocks/, bang bao cao) -> {target}")
    return target


def block_table(blocks, points, scope_mask, dist_coast, coast_km):
    """Moi khoi: is_holdout, land_km2, so diem CV/giu rieng (+ diem CV ven bien), dien tich mat na pham vi
    (+ ven bien)."""
    area = scope_area_by_block(exact_block_frame(blocks)[["block_id", "geometry"]], scope_mask, dist_coast, coast_km)
    t = blocks[["block_id", "land_km2", "is_holdout"]].copy()
    t["block_id"] = t["block_id"].astype(str)
    t["is_holdout"] = t["is_holdout"].astype(bool)
    t = t.merge(area, on="block_id", how="left")
    # lam tron NGAY (boi so 0,0009 km2 -> 4 chu so la du) de gan fold tai lap duoc tu cot trong cv_folds.csv
    for c in ("scope_km2", "coast_scope_km2", "coast_nan_km2"):
        t[c] = t[c].round(4)
    cnt = points.groupby(["block_id", "is_holdout"]).size().unstack(fill_value=0)
    t["n_points"] = t["block_id"].map(cnt.get(False, pd.Series(dtype=int))).fillna(0).astype(int)
    t["n_holdout_points"] = t["block_id"].map(cnt.get(True, pd.Series(dtype=int))).fillna(0).astype(int)
    cv = points[~points["is_holdout"]]
    coast = cv[cv["dist_coast_km"] <= coast_km].groupby("block_id").size()
    t["n_coast_points"] = t["block_id"].map(coast).fillna(0).astype(int)
    return t


def write_folds(df, path, force, prov):
    """Ghi bang fold; da co file KHAC -> loi tru khi --force. Tra ve True neu da ghi."""
    if os.path.exists(path) and not force:
        old = pd.read_csv(path, dtype={"block_id": str, "unit_id": str})
        same = (list(old.columns) == FOLD_COLS and len(old) == len(df)
                and (old["block_id"].to_numpy() == df["block_id"].to_numpy()).all()
                and (old["cv_fold"].to_numpy() == df["cv_fold"].to_numpy()).all()
                and (old["unit_id"].to_numpy() == df["unit_id"].to_numpy()).all())
        if not same:
            raise SystemExit(f"[LOI] {path} da co va KHAC phuong an tinh lai (tham so/khoi/diem doi?). "
                             "Khong ghi de - dung --force neu that su muon gan lai fold (CHI truoc lan CV dau).")
        log(f"Giu nguyen {path} (khop phuong an tinh lai).")
        return False
    df.to_csv(path, index=False)
    write_provenance(path, CANONICAL_BOUNDARY, **prov)
    log(f"Ghi {path}")
    return True


def fold_summary(folds, seed):
    cv = folds[folds["cv_fold"] >= 0]
    fs = cv.groupby("cv_fold").agg(
        n_units=("unit_id", "nunique"), n_blocks=("block_id", "size"),
        scope_km2=("scope_km2", "sum"), coast_scope_km2=("coast_scope_km2", "sum"),
        land_km2=("land_km2", "sum"), n_cv_points=("n_cv_points", "sum"),
        n_cv_coast_points=("n_cv_coast_points", "sum"),
        units=("unit_id", lambda s: " ".join(sorted(set(s)))))
    fs.insert(0, "seed", seed)
    for c, d in (("scope_km2", "scope_dev_pct"), ("coast_scope_km2", "coast_dev_pct"),
                 ("n_cv_points", "points_dev_pct"), ("n_cv_coast_points", "coast_points_dev_pct")):
        fs[d] = (100 * (fs[c] / fs[c].mean() - 1)).round(2)
    fs["coast_share_pct"] = (100 * fs["coast_scope_km2"] / fs["scope_km2"]).round(2)
    fs["coast_points_share_pct"] = (100 * fs["n_cv_coast_points"] / fs["n_cv_points"]).round(2)
    for c in ("scope_km2", "coast_scope_km2", "land_km2"):
        fs[c] = fs[c].round(1)
    return fs.reset_index()


def _dev_pct(v):
    v = np.asarray(v, float)
    return 100 * (v / v.mean() - 1)


def scheme_table(picked, info, a, units):
    """Moi phuong an: vai tro, muc tieu, chenh tuong doi, lech ven bien tung fold (dien tich + so diem CV),
    diem CV / diem CV ven bien tung fold, do trung/Rand/ARI tung cap."""
    seeds = list(picked)
    pts = units.set_index("unit_id")[["n_points", "n_coast_points"]]
    rows = []
    for i, sd in enumerate(seeds):
        at = picked[sd].attrs  # merge (pandas 3) bo attrs -> lay tu frame goc
        f = picked[sd].merge(pts, left_on="unit_id", right_index=True, how="left")
        tot = f.groupby("cv_fold")[[*CRITERIA, "n_points", "n_coast_points"]].sum().sort_index()
        r = {"seed": sd, "vai_tro": "chinh" if i == 0 else "phu", "rank_in_pool": at["rank"],
             "ma_phan_chia": at["code"],
             "objective": round(at["objective"], 6),
             "objective_ratio_vs_best": round(at["objective"] / info["best_objective"], 4)}
        for c in CRITERIA:
            r[f"{c}_rel_range"] = round(float((tot[c].max() - tot[c].min()) / tot[c].mean()), 4)
        r["coast_max_dev_pct"] = round(float(np.abs(_dev_pct(tot["coast_scope_km2"])).max()), 2)
        r["n_cv_points_rel_range"] = round(float((tot["n_points"].max() - tot["n_points"].min())
                                                 / tot["n_points"].mean()), 4)
        r["coast_points_max_dev_pct"] = round(float(np.abs(_dev_pct(tot["n_coast_points"])).max()), 2)
        dev_a, dev_p = _dev_pct(tot["coast_scope_km2"]), _dev_pct(tot["n_coast_points"])
        dev_s = _dev_pct(tot["scope_km2"])
        for j, k in enumerate(tot.index):
            r[f"scope_km2_f{k}"] = round(float(tot.loc[k, "scope_km2"]), 1)
            r[f"scope_dev_pct_f{k}"] = round(float(dev_s[j]), 2)
            r[f"coast_km2_f{k}"] = round(float(tot.loc[k, "coast_scope_km2"]), 1)
            r[f"coast_dev_pct_f{k}"] = round(float(dev_a[j]), 2)
            r[f"coast_share_pct_f{k}"] = round(float(100 * tot.loc[k, "coast_scope_km2"] / tot.loc[k, "scope_km2"]), 2)
            r[f"n_cv_points_f{k}"] = int(tot.loc[k, "n_points"])
            r[f"n_cv_coast_points_f{k}"] = int(tot.loc[k, "n_coast_points"])
            r[f"coast_points_dev_pct_f{k}"] = round(float(dev_p[j]), 2)
            r[f"coast_points_share_pct_f{k}"] = round(float(100 * tot.loc[k, "n_coast_points"] / tot.loc[k, "n_points"]), 2)
        for sd2 in seeds:
            st = fold_pair_stats(f.sort_values("unit_id")["cv_fold"], picked[sd2].sort_values("unit_id")["cv_fold"])
            r[f"overlap_pct_vs_s{sd2}"] = round(100 * st["overlap"], 2)
            r[f"rand_vs_s{sd2}"] = round(st["rand"], 4)
            r[f"ari_vs_s{sd2}"] = round(st["ari"], 4)
        r.update({"mode": info["mode"], "max_objective": info["max_objective"], "n_starts": a.n_starts,
                  "pool_seed": a.pool_seed, "n_pool": info["n_pool"], "n_eligible_alt": info["n_eligible"],
                  "n_valid_alt_combos": info["n_valid_combos"],
                  "tols_thu": " ".join(str(t) for t in a.tols) if a.tols else "",
                  "tol_used": info["tol_used"], "max_overlap": a.max_overlap, "coast_tol": a.coast_tol,
                  "best_objective": round(info["best_objective"], 6)})
        rows.append(r)
    return pd.DataFrame(rows)


def grid_stats(name, seed, table, pts_cv, grid, fold_of_block):
    """Thong ke mot luoi x mot phuong an. Phuong an C: diem p (khoi fold k) du doan bang mo hinh fold k ap len
    o chua p -> hop le khi o chua p KHONG thuoc train fold k (o cham khoi giu rieng, hoac k trong touched_folds)."""
    train_cells = table[~table["touches_holdout"]]
    summ = block_cv_cell_summary(table, rule="touch")
    hit = gpd.sjoin(pts_cv[["point_id", "block_id", "geometry"]], grid[["cell_id", "geometry"]].to_crs(pts_cv.crs),
                    predicate="intersects", how="left")
    n_multi = int(hit["point_id"].duplicated().sum())
    hit = hit.drop_duplicates("point_id").merge(table[["cell_id", "touches_holdout", "cv_fold", "touched_folds"]],
                                                on="cell_id", how="left")
    no_cell = hit["cell_id"].isna()
    kp = hit["block_id"].map(fold_of_block).astype(int).to_numpy()
    touched = parse_touched_folds(hit["touched_folds"].where(~no_cell, "").tolist())
    in_train_k = np.array([(not nc) and (not bool(th)) and (k not in t)
                           for nc, th, k, t in zip(no_cell, hit["touches_holdout"].fillna(False), kp, touched)])
    has = ~no_cell.to_numpy()
    row = {
        "seed": seed, "grid": name, "cells": len(table), "train_cells": len(train_cells),
        "excluded_touch_holdout_cells": int(table["touches_holdout"].sum()),
        "cv_points": len(pts_cv), "cv_points_no_cell": int(no_cell.sum()), "cv_points_multi_cell": n_multi,
        "cv_points_violation_C": int(in_train_k.sum()),
        "cv_points_effective_C": int((has & ~in_train_k).sum()),
        "cv_points_in_cell_touching_holdout": int(hit["touches_holdout"].eq(True).sum()),
        "cv_points_cell_fold_ne_point_fold": int((has & (hit["cv_fold"].fillna(-9).astype(int).to_numpy() != kp)).sum()),
    }
    for k in summ["fold"]:
        r = summ.loc[summ["fold"] == k].iloc[0]
        row[f"train_cells_f{k}"] = int(r["n_train_cells"])
        row[f"val_cells_f{k}"] = int(r["n_val_cells"])
        row[f"excl_touch_cells_f{k}"] = int(r["n_excluded_touch_cells"])
        row[f"points_f{k}"] = int(((kp == k) & has & ~in_train_k).sum())
    return row


def check_existing(a, bt, merged):
    """--check-only: kiem lai dieu kien J/I/D cho cac phuong an DA LUU (cv_folds_s<seed>.csv) tren bang khoi MOI
    (mat na pham vi --scope-mask, diem --points). Khong ghi cv_folds / cell_blocks. Bang -> --report-dir voi tien
    to --check-prefix. Tra ve True neu MOI phuong an dat MOI dieu kien."""
    folds = {}
    for s in a.seeds:
        p = os.path.join(a.out_dir, f"cv_folds_s{s}.csv")
        folds[s] = pd.read_csv(p, dtype={"block_id": str, "unit_id": str})
    ref = folds[a.seeds[0]]
    for s, f in folds.items():
        if not (f.set_index("block_id")["unit_id"].sort_index()
                .equals(ref.set_index("block_id")["unit_id"].sort_index())):
            raise SystemExit(f"[LOI] cv_folds_s{s}.csv co anh xa khoi -> don vi khac s{a.seeds[0]}.")
    cvb = ref[ref["cv_fold"] >= 0]
    unit_of = cvb.set_index("block_id")["unit_id"]
    t = bt[bt["block_id"].isin(unit_of.index)].copy()
    t["unit_id"] = t["block_id"].map(unit_of)
    units = t.groupby("unit_id").agg(scope_km2=("scope_km2", "sum"), coast_scope_km2=("coast_scope_km2", "sum"),
                                     n_points=("n_points", "sum"), n_coast_points=("n_coast_points", "sum"),
                                     n_blocks=("block_id", "size"),
                                     blocks=("block_id", lambda s: " ".join(sorted(s)))).reset_index()
    old_u = cvb.groupby("unit_id").agg(scope_km2_cu=("scope_km2", "sum"), coast_scope_km2_cu=("coast_scope_km2", "sum"),
                                       n_points_cu=("n_cv_points", "sum")).reset_index()
    # Quy tac J chay lai tren diem moi co cho CUNG don vi khong (thong tin; dieu kien J la moi don vi >= min_points)
    m_new = merged.set_index("block_id").loc[unit_of.index, "unit_id"]
    same_merge = bool((m_new == unit_of).all())
    log(f"Don vi da chot: {len(units)}; it diem nhat {units['n_points'].min()} (>= {a.min_points}?); "
        f"chay lai quy tac gop tren diem moi cho cung don vi: {same_merge}")

    pool = unit_pool(units, CRITERIA, a.n_folds, a.n_starts, a.pool_seed)
    keys, _, _, _ = rank_pool(units, pool, CRITERIA, a.n_folds)
    best = pool[keys[0]][0]
    ids = units["unit_id"].astype(str).to_numpy()
    schemes, rank = {}, {}
    for s, f in folds.items():
        g = f[f["cv_fold"] >= 0].groupby("unit_id")["cv_fold"]
        if (g.nunique() != 1).any():
            raise SystemExit(f"[LOI] cv_folds_s{s}.csv: mot don vi nam o nhieu fold.")
        schemes[s] = g.first()
        lab = schemes[s].loc[ids].to_numpy()
        key = next((k for k in pool if scheme_code(k, ids) == scheme_code(lab, ids)), None)
        rank[s] = keys.index(key) + 1 if key is not None else None
    cond = scheme_conditions(units, schemes, best, criteria=CRITERIA, n_folds=a.n_folds, min_points=a.min_points,
                             tol=a.tols[0] if a.tols else 0.10, coast_tol=a.coast_tol, max_overlap=a.max_overlap,
                             max_objective=a.max_objective)
    cond["rank_in_pool_moi"] = cond["seed"].map(rank)
    cond["n_pool_moi"] = len(keys)
    cond["same_merge_rule_J"] = same_merge
    # Neu chay lai DUNG quy trinh D tren bang moi thi chon gi? (thong tin, khong ghi)
    try:
        picked, _, info = fold_schemes(units, a.seeds, criteria=CRITERIA, n_folds=a.n_folds, n_starts=a.n_starts,
                                       pool_seed=a.pool_seed, tols=a.tols, max_overlap=a.max_overlap,
                                       coast_tol=a.coast_tol, pool=pool, max_objective=a.max_objective)
        cond["rerun_pick_overlap_vs_saved"] = [
            fold_overlap(picked[s].set_index("unit_id").loc[ids, "cv_fold"].to_numpy(),
                         schemes[s].loc[ids].to_numpy()) for s in cond["seed"]]
        cond["rerun_tol_used"] = info["tol_used"]
    except ValueError as e:
        cond["rerun_pick_overlap_vs_saved"] = None
        cond["rerun_tol_used"] = f"loi: {e}"
    # gia tri cu (luu trong provenance) de so truoc/sau
    for s in cond["seed"]:
        with open(provenance_path(os.path.join(a.out_dir, f"cv_folds_s{s}.csv")), encoding="utf-8") as fh:
            pv = json.load(fh)
        cond.loc[cond["seed"] == s, "objective_cu"] = pv.get("objective")
        cond.loc[cond["seed"] == s, "scope_mask_cu"] = os.path.basename(pv.get("scope_mask", ""))
    cond["scope_mask_moi"] = os.path.basename(a.scope_mask)
    cond["scope_mask_moi_sha256"] = file_sha256(a.scope_mask)
    cond["points_sha256"] = file_sha256(a.points)
    for c in cond.columns:
        if cond[c].dtype == float:
            cond[c] = cond[c].round(6)
    log(cond.to_string(index=False))

    fold_rows = []
    for s, f in folds.items():
        cv = f[f["cv_fold"] >= 0]
        newb = cv[["block_id", "cv_fold"]].merge(bt[["block_id", "scope_km2", "coast_scope_km2", "n_points",
                                                     "n_coast_points"]], on="block_id")
        fn = newb.groupby("cv_fold")[["scope_km2", "coast_scope_km2", "n_points", "n_coast_points"]].sum()
        fo = cv.groupby("cv_fold")[["scope_km2", "coast_scope_km2", "n_cv_points"]].sum()
        for k in fn.index:
            fold_rows.append({"seed": s, "cv_fold": int(k), "n_units": int(cv.loc[cv["cv_fold"] == k, "unit_id"].nunique()),
                              "scope_km2_cu": round(fo.loc[k, "scope_km2"], 1), "scope_km2_moi": round(fn.loc[k, "scope_km2"], 1),
                              "coast_km2_cu": round(fo.loc[k, "coast_scope_km2"], 1),
                              "coast_km2_moi": round(fn.loc[k, "coast_scope_km2"], 1),
                              "coast_dev_pct_cu": round(100 * (fo.loc[k, "coast_scope_km2"] / fo["coast_scope_km2"].mean() - 1), 2),
                              "coast_dev_pct_moi": round(100 * (fn.loc[k, "coast_scope_km2"] / fn["coast_scope_km2"].mean() - 1), 2),
                              "scope_dev_pct_moi": round(100 * (fn.loc[k, "scope_km2"] / fn["scope_km2"].mean() - 1), 2),
                              "n_cv_points_cu": int(fo.loc[k, "n_cv_points"]), "n_cv_points_moi": int(fn.loc[k, "n_points"]),
                              "n_cv_coast_points_moi": int(fn.loc[k, "n_coast_points"]),
                              "coast_points_dev_pct_moi": round(100 * (fn.loc[k, "n_coast_points"]
                                                                       / fn["n_coast_points"].mean() - 1), 2)})
    fold_tab = pd.DataFrame(fold_rows)
    log(fold_tab.to_string(index=False))
    ut = units.merge(old_u, on="unit_id", how="left")
    for s in a.seeds:
        ut[f"fold_s{s}"] = ut["unit_id"].map(schemes[s])
    for c in ("scope_km2", "coast_scope_km2", "scope_km2_cu", "coast_scope_km2_cu"):
        ut[c] = ut[c].round(2)
    ut = ut[["unit_id", "n_blocks", "blocks", "n_points_cu", "n_points", "n_coast_points", "scope_km2_cu",
             "scope_km2", "coast_scope_km2_cu", "coast_scope_km2"] + [f"fold_s{s}" for s in a.seeds]]
    if a.report_dir:
        os.makedirs(a.report_dir, exist_ok=True)
        cond.to_csv(os.path.join(a.report_dir, f"{a.check_prefix}_dieu_kien_fold.csv"), index=False)
        fold_tab.to_csv(os.path.join(a.report_dir, f"{a.check_prefix}_fold.csv"), index=False)
        ut.to_csv(os.path.join(a.report_dir, f"{a.check_prefix}_don_vi.csv"), index=False)
        log(f"Bang -> {a.report_dir}/{a.check_prefix}_{{dieu_kien_fold,fold,don_vi}}.csv")
    ok = bool(cond["ok_all"].all())
    log("MOI DIEU KIEN DAT - giu nguyen phan chia" if ok else "CO DIEU KIEN VO - can chia lai (--force)")
    return ok


def resolve_rule(a):
    """CHG-14: mac dinh nguong tuyet doi --max-objective 0,183; --tols (quy tac cu) chi khi KHONG co --max-objective."""
    if a.tols and a.max_objective is not None:
        raise SystemExit("[LOI] --tols (quy tac tuong doi cu) va --max-objective (CHG-14) loai tru nhau.")
    if not a.tols and a.max_objective is None:
        a.max_objective = DEFAULT_MAX_OBJECTIVE
    return a


def main(a):
    a = resolve_rule(a)
    blocks = gpd.read_file(a.blocks)
    blocks["block_id"] = blocks["block_id"].astype(str)
    pts = gpd.read_file(a.points)
    pts["is_holdout"] = pts["is_holdout"].astype(bool)
    pts["dist_coast_km"] = dist_coast_at_points(pts, a.dist_coast)
    n_nan = int((~np.isfinite(pts.loc[~pts["is_holdout"], "dist_coast_km"])).sum())
    if n_nan:
        log(f"  CANH BAO: {n_nan} diem CV co dist_coast NaN (khong tinh ven bien)")
    pts_cv = pts[~pts["is_holdout"]].copy()

    log(f"Dien tich mat na pham vi theo khoi ({a.scope_mask}; ven bien dist_coast_km <= {a.coast_km}) ...")
    bt = block_table(blocks, pts, a.scope_mask, a.dist_coast, a.coast_km)
    if bt["coast_nan_km2"].sum() > 0:
        log(f"  CANH BAO: {bt['coast_nan_km2'].sum():.2f} km2 mat na pham vi co dist_coast NaN (khong tinh ven bien)")
    merged, mlog = merge_small_blocks(bt, min_points=a.min_points)
    log(f"--- Gop khoi < {a.min_points} diem CV (J) ---")
    log(mlog.to_string(index=False) if len(mlog) else "(khong khoi nao)")
    elig = (~merged["is_holdout"]) & (merged["land_km2"] > 0)
    units = merged[elig].groupby("unit_id").agg(
        scope_km2=("scope_km2", "sum"), coast_scope_km2=("coast_scope_km2", "sum"), n_points=("n_points", "sum"),
        n_coast_points=("n_coast_points", "sum"),
        n_blocks=("block_id", "size"), blocks=("block_id", lambda s: " ".join(sorted(s)))).reset_index()
    log(f"Don vi CV: {len(units)} (tu {int(elig.sum())} khoi CV); it diem nhat {units['n_points'].min()}; "
        f"diem CV {int(units['n_points'].sum())}, ven bien (<= {a.coast_km:g} km) {int(units['n_coast_points'].sum())}")
    if a.check_only:
        ok = check_existing(a, bt, merged)
        raise SystemExit(0 if ok else 3)

    blocks_sha, scope_sha, pts_sha = file_sha256(a.blocks), file_sha256(a.scope_mask), file_sha256(a.points)
    picked, pool, info = fold_schemes(units, a.seeds, criteria=CRITERIA, n_folds=a.n_folds, n_starts=a.n_starts,
                                      pool_seed=a.pool_seed, tols=a.tols, max_overlap=a.max_overlap,
                                      coast_tol=a.coast_tol, max_objective=a.max_objective)
    rule = (f"muc tieu <= {a.max_objective} (tuyet doi, CHG-14)" if info["mode"] == "tuyet_doi"
            else f"muc tieu <= {1 + info['tol_used']:.2f} x tot nhat (quy tac cu)")
    log(f"Tap ung vien: {info['n_pool']} toi uu dia phuong phan biet tu {a.n_starts} diem xuat phat; tot nhat "
        f"{info['best_objective']:.6f}; nguong DA DUNG: {rule}; {info['n_eligible']} phuong an phu don le hop le "
        f"(ven bien + do trung voi s{a.seeds[0]}); {info['n_valid_combos']} to hop hop le")
    st = scheme_table(picked, info, a, units)
    log(st.T.to_string())

    # --force: sao luu TRUOC moi lan ghi (cv_folds*, cell_blocks/, bang bao cao)
    backup = backup_before_force(a.out_dir, a.report_dir, a.backup_dir) if a.force else None
    schemes, fold_rows = {}, []
    for seed in a.seeds:
        uf = picked[seed]
        bf = unit_folds_to_blocks(merged, uf).rename(columns={"n_points": "n_cv_points",
                                                              "n_coast_points": "n_cv_coast_points"})
        bf["land_km2"] = bf["land_km2"].round(3)
        bf["scope_km2"] = bf["scope_km2"].round(4)
        bf["coast_scope_km2"] = bf["coast_scope_km2"].round(4)
        bf = bf[FOLD_COLS].sort_values("block_id").reset_index(drop=True)
        path = os.path.join(a.out_dir, f"cv_folds_s{seed}.csv")
        prov = dict(blocks=rel(a.blocks), blocks_sha256=blocks_sha, points=rel(a.points), points_sha256=pts_sha,
                    scope_mask=a.scope_mask, scope_mask_sha256=scope_sha, dist_coast=a.dist_coast,
                    coast_km=a.coast_km, min_points=a.min_points, n_folds=a.n_folds, seed=seed,
                    n_starts=a.n_starts, pool_seed=a.pool_seed, rule_mode=info["mode"],
                    max_objective=a.max_objective, tols=list(a.tols) if a.tols else None,
                    tol_used=info["tol_used"], max_overlap=a.max_overlap, coast_tol=a.coast_tol, seeds=list(a.seeds),
                    role="chinh" if seed == a.seeds[0] else "phu (phan tich do nhay)",
                    criteria=list(CRITERIA), objective=round(uf.attrs["objective"], 6), rank_in_pool=uf.attrs["rank"],
                    scheme_code=uf.attrs["code"], coast_max_dev=round(uf.attrs["coast_max_dev"], 6),
                    best_objective=round(info["best_objective"], 6), n_pool=info["n_pool"],
                    n_eligible_alt=info["n_eligible"], n_valid_alt_combos=info["n_valid_combos"],
                    tie_break=TIE_BREAK_RULE, decision="CHG-14 (An chot 2026-10-04)" if info["mode"] == "tuyet_doi"
                    else "CHG-06 (quy tac cu, tai lap)",
                    backup_of_previous=rel(backup) if backup else None,
                    method="merge_small_blocks (gop khoi < min_points vao khoi ke chung canh nhieu diem nhat) + "
                           "fold_schemes (muc tieu: tong chenh tuong doi (max-min)/mean cua scope_km2 va "
                           "coast_scope_km2; tap toi uu dia phuong xep theo tie_break; s42 = hang 1; phuong an phu: "
                           + ("muc tieu <= max_objective (tuyet doi), ven bien +-coast_tol, do trung Hungarian <= "
                              "max_overlap tung cap; cap TONG muc tieu nho nhat" if info["mode"] == "tuyet_doi" else
                              "muc tieu <= (1+tol) x tot nhat, ven bien +-coast_tol, do trung Hungarian <= "
                              "max_overlap tung cap; to hop thu tu tu dien")
                           + "); khoi giu rieng = -1",
                    rule_cells="touch (o cat ngang khoi)", shared_by_all_grids=True)
        write_folds(bf, path, a.force, prov)
        if seed == a.seeds[0]:
            write_folds(bf, os.path.join(a.out_dir, "cv_folds.csv"), a.force,
                        dict(prov, alias_of=os.path.basename(path)))
        schemes[seed] = (path, bf)
        fs = fold_summary(bf, seed)
        fold_rows.append(fs)
        log(f"--- Phuong an seed {seed}: muc tieu {uf.attrs['objective']:.6f} (hang {uf.attrs['rank']}, "
            f"ma {uf.attrs['code']}) ---")
        log(fs.drop(columns=["units"]).to_string(index=False))
    seeds = list(schemes)

    if a.report_dir:  # bang bao cao chi ghi SAU khi cv_folds da ghi/khop (khong --force + khac -> da dung o tren)
        os.makedirs(a.report_dir, exist_ok=True)
        mlog.to_csv(os.path.join(a.report_dir, "dot4_cv_gop_khoi.csv"), index=False)
        pool.to_csv(os.path.join(a.report_dir, "dot4_cv_tap_ung_vien.csv"), index=False)
        bt_out = merged.merge(schemes[seeds[0]][1][["block_id", "cv_fold"]], on="block_id")
        bt_out.to_csv(os.path.join(a.report_dir, "dot4_cv_khoi.csv"), index=False)
        pd.concat(fold_rows).to_csv(os.path.join(a.report_dir, "dot4_cv_folds_theo_fold.csv"), index=False)
        st.to_csv(os.path.join(a.report_dir, "dot4_cv_tap_phuong_an.csv"), index=False)

    cell_dir = os.path.join(a.out_dir, "cell_blocks")
    rows = []
    for gpath in sorted(glob.glob(os.path.join(a.grids_dir, "*.geojson"))):
        name = os.path.splitext(os.path.basename(gpath))[0]
        grid = gpd.read_file(gpath)
        grid["cell_id"] = grid["cell_id"].astype(str)
        gsha = file_sha256(gpath)
        for seed in seeds:
            fpath, folds = schemes[seed]
            table = cell_block_table(grid, blocks, folds)
            targets = [(os.path.join(cell_dir, f"s{seed}", f"{name}.csv"), fpath)]
            if seed == seeds[0]:
                targets.append((os.path.join(cell_dir, f"{name}.csv"), os.path.join(a.out_dir, "cv_folds.csv")))
            for out, fp in targets:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                table.to_csv(out, index=False)
                write_provenance(out, CANONICAL_BOUNDARY, grid=rel(gpath), grid_sha256=gsha, cv_folds=rel(fp),
                                 cv_folds_sha256=file_sha256(fp), blocks_sha256=blocks_sha,
                                 function="eval_design.cell_block_table")
            fold_of_block = folds.set_index("block_id")["cv_fold"].astype(int)
            r = grid_stats(name, seed, table, pts_cv, grid, fold_of_block)
            rows.append(r)
            tr = [r[f"train_cells_f{k}"] for k in range(a.n_folds)]
            log(f"s{seed} {name}: {r['cells']} o, train {r['train_cells']}, train/fold {tr}, diem CV C "
                f"{r['cv_points_effective_C']}/{r['cv_points']} (vi pham {r['cv_points_violation_C']}, "
                f"khong o {r['cv_points_no_cell']})")
    res = pd.DataFrame(rows)
    # o train moi fold theo luoi -> cot "train_cells_<luoi>" = "f0|f1|f2|f3|f4" trong bang phuong an
    for name, g in res.groupby("grid", sort=True):
        tr = g.set_index("seed")[[f"train_cells_f{k}" for k in range(a.n_folds)]]
        st[f"train_cells_{name}"] = st["seed"].map(lambda s: "|".join(str(int(v)) for v in tr.loc[s]))
    if a.report_dir:
        res.to_csv(os.path.join(a.report_dir, "dot4_cv_folds_theo_luoi.csv"), index=False)
        st.to_csv(os.path.join(a.report_dir, "dot4_cv_tap_phuong_an.csv"), index=False)
        log(f"Bang tong hop -> {a.report_dir}")
    if res["cv_points_violation_C"].any() or res["cv_points_no_cell"].any():
        raise SystemExit("[LOI] co diem CV khong cham duoc theo phuong an C (vi pham hoac khong co o).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blocks", default=os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    ap.add_argument("--points", default=os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--backup-dir", default=os.path.join(ROOT, "data", "eval", "deprecated_v2b"),
                    help="--force sao luu cv_folds*, cell_blocks/, bang bao cao vao day TRUOC khi ghi")
    ap.add_argument("--scope-mask", default="A:/Dataset_NCKH/features/scope_mask_v3.tif",
                    help="Mat na pham vi (v3 CHG-10; v2: A:/Dataset_NCKH/features/scope_mask_v2.tif)")
    ap.add_argument("--dist-coast", default="A:/Dataset_NCKH/raw/river/river_distance_90m.tif",
                    help="Raster co band dist_coast_km")
    ap.add_argument("--coast-km", type=float, default=20.0)
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="Moi seed = mot phuong an gan fold; seed dau -> cv_folds.csv")
    ap.add_argument("--n-starts", type=int, default=3000, help="So diem xuat phat dung tap ung vien")
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--max-objective", type=float, default=None,
                    help=f"CHG-14: phuong an phu co diem can bang (muc tieu, lam tron 1e-9) <= nguong TUYET DOI nay "
                         f"(mac dinh {DEFAULT_MAX_OBJECTIVE} khi khong truyen --tols)")
    ap.add_argument("--tols", type=float, nargs="+", default=None,
                    help="Quy tac CU (CHG-06, chi de tai lap): muc tieu <= (1 + tol) x tot nhat, thu lan luot "
                         "(vd 0.10 0.15); loai tru voi --max-objective")
    ap.add_argument("--max-overlap", type=float, default=0.60,
                    help="Do trung toi da (ty le don vi giu fold sau ghep Hungarian) giua moi cap phuong an")
    ap.add_argument("--coast-tol", type=float, default=0.15,
                    help="Dien tich pham vi ven bien moi fold trong +-tol quanh trung binh fold (moi phuong an)")
    ap.add_argument("--force", action="store_true",
                    help="Ghi de cv_folds*.csv da co (gan lai fold - co chu dich); tu sao luu sang --backup-dir")
    ap.add_argument("--check-only", action="store_true",
                    help="Chi kiem lai dieu kien J/I/D cua cac phuong an DA LUU tren mat na/diem hien tai; khong ghi "
                         "fold. Thoat 0 = dat het, 3 = co dieu kien vo")
    ap.add_argument("--check-prefix", default="dot4_scope_v3", help="Tien to bang cua --check-only")
    main(ap.parse_args())

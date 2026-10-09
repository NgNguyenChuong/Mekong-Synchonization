#!/usr/bin/env python
"""Tap cham PHU bo nhan keep6090 (CHG-15, cau V; An chot 2026-10-06 buoc 3): 349 diem goc lop WorldCover 60/90.

- KHONG sinh diem, KHONG lay mau lai: lay DUNG cac diem cua tap goc (data/eval/deprecated_v2b/eval_points.geojson,
  11.766 diem) bi scripts/filter_eval_points_scope.py loc ra (giu <=> scope v3 = 1 tai pixel 30 m chua diem) va co
  wc_class (pixel 30 m chua diem, band wc_class cua scope v3) thuoc {60, 90}. Cung ham scope_values_at_points.
- Kiem truoc khi ghi (sai -> LOI, khong ghi): sha256 nguon = filtered_from_sha256 va sha256 mat na = scope_mask_sha256
  trong provenance tap chinh; tai lap bo loc (scope = 1) ra DUNG tap chinh 10.801 diem (point_id + toa do); so diem
  bi loc theo lop = dropped_by_wc_class; tap point_id = tap added = True cua points_reference_keep6090.csv (kem
  wc_class, block_id, is_holdout trung).
- Khoi / don vi / fold theo VI TRI diem (phuong an C): training.point_eval.point_blocks voi holdout_blocks.geojson va
  cv_folds_s<n>.csv (moi cach chia mot cot cv_fold_s<n>, unit_id_s<n>). Diem ngoai moi khoi / khoi khong co trong file
  fold / is_holdout khac nguon -> LOI (CHG-22), khong bo am tham.
- Giu nguyen point_id, block_id, is_holdout, toa do tu nguon (doc lai sau khi ghi de kiem).
- Mo ta (khong loc): moi luoi - so diem trong luoi, o co trong bang hop nhat keep6090 o MOI mua, o scope_frac > 0
  -> <out_dir>/eval_points_6090_theo_luoi.csv.

Chay:  venv/Scripts/python.exe scripts/build_aux_points_6090.py [--dry-run] [--boundary <ranh gioi v2>]
"""
import argparse
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import CANONICAL_BOUNDARY, file_sha256, provenance_path, write_provenance  # noqa: E402
from scope_mask import scope_values_at_points, select_filtered_points  # noqa: E402
from training.point_eval import point_blocks  # noqa: E402

CLASSES = (60, 90)


def log(msg):
    print(msg, flush=True)


def rel(p):
    """Duong dan tuong doi ROOT; khac o dia (Windows) -> duong dan tuyet doi."""
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:
        return os.path.abspath(p).replace("\\", "/")


def _same_points(a, b) -> bool:
    """Cung point_id (cung thu tu) va toa do trung tuyet doi."""
    return (list(a["point_id"].astype(str)) == list(b["point_id"].astype(str))
            and np.array_equal(a.geometry.x.to_numpy(), b.geometry.x.to_numpy())
            and np.array_equal(a.geometry.y.to_numpy(), b.geometry.y.to_numpy()))


def check_against_main(src, attrs, main, main_prov, src_sha, scope_sha):
    """Tai lap bo loc cua filter_eval_points_scope.py: phai ra dung tap chinh + so loai theo lop. Sai -> ValueError."""
    if main_prov:
        for k, got in (("filtered_from_sha256", src_sha), ("scope_mask_sha256", scope_sha)):
            if main_prov.get(k) and main_prov[k] != got:
                raise ValueError(f"{k} trong provenance tap chinh ({main_prov[k][:12]}) khac file dang dung "
                                 f"({got[:12]}) - nguon/mat na khong phai cai da loc ra tap chinh")
    keep = (attrs["scope"] == 1).to_numpy()
    if not _same_points(src.loc[keep], main):
        raise ValueError(f"Tai lap bo loc (scope = 1) ra {int(keep.sum())} diem, KHONG trung tap chinh {len(main)} diem "
                         "(point_id/thu tu/toa do)")
    dropped = {str(int(k)): int(v) for k, v in attrs.loc[~keep, "wc_class"].value_counts().sort_index().items()}
    want = (main_prov or {}).get("dropped_by_wc_class")
    if want is not None and dropped != {str(k): int(v) for k, v in want.items()}:
        raise ValueError(f"So diem bi loc theo lop {dropped} khac provenance {want}")
    return dropped


def check_against_ref(sel, ref):
    """Tap point_id = tap added = True cua dap an keep6090; wc_class / block_id / is_holdout trung. Sai -> ValueError."""
    need = {"point_id", "added", "wc_class", "block_id", "is_holdout"}
    if not need <= set(ref.columns):
        raise ValueError(f"Dap an thieu cot {sorted(need - set(ref.columns))}")
    added = ref["added"].astype(str).str.lower().map({"true": True, "false": False})
    if added.isna().any():
        raise ValueError("Cot added co gia tri khong phai True/False")
    ra = ref[added.to_numpy()].drop_duplicates("point_id")
    ra = ra.set_index(ra["point_id"].astype(str))
    ids = set(sel["point_id"].astype(str))
    if set(ra.index) != ids:
        raise ValueError(f"Tap diem khac dap an added=True: thieu trong dap an {len(ids - set(ra.index))}, "
                         f"thua trong dap an {len(set(ra.index) - ids)}")
    s = sel.set_index(sel["point_id"].astype(str))
    for c, conv in (("wc_class", int), ("block_id", str), ("is_holdout", lambda v: str(v).lower() == "true")):
        a = s[c].map(conv)
        b = ra[c].map(conv).reindex(s.index)
        if (a != b).any():
            raise ValueError(f"{int((a != b).sum())} diem co {c} khac dap an keep6090")
    # moi diem added phai cung (block_id, is_holdout, wc_class) o MOI mua (khong doi theo mua)
    sub = ref[added.to_numpy()]
    if (sub.groupby("point_id")[["block_id", "is_holdout", "wc_class"]].nunique() > 1).any().any():
        raise ValueError("Dap an keep6090: mot diem added co block_id/is_holdout/wc_class khac nhau giua cac mua")


def assign_blocks(sel, blocks, folds_by_scheme: dict):
    """Them cv_fold_s<n>, unit_id_s<n> theo VI TRI diem (point_blocks). Khoi/is_holdout phai trung nguon."""
    out = sel.copy()
    for scheme, folds in folds_by_scheme.items():
        pb = point_blocks(sel, blocks, folds).set_index("point_id")
        pb = pb.reindex(sel["point_id"].to_numpy())
        if pb["block_id"].isna().any():  # point_blocks da bao loi diem ngoai khoi; giu phong khi doi ham
            raise ValueError(f"s{scheme}: {int(pb['block_id'].isna().sum())} diem khong co khoi")
        if (pb["is_holdout"].astype(bool).to_numpy() != sel["is_holdout"].astype(bool).to_numpy()).any():
            raise ValueError(f"s{scheme}: is_holdout theo vi tri khac is_holdout trong nguon")
        out[f"cv_fold_s{scheme}"] = pb["cv_fold"].astype(int).to_numpy()
        if "unit_id" in pb.columns:
            out[f"unit_id_s{scheme}"] = pb["unit_id"].astype(str).to_numpy()
    return out


STATIC_DESC = ("dem_mean", "dist_main_river_km", "dist_any_water_km", "dist_coast_km")


def describe_grids(points, grids_dir, tables_dir, label_set="keep6090"):
    """Moi luoi (chi MO TA, khong loc): diem trong luoi; o co dong o MOI mua cua bang hop nhat (train.py
    attach_features can); o co pixel dat scope v3 (scope_frac > 0 va scope_n_px > 0 = dieu kien train_ok_scope ve
    phia dac trung); o scope_frac = 0 hoac chi manh vun (scope_n_px = 0); o MOI dac trung tinh NaN; o scope_cx NaN
    (IDW khong dat duoc nhan o)."""
    from training.evaluate import assign_points_to_cells

    rows = []
    for gp in sorted(p for p in os.listdir(grids_dir) if p.endswith(".geojson")):
        grid = gp[:-8]
        tp = os.path.join(tables_dir, f"{grid}_unified_{label_set}.csv")
        if not os.path.exists(tp):
            log(f"  [canh bao] khong co {tp} - bo qua mo ta {grid}")
            continue
        g = gpd.read_file(os.path.join(grids_dir, gp))
        g["cell_id"] = g["cell_id"].astype(str)
        cell = assign_points_to_cells(points, g)
        head = pd.read_csv(tp, nrows=0).columns
        use = ["cell_id", "season", "scope_frac", *[c for c in ("scope_n_px", "scope_cx", *STATIC_DESC) if c in head]]
        t = pd.read_csv(tp, usecols=use, dtype={"cell_id": str})
        n_season = t["season"].nunique()
        stat = [c for c in STATIC_DESC if c in t.columns]
        t["_all_static_nan"] = t[stat].isna().all(axis=1) if stat else False
        per = t.groupby("cell_id").agg(
            n_season=("season", "nunique"), sf_min=("scope_frac", "min"), sf_max=("scope_frac", "max"),
            npx_min=("scope_n_px", "min") if "scope_n_px" in t else ("scope_frac", "size"),
            static_nan=("_all_static_nan", "any"),
            cx_nan=("scope_cx", lambda s: s.isna().any()) if "scope_cx" in t else ("_all_static_nan", "any"))
        c = cell.dropna().astype(str)
        info = per.reindex(c.to_numpy())
        full = (info["n_season"] == n_season).fillna(False).to_numpy(bool)
        land = full & (info["sf_min"] > 0).to_numpy() & (info["npx_min"].fillna(0) > 0).to_numpy()
        rows.append({"grid": grid, "n_diem": len(points), "n_trong_luoi": int(cell.notna().sum()),
                     "n_o_du_moi_mua": int(full.sum()), "n_o_co_dat_scope": int(land.sum()),
                     "n_scope_frac_0": int((full & (info["sf_max"] == 0).to_numpy()).sum()),
                     "n_scope_n_px_0": int((full & (info["sf_max"] > 0).to_numpy()
                                            & (info["npx_min"].fillna(0) == 0).to_numpy()).sum()),
                     "n_moi_dac_trung_tinh_nan": int((full & info["static_nan"].fillna(False).to_numpy(bool)).sum()),
                     "n_scope_cx_nan": int((full & info["cx_nan"].fillna(False).to_numpy(bool)).sum()),
                     "n_mua_bang": int(n_season), "n_o_khac_nhau": int(c.nunique())})
    return pd.DataFrame(rows)


def main(a):
    src_sha, scope_sha = file_sha256(a.source), file_sha256(a.scope)
    src = gpd.read_file(a.source)
    if src["point_id"].duplicated().any():
        raise SystemExit("[LOI] point_id trung trong nguon.")
    src["is_holdout"] = src["is_holdout"].astype(bool)
    main_pts = gpd.read_file(a.main_points)
    main_prov = {}
    if os.path.exists(provenance_path(a.main_points)):
        with open(provenance_path(a.main_points), encoding="utf-8") as f:
            main_prov = json.load(f)
    elif not a.allow_no_provenance:
        raise SystemExit(f"[LOI] Thieu {provenance_path(a.main_points)} - khong doi chieu duoc nguon/mat na.")
    log(f"Nguon {rel(a.source)}: {len(src)} diem (sha {src_sha[:12]}); tap chinh {rel(a.main_points)}: {len(main_pts)}")
    attrs = scope_values_at_points(src, a.scope)
    try:
        dropped = check_against_main(src, attrs, main_pts, main_prov, src_sha, scope_sha)
        log(f"Tai lap bo loc: khop tap chinh {len(main_pts)} diem; bi loc theo lop {dropped}")
        sel = select_filtered_points(src, attrs, a.classes)
        ref = pd.read_csv(a.points_ref, dtype={"point_id": str, "block_id": str})
        check_against_ref(sel, ref)
        log(f"Khop dap an {os.path.basename(a.points_ref)} (added = True): {len(sel)} diem")
        blocks = gpd.read_file(a.blocks)
        folds = {s: pd.read_csv(os.path.join(a.folds_dir, f"cv_folds_s{s}.csv"), dtype={"block_id": str})
                 for s in a.schemes}
        out = assign_blocks(sel, blocks, folds)
    except ValueError as exc:
        raise SystemExit(f"[LOI] {exc}")
    if set(out["point_id"]) & set(main_pts["point_id"]):
        raise SystemExit("[LOI] Tap cham phu giao tap chinh khac rong.")
    by_cls = out["wc_class"].value_counts().sort_index().to_dict()
    n_h = int(out["is_holdout"].sum())
    log(f"Tap cham phu: {len(out)} diem; theo lop {by_cls}; khoi giu rieng {n_h}, CV {len(out) - n_h}")
    for s in a.schemes:
        log(f"  s{s}: diem theo cv_fold {out[f'cv_fold_s{s}'].value_counts().sort_index().to_dict()}")

    desc = describe_grids(out, a.grids_dir, a.tables_dir) if a.tables_dir else pd.DataFrame()
    if len(desc):
        log(f"--- Mo ta theo luoi (bang hop nhat keep6090) ---\n{desc.to_string(index=False)}")
    if a.dry_run:
        log("--dry-run: khong ghi.")
        return out, desc
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cols = ["point_id", "block_id", "is_holdout", "wc_class", "zenodo_n_years",
            *[c for c in out.columns if c.startswith(("cv_fold_s", "unit_id_s"))], "geometry"]
    out = out[cols]
    out.to_file(a.out, driver="GeoJSON")
    back = gpd.read_file(a.out)
    if not (_same_points(back, out) and (back["block_id"].astype(str).to_numpy() == out["block_id"].astype(str).to_numpy()).all()
            and (back["is_holdout"].astype(bool).to_numpy() == out["is_holdout"].to_numpy()).all()):
        raise SystemExit("[LOI] Diem ghi lai khac nguon (toa do/point_id/block_id/is_holdout).")
    src_ids = src.set_index("point_id")
    if not (np.array_equal(src_ids.loc[back["point_id"]].geometry.x.to_numpy(), back.geometry.x.to_numpy())
            and np.array_equal(src_ids.loc[back["point_id"]].geometry.y.to_numpy(), back.geometry.y.to_numpy())):
        raise SystemExit("[LOI] Toa do ghi lai khac nguon.")
    if len(desc):
        desc.to_csv(os.path.join(os.path.dirname(os.path.abspath(a.out)), "eval_points_6090_theo_luoi.csv"), index=False)
    write_provenance(a.out, a.boundary,
                     rule=f"Tap cham PHU bo keep6090 (CHG-15 cau V; An chot 2026-10-06): diem nguon bi LOC khoi bo chinh "
                          f"(scope v3 != 1, cung quy tac filter_eval_points_scope.py) va wc_class (pixel 30 m chua diem) "
                          f"thuoc {sorted(a.classes)}. Khong sinh diem, khong lay mau.",
                     source=rel(a.source), source_sha256=src_sha, scope_mask=a.scope, scope_mask_sha256=scope_sha,
                     main_points=rel(a.main_points), main_points_sha256=file_sha256(a.main_points),
                     points_ref=a.points_ref, points_ref_sha256=file_sha256(a.points_ref),
                     blocks=rel(a.blocks), blocks_sha256=file_sha256(a.blocks),
                     cv_folds={f"s{s}": {"path": rel(os.path.join(a.folds_dir, f"cv_folds_s{s}.csv")),
                                         "sha256": file_sha256(os.path.join(a.folds_dir, f"cv_folds_s{s}.csv"))}
                               for s in a.schemes},
                     n_points=len(out), n_by_wc_class={str(k): int(v) for k, v in by_cls.items()},
                     n_holdout=n_h, n_cv=len(out) - n_h, dropped_by_wc_class_source=dropped,
                     block_rule="phuong an C: khoi/fold/don vi theo VI TRI diem (training.point_eval.point_blocks)",
                     note="point_id/block_id/is_holdout/toa do giu nguyen tu nguon; dap an: points_reference_keep6090.csv "
                          "(added = True) - cham voi train.py --points-ref-variant keep6090 --points-subset-of-ref")
    log(f"Ghi {a.out} ({len(out)} diem) + provenance")
    return out, desc


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=os.path.join(ROOT, "data", "eval", "deprecated_v2b", "eval_points.geojson"),
                    help="Tap diem goc TRUOC khi loc (11.766 diem)")
    ap.add_argument("--main-points", default=os.path.join(ROOT, "data", "eval", "eval_points.geojson"),
                    help="Tap chinh sau khi loc (10.801 diem) + .provenance.json")
    ap.add_argument("--scope", default=data_path("features/scope_mask_v3.tif"))
    ap.add_argument("--points-ref", default=data_path("labels/points_reference_keep6090.csv"))
    ap.add_argument("--classes", nargs="+", type=int, default=list(CLASSES))
    ap.add_argument("--blocks", default=os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44], help="cv_folds_s<n>.csv")
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--tables-dir", default=data_path("features/unified"),
                    help="Bang hop nhat (mo ta so diem co o theo luoi); '' = bo qua mo ta")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval", "aux_6090", "eval_points_6090.geojson"))
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY, help="Ranh gioi ghi vao provenance (sha256)")
    ap.add_argument("--allow-no-provenance", action="store_true", help="Chi cho test: tap chinh khong co provenance")
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    main(build_parser().parse_args())

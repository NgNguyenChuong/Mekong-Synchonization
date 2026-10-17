#!/usr/bin/env python
"""E7 (MO TA): Moran's I va tam semivariogram cua nhan tai diem danh gia, 6 bien x mua; quy tac NHAT_KY 2026-10-09 10:56:38.

Chay:  venv/Scripts/python.exe scripts/analyze_e7_moran.py [--out-dir KE_HOACH/ket-qua]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import spatial_autocorr as sa  # noqa: E402
from analyze_cv_family import FROZEN_MANIFESTS  # noqa: E402
from analyze_spatial_autocorr import point_xy  # noqa: E402
from run_experiments import git_state  # noqa: E402
from settings import data_path  # noqa: E402
from training.dot7_rules import file_sha256, guard_frozen, provenance_path, write_csv_atomic, write_provenance  # noqa: E402

VARIABLES = ["salinity", "ndwi", "rain_chirps", "dsr_mcd18", "t2m_era5", "rh_era5"]
CORE = ("salinity", "ndwi", "rain_chirps", "dsr_mcd18")  # bien hop le -> tap diem chung
N_COMMON = 10424
SRC_RES_KM = {"salinity": 0.03, "ndwi": 0.03, "rain_chirps": 5.0, "dsr_mcd18": 1.0, "t2m_era5": 9.0, "rh_era5": 9.0}
KEY = ["variable", "season"]
COLS = ["variable", "season", "n", "moran_I", "moran_p", "n_islands", "range_km", "ok", "reason", "nugget", "sill"]
SEASON_FILE = "dot7_e7_moran_bien_mua.csv"
SUMMARY_FILE = "dot7_e7_moran_bien.csv"
EVAL_POINTS = os.path.join(ROOT, "data", "eval", "eval_points.geojson")


def loi(msg):
    raise SystemExit(f"LOI: {msg}")


def label_path(t):
    if t == "salinity":
        return data_path("labels", "points_reference.csv")
    return data_path("labels", "dot7", f"points_reference_{t}.csv")


def load_label(t):
    d = pd.read_csv(label_path(t), dtype={"point_id": str}, usecols=["point_id", "season", f"ref_{t}"])
    return d.rename(columns={f"ref_{t}": "value"})


def point_sets(labels: dict, expected=N_COMMON) -> dict:
    """{bien: tap point_id}: giao CORE co nhan huu han >= 1 mua (phai = expected); nhiet/am giao them tap rieng."""
    finite = {t: set(d.loc[np.isfinite(d["value"]), "point_id"]) for t, d in labels.items()}
    common = set.intersection(*(finite[t] for t in CORE))
    print(f"tap diem chung ({'/'.join(CORE)}): {len(common)}", flush=True)
    if len(common) != expected:
        loi(f"tap diem chung {len(common)} != {expected}")
    return {t: common & finite[t] for t in labels}


def season_inputs(labels: dict, pts: dict) -> dict:
    """{(bien, mua): DataFrame(point_id, value)} chi diem trong tap cua bien va nhan huu han mua do."""
    out = {}
    for t, d in labels.items():
        d = d[d["point_id"].isin(pts[t]) & np.isfinite(d["value"])]
        for s, g in d.groupby("season"):
            out[(t, int(s))] = g[["point_id", "value"]]
    return out


def row_for(t, season, xy, v) -> dict:
    vg = sa.variogram_range(xy, v)
    mo = sa.moran_band(xy, v)
    return {"variable": t, "season": season, "n": vg["n"], "moran_I": mo["I"], "moran_p": mo["p_sim"],
            "n_islands": mo["n_islands"], "range_km": vg["range_m"] / 1000, "ok": vg["ok"], "reason": vg["reason"],
            "nugget": vg["nugget"], "sill": vg["sill"]}


def read_done(path) -> pd.DataFrame:
    """CSV ghi dan: trung khoa giong het -> khu (pitfall 44); trung khoa gia tri khac -> LOI."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    res = pd.read_csv(path).drop_duplicates()
    dup = res.duplicated(KEY, keep=False)
    if dup.any():
        loi(f"{int(dup.sum())} dong trung khoa (variable, season) gia tri KHAC, vd "
            f"{res.loc[dup, KEY].drop_duplicates().head(5).to_dict('records')}")
    return res


def check_rows(res: pd.DataFrame, expected_n: dict, complete: bool):
    """Dong (bien, mua) phai thuoc tap ky vong voi n khop so diem tinh lai (dong cu cua tap diem khac -> LOI)."""
    got = {(t, int(s)): int(n) for t, s, n in zip(res["variable"], res["season"], res["n"])}
    thieu = sorted(set(expected_n) - set(got)) if complete else []
    thua = sorted(set(got) - set(expected_n))
    lech = sorted(k for k in set(got) & set(expected_n) if got[k] != expected_n[k])
    if thieu or thua or lech:
        loi(f"ket qua khong khop: thieu {thieu[:5]} thua {thua[:5]} n lech {lech[:5]}")


def summarize(res: pd.DataFrame, n_set: dict) -> pd.DataFrame:
    rows = []
    for t in [v for v in VARIABLES if v in set(res["variable"])]:
        x = res[res["variable"] == t]
        r = x["range_km"].to_numpy(float)
        if np.isnan(r).any() or x["moran_I"].isna().any():
            loi(f"{t}: tam/Moran NaN trong ket qua (NaN khong phai 'vuot')")
        rows.append({"variable": t, "n_seasons": len(x), "n_tap_chung": n_set[t],
                     "n_points_median": float(np.median(x["n"])), "moran_median": float(np.median(x["moran_I"])),
                     "moran_min": float(x["moran_I"].min()), "moran_max": float(x["moran_I"].max()),
                     "range_median_km": float(np.median(r)), "n_range_inf": int(np.isinf(r).sum()),
                     "do_phan_giai_nguon_km": SRC_RES_KM[t]})
    return pd.DataFrame(rows)


def mem():
    try:
        import psutil

        m = psutil.Process().memory_info()
        return f"RSS {m.rss / 1e9:.2f} GB, dinh {getattr(m, 'peak_wset', m.rss) / 1e9:.2f} GB"
    except ImportError:
        return "RSS ?"


def main(a):
    season_path = os.path.join(a.out_dir, SEASON_FILE)
    summary_path = os.path.join(a.out_dir, SUMMARY_FILE)
    outs = [season_path, summary_path]
    guard_frozen(outs + [provenance_path(p) for p in outs], a.frozen_manifest)
    gs = git_state()  # luc bat dau: commit trong luc chay khong doi provenance
    os.makedirs(a.out_dir, exist_ok=True)
    for p in (summary_path, provenance_path(summary_path)):  # dung giua chung khong de lai tong hop cu
        if os.path.exists(p):
            os.remove(p)
    labels = {t: load_label(t) for t in VARIABLES}
    pts = point_sets(labels, a.expected_common)
    inputs = season_inputs(labels, pts)
    expected_n = {k: len(g) for k, g in inputs.items()}
    pxy = point_xy()
    done = read_done(season_path)
    check_rows(done, expected_n, complete=False)
    done_keys = set(zip(done["variable"], done["season"].astype(int)))
    t_start = time.time()
    for (t, s), g in sorted(inputs.items(), key=lambda kv: (VARIABLES.index(kv[0][0]), kv[0][1])):
        if (t, s) in done_keys:
            continue
        t0 = time.time()
        xy, v = sa.season_frame(pxy, g.assign(season=s), "value")[s]
        row = row_for(t, s, xy, v)
        pd.DataFrame([row], columns=COLS).to_csv(season_path, mode="a", header=not os.path.exists(season_path),
                                                 index=False)
        print(f"  {t} {s}: n {row['n']}, I {row['moran_I']:.3f}, tam {row['range_km']:.1f} km {row['reason']}"
              f" ({time.time() - t0:.0f} s; {mem()})", flush=True)
    print(f"tinh xong sau {time.time() - t_start:.0f} s", flush=True)
    res = read_done(season_path)
    check_rows(res, expected_n, complete=True)
    summ = summarize(res, {t: len(p) for t, p in pts.items()})
    write_csv_atomic(summ, summary_path)
    src = {os.path.basename(label_path(t)): file_sha256(label_path(t)) for t in VARIABLES}
    src["eval_points.geojson"] = file_sha256(EVAL_POINTS)
    info = dict(quy_tac_nhat_ky="2026-10-09 10:56:38 E7", vai_tro="mo_ta", sha_dau_vao=src,
                n_tap_chung={t: len(p) for t, p in pts.items()}, moran_band_m=sa.MORAN_BAND_M,
                moran_perm=sa.MORAN_PERM, moran_seed=sa.MORAN_SEED, maxlag_m=sa.MAXLAG_M, n_lags=sa.N_LAGS,
                **gs)
    write_provenance(season_path, **info)
    write_provenance(summary_path, sha_bien_mua=file_sha256(season_path), **info)
    reasons = res["reason"].fillna("").replace("", "ok").value_counts().to_dict()
    print(f"reason: {reasons}")
    with pd.option_context("display.width", 200):
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--expected-common", type=int, default=N_COMMON)
    ap.add_argument("--frozen-manifest", nargs="+", default=FROZEN_MANIFESTS)
    main(ap.parse_args())

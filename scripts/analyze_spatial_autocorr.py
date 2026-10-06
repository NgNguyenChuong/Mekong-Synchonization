#!/usr/bin/env python
"""Tu tuong quan khong gian cua phan du OOF HistGB va cua nhan tai diem (tuan 5 viec 1-2). Quy tac: src/spatial_autocorr.py.

Dau vao: artifacts/experiments/<prefix>__<luoi>__hist_gb__s<n>/cv/oof_points.csv (phan du = err), diem danh gia.
Chi dung diem CV (khoi khong giu rieng, mua != 2020) - giong tap cham CV.
Dau ra (--out-dir, mac dinh KE_HOACH/ket-qua):
  dot6_tu_tuong_quan_phan_du.csv  (luoi, cach chia, mua: tam, nugget, sill, ok, Moran I, p)  - ghi dan, chay tiep duoc
  dot6_tu_tuong_quan_ket_luan.csv (luoi, cach chia: trung vi/max tam, so mua vuot, VI PHAM)
  dot6_tu_tuong_quan_nhan.csv     (nhan tai diem theo mua: tam, Moran)
  dot6_tu_tuong_quan_huong.csv    (bat dang huong 45/135 do: phan du h3_res_7 s42 + nhan)
CHG-22 (3 trang thai): file KET LUAN cua che do dang chay bi XOA ngay khi bat dau va chi ghi lai (file tam ->
os.replace) khi moi buoc xong -> dung giua chung (LOI) khong de lai ket luan cu. Doc lai CSV ket qua: khoa
(grid, scheme, season) trung ma gia tri khac -> LOI; trung giong het -> khu; moi bo (luoi, cach chia) duoc yeu cau
phai du DUNG tap mua cua oof_points -> thieu/thua -> LOI, khong ghi ket luan.

Chay:  venv/Scripts/python.exe scripts/analyze_spatial_autocorr.py --prefix cv1
"""
import argparse
import os
import sys

import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from settings import data_path  # noqa: E402
from spatial_autocorr import BLOCK_KM, detrend_poly, detrend_surface, moran_band, season_frame, variogram_range, violates  # noqa: E402

GRIDS = ["h3_res_5", "h3_res_6", "h3_res_7", "latlon_0.0222deg", "latlon_0.0586deg", "latlon_0.1552deg",
         "s2_level_9", "s2_level_10", "s2_level_11", "s2_level_12", "square_utm_17087m", "square_utm_2441m",
         "square_utm_6458m"]


def point_xy():
    p = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson")).to_crs(32648)
    return pd.DataFrame({"point_id": p["point_id"].astype(str), "x": p.geometry.x, "y": p.geometry.y})


KEY = ["grid", "scheme", "season"]
CONCL_THUONG = ["dot6_tu_tuong_quan_ket_luan.csv", "dot6_tu_tuong_quan_nhan.csv", "dot6_tu_tuong_quan_huong.csv"]
CONCL_BO_XU_HUONG = ["dot6_tu_tuong_quan_bo_xu_huong_ket_luan.csv"]
CONCL_BO_XU_HUONG_XY = ["dot6_tu_tuong_quan_bo_xu_huong_xy_ket_luan.csv"]  # CHG-23 C


def xoa_ket_luan_cu(out_dir, names):
    """CHG-22: xoa file ket luan TRUOC khi tinh -> neu dung giua chung khong con ket luan cu de doc nham."""
    for n in names:
        f = os.path.join(out_dir, n)
        if os.path.exists(f):
            os.remove(f)
            print(f"  da xoa ket luan cu: {n}", flush=True)


def ghi_nguyen_tu(df, path):
    """Ghi ra file tam roi os.replace -> khong bao gio co file ket luan ghi do dang."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def mua_oof(d):
    """Tap mua cua mot oof_points; moi oof phai cung mot tap mua (kiem_cung_tap_mua)."""
    return set(d["season"].astype(int))


def kiem_cung_tap_mua(expected):
    """Moi oof_points duoc yeu cau phai cung mot tap mua (cv1: 12 mua) -> khac / rong -> LOI."""
    tap = {frozenset(v) for v in expected.values()}
    if len(tap) != 1 or not next(iter(tap)):
        raise SystemExit(f"LOI: oof_points khong cung mot tap mua (hoac rong): {[sorted(t) for t in tap]}")


def doc_ket_qua(path, expected):
    """Doc CSV ket qua ghi dan, kiem day du (CHG-22). expected: {(grid, scheme): tap mua cua oof}.

    Trung khoa giong het tung cot -> khu (ghi dan tat dinh, pitfall 44); trung khoa gia tri khac -> LOI;
    bo (grid, scheme) duoc yeu cau thieu / thua mua so voi oof -> LOI. Tra ve chi cac bo duoc yeu cau.
    """
    res = pd.read_csv(path).drop_duplicates()
    dup = res.duplicated(KEY, keep=False)
    if dup.any():
        k = res.loc[dup, KEY].drop_duplicates().head(5).to_dict("records")
        raise SystemExit(f"LOI: {int(dup.sum())} dong trung khoa (grid, scheme, season) nhung gia tri KHAC, vd {k}")
    loi = []
    for (g, s), want in expected.items():
        got = set(res.loc[(res["grid"] == g) & (res["scheme"] == s), "season"].astype(int))
        if got != want:
            loi.append(f"{g} s{s}: thieu {sorted(want - got)} thua {sorted(got - want)}")
    if loi:
        raise SystemExit(f"LOI: {len(loi)}/{len(expected)} bo khong du dung tap mua cua oof -> khong ghi ket luan:\n  "
                         + "\n  ".join(loi))
    keep = pd.Series([(g, s) in expected for g, s in zip(res["grid"], res["scheme"])], index=res.index)
    return res[keep]


def row_for(xy, v, **keys):
    vg = variogram_range(xy, v)
    mo = moran_band(xy, v)
    return {**keys, "n": vg["n"], "range_km": vg["range_m"] / 1000, "ok": vg["ok"], "reason": vg["reason"],
            "nugget": vg["nugget"], "sill": vg["sill"], "moran_I": mo["I"], "moran_p": mo["p_sim"],
            "moran_n_islands": mo["n_islands"]}


def point_dist_coast():
    """dist_coast_km tai pixel chua DIEM (river_distance_90m.tif, cung duong bo Sayre cua dac trung)."""
    import numpy as np
    import rasterio

    p = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson")).to_crs(32648)
    with rasterio.open(data_path("raw", "river", "river_distance_90m.tif")) as src:
        b = list(src.descriptions).index("dist_coast_km")
        v = np.array([x[b] for x in src.sample([(q.x, q.y) for q in p.geometry])])
    return pd.Series(v, index=p["point_id"].astype(str))


def main_detrend(a, mode="coast"):
    """CHG-20 tieu chi 1 (mode "coast"): bo xu huong bac 2 theo khoang cach bo cua diem; CHG-23 C (mode "xy"):
    bo mat xu huong bac 2 theo TOA DO diem (x, y met) - cach bo xu huong bo sung DUY NHAT. Semivariogram cung cau hinh."""
    pxy = point_xy()
    dc = point_dist_coast() if mode == "coast" else None
    exp = os.path.join(ROOT, "artifacts", "experiments")
    os.makedirs(a.out_dir, exist_ok=True)
    suf = "" if mode == "coast" else "_xy"
    xoa_ket_luan_cu(a.out_dir, CONCL_BO_XU_HUONG if mode == "coast" else CONCL_BO_XU_HUONG_XY)
    out_res = os.path.join(a.out_dir, f"dot6_tu_tuong_quan_bo_xu_huong{suf}.csv")
    done = pd.read_csv(out_res) if os.path.exists(out_res) else pd.DataFrame(columns=KEY)
    done_keys = set(zip(done["grid"], done["scheme"], done["season"]))
    expected = {}
    for g in GRIDS:
        for s in a.schemes:
            d = pd.read_csv(os.path.join(exp, f"{a.prefix}__{g}__hist_gb__s{s}", "cv", "oof_points.csv"),
                            dtype={"point_id": str}, usecols=["point_id", "season", "err"])
            expected[(g, s)] = mua_oof(d)
            if mode == "coast":
                d["dc"] = d["point_id"].map(dc)
                d["err_bo_xu_huong"] = d.groupby("season", group_keys=False).apply(
                    lambda x: pd.Series(detrend_poly(x["err"], x["dc"], 2), index=x.index), include_groups=False)
            else:  # CHG-23 C: mat xu huong bac 2 theo toa do, tung mua
                d = d.merge(pxy, on="point_id", how="left", validate="many_to_one")
                d["err_bo_xu_huong"] = d.groupby("season", group_keys=False).apply(
                    lambda x: pd.Series(detrend_surface(x["err"], x["x"], x["y"]), index=x.index), include_groups=False)
                d = d.drop(columns=["x", "y"])
            rows = []
            for season, (xy, v) in sorted(season_frame(pxy, d, "err_bo_xu_huong").items()):
                if (g, s, season) in done_keys:
                    continue
                vg = variogram_range(xy, v)
                rows.append({"grid": g, "scheme": s, "season": season, "n": vg["n"], "range_km": vg["range_m"] / 1000,
                             "ok": vg["ok"], "reason": vg["reason"], "nugget": vg["nugget"], "sill": vg["sill"]})
            if rows:
                pd.DataFrame(rows).to_csv(out_res, mode="a", header=not os.path.exists(out_res), index=False)
            print(f"  {g} s{s}: xong", flush=True)
    kiem_cung_tap_mua(expected)
    res = doc_ket_qua(out_res, expected)
    concl = []
    for (g, s), x in res.groupby(["grid", "scheme"]):
        v = violates(x["range_km"].to_numpy())
        concl.append({"grid": g, "scheme": s, **v, "n_fit_fail_or_bound": int((~x["ok"].astype(bool)).sum())})
    concl = pd.DataFrame(concl)
    ghi_nguyen_tu(concl, os.path.join(a.out_dir, f"dot6_tu_tuong_quan_bo_xu_huong{suf}_ket_luan.csv"))
    with pd.option_context("display.width", 200):
        print(concl.round(2).to_string(index=False))
    print("TIEU CHI 1 (bo xu huong) - VI PHAM (bat ky bo nao):", bool(concl["violates"].any()))


def main(a):
    coast, xy = getattr(a, "detrend_coast", False), getattr(a, "detrend_xy", False)  # Namespace cu khong co detrend_xy
    if coast and xy:
        raise SystemExit("LOI: chon MOT trong --detrend-coast / --detrend-xy")
    if coast:
        return main_detrend(a, "coast")
    if xy:
        return main_detrend(a, "xy")
    os.makedirs(a.out_dir, exist_ok=True)
    xoa_ket_luan_cu(a.out_dir, CONCL_THUONG)
    pxy = point_xy()
    exp = os.path.join(ROOT, "artifacts", "experiments")
    out_res = os.path.join(a.out_dir, "dot6_tu_tuong_quan_phan_du.csv")
    done = pd.read_csv(out_res) if os.path.exists(out_res) else pd.DataFrame(columns=KEY)
    done_keys = set(zip(done["grid"], done["scheme"], done["season"]))
    first_label = None
    expected = {}
    for g in GRIDS:
        for s in a.schemes:
            d = pd.read_csv(os.path.join(exp, f"{a.prefix}__{g}__hist_gb__s{s}", "cv", "oof_points.csv"),
                            dtype={"point_id": str}, usecols=["point_id", "season", "err", "y_ref"])
            expected[(g, s)] = mua_oof(d)
            if first_label is None:
                first_label = d[["point_id", "season", "y_ref"]]
            rows = []
            for season, (xy, v) in sorted(season_frame(pxy, d, "err").items()):
                if (g, s, season) in done_keys:
                    continue
                rows.append(row_for(xy, v, grid=g, scheme=s, season=season))
            if rows:
                pd.DataFrame(rows).to_csv(out_res, mode="a", header=not os.path.exists(out_res), index=False)
            print(f"  {g} s{s}: xong", flush=True)
    kiem_cung_tap_mua(expected)
    res = doc_ket_qua(out_res, expected)
    concl = []
    for (g, s), x in res.groupby(["grid", "scheme"]):
        v = violates(x["range_km"].to_numpy())
        over = sorted(x.loc[~(x["range_km"] <= BLOCK_KM), "season"].astype(int).tolist())
        concl.append({"grid": g, "scheme": s, **v, "seasons_over": " ".join(map(str, over)),
                      "n_fit_fail": int((~x["ok"].astype(bool)).sum()), "moran_I_median": x["moran_I"].median()})
    concl = pd.DataFrame(concl)
    ghi_nguyen_tu(concl, os.path.join(a.out_dir, "dot6_tu_tuong_quan_ket_luan.csv"))
    # nhan tai diem (giong moi luoi) + bat dang huong
    lab = [row_for(xy, v, season=season) for season, (xy, v) in sorted(season_frame(pxy, first_label, "y_ref").items())]
    ghi_nguyen_tu(pd.DataFrame(lab), os.path.join(a.out_dir, "dot6_tu_tuong_quan_nhan.csv"))
    d7 = pd.read_csv(os.path.join(exp, f"{a.prefix}__h3_res_7__hist_gb__s42", "cv", "oof_points.csv"),
                     dtype={"point_id": str}, usecols=["point_id", "season", "err", "y_ref"])
    hrows = []
    for col, ten in (("err", "phan_du_h3_res_7_s42"), ("y_ref", "nhan")):
        for season, (xy, v) in sorted(season_frame(pxy, d7, col).items()):
            for az in (45.0, 135.0):
                r = variogram_range(xy, v, azimuth=az)
                hrows.append({"doi_tuong": ten, "season": season, "azimuth": az, "range_km": r["range_m"] / 1000,
                              "ok": r["ok"], "reason": r["reason"]})
    ghi_nguyen_tu(pd.DataFrame(hrows), os.path.join(a.out_dir, "dot6_tu_tuong_quan_huong.csv"))
    with pd.option_context("display.width", 200):
        print(concl.round(2).to_string(index=False))
    print("VI PHAM (bat ky bo nao):", bool(concl["violates"].any()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--detrend-xy", action="store_true",
                    help="CHG-23 C: bo mat xu huong bac 2 theo toa do diem truoc semivariogram (cach bo sung DUY NHAT)")
    ap.add_argument("--detrend-coast", action="store_true",
                    help="CHG-20 tieu chi 1: bo xu huong bac 2 theo khoang cach bo cua diem truoc semivariogram")
    main(ap.parse_args())

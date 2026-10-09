"""E4 Dot 7: ap 2 quy tac da chot TRUOC (NHAT_KY 2026-10-06, muc "AN CHOT sau phan bien") cho nhan moi.

1) Buc xa mua 2020 (MCD18A1 thieu 03-31/12/2019): o 12 mua con lai, Pearson giua TB mua CO va KHONG co thang 12
   (band dsr_mean vs dsr_mean_no_dec) tai 10.454 diem cham duoc (gia tri pixel nguon chua diem).
   >= 0,95 o MOI mua (12/12) -> (a) giu mua 2020 (TB ngay co du lieu + co "thieu thang 12"); khong -> (b) NaN.
   Do lech muc (TB khong-12 tru TB co-12) bao kem, KHONG dung de quyet.
2) Mua - do phan giai hieu dung CHIRPS v3: khi hau nen = TB 13 mua (ke ca 2020); di thuong = mua - khi hau nen;
   ty le = SS trong o cua di thuong / SS tong cua di thuong, tren pixel CHIRPS (tam pixel trong pham vi), o tung
   luoi tho (h3_res_5, s2_level_9, latlon_0.1552deg, square_utm_17087m), tung mua; trung vi qua mua va luoi.
   Trung vi < 5% -> mua la diem tham chieu (rong) o MOI muc; >= 5% -> giu diem mua muc tho.

Ba trang thai (CHG-22): thieu file/band/diem sai so luong/khong du du lieu de tinh -> LOI (thoat ma 2), khong
coi la dat hay khong dat.

Chay: python scripts/analyze_dot7_label_rules.py [--out KE_HOACH/ket-qua]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import settings  # noqa: E402

SEASONS = tuple(range(2014, 2027))
SEASON_MISSING_DEC = 2020
N_POINTS_EXPECTED = 10454
PEARSON_MIN = 0.95                      # chot truoc
RAIN_RATIO_MAX = 0.05                   # chot truoc: trung vi < 5% -> diem tham chieu
COARSE_GRIDS = ("h3_res_5", "s2_level_9", "latlon_0.1552deg", "square_utm_17087m")


def loi(msg):
    print(f"LOI: {msg}", file=sys.stderr)
    sys.exit(2)


# ------------------------------------------------------------------ ham thuan (co test)
def within_ss_ratio(values, groups) -> float:
    """SS trong nhom / SS tong (trong so deu theo phan tu). Gia tri khong huu han hoac tong SS = 0 -> ValueError."""
    v = np.asarray(values, dtype="float64")
    g = np.asarray(groups)
    if v.shape != g.shape or v.size < 2:
        raise ValueError("values/groups khong cung kich thuoc hoac < 2 phan tu")
    if not np.all(np.isfinite(v)):
        raise ValueError("co gia tri khong huu han")
    ss_tot = float(((v - v.mean()) ** 2).sum())
    if ss_tot <= 0:
        raise ValueError("tong SS = 0")
    s = pd.Series(v).groupby(pd.Series(g).values)
    ss_within = float((s.transform(lambda x: x - x.mean()) ** 2).sum())
    return ss_within / ss_tot


def pearson(a, b) -> tuple[float, int]:
    """Pearson tren cap cung huu han; < 30 cap hoac phuong sai 0 -> ValueError."""
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 30:
        raise ValueError(f"chi {n} cap huu han")
    x, y = a[ok], b[ok]
    if x.std() == 0 or y.std() == 0:
        raise ValueError("phuong sai 0")
    return float(np.corrcoef(x, y)[0, 1]), n


def radiation_decision(r_by_season: dict) -> str:
    """(a) neu MOI mua (tru 2020) co r >= PEARSON_MIN; nguoc lai (b). Thieu mua / r NaN -> ValueError."""
    want = [s for s in SEASONS if s != SEASON_MISSING_DEC]
    miss = [s for s in want if s not in r_by_season or not np.isfinite(r_by_season[s])]
    if miss:
        raise ValueError(f"thieu r cho mua {miss}")
    return "a" if all(r_by_season[s] >= PEARSON_MIN for s in want) else "b"


def rain_decision(ratios) -> tuple[float, str]:
    r = np.asarray(ratios, dtype="float64")
    if r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("ty le rong hoac khong huu han")
    med = float(np.median(r))
    return med, ("diem_tham_chieu" if med < RAIN_RATIO_MAX else "giu_muc_tho")


# ------------------------------------------------------------------ du lieu
def load_points():
    import geopandas as gpd
    pts = gpd.read_file(os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    ref = pd.read_csv(settings.data_path("labels", "points_reference.csv"), dtype={"point_id": str},
                      usecols=["point_id", "ref_salinity"])
    ok = set(ref.loc[ref["ref_salinity"].notna(), "point_id"])
    pts = pts[pts["point_id"].astype(str).isin(ok)].reset_index(drop=True)
    if len(pts) != N_POINTS_EXPECTED:
        loi(f"so diem cham duoc {len(pts)} != {N_POINTS_EXPECTED}")
    return pts


def sample(path, bands, pts):
    """Gia tri pixel nguon CHUA diem (khong noi suy). Diem ngoai raster -> LOI."""
    import rasterio
    with rasterio.open(path) as ds:
        names = list(ds.descriptions)
        if any(b not in names for b in bands):
            loi(f"{path}: thieu band {set(bands) - set(names)}")
        xy = pts.to_crs(ds.crs)
        rows, cols = rasterio.transform.rowcol(ds.transform, xy.geometry.x.values, xy.geometry.y.values)
        rows, cols = np.asarray(rows), np.asarray(cols)
        if (rows < 0).any() or (cols < 0).any() or (rows >= ds.height).any() or (cols >= ds.width).any():
            loi(f"{path}: co diem nam ngoai raster")
        return {b: ds.read(names.index(b) + 1).astype("float64")[rows, cols] for b in bands}


def run_radiation(pts):
    rows = []
    for s in SEASONS:
        p = settings.data_path("raw", "mcd18a1", f"mcd18a1_dsr_{s}.tif")
        if not os.path.exists(p):
            loi(f"thieu {p}")
        import met_labels as ml
        import rasterio
        with rasterio.open(p) as _ds:
            if not ml.is_modis_sphere(_ds.crs):
                loi(f"{p}: CRS khong phai sinusoidal hinh cau MODIS (doc se lech ~15 km)")
        v = sample(p, ("dsr_mean", "dsr_mean_no_dec", "n_days_valid"), pts)
        nan_pt = int((~np.isfinite(v["dsr_mean"])).sum())
        row = {"season": s, "n_diem": len(pts), "n_diem_khong_gia_tri": nan_pt,
               "n_days_valid_min_tai_diem": float(np.nanmin(v["n_days_valid"])),
               "n_days_valid_trung_vi_tai_diem": float(np.nanmedian(v["n_days_valid"]))}
        if s != SEASON_MISSING_DEC:
            try:
                r, n = pearson(v["dsr_mean"], v["dsr_mean_no_dec"])
            except ValueError as e:
                loi(f"buc xa mua {s}: {e}")
            d = v["dsr_mean_no_dec"] - v["dsr_mean"]
            row.update({"pearson_co_vs_khong_t12": r, "n_cap": n, "lech_muc_TB_Wm2": float(np.nanmean(d)),
                        "lech_muc_TB_pct": float(np.nanmean(d / v["dsr_mean"]) * 100)})
        rows.append(row)
    df = pd.DataFrame(rows)
    dec = radiation_decision({int(r.season): r.pearson_co_vs_khong_t12 for r in df.itertuples()
                              if r.season != SEASON_MISSING_DEC})
    return df, dec


def run_rain():
    import geopandas as gpd
    import met_labels as ml
    from preprocessing import CANONICAL_BOUNDARY as boundary
    stack, prof = [], None
    for s in SEASONS:
        p = settings.data_path("raw", "chirps3", f"chirps3_rain_{s}.tif")
        if not os.path.exists(p):
            loi(f"thieu {p}")
        b, pr = ml.read_bands(p)
        if prof is not None and (pr["transform"] != prof["transform"] or pr["crs"] != prof["crs"]):
            loi(f"{p}: luoi pixel khac mua truoc")
        prof = pr
        stack.append(b["rain_sum"])
    stack = np.stack(stack)                       # (mua, h, w)
    center, _ = ml.region_masks(boundary, prof)
    # Pixel bi che o MOI mua (mat na bien/dao co dinh cua CHIRPS, n_pentad_valid = 0) -> loai + dem;
    # thieu chi o MOT SO mua -> LOI (khong tu bo).
    fin = np.isfinite(stack)
    perm = center & ~fin.any(axis=0)
    if (center & ~perm & ~fin.all(axis=0)).any():
        loi("CHIRPS co pixel trong pham vi thieu o mot so mua")
    n_px_che = int(perm.sum())
    center = center & ~perm
    vals = stack[:, center]                       # (mua, n_px)
    clim = vals.mean(axis=0)
    anom = vals - clim
    rr, cc = np.nonzero(center)
    xs, ys = prof["transform"] * (cc + 0.5, rr + 0.5)
    px = gpd.GeoDataFrame({"px": np.arange(len(xs))}, geometry=gpd.points_from_xy(xs, ys), crs=prof["crs"])
    rows = []
    for g in COARSE_GRIDS:
        grid = gpd.read_file(os.path.join(ROOT, "data", "grids", f"{g}.geojson")).to_crs(prof["crs"])
        j = gpd.sjoin(px, grid[["cell_id", "geometry"]], how="left", predicate="within")
        if j.index.duplicated().any():
            loi(f"{g}: tam pixel roi vao > 1 o")
        cell = j.sort_index()["cell_id"]
        inside = cell.notna().values
        npc = cell[inside].value_counts()
        for k, s in enumerate(SEASONS):
            try:
                ratio = within_ss_ratio(anom[k, inside], cell[inside].values)
            except ValueError as e:
                loi(f"mua {g} {s}: {e}")
            rows.append({"grid": g, "season": s, "ty_le_trong_o": ratio, "n_px": int(inside.sum()),
                         "n_px_ngoai_o": int((~inside).sum()), "n_px_che_moi_mua_da_loai": n_px_che, "n_o": int(npc.size),
                         "n_o_1_pixel": int((npc == 1).sum()), "px_moi_o_trung_vi": float(npc.median())})
    df = pd.DataFrame(rows)
    med, dec = rain_decision(df["ty_le_trong_o"].values)
    return df, med, dec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pts = load_points()
    rad, rdec = run_radiation(pts)
    rain, med, wdec = run_rain()
    rad.to_csv(os.path.join(a.out, "dot7_e4_buc_xa_2020.csv"), index=False)
    rain.to_csv(os.path.join(a.out, "dot7_e4_mua_do_phan_giai.csv"), index=False)
    pd.set_option("display.width", 200)
    print(rad.to_string(index=False))
    print(f"\nBUC XA 2020: r min (12 mua) = {rad['pearson_co_vs_khong_t12'].min():.4f} -> phuong an ({rdec})")
    print(rain.groupby("grid")["ty_le_trong_o"].describe().to_string())
    print(f"\nMUA: trung vi ty le trong o = {med:.4f} ({med * 100:.2f}%) -> {wdec}")


if __name__ == "__main__":
    main()

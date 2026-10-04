#!/usr/bin/env python
"""Bang thuy van theo mua kho 2014-2026 (P4) - bien cap vung, dung chung cho moi o cua moi luoi.

Mua kho Y = 01/11/(Y-1) .. 29/04/Y (src/seasons.py). Mot dong moi mua. Khong lap so lieu thieu.

Cot:
  season
  {tch,cdo}_wl_mean       : trung binh muc nuoc ngay (m, theo file MRC) tren cac ngay CO so lieu trong mua.
                            MRC: 0.0 goc = thieu (da doi thanh NaN o fetch_mrc_water_level.py).
  {tch,cdo}_wl_n_days     : so ngay co so lieu trong mua.
  {tch,cdo}_wl_frac_days  : n_days / season_days(Y) (181 ngay o mua co 29/02, 180 ngay o mua khac).
  {tch,cdo}_wl_ok         : frac_days >= 0.8.
                            Mua co `_ok = False` -> MOI cot gia tri cua tram do (`_mean` va cot cuc tri) = NaN
                            (An chot: MRC 2015 chi co 36-37 ngay, toan thang 3-4 -> NaN). n_days/frac_days/ok
                            van ghi de biet ly do. Dong mua van giu (khong xoa dong).
  tch_wl_p20c             : phan vi 20% muc nuoc ngay Tan Chau CO XU LY CHAN (An duyet cau B, 2026-10-03;
                            hydro_season.censored_season_stats). Chan CA chuoi o 0 (gia tri < 0 -> 0, moi
                            nam: tu 2021 co 13-19 ngay am/mua, truoc 2021 so lieu thap bi ghi 0 roi thanh NaN).
                            Ngay thieu "bi chan" = doan NaN <= 8 ngay co hai dau (ngay do lien ke, xet ca
                            ngoai mua) < 0,3 m -> coi la duoi moi gia tri do (-inf), phan vi method "lower";
                            ngay thieu khac bo qua. Cuc tri thap = nuoc thuong nguon yeu.
  tch_wl_n_censored       : so ngay bi chan (ngay thieu bi chan + ngay do <= 0) trong mua. Ty le
                            n_censored / (ngay do + ngay thieu bi chan) >= 20% -> canh bao.
  tch_wl_days_le_0_3      : so ngay <= 0,3 m (ngay do + ngay thieu bi chan) - do nhay, khong phu thuoc
                            gia tri duoi nguong. Mua `_ok = False` -> p20c va days_le_0_3 = NaN.
                            (Bo tch_wl_p10, tch_wl_min7 theo cau B.)
  zos_mean               : CMEMS `zos` (m). Moi ngay lay trung binh vung (trong so cos vi do) tren MOI
                            pixel bien co gia tri trong file (vung 104,4-107,2 E x 8,4-10,8 N); roi trung
                            binh cac ngay trong mua.
  zos_coast_mean          : nhu zos_mean nhung chi tren pixel bien co TAM cach ranh gioi 13 tinh
                            (preprocessing.CANONICAL_BOUNDARY, do trong EPSG:32648) <= 30 km
                            (khoang cach toi da giac ranh gioi; tam nam trong ranh gioi -> 0 km, van tinh).
  zos_coast_p90           : phan vi 90% cua chuoi trung binh vung ven bo theo ngay (cung tap pixel voi
                            zos_coast_mean) tren cac ngay co so lieu trong mua. Cuc tri cao = bien day vao.
  zos_n_days, zos_frac_days, zos_ok : nhu tren, cho chuoi trung binh vung (ngay nao thieu bat ky pixel
                            nao trong tap -> ca ngay coi la thieu). zos_ok = False -> zos_mean, zos_coast_mean,
                            zos_coast_p90 = NaN (cung quy tac voi MRC; ven bo la tap con cua toan vung nen
                            khong thieu nhieu ngay hon).

Chay:  python scripts/build_hydro_season.py [--mrc A:/Dataset_NCKH/mrc/mrc_water_level_daily.csv]
           [--cmems A:/Dataset_NCKH/cmems/cmems_zos_daily_mekong_1999_2026.nc]
           [--out A:/Dataset_NCKH/features/hydro_season_2014_2026.csv] [--years 2014 2026] [--coast-km 30]
           [--censor-runs-within-season]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hydro_season import (  # noqa: E402
    censored_season_stats, coastal_pixel_mask, regional_daily_mean, season_stats)

STATIONS = {"TCH": "tch_wl", "CDO": "cdo_wl"}  # Tan Chau, Chau Doc
# Cuc tri thap co xu ly chan theo tram (xam nhap man do nuoc thuong nguon thap quyet dinh, khong phai
# trung binh). Tham so cua hydro_season.censored_season_stats; An duyet cau B 2026-10-03.
STATION_CENSORED = {"TCH": {"q": 0.20, "suffix": "p20c", "floor": 0.0, "max_gap": 8, "edge_below": 0.3,
                            "low_thresh": 0.3, "warn_frac": 0.20}}
ZOS_COAST_QUANTILES = {"p90": 0.90}


def mrc_season_table(mrc_df, years, stations=STATIONS, censored=STATION_CENSORED,
                     nan_if_not_ok=True, runs_within_season=False) -> pd.DataFrame:
    """Bang mua cho cac tram MRC tu file dang dai (station, date, value).

    Mac dinh mua `_ok = False` -> moi cot gia tri cua tram do = NaN (xem docstring dau file).
    Doan ngay thieu bi chan xet tren TOAN chuoi cua tram (khong cat theo mua) tru khi runs_within_season.
    """
    out = pd.DataFrame({"season": list(years)})
    for st, prefix in stations.items():
        d = mrc_df[mrc_df["station"] == st]
        if d.empty:
            raise ValueError(f"Khong co du lieu tram {st} trong file MRC")
        out = out.merge(season_stats(d["date"], d["value"], years, prefix, nan_if_not_ok=nan_if_not_ok),
                        on="season")
        if st in censored:
            out = out.merge(censored_season_stats(d["date"], d["value"], years, prefix,
                                                  nan_if_not_ok=nan_if_not_ok,
                                                  runs_within_season=runs_within_season, **censored[st]),
                            on="season")
    return out


def zos_season_table(times, zos, lat, lon, boundary_geom, years, coast_km=30.0):
    """Bang mua cho CMEMS zos: trung binh vung toan bo pixel bien va dai ven bo <= coast_km."""
    zos = np.asarray(zos, dtype="float64")
    sea = np.isfinite(zos).any(axis=0)
    coast = sea & coastal_pixel_mask(lat, lon, boundary_geom, coast_km)
    if not coast.any():
        raise ValueError(f"Khong co pixel bien nao cach ranh gioi <= {coast_km} km")
    full = season_stats(times, regional_daily_mean(zos, lat, sea), years, "zos", nan_if_not_ok=True)
    near = season_stats(times, regional_daily_mean(zos, lat, coast), years, "zos_coast",
                        quantiles=ZOS_COAST_QUANTILES, nan_if_not_ok=True)
    qcols = [f"zos_coast_{k}" for k in ZOS_COAST_QUANTILES]
    out = full.merge(near[["season", "zos_coast_mean", *qcols]], on="season")
    # Ven bo la tap con pixel cua toan vung -> ngay du toan vung thi du ven bo; zos_ok (toan vung) chat hon
    out.loc[~out["zos_ok"], ["zos_coast_mean", *qcols]] = np.nan
    cols = ["season", "zos_mean", "zos_coast_mean", *qcols, "zos_n_days", "zos_frac_days", "zos_ok"]
    return out[cols], int(sea.sum()), int(coast.sum())


def main(a):
    import geopandas as gpd
    import xarray as xr
    from shapely.ops import unary_union

    years = list(range(a.years[0], a.years[1] + 1))
    mrc = pd.read_csv(a.mrc, parse_dates=["date"])
    mrc_tab = mrc_season_table(mrc, years, runs_within_season=a.censor_runs_within_season)
    print(f"MRC: {a.mrc} ({len(mrc)} dong)", flush=True)

    boundary = unary_union(gpd.read_file(a.boundary).to_crs(4326).geometry)
    with xr.open_dataset(a.cmems) as ds:
        z = ds["zos"]
        zos_tab, n_sea, n_coast = zos_season_table(
            ds["time"].values, z.values, ds[z.dims[1]].values, ds[z.dims[2]].values,
            boundary, years, a.coast_km)
    print(f"CMEMS: {a.cmems}; pixel bien {n_sea}, pixel <= {a.coast_km:g} km ranh gioi {n_coast}", flush=True)

    tab = mrc_tab.merge(zos_tab, on="season", how="outer").sort_values("season")
    assert tab["season"].is_unique and len(tab) == len(years)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tab.to_csv(a.out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(tab.to_string(index=False))
    # Cuc tri co them thong tin so voi trung binh khong? (Pearson tren cac mua ca hai cot co gia tri)
    for m, e in [("tch_wl_mean", "tch_wl_p20c"), ("tch_wl_p20c", "tch_wl_days_le_0_3"),
                 ("zos_coast_mean", "zos_coast_p90")]:
        sub = tab[[m, e]].dropna()
        print(f"r({m}, {e}) = {sub[m].corr(sub[e]):.3f} (n = {len(sub)} mua)", flush=True)
    print(f"Da ghi: {a.out}", flush=True)


if __name__ == "__main__":
    from preprocessing import CANONICAL_BOUNDARY  # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mrc", default="A:/Dataset_NCKH/mrc/mrc_water_level_daily.csv")
    ap.add_argument("--cmems", default="A:/Dataset_NCKH/cmems/cmems_zos_daily_mekong_1999_2026.nc")
    ap.add_argument("--boundary", default=CANONICAL_BOUNDARY)
    ap.add_argument("--out", default="A:/Dataset_NCKH/features/hydro_season_2014_2026.csv")
    ap.add_argument("--years", nargs=2, type=int, default=[2014, 2026], metavar=("TU", "DEN"))
    ap.add_argument("--coast-km", type=float, default=30.0)
    ap.add_argument("--censor-runs-within-season", action="store_true",
                    help="Xet doan ngay thieu bi chan chi trong cua so mua (doan cham 29/04 khong chan). "
                         "Mac dinh: dau thuc cua doan, ke ca ngoai mua.")
    main(ap.parse_args())

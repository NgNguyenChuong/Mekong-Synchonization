#!/usr/bin/env python
"""Tai muc nuoc ngay tu MRC FFW (T3-V2) va chuan hoa thanh chuoi ngay theo lich that.

Cau truc phan hoi MRC (kiem tra 2026-09-30):
- Mua kho (`fetchdry_new.php`): cot "YYYY.yy" = mua tu 01/11/YYYY den 31/05/YYYY+1.
  `date_gmt` chi la khung mau cua mua hien tai, nen ngay thang 11-12 thuoc nam YYYY,
  thang 1-5 thuoc nam YYYY+1. Khung mau co dong "29/02" ca khi nam mau khong nhuan
  -> chi giu 29/02 o nam nhuan that.
- Mua lu (`fetchwet_new.php`): cot "YYYY" = 01/06 den 31/10 cua nam YYYY.
- Gia tri 0.0 dong vai tro "thieu du lieu" -> NaN. Ngay sau hom nay -> bo.

Chay:  python scripts/fetch_mrc_water_level.py [--stations TCH CDO] [--out data/raw/mrc/mrc_water_level_daily.csv]
"""
import argparse
import os
import re
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "component"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DRY_COL = re.compile(r"^(\d{4})\.\d{2}$")
WET_COL = re.compile(r"^(\d{4})$")


def _month_day(date_gmt):
    # Khong parse ca chuoi: khung mau co the chua "2026-02-29" (ngay khong ton tai).
    _, m, d = str(date_gmt).split("-")
    return int(m), int(d)


def _make_date(year, month, day):
    try:
        return pd.Timestamp(year, month, day)
    except ValueError:  # 29/02 o nam khong nhuan
        return None


def dry_to_long(rows):
    """Chuyen bang mua kho (cot = mua) thanh (date, value) theo lich that."""
    out = []
    for row in rows:
        month, day = _month_day(row["date_gmt"])
        for col, val in row.items():
            m = DRY_COL.match(col)
            if not m:
                continue
            start = int(m.group(1))
            ts = _make_date(start if month >= 11 else start + 1, month, day)
            if ts is not None:
                out.append((ts, val, "dry", col))
    return out


def wet_to_long(rows):
    """Chuyen bang mua lu (cot = nam) thanh (date, value)."""
    out = []
    for row in rows:
        month, day = _month_day(row["date_gmt"])
        for col, val in row.items():
            m = WET_COL.match(col)
            ts = _make_date(int(m.group(1)), month, day) if m else None
            if ts is not None:
                out.append((ts, val, "wet", col))
    return out


def normalize(records, station, today=None):
    today = pd.Timestamp(today or date.today())
    df = pd.DataFrame(records, columns=["date", "value", "season", "source_col"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.loc[df["value"] == 0, "value"] = np.nan
    df = df[df["date"] <= today].copy()
    df["station"] = station
    # Mot ngay chi thuoc mot mua; neu trung (khong mong doi) giu ban ghi co gia tri.
    df = df.sort_values(["date", "value"], na_position="last").drop_duplicates("date", keep="first")
    return df[["station", "date", "value", "season", "source_col"]].reset_index(drop=True)


def fetch_station(station):
    from getWaterLevel import BASE_DRY_URL, BASE_WET_URL, fetch_water_level
    records = dry_to_long(fetch_water_level(station, BASE_DRY_URL))
    records += wet_to_long(fetch_water_level(station, BASE_WET_URL))
    return normalize(records, station)


def coverage(df):
    """Ty le ngay co gia tri theo mua kho (nhan nam N = 01/11/N-1 .. 30/04/N)."""
    d = df.copy()
    d["label_year"] = np.where(d["date"].dt.month >= 11, d["date"].dt.year + 1, d["date"].dt.year)
    d = d[d["date"].dt.month.isin([11, 12, 1, 2, 3, 4])]
    g = d.groupby(["station", "label_year"])["value"]
    return pd.DataFrame({"days": g.size(), "valid": g.count()}).assign(
        valid_pct=lambda x: (100 * x["valid"] / x["days"]).round(1)).reset_index()


def main(args):
    frames = []
    for st in args.stations:
        df = fetch_station(st)
        print(f"{st}: {len(df)} ngay, {df['value'].notna().sum()} co gia tri, "
              f"{df['date'].min().date()} .. {df['date'].max().date()}")
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    allf.to_csv(args.out, index=False, date_format="%Y-%m-%d")
    cov = coverage(allf)
    cov_path = args.out.replace(".csv", "_coverage_dry.csv")
    cov.to_csv(cov_path, index=False)
    print(cov.pivot(index="label_year", columns="station", values="valid_pct").to_string())
    print(f"Da ghi: {args.out}\n        {cov_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stations", nargs="+", default=["TCH", "CDO"])
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw", "mrc", "mrc_water_level_daily.csv"))
    main(ap.parse_args())

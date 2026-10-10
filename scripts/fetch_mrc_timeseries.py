#!/usr/bin/env python
"""Tai muc nuoc thu cong 6 tram MRC (time-series API) mua kho 01/11-30/04, mua 2014-2026 -> <DATA_ROOT>/mrc_ts/<tram>.csv
(datetime gio dia phuong +07:00, value m) + provenance; mua API loi -> lay tu --fallback-csv va ghi nguon.
Chay:  venv/Scripts/python.exe scripts/fetch_mrc_timeseries.py [--stations Tan_Chau ...] [--fallback-csv <csv>] [--force]
"""
import argparse
import datetime
import json
import os
import sys
import time
from http.client import HTTPException
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from seasons import season_of  # noqa: E402
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
from training.dot7_rules import file_sha256, provenance_path, write_provenance  # noqa: E402
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = "https://timeseries.api.mrcmekong.org/api/v1/ts/data/timeSeriesCorrectedData/"
UA = "Mekong-NCKH-research/1.0 (python urllib)"
# ten file: (stationName MRC, uniqueId chuoi "Water Level.Manual", lon, lat) - theo danh sach chuoi cua API
STATIONS = {
    "Tan_Chau": ("Tan Chau", "3bd17f3a15804c76bbad950867370528", 105.2480164, 10.80062008),
    "Chau_Doc": ("Chau Doc", "df56e8456a7b4fd485c1b0e71dafdf53", 105.1335068, 10.7052803),
    "Vam_Nao": ("Vam Nao", "23b97124ade04fea8597cd8fa269dc18", 105.3633728, 10.57865047),
    "My_Thuan": ("My Thuan", "cd6c69a7c33a4440aeb2673d5e7babf1", 105.9263229, 10.27532005),
    "Can_Tho": ("Can Tho", "b8706a811eba406b872601dcbad8e9a0", 105.7871389, 10.05288889),
    "Vam_Kenh": ("Vam Kenh", "54d4a5656582438f9cf2256b4e7ee6d7", 106.7371368, 10.27429962),
}
SEASONS = list(range(2014, 2027))
MIN_DAYS = 150
OFFSET = "+07:00"
NET_ERRORS = (URLError, TimeoutError, ConnectionError, HTTPException, json.JSONDecodeError)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def season_url(uid, s):
    # API nhan gio UTC: 01/11 00:00 +07 = 31/10 17:00Z; het 30/04 gio dia phuong = 30/04 17:00Z
    return f"{API}{uid}?sd={s - 1}-10-31T17:00:00.000Z&ed={s}-04-30T17:00:00.000Z"


def get_json(url, tries=4):
    for k in range(tries):
        try:
            with urlopen(Request(url, headers={"User-Agent": UA}), timeout=120) as r:
                return json.load(r)
        except NET_ERRORS as exc:
            if k == tries - 1:
                raise
            log(f"  thu lai {k + 1}/{tries - 1} sau loi {exc!r}")
            time.sleep(5 * 2 ** k)


def window(df, s):
    """Giu dong co ngay dia phuong trong [01/11/(s-1), 30/04/s]; gio phai cung mui +07:00."""
    if not df["datetime"].str.endswith(OFFSET).all():
        raise ValueError(f"co thoi diem khong theo mui {OFFSET}")
    day = df["datetime"].str[:10]
    return df[(day >= f"{s - 1}-11-01") & (day <= f"{s}-04-30")]


def api_points(d) -> pd.DataFrame:
    rows = [(p["Timestamp"], p["Value"]["Numeric"]) for p in d.get("Points", []) if p["Value"]["Numeric"] is not None]
    return pd.DataFrame(rows, columns=["datetime", "value"])


def season_counts(df) -> dict:
    """{mua: (so dong, so ngay)} theo season_of (30/04 khong thuoc mua)."""
    day = pd.to_datetime(df["datetime"].str[:10])
    s = season_of(day)
    g = pd.DataFrame({"s": s.to_numpy(), "d": day.to_numpy()}).dropna().groupby("s")
    return {int(k): (int(v), int(n)) for k, v, n in zip(g.size().index, g.size(), g["d"].nunique())}


def fetch_station(key, a, fallback):
    name, uid, lon, lat = STATIONS[key]
    parts, info = [], {}
    for s in SEASONS:
        url = season_url(uid, s)
        try:
            d = get_json(url)
            df, src, notes = window(api_points(d), s), "api", d.get("Notes", [])
        except NET_ERRORS as exc:
            if fallback is None:
                raise SystemExit(f"[LOI] {key} mua {s}: API loi {exc!r} va khong co --fallback-csv")
            fb = fallback[(fallback["station"] == name) & (fallback["season"] == s)]
            df, src, notes = window(fb.rename(columns={"ts": "datetime", "wl": "value"})[["datetime", "value"]], s), \
                "fallback", []
            log(f"  {key} mua {s}: API loi {exc!r} -> du phong ({len(df)} dong)")
        parts.append(df)
        info[str(s)] = {"source": src, "url": url, "n_rows_window": len(df), "notes": notes}
        time.sleep(a.sleep)
    out = pd.concat(parts, ignore_index=True)
    dup = out[out.duplicated("datetime", keep=False)]
    if (dup.groupby("datetime")["value"].nunique() > 1).any():
        raise SystemExit(f"[LOI] {key}: cung thoi diem co hai gia tri khac nhau")
    dup = int(out.duplicated("datetime").sum())
    out = out.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    return out, info, dup, (name, uid, lon, lat)


def check(counts, min_days=MIN_DAYS):
    """Danh sach mua thieu hoac < min_days ngay co so lieu."""
    return [s for s in SEASONS if counts.get(s, (0, 0))[1] < min_days]


def done(csv, keys):
    pp = provenance_path(csv)
    if not (os.path.exists(csv) and os.path.exists(pp)):
        return False
    with open(pp, encoding="utf-8") as f:
        p = json.load(f)
    return p.get("csv_sha256") == file_sha256(csv) and sorted(p.get("seasons", {})) == sorted(keys) and not p.get("bad", 1)


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    fallback = None
    if a.fallback_csv:
        fallback = pd.read_csv(a.fallback_csv)
        fb_sha = file_sha256(a.fallback_csv)
    bad_all = {}
    for key in a.stations:
        csv = os.path.join(a.out_dir, f"{key}.csv")
        if not a.force and done(csv, [str(s) for s in SEASONS]):
            log(f"[bo qua] {key}: da co, du {len(SEASONS)} mua")
            continue
        log(f"Tai {key}")
        df, info, dup, (name, uid, lon, lat) = fetch_station(key, a, fallback)
        df.to_csv(csv + ".part", index=False)
        os.replace(csv + ".part", csv)
        counts = season_counts(df)
        for s in SEASONS:
            info[str(s)]["n_rows"], info[str(s)]["n_days"] = counts.get(s, (0, 0))
        bad = check(counts)
        bad_all[key] = bad
        srcs = sorted({v["source"] for v in info.values()})
        write_provenance(csv, station=name, uniqueId=uid, lon=lon, lat=lat, parameter="Water Level", label="Manual",
                         unit="m", user_agent=UA,
                         fetched_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                         sources=srcs, fallback_csv=(a.fallback_csv.replace("\\", "/") if "fallback" in srcs else None),
                         fallback_sha256=(fb_sha if "fallback" in srcs else None),
                         fallback_note="du phong: cua so 01/11 07:00 -> 01/05 07:00 gio dia phuong (thieu doc 0-7h 01/11)"
                         if "fallback" in srcs else None,
                         duplicates_dropped=dup, n_rows=len(df), min_days=MIN_DAYS, bad=bad,
                         season_rule="ngay dia phuong, seasons.season_of (30/04 khong thuoc mua)", seasons=info)
        log(f"  {key}: {len(df)} dong, nguon {srcs}, ngay/mua {[counts.get(s, (0, 0))[1] for s in SEASONS]}"
            + (f"; THIEU {bad}" if bad else ""))
    if any(bad_all.values()):
        raise SystemExit(f"[LOI] mua thieu hoac < {MIN_DAYS} ngay: { {k: v for k, v in bad_all.items() if v} }")
    log("Xong")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stations", nargs="+", choices=list(STATIONS), default=list(STATIONS))
    ap.add_argument("--out-dir", default=data_path("mrc_ts"))
    ap.add_argument("--fallback-csv", default=None, help="CSV station,season,ts,wl da tai truoc (dung khi API loi)")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())

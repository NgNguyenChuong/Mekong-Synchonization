"""scripts/fetch_mrc_timeseries.py: cua so ngay dia phuong 01/11-30/04, dem ngay theo season_of, API loi -> du phong
ghi nguon, cung thoi diem hai gia tri -> loi."""
import argparse
import os
import sys
from urllib.error import URLError

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import fetch_mrc_timeseries as f  # noqa: E402


def _ts(day, h=7):
    return f"{day}T{h:02d}:00:00.0000000+07:00"


def test_cua_so_ngay_dia_phuong_va_mui_gio():
    df = pd.DataFrame({"datetime": [_ts("2019-10-31", 23), _ts("2019-11-01", 0), _ts("2020-04-30", 21),
                                    _ts("2020-05-01", 0)], "value": [1.0, 2.0, 3.0, 4.0]})
    assert list(f.window(df, 2020)["value"]) == [2.0, 3.0]
    with pytest.raises(ValueError, match="mui"):
        f.window(df.assign(datetime=df["datetime"].str.replace("+07:00", "+00:00")), 2020)
    c = f.season_counts(f.window(df, 2020))
    assert c == {2020: (1, 1)}                                       # 30/04 khong thuoc mua (season_of)
    assert f.check({2020: (1, 1)}, min_days=1) == [s for s in f.SEASONS if s != 2020]


def test_api_loi_dung_du_phong_va_ghi_nguon(monkeypatch):
    def boom(url, tries=4):
        if "sd=2015-10-31" in url:                                   # chi mua 2016 loi
            raise URLError("mat mang")
        s = int(url.split("ed=")[1][:4])
        return {"Points": [{"Timestamp": _ts(f"{s}-01-05"), "Value": {"Numeric": 0.0}},
                           {"Timestamp": _ts(f"{s}-01-06"), "Value": {"Numeric": None}}], "Notes": []}

    monkeypatch.setattr(f, "get_json", boom)
    fb = pd.DataFrame({"station": ["Can Tho", "Can Tho", "My Thuan"], "season": [2016, 2016, 2016],
                       "ts": [_ts("2016-02-01"), _ts("2016-05-01"), _ts("2016-02-01")], "wl": [1.5, 9.0, 7.0]})
    out, info, dup, meta = f.fetch_station("Can_Tho", argparse.Namespace(sleep=0), fb)
    assert info["2016"]["source"] == "fallback" and info["2015"]["source"] == "api"
    assert out.loc[out["datetime"] == _ts("2016-02-01"), "value"].tolist() == [1.5]   # dung tram, ngoai cua so bi bo
    assert out["value"].eq(0.0).sum() == len(f.SEASONS) - 1                          # 0.0 that giu, None bo
    assert meta[1] == f.STATIONS["Can_Tho"][1] and dup == 0
    with pytest.raises(SystemExit, match="fallback"):
        f.fetch_station("Can_Tho", argparse.Namespace(sleep=0), None)


def test_cung_thoi_diem_hai_gia_tri_la_loi(monkeypatch):
    monkeypatch.setattr(f, "get_json", lambda url, tries=4: {"Points": [
        {"Timestamp": _ts(f"{int(url.split('ed=')[1][:4])}-01-05"), "Value": {"Numeric": v}} for v in (1.0, 2.0)]})
    with pytest.raises(SystemExit, match="hai gia tri"):
        f.fetch_station("Vam_Kenh", argparse.Namespace(sleep=0), None)

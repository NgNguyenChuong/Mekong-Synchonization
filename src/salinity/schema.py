"""
Schema chuan cho du lieu do man quan trac (nhan de train model).

Day la SCHEMA, khong phai du lieu. Chua co dataset that nao trong repo.
Khi co file CSV do man thuc te, dung loader.py de doc va validate theo
dung schema nay truoc khi dua vao pipeline.
"""
from dataclasses import dataclass

# Cac cot BAT BUOC phai co trong file CSV do man quan trac.
REQUIRED_COLUMNS = [
    "station_id",     # ma tram quan trac (str)
    "station_name",   # ten tram (str, co the trung station_id neu khong co)
    "latitude",       # vi do tram (float, WGS84)
    "longitude",      # kinh do tram (float, WGS84)
    "datetime",        # thoi diem do (ISO 8601 string hoac parse duoc boi pandas)
    "salinity",       # gia tri do man - DON VI PHAI GHI RO trong cot "unit"
    "source",         # nguon du lieu (vd: "MRC", "Vien Khoa hoc Thuy loi mien Nam"...)
]

# Cot khuyen khich nhung khong bat buoc.
OPTIONAL_COLUMNS = [
    "unit",  # "ppt" | "g/L" | ... - PHAI xac dinh truoc khi dung cho training,
             # khong duoc gia dinh don vi neu dataset khong ghi ro.
]


@dataclass(frozen=True)
class SalinityObservation:
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    datetime: str
    salinity: float
    source: str
    unit: str | None = None

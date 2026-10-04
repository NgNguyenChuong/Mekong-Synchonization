"""Mua kho dung chung cho dac trung va nhan.

Mua kho nam Y = 01/11/(Y-1) .. 29/04/Y (bao gom ca hai dau).
Khop ma GEE cua tac gia nhan Zenodo: filterDate('Y-1-11-01', 'Y-04-30') - GEE loai ngay cuoi,
nen 30/04 KHONG thuoc mua. Moi script gop theo mua phai dung ham nay, khong tu viet lai
(loi lech nam cua MRC cu la do moi noi tu dinh nghia mua mot kieu).
"""
import pandas as pd

SEASON_START = (11, 1)   # 01/11 cua nam Y-1
SEASON_END = (4, 29)     # 29/04 cua nam Y (bao gom)


def season_of(dates) -> pd.Series:
    """Tra ve nam mua kho (Int64) cho tung ngay; NA neu ngay khong thuoc mua kho nao."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    m, day = d.dt.month, d.dt.day
    in_late = m >= SEASON_START[0]                                     # 11, 12 -> mua nam sau
    in_early = (m < SEASON_END[0]) | ((m == SEASON_END[0]) & (day <= SEASON_END[1]))  # 1..3, 1-29/04
    out = pd.Series(pd.NA, index=d.index, dtype="Int64")
    out[in_late] = d.dt.year[in_late] + 1
    out[in_early] = d.dt.year[in_early]
    return out


def season_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(ngay dau, ngay cuoi) bao gom ca hai dau cua mua kho nam `year`."""
    return (pd.Timestamp(year - 1, *SEASON_START), pd.Timestamp(year, *SEASON_END))


def season_days(year: int) -> int:
    start, end = season_window(year)
    return (end - start).days + 1

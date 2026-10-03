"""
FastAPI backend cho demo du doan do man DBSCL theo luoi H3.

Chay: uvicorn app:app --port 8002

LUU Y: Khong dung --reload trong moi truong dev nay - watchfiles reloader
bi treo (khong tu nap lai code khi file thay doi), khien server phuc vu
nham code cu ma khong bao loi ro rang. Sua code xong thi tat process cu va
chay lai lenh o tren.
"""
from datetime import date as date_cls

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import h3

from config import get_predictor
from grid import SCOPES, build_grid, cell_centroid

app = FastAPI(title="Mekong Salinity Prediction")

# Predictor dung model that (config.py). Khoi tao ngay luc import (= luc server
# start) de fail-fast khi chua co model, thay vi am tham loi tren tung request.
predictor = get_predictor()

# Dev local: cho phep moi origin de frontend tinh (vd. http://localhost:5500) goi duoc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _validate_date(value: str) -> str:
    try:
        date_cls.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date phai co dinh dang YYYY-MM-DD") from exc
    return value


def _validate_scope(value: str) -> str:
    if value not in SCOPES:
        raise HTTPException(status_code=400, detail=f"scope phai la mot trong: {list(SCOPES)}")
    return value


@app.get("/api/predict")
def predict(
    date: str = Query(..., description="Ngay du doan, dinh dang YYYY-MM-DD"),
    scope: str = Query("mekong", description="Pham vi: 'mekong' (chi tiet) hoac 'world' (toan cau, tho)"),
    resolution: int | None = Query(None, description="Do phan giai H3. Bo trong de dung mac dinh theo scope."),
):
    date = _validate_date(date)
    scope = _validate_scope(scope)
    cells = build_grid(scope, resolution)

    features = []
    for h3_index, boundary in cells:
        centroid_lat, centroid_lon = cell_centroid(h3_index)
        result = predictor.predict(scope, h3_index, centroid_lat, centroid_lon, date)

        # GeoJSON dung (lon, lat), boundary dang tra ve (lat, lon) -> dao nguoc
        ring = [[lon, lat] for lat, lon in boundary]
        ring.append(ring[0])  # dong vong polygon

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "h3_index": h3_index,
                    "salinity_ppt": result.salinity_ppt,
                    "date": date,
                },
            }
        )

    # prediction_mode/model_meta o cap top-level (khong nhet vao tung feature
    # de khong lam nang payload) - frontend doc 1 lan de hien badge Mock/Model.
    return {
        "type": "FeatureCollection",
        "features": features,
        "model_meta": predictor.metadata,
    }


def _month_series_dates(date_from: str, date_to: str) -> list[str]:
    """Sinh danh sach ngay dai dien (ngay 15) cho tung thang trong [date_from, date_to]."""
    start = date_cls.fromisoformat(date_from)
    end = date_cls.fromisoformat(date_to)
    if start > end:
        raise HTTPException(status_code=400, detail="'from' phai <= 'to'")

    dates = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        dates.append(date_cls(year, month, 15).isoformat())
        if len(dates) > 120:
            raise HTTPException(status_code=400, detail="Khoang thoi gian qua dai (toi da 120 thang).")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


@app.get("/api/predict/series")
def predict_series(
    h3_index: str = Query(..., description="Ma o H3 can xem xu huong theo thoi gian"),
    date_from: str = Query(..., alias="from", description="Ngay bat dau, YYYY-MM-DD"),
    date_to: str = Query(..., alias="to", description="Ngay ket thuc, YYYY-MM-DD"),
    scope: str = Query("mekong", description="Pham vi: 'mekong' hoac 'world', phai khop voi scope da dung de lay h3_index"),
):
    if not h3.is_valid_cell(h3_index):
        raise HTTPException(status_code=400, detail="h3_index khong hop le")
    scope = _validate_scope(scope)

    date_from = _validate_date(date_from)
    date_to = _validate_date(date_to)
    dates = _month_series_dates(date_from, date_to)

    centroid_lat, centroid_lon = cell_centroid(h3_index)
    points = [
        {"date": d, "salinity_ppt": predictor.predict(scope, h3_index, centroid_lat, centroid_lon, d).salinity_ppt}
        for d in dates
    ]

    return {"h3_index": h3_index, "points": points, "model_meta": predictor.metadata}


@app.get("/api/health")
def health():
    return {"status": "ok"}

"""Thiet ke danh gia dung chung cho moi khung luoi (T3-V4).

- Khoi giu rieng: luoi khoi vuong neo o boi so canh (toa do UTM cua vung); chon
  ngau nhien (seed co dinh) cho den khi dat ty le DIEN TICH DAT muc tieu.
  Khoi dinh nghia tren toa do dia ly, doc lap voi khung luoi.
- Diem danh gia: gieo ngau nhien deu trong phan dat cua vung kiem tra.
- O huan luyen: moi o GIAO vung kiem tra (ke ca giao mot phan) bi loai.
"""
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box
from shapely.prepared import prep


def make_holdout_blocks(boundary: gpd.GeoDataFrame, block_km=50.0, holdout_frac=0.2, seed=42):
    """Tra ve GeoDataFrame (EPSG:4326): block_id, land_km2, is_holdout, geometry."""
    utm = boundary.estimate_utm_crs()
    land = boundary.to_crs(utm).geometry.union_all()
    size = block_km * 1000.0
    minx, miny, maxx, maxy = land.bounds

    blocks, ids = [], []
    for ix in range(int(np.floor(minx / size)), int(np.ceil(maxx / size))):
        for iy in range(int(np.floor(miny / size)), int(np.ceil(maxy / size))):
            b = box(ix * size, iy * size, (ix + 1) * size, (iy + 1) * size)
            if b.intersects(land):
                blocks.append(b)
                ids.append(f"blk_{ix}_{iy}")

    land_km2 = np.array([b.intersection(land).area / 1e6 for b in blocks])
    target = holdout_frac * land_km2.sum()
    order = np.random.default_rng(seed).permutation(len(blocks))
    is_holdout = np.zeros(len(blocks), dtype=bool)
    acc = 0.0
    for i in order:
        if acc >= target:
            break
        if land_km2[i] <= 0:
            continue
        is_holdout[i] = True
        acc += land_km2[i]

    gdf = gpd.GeoDataFrame(
        {"block_id": ids, "land_km2": np.round(land_km2, 3), "is_holdout": is_holdout},
        geometry=blocks, crs=utm,
    )
    return gdf.to_crs(4326)


def sample_eval_points(boundary: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, n_points=3000, seed=42):
    """Gieo n diem ngau nhien deu (theo dien tich) trong dat thuoc cac khoi giu rieng."""
    utm = boundary.estimate_utm_crs()
    land = boundary.to_crs(utm).geometry.union_all()
    holdout = blocks[blocks["is_holdout"]].to_crs(utm).geometry.union_all()
    region = land.intersection(holdout)
    if region.is_empty:
        raise ValueError("Vung kiem tra rong - kiem tra lai khoi giu rieng.")

    shapely.prepare(region)
    minx, miny, maxx, maxy = region.bounds
    rng = np.random.default_rng(seed)
    xs, ys = np.empty(0), np.empty(0)
    while xs.size < n_points:
        cx = rng.uniform(minx, maxx, 4 * n_points)
        cy = rng.uniform(miny, maxy, 4 * n_points)
        keep = shapely.contains_xy(region, cx, cy)
        xs, ys = np.concatenate([xs, cx[keep]]), np.concatenate([ys, cy[keep]])
    xs, ys = xs[:n_points], ys[:n_points]

    return gpd.GeoDataFrame(
        {"point_id": [f"pt_{i:05d}" for i in range(n_points)]},
        geometry=gpd.points_from_xy(xs, ys), crs=utm,
    ).to_crs(4326)


def training_mask(grid: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame):
    """True = o duoc phep dung de huan luyen (khong giao vung kiem tra nao)."""
    holdout = blocks[blocks["is_holdout"]].to_crs(grid.crs).geometry.union_all()
    touching = prep(holdout)
    return ~grid.geometry.apply(touching.intersects).to_numpy()

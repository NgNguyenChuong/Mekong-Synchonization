import datetime
import hashlib
import json
import os
import sys

import geopandas as gpd
import h3
import numpy as np
import pyproj
import s2sphere
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import transform, unary_union

from config import (
    SHAPEFILE_RAW, SHAPEFILE_CLEAN,
    CRS_METRIC, CRS_WGS84, MIN_ISLAND_AREA_KM2,
    H3_GRID_GEOJSON, H3_RESOLUTION,
)

_BOUNDARY_DIR = os.path.join(os.path.dirname(__file__), "..", "webapp", "backend", "data")
# v1: 13 tinh geoBoundaries (webapp van dung). v2 (An duyet 2026-10-03): v1 noi 3 km phia bien,
# tru lanh tho lang gieng - sinh bang scripts/build_boundary_v2.py. Moi script du lieu dung v2.
BOUNDARY_V1 = os.path.abspath(os.path.join(_BOUNDARY_DIR, "mekong_delta_boundary.geojson"))
CANONICAL_BOUNDARY = os.path.abspath(os.path.join(_BOUNDARY_DIR, "mekong_delta_boundary_v2.geojson"))

# O "giao" ranh gioi khi ty le dien tich giao > nguong nay. Nguong khac 0 chi de bo nhieu so hoc
# (o chi cham canh/dinh ranh gioi cho dien tich giao ~1e-13 sau phep chieu qua lai).
MIN_OVERLAP_FRAC = 1e-9


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_path(output_path) -> str:
    return f"{output_path}.provenance.json"


def write_provenance(output_path, boundary_path, **extra) -> str:
    """Ghi <output>.provenance.json: ranh gioi da dung (duong dan + sha256) va tham so sinh."""
    info = {
        "output": os.path.basename(output_path),
        "boundary": os.path.abspath(boundary_path).replace("\\", "/"),
        "boundary_sha256": file_sha256(boundary_path),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        **extra,
    }
    path = provenance_path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return path


def _resolve_boundary_path(boundary_path=None) -> str:
    if boundary_path:
        if not os.path.exists(boundary_path):
            raise FileNotFoundError(f"Khong tim thay ranh gioi: {boundary_path}")
        return boundary_path
    if os.path.exists(SHAPEFILE_CLEAN):
        return SHAPEFILE_CLEAN
    if os.path.exists(CANONICAL_BOUNDARY):
        return CANONICAL_BOUNDARY
    raise FileNotFoundError(f"Khong tim thay ranh gioi chuan: {CANONICAL_BOUNDARY}")


def _load_union(boundary_path, crs):
    gdf = gpd.read_file(boundary_path).to_crs(crs)
    return unary_union(gdf.geometry)


def _select_intersecting(cell_ids, geoms, union_poly, cell_area=None):
    """Giu o GIAO ranh gioi (overlap_frac > MIN_OVERLAP_FRAC), khong cat hinh o.

    overlap_frac = dien tich (o giao ranh gioi) / dien tich o, tinh trong CRS cua hinh dau vao.
    Tra ve (cell_ids, geoms, fracs) da loc, giu thu tu dau vao.
    """
    arr = np.empty(len(geoms), dtype=object)
    arr[:] = list(geoms)
    if len(arr) == 0:
        return [], [], np.zeros(0)
    shapely.prepare(union_poly)
    hit = shapely.intersects(union_poly, arr)
    inner = hit & shapely.contains_properly(union_poly, arr)
    fracs = np.zeros(len(arr))
    fracs[inner] = 1.0
    edge = hit & ~inner
    if edge.any():
        inter = shapely.intersection(arr[edge], union_poly)
        area = cell_area if cell_area is not None else shapely.area(arr[edge])
        fracs[edge] = np.minimum(shapely.area(inter) / area, 1.0)
    keep = np.flatnonzero(fracs > MIN_OVERLAP_FRAC)
    return [cell_ids[i] for i in keep], list(arr[keep]), fracs[keep]


def _write(gdf, output_path):
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")
    return gdf


# -----------------------------------------------------------
# CLEAN SHAPEFILE (REMOVE SMALL ISLANDS)
# -----------------------------------------------------------
def clean_shapefile():
    """Tra ve duong dan ranh gioi dung cho pipeline.

    Neu khong co data/raw/boundary_input.shp thi dung ranh gioi chuan 13 tinh.
    """
    if not os.path.exists(SHAPEFILE_RAW):
        if os.path.exists(CANONICAL_BOUNDARY):
            print(f"    Input shapefile missing ({SHAPEFILE_RAW}). Dung ranh gioi chuan: {CANONICAL_BOUNDARY}")
            return CANONICAL_BOUNDARY
        raise FileNotFoundError(f"Input shapefile missing: {SHAPEFILE_RAW}")

    if os.path.exists(SHAPEFILE_CLEAN) and os.path.getmtime(SHAPEFILE_CLEAN) >= os.path.getmtime(SHAPEFILE_RAW):
        print(f"    Cleaned shapefile already exists: {SHAPEFILE_CLEAN}")
        return SHAPEFILE_CLEAN

    print("    Cleaning shapefile...")
    gdf = gpd.read_file(SHAPEFILE_RAW)
    gdf_exploded = gdf.to_crs(CRS_METRIC).explode(index_parts=True).reset_index(drop=True)
    gdf_exploded["area_km2"] = gdf_exploded.geometry.area / 1e6

    if MIN_ISLAND_AREA_KM2 > 0:
        gdf_clean = gdf_exploded[gdf_exploded["area_km2"] > MIN_ISLAND_AREA_KM2].copy()
    else:
        gdf_clean = gdf_exploded.copy()

    gdf_final = gdf_clean.dissolve().to_crs(CRS_WGS84)
    os.makedirs(os.path.dirname(SHAPEFILE_CLEAN), exist_ok=True)
    gdf_final.to_file(SHAPEFILE_CLEAN)
    print(f"    Cleaned shapefile saved: {SHAPEFILE_CLEAN}")
    return SHAPEFILE_CLEAN


# -----------------------------------------------------------
# 4 KHUNG LUOI - quy tac chung (An duyet 2026-10-03): o thuoc luoi neu GIAO ranh gioi
# (overlap_frac > 0, ke ca giao 1%), giu nguyen hinh hoc o (khong cat), ghi overlap_frac.
# Moi diem trong ranh gioi deu thuoc it nhat 1 o. Truoc day: theo tam o (bo sot dai ven bien).
# -----------------------------------------------------------
def _h3_polyfill(poly, resolution):
    parts = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    cells = set()
    for part in parts:
        if len(set(part.exterior.coords)) < 3:
            continue
        outer = [(lat, lon) for lon, lat in part.exterior.coords]
        # H3 v4 nhan tung lo la mot doi so rieng: LatLngPoly(outer, *holes).
        holes = [
            [(lat, lon) for lon, lat in ring.coords]
            for ring in part.interiors
            if len(set(ring.coords)) >= 3
        ]
        cells.update(h3.polygon_to_cells(h3.LatLngPoly(outer, *holes), resolution))
    return cells


def generate_h3_grid(boundary_path=None, resolution=None, output_path=None) -> gpd.GeoDataFrame:
    boundary_path = _resolve_boundary_path(boundary_path)
    resolution = H3_RESOLUTION if resolution is None else resolution
    union_poly = _load_union(boundary_path, CRS_WGS84)

    # Ung vien = o co TAM trong ranh gioi noi rong 2,5 canh o (do): moi o giao ranh gioi deu co tam
    # cach ranh gioi <= ban kinh ngoai tiep (= canh o) nen nam trong tap nay; sau do loc theo giao.
    pad_deg = 2.5 * h3.average_hexagon_edge_length(resolution, unit="km") / 110.0
    candidates = sorted(_h3_polyfill(union_poly.buffer(pad_deg), resolution))
    geoms = [Polygon([(p[1], p[0]) for p in h3.cell_to_boundary(c)]) for c in candidates]
    cell_ids, geoms, fracs = _select_intersecting(candidates, geoms, union_poly)

    gdf = gpd.GeoDataFrame({"cell_id": cell_ids, "overlap_frac": fracs}, geometry=geoms, crs=CRS_WGS84)
    return _write(gdf, output_path)


def generate_s2_grid(boundary_path=None, level=11, output_path=None) -> gpd.GeoDataFrame:
    boundary_path = _resolve_boundary_path(boundary_path)
    union_poly = _load_union(boundary_path, CRS_WGS84)
    minx, miny, maxx, maxy = union_poly.bounds

    # Liet ke moi o o dung level phu khung bao (co le), roi loc theo giao ranh gioi.
    coverer = s2sphere.RegionCoverer()
    coverer.min_level = level
    coverer.max_level = level
    coverer.max_cells = 10_000_000
    rect = s2sphere.LatLngRect(
        s2sphere.LatLng.from_degrees(miny - 0.05, minx - 0.05),
        s2sphere.LatLng.from_degrees(maxy + 0.05, maxx + 0.05),
    )

    ids, geoms = [], []
    for cid in coverer.get_covering(rect):
        cell = s2sphere.Cell(cid)
        corners = [s2sphere.LatLng.from_point(cell.get_vertex(i)) for i in range(4)]
        ids.append(f"s2_{cid.to_token()}")
        geoms.append(Polygon([(v.lng().degrees, v.lat().degrees) for v in corners]))
    cell_ids, geoms, fracs = _select_intersecting(ids, geoms, union_poly)

    gdf = gpd.GeoDataFrame({"cell_id": cell_ids, "overlap_frac": fracs}, geometry=geoms, crs=CRS_WGS84)
    return _write(gdf, output_path)


def generate_square_grid(boundary_path=None, resolution_m=6458, output_path=None) -> gpd.GeoDataFrame:
    """O vuong trong EPSG:32648, goc neo o boi so cua resolution_m theo toa do UTM."""
    boundary_path = _resolve_boundary_path(boundary_path)
    union_poly = _load_union(boundary_path, CRS_METRIC)

    minx, miny, maxx, maxy = union_poly.bounds
    size = float(resolution_m)
    xs = np.arange(np.floor(minx / size) * size, maxx + size, size)
    ys = np.arange(np.floor(miny / size) * size, maxy + size, size)

    ids, geoms_utm = [], []
    for x in xs:
        for y in ys:
            ids.append(f"sq_{resolution_m}m_{int(x)}_{int(y)}")
            geoms_utm.append(box(x, y, x + size, y + size))
    cell_ids, geoms_utm, fracs = _select_intersecting(ids, geoms_utm, union_poly, cell_area=size * size)

    to_wgs = pyproj.Transformer.from_crs(CRS_METRIC, CRS_WGS84, always_xy=True)
    geoms = [transform(to_wgs.transform, g) for g in geoms_utm]

    gdf = gpd.GeoDataFrame({"cell_id": cell_ids, "overlap_frac": fracs}, geometry=geoms, crs=CRS_WGS84)
    return _write(gdf, output_path)


def generate_latlon_grid(boundary_path=None, step_deg=0.0586, output_path=None) -> gpd.GeoDataFrame:
    """O deu theo do, goc neo o boi so cua step_deg tinh tu (0, 0)."""
    boundary_path = _resolve_boundary_path(boundary_path)
    union_poly = _load_union(boundary_path, CRS_WGS84)

    minx, miny, maxx, maxy = union_poly.bounds
    step = float(step_deg)
    ix0, ix1 = int(np.floor(minx / step)), int(np.ceil(maxx / step))
    iy0, iy1 = int(np.floor(miny / step)), int(np.ceil(maxy / step))

    ids, geoms = [], []
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            x, y = ix * step, iy * step
            ids.append(f"ll_{int(round(step * 10000))}p_{iy}_{ix}")
            geoms.append(box(x, y, x + step, y + step))
    cell_ids, geoms, fracs = _select_intersecting(ids, geoms, union_poly)

    gdf = gpd.GeoDataFrame({"cell_id": cell_ids, "overlap_frac": fracs}, geometry=geoms, crs=CRS_WGS84)
    return _write(gdf, output_path)


# -----------------------------------------------------------
# MAIN WRAPPER
# -----------------------------------------------------------
def run_preprocessing():
    print("--- [PREPROCESSING] ---")
    boundary = clean_shapefile()
    if os.path.exists(H3_GRID_GEOJSON) and os.path.getmtime(H3_GRID_GEOJSON) >= os.path.getmtime(boundary):
        print(f"    H3 Grid already exists: {H3_GRID_GEOJSON}")
    else:
        gdf = generate_h3_grid(boundary, H3_RESOLUTION, H3_GRID_GEOJSON)
        print(f"    H3 Grid saved: {H3_GRID_GEOJSON} ({len(gdf)} cells)")
    print("-----------------------")

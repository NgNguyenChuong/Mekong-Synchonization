import os

import geopandas as gpd
import h3
import numpy as np
import pyproj
import s2sphere
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform, unary_union
from shapely.prepared import prep

from config import (
    SHAPEFILE_RAW, SHAPEFILE_CLEAN,
    CRS_METRIC, CRS_WGS84, MIN_ISLAND_AREA_KM2,
    H3_GRID_GEOJSON, H3_RESOLUTION,
)

CANONICAL_BOUNDARY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "webapp", "backend", "data", "mekong_delta_boundary.geojson")
)


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


def _overlap_fracs(cell_geoms, union_poly, cell_area=None):
    """Ty le dien tich o nam trong ranh gioi. Chi tinh giao cho o cat duong bien."""
    boundary_line = prep(union_poly.boundary)
    fracs = []
    for geom in cell_geoms:
        if not boundary_line.intersects(geom):
            fracs.append(1.0)
            continue
        inter = geom.intersection(union_poly)
        area = cell_area if cell_area is not None else geom.area
        fracs.append(round(inter.area / area, 4) if not inter.is_empty else 0.0)
    return fracs


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
# 4 KHUNG LUOI - quy tac chung: o thuoc luoi neu tam nam trong ranh gioi,
# giu nguyen hinh hoc o (khong cat), ghi overlap_frac.
# -----------------------------------------------------------
def generate_h3_grid(boundary_path=None, resolution=None, output_path=None) -> gpd.GeoDataFrame:
    boundary_path = _resolve_boundary_path(boundary_path)
    resolution = H3_RESOLUTION if resolution is None else resolution
    union_poly = _load_union(boundary_path, CRS_WGS84)
    inside = prep(union_poly)

    parts = union_poly.geoms if union_poly.geom_type == "MultiPolygon" else [union_poly]
    candidates = set()
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
        candidates.update(h3.polygon_to_cells(h3.LatLngPoly(outer, *holes), resolution))

    cell_ids, geoms = [], []
    for cell in sorted(candidates):
        lat, lon = h3.cell_to_latlng(cell)
        if inside.contains(Point(lon, lat)):
            cell_ids.append(cell)
            geoms.append(Polygon([(p[1], p[0]) for p in h3.cell_to_boundary(cell)]))

    gdf = gpd.GeoDataFrame(
        {"cell_id": cell_ids, "overlap_frac": _overlap_fracs(geoms, union_poly)},
        geometry=geoms, crs=CRS_WGS84,
    )
    return _write(gdf, output_path)


def generate_s2_grid(boundary_path=None, level=11, output_path=None) -> gpd.GeoDataFrame:
    boundary_path = _resolve_boundary_path(boundary_path)
    union_poly = _load_union(boundary_path, CRS_WGS84)
    inside = prep(union_poly)
    minx, miny, maxx, maxy = union_poly.bounds

    # Liet ke moi o o dung level trong khung bao, roi loc theo tam
    # (khong dung covering mac dinh vi no giu ca o chi cham ranh gioi).
    coverer = s2sphere.RegionCoverer()
    coverer.min_level = level
    coverer.max_level = level
    coverer.max_cells = 10_000_000
    rect = s2sphere.LatLngRect(
        s2sphere.LatLng.from_degrees(miny - 0.05, minx - 0.05),
        s2sphere.LatLng.from_degrees(maxy + 0.05, maxx + 0.05),
    )

    cell_ids, geoms = [], []
    for cid in coverer.get_covering(rect):
        cell = s2sphere.Cell(cid)
        centre = s2sphere.LatLng.from_point(cell.get_center())
        if inside.contains(Point(centre.lng().degrees, centre.lat().degrees)):
            cell_ids.append(f"s2_{cid.to_token()}")
            corners = [s2sphere.LatLng.from_point(cell.get_vertex(i)) for i in range(4)]
            geoms.append(Polygon([(v.lng().degrees, v.lat().degrees) for v in corners]))

    gdf = gpd.GeoDataFrame(
        {"cell_id": cell_ids, "overlap_frac": _overlap_fracs(geoms, union_poly)},
        geometry=geoms, crs=CRS_WGS84,
    )
    return _write(gdf, output_path)


def generate_square_grid(boundary_path=None, resolution_m=6458, output_path=None) -> gpd.GeoDataFrame:
    """O vuong trong EPSG:32648, goc neo o boi so cua resolution_m theo toa do UTM."""
    boundary_path = _resolve_boundary_path(boundary_path)
    union_poly = _load_union(boundary_path, CRS_METRIC)
    inside = prep(union_poly)

    minx, miny, maxx, maxy = union_poly.bounds
    size = float(resolution_m)
    xs = np.arange(np.floor(minx / size) * size, maxx + size, size)
    ys = np.arange(np.floor(miny / size) * size, maxy + size, size)

    cell_ids, geoms_utm = [], []
    for x in xs:
        for y in ys:
            if inside.contains(Point(x + size / 2.0, y + size / 2.0)):
                cell_ids.append(f"sq_{resolution_m}m_{int(x)}_{int(y)}")
                geoms_utm.append(box(x, y, x + size, y + size))

    fracs = _overlap_fracs(geoms_utm, union_poly, cell_area=size * size)
    to_wgs = pyproj.Transformer.from_crs(CRS_METRIC, CRS_WGS84, always_xy=True)
    geoms = [transform(to_wgs.transform, g) for g in geoms_utm]

    gdf = gpd.GeoDataFrame({"cell_id": cell_ids, "overlap_frac": fracs}, geometry=geoms, crs=CRS_WGS84)
    return _write(gdf, output_path)


def generate_latlon_grid(boundary_path=None, step_deg=0.0586, output_path=None) -> gpd.GeoDataFrame:
    """O deu theo do, goc neo o boi so cua step_deg tinh tu (0, 0)."""
    boundary_path = _resolve_boundary_path(boundary_path)
    union_poly = _load_union(boundary_path, CRS_WGS84)
    inside = prep(union_poly)

    minx, miny, maxx, maxy = union_poly.bounds
    step = float(step_deg)
    ix0, ix1 = int(np.floor(minx / step)), int(np.ceil(maxx / step))
    iy0, iy1 = int(np.floor(miny / step)), int(np.ceil(maxy / step))

    cell_ids, geoms = [], []
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            x, y = ix * step, iy * step
            if inside.contains(Point(x + step / 2.0, y + step / 2.0)):
                cell_ids.append(f"ll_{int(round(step * 10000))}p_{iy}_{ix}")
                geoms.append(box(x, y, x + step, y + step))

    gdf = gpd.GeoDataFrame(
        {"cell_id": cell_ids, "overlap_frac": _overlap_fracs(geoms, union_poly)},
        geometry=geoms, crs=CRS_WGS84,
    )
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

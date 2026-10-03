"""
Sinh luoi H3 cho 2 pham vi (scope):

- "mekong": ranh gioi hanh chinh THAT cua 13 tinh/thanh DBSCL (An Giang,
  Bac Lieu, Ben Tre, Ca Mau, Can Tho, Hau Giang, Kien Giang, Long An,
  Soc Trang, Tien Giang, Tra Vinh, Vinh Long, Dong Thap), hop nhat tu
  nguon mo geoBoundaries.org. File data/mekong_delta_boundary.geojson
  duoc tao san boi data/build_boundary.py.

- "world": toan bo dat lien the gioi, tu Natural Earth 110m Land (public
  domain). File data/world_land_boundary.geojson duoc tao san boi
  data/build_world_boundary.py. Day la du lieu do phan giai thap (110m),
  chi phu hop de minh hoa tong quan toan cau, KHONG chi tiet nhu pham vi
  mekong.

Ca 2 deu la ranh gioi dia ly THAT, khong con la polygon ve tay xap xi.
"""
import json
import os

import h3
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SCOPES = {
    "mekong": {
        "boundary_path": os.path.join(_DATA_DIR, "mekong_delta_boundary.geojson"),
        "default_resolution": 6,
        "min_resolution": 4,
        "max_resolution": 8,
        # h3.polygon_to_cells chi giu cell co TAM nam trong polygon. Ranh gioi
        # that van co the lam rot mot vien mong cac o sat bien. Buffer nho
        # (~1.5km) de bu sai so nay ma khong lam sai lech dang ke hinh dang.
        "buffer_deg": 0.015,
    },
    "world": {
        "boundary_path": os.path.join(_DATA_DIR, "world_land_boundary.geojson"),
        "default_resolution": 3,
        "min_resolution": 2,
        "max_resolution": 4,
        # Du lieu Natural Earth 110m rat tho, buffer lon hon de tranh sot dai
        # bien mong khi fill o do phan giai thap.
        "buffer_deg": 0.3,
    },
}

_grid_cache: dict[tuple[str, int], list[tuple[str, list[tuple[float, float]]]]] = {}
_boundary_cache: dict[str, list[Polygon]] = {}


def _load_geojson_geometries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        return [shape(feature["geometry"]) for feature in data["features"]]
    if data.get("type") == "Feature":
        return [shape(data["geometry"])]
    return [shape(data)]


def _load_boundary_polygons(scope: str) -> list[Polygon]:
    if scope in _boundary_cache:
        return _boundary_cache[scope]

    cfg = SCOPES[scope]
    geoms = _load_geojson_geometries(cfg["boundary_path"])

    polygons: list[Polygon] = []
    for geom in geoms:
        buffered = geom.buffer(cfg["buffer_deg"])
        if buffered.geom_type == "Polygon":
            polygons.append(buffered)
        else:  # MultiPolygon
            polygons.extend(buffered.geoms)

    _boundary_cache[scope] = polygons
    return polygons


def _polygon_to_h3_loops(poly: Polygon):
    outer = [(lat, lon) for lon, lat in poly.exterior.coords]
    holes = [[(lat, lon) for lon, lat in interior.coords] for interior in poly.interiors]
    return h3.LatLngPoly(outer, *holes)


def clamp_resolution(scope: str, resolution: int) -> int:
    cfg = SCOPES[scope]
    return max(cfg["min_resolution"], min(cfg["max_resolution"], resolution))


def default_resolution(scope: str) -> int:
    return SCOPES[scope]["default_resolution"]


def build_grid(scope: str = "mekong", resolution: int | None = None):
    """Tra ve list (h3_index, boundary) voi boundary la list (lat, lon).

    Ket qua duoc cache trong bo nho theo (scope, resolution) vi ranh gioi
    co dinh.
    """
    if scope not in SCOPES:
        raise ValueError(f"Scope khong hop le: {scope}")

    if resolution is None:
        resolution = default_resolution(scope)
    resolution = clamp_resolution(scope, resolution)

    cache_key = (scope, resolution)
    if cache_key in _grid_cache:
        return _grid_cache[cache_key]

    hex_set: set[str] = set()
    for poly in _load_boundary_polygons(scope):
        h3_poly = _polygon_to_h3_loops(poly)
        hex_set.update(h3.polygon_to_cells(h3_poly, resolution))

    result = []
    for cell in hex_set:
        boundary = h3.cell_to_boundary(cell)  # list of (lat, lon)
        result.append((cell, boundary))

    _grid_cache[cache_key] = result
    return result


def cell_centroid(h3_index: str) -> tuple[float, float]:
    """Tra ve (lat, lon) tam cua 1 o H3."""
    lat, lon = h3.cell_to_latlng(h3_index)
    return lat, lon

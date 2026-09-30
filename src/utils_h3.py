import os
import re
import random
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import math

# -----------------------------------------------------------
# File Helpers
# -----------------------------------------------------------
def parse_year_month(fname):
    """Parse năm-tháng từ tên file"""
    m = re.search(r"(\d{4})_(\d{1,2})", fname)
    if not m:
        # Thử format khác nếu cần, hoặc log warning
        return None
    return int(m.group(1)), int(m.group(2))

def index_files(folder):
    """Index tất cả .tif files theo (year, month)"""
    if not os.path.exists(folder):
        print(f"⚠️ Warning: Folder not found: {folder}")
        return {}
        
    mapping = {}
    for f in os.listdir(folder):
        if f.endswith(".tif") or f.endswith(".tiff"):
            ym = parse_year_month(f)
            if ym:
                mapping[ym] = os.path.join(folder, f)
    return mapping

# -----------------------------------------------------------
# Geometry Helpers
# -----------------------------------------------------------
def get_h3_sample_points(hex_geom):
    """Tạo 7 điểm sample cho 1 ô H3 (1 centroid + 6 midpoints)"""
    pts = []
    c = hex_geom.centroid
    pts.append((c.x, c.y))
    
    # Midpoint của 6 cạnh
    if hex_geom.geom_type == 'Polygon':
        coords = list(hex_geom.exterior.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            pts.append(((x1 + x2) / 2, (y1 + y2) / 2))
    return pts

def load_h3_multipoints(h3_path, crs_metric, crs_wgs84):
    """
    Load grid (H3 hoac generic) và trả về dữ liệu hình học cần thiết
    """
    if not os.path.exists(h3_path):
        raise FileNotFoundError(f"Không tìm thấy file Grid: {h3_path}")

    print(f"loading grid: {h3_path}")
    cells_df = gpd.read_file(h3_path)
    
    # Chuyển đổi CRS để tính toán hình học chính xác
    cells_metric = cells_df.to_crs(crs_metric)
    cells_wgs = cells_metric.to_crs(crs_wgs84)
    
    if "cell_id" in cells_df.columns:
        cell_ids = cells_df["cell_id"].astype(str).tolist()
    elif "h3_index" in cells_df.columns:
        cell_ids = cells_df["h3_index"].astype(str).tolist()
    else:
        raise KeyError(f"Khong tim thay cot cell_id hoac h3_index trong {h3_path}")
    
    # Tạo point groups cho sampling
    point_groups = [get_h3_sample_points(geom) for geom in cells_wgs.geometry]
    
    return cell_ids, point_groups, list(cells_wgs.geometry)

def random_points_in_polygon(poly, n):
    """Sinh n điểm ngẫu nhiên nằm trong polygon"""
    minx, miny, maxx, maxy = poly.bounds
    points = []
    while len(points) < n:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            points.append((p.x, p.y))
    return points

# -----------------------------------------------------------
# Sampling Logic - OPTIMIZED
# -----------------------------------------------------------
def is_nodata(value, nodata):
    if value is None or nodata is None:
        return value is None
    try:
        return math.isclose(float(value), float(nodata), rel_tol=1e-9, abs_tol=1e-9)
    except (ValueError, TypeError):
        return value == nodata

def sample_multiband_robust(tif_path, point_groups, h3_geoms=None, n_random=15):
    """
    Chiến thuật lấy mẫu (OPTIMIZED - Batch sampling):
    1. Lấy tại tâm (batch)
    2. Lấy trung bình 6 điểm cạnh (batch)
    """
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        num_bands = src.count
        num_cells = len(point_groups)

        # Mảng kết quả [num_cells][num_days_in_month]
        vals = [[None] * num_bands for _ in range(num_cells)]

        for band_idx in range(num_bands):
            idx_param = band_idx + 1  # rasterio index bắt đầu từ 1

            # OPTIMIZED: Batch sampling cho centroids
            centroids = [points[0] for points in point_groups]
            centroid_values = list(src.sample(centroids, indexes=idx_param))

            # Track những cell cần fallback sang midpoints
            needs_fallback = []

            for i, v in enumerate(centroid_values):
                v = v[0]  # src.sample trả về array
                if not is_nodata(v, nodata):
                    vals[i][band_idx] = float(v)
                else:
                    needs_fallback.append(i)

            # OPTIMIZED: Batch sampling cho midpoints của những cell cần fallback
            if needs_fallback:
                # Thu thập tất cả midpoints
                all_midpoints = []
                midpoint_cell_map = []  # Map midpoint index -> cell index

                for cell_idx in needs_fallback:
                    midpoints = point_groups[cell_idx][1:]  # Bỏ centroid
                    for pt in midpoints:
                        all_midpoints.append(pt)
                        midpoint_cell_map.append(cell_idx)

                # Batch sample tất cả midpoints
                if all_midpoints:
                    midpoint_values = list(src.sample(all_midpoints, indexes=idx_param))

                    # Gom giá trị theo cell
                    cell_valid_vals = {idx: [] for idx in needs_fallback}
                    for j, v in enumerate(midpoint_values):
                        v = v[0]
                        cell_idx = midpoint_cell_map[j]
                        if not is_nodata(v, nodata):
                            cell_valid_vals[cell_idx].append(v)

                    # Tính trung bình
                    for cell_idx, valid_vals in cell_valid_vals.items():
                        if valid_vals:
                            vals[cell_idx][band_idx] = float(sum(valid_vals) / len(valid_vals))
                            
        return vals, nodata
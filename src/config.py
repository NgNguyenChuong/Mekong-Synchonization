import os
import json


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

# ============================================================
# PATH SETTINGS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
# Cac bien moi truong cho phep chay pipeline tren nhieu khung luoi, moi lan
# mot thu muc ket qua rieng; khong dat thi giu duong dan mac dinh.
DATA_RAW = os.getenv("RAW_DIR") or os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.getenv("OUTPUT_DIR") or os.path.join(DATA_DIR, "processed")
GRID_GEOJSON = os.getenv("GRID_GEOJSON") or None

# Create directories if they don't exist
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)

# --- SHAPEFILE & GRID PATHS ---
# Input boundary is required to be exactly one .shp in data/raw.
# Fallback output path when ENABLE_GEE_FALLBACK=true.
SHAPEFILE_RAW = os.path.join(DATA_RAW, "boundary_input.shp")

# Generated artifacts
VECTOR_WORK_DIR = os.path.join(DATA_PROCESSED, "vector")
os.makedirs(VECTOR_WORK_DIR, exist_ok=True)
SHAPEFILE_CLEAN = os.path.join(VECTOR_WORK_DIR, "boundary_clean.shp")
H3_GRID_GEOJSON = os.path.join(DATA_PROCESSED, "h3_grid.geojson")

# ============================================================
# GEOSPATIAL SETTINGS
# ============================================================
CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:32648"  # UTM Zone 48N for accurate area calculation in Vietnam
BUFFER_DIST = _env_int("BUFFER_DIST", 0)  # meters
MIN_ISLAND_AREA_KM2 = _env_float("MIN_ISLAND_AREA_KM2", 600)
H3_RESOLUTION = _env_int("H3_RESOLUTION", 7)

# ============================================================
# DATASET SPECIFICATIONS
# ============================================================
DEFAULT_DATA_SPECS = {
    "rain": {
        "folder": "daily_rain",
        "col_name": "rain_mm",
        "output_file": "h3_rain_daily_filled.csv"
    },
    "solar": {
        "folder": "daily_solar",
        "col_name": "solar",
        "output_file": "h3_solar_daily_filled.csv"
    },
    "temp_avg": {
        "folder": "daily_temp_avg",
        "col_name": "temp_c",
        "output_file": "h3_temp_daily_filled.csv"
    },
    "temp_max": {
        "folder": "daily_temp_max",
        "col_name": "temp_max_c",
        "output_file": "h3_temp_max_daily_filled.csv"
    },
    "temp_min": {
        "folder": "daily_temp_min",
        "col_name": "temp_min_c",
        "output_file": "h3_temp_min_daily_filled.csv"
    },
    "humidity": {
        "folder": "daily_humid",
        "col_name": "rh_percent",
        "output_file": "h3_rh_daily_filled.csv"
    }
}

DATA_SPECS_OVERRIDE_FILE = os.path.join(DATA_RAW, "dataset_specs.json")


def load_data_specs():
    if not os.path.exists(DATA_SPECS_OVERRIDE_FILE):
        return DEFAULT_DATA_SPECS

    try:
        with open(DATA_SPECS_OVERRIDE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Invalid dataset spec file: {DATA_SPECS_OVERRIDE_FILE}. Error: {exc}") from exc

    if not isinstance(data, dict) or not data:
        raise ValueError(f"{DATA_SPECS_OVERRIDE_FILE} must be a non-empty object.")

    required_fields = {"folder", "col_name", "output_file"}
    for key, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset '{key}' must be an object.")
        missing = required_fields - set(spec.keys())
        if missing:
            raise ValueError(f"Dataset '{key}' is missing required fields: {sorted(missing)}")

    return data


DATA_SPECS = load_data_specs()

# ============================================================
# STATIC FEATURES SPECIFICATIONS (BIẾN TĨNH)
# ============================================================

DEFAULT_STATIC_SPECS = {
    "dem": {
        "folder": "dem",
        "file": "DEM_DBSCL.tif",
        "col_name": "dem_mean",
        "output_file": "h3_dem.csv",
        "method": "mean"  # Phương pháp tính (mean, max, min...)
    },
    "landcover": {
        "folder": "landcover",
        "file": "LandCover_DBSCL_2021.tif",
        "col_name": "landcover_class",
        "output_file": "h3_landcover.csv",
        "method": "all_classes"  ,
        "class_names": {
            10: "Trees",        # Cây cối / Rừng thông thường
            20: "Shrubland",    # Cây bụi
            30: "Grassland",    # Đồng cỏ
            40: "Cropland",     # Đất nông nghiệp (Lúa, hoa màu...)
            50: "Built_up",     # Đô thị / Khu dân cư
            60: "Bareland",     # Đất trống
            80: "Water",        # Nước (Sông, hồ, biển)
            90: "Wetland",      # Đất ngập nước (Đầm lầy cỏ)
            95: "Mangroves"     # Rừng ngập mặn (Đặc trưng vùng ven biển ĐBSCL)
        }
    },
    "river": {
        "folder": "river",
        "file": "River_DBSCL.tif",
        "col_name": "river_proximity",
        "output_file": "h3_river.csv",
        "method": "min_distance"  # Tính khoảng cách đến sông gần nhất
    }

}
STATIC_SPECS_OVERRIDE_FILE = os.path.join(DATA_RAW, "static_specs.json")

def load_static_specs():
    """Load cấu hình biến tĩnh từ file JSON nếu có, không thì dùng Default"""
    if not os.path.exists(STATIC_SPECS_OVERRIDE_FILE):
        return DEFAULT_STATIC_SPECS

    try:
        with open(STATIC_SPECS_OVERRIDE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Invalid static spec file: {STATIC_SPECS_OVERRIDE_FILE}. Error: {exc}") from exc

    if not isinstance(data, dict) or not data:
        raise ValueError(f"{STATIC_SPECS_OVERRIDE_FILE} must be a non-empty object.")

    # Chú ý: Biến tĩnh yêu cầu trường 'file' thay vì chỉ 'folder'
    required_fields = {"folder", "file", "col_name", "output_file"}
    
    for key, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Static Dataset '{key}' must be an object.")
        
        missing = required_fields - set(spec.keys())
        if missing:
            raise ValueError(f"Static Dataset '{key}' is missing required fields: {sorted(missing)}")
            
        # Gán giá trị mặc định cho method nếu người dùng không điền trong file JSON
        if "method" not in spec:
            spec["method"] = "mean"

    return data

# Gọi hàm để khởi tạo cấu hình
STATIC_SPECS = load_static_specs()



# ============================================================
# PERIODIC FEATURES SPECIFICATIONS (DỮ LIỆU THEO CHU KỲ - VD: Sentinel-2)
# ============================================================
# Dùng cho dữ liệu thay đổi theo thời gian nhưng KHÔNG PHẢI hàng ngày
# Ví dụ: NDVI mỗi 16 ngày, ảnh vệ tinh mỗi 5 ngày...

# TAT CO Y (2026-10-03) — RO RI NHAN: nhan do man Zenodo = 28.013*exp(-13.39*NIR),
# NIR la phan xa Landsat B5 trong mua kho (Nguyen et al. 2020; Wang et al. 2025).
# NDVI (NIR+Red) va NDWI (NIR+SWIR) tu Sentinel-2/MODIS do CUNG dai luong phan xa NIR
# be mat trong cung khoang thoi gian -> mo hinh hoc lai cong thuc nhan, sai so dep gia tao.
# Khong dua dac trung quang hoc nao do cung dai luong voi band trong cong thuc nhan,
# cung khoang thoi gian, vao bo dac trung chinh (ke ca qua periodic_specs.json).
# Chi dung trong thi nghiem kiem ro ri co doi chung (oracle, NDVI lech mua) — xem NHAT_KY.
DEFAULT_PERIODIC_SPECS = {
    # "sentinel_ndvi": {
    #     "folder": "sentinel2_ndvi",
    #     "file_pattern": "NDVI_{date}.tif",
    #     "date_pattern": "%Y-%m-%d",
    #     "col_name": "ndvi",
    #     "output_file": "h3_sentinel_ndvi.csv",
    #     "method": "mean",
    #     "typical_interval_days": 30
    # },
    # "sentinel_ndwi": {
    #     "folder": "sentinel2_ndvi",
    #     "file_pattern": "NDWI_{date}.tif",
    #     "date_pattern": "%Y-%m-%d",
    #     "col_name": "ndwi",
    #     "output_file": "h3_sentinel_ndwi.csv",
    #     "method": "mean",
    #     "typical_interval_days": 30
    # },
}

PERIODIC_SPECS_OVERRIDE_FILE = os.path.join(DATA_RAW, "periodic_specs.json")

def load_periodic_specs():
    """Load cấu hình biến periodic từ file JSON nếu có"""
    if not os.path.exists(PERIODIC_SPECS_OVERRIDE_FILE):
        return DEFAULT_PERIODIC_SPECS

    try:
        with open(PERIODIC_SPECS_OVERRIDE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Invalid periodic spec file: {PERIODIC_SPECS_OVERRIDE_FILE}. Error: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{PERIODIC_SPECS_OVERRIDE_FILE} must be an object.")

    if not data:
        return DEFAULT_PERIODIC_SPECS

    required_fields = {"folder", "col_name", "output_file"}

    for key, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Periodic Dataset '{key}' must be an object.")

        missing = required_fields - set(spec.keys())
        if missing:
            raise ValueError(f"Periodic Dataset '{key}' is missing required fields: {sorted(missing)}")

        # Gán giá trị mặc định
        spec.setdefault("date_pattern", "%Y-%m-%d")
        spec.setdefault("file_pattern", "{date}.tif")
        spec.setdefault("method", "mean")
        spec.setdefault("typical_interval_days", 16)

    return data


PERIODIC_SPECS = load_periodic_specs()


"""Trich dac trung tu raster theo tung o luoi va gop thanh bang.

Moi bien (dong, tinh, dinh ky) deu trich bang TRUNG BINH CO TRONG SO THEO DIEN
TICH PHU (exactextract): hinh dang o anh huong truc tiep den gia tri o nhan,
dung cho so sanh cac khung luoi. Pixel NaN/NoData bi bo qua; o khong co pixel
hop le nhan NaN (khong dien 0).
"""
import glob as glob_module
import os
import re
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from scipy.spatial import cKDTree

from config import DATA_PROCESSED, DATA_RAW, DATA_SPECS, STATIC_SPECS, PERIODIC_SPECS
from utils_h3 import index_files


# ---------------------------------------------------------
# CORE: TRUNG BINH THEO DIEN TICH
# ---------------------------------------------------------
def _cells_gdf(cell_data, crs):
    cell_ids, _, geoms = cell_data
    gdf = gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=geoms, crs="EPSG:4326")
    return gdf.to_crs(crs)


def area_weighted_means(tif_path, cell_data):
    """Trung binh co trong so dien tich cho moi band.

    Tra ve DataFrame: cot cell_id + mot cot moi band theo thu tu band (1..n).
    """
    with rasterio.open(tif_path) as ds:
        crs, n_bands = ds.crs, ds.count
    gdf = _cells_gdf(cell_data, crs)
    res = exact_extract(tif_path, gdf, ["mean"], include_cols=["cell_id"], output="pandas")
    if n_bands == 1:
        value_cols = ["mean"]
    else:
        value_cols = [f"band_{i}_mean" for i in range(1, n_bands + 1)]
    out = res[["cell_id"] + value_cols].copy()
    out.columns = ["cell_id"] + [f"b{i}" for i in range(1, n_bands + 1)]
    return out


def area_weighted_class_fractions(tif_path, cell_data):
    """Ty le dien tich tung lop (co trong so) trong moi o. O khong co pixel hop le -> {}."""
    with rasterio.open(tif_path) as ds:
        crs = ds.crs
    gdf = _cells_gdf(cell_data, crs)
    res = exact_extract(tif_path, gdf, ["unique", "frac"], include_cols=["cell_id"], output="pandas")
    fractions = []
    for classes, fracs in zip(res["unique"], res["frac"]):
        fractions.append({int(c): float(f) for c, f in zip(classes, fracs)})
    return list(res["cell_id"]), fractions


# ---------------------------------------------------------
# DU LIEU DONG (moi file = 1 thang, band k = ngay k)
# ---------------------------------------------------------
def extract_generic(spec_name, spec_config, cell_data, raw_root_dir):
    input_dir = os.path.join(raw_root_dir, spec_config["folder"])
    col_name = spec_config["col_name"]

    file_map = index_files(input_dir)
    if not file_map:
        return pd.DataFrame()

    parts = []
    for (year, month), tif_path in sorted(file_map.items()):
        wide = area_weighted_means(tif_path, cell_data)
        long = wide.melt(id_vars="cell_id", var_name="band", value_name=col_name)
        day_offset = long["band"].str[1:].astype(int) - 1
        long["date"] = (pd.Timestamp(year, month, 1) + pd.to_timedelta(day_offset, unit="D")).dt.strftime("%Y-%m-%d")
        parts.append(long[["cell_id", "date", col_name]])
    return pd.concat(parts, ignore_index=True)


def process_single_dataset(args):
    key, spec, cell_data = args
    print(f" [Start] {key.upper()} processing...")
    df = extract_generic(key, spec, cell_data, DATA_RAW)
    if df.empty:
        print(f" [Skip] {key.upper()} - No data found.")
        return key
    out_path = os.path.join(DATA_PROCESSED, spec["output_file"])
    df.to_csv(out_path, index=False)
    print(f"[Done] {key.upper()} saved -> {out_path}")
    return key


def _read_cell_csv(file_path):
    df = pd.read_csv(file_path)
    if "h3_index" in df.columns and "cell_id" not in df.columns:
        df = df.rename(columns={"h3_index": "cell_id"})
    df["cell_id"] = df["cell_id"].astype(str)
    return df


def _merge(specs, keys, output_name, label):
    dfs = []
    for key, spec in specs.items():
        file_path = os.path.join(DATA_PROCESSED, spec["output_file"])
        if not os.path.exists(file_path):
            print(f"[Warning] {label}: bo qua {key}, khong co {spec['output_file']}")
            continue
        dfs.append(_read_cell_csv(file_path).set_index(keys))
    if not dfs:
        print(f"[Warning] {label}: khong co du lieu de gop.")
        return None
    merged = pd.concat(dfs, axis=1, join="outer").reset_index()
    out_path = os.path.join(DATA_PROCESSED, output_name)
    merged.to_csv(out_path, index=False)
    print(f"[Done] {label}: {out_path} {merged.shape}")
    return merged


def merge_dynamic_datasets():
    return _merge(DATA_SPECS, ["cell_id", "date"], "DYNAMIC_MERGE.csv", "MERGE-DYNAMIC")


# ---------------------------------------------------------
# DU LIEU TINH (DEM, lop phu, song)
# ---------------------------------------------------------
def extract_static_generic(spec_name, spec_config, cell_data, raw_root_dir):
    cell_ids = cell_data[0]
    file_path = os.path.join(raw_root_dir, spec_config["folder"], spec_config["file"])
    col_name = spec_config["col_name"]
    method = spec_config.get("method", "mean")

    if not os.path.exists(file_path):
        print(f"Khong tim thay file: {file_path}")
        return pd.DataFrame()

    if method == "all_classes":
        class_map = spec_config.get("class_names", {})
        ids, fractions = area_weighted_class_fractions(file_path, cell_data)
        codes = sorted({c for f in fractions for c in f})
        rows = []
        for f in fractions:
            # O khong co pixel hop le: de NaN, khong gan 0% cho moi lop.
            rows.append({code: (f.get(code, 0.0) if f else np.nan) for code in codes})
        df = pd.DataFrame(rows, columns=codes)
        df.columns = [f"{col_name}_{class_map.get(c, str(c))}" for c in codes]
        df.insert(0, "cell_id", ids)
        return df

    if method == "min_distance":
        water = area_weighted_means(file_path, cell_data)["b1"].to_numpy()
        water_threshold = 0.40
        is_water = np.nan_to_num(water, nan=0.0) > water_threshold
        if not is_water.any():
            print("   Canh bao: khong co o nao co ty le nuoc > 40%.")
            dist_km = np.full(len(cell_ids), np.nan)
        else:
            gdf = _cells_gdf(cell_data, "EPSG:4326")
            metric = gdf.to_crs(gdf.estimate_utm_crs())
            xy = np.column_stack((metric.geometry.centroid.x, metric.geometry.centroid.y))
            dist_m, _ = cKDTree(xy[is_water]).query(xy, k=1)
            dist_km = np.where(is_water, 0.0, dist_m / 1000.0)
        return pd.DataFrame({"cell_id": cell_ids, col_name: dist_km, f"{col_name}_fraction": water})

    if method != "mean":
        raise ValueError(f"{spec_name}: chi ho tro method 'mean', 'all_classes', 'min_distance' (nhan '{method}')")
    return pd.DataFrame({"cell_id": cell_ids, col_name: area_weighted_means(file_path, cell_data)["b1"].to_numpy()})


def process_single_static_dataset(args):
    key, spec, cell_data = args
    print(f"[Start] STATIC {key.upper()} processing...")
    df = extract_static_generic(key, spec, cell_data, DATA_RAW)
    if df.empty:
        print(f"[Skip] STATIC {key.upper()} - khong tim thay {spec['file']}.")
        return key
    out_path = os.path.join(DATA_PROCESSED, spec["output_file"])
    df.to_csv(out_path, index=False)
    print(f"[Done] STATIC {key.upper()} saved -> {out_path}")
    return key


def merge_static_datasets():
    return _merge(STATIC_SPECS, ["cell_id"], "STATIC_MERGED.csv", "MERGE-STATIC")


# ---------------------------------------------------------
# DU LIEU DINH KY (NDVI, NDWI, nhan Zenodo: moi file = 1 moc thoi gian)
# ---------------------------------------------------------
def index_periodic_files(input_dir, file_pattern, date_pattern):
    """Index file theo ngay: {datetime: duong_dan}."""
    regex = re.compile(
        file_pattern.replace(".", r"\.").replace("{date}", r"(?P<date>\d{4}[-_]?\d{2}[-_]?\d{2})")
    )
    file_map = {}
    for fpath in glob_module.glob(os.path.join(input_dir, "*.tif")):
        match = regex.match(os.path.basename(fpath))
        if not match:
            continue
        date_str = match.group("date").replace("_", "-")
        try:
            file_map[datetime.strptime(date_str, date_pattern)] = fpath
        except ValueError:
            file_map[datetime.strptime(date_str.replace("-", ""), "%Y%m%d")] = fpath
    return file_map


def extract_periodic_generic(spec_name, spec_config, cell_data, raw_root_dir, from_date=None, to_date=None):
    input_dir = os.path.join(raw_root_dir, spec_config["folder"])
    col_name = spec_config["col_name"]
    method = spec_config.get("method", "mean")
    if method != "mean":
        raise ValueError(f"{spec_name}: du lieu dinh ky chi ho tro method 'mean' (nhan '{method}')")

    file_map = index_periodic_files(
        input_dir, spec_config.get("file_pattern", "{date}.tif"), spec_config.get("date_pattern", "%Y-%m-%d")
    )
    file_map = {
        dt: p for dt, p in file_map.items()
        if (from_date is None or dt >= from_date) and (to_date is None or dt <= to_date)
    }
    if not file_map:
        print(f"   Khong tim thay file dinh ky nao trong {input_dir}")
        return pd.DataFrame()

    parts = []
    for dt, tif_path in sorted(file_map.items()):
        means = area_weighted_means(tif_path, cell_data)
        parts.append(pd.DataFrame({"cell_id": means["cell_id"], "date": dt.strftime("%Y-%m-%d"), col_name: means["b1"]}))
    return pd.concat(parts, ignore_index=True)


def _month_range(year, month):
    start = datetime(year, month, 1)
    end = datetime(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return start, end


def process_single_periodic_dataset(args):
    key, spec, cell_data, options = args
    print(f"[Start] PERIODIC {key.upper()} processing...")
    from_date = to_date = None
    if options.get("from_date"):
        from_date = _month_range(*options["from_date"])[0]
    if options.get("to_date"):
        to_date = _month_range(*options["to_date"])[1]
    if options.get("single_month"):
        from_date, to_date = _month_range(*options["single_month"])

    df = extract_periodic_generic(key, spec, cell_data, DATA_RAW, from_date, to_date)
    if df.empty:
        print(f"[Skip] PERIODIC {key.upper()} - khong co du lieu.")
        return key
    out_path = os.path.join(DATA_PROCESSED, spec["output_file"])
    df.to_csv(out_path, index=False)
    print(f"[Done] PERIODIC {key.upper()} saved -> {out_path}")
    return key


def merge_periodic_datasets():
    if not PERIODIC_SPECS:
        print("[Warning] Khong co periodic dataset nao duoc cau hinh.")
        return None
    return _merge(PERIODIC_SPECS, ["cell_id", "date"], "PERIODIC_MERGED.csv", "MERGE-PERIODIC")

import os
import pandas as pd
import numpy as np
import h3
import rasterio
import geopandas as gpd
from rasterstats import zonal_stats
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
from utils_h3 import index_files, sample_multiband_robust
from config import DATA_PROCESSED, DATA_RAW,DATA_SPECS, STATIC_SPECS, PERIODIC_SPECS
import re
import glob as glob_module

# ---------------------------------------------------------
# CORE LOGIC: EXTRACT
# ---------------------------------------------------------
def extract_generic(spec_name, spec_config, h3_data, raw_root_dir):
    h3_ids, point_groups, h3_geoms = h3_data
    input_dir = os.path.join(raw_root_dir, spec_config["folder"])
    col_name = spec_config["col_name"]
    
    file_map = index_files(input_dir)
    if not file_map:
        return pd.DataFrame()

    records = []
    sorted_items = sorted(file_map.items())
    
    # print(f"    Extracting {spec_name}...") 
    
    for (year, month), tif_path in sorted_items:
        # Lấy mẫu dữ liệu (Robust sampling)
        vals, _ = sample_multiband_robust(tif_path, point_groups, h3_geoms, n_random=15)
        num_days = len(vals[0]) if vals and vals[0] else 0
        
        for d in range(num_days):
            current_date = datetime(year, month, 1) + timedelta(days=d)
            date_str = current_date.strftime("%Y-%m-%d")
            
            for i, h3_id in enumerate(h3_ids):
                records.append({
                    "cell_id": h3_id,
                    "date": date_str,
                    col_name: vals[i][d]
                })
                
    return pd.DataFrame(records)


# bỏ tính năng fill nodata.
# ---------------------------------------------------------
# CORE LOGIC: SPATIAL FILL (K-RING) - OPTIMIZED
# ---------------------------------------------------------
# def fill_spatial_generic(df, col_name):
#     if df.empty: return df

#     # Xử lý giá trị rác -9999 thành NaN
#     df[col_name] = df[col_name].replace(-9999.0, np.nan)

#     # Nếu không còn thiếu thì trả về luôn
#     if not df[col_name].isna().any():
#         return df

#     h3_ids = df["h3_index"].unique()
#     max_k = FILL_CONFIG["MAX_K"]
#     min_nei = FILL_CONFIG["MIN_NEI"]

#     # Precompute k-ring cho tất cả các ô
#     try:
#         k_ring_map = {h: {k: list(h3.grid_disk(h, k)) for k in range(1, max_k + 1)} for h in h3_ids}
#     except AttributeError:
#         k_ring_map = {h: {k: list(h3.k_ring(h, k)) for k in range(1, max_k + 1)} for h in h3_ids}

#     # OPTIMIZED: Tạo dict tra cứu nhanh bằng vectorized indexing
#     df_indexed = df.set_index(['h3_index', 'date'])[col_name]
#     val_map = df_indexed.to_dict()

#     # Tìm các ô cần fill
#     missing_mask = df[col_name].isna()
#     missing_rows = df.loc[missing_mask, ['h3_index', 'date']].values

#     # Chỉ xử lý các ô thiếu dữ liệu
#     for h, date in missing_rows:
#         key = (h, date)

#         for k in range(1, max_k + 1):
#             neighs = k_ring_map.get(h, {}).get(k, [])
#             valid_vals = [val_map[(n, date)] for n in neighs
#                          if (n, date) in val_map and pd.notna(val_map[(n, date)])]

#             if len(valid_vals) >= min_nei:
#                 val_map[key] = sum(valid_vals) / len(valid_vals)
#                 break

#     # OPTIMIZED: Cập nhật lại DataFrame bằng MultiIndex lookup
#     keys = list(zip(df['h3_index'], df['date']))
#     df[col_name] = [val_map.get(k) for k in keys]

#     return df

# ---------------------------------------------------------
# CORE LOGIC: FINAL FILL (NEAREST NEIGHBOR) - OPTIMIZED
# ---------------------------------------------------------
# def fill_final_nearest(df, col_name):
#     """
#     Chiến lược cứu hộ cuối cùng (Last Resort):
#     Tìm 1 ô hàng xóm gần nhất (Nearest Neighbor) trên toàn bản đồ có dữ liệu
#     và sao chép dữ liệu sang ô bị thiếu.
#     Dùng cho các đảo xa hoặc vùng mây quá lớn mà Spatial Fill (K-Ring) bó tay.
#     """
#     # 1. Kiểm tra xem còn thiếu không
#     missing_mask = df[col_name].isna()
#     if not missing_mask.any():
#         return df

#     print(f"   🔧 [Final Fill] Found {missing_mask.sum()} isolated cells. Running Nearest Neighbor...")

#     # 2. Tách dữ liệu Tốt (Nguồn) và Xấu (Đích)
#     bad_ids = df[missing_mask]["h3_index"].unique()
#     all_ids = df["h3_index"].unique()
#     # Good ids là những ô KHÔNG nằm trong bad_ids
#     good_ids = list(set(all_ids) - set(bad_ids))

#     if not good_ids:
#         return df

#     # 3. Xây dựng cây tìm kiếm khoảng cách (KDTree)
#     # Hỗ trợ cả H3 v3 và v4
#     try:
#         good_coords = [h3.cell_to_latlng(h) for h in good_ids]
#         bad_coords = [h3.cell_to_latlng(h) for h in bad_ids]
#     except AttributeError:
#         good_coords = [h3.h3_to_geo(h) for h in good_ids]
#         bad_coords = [h3.h3_to_geo(h) for h in bad_ids]

#     tree = cKDTree(good_coords)
#     # Tìm 1 điểm gần nhất (k=1)
#     dists, indices = tree.query(bad_coords, k=1)

#     # 4. Map ô Xấu -> ô Tốt gần nhất
#     rescue_map = {bad_ids[i]: good_ids[indices[i]] for i in range(len(bad_ids))}

#     # 5. OPTIMIZED: Điền dữ liệu bằng vectorized merge thay vì apply
#     # Tạo DataFrame tra cứu dữ liệu tốt
#     df_good = df[df["h3_index"].isin(good_ids)][['h3_index', 'date', col_name]].copy()
#     df_good = df_good.rename(columns={'h3_index': 'source_h3', col_name: 'source_val'})

#     # Tạo cột source_h3 cho các ô bad
#     df['source_h3'] = df['h3_index'].map(rescue_map)

#     # Merge để lấy giá trị từ ô nguồn
#     df = df.merge(
#         df_good,
#         left_on=['source_h3', 'date'],
#         right_on=['source_h3', 'date'],
#         how='left'
#     )

#     # Chỉ fill những ô đang thiếu
#     fill_mask = df[col_name].isna() & df['source_val'].notna()
#     df.loc[fill_mask, col_name] = df.loc[fill_mask, 'source_val']

#     # Xóa cột tạm
#     df = df.drop(columns=['source_h3', 'source_val'])

#     # Fill nốt các lỗ hổng thời gian nếu ô nguồn cũng bị thiếu 1 vài ngày
#     if df[col_name].isna().any():
#         df[col_name] = df[col_name].interpolate(method='linear', limit_direction='both')

#     return df

# ---------------------------------------------------------
# WORKER FUNCTION (Cho Multiprocessing)
# ---------------------------------------------------------
def process_single_dataset(args):
    """
    Hàm worker chạy trên Process riêng.
    Args: (key, spec, h3_data_bundle)
    """
    key, spec, h3_data_bundle = args
    print(f" [Start] {key.upper()} processing...")
    
    try:
        # 1. Extract
        df = extract_generic(key, spec, h3_data_bundle, DATA_RAW)
        if df.empty:
            print(f" [Skip] {key.upper()} - No data found.")
            return key
            
        # 2. Check Missing & Cleanup (disabled)
        # col_name = spec["col_name"]
        # # Convert rác thành NaN trước khi check
        # if col_name in df.columns:
        #     df[col_name] = df[col_name].replace(-9999.0, np.nan)
        #     nan_raw = df[col_name].isna().sum()
        # else:
        #     nan_raw = 0
        # 
        # # 3. Fill Strategy (disabled)
        # if nan_raw > 0:
        #     # Bước A: Spatial Fill (Hàng xóm lân cận)
        #     df = fill_spatial_generic(df, col_name)
        # 
        #     # Bước B: Final Fill (Hàng xóm gần nhất toàn cục - Cứu đảo xa)
        #     nan_remaining = df[col_name].isna().sum()
        #     if nan_remaining > 0:
        #         df = fill_final_nearest(df, col_name)
        # 
        # 4. Save
        out_path = os.path.join(DATA_PROCESSED, spec["output_file"])
        df.to_csv(out_path, index=False)
        print(f"✅ [Done] {key.upper()} saved.")
        
    except Exception as e:
        print(f"❌ [Error] {key.upper()}: {e}")
        raise e
        
    return key


# ---------------------------------------------------------
# MERGE FUNCTION
# ---------------------------------------------------------
from config import DATA_SPECS, DATA_PROCESSED

def merge_dynamic_datasets():
    """
    Gộp tất cả các file CSV thành phần (Mưa, Nhiệt, Ẩm...) thành 1 file tổng.
    Join key: ['cell_id', 'date']
    """
    print("🔄 [MERGE] Bắt đầu gộp các file dữ liệu...")
    
    # 1. Tập hợp danh sách file từ Config
    # Chỉ lấy những key có trong DATA_SPECS
    dfs = []
    
    for key, spec in DATA_SPECS.items():
        file_path = os.path.join(DATA_PROCESSED, spec["output_file"])
        
        if not os.path.exists(file_path):
            print(f"⚠️ [Warning] File không tồn tại, bỏ qua: {spec['output_file']}")
            continue
            
        print(f"   📖 Reading {key}...")
        # Đọc file, ép kiểu h3_index về string để tránh lỗi merge
        df = pd.read_csv(file_path)
        if 'h3_index' in df.columns and 'cell_id' not in df.columns:
            df.rename(columns={'h3_index': 'cell_id'}, inplace=True)
        df['cell_id'] = df['cell_id'].astype(str)
        
        # Đảm bảo cột date đúng định dạng
        # df['date'] = pd.to_datetime(df['date']) 
        
        # Set index là (h3_index, date) để chuẩn bị merge
        df = df.set_index(['cell_id', 'date'])
        dfs.append(df)
    
    if not dfs:
        print("❌ Không tìm thấy file dữ liệu nào để gộp!")
        return

    # 2. Thực hiện Merge (Join)
    # Dùng concat axis=1 sẽ nhanh hơn merge từng lần
    print("   🔗 Joining dataframes...")
    try:
        final_df = pd.concat(dfs, axis=1, join='outer')
        
        # Reset index để đưa h3_index và date trở lại thành cột
        final_df = final_df.reset_index()
        
        # 3. Lưu kết quả
        output_csv = os.path.join(DATA_PROCESSED, "DYNAMIC_MERGE.csv")
        # output_parquet = os.path.join(DATA_PROCESSED, "FINAL_MERGED_DATASET.parquet")
        
        print(f"   💾 Saving to {output_csv}...")
        final_df.to_csv(output_csv, index=False)
        # final_df.to_parquet(output_parquet, index=False) # Khuyên dùng Parquet nếu file lớn
        
        print(f"✅ [DONE] Đã gộp thành công! Kích thước: {final_df.shape}")
        return final_df
        
    except Exception as e:
        print(f"❌ [Error] Lỗi khi gộp file: {e}")
        return None

def extract_static_generic(spec_name, spec_config, h3_data_bundle, raw_root_dir):
    h3_ids, _, h3_geoms = h3_data_bundle
    
    folder_path = os.path.join(raw_root_dir, spec_config["folder"])
    file_path = os.path.join(folder_path, spec_config["file"])
    col_name = spec_config["col_name"]
    method = spec_config.get("method", "mean")

    if not os.path.exists(file_path):
        print(f"Không tìm thấy file: {file_path}")
        return pd.DataFrame()

    gdf = gpd.GeoDataFrame({"cell_id": h3_ids}, geometry=h3_geoms, crs="EPSG:4326")

    # -------------------------------------------------------------
    # LOGIC 1: Lấy TẤT CẢ các lớp dưới dạng Tỷ lệ % (Fraction) VÀ GHI TÊN
    # -------------------------------------------------------------
    if method == "all_classes":
        print(f"   🌱 Đang tính toán tỷ lệ % cho tất cả các lớp {col_name}...")
        stats = zonal_stats(gdf, file_path, categorical=True)
        
        # Lấy từ điển dịch tên class từ config (nếu không có thì trả về dict rỗng)
        class_map = spec_config.get("class_names", {})
        
        records = []
        for s in stats:
            if not s: 
                records.append({})
            else:
                total_pixels = sum(s.values())
                row_data = {}
                for k, v in s.items():
                    # Tìm tên trong từ điển. Nếu file có mã lạ (ví dụ 99) mà chưa cấu hình, nó sẽ để tên là "99"
                    class_name = class_map.get(int(k), str(int(k)))
                    
                    # Tạo tên cột, ví dụ: "landcover_Water", "landcover_Rice"
                    column_key = f"{col_name}_{class_name}"
                    
                    row_data[column_key] = round(v / total_pixels, 4)
                records.append(row_data)
                
        df_out = pd.DataFrame(records)
        df_out = df_out.fillna(0) 
        df_out["cell_id"] = h3_ids
        
        cols = ["cell_id"] + [c for c in df_out.columns if c != "cell_id"]
        return df_out[cols]


    # -------------------------------------------------------------
    # LOGIC 2: Tính khoảng cách ngắn nhất đến sông VÀ Tỷ lệ % nước
    # -------------------------------------------------------------
    elif method == "min_distance":
        print(f"   🌊 Đang tính khoảng cách tới ô H3 có tỷ lệ nước > 40%...")

        # 1. TÍNH TỶ LỆ NƯỚC CHO TỪNG Ô H3
        water_stats = zonal_stats(gdf, file_path, stats="mean")
        water_fractions = np.array([
            s['mean'] if s and s.get('mean') is not None else 0.0
            for s in water_stats
        ])

        # Ô H3 được xem là "ô nước" khi tỷ lệ nước > 40%
        water_threshold = 0.40
        has_water_mask = water_fractions > water_threshold

        # 2. ĐO KHOẢNG CÁCH TỪ MỌI Ô H3 ĐẾN Ô H3 NƯỚC GẦN NHẤT (KDTree)
        if not has_water_mask.any():
            print("   ⚠️ Cảnh báo: Không có ô H3 nào có tỷ lệ nước > 40%.")
            vals = np.full(len(h3_ids), np.nan)
        else:
            # Dùng CRS metric để khoảng cách Euclidean có ý nghĩa theo mét
            gdf_metric = gdf.to_crs(epsg=3857)
            centroids = gdf_metric.geometry.centroid
            h3_coords = np.column_stack((centroids.x, centroids.y))

            water_h3_coords = h3_coords[has_water_mask]
            tree = cKDTree(water_h3_coords)
            dists_m, _ = tree.query(h3_coords, k=1)

            # Đổi từ mét sang km
            vals = dists_m / 1000.0

            # Ép khoảng cách = 0 cho các ô đã là ô nước
            vals = np.where(has_water_mask, 0.0, vals)

        mean_dist = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
        print(
            f"   ✅ Xong! (Ngưỡng nước > {water_threshold:.0%} | "
            f"Khoảng cách TB: {mean_dist:.2f} km | "
            f"Ô nước: {has_water_mask.sum()}/{len(h3_ids)})"
        )
        
        # 4. LƯU CẢ 2 CỘT VÀO FILE H3_RIVER.CSV
        return pd.DataFrame({
            "cell_id": h3_ids,
            col_name: vals,                                # Cột 1: river_proximity (Khoảng cách - km)
            f"{col_name}_fraction": water_fractions        # Cột 2: river_proximity_fraction (Tỷ lệ % - từ 0 đến 1)
        })

    # -------------------------------------------------------------
    # LOGIC 3: Zonal Stats cơ bản (mean, max, min cho DEM...)
    # -------------------------------------------------------------
    else:
        stats = zonal_stats(gdf, file_path, stats=method)
        vals = [s[method] if s[method] is not None else np.nan for s in stats]

    return pd.DataFrame({
        "cell_id": h3_ids,
        col_name: vals
    })

def process_single_static_dataset(args):
    """Worker chạy từng biến tĩnh độc lập."""
    key, spec, h3_data_bundle = args
    print(f"🚀 [Start] STATIC {key.upper()} processing...")
    
    try:
        df = extract_static_generic(key, spec, h3_data_bundle, DATA_RAW)
        if df.empty:
            print(f"⚠️ [Skip] STATIC {key.upper()} - Không tìm thấy file {spec['file']}.")
            return key
            
        out_path = os.path.join(DATA_PROCESSED, spec["output_file"])
        df.to_csv(out_path, index=False)
        print(f"✅ [Done] STATIC {key.upper()} saved.")
        
    except Exception as e:
        print(f"❌ [Error] STATIC {key.upper()}: {e}")
        
    return key

def merge_static_datasets():
    """Gộp các file tĩnh riêng lẻ thành file STATIC_MERGED.csv.
    JOIN key: cell_id"""
    print("\n🔄 [MERGE-STATIC] Bắt đầu gộp các file dữ liệu tĩnh...")
    dfs = []

    for key, spec in STATIC_SPECS.items():
        file_path = os.path.join(DATA_PROCESSED, spec["output_file"])
        if os.path.exists(file_path):
            print(f"   📖 Reading {key}...")
            df = pd.read_csv(file_path)
        if 'h3_index' in df.columns and 'cell_id' not in df.columns:
            df.rename(columns={'h3_index': 'cell_id'}, inplace=True)
        df['cell_id'] = df['cell_id'].astype(str)
            df = df.set_index('cell_id')
            dfs.append(df)

    if dfs:
        final_static_df = pd.concat(dfs, axis=1, join='outer').reset_index()
        out_path = os.path.join(DATA_PROCESSED, "STATIC_MERGED.csv")
        final_static_df.to_csv(out_path, index=False)
        print(f"✅ [DONE] Đã lưu bảng danh mục tĩnh tại: {out_path} ({final_static_df.shape})")
    else:
        print("❌ Không có dữ liệu tĩnh nào để gộp.")


# ---------------------------------------------------------
# PERIODIC DATA PROCESSING (Sentinel-2, NDVI, v.v.)
# ---------------------------------------------------------


def index_periodic_files(input_dir, file_pattern, date_pattern):
    """
    Index các file periodic theo ngày.

    Args:
        input_dir: Thư mục chứa file
        file_pattern: Pattern tên file (vd: "NDVI_{date}.tif")
        date_pattern: Format ngày (vd: "%Y-%m-%d")

    Returns:
        dict: {datetime_obj: file_path}
    """
    file_map = {}

    # Tạo regex từ file_pattern
    # Thay {date} bằng regex capture group
    regex_pattern = file_pattern.replace("{date}", r"(?P<date>\d{4}[-_]?\d{2}[-_]?\d{2})")
    regex_pattern = regex_pattern.replace(".", r"\.")
    regex = re.compile(regex_pattern)

    # Tìm tất cả file .tif trong thư mục
    tif_files = glob_module.glob(os.path.join(input_dir, "*.tif"))

    for fpath in tif_files:
        fname = os.path.basename(fpath)
        match = regex.match(fname)

        if match:
            date_str = match.group("date")
            # Normalize date string (thay _ bằng -)
            date_str = date_str.replace("_", "-")

            try:
                dt = datetime.strptime(date_str, date_pattern)
                file_map[dt] = fpath
            except ValueError:
                # Thử các format khác
                for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y_%m_%d"]:
                    try:
                        dt = datetime.strptime(date_str.replace("-", "").replace("_", ""), "%Y%m%d")
                        file_map[dt] = fpath
                        break
                    except ValueError:
                        continue

    return file_map


def extract_periodic_generic(spec_name, spec_config, h3_data, raw_root_dir, from_date=None, to_date=None):
    """
    Extract dữ liệu periodic (Sentinel-2, NDVI, etc.)

    Khác với dynamic (daily):
    - Mỗi file là 1 ngày chụp (single band)
    - Ngày chụp không liên tục
    """
    h3_ids, point_groups, h3_geoms = h3_data
    input_dir = os.path.join(raw_root_dir, spec_config["folder"])
    col_name = spec_config["col_name"]

    file_pattern = spec_config.get("file_pattern", "{date}.tif")
    date_pattern = spec_config.get("date_pattern", "%Y-%m-%d")
    method = spec_config.get("method", "mean")

    # Index files theo ngày
    file_map = index_periodic_files(input_dir, file_pattern, date_pattern)

    if not file_map:
        print(f"   ⚠️ Không tìm thấy file periodic nào trong {input_dir}")
        return pd.DataFrame()

    # Filter theo thời gian nếu có
    if from_date or to_date:
        filtered = {}
        for dt, fpath in file_map.items():
            if from_date and dt < from_date:
                continue
            if to_date and dt > to_date:
                continue
            filtered[dt] = fpath
        file_map = filtered

    print(f"   📁 Tìm thấy {len(file_map)} file periodic cho {spec_name}")

    records = []
    sorted_items = sorted(file_map.items())

    gdf = gpd.GeoDataFrame({"cell_id": h3_ids}, geometry=h3_geoms, crs="EPSG:4326")

    for dt, tif_path in sorted_items:
        date_str = dt.strftime("%Y-%m-%d")
        print(f"      📅 Processing {date_str}...")

        # Dùng zonal_stats cho single band
        stats = zonal_stats(gdf, tif_path, stats=method)

        for i, h3_id in enumerate(h3_ids):
            val = stats[i][method] if stats[i][method] is not None else np.nan
            records.append({
                "cell_id": h3_id,
                "date": date_str,
                col_name: val
            })

    return pd.DataFrame(records)


# bỏ tính năng fill periodic data.
# def fill_periodic_data(df, col_name, fill_method="interpolate", typical_interval_days=16):
#     if fill_method == "none" or df.empty:
#         return df
#     filled_dfs = []
#     for h3_id in df["h3_index"].unique():
#         h3_df = df[df["h3_index"] == h3_id].copy()
#         h3_df = h3_df.sort_values("date")
#         if fill_method == "interpolate":
#             h3_df[col_name] = h3_df[col_name].interpolate(method='linear')
#         elif fill_method == "forward":
#             h3_df[col_name] = h3_df[col_name].ffill()
#         elif fill_method == "backward":
#             h3_df[col_name] = h3_df[col_name].bfill()
#         filled_dfs.append(h3_df)
#     return pd.concat(filled_dfs, ignore_index=True)


def process_single_periodic_dataset(args):
    """Worker xử lý 1 periodic dataset."""
    key, spec, h3_data_bundle, options = args
    print(f"🚀 [Start] PERIODIC {key.upper()} processing...")

    try:
        # Parse date options
        from_date = None
        to_date = None

        if options.get('from_date'):
            y, m = options['from_date']
            from_date = datetime(y, m, 1)
        if options.get('to_date'):
            y, m = options['to_date']
            # Cuối tháng
            if m == 12:
                to_date = datetime(y + 1, 1, 1) - timedelta(days=1)
            else:
                to_date = datetime(y, m + 1, 1) - timedelta(days=1)
        if options.get('single_month'):
            y, m = options['single_month']
            from_date = datetime(y, m, 1)
            if m == 12:
                to_date = datetime(y + 1, 1, 1) - timedelta(days=1)
            else:
                to_date = datetime(y, m + 1, 1) - timedelta(days=1)

        # 1. Extract
        df = extract_periodic_generic(key, spec, h3_data_bundle, DATA_RAW, from_date, to_date)

        if df.empty:
            print(f"⚠️ [Skip] PERIODIC {key.upper()} - Không có dữ liệu.")
            return key

        # 2. Fill (disabled)
        # fill_method = spec.get("fill_method", "none")
        # if fill_method != "none" and not options.get('no_fill', False):
        #     df = fill_periodic_data(df, spec["col_name"], fill_method, spec.get("typical_interval_days", 16))

        # 3. Save
        out_path = os.path.join(DATA_PROCESSED, spec["output_file"])
        df.to_csv(out_path, index=False)
        print(f"✅ [Done] PERIODIC {key.upper()} saved -> {out_path}")

    except Exception as e:
        print(f"❌ [Error] PERIODIC {key.upper()}: {e}")
        import traceback
        traceback.print_exc()

    return key


def merge_periodic_datasets():
    """Gộp các file periodic thành 1 file tổng."""
    print("\n🔄 [MERGE-PERIODIC] Bắt đầu gộp các file periodic...")

    if not PERIODIC_SPECS:
        print("   ⚠️ Không có periodic dataset nào được cấu hình.")
        return

    dfs = []

    for key, spec in PERIODIC_SPECS.items():
        file_path = os.path.join(DATA_PROCESSED, spec["output_file"])
        if os.path.exists(file_path):
            print(f"   📖 Reading {key}...")
            df = pd.read_csv(file_path)
        if 'h3_index' in df.columns and 'cell_id' not in df.columns:
            df.rename(columns={'h3_index': 'cell_id'}, inplace=True)
        df['cell_id'] = df['cell_id'].astype(str)
            df = df.set_index(['cell_id', 'date'])
            dfs.append(df)

    if dfs:
        final_df = pd.concat(dfs, axis=1, join='outer').reset_index()
        out_path = os.path.join(DATA_PROCESSED, "PERIODIC_MERGED.csv")
        final_df.to_csv(out_path, index=False)
        print(f"✅ [DONE] Đã lưu file periodic tổng hợp: {out_path} ({final_df.shape})")
    else:
        print("   ⚠️ Không có dữ liệu periodic nào để gộp.")
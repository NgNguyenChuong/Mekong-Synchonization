#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DGGS H3 Pipeline - Xu ly du lieu dia khong gian voi luoi H3

Chi can chay: python main.py

Cau hinh du lieu tai: src/config.py
- DATA_SPECS: Du lieu dong (mua, nhiet do, do am...)
- STATIC_SPECS: Du lieu tinh (DEM, landcover, river)
- PERIODIC_SPECS: Du lieu ve tinh (Sentinel-2, NDVI, NDWI)
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Fix Unicode on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    DATA_SPECS, STATIC_SPECS, PERIODIC_SPECS,
    H3_GRID_GEOJSON, GRID_GEOJSON, CRS_METRIC, CRS_WGS84, DATA_RAW, DATA_PROCESSED
)
from utils_h3 import load_h3_multipoints
from preprocessing import run_preprocessing
from processing import (
    process_single_dataset,
    process_single_static_dataset,
    process_single_periodic_dataset,
    merge_dynamic_datasets,
    merge_static_datasets,
    merge_periodic_datasets
)


def main():
    start_time = time.time()

    print("=" * 60)
    print("DGGS H3 PIPELINE")
    print("=" * 60)

    # =========================================================
    # STEP 0: PREPROCESSING (bo qua khi da chi dinh luoi qua GRID_GEOJSON)
    # =========================================================
    if GRID_GEOJSON:
        grid_path = GRID_GEOJSON
        print(f"[STEP 0] Dung luoi co san: {grid_path} | ket qua -> {DATA_PROCESSED}")
    else:
        try:
            run_preprocessing()
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return
        grid_path = H3_GRID_GEOJSON

    # =========================================================
    # STEP 1: LOAD GRID
    # =========================================================
    print("\n[STEP 1] Loading Grid Geometry...")

    if not os.path.exists(grid_path):
        print(f"Grid file missing: {grid_path}")
        return

    h3_data_bundle = load_h3_multipoints(grid_path, CRS_METRIC, CRS_WGS84)
    print(f"Loaded {len(h3_data_bundle[0])} cells.")

    # =========================================================
    # STEP 2: PREPARE TASKS
    # =========================================================
    print("\n[STEP 2] Preparing tasks...")

    dynamic_tasks = []
    static_tasks = []
    periodic_tasks = []

    # Dynamic datasets (daily data)
    for key, spec in DATA_SPECS.items():
        input_dir = os.path.join(DATA_RAW, spec["folder"])
        if not os.path.exists(input_dir):
            print(f"  [Skip] {key.upper()} - folder not found: {input_dir}")
            continue
        dynamic_tasks.append((key, spec, h3_data_bundle))
        print(f"  [Add] Dynamic: {key}")

    # Static datasets
    for key, spec in STATIC_SPECS.items():
        input_path = os.path.join(DATA_RAW, spec["folder"], spec["file"])
        if not os.path.exists(input_path):
            print(f"  [Skip] Static {key.upper()} - file not found: {input_path}")
            continue
        static_tasks.append((key, spec, h3_data_bundle))
        print(f"  [Add] Static: {key}")

    # Periodic datasets (satellite data)
    if PERIODIC_SPECS:
        for key, spec in PERIODIC_SPECS.items():
            input_dir = os.path.join(DATA_RAW, spec["folder"])
            if not os.path.exists(input_dir):
                print(f"  [Skip] Periodic {key.upper()} - folder not found: {input_dir}")
                continue
            # Options for periodic (no date filter by default)
            options = {'from_date': None, 'to_date': None, 'single_month': None, 'no_fill': False}
            periodic_tasks.append((key, spec, h3_data_bundle, options))
            print(f"  [Add] Periodic: {key}")

    total_tasks = len(dynamic_tasks) + len(static_tasks) + len(periodic_tasks)
    if total_tasks == 0:
        print("\nNo valid datasets found. Check data/raw folder.")
        return

    print(f"\nTotal: {len(dynamic_tasks)} dynamic, {len(static_tasks)} static, {len(periodic_tasks)} periodic")

    # =========================================================
    # STEP 3: PARALLEL PROCESSING
    # =========================================================
    max_workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"\n[STEP 3] Processing with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Process dynamic datasets
        if dynamic_tasks:
            print(f"\n  Processing {len(dynamic_tasks)} dynamic datasets...")
            list(executor.map(process_single_dataset, dynamic_tasks))

        # Process static datasets
        if static_tasks:
            print(f"\n  Processing {len(static_tasks)} static datasets...")
            list(executor.map(process_single_static_dataset, static_tasks))

        # Process periodic datasets
        if periodic_tasks:
            print(f"\n  Processing {len(periodic_tasks)} periodic datasets...")
            list(executor.map(process_single_periodic_dataset, periodic_tasks))

    # =========================================================
    # STEP 4: MERGE DATASETS
    # =========================================================
    print("\n[STEP 4] Merging datasets...")

    if dynamic_tasks:
        merge_dynamic_datasets()

    if static_tasks:
        merge_static_datasets()

    if periodic_tasks:
        merge_periodic_datasets()

    # =========================================================
    # DONE
    # =========================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"PIPELINE FINISHED in {elapsed:.1f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()

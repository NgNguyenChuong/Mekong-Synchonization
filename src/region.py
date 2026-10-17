"""Tham so theo VUNG cho pipeline nhan/dac trung: dbscl (mac dinh, duong dan cu) + 2 vung doi chung.

Vung doi chung: dau vao <DATA_ROOT>/doi_chung/<vung>/{raw,landsat,static}, dau ra
<DOI_CHUNG_OUT>/<vung>/{grids,labels,features,eval,reports} (DOI_CHUNG_OUT trong .env).
"""
import os

from settings import data_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cung cach tinh ROOT cua scripts/

DEFAULT_REGION = "dbscl"
# Ca 3 vung cung cua so mua kho 01/11 -> 29/04 (tinh bang src/seasons.py).
SEASON_WINDOW = ((11, 1), (4, 29))
WC_PAD_KM = 25  # WorldCover vung doi chung xuat them dem 25 km (fetch_doi_chung --static-buffer-km)

REGIONS = {
    "dbscl": {"label": "MD", "epsg": 32648, "landsat_suffix": "_l8_v2"},
    "mississippi": {"label": "MS", "epsg": 32615, "landsat_suffix": "_l8_sf3"},
    "nile": {"label": "NL", "epsg": 32636, "landsat_suffix": "_l8_sf3"},
}


def get_region(name=DEFAULT_REGION) -> dict:
    if name not in REGIONS:
        raise ValueError(f"Vung '{name}' khong thuoc {sorted(REGIONS)}")
    return REGIONS[name]


def crs_metric(name=DEFAULT_REGION) -> str:
    return f"EPSG:{get_region(name)['epsg']}"


def doi_chung_out() -> str:
    return (os.getenv("DOI_CHUNG_OUT", "").strip() or "C:/NCKH_Mekong/doi_chung_out").rstrip("/\\")


def region_paths(name=DEFAULT_REGION) -> dict:
    """Duong dan mac dinh theo vung; dbscl = DUNG gia tri mac dinh cu cua cac script."""
    get_region(name)
    if name == "dbscl":
        from preprocessing import CANONICAL_BOUNDARY

        repo = ROOT
        return {
            "boundary": CANONICAL_BOUNDARY,
            "grids_dir": os.path.join(repo, "data", "grids"),
            "eval_dir": os.path.join(repo, "data", "eval"),
            "report_dir": os.path.join(repo, "KE_HOACH", "ket-qua"),
            "raw_dir": os.environ.get("RAW_DIR", data_path("raw")),
            "labels_dir": data_path("labels"),
            "features_dir": data_path("features"),
            "scope": data_path("features/scope_mask_v3.tif"),
            "worldcover": data_path("gee/LandCover_DBSCL_2021_v2.tif"),
            "worldcover_static": data_path("gee/LandCover_DBSCL_2021_v2.tif"),
            "dem": data_path("gee/DEM_DBSCL.tif"),
            "dem_flags": data_path("gee/DEM_DBSCL_flags_v2.tif"),
            "jrc": None,
            "river": data_path("raw/river/river_distance_90m.tif"),
        }
    lab = REGIONS[name]["label"]
    src, out = data_path("doi_chung", name), f"{doi_chung_out()}/{name}"
    return {
        "boundary": f"{src}/boundary.geojson",
        "grids_dir": f"{out}/grids",
        "eval_dir": f"{out}/eval",
        "report_dir": f"{out}/reports",
        "raw_dir": f"{src}/raw",
        "labels_dir": f"{out}/labels",
        "features_dir": f"{out}/features",
        "scope": f"{out}/features/scope_mask_v3.tif",
        "worldcover": f"{src}/static/LandCover_{lab}_2021.tif",  # trong ranh gioi: scope/nhan
        "worldcover_static": f"{src}/static/LandCover_{lab}_2021_pad{WC_PAD_KM}.tif",  # dem 25 km nhu DBSCL v2: dac trung tinh
        "dem": f"{src}/static/DEM_{lab}.tif",
        "dem_flags": f"{src}/static/DEM_{lab}_flags.tif",
        "jrc": f"{src}/static/JRC_GSW_{lab}.tif",
        # raster khoang cach 3 band (dist_main_river_km, dist_any_water_km, dist_coast_km) - dung o buoc sau
        "river": f"{out}/features/river/river_distance_90m.tif",
    }


def landsat_file(name, kind, year) -> str:
    """Raster Landsat mua kho cua vung doi chung: kind 'ndwi' (band NDWIchen) | 'watermask' (water_freq, n_clear).

    dbscl khong dung ham nay (nhan Zenodo/GEE do script tu chon theo nam)."""
    if name == "dbscl":
        raise ValueError("dbscl: duong dan nhan theo --zenodo-dir/--gee-dir cua script")
    r = get_region(name)
    stem = {"ndwi": "NDWIchen", "watermask": "watermask"}[kind]
    return data_path("doi_chung", name, "landsat", f"{year}_{r['label']}_dry_{stem}{r['landsat_suffix']}.tif")


def add_region_arg(ap, mapping=None, extra=None, argv=None):
    """Them --region vao parser; dat lai mac dinh theo vung TRUOC khi parse.

    mapping {ten_arg: khoa region_paths | ham(paths)}; extra {ten_arg: gia tri | ham(paths)} chi cho vung doi chung.
    Mac dinh do nguoi dung truyen tren dong lenh van thang."""
    import argparse

    ap.add_argument("--region", choices=list(REGIONS), default=DEFAULT_REGION,
                    help="Vung (mac dinh dbscl = duong dan cu); mississippi/nile = vung doi chung")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--region", default=DEFAULT_REGION)
    name = pre.parse_known_args(argv)[0].region
    if name not in REGIONS:
        return  # argparse se bao loi choices
    if name == DEFAULT_REGION and not mapping:
        return  # dbscl giu nguyen mac dinh cu, khong import preprocessing (geopandas)
    p = region_paths(name)
    defaults = {k: (v(p) if callable(v) else p[v]) for k, v in (mapping or {}).items()}
    if name != DEFAULT_REGION:
        defaults.update({k: (v(p) if callable(v) else v) for k, v in (extra or {}).items()})
    ap.set_defaults(**defaults)

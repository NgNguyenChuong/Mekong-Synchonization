"""
Validate dataset truoc khi dua vao training. Moi ham tra ve so lieu kiem
tra (khong tu drop du lieu) - quyet dinh loai bo record nao la o buoc
khac, de nguoi dung luon thay ro co bao nhieu record co van de truoc khi mat di.
"""
import h3
import numpy as np
import pandas as pd


def validate_dataset(
    df: pd.DataFrame,
    target_col: str = "salinity",
    salinity_col: str | None = None,
    grid_type: str = "generic",
) -> dict:
    """Kiem tra 1 dataset training/feature, tra ve dict bao cao.

    Khong raise loi, khong drop du lieu - chi bao cao de nguoi dung tu
    quyet dinh buoc xu ly tiep theo.
    """
    if salinity_col is not None:
        target_col = salinity_col

    report: dict = {"n_rows": len(df)}

    # 1. Missing values theo tung cot.
    report["missing_by_column"] = df.isna().sum().to_dict()

    # 2. Duplicate rows (toan bo cac cot giong het nhau).
    report["duplicate_rows"] = int(df.duplicated().sum())

    # 3. Duplicate (cell_id, date) - moi cap nay chi nen co 1 ban ghi.
    cell_col = "cell_id" if "cell_id" in df.columns else ("h3_index" if "h3_index" in df.columns else None)
    if cell_col and "date" in df.columns:
        report["duplicate_cell_date"] = int(df.duplicated(subset=[cell_col, "date"]).sum())

    # 4. Invalid dates.
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        report["invalid_dates"] = int(parsed.isna().sum())
        valid_dates = parsed.dropna()
        report["temporal_coverage"] = {
            "min_date": str(valid_dates.min()) if not valid_dates.empty else None,
            "max_date": str(valid_dates.max()) if not valid_dates.empty else None,
            "n_unique_dates": int(valid_dates.nunique()),
        }

    # 5. Invalid coordinates.
    lat_col = "latitude" if "latitude" in df.columns else ("lat" if "lat" in df.columns else None)
    lon_col = "longitude" if "longitude" in df.columns else ("lon" if "lon" in df.columns else None)
    if lat_col and lon_col:
        invalid_coords = (
            ~df[lat_col].between(-90, 90) | ~df[lon_col].between(-180, 180)
        )
        report["invalid_coordinates"] = int(invalid_coords.sum())

    # 6. Invalid cell index.
    if cell_col:
        if grid_type == "h3":
            invalid_cells = df[cell_col].apply(
                lambda v: not isinstance(v, str) or not h3.is_valid_cell(v)
            )
        else:
            invalid_cells = df[cell_col].apply(
                lambda v: not isinstance(v, str) or len(str(v).strip()) == 0
            )
        report["invalid_cell_id"] = int(invalid_cells.sum())

    # 7. Target negative / NaN / infinity.
    if target_col in df.columns:
        col = pd.to_numeric(df[target_col], errors="coerce")
        report["target_negative"] = int((col < 0).sum())
        report["target_nan"] = int(col.isna().sum())
        report["target_infinite"] = int(np.isinf(col.dropna()).sum())
        # Alias tuong thich nguoc neu ten bien la salinity
        if target_col == "salinity":
            report["salinity_negative"] = report["target_negative"]
            report["salinity_nan"] = report["target_nan"]
            report["salinity_infinite"] = report["target_infinite"]

        finite = col.replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            report["target_distribution"] = {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()) if len(finite) > 1 else 0.0,
            }
            if target_col == "salinity":
                report["salinity_distribution"] = report["target_distribution"]

    # 8. NaN / infinity cho toan bo cot numeric (feature).
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report["infinite_by_column"] = {
        col: int(np.isinf(df[col]).sum()) for col in numeric_cols if np.isinf(df[col]).any()
    }

    # 9. Station coverage (neu la dataset quan trac).
    if "station_id" in df.columns:
        report["n_stations"] = int(df["station_id"].nunique())

    return report


def format_report(report: dict) -> str:
    """In bao cao validate duoi dang text de log ra console/file."""
    lines = ["=== DATA VALIDATION REPORT ==="]
    for key, value in report.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)

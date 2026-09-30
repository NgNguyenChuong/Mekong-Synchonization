"""
Training script cho model du doan do man / bien muc tieu tren khung luoi khong gian.
"""
import argparse
import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import H3_RESOLUTION  # noqa: E402
from data_validation import format_report, validate_dataset  # noqa: E402
from dataset import build_feature_dataset, merge_with_salinity_labels  # noqa: E402
from salinity.loader import load_salinity_observations  # noqa: E402
from salinity.preprocessing import clean_salinity_observations  # noqa: E402
from salinity.spatial_mapping import map_observations_to_h3  # noqa: E402
from training.baselines import ClimatologyBaseline, PersistenceBaseline  # noqa: E402
from training.evaluate import compute_metrics  # noqa: E402
from training.features import prepare_matrices  # noqa: E402
from training.split import SplitConfig, split_summary, time_based_split  # noqa: E402

# Cot khong duoc dung lam feature (id/target/leakage/spatial grid artifacts).
NON_FEATURE_COLUMNS = {
    "cell_id", "h3_index", "date", "station_id", "station_name", "source", "unit",
    "lat", "lon", "latitude", "longitude", "overlap_frac",
}

MODEL_REGISTRY = {
    "linear": lambda seed: LinearRegression(),
    "random_forest": lambda seed: RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
    "hist_gb": lambda seed: HistGradientBoostingRegressor(random_state=seed),
    # StandardScaler nam trong pipeline -> chi fit tren tap huan luyen.
    "mlp": lambda seed: make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=seed),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train mo hinh du doan do man / bien muc tieu.")
    parser.add_argument("--label-csv", default=None, help="File CSV nhan quan trac / raster")
    parser.add_argument("--salinity-csv", default=None, help="Alias tuong thich nguoc cho --label-csv")
    parser.add_argument("--target", default="salinity", help="Ten cot bien muc tieu (default: salinity)")
    parser.add_argument("--grid-type", choices=["generic", "h3", "s2", "square", "latlon"], default="generic", help="Loai khung luoi")
    parser.add_argument("--include-coords", action="store_true", default=False, help="Dua toa do tam vao tap dac trung")
    parser.add_argument("--model", choices=["climatology", "persistence", *MODEL_REGISTRY.keys()], default="linear")
    parser.add_argument("--train-end", required=True, help="Ngay cuoi tap train, YYYY-MM-DD")
    parser.add_argument("--val-end", required=True, help="Ngay cuoi tap validation, YYYY-MM-DD")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=H3_RESOLUTION)
    return parser.parse_args()


def build_training_dataset(label_csv: str, resolution: int, target_col: str = "salinity") -> pd.DataFrame:
    raw_obs = load_salinity_observations(label_csv)
    clean_obs, clean_report = clean_salinity_observations(raw_obs)
    print("--- Salinity observation cleaning report ---")
    print(clean_report)

    obs_mapped = map_observations_to_h3(clean_obs, resolution=resolution)
    feature_df = build_feature_dataset()

    merged = merge_with_salinity_labels(feature_df, obs_mapped, target_col=target_col)
    return merged


def select_feature_columns(df: pd.DataFrame, target_col: str = "salinity", include_coords: bool = False) -> list[str]:
    excluded = set(NON_FEATURE_COLUMNS) | {target_col}
    if include_coords:
        excluded -= {"lat", "lon", "latitude", "longitude"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def main():
    args = parse_args()
    label_csv = args.label_csv or args.salinity_csv
    target_col = args.target

    try:
        if label_csv:
            dataset = build_training_dataset(label_csv, args.resolution, target_col=target_col)
        else:
            dataset = build_feature_dataset()
            if target_col not in dataset.columns:
                raise ValueError(
                    f"Khong truyen --label-csv va bien muc tieu '{target_col}' khong co trong feature dataset."
                )
    except FileNotFoundError as exc:
        print(f"BLOCKED BY REAL DATA: {exc}")
        sys.exit(1)

    validation_report = validate_dataset(dataset, target_col=target_col, grid_type=args.grid_type)
    print(format_report(validation_report))

    n_before = len(dataset)
    dataset = dataset.dropna(subset=[target_col])
    print(f"Loai {n_before - len(dataset)} record khong co nhan {target_col} (khong the dung de train/eval).")
    if len(dataset) == 0:
        print(f"BLOCKED BY REAL DATA: khong co record nao co nhan {target_col} hop le.")
        sys.exit(1)

    split_cfg = SplitConfig(date_col="date", train_end=args.train_end, val_end=args.val_end, random_seed=args.seed)
    train_df, val_df, test_df = time_based_split(dataset, split_cfg)
    print("--- Data split summary ---")
    print(split_summary(train_df, val_df, test_df))

    feature_cols = select_feature_columns(train_df, target_col=target_col, include_coords=args.include_coords)
    print(f"Features duoc dung ({len(feature_cols)}): {feature_cols}")

    missing_handler = None
    if args.model == "climatology":
        model = ClimatologyBaseline().fit(train_df["date"], train_df[target_col])
        val_pred = model.predict(val_df["date"])
        test_pred = model.predict(test_df["date"])
    elif args.model == "persistence":
        cell_col = "cell_id" if "cell_id" in train_df.columns else "h3_index"
        model = PersistenceBaseline().fit(train_df[cell_col], train_df["date"], train_df[target_col])
        # Lich su = moi quan trac da biet; predict chi dung gia tri co ngay < ngay can du bao.
        full = pd.concat([train_df, val_df, test_df], ignore_index=True)
        history = PersistenceBaseline._frame(full[cell_col], full["date"], full[target_col])
        val_pred = model.predict(val_df[cell_col], val_df["date"], history)
        test_pred = model.predict(test_df[cell_col], test_df["date"], history)
    else:
        train_X, val_X, test_X, missing_handler = prepare_matrices(
            args.model, train_df[feature_cols], val_df[feature_cols], test_df[feature_cols]
        )
        model = MODEL_REGISTRY[args.model](args.seed)
        model.fit(train_X, train_df[target_col])
        val_pred = model.predict(val_X)
        test_pred = model.predict(test_X)

    val_metrics = compute_metrics(val_df[target_col], val_pred)
    test_metrics = compute_metrics(test_df[target_col], test_pred)
    print("--- Validation metrics ---")
    print(val_metrics)
    print("--- Test metrics ---")
    print(test_metrics)

    out_dir = os.path.join("artifacts", "experiments", args.experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))
    if missing_handler is not None:
        joblib.dump(missing_handler, os.path.join(out_dir, "missing_handler.joblib"))

    experiment_config = {
        "experiment_name": args.experiment_name,
        "model": args.model,
        "target": target_col,
        "grid_type": args.grid_type,
        "include_coords": args.include_coords,
        "features": feature_cols,
        "split": split_cfg.to_dict(),
        "resolution": args.resolution,
        "seed": args.seed,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)

    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()

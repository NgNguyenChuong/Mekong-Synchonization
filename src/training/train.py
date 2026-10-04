"""
Training script cho model du doan do man / bien muc tieu tren khung luoi khong gian.
"""
import argparse
import hashlib
import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import H3_RESOLUTION  # noqa: E402
from data_validation import format_report, validate_dataset  # noqa: E402
from dataset import build_feature_dataset, build_season_feature_dataset, merge_season_labels  # noqa: E402
from salinity.loader import load_salinity_observations  # noqa: E402
from salinity.preprocessing import clean_salinity_observations  # noqa: E402
from salinity.spatial_mapping import map_observations_to_h3  # noqa: E402
from seasons import season_of  # noqa: E402
from training.baselines import ClimatologyBaseline, PersistenceBaseline  # noqa: E402
from training.evaluate import compute_metrics  # noqa: E402
from training.features import (  # noqa: E402,F401  (assert_no_leak_columns re-export cho script khac)
    DEFAULT_ALLOWED_FEATURES,
    assert_no_leak_columns,
    prepare_matrices,
    read_feature_list,
    resolve_feature_list,
    select_feature_columns,
)
from training.split import (  # noqa: E402
    MAIN_HOLDOUT_SEASON,
    TEST_GROUPS,
    EvalSplitConfig,
    SplitConfig,
    assign_eval_split,
    block_cv_membership,
    block_cv_splits,
    block_cv_summary,
    season_sequential_split,
    split_summary,
    time_based_split,
)

COORD_COLUMNS = ("lat", "lon", "latitude", "longitude")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Sieu tham so mac dinh (V7 soat 2026-10-03: TAT early stopping - early stopping noi bo chon vong lap
# tren 10% dong NGAU NHIEN = chia ngau nhien gian tiep). HistGB = DE XUAT CHO AN DUYET (NHAT_KY
# 2026-10-03, muc 3); phai chot truoc lan chay `final` dau tien. Ghi de bang --model-params <json>.
DEFAULT_MODEL_PARAMS = {
    "linear": {},
    "random_forest": {"n_estimators": 200, "n_jobs": -1},
    "hist_gb": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                "l2_regularization": 1.0, "early_stopping": False},
    "mlp": {"hidden_layer_sizes": [64, 32], "early_stopping": False, "max_iter": 500},
}
MODEL_PARAMS_STATUS = "de_xuat_CHO_AN_DUYET"


def build_model(name: str, seed: int, params: dict | None = None):
    """Mo hinh sklearn voi tham so `params` (None -> DEFAULT_MODEL_PARAMS[name]) va random_state = seed."""
    p = dict(DEFAULT_MODEL_PARAMS[name] if params is None else params)
    if name == "linear":
        return LinearRegression(**p)
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, **p)
    if name == "hist_gb":
        return HistGradientBoostingRegressor(random_state=seed, **p)
    if name == "mlp":
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        # StandardScaler nam trong pipeline -> chi fit tren tap huan luyen (cua fold).
        return make_pipeline(StandardScaler(), MLPRegressor(random_state=seed, **p))
    raise ValueError(f"Mo hinh khong ho tro: {name}")


MODEL_REGISTRY = {name: (lambda seed, _n=name: build_model(_n, seed)) for name in DEFAULT_MODEL_PARAMS}


def load_model_params(path: str | None, model: str) -> tuple[dict, str]:
    """(tham so, nguon). File JSON: {"<model>": {...}, ...} hoac dict phang cho mo hinh dang chay."""
    if model not in DEFAULT_MODEL_PARAMS:
        return {}, "khong_ap_dung"
    if not path:
        return dict(DEFAULT_MODEL_PARAMS[model]), f"default:{MODEL_PARAMS_STATUS}"
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: phai la JSON object.")
    if any(k in DEFAULT_MODEL_PARAMS for k in cfg):
        if model not in cfg:
            raise ValueError(f"{path}: khong co muc cho mo hinh '{model}' (co {list(cfg)}).")
        cfg = cfg[model]
    return dict(cfg), f"file:{path}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train mo hinh du doan do man / bien muc tieu.")
    parser.add_argument("--label-csv", default=None, help="File CSV nhan quan trac / raster")
    parser.add_argument("--salinity-csv", default=None, help="Alias tuong thich nguoc cho --label-csv")
    parser.add_argument("--target", default="salinity", help="Ten cot bien muc tieu (default: salinity)")
    parser.add_argument("--grid-type", choices=["generic", "h3", "s2", "square", "latlon"], default="generic", help="Loai khung luoi")
    parser.add_argument("--include-coords", action="store_true", default=False, help="Dua toa do tam vao tap dac trung")
    feat = parser.add_mutually_exclusive_group()
    feat.add_argument("--features", nargs="+", default=None,
                      help="Danh sach CHO PHEP cot dac trung (ten hoac tien to 'x_*'). "
                           f"Mac dinh: {' '.join(DEFAULT_ALLOWED_FEATURES)}")
    feat.add_argument("--features-file", default=None, help="File danh sach cho phep, moi dong mot ten")
    parser.add_argument("--model", choices=["climatology", "persistence", *MODEL_REGISTRY.keys()], default="linear")
    parser.add_argument("--train-end", default=None,
                        help="Moc cuoi tap train: YYYY-MM-DD (bang theo ngay) hoac nam mua kho YYYY (bang theo mua)")
    parser.add_argument("--val-end", default=None,
                        help="Moc cuoi tap validation: YYYY-MM-DD hoac nam mua kho YYYY")
    parser.add_argument("--provisional-split", action="store_true", default=False,
                        help="Bang theo mua: cho phep chia TAM tuan tu theo nam (season_sequential_split). "
                             "KHONG phai thiet ke danh gia (khoi 50 km + nam giu rieng); ket qua ghi vao "
                             "provisional_test_metrics, khong dung de so khung luoi.")
    parser.add_argument("--mode", choices=["cv", "final"], default=None,
                        help="Bang theo mua, thiet ke danh gia chinh: 'cv' = CV khoi tren split=train, ghi du doan "
                             "out-of-fold (KHONG dung tap test); 'final' = fit toan bo train, cham test theo nhom.")
    parser.add_argument("--grid", default=os.getenv("GRID_GEOJSON"),
                        help="Luoi cua bang (de dung bang o -> khoi/fold); mac dinh bien moi truong GRID_GEOJSON")
    parser.add_argument("--blocks", default=os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    parser.add_argument("--cv-folds", default=os.path.join(ROOT, "data", "eval", "cv_folds.csv"),
                        help="Fold cap khoi dung chung 13 luoi (scripts/build_cv_folds.py)")
    parser.add_argument("--cell-table", default=None,
                        help="Bang o -> khoi/fold da dung (data/eval/cell_blocks/<luoi>.csv); thay cho --grid")
    parser.add_argument("--holdout-seasons", nargs="+", type=int, default=[MAIN_HOLDOUT_SEASON])
    parser.add_argument("--allow-holdout-season-override", action="store_true", default=False,
                        help=f"Cho phep --holdout-seasons KHONG chua {MAIN_HOLDOUT_SEASON} (thiet ke chinh). Ket qua "
                             "ghi config analysis_kind=phan_tich_do_nhay - khong dung de so khung chinh.")
    parser.add_argument("--cv-rule", choices=["touch", "centroid"], default="touch")
    parser.add_argument("--min-val-cells", type=int, default=5)
    parser.add_argument("--model-params", default=None,
                        help="JSON sieu tham so ({\"hist_gb\": {...}} hoac dict phang); mac dinh DEFAULT_MODEL_PARAMS")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Nhieu seed cho --mode cv|final (mac dinh: [--seed])")
    parser.add_argument("--resolution", type=int, default=H3_RESOLUTION)
    return parser.parse_args()


def load_season_labels(label_csv: str, resolution: int) -> pd.DataFrame:
    """Nhan don vi (cell_id, season).

    - CSV co `cell_id` (nhan trich theo o): dung truc tiep; thieu `season` thi suy tu date/datetime.
    - CSV quan trac tram (schema salinity/, co lat/lon): lam sach + map sang o H3 (luu y V8:
      chi dung voi luoi H3) roi suy season tu datetime.
    Ngay ngoai mua kho bi bo. Nhieu nhan cung (o, mua) -> merge_season_labels bao loi, khong tu
    lay trung binh (cach gop nhan tram la quyet dinh nghiep vu chua chot).
    """
    head = pd.read_csv(label_csv, nrows=0).columns
    if "cell_id" in head:
        labels = pd.read_csv(label_csv)
    else:
        raw_obs = load_salinity_observations(label_csv)
        labels, clean_report = clean_salinity_observations(raw_obs)
        print("--- Salinity observation cleaning report ---")
        print(clean_report)
        labels = map_observations_to_h3(labels, resolution=resolution)

    if "season" not in labels.columns:
        time_col = "date" if "date" in labels.columns else ("datetime" if "datetime" in labels.columns else None)
        if time_col is None:
            raise ValueError(f"{label_csv}: nhan can cot 'season' hoac 'date'/'datetime'.")
        labels = labels.copy()
        labels["season"] = season_of(labels[time_col]).to_numpy()
    n_out = int(labels["season"].isna().sum())
    if n_out:
        print(f"Bo {n_out} nhan ngoai mua kho (01/11..29/04).")
    return labels[labels["season"].notna()].reset_index(drop=True)


def build_training_dataset(label_csv: str, resolution: int, target_col: str = "salinity") -> pd.DataFrame:
    """Bang huan luyen don vi (cell_id, season): dac trung gop mua + nhan ghep chinh xac theo khoa."""
    labels = load_season_labels(label_csv, resolution)
    features = build_season_feature_dataset()
    merged = merge_season_labels(features, labels, target_col=target_col)
    print(f"Ghep nhan mua: {len(labels)} nhan, {len(features)} dong dac trung (o, mua), "
          f"{int(merged[target_col].notna().sum())} dong co nhan.")
    return merged


def resolve_allowed_features(args, target_col: str, columns) -> tuple[list[str], tuple[str, ...], str]:
    """(danh sach cho phep, cot cam duoc phep co y, nguon danh sach) tu tham so CLI.

    `--include-coords`: them cac cot toa do CO trong bang; khong co cot nao -> loi.
    """
    if args.features_file:
        allowed, source = read_feature_list(args.features_file), f"file:{args.features_file}"
    elif args.features:
        allowed, source = list(args.features), "cli"
    else:
        allowed, source = list(DEFAULT_ALLOWED_FEATURES), "default"
    if target_col in allowed:
        print(f"Bo bien muc tieu '{target_col}' khoi danh sach dac trung cho phep.")
        allowed.remove(target_col)
    allow_leak: tuple[str, ...] = ()
    if args.include_coords:
        coords = [c for c in COORD_COLUMNS if c in columns]
        if not coords:
            raise ValueError(f"--include-coords nhung bang khong co cot toa do nao trong {list(COORD_COLUMNS)}.")
        allowed += coords
        allow_leak = COORD_COLUMNS
    return allowed, allow_leak, source


def _time_key(df: pd.DataFrame) -> pd.Series:
    """Truc thoi gian cho baseline: date (bang ngay) hoac 01/01 cua nam mua (bang mua)."""
    if "date" in df.columns:
        return pd.to_datetime(df["date"])
    return pd.to_datetime(df["season"].astype("int64").astype(str) + "-01-01")


# -----------------------------------------------------------------------------------------------
# --mode cv|final: thiet ke danh gia chinh (An duyet 2026-10-03; soat chia tap V1-V3, V6, V7).
#   cv    : CHI dong split=train; CV 5 fold cap khoi (cv_folds.csv chung 13 luoi, quy tac "cham");
#           ghi oof_predictions.csv. Khong doc/in/ghi gi cua tap test.
#   final : fit toan bo split=train, du doan toan bo split=test -> final_predictions.csv; chi so test
#           theo test_group (khong_gian / thoi_gian / ca_hai).
# -----------------------------------------------------------------------------------------------
def _sha256(path) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def holdout_season_policy(holdout_seasons, allow_override: bool) -> bool:
    """True neu dang chay phan tich do nhay (mua giu rieng khac thiet ke chinh).

    Thiet ke chinh giu rieng mua MAIN_HOLDOUT_SEASON (2020); thieu 2020 trong holdout_seasons -> 2020 vao
    CV/train -> ValueError, tru khi co co tuong minh --allow-holdout-season-override (V3 soat 2026-10-03).
    """
    seasons = [int(s) for s in holdout_seasons or []]
    if not seasons:
        raise ValueError("--holdout-seasons rong.")
    if MAIN_HOLDOUT_SEASON in seasons:
        return False
    if not allow_override:
        raise ValueError(f"--holdout-seasons {seasons} khong chua {MAIN_HOLDOUT_SEASON} (mua giu rieng cua thiet ke "
                         f"chinh) -> {MAIN_HOLDOUT_SEASON} se vao CV/train. Phan tich do nhay: them "
                         "--allow-holdout-season-override.")
    return True


def check_cell_table_provenance(cell_table_path, blocks_path, grid_path, cv_folds_path) -> dict:
    """Doi chieu <cell_table>.provenance.json voi file THUC DUNG (V5 soat 2026-10-03).

    blocks_sha256 / grid_sha256 / cv_folds_sha256 trong provenance phai bang sha256 cua --blocks /
    --grid / --cv-folds. Thieu provenance, thieu khoa, thieu --grid hoac khac -> ValueError.
    Tra ve provenance da doc.
    """
    from preprocessing import provenance_path

    prov_path = provenance_path(cell_table_path)
    if not os.path.exists(prov_path):
        raise ValueError(f"Khong co {prov_path} - khong doi chieu duoc bang o voi khoi/luoi/fold dang dung.")
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    if not grid_path:
        raise ValueError("--cell-table can --grid (hoac GRID_GEOJSON) de doi chieu grid_sha256 trong provenance.")
    errs = []
    for key, path, flag in (("blocks_sha256", blocks_path, "--blocks"), ("grid_sha256", grid_path, "--grid"),
                            ("cv_folds_sha256", cv_folds_path, "--cv-folds")):
        want = prov.get(key)
        if not want:
            errs.append(f"provenance thieu '{key}'")
            continue
        got = _sha256(path)
        if got is None:
            errs.append(f"khong doc duoc {flag} {path}")
        elif got != want:
            errs.append(f"{key} khac: provenance {want[:12]}... vs {flag} {path} {got[:12]}...")
    if errs:
        raise ValueError(f"{cell_table_path}: bang o khong dung tu file dang dung - " + "; ".join(errs))
    return prov


def load_cell_table(args) -> tuple[pd.DataFrame, str]:
    """Bang o -> khoi/fold: doc --cell-table (doi chieu provenance + --cv-folds) hoac dung tu --grid."""
    from eval_design import cell_block_table, read_cell_block_table

    if not os.path.exists(args.cv_folds):
        raise FileNotFoundError(f"Khong co {args.cv_folds} - chay scripts/build_cv_folds.py truoc.")
    folds = pd.read_csv(args.cv_folds, dtype={"block_id": str})
    if args.cell_table:
        check_cell_table_provenance(args.cell_table, args.blocks, args.grid, args.cv_folds)
        table = read_cell_block_table(args.cell_table)
        fold_of = folds.set_index("block_id")["cv_fold"].astype(int)
        mapped = table["block_id"].map(fold_of)
        if mapped.isna().any() or (mapped.to_numpy() != table["cv_fold"].to_numpy()).any():
            raise ValueError(f"{args.cell_table}: cv_fold khong khop {args.cv_folds} (bang o dung tu fold khac?).")
        return table, f"file:{args.cell_table}"
    if not args.grid:
        raise ValueError("--mode can --cell-table hoac --grid (hoac bien moi truong GRID_GEOJSON).")
    import geopandas as gpd

    grid = gpd.read_file(args.grid)
    grid["cell_id"] = grid["cell_id"].astype(str)
    print(f"Dung bang o -> khoi/fold tu {args.grid} ({len(grid)} o)...", flush=True)
    return cell_block_table(grid, gpd.read_file(args.blocks), folds), f"grid:{args.grid}"


def _fit_predict(model_name, params, seed, train, other, feature_cols, target_col):
    """Fit tren `train` (MissingValueHandler fit LAI tren chinh tap nay), du doan `other`."""
    if model_name == "climatology":
        m = ClimatologyBaseline().fit(_time_key(train), train[target_col])
        return m.predict(_time_key(other)), m, None
    X_tr, X_ot, handler = prepare_matrices(model_name, train[feature_cols], other[feature_cols])
    m = build_model(model_name, seed, params)
    m.fit(X_tr, train[target_col])
    return np.asarray(m.predict(X_ot), dtype=float), m, handler


def _metrics_or_none(y, p) -> dict | None:
    ok = np.isfinite(np.asarray(p, dtype=float))
    return compute_metrics(np.asarray(y)[ok], np.asarray(p)[ok]) if ok.any() else None


def _mean_over_seeds(per_seed: dict) -> dict:
    vals = [m for m in per_seed.values() if m]
    out = {"n_seeds": len(vals)}
    for k in ("mae", "rmse", "r2"):
        xs = [m[k] for m in vals if m.get(k) is not None]
        out[k] = float(np.mean(xs)) if xs else None
    return out


def run_eval_mode(args, dataset: pd.DataFrame, target_col: str) -> None:
    if "date" in dataset.columns or "season" not in dataset.columns:
        print("--mode chi ap dung cho bang theo mua (cell_id, season).", file=sys.stderr, flush=True)
        sys.exit(2)
    if args.include_coords:
        print("--mode: bo chinh KHONG dung toa do (An chot 2026-10-03) - bo --include-coords.",
              file=sys.stderr, flush=True)
        sys.exit(2)
    if args.mode == "cv" and args.model == "persistence":
        print("Persistence khong co trong CV khoi khong gian: o val khong co lich su thuoc train fold "
              "(khong co duong co so persistence cho nhom khong gian).", file=sys.stderr, flush=True)
        sys.exit(2)
    seeds = list(dict.fromkeys(args.seeds or [args.seed]))
    try:
        holdout_override = holdout_season_policy(args.holdout_seasons, args.allow_holdout_season_override)
        params, params_source = load_model_params(args.model_params, args.model)
        cell_table, table_source = load_cell_table(args)
        labelled = dataset[dataset[target_col].notna()]
        split_df = assign_eval_split(labelled, cell_table, holdout_seasons=tuple(args.holdout_seasons))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Loi thiet ke danh gia: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True) if args.mode == "final" else None
    del split_df, labelled, dataset  # cv: tu day khong con tham chieu toi dong test

    print(format_report(validate_dataset(train_df if test_df is None else pd.concat([train_df, test_df]),
                                         target_col=target_col, grid_type=args.grid_type)))
    print(f"Tap huan luyen (split=train): {len(train_df)} dong, {train_df['cell_id'].nunique()} o, "
          f"mua {sorted(train_df['season'].astype(int).unique().tolist())}")

    candidates = train_df.drop(columns=[target_col, "split", "test_group"])
    try:
        allowed, allow_leak, features_source = resolve_allowed_features(args, target_col, candidates.columns)
        feature_cols = select_feature_columns(candidates, allowed=allowed, allow_leak=allow_leak,
                                              require_all=features_source != "default")
    except ValueError as exc:
        print(f"Loi chon dac trung: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    _, features_missing = resolve_feature_list(candidates.columns, allowed)
    print(f"Features duoc dung ({len(feature_cols)}): {feature_cols}")
    print(f"Mo hinh {args.model}, tham so ({params_source}): {params}; seeds {seeds}", flush=True)

    eval_cfg = EvalSplitConfig(
        holdout_seasons=tuple(args.holdout_seasons), cv_rule=args.cv_rule, min_val_cells=args.min_val_cells,
        n_folds=int((cell_table["cv_fold"] >= 0).groupby(cell_table["cv_fold"]).any().sum()),
        grid=args.grid, blocks=args.blocks, blocks_sha256=_sha256(args.blocks),
        cv_folds=args.cv_folds, cv_folds_sha256=_sha256(args.cv_folds), cell_table=table_source,
        grid_sha256=_sha256(args.grid), holdout_season_override=holdout_override,
        analysis_kind="phan_tich_do_nhay" if holdout_override else "thiet_ke_chinh")
    if holdout_override:
        print(f"[CANH BAO] Mua giu rieng {list(args.holdout_seasons)} KHONG chua {MAIN_HOLDOUT_SEASON}: phan tich do "
              "nhay (--allow-holdout-season-override), khong dung de so khung chinh.", file=sys.stderr, flush=True)
    config = {
        "experiment_name": args.experiment_name, "mode": args.mode, "model": args.model, "target": target_col,
        "grid_type": args.grid_type, "include_coords": False,
        "features": feature_cols, "features_requested": allowed, "features_source": features_source,
        "features_missing": features_missing,
        "model_params": params, "model_params_source": params_source, "seeds": seeds,
        "split_kind": f"eval_design_{args.mode}", "eval_split": eval_cfg.to_dict(),
        "n_train_rows": len(train_df), "n_train_cells": int(train_df["cell_id"].nunique()),
    }
    out_dir = os.path.join("artifacts", "experiments", args.experiment_name, args.mode)
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "cv":
        try:
            hs = tuple(args.holdout_seasons)
            splits = block_cv_splits(train_df, cell_table, holdout_seasons=hs, rule=args.cv_rule,
                                     min_val_cells=args.min_val_cells)
            summary = block_cv_summary(train_df, cell_table, holdout_seasons=hs, rule=args.cv_rule)
            membership = block_cv_membership(train_df, splits, summary["fold"].tolist())
        except ValueError as exc:
            print(f"Loi CV khoi: {exc}", file=sys.stderr, flush=True)
            sys.exit(2)
        print("--- CV khoi (o bi loai khoi train do quy tac 'cham') ---")
        print(summary.to_string(index=False), flush=True)
        parts, cv_metrics = [], {}
        for seed in seeds:
            fold_metrics = {}
            for (tr_idx, va_idx), k in zip(splits, summary["fold"]):
                tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
                pred, _, _ = _fit_predict(args.model, params, seed, tr, va, feature_cols, target_col)
                parts.append(pd.DataFrame({
                    "cell_id": va["cell_id"].astype(str).to_numpy(), "season": va["season"].astype(int).to_numpy(),
                    "y_true": va[target_col].to_numpy(dtype=float), "y_pred": pred, "fold": int(k),
                    "seed": seed, "model": args.model, "pred_source": "oof"}))
                fold_metrics[str(int(k))] = _metrics_or_none(va[target_col], pred)
                print(f"  seed {seed} fold {int(k)}: train {len(tr)} dong, val {len(va)} dong, "
                      f"MAE {fold_metrics[str(int(k))]['mae']:.4f}", flush=True)
            oof_seed = pd.concat(parts[-len(splits):])
            cv_metrics[str(seed)] = {"oof": _metrics_or_none(oof_seed["y_true"], oof_seed["y_pred"]),
                                     "by_fold": fold_metrics}
        oof = pd.concat(parts, ignore_index=True)
        oof.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
        summary.to_csv(os.path.join(out_dir, "cv_folds_summary.csv"), index=False)
        # (cell_id, fold, role train/val/loai) - kiem doc lap bang check_split.py --membership --grid --cv-folds
        membership.to_csv(os.path.join(out_dir, "cv_membership.csv"), index=False)
        config["cv_summary"] = summary.to_dict(orient="records")
        config["cv_metrics"] = cv_metrics
        config["cv_metrics_mean_over_seeds"] = _mean_over_seeds({s: m["oof"] for s, m in cv_metrics.items()})
        print("--- CV (out-of-fold) ---")
        print(config["cv_metrics_mean_over_seeds"])
    else:
        print(f"Tap test: {len(test_df)} dong; theo nhom: {test_df['test_group'].value_counts().to_dict()}")
        parts, by_group = [], {g: {} for g in TEST_GROUPS}
        notes = {}
        for seed in seeds:
            if args.model == "persistence":
                # V6: lich su CHI tu split=train; o test khong gian / ca hai khong co lich su train ->
                # khong co duong co so persistence cho nhom do (NaN, khong dien trung binh).
                model = PersistenceBaseline().fit(train_df["cell_id"], _time_key(train_df), train_df[target_col])
                history = PersistenceBaseline._frame(train_df["cell_id"], _time_key(train_df), train_df[target_col])
                pred = model.predict(test_df["cell_id"], _time_key(test_df), history)
                first = train_df.groupby(train_df["cell_id"].astype(str))["season"].min()
                has_hist = test_df["cell_id"].astype(str).map(first).lt(test_df["season"]).fillna(False).to_numpy()
                pred = np.where((test_df["test_group"] == "thoi_gian").to_numpy() & has_hist, pred, np.nan)
                notes["persistence"] = ("Lich su chi tu split=train; nhom khong_gian/ca_hai KHONG co duong co so "
                                        "persistence (o chua tung o tap train); o thoi_gian khong co mua train "
                                        f"truoc do -> NaN ({int(((test_df['test_group'] == 'thoi_gian') & ~has_hist).sum())} dong).")
                handler = None
            else:
                pred, model, handler = _fit_predict(args.model, params, seed, train_df, test_df, feature_cols, target_col)
            joblib.dump(model, os.path.join(out_dir, f"model_seed{seed}.joblib"))
            if handler is not None:
                joblib.dump(handler, os.path.join(out_dir, f"missing_handler_seed{seed}.joblib"))
            parts.append(pd.DataFrame({
                "cell_id": test_df["cell_id"].astype(str).to_numpy(), "season": test_df["season"].astype(int).to_numpy(),
                "y_true": test_df[target_col].to_numpy(dtype=float), "y_pred": pred, "seed": seed,
                "model": args.model, "test_group": test_df["test_group"].to_numpy(), "pred_source": "final"}))
            for g in TEST_GROUPS:
                m = (test_df["test_group"] == g).to_numpy()
                by_group[g][str(seed)] = _metrics_or_none(test_df[target_col].to_numpy()[m], pred[m])
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(out_dir, "final_predictions.csv"), index=False)
        config["test_metrics_by_group"] = by_group
        config["test_metrics_by_group_mean_over_seeds"] = {g: _mean_over_seeds(v) for g, v in by_group.items()}
        config["n_test_rows_by_group"] = test_df["test_group"].value_counts().to_dict()
        if notes:
            config["notes"] = notes
        print("--- Test theo nhom (trung binh qua seed) ---")
        for g, m in config["test_metrics_by_group_mean_over_seeds"].items():
            print(f"  {g}: {m}")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Artifacts saved to {out_dir}")


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

    if args.mode:
        run_eval_mode(args, dataset, target_col)
        return

    validation_report = validate_dataset(dataset, target_col=target_col, grid_type=args.grid_type)
    print(format_report(validation_report))

    n_before = len(dataset)
    dataset = dataset.dropna(subset=[target_col])
    print(f"Loai {n_before - len(dataset)} record khong co nhan {target_col} (khong the dung de train/eval).")
    if len(dataset) == 0:
        print(f"BLOCKED BY REAL DATA: khong co record nao co nhan {target_col} hop le.")
        sys.exit(1)

    if "date" in dataset.columns:
        if args.provisional_split:
            print("--provisional-split chi ap dung cho bang theo mua; bang theo ngay dung time_based_split.")
        if not args.train_end or not args.val_end:
            print("Bang theo ngay: can --train-end va --val-end (YYYY-MM-DD).", file=sys.stderr, flush=True)
            sys.exit(2)
        split_cfg = SplitConfig(date_col="date", train_end=args.train_end, val_end=args.val_end, random_seed=args.seed)
        train_df, val_df, test_df = time_based_split(dataset, split_cfg)
        split_kind, test_key = "time_based_daily", "test_metrics"
    else:
        # Bang (o, mua): CHUA co thiet ke danh gia (khoi 50 km + nam giu rieng + CV khoi).
        # Chia tuan tu theo nam chi la cau noi TAM, phai bat co ro rang (soat 2026-10-03, muc 1).
        if not args.provisional_split:
            print("Bang theo mua: dung --mode cv|final (thiet ke danh gia: assign_eval_split/block_cv_splits). "
                  "Hoac --provisional-split de chia TAM tuan tu theo nam (ket qua khong dung de so khung).",
                  file=sys.stderr, flush=True)
            sys.exit(2)
        split_kind, test_key = "season_sequential_PROVISIONAL", "provisional_test_metrics"
        try:
            train_end, val_end = int(args.train_end), int(args.val_end)
        except (TypeError, ValueError):
            print("Bang theo mua: --train-end/--val-end phai la nam mua kho (vd 2019 2021), "
                  f"nhan '{args.train_end}' '{args.val_end}'.")
            sys.exit(2)
        split_cfg = SplitConfig(date_col="season", train_end=str(train_end), val_end=str(val_end), random_seed=args.seed)
        train_df, val_df, test_df = season_sequential_split(dataset, train_end, val_end)
    print("--- Data split summary ---")
    print(split_summary(train_df, val_df, test_df))

    candidates = train_df.drop(columns=[target_col])
    try:
        allowed, allow_leak, features_source = resolve_allowed_features(args, target_col, candidates.columns)
        # Danh sach tuong minh (--features/--features-file): muc thieu -> loi. Mac dinh: canh bao + ghi config.
        feature_cols = select_feature_columns(candidates, allowed=allowed, allow_leak=allow_leak,
                                              require_all=features_source != "default")
    except ValueError as exc:
        print(f"Loi chon dac trung: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    _, features_missing = resolve_feature_list(candidates.columns, allowed)
    print(f"Features duoc dung ({len(feature_cols)}): {feature_cols}")
    if features_missing:
        print(f"[CANH BAO] Muc trong danh sach mac dinh khong co trong bang: {features_missing} "
              "(ghi vao config 'features_missing' - kiem bo dac trung giong nhau giua cac luoi).",
              file=sys.stderr, flush=True)

    missing_handler = None
    if args.model == "climatology":
        model = ClimatologyBaseline().fit(_time_key(train_df), train_df[target_col])
        val_pred = model.predict(_time_key(val_df))
        test_pred = model.predict(_time_key(test_df))
    elif args.model == "persistence":
        cell_col = "cell_id" if "cell_id" in train_df.columns else "h3_index"
        model = PersistenceBaseline().fit(train_df[cell_col], _time_key(train_df), train_df[target_col])
        # Lich su = moi quan trac da biet; predict chi dung gia tri co ngay < ngay can du bao.
        full = pd.concat([train_df, val_df, test_df], ignore_index=True)
        history = PersistenceBaseline._frame(full[cell_col], _time_key(full), full[target_col])
        val_pred = model.predict(val_df[cell_col], _time_key(val_df), history)
        test_pred = model.predict(test_df[cell_col], _time_key(test_df), history)
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
    print("--- PROVISIONAL test metrics (chia tam, khong dung de so khung) ---" if test_key != "test_metrics"
          else "--- Test metrics ---")
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
        "features": feature_cols,              # ten day du DA GIAI tien to (dung de so giua luoi)
        "features_requested": allowed,         # danh sach cho phep truoc khi giai
        "features_source": features_source,
        "features_missing": features_missing,  # chi co the khac rong voi danh sach mac dinh
        "split_kind": split_kind,
        "split": split_cfg.to_dict(),
        "resolution": args.resolution,
        "seed": args.seed,
        "model_params": DEFAULT_MODEL_PARAMS.get(args.model, {}),
        "val_metrics": val_metrics,
        test_key: test_metrics,
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)

    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()

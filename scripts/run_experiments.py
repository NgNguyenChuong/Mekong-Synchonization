#!/usr/bin/env python
"""Chay hang loat train.py --mode cv (hoac final) cho luoi x mo hinh x cach chia fold (tuan 4).

Moi lan chay = mot thu muc artifacts/experiments/<ten>/<mode>/ (do train.py ghi) + run_meta.json:
  git commit, tag tren HEAD (git describe --exact-match), trang thai cay lam viec, sha256 bang hop nhat,
  sha256 dap an diem, sha256 file fold, lenh day du, thoi gian, ma thoat.
Chay TIEP DUOC: lan chay da xong (run_meta.json returncode 0 + config.json) VA trung MOI khoa: commit, cay
lam viec sach/ban, co thu nghiem (allow_*), seeds, sha256 bang / fold / dap an / luoi / khoi / diem, mode -> bo
qua; khac bat ky khoa nao hoac chua xong -> chay lai (thu muc cu chuyen sang artifacts/experiments/_cu/, khong
xoa, khong nam canh thu muc ket qua). Lan chay co --allow-dirty/--allow-untagged KHONG BAO GIO duoc tinh la
xong cho lan chay chinh thuc (soat 2026-10-04 muc 1).
Sau moi lan chay: danh sach dac trung (config.features) phai GIONG NHAU giua moi luoi cua cung mo hinh va
khong thieu muc nao (features_missing rong) - khac -> danh dau LOI.
Bat buoc truoc khi chay that: HEAD co tag va cay lam viec sach o src/ scripts/ (tru --allow-untagged /
--allow-dirty, chi de thu nghiem; ghi vao run_meta).
Bien the dap an (--points-ref-variant, mac dinh main = cham chinh moi bo bang dap an bo chinh; S2 2026-10-06):
cham phu (vd keep6090 tren 349 diem 60/90) PHAI dung tien to rieng - ten lan chay khong chua bien the, nen neu
cung tien to + label_set da co lan chay bien the khac (meta thieu khoa = main) -> LOI truoc khi chay (khong ghi de,
khong chuyen sang _cu). Dap an co provenance ma 'variant' khac yeu cau -> LOI som (train.py cung kiem).
Bien muc tieu (--target, Dot 7; mac dinh salinity = hanh vi cu, ten lan chay va lenh KHONG doi): bien khac -> ten lan
chay them "__t-<target>", truyen --target xuong train.py, khoa 'target' (meta cu thieu khoa = salinity); provenance
bang / dap an co 'target' khac --target -> LOI som. Bang theo bien o --tables-dir rieng
(<DATA_ROOT>/features/unified_dot7/<target>), dap an <DATA_ROOT>/labels/dot7/points_reference_<target>.csv.
--feature-set CUNG --target: bo dat ten tru cot cam theo bien (TARGET_FORBIDDEN) -> truyen --features da tru (train.py
van tu choi danh sach tuong minh co cot cam); run_meta ghi features_requested + features_dropped_forbidden.
Nhom test rong co chu dich (CHG-25 muc 4): provenance bang co khai bao 'holdout_season_nan' (target khop) -> truyen
--allow-empty-test-groups xuong train.py + khoa 'empty_test_groups_declared'; bien Dot 7 ma bang thieu provenance -> LOI
truoc khi chay luot nao.
Tong hop -> artifacts/experiments/<prefix>_manifest.csv (gop don theo ten lan chay + mode, khong ghi de dong
cu; KHONG in chi so sai so). run_meta ghi phien ban python/numpy/pandas/sklearn.

Chay:  venv/Scripts/python.exe scripts/run_experiments.py --prefix cv1 [--grids h3_res_5 ...]
           [--models hist_gb linear idw] [--schemes 42 43 44] [--mode cv]
       Bien Dot 7: ... --prefix cv1 --target rain_chirps --tables-dir <DATA_ROOT>/features/unified_dot7/rain_chirps
           --points-ref <DATA_ROOT>/labels/dot7/points_reference_rain_chirps.csv
       Cham phu 60/90: ... --prefix phu6090 --label-set keep6090 --models hist_gb --points-ref-variant keep6090
           --points-subset-of-ref --points data/eval/aux_6090/eval_points_6090.geojson
           --points-ref <DATA_ROOT>/labels/points_reference_keep6090.csv
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

KEY_FIELDS = ("run", "grid", "model", "scheme", "label_set", "mode", "feature_set", "features_sha256",
              "git_commit", "git_dirty_src_scripts", "allow_dirty", "allow_untagged", "seeds", "table_sha256",
              "cv_folds_sha256", "points_ref_sha256", "grid_sha256", "blocks_sha256", "points_sha256",
              "points_ref_variant", "points_subset_of_ref", "target", "empty_test_groups_declared")
# Gia tri mac dinh khi run_meta CU thieu khoa (lan chay truoc khi them khoa) - de khong chay lai do man.
META_DEFAULTS = {"target": "salinity", "empty_test_groups_declared": []}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from settings import data_path  # noqa: E402  (.env: DATA_ROOT)
from training.features import drop_target_forbidden, read_feature_list  # noqa: E402
from training.split import MAIN_HOLDOUT_SEASON, declared_empty_groups  # noqa: E402

DATA = data_path()
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def git_state():
    _, commit = git("rev-parse", "HEAD")
    rc, tag = git("describe", "--tags", "--exact-match", "HEAD")
    _, dirty = git("status", "--porcelain", "--", "src", "scripts")
    return {"git_commit": commit, "git_tag": tag if rc == 0 else None, "git_dirty_src_scripts": bool(dirty)}


def run_name(prefix, grid, model, scheme, label_set, feature_set=None, target=None):
    return (f"{prefix}__{grid}__{model}__s{scheme}" + (f"__{label_set}" if label_set else "")
            + (f"__fs-{feature_set}" if feature_set else "")
            + (f"__t-{target}" if target and target != "salinity" else ""))


def versions():
    import numpy
    import sklearn

    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__}


def check_features(out_dir, model, ref_features: dict):
    """Loi (chuoi) neu config.features thieu muc hoac khac luoi truoc cua cung mo hinh; None neu dat."""
    with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("features_missing"):
        return f"thieu dac trung {cfg['features_missing']}"
    feats = cfg.get("features")
    if feats is None:  # CHG-22: thieu khoa -> truoc day None == None la "giong luoi truoc"
        return "config.json thieu khoa features"
    if model in ref_features and ref_features[model] != feats:
        return f"danh sach dac trung khac luoi truoc ({len(feats)} vs {len(ref_features[model])})"
    ref_features.setdefault(model, feats)
    return None


def variant_conflicts(exp_root, prefix, label_set, variant) -> list:
    """Lan chay da co (artifacts/experiments/<prefix>__*/<mode>/run_meta.json) CUNG label_set nhung bien the dap an
    khac `variant` (meta khong co khoa points_ref_variant = lan chay truoc S2 = 'main'). Meta khong doc duoc -> tinh
    la xung dot (CHG-22: khong doan). Tra danh sach (duong dan, ly do)."""
    want_set = label_set or "chinh"
    out = []
    for mp in sorted(glob.glob(os.path.join(exp_root, f"{glob.escape(prefix)}__*", "*", "run_meta.json"))):
        try:
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError) as exc:
            out.append((mp, f"khong doc duoc run_meta ({exc})"))
            continue
        if meta.get("label_set", "chinh") != want_set:
            continue
        got = meta.get("points_ref_variant") or "main"
        if got != variant:
            out.append((mp, f"points_ref_variant {got} != {variant}"))
    return out


def ref_variant_of(points_ref):
    """'variant' trong <points_ref>.provenance.json; khong co provenance -> None (train.py xu ly tiep)."""
    return provenance_of(points_ref).get("variant")


def provenance_of(path) -> dict:
    """Noi dung <path>.provenance.json ({} neu khong co)."""
    pp = f"{path}.provenance.json"  # = preprocessing.provenance_path (khong import geopandas/h3 o runner)
    if not os.path.exists(pp):
        return {}
    with open(pp, encoding="utf-8") as f:
        return json.load(f)


def target_mismatch(path, target, default=None):
    """Chuoi loi neu provenance cua `path` co 'target' (thieu -> default) khac `target`; None neu khop/khong ro."""
    info = provenance_of(path)
    got = info.get("target", default) if info else None
    if got is not None and got != target:
        return f"{os.path.basename(path)}: target '{got}' khac --target '{target}'"
    return None


def table_empty_groups(table, target) -> list:
    """Nhom test rong khai bao trong provenance bang (train.py mac dinh giu rieng MAIN_HOLDOUT_SEASON).
    Bien Dot 7 (target != salinity) ma bang thieu provenance, hoac khai bao sai dang -> ValueError."""
    prov = provenance_of(table)
    if not prov and target != "salinity":
        raise ValueError(f"{os.path.basename(table)}: thieu provenance (bang bien Dot 7 bat buoc co)")
    return list(declared_empty_groups(prov, target, (MAIN_HOLDOUT_SEASON,)))


def feature_set_lists(feat_file, target):
    """(muc trong file, muc giu, muc bo vi cam theo bien): bo dat ten giao voi danh sach cho phep cua bien."""
    req = read_feature_list(feat_file)
    keep, dropped = drop_target_forbidden(req, target)
    if not keep:
        raise ValueError(f"{os.path.basename(feat_file)}: moi muc bi cam voi bien '{target}'")
    return req, keep, dropped


def is_done(meta_path, cfg_path, want: dict) -> bool:
    if not (os.path.exists(meta_path) and os.path.exists(cfg_path)):
        return False
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("allow_dirty") or meta.get("allow_untagged") or meta.get("git_dirty_src_scripts"):
        if not (want["allow_dirty"] or want["allow_untagged"]):
            return False  # ket qua thu nghiem khong thay cho lan chay chinh thuc
    # CHG-22: config thieu khoa features hoac (co cham theo diem) thieu file diem cua mode -> CHUA xong
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "features" not in cfg:
        return False
    if want.get("points_ref_sha256") is not None:
        pts = "oof_points.csv" if want.get("mode") == "cv" else "final_points.csv"
        if not os.path.exists(os.path.join(os.path.dirname(cfg_path), pts)):
            return False
    return meta.get("returncode") == 0 and all(meta.get(k, META_DEFAULTS.get(k)) == want.get(k) for k in KEY_FIELDS)


def main(a):
    state = git_state()
    if not state["git_tag"] and not a.allow_untagged:
        sys.exit("HEAD chua co tag git - gan tag truoc lan chay that (hoac --allow-untagged de thu nghiem).")
    if state["git_dirty_src_scripts"] and not a.allow_dirty:
        sys.exit("Cay lam viec co thay doi chua commit trong src/ scripts/ (hoac --allow-dirty de thu nghiem).")
    grids = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(a.grids_dir, "*.geojson")))
    if a.grids:
        miss = sorted(set(a.grids) - set(grids))
        if miss:
            sys.exit(f"Khong co luoi {miss} trong {a.grids_dir}")
        grids = [g for g in grids if g in a.grids]
    if not a.points_ref and not a.no_point_eval:
        sys.exit("Thieu --points-ref: cham theo diem la ket qua chinh; bo phai ghi ro --no-point-eval.")
    for k in ("tables_dir", "points", "points_ref", "grids_dir", "folds_dir", "blocks", "cwd"):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    suffix = f"_{a.label_set}" if a.label_set else ""
    ref_sha = sha256(a.points_ref) if (a.points_ref and not a.no_point_eval) else None
    blocks_sha, points_sha = sha256(a.blocks), sha256(a.points)
    feat_file, feat_sha, feat_req, feat_keep, feat_dropped = None, None, None, None, []
    if a.feature_set:  # bo dac trung dat ten: configs/feature_sets/<ten>.txt (--features-file -> moi muc phai co)
        feat_file = os.path.join(ROOT, "configs", "feature_sets", f"{a.feature_set}.txt")
        if not os.path.exists(feat_file):
            sys.exit(f"Khong co bo dac trung {feat_file}")
        feat_sha = sha256(feat_file)
        try:
            feat_req, feat_keep, feat_dropped = feature_set_lists(feat_file, a.target)
        except ValueError as exc:
            sys.exit(f"[LOI] {exc}")
    exp_root = os.path.join(a.cwd, "artifacts", "experiments")
    if ref_sha is not None:
        v = ref_variant_of(a.points_ref)
        if v is not None and v != a.points_ref_variant:
            sys.exit(f"--points-ref co variant '{v}' khac --points-ref-variant '{a.points_ref_variant}'.")
        err = target_mismatch(a.points_ref, a.target, default="salinity")   # dap an cu khong ghi target = do man
        if err:
            sys.exit(f"[LOI] {err}")
    conflicts = variant_conflicts(exp_root, a.prefix, a.label_set, a.points_ref_variant)
    if conflicts:
        sys.exit(f"[LOI] Tien to '{a.prefix}' (label_set {a.label_set or 'chinh'}) da co {len(conflicts)} lan chay "
                 f"bien the dap an khac (vd {conflicts[0][0]}: {conflicts[0][1]}) - cham phu phai dung tien to rieng "
                 "(vd --prefix phu6090), khong ghi de lan cham chinh.")
    empty_by_grid = {}   # kiem moi bang truoc khi chay luot nao
    for grid in grids:
        table = os.path.join(a.tables_dir, f"{grid}_unified{suffix}.csv")
        if not os.path.exists(table):
            sys.exit(f"Khong co bang {table}")
        err = target_mismatch(table, a.target)
        if err:
            sys.exit(f"[LOI] {err}")
        try:
            empty_by_grid[grid] = table_empty_groups(table, a.target)
        except ValueError as exc:
            sys.exit(f"[LOI] {grid}: {exc}")
    rows, ref_features, vers = [], {}, versions()
    for grid in grids:
        table = os.path.join(a.tables_dir, f"{grid}_unified{suffix}.csv")
        grid_path = os.path.join(a.grids_dir, f"{grid}.geojson")
        empty_groups = empty_by_grid[grid]
        table_sha, grid_sha = sha256(table), sha256(grid_path)
        for scheme in a.schemes:
            folds = os.path.join(a.folds_dir, f"cv_folds_s{scheme}.csv")
            folds_sha = sha256(folds)
            for model in a.models:
                name = run_name(a.prefix, grid, model, scheme, a.label_set, a.feature_set, a.target)
                out = os.path.join(exp_root, name, a.mode)
                meta_path = os.path.join(out, "run_meta.json")
                want = {"run": name, "grid": grid, "model": model, "scheme": scheme, "label_set": a.label_set or "chinh",
                        "feature_set": a.feature_set, "features_sha256": feat_sha,
                        "git_commit": state["git_commit"], "git_dirty_src_scripts": state["git_dirty_src_scripts"],
                        "allow_dirty": a.allow_dirty, "allow_untagged": a.allow_untagged, "seeds": list(a.seeds),
                        "mode": a.mode, "table_sha256": table_sha, "cv_folds_sha256": folds_sha,
                        "points_ref_sha256": ref_sha, "grid_sha256": grid_sha, "blocks_sha256": blocks_sha,
                        "points_sha256": points_sha, "points_ref_variant": a.points_ref_variant,
                        "points_subset_of_ref": a.points_subset_of_ref, "target": a.target,
                        "empty_test_groups_declared": empty_groups}
                row = {"run": name, "grid": grid, "model": model, "scheme": scheme, "label_set": a.label_set or "chinh",
                       **want, "seeds": " ".join(map(str, a.seeds)), "git_tag": state["git_tag"],
                       "empty_test_groups_declared": " ".join(empty_groups)}
                if is_done(meta_path, os.path.join(out, "config.json"), want):
                    err = check_features(out, model, ref_features)
                    rows.append({**row, "status": "bo_qua_da_xong" if not err else f"LOI: {err}",
                                 "returncode": 0 if not err else 3})
                    print(f"[bo qua] {name}", flush=True)
                    continue
                if os.path.exists(out):
                    old = os.path.join(exp_root, "_cu", f"{name}__{a.mode}__{datetime.now():%Y%m%d_%H%M%S}")
                    os.makedirs(os.path.dirname(old), exist_ok=True)
                    os.rename(out, old)
                cmd = [sys.executable, os.path.join(ROOT, "src", "training", "train.py"), "--table", table,
                       "--mode", a.mode, "--grid", grid_path, "--blocks", a.blocks, "--cv-folds", folds,
                       "--model", model, "--seeds", *map(str, a.seeds), "--experiment-name", name]
                if feat_file:  # train.py tu choi danh sach tuong minh co cot cam -> truyen danh sach da tru
                    cmd += ["--features", *feat_keep] if feat_dropped else ["--features-file", feat_file]
                if a.target != "salinity":
                    cmd += ["--target", a.target]
                if empty_groups:
                    cmd += ["--allow-empty-test-groups", *empty_groups]
                if a.no_point_eval:
                    cmd += ["--no-point-eval"]
                else:
                    cmd += ["--points", a.points, "--points-ref", a.points_ref,
                            "--points-ref-variant", a.points_ref_variant]
                    if a.points_subset_of_ref:
                        cmd += ["--points-subset-of-ref"]
                started = datetime.now().isoformat(timespec="seconds")
                print(f"[chay] {name}", flush=True)
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "train_stdout.log"), "w", encoding="utf-8") as log:
                    rc = subprocess.run(cmd, cwd=a.cwd, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
                meta = {**row, **state, **want, "features_requested": feat_req,
                        "features_dropped_forbidden": feat_dropped, "returncode": rc, "cmd": cmd, "started": started,
                        "finished": datetime.now().isoformat(timespec="seconds"), "versions": vers}
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                err = check_features(out, model, ref_features) if rc == 0 else None
                if err:
                    rc = 3
                rows.append({**row, "status": "xong" if rc == 0 else f"LOI{': ' + err if err else ''}", "returncode": rc})
                print(f"  -> ma thoat {rc}", flush=True)
    man = pd.DataFrame(rows)
    os.makedirs(exp_root, exist_ok=True)
    man_path = os.path.join(exp_root, f"{a.prefix}_manifest.csv")
    if os.path.exists(man_path):  # gop don: dong moi thay dong cu cung (run, mode), giu dong khac
        old = pd.read_csv(man_path, dtype=str)
        keep = ~old.set_index(["run", "mode"]).index.isin(man.set_index(["run", "mode"]).index)
        man = pd.concat([old[keep], man.astype(str)], ignore_index=True)
    man.to_csv(man_path, index=False)
    man["returncode"] = man["returncode"].astype(int)
    n_err = int((man["returncode"] != 0).sum())
    print(f"Manifest: {man_path} | {len(man)} lan chay, {n_err} loi")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", required=True, help="Tien to ten thi nghiem, vd cv1")
    ap.add_argument("--grids", nargs="*")
    ap.add_argument("--models", nargs="+", default=["hist_gb", "linear", "idw"])
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44], help="cv_folds_s<n>.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="seed mo hinh (HistGB tat dinh)")
    ap.add_argument("--mode", choices=["cv", "final"], default="cv")
    ap.add_argument("--label-set", default="", help='"" = bo chinh; keepwater / keepmangrove / keep6090')
    ap.add_argument("--target", default="salinity",
                    help="Cot muc tieu (Dot 7: ndwi, rain_chirps, dsr_mcd18, t2m_era5, rh_era5); mac dinh salinity")
    ap.add_argument("--feature-set", default=None,
                    help="Ten bo dac trung trong configs/feature_sets/<ten>.txt (mac dinh: danh sach cho phep mac dinh)")
    ap.add_argument("--tables-dir", default=f"{DATA}/features/unified")
    ap.add_argument("--points", default=os.path.join(ROOT, "data", "eval", "eval_points.geojson"))
    ap.add_argument("--points-ref", default=f"{DATA}/labels/points_reference.csv",
                    help="Dap an bo chinh cho MOI bo nhan (An chot 2026-10-04)")
    ap.add_argument("--points-ref-variant", default="main",
                    help="Bien the dap an (provenance 'variant'), truyen xuong train.py. main = cham chinh; khac main "
                         "(vd keep6090) = cham phu, BAT BUOC tien to rieng")
    ap.add_argument("--points-subset-of-ref", action="store_true",
                    help="--points la tap con diem cua --points-ref (cham phu 349 diem 60/90)")
    ap.add_argument("--no-point-eval", action="store_true", help="Bo cham theo diem - KHONG dung cho ket qua chinh")
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "data", "grids"))
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--blocks", default=os.path.join(ROOT, "data", "eval", "holdout_blocks.geojson"))
    ap.add_argument("--cwd", default=ROOT, help="Thu muc goc ghi artifacts/experiments")
    ap.add_argument("--allow-untagged", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    main(ap.parse_args())

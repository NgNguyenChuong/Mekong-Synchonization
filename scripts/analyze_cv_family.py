#!/usr/bin/env python
"""Kiem dinh so sanh khung luoi THEO KHOI tren ket qua CV (mode cv) - CHG-06, A2-A4, D; Dot 7 them --target (CHG-25).

Thiet ke da chot TRUOC (NHAT_KY 2026-10-03/04):
  - Ho chinh F1: HistGB, MAE, cap CUNG muc co ti le dien tich o TB <= 1,2 -> 12 cap; alpha 0,05.
  - Don vi kiem dinh = don vi CV (khoi 50 km, khoi < 30 diem gop vao khoi ke: cv_folds.csv unit_id) - 19 don vi.
  - "seed" = 3 cach gan khoi -> fold (42, 43, 44): sai so trung binh qua 3 cach; nhan can cung chieu ca 3.
  - Delta_min = 5% MAE tuong doi, MOT so moi muc (level_mean tren cac luoi trong ho).
  - Mode cv: chua co khoi giu rieng -> nhan cao nhat "cho_giu_rieng" (Holm dat + cung chieu moi cach chia).
Phu (khong vao Holm cua F1): linear, IDW (cung 12 cap); cap S2 L9 voi luoi tho (ti le 1,35); do nhay tung cach
chia rieng. Tap diem chung tinh THEO MO HINH (An 2026-10-04). Delta < 0: luoi A sai so THAP hon B.
CHG-25: Holm F1 chi tren cap duoc kiem dinh; con lai label mo_ta (p/Holm/TOST NaN). Bien khac salinity bat buoc co
dot7_cong_kiem_dinh.csv (analyze_learnability.py). --holm-scope cong (mac dinh): ho = muc thuoc TESTED_TIERS va hop le
o cong; muc_kiem_dinh (do nhay): ho = moi muc thuoc TESTED_TIERS, muc truot cong ep mo_ta sau Holm. Hai pham vi khac
nhau -> ghi them file __holm_muc_kiem_dinh; pham vi Holm ghi o <file>.provenance.json (khong them cot).
Moi luot doc vao: config.target == --target, cung points_ref_sha256, cung git_tag (ghi vao output). File dau ra nam
trong manifest dong bang -> LOI ma 2. --no-table: khong in bang so (buoc mo niem phong).
--model rf|mlp: chi ho do_nhay_<model> (+ tung cach chia), cot vai_tro, file hau to __<model>, cong cua chinh mo hinh.
--region mississippi|nile: luot/bang dien tich/fold/diem cua vung, ra KE_HOACH/ket-qua/doi_chung/<vung>/; run_meta.region
phai khop; --pairs-mode bat buoc (dbscl12 | ratio); --exclude-points (mac dinh configs/exclude_points/<vung>.txt).

Chay:  venv/Scripts/python.exe scripts/analyze_cv_family.py --prefix cv1 [--target ndwi] [--out-dir KE_HOACH/ket-qua]
           [--model hist_gb|rf|mlp] [--holm-scope cong|muc_kiem_dinh] [--allowed-tags TAG ...] [--no-table]
       Vung:  ... --region mississippi --target ndwi --pairs-mode ratio --frozen-manifest <manifest...>
"""
import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from region import DEFAULT_REGION, REGIONS, region_paths  # noqa: E402
from run_experiments import run_name  # noqa: E402
from training.block_stats import (EXACT_MAX_G, area_levels, common_point_set, compare_family,  # noqa: E402,F401
                                  pairs_by_area)
from training.dot7_rules import (HOLM_SCOPES, MODELS, MULTI_TAG_REASON, SENS_ROLE, TESTED_TIERS,  # noqa: E402
                                 all_scoped_names, apply_gate, compare_tiered, expected_f1_m, file_sha256,
                                 gate_required,
                                 guard_frozen, holm_provenance, holm_tiers, load_gate, main_family, model_name,
                                 model_provenance, provenance_path, run_model, scoped_name, scopes_differ,
                                 write_csv_atomic, write_provenance)

GRIDS = ["h3_res_5", "h3_res_6", "h3_res_7", "latlon_0.0222deg", "latlon_0.0586deg", "latlon_0.1552deg",
         "s2_level_9", "s2_level_10", "s2_level_11", "s2_level_12", "square_utm_17087m", "square_utm_2441m",
         "square_utm_6458m"]
RESULTS = os.path.join(ROOT, "KE_HOACH", "ket-qua")
FROZEN_MANIFESTS = [os.path.join(RESULTS, "dot_dong_bang_do_man_manifest.csv"),
                    os.path.join(RESULTS, "dot7_dong_bang_histgb_manifest.csv"),
                    os.path.join(RESULTS, "dot7_dong_bang_rfmlp_manifest.csv")]
GATE_FILE = "dot7_cong_kiem_dinh.csv"
EXP_ROOT = os.path.join(ROOT, "artifacts", "experiments")
AREA_FILE = "dot4_doi_chieu_dien_tich.csv"
PAIRS_MODES = ("dbscl12", "ratio")
MAX_RATIO = 1.2
# Ho F1 DBSCL (pairs_by_area 1,2 tren bang DBSCL, giu chieu A/B) - ap nguyen cho vung o che do dbscl12
DBSCL_PAIRS = (("h3_res_5", "square_utm_17087m"), ("h3_res_5", "latlon_0.1552deg"), ("h3_res_6", "square_utm_6458m"),
               ("h3_res_6", "latlon_0.0586deg"), ("h3_res_7", "s2_level_12"), ("h3_res_7", "square_utm_2441m"),
               ("h3_res_7", "latlon_0.0222deg"), ("s2_level_12", "square_utm_2441m"),
               ("s2_level_12", "latlon_0.0222deg"), ("square_utm_17087m", "latlon_0.1552deg"),
               ("square_utm_6458m", "latlon_0.0586deg"), ("square_utm_2441m", "latlon_0.0222deg"))
KNOWN_G = {"dbscl": 19, "mississippi": 27, "nile": 15}  # so don vi CV (cv_folds) - chi de canh bao


def loi(msg):
    raise SystemExit(f"LOI: {msg}")


def out_name(target, prefix):
    return f"dot5_{prefix}_kiem_dinh_khoi.csv" if target == "salinity" else f"dot7_{target}_{prefix}_kiem_dinh_khoi.csv"


def gate_name(model="hist_gb"):
    return model_name(GATE_FILE, model)


def scope_outputs(target, prefix, gate, scope):
    """[(pham vi, ten file)]: cong -> file chinh (+ file do nhay khi hai pham vi khac nhau); muc_kiem_dinh -> mot file."""
    differ = scopes_differ(target, gate)
    scopes = ["cong", "muc_kiem_dinh"] if scope == "cong" and differ else [scope]
    return [(s, scoped_name(out_name(target, prefix), s, differ)) for s in scopes]


def area_table(path):
    t = pd.read_csv(path)
    t["grid"] = t["file"].map(lambda f: os.path.splitext(os.path.basename(str(f)))[0])
    return t


# ---------------------------------------------------------
# Vung: duong dan mac dinh, diem loai, tap cap
# ---------------------------------------------------------
def region_defaults(region):
    """Mac dinh theo vung; dbscl = DUNG gia tri cu."""
    if region == DEFAULT_REGION:
        return {"exp_root": EXP_ROOT, "area_table": os.path.join(RESULTS, AREA_FILE), "out_dir": RESULTS,
                "folds_csv": os.path.join(ROOT, "data", "eval", "cv_folds.csv"), "exclude_points": None,
                "eval_dir": None}
    p = region_paths(region)
    return {"exp_root": os.path.join(EXP_ROOT, region), "area_table": f"{p['report_dir']}/{AREA_FILE}",
            "out_dir": os.path.join(RESULTS, "doi_chung", region), "folds_csv": f"{p['eval_dir']}/cv_folds.csv",
            "exclude_points": os.path.join(ROOT, "configs", "exclude_points", f"{region}.txt"),
            "eval_dir": p["eval_dir"]}


def resolve_region(a):
    """Dien mac dinh theo --region cho tham so chua truyen; vung doi chung bat buoc --pairs-mode."""
    for k, v in region_defaults(a.region).items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    if a.pairs_mode is None:
        if a.region != DEFAULT_REGION:
            loi(f"--region {a.region}: bat buoc --pairs-mode {'|'.join(PAIRS_MODES)} (tap cap vung chua chot)")
        a.pairs_mode = "dbscl12"
    a.exclude = read_point_list(a.exclude_points) if a.exclude_points else []
    a.eval_kw = {} if a.region == DEFAULT_REGION else {"eval_dir": a.eval_dir}
    return a


def read_point_list(path):
    """Moi dong mot point_id; dong trong va phan sau '#' bo qua."""
    if not os.path.isfile(path):
        loi(f"thieu file diem loai {path}")
    with open(path, encoding="utf-8") as f:
        ids = [ln.split("#")[0].strip() for ln in f]
    ids = [x for x in ids if x]
    if len(set(ids)) != len(ids):
        loi(f"{path}: point_id trung")
    return ids


def drop_points(df, exclude):
    return df[~df["point_id"].isin(set(exclude))] if len(exclude) else df


def check_excluded(exclude, pu):
    """Diem loai phai thuoc tap diem danh gia cua vung (go nham ma -> loi, khong lang le bo qua)."""
    miss = sorted(set(exclude) - set(pu.index))
    if miss:
        loi(f"diem loai khong co trong tap diem danh gia: {miss}")


def _posix(path):
    return os.path.abspath(path).replace(os.sep, "/")


def select_pairs(at, region, mode):
    """(cap F1, canh bao). dbscl12: 12 cap DBSCL voi ti le dien tich thuc cua vung; ratio: pairs_by_area(1,2)."""
    if mode not in PAIRS_MODES:
        loi(f"--pairs-mode '{mode}' khong thuoc {PAIRS_MODES}")
    if region != DEFAULT_REGION and "file" in at.columns:
        bad = [f for f in at["file"].astype(str) if f"/{region}/" not in f.replace(os.sep, "/")]
        if bad:
            loi(f"bang dien tich khong phai cua vung {region}: {bad[:3]}")
    if mode == "ratio":
        return pairs_by_area(at, max_ratio=MAX_RATIO), []
    if region == DEFAULT_REGION:
        return pairs_by_area(at, max_ratio=MAX_RATIO, expected_m=12), []
    lv = area_levels(at)
    area = at.set_index("grid")["mean_area_km2"].astype(float)
    rows = []
    for a, b in DBSCL_PAIRS:
        if a not in lv.index or b not in lv.index:
            loi(f"bang dien tich vung {region} thieu luoi {a}/{b}")
        if lv[a] != lv[b]:
            loi(f"cap DBSCL {a}/{b} khac muc o vung {region}")
        rows.append({"grid_a": a, "grid_b": b, "tier": lv[a], "area_a": area[a], "area_b": area[b],
                     "area_ratio": max(area[a], area[b]) / min(area[a], area[b])})
    out = pd.DataFrame(rows)
    warn = [f"{r.grid_a}/{r.grid_b} ti le {r.area_ratio:.4f} > {MAX_RATIO}" for r in out.itertuples()
            if r.area_ratio > MAX_RATIO * (1 + 1e-12)]
    return out, warn


def side_pairs(at, main_pairs):
    """Ho phu: cap ti le (1,2; 1,5] ngoai ho chinh."""
    wide = pairs_by_area(at, max_ratio=1.5)
    key_main = set(zip(main_pairs["grid_a"], main_pairs["grid_b"]))
    keep = [(x, y) not in key_main and r > MAX_RATIO * (1 + 1e-12)
            for x, y, r in zip(wide["grid_a"], wide["grid_b"], wide["area_ratio"])]
    return wide[keep].reset_index(drop=True)


def f1_m(tiers, pairs, region):
    """m Holm F1: DBSCL theo hang F1_PAIRS_PER_TIER (kiem lai 3/3/6); vung theo so cap thuc o cac muc Holm."""
    return expected_f1_m(tiers) if region == DEFAULT_REGION else int(pairs["tier"].isin(set(tiers)).sum())


def region_info(a, cv_units, res=None, pairs=None, warn=()):
    """Khoa provenance: vung, tap cap, diem loai, G, sign-flip."""
    excl = a.exclude_points
    info = {"region": a.region, "pairs_mode": a.pairs_mode,
            "exclude_points_file": _posix(excl) if excl else None,
            "exclude_points_sha256": file_sha256(excl) if excl else None,
            "diem_loai": list(a.exclude), "n_diem_loai": len(a.exclude), "G_don_vi_cv": len(cv_units)}
    if pairs is not None:
        info["cap_f1"] = [{"grid_a": r.grid_a, "grid_b": r.grid_b, "tier": int(r.tier),
                           "area_ratio": float(r.area_ratio)} for r in pairs.itertuples()]
        info["canh_bao_ti_le"] = list(warn)
    if res is not None and "G" in res.columns:
        info["G_theo_cap"] = {"min": int(res["G"].min()), "max": int(res["G"].max())}
        info["sign_flip"] = {"exact_max_G": EXACT_MAX_G, "exact": sorted(bool(x) for x in set(res["p_exact"])),
                             "n_perm": sorted(int(x) for x in set(res["n_perm"])),
                             "seed": "comparison_seed(cmp_id, base_seed=42)"}
    return info


def warn_g(region, cv_units):
    if len(cv_units) != KNOWN_G[region]:
        print(f"CANH BAO: {len(cv_units)} don vi CV, ky vong {KNOWN_G[region]} cho {region}", flush=True)


# ---------------------------------------------------------
# Doc + kiem luot chay
# ---------------------------------------------------------
def _read_json(path, name):
    if not os.path.isfile(path):
        loi(f"{name}: thieu {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_run(run_dir, name, target, mode, region=DEFAULT_REGION):
    """run_meta + config cua mot luot: ma 0, dung mode, khong co allow_*, target (thieu = salinity) == --target,
    region (thieu = dbscl) == --region."""
    meta = _read_json(os.path.join(run_dir, "run_meta.json"), name)
    cfg = _read_json(os.path.join(run_dir, "config.json"), name)
    if str(meta.get("returncode")) != "0":
        loi(f"{name}: returncode {meta.get('returncode')}")
    if meta.get("mode", mode) != mode:
        loi(f"{name}: mode '{meta.get('mode')}' khac '{mode}'")
    if meta.get("allow_dirty") or meta.get("allow_untagged"):
        loi(f"{name}: luot thu nghiem (allow_dirty/allow_untagged)")
    for src, got in (("run_meta", meta.get("target", "salinity")), ("config", cfg.get("target", "salinity"))):
        if got != target:
            loi(f"{name}: {src}.target '{got}' khac --target '{target}'")
    for src, got in (("run_meta", meta.get("region", DEFAULT_REGION)), ("config", cfg.get("region", DEFAULT_REGION))):
        if got != region:
            loi(f"{name}: {src}.region '{got}' khac --region '{region}'")
    return {"run": name, "git_tag": meta.get("git_tag"), "points_ref_sha256": meta.get("points_ref_sha256"),
            "meta": meta, "config": cfg}


def check_consistent(infos, allowed_tags=None):
    """Moi luot cung points_ref_sha256; git_tag duy nhat, hoac (allowed_tags) tap git_tag la tap con cua danh sach
    (git_tag ghi ra = cac tag noi bang ';')."""
    if not infos:
        loi("khong co luot nao")
    out = {}
    for k in ("git_tag", "points_ref_sha256"):
        vals = {i[k] for i in infos}
        if None in vals or "" in vals:
            loi(f"{k} thieu o {[i['run'] for i in infos if not i[k]][:5]}")
        if k == "git_tag" and allowed_tags:
            extra = sorted(vals - set(allowed_tags))
            if extra:
                loi(f"git_tag {extra} ngoai --allowed-tags {sorted(set(allowed_tags))}")
            out["git_tags_seen"] = sorted(vals)
            out[k] = ";".join(sorted(vals))
            continue
        if len(vals) != 1:
            loi(f"{k} khac nhau giua cac luot: {sorted(vals)}")
        out[k] = vals.pop()
    return out


def tag_info(same, allowed_tags, reason=MULTI_TAG_REASON) -> dict:
    """Khoa provenance khi cho phep nhieu tag; khong khai bao -> {}."""
    if not allowed_tags:
        return {}
    return {"git_tags_allowed": sorted(set(allowed_tags)), "git_tags_seen": same["git_tags_seen"],
            "ly_do_nhieu_tag": reason}


def load_errors(exp_root, prefix, model, schemes, grids=GRIDS, target="salinity", infos=None, region=DEFAULT_REGION,
                exclude=()):
    """err_long (grid, model, seed=cach chia, point_id, season, err, pred_source, unit_id) tu oof_points.csv;
    bo dong cua diem trong `exclude`."""
    parts = []
    for g in grids:
        for s in schemes:
            name = run_name(prefix, g, model, s, "", None, target)
            run_dir = os.path.join(exp_root, name, "cv")
            info = check_run(run_dir, name, target, "cv", region)
            if infos is not None:
                infos.append(info)
            p = os.path.join(run_dir, "oof_points.csv")
            if not os.path.isfile(p):
                loi(f"{name}: thieu {p}")
            d = pd.read_csv(p, dtype={"point_id": str, "unit_id": str},
                            usecols=["grid", "point_id", "season", "unit_id", "err", "pred_source"])
            if (d["pred_source"] != "oof").any():
                raise ValueError(f"{p}: co dong khong phai oof")
            d["model"], d["seed"] = model, s
            parts.append(drop_points(d, exclude))
    return pd.concat(parts, ignore_index=True)


def point_units(err, folds_csv, eval_dir=None):
    """point_id -> unit_id cho MOI diem danh gia (gom diem giu rieng - chi de doi chieu don vi, khong dung gia tri),
    tinh lai tu vi tri diem (point_eval.point_blocks); kiem khop unit_id trong oof_points."""
    import geopandas as gpd

    from training.point_eval import point_blocks

    folds = pd.read_csv(folds_csv, dtype={"block_id": str, "unit_id": str})
    eval_dir = eval_dir or os.path.join(ROOT, "data", "eval")
    pts = gpd.read_file(os.path.join(eval_dir, "eval_points.geojson"))
    blocks = gpd.read_file(os.path.join(eval_dir, "holdout_blocks.geojson"))
    pb = point_blocks(pts, blocks, folds).set_index("point_id")
    pu = pb["unit_id"].astype(str)
    got = err.drop_duplicates(["point_id", "unit_id"]).set_index("point_id")["unit_id"]
    if got.index.duplicated().any() or (got != pu.reindex(got.index)).any():
        raise ValueError("unit_id trong oof_points khac don vi tinh tu vi tri diem")
    cv_units = sorted(folds.loc[folds["cv_fold"] >= 0, "unit_id"].unique())
    ho_units = sorted(folds.loc[folds["cv_fold"] < 0, "unit_id"].unique())
    return pu, cv_units, ho_units


def common_subset(err):
    """Loc err theo tap (diem, mua) hop le o MOI luoi cua err (hop le = err huu han o moi mo hinh / cach chia)."""
    keys = [err["grid"], err["point_id"], err["season"]]
    valid = err["err"].notna().groupby(keys).all().rename("valid").reset_index()
    common, _ = common_point_set(valid)
    key = pd.MultiIndex.from_frame(err[["point_id", "season"]])
    return err[key.isin(common)], len(common)


def run_family(err, pu, pairs, levels, cv_units, ho_units, model, family, alpha, delta_min, target="salinity",
               gate=None, expected_m=None, holm_tiers=None):
    e, n_common = common_subset(err)
    out = compare_tiered(e, pu, pairs, target, levels=levels, expected_m=expected_m, holm_tiers=holm_tiers,
                         cv_units=cv_units,
                         holdout_units=ho_units, holdout_seasons=(2020,), mode="cv", delta_min=delta_min,
                         delta_min_kind="rel", alpha=alpha, delta_ref="level_mean", require_practical=True,
                         model=model, family=family)
    out = apply_gate(out, gate)
    out.insert(0, "n_common_point_seasons", n_common)
    return out


def main(a):
    if a.target not in TESTED_TIERS:
        loi(f"--target '{a.target}' khong co trong TESTED_TIERS")
    resolve_region(a)
    os.makedirs(a.out_dir, exist_ok=True)
    names = all_scoped_names(os.path.join(a.out_dir, out_name(a.target, a.prefix)), a.model)
    guard_frozen(names + [provenance_path(p) for p in names], a.frozen_manifest)
    gate_path = a.gate or os.path.join(a.out_dir, gate_name(a.model))
    try:
        gate = load_gate(gate_path, a.target, required=gate_required(a.target, a.model), model=a.model)
    except ValueError as exc:
        loi(str(exc))
    outs = [(s, os.path.join(a.out_dir, model_name(n, a.model)))
            for s, n in scope_outputs(a.target, a.prefix, gate, a.holm_scope)]
    at = area_table(a.area_table)
    main_pairs, warn = select_pairs(at, a.region, a.pairs_mode)
    for w in warn:
        print(f"CANH BAO cap dbscl12: {w}", flush=True)
    levels = area_levels(at, pairs=main_pairs)
    s2_pairs = side_pairs(at, main_pairs)
    infos, results = [], {s: [] for s, _ in outs}
    main_fam = main_family(a.model)
    fams = ((("hist_gb", main_fam), ("linear", "phu_linear"), ("idw", "phu_idw")) if a.model == "hist_gb"
            else ((run_model(a.model), main_fam),))
    for model, fam in fams:
        err = load_errors(a.exp_root, a.prefix, model, a.schemes, target=a.target, infos=infos, region=a.region,
                          exclude=a.exclude)
        pu, cv_units, ho_units = point_units(err, a.folds_csv, **a.eval_kw)
        check_excluded(a.exclude, pu)
        for scope, _ in outs:
            ht = holm_tiers(a.target, gate, scope)
            m_main = f1_m(ht, main_pairs, a.region)
            kw = dict(target=a.target, gate=gate, holm_tiers=ht)
            rs = results[scope]
            rs.append(run_family(err, pu, main_pairs, levels, cv_units, ho_units, model, fam, a.alpha, a.delta_min,
                                 expected_m=m_main, **kw))
            if fam == main_fam:
                if a.model == "hist_gb" and len(s2_pairs):
                    lv2 = area_levels(at, pairs=s2_pairs)
                    rs.append(run_family(err, pu, s2_pairs, lv2, cv_units, ho_units, model, "phu_S2_ti_le_1.2-1.5",
                                         a.alpha, a.delta_min, **kw))
                for s in a.schemes:  # do nhay: tung cach chia rieng (khong trung binh)
                    es = err[err["seed"] == s]
                    rs.append(run_family(es, pu, main_pairs, levels, cv_units, ho_units, model,
                                         f"do_nhay_chi_s{s}", a.alpha, a.delta_min, expected_m=m_main, **kw))
    same = check_consistent(infos, a.allowed_tags)
    warn_g(a.region, cv_units)
    for i, (scope, out) in enumerate(outs):
        res = pd.concat(results[scope], ignore_index=True)
        res["git_tag"], res["points_ref_sha256"] = same["git_tag"], same["points_ref_sha256"]
        res["n_luot"] = len(infos)
        res["cong_file"] = gate_path if gate is not None else ""
        if a.model != "hist_gb":
            res["vai_tro"] = SENS_ROLE
        write_csv_atomic(res, out)
        hp = holm_provenance(a.target, gate, gate_path, scope, res)
        hp["holm_m_f1"] = f1_m(holm_tiers(a.target, gate, scope), main_pairs, a.region)
        write_provenance(out, **hp, prefix=a.prefix,
                         git_tag=same["git_tag"], points_ref_sha256=same["points_ref_sha256"], n_luot=len(infos),
                         **model_provenance(a.model), **tag_info(same, a.allowed_tags, a.tags_reason),
                         **region_info(a, cv_units, res, main_pairs, warn))
        if i == 0 and not a.no_table:
            cols = [c for c in ["family", "grid_a", "grid_b", "kiem_dinh", "delta_hat", "ci_low", "ci_high",
                                "p_holm", "seeds_same_dir", "k_over_G", "label"] if c in res.columns]
            with pd.option_context("display.width", 250, "display.max_columns", 30,
                                   "display.float_format", "{:.4f}".format):
                print(res[cols].to_string(index=False))
    print(f"\ntarget {a.target} | model {a.model} | git_tag {same['git_tag']} | {len(infos)} luot | cong: {gate}")
    for scope, out in outs:
        ht = holm_tiers(a.target, gate, scope)
        print(f"Holm {scope}: muc {sorted(ht)}, m F1 = {f1_m(ht, main_pairs, a.region)} | Ghi: {out}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="cv1")
    ap.add_argument("--target", default="salinity", help="salinity (mac dinh) | ndwi | rain_chirps | dsr_mcd18 | ...")
    ap.add_argument("--schemes", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--area-table", default=None, help=f"mac dinh theo vung (dbscl: KE_HOACH/ket-qua/{AREA_FILE})")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta-min", type=float, default=0.05)
    ap.add_argument("--out-dir", default=None, help="mac dinh theo vung (dbscl: KE_HOACH/ket-qua)")
    ap.add_argument("--gate", default=None, help=f"bang cong (mac dinh <out-dir>/{GATE_FILE}; rf/mlp: dot7_cong_kiem_dinh__<model>.csv)")
    ap.add_argument("--exp-root", default=None, help="mac dinh theo vung (dbscl: artifacts/experiments)")
    ap.add_argument("--folds-csv", default=None, help="mac dinh theo vung (dbscl: data/eval/cv_folds.csv)")
    add_common_args(ap)
    add_region_args(ap)
    ap.add_argument("--holm-scope", choices=HOLM_SCOPES, default="cong",
                    help="cong: Holm tren muc hop le o cong (mac dinh); muc_kiem_dinh: moi muc kiem dinh (do nhay)")
    ap.add_argument("--no-table", action="store_true", help="khong in bang so ra stdout")
    return ap


def add_region_args(ap):
    """--region, --pairs-mode, --exclude-points (analyze_cv_family / analyze_final / analyze_learnability)."""
    ap.add_argument("--region", choices=list(REGIONS), default=DEFAULT_REGION,
                    help="dbscl (mac dinh, duong dan cu) | mississippi | nile")
    ap.add_argument("--pairs-mode", choices=PAIRS_MODES, default=None,
                    help="dbscl12: 12 cap DBSCL; ratio: pairs_by_area(1,2) cua vung. dbscl mac dinh dbscl12; vung: bat buoc")
    ap.add_argument("--exclude-points", default=None,
                    help="file point_id loai khoi tap diem chung (vung: mac dinh configs/exclude_points/<vung>.txt)")


def add_common_args(ap):
    """--model, --frozen-manifest, --allowed-tags, --tags-reason dung chung 4 script tong hop."""
    ap.add_argument("--model", choices=MODELS, default="hist_gb",
                    help="hist_gb (chinh) | rf | mlp (do nhay theo mo hinh, file ra hau to __<model>)")
    ap.add_argument("--frozen-manifest", nargs="+", default=FROZEN_MANIFESTS,
                    help="mot hoac nhieu manifest dong bang (thieu file nao -> ma 2)")
    ap.add_argument("--allowed-tags", nargs="+", default=None,
                    help="cho phep nhieu git_tag giua cac luot (tap tag thuc te phai la tap con)")
    ap.add_argument("--tags-reason", default=MULTI_TAG_REASON, help="ly do nhieu tag ghi vao provenance")


if __name__ == "__main__":
    main(build_parser().parse_args())

#!/usr/bin/env python
"""Do nhay kich thuoc khoi cua boc tach hybrid - CHI MO TA (CHG-21, An chot 2026-10-05 21:22, TRUOC khi mo bat ky
ket qua khoi 100 km nao cua boc tach hybrid).

I = MAE(b) - MAE(a), MAE gop theo (diem, mua); (b) = bo dac trung `khong_diem`, (a) = day du; HistGB.
Moi luoi:
  I50_s   (s = s42/s43/s44, khoi 50 km, prefix cv1); dai nhieu cach chia 50 km = [min, max] I50_s; I_50 = TB I50_s
  I100_s  (P1 = s42, P2 = s43, khoi ~100 km, prefix k100); I_100 = TB I100_s
  trong_dai_nhieu_50km = min I50_s <= I_100 <= max I50_s
Tap (diem, mua): CUNG MOT tap o MOI lan chay cua ca 4 nhom (a/b x 50/100 km) va = --expected-n (86.073).
Khong co nhan "vung"/"phu thuoc" (quy tac 21:06 thay bang CHG-21 - o khoi 50 km khong co hieu ung de lam moc).
Tom tat in ra (mo ta): khoang I_100, so luoi I_100 > 0, so luoi trong dai nhieu 50 km. Luoi trong dai -> "khong co
bang chung hieu ung thay doi theo kich thuoc khoi"; ngoai dai -> chi bao so, khong gan nhan.
Kiem dinh chinh thuc cua boc tach van o khoi 50 km (analyze_hybrid.py).
CHG-22 (3 trang thai): thieu file / pred_source khac oof / err NaN / trung (diem, mua) / tap (diem, mua) khac nhau
giua cac lan chay / tap chung rong hoac != --expected-n -> LOI, dung, khong ghi ket qua (file ket qua cu bi xoa
khi bat dau; ghi file tam roi os.replace).
--target (Dot 7; mac dinh salinity = ten file/byte cu): luot run_name ...__t-<t>, (b) ...__fs-khong_diem__t-<t>;
  run_meta/config ma 0, dung target + feature_set, tag theo --allowed-tags; ra dot7_<t>_hybrid_khoi100.csv + provenance.
  File ra trong manifest dong bang -> ma 2 truoc khi doc/ghi.

Chay:  venv/Scripts/python.exe scripts/analyze_hybrid_block_size.py
       Bien Dot 7: ... --target ndwi --allowed-tags nckh-dot7-e6a nckh-dot7-hyb
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze_cv_family import FROZEN_MANIFESTS, GRIDS, check_consistent, check_run, loi, tag_info  # noqa: E402
from analyze_hybrid import TAGS_REASON  # noqa: E402
from run_experiments import run_name  # noqa: E402
from training.dot7_rules import TESTED_TIERS, guard_frozen, provenance_path, write_provenance  # noqa: E402

FEATURE_SET = "khong_diem"
EXPECTED_N = 86073
SEEDS = {"k50": (42, 43, 44), "k100": (42, 43)}  # k100: P1 = s42, P2 = s43
OUT_NAME = "dot6_hybrid_khoi100.csv"
CAU_TRONG_DAI = "khong co bang chung hieu ung thay doi theo kich thuoc khoi"


def out_name(target):
    return OUT_NAME if target == "salinity" else f"dot7_{target}_hybrid_khoi100.csv"


def run_dir(exp_root, prefix, grid, scheme, feature_set=None, target="salinity"):
    return os.path.join(exp_root, run_name(prefix, grid, "hist_gb", scheme, "", feature_set, target), "cv",
                        "oof_points.csv")


def read_err(path) -> pd.Series:
    """err theo (point_id, season), da sap xep. Thieu file / khong phai oof / NaN / trung -> LOI."""
    if not os.path.exists(path):
        raise SystemExit(f"LOI: thieu {path}")
    d = pd.read_csv(path, dtype={"point_id": str}, usecols=["point_id", "season", "err", "pred_source"])
    if (d["pred_source"] != "oof").any():
        raise SystemExit(f"LOI: {path} co dong khong phai oof")
    if not np.isfinite(d["err"].to_numpy(float)).all():
        raise SystemExit(f"LOI: {path} co {int((~np.isfinite(d['err'].to_numpy(float))).sum())} err NaN/khong huu han")
    s = d.set_index(["point_id", "season"])["err"].sort_index()
    if s.index.duplicated().any():
        raise SystemExit(f"LOI: {path} trung (diem, mua)")
    return s


def runs(a):
    for design, prefix in (("k50", a.prefix50), ("k100", a.prefix100)):
        for arm, fs in (("a", None), ("b", a.feature_set)):
            for g in GRIDS:
                for s in SEEDS[design]:
                    yield design, arm, g, s, fs, run_dir(a.exp_root, prefix, g, s, fs, a.target)


def mae_table(a, infos=None):
    """MAE moi lan chay tren tap (diem, mua) chung - moi lan chay phai CUNG tap (khong cat giao am tham).
    infos (list) -> kiem run_meta/config tung luot va noi vao infos."""
    keys, ref_path, rows = None, None, []
    expected_n = a.expected_n if a.expected_n is not None or a.target != "salinity" else EXPECTED_N
    for design, arm, g, s, fs, p in runs(a):
        if infos is not None:
            name = os.path.basename(os.path.dirname(os.path.dirname(p)))
            info = check_run(os.path.dirname(p), name, a.target, "cv")
            if info["meta"].get("feature_set") != fs:
                loi(f"{name}: feature_set '{info['meta'].get('feature_set')}' khac '{fs}'")
            infos.append(info)
        e = read_err(p)
        if keys is None:
            keys, ref_path = e.index, p
            if len(keys) == 0:
                raise SystemExit("LOI: tap (diem, mua) chung rong")
            if expected_n is not None and len(keys) != expected_n:
                raise SystemExit(f"LOI: tap (diem, mua) {len(keys)} != ky vong {expected_n} ({p})")
        elif not e.index.equals(keys):
            raise SystemExit(f"LOI: {p} khac tap (diem, mua) voi {ref_path} "
                             f"({len(e.index.difference(keys))} chi o file nay, {len(keys.difference(e.index))} thieu)")
        rows.append({"design": design, "arm": arm, "grid": g, "seed": s, "MAE": float(e.abs().mean())})
    return pd.DataFrame(rows), len(keys)


def block_size_table(m: pd.DataFrame, n_keys: int) -> pd.DataFrame:
    piv = m.pivot_table(index=["design", "grid", "seed"], columns="arm", values="MAE", aggfunc="first")
    i = (piv["b"] - piv["a"]).unstack(["design", "seed"])
    want = [("k50", s) for s in SEEDS["k50"]] + [("k100", s) for s in SEEDS["k100"]]
    i = i.reindex(index=GRIDS, columns=pd.MultiIndex.from_tuples(want, names=["design", "seed"]))
    if not np.isfinite(i.to_numpy(float)).all():
        raise SystemExit("LOI: thieu / khong huu han I o mot luoi / lan chay")
    t = pd.DataFrame({"grid": GRIDS, "n_diem_mua_chung": n_keys})
    for s in SEEDS["k50"]:
        t[f"I50_s{s}"] = i[("k50", s)].to_numpy()
    i50 = t[[f"I50_s{s}" for s in SEEDS["k50"]]]
    t["dai50_min"], t["dai50_max"], t["I_50"] = i50.min(axis=1), i50.max(axis=1), i50.mean(axis=1)
    for s, ten in zip(SEEDS["k100"], ("P1", "P2")):
        t[f"I100_{ten}_s{s}"] = i[("k100", s)].to_numpy()
    t["I_100"] = t[[f"I100_{ten}_s{s}" for s, ten in zip(SEEDS["k100"], ("P1", "P2"))]].mean(axis=1)
    t["trong_dai_nhieu_50km"] = (t["dai50_min"] <= t["I_100"]) & (t["I_100"] <= t["dai50_max"])
    return t


def summary_lines(t: pd.DataFrame) -> list:
    n = len(t)
    trong = t.loc[t["trong_dai_nhieu_50km"], "grid"].tolist()
    ngoai = t.loc[~t["trong_dai_nhieu_50km"]]
    lines = [f"I_100 (khoi ~100 km, TB P1/P2): tu {t['I_100'].min():+.4f} den {t['I_100'].max():+.4f} dS/m",
             f"So luoi I_100 > 0: {int((t['I_100'] > 0).sum())}/{n}",
             f"So luoi I_100 trong dai nhieu cach chia 50 km [min, max] I50_s: {len(trong)}/{n}"]
    if trong:
        lines.append(f"  Trong dai ({', '.join(trong)}): {CAU_TRONG_DAI}.")
    for r in ngoai.itertuples():  # ngoai dai: chi bao so, khong gan nhan
        lines.append(f"  Ngoai dai: {r.grid}: I_100 = {r.I_100:+.4f}, dai 50 km [{r.dai50_min:+.4f}; {r.dai50_max:+.4f}]")
    return lines


def main(a):
    salt = a.target == "salinity"
    if a.target not in TESTED_TIERS:
        loi(f"--target '{a.target}' khong co trong TESTED_TIERS {sorted(TESTED_TIERS)}")
    out = os.path.join(a.out_dir, out_name(a.target))
    guard_frozen([out, provenance_path(out)], a.frozen_manifest)  # truoc khi xoa ket qua cu
    os.makedirs(a.out_dir, exist_ok=True)
    for p in (out, provenance_path(out)):
        if os.path.exists(p):  # dung giua chung (LOI) khong de lai ket qua cu
            os.remove(p)
    infos = None if salt else []
    m, n_keys = mae_table(a, infos)
    same = None if salt else check_consistent(infos, a.allowed_tags)
    print(f"(diem, mua) chung: {n_keys}", flush=True)
    t = block_size_table(m, n_keys)
    tmp = out + ".tmp"
    t.to_csv(tmp, index=False)
    os.replace(tmp, out)
    if not salt:  # do man: giu dung file cu (khong sidecar)
        write_provenance(out, target=a.target, quy_tac="CHG-21 mo ta: I_100 so voi dai nhieu cach chia 50 km",
                         prefix50=a.prefix50, prefix100=a.prefix100, feature_set_b=a.feature_set,
                         schemes={k: list(v) for k, v in SEEDS.items()}, grids=list(GRIDS), expected_n=a.expected_n,
                         n_diem_mua_chung=n_keys, git_tag=same["git_tag"], points_ref_sha256=same["points_ref_sha256"],
                         n_luot=len(infos), **tag_info(same, a.allowed_tags, a.tags_reason))
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(t.round(4).to_string(index=False))
    print("\n".join(summary_lines(t)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix50", default="cv1")
    ap.add_argument("--prefix100", default="k100")
    ap.add_argument("--target", default="salinity", help="salinity (mac dinh, file dot6 cu) | ndwi | rain_chirps | ...")
    ap.add_argument("--feature-set", default=FEATURE_SET)
    ap.add_argument("--expected-n", type=int, default=None,
                    help=f"so (diem, mua) chung ky vong; khac -> LOI. Mac dinh salinity {EXPECTED_N}, bien khac "
                         "chi doi chieu cung tap giua moi luot")
    ap.add_argument("--exp-root", default=os.path.join(ROOT, "artifacts", "experiments"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "KE_HOACH", "ket-qua"))
    ap.add_argument("--frozen-manifest", nargs="+", default=FROZEN_MANIFESTS)
    ap.add_argument("--allowed-tags", nargs="+", default=None, help="cho phep nhieu git_tag (bien Dot 7: (a) + (b))")
    ap.add_argument("--tags-reason", default=TAGS_REASON)
    main(ap.parse_args())

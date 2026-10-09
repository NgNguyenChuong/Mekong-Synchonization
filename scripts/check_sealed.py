#!/usr/bin/env python
"""So mot file ket qua moi voi ban niem phong: sha256 trung -> DAT; khac -> so tung o cung khoa (so: |d| <= tol,
chuoi: bang nhau). Chi in thong ke lech (so o, max |d|, ten cot), KHONG in gia tri. Ma thoat: 0 DAT, 2 KHONG DAT.

Chay:  venv/Scripts/python.exe scripts/check_sealed.py <file_moi.csv> <file_niem_phong.csv>
           [--key family cmp_id] [--cols delta_hat ci_low ...] [--exclude cong_file] [--tol 1e-9]
"""
import argparse
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from training.dot7_rules import file_sha256  # noqa: E402

FAIL = 2
USAGE = 1


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # ma 2 danh cho KHONG DAT
        self.print_usage(sys.stderr)
        self.exit(USAGE, f"LOI tham so: {message}\n")


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def compare_cells(a: pd.Series, b: pd.Series, tol: float):
    """(n_o, n_vuot, max_abs_d) giua hai cot chuoi tho; o rong = NaN, NaN chi bang NaN."""
    n_bad, dmax = 0, 0.0
    for x, y in zip(a, b):
        if x == y:
            continue
        fx, fy = _num(x), _num(y)
        if fx is None or fy is None:
            n_bad += 1
            continue
        if math.isnan(fx) and math.isnan(fy):
            continue
        if math.isnan(fx) or math.isnan(fy) or math.isinf(fx) or math.isinf(fy):
            n_bad += 1
            continue
        d = abs(fx - fy)
        dmax = max(dmax, d)
        if d > tol:
            n_bad += 1
    return len(a), n_bad, dmax


def check(new_path, sealed_path, key=("family", "cmp_id"), cols=None, exclude=(), tol=1e-9, out=print) -> int:
    for p in (new_path, sealed_path):
        if not os.path.isfile(p):
            out(f"LOI: khong co file {p}")
            return USAGE
    sha_new, sha_old = file_sha256(new_path), file_sha256(sealed_path)
    out(f"sha256 moi {sha_new}")
    out(f"sha256 niem phong {sha_old}")
    if sha_new == sha_old:
        out("DAT: sha256 trung")
        return 0
    out("sha256 khac -> so tung o")
    rd = dict(dtype=str, keep_default_na=False)
    n, s = pd.read_csv(new_path, **rd), pd.read_csv(sealed_path, **rd)
    key = list(key)
    problems = []
    for name, df in (("moi", n), ("niem phong", s)):
        miss = [k for k in key if k not in df.columns]
        if miss:
            out(f"KHONG DAT: file {name} thieu cot khoa {miss}")
            return FAIL
        dup = int(df.duplicated(key).sum())
        if dup:
            problems.append(f"file {name} co {dup} dong trung khoa")
    if list(n.columns) != list(s.columns):
        out("ghi chu: danh sach / thu tu cot khac nhau")
    if cols:
        use = [c for c in cols if c not in key]
        miss = [c for c in use if c not in n.columns or c not in s.columns]
        if miss:
            problems.append(f"thieu cot can so {miss}")
            use = [c for c in use if c not in miss]
    else:
        only_n = [c for c in n.columns if c not in s.columns and c not in exclude]
        only_s = [c for c in s.columns if c not in n.columns and c not in exclude]
        if only_n or only_s:
            problems.append(f"cot chi o file moi {only_n}, chi o ban niem phong {only_s}")
        use = [c for c in s.columns if c in n.columns and c not in key]
    use = [c for c in use if c not in exclude]
    if exclude:
        out(f"bo qua cot: {sorted(set(exclude))}")
    kn = set(map(tuple, n[key].to_numpy()))
    ks = set(map(tuple, s[key].to_numpy()))
    if kn != ks:
        problems.append(f"khoa chi o file moi: {len(kn - ks)}, chi o ban niem phong: {len(ks - kn)}")
    if not problems and list(map(tuple, n[key].to_numpy())) != list(map(tuple, s[key].to_numpy())):
        out("ghi chu: thu tu dong khac nhau (so theo khoa)")
    m = n.drop_duplicates(key).merge(s.drop_duplicates(key), on=key, how="inner", suffixes=("__moi", "__np"))
    total, bad_total, dmax_all, bad_cols = 0, 0, 0.0, []
    for c in use:
        k, nb, dm = compare_cells(m[f"{c}__moi"], m[f"{c}__np"], tol)
        total += k
        bad_total += nb
        dmax_all = max(dmax_all, dm)
        if nb or dm > 0:
            out(f"  cot {c}: {nb} o vuot dung sai, max |d| = {dm:.3g}")
        if nb:
            bad_cols.append(c)
    out(f"so {len(use)} cot x {len(m)} dong = {total} o; vuot dung sai {tol:g}: {bad_total}; max |d| = {dmax_all:.3g}")
    for p in problems:
        out(f"KHONG DAT: {p}")
    if bad_cols:
        out(f"KHONG DAT: cot lech {bad_cols}")
    if problems or bad_cols:
        return FAIL
    out("DAT: moi o trong dung sai")
    return 0


def build_parser():
    ap = _Parser(description=__doc__.splitlines()[0])
    ap.add_argument("new")
    ap.add_argument("sealed")
    ap.add_argument("--key", nargs="+", default=["family", "cmp_id"])
    ap.add_argument("--cols", nargs="+", default=None, help="chi so cac cot nay (mac dinh: moi cot)")
    ap.add_argument("--exclude", nargs="+", default=[], help="bo qua cac cot nay (ghi ro trong ket qua)")
    ap.add_argument("--tol", type=float, default=1e-9)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return check(a.new, a.sealed, key=a.key, cols=a.cols, exclude=a.exclude, tol=a.tol,
                 out=lambda msg: print(msg, flush=True))


if __name__ == "__main__":
    sys.exit(main())

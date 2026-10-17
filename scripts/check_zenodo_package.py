#!/usr/bin/env python
"""Kiem goi Zenodo nhap: sha/bytes theo MANIFEST, file thua/thieu, tung muc ke hoach so voi nguon (copy: sha; text: chuan hoa; csvdrop: bo cot MRC).

Chay:  venv/Scripts/python.exe scripts/check_zenodo_package.py [--pkg DIR]
"""
import argparse
import csv
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_zenodo_package as bz  # noqa: E402


def main(pkg):
    errs = []
    man = {r["path"]: r for r in csv.DictReader(open(os.path.join(pkg, "MANIFEST.csv"), encoding="utf-8"))}
    on_disk = {os.path.relpath(os.path.join(d, f), pkg).replace("\\", "/") for d, _, fs in os.walk(pkg) for f in fs} - {"MANIFEST.csv"}
    errs += [f"khong co trong MANIFEST: {p}" for p in sorted(on_disk - set(man))]
    errs += [f"MANIFEST co nhung thieu file: {p}" for p in sorted(set(man) - on_disk)]
    for p in sorted(on_disk & set(man)):
        f = os.path.join(pkg, p)
        if os.path.getsize(f) != int(man[p]["bytes"]) or bz.file_sha256(f) != man[p]["sha256"]:
            errs.append(f"sha/bytes lech MANIFEST: {p}")
    roots, n = bz._roots(), 0
    with tempfile.TemporaryDirectory() as tmp:
        for it in bz.build_plan():
            dst, n = os.path.join(pkg, it.dest), n + 1
            if not os.path.isfile(dst):
                errs.append(f"thieu muc ke hoach: {it.dest}")
                continue
            if it.mode == "csvdrop":
                t = os.path.join(tmp, "x.csv")
                bz.drop_columns_csv(it.src, t)
                want = bz.file_sha256(t)
            elif it.mode == "text":
                raw = open(it.src, "rb").read()
                try:
                    want = bz.hashlib.sha256(bz.normalize_text(raw.decode("utf-8"), roots).encode("utf-8")).hexdigest()
                except UnicodeDecodeError:
                    want = bz.file_sha256(it.src)
            else:
                want = bz.file_sha256(it.src)
            if want != bz.file_sha256(dst):
                errs.append(f"khac nguon ({it.mode}): {it.dest}")
    print(f"MANIFEST {len(man)} dong, {len(on_disk)} file, {n} muc ke hoach so voi nguon: {len(errs)} loi", flush=True)
    for e in errs[:40]:
        print("  LOI", e)
    return 1 if errs else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkg", default=os.getenv("ZENODO_PKG_DIR") or os.path.join(os.path.dirname(bz.doi_chung_out()), "zenodo_pkg"))
    sys.exit(main(ap.parse_args().pkg))

#!/usr/bin/env python
"""Dung goi Zenodo NHAP (phuong an B): du lieu dan xuat + bang dac trung + ma, KHONG luot mo hinh, du lieu tho, MRC.
Cot MRC (tch_*, tram_*) bi bo khoi bang dac trung; provenance chuan hoa duong dan may; cuoi cung quet nhay cam.

Chay:  venv/Scripts/python.exe scripts/build_zenodo_package.py [--out DIR] [--ref HEAD] [--license-data-file F] [--license-code {MIT,Apache-2.0}] [--dry-run] [--force]
"""
import argparse
import csv
import dataclasses
import datetime
import getpass
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from region import doi_chung_out, region_paths  # noqa: E402
from settings import require_data_root, data_path  # noqa: E402  (.env: DATA_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REGIONS = ("dbscl", "mississippi", "nile")
DOT7_VARS = ("dsr_mcd18", "ndwi", "rain_chirps", "rh_era5", "t2m_era5")
MRC_PREFIXES = ("tch_", "tram_")  # chuoi muc nuoc MRC (Tan Chau, 6 tram): khong phan phoi lai dan xuat
CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"
CODE_LICENSE_URL = {"Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0.txt"}
MIT_TEXT = """MIT License

Copyright (c) 2026 The authors of this record

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
CRMS_ZIP_URL = "https://cims.coastal.louisiana.gov/RequestedDownloads/ZippedFiles/CRMS_Discrete_Hydrographic.zip"
CRMS_ZIP_SHA256 = "a599cdf7c02d56a1352457273393701d26ddfcf45424760a0c546cd9b564931f"  # zip tai 2026-10-17 (Last-Modified 2026-10-11)
RESTORE_TEXT = ("Some or all of the associated data and related items of information shown or delivered as part of this "
                "data/information inquiry were prepared by the Coastal Protection and Restoration Authority under an award "
                "from the Gulf Coast Ecosystem Restoration Council (RESTORE Council). The data, statements, findings, "
                "conclusions, and recommendations are those of the author[s] and do not necessarily reflect any "
                "determinations, views, or policies of the RESTORE Council.")
FETCH_SCRIPTS = ("fetch_doi_chung.py", "gee_fetch.py", "fetch_mrc_timeseries.py", "fetch_mrc_water_level.py")
CODE_PATHS = ["src", "scripts", "configs", "tests", "component", "main.py", "pytest.ini", "requirements.txt", "README.md", ".env.example"]


@dataclasses.dataclass
class Item:
    src: str
    dest: str  # tuong doi goc goi
    region: str = ""
    variable: str = ""
    origin: str = ""  # script sinh (neu sidecar khong ghi)
    mode: str = "copy"  # copy | csvdrop | text


# ---------------------------------------------------------------- chuan hoa duong dan / loc cot
def _roots():
    out = doi_chung_out().replace("\\", "/")
    return [(out, "<DOI_CHUNG_OUT>"), (os.path.dirname(out), "<OUT_ROOT>"),
            (require_data_root().replace("\\", "/"), "<DATA_ROOT>"), (ROOT.replace("\\", "/"), "<REPO>")]


ABS_PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;)\]]*")


def normalize_text(s, roots=None):
    """Duong dan tuyet doi kieu o dia -> <DATA_ROOT>/<DOI_CHUNG_OUT>/<OUT_ROOT>/<REPO>; con lai -> <LOCAL>/<ten file>."""
    roots = _roots() if roots is None else roots

    def sub(m):
        p = m.group(0).replace("\\", "/")
        for root, tok in roots:
            if p.lower() == root.lower() or p.lower().startswith(root.lower() + "/"):
                return tok + p[len(root):]
        return "<LOCAL>/" + p.rstrip("/").rsplit("/", 1)[-1]
    return ABS_PATH.sub(sub, s)


def normalize_json(o, roots=None):
    if isinstance(o, str):
        return normalize_text(o, roots)
    if isinstance(o, list):
        return [normalize_json(x, roots) for x in o]
    if isinstance(o, dict):
        return {normalize_text(k, roots): normalize_json(v, roots) for k, v in o.items()}
    return o


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def drop_columns_csv(src, dst, prefixes=MRC_PREFIXES):
    """Ghi lai CSV khong co cot bat dau bang prefixes (giu nguyen chuoi cua cac cot con lai). Tra ve danh sach cot da bo."""
    with open(src, "rb") as f:
        head = f.read(65536)
    eol = "\r\n" if b"\r\n" in head else "\n"
    with open(src, newline="", encoding="utf-8") as f, open(dst, "w", newline="", encoding="utf-8") as g:
        r = csv.reader(f)
        header = next(r)
        keep = [i for i, c in enumerate(header) if not c.startswith(prefixes)]
        dropped = [c for c in header if c.startswith(prefixes)]
        w = csv.writer(g, lineterminator=eol)
        w.writerow([header[i] for i in keep])
        for row in r:
            w.writerow([row[i] for i in keep])
    return dropped


# ---------------------------------------------------------------- ke hoach: file nao vao goi
def _glob(items, d, pats, dest, region, var="", origin="", mode="copy", exclude=()):
    for pat in pats:
        for p in sorted(glob.glob(os.path.join(d, pat))):
            n = os.path.basename(p)
            if os.path.isfile(p) and not n.endswith(".provenance.json") and not any(re.search(e, n) for e in exclude):
                items.append(Item(p.replace("\\", "/"), f"{dest}/{n}", region, var, origin, mode))


def build_plan(repo=ROOT):
    items = []
    ket_qua = os.path.join(repo, "KE_HOACH", "ket-qua").replace("\\", "/")
    for r in REGIONS:
        P = region_paths(r)
        sal = "salinity" if r == "dbscl" else "ndwi"
        _glob(items, P["grids_dir"], ["*.geojson"], f"grids/{r}", r, origin="generate_all_grids.py")
        items.append(Item(P["boundary"].replace("\\", "/"), f"grids/{r}/boundary.geojson", r, origin="build_boundary_v2.py / fetch_doi_chung.py"))
        for sub in ("cell_blocks", "khoi100"):
            _glob(items, os.path.join(P["eval_dir"], sub), ["*.csv"], f"eval/{r}/{sub}", r, origin="build_cv_folds.py")
        _glob(items, P["eval_dir"], ["eval_points.geojson", "holdout_blocks.geojson", "cv_folds*.csv"], f"eval/{r}", r, origin="build_eval_design.py")
        L, F = P["labels_dir"], P["features_dir"]
        if r == "dbscl":
            _glob(items, L, ["*_labels_season.csv", "points_reference.csv"], f"labels/{r}/salinity", r, sal, "build_labels_season.py",
                  exclude=[r"keep"])
        _glob(items, os.path.join(L, "dot7"), ["*.csv"], f"labels/{r}/dot7", r, "", "build_met_labels.py")
        _glob(items, os.path.join(F, "unified"), ["*_unified.csv"], f"features/{r}/unified", r, sal, "build_unified_table.py", "csvdrop",
              exclude=[r"keep"])
        for v in DOT7_VARS:
            _glob(items, os.path.join(F, "unified_dot7", v), ["*_unified.csv"], f"features/{r}/unified_dot7/{v}", r, v,
                  "build_unified_dot7.py", "csvdrop")
        area = f"{ket_qua}/dot4_doi_chieu_dien_tich.csv" if r == "dbscl" else f"{P['report_dir']}/dot4_doi_chieu_dien_tich.csv".replace("\\", "/")
        items.append(Item(area, f"grids/{r}/dot4_doi_chieu_dien_tich.csv", r, origin="generate_all_grids.py", mode="text"))
        res = ket_qua if r == "dbscl" else f"{ket_qua}/doi_chung/{r}"
        dest = {"dbscl": "results/dbscl_a"}.get(r, f"results/{r}")
        pats = ["*.csv", "*.md"] if r != "dbscl" else [
            "comparison_ledger.csv", "comparison_ledger.md", "dot7_cong_kiem_dinh.csv", "dot5_cv1_*.csv", "dot5_final_*.csv",
            "dot5_chg19_*.csv", "dot7_*_cv1_kiem_dinh_khoi*.csv", "dot7_*_cv1_final_cap*.csv", "dot7_*_cv1_final_nhom*.csv",
            "dot7_*_hoc_duoc*.csv", "dot7_salinity_*.csv"]
        _glob(items, res, pats, dest, r, "", "", "text")
    _glob(items, f"{ket_qua}/dbscl_fs_khong_diem", ["*.csv", "*.md"], "results/dbscl_b", "dbscl", "", "", "text")
    _glob(items, ket_qua, ["bang_3_vung.*", "pc1_ci.*", "do_nhay_nguong.*", "bang_nhan_dac_trung.md", "dot7_pc1_*.csv"], "results", "", "", "", "text", exclude=[r"\.provenance"])
    _glob(items, ket_qua, ["dot4_scope_v3_dien_tich.csv"], "grids/dbscl", "dbscl", "", "build_scope_mask.py", "text")
    _glob(items, f"{ket_qua}/hinh/paper", ["fig*.png", "fig*.pdf", "table*.csv", "table*.md"], "results/figures", "", "", "plot_paper_figures.py",
          exclude=[r"_thumb"])
    st = data_path("doi_chung", "mississippi", "stations").replace("\\", "/")
    _glob(items, st, ["noaa_*.csv", "usgs_*.csv", "NOAA_khong_co_du_lieu.json"], "stations/noaa_usgs_mississippi", "mississippi", "water_level",
          "fetch_doi_chung.py")
    _glob(items, data_path("doi_chung", "mississippi", "raw", "crms").replace("\\", "/"), ["crms_*.csv"], "stations/crms", "mississippi",
          "porewater_salinity", "fetch_doi_chung.py")
    uniq = {}
    for i in items:  # cung (dich, nguon) do mau chong nhau thi gop; cung dich khac nguon la loi
        assert uniq.setdefault(i.dest, i).src == i.src, f"khoa dich trung: {i.dest}"
    return list(uniq.values())


# ---------------------------------------------------------------- dung
def _sidecar_write(src_sc, dst_sc, extra=None):
    d = normalize_json(json.load(open(src_sc, encoding="utf-8")))
    if extra:
        d["zenodo_package"] = extra
    with open(dst_sc, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def materialize(it, out, force=False):
    """Copy 1 muc + sidecar; tra ve {'dropped': [cot da bo]}."""
    dst = os.path.join(out, it.dest)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sc_src, sc_dst = it.src + ".provenance.json", dst + ".provenance.json"
    if os.path.exists(dst) and not force:  # chay lai: bo qua, van bao cot da bo (doc tu sidecar)
        try:
            return {"dropped": json.load(open(sc_dst, encoding="utf-8"))["zenodo_package"]["dropped_columns"]}
        except (OSError, KeyError, ValueError):
            return {"dropped": []}
    extra = None
    if it.mode == "csvdrop":
        dropped = drop_columns_csv(it.src, dst)
        if dropped:
            extra = {"dropped_columns": dropped, "reason": "MRC water-level derived, not redistributed",
                     "source_sha256_before_filter": file_sha256(it.src), "note": "file sha differs from csv_sha256/sha256 recorded below"}
    elif it.mode == "text":
        raw = open(it.src, "rb").read()
        try:
            open(dst, "wb").write(normalize_text(raw.decode("utf-8")).encode("utf-8"))
        except UnicodeDecodeError:
            shutil.copyfile(it.src, dst)
    else:
        shutil.copyfile(it.src, dst)
    if os.path.exists(sc_src):
        _sidecar_write(sc_src, sc_dst, extra)
    return {"dropped": (extra or {}).get("dropped_columns", [])}


def write_code_zip(out, ref):
    d = os.path.join(out, "code")
    os.makedirs(d, exist_ok=True)
    full = subprocess.check_output(["git", "-C", ROOT, "rev-parse", ref], text=True).strip()
    desc = subprocess.check_output(["git", "-C", ROOT, "describe", "--tags", "--always", ref], text=True).strip()
    tags = subprocess.check_output(["git", "-C", ROOT, "tag", "--points-at", ref], text=True).split()
    when = subprocess.check_output(["git", "-C", ROOT, "log", "-1", "--format=%cI", ref], text=True).strip()
    z = os.path.join(d, f"mekong-nckh-code-{full[:8]}.zip")
    subprocess.check_call(["git", "-C", ROOT, "archive", "--format=zip", f"--prefix=mekong-nckh-code/", "-o", z, ref, "--"] + CODE_PATHS)
    with open(os.path.join(d, "COMMIT.txt"), "w", encoding="utf-8") as f:
        f.write(f"commit: {full}\ndescribe: {desc}\ntags_on_commit: {', '.join(tags) or '(none - create a release tag before upload)'}\n"
                f"commit_date: {when}\npaths: {' '.join(CODE_PATHS)}\n")
    return z


def _download(url):
    from urllib.request import Request, urlopen
    return urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode("utf-8")


def write_extras(out, license_file, license_code=None):
    w = lambda rel, txt: (os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True),
                          open(os.path.join(out, rel), "w", encoding="utf-8", newline="\n").write(txt))
    if license_file:
        lic = open(license_file, encoding="utf-8").read()
    else:
        lic = _download(CC_BY_URL)
    if "Attribution 4.0 International" not in lic or len(lic) < 10000:
        raise SystemExit("LICENSE_DATA: van ban CC-BY-4.0 khong day du; dung --license-data-file")
    w("LICENSE_DATA", "Data in this record (grids/, eval/, labels/, features/, results/, stations/): CC-BY-4.0 unless a file or README says otherwise.\n"
      "Third-party components keep their own terms (see README, section Licences and sources).\n\n" + lic)
    w("LICENSE_CODE", {None: "[PLACEHOLDER: MIT or Apache-2.0 - to be chosen by the author before upload]\n", "MIT": MIT_TEXT}.get(license_code)
      or _download(CODE_LICENSE_URL[license_code]))
    w("stations/crms/RESTORE_disclaimer.txt", RESTORE_TEXT + "\n")
    w("stations/crms/CRMS_SOURCE.json", json.dumps({
        "source": "CPRA CRMS Discrete Hydrographic (CIMS), column Soil Porewater Salinity (ppt), filtered extract",
        "url": CRMS_ZIP_URL, "original_zip_sha256": CRMS_ZIP_SHA256, "original_zip_last_modified": "Sun, 11 Oct 2026 08:05:30 GMT",
        "filter": "rows with porewater salinity, 2013-11-01..2026-04-30, inside the Louisiana study boundary (scripts/fetch_doi_chung.py crms_table)",
        "licence": "CPRA data are public domain (CIMS Disclaimer https://cims.coastal.louisiana.gov/About.aspx); credit requested; RESTORE disclaimer kept",
        "citation": "Coastal Protection and Restoration Authority (CPRA) of Louisiana. 2026. Coastwide Reference Monitoring System. "
                    "Coastal Information Management System (CIMS) database. https://cims.coastal.la.gov. Accessed 15 October 2026."},
        indent=2) + "\n")
    for s in FETCH_SCRIPTS:
        p = os.path.join(ROOT, "scripts", s)
        if os.path.exists(p):
            os.makedirs(os.path.join(out, "fetch_scripts"), exist_ok=True)
            shutil.copyfile(p, os.path.join(out, "fetch_scripts", s))
    w("fetch_scripts/FETCH.md", FETCH_MD)
    w("fetch_scripts/SOURCES.csv", sources_csv())


def sources_csv():
    rows = [("name", "url_or_id", "used_for", "local_file_sha256", "note")]
    osm_dir = os.path.join(os.path.dirname(doi_chung_out()), "doi_chung_osm")
    for nm, url, reg in (("OSM Geofabrik Louisiana", "https://download.geofabrik.de/north-america/us/louisiana-latest-free.shp.zip", "louisiana"),
                         ("OSM Geofabrik Egypt", "https://download.geofabrik.de/africa/egypt-latest-free.shp.zip", "egypt")):
        p = os.path.join(osm_dir, f"{reg}-latest-free.shp.zip")
        rows.append((nm, url, "river-distance features (ODbL)", file_sha256(p) if os.path.exists(p) else "not-available-at-build",
                     "snapshot of 2026-10-16; 'latest' changes, verify sha before reuse"))
    rows += [("OSM Geofabrik Viet Nam", "https://download.geofabrik.de/asia/vietnam.html", "river-distance features (ODbL)", "not-recorded",
              "dated extract vietnam-YYMMDD-free.shp.zip used by scripts/extract_osm_waterways.py; extract date not recorded"),
             ("HydroRIVERS v1.0", "https://www.hydrosheds.org/products/hydrorivers", "cross-check only (not a feature)", "not-recorded", "Lehner & Grill 2013"),
             ("Wang et al. 2025 salinity labels", "https://zenodo.org/records/15653696", "labels 2014-2023", "see zenodo API checksums", "DOI 10.5281/zenodo.15653696"),
             ("CRMS Discrete Hydrographic", CRMS_ZIP_URL, "Mississippi porewater salinity (reference)", CRMS_ZIP_SHA256, "zip of 2026-10-17"),
             ("MRC time series API", "https://timeseries.api.mrcmekong.org/", "NOT redistributed", "n/a", "run scripts/fetch_mrc_timeseries.py subject to MRC terms")]
    return "".join(",".join('"' + str(c).replace('"', '""') + '"' for c in r) + "\n" for r in rows)


FETCH_MD = """# Re-fetching the raw inputs (not redistributed here)

All raw inputs are public or free-of-charge sources, except the MRC water levels (licence forbids redistribution).
Run from the unpacked `code/` tree after creating `.env` from `.env.example` (DATA_ROOT, EE_PROJECT, GEE_DRIVE_FOLDER).

| Input | Script | Notes |
| --- | --- | --- |
| ERA5-Land, CHIRPS v3, MCD18A1, Landsat 8 C2 L2, Copernicus DEM, WorldCover, JRC GSW (Mekong Delta) | `gee_fetch.py` | Google Earth Engine; needs an EE project. |
| Same layers for Mississippi (Louisiana) and Nile Delta, NOAA CO-OPS, USGS NWIS, CRMS | `fetch_doi_chung.py` | boundaries: TIGER/2018 (LA), FAO GAUL 2015 (Egypt) via GEE. |
| MRC water levels (Tan Chau + 5 stations) | `fetch_mrc_timeseries.py`, `fetch_mrc_water_level.py` | Obtain under MRC terms; NOT in this record. Needed only to rebuild `tch_*`/`tram_*` columns. |
| OSM waterways, HydroRIVERS | see `SOURCES.csv` | Geofabrik extracts change over time; the sha256 of the files used is listed where recorded. |
| Wang et al. 2025 labels | https://zenodo.org/records/15653696 | CC-BY; used for 2014-2023, 2024-2026 re-created with the same GEE recipe. |
"""


def group_table(rows):
    g = {}
    for r in rows:
        parts = r["path"].split("/")
        k = "/".join(parts[:2]) if len(parts) > 2 else parts[0] + "/(top level)"
        e = g.setdefault(k, {"n": 0, "b": 0, "o": set()})
        e["n"] += 1
        e["b"] += r["bytes"]
        if r["origin"]:
            e["o"].add(r["origin"])
    lines = ["| Folder | Files | MB | Generated by (script) |", "| --- | ---: | ---: | --- |"]
    for k in sorted(g):
        lines.append(f"| `{k}/` | {g[k]['n']} | {g[k]['b'] / 1e6:.1f} | {', '.join(sorted(g[k]['o'])) or '-'} |")
    return "\n".join(lines)


def write_readme(out, rows, dropped_summary, code_commit):
    txt = README_TMPL.replace("{{GROUPS}}", group_table(rows)).replace("{{DROPPED}}", dropped_summary).replace("{{COMMIT}}", code_commit)
    open(os.path.join(out, "README.md"), "w", encoding="utf-8", newline="\n").write(txt)


# ---------------------------------------------------------------- manifest + quet nhay cam
def sidecar_script(p):
    try:
        return json.load(open(p + ".provenance.json", encoding="utf-8")).get("script", "")
    except Exception:
        return ""


def write_manifest(out, meta):
    rows = []
    for d, _, fs in os.walk(out):
        for f in sorted(fs):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, out).replace("\\", "/")
            if rel == "MANIFEST.csv":
                continue
            base = rel[:-len(".provenance.json")] if rel.endswith(".provenance.json") else rel
            m = meta.get(base, {})
            rows.append({"path": rel, "bytes": os.path.getsize(p), "sha256": file_sha256(p),
                         "origin": sidecar_script(os.path.join(out, base)) or m.get("origin", ""),
                         "region": m.get("region", ""), "variable": m.get("variable", "")})
    rows.sort(key=lambda r: r["path"])
    with open(os.path.join(out, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "origin", "region", "variable"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def _patterns():
    ee = os.getenv("EE_PROJECT", "").strip()
    pats = {"email": re.compile(rb"[\w.+-]+@[\w-]+\.[A-Za-z][\w.]*"),
            "abs_path": re.compile(rb"\b[A-Za-z]:[\\/][A-Za-z_$~]"),
            "gee_project_id": re.compile(rb"project-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}")}
    terms = [t for t in (ee, getpass.getuser()) if len(t) >= 4]
    if terms:
        pats["private_term"] = re.compile("|".join(re.escape(t) for t in terms).encode(), re.I)
    return pats


def _scan(data, pats, label, hits):
    need = {"email": b"@", "abs_path": b":", "gee_project_id": b"project-"}
    binary = label.lower().endswith((".png", ".jpg"))  # luong nen nen: khop duong dan o dia la ngau nhien
    for k, rx in pats.items():
        if (need.get(k) and need[k] not in data) or (binary and k == "abs_path"):
            continue
        for m in rx.finditer(data):
            hits.append((k, label, m.group(0)[:60].decode("utf-8", "replace")))


def audit(out):
    """Quet nhay cam: email, ID project GEE, thuoc tinh ca nhan, duong dan o dia; ten file .env; cot tch_/tram_ trong features.
    Tra ve (loi, canh_bao): duong dan o dia trong MA (zip) chi la canh bao (mac dinh duong dan trong code)."""
    pats, errs, warns = _patterns(), [], []
    for d, _, fs in os.walk(out):
        for f in fs:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, out).replace("\\", "/")
            if f == ".env" or (f.startswith(".env.") and f != ".env.example"):
                errs.append(("env_file", rel, f))
            if rel.startswith("features/") and f.endswith(".csv"):
                head = open(p, encoding="utf-8").readline()
                bad = [c for c in head.strip().split(",") if c.startswith(MRC_PREFIXES)]
                if bad:
                    errs.append(("mrc_column", rel, ",".join(bad)))
            if f.lower().endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    for n in z.namelist():
                        if os.path.basename(n) == ".env":
                            errs.append(("env_file", f"{rel}!{n}", n))
                        h = []
                        _scan(z.read(n), pats, f"{rel}!{n}", h)
                        for x in h:
                            (warns if x[0] == "abs_path" else errs).append(x)
                continue
            h = []
            with open(p, "rb") as fh:
                prev = b""
                for chunk in iter(lambda: fh.read(1 << 26), b""):
                    _scan(prev + chunk, pats, rel, h)
                    prev = chunk[-128:]
            seen = set()
            for x in h:  # phan chong lap do chunk gac nhau
                if x not in seen:
                    seen.add(x)
                    errs.append(x)
    return errs, warns


def run(out, ref="HEAD", license_file=None, dry_run=False, force=False, license_code=None):
    items = build_plan()
    est = sum(os.path.getsize(i.src) for i in items)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out))).free if os.path.isdir(os.path.dirname(os.path.abspath(out))) else 0
    print(f"{len(items)} muc ~ {est / 1e9:.2f} GB nguon; trong ~ {free / 1e9:.0f} GB", flush=True)
    if dry_run:
        return items
    if free and free < est * 1.3:
        raise SystemExit("Khong du cho trong")
    meta, dropped = {}, {}
    for k, it in enumerate(items, 1):
        r = materialize(it, out, force)
        meta[it.dest] = {"origin": it.origin, "region": it.region, "variable": it.variable}
        if r and r["dropped"]:
            dropped.setdefault(tuple(r["dropped"]), []).append(it.dest)
        if k % 100 == 0:
            print(f"  {k}/{len(items)}", flush=True)
    z = write_code_zip(out, ref)
    write_extras(out, license_file, license_code)
    meta[os.path.relpath(z, out).replace("\\", "/")] = {"origin": "git archive", "region": "", "variable": ""}
    dsum = "\n".join(f"- `{', '.join(c)}` removed from {len(v)} files (e.g. `{v[0]}`)" for c, v in dropped.items()) or "- (none)"
    commit = open(os.path.join(out, "code", "COMMIT.txt"), encoding="utf-8").read().splitlines()[0].split(": ")[1][:12]
    tmp = [{"path": i.dest, "bytes": os.path.getsize(os.path.join(out, i.dest)),
            "origin": sidecar_script(os.path.join(out, i.dest)) or i.origin} for i in items]
    write_readme(out, tmp, dsum, commit)
    rows = write_manifest(out, meta)
    print(f"{len(rows)} file, {sum(r['bytes'] for r in rows) / 1e9:.2f} GB", flush=True)
    return rows


README_TMPL = """# Mekong Delta soil-salinity grid-framework comparison: derived data, features and code (DRAFT)

Derived data for the study comparing spatial grid frameworks (H3, S2, lat/lon, UTM squares; 13 configurations) for predicting
dry-season soil salinity in the Vietnamese Mekong Delta, with two control regions (Mississippi Delta, Nile Delta).
Unit of analysis: (grid cell, dry season), dry season of year Y = 1 Nov (Y-1) to 29 Apr (Y), Landsat 8 OLI years 2014-2026.

**Draft package, not uploaded. Version/DOI to be assigned at upload.**

## What is and is not in this record

Included: grids, evaluation design (held-out blocks, evaluation points, cross-validation folds), per-cell season labels,
per-cell season feature tables, final result tables and paper figures, public station extracts (NOAA, USGS, CRMS), the code.
Not included: raw rasters, per-run model outputs and predictions (so the auxiliary feature subsets `keep*`, the scope mask raster and
the per-run model results are not shipped: they are intermediate, large and regenerated by the scripts), and **Mekong River Commission (MRC) water-level data**
(not redistributed; see `fetch_scripts/fetch_mrc_timeseries.py`; obtain under MRC terms). The MRC-derived feature columns were
removed from the feature tables:

{{DROPPED}}

Consequence: the released feature tables lack `tch_*`/`tram_*`; the Delta experiments that used `tch_wl_p20c` cannot be re-run from
this record alone (re-fetch MRC data and run `scripts/build_hydro_season.py`, then `scripts/build_unified_table.py`). Result tables are
released as computed. `zos_*` columns (CMEMS sea level) and all other columns are kept unchanged.

## Layout

```
README.md  LICENSE_DATA  LICENSE_CODE  MANIFEST.csv   (path, bytes, sha256, origin script, region, variable)
code/            git archive of the repository at {{COMMIT}} (see code/COMMIT.txt)
grids/<region>/  13 grid GeoJSON + study boundary + `dot4_doi_chieu_dien_tich.csv` (cell-area table per grid; `file` column = source path, normalised);
                 Delta also `dot4_scope_v3_dien_tich.csv` (land-cover area lost per class when the scope mask is applied, not a grid-area table)
eval/<region>/   eval_points, holdout_blocks, cv_folds (seeds 42/43/44), cell_blocks (+ khoi100 for the Delta)
labels/<region>/ per-cell season labels: salinity (Delta only, labels/dbscl/salinity) and 5 labels in dot7/ (ndwi, dsr_mcd18, rain_chirps, rh_era5, t2m_era5)
features/<region>/ unified (base table) and unified_dot7/<variable>/ (tables per label), 13 grids each
results/         ledgers (dbscl_a, dbscl_b, mississippi, nile), three-region table, block tests, final-set tables, PC-1 CI, threshold sensitivity, figures
stations/        CRMS extract (+ RESTORE disclaimer, original zip sha256), NOAA/USGS CSV; no MRC
fetch_scripts/   scripts and notes to re-fetch raw inputs (FETCH.md, SOURCES.csv)
```

Regions: `dbscl` = Mekong Delta; `dbscl_a`/`dbscl_b` = feature set (a) full, (b) without point-observation-derived columns (`configs/feature_sets`);
`mississippi`, `nile` = control regions. Only the Delta has soil-salinity labels (Wang et al. 2025); the control regions have the five
meteorological/optical labels only (no salinity labels exist for them).

## Files and the script that produced them

Every sidecar `*.provenance.json` records script, inputs and sha256. Local absolute paths were normalised to the tokens
`<DATA_ROOT>`, `<DOI_CHUNG_OUT>`, `<OUT_ROOT>`, `<REPO>` (anything else: `<LOCAL>/<name>`); the same applies to `.md`/`.csv` result files.
The sha256 fields inside a sidecar refer to the file as generated; for feature tables with removed columns the sidecar block
`zenodo_package` gives the sha256 before filtering and the removed columns, and `MANIFEST.csv` gives the sha256 of the released file.

{{GROUPS}}

## Reproduction (order of scripts, run from the unpacked `code/` tree; paths via `.env`)

1. Environment: Python (see `requirements.txt`), copy `.env.example` to `.env`, set `DATA_ROOT`, `EE_PROJECT`, `GEE_DRIVE_FOLDER`, `DOI_CHUNG_OUT`.
2. Raw inputs: `fetch_scripts/FETCH.md` (`gee_fetch.py`, `fetch_doi_chung.py`; MRC separately).
3. Grids: `scripts/generate_all_grids.py` (boundary: `scripts/build_boundary_v2.py`).
4. Labels: `scripts/build_labels_season.py` (salinity), `scripts/build_met_labels.py` (5 labels), `scripts/clip_labels_footprint.py`.
5. Features: `scripts/build_era5_season.py`, `build_static_features.py`, `build_river_features.py` (`build_river_features_doi_chung.py`), `build_scope_mask.py`, `build_hydro_season.py` (MRC), `build_unified_table.py`, `build_unified_dot7.py`.
6. Evaluation design: `scripts/build_eval_design.py` / `rebuild_eval_design_v2.py`, `build_cv_folds.py`.
7. Experiments: `scripts/run_experiments.py --prefix cv1 --target <variable> ...` (HistGB and baseline; RF/MLP as sensitivity).
8. Analysis: `scripts/analyze_final.py`, `build_comparison_ledger.py`, `build_bang_3_vung.py`, `pc1_wilson_ci.py`, `threshold_sensitivity.py`, `plot_paper_figures.py`.
9. Tests: `pytest -q` (synthetic data; needs `.env` with `DATA_ROOT` set, otherwise collection fails).

Rebuilding the three-region table and the control-region ledgers from this record (folder layout differs from the working repository):

```
python scripts/build_bang_3_vung.py --package-root <record root> --out-dir <new dir>
python scripts/build_comparison_ledger.py --region nile --res-dir <record root>/results/nile --area-table <record root>/grids/nile/dot4_doi_chieu_dien_tich.csv
```

(same for `--region mississippi`; the ledger is written into `--res-dir`, so work on a copy.) Folder map: `results/dbscl_a` = `KE_HOACH/ket-qua`,
`results/dbscl_b` = `KE_HOACH/ket-qua/dbscl_fs_khong_diem`, `results/<region>` = `KE_HOACH/ket-qua/doi_chung/<region>`.

This draft is a snapshot of the working repository; a refactored publication tree is planned. The released files are frozen outputs;
`MANIFEST.csv` sha256 values are the reference for byte-identity.

## Licences and sources

- Data created here (grids, eval design, labels, features, results, figures): CC-BY-4.0 (`LICENSE_DATA`).
- Code: `LICENSE_CODE` [PLACEHOLDER: MIT or Apache-2.0 - to be chosen].
- Salinity labels 2014-2023 re-aggregated from Wang et al. (2025), Commun. Earth Environ. 6, 503; data: Zenodo https://doi.org/10.5281/zenodo.15653696 (CC-BY); formula of Nguyen et al. (2020). Labels 2024-2026 re-created with the same recipe from Landsat 8 (NDWIchen, Salinity = EC1:5 dS/m).
- Landsat 8 Collection 2 Level 2 (USGS), via Google Earth Engine (`LANDSAT/LC08/C02/T1_L2`); Landsat 9 is not used.
- ERA5-Land (Copernicus Climate Change Service; Munoz-Sabater et al. 2021, Earth Syst. Sci. Data 13, 4349).
- CHIRPS v3 (Climate Hazards Center, UC Santa Barbara; `CHC/CHIRPS/V3/PENTAD`).
- MODIS MCD18A1 v6.2 downward shortwave radiation (NASA LP DAAC; `MODIS/062/MCD18A1`).
- Copernicus DEM GLO-30 (ESA; `COPERNICUS/DEM/GLO30_2024_1`).
- ESA WorldCover v200 (Zanaga et al. 2022; `ESA/WorldCover/v200`).
- JRC Global Surface Water v1.4 (Pekel et al. 2016, Nature 540, 418-422).
- CMEMS global ocean physics reanalysis, sea surface height `zos` (`cmems_mod_glo_phy_my_0.083deg_P1D-m`).
- OpenStreetMap waterways (c) OpenStreetMap contributors, ODbL 1.0, via Geofabrik; released river-distance features are derived values, the OSM extracts are not redistributed.
  Contains information from OpenStreetMap, (c) OpenStreetMap contributors. Open question for the authors: whether the ODbL share-alike clause applies to
  these derived features (and thus to the record) has not been decided; do not treat the CC-BY-4.0 label of the features as final until resolved.
- HydroRIVERS v1.0 (Lehner & Grill 2013, Hydrol. Process. 27, 2171-2186): cross-check only, not a feature.
- Coastline: Sayre et al. (2019). Boundaries: geoBoundaries gbOpen (Mekong Delta), TIGER/2018 States (Louisiana), FAO GAUL 2015 (Nile Delta).
- NOAA CO-OPS tide gauges and USGS NWIS gauges (US Government public data), `stations/noaa_usgs_mississippi/`.
- CRMS: Coastal Protection and Restoration Authority (CPRA) of Louisiana. 2026. Coastwide Reference Monitoring System. Coastal Information
  Management System (CIMS) database. https://cims.coastal.la.gov. Accessed 15 October 2026. Public domain; preliminary data, no warranty;
  RESTORE disclaimer in `stations/crms/RESTORE_disclaimer.txt`; filtered extract (see `stations/crms/CRMS_SOURCE.json` for the original zip sha256).
- MRC (Mekong River Commission) water levels: **not redistributed**; see `fetch_scripts/fetch_mrc_timeseries.py`.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.getenv("ZENODO_PKG_DIR") or os.path.join(os.path.dirname(doi_chung_out()), "zenodo_pkg"))
    ap.add_argument("--ref", default="HEAD", help="commit/tag cua ma nguon dong goi")
    ap.add_argument("--license-data-file", help="van ban CC-BY-4.0 (mac dinh tai tu creativecommons.org)")
    ap.add_argument("--license-code", choices=["MIT", "Apache-2.0"], help="giay phep ma (mac dinh: placeholder cho den khi tac gia chot)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="ghi de file da co")
    ap.add_argument("--audit-only", action="store_true")
    a = ap.parse_args()
    if not a.audit_only:
        run(a.out, a.ref, a.license_data_file, a.dry_run, a.force, a.license_code)
    if a.dry_run:
        return
    errs, warns = audit(a.out)
    print(f"QUET NHAY CAM: {len(errs)} loi, {len(warns)} canh bao (duong dan trong ma)", flush=True)
    for e in errs[:30]:
        print("  LOI", e)
    for w in warns[:10]:
        print("  CANH BAO", w)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()

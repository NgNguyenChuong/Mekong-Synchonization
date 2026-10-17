#!/usr/bin/env python
"""Bang tong hop 3 vung (DBSCL (a)/(b), Mississippi, Nile) tu 4 comparison_ledger: dem Eq/Inc/Desc(/Pend) theo bien x muc
+ bang cong hoc duoc. Chi doc ledger da mo.

Chay:  venv/Scripts/python.exe scripts/build_bang_3_vung.py [--results-dir KE_HOACH/ket-qua] [--out-dir <results-dir>]
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from md_util import md_table  # noqa: E402
from training.dot7_rules import file_sha256, write_provenance  # noqa: E402

RESULTS = os.path.join(ROOT, "KE_HOACH", "ket-qua")
LEDGERS = {"dbscl_a": ("ĐBSCL (a)", "comparison_ledger.csv"),
           "dbscl_b": ("ĐBSCL (b)", os.path.join("dbscl_fs_khong_diem", "comparison_ledger__fs-khong_diem.csv")),
           "ms": ("Mississippi", os.path.join("doi_chung", "mississippi", "comparison_ledger.csv")),
           "nl": ("Nile", os.path.join("doi_chung", "nile", "comparison_ledger.csv"))}
VARS = ("salinity", "ndwi", "dsr_mcd18", "rain_chirps", "t2m_era5", "rh_era5")
LEVELS = ("coarse", "medium", "fine")
PAIRS_PER_LEVEL = {"coarse": 3, "medium": 3, "fine": 6}  # 12 cap/bien/vung
RATIO_MAX = 1.2
OUT_NAME = "bang_3_vung"
NEED = ["region", "variable", "grid_a", "grid_b", "level", "area_ratio", "eligible", "ineligible_reason", "G",
        "decision"]
KNOWN = {"Equivalent", "Inconclusive", "Descriptive", "PendingFinal", "Trivial"}


def loi(msg):
    sys.exit(f"LOI: {msg}")


def load_ledgers(results_dir) -> dict:
    """key -> ledger da kiem: du cot, 12 cap/bien (3/3/6 theo muc), decision thuoc tap biet, eligible bool."""
    out = {}
    for k, (_, rel) in LEDGERS.items():
        path = os.path.join(results_dir, rel)
        if not os.path.isfile(path):
            loi(f"thieu ledger {path}")
        d = pd.read_csv(path)
        if (miss := [c for c in NEED if c not in d]):
            loi(f"{path}: thieu cot {miss}")
        if bad := set(d["decision"]) - KNOWN:
            loi(f"{path}: decision {sorted(bad)} chua duoc bang xu ly (Different?)")
        d["eligible"] = d["eligible"].astype(str).str.lower().eq("true")
        d["ineligible_reason"] = d["ineligible_reason"].fillna("")
        if d.duplicated(["variable", "grid_a", "grid_b"]).any():
            loi(f"{path}: khoa (variable, grid_a, grid_b) trung")
        n = d.groupby(["variable", "level"]).size()
        if not (n == n.index.get_level_values("level").map(PAIRS_PER_LEVEL)).all():
            loi(f"{path}: so cap theo bien x muc khong phai 3/3/6:\n{n}")
        if not (d.groupby("variable").size() == 12).all():
            loi(f"{path}: moi bien phai du 12 cap")
        if (bad := d[d.eligible & (d.decision == "Descriptive")]).size:
            loi(f"{path}: eligible nhung decision Descriptive ({len(bad)} dong)")
        d.attrs["path"] = path
        out[k] = d
    return out


def classify(d) -> pd.Series:
    """Eq/Inc/Pend chi cho cap eligible; cap khong du dieu kien luon la Desc (ke ca khi ledger ghi decision khac)."""
    cls = d["decision"].map({"Equivalent": "Eq", "Inconclusive": "Inc", "PendingFinal": "Pend", "Trivial": "Triv"})
    return cls.where(d["eligible"], "Desc")


def _reasons(s) -> str:
    c = s[s != ""].value_counts()
    return "; ".join(f"{k} {v}" if len(c) > 1 or len(s) != v else k for k, v in c.sort_index().items())


def cell(d) -> dict:
    """Mot o (tap cap con cua 1 bien x muc hoac tong)."""
    if d.empty:
        return dict(text="–", eq=None, inc=None, desc=None, pend=None, triv=None, reason="", flags="")
    cls = classify(d)
    n = {k: int((cls == k).sum()) for k in ("Eq", "Inc", "Desc", "Pend", "Triv")}
    assert sum(n.values()) == len(d)
    reason = _reasons(d.loc[cls == "Desc", "ineligible_reason"])
    flags = ("†" if (d["area_ratio"] > RATIO_MAX).any() else "") + ("‡" if n["Pend"] else "") + \
            ("§" if ((cls == "Desc") & (d["decision"] != "Descriptive")).any() else "")
    txt = f"{n['Eq']}/{n['Inc']}/{n['Desc']}" + (f"/{n['Pend']}" if n["Pend"] else "") + (f" ◊{n['Triv']}" if n["Triv"] else "")
    if reason:
        txt += f" ({reason})"
    return dict(text=txt + flags, eq=n["Eq"], inc=n["Inc"], desc=n["Desc"], pend=n["Pend"], triv=n["Triv"], reason=reason, flags=flags)


def table1(led) -> pd.DataFrame:
    rows = []
    for v in VARS:
        for lv in LEVELS:
            rows.append(dict(variable=v, level=lv, **{f"{k}__{f}": x for k, d in led.items()
                             for f, x in cell(d[(d.variable == v) & (d.level == lv)]).items()}))
    for lv in (*LEVELS, "all"):
        sel = (lambda d: d[d.level == lv]) if lv != "all" else (lambda d: d)
        rows.append(dict(variable="TOTAL", level=lv, **{f"{k}__{f}": x for k, d in led.items()
                         for f, x in cell(sel(d)).items()}))
    return pd.DataFrame(rows)


def g_row(led) -> dict:
    out = {}
    for k, d in led.items():
        g = d.groupby("variable")["G"].first()
        main = g.mode().iloc[0]
        odd = g[g != main]
        out[k] = str(main) + "".join(f" ({int(x)}: {', '.join(odd[odd == x].index)})" for x in sorted(odd.unique()))
    return out


def table2(led) -> pd.DataFrame:
    """Cong hoc duoc theo bien x muc x vung: QUA x/n | TRUOT | Khong kiem dinh (muc khong dang ky kiem dinh);
    salinity (chi DBSCL) khong ap cong."""
    rows = []
    for v in VARS:
        for lv in LEVELS:
            r = dict(variable=v, level=lv)
            for k, d in led.items():
                s = d[(d.variable == v) & (d.level == lv)]
                if s.empty:
                    r[k] = "–"
                elif v == "salinity":
                    r[k] = "Không áp cổng"
                elif (x := int(s.eligible.sum())):
                    r[k] = f"QUA {x}/{len(s)}"
                else:
                    r[k] = "TRƯỢT" if (s.ineligible_reason == "truot_cong").any() else "Không kiểm định"
            rows.append(r)
    return pd.DataFrame(rows)


NOTES = ("Ô = nEq/nInc/nDesc(/nPend): Equivalent / Inconclusive / Descriptive / PendingFinal theo ledger, số cặp cùng mức "
         "(coarse 3, medium 3, fine 6 mỗi biến; 12 cặp/biến/vùng). Descriptive kèm lý do: `khong_kiem_dinh` = mức không "
         "đăng ký kiểm định; `truot_cong` = mức kiểm định nhưng trượt cổng học được. Cặp không eligible luôn tính Descriptive.  \n"
         "† ô chứa cặp có tỉ lệ diện tích > 1,2 (MS: S2 L12 vs Square 2,4 km; NL: H3 r7 vs S2 L12; vẫn tính trong số đếm).  \n"
         "‡ có cặp PendingFinal (cho_giu_rieng ở CV, biến không có lượt final).  \n"
         "◊n = n cặp Trivial: Holm đạt, giữ riêng cùng chiều nhưng |Δ| < Δmin (có ý nghĩa thống kê nhưng nằm trong biên tương đương; đếm riêng).  \n"
         "§ ledger ghi decision khác Descriptive cho cặp không eligible (không tính vào Eq/Inc; xem cột *_raw trong CSV).  \n"
         "– biến không có ở vùng đó (salinity chỉ ĐBSCL).")


def to_md(led, t1, t2, g) -> str:
    names = {k: LEDGERS[k][0] for k in led}
    a = t1.copy()
    a.insert(0, "Biến × mức", a["variable"] + " · " + a["level"])
    show = a[["Biến × mức"] + [f"{k}__text" for k in led]].rename(columns={f"{k}__text": names[k] for k in led})
    gr = pd.DataFrame([{"Biến × mức": "G (số khối CV)", **{names[k]: g[k] for k in led}}])
    b = t2.copy()
    b.insert(0, "Biến × mức", b["variable"] + " · " + b["level"])
    b = b[["Biến × mức", *led]].rename(columns=names)
    return "\n\n".join([
        "# Bảng 1. Tổng hợp 3 vùng: Eq/Inc/Desc(/Pend) theo biến × mức", md_table(pd.concat([show, gr])), NOTES,
        "# Bảng 2. Cổng học được theo biến × mức × vùng (QUA x/n = số cặp eligible/số cặp của ô; độ mặn không áp cổng)", md_table(b)]) + "\n"


def with_raw(t1, led) -> pd.DataFrame:
    """Them cot *_raw: so dong ledger co decision != Descriptive trong nhom khong eligible (kiem nhat quan)."""
    t = t1.copy()
    for k, d in led.items():
        raw = []
        for r in t.itertuples():
            s = d if r.variable == "TOTAL" and r.level == "all" else d[d.level == r.level] if r.variable == "TOTAL" \
                else d[(d.variable == r.variable) & (d.level == r.level)]
            raw.append(int((~s.eligible & (s.decision != "Descriptive")).sum()))
        t[f"{k}__raw"] = raw
    return t


def main(a):
    led = load_ledgers(a.results_dir)
    t1, t2, g = with_raw(table1(led), led), table2(led), g_row(led)
    for k, d in led.items():  # tong khop so ledger: 12 cap/bien/vung
        tot = t1[(t1.variable == "TOTAL") & (t1.level == "all")].iloc[0]
        assert tot[f"{k}__eq"] + tot[f"{k}__inc"] + tot[f"{k}__desc"] + tot[f"{k}__pend"] == len(d) == 12 * d.variable.nunique()
    os.makedirs(a.out_dir, exist_ok=True)
    csv, md = (os.path.join(a.out_dir, f"{OUT_NAME}.{e}") for e in ("csv", "md"))
    gline = pd.DataFrame([{"variable": "G", "level": "", **{f"{k}__text": v for k, v in g.items()}}])
    pd.concat([t1, gline]).to_csv(csv, index=False, encoding="utf-8-sig")
    with open(md, "w", encoding="utf-8") as f:
        f.write(to_md(led, t1, t2, g))
    src = {os.path.relpath(d.attrs["path"], ROOT) if d.attrs["path"].startswith(ROOT) else d.attrs["path"]:
           file_sha256(d.attrs["path"]) for d in led.values()}
    for p in (csv, md):
        write_provenance(p, sha_dau_vao=src)
    print(open(md, encoding="utf-8").read(), flush=True)
    print(f"ghi {csv}\nghi {md}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--out-dir", default=None, help="mac dinh = --results-dir")
    args = ap.parse_args()
    args.out_dir = args.out_dir or args.results_dir
    main(args)

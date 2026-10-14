"""Do nhay nguong TOST (2/5/10 % delta_ref) cho cac cap kiem dinh, ho F1_chinh, tu CI 90 % da luu.
Chay: venv/Scripts/python.exe scripts/threshold_sensitivity.py [--dir KE_HOACH/ket-qua]
"""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

MAIN_DIR = Path(__file__).resolve().parents[1] / "KE_HOACH" / "ket-qua"
VARS = {"salinity": "dot5_cv1_kiem_dinh_khoi.csv", **{t: f"dot7_{t}_cv1_kiem_dinh_khoi.csv"
        for t in ("ndwi", "rain_chirps", "dsr_mcd18", "t2m_era5", "rh_era5")}}
MS = (0.02, 0.05, 0.10)
LEVEL = {5: "coarse", 6: "medium", 7: "fine"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(d):
    rows, shas = [], {}
    for v, fn in VARS.items():
        p = d / fn
        shas[fn] = sha(p)
        x = pd.read_csv(p)
        x = x[x.family == "F1_chinh"]
        if "kiem_dinh" in x:  # dot5 khong co cot nay: moi cap deu kiem dinh
            x = x[x.kiem_dinh.astype(str) == "True"]
        o = pd.DataFrame({"bien": v, "muc": x.delta_ref_level.map(LEVEL), "grid_a": x.grid_a, "grid_b": x.grid_b,
                          "delta_rel": x.delta_hat / x.delta_ref,
                          "ci_rel_low": x.ci_tost_low / x.delta_ref, "ci_rel_high": x.ci_tost_high / x.delta_ref,
                          "tost_luu_5": x.tost_tuong_duong.astype(str) == "True"})
        rows.append(o)
    return pd.concat(rows, ignore_index=True), shas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=MAIN_DIR)
    a = ap.parse_args()
    r, shas = load(a.dir)
    r["ci_half_rel"] = (r.ci_rel_high - r.ci_rel_low) / 2
    r["m_min"] = r[["ci_rel_low", "ci_rel_high"]].abs().max(axis=1)
    for m in MS:  # tuong duong khi CI 90 % nam tron trong (-m, m)
        r[f"td_{round(m*100)}"] = (r.ci_rel_low > -m) & (r.ci_rel_high < m)
    mismatch = r[r.td_5 != r.tost_luu_5]
    r.drop(columns="tost_luu_5").to_csv(a.dir / "do_nhay_nguong.csv", index=False)
    (a.dir / "do_nhay_nguong.csv.provenance.json").write_text(json.dumps(
        {"script": "scripts/threshold_sensitivity.py", "inputs_sha256": shas}, indent=1))

    mf = r[r.muc.isin(["medium", "fine"])]
    t = r.groupby(["bien", "muc"]).agg(n=("td_5", "size"), td_2=("td_2", "sum"), td_5=("td_5", "sum"),
                                       td_10=("td_10", "sum")).reset_index()
    tot = [int(mf[f"td_{k}"].sum()) for k in (2, 5, 10)]
    md = ["# Do nhay nguong TOST", "", "Tuong duong neu CI 90 % (ci_tost) nam trong (-m, +m) x delta_ref; p/Holm khong doi.", "",
          "| bien | muc | n | 2 % | 5 % | 10 % |", "|---|---|---|---|---|---|"]
    md += [f"| {x.bien} | {x.muc} | {x.n} | {x.td_2} | {x.td_5} | {x.td_10} |" for x in t.itertuples()]
    md += ["", f"medium+fine du dieu kien: {tot[0]}/{len(mf)} o 2 %, {tot[1]}/{len(mf)} o 5 %, {tot[2]}/{len(mf)} o 10 %", ""]
    md.append(f"Doi chieu cot tost_tuong_duong da luu (5 %): {len(mismatch)} cap lech.")
    for x in mismatch.itertuples():
        md.append(f"- {x.bien} {x.grid_a} vs {x.grid_b} ({x.muc}): tinh lai td_5={x.td_5}, luu={x.tost_luu_5}")
    md += ["", "Cap tuong duong yeu nhat (m_min lon nhat, medium+fine):", ""]
    md += [f"- {x.bien} {x.muc} {x.grid_a} vs {x.grid_b}: m_min = {x.m_min:.3f}" for x in mf.nlargest(5, "m_min").itertuples()]
    md += ["", "Input sha256:"] + [f"- {k}: {v}" for k, v in shas.items()]
    (a.dir / "do_nhay_nguong.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

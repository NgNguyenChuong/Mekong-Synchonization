"""build_bang_3_vung + fig7 tren ledger gia: dem theo eligible (khong theo decision), tong 12 cap/bien, co †‡§, G le,
cong hoc duoc, fig7 chi ve cap eligible o muc vua/nho voi CI quy ve % MAE muc."""
import importlib.util
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import build_bang_3_vung as b3  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAIRS = [("coarse", 0), ("coarse", 1), ("coarse", 2), ("medium", 0), ("medium", 1), ("medium", 2),
         *[("fine", i) for i in range(6)]]


def _ledger(variables, over):
    """over[(variable, level, i)] = dict(cot=gia tri) ghi de len dong mac dinh (eligible, Equivalent)."""
    rows = []
    for v in variables:
        for lv, i in PAIRS:
            r = dict(region="X", variable=v, grid_a=f"{lv}_a{i}", grid_b=f"{lv}_b{i}", level=lv, area_ratio=1.0,
                     eligible=True, ineligible_reason="", G=10, decision="Equivalent", delta_ref=10.0,
                     delta_min_thr=0.5, delta_rel=0.01, ci90_low=-0.1, ci90_high=0.2, ci95_low=-0.2, ci95_high=0.3)
            r.update(over.get((v, lv, i), {}))
            rows.append(r)
    return pd.DataFrame(rows)


def _desc(reason):
    return dict(eligible=False, ineligible_reason=reason, decision="Descriptive")


@pytest.fixture()
def results(tmp_path):
    spec = {"dbscl_a": ("comparison_ledger.csv", ["salinity", "ndwi"], {}),
            "dbscl_b": (b3.LEDGERS["dbscl_b"][1], ["salinity", "ndwi"],
                        {("salinity", lv, i): dict(eligible=False, ineligible_reason="truot_cong")  # decision van Equivalent
                         for lv, i in PAIRS}),
            "ms": (b3.LEDGERS["ms"][1], ["ndwi", "rain_chirps", "rh_era5"],
                   {(v, lv, i): _desc("khong_kiem_dinh") for v in ("rain_chirps", "rh_era5") for lv, i in PAIRS}
                   | {("ndwi", "fine", 5): dict(area_ratio=1.25, decision="Inconclusive")}),
            "nl": (b3.LEDGERS["nl"][1], ["ndwi"], {("ndwi", "fine", 0): dict(decision="PendingFinal")})}
    for rel, vars_, over in spec.values():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        _ledger(vars_, over).to_csv(p, index=False)
    return tmp_path


def test_o_dem_theo_eligible_va_co(results, monkeypatch):
    monkeypatch.setattr(b3, "VARS", ("salinity", "ndwi", "rain_chirps", "rh_era5"))
    led = b3.load_ledgers(str(results))
    c = b3.cell(led["dbscl_b"][(led["dbscl_b"].variable == "salinity") & (led["dbscl_b"].level == "fine")])
    assert (c["eq"], c["inc"], c["desc"]) == (0, 0, 6) and c["text"] == "0/0/6 (truot_cong)§"  # § = decision Equivalent nhung truot cong
    c = b3.cell(led["ms"][(led["ms"].variable == "ndwi") & (led["ms"].level == "fine")])
    assert c["text"] == "5/1/0†"
    c = b3.cell(led["nl"][(led["nl"].variable == "ndwi") & (led["nl"].level == "fine")])
    assert c["text"] == "5/0/0/1‡" and c["pend"] == 1
    assert b3.cell(led["nl"][led["nl"].variable == "salinity"])["text"] == "–"
    c = b3.cell(led["ms"][led["ms"].variable == "rain_chirps"])
    assert c["text"] == "0/0/12 (khong_kiem_dinh)"


def test_tong_12_cap_moi_bien_va_g_le(results, monkeypatch):
    monkeypatch.setattr(b3, "VARS", ("salinity", "ndwi", "rain_chirps", "rh_era5"))
    led = b3.load_ledgers(str(results))
    led["ms"].loc[led["ms"].variable == "rain_chirps", "G"] = 9
    t = b3.table1(led)
    tot = t[(t.variable == "TOTAL") & (t.level == "all")].iloc[0]
    assert tot["dbscl_a__eq"] == 24 and tot["ms__desc"] == 24 and tot["nl__pend"] == 1
    assert sum(tot[f"nl__{k}"] for k in ("eq", "inc", "desc", "pend")) == 12
    assert b3.g_row(led)["ms"] == "10 (9: rain_chirps)" and b3.g_row(led)["nl"] == "10"


def test_cong_hoc_duoc(results, monkeypatch):
    monkeypatch.setattr(b3, "VARS", ("salinity", "ndwi", "rain_chirps", "rh_era5"))
    t = b3.table2(b3.load_ledgers(str(results))).set_index(["variable", "level"])
    assert t.loc[("salinity", "coarse"), "dbscl_a"] == "QUA 3/3" and t.loc[("salinity", "coarse"), "dbscl_b"] == "TRƯỢT"
    assert t.loc[("rain_chirps", "fine"), "ms"] == "Không kiểm định" and t.loc[("rain_chirps", "fine"), "nl"] == "–"


def test_ledger_thieu_cap_hoac_decision_la_bi_chan(results):
    p = results / b3.LEDGERS["nl"][1]
    d = pd.read_csv(p)
    d.iloc[:-1].to_csv(p, index=False)
    with pytest.raises(SystemExit):
        b3.load_ledgers(str(results))
    d.assign(decision="Different").to_csv(p, index=False)
    with pytest.raises(SystemExit):
        b3.load_ledgers(str(results))


def test_fig7_chi_cap_eligible_muc_vua_nho(results):
    spec = importlib.util.spec_from_file_location("ppf", os.path.join(ROOT, "scripts", "plot_paper_figures.py"))
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    f = pp.fig7_pairs(b3.load_ledgers(str(results)))
    assert set(f["level"]) == {"medium", "fine"}
    assert set(f["reg"]) == {"dbscl_b", "ms", "nl"} and not ((f.reg == "dbscl_b") & (f.variable == "salinity")).any()
    assert not ((f.reg == "ms") & (f.variable == "rain_chirps")).any()  # khong eligible
    r = f.iloc[0]
    assert r["x"] == pytest.approx(1.0) and r["ci90_low"] == pytest.approx(-1.0) and r["ci95_high"] == pytest.approx(3.0)
    assert f[f.dag].shape[0] == 1 and f[f.cls == "pend"].shape[0] == 1 and f[f.cls == "und"].shape[0] == 1

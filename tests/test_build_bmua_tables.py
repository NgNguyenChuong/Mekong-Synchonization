"""scripts/build_bmua_tables.py: zos_mua_mean = TB khong gian theo mua (bo NaN, khong dien 0), sluice_flag theo
SLUICE_FIRST_SEASON (2021 -> 0, 2022 -> 1), cot cu giu nguyen, provenance giu khoa runner can, chay lai bo qua."""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def bm():
    spec = importlib.util.spec_from_file_location("build_bmua_tables",
                                                  os.path.join(ROOT, "scripts", "build_bmua_tables.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _table(label="salinity"):
    # 3 o x 4 mua; o c3 khong co zos (NaN); chi c1 sau cong
    rows = []
    for s in (2020, 2021, 2022, 2023):
        for c, z in (("c1", 1.0), ("c2", 2.0), ("c3", np.nan)):
            rows.append({"cell_id": c, "season": s, label: 0.5, "train_ok_scope": c != "c3",
                         "zos_mouth_p90": z + 0.1 * (s - 2020) if np.isfinite(z) else np.nan,
                         "sluice_frac": 0.4 if (c == "c1" and s >= 2022) else 0.0,
                         "tch_wl_p20c": np.nan if s == 2021 else 0.5, "dem_mean": 1.234567891})
    return pd.DataFrame(rows)


def test_gia_tri_biet_truoc(bm):
    t = _table()
    out = bm.add_bmua_columns(t)
    z = out.groupby("season")["zos_mua_mean"].unique()
    assert all(len(v) == 1 for v in z)
    assert z[2020][0] == pytest.approx(1.5) and z[2023][0] == pytest.approx(1.8)  # TB (1,3 + 2,3)/2, NaN bi bo qua
    flag = out.groupby("season")["sluice_flag"].unique()
    assert list(flag[2021]) == [0] and list(flag[2022]) == [1] and list(flag[2023]) == [1]  # moi o, ke ca sluice_frac 0
    assert set(out["sluice_flag"]) <= {0, 1}
    pd.testing.assert_frame_equal(out[list(t.columns)], t)  # cot cu giu nguyen, NaN khong dien 0
    assert out["tch_wl_p20c"].isna().sum() == 3 and list(out.columns[-2:]) == ["zos_mua_mean", "sluice_flag"]


@pytest.mark.parametrize("hong, khop", [
    (lambda t: t.assign(sluice_frac=np.where((t.season == 2021) & (t.cell_id == "c1"), 0.1, t.sluice_frac)),
     "SLUICE_FIRST_SEASON"),
    (lambda t: t.assign(zos_mouth_p90=np.where(t.season == 2021, np.nan, t.zos_mouth_p90)), "zos_mouth_p90"),
    (lambda t: t.assign(sluice_flag=0), "da co cot"),
])
def test_du_lieu_trai_dinh_nghia_la_loi(bm, hong, khop):
    with pytest.raises(ValueError, match=khop):
        bm.add_bmua_columns(hong(_table()))


def _write(path, df, prov):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    with open(f"{path}.provenance.json", "w", encoding="utf-8") as f:
        json.dump(prov, f)


def test_main_ghi_ban_sao_provenance_va_chay_lai(bm, tmp_path, capsys):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_experiments as rx

    uni, d7, out = tmp_path / "unified", tmp_path / "unified_dot7", tmp_path / "unified_bmua"
    _write(str(uni / "g1_unified.csv"), _table(), {"label_set": "chinh", "grid": "g1"})
    decl = {"target": "dsr_mcd18", "seasons": [2020], "reason": "r", "source": "s"}
    _write(str(d7 / "dsr_mcd18" / "g1_unified.csv"), _table("dsr_mcd18"),
           {"target": "dsr_mcd18", "label_set": "chinh", "holdout_season_nan": decl})
    a = argparse.Namespace(targets=["salinity", "dsr_mcd18"], grids=["g1"], unified_dir=str(uni), dot7_dir=str(d7),
                           out_root=str(out))
    bm.main(a)
    for t in ("salinity", "dsr_mcd18"):
        p = out / t / "g1_unified.csv"
        prov = json.loads((out / t / "g1_unified.csv.provenance.json").read_text(encoding="utf-8"))
        assert prov["bmua"]["output_sha256"] == bm.file_sha256(str(p)) and prov["bmua"]["formula"]
        src = uni / "g1_unified.csv" if t == "salinity" else d7 / t / "g1_unified.csv"
        assert prov["bmua"]["source"]["sha256"] == bm.file_sha256(str(src))
        assert rx.target_mismatch(str(p), t) is None  # runner chap nhan bang
        back = pd.read_csv(p, dtype={"cell_id": str})
        assert back["zos_mua_mean"].nunique() == 4 and set(back["sluice_flag"]) == {0, 1}
    assert "target" not in json.loads((out / "salinity" / "g1_unified.csv.provenance.json").read_text("utf-8"))
    groups = rx.table_empty_groups(str(out / "dsr_mcd18" / "g1_unified.csv"), "dsr_mcd18")
    assert groups and groups == rx.table_empty_groups(str(d7 / "dsr_mcd18" / "g1_unified.csv"), "dsr_mcd18")
    mtime = os.path.getmtime(out / "salinity" / "g1_unified.csv")
    capsys.readouterr()
    bm.main(a)  # chay lai: nguon khong doi -> bo qua
    assert "bo qua" in capsys.readouterr().out and os.path.getmtime(out / "salinity" / "g1_unified.csv") == mtime

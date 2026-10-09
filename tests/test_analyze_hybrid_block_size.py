"""scripts/analyze_hybrid_block_size.py - CHG-21 (chi mo ta, so voi dai nhieu cach chia 50 km) + CHG-22 (LOI).

Du lieu gia nho (tmp_path) co I biet truoc: (a) |err| = 1; (b) |err| = 1 + c[design, seed] -> I = c. KHONG chay tren
artifacts that (ket qua k100 hybrid chua duoc mo theo quy dinh chot truoc).
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
N_PTS, SEASONS = 20, (2014, 2015)

# I theo luoi: g_trong -> I_100 = 0,01 nam trong [0; 0,02]; g_ngoai -> I_100 = 0,05 ngoai [-0,01; 0,01]
C = {"g_trong": {("k50", 42): 0.0, ("k50", 43): 0.02, ("k50", 44): 0.01, ("k100", 42): 0.0, ("k100", 43): 0.02},
     "g_ngoai": {("k50", 42): -0.01, ("k50", 43): 0.01, ("k50", 44): 0.0, ("k100", 42): 0.04, ("k100", 43): 0.06}}


@pytest.fixture()
def mod(monkeypatch):
    spec = importlib.util.spec_from_file_location("analyze_hybrid_block_size",
                                                  os.path.join(ROOT, "scripts", "analyze_hybrid_block_size.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "GRIDS", list(C))
    return m


def _write(tmp_path, mutate=None):
    exp = tmp_path / "experiments"
    rng = np.random.default_rng(0)
    for design, prefix in (("k50", "cv1"), ("k100", "k100")):
        for arm, fs in (("a", None), ("b", "khong_diem")):
            for g in C:
                for s in ((42, 43, 44) if design == "k50" else (42, 43)):
                    mag = 1.0 if arm == "a" else 1.0 + C[g][(design, s)]
                    d = pd.DataFrame([(f"p{i}", se) for i in range(N_PTS) for se in SEASONS],
                                     columns=["point_id", "season"])
                    d["err"] = mag * rng.choice([-1.0, 1.0], len(d))
                    d["pred_source"] = "oof"
                    name = f"{prefix}__{g}__hist_gb__s{s}" + (f"__fs-{fs}" if fs else "")
                    if mutate is not None:
                        d = mutate(name, d)
                    if d is None:
                        continue
                    (exp / name / "cv").mkdir(parents=True)
                    d.to_csv(exp / name / "cv" / "oof_points.csv", index=False)
    out = tmp_path / "ket-qua"
    return argparse.Namespace(prefix50="cv1", prefix100="k100", feature_set="khong_diem",
                              expected_n=N_PTS * len(SEASONS), exp_root=str(exp), out_dir=str(out))


def test_gia_tri_biet_truoc_va_dai_nhieu(mod, tmp_path, capsys):
    a = _write(tmp_path)
    mod.main(a)
    t = pd.read_csv(os.path.join(a.out_dir, mod.OUT_NAME)).set_index("grid")
    assert not os.path.exists(os.path.join(a.out_dir, mod.OUT_NAME + ".tmp"))
    g = t.loc["g_trong"]
    assert g["I50_s42"] == pytest.approx(0.0) and g["I50_s43"] == pytest.approx(0.02)
    assert g["dai50_min"] == pytest.approx(0.0) and g["dai50_max"] == pytest.approx(0.02)
    assert g["I_50"] == pytest.approx(0.01) and g["I_100"] == pytest.approx(0.01)
    assert g["I100_P1_s42"] == pytest.approx(0.0) and g["I100_P2_s43"] == pytest.approx(0.02)
    assert bool(g["trong_dai_nhieu_50km"])
    h = t.loc["g_ngoai"]
    assert h["I_100"] == pytest.approx(0.05) and not bool(h["trong_dai_nhieu_50km"])
    assert (t["n_diem_mua_chung"] == N_PTS * len(SEASONS)).all()
    assert not {"vung", "cung_dau", "ti_le"} & set(t.columns)  # CHG-21: khong con nhan "vung"
    txt = capsys.readouterr().out
    assert "So luoi I_100 > 0: 2/2" in txt and "trong dai nhieu cach chia 50 km [min, max] I50_s: 1/2" in txt
    assert f"Trong dai (g_trong): {mod.CAU_TRONG_DAI}" in txt
    assert "Ngoai dai: g_ngoai" in txt
    assert "vung" not in txt.lower() and "phu thuoc" not in txt.lower()


def test_bien_dai_tinh_la_trong_dai(mod):
    m = pd.DataFrame([{"design": d, "arm": arm, "grid": g, "seed": s,
                       "MAE": 1.0 + (0 if arm == "a" else {("k50", 42): 0.0, ("k50", 43): 0.02, ("k50", 44): 0.01,
                                                          ("k100", 42): 0.02, ("k100", 43): 0.02}[(d, s)])}
                      for g in C for d, ss in mod.SEEDS.items() for s in ss for arm in ("a", "b")])
    t = mod.block_size_table(m, 40)
    assert t["I_100"].to_numpy() == pytest.approx([0.02, 0.02]) and t["trong_dai_nhieu_50km"].all()


@pytest.mark.parametrize("mutate, khop", [
    (lambda n, d: None if n == "k100__g_ngoai__hist_gb__s43__fs-khong_diem" else d, "thieu"),
    (lambda n, d: d.assign(err=np.where(d.index == 0, np.nan, d["err"])) if n.startswith("k100__g_trong") else d,
     "NaN"),
    (lambda n, d: d.assign(pred_source="final") if n == "cv1__g_trong__hist_gb__s44" else d, "oof"),
    (lambda n, d: pd.concat([d, d.iloc[[0]]]) if n == "cv1__g_trong__hist_gb__s43" else d, "trung"),
    (lambda n, d: d.iloc[1:] if n == "k100__g_trong__hist_gb__s42" else d, "khac tap"),   # khong cat giao lang
])
def test_loi_khong_ghi_ket_qua(mod, tmp_path, mutate, khop):
    a = _write(tmp_path, mutate)
    os.makedirs(a.out_dir)
    cu = os.path.join(a.out_dir, mod.OUT_NAME)
    with open(cu, "w", encoding="utf-8") as f:
        f.write("ket qua cu\n")
    with pytest.raises(SystemExit, match=khop):
        mod.main(a)
    assert not os.path.exists(cu)  # ket qua cu da bi xoa, khong co ket qua moi


def test_tap_chung_khac_ky_vong_la_loi(mod, tmp_path):
    a = _write(tmp_path)
    a.expected_n = 86073
    with pytest.raises(SystemExit, match="86073"):
        mod.main(a)
    a2 = _write(tmp_path / "rong", lambda n, d: d.iloc[:0])
    a2.expected_n = None
    with pytest.raises(SystemExit, match="rong"):
        mod.main(a2)

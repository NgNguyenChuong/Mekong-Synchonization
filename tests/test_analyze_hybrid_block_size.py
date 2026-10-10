"""scripts/analyze_hybrid_block_size.py - CHG-21 (chi mo ta, so voi dai nhieu cach chia 50 km) + CHG-22 (LOI).

Du lieu gia nho (tmp_path) co I biet truoc: (a) |err| = 1; (b) |err| = 1 + c[design, seed] -> I = c. KHONG chay tren
artifacts that (ket qua k100 hybrid chua duoc mo theo quy dinh chot truoc).
"""
import argparse
import importlib.util
import json
import os
import subprocess

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


TAGS = {None: "nckh-dot7-e6a", "khong_diem": "nckh-dot7-hyb"}


def _name(prefix, g, s, fs, target):
    """Ten luot cua runner: ...__sN[__fs-khong_diem][__t-<t>]; do man khong co __t-."""
    return f"{prefix}__{g}__hist_gb__s{s}" + (f"__fs-{fs}" if fs else "") + (
        "" if target == "salinity" else f"__t-{target}")


def _manifest(tmp_path, files=("KE_HOACH/ket-qua/khac.csv",)):
    man = tmp_path / "KE_HOACH" / "ket-qua" / "manifest_gia.csv"   # cot file tuong doi tmp_path (2 cap tren)
    man.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file": list(files)}).to_csv(man, index=False)
    return str(man)


def _write(tmp_path, mutate=None, target="salinity", mutate_meta=None, frozen_files=("KE_HOACH/ket-qua/khac.csv",)):
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
                    name = _name(prefix, g, s, fs, target)
                    if mutate is not None:
                        d = mutate(name, d)
                    if d is None:
                        continue
                    (exp / name / "cv").mkdir(parents=True)
                    d.to_csv(exp / name / "cv" / "oof_points.csv", index=False)
                    if target == "salinity":
                        continue
                    meta = {"returncode": 0, "mode": "cv", "target": target, "feature_set": fs, "git_tag": TAGS[fs],
                            "points_ref_sha256": "ref1"}
                    if mutate_meta is not None:
                        meta = mutate_meta(name, meta)
                    (exp / name / "cv" / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
                    (exp / name / "cv" / "config.json").write_text(json.dumps({"target": target}), encoding="utf-8")
    out = tmp_path / "ket-qua"
    return argparse.Namespace(prefix50="cv1", prefix100="k100", feature_set="khong_diem", target=target,
                              expected_n=N_PTS * len(SEASONS), exp_root=str(exp), out_dir=str(out),
                              frozen_manifest=[_manifest(tmp_path, frozen_files)],
                              allowed_tags=None if target == "salinity" else sorted(TAGS.values()), tags_reason="r")


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


def test_dot7_ten_luot_gia_tri_va_provenance(mod, tmp_path):
    a = _write(tmp_path, target="ndwi")
    mod.main(a)
    out = os.path.join(a.out_dir, "dot7_ndwi_hybrid_khoi100.csv")
    assert sorted(os.listdir(a.out_dir)) == ["dot7_ndwi_hybrid_khoi100.csv", "dot7_ndwi_hybrid_khoi100.csv.provenance.json"]
    t = pd.read_csv(out).set_index("grid")
    assert t.loc["g_trong", "I_100"] == pytest.approx(0.01) and bool(t.loc["g_trong", "trong_dai_nhieu_50km"])
    assert t.loc["g_ngoai", "I_100"] == pytest.approx(0.05) and not bool(t.loc["g_ngoai", "trong_dai_nhieu_50km"])
    prov = json.loads(open(out + ".provenance.json", encoding="utf-8").read())
    assert prov["target"] == "ndwi" and prov["n_luot"] == len(C) * (3 + 2) * 2 and prov["n_diem_mua_chung"] == 40
    assert prov["git_tags_seen"] == ["nckh-dot7-e6a", "nckh-dot7-hyb"] and prov["schemes"]["k100"] == [42, 43]


@pytest.mark.parametrize("kw, khop", [
    (dict(allowed=None), "git_tag"),                                                          # 2 tag khong khai bao
    (dict(mutate_meta=lambda n, m: {**m, "feature_set": None}
          if n == "k100__g_ngoai__hist_gb__s43__fs-khong_diem__t-ndwi" else m), "feature_set"),
    (dict(mutate_meta=lambda n, m: {**m, "target": "salinity"} if n == "cv1__g_trong__hist_gb__s44__t-ndwi" else m),
     "target"),
    (dict(mutate_meta=lambda n, m: {**m, "returncode": 1} if n == "k100__g_trong__hist_gb__s42__t-ndwi" else m),
     "returncode"),
    (dict(mutate=lambda n, d: None if n == "k100__g_trong__hist_gb__s43__fs-khong_diem__t-ndwi" else d), "thieu"),
])
def test_dot7_meta_sai_la_loi_khong_ghi(mod, tmp_path, kw, khop):
    allowed = kw.pop("allowed", "mac_dinh")
    a = _write(tmp_path, target="ndwi", **kw)
    if allowed is None:
        a.allowed_tags = None
    with pytest.raises(SystemExit, match=khop):
        mod.main(a)
    assert not os.listdir(a.out_dir)


def test_manifest_dong_bang_ma_2_khong_xoa_file_cu(mod, tmp_path):
    a = _write(tmp_path, target="ndwi", frozen_files=("ket-qua/dot7_ndwi_hybrid_khoi100.csv",))
    os.makedirs(a.out_dir)
    cu = os.path.join(a.out_dir, "dot7_ndwi_hybrid_khoi100.csv")
    with open(cu, "w", encoding="utf-8") as f:
        f.write("dong bang\n")
    with pytest.raises(SystemExit) as e:
        mod.main(a)
    assert e.value.code == 2 and open(cu, encoding="utf-8").read() == "dong bang\n"
    a.frozen_manifest = [str(tmp_path / "khong_co.csv")]   # thieu manifest -> khong kiem duoc -> ma 2
    with pytest.raises(SystemExit) as e:
        mod.main(a)
    assert e.value.code == 2 and os.path.exists(cu)
    with pytest.raises(SystemExit, match="TESTED_TIERS"):
        mod.main(_write(tmp_path / "x", target="khong_co"))


def test_do_man_giu_ten_va_byte_nhu_ban_cu(mod, tmp_path, monkeypatch):
    """Mac dinh salinity: dot6_hybrid_khoi100.csv trung tung byte voi ban truoc --target (tag nckh-dot7-hyb)."""
    src = subprocess.run(["git", "show", "nckh-dot7-hyb:scripts/analyze_hybrid_block_size.py"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8")
    if src.returncode != 0:
        pytest.skip("khong doc duoc ban cu tu git")
    (tmp_path / "scripts").mkdir()
    old_path = tmp_path / "scripts" / "analyze_hybrid_block_size_cu.py"
    old_path.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("analyze_hybrid_block_size_cu", str(old_path))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    monkeypatch.setattr(old, "GRIDS", list(C))
    a = _write(tmp_path)
    old.main(argparse.Namespace(prefix50="cv1", prefix100="k100", feature_set="khong_diem", expected_n=a.expected_n,
                                exp_root=a.exp_root, out_dir=str(tmp_path / "out_cu")))
    mod.main(a)
    assert os.listdir(a.out_dir) == [mod.OUT_NAME] == [mod.out_name("salinity")]   # khong sidecar
    with open(os.path.join(a.out_dir, mod.OUT_NAME), "rb") as f1, open(tmp_path / "out_cu" / mod.OUT_NAME, "rb") as f2:
        assert f1.read() == f2.read()

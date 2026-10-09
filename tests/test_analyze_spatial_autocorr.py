"""scripts/analyze_spatial_autocorr.py - CHG-22 (3 trang thai): doc lai CSV ket qua ghi dan va file ket luan.

Khoa trung gia tri khac -> LOI; trung giong het -> khu; thieu/thua mua so voi oof -> LOI; thieu bo -> LOI;
file ket luan cu bi xoa khi bat dau (dung giua chung khong de lai ket luan cu). Du lieu gia nho (tmp_path),
semivariogram/Moran duoc thay bang ham gia (khong fit that).
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_spatial_autocorr",
                                                  os.path.join(ROOT, "scripts", "analyze_spatial_autocorr.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SEASONS = {2014, 2015, 2016}


def _res(rows):
    return pd.DataFrame(rows, columns=["grid", "scheme", "season", "range_km", "ok"])


def _write(tmp_path, df):
    p = tmp_path / "res.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_doc_ket_qua_du_dung(mod, tmp_path):
    rows = [("g1", 42, s, 20.0 + s % 10, True) for s in SEASONS] + [("g1", 43, s, 30.0, True) for s in SEASONS]
    exp = {("g1", 42): SEASONS, ("g1", 43): SEASONS}
    r = mod.doc_ket_qua(_write(tmp_path, _res(rows)), exp)
    assert len(r) == 6


def test_trung_giong_het_duoc_khu_trung_khac_la_loi(mod, tmp_path):
    rows = [("g1", 42, s, 20.0, True) for s in SEASONS]
    exp = {("g1", 42): SEASONS}
    r = mod.doc_ket_qua(_write(tmp_path, _res(rows + [rows[0]])), exp)   # trung giong het -> khu
    assert len(r) == 3
    bad = rows + [("g1", 42, 2014, 99.0, True)]                         # cung khoa, gia tri khac
    with pytest.raises(SystemExit, match="trung khoa"):
        mod.doc_ket_qua(_write(tmp_path, _res(bad)), exp)


def test_thieu_mua_thua_mua_thieu_bo_la_loi(mod, tmp_path):
    rows = [("g1", 42, s, 20.0, True) for s in sorted(SEASONS)]
    with pytest.raises(SystemExit, match="thieu \\[2016\\]"):
        mod.doc_ket_qua(_write(tmp_path, _res(rows[:2])), {("g1", 42): SEASONS})
    with pytest.raises(SystemExit, match="thua \\[2030\\]"):
        mod.doc_ket_qua(_write(tmp_path, _res(rows + [("g1", 42, 2030, 20.0, True)])), {("g1", 42): SEASONS})
    with pytest.raises(SystemExit, match="g1 s43"):                    # bo duoc yeu cau khong co dong nao
        mod.doc_ket_qua(_write(tmp_path, _res(rows)), {("g1", 42): SEASONS, ("g1", 43): SEASONS})


def test_bo_khong_yeu_cau_bi_bo_qua(mod, tmp_path):
    rows = [("g1", 42, s, 20.0, True) for s in SEASONS] + [("g2", 44, 2014, 20.0, True)]
    r = mod.doc_ket_qua(_write(tmp_path, _res(rows)), {("g1", 42): SEASONS})
    assert set(r["grid"]) == {"g1"}


def test_oof_khac_tap_mua_la_loi(mod):
    mod.kiem_cung_tap_mua({("a", 42): SEASONS, ("b", 42): set(SEASONS)})
    with pytest.raises(SystemExit):
        mod.kiem_cung_tap_mua({("a", 42): SEASONS, ("b", 42): SEASONS - {2016}})
    with pytest.raises(SystemExit):
        mod.kiem_cung_tap_mua({("a", 42): set()})


def _fake_world(mod, tmp_path, monkeypatch, grids=("h3_res_7",), schemes=(42,)):
    """Gia lap: diem, oof_points, semivariogram/Moran (ham gia) -> chay main() tren tmp_path.

    main() doc them h3_res_7 s42 cho bat dang huong -> grids phai co h3_res_7 va schemes co 42.
    """
    assert "h3_res_7" in grids and 42 in schemes
    rng = np.random.default_rng(0)
    pid = [f"p{i}" for i in range(40)]
    pxy = pd.DataFrame({"point_id": pid, "x": 500_000 + rng.uniform(0, 1e5, 40), "y": 1e6 + rng.uniform(0, 1e5, 40)})
    exp = tmp_path / "artifacts" / "experiments"
    for g in grids:
        for s in schemes:
            d = exp / f"cv1__{g}__hist_gb__s{s}" / "cv"
            d.mkdir(parents=True)
            o = pd.DataFrame([(p, se) for se in sorted(SEASONS) for p in pid], columns=["point_id", "season"])
            o["err"] = rng.normal(size=len(o))
            o["y_ref"] = rng.uniform(0, 5, len(o))
            o.to_csv(d / "oof_points.csv", index=False)
    monkeypatch.setattr(mod, "ROOT", str(tmp_path))
    monkeypatch.setattr(mod, "GRIDS", list(grids))
    monkeypatch.setattr(mod, "point_xy", lambda: pxy)
    calls = {"n": 0}

    def fake_vg(xy, v, azimuth=None, **k):
        calls["n"] += 1
        return {"n": len(v), "range_m": 20_000.0, "sill": 1.0, "nugget": 0.1, "ok": True, "reason": ""}

    monkeypatch.setattr(mod, "variogram_range", fake_vg)
    monkeypatch.setattr(mod, "moran_band", lambda xy, v: {"I": 0.1, "p_sim": 0.5, "n_islands": 0})
    out = tmp_path / "ket-qua"
    out.mkdir()
    return argparse.Namespace(prefix="cv1", schemes=list(schemes), out_dir=str(out), detrend_coast=False), out, calls


def test_main_xoa_ket_luan_cu_va_dung_khi_loi(mod, tmp_path, monkeypatch):
    a, out, calls = _fake_world(mod, tmp_path, monkeypatch, grids=("h3_res_7",))
    for n in mod.CONCL_THUONG:
        (out / n).write_text("ket luan cu\n", encoding="utf-8")
    # CSV phan du co san mot dong cung khoa nhung gia tri khac voi dong se ghi -> LOI khi doc lai
    pd.DataFrame([{"grid": "h3_res_7", "scheme": 42, "season": 2014, "n": 40, "range_km": 99.0, "ok": True,
                   "reason": "", "nugget": 0.1, "sill": 1.0, "moran_I": 0.1, "moran_p": 0.5,
                   "moran_n_islands": 0}]).to_csv(out / "dot6_tu_tuong_quan_phan_du.csv", index=False)
    # dong tren duoc coi la da xong (bo qua mua 2014) -> khong trung; them dong trung khac gia tri
    df = pd.read_csv(out / "dot6_tu_tuong_quan_phan_du.csv")
    pd.concat([df, df.assign(range_km=12.0)]).to_csv(out / "dot6_tu_tuong_quan_phan_du.csv", index=False)
    with pytest.raises(SystemExit, match="trung khoa"):
        mod.main(a)
    for n in mod.CONCL_THUONG:  # ket luan cu da bi xoa, khong co ket luan moi
        assert not (out / n).exists()


def test_main_du_lieu_dung_ghi_ket_luan(mod, tmp_path, monkeypatch):
    a, out, calls = _fake_world(mod, tmp_path, monkeypatch, grids=("h3_res_7",))
    mod.main(a)
    k = pd.read_csv(out / "dot6_tu_tuong_quan_ket_luan.csv")
    assert len(k) == 1 and not k["violates"].iloc[0] and k["n_over"].iloc[0] == 0
    assert len(pd.read_csv(out / "dot6_tu_tuong_quan_nhan.csv")) == 3
    assert len(pd.read_csv(out / "dot6_tu_tuong_quan_huong.csv")) == 3 * 2 * 2
    assert not list(out.glob("*.tmp"))
    # chay lai: phan du da du -> khong fit lai, ket luan giong het
    n1 = calls["n"]
    before = (out / "dot6_tu_tuong_quan_ket_luan.csv").read_bytes()
    mod.main(a)
    assert (out / "dot6_tu_tuong_quan_ket_luan.csv").read_bytes() == before
    assert calls["n"] - n1 == 3 + 3 * 2 * 2  # chi nhan + bat dang huong tinh lai (khong ghi dan)


def test_detrend_xoa_ket_luan_cu_khi_thieu_mua(mod, tmp_path, monkeypatch):
    a, out, calls = _fake_world(mod, tmp_path, monkeypatch, grids=("h3_res_7",))
    a.detrend_coast = True
    (out / "dot6_tu_tuong_quan_bo_xu_huong_ket_luan.csv").write_text("cu\n", encoding="utf-8")
    pxy = mod.point_xy()
    monkeypatch.setattr(mod, "point_dist_coast", lambda: pd.Series(np.linspace(1, 50, len(pxy)),
                                                                   index=pxy["point_id"]))

    def vg_loi(xy, v, **k):  # mua thu 2 gap LOI -> dung, khong ghi ket luan
        if calls["n"] == 1:
            raise MemoryError("gia lap")
        calls["n"] += 1
        return {"n": len(v), "range_m": 20_000.0, "sill": 1.0, "nugget": 0.1, "ok": True, "reason": ""}

    monkeypatch.setattr(mod, "variogram_range", vg_loi)
    with pytest.raises(MemoryError):
        mod.main(a)
    assert not (out / "dot6_tu_tuong_quan_bo_xu_huong_ket_luan.csv").exists()
    # CSV ghi dan dang thieu 2 mua (khong co dong nao duoc ghi vi loi giua bo) -> doc lai bang tay -> LOI thieu mua
    pd.DataFrame([{"grid": "h3_res_7", "scheme": 42, "season": 2014, "n": 40, "range_km": 20.0, "ok": True,
                   "reason": "", "nugget": 0.1, "sill": 1.0}]).to_csv(out / "dot6_tu_tuong_quan_bo_xu_huong.csv",
                                                                     index=False)
    with pytest.raises(SystemExit, match="thieu"):
        mod.doc_ket_qua(str(out / "dot6_tu_tuong_quan_bo_xu_huong.csv"), {("h3_res_7", 42): SEASONS})
    # chay lai binh thuong -> tinh tiep 2 mua con lai, du -> ghi ket luan
    calls["n"] = 2
    mod.main(a)
    k = pd.read_csv(out / "dot6_tu_tuong_quan_bo_xu_huong_ket_luan.csv")
    assert len(k) == 1 and k["n_over"].iloc[0] == 0

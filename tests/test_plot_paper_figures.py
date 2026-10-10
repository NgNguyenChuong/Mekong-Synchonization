"""plot_paper_figures.py tren du lieu gia: dai null dung phan vi, nhan cuoi 4 lop, guard manifest, file ra du."""
import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from test_plot_dot7_figures import F1_PAIRS, REF, TIER, _area

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def pp():
    spec = importlib.util.spec_from_file_location("plot_paper_figures",
                                                  os.path.join(ROOT, "scripts", "plot_paper_figures.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_null_band_dung_phan_vi_ly_thuyet(pp):
    # 3 cap cung se: P(max|Z| <= x) = (2 Phi(x) - 1)^3
    lo, hi = pp.null_band([1.0, 1.0, 1.0], 1.0)
    q = lambda p: norm.ppf((1 + p ** (1 / 3)) / 2)  # noqa: E731
    assert lo == pytest.approx(q(0.025), rel=0.02) and hi == pytest.approx(q(0.975), rel=0.01)
    # 1 cap, se 2, Delta_min 4: |N(0, 2)| / 4
    lo, hi = pp.null_band([2.0], 4.0)
    assert lo == pytest.approx(0.5 * norm.ppf(0.5125), rel=0.03) and hi == pytest.approx(0.5 * norm.ppf(0.9875),
                                                                                         rel=0.01)
    assert pp.null_band([1.0, 2.0], 1.0) == pp.null_band([1.0, 2.0], 1.0)  # tat dinh (seed 42)
    for bad in ([0.0, 1.0], [np.nan], []):
        with pytest.raises(SystemExit):
            pp.null_band(bad, 1.0)


def _final_cap(t, labels):
    return pd.DataFrame([{"target": t, "family": "F1_chinh", "grid_a": a, "grid_b": b, "delta_hat": 0.1, "label": lab}
                         for (_, a, b), lab in zip(F1_PAIRS, labels)])


def test_nhan_cuoi_anh_xa_4_lop(pp, tmp_path):
    labs = ["tuong_duong", "khong_tai_lap", "chua_phan_dinh", "mo_ta", "khong_tai_lap_cap_ho"] + ["mo_ta"] * 7
    _final_cap("rain_chirps", labs).to_csv(tmp_path / "dot7_rain_chirps_cv1_final_cap.csv", index=False)
    st = {"notes": [], "used": []}
    d = pp.final_labels(str(tmp_path), "rain_chirps", st)
    assert list(d["final"][:5]) == ["eq", "nrep", "und", "desc", "nrep"]
    assert set(d["final"]) <= {"eq", "nrep", "und", "desc"}
    pd.DataFrame({"grid_a": ["h3_res_7"], "grid_b": ["s2_level_12"], "delta_hat": [0.1], "label": ["tuong_duong"]}).to_csv(
        tmp_path / "dot5_final_so_cap_voi_cv.csv", index=False)
    assert list(pp.final_labels(str(tmp_path), "salinity", st)["final"]) == ["eq"]
    _final_cap("ndwi", ["xac_nhan"] + ["mo_ta"] * 11).to_csv(tmp_path / "dot7_ndwi_cv1_final_cap.csv", index=False)
    with pytest.raises(SystemExit, match="chua co ky hieu"):  # lop chua ve -> LOI, khong bia ky hieu
        pp.final_labels(str(tmp_path), "ndwi", st)
    assert pp.final_labels(str(tmp_path), "t2m_era5", st) is None and st["notes"]


def _args(tmp_path, manifest, only, thumb=False):
    return argparse.Namespace(results_dir=str(tmp_path / "res"), out_dir=str(tmp_path / "out"),
                              grids_dir=str(tmp_path), eval_dir=str(tmp_path), boundary=str(tmp_path / "b.geojson"),
                              only=only, frozen_manifest=[manifest], thumb=thumb)


def _manifest(tmp_path, files=()):
    d = tmp_path / "KE_HOACH" / "ket-qua"  # cot file tuong doi goc = 2 cap tren thu muc manifest
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file": list(files)}).to_csv(d / "m.csv", index=False)
    return str(d / "m.csv")


def test_guard_chan_manifest_gia(pp, tmp_path):
    m = _manifest(tmp_path, ["out/fig6_main.png"])
    with pytest.raises(SystemExit) as e:
        pp.main(_args(tmp_path, m, ["fig6"]))
    assert e.value.code == 2 and not (tmp_path / "out").exists()
    with pytest.raises(SystemExit) as e:  # thieu manifest -> khong ghi
        pp.main(_args(tmp_path, str(tmp_path / "khong_co.csv"), ["fig6"]))
    assert e.value.code == 2


def _fake_results(pp, res):
    res.mkdir()
    _area().to_csv(res / pp.AREA_FILE, index=False)
    eff = []
    for i, t in enumerate(pp.ORDER):
        rows = [{"target": t, "family": "F1_chinh", "grid_a": a, "grid_b": b, "delta_ref_level": m, "delta_ref": REF[m],
                 "delta_hat": 0.05 * (k % 3 - 1), "ci_low": -0.3, "ci_high": 0.3, "ci_tost_low": -0.25,
                 "ci_tost_high": 0.25, "se": 0.1 + 0.01 * k, "delta_min_thr": 0.2, "p_holm": 1.0,
                 "label": "tuong_duong"} for k, (m, a, b) in enumerate(F1_PAIRS)]
        f1 = pd.DataFrame(rows)
        f1.to_csv(res / pp.out_name(t, "cv1"), index=False)
        if t == "salinity":
            f1[["grid_a", "grid_b", "delta_hat", "label"]].to_csv(res / "dot5_final_so_cap_voi_cv.csv", index=False)
        else:
            f1[["target", "family", "grid_a", "grid_b", "delta_hat", "label"]].to_csv(
                res / f"dot7_{t}_cv1_final_cap.csv", index=False)
        for m, g in f1.groupby("delta_ref_level"):
            eff.append({"variable": t, "muc": m, "anh_huong_khung": g["delta_hat"].abs().max() / 0.2,
                        "kiem_dinh": i < 3, "n_cap": len(g), "nguon_tren_canh_median": 10.0 ** (i - 3) * (8 - m)})
    pd.DataFrame(eff).to_csv(res / pp.MAIN_FILE, index=False)
    pd.DataFrame({"variable": list(pp.ORDER), "moran_median": 0.5, "moran_min": 0.4, "moran_max": 0.6}).to_csv(
        res / pp.SUMMARY_FILE, index=False)


def test_file_ra_du(pp, tmp_path):
    assert len(pp.outputs("x")) == 16 and {os.path.basename(p) for p in pp.outputs("x")} >= {
        "fig1_study_design.pdf", "fig6_main.png", "figS1_model_frame.pdf", "figS2_model_frame_mlp.png"}
    _fake_results(pp, tmp_path / "res")
    pp.main(_args(tmp_path, _manifest(tmp_path), ["fig3", "fig6"]))
    for name in ("fig3_same_level_pairs", "fig6_main"):
        for ext in ("pdf", "png"):
            p = tmp_path / "out" / f"{name}.{ext}"
            assert p.stat().st_size > 0
            prov = json.loads((tmp_path / "out" / f"{name}.{ext}.provenance.json").read_text(encoding="utf-8"))
            assert prov["chu_thich_en"] and prov["sha_dau_vao"] and prov["phan_bo"] == []
    prov = json.loads((tmp_path / "out" / "fig6_main.png.provenance.json").read_text(encoding="utf-8"))
    vals = prov["dai_null"]["gia_tri"]
    assert len(vals) == 18 and all(0 < v["null_lo"] < v["null_hi"] for v in vals)
    assert not list((tmp_path / "out").glob("*_thumb.png"))  # khong --thumb -> khong anh nho


def test_thumb_150dpi_nho_va_trong_guard(pp, tmp_path):
    from PIL import Image

    assert len(pp.outputs("x", thumb=True)) == 24
    with pytest.raises(SystemExit) as e:  # anh nho cung bi chan neu nam trong manifest
        pp.main(_args(tmp_path, _manifest(tmp_path, ["out/fig3_same_level_pairs_thumb.png"]), ["fig3"], thumb=True))
    assert e.value.code == 2
    _fake_results(pp, tmp_path / "res")
    pp.main(_args(tmp_path, _manifest(tmp_path), ["fig3"], thumb=True))
    p = tmp_path / "out" / "fig3_same_level_pairs_thumb.png"
    w_mm = pp.SIZE_MM["fig3"][0]
    assert Image.open(p).size[0] == pytest.approx(w_mm / 25.4 * 150, abs=2)
    assert p.stat().st_size <= pp.THUMB_MAX_KB * 1024 and (tmp_path / "out" / f"{p.name}.provenance.json").is_file()


def test_thuoc_ti_le_dung_50_km_khong_doi_khung(pp):
    from pyproj import Geod

    fig = pp._plt().figure()
    ax = fig.add_subplot()
    ax.set_xlim(104.3, 107.0)
    ax.set_ylim(8.4, 11.2)
    pp.north_scale(ax)
    (xs, xe), (y, _) = ax.lines[0].get_data()
    assert Geod(ellps="WGS84").inv(xs, y, xe, y)[2] == pytest.approx(50_000, rel=1e-3)
    assert ax.get_xlim() == (104.3, 107.0) and ax.get_ylim() == (8.4, 11.2)
    assert [t.get_text() for t in ax.texts if t.get_text()] == ["50 km", "N"]  # mui ten = Annotation chu rong
    pp._plt().close(fig)

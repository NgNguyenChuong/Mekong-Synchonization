"""Cat nhan GEE theo dau chan Zenodo (src/labels.py): can luoi lech nguyen pixel theo transform,
khong noi suy; NaN ngoai dau chan; NaN san co khong bi dien; doc band theo ten.

Du lieu la raster TONG HOP (gia tri biet truoc), luoi 30 m EPSG:32648 nhu that nhung nho.
"""
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from labels import (build_footprint, clip_label_file, clip_to_footprint, footprint_on_grid,  # noqa: E402
                    pixel_offset, zenodo_label_paths)

# Luoi "Zenodo" 4 hang x 5 cot, goc (438540, 1219680); luoi "GEE" 6 x 8, goc lech -2 cot, -1 hang
# (bo so 30 m) -> Zenodo nam tai hang 1..4, cot 2..6 cua GEE. Cung kieu lech that (2292 cot, 14 hang).
Z_TR = from_origin(438540, 1219680, 30, 30)
G_TR = from_origin(438540 - 2 * 30, 1219680 + 1 * 30, 30, 30)


def _write(path, bands, transform, names=("NDWIchen", "Salinity")):
    bands = [np.asarray(b, dtype="float32") for b in bands]
    h, w = bands[0].shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=len(bands), dtype="float32",
                       crs="EPSG:32648", transform=transform) as ds:   # khong tag nodata, nhu file GEE
        for i, (b, n) in enumerate(zip(bands, names), start=1):
            ds.write(b, i)
            ds.set_band_description(i, n)
    return str(path)


def _zenodo_pair(tmp_path):
    """Hai nam Zenodo: dau chan = hop pixel Salinity huu han. Band 1 (NDWI) huu han o MOI pixel de bat
    loi doc band theo so thu tu."""
    nd = np.zeros((4, 5))
    s1 = np.full((4, 5), np.nan)
    s2 = np.full((4, 5), np.nan)
    s1[0, :3] = 1.0          # nam 1: hang 0 cot 0..2
    s2[0, 2:4] = 2.0         # nam 2: hang 0 cot 2..3 -> hop: cot 0..3
    s2[2, 1] = 3.0
    p1 = _write(tmp_path / "2014_MD_dry_NDWIchen_Salinity.tif", [nd, s1], Z_TR)
    p2 = _write(tmp_path / "2015_MD_dry_NDWIchen_Salinity.tif", [nd, s2], Z_TR)
    fp_expect = np.zeros((4, 5), bool)
    fp_expect[0, :4] = True
    fp_expect[2, 1] = True
    return [p1, p2], fp_expect


def test_dau_chan_la_hop_salinity_huu_han_doc_band_theo_ten(tmp_path):
    paths, fp_expect = _zenodo_pair(tmp_path)
    fp, tr, _ = build_footprint(paths)
    assert tr == Z_TR
    np.testing.assert_array_equal(fp, fp_expect)   # neu doc band 1 (NDWI) -> toan True
    # Doi thu tu band (Salinity truoc) van doc dung theo ten
    nd = np.zeros((4, 5))
    s = np.full((4, 5), np.nan)
    s[3, 4] = 1.0
    p = _write(tmp_path / "swap.tif", [s, nd], Z_TR, names=("Salinity", "NDWIchen"))
    fp2, _, _ = build_footprint([p])
    assert fp2.sum() == 1 and fp2[3, 4]


def test_pixel_offset_lech_nguyen_va_tu_choi_lech_le():
    assert pixel_offset(Z_TR, G_TR) == (1, 2)
    # Lech that: GEE v2 369780/1220100 vs Zenodo 438540/1219680
    assert pixel_offset(from_origin(438540, 1219680, 30, 30), from_origin(369780, 1220100, 30, 30)) == (14, 2292)
    with pytest.raises(ValueError, match="khong nguyen"):
        pixel_offset(from_origin(438555, 1219680, 30, 30), G_TR)         # lech 15 m -> khong can, khong noi suy
    with pytest.raises(ValueError, match="Kich thuoc pixel"):
        pixel_offset(from_origin(438540, 1219680, 10, 10), G_TR)


def test_footprint_on_grid_dich_nguyen_pixel_ngoai_khung_la_false():
    fp = np.zeros((4, 5), bool)
    fp[0, 0] = fp[3, 4] = True
    out = footprint_on_grid(fp, Z_TR, G_TR, (6, 8))
    assert out.sum() == 2 and out[1, 2] and out[4, 6]
    # Luoi dich nho hon, cat ngang khung fp: chi phan chong lan
    out2 = footprint_on_grid(fp, Z_TR, from_origin(438540 + 30, 1219680, 30, 30), (4, 3))
    assert out2.sum() == 0                                              # (0,0) va (3,4) deu ngoai cua so
    out3 = footprint_on_grid(fp, Z_TR, from_origin(438540 + 4 * 30, 1219680 - 3 * 30, 30, 30), (2, 2))
    assert out3.sum() == 1 and out3[0, 0]


def test_clip_to_footprint_nan_ngoai_dau_chan_va_khong_dien_nan_san_co(tmp_path):
    paths, fp_expect = _zenodo_pair(tmp_path)
    fp, ztr, _ = build_footprint(paths)
    lab = np.arange(48, dtype="float32").reshape(6, 8) + 0.5
    lab[1, 2] = np.nan                    # trong dau chan nhung NaN san -> van NaN, khong dien
    out, n_cut = clip_to_footprint(lab, G_TR, fp, ztr)
    inside = np.zeros((6, 8), bool)
    inside[1:5, 2:7] = fp_expect
    assert np.isnan(out[~inside]).all()
    keep = inside & np.isfinite(lab)
    np.testing.assert_array_equal(out[keep], lab[keep])           # gia tri trong dau chan giu nguyen
    assert np.isnan(out[1, 2])
    assert n_cut == int((~inside).sum())                           # moi pixel ngoai deu huu han truoc khi cat
    assert np.isfinite(lab[0, 0])                                  # khong sua mang dau vao
    # Mang 3 chieu (band, h, w): cat moi band
    out3, n3 = clip_to_footprint(np.stack([lab, lab]), G_TR, fp, ztr)
    assert n3 == n_cut and np.isnan(out3[:, ~inside]).all()


def test_clip_label_file_dem_va_ghi_ca_hai_band(tmp_path):
    paths, fp_expect = _zenodo_pair(tmp_path)
    fp, ztr, _ = build_footprint(paths)
    s = np.full((6, 8), 1.5, dtype="float32")
    s[0, 0] = np.nan                       # ngoai dau chan, da NaN -> khong tinh la "bi cat"
    nd = np.full((6, 8), 0.2, dtype="float32")
    src = _write(tmp_path / "2024_MD_dry_NDWIchen_Salinity_l8_v2.tif", [nd, s], G_TR)
    inside = np.zeros((6, 8), bool)
    inside[1:5, 2:7] = fp_expect
    region = np.zeros((6, 8), bool)
    region[:3] = True                      # "vung" = 3 hang dau
    dst = tmp_path / "out_fp.tif"
    st = clip_label_file(src, fp, ztr, dst_path=str(dst), regions={"r": region}, chunk_rows=2)  # nhieu khoi
    n_out = int((~inside).sum())
    assert st["n_cut"] == n_out - 1
    assert st["n_finite_before"] == 47 and st["n_finite_after"] == int(inside.sum())
    assert st["n_cut_r"] == int((~inside & region).sum()) - 1
    assert st["n_finite_after_r"] == int((inside & region).sum())
    with rasterio.open(dst) as r:
        assert r.descriptions == ("NDWIchen", "Salinity") and r.transform == G_TR
        a = r.read()
    assert np.isnan(a[:, ~inside]).all()
    assert (a[1][inside] == 1.5).all() and (a[0][inside] == np.float32(0.2)).all()
    # Chi dem (khong ghi) cho cung so
    assert clip_label_file(src, fp, ztr)["n_cut"] == st["n_cut"]


def test_zenodo_label_paths_thieu_nam_bao_loi(tmp_path):
    _zenodo_pair(tmp_path)
    assert len(zenodo_label_paths(str(tmp_path), (2014, 2015))) == 2
    with pytest.raises(FileNotFoundError, match="2016"):
        zenodo_label_paths(str(tmp_path), (2014, 2015, 2016))

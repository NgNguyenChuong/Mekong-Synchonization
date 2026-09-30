"""Doi chieu nhan: anh giong het -> r = 1, MAE = 0; lech hang so -> bias dung; lech luoi van canh dung pixel."""
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from compare_salinity_labels import compare  # noqa: E402


def _write(path, ndwi, sal, x0=500000.0, y0=1100000.0):
    h, w = sal.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=2, dtype="float32",
                       crs="EPSG:32648", transform=from_origin(x0, y0, 30, 30), nodata=np.nan) as dst:
        dst.write(ndwi.astype("float32"), 1)
        dst.write(sal.astype("float32"), 2)


@pytest.fixture
def base():
    rng = np.random.default_rng(0)
    sal = rng.uniform(0.2, 5, (40, 50))
    sal[0, :5] = np.nan
    return rng.uniform(-0.5, 0.5, (40, 50)), sal


def test_identical_rasters(tmp_path, base):
    _write(tmp_path / "z.tif", *base)
    _write(tmp_path / "o.tif", *base)
    df = compare(tmp_path / "z.tif", tmp_path / "o.tif").set_index("band")
    assert df.loc["Salinity", "pearson_r"] == pytest.approx(1.0)
    assert df.loc["Salinity", "mae"] == 0 and df.loc["Salinity", "n_both"] == 40 * 50 - 5


def test_constant_bias_and_shifted_grid(tmp_path, base):
    ndwi, sal = base
    _write(tmp_path / "z.tif", ndwi, sal)
    # Anh cua minh lech 1 cot (30 m) sang trai nhung cung gia tri tai cung toa do + bias 0.5.
    pad = np.full((40, 1), 1.0)
    _write(tmp_path / "o.tif", np.hstack([pad, ndwi]), np.hstack([pad, sal]) + 0.5, x0=500000.0 - 30)
    df = compare(tmp_path / "z.tif", tmp_path / "o.tif").set_index("band")
    assert df.loc["Salinity", "bias_ours_minus_zenodo"] == pytest.approx(0.5, abs=1e-5)
    assert df.loc["Salinity", "pearson_r"] == pytest.approx(1.0)
    assert df.loc["NDWIchen", "mae"] == pytest.approx(0.0, abs=1e-6)

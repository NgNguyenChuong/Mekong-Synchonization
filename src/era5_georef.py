"""Sua georeference ERA5-Land tai bang `scale=11132` + EPSG:4326 (known-pitfalls 1a).

Luoi goc ERA5-Land tren GEE: crs_transform [0.1, 0, -180.05, 0, -0.1, 90.05] (tam pixel o boi so 0,1 do).
File tai bang `scale=` co transform (0.10000046, 103.700474, 11.200051): tam pixel xuat roi SAT mep
pixel goc (lech 0,0005 do ve Dong / 0,00005 do ve Bac) -> nearest chon pixel goc chua tam do, tuc pixel
Dong-Bac. Gia tri cua pixel xuat (r, c) la gia tri pixel goc chua tam (r, c) -> georeference dung la
CHINH pixel goc do: transform moi = luoi goc 0,1 do tron, goc = mep tren-trai cua pixel goc chua tam
pixel (0, 0). Voi file hien tai: (103.75, 11.25) = goc cu + ~0,05 do Dong, + ~0,05 do Bac.

Chi doi transform trong header, khong doi du lieu. Ham `native_transform` idempotent: ap len transform
da nam tren luoi goc tra ve chinh no.
"""
import hashlib
import math

import numpy as np
from rasterio.transform import Affine

NATIVE_RES = 0.1
NATIVE_X0 = -180.05
NATIVE_Y0 = 90.05
NATIVE_CRS_TRANSFORM = [NATIVE_RES, 0, NATIVE_X0, 0, -NATIVE_RES, NATIVE_Y0]
# Sai so cho phep khi coi mot toa do la "nam tren mep luoi goc" (do). Transform luu float64 -> ~1e-12.
EDGE_TOL = 1e-7
# Tam pixel xuat phai cach mep pixel goc it nhat chung nay (don vi pixel goc; 1e-4 px ~ 1 m) thi "nearest"
# moi xac dinh duy nhat - neu tam roi dung mep, khong biet GEE chon pixel nao -> tu choi sua.
MIN_MARGIN_PX = 1e-4


def _native_index(coord, origin, step):
    """Chi so pixel goc chua toa do `coord` (step > 0 cho x, < 0 cho y) va khoang cach toi mep gan nhat."""
    u = (coord - origin) / step
    k = math.floor(u)
    return k, min(u - k, k + 1 - u)


def is_native_aligned(t: Affine, tol: float = EDGE_TOL) -> bool:
    """True neu transform la luoi goc 0,1 do tron (khong xoay) va goc nam tren mep pixel goc."""
    if abs(t.b) > 0 or abs(t.d) > 0:
        return False
    if abs(t.a - NATIVE_RES) > 1e-12 or abs(t.e + NATIVE_RES) > 1e-12:
        return False
    ux = (t.c - NATIVE_X0) / NATIVE_RES
    uy = (NATIVE_Y0 - t.f) / NATIVE_RES
    return abs(ux - round(ux)) * NATIVE_RES < tol and abs(uy - round(uy)) * NATIVE_RES < tol


def native_transform(t: Affine, width: int, height: int) -> Affine:
    """Transform luoi goc ung voi gia tri da xuat theo `t` (nearest tai tam pixel xuat).

    Kiem: pixel (0,0) va (height-1, width-1) phai roi vao pixel goc lien tiep (khong bo/lap pixel do
    sai khac do phan giai) va tam pixel khong qua sat mep goc (mo ho). Sai -> ValueError.
    """
    if is_native_aligned(t):
        return Affine(NATIVE_RES, 0.0, t.c, 0.0, -NATIVE_RES, t.f)
    if abs(t.b) > 0 or abs(t.d) > 0:
        raise ValueError(f"Transform co xoay: {t}")
    out = {}
    for name, n, origin, step, start, res in (("x", width, NATIVE_X0, NATIVE_RES, t.c, t.a),
                                              ("y", height, NATIVE_Y0, -NATIVE_RES, t.f, t.e)):
        k0, m0 = _native_index(start + 0.5 * res, origin, step)
        k1, m1 = _native_index(start + (n - 0.5) * res, origin, step)
        if k1 - k0 != n - 1:
            raise ValueError(f"Truc {name}: pixel xuat 0..{n - 1} roi vao pixel goc {k0}..{k1} (khong lien tiep)")
        if min(m0, m1) < MIN_MARGIN_PX:
            raise ValueError(f"Truc {name}: tam pixel xuat cach mep pixel goc {min(m0, m1):.2e} px - mo ho")
        out[name] = origin + k0 * step
    return Affine(NATIVE_RES, 0.0, out["x"], 0.0, -NATIVE_RES, out["y"])


def data_digest(ds) -> str:
    """sha256 cua moi band (bytes thuan du lieu) - dung de chung minh sua header khong doi du lieu."""
    h = hashlib.sha256()
    for b in range(1, ds.count + 1):
        h.update(np.ascontiguousarray(ds.read(b)).tobytes())
    return h.hexdigest()


def transform_list(t: Affine) -> list:
    return [float(v) for v in tuple(t)[:6]]

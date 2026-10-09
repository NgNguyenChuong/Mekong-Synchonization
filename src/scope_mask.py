"""Mat na pham vi TINH (An duyet M 2026-10-03; CHG-10 2026-10-03) - dung chung cho diem danh gia, nhan, dac trung.

pham vi v3 (CHG-10, MAC DINH) = ranh gioi v2 (tam pixel nam trong)
        ∩ WorldCover 2021 THUOC {10 cay, 20 bui, 30 co, 40 dat canh tac} (gia tri tai TAM pixel 30 m)
        ∩ dau chan Zenodo: Salinity HUU HAN o >= 1 nam trong cac file nhan dua vao (2014-2023).
pham vi v2 (truoc CHG-10; chay lai bang exclude=SCOPE_EXCLUDE_LC) = v2 ∩ WorldCover KHONG thuoc {0, 80, 95}
        ∩ dau chan Zenodo.
Khong phai "co nhan moi nam": pixel khong hop le theo nam do quy tac 5/9 o buoc danh gia xu ly.
Band wc_class (v3): lop WorldCover tai tam pixel 30 m TRONG ranh gioi v2 (0 = ngoai ranh gioi / ngoai khung WC)
-> bo phu dung lai duoc: isin(wc_class, cac_lop) & zenodo_n_years >= 1.

Luoi dich = luoi pixel 30 m cua file nhan Zenodo (EPSG:32648; luoi GEE v2 cung goc neo boi so 30 m).
Doc band theo TEN (Zenodo: band 1 = NDWIchen, band 2 = Salinity). File GEE khong co tag nodata -> khong
dua vao masked=True; dung np.isfinite.
"""
import numpy as np
import pandas as pd

SCOPE_EXCLUDE_LC = (0, 80, 95)          # quy tac v2 (truoc CHG-10) - giu de chay lai v2
SCOPE_INCLUDE_LC = (10, 20, 30, 40)     # quy tac v3 (CHG-10): bo chinh chi 4 lop nay
SCOPE_BANDS = ("scope", "zenodo_n_years")
SCOPE_BANDS_V3 = ("scope", "zenodo_n_years", "wc_class")


def band_index(src, band) -> int:
    """Chi so band (1-based) theo ten (descriptions) hoac so; ten khong co -> loi."""
    if isinstance(band, str):
        names = list(src.descriptions)
        if band not in names:
            raise ValueError(f"{src.name}: khong co band '{band}' (co {names}).")
        return names.index(band) + 1
    return int(band)


def nearest_on_grid(src, band, transform, shape, row0=0, fill=0):
    """Gia tri pixel nguon CHUA tam tung pixel cua khoi hang luoi dich [row0, row0 + shape[0]).

    src: rasterio dataset da mo (cung CRS voi luoi dich). Ngoai khung nguon = fill.
    Doc mot cua so nguon bao trum khoi hang -> khong nap ca raster.
    """
    from rasterio.windows import Window

    h, w = shape
    xs = transform.c + transform.a * (np.arange(w) + 0.5)
    ys = transform.f + transform.e * (row0 + np.arange(h) + 0.5)
    inv = ~src.transform
    cols = np.floor(inv.a * xs + inv.c).astype(np.int64)
    rows = np.floor(inv.e * ys + inv.f).astype(np.int64)
    okc = (cols >= 0) & (cols < src.width)
    okr = (rows >= 0) & (rows < src.height)
    b = band_index(src, band)
    out = np.full(shape, fill, dtype=src.dtypes[b - 1])
    if not okc.any() or not okr.any():
        return out
    c0, c1 = int(cols[okc].min()), int(cols[okc].max())
    r0, r1 = int(rows[okr].min()), int(rows[okr].max())
    data = src.read(b, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
    out[np.ix_(okr, okc)] = data[np.ix_(rows[okr] - r0, cols[okc] - c0)]
    return out


def wc_at_centers(wc_path, transform, crs, shape, chunk_rows=1024):
    """Lop WorldCover (band 'Map' hoac 1) tai TAM tung pixel luoi dich; ngoai khung WC = 0. CRS khac -> loi."""
    import rasterio

    h, w = shape
    out = np.zeros(shape, np.uint8)
    with rasterio.open(wc_path) as wc:
        if wc.crs != crs:
            raise ValueError(f"{wc_path} CRS {wc.crs} khac luoi dich {crs}.")
        b = "Map" if "Map" in wc.descriptions else 1
        for r0 in range(0, h, chunk_rows):
            n = min(chunk_rows, h - r0)
            out[r0:r0 + n] = nearest_on_grid(wc, b, transform, (n, w), r0, fill=0)
    return out


def wc_rule(include=None, exclude=None):
    """(include, exclude, mo ta) cua quy tac WorldCover. Ca hai None -> v3: include SCOPE_INCLUDE_LC (CHG-10).
    Truyen ca hai -> loi (quy tac mo ho)."""
    if include is not None and exclude is not None:
        raise ValueError("Chi truyen MOT trong include / exclude.")
    if include is None and exclude is None:
        include = SCOPE_INCLUDE_LC
    if include is not None:
        include = tuple(int(v) for v in include)
        return include, None, "WorldCover thuoc {" + ",".join(map(str, include)) + "}"
    exclude = tuple(int(v) for v in exclude)
    return None, exclude, "WorldCover khong thuoc {" + ",".join(map(str, exclude)) + "}"


def build_scope_mask(boundary_geom, wc_path, label_paths, ref_path=None, exclude=None, include=None,
                     label_band="Salinity", chunk_rows=1024, log=None):
    """Dung mat na pham vi tren luoi cua ref_path (mac dinh file nhan dau tien).

    boundary_geom: hinh ranh gioi (shapely) DA o CRS cua luoi dich.
    Quy tac WorldCover (gia tri tai TAM pixel 30 m): include (GIU lop thuoc tap) HOAC exclude (BO lop thuoc tap);
    ca hai None -> v3 = include SCOPE_INCLUDE_LC (CHG-10). Chay lai v2: exclude=SCOPE_EXCLUDE_LC.
    Tra ve dict: scope (uint8 0/1), zenodo_n_years (uint8: so nam Salinity huu han), wc_class (uint8: lop WC tai
    tam pixel, 0 ngoai ranh gioi), in_boundary (bool), wc_ok (bool), rule (chuoi), include, exclude, transform,
    crs, shape. Moi file nhan phai cung luoi voi ref (khac -> loi).
    """
    include, exclude, rule_txt = wc_rule(include, exclude)
    import rasterio
    from rasterio import features
    from rasterio.windows import Window

    label_paths = list(label_paths)
    if not label_paths:
        raise ValueError("Khong co file nhan nao de dung dau chan.")
    ref_path = ref_path or label_paths[0]
    with rasterio.open(ref_path) as ref:
        transform, crs, shape = ref.transform, ref.crs, (ref.height, ref.width)
    for p in label_paths:
        with rasterio.open(p) as s:
            if (s.transform, s.crs, (s.height, s.width)) != (transform, crs, shape):
                raise ValueError(f"{p} khac luoi voi {ref_path} - can can luoi truoc.")
            band_index(s, label_band)

    in_b = features.rasterize([(boundary_geom, 1)], out_shape=shape, transform=transform, fill=0,
                              dtype="uint8", all_touched=False).astype(bool)
    n_years = np.zeros(shape, np.uint8)
    h, w = shape
    wc_cls = wc_at_centers(wc_path, transform, crs, shape, chunk_rows)
    wc_ok = np.isin(wc_cls, list(include)) if include is not None else ~np.isin(wc_cls, list(exclude))
    wc_cls[~in_b] = 0
    for p in label_paths:
        with rasterio.open(p) as s:
            b = band_index(s, label_band)
            for r0 in range(0, h, chunk_rows):
                n = min(chunk_rows, h - r0)
                n_years[r0:r0 + n] += np.isfinite(s.read(b, window=Window(0, r0, w, n))).astype(np.uint8)
        if log:
            log(f"  dau chan: {p}")
    scope = (in_b & wc_ok & (n_years >= 1)).astype(np.uint8)
    rule = f"v2 (tam pixel) ∩ {rule_txt} (gia tri tai tam pixel 30 m) ∩ Salinity huu han >= 1 nam"
    return {"scope": scope, "zenodo_n_years": n_years, "wc_class": wc_cls, "in_boundary": in_b, "wc_ok": wc_ok,
            "rule": rule, "include": include, "exclude": exclude, "transform": transform, "crs": crs,
            "shape": shape}


def write_scope_mask(path, res, with_wc_class=True) -> None:
    """Ghi GeoTIFF uint8 (scope, zenodo_n_years[, wc_class]), nen LZW, khong tag nodata (0 = ngoai pham vi).

    with_wc_class=False (hoac res khong co wc_class) -> chi 2 band nhu ban v2 cu. Thu tu band giu nguyen
    (band 1 scope, band 2 zenodo_n_years) de code doc theo so band van dung."""
    import rasterio

    h, w = res["shape"]
    names = SCOPE_BANDS_V3 if with_wc_class and "wc_class" in res else SCOPE_BANDS
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=len(names), dtype="uint8",
                       crs=res["crs"], transform=res["transform"], compress="lzw", tiled=True, blockxsize=512,
                       blockysize=512) as ds:
        for i, name in enumerate(names, start=1):
            ds.write(np.asarray(res[name], np.uint8), i)
            ds.set_band_description(i, name)
        tags = {"scope": "1 = " + res.get("rule", "v2 ∩ WorldCover khong {0,80,95} (tam pixel) ∩ Salinity huu han "
                                                  ">= 1 nam"),
                "zenodo_n_years": "so nam 2014-2023 Salinity huu han"}
        if "wc_class" in names:
            tags["wc_class"] = "lop WorldCover 2021 tai tam pixel 30 m trong ranh gioi v2 (0 = ngoai ranh gioi/khung)"
        ds.update_tags(**tags)


def compare_worldcover(new_path, old_path, boundary_geom, chunk_rows=2048):
    """So hai file WorldCover CUNG kich thuoc pixel, goc lech SO NGUYEN pixel, tren pixel co TAM trong ranh gioi.

    Duyet tren luoi file CU (ban moi phu rong hon). boundary_geom o CRS cua hai file.
    Tra ve dict: n_px_in_boundary, n_diff, diff_pairs {(cu, moi): so pixel}, offset (cot, hang cua goc cu tren
    luoi moi), n_outside_new (pixel trong ranh gioi nam ngoai khung moi - tinh vao n_diff voi moi = 255).
    Luoi khong khop nguyen pixel / khac CRS -> loi.
    """
    import rasterio
    from rasterio import features
    from rasterio.windows import Window

    with rasterio.open(old_path) as old, rasterio.open(new_path) as new:
        if old.crs != new.crs or old.transform.a != new.transform.a or old.transform.e != new.transform.e:
            raise ValueError("Hai file WorldCover khac CRS hoac kich thuoc pixel.")
        dc = (old.transform.c - new.transform.c) / new.transform.a
        dr = (old.transform.f - new.transform.f) / new.transform.e
        if abs(dc - round(dc)) > 1e-6 or abs(dr - round(dr)) > 1e-6:
            raise ValueError(f"Goc lech khong nguyen pixel ({dc}, {dr}).")
        dc, dr = int(round(dc)), int(round(dr))
        bo = band_index(old, "Map") if "Map" in old.descriptions else 1
        bn = band_index(new, "Map") if "Map" in new.descriptions else 1
        n_in = n_diff = n_out = 0
        pairs = {}
        for r0 in range(0, old.height, chunk_rows):
            n = min(chunk_rows, old.height - r0)
            win = Window(0, r0, old.width, n)
            inb = features.rasterize([(boundary_geom, 1)], out_shape=(n, old.width),
                                     transform=old.window_transform(win), fill=0, dtype="uint8").astype(bool)
            if not inb.any():
                continue
            a = old.read(bo, window=win).astype(np.int16)
            b = np.full(a.shape, 255, np.int16)          # ngoai khung moi = 255
            rr0 = r0 + dr
            rs, re_ = max(rr0, 0), min(rr0 + n, new.height)
            cs, ce = max(dc, 0), min(dc + old.width, new.width)
            if rs < re_ and cs < ce:
                b[rs - rr0:re_ - rr0, cs - dc:ce - dc] = new.read(bn, window=Window(cs, rs, ce - cs, re_ - rs))
            n_in += int(inb.sum())
            n_out += int((inb & (b == 255)).sum())
            d = inb & (a != b)
            n_diff += int(d.sum())
            if d.any():
                key = a[d].astype(np.int64) * 1000 + b[d]
                for k, c in zip(*np.unique(key, return_counts=True)):
                    pk = (int(k // 1000), int(k % 1000))
                    pairs[pk] = pairs.get(pk, 0) + int(c)
    return {"n_px_in_boundary": n_in, "n_diff": n_diff, "diff_pairs": pairs, "offset": (dc, dr),
            "n_outside_new": n_out}


def scope_values_at_points(points, scope_path):
    """Gia tri mat na pham vi tai PIXEL 30 m CHUA tung diem (points: GeoDataFrame, CRS bat ky).

    Tra ve DataFrame (cung index voi points): scope, zenodo_n_years, wc_class (neu raster co band nay, nguoc lai
    -1). Diem ngoai khung raster -> 0 (ngoai pham vi)."""
    import rasterio
    from eval_design import raster_values_at

    with rasterio.open(scope_path) as src:
        crs, names = src.crs, list(src.descriptions)
    p = points.to_crs(crs)
    x, y = p.geometry.x.to_numpy(), p.geometry.y.to_numpy()
    out = pd.DataFrame(index=points.index)
    out["scope"] = raster_values_at(scope_path, x, y, band=names.index("scope") + 1, fill=0)
    out["zenodo_n_years"] = raster_values_at(scope_path, x, y, band=names.index("zenodo_n_years") + 1, fill=0)
    out["wc_class"] = (raster_values_at(scope_path, x, y, band=names.index("wc_class") + 1, fill=0).astype(int)
                       if "wc_class" in names else -1)
    return out


def select_filtered_points(points, attrs, classes):
    """Diem bi LOC khoi bo chinh theo DUNG quy tac scripts/filter_eval_points_scope.py (giu <=> scope == 1) VA co
    wc_class (pixel 30 m chua diem) thuoc `classes` - tap cham phu cua bo nhan phu (CHG-15, cau V).

    points: GeoDataFrame nguon (truoc khi loc); attrs: scope_values_at_points(points, scope_v3) (CUNG index).
    Khong sinh diem, khong lay mau: tra ve cac dong cua `points` (giu cot, toa do, thu tu) + cot wc_class,
    zenodo_n_years. Loi: attrs khong cung index; thieu cot; lop yeu cau ma diem van scope == 1 (mat na khong loai
    lop do -> sai mat na); classes rong.
    """
    classes = sorted({int(c) for c in classes})
    if not classes:
        raise ValueError("classes rong")
    need = {"scope", "wc_class", "zenodo_n_years"}
    if not need <= set(attrs.columns):
        raise ValueError(f"attrs thieu cot {sorted(need - set(attrs.columns))}")
    if not attrs.index.equals(points.index):
        raise ValueError("attrs khong cung index voi points")
    in_cls = attrs["wc_class"].astype(int).isin(classes)
    kept = attrs["scope"] == 1
    if (in_cls & kept).any():
        raise ValueError(f"{int((in_cls & kept).sum())} diem lop {classes} van co scope = 1 - mat na khong loai lop "
                         "nay (sai mat na?)")
    sel = (in_cls & ~kept).to_numpy()
    out = points.loc[sel].copy()
    out["wc_class"] = attrs.loc[sel, "wc_class"].astype(int).to_numpy()
    out["zenodo_n_years"] = attrs.loc[sel, "zenodo_n_years"].astype(int).to_numpy()
    return out


def scope_change_by_class(old_scope, new_res):
    """So mat na pham vi cu (mang 0/1, cung luoi) voi ket qua build_scope_mask moi, theo lop wc_class moi.

    Tra ve DataFrame: wc_class, old_px, new_px, lost_px (cu 1 -> moi 0), gained_px (cu 0 -> moi 1)."""
    old = np.asarray(old_scope).astype(bool)
    new = np.asarray(new_res["scope"]).astype(bool)
    if old.shape != new.shape:
        raise ValueError(f"Hai mat na khac kich thuoc {old.shape} vs {new.shape}.")
    cls = np.asarray(new_res["wc_class"])
    rows = []
    for c in np.unique(cls[old | new]):
        m = cls == c
        rows.append({"wc_class": int(c), "old_px": int((old & m).sum()), "new_px": int((new & m).sum()),
                     "lost_px": int((old & ~new & m).sum()), "gained_px": int((~old & new & m).sum())})
    return pd.DataFrame(rows, columns=["wc_class", "old_px", "new_px", "lost_px", "gained_px"])


def scope_area_by_block(blocks_utm, scope_path, dist_coast_path=None, coast_km=20.0, chunk_rows=1024):
    """Dien tich (km2) mat na pham vi theo khoi, va phan ven bien (dist_coast_km <= coast_km).

    blocks_utm: GeoDataFrame (block_id, geometry) - nen la hinh khoi CHINH XAC (eval_design.exact_block_frame);
    pixel gan cho khoi chua TAM pixel. dist_coast lay gia tri pixel 90 m chua tam pixel 30 m (NaN -> khong
    tinh ven bien, dem rieng). Tra ve DataFrame: block_id, scope_km2, coast_scope_km2, coast_nan_km2.
    """
    import rasterio
    from rasterio import features
    from rasterio.windows import Window

    with rasterio.open(scope_path) as src:
        bl = blocks_utm.to_crs(src.crs)
        shapes = [(g, i + 1) for i, g in enumerate(bl.geometry)]
        nb = len(bl) + 1
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
        cnt, cst, cnan = (np.zeros(nb, np.int64) for _ in range(3))
        dc = rasterio.open(dist_coast_path) if dist_coast_path else None
        try:
            if dc is not None and dc.crs != src.crs:
                raise ValueError(f"{dist_coast_path} CRS khac {scope_path}.")
            for r0 in range(0, src.height, chunk_rows):
                n = min(chunk_rows, src.height - r0)
                win = Window(0, r0, src.width, n)
                sc = src.read(band_index(src, "scope") if "scope" in src.descriptions else 1, window=win) == 1
                if not sc.any():
                    continue
                lab = features.rasterize(shapes, out_shape=(n, src.width), transform=src.window_transform(win),
                                         fill=0, dtype="int32")
                keep = sc & (lab > 0)
                cnt += np.bincount(lab[keep], minlength=nb)
                if dc is not None:
                    d = nearest_on_grid(dc, "dist_coast_km", src.transform, (n, src.width), r0, fill=np.nan)
                    cst += np.bincount(lab[keep & (d <= coast_km)], minlength=nb)
                    cnan += np.bincount(lab[keep & ~np.isfinite(d)], minlength=nb)
        finally:
            if dc is not None:
                dc.close()
    out = pd.DataFrame({"block_id": bl["block_id"].astype(str).to_numpy(), "scope_km2": cnt[1:] * px_km2})
    if dist_coast_path:
        out["coast_scope_km2"] = cst[1:] * px_km2
        out["coast_nan_km2"] = cnan[1:] * px_km2
    return out

"""Thiet ke danh gia dung chung cho moi khung luoi (T3-V4).

- Khoi giu rieng: luoi khoi vuong neo o boi so canh (toa do UTM cua vung); chon
  ngau nhien (seed co dinh) cho den khi dat ty le DIEN TICH DAT muc tieu.
  Khoi dinh nghia tren toa do dia ly, doc lap voi khung luoi.
- Diem danh gia: gieo ngau nhien deu trong phan dat cua vung kiem tra.
- O huan luyen: moi o GIAO vung kiem tra (ke ca giao mot phan) bi loai.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box
from shapely.prepared import prep


def make_holdout_blocks(boundary: gpd.GeoDataFrame, block_km=50.0, holdout_frac=0.2, seed=42):
    """Tra ve GeoDataFrame (EPSG:4326): block_id, land_km2, is_holdout, geometry."""
    utm = boundary.estimate_utm_crs()
    land = boundary.to_crs(utm).geometry.union_all()
    size = block_km * 1000.0
    minx, miny, maxx, maxy = land.bounds

    blocks, ids = [], []
    for ix in range(int(np.floor(minx / size)), int(np.ceil(maxx / size))):
        for iy in range(int(np.floor(miny / size)), int(np.ceil(maxy / size))):
            b = box(ix * size, iy * size, (ix + 1) * size, (iy + 1) * size)
            if b.intersects(land):
                blocks.append(b)
                ids.append(f"blk_{ix}_{iy}")

    land_km2 = np.array([b.intersection(land).area / 1e6 for b in blocks])
    target = holdout_frac * land_km2.sum()
    order = np.random.default_rng(seed).permutation(len(blocks))
    is_holdout = np.zeros(len(blocks), dtype=bool)
    acc = 0.0
    for i in order:
        if acc >= target:
            break
        if land_km2[i] <= 0:
            continue
        is_holdout[i] = True
        acc += land_km2[i]

    gdf = gpd.GeoDataFrame(
        {"block_id": ids, "land_km2": np.round(land_km2, 3), "is_holdout": is_holdout},
        geometry=blocks, crs=utm,
    )
    return gdf.to_crs(4326)


def sample_eval_points(boundary: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, n_points=3000, seed=42):
    """Gieo n diem ngau nhien deu (theo dien tich) trong dat thuoc cac khoi giu rieng."""
    utm = boundary.estimate_utm_crs()
    land = boundary.to_crs(utm).geometry.union_all()
    holdout = blocks[blocks["is_holdout"]].to_crs(utm).geometry.union_all()
    region = land.intersection(holdout)
    if region.is_empty:
        raise ValueError("Vung kiem tra rong - kiem tra lai khoi giu rieng.")

    shapely.prepare(region)
    minx, miny, maxx, maxy = region.bounds
    rng = np.random.default_rng(seed)
    xs, ys = np.empty(0), np.empty(0)
    while xs.size < n_points:
        cx = rng.uniform(minx, maxx, 4 * n_points)
        cy = rng.uniform(miny, maxy, 4 * n_points)
        keep = shapely.contains_xy(region, cx, cy)
        xs, ys = np.concatenate([xs, cx[keep]]), np.concatenate([ys, cy[keep]])
    xs, ys = xs[:n_points], ys[:n_points]

    return gpd.GeoDataFrame(
        {"point_id": [f"pt_{i:05d}" for i in range(n_points)]},
        geometry=gpd.points_from_xy(xs, ys), crs=utm,
    ).to_crs(4326)


def training_mask(grid: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame):
    """True = o duoc phep dung de huan luyen (khong giao vung kiem tra nao)."""
    holdout = blocks[blocks["is_holdout"]].to_crs(grid.crs).geometry.union_all()
    touching = prep(holdout)
    return ~grid.geometry.apply(touching.intersects).to_numpy()


# -----------------------------------------------------------------------------------------------
# Thiet ke danh gia tren ranh gioi v2 (An duyet 2026-10-03): GIU khoi giu rieng, diem tren mat na dat.
# -----------------------------------------------------------------------------------------------
def refresh_blocks(blocks: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, block_km=50.0):
    """Giu nguyen block_id / is_holdout; tinh lai land_km2 = dien tich (khoi giao ranh gioi moi).

    Khoi cung luoi (neo boi so canh) ma ranh gioi moi cham toi nhung chua co trong file duoc THEM voi
    is_holdout = False (khong doi tap giu rieng). Tra ve (blocks_moi EPSG:4326, danh sach id them).
    """
    utm = boundary.estimate_utm_crs()
    land = boundary.to_crs(utm).geometry.union_all()
    old = blocks.to_crs(utm).copy()
    size = block_km * 1000.0
    # Dung lai hinh khoi CHINH XAC tu block_id (o vuong UTM neo boi so canh) thay vi chieu qua lai
    # 4326 -> UTM (troi vai nanomet moi lan chay -> file khong tai lap).
    ij = old["block_id"].str.extract(r"^blk_(-?\d+)_(-?\d+)$").astype(float)
    exact = [box(i * size, j * size, (i + 1) * size, (j + 1) * size) if not np.isnan(i) else g
             for i, j, g in zip(ij[0], ij[1], old.geometry)]
    old = old.set_geometry(exact)
    known = set(old["block_id"])
    minx, miny, maxx, maxy = land.bounds
    added = []
    for ix in range(int(np.floor(minx / size)), int(np.ceil(maxx / size))):
        for iy in range(int(np.floor(miny / size)), int(np.ceil(maxy / size))):
            bid = f"blk_{ix}_{iy}"
            b = box(ix * size, iy * size, (ix + 1) * size, (iy + 1) * size)
            # Chi them khoi GIAO that (> 1 m2): cham canh / sai so chieu qua lai khong tinh.
            if bid not in known and b.intersects(land) and b.intersection(land).area > 1.0:
                added.append({"block_id": bid, "land_km2": 0.0, "is_holdout": False, "geometry": b})
    if added:
        old = pd.concat([old, gpd.GeoDataFrame(added, geometry="geometry", crs=utm)], ignore_index=True)
    old["land_km2"] = np.round([g.intersection(land).area / 1e6 for g in old.geometry], 3)
    old["is_holdout"] = old["is_holdout"].astype(bool)
    return gpd.GeoDataFrame(old[["block_id", "land_km2", "is_holdout", "geometry"]], crs=utm).to_crs(4326), \
        [a["block_id"] for a in added]


def raster_values_at(path, xs, ys, band=1, fill=0, chunk_rows=2048):
    """Gia tri pixel chua toa do (xs, ys cung CRS voi raster), doc theo khoi hang; ngoai khung = fill."""
    import rasterio
    from rasterio.windows import Window

    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    with rasterio.open(path) as src:
        inv = ~src.transform
        cols = np.floor(inv.a * xs + inv.b * ys + inv.c).astype(np.int64)
        rows = np.floor(inv.d * xs + inv.e * ys + inv.f).astype(np.int64)
        ok = (cols >= 0) & (cols < src.width) & (rows >= 0) & (rows < src.height)
        out = np.full(xs.shape, fill, dtype=src.dtypes[band - 1])
        idx = np.flatnonzero(ok)
        for r0 in np.unique(rows[idx] // chunk_rows) * chunk_rows:
            sel = idx[(rows[idx] >= r0) & (rows[idx] < r0 + chunk_rows)]
            c0, c1 = cols[sel].min(), cols[sel].max()
            r1 = rows[sel].max()
            data = src.read(band, window=Window(int(c0), int(r0), int(c1 - c0 + 1), int(r1 - r0 + 1)))
            out[sel] = data[rows[sel] - r0, cols[sel] - c0]
    return out


def dist_coast_at_points(points: gpd.GeoDataFrame, dist_raster, band="dist_coast_km") -> np.ndarray:
    """Khoang cach bien (km) tai pixel CHUA diem (raster khoang cach 90 m, band theo TEN); ngoai khung -> NaN.
    Cung cach doc voi filter_eval_points_scope.point_attrs (bang diem bi loai theo dai ven bien)."""
    import rasterio

    from scope_mask import band_index

    with rasterio.open(dist_raster) as src:
        crs, b = src.crs, band_index(src, band)
    p = points.to_crs(crs)
    return raster_values_at(dist_raster, p.geometry.x.to_numpy(), p.geometry.y.to_numpy(), band=b,
                            fill=np.nan).astype(float)


def mask_area_by_block(blocks: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, lc_path, exclude=(0, 80, 95),
                       chunk_rows=1024):
    """Dien tich (km2) mat na dat theo khoi: trong ranh gioi va lop phu (WorldCover) KHONG thuoc exclude."""
    import rasterio
    from rasterio import features
    from rasterio.windows import Window

    with rasterio.open(lc_path) as src:
        crs = src.crs
        land = boundary.to_crs(crs).geometry.union_all()
        bl = blocks.to_crs(crs)
        shapes = [(g.intersection(land), i + 1) for i, g in enumerate(bl.geometry) if g.intersects(land)]
        counts = np.zeros(len(bl) + 1, np.int64)
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
        for r0 in range(0, src.height, chunk_rows):
            n = min(chunk_rows, src.height - r0)
            win = Window(0, r0, src.width, n)
            lab = features.rasterize(shapes, out_shape=(n, src.width), transform=src.window_transform(win),
                                     fill=0, dtype="int32")
            if not lab.any():
                continue
            lc = src.read(1, window=win)
            keep = (lab > 0) & ~np.isin(lc, list(exclude))
            counts += np.bincount(lab[keep], minlength=len(bl) + 1)
    return pd.Series(counts[1:] * px_km2, index=blocks["block_id"].to_numpy(), name="mask_km2")


def sample_points_in_mask(region_utm, utm_crs, n_points, seed, accept, batch=None):
    """Gieo n diem deu (theo dien tich) trong region_utm, chi giu diem accept(xs, ys) = True.

    Lay mau loai bo: rut ung vien deu trong khung bao, giu diem trong vung VA tren mat na -> mat do deu
    tren phan mat na. Tat dinh theo seed. Tra ve GeoDataFrame EPSG:4326 (chi geometry).
    """
    if region_utm.is_empty:
        raise ValueError("Vung gieo diem rong.")
    shapely.prepare(region_utm)
    minx, miny, maxx, maxy = region_utm.bounds
    rng = np.random.default_rng(seed)
    batch = batch or max(20000, 8 * n_points)
    xs, ys = np.empty(0), np.empty(0)
    for _ in range(200):
        if xs.size >= n_points:
            break
        cx, cy = rng.uniform(minx, maxx, batch), rng.uniform(miny, maxy, batch)
        keep = shapely.contains_xy(region_utm, cx, cy)
        cx, cy = cx[keep], cy[keep]
        keep = np.asarray(accept(cx, cy), bool)
        xs, ys = np.concatenate([xs, cx[keep]]), np.concatenate([ys, cy[keep]])
    if xs.size < n_points:
        raise RuntimeError(f"Chi gieo duoc {xs.size}/{n_points} diem - mat na qua nho?")
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs[:n_points], ys[:n_points]), crs=utm_crs).to_crs(4326)


# -----------------------------------------------------------------------------------------------
# CV khoi khong gian (An duyet 2026-10-03): 5 fold gan o CAP KHOI 50 km, can bang dien tich dat,
# seed co dinh, ghi MOT lan ra data/eval/cv_folds.csv va DUNG CHUNG cho ca 13 luoi.
# O cat ngang khoi: quy tac "cham" (nhat quan voi test khong gian).
# -----------------------------------------------------------------------------------------------
HOLDOUT_FOLD = -1


def exact_block_frame(blocks: gpd.GeoDataFrame, utm=None, block_km=50.0) -> gpd.GeoDataFrame:
    """Khoi trong UTM voi hinh CHINH XAC dung lai tu block_id (o vuong neo boi so canh).

    File khoi luu EPSG:4326 chi 4 dinh -> canh lech vai met so voi o vuong UTM (known-pitfalls 4k);
    gan khoi/cham khoi tren hinh nay. block_id khong theo mau blk_i_j -> giu hinh chieu sang UTM.
    """
    utm = utm or blocks.estimate_utm_crs()
    out = blocks.to_crs(utm).copy()
    size = block_km * 1000.0
    ij = out["block_id"].astype(str).str.extract(r"^blk_(-?\d+)_(-?\d+)$").astype(float)
    exact = [box(i * size, j * size, (i + 1) * size, (j + 1) * size) if not np.isnan(i) else g
             for i, j, g in zip(ij[0], ij[1], out.geometry)]
    return out.set_geometry(gpd.GeoSeries(exact, index=out.index, crs=utm))


def assign_cv_folds(blocks: gpd.GeoDataFrame, n_folds=5, seed=42, n_trials=5000) -> pd.DataFrame:
    """Gan fold CV o cap khoi: (block_id, cv_fold, land_km2); khoi giu rieng -> cv_fold = -1.

    Chi khoi KHONG giu rieng co dat (land_km2 > 0) duoc gan fold 0..n_folds-1; khoi khong dat ->
    -1 (cell_block_table bao loi neu o huan luyen roi vao do). Can bang dien tich dat: thu
    `n_trials` thu tu ngau nhien (rng theo seed), moi thu tu gan tham lam vao fold dang co it dat
    nhat; giu phuong an co (max - min) dien tich dat theo fold nho nhat (hoa -> phuong an gap truoc).
    Nhan fold danh lai theo block_id nho nhat trong fold. Tat dinh theo (blocks, n_folds, seed, n_trials).
    """
    df = pd.DataFrame({"block_id": blocks["block_id"].astype(str).to_numpy(),
                       "land_km2": blocks["land_km2"].astype(float).to_numpy(),
                       "is_holdout": blocks["is_holdout"].astype(bool).to_numpy()})
    if df["block_id"].duplicated().any():
        raise ValueError(f"block_id trung: {df.loc[df['block_id'].duplicated(), 'block_id'].tolist()}")
    elig = np.flatnonzero((~df["is_holdout"].to_numpy()) & (df["land_km2"].to_numpy() > 0))
    if len(elig) < n_folds:
        raise ValueError(f"Chi {len(elig)} khoi khong giu rieng co dat < n_folds = {n_folds}.")
    land = df["land_km2"].to_numpy()
    rng = np.random.default_rng(seed)
    best, best_range = None, np.inf
    for _ in range(int(n_trials)):
        order = rng.permutation(elig)
        tot = np.zeros(n_folds)
        fold = {}
        for i in order:
            k = int(np.argmin(tot))  # fold rong (0 km2) duoc lap truoc -> moi fold >= 1 khoi
            fold[int(i)] = k
            tot[k] += land[i]
        r = tot.max() - tot.min()
        if r < best_range - 1e-9:
            best, best_range = fold, r
    ids = df["block_id"].to_numpy()
    first = {k: min(ids[i] for i, kk in best.items() if kk == k) for k in range(n_folds)}
    remap = {old: new for new, old in enumerate(sorted(first, key=first.get))}
    cv = np.full(len(df), HOLDOUT_FOLD, dtype=int)
    for i, k in best.items():
        cv[i] = remap[k]
    return pd.DataFrame({"block_id": df["block_id"], "cv_fold": cv, "land_km2": land.round(3)})


def cell_block_table(grid: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, cv_folds: pd.DataFrame,
                     block_km=50.0) -> pd.DataFrame:
    """Bang o -> khoi/fold cho mot luoi (cv_folds dung chung moi luoi).

    Cot: cell_id, touches_holdout, block_id, block_assign, cv_fold, touched_folds.
    - touches_holdout = ~training_mask(grid, blocks) (cung ham voi test khong gian va check_split).
    - block_id: khoi chua TAM o (hinh khoi chinh xac trong UTM); tam ngoai moi khoi hoac nam dung
      tren canh -> khoi giao DIEN TICH LON NHAT (hoa -> block_id nho hon); block_assign ghi
      'centroid' / 'max_overlap'.
    - cv_fold: fold cua block_id (-1 neu khoi giu rieng).
    - touched_folds: cac fold CV (>= 0) ma o CHAM (giao, ke ca cham canh), chuoi tang dan vd "1|3".
    """
    cv = cv_folds[["block_id", "cv_fold"]].copy()
    cv["block_id"] = cv["block_id"].astype(str)
    missing = sorted(set(blocks["block_id"].astype(str)) - set(cv["block_id"]))
    if missing:
        raise ValueError(f"cv_folds thieu khoi: {missing[:10]} - file fold khong khop file khoi.")
    fold_of = cv.set_index("block_id")["cv_fold"].astype(int)
    if grid["cell_id"].astype(str).duplicated().any():
        raise ValueError("cell_id trung trong luoi.")

    exact = exact_block_frame(blocks, block_km=block_km)[["block_id", "is_holdout", "geometry"]]
    exact["block_id"] = exact["block_id"].astype(str)
    g = grid[["cell_id", "geometry"]].to_crs(exact.crs).reset_index(drop=True)
    g["cell_id"] = g["cell_id"].astype(str)

    touches = ~np.asarray(training_mask(grid, blocks), bool)

    cen = gpd.GeoDataFrame({"cell_id": g["cell_id"]}, geometry=g.geometry.centroid, crs=exact.crs)
    within = gpd.sjoin(cen, exact[["block_id", "geometry"]], predicate="within", how="inner")
    n_hit = within.groupby("cell_id")["block_id"].nunique()
    by_centroid = within[within["cell_id"].map(n_hit) == 1].drop_duplicates("cell_id") \
        .set_index("cell_id")["block_id"]

    inter = gpd.sjoin(g, exact[["block_id", "geometry"]], predicate="intersects", how="inner")
    no_block = sorted(set(g["cell_id"]) - set(inter["cell_id"]))
    if no_block:
        raise ValueError(f"{len(no_block)} o khong giao khoi nao (vd {no_block[:5]}) - file khoi thieu khoi?")

    block_id = g["cell_id"].map(by_centroid)
    assign = np.where(block_id.notna(), "centroid", "max_overlap")
    if block_id.isna().any():
        geo = exact.set_index("block_id").geometry
        cell_geo = g.set_index("cell_id").geometry
        sub = inter[inter["cell_id"].isin(set(g.loc[block_id.isna(), "cell_id"]))]
        areas = [cell_geo[c].intersection(geo[b]).area for c, b in zip(sub["cell_id"], sub["block_id"])]
        sub = pd.DataFrame({"cell_id": sub["cell_id"].to_numpy(), "block_id": sub["block_id"].to_numpy(),
                            "a": areas}).sort_values(["cell_id", "a", "block_id"], ascending=[True, False, True])
        block_id = block_id.fillna(g["cell_id"].map(sub.drop_duplicates("cell_id").set_index("cell_id")["block_id"]))

    inter = inter.assign(fold=inter["block_id"].map(fold_of).astype(int).to_numpy())
    tf = inter[inter["fold"] >= 0].groupby("cell_id")["fold"].apply(
        lambda s: "|".join(str(k) for k in sorted(set(s))))

    out = pd.DataFrame({
        "cell_id": g["cell_id"],
        "touches_holdout": touches,
        "block_id": block_id.to_numpy(),
        "block_assign": assign,
        "cv_fold": block_id.map(fold_of).astype(int).to_numpy(),
        "touched_folds": g["cell_id"].map(tf).fillna("").to_numpy(),
    })
    bad = out[(out["cv_fold"] == HOLDOUT_FOLD) & ~out["touches_holdout"]]
    if len(bad):
        hold = set(exact.loc[exact["is_holdout"].astype(bool), "block_id"])
        no_fold = bad[~bad["block_id"].isin(hold)]
        if len(no_fold):
            raise ValueError(f"{len(no_fold)} o huan luyen thuoc khoi khong co fold (khoi khong dat?): "
                             f"{no_fold['block_id'].unique().tolist()[:5]}")
        raise ValueError(f"{len(bad)} o co tam trong khoi giu rieng nhung training_mask khong coi la cham "
                         "(lech hinh khoi 4326/UTM?)")
    return out


def parse_touched_folds(values) -> list[set]:
    """Chuoi touched_folds ("1|3", "") -> danh sach tap so nguyen."""
    out = []
    for v in values:
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            out.append(set())
        elif isinstance(v, (int, float, np.integer, np.floating)):  # CSV doc khong dtype=str: "3" -> 3
            out.append({int(v)})
        else:
            out.append({int(x) for x in str(v).split("|") if x.strip() != ""})
    return out


def read_cell_block_table(path) -> pd.DataFrame:
    """Doc bang cell_block_table da ghi (giu cell_id, touched_folds la chuoi)."""
    df = pd.read_csv(path, dtype={"cell_id": str, "block_id": str, "touched_folds": str},
                     keep_default_na=False, na_values=[])
    df["touches_holdout"] = df["touches_holdout"].astype(str).str.lower().map({"true": True, "false": False})
    if df["touches_holdout"].isna().any():
        raise ValueError(f"{path}: cot touches_holdout co gia tri khong phai True/False.")
    df["cv_fold"] = df["cv_fold"].astype(int)
    return df


# -----------------------------------------------------------------------------------------------
# Don vi kiem dinh + fold CV cap don vi (An duyet J, I, D 2026-10-03; truoc lan CV dau tien).
# -----------------------------------------------------------------------------------------------
def _block_ij(block_id: str):
    m = pd.Series([block_id]).str.extract(r"^blk_(-?\d+)_(-?\d+)$").iloc[0]
    if m.isna().any():
        raise ValueError(f"block_id '{block_id}' khong theo mau blk_i_j - khong xac dinh duoc khoi ke.")
    return int(m[0]), int(m[1])


def merge_small_blocks(table: pd.DataFrame, min_points: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gop khoi CV it diem vao khoi KE -> don vi kiem dinh (unit_id). Lam TRUOC khi gan fold.

    Dau vao `table`: block_id (blk_i_j), is_holdout, land_km2, n_points (so diem CV trong khoi).
    Khoi CV = khong giu rieng va land_km2 > 0; khoi giu rieng / khoi khong dat KHONG bao gio gop
    (unit_id = block_id cua chinh no).
    Quy tac (lap den khi moi don vi >= min_points hoac chi con mot don vi):
      1. Chon don vi CV co it diem nhat (< min_points); hoa -> unit_id nho hon (thu tu chuoi).
      2. Don vi ke = don vi CV khac co it nhat mot khoi CHUNG CANH (lang gieng 4 huong |di| + |dj| = 1)
         voi mot khoi cua don vi dang xet. Cham goc (cheo) KHONG tinh la ke.
      3. Gop vao don vi ke co NHIEU DIEM NHAT; hoa -> unit_id nho hon. Don vi nhan GIU unit_id.
      4. Khong co don vi ke chung canh (khoi bi co lap) -> gop vao don vi CV GAN NHAT theo khoang
         cach Chebyshev nho nhat giua chi so khoi (roi nhieu diem nhat, roi unit_id nho hon); ghi
         rule = "gan_nhat" de bao cao.
    unit_id ban dau = block_id. Tra ve (bang khoi + cot unit_id, nhat ky gop: buoc, from_unit,
    from_blocks, from_points, into_unit, into_points_truoc, rule).
    """
    df = table.copy()
    df["block_id"] = df["block_id"].astype(str)
    if df["block_id"].duplicated().any():
        raise ValueError("block_id trung.")
    df["unit_id"] = df["block_id"]
    elig = (~df["is_holdout"].astype(bool)) & (df["land_km2"].astype(float) > 0)
    ij = {b: _block_ij(b) for b in df.loc[elig, "block_id"]}
    members = {b: {b} for b in ij}
    pts = dict(zip(df.loc[elig, "block_id"], df.loc[elig, "n_points"].astype(int)))
    log = []

    def edge_adjacent(u, v):
        return any(abs(ij[a][0] - ij[b][0]) + abs(ij[a][1] - ij[b][1]) == 1
                   for a in members[u] for b in members[v])

    def cheb(u, v):
        return min(max(abs(ij[a][0] - ij[b][0]), abs(ij[a][1] - ij[b][1])) for a in members[u] for b in members[v])

    while len(members) > 1:
        small = [u for u in members if pts[u] < min_points]
        if not small:
            break
        u = min(small, key=lambda x: (pts[x], x))
        others = [v for v in members if v != u]
        nbrs = [v for v in others if edge_adjacent(u, v)]
        rule = "chung_canh"
        if not nbrs:
            dmin = min(cheb(u, v) for v in others)
            nbrs = [v for v in others if cheb(u, v) == dmin]
            rule = "gan_nhat"
        v = min(nbrs, key=lambda x: (-pts[x], x))
        log.append({"buoc": len(log) + 1, "from_unit": u, "from_blocks": " ".join(sorted(members[u])),
                    "from_points": pts[u], "into_unit": v, "into_points_truoc": pts[v], "rule": rule})
        members[v] |= members.pop(u)
        pts[v] += pts.pop(u)
    unit_of = {b: u for u, bs in members.items() for b in bs}
    df.loc[elig, "unit_id"] = df.loc[elig, "block_id"].map(unit_of)
    return df, pd.DataFrame(log, columns=["buoc", "from_unit", "from_blocks", "from_points", "into_unit",
                                          "into_points_truoc", "rule"])


def fold_balance_objective(tot) -> float:
    """Tong chenh tuong doi (max - min) / trung binh cua tung tieu chi (tot: n_folds x n_tieu_chi)."""
    tot = np.asarray(tot, float)
    mean = tot.mean(axis=0)
    if not (np.isfinite(tot).all() and (mean > 0).all()):  # CHG-22: truoc day mean 0 -> 1, NaN -> muc tieu NaN "dat"
        raise ValueError(f"fold_balance_objective: tong theo fold khong huu han hoac trung binh <= 0 ({mean})")
    return float(((tot.max(axis=0) - tot.min(axis=0)) / mean).sum())


def _unit_matrix(units: pd.DataFrame, criteria, n_folds):
    u = units.reset_index(drop=True)
    if u["unit_id"].duplicated().any():
        raise ValueError("unit_id trung.")
    if len(u) < n_folds:
        raise ValueError(f"Chi {len(u)} don vi < n_folds = {n_folds}.")
    return u, u[list(criteria)].to_numpy(float)


def _unit_local_optima(X: np.ndarray, n_folds: int, rng, n_starts: int):
    """Sinh (assign, objective) cho tung diem xuat phat: hoan vi ngau nhien -> gan tham lam vao fold co tai
    chuan hoa (sum_c tot/mean) nho nhat (fold rong truoc) -> leo doi: lap ap dung buoc (chuyen 1 don vi hoac
    doi cho 2 don vi khac fold) giam muc tieu NHIEU NHAT, giu moi fold >= 1 don vi, dung khi khong con buoc giam.
    """
    n = len(X)
    mean_fold = X.sum(axis=0) / n_folds
    mean_fold[mean_fold == 0] = 1.0
    w = (X / mean_fold).sum(axis=1)  # tai chuan hoa moi don vi

    def totals(assign):
        t = np.zeros((n_folds, X.shape[1]))
        np.add.at(t, assign, X)
        return t

    for _ in range(int(n_starts)):
        assign = np.empty(n, int)
        load = np.zeros(n_folds)
        cnt = np.zeros(n_folds, int)
        for i in rng.permutation(n):
            empty = np.flatnonzero(cnt == 0)
            k = int(empty[0]) if len(empty) else int(np.argmin(load))
            assign[i] = k
            load[k] += w[i]
            cnt[k] += 1
        obj = fold_balance_objective(totals(assign))
        while True:
            cur_best, move = obj, None
            for i in range(n):
                ki = assign[i]
                if cnt[ki] > 1:
                    for k in range(n_folds):
                        if k == ki:
                            continue
                        assign[i] = k
                        o = fold_balance_objective(totals(assign))
                        assign[i] = ki
                        if o < cur_best - 1e-12:
                            cur_best, move = o, ("mv", i, k)
                for j in range(i + 1, n):
                    kj = assign[j]
                    if kj == ki:
                        continue
                    assign[i], assign[j] = kj, ki
                    o = fold_balance_objective(totals(assign))
                    assign[i], assign[j] = ki, kj
                    if o < cur_best - 1e-12:
                        cur_best, move = o, ("sw", i, j)
            if move is None:
                break
            if move[0] == "mv":
                _, i, k = move
                cnt[assign[i]] -= 1
                cnt[k] += 1
                assign[i] = k
            else:
                _, i, j = move
                assign[i], assign[j] = assign[j], assign[i]
            obj = cur_best
        yield assign.copy(), obj


def _canonical(assign: np.ndarray, ids: np.ndarray, n_folds: int) -> np.ndarray:
    """Danh lai nhan fold theo unit_id nho nhat trong fold (hai cach gan chi khac nhan -> cung mot mang)."""
    first = {k: min(ids[assign == k]) for k in range(n_folds)}
    remap = {old: new for new, old in enumerate(sorted(first, key=first.get))}
    return np.array([remap[int(k)] for k in assign])


def _unit_fold_frame(u, criteria, assign, obj):
    out = u[["unit_id", *criteria]].copy()
    out.insert(1, "cv_fold", assign)
    out.attrs["objective"] = obj
    return out


def assign_unit_folds(units: pd.DataFrame, criteria=("scope_km2", "coast_scope_km2"), n_folds=5, seed=42,
                      n_starts=200) -> pd.DataFrame:
    """Gan fold cho DON VI kiem dinh, can bang dong thoi cac cot `criteria` (vd dien tich mat na pham vi
    va dien tich mat na pham vi ven bien) - phuong an TOT NHAT tim duoc.

    Muc tieu: tong chenh tuong doi sum_c (max_k - min_k) / mean_k cua tung tieu chi (fold_balance_objective).
    n_starts diem xuat phat (rng(seed)) + leo doi (_unit_local_optima); giu phuong an tot nhat (hoa -> gap
    truoc). Nhan fold danh lai theo unit_id nho nhat trong fold. Tra ve: unit_id, cv_fold, + cot criteria;
    attrs["objective"]. Luu y: n_starts du lon -> seed khac thuong ra CUNG toi uu -> khong dung seed ham nay
    de tao cac phuong an khac nhau (dung fold_schemes).
    """
    u, X = _unit_matrix(units, criteria, n_folds)
    best, best_obj = None, np.inf
    for assign, obj in _unit_local_optima(X, n_folds, np.random.default_rng(seed), n_starts):
        if obj < best_obj - 1e-12:
            best, best_obj = assign, obj
    ids = u["unit_id"].astype(str).to_numpy()
    return _unit_fold_frame(u, criteria, _canonical(best, ids, n_folds), best_obj)


def fold_overlap(a, b) -> float:
    """Do trung hai cach gan fold tren CUNG danh sach don vi: ty le don vi GIU fold sau khi ghep nhan fold toi uu
    (Hungarian, toi da so don vi trung). 1 = giong het (chi khac nhan); 0,6 = 60% don vi giu fold."""
    from scipy.optimize import linear_sum_assignment

    a, b = np.asarray(a, int), np.asarray(b, int)
    if a.shape != b.shape:
        raise ValueError("Hai cach gan fold khac so don vi.")
    k = int(max(a.max(), b.max())) + 1
    m = np.zeros((k, k))
    np.add.at(m, (a, b), 1)
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / len(a))


def fold_pair_stats(a, b) -> dict:
    """Do trung (Hungarian), Rand index, ARI giua hai cach gan fold (Rand/ARI chi de tham khao)."""
    from sklearn.metrics import adjusted_rand_score, rand_score

    return {"overlap": fold_overlap(a, b), "rand": float(rand_score(a, b)), "ari": float(adjusted_rand_score(a, b))}


def fold_max_rel_dev(assign, values, n_folds) -> float:
    """max_k |tong_k / trung binh cac fold - 1| cua mot cot gia tri theo don vi."""
    t = np.zeros(n_folds)
    np.add.at(t, np.asarray(assign, int), np.asarray(values, float))
    if not (np.isfinite(t).all() and t.mean() > 0):  # CHG-22: truoc day mean 0/NaN -> 0 (= "dat" dieu kien ven bien)
        raise ValueError(f"fold_max_rel_dev: tong theo fold khong huu han hoac trung binh <= 0 ({t})")
    return float(np.abs(t / t.mean() - 1).max())


def unit_pool(units: pd.DataFrame, criteria=("scope_km2", "coast_scope_km2"), n_folds=5, n_starts=3000,
              pool_seed=0) -> dict:
    """Tap ung vien = toi uu dia phuong PHAN BIET (nhan fold danh lai theo unit_id, thu tu dong cua `units`)
    tu n_starts diem xuat phat (rng(pool_seed)). Tra ve {tuple fold theo dong units: [muc tieu, so lan gap]}."""
    u, X = _unit_matrix(units, criteria, n_folds)
    ids = u["unit_id"].astype(str).to_numpy()
    pool = {}
    for assign, obj in _unit_local_optima(X, n_folds, np.random.default_rng(pool_seed), n_starts):
        key = tuple(_canonical(assign, ids, n_folds))
        if key in pool:
            pool[key][1] += 1
        else:
            pool[key] = [obj, 1]
    return pool


# Pha hoa (CHG-14, An chot 2026-10-04) - dung o MOI cho xep hang phuong an.
TIE_ROUND = 9
TIE_BREAK_RULE = ("muc tieu lam tron 1e-9 -> lech ven bien max_k |tong_k/TB - 1| nho hon (lam tron 1e-9) -> "
                  "ma phan chia nho hon (tuple nhan fold theo unit_id sap tang; nhan danh lai theo unit_id nho "
                  "nhat trong fold). To hop phuong an phu: tong muc tieu (moi muc tieu lam tron 1e-9) -> max lech "
                  "ven bien trong to hop -> danh sach ma (sap tang) nho hon")


def scheme_code(assign, ids) -> tuple:
    """Ma phan chia ON DINH: nhan fold (danh lai theo unit_id nho nhat trong fold) xep theo unit_id SAP TANG.
    Khong phu thuoc thu tu dong cua bang don vi hay nhan fold goc."""
    ids = np.asarray(ids).astype(str)
    a = np.asarray(assign, int)
    n_folds = int(a.max()) + 1
    canon = _canonical(a, ids, n_folds)
    return tuple(int(x) for x in canon[np.argsort(ids, kind="stable")])


def scheme_code_str(code) -> str:
    """Ma phan chia dang chuoi ('0-0-1-...') de ghi bang/provenance."""
    return "-".join(str(int(x)) for x in code)


def scheme_rank_key(objective, coast_dev, code) -> tuple:
    """Khoa xep hang (nho = tot): (muc tieu lam tron 1e-9, lech ven bien lam tron 1e-9, ma phan chia)."""
    return (round(float(objective), TIE_ROUND), round(float(coast_dev), TIE_ROUND), tuple(int(x) for x in code))


def rank_pool(units: pd.DataFrame, pool: dict, criteria=("scope_km2", "coast_scope_km2"), n_folds=5,
              coast_col="coast_scope_km2") -> tuple[list, dict, dict, dict]:
    """Xep hang tap ung vien theo scheme_rank_key (pha hoa CHG-14).
    pool: {tuple fold theo dong `units`: [muc tieu, so lan gap]}. Tra ve (keys da xep, lech ven bien, ma, khoa)."""
    u, _ = _unit_matrix(units, criteria, n_folds)
    ids = u["unit_id"].astype(str).to_numpy()
    coast = u[coast_col].to_numpy(float)
    dev = {k: fold_max_rel_dev(k, coast, n_folds) for k in pool}
    code = {k: scheme_code(k, ids) for k in pool}
    rkey = {k: scheme_rank_key(pool[k][0], dev[k], code[k]) for k in pool}
    return sorted(pool, key=rkey.get), dev, code, rkey


def pick_alternates(cands, rkey: dict, overlap, n_alt: int, max_overlap: float):
    """Chon to hop n_alt phuong an phu (CHG-14) trong `cands` (da loc dieu kien don le) sao cho do trung giua
    MOI cap <= max_overlap; trong cac to hop hop le chon TONG muc tieu nho nhat (muc tieu lam tron 1e-9),
    hoa -> max lech ven bien trong to hop nho hon -> danh sach ma phan chia (sap tang) nho hon.
    rkey: {cand: scheme_rank_key}; overlap(a, b) -> do trung. Tra ve (danh sach xep theo rkey, so to hop hop le)
    hoac (None, 0)."""
    from itertools import combinations

    cache = {}

    def ov(a, b):
        k = (a, b) if rkey[a] <= rkey[b] else (b, a)
        if k not in cache:
            cache[k] = overlap(*k)
        return cache[k]

    best, best_key, n_ok = None, None, 0
    for combo in combinations(cands, n_alt):
        if not all(ov(a, b) <= max_overlap + 1e-12 for a, b in combinations(combo, 2)):
            continue
        n_ok += 1
        key = (round(sum(rkey[c][0] for c in combo), TIE_ROUND), max((rkey[c][1] for c in combo), default=0.0),
               tuple(sorted(rkey[c][2] for c in combo)))
        if best_key is None or key < best_key:
            best, best_key = combo, key
    if best is None:
        return None, 0
    return sorted(best, key=rkey.get), n_ok


def scheme_conditions(units: pd.DataFrame, schemes: dict, best_objective: float, criteria=("scope_km2",
                      "coast_scope_km2"), n_folds=5, min_points=30, tol=0.10, coast_tol=0.15, max_overlap=0.60,
                      coast_col="coast_scope_km2", points_col="n_points", max_objective=None) -> pd.DataFrame:
    """Kiem lai dieu kien D/I/J cho cac phuong an fold DA CHON tren bang don vi MOI (vd sau khi doi mat na).

    units: unit_id, criteria, points_col. schemes: {seed: Series unit_id -> cv_fold}; seed dau = phuong an chinh.
    best_objective: muc tieu tot nhat tinh lai tren bang moi (tu unit_pool).
    Moi phuong an mot dong: objective, ratio_vs_best, chenh tuong doi tung tieu chi, lech ven bien max/fold,
    do trung voi tung phuong an khac; cot dieu kien (True = dat):
      ok_J_min_points (moi don vi >= min_points), ok_I_coast (lech ven bien moi fold <= coast_tol),
      ok_D_ratio (quy tac CU: phuong an phu muc tieu <= (1 + tol) x tot nhat; phuong an chinh luon True, xem
      main_is_best), ok_D_objective (= ok_D_ratio khi max_objective None; CHG-14: MOI phuong an co muc tieu lam
      tron 1e-9 <= max_objective), ok_D_overlap (do trung <= max_overlap voi moi phuong an khac),
      ok_all (dung ok_D_objective).
    """
    u, X = _unit_matrix(units, criteria, n_folds)
    ids = u["unit_id"].astype(str)
    seeds = list(schemes)
    assign = {}
    for s in seeds:
        f = pd.Series(schemes[s]).astype(int)
        f.index = f.index.astype(str)
        if set(f.index) != set(ids):
            raise ValueError(f"Phuong an s{s}: tap don vi khac bang don vi (thieu {sorted(set(ids) - set(f.index))}, "
                             f"thua {sorted(set(f.index) - set(ids))}).")
        assign[s] = f.loc[ids].to_numpy()
    min_pts = int(u[points_col].min())
    rows = []
    for i, s in enumerate(seeds):
        a = assign[s]
        tot = np.zeros((n_folds, X.shape[1]))
        np.add.at(tot, a, X)
        obj = fold_balance_objective(tot)
        r = {"seed": s, "vai_tro": "chinh" if i == 0 else "phu", "objective": obj,
             "best_objective": best_objective, "ratio_vs_best": obj / best_objective,
             "main_is_best": bool(round(obj, TIE_ROUND) <= round(best_objective, TIE_ROUND)) if i == 0 else None,
             "min_unit_points": min_pts}
        for j, c in enumerate(criteria):
            r[f"{c}_rel_range"] = float((tot[:, j].max() - tot[:, j].min()) / tot[:, j].mean())
        r["coast_max_dev"] = fold_max_rel_dev(a, u[coast_col].to_numpy(float), n_folds)
        ov = {s2: fold_overlap(a, assign[s2]) for s2 in seeds if s2 != s}
        for s2, v in ov.items():
            r[f"overlap_vs_s{s2}"] = v
        r["ok_J_min_points"] = min_pts >= min_points
        r["ok_I_coast"] = r["coast_max_dev"] <= coast_tol + 1e-12
        r["ok_D_ratio"] = True if i == 0 else bool(obj <= best_objective * (1 + tol) + 1e-12)
        if max_objective is None:
            r["D_rule"] = f"tuong_doi <= {1 + tol:.2f} x tot nhat"
            r["ok_D_objective"] = r["ok_D_ratio"]
        else:
            r["D_rule"] = f"tuyet_doi <= {max_objective}"
            r["ok_D_objective"] = bool(round(obj, TIE_ROUND) <= max_objective + 1e-12)
        r["ok_D_overlap"] = all(v <= max_overlap + 1e-12 for v in ov.values())
        r["ok_all"] = r["ok_J_min_points"] and r["ok_I_coast"] and r["ok_D_objective"] and r["ok_D_overlap"]
        rows.append(r)
    return pd.DataFrame(rows)


def fold_schemes(units: pd.DataFrame, seeds, criteria=("scope_km2", "coast_scope_km2"), n_folds=5, n_starts=3000,
                 pool_seed=0, tols=(0.10, 0.15), max_overlap=0.60, coast_col="coast_scope_km2",
                 coast_tol=0.15, pool=None, max_objective=None) -> tuple[dict, pd.DataFrame, dict]:
    """Phuong an chinh + phuong an phu gan fold, dung chung 13 luoi.
    pool: tap ung vien da tinh san (unit_pool cung units/tham so) - None -> tinh lai.

    1. Tap ung vien = toi uu dia phuong PHAN BIET (sau khi danh lai nhan) tu n_starts diem xuat phat
       (rng(pool_seed)) cua _unit_local_optima. Xep hang theo scheme_rank_key (pha hoa CHG-14: muc tieu lam
       tron 1e-9 -> lech ven bien nho hon -> ma phan chia nho hon).
    2. seeds[0] = phuong an CHINH = hang 1 (muc tieu thap nhat; ket qua chinh).
    3. Rang buoc cho MOI phuong an (ca chinh): dien tich `coast_col` moi fold trong +-coast_tol quanh trung
       binh cac fold (phuong an chinh vi pham -> ValueError, khong tu noi).
    4. Phuong an phu (seeds[1:], phan tich do nhay); do trung (fold_overlap) <= max_overlap voi phuong an chinh
       VA giua moi cap phuong an phu (khong noi).
       - max_objective (CHG-14, MAC DINH duong chinh): muc tieu lam tron 1e-9 <= max_objective (nguong TUYET
         DOI; phuong an chinh vuot -> ValueError); trong cac to hop hop le chon TONG muc tieu nho nhat
         (pick_alternates; hoa -> lech ven bien -> ma).
       - max_objective None (quy tac CU CHG-06, giu de tai lap): muc tieu <= (1 + tol) x tot nhat, tol lan luot
         trong `tols`, dung tol DAU TIEN co du phuong an; chon to hop thu tu tu dien theo hang.
       Seed chi la NHAN phuong an (lua chon tat dinh, khong rut ngau nhien); trong to hop, hang nho -> seed truoc.
    Khong du phuong an -> ValueError. Tra ve ({seed: DataFrame unit_id, cv_fold, criteria (attrs objective, rank,
    code)}, bang tap ung vien, thong tin {mode, max_objective, tol_used, best_objective, n_pool, n_eligible,
    n_valid_combos, tie_break}).
    """
    if max_objective is None and not tols:
        raise ValueError("Can max_objective (CHG-14) hoac tols (quy tac cu) de chon phuong an phu.")
    u, X = _unit_matrix(units, criteria, n_folds)
    pool = unit_pool(units, criteria, n_folds, n_starts, pool_seed) if pool is None else pool
    keys, dev, code, rkey = rank_pool(u, pool, criteria, n_folds, coast_col)
    best = pool[keys[0]][0]
    main = keys[0]
    if dev[main] > coast_tol + 1e-12:
        raise ValueError(f"Phuong an chinh vi pham ven bien: lech {dev[main]:.3f} > {coast_tol}.")
    if max_objective is not None and rkey[main][0] > max_objective + 1e-12:
        raise ValueError(f"Phuong an chinh (tot nhat) co muc tieu {best:.6f} > nguong tuyet doi {max_objective}.")
    ov_main = {k: fold_overlap(k, main) for k in keys}
    n_alt = len(seeds) - 1
    picked_keys, tol_used, n_elig, n_combos = None, None, 0, 0
    if max_objective is not None:
        mode = "tuyet_doi"
        elig = [k for k in keys[1:] if rkey[k][0] <= max_objective + 1e-12 and dev[k] <= coast_tol + 1e-12
                and ov_main[k] <= max_overlap + 1e-12]
        picked_keys, n_combos = pick_alternates(elig, rkey, fold_overlap, n_alt, max_overlap)
        n_elig = len(elig)
        if picked_keys is None:
            raise ValueError(f"Khong du {n_alt} phuong an phu thoa muc tieu <= {max_objective} (tuyet doi), ven bien "
                             f"+-{coast_tol}, do trung <= {max_overlap} ({n_elig} ung vien don le hop le / tap "
                             f"{len(keys)}).")
    else:
        mode = "tuong_doi"

        def search(cands, chosen):
            if len(chosen) == n_alt:
                return chosen
            for i, k in enumerate(cands):
                if all(fold_overlap(k, c) <= max_overlap + 1e-12 for c in chosen):
                    r = search(cands[i + 1:], chosen + [k])
                    if r:
                        return r
            return None

        for tol in tols:
            elig = [k for k in keys[1:] if pool[k][0] <= best * (1 + tol) + 1e-12 and dev[k] <= coast_tol + 1e-12
                    and ov_main[k] <= max_overlap + 1e-12]
            picked_keys = search(elig, [])
            if picked_keys is not None:
                tol_used, n_elig = tol, len(elig)
                break
        if picked_keys is None:
            raise ValueError(f"Khong du {n_alt} phuong an phu thoa muc tieu <= (1 + {max(tols)}) x tot nhat, ven "
                             f"bien +-{coast_tol}, do trung <= {max_overlap} (tap {len(keys)} ung vien).")
    out = {}
    for s, k in zip(seeds, [main] + list(picked_keys)):
        f = _unit_fold_frame(u, criteria, np.array(k), pool[k][0])
        f.attrs["rank"] = keys.index(k) + 1
        f.attrs["code"] = scheme_code_str(code[k])
        f.attrs["coast_max_dev"] = dev[k]
        out[s] = f
    table = pd.DataFrame({"rank": range(1, len(keys) + 1), "objective": [pool[k][0] for k in keys],
                          "objective_round": [rkey[k][0] for k in keys],
                          "n_hits": [pool[k][1] for k in keys], "coast_max_dev": [dev[k] for k in keys],
                          "overlap_vs_main": [ov_main[k] for k in keys],
                          "ma_phan_chia": [scheme_code_str(code[k]) for k in keys]})
    if max_objective is not None:
        table[f"obj_le_{max_objective}"] = table["objective_round"] <= max_objective + 1e-12
    for t in (tols or ()):
        table[f"obj_le_{1 + t:.2f}x"] = table["objective"] <= best * (1 + t) + 1e-12
    table["chon"] = ""
    for s, k in zip(seeds, [main] + list(picked_keys)):
        table.loc[keys.index(k), "chon"] = f"s{s}"
    info = {"mode": mode, "max_objective": max_objective, "tol_used": tol_used, "best_objective": best,
            "n_pool": len(keys), "n_eligible": n_elig, "n_valid_combos": n_combos, "tie_break": TIE_BREAK_RULE}
    return out, table, info


def unit_folds_to_blocks(blocks_units: pd.DataFrame, unit_folds: pd.DataFrame) -> pd.DataFrame:
    """Ghep fold don vi -> bang khoi (them cot cv_fold); khoi khong thuoc don vi CV (giu rieng / khong dat) -> -1."""
    fold_of = unit_folds.set_index("unit_id")["cv_fold"].astype(int)
    out = blocks_units.copy()
    out["cv_fold"] = out["unit_id"].map(fold_of).fillna(HOLDOUT_FOLD).astype(int)
    hold = out["is_holdout"].astype(bool)
    if (out.loc[hold, "cv_fold"] != HOLDOUT_FOLD).any():
        raise ValueError("Khoi giu rieng bi gan fold CV.")
    return out

"""Do thi song chinh va khoang cach DOC SONG toi cua song (tuan 3 viec 2, CHG-16 - An chot 2026-10-04).

Quy tac (ghi truoc khi tinh dac trung, giong nhau cho ca 13 luoi - luoi chi khac o buoc gop pixel -> o):
  - Mang: OSM `fclass == river` co ten khop MAIN_RIVERS (da duyet) + GRAPH_EXTRA (Vam Nao noi Tien-Hau,
    Soai Rap la cua Vam Co, Bay Hap, My Thanh, Dam Doi, Trem, Cai Tau - song tu nhien ban dao Ca Mau; KHONG
    Nhu Gia: OSM khong co duong ra bien). KHONG kenh (kenh co cong ngan man -> duong tat khong that).
    Ten OSM chuan hoa NFC truoc khi so.
  - Dinh: duong OSM duoc noi tai giao diem (shapely union_all) roi chia deu moi doan <= SPACING_M; trong so
    canh = chieu dai doc duong.
  - Cua song = DAU MUT CUA MOT SONG (dinh ma song do chi co 1 canh) cach duong bo <= MOUTH_TOL_M - khong phai
    moi diem song trong khoang do (song chay song song bo se sinh ca dai "cua"). Xet theo tung song chu khong
    theo bac dinh: Dam Doi va Cua Lon cung ket thuc tai cua Bo De -> bac 2 nhung van la cua. Ngoai le thu
    cong MOUTH_TOL_EXCEPTIONS (Bay Hap: OSM dung cach bo 10,1 km, song co cua that). Diem goc mang them doan
    thang dau mut -> bo. Loai song trong EXCLUDED_MOUTH_RIVERS (Ba Lai: cong dap 2002).
  - Noi cho ho: chi tu dau mut bac 1 KHONG phai cua song, toi dinh gan nhat cua MANH KHAC trong <= GAP_TOL_M
    (xet manh truoc khi noi, mot luot). Them MANUAL_JOINS: lo du lieu OSM tren CUNG mot song. Danh sach cho
    noi duoc in de An duyet.
  - Cong (SLUICES): cat song tai dinh cua song do gan toa do cong nhat (bo moi canh cua dinh do). Dinh "sau
    cong" = ra bien duoc khi thong nhung KHONG ra duoc khi cat. Khoang cach + zos LUON theo ban thong (neu cat
    thi phan thuong luu khong con duong ra bien -> pixel bi gan sang song khac, sai chieu: Vi Thanh 79,5 ->
    13,3 km); tac dong cong vao dac trung rieng `sluice_frac` (= 0 truoc SLUICE_FIRST_SEASON; do nhay
    SLUICE_ALT_SEASON).
  - Khoang cach doc song cua mot dinh = min tren cac cua (doan thang cua -> bo + duong ngan nhat tren do
    thi). Pixel nhan gia tri cua DINH GAN NHAT ra bien duoc (ban thong) - khong cong khoang cach ngang;
    khoang cach ngang la cot rieng `graph_lateral_km`, chi de kiem/phan tang.
  - Bat bien: dist[v] >= dist_coast(cua duoc gan) + khoang cach thang (v, dinh cua)  (check_invariant).
"""
import re
import unicodedata

import numpy as np
import pandas as pd

GRAPH_EXTRA = ("Vàm Nao", "Soài Rạp", "Bảy Háp", "Mỹ Thanh", "Đầm Dơi", "Trẹm", "Cái Tàu")
SPACING_M = 90.0
GAP_TOL_M = 1000.0
MOUTH_TOL_M = 5000.0
EXCLUDED_MOUTH_RIVERS = ("Ba Lai",)
# Ngoai le thu cong (An duyet 2026-10-04): {song: nguong cach bo (m)} cho dau mut cua song do.
MOUTH_TOL_EXCEPTIONS = {"Bảy Háp": 12_000.0}
# Lo du lieu OSM tren cung mot song: (song, (lon, lat) gan dau mut can noi) -> noi toi dinh gan nhat cua CUNG
# song o manh khac. Cai Lon: manh thuong luu 16,8 km (Vi Thanh - Long My) ho 5,28 km voi than song.
MANUAL_JOINS = (("Cái Lớn", (105.295, 9.637)),)
# OSM way 1009147705 / 1009147704 (landuse=construction, "Cống Cái Lớn" / "Cống Cái Bé"). (lon, lat) EPSG:4326.
SLUICES = {"Cái Lớn": (105.1316723, 9.8396875), "Cái Bé": (105.1423185, 9.8505499)}
SLUICE_FIRST_SEASON = 2022  # ban giao van hanh 20/01/2022 (vi.wikipedia "Cong trinh thuy loi Cai Lon - Cai Be")
SLUICE_ALT_SEASON = 2021    # do nhay: mot hang muc van hanh tu dau 2021


def nfc(s) -> str:
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else ""


def graph_pattern(main_pattern: str, extra=GRAPH_EXTRA) -> re.Pattern:
    """Regex ten song cua do thi = MAIN_RIVERS (chuoi regex) them cac song trong `extra`."""
    alts = "|".join(re.escape(nfc(e)) for e in extra)
    return re.compile(f"(?:{nfc(main_pattern)})|(?:^Sông (?:{alts})\\b)")


def river_key(name: str) -> str:
    """'Sông Cái Lớn' -> 'Cái Lớn' (bo tien to 'Sông ', phan sau '/' hoac '(')."""
    s = nfc(name)
    s = re.split(r"[/(]", s)[0].strip()
    return s[5:].strip() if s.startswith("Sông ") else s


def select_graph_lines(osm, pattern):
    """Duong OSM thuoc do thi: fclass river, ten (NFC) khop `pattern`. Them cot `river` (khoa ten)."""
    names = osm["name"].map(nfc)
    sel = osm[(osm["fclass"] == "river") & names.map(lambda s: bool(pattern.match(s)))].copy()
    sel["river"] = sel["name"].map(river_key)
    return sel


class RiverGraph:
    """Do thi dinh chia deu. xy (N, 2) m; river (N,); canh (ei, ej, ew, er = song cua canh); degree theo canh."""

    def __init__(self, xy, river, ei, ej, ew, er=None):
        self.xy = np.asarray(xy, float)
        self.river = np.asarray(river, object)
        self.ei, self.ej, self.ew = (np.asarray(a) for a in (ei, ej, ew))
        self.er = np.asarray(er if er is not None else self.river[self.ei], object)
        self.degree = np.bincount(np.concatenate([self.ei, self.ej]), minlength=len(self.xy))

    @property
    def n(self):
        return len(self.xy)

    def matrix(self, extra=None, drop_vertices=()):
        """Ma tran ke thua (n+1 x n+1 neu extra co dinh n = nguon ao). extra: (i, j, w) them vao."""
        from scipy.sparse import coo_matrix

        ei, ej, ew = self.ei, self.ej, self.ew
        if len(drop_vertices):
            keep = ~(np.isin(ei, drop_vertices) | np.isin(ej, drop_vertices))
            ei, ej, ew = ei[keep], ej[keep], ew[keep]
        size = self.n
        if extra is not None:
            xi, xj, xw = extra
            ei, ej, ew = np.concatenate([ei, xi]), np.concatenate([ej, xj]), np.concatenate([ew, xw])
            size = max(size, int(max(xi.max(initial=0), xj.max(initial=0))) + 1)
        return coo_matrix((ew, (ei, ej)), shape=(size, size)).tocsr()

    def components(self, extra=None, drop_vertices=()):
        from scipy.sparse.csgraph import connected_components

        m = self.matrix(extra, drop_vertices)[: self.n, : self.n]
        return connected_components(m, directed=False)[1]

    def add_edges(self, i, j, w, label="~noi"):
        return RiverGraph(self.xy, self.river, np.concatenate([self.ei, i]), np.concatenate([self.ej, j]),
                          np.concatenate([self.ew, w]), np.concatenate([self.er, np.full(len(i), label, object)]))

    def river_endpoints(self):
        """{dinh: [song ...]}: dinh la dau mut cua song do (song chi co dung 1 canh tai dinh; bo canh noi '~')."""
        cnt = {}
        for a, b, r in zip(self.ei.tolist(), self.ej.tolist(), self.er.tolist()):
            if str(r).startswith("~"):
                continue
            for v in (a, b):
                cnt[(v, r)] = cnt.get((v, r), 0) + 1
        out = {}
        for (v, r), c in cnt.items():
            if c == 1:
                out.setdefault(v, []).append(r)
        return out


def build_graph(lines, spacing=SPACING_M):
    """Noi duong tai giao diem, chia deu moi doan <= spacing m. lines: GeoDataFrame (CRS met) co cot river."""
    import geopandas as gpd
    import shapely

    noded = shapely.get_parts(shapely.union_all(lines.geometry.values))
    noded = [g for g in noded if g.length > 0]
    parts = gpd.GeoDataFrame({"geometry": noded}, crs=lines.crs)
    mids = gpd.GeoDataFrame(geometry=[g.interpolate(0.5, normalized=True) for g in noded], crs=lines.crs)
    near = gpd.sjoin_nearest(mids, lines[["river", "geometry"]], how="left")
    near = near[~near.index.duplicated()]
    parts["river"] = near["river"].values

    xy, river, ei, ej, ew, er = [], [], [], [], [], []
    end_id = {}

    def vid(pt, rv, endpoint):
        if endpoint:
            key = (round(pt[0], 2), round(pt[1], 2))
            if key in end_id:
                return end_id[key]
            end_id[key] = len(xy)
        xy.append(pt)
        river.append(rv)
        return len(xy) - 1

    for geom, rv in zip(parts.geometry, parts["river"]):
        L = geom.length
        k = max(1, int(np.ceil(L / spacing)))
        ts = np.linspace(0, L, k + 1)
        ids = []
        for t_i, t in enumerate(ts):
            p = geom.interpolate(t)
            ids.append(vid((p.x, p.y), rv, t_i in (0, k)))
        ei += ids[:-1]
        ej += ids[1:]
        ew += [L / k] * k
        er += [rv] * k
    return RiverGraph(np.array(xy), river, np.array(ei), np.array(ej), np.array(ew, float), er)


def find_mouths(g, coast_geom, tol=MOUTH_TOL_M, excluded=EXCLUDED_MOUTH_RIVERS, exceptions=None):
    """Cua song: dau mut cua mot song cach bo <= tol (hoac nguong ngoai le cua song do), song khong bi loai.

    Tra (mouths: mouth_id, vertex, river, dist_coast_m, manual; excluded: vertex, river, dist_coast_m).
    `river` cua mot cua = cac song ket thuc tai dinh do, noi bang '/'. manual = chi dat nho ngoai le.
    """
    import shapely

    exceptions = MOUTH_TOL_EXCEPTIONS if exceptions is None else exceptions
    ends = g.river_endpoints()
    cols = ["vertex", "river", "dist_coast_m", "manual"]
    vs = np.array(sorted(ends), dtype=np.int64)
    rows, excl = [], []
    if len(vs):
        d = shapely.distance(shapely.points(g.xy[vs]), coast_geom)
        for v, dc in zip(vs.tolist(), d.tolist()):
            rs = sorted(ends[v])
            ok = [r for r in rs if dc <= tol and r not in excluded]
            manual = [r for r in rs if tol < dc <= exceptions.get(r, -1.0) and r not in excluded]
            if ok or manual:
                rows.append((v, "/".join(ok + manual), dc, bool(manual and not ok)))
            elif dc <= tol and any(r in excluded for r in rs):
                excl.append((v, "/".join(rs), dc))
    df = pd.DataFrame(rows, columns=cols).sort_values(["river", "vertex"]).reset_index(drop=True)
    df.insert(0, "mouth_id", np.arange(len(df)))
    return df, pd.DataFrame(excl, columns=cols[:3])


def bridge_gaps(g, mouth_vertices, tol=GAP_TOL_M):
    """Noi dau mut bac 1 (khong phai cua) toi dinh gan nhat cua manh khac trong <= tol. Tra (graph moi, bang noi)."""
    from scipy.spatial import cKDTree

    comp = g.components()
    tree = cKDTree(g.xy)
    mset = set(int(v) for v in mouth_vertices)
    ends = [int(v) for v in np.nonzero(g.degree == 1)[0] if int(v) not in mset]
    rows, seen = [], set()
    for v in ends:
        cand = [c for c in tree.query_ball_point(g.xy[v], tol) if comp[c] != comp[v]]
        if not cand:
            continue
        dd = np.hypot(*(g.xy[cand] - g.xy[v]).T)
        k = int(np.argmin(dd))
        u = int(cand[k])
        key = tuple(sorted((v, u)))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"from_vertex": v, "to_vertex": u, "gap_m": float(dd[k]),
                     "river_from": g.river[v], "river_to": g.river[u], "manual": False})
    tab = pd.DataFrame(rows, columns=["from_vertex", "to_vertex", "gap_m", "river_from", "river_to", "manual"])
    if tab.empty:
        return g, tab
    return g.add_edges(tab["from_vertex"].to_numpy(), tab["to_vertex"].to_numpy(), tab["gap_m"].to_numpy()), tab


def manual_joins(g, joins, to_xy, search_m=1000.0):
    """Noi thu cong lo du lieu tren CUNG song: dau mut cua song do gan (lon, lat) nhat (<= search_m) -> dinh gan
    nhat cua cung song o manh khac. to_xy: ham list[(lon, lat)] -> mang (n, 2) met. Tra (graph moi, bang noi)."""
    comp = g.components()
    ends = g.river_endpoints()
    rows = []
    for rv, ll in joins:
        p = np.asarray(to_xy([ll]))[0]
        cand = [v for v, rs in ends.items() if rv in rs]
        if not cand:
            raise ValueError(f"MANUAL_JOINS: song {rv} khong co dau mut")
        dd = np.hypot(*(g.xy[cand] - p).T)
        k = int(np.argmin(dd))
        if dd[k] > search_m:
            raise ValueError(f"MANUAL_JOINS: dau mut {rv} gan {ll} nhat cach {dd[k]:.0f} m > {search_m:.0f} m")
        v = int(cand[k])
        same = np.nonzero((g.river == rv) & (comp != comp[v]))[0]
        if not len(same):
            raise ValueError(f"MANUAL_JOINS: {rv} khong co manh khac de noi")
        d2 = np.hypot(*(g.xy[same] - g.xy[v]).T)
        rows.append({"from_vertex": v, "to_vertex": int(same[int(np.argmin(d2))]), "gap_m": float(d2.min()),
                     "river_from": rv, "river_to": rv, "manual": True})
    tab = pd.DataFrame(rows, columns=["from_vertex", "to_vertex", "gap_m", "river_from", "river_to", "manual"])
    if tab.empty:
        return g, tab
    return (g.add_edges(tab["from_vertex"].to_numpy(), tab["to_vertex"].to_numpy(), tab["gap_m"].to_numpy(),
                        label="~noi_tay"), tab)


def sluice_vertices(g, sluices_xy):
    """{ten song: (dinh cat, khoang cach m tu toa do cong)} - dinh gan nhat thuoc dung song do."""
    out = {}
    for rv, (x, y) in sluices_xy.items():
        idx = np.nonzero(g.river == rv)[0]
        if not len(idx):
            raise ValueError(f"Khong co dinh nao thuoc song {rv} de cat cong")
        d = np.hypot(g.xy[idx, 0] - x, g.xy[idx, 1] - y)
        k = int(np.argmin(d))
        out[rv] = (int(idx[k]), float(d[k]))
    return out


def mouth_distances(g, mouths, drop_vertices=()):
    """(dist_m (N,), mouth_id (N,) -1 neu khong toi duoc) - Dijkstra tu nguon ao noi moi cua (w = dist_coast_m)."""
    from scipy.sparse.csgraph import dijkstra

    n = g.n
    mv = mouths["vertex"].to_numpy()
    extra = (np.full(len(mv), n), mv, mouths["dist_coast_m"].to_numpy(float) + 1e-9)
    dist, pred = dijkstra(g.matrix(extra, drop_vertices), directed=False, indices=n, return_predecessors=True)
    dist, pred = dist[:n], pred[:n]
    mid_of = dict(zip(mv.tolist(), mouths["mouth_id"].tolist()))
    lab = np.full(n, -1, np.int64)
    for v in np.argsort(dist, kind="stable"):
        if not np.isfinite(dist[v]):
            break
        p = pred[v]
        lab[v] = mid_of[int(v)] if p == n else lab[p]
    return dist, lab


def check_invariant(g, mouths, dist, lab, tol_m=1.0):
    """(so dinh vi pham, muc vi pham lon nhat m): dist[v] >= dist_coast(cua) + |v - dinh cua| - tol_m."""
    ok = lab >= 0
    m = mouths.set_index("mouth_id")
    mv = m.loc[lab[ok], "vertex"].to_numpy()
    lower = m.loc[lab[ok], "dist_coast_m"].to_numpy(float) + np.hypot(*(g.xy[ok] - g.xy[mv]).T)
    gap = lower - dist[ok]
    return int((gap > tol_m).sum()), float(max(gap.max(initial=0.0), 0.0))


def behind_sluice(lab_open, lab_cut):
    """Dinh sau cong: ra bien duoc khi thong, KHONG ra duoc khi cat tai cong."""
    return (lab_open >= 0) & (lab_cut < 0)


def snap_index(xy, transform, shape, chunk_rows=256):
    """Moi tam pixel -> chi so dinh gan nhat trong xy (int32) + khoang cach (m, float32)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    h, w = shape
    idx = np.empty(shape, "int32")
    lat = np.empty(shape, "float32")
    xs = transform.c + transform.a * (np.arange(w) + 0.5)
    for r0 in range(0, h, chunk_rows):
        rows = np.arange(r0, min(h, r0 + chunk_rows))
        ys = transform.f + transform.e * (rows + 0.5)
        X, Y = np.meshgrid(xs, ys)
        d, k = tree.query(np.column_stack([X.ravel(), Y.ravel()]))
        idx[rows] = k.reshape(len(rows), w)
        lat[rows] = d.reshape(len(rows), w)
    return idx, lat


def nearest_valid_sea_pixel(lat, lon, valid, pts_lonlat, crs_metric="EPSG:32648"):
    """Cho moi diem (lon, lat): (i, j) pixel CMEMS hop le gan nhat + khoang cach km (tam pixel, CRS met)."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    tf = Transformer.from_crs("EPSG:4326", crs_metric, always_xy=True)
    LON, LAT = np.meshgrid(lon, lat)
    ii, jj = np.nonzero(valid)
    px, py = tf.transform(LON[ii, jj], LAT[ii, jj])
    qx, qy = tf.transform(*np.asarray(pts_lonlat, float).T)
    d, k = cKDTree(np.column_stack([px, py])).query(np.column_stack([qx, qy]))
    return ii[k], jj[k], d / 1000.0


def sluice_active(season: int, first=SLUICE_FIRST_SEASON) -> bool:
    return season >= first

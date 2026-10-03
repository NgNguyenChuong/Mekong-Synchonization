"""
Script mot-lan de tao data/mekong_delta_boundary.geojson tu ranh gioi hanh
chinh THAT cua Viet Nam (nguon mo: geoBoundaries.org, public domain).

Khong chay script nay khi server dang hoat dong (chi chay lai thu cong neu
can cap nhat nguon du lieu). Ket qua da duoc luu san trong
mekong_delta_boundary.geojson, backend doc truc tiep file do, KHONG goi
mang moi lan khoi dong.

Cach chay lai (neu can):
    cd webapp/backend/data
    python build_boundary.py
"""
import json
import os
import urllib.request

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

SOURCE_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
    "releaseData/gbOpen/VNM/ADM1/geoBoundaries-VNM-ADM1.geojson"
)

# 13 tinh/thanh chinh thuc thuoc Dong bang song Cuu Long (DBSCL).
MEKONG_DELTA_PROVINCES = {
    "An Giang", "Bạc Liêu", "Bến Tre", "Cà Mau", "Cần Thơ", "Hậu Giang",
    "Kiên Giang", "Long An", "Sóc Trăng", "Tiền Giang", "Trà Vinh",
    "Vĩnh Long", "Đồng Tháp",
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "mekong_delta_boundary.geojson")


def main():
    with urllib.request.urlopen(SOURCE_URL) as resp:
        vn_provinces = json.loads(resp.read().decode("utf-8"))

    geoms = []
    matched = set()
    for feature in vn_provinces["features"]:
        name = feature["properties"].get("shapeName", "").strip()
        if name in MEKONG_DELTA_PROVINCES:
            geoms.append(shape(feature["geometry"]))
            matched.add(name)

    missing = MEKONG_DELTA_PROVINCES - matched
    if missing:
        raise RuntimeError(f"Thieu tinh trong nguon du lieu: {missing}")

    union = unary_union(geoms)

    output = {
        "type": "Feature",
        "properties": {
            "name": "Mekong Delta (13 provinces)",
            "source": SOURCE_URL,
        },
        "geometry": mapping(union),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"Saved {OUTPUT_PATH} ({union.geom_type}, {len(geoms)} provinces merged)")


if __name__ == "__main__":
    main()

"""
Script mot-lan de tao data/world_land_boundary.geojson tu Natural Earth
110m Land (public domain), dung cho che do xem "Toan cau" cua webapp.

Khong chay khi server dang hoat dong. Ket qua da luu san trong
world_land_boundary.geojson, backend doc truc tiep file do.

Cach chay lai (neu can):
    cd webapp/backend/data
    python build_world_boundary.py
"""
import json
import os
import urllib.request

SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_land.geojson"
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "world_land_boundary.geojson")


def main():
    with urllib.request.urlopen(SOURCE_URL) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Saved {OUTPUT_PATH} ({len(data['features'])} landmass polygons)")


if __name__ == "__main__":
    main()

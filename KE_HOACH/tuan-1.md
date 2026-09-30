# Tuần 1 — Nhãn + 4 khung lưới × 3 độ phân giải

**Thời gian:** 29/09 – 05/10/2026

**Tuần nút thắt.** Mọi việc sau đều đứng trên kết quả tuần này.

## Mục tiêu

1. Biết chắc bộ nhãn Zenodo dùng được, nắm rõ đặc điểm của nó.
2. Code hết gắn cứng với H3 và với biến `salinity`.
3. Có 12 file lưới (4 khung × 3 độ phân giải) với diện tích ô được kiểm soát.

## Kết quả kỳ vọng

- [ ] Bảng khảo sát bộ Zenodo
- [ ] Code đã gỡ gắn cứng, `pytest tests/` pass
- [ ] 12 file lưới cùng schema
- [ ] Bảng đối chiếu diện tích ô giữa 4 khung

---

## Việc 1 — Tải và khảo sát bộ Zenodo (ngày 1–2)

**Làm trước tiên** vì đây là thông tin quyết định nhiều nhất trên mỗi giờ bỏ ra.

**Nguồn:** https://zenodo.org/records/15653696 — *The dataset of soil salinity and NDWI index for Vietnam's Mekong Delta (2000–2023)*, đi kèm bài báo Communications Earth & Environment 2025 (doi: 10.1038/s43247-025-02490-z). Cũng có trên Google Earth Engine.

**Đặc tả:**
1. Tải về `data/raw/zenodo_salinity/`. Đọc file mô tả và file code kèm theo; ghi lại **thuật toán nghịch đảo độ mặn** dùng công thức gì và **kỳ mùa khô** được định nghĩa thế nào.
2. Viết `scripts/survey_zenodo.py`, với mỗi raster in ra: CRS, độ phân giải điểm ảnh, khung bao, giá trị NoData, **tỉ lệ % điểm ảnh NoData**, min / max / trung bình / trung vị, đơn vị (tra trong file mô tả — không đoán).
3. Đối chiếu khung bao với ranh giới 13 tỉnh trong `webapp/backend/data/mekong_delta_boundary.geojson`.
4. Lưu bảng kết quả vào `KE_HOACH/ket-qua/tuan1_khao_sat_zenodo.csv`.

**Xử lý năm thiếu dữ liệu:** dùng **toàn bộ 2000–2023**. Chỉ loại những năm có tỉ lệ NoData vượt ngưỡng cố định (đề xuất 30% — chốt ngưỡng trong `GIAO_TIEP` trước khi áp dụng), và ghi rõ năm nào bị loại, vì sao.

> ⚠️ **Phương pháp luận:** lớp độ mặn trong bộ này là **ước lượng bằng nghịch đảo viễn thám**, không phải số đo thực địa. Trong bài báo phải gọi đúng tên, **không gọi là ground truth**.

### Tiêu chí review
- Đơn vị của lớp độ mặn có được trích dẫn từ tài liệu gốc không, hay đoán?
- Tỉ lệ NoData có được tính trên đúng vùng ĐBSCL (cắt theo ranh giới), hay trên cả khung chữ nhật bao ngoài? *Tính trên khung chữ nhật sẽ làm tỉ lệ thiếu cao giả tạo do phần biển và nước ngoài vùng.*

---

## Việc 2 — Gỡ gắn cứng trong code (ngày 3)

**Danh sách chỗ phải sửa (đã rà):**

| File | Gắn cứng |
|---|---|
| `src/dataset.py` | `dtype={"h3_index": str}`, merge theo `["h3_index", "date"]` |
| `src/utils_h3.py` | `h3["h3_index"].tolist()` trong `load_h3_multipoints` |
| `src/training/split.py` | `df["h3_index"]` trong `spatial_holdout_split` |
| `src/training/train.py` | `NON_FEATURE_COLUMNS`, `dropna(subset=["salinity"])`, `"target": "salinity"`, `train_df["h3_index"]`, tham số `--salinity-csv` bắt buộc |
| `src/training/baselines.py` | tham số `h3_index` trong `PersistenceBaseline` |
| `src/processing.py` | cột `"h3_index"` khi xuất CSV đặc trưng |
| `src/salinity/spatial_mapping.py` | gán cột `h3_index` |

**Đặc tả:**
1. Đổi tên cột thành `cell_id` ở tất cả chỗ trên. Kiểm tra đủ bằng `grep -rn "h3_index" src/ tests/ main.py`.
2. `train.py`: thêm `--target` (mặc định `salinity`), thay mọi chỗ viết cứng `"salinity"`. Đổi `--salinity-csv` thành `--label-csv`.
3. Cập nhật test tương ứng trong `tests/`.

### Tiêu chí review
- `grep` còn sót `h3_index` ở đâu không (trừ những chỗ thật sự gọi thư viện H3)?
- Test có bị **sửa kỳ vọng cho khớp code** không? *Chỉ được đổi tên cột trong test, không được đổi logic kiểm tra.*
- Tên biến `h3` trong `load_h3_multipoints` đang đặt trùng tên thư viện `h3` — đổi luôn cho khỏi nhầm.

---

## Việc 3 — Sinh 4 khung lưới × 3 độ phân giải (ngày 4–6)

### Chọn độ phân giải

Dùng 3 mức tương ứng **H3 res 5, 6, 7**.

> Ghi chú cho bài báo: dữ liệu khí tượng ~9 km, nên ở mức res 7 các biến khí tượng bị **phân giải quá mịn so với nguồn** — nhiều ô cùng nhận giá trị từ một điểm ảnh gốc. Vẫn giữ res 7 vì lớp độ mặn/NDWI (30 m) và địa hình vẫn có biến thiên thật ở mức này. Hiện tượng này **là một phần kết quả cần báo cáo**, không phải lỗi.

### Kích thước ô

**Không tự đoán số — tính bằng code:**

```python
import h3
for res in (5, 6, 7):
    print(res, h3.average_hexagon_area(res, unit="km^2"))
```

| Khung | Cách xác định kích thước |
|---|---|
| **H3** | Theo resolution — đã có `generate_h3_grid()` |
| **Ô vuông** | Cạnh = căn bậc hai diện tích ô H3 tương ứng. Sinh trong **EPSG:32648** (UTM 48N, đã có sẵn trong `src/config.py` là `CRS_METRIC`), rồi chuyển về EPSG:4326 |
| **Kinh–vĩ độ** | Bước (độ) sao cho diện tích tương đương ở vĩ độ trung tâm vùng |
| **S2** | Chọn **level gần nhất** về diện tích — xem cảnh báo bên dưới |

### ⚠️ Vấn đề với S2 — phải xử lý đúng

H3 chia mỗi cấp thành **7 ô con**, S2 chia thành **4 ô con**. Hai dãy cấp độ **không khớp nhau** — không phải lúc nào cũng tìm được một level S2 có diện tích bằng ô H3.

Ước lượng thô (Antigravity phải tính lại số thật trên vùng ĐBSCL):

| H3 | Diện tích ~ | S2 gần nhất | Diện tích ~ | Tỉ lệ lệch |
|---|---|---|---|---|
| res 5 | 253 km² | level 9 | 324 km² | ~1,3 lần |
| res 6 | 36 km² | level 10 hoặc 11 | 81 hoặc 20 km² | **~2,2 hoặc ~0,56 lần** |
| res 7 | 5,2 km² | level 12 | 5,1 km² | gần khớp |

Ở res 6, S2 **lệch xa theo cả hai hướng**. Hướng xử lý đề xuất: chọn level gần nhất theo tỉ lệ logarit, **báo cáo trung thực diện tích thật** của mọi khung, và ở bước phân tích đưa **diện tích ô vào làm biến kiểm soát**. Thống nhất hướng này trong `GIAO_TIEP` trước khi code.

Thêm một điểm: ô S2 **không đều diện tích** trên toàn cầu. Trên vùng nhỏ như ĐBSCL mức biến thiên nhỏ, nhưng vẫn phải đo và báo cáo.

**Thư viện:** `s2sphere` (thuần Python, cài được trên Windows). Nếu gặp vấn đề, chạy phần S2 trên WSL hoặc Colab.

### Đặc tả hàm

Thêm vào `src/preprocessing.py`, cạnh `generate_h3_grid()`:
- `generate_square_grid(boundary, cell_area_km2)`
- `generate_latlon_grid(boundary, cell_area_km2)`
- `generate_s2_grid(boundary, level)`

Cả 4 hàm:
- cắt theo đúng ranh giới như hàm H3 (dùng lại STRtree đang có)
- xuất **cùng schema**: cột `cell_id` + `geometry`, CRS EPSG:4326
- lưu vào `data/grids/{khung}_r{mức}.geojson` — ví dụ `s2_r6.geojson`

### Bảng đối chiếu diện tích

Với cả 12 file, tính: số ô, diện tích trung bình, độ lệch chuẩn diện tích, diện tích nhỏ nhất và lớn nhất. **Tính trong hệ toạ độ mét** (EPSG:32648), không tính trên toạ độ độ. Lưu `KE_HOACH/ket-qua/tuan1_doi_chieu_dien_tich.csv`.

> ⚠️ **Hạn chế cần ghi cho bài báo:** ĐBSCL hẹp và gần xích đạo, nên lưới kinh–vĩ độ ở đây **gần như đều diện tích** (cos 10° ≈ 0,985). Thí nghiệm trên riêng ĐBSCL **không bộc lộ được hiện tượng méo diện tích theo vĩ độ**. Đây là một lý do chính cho việc mở rộng sang vùng đối chứng ở tuần 6 — nên chọn ít nhất một vùng ở **vĩ độ cao hơn**.

### Tiêu chí review
- Diện tích có được tính trong hệ toạ độ mét không? *Tính trên độ là sai hoàn toàn.*
- Ô ở biên bị cắt theo ranh giới có bị tính vào diện tích trung bình không? *Nên báo cáo riêng ô nguyên vẹn và ô bị cắt — ô bị cắt nhỏ hơn nhiều, kéo lệch trung bình.*
- 4 khung có phủ **cùng một vùng** không? So tổng diện tích phủ của 4 khung — phải xấp xỉ bằng nhau.
- Lưới ô vuông có được sinh trong hệ toạ độ mét rồi mới chuyển sang độ không? *Sinh thẳng trên độ thì ra hình chữ nhật méo, không phải hình vuông.*

---

## Ngày 7 — Cổng kiểm tra

1. Bộ Zenodo dùng được không? Loại những năm nào, vì sao?
2. `pytest tests/` pass hết không?
3. Đủ 12 file lưới chưa?
4. Bảng đối chiếu diện tích: ngoài S2 ở res 6, các khung có lệch nhau quá 5% không?

## Nhật ký quyết định

### Trạng thái (cập nhật 2026-09-30)

| Việc | Trạng thái | Bằng chứng / còn thiếu |
|---|---|---|
| 1 — Zenodo | **ĐANG DỞ** (tạm hoãn chờ mạng) | Có: `scripts/survey_zenodo.py` (tải + MD5, đọc theo khối, footprint ≥ 50% số năm, suy chế độ cảm biến), chạy thử trên raster tổng hợp nhận đúng TM và OLI; danh sách MD5 chính thức 19 file (7,43 GB) lấy từ API Zenodo; năm 2000 đã tải, MD5 đúng. **Thiếu:** 18 năm còn lại, bảng khảo sát thật, chốt ngưỡng và năm bị loại, sửa `docs/reports/zenodo_dataset_metadata.md` (vẫn còn câu chưa có nguồn về lý do thiếu năm, dS/m ghi như sự thật, chưa có mục Comment/Reply) |
| 2 — Gỡ gắn cứng | **XONG** | `cell_id`, `--target`, `--label-csv` (tuỳ chọn), `--grid-type`, `--include-coords`; `tests/test_generic_grid.py` dùng ID không phải H3 |
| 3 — 13 lưới | **XONG** | `scripts/generate_all_grids.py` + 4 hàm trong `src/preprocessing.py`; `KE_HOACH/ket-qua/tuan1_doi_chieu_dien_tich.csv`; `tests/test_grid_validation.py` (19 test) |
| Cổng kiểm tra | Câu 1 **chưa**; câu 2–4 **đạt** | 2: `pytest -q` một phiên — 42 passed (lúc đóng tuần 1), 50 passed (sau tuần 2). 3: 13 file. 4: ô vuông và kinh–vĩ độ lệch H3 0,05–0,31%; S2 lệch theo đặc tính hệ (L9 +35%, L10 +137%, L11 −41%, L12 +3,5%) |

### Quyết định

- **2026-09-29 — 13 lưới thay vì 12:** thêm S2 L10 làm lưới kẹp ở res 6 (T1-03). Kéo theo tuần 4 có 858 cấu hình thay vì 792.
- **2026-09-29 — Khớp diện tích theo diện tích CỤC BỘ** của lưới H3 trên ranh giới chuẩn, không theo trung bình toàn cầu của `h3.average_hexagon_area` (tại ĐBSCL ô H3 lớn hơn trung bình toàn cầu ~15%). Mục tiêu: res 5 = 291,98 km², res 6 = 41,71, res 7 = 5,96 → cạnh ô vuông 17.087 / 6.458 / 2.441 m; bước kinh–vĩ độ 0,1552 / 0,0586 / 0,0222°.
- **2026-09-29 — Ranh giới chuẩn duy nhất:** `webapp/backend/data/mekong_delta_boundary.geojson` (39.867,81 km²), giữ toàn bộ đảo, không dùng `MIN_ISLAND_AREA_KM2`.
- **2026-09-29 — Quy tắc chọn ô chung cho 4 khung:** ô thuộc lưới khi tâm nằm trong ranh giới; không cắt hình học ô; ghi `overlap_frac`. Neo gốc ô vuông ở bội số cạnh (UTM 48N), kinh–vĩ độ ở bội số bước tính từ (0°, 0°).
- **2026-09-29 — Bỏ NDWI khỏi đặc trưng khi mục tiêu là độ mặn** (và ngược lại): mã GEE của tác giả cho thấy cả hai cùng dựng từ một band NIR.
- **2026-09-30 — Đổi vai trò:** Antigravity dừng; Claude thực hiện, An review.
- **2026-09-30 — Sự cố ghi đè file:** 5 lần (29/09: 14:00:55, 14:49:53, 15:07:39, 15:56:08, 16:21:19) — nhiều file bị ghi về bản cũ hoặc 0 byte cùng lúc. Nghi do editor (Antigravity/VS Code) lưu bộ đệm cũ; **chưa chứng minh**. Khôi phục từ `artifacts/backup_baseline_code/`.
- **2026-09-30 — Sửa khi khôi phục:** `LatLngPoly(outer, holes)` sai cách gọi H3 v4 → `LatLngPoly(outer, *holes)`; lọc vòng lỗ suy biến theo số điểm phân biệt; `generate_h3_grid()` không tự đọc cache; script sinh lưới gọi lại hàm trong `preprocessing.py` thay vì chép logic; ID lưới kinh–vĩ độ đổi sang chỉ số nguyên `(iy, ix)` để tránh sai số dấu phẩy động.
- **2026-09-30 — Xung đột tên module `config`:** `tests/conftest.py` đăng ký một module `config` hợp nhất (gốc `src/config.py` + thuộc tính riêng của `webapp/backend/config.py`), chỉ có tác dụng khi chạy test.
- **2026-09-30 — Đối chiếu lưới H3 res 7 cũ** (`data/reference/h3_grid_dbscl_old_res7.geojson`, 6.959 ô): 98,3% ô mới có trong lưới cũ; 382 ô chỉ có ở lưới cũ đều có tâm ngoài ranh giới chuẩn (lưới cũ dùng ranh giới rộng hơn).
- **2026-09-30 — Lưu dữ liệu Zenodo ở `A:\Dataset_NCKH\zenodo_15653696`**: ổ D chỉ còn 4,1 GB trống.
- **2026-09-30 — Git:** nhánh mới `feature/nckh-tuan1-multigrid` (nhánh `feature/create-Ui` trùng `main`, không có commit riêng); commit `4c68d945`; `.gitignore` thêm ngoại lệ cho `KE_HOACH/` và `GIAO_TIEP/`. Chưa push. `webapp/`, `tests/test_api.py`, `tests/test_predictor.py` **chưa commit** (phần demo giao diện — An quyết định nhánh).

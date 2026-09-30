# Tuần 2 — Dữ liệu dạng lưới + trích theo diện tích

**Thời gian:** 06/10 – 12/10/2026

**Điều kiện bắt đầu:** qua cổng kiểm tra tuần 1.

## Mục tiêu

1. Có đầy đủ dữ liệu khí tượng, địa hình, thực vật cho toàn bộ 2000–2023, đúng định dạng pipeline.
2. **Sửa phương pháp trích đặc trưng** để hình dạng ô thực sự ảnh hưởng đến giá trị.
3. Chạy pipeline cho cả 12 cấu hình lưới.

## Kết quả kỳ vọng

- [ ] Raster khí tượng, địa hình, NDVI cho toàn bộ phạm vi thời gian
- [ ] `processing.py` trích mọi biến bằng trung bình có trọng số theo diện tích
- [ ] 12 bộ đặc trưng dạng lưới, mỗi bộ một thư mục riêng
- [ ] Bảng kiểm tra chất lượng cho 12 bộ

---

## Việc 1 — Cho `main.py` chạy được nhiều khung lưới (ngày 1, sáng)

**Vấn đề đã rà:**
- `main.py` luôn gọi `run_preprocessing()` ở bước 0 → **sinh lại lưới H3, đè mất lưới được đưa vào**.
- `H3_GRID_GEOJSON` và `DATA_PROCESSED` viết cứng trong `src/config.py` → chạy nhiều lần thì **lần sau đè kết quả lần trước**.

**Đặc tả:** trong `src/config.py`, cho hai giá trị đọc được từ biến môi trường (dùng các hàm `_env_*` có sẵn): `GRID_GEOJSON` và `OUTPUT_DIR`. Trong `main.py`, nếu `GRID_GEOJSON` được đặt thì **bỏ qua bước 0**.

```bash
GRID_GEOJSON=data/grids/s2_r6.geojson OUTPUT_DIR=data/processed/s2_r6 python main.py
```

### Tiêu chí review
- Không đặt biến môi trường thì hành vi cũ có giữ nguyên không? *Không được làm hỏng cách chạy hiện tại.*

---

## Việc 2 — Thu thập dữ liệu dạng lưới (ngày 1 chiều – ngày 3)

### Quy ước định dạng pipeline đang đòi (đã rà trong code — sai là không đọc được)

| Loại | Quy ước |
|---|---|
| Theo ngày (khí tượng) | **Một GeoTIFF cho mỗi tháng**. Tên chứa `YYYY_MM` (ví dụ `rain_2020_03.tif`). **Band thứ k = ngày thứ k của tháng.** |
| Định kỳ (NDVI, NDWI, độ mặn) | Một GeoTIFF cho mỗi mốc thời gian, tên theo `file_pattern` trong `PERIODIC_SPECS` (ví dụ `NDVI_{date}.tif`, `date_pattern` = `%Y-%m-%d`) |
| Tĩnh (DEM, lớp phủ, sông) | Một file, khai báo trong `STATIC_SPECS` |

### Nguồn dữ liệu — tất cả lấy qua Google Earth Engine

Lý do dùng GEE: xuất thẳng ra GeoTIFF, khớp định dạng pipeline, **không phải viết thêm tầng đọc NetCDF**.

| Nhóm | Bộ dữ liệu GEE | Ghi chú |
|---|---|---|
| Khí tượng | `ECMWF/ERA5_LAND/DAILY_AGGR` | Mưa, nhiệt độ TB/max/min, độ ẩm, bức xạ. ~9 km |
| Độ cao | `COPERNICUS/DEM/GLO30` | 30 m |
| Lớp phủ | `ESA/WorldCover/v200` | 10 m. Mã lớp khớp sẵn với `class_names` trong `STATIC_SPECS` (10, 20, …, 95) |
| Mặt nước (cho khoảng cách sông) | `JRC/GSW1_4/GlobalSurfaceWater` | 30 m |
| **Thực vật** | **`MODIS/061/MOD13Q1`** | NDVI 250 m, 16 ngày — xem cảnh báo |

> ⚠️ **Sentinel-2 không phủ đủ phạm vi thời gian.** `PERIODIC_SPECS` hiện lấy NDVI từ Sentinel-2 — vệ tinh này **chỉ có dữ liệu từ năm 2015**. Làm đủ 2000–2023 thì 15 năm đầu trống. Đổi sang **MODIS MOD13Q1** (có từ năm 2000, nhất quán suốt cả phạm vi).
>
> Thêm một lý do: nhãn độ mặn Zenodo được dựng từ **Landsat**. Nếu NDVI cũng lấy từ Landsat thì đặc trưng và nhãn dùng chung nguồn cảm biến — dễ rò rỉ thông tin. Dùng MODIS (cảm biến khác) tránh được vấn đề này.

**Lưu ý đơn vị:** ERA5-Land lưu nhiệt độ theo **Kelvin**, lượng mưa theo **mét**, bức xạ theo **J/m²**. Đổi về °C, mm, W/m² ngay trong GEE trước khi xuất, và ghi lại công thức đổi.

**Về hạn mức GEE:** xuất theo từng lô (từng năm, từng biến), không xuất một lần cả 24 năm. Nếu chậm thì chạy xuất qua đêm — **không giảm phạm vi thời gian**.

### Tiêu chí review
- Số band của mỗi file tháng có đúng bằng số ngày trong tháng không (kể cả tháng 2 năm nhuận)?
- Đơn vị sau khi đổi có hợp lý không? *Kiểm tra: lượng mưa ngày không âm, nhiệt độ ĐBSCL nằm trong khoảng 18–40 °C.*
- Cả 4 khung dùng **cùng một bộ file raster** không? *Phải cùng — nếu không thì không còn là so sánh khung nữa.*

---

## Việc 3 — Đổi phương pháp trích sang trung bình theo diện tích (ngày 4–5) ⚠️ QUAN TRỌNG NHẤT TUẦN

**Vấn đề:** dữ liệu theo ngày hiện được trích bằng `sample_multiband_robust()` — **chỉ lấy giá trị tại tâm ô**, chỉ khi tâm rơi vào NoData mới lấy thêm các điểm giữa cạnh.

**Vì sao đây là lỗi nghiêm trọng với đề tài này:** nếu mỗi ô chỉ nhận giá trị của một điểm tâm, **hình dạng ô gần như không ảnh hưởng gì** đến giá trị ô nhận được. Thí nghiệm "so sánh các khung lưới" khi đó thực chất chỉ so sánh vị trí các điểm tâm — **không đo đúng thứ đề tài tuyên bố đo**. Phản biện có kinh nghiệm sẽ bắt lỗi này ngay.

Thêm một thiên lệch phụ: `get_h3_sample_points()` sinh điểm giữa theo **số cạnh thật** của hình — lục giác được 6 điểm, ô vuông 4 điểm, ô biên bị cắt méo được rất nhiều điểm. Mật độ lấy mẫu không đồng đều giữa các khung.

**Đặc tả:** trích **mọi biến** (khí tượng, địa hình, thực vật, nhãn) bằng **trung bình có trọng số theo diện tích phủ** của từng điểm ảnh trên ô.

**Công cụ:** thư viện **`exactextract`** — tính chính xác phần diện tích mỗi điểm ảnh nằm trong ô, xử lý đúng khi ô nhỏ hơn điểm ảnh, và đọc được ảnh nhiều band trong một lần (nhanh với dữ liệu tháng).

> ⚠️ **Không dùng `rasterstats` mặc định:** nó chỉ tính các điểm ảnh có **tâm** nằm trong ô. Với dữ liệu ~9 km và ô H3 res 6 (~6 km bề ngang), **nhiều ô sẽ không chứa tâm điểm ảnh nào → trả về rỗng**.

**Riêng biến lớp phủ đất:** không lấy trung bình mã lớp (vô nghĩa — trung bình của "rừng" và "đô thị" không phải là gì cả). Giữ cách tính **tỉ lệ % diện tích từng lớp** như `method="all_classes"` đang làm, nhưng tính theo trọng số diện tích.

### Tiêu chí review
- Có còn biến nào được trích theo điểm tâm không?
- **Kiểm tra bằng tay:** chọn 3–5 ô bất kỳ ở mỗi khung, tính tay trung bình theo diện tích, so với kết quả code. *Bắt buộc có bằng chứng kiểm tra này trong `GIAO_TIEP`.*
- Ô không giao với điểm ảnh hợp lệ nào thì trả về NaN hay 0? *Phải là NaN.*

---

## Việc 4 — Chạy pipeline 12 lần (ngày 6)

Mỗi tổ hợp khung × độ phân giải một thư mục output: `data/processed/{khung}_r{mức}/` — ví dụ `h3_r5`, `s2_r6`, `square_r7`, `latlon_r5`.

Chạy nền qua đêm. **Ghi lại thời gian chạy và dung lượng output của từng cấu hình** — đây là số liệu về chi phí tính toán của mỗi khung, dùng được trong bài báo.

---

## Ngày 7 — Cổng kiểm tra

1. Đủ 12 bộ đặc trưng chưa?
2. 12 bộ có **cùng phạm vi ngày** và **cùng danh sách cột** không?
3. Tỉ lệ thiếu của từng cột ở mỗi bộ (chạy `src/data_validation.py`) — ghi vào `KE_HOACH/ket-qua/tuan2_chat_luong.csv`
4. Đã có bằng chứng kiểm tra tay phép trích theo diện tích chưa?

**Nếu câu 2 không đạt:** dừng lại tìm nguyên nhân — so sánh giữa các bộ không cùng điều kiện là vô nghĩa.

## Nhật ký quyết định

### Trạng thái (cập nhật 2026-09-30)

| Việc | Trạng thái | Bằng chứng / còn thiếu |
|---|---|---|
| 1 — `main.py` nhiều lưới | **XONG** | Biến môi trường `GRID_GEOJSON`, `OUTPUT_DIR` (+ `RAW_DIR`, xem quyết định). Test chạy `main.py` đầu–cuối trên lưới và raster tổng hợp, kiểm tra thư mục mặc định `data/processed/` không bị ghi. **Chưa** chạy nhánh cũ (không đặt biến) với dữ liệu thật |
| 2 — Dữ liệu dạng lưới | **CHƯA LÀM** | Cần mạng ổn định và tài khoản Google Earth Engine của An |
| 3 — Trích theo diện tích | **XONG phần code** | `src/processing.py` dùng `exactextract` 0.3.0 cho mọi biến; 8 test trên raster tổng hợp giá trị biết trước (ô phủ nửa pixel 1 + nửa pixel 2 → 1,5; ô nhỏ hơn pixel → giá trị pixel; NaN/NoData bị bỏ; ô không có pixel hợp lệ → NaN). **Còn thiếu theo tiêu chí review:** kiểm tay 3–5 ô mỗi khung trên dữ liệu thật — cần Việc 2 |
| 4 — Chạy 13 cấu hình | **CHƯA LÀM** | Cần Việc 2 |
| Cổng kiểm tra | **Chưa** | — |

### Quyết định

- **2026-09-30 — Sửa hai lỗi có sẵn trong `src/processing.py`:** (1) file không import được (`IndentationError` dòng 433, do dòng bị chèn sai thụt lề); (2) hàm gộp dữ liệu tĩnh và định kỳ dùng lại `df` cũ khi thiếu file.
- **2026-09-30 — Thêm biến `RAW_DIR`** (ngoài đặc tả, cùng loại với `GRID_GEOJSON`/`OUTPUT_DIR`) để đặt dữ liệu thô ngoài ổ D, vì ổ D không đủ chỗ cho raster 24 năm.
- **2026-09-30 — Thay đổi hành vi đáng chú ý của tầng trích:**
  - Lớp phủ đất: ô không có pixel hợp lệ nhận NaN; trước đây bị điền 0 cho mọi lớp.
  - Khoảng cách tới sông tính trên UTM ước lượng theo vùng (`estimate_utm_crs`) thay cho EPSG:3857 — dùng được cho vùng đối chứng ở tuần 6; ở vĩ độ ~10° giá trị nhỏ hơn bản cũ khoảng 1–2%.
  - `method` khác `mean` / `all_classes` / `min_distance` giờ báo lỗi rõ ràng, thay vì âm thầm tính thống kê khác.
  - Lỗi trong worker dữ liệu tĩnh và định kỳ giờ được đưa ra ngoài, không còn bị nuốt.
- **2026-09-30 — Bỏ mã fill dữ liệu đã bị comment** trong `processing.py`; khôi phục được từ commit `4c68d945`.
- **2026-09-30 — Git:** commit `820ec7ac` trên `feature/nckh-tuan1-multigrid`; `exactextract==0.3.0` thêm vào `requirements.txt`.
- Ghi chú: cảnh báo `PendingDeprecationWarning` khi chạy test phát sinh bên trong thư viện `rasterio`, không từ mã dự án.

# Tuần 4 — Chạy toàn bộ thực nghiệm ĐBSCL

**Thời gian:** 20/10 – 26/10/2026 · **Trình GVHD:** ~28/10/2026

**Điều kiện bắt đầu:** qua cổng kiểm tra tuần 3 — đặc biệt là đã **chạy thử trọn quy trình trên một cấu hình nhỏ** thành công.

## Mục tiêu

1. Chốt đơn vị phân tích theo thời gian — quyết định ảnh hưởng đến toàn bộ khối lượng tính toán.
2. Chạy toàn bộ lưới thực nghiệm cho ĐBSCL.
3. Có bảng sai số đầy đủ kèm kiểm định → mang đi trình GVHD.

## Kết quả kỳ vọng

- [ ] Đơn vị phân tích đã thống nhất, có ghi lý do
- [ ] Script chạy thực nghiệm tự động
- [ ] `artifacts/experiments/results_all.csv` — toàn bộ kết quả
- [ ] Bảng kiểm định giữa các cặp khung (p-value đã hiệu chỉnh Holm)
- [ ] Nội dung trình GVHD

---

## Việc 1 — Chốt đơn vị phân tích (ngày 1, sáng) ⚠️ PHẢI THỐNG NHẤT TRƯỚC KHI CHẠY

**Vấn đề:** các nguồn dữ liệu có nhịp thời gian khác nhau.

| Nguồn | Nhịp thời gian |
|---|---|
| Khí tượng ERA5-Land | Theo ngày |
| Dòng chảy, mực nước biển | Theo ngày |
| **Nhãn độ mặn, NDWI (Zenodo)** | **Một bản đồ cho mỗi mùa khô** — tổng cộng 24 mốc |
| NDVI MODIS | 16 ngày |
| Địa hình, lớp phủ | Tĩnh |

**Khối lượng nếu chạy thẳng theo ngày** (ước lượng thô, 24 năm × ~180 ngày mùa khô):

| Độ phân giải | Số ô ~ | Số dòng ~ |
|---|---|---|
| res 5 | 160 | 0,7 triệu |
| res 6 | 1.100 | 4,8 triệu |
| res 7 | 7.700 | **33 triệu** |

Ở res 7, Random Forest và MLP trên laptop sẽ tốn **hàng giờ cho mỗi lần chạy** — nhân với hàng trăm cấu hình thì không khả thi.

**Đề xuất:**

1. **Thí nghiệm chính — đơn vị (ô, mùa khô):** mọi biến gộp về mức mùa khô — nhiệt độ, độ ẩm, bức xạ, dòng chảy, mực nước biển lấy **trung bình**; lượng mưa lấy **tổng**. Độ mặn, NDWI dùng đúng giá trị Zenodo của mùa đó. Số dòng ở res 7 còn khoảng 185 nghìn — mọi mô hình đều chạy được. Và quan trọng: **khớp với nhịp của nhãn**, không có chuyện lặp một giá trị nhãn cho nhiều ngày (lặp như vậy là nhân bản giả mẫu, làm kiểm định thống kê sai).
2. **Thí nghiệm phụ — đơn vị (ô, ngày), chỉ cho 4 biến khí tượng, ở res 5 và 6, chỉ HistGB:** mục đích là giữ lại **độ tương phản về biến động không gian**. Mưa theo ngày rất lộn xộn, còn gộp theo mùa thì mượt đi nhiều. Có cả hai thang thời gian thì biểu đồ chính ở tuần 5 có **thêm điểm dữ liệu** với dải biến động rộng hơn — làm kiểm định giả thuyết GT2 chắc hơn.

**Không chạy gì cho đến khi đề xuất này được thống nhất trong `GIAO_TIEP`.**

### Tiêu chí review
- Gộp mùa khô có dùng **đúng định nghĩa kỳ mùa khô của bài báo Zenodo** không? *Phải khớp — nếu không thì đặc trưng và nhãn nói về hai khoảng thời gian khác nhau.*
- Lượng mưa gộp bằng tổng, các biến khác bằng trung bình — có nhất quán không?

---

## Việc 2 — Script chạy thực nghiệm (ngày 1 chiều – ngày 2)

**Không chạy tay từng lệnh** — hàng trăm cấu hình mà chạy tay thì chắc chắn sót hoặc nhầm.

**Đặc tả `scripts/run_experiments.py`:** lặp qua mọi tổ hợp, **ghi nối tiếp vào một file** `artifacts/experiments/results_all.csv`, mỗi dòng một lần chạy, các cột: `thi_nghiem` (chính / phụ), `khung`, `do_phan_giai`, `bien_muc_tieu`, `mo_hinh`, `seed`, `n_train`, `n_eval`, `mae`, `rmse`, `r2`, `thoi_gian_giay`, `trang_thai`, `loi`.

Đồng thời lưu **sai số tại từng điểm đánh giá** của mỗi lần chạy vào `artifacts/experiments/per_point/` — cần cho kiểm định theo cặp.

**Ba tính năng bắt buộc:**
1. **Bỏ qua cấu hình đã chạy xong** (đọc lại `results_all.csv` trước) — bị ngắt giữa chừng thì chạy lại không phải làm từ đầu.
2. **Ghi lỗi rồi chạy tiếp** — không dừng cả loạt vì một cấu hình lỗi.
3. **Lưu cấu hình đầy đủ** mỗi lần chạy để tái lập được.

**Quy mô thí nghiệm chính:**

| Thành phần | Số lượng |
|---|---|
| Khung | 4 |
| Độ phân giải | 3 |
| Biến mục tiêu | 6 |
| Mô hình | IDW, hồi quy tuyến tính (1 lần — tất định); Random Forest, HistGB, MLP (mỗi mô hình 3 seed) |
| **Tổng** | **4 × 3 × 6 × (1 + 1 + 3 + 3 + 3) = 792 lần chạy** |

**Thí nghiệm phụ:** 4 khung × 2 độ phân giải × 4 biến × HistGB 3 seed = **96 lần chạy**.

**Lưu ý riêng cho MLP:** cần chuẩn hoá đặc trưng (`StandardScaler`) — **fit chỉ trên tập huấn luyện**. MLP không nhận NaN, dùng bản dữ liệu đã điền trung vị (từ tuần 3).

### Tiêu chí review
- Cấu hình bị lỗi có được ghi lại, hay bị bỏ qua lặng lẽ?
- `StandardScaler` có bị fit trên toàn bộ dữ liệu không? *Nếu có là rò rỉ.*
- Kiểm tra rò rỉ biến mục tiêu: khi dự đoán mưa, **cột mưa không được nằm trong đặc trưng**. Khi dự đoán độ mặn, cân nhắc bỏ NDWI khỏi đặc trưng vì hai lớp này cùng dựng từ ảnh Landsat trong bộ Zenodo. Ghi rõ quyết định.

---

## Việc 3 — Chạy thử quy mô nhỏ (ngày 3, sáng)

Chạy khoảng 20 cấu hình đầu (ví dụ toàn bộ res 5 cho 2 biến). Mở `results_all.csv` kiểm tra:
- R² âm hàng loạt, hoặc MAE bằng 0 → có lỗi
- Học máy **thua cả trung bình toàn cục** → có lỗi
- Thời gian chạy mỗi cấu hình → ước lượng tổng thời gian cho cả 792 lần

**Có dấu hiệu bất thường thì dừng lại sửa**, đừng để chạy hết mới phát hiện.

---

## Việc 4 — Chạy toàn bộ (ngày 3 chiều – ngày 5)

Chạy nền, qua đêm. Ước lượng thô cho thí nghiệm chính khoảng 1–2 ngày máy; thí nghiệm phụ thêm khoảng nửa ngày.

**Nếu chậm hơn ước lượng nhiều:** báo qua `GIAO_TIEP`. Hướng xử lý là **chạy trên máy mạnh hơn** (Colab, máy phòng lab) — **không giảm số cấu hình**.

Trong lúc chờ: bắt đầu dựng hình minh hoạ 4 khung chia lưới (hình 1 của tuần 5).

---

## Việc 5 — Kiểm định và bảng tổng hợp (ngày 6)

1. Với mỗi (biến × độ phân giải × mô hình): kiểm định Wilcoxon theo cặp cho **6 cặp khung** trên sai số tại từng điểm.
2. Hiệu chỉnh Holm trên **toàn bộ** tập p-value.
3. Bảng tổng hợp chính: hàng = biến mục tiêu, cột = khung, ô = MAE trung bình (± độ lệch chuẩn qua các seed), đánh dấu khác biệt có ý nghĩa sau hiệu chỉnh.
4. Trả lời thẳng: **học máy có thắng IDW không?** Ở biến nào thắng, biến nào thua?

---

## Việc 6 — Chuẩn bị trình GVHD (ngày 7)

Buổi này trình **kết quả so sánh đầy đủ cho ĐBSCL**. Phân tích Moran's I và mở rộng vùng làm tiếp ở tuần 5–6.

Bố cục gợi ý:
1. Câu hỏi nghiên cứu — một câu
2. Dữ liệu: nguồn, phạm vi, **cách đồng bộ hybrid** (điểm → lưới theo đường sông)
3. Thiết kế thí nghiệm: 4 khung, **3 cơ chế bảo đảm so sánh công bằng** (khớp diện tích, cùng tập điểm đánh giá, trích theo diện tích)
4. Bảng kết quả tổng hợp
5. Nhận xét bước đầu
6. Kế hoạch tuần 5–7 và lịch bài báo
7. Câu hỏi xin ý kiến

**Chuẩn bị sẵn câu trả lời** cho câu gần như chắc chắn sẽ bị hỏi: *"Sao biết khác biệt là do khung lưới, không phải do yếu tố khác?"* — câu trả lời nằm ở mục 3.

---

## Cổng kiểm tra

1. **Có bảng sai số đầy đủ cho cả 4 khung chưa?** ← câu quyết định
2. Bao nhiêu cấu hình lỗi, lỗi vì sao?
3. Học máy có thắng IDW không?
4. Kết quả kiểm định cho thấy điều gì — tóm lại trong một câu.

**Nếu học máy thua IDW ở hầu hết các biến:** đó không phải thất bại mà là một phát hiện. Ghi nhận trung thực.

## Nhật ký quyết định

-

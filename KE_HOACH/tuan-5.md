# Tuần 5 — Phân tích + bộ hình (ĐBSCL)

**Thời gian:** 27/10 – 02/11/2026

**Điều kiện bắt đầu:** có bảng sai số đầy đủ từ tuần 4, đã nhận góp ý của GVHD.

## Mục tiêu

1. Kiểm chứng giả thuyết chính: quan hệ giữa mức biến động không gian của biến và mức ảnh hưởng của lựa chọn khung.
2. **Kiểm chứng riêng phần hybrid:** khung biểu diễn có ảnh hưởng đến việc hợp nhất dữ liệu điểm không.
3. Kiểm tra lại tính hợp lệ của cách chia khối đánh giá.
4. Dựng bộ hình hoàn chỉnh cho ĐBSCL.

## Kết quả kỳ vọng

- [ ] Moran's I + tầm semivariogram cho từng biến, tính trên chuẩn chung
- [ ] Xác nhận kích thước khối giữ riêng đủ lớn — hoặc đã dựng lại và chạy lại
- [ ] Kết quả thí nghiệm bóc tách phần hybrid
- [ ] **Biểu đồ chính** + bộ hình hoàn chỉnh

---

## Việc 1 — Tính Moran's I và semivariogram (ngày 1) ⚠️

**Moran's I đo gì:** mức độ các vị trí gần nhau có giá trị giống nhau. Cao (gần 1) = biến thiên trơn (như nhiệt độ); thấp = biến động cục bộ, lộn xộn (như mưa rào).

**Bẫy phải tránh — tính trên chính từng khung lưới:** đó là **lập luận vòng quanh**. Khung lưới vừa ảnh hưởng đến Moran's I, vừa ảnh hưởng đến sai số — đem hai thứ đó so với nhau thì quan hệ tìm được là do chính cái khung tạo ra, không phải đặc tính thật của biến.

**Cách đúng — Moran's I phải là đặc tính của biến, không phụ thuộc khung:**
- Dùng tập `eval_points.geojson`
- Lấy giá trị từng biến tại các điểm từ raster gốc
- Dùng **cùng một ma trận trọng số không gian cho mọi biến** (theo dải khoảng cách cố định — chốt giá trị trong `GIAO_TIEP`)
- Tính riêng cho từng thang thời gian (mùa khô ở thí nghiệm chính, ngày ở thí nghiệm phụ), rồi lấy trung bình theo thời gian

**Thước đo thứ hai:** tầm ảnh hưởng (range) của **semivariogram**. Nếu cả hai thước đo cho cùng xu hướng thì kết luận vững hơn nhiều.

**Công cụ:** `esda` + `libpysal` (Moran's I), `scikit-gstat` (semivariogram). Thêm vào `requirements.txt`.

### Tiêu chí review
- Có chỗ nào Moran's I được tính trên dữ liệu đã gộp theo khung lưới không? *Nếu có là sai.*
- Ma trận trọng số có giống hệt nhau cho mọi biến không?

---

## Việc 2 — Kiểm tra lại kích thước khối đánh giá (ngày 1 chiều) ⚠️

Ở tuần 3 đã chọn khối giữ riêng cạnh 50 km **trước khi biết** tầm tự tương quan của dữ liệu. Giờ đã có tầm semivariogram — **phải kiểm tra lại**.

**Quy tắc:** kích thước khối phải **lớn hơn** tầm tự tương quan của các biến (Roberts et al. 2017). Nếu khối nhỏ hơn thì ô kiểm tra vẫn "nhìn trộm" được ô huấn luyện qua tự tương quan → kết quả tuần 4 quá lạc quan.

**Nếu khối không đủ lớn:** dựng lại khối lớn hơn và **chạy lại tuần 4**. Nhờ script có tính năng tiếp tục và đơn vị mùa khô gọn nhẹ, việc chạy lại mất khoảng 1–2 ngày máy — chấp nhận giãn lịch, **không được bỏ qua bước này**. Một kết quả đánh giá quá lạc quan là lỗi phương pháp mà phản biện sẽ bắt.

---

## Việc 3 — Thí nghiệm bóc tách phần hybrid (ngày 2) ⚠️ BẰNG CHỨNG CHO ĐÓNG GÓP MỚI

**Câu hỏi:** khung biểu diễn có ảnh hưởng đến việc hợp nhất dữ liệu điểm (dòng chảy, mực nước biển) không? Đây là phần "hybrid synchronization" — **cần bằng chứng riêng**, không thể chỉ suy ra từ kết quả tổng.

**Thiết kế:** chạy thêm HistGB **không có đặc trưng từ dữ liệu điểm** — chỉ có đặc trưng lưới — cho mọi khung × độ phân giải × biến mục tiêu, 3 seed.

**Thước đo:** với mỗi khung, **mức cải thiện khi thêm dữ liệu điểm** = MAE (không có điểm) − MAE (có điểm).

**Đọc kết quả:**
- Mức cải thiện **khác nhau giữa các khung** → khung biểu diễn **có** ảnh hưởng đến hợp nhất dữ liệu điểm. *Đây là bằng chứng trực tiếp cho đóng góp của đề tài.*
- Mức cải thiện **như nhau** → khung không ảnh hưởng đến phần này. Vẫn là kết quả, báo cáo trung thực.

**Dự đoán cần kiểm chứng:** hiệu ứng rõ nhất ở **độ mặn** — vì biến này phụ thuộc mạnh vào khoảng cách dọc sông từ cửa biển, mà khoảng cách này tính từ tâm ô nên đổi theo khung.

**Khối lượng:** 4 × 3 × 6 × 3 = 216 lần chạy trên dữ liệu mùa khô — nhanh.

---

## Việc 4 — Biểu đồ chính (ngày 3)

**Hình quan trọng nhất của cả đề tài.**

- **Trục hoành:** Moran's I của biến (Việc 1)
- **Trục tung:** mức ảnh hưởng của lựa chọn khung — độ chênh MAE tương đối giữa khung tốt nhất và kém nhất: `(MAE_max - MAE_min) / MAE_min`, lấy trên HistGB
- **Mỗi điểm** = một biến × một độ phân giải × một thang thời gian → khoảng **26 điểm** (18 từ thí nghiệm chính + 8 từ thí nghiệm phụ)
- Đường hồi quy kèm dải tin cậy

**Tách thêm hai hiệu ứng nhờ thiết kế 4 khung** — đây là điểm mạnh riêng của đề tài:

| Hiệu ứng | So sánh | Ý nghĩa |
|---|---|---|
| **Hình dạng ô** | H3 với S2 | Cùng là DGGS, chỉ khác hình dạng lục giác / tứ giác |
| **DGGS so với lưới phẳng** | S2 với ô vuông | Cùng dạng tứ giác, chỉ khác hệ quy chiếu |

Vẽ riêng hai hiệu ứng này theo Moran's I. Nếu hai đường có xu hướng khác nhau thì đó là một phát hiện đáng giá: *biết được thành phần nào của khung biểu diễn thực sự quan trọng*.

> ⚠️ **Nhớ biến kiểm soát diện tích:** ở res 6, diện tích ô S2 lệch xa H3 (xem tuần 1). Khi so H3 với S2, phải đưa **diện tích ô vào làm biến kiểm soát** trong hồi quy, không thì hiệu ứng "hình dạng" sẽ lẫn với hiệu ứng "kích thước".

**Thống kê:** tương quan **Spearman** kèm khoảng tin cậy bootstrap.

> ⚠️ **Trung thực về cỡ mẫu:** khoảng 26 điểm là **ít**. Không viết "chứng minh được quy luật". Viết: *"kết quả cho thấy xu hướng…, cần kiểm chứng thêm"*. Tuần 6 sẽ bổ sung điểm từ các vùng khác.

**Nếu không thấy xu hướng:** giả thuyết GT2 bị bác bỏ — ghi nhận thẳng. Bài báo chuyển trọng tâm sang mức ảnh hưởng **trung bình** của lựa chọn khung.

---

## Việc 5 — Bộ hình hoàn chỉnh (ngày 4–5)

| Hình | Nội dung |
|---|---|
| 1 | Bản đồ ĐBSCL với 4 kiểu chia ô — phóng to một vùng nhỏ cho dễ nhìn |
| 2 | Sơ đồ quy trình đồng bộ hybrid: nguồn lưới + nguồn điểm → quy tắc chuyển theo đường sông → khung chung |
| 3 | Bảng nhiệt MAE: hàng = biến, cột = khung |
| 4 | **Biểu đồ chính** (Việc 4) — kèm hai hình tách hiệu ứng |
| 5 | Kết quả bóc tách hybrid (Việc 3) |
| 6 | Sai số theo độ phân giải cho từng khung |

**Quy ước:** mỗi khung giữ **một màu cố định** ở mọi hình. Xuất tối thiểu 300 dpi — tạp chí sẽ đòi. Lưu cả mã vẽ hình để vẽ lại được khi phản biện yêu cầu chỉnh.

---

## Cổng kiểm tra

1. Khối đánh giá đã được xác nhận đủ lớn (hoặc đã chạy lại) chưa?
2. Biểu đồ chính cho thấy gì — viết kết luận trong **một câu**.
3. Thí nghiệm bóc tách hybrid cho thấy gì — viết kết luận trong **một câu**.
4. Đủ 6 hình chưa?

## Nhật ký quyết định

-

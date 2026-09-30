# Kế hoạch thực hiện — phạm vi đầy đủ

> Mục tiêu: một nghiên cứu **hoàn chỉnh** đủ chuẩn công bố Scopus.
> Nguyên tắc: **không cắt phạm vi để kịp lịch — thiếu thời gian thì giãn lịch.**
>
> **Đọc trước:** [`00-quy-dao-va-milestone.md`](00-quy-dao-va-milestone.md) — đích đến, phạm vi khoá, milestone, cổng đi/dừng. Khi mâu thuẫn, file đó ưu tiên hơn file này.

## Cách phối hợp

- **Antigravity** viết code theo đặc tả trong các file tuần.
- **Claude** chỉ đạo và review: duyệt đề xuất trước khi code, review code sau khi code.
- Mọi trao đổi đi qua thư mục [`GIAO_TIEP/`](../GIAO_TIEP/) — xem quy trình ở đó. **Chưa thống nhất thì chưa code.**

Mỗi file tuần có mục **"Tiêu chí review"** — đó là những gì sẽ bị kiểm tra. Antigravity nên đọc mục này **trước** khi viết code để khỏi phải làm lại.

## Lịch tổng thể

| Tuần | Nội dung | Thời gian |
|---|---|---|
| 1 | Nhãn + 4 khung lưới × 3 độ phân giải + gỡ gắn cứng | 29/09 – 05/10 |
| 2 | Dữ liệu khí tượng – địa hình đầy đủ + trích theo diện tích | 06/10 – 12/10 |
| 3 | **Đồng bộ hybrid** (dòng chảy, mực nước biển) + tập đánh giá + đường cơ sở | 13/10 – 19/10 |
| 4 | Chạy toàn bộ thực nghiệm ĐBSCL + kiểm định | 20/10 – 26/10 |
| — | **Trình GVHD: bảng so sánh đầy đủ cho ĐBSCL** | **~28/10** |
| 5 | Phân tích Moran's I + bộ hình (ĐBSCL) | 27/10 – 02/11 |
| 6 | Mở rộng vùng khí hậu khác + kiểm chứng trên bài toán dự báo | 03/11 – 09/11 |
| 7 | Tổng hợp toàn bộ + đệm | 10/11 – 16/11 |
| — | Viết bài báo | 17/11 – 07/12 |
| — | **Nộp tạp chí** | **~10/12/2026** |
| — | Kết quả phản biện vòng 1 | ~3–4/2027 |
| — | Nếu bị từ chối: nộp lại, kết quả vòng 2 | ~8/2027 — vẫn trước hạn 10/2027 |

Nếu tuần nào trễ, **giãn tuần đó ra**, không bỏ việc. Tuần 7 là đệm dành cho việc này. Nếu vượt cả tuần 7 thì lùi lịch viết bài báo tương ứng — vẫn còn đủ khoảng cho một lần bị từ chối.

## Câu hỏi nghiên cứu

> Khi đồng bộ dữ liệu đa nguồn — có nguồn dạng lưới, có nguồn dạng điểm, có nguồn thưa theo thời gian — về một khung biểu diễn không gian chung, việc chọn khung đó ảnh hưởng thế nào đến độ chính xác của mô hình học máy? Và ảnh hưởng ấy phụ thuộc vào đặc tính gì của dữ liệu?

## Phạm vi đầy đủ

| Hạng mục | Phạm vi |
|---|---|
| **Khung biểu diễn** | H3 (lục giác, DGGS) · S2 (tứ giác, DGGS) · ô vuông (phẳng, theo hệ toạ độ chiếu) · kinh–vĩ độ |
| **Độ phân giải** | 3 mức, tương ứng H3 res 5, 6, 7 |
| **Thời gian** | Toàn bộ 2000–2023 theo bộ Zenodo (các mùa khô) |
| **Nguồn dạng lưới** | Khí tượng ERA5-Land, địa hình, lớp phủ đất, NDWI/độ mặn Zenodo |
| **Nguồn dạng điểm — hybrid** | Dòng chảy Tân Châu, Châu Đốc (MRC) · mực nước biển (CMEMS) |
| **Biến mục tiêu** | Lượng mưa, nhiệt độ, độ ẩm, bức xạ, NDWI, độ mặn |
| **Mô hình** | IDW, hồi quy tuyến tính, Random Forest, HistGradientBoosting, mạng nơ-ron MLP |
| **Vùng** | ĐBSCL (chính) + 2 vùng đối chứng khác đới khí hậu (khô hạn, núi cao) |
| **Thiết lập bài toán** | Ước lượng không gian (chính) + dự báo theo thời gian (kiểm chứng) |

**Vì sao 4 khung này:** chúng tạo thành một thiết kế đối chứng rõ ràng. **H3 với S2** — cùng là DGGS, khác hình dạng ô → tách riêng ảnh hưởng của *hình dạng*. **S2 với ô vuông** — cùng dạng tứ giác, khác hệ quy chiếu → tách riêng ảnh hưởng của *DGGS so với lưới phẳng*. Kinh–vĩ độ là mốc so sánh với cách làm phổ biến nhất hiện nay.

**Hai lựa chọn có chủ đích (không phải cắt bớt):**
- Không thêm geohash — về bản chất là cây tứ phân trên lưới kinh–vĩ độ, trùng với khung kinh–vĩ độ đã có.
- Không tự tính lại công thức NDWI → độ mặn — dùng thuật toán nghịch đảo đã công bố và kiểm chứng trong bài báo gốc là cách làm chặt chẽ hơn.

## Các cổng kiểm tra

| Cuối tuần | Phải có |
|---|---|
| 1 | 12 file lưới (4 khung × 3 độ phân giải) + xác nhận bộ Zenodo dùng được |
| 2 | Bộ đặc trưng dạng lưới cho cả 12 cấu hình |
| 3 | Dữ liệu dạng điểm đã đồng bộ vào lưới + tập đánh giá chung + đường cơ sở IDW |
| 4 | **Bảng sai số đầy đủ cho ĐBSCL** ← mang đi trình GVHD |
| 5 | Biểu đồ chính + bộ hình ĐBSCL |
| 6 | Kết quả 2 vùng đối chứng + kết quả bài toán dự báo |
| 7 | Toàn bộ kết quả đã tổng hợp, danh sách hạn chế, dàn ý bài báo |

Không qua cổng thì dừng lại xử lý, không chồng việc lên.

## Nguyên tắc làm việc

1. **Chưa thống nhất đề xuất thì chưa code** — tránh viết xong mới phát hiện sai hướng.
2. **Ghi lại mọi quyết định** vào mục "Nhật ký quyết định" cuối file tuần — viết bài báo cần đúng những thứ này.
3. **Gặp bế tắc quá 1 ngày thì báo** qua `GIAO_TIEP/` để cùng tìm hướng, không đâm mãi.
4. **Không có số thì không có kết luận** — mọi nhận định trong bài báo phải truy được về một file kết quả cụ thể.

## Phương án dự phòng

Giãn lịch là lựa chọn đầu tiên. Chỉ dùng các phương án dưới đây khi **không thể** làm được theo cách chính, và phải ghi rõ lý do:

| Tình huống | Cách xử lý |
|---|---|
| Bộ Zenodo thiếu dữ liệu do mây quá nhiều ở một số năm | Loại riêng các năm đó, ghi rõ tiêu chí loại. Nếu thiếu nhiều năm: dùng NDWI (quan sát trực tiếp) làm mục tiêu chính thay cho độ mặn |
| Thư viện S2 không cài được trên Windows | Chạy phần S2 trên WSL hoặc Google Colab |
| Không lấy được dữ liệu cho vùng đối chứng | Chọn vùng khác cùng đới khí hậu có dữ liệu tốt hơn |
| Kết quả cho thấy các khung không khác nhau đáng kể | **Vẫn công bố được** — đó là một phát hiện: lựa chọn khung không ảnh hưởng đáng kể ở quy mô này |

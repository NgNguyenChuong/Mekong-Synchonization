# Quỹ đạo dự án và milestone — file "la bàn"

> Cập nhật: 2026-09-29 · Chủ nhiệm: Trần Quốc An
>
> **Khi có mâu thuẫn hoặc nghi ngờ đang đi lệch: dừng lại, đọc file này, rồi hỏi An.**
> Thứ tự ưu tiên khi các tài liệu mâu thuẫn: **Thuyết minh đã nộp** (cam kết) → **file này** → `README.md` và `tuan-N.md` → `GIAO_TIEP/trao-doi.md`.

## 1. Đích đến (đọc riêng phần này là đủ nhớ)

**Một câu hỏi nghiên cứu:** khi đồng bộ dữ liệu đa nguồn (dạng lưới, dạng điểm, thưa theo thời gian) về một khung biểu diễn không gian chung, việc chọn khung đó (H3, S2, ô vuông, kinh–vĩ độ) ảnh hưởng thế nào đến độ chính xác của mô hình học máy — và ảnh hưởng đó phụ thuộc vào đặc tính gì của dữ liệu?

**Một sản phẩm chính:** bài báo WoS/Scopus có bảng so sánh 4 khung trên ĐBSCL, kiểm định thống kê, quan hệ giữa mức tự tương quan không gian của biến (Moran's I) và mức ảnh hưởng của khung, bóc tách phần hybrid, và kiểm chứng ở vùng khác.

**Ba kết quả đều công bố được** — không được chỉnh phân tích để "ra kết quả đẹp":
1. Các khung khác nhau có ý nghĩa → báo cáo khung nào, ở biến nào, vì sao.
2. Các khung không khác nhau đáng kể → báo cáo như một phát hiện: chọn khung theo tiêu chí kỹ thuật khác.
3. Hỗn hợp (khác ở biến này, không khác ở biến kia) → đó chính là quan hệ với Moran's I.

## 2. Sản phẩm cuối cùng (cam kết ở mục 14 của thuyết minh)

| # | Sản phẩm | Xong khi | Hạn |
|---|---|---|---|
| 1 | Bộ dữ liệu chuẩn hoá đa khung + báo cáo phân tích khám phá dữ liệu | Đủ 13 lưới, cùng phạm vi thời gian và cùng bộ cột; báo cáo chất lượng dữ liệu; đóng gói kèm DOI (Zenodo) | 12/2026 |
| 2 | Mã nguồn + giao diện trực quan hoá | `pytest` toàn bộ pass trong **một phiên**; tái lập được kết quả từ máy sạch; giao diện tối thiểu: xem được 4 khung và bản đồ sai số theo ô trên globe hiện có | 05/2027 |
| 3 | Bài báo WoS/Scopus | Nộp ~10/12/2026; **được đăng** trước nghiệm thu (yêu cầu tối thiểu của loại A) | 10/2027 |

## 3. Phạm vi KHOÁ

### 3.1 Đã cam kết — không cắt để kịp lịch (thiếu thời gian thì giãn lịch, tuần 7 là đệm)

- **Khung:** H3 · S2 · ô vuông (UTM 48N) · kinh–vĩ độ.
- **Độ phân giải:** ba mức ứng với H3 res 5, 6, 7. S2 tương ứng L9, L11, L12; thêm **S2 L10 làm lưới kẹp ở res 6** → **13 lưới**.
- **Vùng:** ĐBSCL (13 tỉnh, ranh giới chuẩn duy nhất `webapp/backend/data/mekong_delta_boundary.geojson`, giữ toàn bộ đảo) + 2 vùng đối chứng khác đới khí hậu và vĩ độ (tuần 6).
- **Nhãn:** bộ Zenodo 15653696 (19 mùa khô; loại năm theo quy tắc NoData đã thống nhất). Gọi đúng tên "độ mặn ước lượng bằng nghịch đảo viễn thám", **không gọi là ground truth**.
- **Biến mục tiêu:** lượng mưa, nhiệt độ, độ ẩm, bức xạ, NDWI, độ mặn.
- **Hybrid:** dòng chảy Tân Châu và Châu Đốc (API MRC) + mực nước biển (CMEMS). Thí nghiệm bóc tách chạy trên ~13 năm có đủ dữ liệu; các phần khác dùng đủ 19 năm.
- **Mô hình:** IDW, hồi quy tuyến tính, Random Forest, HistGradientBoosting, MLP.
- **Đơn vị phân tích:** (ô, mùa khô) là chính; (ô, ngày) là phụ cho 4 biến khí tượng.
- **Thiết lập bài toán:** ước lượng không gian là chính; dự báo ngắn hạn để kiểm chứng độ ổn định (tuần 6).

### 3.2 Ngoài phạm vi — KHÔNG làm, KHÔNG thêm

- Kiến trúc mô hình mới (GNN, mạng chuyên cho lưới lục giác) — đó là bài báo thứ hai.
- Dữ liệu đập MDM, ENSO/PDO, bão, sụt lún, SoilGrids, gió, thuỷ triều riêng.
- Dữ liệu trong Drive cũ, **trừ** hai thứ: cột lớp phủ của `H3_Static_Features.csv` và lưới `h3_grid_dbscl.geojson` (chỉ để kiểm tra chéo).
- Tự tính lại công thức NDWI → độ mặn (dùng nhãn Zenodo đã công bố).
- Thêm DGGS khác (geohash, rHEALPix…), thêm vùng hoặc biến ngoài danh sách.
- Mô hình thuỷ động lực, hệ thống dự báo nghiệp vụ, tích hợp webapp thời gian thực.

## 4. Giao thức khoá (đổi = làm hỏng tính so sánh)

- **Chọn ô:** ô thuộc lưới nếu tâm nằm trong ranh giới; **không cắt hình học ô**; ghi `overlap_frac`. Cùng một quy tắc cho cả 4 khung.
- **Kích thước ô:** khớp diện tích trắc địa với H3; neo gốc lưới ô vuông ở bội số của cạnh (toạ độ UTM), lưới kinh–vĩ độ ở bội số của Δ tính từ (0°, 0°).
- **Trích đặc trưng:** **theo diện tích** (`exactextract`), không lấy điểm tâm; lớp phủ là tỉ lệ % theo diện tích; ô không có pixel hợp lệ = NaN.
- **Mặt nạ:** lọc nước và đô thị (ESA WorldCover) như nhau cho cả 4 khung.
- **Đánh giá:** cùng một tập ~3.000 điểm (seed cố định) trong khối giữ riêng độc lập với lưới, nhãn đọc từ raster gốc 30 m; loại khỏi huấn luyện mọi ô giao vùng kiểm tra.
- **Đặc trưng:** bỏ NDWI khi mục tiêu là độ mặn (và ngược lại); toạ độ tâm ô không làm đặc trưng mặc định; `overlap_frac` không làm đặc trưng.
- **Thiếu dữ liệu:** không `fillna(0)`; trung vị chỉ tính trên tập huấn luyện.
- **Thống kê:** Wilcoxon theo cặp trên sai số tại từng điểm; hiệu chỉnh Holm trên **toàn bộ** tập p-value; báo cáo độ lớn hiệu ứng kèm khoảng tin cậy bootstrap.
- **Moran's I:** tính trên chuẩn chung (cùng điểm, cùng ma trận trọng số cho mọi biến), **không** tính trên từng khung.

## 5. Phân tích đã chốt trước (không thêm sau khi thấy kết quả)

1. So sánh sai số giữa 4 khung (Wilcoxon theo cặp, Holm, hiệu ứng + bootstrap).
2. Quan hệ Moran's I và tầm semivariogram với mức ảnh hưởng của khung (Spearman + bootstrap).
3. Tách hiệu ứng hình dạng (H3 với S2) và hiệu ứng DGGS so với lưới phẳng (S2 với ô vuông), có biến kiểm soát diện tích.
4. Bóc tách hybrid: cùng mô hình có và không có đặc trưng từ dữ liệu điểm.
5. Ảnh hưởng của vĩ độ lên khung kinh–vĩ độ (3 vùng).
6. Kiểm chứng trên bài toán dự báo ngắn hạn.

Mọi phân tích khác **phải gắn nhãn "thăm dò"** và không được dùng làm kết luận chính.

## 6. Milestone và checklist

Đánh dấu `[x]` chỉ khi có bằng chứng (nguyên văn lệnh và kết quả) trong `GIAO_TIEP/trao-doi.md`.

### M0 — Nền tảng · 29/09 – 05/10/2026 (T1-01, T1-02, T1-03)
- [ ] Mốc phục hồi (commit hoặc bản sao) cho `src/`, `tests/`, `scripts/`
- [ ] Zenodo: 19 file tải xong, MD5 khớp; bảng khảo sát thật (min, max, mean, p1, p50, p99; NoData theo footprint; số pixel bằng 0; số pixel > 28,013; chế độ cảm biến suy ra theo năm); danh sách năm bị loại kèm lý do; báo cáo metadata (kể cả Comment/Reply hoặc ghi "chưa đọc")
- [ ] Code: `cell_id`, `--target`, `--label-csv` tuỳ chọn; test dùng ID không phải H3; `pytest -q` toàn bộ trong 1 phiên pass; `tests/conftest.py` không rỗng
- [ ] 13 lưới cùng schema (`cell_id`, `geometry`, `overlap_frac`); bảng đối chiếu diện tích; các kiểm thử lưới pass
- [ ] **Cổng G1**

### M1 — Đặc trưng dạng lưới · → 12/10
- [ ] `main.py` chạy được nhiều lưới bằng biến môi trường; hành vi cũ giữ nguyên
- [ ] Raster 2000–2023 (ERA5-Land, Copernicus DEM, ESA WorldCover, MODIS NDVI, JRC mặt nước) đúng quy ước file và đã đổi đơn vị
- [ ] Trích theo diện tích; kiểm tay ≥ 3 ô mỗi khung
- [ ] 13 bộ đặc trưng cùng phạm vi ngày và cùng cột; báo cáo chất lượng

### M2 — Hybrid và đánh giá · → 19/10
- [ ] MRC (`CDO`, `TCH`) và CMEMS (4 điểm) đã lấy; xử lý `0.0`; tỉ lệ thiếu theo năm
- [ ] Mạng sông + khoảng cách dọc sông; đặc trưng hybrid cho 13 cấu hình dùng chung một đồ thị sông
- [ ] Đã bỏ `fillna(0)`
- [ ] Khối giữ riêng + điểm đánh giá; kiểm tra tự động: 0 ô huấn luyện giao vùng kiểm tra
- [ ] IDW; `evaluate.py` (theo điểm, Wilcoxon, Holm, bootstrap) kèm test
- [ ] Chạy thử 1 cấu hình đầu–cuối ra số hợp lý · **Cổng G2**

### M3 — Thực nghiệm ĐBSCL · → 26/10 (trình GVHD ~28/10)
- [ ] Đơn vị phân tích thống nhất; `run_experiments.py` tiếp tục được và ghi lỗi mà không dừng
- [ ] `results_all.csv` đủ 858 cấu hình chính (13 lưới × 6 biến × 11) và thí nghiệm phụ, hoặc ghi rõ lỗi từng cấu hình
- [ ] Bảng kiểm định (Holm) và bảng tổng hợp; trả lời: học máy có thắng IDW không
- [ ] Nội dung trình GVHD · **Cổng G3**

### M4 — Phân tích · → 02/11
- [ ] Moran's I và semivariogram trên chuẩn chung; xác nhận kích thước khối đánh giá đủ lớn (nếu không: dựng lại và chạy lại)
- [ ] Bóc tách hybrid; biểu đồ chính; tách hai hiệu ứng; 6 hình ≥ 300 dpi

### M5 — Mở rộng · → 09/11
- [ ] 2 vùng đối chứng (đúng múi UTM từng vùng) và phân tích vĩ độ
- [ ] Dự báo ngắn hạn: test chống rò rỉ tương lai pass; sửa `PersistenceBaseline`

### M6 — Tổng hợp · → 16/11
- [ ] Đệm cho việc trễ; tái lập 3 cấu hình trên máy sạch trùng khớp `results_all.csv`
- [ ] Đóng băng phiên bản (git tag); `requirements.txt` cố định phiên bản
- [ ] Bảng tổng hợp: mỗi phần một câu kết luận kèm đường dẫn số liệu gốc; danh sách hạn chế
- [ ] Dàn ý bài báo; 3 tạp chí theo thứ tự ưu tiên (kiểm tra danh mục Scopus chính thức, phí, thời gian phản biện) · **Cổng G4**

### M7 — Viết bài báo · 17/11 – 07/12
### M8 — Nộp · ~10/12/2026 (kèm công bố dữ liệu lên Zenodo)
### M9 — Sau nộp · 12/2026 – 10/2027
- [ ] Bộ dữ liệu + báo cáo khám phá dữ liệu (12/2026)
- [ ] Phản biện vòng 1 (~3–4/2027); nếu bị từ chối thì nộp lại ngay (kết quả ~8/2027)
- [ ] Giao diện trực quan hoá tối thiểu (05/2027)
- [ ] Bài báo được đăng trước nghiệm thu (10/2027)

## 7. Cổng đi/dừng

| Cổng | Khi | Đi tiếp nếu | Nếu không |
|---|---|---|---|
| G1 | 05/10 | Zenodo tải được và còn đủ năm sau khi lọc NoData (ngưỡng số năm chốt ở T1-01 sau khi thấy phân bố) | Phương án dự phòng 1, rồi họp GVHD |
| G2 | 19/10 | Một cấu hình chạy đầu–cuối ra số hợp lý | Báo GVHD; **không** chạy 858 cấu hình |
| G3 | 26–28/10 | Có bảng sai số đầy đủ cho cả 4 khung | Họp GVHD quyết định; không tự thu hẹp |
| G4 | 16/11 | Tái lập được, code đã đóng băng, đã chốt tạp chí | Lùi lịch viết bài tương ứng (vẫn còn chỗ cho một lần bị từ chối) |

**Phương án dự phòng đã chốt** (chỉ dùng khi không làm được theo cách chính; phải ghi lý do):
1. Nhãn Zenodo thiếu quá nhiều → loại năm theo tiêu chí; nếu thiếu nhiều năm → dùng NDWI làm mục tiêu chính.
2. Thư viện S2 không chạy trên Windows → chạy phần S2 trên WSL hoặc Colab.
3. Không lấy được dữ liệu cho vùng đối chứng → chọn vùng khác cùng đới khí hậu.
4. MRC thiếu năm → hybrid chạy trên tập con (đã chấp nhận).
5. Các khung không khác nhau đáng kể → vẫn công bố như một phát hiện.

## 8. Dấu hiệu đang lệch quỹ đạo

- Thêm nguồn dữ liệu, biến, vùng, khung hoặc mô hình không có trong mục 3.1.
- Đổi định nghĩa nhãn, quy tắc chọn ô, kích thước ô hoặc giao thức đánh giá sau khi đã chạy.
- Các khung được chấm trên tập điểm khác nhau; đặc trưng hoặc nhãn có rò rỉ.
- Sửa kỳ vọng của test cho khớp code thay vì sửa code.
- Xuất hiện phân tích không nằm trong mục 5 nhưng được dùng làm kết luận chính.
- Code hoặc file được tạo khi mục tương ứng chưa "ĐÃ THỐNG NHẤT".
- Trễ một tuần quá 3 ngày mà không có mục ghi chú trong `GIAO_TIEP`.
- File quan trọng có dung lượng 0 byte hoặc bị mất.

## 9. Quy tắc thay đổi phạm vi (CHG)

Muốn thay đổi bất cứ điều gì ở mục 3 hoặc 4:
1. Mở mục `[CHG-xx]` trong `GIAO_TIEP/trao-doi.md`: đổi gì, vì sao (kèm bằng chứng), ảnh hưởng đến câu hỏi nghiên cứu, thời gian, kinh phí và các sản phẩm ở mục 2.
2. Claude review; **An quyết định**; nếu chạm tới sản phẩm cam kết hoặc nghiệm thu thì hỏi GVHD.
3. Chỉ sau khi được duyệt mới sửa file này (ghi vào bảng dưới) và `KE_HOACH/`.

## 10. Định nghĩa "xong" cho mọi việc

- Có bằng chứng: nguyên văn lệnh đã chạy và kết quả — không chỉ lời khẳng định.
- Test liên quan pass; có test mới cho mọi hàm mới.
- Claude review "ĐẠT"; nhật ký quyết định đã ghi.
- File tạo ra không rỗng (`wc -c` > 0) và còn nguyên sau một khoảng thời gian.

## 11. Bảng thay đổi phạm vi

| Mã | Ngày | Thay đổi | Lý do | Duyệt bởi |
|---|---|---|---|---|
| — | — | Chưa có | — | — |

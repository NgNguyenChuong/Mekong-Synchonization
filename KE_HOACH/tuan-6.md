# Tuần 6 — Mở rộng vùng + kiểm chứng trên bài toán dự báo

**Thời gian:** 03/11 – 09/11/2026

**Điều kiện bắt đầu:** qua cổng kiểm tra tuần 5.

## Mục tiêu

Kiểm chứng xem kết luận rút ra từ ĐBSCL có **đứng vững khi đổi điều kiện** không:

1. **Đổi vùng:** 2 vùng khác đới khí hậu và khác vĩ độ.
2. **Đổi bài toán:** từ ước lượng không gian sang dự báo theo thời gian.

Hai phần này trả lời câu hỏi phản biện chắc chắn sẽ đặt ra: *"kết quả chỉ đúng cho ĐBSCL, hay đúng nói chung?"*

## Kết quả kỳ vọng

- [ ] 2 vùng đối chứng đã chốt, có lý do chọn
- [ ] Dữ liệu và 12 cấu hình lưới cho mỗi vùng
- [ ] Kết quả so sánh khung cho 2 vùng
- [ ] Phân tích ảnh hưởng của vĩ độ đến khung kinh–vĩ độ
- [ ] Code tạo đặc trưng trễ cho bài toán dự báo, không rò rỉ tương lai
- [ ] Kết quả so sánh khung trên bài toán dự báo

---

## Việc 1 — Chọn 2 vùng đối chứng (ngày 1, sáng)

Đây đúng là hướng nhóm đã đặt ra từ sau cuộc thi: *"benchmark cho từng khu vực nhất định — sa mạc, nhiều núi"*.

**Tiêu chí chọn:**

| Tiêu chí | Lý do |
|---|---|
| **Khác đới khí hậu** với ĐBSCL: một vùng khô hạn, một vùng núi cao | Kiểm tra kết luận có phụ thuộc kiểu khí hậu không |
| **Trải rộng vĩ độ:** ít nhất một vùng quanh **40–45°** | Khắc phục đúng hạn chế đã nêu ở tuần 1: ĐBSCL gần xích đạo nên khung kinh–vĩ độ gần như đều diện tích. Ở 45°, ô kinh–vĩ độ hẹp đi khoảng 30% theo chiều đông–tây — lúc đó mới thấy được hiện tượng méo |
| **Diện tích tương đương ĐBSCL** (~40.000 km²) | Giữ số ô gần bằng nhau — là một biến kiểm soát |
| **Dữ liệu toàn cầu phủ đủ** | ERA5-Land, Copernicus DEM, ESA WorldCover, MODIS — đều phủ toàn cầu, không phải xin cấp |

**Gợi ý để bàn** (chốt trong `GIAO_TIEP`): một vùng khô hạn vĩ độ cao như một phần sa mạc Gobi / Nội Mông (~42°N); một vùng núi như vùng núi Tây Bắc Việt Nam (~21–22°N — gần gũi, có ý nghĩa với bối cảnh trong nước) hoặc một đoạn dãy Alps (~46°N — vừa núi vừa vĩ độ cao).

**Có ba vùng trải từ ~10° đến ~45°** thì phân tích thêm được **ảnh hưởng của vĩ độ** — một kết quả rất rõ ràng về mặt hình học.

---

## Việc 2 — Chạy lại quy trình cho 2 vùng (ngày 1 chiều – ngày 3)

Nhờ đã gỡ gắn cứng ở tuần 1–2, quy trình chạy được cho vùng mới chỉ bằng cách **đổi file ranh giới**.

**Đặc tả:** với mỗi vùng: sinh 12 cấu hình lưới → xuất dữ liệu GEE → trích theo diện tích → tập đánh giá chung (chia khối như ĐBSCL) → chạy thí nghiệm.

**Hai điểm khác ĐBSCL:**
- **Biến mục tiêu chỉ gồm 4 biến khí tượng.** Bộ Zenodo chỉ phủ ĐBSCL, nên độ mặn và NDWI không có ở vùng khác.
- **Không có phần hybrid** (dòng chảy, mực nước biển) — dữ liệu trạm là đặc thù từng vùng. Phần này kiểm chứng **hiệu ứng khung lưới**, không kiểm chứng hybrid. Ghi rõ trong bài báo.

**Khối lượng mỗi vùng:** 4 khung × 3 độ phân giải × 4 biến × 11 lần chạy = **528 lần**, đơn vị mùa khô → nhanh. *Vùng khác không có "mùa khô" theo nghĩa của ĐBSCL — dùng một kỳ cố định cùng độ dài, chốt trong `GIAO_TIEP`.*

### Tiêu chí review
- Hệ toạ độ mét dùng cho vùng mới có đúng không? *EPSG:32648 là UTM 48N, chỉ đúng cho ĐBSCL. Mỗi vùng phải dùng **đúng múi UTM của nó** — sai múi là diện tích ô vuông sai.*
- Các tham số cố định (kích thước khối, dải khoảng cách của Moran's I) có giữ nguyên như ĐBSCL không? *Phải giữ nguyên để so được giữa các vùng.*

---

## Việc 3 — Phân tích ảnh hưởng của vĩ độ (ngày 4, sáng)

Với 3 vùng ở 3 vĩ độ khác nhau: vẽ **sai số của khung kinh–vĩ độ so với H3** theo vĩ độ.

**Kỳ vọng từ hình học:** khung kinh–vĩ độ kém dần khi lên vĩ độ cao, vì ô bị méo diện tích nhiều hơn.

**Nếu đúng như vậy:** đây là kết quả mạnh vì **có giải thích cơ chế rõ ràng**. **Nếu không:** càng đáng bàn luận — nghĩa là mô hình học máy đã tự bù được hiện tượng méo.

**Cập nhật biểu đồ chính ở tuần 5:** thêm điểm từ 2 vùng mới → số điểm tăng từ ~26 lên ~50. Kết luận chắc hơn nhiều.

---

## Việc 4 — Bài toán dự báo: tạo đặc trưng trễ (ngày 4 chiều – ngày 5) ⚠️

**Khác biệt so với phần trước:** ước lượng không gian = đoán giá trị ở **chỗ chưa thấy**, cùng thời điểm. Dự báo = đoán giá trị ở **thời điểm tương lai**, từ thông tin quá khứ.

**Phần này là code mới hoàn toàn** — code hiện có chỉ làm ước lượng không gian.

**Đặc tả** — module mới `src/training/forecast_features.py`:
- **Đặc trưng trễ:** giá trị tại t-1, t-3, t-7
- **Trung bình trượt:** 7 ngày, 30 ngày
- **Chuẩn khí hậu:** trung bình nhiều năm của cùng ngày trong năm — **chỉ tính trên các năm huấn luyện**
- **Mục tiêu dịch theo tầm dự báo:** h = 1, 3, 7 ngày

**Phạm vi:** 4 biến khí tượng, theo ngày, ĐBSCL, res 5 và 6, HistGB 3 seed.

> ⚠️ **Rò rỉ tương lai — lỗi nguy hiểm nhất của bài toán dự báo:** mọi đặc trưng tại thời điểm t **chỉ được dùng dữ liệu đến t**. Lỗi hay gặp: trung bình trượt dùng cửa sổ **căn giữa** (lấy cả dữ liệu sau t), hoặc chuẩn khí hậu tính trên **toàn bộ** các năm kể cả năm kiểm tra. Cả hai đều làm mô hình trông tốt giả tạo.

### Tiêu chí review
- **Viết test tự động chống rò rỉ:** đổi giá trị của mọi ngày sau t thành số rác, đặc trưng tại t **phải không đổi**. *Bắt buộc có test này.*
- Trung bình trượt có phải cửa sổ **lùi** (chỉ dùng quá khứ) không?
- Chuẩn khí hậu có chỉ tính trên các năm huấn luyện không?

---

## Việc 5 — Chạy thí nghiệm dự báo (ngày 6)

**Chia tập:** theo thời gian — huấn luyện trên các năm đầu, kiểm tra trên các năm cuối. Dùng `time_based_split` đã có.

**Đường cơ sở cho bài toán dự báo:**
- **Persistence** — lấy giá trị hôm nay làm dự báo. **Ở bài toán này persistence mới có nghĩa**, khác với bài toán ước lượng không gian.
- **Climatology** — chuẩn khí hậu của ngày cần dự báo.

> ⚠️ **Sửa lỗi `PersistenceBaseline` trước khi dùng.** Bản hiện tại lấy giá trị **cuối cùng của tập huấn luyện** rồi dùng cho **mọi ngày** trong tập kiểm tra. Đúng ra phải lấy giá trị **liền trước từng ngày cần dự báo**. Nếu không sửa, đường cơ sở này yếu đi nhanh theo thời gian, làm mô hình chính trông tốt hơn thực tế.

**Câu hỏi cần trả lời:** hiệu ứng khung lưới tìm được ở bài toán ước lượng không gian **có còn** ở bài toán dự báo không?
- Còn → kết luận mạnh: không phụ thuộc loại bài toán.
- Mất → cũng là phát hiện: ảnh hưởng của khung chủ yếu nằm ở chiều không gian, còn khi dự báo thì thông tin theo thời gian lấn át.

---

## Cổng kiểm tra

1. Đủ kết quả cho 2 vùng đối chứng chưa?
2. Ảnh hưởng của vĩ độ đến khung kinh–vĩ độ ra sao — một câu.
3. Test chống rò rỉ tương lai có pass không?
4. Hiệu ứng khung có còn ở bài toán dự báo không — một câu.

## Nhật ký quyết định

-

# Tuần 3 — Đồng bộ hybrid + tập đánh giá + đường cơ sở

**Thời gian:** 13/10 – 19/10/2026

**Điều kiện bắt đầu:** qua cổng kiểm tra tuần 2 (12 bộ đặc trưng dạng lưới).

**Đây là tuần lõi của hướng hybrid** — phần đóng góp mới của đề tài nằm ở đây.

## Mục tiêu

1. Đưa dữ liệu **dạng điểm** (dòng chảy, mực nước biển) vào các khung lưới bằng quy tắc có cơ sở vật lý.
2. Dựng tập đánh giá chung, bảo đảm so sánh công bằng giữa 4 khung.
3. Cài đặt đúng bộ đường cơ sở cho bài toán ước lượng không gian.
4. Hoàn thiện module đánh giá: đánh giá theo điểm, kiểm định thống kê.

## Kết quả kỳ vọng

- [ ] Dữ liệu dòng chảy + mực nước biển cho toàn bộ phạm vi thời gian
- [ ] Mạng sông dạng đồ thị + khoảng cách dọc sông cho mọi ô của 12 cấu hình
- [ ] 12 bộ dữ liệu hợp nhất đầy đủ (lưới + điểm)
- [ ] Đã bỏ `fillna(0)`
- [ ] Khối giữ riêng + tập điểm đánh giá chung
- [ ] Đường cơ sở IDW
- [ ] `evaluate.py` có đánh giá theo điểm + kiểm định theo cặp + hiệu chỉnh so sánh bội

---

## Việc 1 — Thu thập dữ liệu dạng điểm (ngày 1)

Công cụ đã có sẵn và chạy được:

| Nguồn | Công cụ | Ghi chú |
|---|---|---|
| Dòng chảy / mực nước Tân Châu, Châu Đốc | `component/getWaterLevel.py` | Gọi API MRC (`ffw.mrcmekong.org`), tách mùa mưa / mùa khô |
| Chuẩn hoá theo trạm | `component/format_waterlevel.py` | Mã trạm lấy từ tên file (ví dụ `CDO` = Châu Đốc) |
| Mực nước biển | `component/format_sealevel.py` | Dữ liệu CMEMS, trường ADT theo toạ độ |

**Đặc tả:** chạy lấy đủ **2000–2023**. Ghi lại các khoảng thời gian trạm bị mất số liệu — không tự nội suy ở bước này, chỉ ghi nhận.

### Tiêu chí review
- API MRC có trả về đủ 24 năm không, hay chỉ vài năm gần đây? *Nếu thiếu, phải ghi rõ phạm vi thực tế có được — không được lặng lẽ thu hẹp.*
- Đơn vị mực nước và mốc cao độ (datum) của hai trạm có giống nhau không?

---

## Việc 2 — Đồng bộ hybrid: đưa dữ liệu điểm vào lưới (ngày 2–3) ⚠️ LÕI CỦA ĐỀ TÀI

**Vấn đề gốc:** dữ liệu khí tượng là lưới, phủ kín mọi nơi. Nhưng dòng chảy chỉ đo tại **2 điểm** ở thượng nguồn, mực nước biển chỉ có ở **các điểm ven biển**. Muốn đưa vào mô hình theo ô thì phải có **quy tắc chuyển** từ điểm sang ô.

**Điểm mấu chốt cho bài báo:** kết quả của quy tắc chuyển này **phụ thuộc vào khung lưới** — vì nó dựa trên vị trí tâm ô và cách các ô nối với nhau dọc theo sông. Nghĩa là **khung biểu diễn không chỉ ảnh hưởng đến dữ liệu lưới, mà còn ảnh hưởng đến cách dữ liệu điểm được hợp nhất**. Đây chính là phần "hybrid synchronization" chưa ai khảo sát.

### Bước 1 — Dựng mạng sông dạng đồ thị

- **Nguồn:** HydroRIVERS (hydrosheds.org) — mạng sông dạng vector toàn cầu, dữ liệu mở.
- Dựng đồ thị bằng `networkx`: mỗi đoạn sông là một cạnh, trọng số = chiều dài đoạn.

> Giới hạn cần ghi lại: HydroRIVERS nắm được các nhánh chính (Tiền, Hậu và các phân lưu lớn) nhưng **không có hệ thống kênh rạch nhỏ** của ĐBSCL.

### Bước 2 — Quy tắc chuyển dòng chảy (2 trạm thượng nguồn → ô)

**Vì sao không gán trạm gần nhất theo đường chim bay:** nước đi theo sông, không đi theo đường thẳng. Hai ô cách trạm cùng một khoảng theo đường thẳng có thể cách nhau rất xa theo đường sông.

**Đặc tả:**
1. Gắn mỗi tâm ô vào nút sông gần nhất.
2. Xác định ô thuộc **nhánh Tiền** (trạm Tân Châu) hay **nhánh Hậu** (trạm Châu Đốc) theo đường đi ngắn nhất trên đồ thị.
3. Tạo cho mỗi ô các đặc trưng:
   - giá trị dòng chảy của trạm tương ứng
   - **khoảng cách dọc sông** từ trạm tới ô
   - dòng chảy **trễ theo thời gian truyền** (ước lượng từ khoảng cách dọc sông chia cho vận tốc dòng chảy điển hình — chốt giá trị vận tốc trong `GIAO_TIEP`)

### Bước 3 — Quy tắc chuyển mực nước biển (điểm ven biển → ô)

**Đặc tả:** với mỗi ô, tạo các đặc trưng:
- mực nước biển tại **cửa sông gần nhất theo đường sông** (không phải điểm ven biển gần nhất theo đường thẳng)
- **khoảng cách dọc sông từ cửa sông vào tới ô** — đây là đặc trưng then chốt cho độ mặn, vì nước mặn xâm nhập ngược theo sông

### Tiêu chí review
- Có dùng khoảng cách đường chim bay ở đâu không? *Phải là khoảng cách dọc sông.*
- Ô **không gắn được vào mạng sông** (quá xa sông) xử lý thế nào? *Phải có quy tắc rõ ràng và áp dụng giống nhau cho cả 4 khung.*
- **Cả 4 khung dùng chung một đồ thị sông** không? *Bắt buộc — khác nhau duy nhất phải là vị trí tâm ô.*
- Thời gian truyền có hợp lý về độ lớn không? *Từ Tân Châu ra biển khoảng 200–250 km — trễ vài ngày là hợp lý, trễ vài tuần là sai.*

---

## Việc 3 — Hợp nhất và bỏ `fillna(0)` (ngày 4, sáng)

Ghép đặc trưng điểm vào 12 bộ đặc trưng lưới từ tuần 2.

**Bỏ `fillna(0)`** ở `src/training/train.py` (dòng 131 và 134).
- *Vì sao:* với lượng mưa, **0 là giá trị có nghĩa** ("không mưa"). Điền 0 vào chỗ thiếu khiến mô hình không phân biệt được "không mưa" với "không có dữ liệu".
- *Thay bằng:* HistGradientBoosting xử lý NaN trực tiếp. Các mô hình khác: điền trung vị tính **chỉ trên tập huấn luyện** (tránh rò rỉ) + thêm cột đánh dấu "bị thiếu".

### Tiêu chí review
- Trung vị có được tính trên tập huấn luyện, hay trên toàn bộ dữ liệu? *Tính trên toàn bộ là rò rỉ thông tin từ tập kiểm tra.*

---

## Việc 4 — Tập đánh giá chung (ngày 4 chiều)

**Đây là thứ bảo đảm so sánh giữa 4 khung là công bằng.**

1. **Chia khối không gian:** chia ĐBSCL thành các khối vuông liền kề (đề xuất cạnh 50 km), giữ riêng khoảng 20% số khối làm vùng kiểm tra, chọn ngẫu nhiên với seed cố định. Lưu `data/eval/holdout_blocks.geojson`.
   - *Vì sao chia theo khối:* dữ liệu có tự tương quan không gian — ô kiểm tra nằm sát ô huấn luyện thì mô hình "nhìn trộm" được, cho kết quả quá lạc quan (Roberts et al. 2017; Ploton et al. 2020).
   - *Kích thước khối phải lớn hơn tầm tự tương quan của dữ liệu.* Tầm này sẽ được đo ở tuần 5 — nếu lúc đó thấy khối 50 km nhỏ hơn tầm tự tương quan, **phải dựng lại khối và chạy lại**. Ghi chú sẵn rủi ro này.
2. **Sinh điểm đánh giá:** gieo ngẫu nhiên khoảng 3.000 điểm trong vùng kiểm tra, seed cố định. Lưu `data/eval/eval_points.geojson` (chỉ vị trí, chưa có giá trị).
3. **Giá trị tham chiếu:** với mỗi biến mục tiêu, đọc giá trị tại các điểm này **trực tiếp từ raster gốc ở độ phân giải gốc**. Đây là tham chiếu chung, không phụ thuộc khung nào.
4. **Quy tắc loại trừ:** với **mọi khung**, ô nào **giao** với vùng kiểm tra đều bị loại khỏi tập huấn luyện — kể cả ô chỉ giao một phần.

### Tiêu chí review
- Khối giữ riêng có được định nghĩa **độc lập với khung lưới** không? *Phải định nghĩa trên toạ độ địa lý, dùng chung cho cả 4 khung.*
- Có ô huấn luyện nào giao với vùng kiểm tra không? *Viết kiểm tra tự động — kết quả phải bằng 0.*

---

## Việc 5 — Đường cơ sở (ngày 5)

**Vấn đề:** `PersistenceBaseline` lấy giá trị cuối cùng **của chính ô đó** trong tập huấn luyện. Nhưng với cách giữ riêng theo khối, **ô trong vùng kiểm tra không có lịch sử huấn luyện nào** → nó chỉ trả về trung bình toàn cục. **Vô nghĩa trong thiết lập này.** Giữ lại cho bài toán dự báo ở tuần 6, không dùng cho bài toán ước lượng không gian.

**Bộ đường cơ sở cho bài toán ước lượng không gian:**

| Đường cơ sở | Ý nghĩa |
|---|---|
| Trung bình toàn cục | Mức sàn tuyệt đối |
| Climatology theo tháng | Có sẵn |
| **IDW** — nội suy nghịch đảo khoảng cách | **Quan trọng nhất.** Nếu học máy không thắng được IDW thì không đóng góp gì hơn nội suy thuần tuý |

**Đặc tả IDW:** với mỗi ô cần dự đoán, lấy trung bình có trọng số `1 / khoảng_cách^p` của `k` ô huấn luyện gần nhất. Dùng `scipy.spatial.cKDTree`. Chọn `p` và `k` bằng kiểm định chéo **trên tập huấn luyện**, không chọn trên tập kiểm tra.

**Điểm đáng lưu ý:** IDW dùng khoảng cách giữa các tâm ô, nên **bản thân IDW cũng chịu ảnh hưởng của khung lưới** — kết quả IDW trên 4 khung cũng là một phần của phép so sánh.

### Tiêu chí review
- `p` và `k` có bị chọn dựa trên kết quả tập kiểm tra không? *Nếu có là rò rỉ.*
- Khoảng cách có tính trong hệ toạ độ mét không?

---

## Việc 6 — Hoàn thiện `evaluate.py` (ngày 6)

**a) Đánh giá theo điểm chung:**
- Nhận: dự đoán theo ô, file lưới, `eval_points.geojson`, giá trị tham chiếu
- Với mỗi điểm: tìm ô chứa nó (spatial join), lấy dự đoán của ô đó
- Trả về **sai số tại từng điểm** — cần cho kiểm định

**b) Kiểm định theo cặp:**
- Cả 4 khung chấm trên **cùng một bộ điểm** → dùng kiểm định theo cặp trên sai số tuyệt đối tại từng điểm.
- **Wilcoxon signed-rank** (`scipy.stats.wilcoxon`) — không đòi phân phối chuẩn, phù hợp với sai số thường bị lệch.
- **Độ lớn hiệu ứng:** chênh lệch trung vị sai số + khoảng tin cậy bootstrap. Không chỉ báo p-value.

**c) Hiệu chỉnh so sánh bội:** 6 cặp khung × 6 biến × 3 độ phân giải × nhiều mô hình = hàng trăm kiểm định. Chạy đủ nhiều kiểm định thì kiểu gì cũng có vài cái "có ý nghĩa" do ngẫu nhiên. Áp dụng **Holm** (`statsmodels.stats.multitest.multipletests`) cho toàn bộ tập p-value.

**d) Viết test** cho các hàm mới trong `tests/`.

### Tiêu chí review
- Spatial join có xử lý điểm rơi đúng **cạnh chung** của hai ô không? *Phải có quy tắc cố định, không để phụ thuộc thứ tự dữ liệu.*
- Hiệu chỉnh Holm có áp dụng trên **toàn bộ** tập p-value, hay chia nhỏ từng nhóm? *Chia nhỏ là hiệu chỉnh thiếu.*
- Bootstrap có seed cố định không?

---

## Ngày 7 — Cổng kiểm tra

1. Đủ 12 bộ dữ liệu hợp nhất (lưới + điểm) chưa?
2. Kiểm tra tự động loại trừ vùng kiểm tra có ra kết quả 0 không?
3. IDW và `evaluate.py` đã có test, test pass chưa?
4. Chạy thử toàn bộ quy trình trên **một** cấu hình nhỏ (ví dụ `h3_r5`, một biến, một mô hình) từ đầu đến cuối — có ra được con số hợp lý không?

**Câu 4 là bắt buộc** — phát hiện lỗi trên một cấu hình rẻ hơn nhiều so với phát hiện sau khi chạy hàng trăm cấu hình ở tuần 4.

## Nhật ký quyết định

-

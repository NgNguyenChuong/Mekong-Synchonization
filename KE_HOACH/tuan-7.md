# Tuần 7 — Tổng hợp + đệm

**Thời gian:** 10/11 – 16/11/2026

## Mục tiêu

1. **Đệm** cho những việc trễ từ các tuần trước — gần như chắc chắn sẽ có.
2. Tổng hợp toàn bộ kết quả, kiểm tra tái lập.
3. Chuẩn bị đầy đủ để bắt tay viết bài báo ngay tuần sau.

## Kết quả kỳ vọng

- [ ] Mọi việc trễ từ tuần 1–6 đã xong
- [ ] Đã kiểm tra tái lập thành công
- [ ] Phiên bản code đã được đóng băng
- [ ] Bảng tổng hợp toàn bộ kết quả
- [ ] Danh sách hạn chế
- [ ] Dàn ý bài báo + danh sách tạp chí dự kiến

---

## Ngày 1–3 — Đệm

**Không lên kế hoạch gì cho ba ngày này.** Dùng để hoàn thành việc trễ. Nếu không có gì trễ thì chuyển sang các việc bên dưới sớm hơn.

Nếu việc trễ cần nhiều hơn 3 ngày: **giãn tuần này ra và lùi lịch viết bài báo tương ứng** — không bỏ việc. Lịch tổng vẫn còn khoảng cho một lần bị từ chối.

---

## Việc 1 — Kiểm tra tái lập (ngày 4, sáng)

**Vì sao:** phản biện có thể yêu cầu chạy lại hoặc chỉnh một phần. Nếu lúc đó không tái lập được kết quả cũ thì rất khó xử lý.

**Đặc tả:**
1. Clone repo ra một thư mục **mới hoàn toàn**, cài lại môi trường từ `requirements.txt`.
2. Chạy lại **3 cấu hình bất kỳ** từ đầu đến cuối.
3. So kết quả với `results_all.csv` — **phải trùng khớp** (cùng seed thì cùng số).

**Nếu không khớp:** tìm nguyên nhân ngay — thường là thiếu seed ở đâu đó, thứ tự dữ liệu không cố định, hoặc phiên bản thư viện khác. Đây là lỗi phải sửa trước khi viết bài báo.

### Tiêu chí review
- `requirements.txt` có ghi **phiên bản cụ thể** của từng thư viện không? *Không ghi phiên bản thì vài tháng sau cài lại sẽ ra phiên bản khác, có thể ra số khác.*

---

## Việc 2 — Đóng băng phiên bản code (ngày 4, chiều)

Gắn thẻ git cho phiên bản đã tạo ra toàn bộ kết quả, ví dụ `v1.0-ket-qua-bai-bao`.

**Vì sao quan trọng:** mọi con số trong bài báo phải truy được về **đúng phiên bản code** đã tạo ra nó. Sau này sửa code thì vẫn quay lại được phiên bản gốc.

---

## Việc 3 — Bảng tổng hợp toàn bộ kết quả (ngày 5)

Gom kết quả từ tuần 4, 5, 6 thành một tài liệu duy nhất `KE_HOACH/ket-qua/tong-hop.md`, gồm:

| Phần | Nguồn |
|---|---|
| So sánh 4 khung — ĐBSCL | Tuần 4 |
| Quan hệ Moran's I với ảnh hưởng của khung | Tuần 5, bổ sung điểm từ tuần 6 |
| Tách hiệu ứng hình dạng và hiệu ứng DGGS | Tuần 5 |
| Bóc tách phần hybrid | Tuần 5 |
| So sánh giữa 3 vùng + ảnh hưởng vĩ độ | Tuần 6 |
| Bài toán dự báo | Tuần 6 |

Với **mỗi phần**, viết đúng **một câu kết luận**, kèm đường dẫn tới file số liệu gốc. Đây sẽ là bộ khung cho mục Kết quả của bài báo.

**Nguyên tắc:** không có số thì không có kết luận. Mọi câu kết luận phải chỉ ra được con số cụ thể ở file nào.

---

## Việc 4 — Danh sách hạn chế (ngày 6, sáng)

Viết sẵn cho mục Thảo luận. Mỗi hạn chế **kèm một hướng khắc phục** — biến thành "hướng phát triển" ở cuối bài.

Các hạn chế đã biết từ trước:
1. **Nhãn độ mặn là ước lượng viễn thám**, không phải số đo thực địa — sai số của thuật toán nghịch đảo lan vào kết quả.
2. **Độ phân giải bị giới hạn** bởi dữ liệu khí tượng ~9 km — ở res 7 các biến khí tượng bị phân giải quá mịn so với nguồn.
3. **Diện tích ô S2 không khớp được hoàn toàn** với các khung khác ở res 6, do S2 chia 4 còn H3 chia 7 — đã xử lý bằng biến kiểm soát, nhưng vẫn là một giới hạn.
4. **Mạng sông HydroRIVERS** không có hệ thống kênh rạch nhỏ của ĐBSCL — khoảng cách dọc sông chỉ tính trên các nhánh chính.
5. **Phần hybrid chỉ kiểm chứng được ở ĐBSCL** — vùng đối chứng không có dữ liệu trạm tương ứng.
6. **Số vùng và số biến còn hạn chế** — quy luật tìm được cần thêm kiểm chứng.

Bổ sung các hạn chế phát sinh trong quá trình làm — xem mục "Nhật ký quyết định" của các tuần.

---

## Việc 5 — Dàn ý bài báo và chọn tạp chí (ngày 6 chiều – ngày 7)

### Dàn ý

1. **Giới thiệu** — vấn đề, khoảng trống, câu hỏi nghiên cứu, đóng góp
2. **Tổng quan** — học máy cho dữ liệu môi trường; DGGS; đánh giá mô hình trên dữ liệu không gian
3. **Dữ liệu** — vùng nghiên cứu, nguồn, cách đồng bộ hybrid
4. **Phương pháp** — 4 khung, cơ chế bảo đảm so sánh công bằng, mô hình, giao thức đánh giá, kiểm định
5. **Kết quả** — theo bảng tổng hợp ở Việc 3
6. **Thảo luận** — giải thích cơ chế, hạn chế, khuyến nghị lựa chọn khung
7. **Kết luận**

**Nêu đóng góp đúng mức:** đây là đóng góp dạng **nghiên cứu so sánh thực nghiệm có kiểm soát**, không phải đề xuất thuật toán mới. Viết khoa trương sẽ bị phản biện bắt.

### Tiêu chí chọn tạp chí

| Tiêu chí | Lý do |
|---|---|
| Thuộc danh mục **WoS hoặc Scopus** | Yêu cầu bắt buộc của đề tài loại A |
| **Thời gian phản biện ngắn** | Lịch chỉ chịu được một lần bị từ chối. Tạp chí phản biện trên 6 tháng là không phù hợp |
| **Phí đăng bài** trong khả năng | Nhiều tạp chí truy cập mở thu 2.000–3.000 USD, vượt xa kinh phí 20 triệu đồng. Ưu tiên tạp chí theo mô hình đăng ký, không thu phí nếu không chọn truy cập mở |
| **Đúng lĩnh vực** | Tin học địa không gian, khoa học dữ liệu Trái Đất, viễn thám, mô hình hoá môi trường |

**Kiểm tra bắt buộc:** tra tên tạp chí **trực tiếp trên danh mục nguồn Scopus chính thức** tại thời điểm nộp — không tin thông tin trên website tạp chí. Tránh tạp chí "săn mồi" hứa duyệt trong vài tuần, và tạp chí đã bị loại khỏi danh mục.

**Chốt danh sách 3 tạp chí theo thứ tự ưu tiên** và xin ý kiến GVHD — để nếu bị từ chối ở tạp chí thứ nhất thì chuyển ngay sang tạp chí thứ hai, không mất thời gian chọn lại.

### Chuẩn bị công bố dữ liệu

Bộ dữ liệu đã đồng bộ đa khung là **sản phẩm số 1 trong thuyết minh**. Chuẩn bị công bố lên **Zenodo để lấy DOI**, đi kèm bài báo — vừa là sản phẩm được trích dẫn riêng, vừa đáp ứng yêu cầu công khai dữ liệu mà nhiều tạp chí đòi hỏi. Kiểm tra giấy phép của các nguồn dữ liệu gốc trước khi công bố lại.

---

## Cổng kiểm tra — sẵn sàng viết bài báo chưa?

1. Mọi việc trễ đã xong chưa?
2. Tái lập thành công chưa?
3. Code đã đóng băng phiên bản chưa?
4. Bảng tổng hợp có đủ các phần, mỗi phần một câu kết luận kèm số liệu gốc chưa?
5. Đã chốt được danh sách 3 tạp chí chưa?

Qua được cả 5 câu thì bắt đầu viết bài báo từ ~17/11.

## Nhật ký quyết định

-

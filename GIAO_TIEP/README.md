# Quy trình phối hợp Antigravity – Claude

> **Đọc trước khi làm bất cứ việc gì:** [`../KE_HOACH/00-quy-dao-va-milestone.md`](../KE_HOACH/00-quy-dao-va-milestone.md). Muốn đổi phạm vi thì phải mở mục `[CHG-xx]` trong `trao-doi.md` (mục 9 của file đó), không tự ý đổi.

## Vai trò

| Bên | Làm gì |
|---|---|
| **Antigravity** | Đề xuất cách làm → viết code → báo cáo kèm bằng chứng |
| **Claude** | Review đề xuất → review code → chỉ ra chỗ cần sửa |
| **An** | Chuyển lượt giữa hai bên, quyết định khi hai bên không thống nhất |

Mọi trao đổi ghi vào **một file duy nhất**: [`trao-doi.md`](trao-doi.md).

## Luồng một đầu việc

```
ĐỀ XUẤT ──► Claude review ──► CẦN SỬA ĐỀ XUẤT ─┐
   ▲                                            │
   └────────────────────────────────────────────┘
                     │ đồng ý
                     ▼
              ĐÃ THỐNG NHẤT ──► Antigravity code
                                      │
                                      ▼
                          ĐÃ CODE — CHỜ REVIEW ──► Claude review ──► CẦN SỬA CODE ─┐
                                      ▲                                             │
                                      └─────────────────────────────────────────────┘
                                                        │ đạt
                                                        ▼
                                                    HOÀN TẤT
```

## Năm quy tắc

1. **Chưa "ĐÃ THỐNG NHẤT" thì chưa code.** Tránh viết xong mới phát hiện sai hướng.
2. **Mỗi mục là một thay đổi vừa phải** — không gộp "làm hết tuần 1" vào một mục. Mục càng nhỏ, review càng kỹ và càng nhanh.
3. **Báo cáo phải có bằng chứng, không chỉ lời khẳng định.** "Test pass" phải kèm **nguyên văn kết quả chạy test**. "Đã kiểm tra tay" phải kèm **con số đã tính**. Không có bằng chứng thì coi như chưa làm.
4. **Làm khác đề xuất thì phải ghi rõ** ở mục "Lệch so với đề xuất" — kể cả khác nhỏ.
5. **Review đọc mục "Tiêu chí review"** trong file tuần tương ứng ở `KE_HOACH/`. Antigravity nên đọc mục này **trước khi viết code**.

## Cách chuyển lượt

Hai bên không nói chuyện trực tiếp — đều đọc và ghi vào `trao-doi.md`:

1. Antigravity ghi xong phần của mình → An nhắn Claude: *"review mục T1-02"*
2. Claude đọc file, review, ghi kết quả vào đúng mục đó
3. An chuyển lại cho Antigravity

## Đánh số

`T{tuần}-{số thứ tự}` — ví dụ `T1-01`, `T1-02`, `T3-05`.

## Khi hai bên không thống nhất

Mỗi bên ghi lý do vào mục đó, **An quyết định**. Ghi quyết định cuối cùng và lý do vào mục "Nhật ký quyết định" của file tuần tương ứng.

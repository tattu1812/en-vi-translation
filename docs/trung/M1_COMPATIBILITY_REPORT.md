# M1 — Báo cáo tương thích proposal và Data Pipeline V3

Ngày 21/11/2026; repository `C:\Users\MSI\Desktop\en-vi-translation`, branch `trungnguyen`, HEAD `9ecef1a32f19767d1176b959a3326a8f95e870aa`. Nguồn chuẩn: `docs/proposal/main.tex` (PDF đi kèm `docs/proposal/main.pdf`); proposal được nhóm phê duyệt theo chỉ dẫn người yêu cầu. Tài liệu M1 chi tiết chưa có bằng chứng review. VERIFIED = quan sát code/file hoặc test như nêu; PROPOSED — PENDING APPROVAL = chi tiết chưa được proposal quy định; UNKNOWN = chưa có bằng chứng.

## Đối chiếu proposal với M1 trước khi sửa

| Parameter | Approved Proposal | Current M1 trước sửa | Required Correction / trạng thái mới |
|---|---|---|---|
| Encoder | 2-layer BiLSTM, 512/hướng | 2-layer, 256/hướng | Đổi sang 512/hướng, output 1024 |
| Decoder | 2-layer LSTM hidden 512 | Đúng 2-layer 512 | Giữ |
| Embedding | 512, chung encoder/decoder | 256 shared | Đổi thành 512 shared |
| Attention | Luong general, input feeding | Bahdanau additive trước LSTM | Đổi Luong general sau decoder, feed attentional vector bước trước |
| Bridge | Không quy định | Linear+tanh 512→512 | D01: concat 1024→512 cho từng tầng/h/c, `TRUNG_ACCEPTED — PENDING_TEAM_REVIEW` |
| Output tying | Decoder embedding dùng chung lớp output | Tied qua adapter 256 | Tied trực tiếp `a_t[512] @ E.T`; cách tạo `a_t` là đề xuất |
| Dropout | 0.3 | 0.3 | Giữ; vị trí cụ thể chờ duyệt |
| Optimizer/clip | Adam 1e-3, clip 1.0 | Chưa ấn định | Ghi đúng theo proposal |
| Loss / teacher forcing | Label smoothing 0.1; không quy định cụ thể loss hay tỉ lệ teacher forcing | CE ignore PAD, smoothing 0; teacher forcing 100% | Smoothing 0.1; CE ignore PAD và 100% teacher forcing là PROPOSED — PENDING APPROVAL |
| Batch/training | Batch size không nêu; mixed precision, ≤30 epochs, patience 5, checkpoint tốt nhất validation, một seed/mô hình | Batch và config chưa chốt | Bổ sung chính xác; batch/scheduler/tiêu chí cụ thể chưa nêu không đoán |
| Decoding | Beam 5, length normalization, max source length + 50; greedy so beam ở A1 | Chưa có implementation | Giữ yêu cầu; công thức length penalty, tie-breaking và interface chờ duyệt |
| Metrics | SacreBLEU corpus và theo 4 nhóm chiều dài, số tham số, thời gian, GPU peak | BLEU nêu chung | Ghi đủ; chưa có kết quả model |
| Estimated parameters | ~22M, đo lại bằng code | Không đo | Ghi là ước tính, đo khi có model; bridge và feeding ảnh hưởng kết quả |

## Bằng chứng thực tế

- `docs/proposal/main.tex:57-69,79-108,113-151` quy định các thông số nêu trên. Proposal ghi 133317 cặp train gốc; `docs/dataset_manifest.md` và data V3 cho 131000 sau lọc. Filter 100 trong `scripts/clean_iwslt15.py` là whitespace units, không phải BPE; validation/test không bị lọc chiều dài.
- `src/data/dataloader.py:18-22,30-80,95-149` tạo source + EOS, target input BOS+tokens và target target tokens+EOS, pad phải, mask `src_ids.eq(PAD_ID)`/`tgt_input_ids.eq(PAD_ID)` — **True=PAD**. Lấy cùng model tokenizer cho cả EN/VI. Không có truncation BPE.
- `scripts/train_tokenizer.py` gộp train hai ngôn ngữ, BPE vocab 8000 và IDs 0/1/2/3. `shared_assets/tokenizer/spm_envi_bpe_8000.model` được load trực tiếp trong lần test này.
- `README.md:261` nói mỗi model tự triển khai loop; `Sơ đồ công việc nhóm.html:174-176,193-195` giao Đức D3 loop chung và nối tới M4. Chưa thấy loop hoặc hai model trong `src/` hiện có. Đây là phân công cần thống nhất, không ngăn bắt đầu modules M2.

## Runtime validation — VERIFIED

Dùng `.venv` tạo từ `C:\Users\MSI\miniconda3\envs\deep_learning` với `--system-site-packages`, CPU; Python 3.11.16, PyTorch `2.14.1+cpu` kế thừa từ môi trường đó và SentencePiece 0.2.3 cài trong `.venv`. Vì kế thừa site packages, môi trường không hoàn toàn độc lập. Không tải lại dataset/model, không GPU/training và không thay đổi dependency chung. Set `PYTHONDONTWRITEBYTECODE=1`.

- Tokenizer load thành công; vocab 8000, PAD=0, UNK=1, BOS=2, EOS=3. Encode/decode round trip với ví dụ EN/VI thành công.
- Dùng `get_dataloader(split,batch_size=4,shuffle=False)`; `len(dataset)` train/validation/test lần lượt 131000/1553/1268. Batch đầu có `(src,tgt)` shapes train `(4,68)/(4,83)`, validation `(4,108)/(4,131)`, test `(4,47)/(4,57)`.
- Với từng batch: ID/length int64, mask bool và bằng `ids.eq(0)`; source EOS tại `src_lengths-1`, target bắt đầu BOS/kết thúc EOS tại `tgt_lengths-1`, phần pad là 0, input dịch một token so với target. Mỗi assert PASS. Đây là kiểm tra ba batch đầu và không thay thế test model hay mọi trường hợp dữ liệu.
- Tương thích tensor theo thiết kế: dữ liệu batch-first, S/T độc lập, source mask và lengths đáp ứng encoder/attention, target shifting đáp ứng teacher forcing đề xuất. Test không thực thi LSTM/Luong.

## Khả năng tương thích và điểm còn mở

- **VERIFIED theo đại số:** 2×512=1024 encoder features; bridge theo layer 1024→512; `q[512]^T W[512,1024] H[1024]` ra scalar cho từng S; context 1024; `[q;context]` 1536→512; `a[512] E.T[512,8000]` ra logits 8000. Không có model thực thi để khẳng định implementation chính xác.
- **D01 — TRUNG_ACCEPTED — PENDING_TEAM_REVIEW:** bridge nối hai hướng theo layer rồi projection 1024→512 + tanh, riêng h/c; shape và test trong tài liệu kiến trúc.
- **D02 — TRUNG_ACCEPTED — PENDING_DUC_CONFIRMATION:** `encode()`, `init_decoder_state()`, `decode_step()`, `reorder_state()` là API chung; mỗi model tự quản state, beam controller không truy cập trực tiếp hidden/cell hay KV-cache.
- **D03 — TRUNG_ACCEPTED — PENDING_DUC_CONFIRMATION:** Đức phụ trách framework training chung (optimizer, scheduler, validation, checkpoint, logging); Trung tích hợp LSTM, thống nhất `forward()`, logits, targets, loss; không tự tạo framework LSTM riêng. README và sơ đồ vẫn có khác biệt cần Đức xác nhận.
- **D04 — TRUNG_ACCEPTED — PENDING_TU_CONFIRMATION:** SHA-256 byte file gốc là chuẩn chính; checksum sau chuẩn hóa LF chỉ là kiểm tra phụ. Tránh Git tự chuyển line endings file dữ liệu; không thay đổi Git/config hay dataset trong task này.
- **PROPOSED — PENDING APPROVAL:** input feeding khởi tạo zero, công thức vector attention/projection cụ thể, cách đặt dropout, CE ignore PAD, teacher forcing 100%, hậu xử lý special token. Những chi tiết này không được proposal quy định.
- **UNKNOWN:** GPU VRAM/thời gian/hội tụ, max BPE toàn tập, kích thước batch và parameter count thực tế; tương thích Transformer–beam thực tế khi chưa có adapter; nhóm duyệt các chi tiết M1. Proposal ước tính ~22M, không thay thế phép đếm model thực tế.
- Theo audit trước, 7 file text khác checksum byte do CRLF nhưng khớp sau chuẩn hóa LF; `.model` khớp trực tiếp. Theo D04, 7 file text hiện chưa đạt kiểm tra byte chính; kết quả LF chỉ là bằng chứng phụ, chờ Tú xác nhận hướng xử lý. Không sửa file trong task.
- README nhắc checkpoint V1 không dùng với tokenizer V3. Đề xuất xác thực tokenizer hash, config, vocabulary và dataset trước load; không thấy checkpoint model trong repo.

## Rủi ro và kiểm thử M2

| Rủi ro | Kiểm thử cụ thể đề xuất |
|---|---|
| Bridge 1024→512 và chiều hướng | B=1/B>1, S≠T; kiểm tra từng layer h/c; so packed batch với mẫu riêng |
| Mask attention | PAD weight=0, tổng valid=1; all-PAD báo lỗi; thay memory PAD không đổi kết quả hợp lệ |
| Input feeding | Kiểm tra `a_0=0`, state lưu `a_t`, thay `a_prev` ảnh hưởng bước sau; reorder bao gồm `a_prev` |
| Target shift/loss | `BOS,x,y` so `x,y,EOS`, ignore PAD và label smoothing .1; so loss khi thêm PAD |
| Tied weights/PAD | Xác nhận encoder/decoder/output cùng parameter; kiểm tra gradient hàng PAD qua output và chính sách lựa chọn |
| Beam/Transformer | Fake adapter hai model, duplicate parent/reorder cache và isolation theo sample; beam 1 so greedy cùng tie-break |
| Numeric/chiều dài | softmax FP32, reject all-PAD, kiểm tra finite gradient/AMP; benchmark S/T lớn không truncate ngầm |
| V1/V3 checkpoint | kiểm tra hash/model config và từ chối mapping không khớp, round-trip output cùng state |
| Reproducibility | ghi seed/RNG, commit, versions, hashes, config và hardware; kiểm tra lặp |

## Yêu cầu Tú và Đức

- Tú xác nhận checksum LF/CRLF, chuẩn tokenizer V3 và normalization inference; runtime contract cơ bản đã PASS.
- Đức xác nhận D02/D03, adapter Transformer, parent indexing, EOS, length normalization và interface `forward`/loss; framework chung do Đức phụ trách theo D03 đang chờ xác nhận.
- Nhóm review D01, input feeding cụ thể, teacher forcing, CE, dropout placement và các tham số chưa ghi trong proposal. Những mục này không thay đổi phần đã được duyệt.

## Kết luận

**M1_DESIGN_READY: YES**: kiến trúc theo proposal, shape nhất quán qua suy luận, tokenizer và DataLoader test CPU PASS; có thể bắt đầu M2 với các chi tiết còn nhãn đề xuất. **M1_TEAM_APPROVED: PENDING_REVIEW**: proposal được phê duyệt theo thông tin người yêu cầu, nhưng không có bằng chứng nhóm đã review tài liệu M1 chi tiết. Không có training, mô hình, checkpoint hoặc đánh giá chất lượng trong task này.

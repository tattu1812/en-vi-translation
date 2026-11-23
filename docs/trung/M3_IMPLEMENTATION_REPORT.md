# M3 Implementation Report

Ngày 23/11/2026; branch `trungnguyen`, HEAD nền `c96842bc38c3dba122295dcbc9237210e92d943e`. **M3_IMPLEMENTATION_READY=YES** cho controller và contract theo các quy ước được công bố; **M3_TRANSFORMER_INTEGRATION_READY=PENDING** (mới mock), **M3_TEAM_APPROVED=PENDING_REVIEW**. Length normalization là `TRUNG_ACCEPTED — PENDING_TEAM_REVIEW`; interface D02 cần Đức xác nhận trước thực nghiệm chính; proposal không nêu exponent.

## 1. Task Overview

Code chung trong `src/decoding/`, tests `tests/test_shared_decoding.py`; không sửa kiến trúc M2. Chưa huấn luyện, tính BLEU, tạo checkpoint, commit hay push.

## 2. Proposal Compliance

Beam size mặc định 5; `max_length=None` lấy từng source length + 50; có chuẩn hóa độ dài và dùng chung controller. Proposal/M1 không chốt công thức: `normalized_score = cumulative_log_probability / (generated_length ** alpha)` với alpha mặc định 1.0 và cấu hình được là **TRUNG_ACCEPTED — PENDING_TEAM_REVIEW**; generated length tối thiểu 1, gồm EOS nếu có, không tính BOS/PAD; giả thuyết chưa EOS tại cap vẫn được xếp hạng. Do đó chưa thể nói tham số này đã được nhóm phê duyệt. V3 PAD/UNK/BOS/EOS 0/1/2/3; PAD/BOS cấm sinh, EOS ngừng và UNK được phép.

## 3. Shared Decoding Architecture

`interface.py` định nghĩa `DecodingModel` Protocol, kiểm tra input và `DecodeResult`; context và state opaque. `greedy.py`/`beam_search.py` chỉ gọi encode/init/step/reorder. LSTM M2 trực tiếp tuân theo contract, không cần adapter hay sửa model. Mock Transformer adapter trong test dùng prefix, không phải Transformer thật. Việc xác nhận tương thích thực tế vẫn PENDING.

## 4. Greedy Implementation

Greedy chọn argmax hợp lệ từ log-softmax, hòa chọn token ID nhỏ nhất, loại sample hoàn thành khỏi active state; EOS không bị mở rộng. Dừng từng mẫu tại EOS hoặc cap; trả token ids padding phải, lengths và finished cùng raw/normalized scores. Chạy inference mode và yêu cầu caller đã `.eval()`.

## 5. Beam Search Implementation

Tổng log probabilities trên vocab đầy đủ, sau đó cấm PAD/BOS không renormalize. Xử lý từng source riêng, live prune top K trên raw score tại cùng độ dài; mọi EOS được lưu riêng trong tối đa K completed theo normalized score, không mất vì live beam. Reorder state theo parent từng bước. Không heuristic early stop; tại cap so sánh completed và unfinished theo normalized score, tie lexicographic; live tie parent rồi token ID. K=1 định nghĩa đúng greedy; K mặc định 5. Tìm kiếm là xấp xỉ theo beam, không đảm bảo global optimum; width K áp dụng riêng cho live và completed, không phải gộp chung K. Không dừng EOS như content hay trộn câu.

## 6. Interface Contract

Chi tiết signatures, shape/dtype/device, EOS/PAD, state ownership, reordering và schema kết quả trong `docs/trung/SHARED_DECODING_INTERFACE.md`. Chưa áp đặt định dạng hidden/cell hay KV-cache; source phải pad phải và mask True=PAD. Không hỗ trợ logits có NaN/Inf hoặc model đang train.

## 7. Unit Test Results

Môi trường: Python 3.11.16, torch 2.14.1+cpu, SentencePiece 0.2.3, CPU; CUDA unavailable. `PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest tests.test_shared_decoding -v`: **15 PASS, 0 FAIL, 0 BLOCKED trong suite**, 3.159 giây; test GPU thực là BLOCKED/NOT_RUN riêng. Các bài kiểm thử gồm fake autoregressive oracle và LSTM chưa train.

| Test | Expected | Actual | Status |
|---|---|---|---|
| Greedy | argmax, BOS, EOS, batch và loại sample xong | đúng sequence, padding và state sample còn sống | PASS |
| Cap/default | không EOS dừng tại cap, source length + 50 | max tùy chỉnh và lengths 52/53 | PASS |
| Beam size 1 | giống greedy mọi field | ba loại logits, scores/length/token bằng nhau | PASS |
| Beam search scoring | cumulative logP, normalize, top candidates | so exhaustive oracle nhỏ K=5, score khớp | PASS |
| Completed hypotheses | EOS tốt được giữ | EOS thắng sau các bước tiếp | PASS |
| Beam reorder/batch | parent đúng, không trộn hai câu | batch và gọi riêng từng mẫu khớp; duplicate parent | PASS |
| Length norm | raw và score tách biệt | alpha 0/0.6/1, test EOS/BOS/PAD, score và sequence; alpha đổi kết quả | PASS |
| Tie/PAD/BOS | deterministic và không sinh PAD/BOS | chạy lặp nhất quán, tie chọn ID nhỏ | PASS |
| LSTM M2 | greedy/beam qua bốn API | batch 2, state cập nhật, vocab đúng | PASS |
| Mock Transformer | prefix state encode/init/step/reorder | chạy greedy và beam, exhaustive/batch test | PASS |
| CUDA/Transformer thật | integration trên adapter thực | CUDA unavailable, adapter chưa có | BLOCKED / NOT_RUN |

## 8. LSTM Integration Results

`Seq2SeqLSTM().eval()` chưa huấn luyện, input giả batch 2, greedy/beam K=5 max 3, output IDs trong [0,8000), không PAD/BOS bên trong lengths; `decode_step` cập nhật hidden state. Test này không có ý nghĩa đánh giá chất lượng dịch.

## 9. Mock Transformer Integration Results

`MockTransformer` trong test lưu `source` và `prefix` trong state, reorder cả hai bằng index_select; greedy và beam chạy với batch 1/2, duplicate/permutation và source isolation. Mock không phải Transformer thật và không kiểm chứng causal mask, KV-cache hoặc output của Đức.

## 10. Edge Cases

Mask sai, lengths sai, all PAD, invalid beam/max/alpha, train mode, logits không hữu hạn hoặc sai vocab bị từ chối; EOS dừng riêng từng mẫu; cap không EOS trả `finished=False`; tie ổn định và beam 1 khớp greedy. Empty batch và max_length≤0 bị từ chối. `DecodeResult` IDs sau `length` là PAD, trước đó không có PAD/BOS. UNK được giữ.

## 11. Performance Observations

Suite 15 test 3.159s CPU trong lần chạy ghi nhận, 2 PyTorch threads; không phải benchmark production. Beam tuần tự theo source, stable argsort qua vocab và chuyển chỉ số sang Python, phù hợp xác nhận logic nhưng chưa tối ưu throughput/GPU; độ dài source+50 có thể tốn thời gian. Peak memory và performance GPU UNKNOWN.

## 12. Known Issues

- Công thức/exponent length normalization, tie-breaking và quy ước tính EOS chưa được proposal chốt; cần review trước báo cáo thí nghiệm.
- M3 không tự đổi `.eval()` để tránh side effect; caller phải gọi `.eval()` và chuyển tensors/device phù hợp.
- GPU, mixed precision và Transformer thật chưa test. Model untrained, không phải checkpoint dịch.
- Search không đảm bảo nghiệm toàn cục ngoài oracle test nhỏ; chỉ trả 1-best, không có n-best hay attention trace. Memory/runtime tăng theo K và độ dài.
- Trong scoring, PAD/BOS cấm sau log-softmax, score không renormalize; beam width 1 cố ý tương đương greedy.

## 13. Dependencies

Đức xác nhận D02, normalized scoring, EOS/cap, adapter Transformer/cache/reorder và chạy contract tests thật. Tú xác nhận tokenizer V3/hash và decode text/normalization. Team review pending. D04 checksum text CRLF cần Tú xác nhận; task này không sửa dataset.

## 14. Handover to Đức

Đọc `docs/trung/M3_TRANSFORMER_INTEGRATION_GUIDE.md` và `docs/trung/SHARED_DECODING_INTERFACE.md`; thực hiện bốn API encode/init/step/reorder, test prefix/KV-cache/reorder và cả hai controller với Transformer thực. Không có code Transformer được tạo trong task.

## 15. M3 Acceptance Checklist

- [x] Shared interface và tài liệu; greedy, beam K=5, cap, scoring length normalization TRUNG_ACCEPTED, chờ nhóm review.
- [x] BOS/EOS/PAD, batch, reorder, beam=1, completed, tie và exhaustive toy oracle được test CPU.
- [x] LSTM untrained integration và mock Transformer PASS; tài liệu bàn giao Đức.
- [ ] Đức xác nhận D02 và công thức normalization; test Transformer thực, CUDA và team review.

`M3_IMPLEMENTATION_READY=YES` đánh giá implementation hiện tại theo quy ước đã nêu; **không** là phê duyệt quy ước thực nghiệm. `M3_TRANSFORMER_INTEGRATION_READY=PENDING`; `M3_TEAM_APPROVED=PENDING_REVIEW`.

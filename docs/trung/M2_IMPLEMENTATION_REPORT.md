# M2 Implementation Report — Seq2Seq LSTM + Luong General Attention

Ngày: 22/11/2026. Repository: `trungnguyen`, HEAD nền `32be78adde716401e217eb0a586a940459299b3e`. M2 là thay đổi chưa commit. **M2_IMPLEMENTATION_READY = YES** theo tiêu chí M2 CPU bên dưới; **M2_TEAM_APPROVED = PENDING_REVIEW**; **M3_READY = NO** cho tích hợp dùng chung đến khi Đức xác nhận D02 và có adapter Transformer để kiểm thử. Đây không phải bằng chứng đã huấn luyện hay chất lượng dịch.

## 1. M2 Overview

Triển khai model, helper loss, 10 test và integration V3. Không sửa DataLoader, tokenizer, proposal, dataset; không cài thêm dependency, không chạy training loop, không lưu checkpoint huấn luyện, không implement greedy/beam. `TRUNG_TASK_AUDIT.md` chưa được theo dõi và giữ nguyên.

## 2. Proposal Compliance

Theo `docs/proposal/main.tex` và `docs/trung/M1_LSTM_ARCHITECTURE.md`: BiLSTM 2 lớp hidden 512 mỗi hướng; decoder LSTM 2 lớp hidden 512; embedding 512, vocab 8000, dropout 0.3; Luong general + input feeding; encoder/decoder dùng chung embedding và output tie cùng Parameter với embedding decoder. Chia sẻ encoder–decoder và output–decoder là hai quan hệ được test riêng. D01 bridge 1024→512 tanh riêng h/c, theo layer và hai hướng, là quyết định Trung chờ nhóm review. Softmax FP32 trong half precision, bridge layer-specific, CE ignore PAD và teacher forcing 100% là cách hiện thực M1; proposal không định nghĩa đầy đủ các chi tiết này. Label smoothing mặc định 0.1 theo proposal.

## 3. Implemented Modules

`src/models/lstm_seq2seq.py`:

- `Encoder`: shared embedding, packed BiLSTM với lengths CPU, giữ thứ tự gốc, output padded đến S.
- `Bridge`: nối hai chiều từng layer, affine+tanh 1024→512 độc lập hidden/cell.
- `LuongGeneralAttention`: `qᵀ W_a H_s`, source mask True=PAD, all-PAD raise `ValueError`, context và weights.
- `Decoder`: LSTM hai lớp, input `[embedding, a_prev]`, attention sau decoder, `a_t=tanh(W_c[q_t;context_t]+b_c)`, output projection tied thật với embedding; state giữ `a_t` cho bước sau.
- `Seq2SeqLSTM`: `encode`, `init_decoder_state`, `decode_step`, `reorder_state`, forward teacher forcing; state gồm hidden, cell, input feed, memory và mask. `reorder_state` index h/c dim 1, các trường còn lại dim 0 và hỗ trợ chỉ số lặp. Interface D02 chờ Đức xác nhận; chưa có beam controller.
- `translation_loss`: raw logits và target đã shift, CE ignore PAD=0, EOS là nhãn hợp lệ, smoothing mặc định 0.1; từ chối toàn PAD. Tích hợp training loop theo D03 vẫn chờ Đức.

## 4. Model Architecture

Với `B,S,T,V=8000`, input feeding nhận embedding `[B,512]` nối vector attention bước trước `[B,512]`. LSTM output `q_t [B,512]`; `W_a [512,1024]` nhân encoder memory `[B,S,1024]` cho scores `[B,S]`. Weights softmax qua token nguồn hợp lệ, context `[B,1024]`. Concatenate query và context `[B,1536]`, linear 1536→512 + tanh tạo attention vector; output logits `[B,8000]` qua embedding weight transpose và bias output. Dùng linear 1536→512 cho attentional vector theo M1, không cần ma trận output weight độc lập.

## 5. Tensor Shape Verification

| Tensor | Expected | Actual | Status |
|---|---|---|---|
| encoder output / h,c | `[B,S,1024]` / `[4,B,512]` | `[3,4,1024]` / `[4,3,512]` | PASS |
| bridge h,c | `[2,B,512]` | `[2,1,512]` với test direction độc lập | PASS |
| context / weights | `[B,1024]` / `[B,S]` | `[2,1024]` / `[2,4]` | PASS |
| decoder step logits | `[B,8000]` | `[2,8000]` | PASS |
| synthetic forward / attention | `[B,T,8000]` / `[B,T,S]` | `[2,3,8000]` / `[2,3,4]` | PASS |
| V3 forward / attention | `[B,T,8000]` / `[B,T,S]` | `[1,11,8000]` / `[1,11,14]` | PASS |

## 6. Unit Test Results

Lệnh: `.venv\\Scripts\\python.exe -B -m unittest tests.test_lstm_seq2seq -v`, `PYTHONDONTWRITEBYTECODE=1`; 22/11/2026, **10 tests PASS, 0 FAIL, 0 BLOCKED**, 23.379 giây. Bao gồm 9 unit tests và 1 integration test; các mục GPU/AMP riêng bên dưới không tính như test đã chạy. Test dùng `torch.set_num_threads(2)`.

| Test | Expected | Actual | Status |
|---|---|---|---|
| Encoder | shapes, lengths khác nhau, B=1, padding, không đổi thứ tự | So sánh unsorted batch với chạy từng câu; shapes đúng | PASS |
| Bridge | đúng layer/direction, h/c riêng, gradient | Kiểm tra giá trị đầu ra theo input có dấu khác, gradient hữu hạn | PASS |
| Luong general | score công thức, mask=0, tổng=1, all-PAD lỗi | So tham chiếu einsum, không NaN/Inf, all-PAD raise | PASS |
| Decoder/tying | logits đúng, state thay đổi, hai quan hệ shared Parameter, input feeding có tác dụng | Identity `is`, thay feed làm thay logits, attention đúng shape | PASS |
| Forward | teacher forcing và batch một | Logits khớp vòng `decode_step` ở eval, B=1 PASS | PASS |
| Reorder | lặp parent, đủ state, lỗi index | h/c, feed, memory, mask khớp index_select | PASS |
| Loss | bỏ PAD, giữ EOS, không softmax trước loss | Khớp CE tham chiếu smoothing .1, gradient tại EOS khác 0, PAD=0 | PASS |
| Save/load | eval output trùng và vẫn tied | `state_dict` round-trip bằng `BytesIO`, output trùng tuyệt đối | PASS |
| Forward/backward + dropout | gradient hữu hạn ở train mode | synthetic batch, dropout train, backward hợp lệ | PASS |
| CPU V3 | batch thật forward/loss/backward | Xem mục 7–8 | PASS |
| CUDA / VRAM | GPU forward/backward | `.venv` là torch `2.14.1+cpu`, CUDA unavailable | BLOCKED |
| Mixed precision CUDA | kiểm tra FP16/BF16 khi có CUDA | Không có CUDA; code softmax float32 với half/bfloat16 | NOT_RUN |
| Chất lượng dịch/BLEU | đánh giá model đã train | Ngoài M2; model chưa train | NOT_RUN |

## 7. DataLoader Integration Test

Python 3.11.16, PyTorch 2.14.1+cpu, SentencePiece 0.2.3 trong `.venv` kế thừa site packages. Load trực tiếp model SentencePiece V3, vocab/ID `(8000,0,1,2,3)` tương ứng `(vocab,PAD,UNK,BOS,EOS)`. `get_dataloader('train', batch_size=1, shuffle=False)` trả batch đầu source `[1,14]`, target input/target `[1,11]`; không truncate. Mask bằng `src_ids.eq(0)`, BOS/EOS và target shifting được kiểm tra. Forward `[1,11,8000]`, attention `[1,11,14]`, loss và backward PASS; loss 9.082489 chỉ là giá trị trên trọng số khởi tạo, không phải chất lượng dịch. Integration chạy model `eval()` để có kết quả ổn định nhưng vẫn bật autograd; test thứ 10 riêng kiểm tra `train()` có dropout và backward trên dữ liệu giả. Chưa đưa cả tập dữ liệu qua model.

## 8. Loss and Gradient Validation

Unit test so `translation_loss` với `torch.nn.functional.cross_entropy` có smoothing 0.1; target EOS đóng góp gradient, PAD không đóng góp, toàn PAD được từ chối. Integration `loss.backward()` cho gradient hữu hạn và khác 0 đối với mọi Parameter trainable. Mỗi lần test không gọi optimizer.step, không tạo training checkpoint.

## 9. Model Parameter Count

Đếm qua `model.parameters()` loại bỏ Parameter chia sẻ trùng: **23.267.648 tổng và trainable**. Giá trị cao hơn ước tính `~22M` của proposal; proposal nói phải đo lại bằng code, không thay đổi kích thước model. FP32 parameters chiếm **93.070.592 bytes ≈ 88,76 MiB**; không gồm activations, gradients, optimizer, dataset hoặc runtime.

## 10. Runtime Resource Observations

Một process CPU riêng, 2 PyTorch threads, batch thật đầu train `[1,14]`→`[1,11]`, `eval()` với autograd: forward + loss + backward **0,562 giây**, Windows peak process working set **508.977.152 bytes ≈ 485,40 MiB**; bao gồm Python, thư viện, cả Dataset đã load, model, gradients và allocations trước đó, không phải delta riêng của model. Số đo một lần, không phải benchmark thống kê. GPU/VRAM **BLOCKED / UNKNOWN** vì PyTorch CPU-only, không suy đoán VRAM.

## 11. Known Issues and Limitations

- `forward` teacher forcing 100%; scheduled sampling chưa triển khai. CE với smoothing=0.1 là helper, chưa phải framework training chung.
- Encoder và output thực sự share Parameter; `padding_idx=0` chỉ chặn gradient lookup embedding, hàng PAD vẫn có thể nhận gradient qua output tied. Không cam kết hàng PAD luôn bằng 0.
- Mask phải True=PAD và input source phải pad phải nhất quán với lengths; source toàn PAD bị từ chối. Seq2seq lưu attention cho từng bước chỉ khi `return_attention=True`.
- Chưa đo tốc độ/bộ nhớ cho batch dài thực tế hoặc hội tụ; CPU integration ở B=1. CUDA/mixed precision chưa được kiểm tra.
- D04: theo audit, 7 file text V3 checkout CRLF không khớp SHA-256 byte gốc nhưng khớp sau chuẩn hóa LF; byte gốc vẫn là chuẩn chính, kiểm tra LF là phụ, cần Tú xác nhận. Không sửa data.
- Kết quả load model thực tế chưa huấn luyện; checkpoint tương thích tokenizer V1/V3 cần kiểm tra metadata khi framework của Đức hỗ trợ.

## 12. Dependencies with Đức/Tú

- **Đức:** D02 `TRUNG_ACCEPTED — PENDING_DUC_CONFIRMATION` cho API opaque state, bước attention, dtypes, reorder/beam và adapter Transformer; D03 training framework chung, logits/target/loss, optimizer/scheduler/validation/checkpoint/logging. Không tự tạo loop riêng.
- **Tú:** xác nhận hash byte chuẩn và cách tránh chuyển LF/CRLF, tokenizer V3 và quy ước input inference. D04 `PENDING_TU_CONFIRMATION`.
- **Nhóm:** D01 `PENDING_TEAM_REVIEW`; code M2 và các chi tiết triển khai chưa có bằng chứng được review.

## 13. M2 Acceptance Checklist

- [x] Encoder variable-length và order, bridge directional, attention Luong/mask/all-PAD.
- [x] Decoder input feeding và trọng số embedding encoder/decoder/output chia sẻ thật.
- [x] Forward teacher forcing, logits và attention shapes; CE ignore PAD/EOS, backward và finite gradient.
- [x] 9 unit tests và 1 V3 integration PASS trên CPU; save/load `state_dict` trong RAM.
- [x] Không có blocker nghiêm trọng đã biết trong phạm vi triển khai M2 CPU.
- [ ] GPU/AMP chưa test (môi trường CPU-only); team review và D02–D04 chưa xác nhận.

## 14. M3 Readiness

Các API `encode()`, `init_decoder_state()`, `decode_step()` và `reorder_state()` đã có, có unit test trên LSTM. **M3_READY = NO** để tích hợp dùng chung: Đức chưa xác nhận D02, chưa có Transformer adapter và test liên mô hình. Có thể review interface trước; không coi greedy/beam đã được triển khai, không bắt đầu M3 trong task này.

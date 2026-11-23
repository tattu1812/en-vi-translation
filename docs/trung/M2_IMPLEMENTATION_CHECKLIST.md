# M2 — Checklist triển khai theo proposal đã duyệt

Trạng thái M1: `M1_DESIGN_READY=YES`; `M1_TEAM_APPROVED=PENDING_REVIEW`. Proposal chính thức `docs/proposal/main.tex`: 2-layer BiLSTM encoder 512/hướng, decoder LSTM 2 lớp 512, embedding 512 dùng chung, Luong general input feeding, output tied, dropout 0.3. D01–D04 ghi nhận quyết định của Trung với trạng thái chờ bên liên quan xác nhận; các chi tiết teacher forcing/loss khác là **PROPOSED — PENDING APPROVAL**. Không tự coi các quyết định này là nhóm đã phê duyệt. Checklist này không phải code M2.

## Modules

- [ ] Shared embedding E `[8000,512]` cho EN/VI; tie decoder output weight với E; xác định gradient hàng PAD khi tying.
- [ ] Encoder 2-layer BiLSTM 512/hướng, output `[B,S,1024]`, h/c `[4,B,512]`; packed lengths, bất biến với padding.
- [ ] D01 `TRUNG_ACCEPTED — PENDING_TEAM_REVIEW`: ghép hai hướng từng tầng, affine h/c riêng 1024→512 + tanh, trả `[2,B,512]`; kiểm tra shape/hướng và chờ nhóm review.
- [ ] Decoder 2-layer LSTM hidden 512, input `[embedding; a_prev] [B,1024]`, khởi đầu `a_prev=0` theo đề xuất.
- [ ] Luong general: query h sau decoder `[B,512]`, `W_a[512,1024]`, scores `[B,S]`, mask True=PAD, softmax theo S, context `[B,1024]`.
- [ ] Attentional vector đề xuất `tanh(W_c[h;context])` 1536→512; feed vector này vào bước kế và xuất logits bằng E.T; trả attention weights.
- [ ] Dropout 0.3; forward trả logits `[B,T,8000]` và weights `[B,T,S]`; phương án 100% teacher forcing và CE ignore PAD/smoothing 0.1 cần review, smoothing 0.1 là approved.
- [ ] Adapter `encode`, `init_decoder_state`, `decode_step`, `reorder_state` theo M1; không triển khai controller M3 trong M2.

## Unit tests bắt buộc

- [ ] Test B=1 và B>1, S≠T; output encoder 1024, states `[4,B,512]`, bridge `[2,B,512]`, decoder input 1024, attention, output `[B,T,8000]`; h/c đúng layer và hướng.
- [ ] So sánh encoder packed batch unsorted với từng mẫu; từ chối lengths, dtype, shape, device hoặc token không hợp lệ.
- [ ] General score kiểm tra bằng phép tính tham chiếu; source mask PAD weight=0, valid sum=1, all-PAD báo lỗi; masked memory thay đổi không ảnh hưởng context.
- [ ] Input feeding: `a_0` zero, `a_t` đúng state cho bước tiếp; thay `a_prev` ảnh hưởng kết quả, gradient lan qua bước trước; test `forward` so vòng `decode_step` ở eval.
- [ ] Shared/tied embedding cùng parameter; kiểm tra gradient PAD qua output; `BOS,x,y` so `x,y,EOS`, không shift lần hai; thêm PAD không đổi loss trên non-PAD, smoothing 0.1 nếu dùng CE.
- [ ] Step/reorder: duplicate parent, expand K, permute và N thay đổi; h/c dim 1, memory/mask/**a_prev**/cache dim 0; không trộn sample, finite outputs/gradients, mixed precision khi phù hợp.
- [ ] Checkpoint round-trip và fail-fast khác tokenizer V1/V3, config, vocab hoặc weight tying.

## Integration DataLoader V3

- [x] M1 đã kiểm thử ba batch đầu B=4 CPU (train, validation, test), tokenizer 8000, IDs 0/1/2/3, dtype/mask/length/shift PASS; xem compatibility report.
- [ ] M2: đưa batch thật qua model, logits và loss đúng shape/hữu hạn; backward một batch kiểm thử nếu phù hợp, không nhầm với training đầy đủ.
- [ ] Đo max subword và benchmark memory/batch; kiểm tra `src_padding_mask=True` tại PAD và không truncate ngầm. Xác nhận checksum LF/CRLF với Tú.

## Interface Beam Search

- [ ] D02 `TRUNG_ACCEPTED — PENDING_DUC_CONFIRMATION`: Trung/Đức chốt `encode`, `init_decoder_state`, `decode_step`, `reorder_state`, state opaque và dtype/device; mỗi model tự quản lý state, beam controller không truy cập trực tiếp hidden/cell LSTM hay KV-cache Transformer.
- [ ] LSTM state giữ h/c, memory/mask và `a_prev`; duplicate/reorder tất cả cùng batch. Test fake adapter của Transformer và LSTM, beam=1 so greedy trong cùng chính sách tie-break.
- [ ] Proposal yêu cầu beam 5, chuẩn hóa độ dài và giới hạn source+50; thống nhất công thức penalty, EOS/BOS, finished hypotheses và tie-break; không giả định các chi tiết đó đã approved.

## Điều kiện bàn giao M2

- [ ] Nhóm review chi tiết M1 đề xuất; triển khai đúng Luong general/input feeding và các kích thước proposal.
- [ ] Unit/integration tests được ghi kết quả, môi trường/versions được ghi; attention, logits, loss, gradients hữu hạn; shared/tied weights đúng.
- [ ] Có API, config, tokenizer hash, checkpoint metadata và bằng chứng test; nêu các hạn chế/chưa thống nhất với Tú, Đức. Chỉ báo đã huấn luyện khi có checkpoint/kết quả tương ứng.
- [ ] D03 `TRUNG_ACCEPTED — PENDING_DUC_CONFIRMATION`: Đức phụ trách framework chung (optimizer, scheduler, validation, checkpoint, logging); Trung tích hợp LSTM, chốt `forward()`, logits/targets/loss với Đức. Không tự xây framework riêng; xác nhận khác biệt README/sơ đồ trước tích hợp training.
- [ ] D04 `TRUNG_ACCEPTED — PENDING_TU_CONFIRMATION`: SHA-256 byte file gốc là chuẩn chính, checksum sau chuẩn hóa LF chỉ phụ; thống nhất tránh Git chuyển line endings, không sửa/chuẩn hóa dataset trong M1. Các file text CRLF hiện chưa khớp byte manifest.

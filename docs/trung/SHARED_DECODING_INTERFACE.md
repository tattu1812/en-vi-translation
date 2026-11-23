# Shared Decoding Interface — D02

Ngày 23/11/2026. Chủ sở hữu: Trung. Trạng thái `TRUNG_ACCEPTED — PENDING_DUC_CONFIRMATION`; interface được test với LSTM M2 và mock prefix-based, chưa được Đức xác nhận hoặc test Transformer thực. Nguồn: `docs/proposal/main.tex`, `docs/trung/M1_LSTM_ARCHITECTURE.md`. Implement tại `src/decoding/`. Controller không truy cập fields nội bộ của LSTM hay Transformer.

## Hợp đồng input và model

Các hàm public `greedy_decode(model, src_ids, src_lengths, src_padding_mask, *, max_length=None, pad_id=0, bos_id=2, eos_id=3, length_alpha=1.0)` và `beam_search(..., beam_size=5, ...)`. `model.eval()` phải được gọi trước: controller không thay đổi mode; chạy dưới `torch.inference_mode()`. Model và source cùng device; `src_ids int64 [B,S]`, `src_lengths int64 [B]` CPU hay cùng device, `src_padding_mask bool [B,S]` trên device source. B,S≥1; lengths trong `[1,S]`, pad phải, mask True=PAD, khớp `src_ids==pad_id`. Special IDs mặc định V3: PAD=0, UNK=1, BOS=2, EOS=3. UNK là output hợp lệ. Token ID hợp lệ phải nhỏ hơn V, ba special IDs phải khác nhau. Các kiểm tra vi phạm raise `ValueError`.

`model` thực hiện bốn phương thức (context/state là opaque object, không yêu cầu dataclass cụ thể):

1. `encode(src_ids [N,S], src_lengths [N], src_padding_mask [N,S]) -> encoded`: context tùy model, mang đủ memory/mask ban đầu để khởi tạo state. Không reorder batch ban đầu. N=B tại thời điểm này.
2. `init_decoder_state(encoded) -> state`: state bao gồm mọi dữ liệu cần cho từng bước: encoder context, mask, cache hoặc prefix, vị trí, v.v. LSTM M2 bao gồm memory, mask, h/c và input feeding; Transformer có thể chứa memory, mask và prefix/KV-cache. Controller không kiểm tra hay chỉnh bên trong state.
3. `decode_step(last_token int64 [N], state) -> (logits [N,V], new_state, attention_or_None)`: lần đầu `last_token=BOS`; mỗi lần consume token được đưa vào, trả logits cho token tiếp theo và state đã consume token. Tất cả logits là float hữu hạn, trên source device và có V lớn hơn mọi special ID. Không mutate state cũ; logits không softmax. Attention không được controller sử dụng và có thể là `None`. Nếu logits có NaN/Inf hoặc sai shape/device, controller báo lỗi.
4. `reorder_state(state, indices int64 [N_new] trên source device) -> new_state`: mọi thành phần state có batch dimension phải index theo indices, cho phép lặp, permute, giảm/tăng N. Với LSTM M2, h/c index dim=1, memory/mask/input_feed dim=0; với Transformer, tự xử lý mọi KV cache, prefix, position và memory. Phải hỗ trợ index lặp `[0,0,1,1]` để expand, `[3,0,3]` để chọn parent, và không thay đổi state cũ.

Một beam có thể gồm nhiều state có chung encoder ban đầu; indices là **phẳng, zero-based trong batch state hiện tại**, không phải token ID. Implementation hiện xử lý từng source độc lập sau encode batch: ban đầu `reorder_state(initial,[sample])`, sau đó parent index nằm trong danh sách live beam của sample đó. Controller không trộn sample. Kích thước live N biến thiên từ 1 đến K; đừng giả định luôn là B×K. Greedy cũng dùng `reorder_state` để loại sample đã kết thúc.

## Kết quả và token

Hai controller trả `DecodeResult` cùng schema: `token_ids int64 [B,L]` không chứa BOS ban đầu, có EOS nếu sinh, pad phải bằng PAD sau `lengths`; `lengths int64 [B]` đếm token sinh gồm EOS nhưng không PAD; `finished bool [B]` đúng nếu đã sinh EOS; `log_scores float64 [B]` là tổng log probability của token được chọn; `scores float64 [B] = log_scores / lengths**length_alpha`. Tất cả trên source device. Sau EOS không sinh thêm token; record không EOS tại giới hạn có `finished=False` và vẫn là kết quả hợp lệ. Để decode text, lấy `token_ids[i,:lengths[i]]`, dừng/loại EOS, không decode PAD/BOS; UNK giữ lại.

Tại mỗi bước logits được `log_softmax` trên **toàn vocab**, sau đó PAD và BOS bị cấm chọn; không renormalize lại sau khi cấm. Raw score do đó là log probability model gốc của chuỗi hợp lệ. Greedy chọn argmax token hợp lệ, hòa chọn ID nhỏ nhất. Với beam: live sequences được prune top K theo raw log score tại mỗi bước (cùng chiều dài), EOS được ghi riêng vào tối đa K completed hypotheses theo normalized score; đến giới hạn so sánh completed và unfinished bằng normalized score. Không dùng early stopping heuristic. Đây là beam search hữu hạn, không đảm bảo global optimum tổng quát; width 1 được định nghĩa chính xác bằng greedy. Điểm hòa live: parent index tăng rồi token ID tăng; điểm hòa completed/final: chuỗi token lexicographic tăng. Không coi điểm hòa này là chính sách đã được nhóm phê duyệt.

## Độ dài và chuẩn hóa

Proposal quy định beam 5, chuẩn hóa độ dài, maximum bằng source length + 50. Không nêu công thức/exponent, không nêu tính EOS/BOS. **TRUNG_ACCEPTED — PENDING_TEAM_REVIEW:** `normalized_score = cumulative_log_probability / (generated_length ** alpha)`, beam mặc định 5, alpha mặc định 1.0 và có thể cấu hình bằng `length_alpha>=0`; length tính EOS nếu có, không tính BOS/PAD. Độ dài tối thiểu dùng trong phép chia là 1: mọi giả thuyết được xếp hạng hiện đều sinh ít nhất một token vì `max_length>=1`. Giả thuyết chưa EOS tại cap vẫn được xếp hạng. `max_length=None` áp dụng riêng từng mẫu `src_lengths[i]+50` (gồm EOS nguồn trong source length; generated length gồm EOS đích); số nguyên dương override cho tất cả mẫu. Các quyết định này cần xác nhận trước so sánh chính; không tự coi là thông số proposal chính thức.

## LSTM, M3 và giới hạn

`Seq2SeqLSTM` đã đáp ứng trực tiếp bốn phương thức, không cần sửa code hay adapter wrapper; phải `.eval()` trước khi gọi. Beam width>1 hiện tuần tự theo sample; có `.tolist()` và đồng bộ CPU/GPU nên ưu tiên đúng đắn/handoff hơn throughput; không khẳng định hiệu quả GPU. Chưa có output n-best/attention lịch sử, beam không cung cấp chứng minh tối ưu toàn cục. Chưa có Transformer thực và CUDA test. Khả năng cùng interface cho Transformer là hợp đồng đề xuất cần test thực tế.

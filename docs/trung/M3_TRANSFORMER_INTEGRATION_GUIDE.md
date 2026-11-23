# M3 Transformer Integration Guide — bàn giao cho Đức

Trạng thái D02 **PENDING_DUC_CONFIRMATION**. Đọc `docs/trung/SHARED_DECODING_INTERFACE.md` trước: đây là tài liệu hợp đồng đầy đủ, không cần đọc nội bộ LSTM. `tests/test_shared_decoding.py` có adapter prefix giả và các regression cases, không phải Transformer thật.

## API cần triển khai

- `encode(src_ids[B,S], src_lengths[B], src_padding_mask[B,S]) -> encoded`: source long, lengths long, mask bool True=PAD; giữ thứ tự batch, context chứa encoder memory/mask.
- `init_decoder_state(encoded) -> state`: state opaque chứa memory, mask, prefix hoặc KV-cache và position. Tất cả tensors cùng device model; lengths nếu pack có thể CPU.
- `decode_step(last_token[N], state) -> (logits[N,V], state_next, attention_or_None)`: token long trên device, lần đầu BOS=2; cập nhật chính xác prefix hoặc KV-cache và position; trả logits float hữu hạn trên device cho token tiếp. Không dựa vào LSTM hidden/cell. Không mutate state cũ.
- `reorder_state(state, indices[N_new]) -> state_new`: indices long cùng device, cho phép lặp, chọn, đảo thứ tự; reorder memory, mask, prefix/cache, positions đồng bộ. Controller quyết định expansion, state là opaque.

Hỗ trợ N biến thiên, không giả định fixed beam size, cả step đầu và sau reorder. Kiểm tra `model.eval()` trước gọi; `src_ids` right-padded, mask True=PAD, IDs V3 PAD=0 UNK=1 BOS=2 EOS=3, mọi logit hữu hạn.

## Pseudocode adapter tối thiểu

```python
class TransformerAdapter:
    def encode(self, src_ids, src_lengths, src_padding_mask):
        memory = transformer_encoder(src_ids, src_padding_mask)
        return (memory, src_padding_mask)

    def init_decoder_state(self, encoded):
        memory, mask = encoded
        return State(memory=memory, mask=mask, prefix=empty_prefix())

    def decode_step(self, token, state):
        prefix = concatenate_token(state.prefix, token)
        logits = transformer_decode_last(prefix, state.memory, state.mask)
        return logits, State(state.memory, state.mask, prefix), None

    def reorder_state(self, state, indices):
        return State(select(state.memory, indices), select(state.mask, indices),
                     select(state.prefix, indices))
```

Đây là pseudocode: `select` chọn batch dimension thật, cách tạo prefix và causal mask phụ thuộc implementation của Đức. Nếu dùng KV-cache, index đúng batch dimension của từng layer/head, reorder cả memory/mask và position; không append token hai lần.

## Gọi controller

```python
from src.decoding import greedy_decode, beam_search
model.eval()
greedy = greedy_decode(model, batch['src_ids'], batch['src_lengths'],
                       batch['src_padding_mask'], max_length=80)
beam = beam_search(model, batch['src_ids'], batch['src_lengths'],
                   batch['src_padding_mask'], beam_size=5, length_alpha=1.0)
# token_ids không gồm BOS; lengths gồm EOS nếu finished=True.
```

`max_length=None` dùng source length + 50 theo từng câu. **TRUNG_ACCEPTED — PENDING_TEAM_REVIEW:** `normalized_score = cumulative_log_probability / (generated_length ** alpha)`; beam mặc định 5, alpha mặc định 1.0 và có thể cấu hình. Generated length tối thiểu 1, không tính BOS/PAD và tính EOS nếu có. Completed được giữ riêng, unfinished tại cap vẫn được xếp hạng; live prune theo raw logP ở cùng độ dài, final so sánh normalized score. Tie và API chi tiết mô tả trong hợp đồng. Proposal chưa chốt công thức và nhóm chưa review quyết định của Trung.

## Contract tests phải PASS trên Transformer thật

1. B=1 và B>1, source lengths khác nhau, padding mask không bị đảo và đúng thứ tự batch.
2. BOS bước đầu, prefix tăng một token mỗi lần; `decode_step` tương đương full-prefix eval logits (dropout off), cache không dùng token tương lai.
3. `reorder_state` với parent lặp `[0,0,1]`, permutation và cắt giảm; memory/mask/prefix/cache/position luôn cùng sample; state cũ không bị mutate.
4. Greedy và beam chạy với vocab V3, không sinh BOS/PAD, EOS chấm dứt, no EOS trước cap; kết quả output schema/score đúng.
5. Beam=1 bằng greedy; beam K=5 không trộn câu; kiểm tra completed hypothesis và normalized/raw score theo oracle nhỏ; deterministic tie.
6. CPU và CUDA nếu có; verify dtype/device và finite logits; source all PAD bị reject. Chạy thêm `python -m unittest tests.test_shared_decoding -v` như regression controller, rồi thêm bài test gọi **adapter thật**, không coi mock thay thế được test này.

## Lỗi thường gặp

Mask đảo nghĩa; không reorder KV cache hoặc memory; append BOS/token hai lần; reorder axis sai; normalize trước khi tính log-softmax; tiếp tục mở rộng EOS; đổi model sang train mode; lấy PAD sau length làm token; dùng thông số alpha chưa được nhóm duyệt. Chưa có xác nhận của Đức hay test Transformer thật, nên integration vẫn PENDING.

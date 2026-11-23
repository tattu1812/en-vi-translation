# M1 — Kiến trúc Seq2Seq BiLSTM + Luong General

Ngày 21/11/2026. Nguồn chuẩn được người yêu cầu xác nhận đã được nhóm phê duyệt: `docs/proposal/main.tex` (bản `docs/proposal/main.pdf` là tài liệu đi kèm). Các chi tiết chưa có trong proposal gồm D01–D04 theo trạng thái quyết định dưới đây; những chi tiết khác được đánh dấu **PROPOSED — PENDING APPROVAL**. Không có code model trong task này.

**M1_DESIGN_READY: YES** theo điều kiện nghiệm thu của task: kiến trúc bám proposal, shape đã kiểm tra bằng đại số và batch/tokenizer V3 đã kiểm thử CPU; không còn blocker cho việc bắt đầu M2 theo phương án được gắn nhãn đề xuất. **M1_TEAM_APPROVED: PENDING_REVIEW**: proposal đã được nhóm duyệt theo chỉ dẫn, chưa có bằng chứng review/chấp thuận chi tiết của bộ tài liệu M1 này. Các phương án chưa được duyệt không được mô tả là quyết định đã thông qua.

## 1. Bối cảnh và phạm vi

Dự án so sánh dịch EN→VI giữa Seq2Seq LSTM và Transformer, cùng IWSLT'15, tokenizer, GPU và cách giải mã. Proposal quy định BiLSTM 2 lớp hidden 512 mỗi chiều, LSTM decoder 2 lớp hidden 512, embedding 512, attention Luong general và input feeding; dropout 0.3, embedding chung encoder/decoder và output dùng chung trọng số embedding decoder. Tài liệu cũ Bahdanau/256 được thay bằng cấu hình này. `docs/proposal/main.tex:57-69,79-108` là nguồn; chi tiết không có trong proposal ghi rõ dưới đây.

## 2. Data và tokenizer V3

`src/data/dataloader.py:10-22,30-80,95-175`: `shared_assets/data/iwslt15/{train,validation,test}.{en,vi}` UTF-8, cặp theo dòng; có 131000/1553/1268 cặp V3 theo kiểm tra runtime. Train được lọc theo 100 đơn vị whitespace, ratio 2.5 (`scripts/clean_iwslt15.py`), khác 100 subword; validation/test không bị lọc chiều dài. DataLoader không truncate, S/T phụ thuộc batch. Proposal ghi số train gốc 133317; V3 train sau lọc là 131000, xem `docs/dataset_manifest.md`.

`shared_assets/tokenizer/spm_envi_bpe_8000.model`: SentencePiece BPE chung EN/VI, vocab V=8000; **runtime xác minh** PAD=0, UNK=1, BOS=2, EOS=3; `scripts/train_tokenizer.py` mô tả BPE trên train hai ngôn ngữ. Encode source + EOS; target encode không BOS/EOS trong dataset. `collate_fn` thêm BOS vào input, EOS vào target, pad phải bằng 0. Không có bộ hậu xử lý giải mã trong repo; **PROPOSED — PENDING APPROVAL**: dừng tại EOS, bỏ BOS/PAD/EOS trước `sp.decode`, giữ UNK. Không sửa pipeline.

| Trường | Shape | dtype | Quy ước |
|---|---|---|---|
| `src_ids` | `[B,S]` | int64 | EN + EOS, pad phải |
| `src_lengths` | `[B]` | int64 | số token thực gồm EOS |
| `src_padding_mask` | `[B,S]` | bool | **True = PAD** |
| `tgt_input_ids` | `[B,T]` | int64 | BOS + VI |
| `tgt_target_ids` | `[B,T]` | int64 | VI + EOS |
| `tgt_padding_mask` | `[B,T]` | bool | True = PAD theo target input |
| `tgt_lengths` | `[B]` | int64 | gồm BOS hoặc EOS |

B=batch, S=max source, T=max target, V=8000; S và T độc lập. Runtime batch đầu `shuffle=False`, B=4: train S/T=68/83, validation 108/131, test 47/57. Chỉ là ba batch đầu, không phải giới hạn toàn tập.

## 3. Kiến trúc tổng quan

Phần **approved theo proposal**: shared embedding `E [V,512]`; encoder 2 lớp BiLSTM 512/hướng; decoder 2 lớp LSTM 512; Luong general với input feeding; output weight tied với embedding decoder; dropout 0.3. Bridge và API tuân theo D01–D02 dưới đây; các chi tiết khác về thứ tự operation và khởi tạo là **PROPOSED — PENDING APPROVAL**, vì proposal không quy định công thức chi tiết. Chúng không thay đổi các thông số đã duyệt.

## 4. Encoder

**PROPOSED — PENDING APPROVAL:** lookup `E` cho source `[B,S,512]`, encode packed sequence với `src_lengths` CPU int64, `enforce_sorted=False`, unpack `total_length=S`. BiLSTM PyTorch `batch_first=True`, hidden=512, num_layers=2, bidirectional=True; output `H [B,S,1024]`; hidden/cell mỗi tensor `[4,B,512]` theo thứ tự layer0 forward/backward rồi layer1 forward/backward. `src_padding_mask` True=PAD, lengths gồm EOS. Không để padding làm nhiễm hidden backward.

## 5. Encoder–decoder Bridge

Proposal không xác định bridge. **D01: TRUNG_ACCEPTED — PENDING_TEAM_REVIEW:** mỗi tầng l nối states encoder hai chiều `hcat_l = concat(h[2l], h[2l+1]) [B,1024]`, tương tự c; hai affine riêng cho h và c của mỗi tầng: `hD_l=tanh(W_h_l hcat_l+b_h_l)` và `cD_l=tanh(W_c_l ccat_l+b_c_l)`, W `[512,1024]`, output `[B,512]`; stack h/c thành `[2,B,512]`. Đây là quyết định D01 của Trung, chưa được nhóm review; projection và tanh không được quy định cụ thể trong proposal. Reject lengths không hợp lệ và batch/mask không khớp.

## 6. Luong general và input feeding

Luong **general** là biến thể đã ghi trong proposal. Công thức hiện thực cụ thể sau là **PROPOSED — PENDING APPROVAL** (định hướng `W` tương đương các dạng chuyển vị):

1. Tại bước t, LSTM nhận `x_t = concat(E[y_(t-1)], a_(t-1)) [B,1024]`, với `a_0=zeros[B,512]`; dùng hidden/cell trước đó tạo `h_t,c_t`, query là **top-layer decoder output sau cập nhật** `q_t=h_t[-1] [B,512]`.
2. `score_ts = q_t^T W_a H_s`, với `H_s [B,1024]`, `W_a [512,1024]`; vectorized `q_t W_a H_s^T` cho scores `[B,S]`.
3. Mask source `True=PAD` bằng `-inf` trước softmax; từ chối hàng toàn PAD. `alpha_t=softmax(scores, dim=S) [B,S]`, `context_t=Σ_s alpha_ts H_s [B,1024]`. Đề xuất tính softmax float32 trong mixed precision; PAD phải có trọng số 0.
4. `a_t=tanh(W_c concat(q_t,context_t)+b_c) [B,512]` với `W_c [512,1536]`. `a_t` là attentional vector được **feed sang bước t+1** và là đầu vào projection output. Có thể xuất alpha cho M7. Đây là input feeding, không phải attention Bahdanau trước bước LSTM.

## 7. Decoder và output

Decoder 2-layer unidirectional LSTM hidden 512; input kích thước 1024 do concat embedding 512 và `a_prev` 512. Tied output **PROPOSED — PENDING APPROVAL** cho chi tiết công thức: `logits_t=a_t E^T+b_vocab [B,V]`, `E [V,512]` đồng thời là embedding encoder và decoder theo proposal, `b_vocab [V]`. Không cần adapter 256. Có thể tồn tại gradient đến hàng PAD của E thông qua output projection dù embedding dùng `padding_idx`; xác định/test chính sách PAD riêng. Dropout 0.3 được proposal quy định, vị trí dropout cụ thể đề xuất tại embedding, giữa LSTM layers và attentional output, cần duyệt.

## 8. Training forward

Proposal quy định PyTorch, Adam `lr=1e-3`, gradient clip 1.0, label smoothing 0.1, mixed precision, theo dõi train/validation loss, best checkpoint validation, tối đa 30 epoch và early stopping patience 5; cùng GPU và training budget cho hai model. Proposal **không ghi rõ** batch size, tỉ lệ teacher forcing, công thức loss, optimizer betas, scheduler, hay tiêu chí cụ thể của early stopping: không coi giá trị nào là approved.

**PROPOSED — PENDING APPROVAL:** 100% teacher forcing dùng nguyên `tgt_input_ids`, không shift lần hai; `forward` lần lượt chạy T bước trả `[B,T,V]`, `[B,T,S]`. `CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)` trên logits reshape `[B*T,V]` và target `[B*T]` long; bỏ target PAD khi phân tích attention; không softmax trước loss. Batch size chọn sau benchmark và thống nhất; loss CE/ignore PAD là phương án phù hợp pipeline, không phải trích dẫn nguyên văn từ proposal. Mixed precision thực hiện khi có GPU hỗ trợ, không chạy huấn luyện ở đây.

**D03 — TRUNG_ACCEPTED — PENDING_DUC_CONFIRMATION:** Đức phụ trách framework training chung gồm optimizer, scheduler, validation, checkpoint và logging; Trung phụ trách tích hợp LSTM, thống nhất `forward()`, logits `[B,T,V]`, targets `[B,T]` và interface loss với Đức. Không tự xây training framework riêng cho LSTM. README hiện nói mỗi mô hình có loop riêng, trong khi sơ đồ giao D3 cho Đức; quyết định phân công này chờ Đức xác nhận.

## 9. Bảng tensor shapes

| Tensor | Shape | Diễn giải |
|---|---|---|
| source / decoder input / target | `[B,S]` / `[B,T]` / `[B,T]` | long, S và T độc lập |
| source / target embedding | `[B,S,512]` / `[B,T,512]` | shared E `[8000,512]` |
| encoder output H | `[B,S,1024]` | 2 hướng ×512 |
| encoder h/c | `[4,B,512]` | 2 tầng ×2 hướng |
| nối h hoặc c theo tầng | `[B,1024]` | 512+512 |
| bridge h/c | `[2,B,512]` | W `[512,1024]` mỗi layer |
| previous attentional vector | `[B,512]` | `a_0=0` đề xuất |
| decoder input / top output | `[B,1024]` / `[B,512]` | 512+512 / hidden 512 |
| decoder h/c | `[2,B,512]` | 2 layer |
| general W / scores / weights | `[512,1024]` / `[B,S]` / `[B,S]` | query sau decoder |
| context | `[B,1024]` | weighted sum encoder |
| concat query/context → attentional | `[B,1536]` → `[B,512]` | `W_c [512,1536]` |
| step / forward logits | `[B,8000]` / `[B,T,8000]` | output tied E |
| attention history | `[B,T,S]` | loại target PAD khi hiển thị |

Những phép nhân/nối ở trên nhất quán bằng đại số; chưa có model để kiểm thử shape qua forward thực tế. Ba batch V3 thật có shape/dtype phù hợp signature dự kiến, không chứng minh implementation M2.

## 10. Interface model và decoding

D02 **TRUNG_ACCEPTED — PENDING_DUC_CONFIRMATION**: Trung đề xuất API dưới đây, Đức chưa xác nhận; N=B trước beam và có thể N=B×K sau expand. IDs/indices int64, masks bool True=PAD; states/memory float cùng device model; lengths CPU int64 cho pack. Từ chối shape/device/token-range/length sai, batch không khớp và all-PAD; không broadcast ngầm.

- `encode(src_ids[N,S],src_lengths[N],src_padding_mask[N,S]) -> encoded`: trả memory `[N,S,1024]`, mask `[N,S]`, encoder h/c `[4,N,512]`. Kiểm tra lengths và mask.
- `init_decoder_state(encoded) -> state`: bridge h/c `[2,N,512]`, `a_prev=zeros[N,512]`, memory `[N,S,1024]`, mask `[N,S]`; state opaque với controller. Đây là nơi khởi tạo input feeding.
- `decode_step(last_token[N],state) -> (logits[N,V],new_state,attention[N,S])`: consume token hiện tại (BOS tại bước đầu); LSTM dùng embedding + `a_prev`, rồi general attention theo output sau LSTM; lưu `a_t` trong state mới. Controller chọn token tiếp theo, model không tự thực hiện greedy/beam, không mutate in-place. Với adapter Transformer attention có thể `None`.
- `forward(src_ids,src_lengths,src_padding_mask,tgt_input_ids) -> (logits[B,T,V],attention[B,T,S])`: teacher forcing theo lựa chọn đề xuất; không yêu cầu Transformer dùng cùng training signature.
- `reorder_state(state,indices[N_new]) -> state_new`: indices long hợp lệ trên device của state, được phép lặp; h/c index dim 1; `a_prev`, memory, mask, cache index dim 0. Expand K bằng `[0]*K+[1]*K+...`; parent phẳng `b*K+parent`, reorder mọi thành phần và finished/scores/tokens ở controller cùng thứ tự, không trộn sample. Transformer tự quản lý prefix/KV cache. Beam Search chỉ gọi interface chung; mỗi mô hình tự quản lý state, controller không truy cập trực tiếp hidden/cell LSTM hoặc KV-cache Transformer. Tương thích thực tế UNKNOWN cho đến khi các adapter được kiểm thử.

Proposal quy định beam=5, length normalization, max length nguồn+50 cho đánh giá chính; công thức normalization, nghĩa số bước (EOS/BOS), tie-breaking và hình thức greedy chưa được quy định, cần Đức/nhóm duyệt. A1 so greedy với beam; không mô tả beam search đã tồn tại.

## 11. Baseline & đánh giá

Approved proposal: 2-layer BiLSTM 512/hướng; 2-layer LSTM 512; embedding 512 shared source/target và tied decoder/output; Luong general + input feeding; dropout .3; Adam 1e-3, clip 1.0; label smoothing .1, mixed precision; tối đa 30 epoch, patience 5, 1 seed/model, validation chọn best. Số tham số `~22M` chỉ là ước tính của proposal: với bridge và input feeding như đề xuất phải đo thực tế khi M2 được cài đặt; không bảo đảm khớp. Kích thước batch, target device, chi tiết AMP và teacher forcing là chưa quy định; hardware miễn phí Kaggle/Colab dự kiến, memory và runtime UNKNOWN.

Proposal yêu cầu corpus SacreBLEU trên test đã detokenize, BLEU theo độ dài nguồn ≤10/11–20/21–30/>30 từ; số tham số, thời gian epoch, tổng thời gian tới best và peak GPU memory. `src/evaluation/bleu.py` đã có evaluator; kết quả perfect-match smoke không phải kết quả mô hình. Đánh giá thực và huấn luyện nằm ngoài task.

## 12. Checkpoint design

**PROPOSED — PENDING APPROVAL:** lưu model state, architecture/bridge/attention/weight sharing, optimizer/scheduler/scaler nếu dùng, epoch/best validation, seed/RNG, versions/commit, dataset V3 hashes và tokenizer model SHA-256, special IDs, evaluation configuration. Fail-fast nếu hash/token mapping hoặc config khác. Không sử dụng checkpoint V1 với V3; checkpoint và code load/save không có trong M1.

## 13. M2 test requirements

Kiểm thử các shape của encoder/bridge/decoder, mask và softmax, tính đúng input feeding (thay `a_prev` làm thay logits), teacher forcing shift, weight tying cùng parameter, ignore PAD/smoothing, gradient hữu hạn, packed sequence, `forward` khớp `decode_step`, reorder duplicate indices bao gồm `a_prev`, và integration batch V3 CPU; xem checklist. Những test model chỉ dự kiến cho M2, chưa được chạy.

## 14. Dependencies và rủi ro

**D04 — TRUNG_ACCEPTED — PENDING_TU_CONFIRMATION:** SHA-256 theo byte file gốc là kiểm tra chính; SHA-256 sau chuẩn hóa LF chỉ là kiểm tra phụ. Cần tránh Git tự chuyển line ending của file dữ liệu trong tương lai, theo giải pháp Tú xác nhận; task này không sửa cấu hình Git, `.gitattributes` hay chuẩn hóa dataset. Runtime ở máy này đã PASS; file text checkout hiện không khớp hash byte, chỉ khớp sau chuẩn hóa LF, vì vậy không coi checksum byte là PASS. Đức cần xác nhận D02/D03, EOS và length normalization. Bridge D01 chờ nhóm review; dtype, dropout placement, CE và teacher forcing là chi tiết chờ review. Giám sát gradient, all-PAD, batch dài và memory; không xem kết quả test tokenizer là kiểm chứng chất lượng mô hình.

## 15. M1 acceptance criteria

- [x] Thiết kế Luong general, input feeding, kích thước và training parameters được proposal nêu đã phản ánh đúng.
- [x] Shape nhất quán theo đại số và contract batch thật được kiểm tra CPU.
- [x] Tokenizer và ba batch DataLoader V3 kiểm tra thành công; không đào tạo mô hình.
- [ ] Nhóm review D01 và tài liệu; Đức xác nhận D02/D03; Tú xác nhận D04; review các chi tiết **PROPOSED — PENDING APPROVAL**.

`M1_DESIGN_READY=YES` cho phép bắt đầu M2 có kiểm soát theo các điểm được gắn nhãn; `M1_TEAM_APPROVED=PENDING_REVIEW` không giả định nhóm đã duyệt chi tiết. Nếu nhóm bác bỏ bridge/API được đề xuất, cập nhật lại trước khi đóng băng tích hợp.

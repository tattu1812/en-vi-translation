# Dataset Manifest — IWSLT'15 English–Vietnamese

## 1. Thông tin tổng quan

- **Tên bộ dữ liệu:** IWSLT'15 English–Vietnamese.
- **Bài toán:** Dịch máy thần kinh từ tiếng Anh sang tiếng Việt (English → Vietnamese).
- **Nguồn dữ liệu:** Stanford NLP — Neural Machine Translation.
- **Repository cung cấp bản dữ liệu:** https://github.com/stefan-it/nmt-en-vi
- **Trang Stanford NLP:** https://nlp.stanford.edu/projects/nmt/
- **File nhóm đã nhận:** `nmt-en-vi-master.zip`.
- **Mục đích sử dụng:** Huấn luyện, kiểm thử và so sánh hai mô hình Seq2Seq LSTM với Attention và Transformer Encoder–Decoder.

Nhóm sử dụng dữ liệu IWSLT'15 thay cho PhoMT nhằm phù hợp hơn với quy mô thực nghiệm và tài nguyên GPU miễn phí.

## 2. Cấu trúc dữ liệu gốc

File ZIP được cung cấp có cấu trúc:

```text
nmt-en-vi-master/
├── README.md
├── data/
│   ├── train-en-vi.tgz
│   ├── dev-2012-en-vi.tgz
│   └── test-2013-en-vi.tgz
└── tensorboard_envi.png
```

Nội dung các file nén:

| File nén | File tiếng Anh | File tiếng Việt |
|---|---|---|
| `train-en-vi.tgz` | `train.en` | `train.vi` |
| `dev-2012-en-vi.tgz` | `tst2012.en` | `tst2012.vi` |
| `test-2013-en-vi.tgz` | `tst2013.en` | `tst2013.vi` |

Các cặp câu được căn chỉnh theo thứ tự dòng: dòng thứ `i` của file `.en` tương ứng với dòng thứ `i` của file `.vi`.

## 3. Thống kê dữ liệu

Mỗi số cặp trong bảng cũng là số dòng của từng file tiếng Anh và tiếng Việt tương ứng.

| Tập | Số cặp gốc | Số cặp sau làm sạch V3 |
|---|---:|---:|
| Train | 133.317 | 131.000 |
| Validation | 1.553 | 1.553 |
| Test | 1.268 | 1.268 |
| **Tổng** | **136.138** | **133.821** |

Train giảm 2.317 cặp. Validation và test được chuẩn hóa và giữ nguyên số lượng. Sau chuẩn hóa, train không có cặp song ngữ trùng nguyên văn với validation hoặc test. Validation và test còn 3 cặp trùng nhau. Chi tiết nằm trong [báo cáo làm sạch V3](../results/data/iwslt15_cleaning_v3.json).

## 4. Tiền xử lý V3

`scripts/clean_iwslt15.py` gọi `src/data/clean_iwslt15.py` để đọc từng cặp EN–VI theo dòng và báo lỗi nếu hai file lệch số dòng. Hàm `normalize_text()` trong `src/data/text_normalization.py` áp dụng các bước sau cho cả ba tập.

1. Chuẩn hóa Unicode NFC.
2. Giải mã HTML entity bằng `html.unescape()`, chẳng hạn `&quot;` thành dấu ngoặc kép.
3. Chuẩn hóa khoảng trắng, gồm cả khoảng trắng không ngắt dòng.
4. Detokenize bằng Moses với cấu hình ngôn ngữ EN hoặc VI, sau đó chuẩn hóa lại khoảng trắng và NFC. Giữ nguyên chữ hoa và chữ thường.

Chỉ tập train được lọc. Mỗi cặp bị loại được ghi nhận theo lý do đầu tiên trong thứ tự sau.

1. Loại cặp có ít nhất một câu rỗng.
2. **Length Filtering.** Loại nếu câu EN hoặc VI dài hơn 100 đơn vị tách theo khoảng trắng.
3. **Length-ratio Filtering.** Loại nếu tỷ lệ giữa độ dài câu dài hơn và câu ngắn hơn lớn hơn 2,5.
4. **Cross-split Overlap Removal.** Loại cặp EN–VI trùng nguyên văn sau chuẩn hóa với một cặp trong validation hoặc test.
5. **Exact-match Deduplication.** Giữ lần xuất hiện đầu tiên của mỗi cặp EN–VI còn lại trong train.

SentencePiece BPE V3 được huấn luyện trên hai file train đã làm sạch, với 8.000 token và các ID PAD=0, UNK=1, BOS=2, EOS=3. Validation và test không được dùng để huấn luyện tokenizer hay cập nhật trọng số mô hình.

## 5. Sử dụng dữ liệu trong thực nghiệm

### Mô hình 1: Seq2Seq LSTM với Attention

- Encoder LSTM xử lý chuỗi token tiếng Anh.
- Decoder LSTM kết hợp Attention tạo bản dịch tiếng Việt.
- Nhóm tự xây dựng các thành phần cốt lõi bằng PyTorch.

### Mô hình 2: Transformer

- Transformer Encoder–Decoder với self-attention.
- Bao gồm positional encoding, multi-head attention, feed-forward network và masking.
- Nhóm tự xây dựng các thành phần cốt lõi bằng PyTorch.

### Nguyên tắc so sánh

- Cùng bộ dữ liệu train/validation/test.
- Cùng tokenizer và quy trình tiền xử lý trong thí nghiệm so sánh chính.
- Ghi nhận seed, số tham số, cấu hình huấn luyện, GPU và thời gian thực nghiệm.
- Đánh giá chất lượng trên cùng tập test bằng corpus BLEU với SacreBLEU.
- Phân tích ví dụ dịch sai và ảnh hưởng của độ dài câu.
- Không sử dụng checkpoint pretrained để thay thế việc tự xây dựng và huấn luyện hai mô hình.

## 6. Quản lý dữ liệu và tái tạo pipeline

| Tài nguyên | Đường dẫn |
|---|---|
| Sáu file EN/VI gốc sau khi giải nén | `data/iwslt15/nmt-en-vi-master/data/` |
| Dữ liệu V3 cục bộ | `data/iwslt15/cleaned_v3/` |
| Corpus train, tokenizer và vocabulary cục bộ | `data/tokenizer_v3/` |
| Dữ liệu V3 dùng chung | `shared_assets/data/iwslt15/` |
| Model SentencePiece và vocabulary V3 dùng chung | `shared_assets/tokenizer/` |

Để tái tạo pipeline từ dữ liệu gốc, giải nén các file được liệt kê ở mục 2 vào đường dẫn tương ứng rồi chạy các lệnh sau từ thư mục gốc của repository.

```bash
python -m scripts.check_iwslt15
python -m scripts.clean_iwslt15
python -m scripts.train_tokenizer
```

`data/` và `local_artifacts/` được Git bỏ qua. Nhánh `tu/8-detokenize-iwslt15` đã chứa sáu file dữ liệu V3 và hai file tokenizer V3 trong `shared_assets/`. Bản dùng chung có thể được kiểm tra bằng `(cd shared_assets && sha256sum -c SHA256SUMS)`. Bài kiểm thử tokenizer tải trực tiếp file `.model` đã huấn luyện và kiểm tra hành vi encode/decode. Việc tái tạo artifact từ dữ liệu gốc dùng các script trên.

Nhóm cần xác minh điều khoản sử dụng và phân phối dữ liệu từ nguồn tương ứng trước khi chia sẻ dữ liệu ra bên ngoài. Việc lưu tài nguyên trên nhánh làm việc không xác nhận quyền phân phối lại.

## 7. Nguồn tham khảo

- Stanford NLP — Neural Machine Translation: https://nlp.stanford.edu/projects/nmt/
- Repository nmt-en-vi: https://github.com/stefan-it/nmt-en-vi
- Luong and Manning (2015), *Stanford Neural Machine Translation Systems for Spoken Language Domains*: https://aclanthology.org/2015.iwslt-evaluation.11/

## 8. Trạng thái

- **T1:** Đã kiểm tra số dòng và cách ghép cặp EN–VI trong dữ liệu gốc.
- **T2:** Đã làm sạch V3, kiểm tra số lượng và đối chiếu các cặp trùng giữa các tập.
- **T3:** Đã huấn luyện SentencePiece BPE V3 và kiểm tra model, vocabulary, ID đặc biệt và encode/decode.
- **T4:** DataLoader đã qua kiểm thử trên cả ba tập với tài nguyên V3 dùng chung.
- **T5:** SacreBLEU đã qua kiểm thử. Smoke test dùng tham chiếu làm dự đoán đạt BLEU 100 trên 1.268 câu. Điểm BLEU của hai mô hình sẽ được xác định sau khi đánh giá bản dịch thực tế.

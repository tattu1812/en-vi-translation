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

| Tập | Số câu tiếng Anh | Số câu tiếng Việt | Số cặp câu |
|---|---:|---:|---:|
| Train | 133.317 | 133.317 | 133.317 |
| Validation | 1.553 | 1.553 | 1.553 |
| Test | 1.268 | 1.268 | 1.268 |
| **Tổng** | **136.138** | **136.138** | **136.138** |

Các số liệu được kiểm tra trực tiếp bằng cách đọc nội dung các file nén trong `nmt-en-vi-master.zip`.

Kết quả ban đầu: số dòng của hai ngôn ngữ khớp nhau trong cả ba tập. Chưa hoàn tất kiểm tra câu rỗng, câu trùng lặp, mức độ tương ứng ngữ nghĩa và khả năng trùng lặp giữa các tập.

## 4. Kế hoạch tiền xử lý

Nhóm dự kiến thực hiện:

1. Giải nén và đọc các file song ngữ.
2. Kiểm tra số dòng, câu rỗng và cặp câu không hợp lệ.
3. Kiểm tra câu trùng lặp và khả năng rò rỉ giữa các tập.
4. Chuẩn hóa Unicode, khoảng trắng và các HTML entity khi cần thiết.
5. Xây dựng tokenizer hoặc subword vocabulary chỉ từ tập train.
6. Bổ sung các token đặc biệt như PAD, BOS, EOS và UNK theo thiết kế.
7. Chuyển câu thành token ID, padding và tạo batch.
8. Kiểm soát độ dài chuỗi và ghi lại số lượng câu bị loại nếu có.

Các bước làm sạch cần áp dụng nhất quán cho dữ liệu đầu vào của hai mô hình.

Không sử dụng validation hoặc test để huấn luyện tokenizer hay cập nhật trọng số mô hình.

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

## 6. Quản lý dữ liệu trong repository

Dữ liệu gốc và dữ liệu đã giải nén được lưu tại:

`local_artifacts/data/`

Các file dữ liệu không được commit lên GitHub. Repository chỉ lưu mã nguồn, cấu hình, tài liệu và các kết quả nhỏ phục vụ tái lập thí nghiệm.

Nhóm cần xác minh điều khoản sử dụng và phân phối dữ liệu từ nguồn tương ứng trước khi chia sẻ dữ liệu ra bên ngoài.

## 7. Nguồn tham khảo

- Stanford NLP — Neural Machine Translation: https://nlp.stanford.edu/projects/nmt/
- Repository nmt-en-vi: https://github.com/stefan-it/nmt-en-vi
- Luong and Manning (2015), *Stanford Neural Machine Translation Systems for Spoken Language Domains*: https://aclanthology.org/2015.iwslt-evaluation.11/

## 8. Trạng thái

- Đã xác định tập train, validation và test.
- Đã xác nhận số dòng tiếng Anh và tiếng Việt khớp nhau.
- Chưa hoàn thành bước kiểm tra chất lượng chi tiết.
- Chưa xác nhận tokenizer và cấu hình huấn luyện cuối cùng.
- Dữ liệu chính thức của dự án: **IWSLT'15 English–Vietnamese**.
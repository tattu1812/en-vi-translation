# Hướng dẫn sử dụng Data Pipeline IWSLT'15

## 1. Giới thiệu

Data Pipeline được xây dựng để phục vụ dự án dịch máy Anh–Việt, gồm:

- Bộ dữ liệu IWSLT'15 English–Vietnamese đã làm sạch.
- Tokenizer SentencePiece BPE với vocabulary size 8.000.
- PyTorch Dataset và DataLoader dùng chung.
- Các script kiểm tra, làm sạch và huấn luyện tokenizer.
- File kiểm thử DataLoader.

**Mục tiêu:** Bảo đảm Seq2Seq LSTM + Attention và Transformer sử dụng cùng dữ liệu, tokenizer và định dạng batch khi huấn luyện.

## 2. Yêu cầu môi trường

- Python và pip trong môi trường ảo (khuyến nghị Python 3.11–3.13).
- Git.
- PyTorch.
- SentencePiece.
- Ubuntu, WSL2, Google Colab hoặc môi trường Linux tương thích.

Việc kiểm thử DataLoader có thể thực hiện bằng CPU, không bắt buộc có GPU.

## 3. Tải phiên bản mới nhất từ GitHub

Repository:

https://github.com/tattu1812/en-vi-translation

**Nếu đã clone dự án:**

```bash
cd ~/projects/en-vi-translation
git switch main
git pull --ff-only origin main
```

**Nếu chưa clone dự án:**

```bash
git clone https://github.com/tattu1812/en-vi-translation.git
cd en-vi-translation
```

## 4. Tạo môi trường Python

Trên Ubuntu hoặc WSL2:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
```

Cài các thư viện cần thiết:

```bash
python -m pip install --upgrade pip
python -m pip install sentencepiece
python -m pip install torch
```

Nếu chỉ cần kiểm thử DataLoader trên CPU, có thể dùng bản PyTorch CPU phù hợp với môi trường:

```bash
python -m pip install torch \
  --index-url https://download.pytorch.org/whl/cpu
```

Chỉ cần chọn một trong hai cách cài PyTorch. Khi huấn luyện trên GPU, sử dụng bản PyTorch phù hợp với CUDA của môi trường thay vì bản CPU.

Kiểm tra:

```bash
python -c "import torch, sentencepiece; print('Dependencies OK')"
```

## 5. Cấu trúc dữ liệu dùng chung

Các file đã có trên nhánh `main`:

```text
shared_assets/
├── SHA256SUMS
├── data/
│   └── iwslt15/
│       ├── train.en
│       ├── train.vi
│       ├── validation.en
│       ├── validation.vi
│       ├── test.en
│       └── test.vi
└── tokenizer/
    ├── spm_envi_bpe_8000.model
    └── spm_envi_bpe_8000.vocab
```

**Thống kê dữ liệu sau làm sạch:**

| Tập | Số cặp câu |
|---|---:|
| Train | 131.189 |
| Validation | 1.553 |
| Test | 1.268 |

Tokenizer đã thống nhất các ID đặc biệt:

| Token | ID |
|---|---:|
| PAD | 0 |
| UNK | 1 |
| BOS | 2 |
| EOS | 3 |

Không cần tải dữ liệu từ nguồn gốc hoặc huấn luyện lại tokenizer để chạy DataLoader. Các file dùng chung đã có trong repository.

## 6. Kiểm tra dữ liệu và DataLoader

**Kiểm tra tính toàn vẹn của các file:**

```bash
cd shared_assets
sha256sum -c SHA256SUMS
cd ..
```

**Chạy kiểm thử DataLoader:**

```bash
python -m tests.test_dataloader
```

Kết quả mong đợi:

```text
PASS train: 131189 pairs, src=(4, 68), tgt=(4, 83)
PASS validation: 1553 pairs, src=(4, 108), tgt=(4, 131)
PASS test: 1268 pairs, src=(4, 47), tgt=(4, 57)

ALL DATALOADER TESTS PASSED
```

Nếu xuất hiện kết quả trên thì Dataset và DataLoader đã vượt qua các kiểm tra cơ bản, sẵn sàng để thử tích hợp với mô hình.

## 7. Cách sử dụng DataLoader trong code

Import:

```python
from src.data.dataloader import get_dataloader

train_loader = get_dataloader("train", batch_size=32)
val_loader = get_dataloader("validation", batch_size=32)
test_loader = get_dataloader("test", batch_size=32)
```

Đọc một batch:

```python
batch = next(iter(train_loader))

for key, value in batch.items():
    print(key, value.shape)
```

Các trường được trả về:

| Trường | Ý nghĩa |
|---|---|
| `src_ids` | Token ID của câu tiếng Anh |
| `src_lengths` | Độ dài thực tế của câu nguồn |
| `src_padding_mask` | Vị trí padding ở câu nguồn |
| `tgt_input_ids` | Chuỗi đầu vào decoder |
| `tgt_target_ids` | Chuỗi nhãn dùng để tính loss |
| `tgt_padding_mask` | Vị trí padding của decoder |
| `tgt_lengths` | Độ dài thực tế chuỗi đích |

Quy ước:
- `batch_first=True`: tensor có dạng `[batch_size, sequence_length]`.
- Padding mask: `True` nghĩa là PAD.
- Câu nguồn kết thúc bằng EOS.
- Decoder input bắt đầu bằng BOS.
- Decoder target kết thúc bằng EOS.
- Train được shuffle; validation/test giữ nguyên thứ tự.

## 8. Tích hợp vào Seq2Seq và Transformer

Lấy tensor từ batch:

```python
src = batch["src_ids"]
src_lengths = batch["src_lengths"]

tgt_input = batch["tgt_input_ids"]
tgt_target = batch["tgt_target_ids"]

src_pad = batch["src_padding_mask"]
tgt_pad = batch["tgt_padding_mask"]
```

Khi đưa lên GPU hoặc CPU:

```python
src = src.to(device)
tgt_input = tgt_input.to(device)
tgt_target = tgt_target.to(device)

src_pad = src_pad.to(device)
tgt_pad = tgt_pad.to(device)
```

**Trung — Seq2Seq LSTM + Attention:**

- Sử dụng `src_ids` làm đầu vào encoder.
- Sử dụng `src_lengths` khi xử lý các chuỗi có độ dài khác nhau.
- Sử dụng `tgt_input_ids` để thực hiện teacher forcing.
- So sánh logits với `tgt_target_ids` khi tính loss.
- Sử dụng padding mask để loại bỏ các vị trí PAD khỏi attention.

**Đức — Transformer:**

- Sử dụng `src_ids` cho encoder và `tgt_input_ids` cho decoder.
- Sử dụng `src_padding_mask` và `tgt_padding_mask`.
- Tạo thêm causal mask trong decoder để che các token tương lai.
- So sánh logits với `tgt_target_ids` khi tính loss.

Ví dụ cấu hình loss thông thường:

```python
import torch

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=0
)
```

Nếu áp dụng label smoothing theo Proposal, có thể sử dụng:

```python
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=0.1
)
```

Cần bảo đảm logits và target được reshape đúng kích thước khi tính loss. Hai mô hình tự triển khai forward pass và training loop tương ứng.

## 9. Những lưu ý quan trọng

- Không tự ý huấn luyện lại tokenizer hoặc thay đổi vocabulary size khi so sánh hai mô hình.
- Không thay đổi số lượng hoặc thứ tự câu validation/test.
- Không dùng tập test để huấn luyện hoặc lựa chọn siêu tham số.
- Giữ nguyên quy ước PAD/BOS/EOS và batch format.
- Nếu muốn thay đổi DataLoader, tạo nhánh riêng và thông báo cho cả nhóm.
- File `local_artifacts/` không cần thiết khi sử dụng bản dữ liệu dùng chung trên GitHub.

## 10. Kết luận

Pipeline dữ liệu T1–T4 đã được triển khai, đưa lên `main` và vượt qua các kiểm thử DataLoader cơ bản trên máy Tú.

Trung và Đức chỉ cần cập nhật repository, cài dependencies, kiểm tra checksum và chạy `python -m tests.test_dataloader` trước khi tích hợp với mô hình của mình.

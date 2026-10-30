# Báo cáo khảo sát bộ dữ liệu PhoMT

## 1. Thông tin bộ dữ liệu

- **Tên bộ dữ liệu:** PhoMT – A High-Quality and Large-Scale Benchmark Dataset for Vietnamese-English Machine Translation
- **Đơn vị phát triển:** VinAI Research
- **Công bố khoa học:** Hội nghị EMNLP 2021
- **Nguồn dữ liệu:** https://huggingface.co/datasets/vinai/PhoMT
- **Bài báo gốc:** https://aclanthology.org/2021.emnlp-main.369/
- **Bài toán:** Dịch máy thần kinh từ tiếng Anh sang tiếng Việt
- **Ngôn ngữ đầu vào:** Tiếng Anh (`en`)
- **Ngôn ngữ đầu ra:** Tiếng Việt (`vi`)
- **Tổng số cặp câu:** 3.015.869

## 2. Cấu trúc bộ dữ liệu

Bộ dữ liệu PhoMT được cung cấp dưới dạng file nén `PhoMT.zip`, bao gồm hai thư mục chính:

- `tokenization/`: Dữ liệu văn bản đã được tách token.
- `detokenization/`: Dữ liệu văn bản ở dạng đã ghép token thành câu.

Mỗi thư mục gồm ba tập dữ liệu:

- `train/`: Dùng để huấn luyện mô hình.
- `dev/`: Dùng để đánh giá trong quá trình huấn luyện, lựa chọn mô hình và điều chỉnh siêu tham số.
- `test/`: Dùng để đánh giá cuối cùng.

Trong dự án này, nhóm dự kiến sử dụng dữ liệu từ thư mục `detokenization/` để xây dựng tokenizer và chuẩn bị đầu vào cho hai mô hình Seq2Seq Attention và Transformer.

## 3. Thống kê dữ liệu

| Tập dữ liệu | Số câu tiếng Anh | Số câu tiếng Việt | Câu rỗng | Dòng không khớp |
|---|---:|---:|---:|---:|
| Train | 2.977.999 | 2.977.999 | 0 | 0 |
| Dev | 18.719 | 18.719 | 0 | 0 |
| Test | 19.151 | 19.151 | 0 | 0 |
| **Tổng** | **3.015.869** | **3.015.869** | **0** | **0** |

Các số liệu được thu thập bằng cách kiểm tra trực tiếp bộ dữ liệu PhoMT đã tải về, sử dụng các lệnh Linux và chương trình Python.

## 4. Kết quả kiểm tra dữ liệu ban đầu

Những nội dung đã kiểm tra:

- File `PhoMT.zip` đã được tải và giải nén thành công.
- Xác định được đầy đủ ba tập train, dev và test.
- Số dòng tiếng Anh và tiếng Việt trong từng tập bằng nhau.
- Không phát hiện câu tiếng Anh rỗng.
- Không phát hiện câu tiếng Việt rỗng.
- Không phát hiện dòng không khớp giữa hai ngôn ngữ.
- Đã kiểm tra thủ công năm cặp câu mẫu.

Nhận xét: Bộ dữ liệu đáp ứng các yêu cầu cơ bản về cấu trúc và số lượng cặp câu song ngữ. Tuy nhiên, các kiểm tra trên chưa đủ để kết luận toàn bộ các cặp câu đều có chất lượng dịch chính xác.

Các nội dung cần kiểm tra bổ sung:

- Các cặp câu bị trùng lặp.
- Dữ liệu trùng lặp giữa train, dev và test.
- Phân bố độ dài câu.
- Các cặp câu có độ dài bất thường.
- Các trường hợp bản dịch không tương ứng về ngữ nghĩa.

## 5. Kế hoạch sử dụng dữ liệu

Nhóm dự kiến sử dụng PhoMT theo phương án sau:

1. Giữ nguyên cách chia train, dev và test của bộ dữ liệu gốc.
2. Sử dụng tiếng Anh làm ngôn ngữ nguồn và tiếng Việt làm ngôn ngữ đích.
3. Xây dựng tokenizer BPE hoặc SentencePiece dựa trên tập train.
4. Sử dụng một tập con dữ liệu train nếu tài nguyên GPU miễn phí không đủ để huấn luyện toàn bộ.
5. Không sử dụng tập test để huấn luyện hoặc lựa chọn siêu tham số.
6. Huấn luyện Seq2Seq LSTM Attention và Transformer trên cùng cấu hình dữ liệu để đảm bảo tính công bằng.
7. Đánh giá hai mô hình bằng BLEU trên cùng tập test.

Quy mô dữ liệu huấn luyện chính thức sẽ được quyết định sau khi nhóm chạy thử và đo thời gian huấn luyện trên GPU miễn phí.

## 6. Điều khoản sử dụng

Theo README chính thức của PhoMT:

- Bộ dữ liệu chỉ được sử dụng cho mục đích nghiên cứu hoặc giáo dục.
- Không được phân phối lại toàn bộ hoặc một phần dữ liệu, kể cả phiên bản đã chỉnh sửa.
- Phải trích dẫn bài báo PhoMT (EMNLP 2021) khi sử dụng dữ liệu để tạo ra kết quả nghiên cứu được công bố.

Do đó, nhóm sẽ không đưa dữ liệu PhoMT gốc hoặc đã xử lý lên GitHub. Repository chỉ lưu mã nguồn, cấu hình, hướng dẫn tải dữ liệu và các thông tin thống kê cần thiết.

## 7. Tài liệu tham khảo

Doan, Long, Linh The Nguyen, Nguyen Luong Tran, Thai Hoang, and Dat Quoc Nguyen. 2021. *PhoMT: A High-Quality and Large-Scale Benchmark Dataset for Vietnamese-English Machine Translation*. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021).

Nguồn: https://aclanthology.org/2021.emnlp-main.369/
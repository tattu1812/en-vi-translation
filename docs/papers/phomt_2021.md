# Ghi chú bài báo: PhoMT (EMNLP 2021)

## 1. Thông tin bài báo

- **Tên:** PhoMT: A High-Quality and Large-Scale Benchmark Dataset for Vietnamese-English Machine Translation
- **Tác giả:** Long Doan, Linh The Nguyen, Nguyen Luong Tran, Thai Hoang, Dat Quoc Nguyen.
- **Hội nghị:** EMNLP 2021.
- **Nhà xuất bản:** Association for Computational Linguistics.
- **DOI:** 10.18653/v1/2021.emnlp-main.369
- **Nguồn:** https://aclanthology.org/2021.emnlp-main.369/

## 2. Vấn đề nghiên cứu

Nghiên cứu tập trung vào tình trạng thiếu bộ dữ liệu song ngữ Việt–Anh có quy mô lớn và chất lượng tốt để huấn luyện, đánh giá các mô hình dịch máy thần kinh.

Các bộ dữ liệu nhỏ như IWSLT'15 có quy mô hạn chế, trong khi một số bộ dữ liệu lớn hơn tồn tại nhiều cặp câu không tương ứng về ngữ nghĩa hoặc có chất lượng bản dịch chưa tốt.

## 3. Đóng góp chính

Nghiên cứu giới thiệu PhoMT, bộ dữ liệu song ngữ Việt–Anh gồm 3.015.869 cặp câu.

Quy trình xây dựng dữ liệu gồm bốn giai đoạn:

1. Thu thập các cặp tài liệu song ngữ.
2. Tiền xử lý và trích xuất câu.
3. Căn chỉnh các cặp câu tương ứng.
4. Hậu xử lý, loại bỏ các cặp trùng lặp và kiểm tra thủ công chất lượng dữ liệu validation/test.

Dữ liệu được thu thập từ nhiều lĩnh vực như tin tức, bài viết hướng dẫn, TED Talks, MediaWiki và phụ đề phim.

## 4. Thống kê dữ liệu

| Tập dữ liệu | Số cặp câu |
|---|---:|
| Train | 2.977.999 |
| Validation | 18.719 |
| Test | 19.151 |
| Tổng | 3.015.869 |

Các số liệu trên đã được đối chiếu với bộ dữ liệu nhóm tải từ Hugging Face.

## 5. Phương pháp thực nghiệm trong bài báo

Các tác giả đánh giá nhiều hệ thống dịch máy, bao gồm Transformer-base, Transformer-big và mBART được fine-tune.

Kết quả được đánh giá bằng các phương pháp tự động và đánh giá của con người.

Trong phạm vi các hệ thống được nghiên cứu, mBART sau fine-tuning đạt kết quả tốt nhất theo các hình thức đánh giá này.

## 6. Ứng dụng vào dự án Deep Learning

Nhóm sử dụng PhoMT cho bài toán dịch Anh sang Việt.

Những nội dung kế thừa từ nghiên cứu:

- Sử dụng tập dữ liệu song ngữ có nguồn gốc học thuật rõ ràng.
- Giữ nguyên các tập train, validation và test đã được công bố.
- Tham khảo quy trình làm sạch và kiểm tra chất lượng dữ liệu.
- Đánh giá các mô hình trên cùng tập kiểm thử.
- Tham khảo Transformer và mBART làm các hướng nghiên cứu liên quan.

Những nội dung nhóm tự thực hiện:

- Xây dựng pipeline dữ liệu, tokenizer và DataLoader.
- Tự cài đặt Seq2Seq LSTM kết hợp Attention.
- Tự cài đặt Transformer nhỏ.
- Huấn luyện và đánh giá trên tập con PhoMT phù hợp với GPU miễn phí.
- So sánh BLEU và trực quan hóa attention map.

## 7. Hạn chế và lưu ý

- Không thể đảm bảo mọi cặp câu huấn luyện đều có chất lượng hoàn hảo.
- PhoMT có dữ liệu thuộc nhiều lĩnh vực nên chất lượng dịch có thể khác nhau theo miền nội dung.
- Việc sử dụng tập con dữ liệu có thể làm giảm chất lượng dịch so với huấn luyện toàn bộ.
- Kết quả của nhóm không được so sánh trực tiếp với kết quả bài báo gốc nếu thiết lập dữ liệu, mô hình và đánh giá khác nhau.

## 8. Kết luận

PhoMT phù hợp làm tập dữ liệu chính cho dự án vì có quy mô lớn, có cách chia train/validation/test sẵn và được công bố tại hội nghị EMNLP 2021.

Bài báo cung cấp cơ sở học thuật để nhóm giải thích việc lựa chọn dữ liệu và thiết kế quy trình thực nghiệm cho hai mô hình Seq2Seq Attention và Transformer.
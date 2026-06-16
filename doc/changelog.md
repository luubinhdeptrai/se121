# Changelog

Tài liệu này ghi lại tất cả các thay đổi đáng chú ý của dự án **Multimodal Sentiment Analysis**.
Định dạng được tham khảo từ [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-06-16 16:40

### Added (Đã thêm mới)
- **Gradient Accumulation**: Bổ sung cơ chế `grad_accum_steps` vào `Trainer.py` để giải quyết triệt để lỗi Out-of-Memory (OOM) do VRAM bị quá tải khi xử lý batch size lớn (đặc thù mỗi review có 4-5 hình ảnh độ phân giải cao).
- **Cosine LR Scheduler & Warmup**: Tích hợp hàm `get_cosine_schedule_with_warmup` vào `Trainer.py`. LR sẽ tăng dần ở những bước đầu (warmup) và giảm dần theo hình sin về cuối, giúp mô hình hội tụ sâu hơn và tránh phá hỏng trọng số pre-trained.
- **Layer-wise Unfreezing (Mở khóa theo layer)**: Thêm tham số `unfreeze_text_layers` và `unfreeze_image_layers` vào `FusionModel.py`. Cho phép "tan băng" dần dần một số layer cuối của Backbone (`XLM-RoBERTa` và `ConvNeXt`) để fine-tune thay vì khóa cứng hoàn toàn.
- **Tự động hóa Input (`_prepare_inputs`)**: Thêm hàm trợ giúp vào `Trainer.py` để tự động map dữ liệu từ batch vào đúng kwargs của mô hình, giúp hỗ trợ nhiều mode train (`train_text`, `train_image`, `train_fusion`) mà không cần `if-else`.
- **Pipeline Data Chuẩn 6000 Mẫu**: Viết lại hoàn toàn `preprocess_data.py` và `download_images.py`. Pipeline mới sẽ chạy qua toàn bộ tập dữ liệu thô, lọc bỏ URL chết, tải ảnh, cắt gọt chính xác đúng 6000 mẫu hoàn hảo (4800 Train / 600 Val / 600 Test).
- **Verify Script**: Thêm script `verify_dataset_script.py` để tự động kiểm định tính toàn vẹn của dữ liệu sau khi tải xong, trước khi đưa vào huấn luyện.

### Changed (Đã thay đổi / Cải tiến)
- **Gỡ bỏ `torch.no_grad()` tĩnh**: Chỉnh sửa hàm `forward` của `FusionModel.py` để PyTorch Autograd có thể tính toán Gradient cho các Layer đã được mở khóa.
- **Tối ưu Pooling Ảnh (`ImageModel.py`)**: Đưa toàn bộ logic xử lý Tensor 5D `[B, N, C, H, W]` và `Average Pooling` gộp nhiều ảnh vào bên trong lõi của `ImageModel`. Việc này giúp làm sạch hoàn toàn nhánh tính toán của `FusionModel`.
- **Clean Code & Refactor**: Gỡ bỏ hàng loạt các khối lệnh `if-else` lồng nhau phức tạp trong các file Model và vòng lặp Huấn luyện. Giúp mã nguồn Pythonic, ngắn gọn và đạt hiệu năng tính toán cao hơn.

### Removed (Đã gỡ bỏ)
- Loại bỏ các Baseline cũ kĩ, quá đơn giản (chỉ lấy 1 ảnh đại diện). Thay vào đó, mọi mô hình hiện tại đều bắt buộc tuân theo chuẩn xử lý toàn bộ hình ảnh trong review (Average Pooling).

---

## [Planned] - Kế hoạch Kiến trúc Sắp tới (Thử nghiệm Nhóm 3)

### Added (Dự kiến triển khai)
- **Bi-directional Cross-Attention**: Đưa `nn.MultiheadAttention` vào `FusionModel.py`. Text sẽ "nhìn" vào Image để trích xuất ngữ cảnh ảnh, và Image sẽ "nhìn" vào Text để tập trung vùng ảnh tương ứng.
- **Gating Mechanism (Cổng kiểm soát)**: Triển khai một cơ chế Sigmoid Gate tự động học cách đánh giá độ tin cậy của nhánh Text so với nhánh Image trên từng mẫu dữ liệu cụ thể. Mô hình cuối cùng sẽ là một trung bình có trọng số động (`combined_features = gate * text_features + (1 - gate) * image_features`).

*Ghi chú: Những kiến trúc dự kiến này được đúc kết từ phân tích thực tế mã nguồn SOTA dự án SE363.*

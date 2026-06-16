# Changelog

Tài liệu này ghi lại tất cả các thay đổi đáng chú ý của dự án **Multimodal Sentiment Analysis**.
Định dạng được tham khảo từ [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0] - 2026-06-16 20:48

### Added (Đã thêm mới)
- **Hỗ trợ Colab siêu tốc**: Nâng cấp notebook Colab hỗ trợ lệnh `gdown` để tải trực tiếp file nén `data.zip` và xả nén ngay trên ổ SSD máy ảo, rút ngắn thời gian chuẩn bị dữ liệu. Cập nhật các tham số train để tối ưu VRAM.
- **Early Stopping**: Bổ sung cơ chế dừng sớm (`patience=3`) trong vòng lặp của `Trainer.py`. Tự động ngắt huấn luyện nếu Val Loss không cải thiện sau 3 epochs, giữ lại checkpoint tốt nhất.
- **Cosine LR Scheduler & Warmup**: Tích hợp hàm `get_cosine_schedule_with_warmup` vào `Trainer.py`. LR sẽ tăng dần ở những bước đầu và giảm dần theo hình sin về cuối.
- **Layer-wise Unfreezing (Mở khóa theo layer)**: Thêm tham số `unfreeze_text_layers` và `unfreeze_image_layers` vào `FusionModel.py` để fine-tune.
- **Gradient Accumulation**: Bổ sung cơ chế `grad_accum_steps` vào `Trainer.py` để giải quyết triệt để lỗi Out-of-Memory (OOM) do VRAM bị quá tải khi xử lý batch size lớn.

---

## [1.0.1] - 2026-06-16 16:40

### Added (Đã thêm mới)
- **Pipeline Data Chuẩn 6000 Mẫu**: Viết lại hoàn toàn `preprocess_data.py` và `download_images.py`. Pipeline mới cắt gọt chính xác đúng 6000 mẫu hoàn hảo (4800 Train / 600 Val / 600 Test).
- **Verify Script**: Thêm script `verify_dataset_script.py` để tự động kiểm định tính toàn vẹn của dữ liệu sau khi tải.
- **Tự động hóa Input (`_prepare_inputs`)**: Thêm hàm trợ giúp vào `Trainer.py` để tự động map dữ liệu từ batch, gỡ bỏ `if-else`.

### Changed (Đã thay đổi / Cải tiến)
- **Gỡ bỏ `torch.no_grad()` tĩnh**: Chỉnh sửa hàm `forward` của `FusionModel.py` để PyTorch Autograd hoạt động chính xác cho các Layer được mở khóa.
- **Tối ưu Pooling Ảnh (`ImageModel.py`)**: Đưa toàn bộ logic xử lý Tensor 5D và `Average Pooling` vào trong lõi của `ImageModel`.
- **Clean Code & Refactor**: Gỡ bỏ hàng loạt các khối lệnh lồng nhau phức tạp trong vòng lặp Huấn luyện.

### Removed (Đã gỡ bỏ)
- Loại bỏ hoàn toàn các Baseline cũ kĩ, bắt buộc tuân theo chuẩn xử lý toàn bộ hình ảnh trong review (Average Pooling).

---

## [Planned] - Kế hoạch Kiến trúc Sắp tới (Thử nghiệm Nhóm 3)

### Added (Dự kiến triển khai)
- **Bi-directional Cross-Attention**: Đưa `nn.MultiheadAttention` vào `FusionModel.py`. Text sẽ "nhìn" vào Image để trích xuất ngữ cảnh ảnh, và Image sẽ "nhìn" vào Text để tập trung vùng ảnh tương ứng.
- **Gating Mechanism (Cổng kiểm soát)**: Triển khai một cơ chế Sigmoid Gate tự động học cách đánh giá độ tin cậy của nhánh Text so với nhánh Image trên từng mẫu dữ liệu cụ thể. Mô hình cuối cùng sẽ là một trung bình có trọng số động (`combined_features = gate * text_features + (1 - gate) * image_features`).


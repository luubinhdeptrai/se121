# Multimodal Food Review Prediction

Kho mã nguồn này triển khai một kiến trúc Late Fusion Multimodal bằng PyTorch để dự đoán điểm đánh giá đồ ăn (Điểm tổng quan - Overall và các Yếu tố cụ thể - Factors) dựa trên bình luận của người dùng và hình ảnh đi kèm.

## Kiến trúc Mô hình
- **Text Encoder (Xử lý văn bản):** XLM-RoBERTa (`xlm-roberta-base`)
- **Image Encoder (Xử lý hình ảnh):** ConvNeXt (`convnext_base`)
- **Fusion Module (Mô-đun kết hợp):** Concatenation (Nối vector) + Dense Layers (Các lớp kết nối đầy đủ)
- **Prediction Heads (Các nhánh dự đoán):** 
  - Điểm tổng quan (Overall Score từ 0-10)
  - Điểm thành phần (Food Quality - Chất lượng món ăn, Price - Giá cả, Atmosphere - Không gian)

## Chiến lược Huấn luyện (3 Giai đoạn)
Kho mã nguồn này sử dụng chiến lược huấn luyện từng bước để tránh hiện tượng một phương thức (như text) lấn át phương thức còn lại:

1. **Giai đoạn 1 (Train Text Model):** Huấn luyện độc lập nhánh Text Encoder.
2. **Giai đoạn 2 (Train Image Model):** Huấn luyện độc lập nhánh Image Encoder.
3. **Giai đoạn 3 (Train Fusion Model):** Tải lại trọng số đã được train tốt nhất từ Giai đoạn 1 & 2, đóng băng (freeze) các encoders này lại, và chỉ huấn luyện lớp Fusion và các nhánh Prediction cuối cùng.

## Cài đặt Môi trường
Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install torch transformers timm pandas pillow requests tqdm
```

## Hướng dẫn Chạy

**1. Chuẩn bị Dữ liệu**
Hãy chắc chắn rằng tập dữ liệu đã qua xử lý của bạn nằm ở thư mục `data/processed_reviews.csv` (chứa các cột `comment_clean`, `image_url`, `avg_rating`, `food_score`, `price_score`, `atmosphere_score`).

**2. Giai đoạn 1: Chỉ Train nhánh Text**
```bash
python main.py --mode train_text --epochs 5 --lr 2e-5
```

**3. Giai đoạn 2: Chỉ Train nhánh Image**
```bash
python main.py --mode train_image --epochs 5 --lr 2e-5
```

**4. Giai đoạn 3: Train nhánh Fusion**
*(Hãy đảm bảo rằng Giai đoạn 1 & 2 đã chạy xong để các file `checkpoints/best_model_train_text.pth` và `checkpoints/best_model_train_image.pth` tồn tại)*
```bash
python main.py --mode train_fusion --epochs 10 --lr 1e-4
```

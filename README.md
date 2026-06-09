# Multimodal Food Review Prediction

Kho mã nguồn này triển khai một kiến trúc Late Fusion Multimodal bằng PyTorch để dự đoán điểm đánh giá đồ ăn (Điểm tổng quan - Overall và các Yếu tố cụ thể - Factors) dựa trên bình luận của người dùng và hình ảnh đi kèm.

## Kiến trúc Mô hình
- **Text Encoder (Xử lý văn bản):** XLM-RoBERTa (`xlm-roberta-base`)
- **Image Encoder (Xử lý hình ảnh):** ConvNeXt (`convnext_base`)
- **Fusion Module (Mô-đun kết hợp):** Concatenation (Nối vector) + Dense Layers (Các lớp kết nối đầy đủ)
- **Prediction Heads (Các nhánh dự đoán):** 
  - Điểm tổng quan (Overall Score từ 0-10)
  - Điểm thành phần (Food Quality, Price, Atmosphere)

## Chiến lược Huấn luyện (3 Giai đoạn)
Kho mã nguồn này sử dụng chiến lược huấn luyện từng bước để tránh hiện tượng một phương thức (như text) lấn át phương thức còn lại:
1. **Giai đoạn 1 (Train Text Model):** Huấn luyện độc lập nhánh Text Encoder.
2. **Giai đoạn 2 (Train Image Model):** Huấn luyện độc lập nhánh Image Encoder.
3. **Giai đoạn 3 (Train Fusion Model):** Tải lại trọng số đã được train tốt nhất từ Giai đoạn 1 & 2, đóng băng (freeze) các encoders này lại, và chỉ huấn luyện lớp Fusion và các nhánh Prediction cuối cùng.

## Hướng dẫn chạy trên Kaggle / Máy cá nhân (Local)

Dù bạn chạy trên Kaggle hay trên máy cá nhân, quy trình chạy đều y hệt nhau vì mã nguồn sử dụng đường dẫn tương đối (`./data/...`). Trên Kaggle, bạn nên bật **Accelerator là GPU T4 x2**.

### Bước 1: Clone Code từ Github
Mở Terminal (hoặc tạo Cell mới trên Kaggle) và gõ:
```bash
git clone https://github.com/lechihoang/SE365.git
cd SE365
pip install -r requirements.txt
```

### Bước 2: Tải Dữ liệu Hình ảnh
Chạy script để tự động tải toàn bộ hình ảnh dựa vào các URL trong file CSV. Kaggle có tốc độ mạng cực nhanh nên việc này chỉ tốn vài giây:
```bash
python download_images.py
```

### Bước 3: Chạy Train các mô hình
Bạn hoàn toàn có thể tuỳ chỉnh siêu tham số (hyperparameters) bằng cách truyền argument vào lệnh chạy (giống repo gốc). Dưới đây là các tham số bạn có thể điều chỉnh:
- `--epochs`: Số vòng lặp huấn luyện (Mặc định: 5)
- `--batch_size`: Kích thước batch (Mặc định: 16)
- `--lr`: Learning rate (Mặc định: 2e-5)
- `--alpha`: Trọng số loss (Mặc định: 0.5)

Lần lượt chạy các lệnh sau:

**Giai đoạn 1: Train Text**
```bash
python main.py --mode train_text --epochs 5 --batch_size 16 --lr 2e-5
```

**Giai đoạn 2: Train Image**
```bash
python main.py --mode train_image --epochs 5 --batch_size 16 --lr 2e-5
```

**Giai đoạn 3: Train Fusion**
```bash
python main.py --mode train_fusion --epochs 10 --batch_size 16 --lr 1e-4
```

### Bước 4: Test Báo Cáo Kết Quả
Sau khi train xong, chạy lệnh test để báo cáo sai số MAE/MSE trên tập độc lập:
```bash
python test.py --mode train_fusion
```

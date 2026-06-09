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

## Hướng dẫn chạy trên Kaggle (GPU)

Do model xử lý hình ảnh ConvNeXt khá nặng, khuyến nghị chạy trên Kaggle GPU (T4 x2) thay vì máy cá nhân.

### Bước 1: Chuẩn bị Data trên Kaggle
1. Tạo một Kaggle Dataset mới, tải file `data.zip` chứa ảnh và file csv lên. Đặt tên Dataset ví dụ là `food-review-data`.
2. Trong Kaggle Notebook, bật **Accelerator là GPU T4 x2**.
3. Bấm Add Data và thêm dataset vừa tạo vào Notebook. (Đường dẫn data trên Kaggle lúc này sẽ là `/kaggle/input/food-review-data/data/...`)

### Bước 2: Clone Code từ Github
Tạo một block Code (Cell) và chạy lệnh sau để kéo code về:
```bash
!git clone https://github.com/lechihoang/SE365.git
%cd SE365
!pip install transformers timm
```

### Bước 3: Chạy Train các mô hình
Lần lượt chạy từng Cell để train (Lưu ý đường dẫn dữ liệu `--train_path`, `--val_path` và `--image_dir` phải trỏ đúng vào Kaggle Dataset của bạn):

**Giai đoạn 1: Train Text**
```bash
!python main.py --mode train_text \
  --train_path /kaggle/input/food-review-data/data/train.csv \
  --val_path /kaggle/input/food-review-data/data/val.csv \
  --image_dir /kaggle/input/food-review-data/data/images
```

**Giai đoạn 2: Train Image**
```bash
!python main.py --mode train_image \
  --train_path /kaggle/input/food-review-data/data/train.csv \
  --val_path /kaggle/input/food-review-data/data/val.csv \
  --image_dir /kaggle/input/food-review-data/data/images
```

**Giai đoạn 3: Train Fusion**
```bash
!python main.py --mode train_fusion \
  --train_path /kaggle/input/food-review-data/data/train.csv \
  --val_path /kaggle/input/food-review-data/data/val.csv \
  --image_dir /kaggle/input/food-review-data/data/images
```

### Bước 4: Test Báo Cáo Kết Quả
Sau khi train xong, dùng tập `test.csv` để chạy báo cáo kết quả cuối cùng:
```bash
!python test.py --mode train_fusion \
  --test_path /kaggle/input/food-review-data/data/test.csv \
  --image_dir /kaggle/input/food-review-data/data/images
```

## Chạy Local (Máy tính cá nhân)
Nếu muốn chạy test nhanh trên local, chỉ cần chạy các lệnh mặc định:
```bash
pip install -r requirements.txt
python main.py --mode train_text
```
(Các đường dẫn mặc định trong `Config.py` đã trỏ sẵn vào thư mục `./data/` ở local)

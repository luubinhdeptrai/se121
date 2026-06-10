# Multimodal Food Review Prediction

Kho mã nguồn này triển khai một kiến trúc Late Fusion Multimodal bằng PyTorch để dự đoán điểm đánh giá đồ ăn (Điểm tổng quan - Overall và các Yếu tố cụ thể - Factors) dựa trên bình luận của người dùng và hình ảnh đi kèm.

## Kiến trúc Mô hình
- **Text Encoder (Xử lý văn bản):** XLM-RoBERTa (`xlm-roberta-base`)
- **Image Encoder (Xử lý hình ảnh):** ConvNeXt (`convnext_base`)
- **Fusion Module (Mô-đun kết hợp):** Concatenation (Nối vector) + Dense Layers (Các lớp kết nối đầy đủ)
- **Prediction Heads (Các nhánh dự đoán):** 
  - Điểm tổng quan (Overall Score từ 0-10)
  - Điểm thành phần (Food Quality, Price, Atmosphere)

## 🎯 Quyết định Thiết kế (Design Choices) & Best Practices
Để mô hình đạt hiệu năng ổn định nhất với dữ liệu thực tế, repo này áp dụng 3 quy chuẩn (industry standards) trong Machine Learning:

1. **Phương pháp kết hợp Late Fusion (Concatenation)**:
   Thay vì dùng Cross-Attention đắt đỏ và dễ lỗi gradient vanishing, mô hình trích xuất đặc trưng của Text và Image độc lập rồi mới nối (concat) lại ở lớp cuối. Đây là Strong Baseline cực kỳ ổn định, ít tốn RAM GPU và thường được sử dụng rộng rãi trong các hệ thống gợi ý thực tế.
2. **Chiến lược Huấn luyện 3 Giai đoạn (Modality Warming)**:
   Huấn luyện End-to-end ngay từ đầu thường dẫn đến hiện tượng **"Modality Collapse"** (mô hình chỉ lười biếng nhìn vào Text mà bỏ qua Image). Do đó, repo này áp dụng chiến thuật:
   - **Giai đoạn 1 (Train Text):** Ép mô hình chỉ học từ chữ.
   - **Giai đoạn 2 (Train Image):** Ép mô hình chỉ học từ ảnh.
   - **Giai đoạn 3 (Train Fusion):** Đóng băng cả hai, chỉ train lớp nối ghép cuối cùng. Cách này đảm bảo cả hai phương thức đều phát huy tối đa 100% năng lực.
3. **Bài toán Hồi quy (Regression) thay vì Phân loại (Classification)**:
   Điểm số (ví dụ: Food Score 1-10) là dữ liệu có tính thứ tự liên tục (Ordinal). Việc dự đoán sai 8 thành 7 ít nghiêm trọng hơn 8 thành 1. Do đó, mô hình sử dụng hàm mất mát **MSELoss** (Mean Squared Error) để hiểu khoảng cách toán học của điểm số, thay vì biến chúng thành các nhãn phân loại rời rạc cứng nhắc.
## Hướng dẫn chạy trên Kaggle / Máy cá nhân (Local)

Dù bạn chạy trên Kaggle hay trên máy cá nhân, quy trình chạy đều y hệt nhau vì mã nguồn sử dụng đường dẫn tương đối (`./data/...`). Trên Kaggle, bạn nên bật **Accelerator là GPU T4 x2**.

### Bước 1: Clone Code từ Github
Mở Terminal (hoặc tạo Cell mới trên Kaggle) và gõ:
```bash
git clone https://github.com/lechihoang/SE365.git
cd SE365
pip install -r requirements.txt
```

### Bước 2: Tải & Nạp Dữ Liệu (Chuẩn Data Engineer)
Kho lưu trữ này không chứa dữ liệu (Chỉ chứa Code) để đảm bảo tốc độ clone và sự chuyên nghiệp. Cấu trúc dữ liệu yêu cầu như sau:
```text
data/
├── text/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── image/
    └── (5000 file .jpg)
```

**Cách đưa Data lên Kaggle cực nhanh (Không cần tải lại ảnh):**
1. Chạy lệnh `python download_images.py` sẵn trên máy cá nhân để tải ảnh vào `data/image`.
2. Nén **toàn bộ thư mục `data`** thành `data.zip` và Upload lên Kaggle thành một **Kaggle Dataset** (Ví dụ tên là `food-review-dataset`).
3. Mở Kaggle Notebook, chọn **Add Input** -> Chọn Dataset vừa tạo.
4. Tạo lối tắt (Symlink) nối Kaggle Dataset thẳng vào thư mục code bằng lệnh sau trong Notebook:
```bash
!ln -s /kaggle/input/food-review-dataset ./data
```
Lúc này, toàn bộ Code sẽ tự động nhìn thấy cả Text và Image mà không cần chỉnh sửa bất kỳ tham số đường dẫn nào!

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

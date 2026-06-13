# Multimodal Food Review Prediction

Kho mã nguồn này triển khai một kiến trúc Intermediate Fusion Multimodal bằng PyTorch để dự đoán điểm đánh giá đồ ăn (Điểm tổng quan - Overall và các Yếu tố cụ thể - Factors) dựa trên bình luận của người dùng và hình ảnh đi kèm.

## 📚 Hệ thống Tài liệu (Documentation)
Để giữ cho file README này ngắn gọn, toàn bộ các chi tiết kỹ thuật sâu hơn và hướng dẫn chạy thực tế đã được phân tách rõ ràng vào thư mục [`doc/`](./doc/). Tuỳ theo nhu cầu của bạn, hãy đọc các file sau:

- 🧠 **Cần tìm hiểu về Kiến trúc Mô hình (Intermediate Fusion, XLM-RoBERTa, ConvNeXt) & Các Quyết định Thiết kế (Joint MSE Loss):** Hãy đọc file [`doc/ARCHITECTURE_AND_METRICS.md`](./doc/ARCHITECTURE_AND_METRICS.md)
- 🚀 **Cần Hướng dẫn Train/Test trực tiếp trên Google Colab siêu tốc độ (Tích hợp Google Drive):** Hãy đọc file [`doc/COLAB_GUIDE.md`](./doc/COLAB_GUIDE.md)
- 📊 **Cần tìm hiểu cách xử lý, làm sạch và nạp dữ liệu từ Raw Data (Foody/ShopeeFood):** Hãy đọc file [`doc/DATA_SETUP.md`](./doc/DATA_SETUP.md)

## Hướng dẫn chạy & Cài đặt (Setup Guide)

Quy trình chạy và huấn luyện mô hình được thiết kế để hoạt động ổn định trên mọi môi trường (Máy cá nhân, Server, Cloud) nhờ việc sử dụng đường dẫn tương đối (`./data/...`). Để huấn luyện hiệu quả, bạn nên sử dụng máy có GPU (khuyên dùng GPU có VRAM từ 16GB trở lên).

### Bước 1: Clone Code từ Github
Mở Terminal và gõ các lệnh sau:
```bash
git clone https://github.com/lechihoang/SE365.git
cd SE365
pip install -r requirements.txt
```

### Bước 2: Tải & Nạp Dữ Liệu 
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

**Cách chuẩn bị dữ liệu từ Raw Data (nếu bạn chạy từ đầu):**
Nếu bạn có thư mục `data_raw/` chứa các file cào về từ Foody/ShopeeFood:
1. Chạy lệnh `python preprocess_data.py` để tự động lọc, merge và chia 5000 mẫu thành `train.csv`, `val.csv`, `test.csv` lưu vào `data/text/`.
2. Chạy lệnh `python download_images.py` để tự động tải tất cả hình ảnh từ các link có trong CSV về thư mục `data/image/`.

**Cách cấu hình đường dẫn Data (Cho mọi môi trường):**
Mặc định, mã nguồn sẽ tự động đọc dữ liệu từ thư mục `./data/` (ví dụ: `./data/text/train.csv` và `./data/image/`). Nếu dữ liệu của bạn nằm ở một thư mục khác, thiết lập như sau:

Khai báo trực tiếp đường dẫn của từng file thông qua các tham số của `main.py`:
- `--train_path`: Đường dẫn đến file train.csv *(Mặc định: `./data/text/train.csv`)*
- `--val_path`: Đường dẫn đến file val.csv *(Mặc định: `./data/text/val.csv`)*
- `--test_path`: Đường dẫn đến file test.csv *(Mặc định: `./data/text/test.csv`)*
- `--image_dir`: Đường dẫn đến thư mục ảnh *(Mặc định: `./data/image`)*

*Ví dụ chạy lệnh với data nằm ở ổ cứng ngoài:*
```bash
python main.py --mode train_text \
    --train_path /Volumes/External/data/text/train.csv \
    --val_path /Volumes/External/data/text/val.csv \
    --test_path /Volumes/External/data/text/test.csv \
    --image_dir /Volumes/External/data/image
```

### Bước 3: Chạy Train các mô hình
Bạn hoàn toàn có thể tuỳ chỉnh siêu tham số (hyperparameters) bằng cách truyền argument vào lệnh chạy (giống repo gốc). Dưới đây là các tham số bạn có thể điều chỉnh:
- `--mode`: Chế độ chạy (`train_text`, `train_image`, `train_fusion`)
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

## Kết Quả Thử Nghiệm (Benchmark)
Mô hình đã được chạy thử nghiệm trên tập dữ liệu 5000 mẫu với Notebook tham khảo tại: [`notebook/baseline.ipynb`](./notebook/baseline.ipynb). Dưới đây là kết quả của các cấu hình mô hình (Architecture) và hàm mất mát (Loss) khác nhau trên tập Test độc lập, được phân chia theo từng loại độ đo:

### 1. Bảng sai số tuyệt đối trung bình (MAE)
*Độ đo thực tế và dễ hiểu nhất, cho biết trung bình máy đoán lệch bao nhiêu điểm (trên thang 1-10).*
| Model / Architecture | Loss Function | Food | Price | Atmos | Service |
|----------------------|---------------|------|-------|-------|---------|
| XLM-R + ConvNeXt | Joint MSE | **1.0098** | **1.0483** | **1.0135** | **1.0330** |
| mDeBERTa + SigLIP | Joint MSE | 1.1016 | 1.0694 | 1.0547 | 1.1185 |

### 2. Bảng căn bậc hai sai số bình phương (RMSE)
*Độ đo phạt nặng các dự đoán sai lệch lớn (Outliers).*
| Model / Architecture | Loss Function | Food | Price | Atmos | Service |
|----------------------|---------------|------|-------|-------|---------|
| XLM-R + ConvNeXt | Joint MSE | **1.3870** | **1.4537** | **1.3673** | **1.3925** |
| mDeBERTa + SigLIP | Joint MSE | 1.4941 | 1.4820 | 1.4252 | 1.5198 |

### 3. Bảng sai số bình phương (MSE)
*Độ đo cơ sở để tối ưu hóa trong quá trình huấn luyện.*
| Model / Architecture | Loss Function | Food | Price | Atmos | Service |
|----------------------|---------------|------|-------|-------|---------|
| XLM-R + ConvNeXt | Joint MSE | **1.9238** | **2.1133** | **1.8695** | **1.9391** |
| mDeBERTa + SigLIP | Joint MSE | 2.2324 | 2.1963 | 2.0311 | 2.3099 |

**Đánh giá chung:** 
- Trung bình, mô hình Baseline (XLM-R + ConvNeXt) đang dẫn đầu và chỉ đoán sai khoảng **~1.01 - 1.04 điểm** trên các tiêu chí (tương đương với mức MAE ở bảng 1) so với điểm thực tế mà người dùng đánh giá. 
- Mặc dù lý thuyết mDeBERTa + SigLIP mạnh hơn, nhưng ConvNeXt lại tỏ ra ổn định và chống overfitting cực tốt hơn SigLIP trong bài toán Regression này với cùng chế độ huấn luyện.
- Việc tách thành 3 bảng riêng biệt giúp bạn dễ dàng so sánh một cách khoa học khi làm báo cáo.

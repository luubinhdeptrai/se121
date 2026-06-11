# Multimodal Food Review Prediction

Kho mã nguồn này triển khai một kiến trúc Late Fusion Multimodal bằng PyTorch để dự đoán điểm đánh giá đồ ăn (Điểm tổng quan - Overall và các Yếu tố cụ thể - Factors) dựa trên bình luận của người dùng và hình ảnh đi kèm.

## Tài liệu & Kiến trúc Mô hình
Chi tiết về Kiến trúc Mô hình (XLM-RoBERTa + ConvNeXt), các Quyết định Thiết kế (Late Fusion, Joint MSE) và Hướng dẫn chạy trên Google Colab đã được chuyển sang thư mục [`doc/`](./doc/). Vui lòng tham khảo các tài liệu trong đó để hiểu rõ cách thức hoạt động của mô hình.
## Hướng dẫn chạy & Cài đặt (Setup Guide)

Quy trình chạy và huấn luyện mô hình được thiết kế để hoạt động ổn định trên mọi môi trường (Máy cá nhân, Server, Cloud) nhờ việc sử dụng đường dẫn tương đối (`./data/...`). Để huấn luyện hiệu quả, bạn nên sử dụng máy có GPU (khuyên dùng GPU có VRAM từ 16GB trở lên).

### Bước 1: Clone Code từ Github
Mở Terminal và gõ các lệnh sau:
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

**Cách chuẩn bị dữ liệu từ Raw Data (nếu bạn chạy từ đầu):**
Nếu bạn có thư mục `data_raw/` chứa các file cào về từ Foody/ShopeeFood:
1. Chạy lệnh `python preprocess_data.py` để tự động lọc, merge và chia 5000 mẫu thành `train.csv`, `val.csv`, `test.csv` lưu vào `data/text/`.
2. Chạy lệnh `python download_images.py` để tự động tải tất cả hình ảnh từ các link có trong CSV về thư mục `data/image/`.

**Cách cấu hình đường dẫn Data (Cho mọi môi trường):**
Mặc định, mã nguồn sẽ tự động đọc dữ liệu từ thư mục `./data/` (ví dụ: `./data/text/train.csv` và `./data/image/`). Nếu dữ liệu của bạn nằm ở một thư mục khác, bạn có hai cách để thiết lập:

1. **Cách 1 (Khuyên dùng - Dùng Symlink):** 
   Tạo một lối tắt ảo nối thư mục dữ liệu thật của bạn vào thư mục code. Cách này giúp bạn gõ lệnh ngắn gọn hơn và không cần nhớ cấu trúc đường dẫn dài dòng:
   ```bash
   ln -s /đường/dẫn/thực/tế/đến/thư/mục/data ./data
   ```

2. **Cách 2 (Sử dụng tham số khi chạy lệnh):** 
   Bạn có thể khai báo trực tiếp đường dẫn của từng file thông qua các tham số của `main.py`:
   - `--train_path`: Đường dẫn đến file train.csv
   - `--val_path`: Đường dẫn đến file val.csv
   - `--test_path`: Đường dẫn đến file test.csv
   - `--image_dir`: Đường dẫn đến thư mục chứa 5000 file ảnh

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
Mô hình đã được chạy thử nghiệm trên tập dữ liệu 5000 mẫu với Notebook tham khảo tại: `notebook/baseline.ipynb`. Dưới đây là kết quả của các cấu hình mô hình (Architecture) và hàm mất mát (Loss) khác nhau trên tập Test độc lập, được phân chia theo từng loại độ đo:

### 1. Bảng sai số tuyệt đối trung bình (MAE)
*Độ đo thực tế và dễ hiểu nhất, cho biết trung bình máy đoán lệch bao nhiêu điểm (trên thang 1-10).*
| Model / Architecture | Loss Function | Overall | Food | Price | Atmos |
|----------------------|---------------|---------|------|-------|-------|
| Late Fusion (XLM-R + ConvNeXt) | Joint MSE | **0.9968** | **1.2220** | **1.2173** | **1.2327** |

### 2. Bảng căn bậc hai sai số bình phương (RMSE)
*Độ đo phạt nặng các dự đoán sai lệch lớn (Outliers).*
| Model / Architecture | Loss Function | Overall | Food | Price | Atmos |
|----------------------|---------------|---------|------|-------|-------|
| Late Fusion (XLM-R + ConvNeXt) | Joint MSE | 1.3950 | 1.6424 | 1.6432 | 1.7008 |

### 3. Bảng sai số bình phương (MSE)
*Độ đo cơ sở để tối ưu hóa trong quá trình huấn luyện.*
| Model / Architecture | Loss Function | Overall | Food | Price | Atmos |
|----------------------|---------------|---------|------|-------|-------|
| Late Fusion (XLM-R + ConvNeXt) | Joint MSE | 1.9460 | 2.6976 | 2.7000 | 2.8927 |

**Đánh giá chung:** 
- Trung bình, mô hình baseline hiện tại chỉ đoán sai **~0.99 điểm** (tương đương với mức MAE ở bảng 1) so với điểm Overall thực tế mà người dùng đánh giá.
- Việc tách thành 3 bảng riêng biệt giúp bạn dễ dàng so sánh một cách khoa học khi bổ sung các mô hình mới (ví dụ: PhoBERT, ViT) vào báo cáo.

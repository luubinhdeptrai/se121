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
Bạn hoàn toàn có thể tuỳ chỉnh siêu tham số (hyperparameters) bằng cách truyền argument vào lệnh chạy (giống repo gốc). Dưới đây là các tham số nổi bật bạn có thể điều chỉnh:
- `--mode`: Chế độ chạy (`train_text`, `train_image`, `train_fusion`)
- `--epochs`: Số vòng lặp huấn luyện (Mặc định: 5)
- `--batch_size`: Kích thước batch (Mặc định: 16)
- `--grad_accum_steps`: Tích luỹ gradient chống văng lỗi vRAM (Mặc định: 1)
- `--lr`: Learning rate (Mặc định: 2e-5)
- `--patience`: Số epoch Early Stopping chờ đợi (Mặc định: 3)
- `--warmup_ratio`: Tỷ lệ Warmup cho Scheduler (Mặc định: 0.1)
- `--unfreeze_text_layers`: Số layer cuối của Text Model để "tan băng" khi chạy Fusion (Mặc định: 0)
- `--unfreeze_image_layers`: Số block cuối của Image Model để "tan băng" khi chạy Fusion (Mặc định: 0)

Lần lượt chạy các lệnh sau:

**Giai đoạn 1: Train Text**
```bash
python main.py --mode train_text --epochs 15 --batch_size 8 --grad_accum_steps 2 --lr 2e-5
```

**Giai đoạn 2: Train Image**
```bash
python main.py --mode train_image --epochs 15 --batch_size 4 --grad_accum_steps 4 --lr 2e-5
```

**Giai đoạn 3: Train Fusion**
```bash
python main.py --mode train_fusion --epochs 10 --batch_size 4 --grad_accum_steps 4 --lr 1e-4 --unfreeze_text_layers 1 --unfreeze_image_layers 1
```

### Bước 4: Test Báo Cáo Kết Quả
Sau khi train xong, chạy lệnh test để báo cáo sai số MAE/MSE trên tập độc lập:
```bash
python test.py --mode train_fusion
```

## Kết Quả Thử Nghiệm (Benchmark)
Mô hình đã được chạy thử nghiệm trên tập dữ liệu đa phương thức (gồm cả ảnh và đánh giá). Dưới đây là kết quả của các cấu hình mô hình (Architecture) khác nhau trên tập Test độc lập, được phân chia theo từng loại độ đo:

### 1. Bảng sai số tuyệt đối trung bình (MAE)
*Độ đo thực tế và dễ hiểu nhất, cho biết trung bình máy đoán lệch bao nhiêu điểm (trên thang 1-10).*
| Mô hình | Food MAE | Price MAE | Atmos MAE | Service MAE | Overall MAE |
|---|---:|---:|---:|---:|---:|
| Nhóm 1.1: RoBERTa + CLIP | 1.4866 | 1.4212 | 1.3496 | 1.4606 | 1.2671 |
| Nhóm 1.2: ViSoBERT + ConvNeXt | **1.2212** | **1.2009** | **1.2452** | **1.2290** | **1.0103** |
| Nhóm 1.3: DeBERTa + SigLIP | 1.3944 | 1.3070 | 1.2949 | 1.3430 | 1.1653 |

### 2. Bảng căn bậc hai sai số bình phương (RMSE)
*Độ đo phạt nặng các dự đoán sai lệch lớn (Outliers).*
| Mô hình | Food RMSE | Price RMSE | Atmos RMSE | Service RMSE | Overall RMSE |
|---|---:|---:|---:|---:|---:|
| Nhóm 1.1: RoBERTa + CLIP | 1.9492 | 1.8256 | 1.7036 | 1.9093 | 1.6432 |
| Nhóm 1.2: ViSoBERT + ConvNeXt | **1.6453** | **1.5733** | **1.5872** | **1.6392** | **1.3317** |
| Nhóm 1.3: DeBERTa + SigLIP | 1.8544 | 1.7260 | 1.6766 | 1.8017 | 1.5368 |

### 3. Bảng sai số bình phương (MSE)
*Độ đo cơ sở để tối ưu hóa trong quá trình huấn luyện.*
| Mô hình | Food MSE | Price MSE | Atmos MSE | Service MSE | Overall MSE |
|---|---:|---:|---:|---:|---:|
| Nhóm 1.1: RoBERTa + CLIP | 3.7995 | 3.3327 | 2.9023 | 3.6454 | 2.7000 |
| Nhóm 1.2: ViSoBERT + ConvNeXt | **2.7071** | **2.4753** | **2.5193** | **2.6868** | **1.7734** |
| Nhóm 1.3: DeBERTa + SigLIP | 3.4386 | 2.9789 | 2.8111 | 3.2462 | 2.3619 |

**Đánh giá chung:** 
- **Nhóm 1.2 (ViSoBERT + ConvNeXt)** hiện đang giữ ngôi vương với MAE tổng thể (Overall) chỉ khoảng **1.01**. Điều này chứng tỏ việc kết hợp một mô hình ngôn ngữ chuyên sâu cho Tiếng Việt (ViSoBERT) và kiến trúc ảnh mạnh mẽ (ConvNeXt) là hướng đi tối ưu.
- **Nhóm 1.3 (DeBERTa + SigLIP)** cũng rất mạnh với MAE Overall **1.16**, chứng minh hiệu quả vượt trội của các mô hình ngôn ngữ lớn kết hợp chuẩn hóa ảnh đa phương thức (SigLIP).
- So với các cấu hình cũ (chỉ gồm 4 nhãn), việc dự đoán thêm nhãn tổng quát (Overall) giúp kiểm soát chất lượng mô hình một cách toàn diện. Việc loại bỏ hoàn toàn lỗi "văng điểm" (outliers) đã giúp MSE/RMSE giảm mạnh mẽ, ổn định hệ thống để chuẩn bị cho các module Giải thích (Explainable AI - XAI).

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
Để mô hình đạt hiệu năng ổn định nhất và dễ hiểu cho người mới bắt đầu, repo này sử dụng một kiến trúc **Baseline (Nền tảng cơ sở)** cực kỳ tường minh cùng với các chuẩn mực thực tế (industry standards) sau:

### 1. Kiến trúc Baseline: Sự kết hợp muộn (Late Fusion)
Thay vì dùng các cấu trúc phức tạp như Cross-Attention (dễ gây lỗi và quá tải phần cứng), chúng tôi chọn phương pháp **Late Fusion**:
- **Chuyên gia Văn bản (Text Encoder - XLM-RoBERTa):** Đóng vai trò đọc hiểu câu chữ. XLM-RoBERTa được chọn vì nó hỗ trợ tiếng Việt cực tốt, hiểu được các từ lóng ẩm thực.
- **Chuyên gia Hình ảnh (Image Encoder - ConvNeXt):** Đóng vai trò "nhìn" các bức ảnh món ăn, không gian quán. ConvNeXt là thế hệ mạng CNN hiện đại nhất, nhẹ và bén hơn ResNet truyền thống.
- **Tổ trưởng Tổng hợp (Fusion Module):** Hai chuyên gia trên làm việc hoàn toàn độc lập để rút ra nhận xét riêng. Sau đó, "Tổ trưởng" sẽ gom 2 nhận xét này lại (Concatenation) và đưa qua một lớp mạng nơ-ron đơn giản để chốt số điểm cuối cùng. Việc độc lập này giúp mô hình ổn định, tốn ít RAM và dễ bắt bệnh.

### 2. Hàm mất mát Baseline (Loss Function): Tại sao lại dùng Joint MSE?
Bài toán dự đoán điểm đánh giá (1-10) là bài toán **Hồi quy (Regression)** chứ không phải Phân loại (Classification) vì điểm số có tính liên tục (sai 8 thành 7 thì có thể chấp nhận, nhưng sai 8 thành 1 thì là thảm họa).
- **Hàm MSE (Mean Squared Error):** Phạt rất nặng các trường hợp máy đoán lệch quá xa so với thực tế (vì sai số bị bình phương lên). 
- **Joint Loss (Mất mát kết hợp):** Thay vì chỉ dạy máy dự đoán điểm "Overall", chúng tôi bắt máy phải học cách dự đoán cả 3 yếu tố chi tiết (Food, Price, Atmosphere) cùng lúc. Công thức Baseline: 
  `Tổng Loss = 0.5 * MSE_Overall + 0.5 * MSE_Factors`
  Việc này giúp mô hình "hiểu sâu" hơn: Nó sẽ nhận ra rằng điểm Overall thấp thường là do Food tệ hoặc Price quá đắt.

### 3. Chiến lược Huấn luyện 3 Giai đoạn (Modality Warming)
Nếu "bắt" mô hình học cả Text và Image ngay từ đầu, nó thường sinh ra tật lười biếng: Chuyên gia Text quá giỏi nên "Tổ trưởng" chỉ nghe Text mà lơ đi Image (Hiện tượng Modality Collapse). Do đó, ta áp dụng chiến thuật:
- **Giai đoạn 1:** Chỉ dạy chuyên gia Text.
- **Giai đoạn 2:** Chỉ dạy chuyên gia Image.
- **Giai đoạn 3:** Đóng băng 2 chuyên gia, chỉ dạy tổ trưởng cách tổng hợp ý kiến.
Cách này ép cả hai nguồn dữ liệu đều phát huy tối đa 100% năng lực.
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

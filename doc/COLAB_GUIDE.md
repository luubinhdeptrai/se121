# Hướng dẫn Chạy mô hình trên Google Colab (1-Click Run)

Tài liệu này hướng dẫn chi tiết cách chạy rèn luyện (Train) và đánh giá (Test) mô hình Multimodal trên Google Colab.


## BƯỚC 1: Chuẩn bị Dữ liệu bằng Lối tắt (Shortcut)

Nhóm đã chuẩn bị sẵn toàn bộ dữ liệu trên Google Drive. Bạn không cần tải về máy, chỉ cần tạo "Lối tắt" sang Drive của bạn:

1. Bấm vào link Drive được share sẵn của nhóm: https://drive.google.com/drive/folders/1uwKnhb2d3lOLlMIpRU-qnW67ar4LQWk2?usp=sharing
2. Nhấn vào tên thư mục ở trên cùng -> Chọn **Thêm lối tắt vào Drive (Add shortcut to Drive)**.
3. Chọn vị trí lưu là **Drive của tôi (MyDrive)**. Hãy đặt tên cho lối tắt này là `SE365` để khớp với mã nguồn.

---

## BƯỚC 2: Chạy trực tiếp từ Notebook có sẵn

Trong mã nguồn, mình đã chuẩn bị sẵn một file Notebook chuyên dụng dành cho môi trường Colab: `notebook/colab.ipynb`.

Bạn chỉ cần mở file `notebook/colab.ipynb` trên Google Colab và chạy lần lượt các Cell từ trên xuống dưới. Notebook đã tự động hóa mọi thứ từ việc kết nối Drive, copy dữ liệu đến việc Train các mô hình và Đánh giá (Test).

---

## BƯỚC 3: Cách làm thủ công (Dành cho người muốn tự cấu hình)

Nếu bạn không muốn xài Notebook có sẵn mà muốn tự gõ lệnh trên Colab, hãy làm theo trình tự chuẩn sau:

**1. Kết nối với Google Drive**
Thêm một cell Code và gõ:
```python
from google.colab import drive
drive.mount('/content/drive')
```

**2. Tải mã nguồn & Cài đặt thư viện**
Thêm một cell khác:
```bash
!git clone https://github.com/lechihoang/SE365.git
%cd SE365
!pip install -r requirements.txt
```

**3. Tải Dữ liệu từ Lối tắt Drive vào Máy ảo (Tăng tốc tối đa)**
Thay vì trỏ Symlink, ta sẽ copy toàn bộ thư mục `data` từ Lối tắt Drive vào thẳng ổ cứng của Colab để tốc độ huấn luyện đạt mức tối đa (nhanh như Kaggle):
```bash
!rm -rf ./data
!cp -r /content/drive/MyDrive/SE365/data ./data
```
*(Nếu lối tắt của bạn không tên là `SE365`, hãy sửa đường dẫn trên cho phù hợp)*

**4. Khởi tạo Thư mục Lưu trữ cho Phiên chạy**
Để đảm bảo tất cả các mô hình (Text, Image, Fusion) trong cùng một lần chạy được lưu chung vào một thư mục, hãy khởi tạo một biến môi trường:
```python
import os
import datetime

run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
drive_ckpt_path = f'/content/drive/MyDrive/SE365/checkpoints/{run_id}'
os.environ['DRIVE_CKPT'] = drive_ckpt_path

# Tạo thư mục con checkpoints và thư mục run_id bằng os.makedirs
os.makedirs(drive_ckpt_path, exist_ok=True)
print(f'Mọi checkpoint trong phiên này sẽ được lưu chung vào: {drive_ckpt_path}')
```

**5. Chạy Training và Đánh Giá**

Ngoài các tham số đường dẫn, bạn cũng có thể tùy chỉnh các siêu tham số (hyperparameters) để tối ưu mô hình. Dưới đây là danh sách các tham số chính của `main.py`:

**Tham số đường dẫn:**
- `--train_path`: Đường dẫn đến file train.csv
- `--val_path`: Đường dẫn đến file val.csv
- `--test_path`: Đường dẫn đến file test.csv
- `--image_dir`: Đường dẫn đến thư mục chứa 5000 file ảnh

**Tham số huấn luyện (Hyperparameters):**
- `--epochs`: Số vòng lặp huấn luyện (Mặc định: 5)
- `--batch_size`: Kích thước batch (Mặc định: 16)
- `--lr`: Learning rate (Mặc định: 2e-5)
- `--alpha`: Trọng số loss (Mặc định: 0.5)

Lưu ý: Bạn có thể copy từng cụm lệnh dưới đây vào **các cell riêng biệt** trên Colab để dễ dàng theo dõi quá trình chạy. Sau mỗi bước Training, trọng số sẽ tự động được copy vào chung thư mục `$DRIVE_CKPT` trên Google Drive.

**5.1. Huấn luyện mô hình Text**
```bash
!python main.py --mode train_text --epochs 5 --batch_size 16 --lr 2e-5 --train_path ./data/text/train.csv --val_path ./data/text/val.csv --test_path ./data/text/test.csv
!cp ./checkpoints/best_model_train_text.pth $DRIVE_CKPT/
```

**5.2. Huấn luyện mô hình Image**
```bash
!python main.py --mode train_image --epochs 10 --batch_size 16 --lr 2e-5 --train_path ./data/text/train.csv --val_path ./data/text/val.csv --test_path ./data/text/test.csv --image_dir ./data/image
!cp ./checkpoints/best_model_train_image.pth $DRIVE_CKPT/
```

**5.3. Huấn luyện mô hình Fusion kết hợp**
```bash
!python main.py --mode train_fusion --epochs 15 --batch_size 16 --lr 2e-5 --alpha 0.5 --train_path ./data/text/train.csv --val_path ./data/text/val.csv --test_path ./data/text/test.csv --image_dir ./data/image
!cp ./checkpoints/best_model_train_fusion.pth $DRIVE_CKPT/
```

**5.4. Đánh giá kiểm thử (Test) trên mô hình tốt nhất**
```bash
!python test.py --mode train_fusion --test_path ./data/text/test.csv --image_dir ./data/image
```


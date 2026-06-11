# Hướng dẫn Chạy mô hình trên Google Colab (1-Click Run)

Tài liệu này hướng dẫn chi tiết cách chạy rèn luyện (Train) và đánh giá (Test) mô hình Multimodal trên Google Colab.

Được thiết kế theo cấu trúc "Code một nơi, Data một nẻo", luồng thực thi này giúp bạn tránh phải tải đi tải lại dữ liệu khổng lồ lên hệ thống của Colab.

## BƯỚC 1: Chuẩn bị Dữ liệu bằng Lối tắt (Shortcut)

Nhóm đã chuẩn bị sẵn toàn bộ dữ liệu trên Google Drive. Bạn không cần tải về máy, chỉ cần tạo "Lối tắt" sang Drive của bạn:

1. Bấm vào link Drive được share sẵn của nhóm: [Link Drive Share Dữ liệu](https://drive.google.com/drive/folders/1uwKnhb2d3lOLlMIpRU-qnW67ar4LQWk2?usp=sharing).
2. Nhấn vào tên thư mục ở trên cùng -> Chọn **Thêm lối tắt vào Drive (Add shortcut to Drive)**.
3. Chọn vị trí lưu là **Drive của tôi (MyDrive)**. Hãy đặt tên cho lối tắt này là `SE365` để khớp với mã nguồn.

---

## BƯỚC 2: Chạy trực tiếp từ Notebook có sẵn

Trong mã nguồn, mình đã chuẩn bị sẵn một file Notebook chuyên dụng dành cho môi trường Colab: `notebook/colab.ipynb`.

1. Bạn lên Github, mở file `notebook/colab.ipynb`.
2. Tải file này về máy tính và đưa lên Google Colab (vào File > Upload notebook).
3. Chạy lần lượt các Cell từ trên xuống dưới. Notebook đã tự động hóa mọi thứ bao gồm:
   - Kết nối vào Google Drive của bạn.
   - Kéo (Clone) mã nguồn mới nhất từ Github.
   - Copy dữ liệu từ Lối tắt vào máy ảo Colab.
   - Train các mô hình và Đánh giá (Test).

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
Lưu ý: Sau mỗi bước Training, trọng số sẽ tự động được copy ngay lập tức vào chung thư mục `$DRIVE_CKPT` trên Google Drive.

```bash
# 5.1. Huấn luyện mô hình Text
!python main.py --mode train_text --epochs 5
!cp ./checkpoints/best_model_train_text.pth $DRIVE_CKPT/

# 5.2. Huấn luyện mô hình Image
!python main.py --mode train_image --epochs 10
!cp ./checkpoints/best_model_train_image.pth $DRIVE_CKPT/

# 5.3. Huấn luyện mô hình Fusion kết hợp
!python main.py --mode train_fusion --epochs 15
!cp ./checkpoints/best_model_train_fusion.pth $DRIVE_CKPT/

# 5.4. Đánh giá kiểm thử (Test) trên mô hình tốt nhất
!python test.py --mode train_fusion
```

Lúc này, toàn bộ quá trình sẽ diễn ra hoàn toàn tự động từ A tới Z!

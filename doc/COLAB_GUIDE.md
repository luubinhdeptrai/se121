# Hướng dẫn Chạy mô hình trên Google Colab (1-Click Run)

Tài liệu này hướng dẫn chi tiết cách chạy rèn luyện (Train) và đánh giá (Test) mô hình Multimodal trên Google Colab.

Được thiết kế theo cấu trúc "Code một nơi, Data một nẻo", luồng thực thi này giúp bạn tránh phải tải đi tải lại file `data.zip` khổng lồ lên hệ thống của Colab.

## BƯỚC 1: Chuẩn bị Dữ liệu trên Google Drive

1. Nén toàn bộ thư mục `data` thành file `data.zip` (để tối ưu hóa tốc độ truyền tải trên Colab).
2. Tải file `data.zip` lên Google Drive của bạn, hoặc bạn có thể sử dụng link Drive được share sẵn của nhóm: [Link Drive Share](https://drive.google.com/drive/folders/1uwKnhb2d3lOLlMIpRU-qnW67ar4LQWk2?usp=sharing).
3. Nếu dùng link share, hãy chuột phải vào file `data.zip` bên trong thư mục đó -> Chọn **Thêm lối tắt vào Drive (Add shortcut to Drive)** -> Lưu vào thư mục gốc `MyDrive` của bạn.

---

## BƯỚC 2: Chạy trực tiếp từ Notebook có sẵn

Trong mã nguồn, mình đã chuẩn bị sẵn một file Notebook chuyên dụng dành cho môi trường Colab: `notebook/colab.ipynb`.

1. Bạn lên Github, mở file `notebook/colab.ipynb`.
2. Tải file này về máy tính và đưa lên Google Colab (vào File > Upload notebook).
3. Chạy lần lượt các Cell từ trên xuống dưới. Notebook đã tự động hóa mọi thứ bao gồm:
   - Kết nối vào Google Drive của bạn.
   - Kéo (Clone) mã nguồn mới nhất từ Github.
   - Trỏ dữ liệu ảo (Symlink) để kết nối code với Google Drive.
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

**3. Tải và Giải nén File Dữ liệu (Phương pháp Gold Standard)**
Thay vì copy từng file nhỏ, ta sẽ copy thẳng file `data.zip` từ Drive xuống ổ cứng cục bộ và giải nén. Tốc độ sẽ tăng từ vài phút xuống còn chưa tới 30 giây:
```bash
!rm -rf ./data
# Nếu bạn tự up file data.zip lên Drive của mình (ví dụ thư mục SE365)
!cp /content/drive/MyDrive/SE365/data.zip ./data.zip

# Nếu bạn dùng Lối tắt (Shortcut) từ link Drive Share ở Bước 1
# !cp /content/drive/MyDrive/data.zip ./data.zip

!unzip -q ./data.zip -d ./
```
*(Bạn hãy chỉnh sửa tham số đường dẫn `/content/drive/MyDrive/...` ở trên sao cho trỏ đúng tới vị trí file `data.zip` hoặc Lối tắt trong Drive của bạn)*

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

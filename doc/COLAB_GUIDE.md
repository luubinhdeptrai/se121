# Hướng dẫn Chạy mô hình trên Google Colab (1-Click Run)

Tài liệu này hướng dẫn chi tiết cách chạy rèn luyện (Train) và đánh giá (Test) mô hình Multimodal trên Google Colab.


## BƯỚC 1: Chuẩn bị Dữ liệu bằng Lối tắt (Shortcut)

Nhóm đã chuẩn bị sẵn toàn bộ dữ liệu trên Google Drive. Bạn không cần tải về máy, chỉ cần tạo "Lối tắt" sang Drive của bạn:

1. Bấm vào link Drive được share sẵn của nhóm: https://drive.google.com/drive/folders/1uwKnhb2d3lOLlMIpRU-qnW67ar4LQWk2?usp=sharing
2. Nhấn vào tên thư mục ở trên cùng -> Chọn **Thêm lối tắt vào Drive (Add shortcut to Drive)**.
3. Chọn vị trí lưu là **Drive của tôi (MyDrive)**. Hãy đặt tên cho lối tắt này là `SE365` để khớp với mã nguồn.

---

## BƯỚC 2: Chạy trực tiếp từ Notebook có sẵn

Trong mã nguồn, mình đã chuẩn bị sẵn một file Notebook chuyên dụng dành cho môi trường Colab: [`notebook/colab.ipynb`](../notebook/colab.ipynb).

Bạn chỉ cần mở file [`notebook/colab.ipynb`](../notebook/colab.ipynb) trên Google Colab và chạy lần lượt các Cell từ trên xuống dưới. Notebook đã tự động hóa mọi thứ từ việc kết nối Drive, copy dữ liệu đến việc Train các mô hình và Đánh giá (Test).


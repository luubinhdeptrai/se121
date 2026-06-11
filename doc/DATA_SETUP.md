# Cấu trúc và Thiết lập Dữ liệu (Data Setup Guide)

Trong bài toán Multimodal (Text + Image), việc quản lý dữ liệu hiệu quả là yếu tố sống còn. Tài liệu này sẽ hướng dẫn bạn cách thiết lập dữ liệu chuẩn nhất trên các môi trường khác nhau.

## 1. Cấu trúc Dữ liệu Chuẩn
Mã nguồn mong đợi một cấu trúc dữ liệu như sau:
```text
data/
├── text/
│   ├── train.csv (Tập huấn luyện)
│   ├── val.csv   (Tập kiểm thử trong quá trình train)
│   └── test.csv  (Tập đánh giá độc lập cuối cùng)
└── image/
    ├── img_0001.jpg
    ├── img_0002.jpg
    └── ... (5000+ files ảnh)
```

**Tại sao lại gộp chung ảnh vào một thư mục `image/`?**
Khác với các bài toán Image Classification cơ bản chia ảnh ra thư mục `train/`, `val/`, bài toán Multimodal sử dụng file `.csv` làm hệ thống Mục lục (Index). 
Trong file CSV sẽ ghi rõ `image_id` khớp với dòng Text nào và thuộc tập rèn luyện nào. Việc gộp chung ảnh giúp bạn dễ dàng thay đổi tỷ lệ Train/Val/Test (bằng cách chạy lại tập lệnh chia CSV) mà không phải mất thời gian Copy/Move hàng ngàn file ảnh vật lý trên ổ cứng.

---

## 2. Hướng dẫn thiết lập trên Google Colab

Quy trình chạy và kết nối dữ liệu trên Google Colab (sử dụng Google Drive và Symlink) đã được tách riêng thành một tài liệu chi tiết. 

👉 **Vui lòng xem tại: [Hướng dẫn Google Colab (COLAB_GUIDE.md)](./COLAB_GUIDE.md)**

---

## 3. Hướng dẫn thiết lập trên Local (Máy tính cá nhân/Server)

Nếu bạn tải dữ liệu về máy, bạn có hai cách để cấu hình:

**Cách 1: Sử dụng Symlink (Khuyên dùng trên Linux/Mac)**
```bash
ln -s /đường/dẫn/thực/tế/đến/thư/mục/data ./data
```

**Cách 2: Ghi đè tham số dòng lệnh**
Nếu bạn dùng Windows hoặc không thích Symlink, hãy truyền đường dẫn thẳng vào lệnh chạy:
```bash
python main.py --mode train_text \
   --train_path "D:/AI_Data/SE365/text/train.csv" \
   --val_path "D:/AI_Data/SE365/text/val.csv" \
   --test_path "D:/AI_Data/SE365/text/test.csv" \
   --image_dir "D:/AI_Data/SE365/image"
```

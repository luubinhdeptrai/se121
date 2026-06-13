# Hướng dẫn tạo Final Dataset (Dữ liệu huấn luyện cuối cùng)

Tài liệu này mô tả toàn bộ vòng đời và quy trình từng bước để từ con số 0 (chưa có dữ liệu), bạn có thể sinh ra bộ dataset cuối cùng (`train.csv`, `val.csv`, `test.csv` và thư mục ảnh `image/`) phục vụ cho việc huấn luyện mô hình dự đoán (kể cả nhãn `overall_satisfaction` mới).

## Quy trình 4 bước tổng quan:

### Bước 1: Thu thập dữ liệu (Crawl Data)
- **Công cụ:** Chạy file notebook `notebook/crawl_data_from_foody.ipynb`
- **Chức năng:** Cào dữ liệu thô từ trang Foody / ShopeeFood bao gồm thông tin đánh giá, điểm số (Food, Price, Atmosphere, Service), bình luận và đường dẫn hình ảnh (image_url).
- **Đầu ra (Output):** 
  - `data_raw/multimodal_reviews.csv` (Bảng map giữa comment và ảnh)
  - `data_raw/reviews_raw.csv` (Dữ liệu đánh giá thô)

### Bước 2: Làm sạch dữ liệu văn bản (Clean Text)
- **Công cụ:** Chạy file notebook `notebook/clean_foody_dataset.ipynb`
- **Chức năng:** Xoá bỏ các ký tự rác, emoji thừa, chuẩn hoá text tiếng Việt, loại bỏ các dòng bị thiếu dữ liệu hoặc không hợp lệ.
- **Đầu ra (Output):** 
  - `data_raw/reviews_clean.csv`
  - `data_raw/reviews_clean.json`

### Bước 3: Nội suy điểm hài lòng tổng quan (Generate Overall Satisfaction)
- **Công cụ:** Chạy file notebook `notebook/01_generate_overall_satisfaction.ipynb`
- **Chức năng:** Ứng dụng một cỗ máy quy tắc (Rule Engine) và phân tích ngôn ngữ học (Corpus Analysis) để tự động sinh ra trường `overall_satisfaction` mới. Điểm này phản ánh mức độ hài lòng thực sự của người dùng một cách công bằng hơn là điểm `avg_rating` trung bình cộng thô cứng.
- **Đầu ra (Output):** 
  - `data_processed/reviews_clean_enhanced.csv` (Đã được bổ sung thêm cột `overall_satisfaction`)
  - `data_processed/reviews_clean_enhanced.json`

### Bước 4: Trộn, Lọc và Chia Tập Huấn Luyện (Preprocess & Split)
- **Công cụ:** Chạy script `python preprocess_data.py`
- **Chức năng:** 
  - Đọc file `data_raw/multimodal_reviews.csv` và merge với file `data_processed/reviews_clean_enhanced.csv` thông qua `review_id`.
  - Lọc bỏ tất cả các dòng bị thiếu điểm đánh giá (NaN) ở các cột quan trọng (bao gồm cả `overall_satisfaction`).
  - Lấy mẫu ngẫu nhiên (sample) đúng 5000 dòng để cân bằng và giảm tải phần cứng.
  - Tự động chia thành 3 tệp theo tỷ lệ 80-10-10.
- **Đầu ra (Output):** 
  - `data/text/train.csv` (4000 mẫu)
  - `data/text/val.csv` (500 mẫu)
  - `data/text/test.csv` (500 mẫu)

### Bước 5: Tải hình ảnh về máy (Download Images)
- **Công cụ:** Chạy script `python download_images.py`
- **Chức năng:** Duyệt qua tất cả các URL hình ảnh (`image_url`) trong 3 file CSV ở Bước 4, tải chúng về máy cục bộ.
- **Đầu ra (Output):** 
  - Thư mục `data/image/` chứa đầy đủ 5000 tấm ảnh định dạng JPG tương ứng với 5000 mẫu dữ liệu.

---

**Kết luận:**
Sau khi hoàn tất 5 bước trên, cấu trúc thư mục `./data/` của bạn đã hoàn hảo và sẵn sàng để chạy `python main.py --mode train_text/image/fusion`. Mọi quy trình đều được tự động hoá thông qua các script!

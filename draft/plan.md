# Kế hoạch Xây dựng Multimodal Model cho Review Đồ ăn

Tài liệu này mô tả chi tiết kế hoạch xây dựng một mô hình Multimodal (kết hợp Text và Image) dựa trên tập dữ liệu đánh giá đồ ăn tại `/Users/abc/Documents/SE/checkpoints_clean` và kiến trúc hệ thống mục tiêu.

## 1. Mục tiêu Dự án
Xây dựng một mô hình sử dụng cả **Hình ảnh (Image)** và **Văn bản (Text)** từ các review đồ ăn để dự đoán:
- **Overall Score** (Điểm tổng quan từ 0-10)
- **Factor Scores** (Các điểm thành phần như Chất lượng món ăn - Quality, Giá cả - Price, Hình thức - Appearance/Không gian)
Trong giai đoạn này, mục tiêu chính là tập trung vào việc **huấn luyện mô hình (Training Model)** đến phần kết hợp (Fusion) và dự đoán các điểm số này.

## 2. Phân tích Dữ liệu (Dataset)
Tập dữ liệu sử dụng nằm trong `/Users/abc/Documents/SE/checkpoints_clean/`.
Các file quan trọng bao gồm:
- `multimodal_reviews.csv`: Chứa dữ liệu review đã được map với hình ảnh (gồm `comment_clean`, `avg_rating`, `image_url`).
- `reviews_clean.csv`: Chứa dữ liệu review văn bản chi tiết kèm các điểm thành phần (`food_score`, `service_score`, `atmosphere_score`, `price_score`, `avg_rating`).
- `review_images_clean.csv` / `json`: Chứa metadata của các hình ảnh.

**Kế hoạch xử lý dữ liệu:**
1. **Merge dữ liệu:** Kết hợp `multimodal_reviews.csv` và `reviews_clean.csv` thông qua `review_id` để có đầy đủ cả văn bản, hình ảnh, điểm tổng quan và các điểm thành phần.
2. **Tiền xử lý Text:** Làm sạch văn bản `comment_clean`, loại bỏ các ký tự không cần thiết, tokenization chuẩn bị cho mô hình ngôn ngữ.
3. **Tiền xử lý Image:** Tải ảnh từ `image_url` (nếu chưa có sẵn local) hoặc xử lý ảnh local, resize (ví dụ: 224x224), chuẩn hóa (normalize) theo yêu cầu của mô hình ConvNeXt.
4. **Chia tập dữ liệu:** Tách dữ liệu thành các tập Train / Validation / Test (ví dụ: 80-10-10).

## 3. Kiến trúc Mô hình (Architecture)
Theo sơ đồ kiến trúc `/Users/abc/Documents/SE/architechture.png`, hệ thống sẽ gồm các thành phần sau:

### 3.1. Feature Extraction (Trích xuất Đặc trưng)
- **Image Encoder:** Sử dụng **ConvNeXt** (pretrained trên ImageNet). Đầu ra là các feature vectors đại diện cho đặc trưng hình ảnh đồ ăn/quán ăn.
- **Text Encoder:** Sử dụng **XLM-RoBERTa** (hỗ trợ đa ngôn ngữ, đặc biệt tốt cho tiếng Việt). Đầu ra là các embedding vectors đại diện cho nội dung bình luận.

### 3.2. Fusion Module (Mô-đun Kết hợp)
Sử dụng phương pháp **Concatenation** hoặc **Cross-Attention**:
- *Concatenation:* Nối trực tiếp vector đặc trưng của ảnh và văn bản.
- *Cross-Attention:* Cho phép mô hình học được mối liên hệ chéo giữa các từ trong văn bản và các vùng trong hình ảnh.

### 3.3. Prediction Heads (Các nhánh Dự đoán)
Từ biểu diễn đa phương thức (multimodal representation), đưa qua các lớp Fully Connected (Dense layers) để dự đoán:
- **Overall Score:** Điểm tổng quan (từ 0 đến 10). Sử dụng hàm mất mát MSE hoặc MAE.
- **Factor Scores:** Điểm cho các yếu tố cụ thể: Chất lượng (Quality/Food Score), Giá cả (Price Score), Hình thức/Không gian (Appearance/Atmosphere).



## 4. Công nghệ & Framework
- **Deep Learning Framework:** PyTorch.
- **Thư viện AI:** 
  - `transformers` (HuggingFace) cho XLM-RoBERTa.
  - `timm` cho ConvNeXt.

## 5. Các bước Triển khai
1. **Giai đoạn 1: Chuẩn bị Dữ liệu**
   - Khảo sát kỹ dữ liệu, merge các CSV, viết script tải/đọc hình ảnh.
   - Xây dựng Dataloader (PyTorch Dataset).
2. **Giai đoạn 2: Xây dựng & Huấn luyện Mô hình**
   - Xây dựng Image Encoder (ConvNeXt) và Text Encoder (XLM-RoBERTa).
   - Ghép nối qua Fusion Module (Concatenation).
   - Xây dựng Prediction Heads cho Overall Score và Factor Scores.
   - Viết vòng lặp huấn luyện (Training loop) và đánh giá (Validation).

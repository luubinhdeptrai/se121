# Quyết định Thiết kế và Đánh giá (Architecture & Metrics)

Tài liệu này giải thích chi tiết các quyết định đằng sau việc lựa chọn mô hình và các hàm Loss trong mã nguồn.

## 1. Kiến trúc Mô hình (Model Architecture)

Mô hình hiện tại đang sử dụng phương pháp **Intermediate Fusion** (Kết hợp trung gian), trong đó Dữ liệu văn bản (Text) và Hình ảnh (Image) được xử lý song song bởi hai Backbone độc lập để trích xuất đặc trưng rồi mới gộp lại.

### Text Backbone: XLM-RoBERTa
- **Lý do lựa chọn:** Dữ liệu có thể chứa ngôn ngữ đa dạng (ví dụ: đánh giá nhà hàng bằng tiếng Việt, Anh, v.v.). XLM-RoBERTa là mô hình ngôn ngữ đa ngữ (Multilingual) hàng đầu hiện nay, vượt trội hơn so với mBERT truyền thống, đặc biệt trong việc hiểu ngữ cảnh cảm xúc của câu.
- **Vai trò:** Trích xuất đặc trưng ngữ nghĩa từ câu đánh giá, tạo ra một Vector `[CLS]` đại diện cho toàn bộ nội dung.

### Image Backbone: ConvNeXt
- **Lý do lựa chọn:** ConvNeXt là kiến trúc lai giữa sức mạnh của Transformer và tốc độ của CNN. Nó đem lại hiệu suất vượt trội trên ảnh tự nhiên so với các mô hình ResNet cũ kỹ, đồng thời tiết kiệm tài nguyên tính toán hơn so với ViT (Vision Transformer) thuần túy.
- **Vai trò:** Trích xuất đặc trưng thị giác từ ảnh món ăn/nhà hàng (ví dụ: màu sắc món ăn, độ sáng không gian).

### Cơ chế Kết hợp (Fusion Mechanism)
- Vector đặc trưng từ Text (768 chiều) và Image (1024 chiều) được nối lại với nhau (Concatenate).
- Đi qua các lớp Dense Network với hàm kích hoạt GELU và Dropout để học ra sự tương tác chéo giữa Text và Image.
- Output cuối cùng xuất ra N factor scores (Ví dụ: Food, Price, Atmosphere).

---

## 2. Hàm Mất mát (Joint Loss Function)

Bài toán của chúng ta không phải là phân loại, mà là **Dự đoán điểm số liên tục (Regression)**.

Mô hình dự đoán cùng lúc 2 loại điểm:
1. **Điểm tổng quan (Overall Score)**
2. **Điểm thành phần (Factor Scores: Food, Price, Atmosphere)**

Hàm Loss được thiết kế là sự kết hợp của hai thành phần:
```math
Loss = \alpha \times MSE_{Overall} + (1 - \alpha) \times \frac{1}{N} \sum_{i=1}^{N} MSE_{Factor_i}
```
*Trong đó \alpha là hệ số cân bằng (Mặc định 0.5).*

**Lý do sử dụng Joint Loss:** 
Việc ép mô hình học cách dự đoán chính xác cả điểm thành phần giúp mô hình không bị "lười biếng" chỉ nhìn vào một khía cạnh. Mối tương quan logic (ví dụ: Món ăn ngon + Giá rẻ -> Tổng quan cao) sẽ được mạng nơ-ron tự động khám phá và liên kết thông qua cấu trúc Loss này.

---

## 3. Hệ thống Đo lường (Evaluation Metrics)

Kết quả mô hình được đo lường khắt khe bằng 3 hệ metic chuẩn của bài toán Hồi quy, giúp cung cấp góc nhìn đa chiều về sai số.

### MAE (Mean Absolute Error)
- **Ý nghĩa:** Khoảng cách lệch trung bình tuyệt đối giữa điểm dự đoán và điểm thực tế.
- **Giá trị thực tiễn:** Nếu $MAE = 1.0$, nghĩa là trung bình mô hình chấm sai lệch khoảng 1 điểm (ví dụ thật là 7 điểm, dự đoán là 8 điểm). Đây là metric thân thiện và dễ hiểu nhất đối với con người.

### RMSE (Root Mean Square Error) & MSE (Mean Squared Error)
- **Ý nghĩa:** Bình phương sai số. 
- **Giá trị thực tiễn:** MSE và RMSE "trừng phạt" các dự đoán sai lệch quá lớn. Ví dụ, nếu đa số mô hình dự đoán lệch 0.5 điểm, nhưng có một vài mẫu lệch tới 4 điểm, thì MAE có thể vẫn thấp nhưng RMSE sẽ vọt lên rất cao. Do đó, việc công bố cả MAE và RMSE giúp chứng minh độ ổn định (Robustness) của mô hình trước các đánh giá Outliers (Ngoại lai).

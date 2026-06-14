# Tổng hợp Nghiên cứu & Kế hoạch Thử nghiệm (Experimental Plan)

Tài liệu này tóm tắt các kết quả từ đợt Deep Research (trong `finding.md`) và quy hoạch lộ trình triển khai thành 3 nhóm thử nghiệm độc lập, nhằm nâng cấp toàn diện mô hình Baseline hiện tại (XLM-R + ConvNeXt + MSE).

---

## 1. Tóm tắt Đề xuất Nâng cấp Cốt lõi
Phân tích kỹ thuật đã chỉ ra các điểm nghẽn nghiêm trọng của mô hình hiện tại và đề xuất các giải pháp triệt để:
- **Về Backbone:** Các mô hình đa ngôn ngữ (XLM-R) và CNN cơ bản (ConvNeXt) không đủ sức nắm bắt ngữ pháp phức tạp của tiếng Việt và bối cảnh toàn cục của không gian nhà hàng. Cần chuyển sang mô hình đơn ngữ (ViDeBERTa/PhoBERT) và Transformer phân cấp (Swin).
- **Về Loss Function:** Hàm MSE trung bình rất dễ bị bóp méo bởi các bình luận ngoại lai (review bombing) và tạo ra hiện tượng "cạnh tranh tiêu cực" giữa 5 nhãn. Cần áp dụng Robust Loss và Multi-task Learning Loss.
- **Về Fusion:** Nối vector (Concatenation) là quá thô sơ. Cần một cơ chế hòa trộn phi tuyến tính cho phép Văn bản và Hình ảnh "giao tiếp" và cộng hưởng ngữ nghĩa với nhau.

---

## 2. Các Nhóm Thử nghiệm (Experimental Groups)
Để tối ưu hóa khoa học và có kiểm soát, đội ngũ kỹ sư sẽ lần lượt triển khai và đo lường 3 nhóm thử nghiệm sau:

### Nhóm 1: Thử nghiệm Backbone (Cốt lõi trích xuất)
Thay vì dùng XLM-R và ConvNeXt, chúng ta sẽ thử nghiệm các cặp Backbone mới:
- **Thử nghiệm 1.1 (Ưu tiên số 1 - The Sweet Spot):** `ViDeBERTa-base` (Text) + `Swin-B` (Image). Đảm bảo sự phân tách từ vựng tiếng Việt hoàn hảo và khả năng nhìn nhận cả chi tiết món ăn lẫn không gian, trong khi vẫn giữ tốc độ xử lý nhanh.
- **Thử nghiệm 1.2 (Tối đa Chính xác):** `PhoBERT-large` (Text) + `EVA-CLIP` hoặc `ViT-L` (Image). Tận dụng khả năng liên kết hình ảnh-ngôn ngữ đã được align sẵn của EVA-CLIP. Phù hợp nếu tài nguyên GPU dồi dào.
- **Thử nghiệm 1.3 (Tối đa Tốc độ):** `ViDeBERTa-xsmall` (Text) + `MobileViT` (Image). Tối giản tham số cho môi trường triển khai thực tế có tài nguyên tính toán hẹp.

### Nhóm 2: Thử nghiệm Hàm Mất Mát (Loss Functions)
Giữ nguyên kiến trúc mạng, chỉ thay đổi hàm tính Loss ở khâu cuối để cải thiện trọng số tự động:
- **Thử nghiệm 2.1 (Kháng ngoại lai):** Thay thế toàn bộ MSE bằng **Huber Loss** hoặc **Log-Cosh Loss** cho từng nhãn mục tiêu để tránh gradient bùng nổ khi gặp dữ liệu nhiễu.
- **Thử nghiệm 2.2 (Cân bằng Đa tác vụ):** Áp dụng **Homoscedastic Task Uncertainty Loss**. Thay vì fix cứng trọng số $1:1:1:1:1$, mô hình sẽ tự định nghĩa 5 tham số phương sai ($s_1 \dots s_5$) để tự động giảm trọng số của các nhãn khó đoán (như overall_satisfaction) và tăng trọng lượng cho các nhãn dễ đoán (như food).

### Nhóm 3: Thử nghiệm Kiến trúc Fusion
Giữ nguyên Backbones, thay thế lớp Concatenation hiện tại:
- **Thử nghiệm 3.1 (Đề xuất chính):** Triển khai **Gated Cross-Modal Fusion** (như mã giả PyTorch đã được AI viết sẵn). Kiến trúc này dùng `nn.Bilinear` để bắt tương tác chéo, sau đó dùng cổng Gate (GMU) để quyết định sẽ tin tưởng vào Hình ảnh hay Văn bản hơn.
- **Thử nghiệm 3.2:** Áp dụng **Feature-wise Linear Modulation (FiLM)**. Dùng vector Văn bản sinh ra các hệ số $\gamma, \beta$ để Scale và Shift đặc trưng của Hình ảnh (Bật/tắt các vùng nhìn dựa theo review).
- **Thử nghiệm 3.3:** Áp dụng **Cross-Attention** cơ bản (yêu cầu Image Backbone phải xuất ra dạng Patches thay vì Pooled Vector).

---

## 3. Ngưỡng Đánh giá (Benchmark Expectations)
Tất cả các thử nghiệm trên sẽ được đối chiếu trên thang điểm dự đoán 1-10. Dưới đây là các mốc Benchmark (trung bình trên 5 nhãn) để so sánh hiệu năng:

- **Mức Baseline (Hiện tại):** MAE $\approx 1.8 - 2.1$ điểm | $R^2 \approx 0.40 - 0.55$
- **Mức Tốt (Kỳ vọng đạt được sau Nhóm 2 & 3):** MAE $\approx 1.4 - 1.7$ điểm | $R^2 \approx 0.55 - 0.65$
- **Mức SOTA (Kỳ vọng đạt được nếu kết hợp đủ 3 Nhóm):** MAE $< 1.3$ điểm | $R^2 > 0.70$

*(Lưu ý thực tiễn: Khi đo lường thực tế, MAE của nhãn `price` và `food` có thể xuống thấp rất nhanh vì dễ quan sát, trong khi `service` và `overall_satisfaction` sẽ luôn có MAE cao hơn do phụ thuộc tâm lý chủ quan).*

# Báo cáo Lựa chọn Backbone - Phase 2 (Image Backbone Ablation)

Tài liệu này ghi nhận kết quả đánh giá và quyết định lựa chọn mô hình lõi (Backbone) cho nhánh Ảnh (Image Branch) sau khi hoàn thành Phase 2. Mọi đánh giá đều dựa trên kết quả chạy thực tế với tập Validation.

## 1. Kết quả thử nghiệm (Metrics)

Trong Phase 2, chúng ta đã giữ cố định nhánh Text là `XLM-R` và hàm Loss là `MSE`, thay đổi nhánh Image với 3 cấu hình SOTA khác nhau:

| Tiêu chí / Model | Swin-B (`EXP_020B`) 🏆 | SigLIP (`EXP_020E`) 🥈 | EfficientNet-B3 (`EXP_020D`) |
|------------------|------------------------|-----------------------|------------------------------|
| **Mean MAE** (Trung bình 5 yếu tố) | **1.2169** | 1.2296 | 1.2799 |
| **Overall MAE** (Điểm hài lòng chung) | **1.0667** | 1.0703 | 1.1295 |
| **Aspect MAE** (Chỉ tính 4 yếu tố phụ) | **1.2544** | 1.2694 | 1.3175 |
| **R² Overall** (Độ phù hợp) | **0.4874** | 0.4715 | 0.4235 |
| **Loss** (MSE) | **2.7092** | 2.7928 | 2.9468 |

*Ghi chú: MAE, RMSE, Loss càng thấp càng tốt. R² càng cao (gần 1) càng tốt.*

## 2. Phân tích & Lựa chọn

Dựa trên các con số cụ thể trên, ta có thể rút ra các kết luận:
1. **Swin-B (Swin Transformer) là mô hình chiến thắng tuyệt đối:** Nó dẫn đầu ở TẤT CẢ các tiêu chí đánh giá. Mức sai số trung bình (Mean MAE) đạt **1.2169**, và đặc biệt sai số cho điểm tổng thể (Overall MAE) chỉ là **1.0667**. Chỉ số $R^2$ đạt **0.4874**, chứng tỏ mô hình có khả năng giải thích phương sai tốt nhất. Sự vượt trội của Swin-B có thể do kiến trúc Hierarchical Vision Transformer (cửa sổ trượt) giúp nó nắm bắt tốt cả đặc trưng cục bộ (món ăn) lẫn toàn cục (không gian quán).
2. **SigLIP đứng ở vị trí á quân rất sát sao:** Dù rất mạnh trong việc biểu diễn đa phương thức (nhờ Sigmoid Loss), SigLIP (Mean MAE: 1.2296) vẫn thua Swin-B một chút trong bài toán chấm điểm review nhà hàng này.
3. **EfficientNet-B3 đuối sức nhất:** Dù là mạng CNN cực kỳ tối ưu, nhưng cấu trúc CNN truyền thống tỏ ra yếu thế hơn so với họ nhà Transformer (Swin/SigLIP) trong việc trích xuất đặc trưng hình ảnh review ẩm thực phức tạp (Mean MAE tận 1.2799).

## 3. Quyết định cho Phase tiếp theo

👉 **CHỐT:** Chọn **`Swin-B`** làm Image Backbone chính thức cho toàn bộ các thử nghiệm từ Phase 3 trở về sau. 

Các notebook ở Phase 3 (`EXP_030B` và `EXP_030D`) sẽ được nạp trọng số tốt nhất của Swin-B (`best_model_train_image.pth` từ `EXP_020B`) để tiếp tục gọt giũa nhánh Text (PhoBERT / ViSoBERT).

---

# Báo cáo Lựa chọn Text Backbone - Phase 3 (Text Backbone Ablation)

Tài liệu này ghi nhận kết quả đánh giá và quyết định lựa chọn mô hình lõi cho nhánh Chữ (Text Branch) sau khi cố định nhánh Ảnh là `Swin-B` (từ Phase 2).

## 1. Kết quả thử nghiệm (Metrics)

| Tiêu chí / Model | PhoBERT (`EXP_030B`) 🏆 | XLM-RoBERTa (`EXP_020B`) 🥈 | ViSoBERT (`EXP_030D`) |
|------------------|-------------------------|-----------------------------|-----------------------|
| **Mean MAE** | **1.1145** | 1.2169 | 1.2328 |
| **Overall MAE** | **0.9300** | 1.0667 | 1.0923 |
| **Aspect MAE** | **1.1607** | 1.2545 | 1.2679 |
| **R² Overall** | **0.6220** | 0.4874 | 0.4589 |
| **Loss** (MSE) | **2.2034** | 2.7093 | 2.8100 |

## 2. Phân tích & Lựa chọn

1. **PhoBERT hủy diệt mọi đối thủ:** Khi ghép với Swin-B, PhoBERT thể hiện sức mạnh vượt trội với Mean MAE giảm sâu xuống **1.1145**. Đặc biệt, sai số cho điểm tổng thể (Overall MAE) chỉ còn **0.9300** (lần đầu tiên phá vỡ mốc 1.0). R² tăng vọt lên **0.6220** (so với 0.4874 của XLM-R), chứng tỏ model bằng tiếng Việt thuần túy (PhoBERT) kết hợp với Swin-B tạo ra sự cộng hưởng cực kỳ mạnh mẽ.
2. **XLM-RoBERTa về nhì:** Dù là model đa ngôn ngữ, XLM-R vẫn giữ được độ ổn định rất tốt (Mean MAE 1.2169), vượt qua cả ViSoBERT.
3. **ViSoBERT đuối sức:** Dù cũng là model chuyên tiếng Việt, ViSoBERT (Mean MAE 1.2328) có dấu hiệu overfit nhanh trên tập train nên khi đánh giá trên tập Val lại kém hơn PhoBERT khá nhiều.

## 3. Quyết định cho Phase tiếp theo

👉 **CHỐT:** Chọn cặp bài trùng **`Swin-B` + `PhoBERT`** làm kiến trúc cốt lõi (Best Image + Best Text) cho toàn bộ các thử nghiệm từ Phase 4 trở về sau.

Các thử nghiệm ở Phase 4 (Advanced Fusion như GMU, FiLM, Cross-Attention) sẽ kế thừa trực tiếp cặp đôi vô địch này.

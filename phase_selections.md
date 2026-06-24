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

---

# Báo cáo Lựa chọn Fusion Architecture - Phase 4

Tài liệu này ghi nhận kết quả đánh giá các cơ chế kết hợp đặc trưng (Fusion Mechanisms) phức tạp hơn so với Concatenation đơn thuần. Nhánh Image được cố định là `Swin-B`, nhánh Text được cố định là `PhoBERT`.

## 1. Kết quả thử nghiệm (Metrics)

| Tiêu chí / Fusion | Baseline (Concat) | GMU (`EXP_040B`) | Gated Cross (`EXP_040C`) | FiLM (`EXP_041A`) | Cross-Attention (`EXP_041B`) 🏆 |
|-------------------|-------------------|------------------|--------------------------|-------------------|---------------------------------|
| **Mean MAE** | 1.1145 | 1.1160 | 1.1082 | 1.1195 | **1.1079** |
| **Overall MAE** | 0.9300 | 0.9289 | 0.9198 | 0.9278 | **0.9143** |
| **Aspect MAE** | 1.1607 | 1.1628 | 1.1553 | 1.1675 | **1.1563** |
| **R² Overall** | 0.6220 | 0.6246 | 0.6309 | 0.6215 | **0.6335** |
| **Loss** (MSE) | 2.2034 | 2.2047 | 2.1740 | 2.2243 | 2.1750 |

## 2. Phân tích & Lựa chọn

1. **Cross-Attention lên ngôi vô địch:** Bằng cách cho phép Text và Image liên tục rà soát đặc trưng của nhau thông qua thuật toán Attention, mô hình đã tìm được những liên kết ngầm sâu sắc nhất. Chỉ số Overall MAE giảm kỷ lục xuống còn **0.9143**, và R² đạt đỉnh **0.6335**. Đây là kiến trúc tối ưu nhất trong toàn bộ 4 Phase.
2. **Gated Cross-Modal bám sát:** Kiến trúc "Lọc nhiễu chéo" này cũng thể hiện sức mạnh rất tốt (Mean MAE 1.1082), bám đuổi sát nút Cross-Attention và vượt xa Baseline.
3. **GMU và FiLM gây thất vọng:** Có vẻ như việc chia tỷ lệ (GMU) hoặc dùng chữ để xoay/tịnh tiến ảnh (FiLM) không hoạt động tốt trên bộ dữ liệu review ăn uống này, khiến kết quả thậm chí còn thụt lùi hoặc chỉ ngang ngửa so với việc ghép nối (Concat) thô sơ ban đầu.

## 3. Quyết định cho Phase tiếp theo

👉 **CHỐT:** Chọn **`Cross-Attention`** làm kiến trúc Fusion chính thức. 

Đội hình hoàn hảo nhất hiện tại: **`Swin-B` + `PhoBERT` + `Cross-Attention`**.
Đội hình này sẽ được đem đi thử lửa với các hàm Loss chống nhiễu (Huber, Log-Cosh, Uncertainty Weighted) tại Phase 5.

---

# Báo cáo Lựa chọn Loss Function - Phase 5

Tài liệu này ghi nhận kết quả đánh giá các hàm Loss Function phức tạp nhằm xử lý nhiễu (outliers) tốt hơn MSE truyền thống. Nhánh Image cố định là `Swin-B`, nhánh Text cố định là `PhoBERT`, Fusion cố định là `Cross-Attention`.

## 1. Kết quả thử nghiệm (Metrics)

| Tiêu chí | Baseline (MSE) 🏆 Mean | Huber (`EXP_050B`) | Log-Cosh (`EXP_050C`) 🏆 Overall | Uncertainty Weighted (`EXP_051D`) 🏆 R² |
|---|---|---|---|---|
| **Mean MAE** | **1.1078** | 1.1085 | 1.1079 | 1.1080 |
| **Overall MAE** | 0.9142 | 0.9131 | **0.9130** | 0.9143 |
| **R² Overall** | 0.6335 | 0.6307 | 0.6312 | **0.6336** |

## 2. Phân tích & Lựa chọn

1. Mặc dù **MSE** vẫn giữ được Mean MAE thấp nhất (1.1078), tuy nhiên **Log-Cosh** đã cho thấy khả năng vượt trội ở **Overall MAE** (0.9130). 
2. Trong bối cảnh review nhà hàng, điểm số đánh giá tổng thể (Overall) mang tính chất quyết định nhất đối với trải nghiệm người dùng, do đó việc tối ưu tốt nhất cho Overall được ưu tiên.
3. Chênh lệch Mean MAE giữa Log-Cosh và MSE là vô cùng nhỏ (0.0001), hoàn toàn có thể chấp nhận đánh đổi.

## 3. Quyết định cho Phase tiếp theo

👉 **CHỐT:** Chọn **`Log-Cosh`** làm hàm Loss cuối cùng.

Cấu hình "Vô địch" (Best Sequential Full Configuration) sẽ là:
- **Image:** Swin-B
- **Text:** PhoBERT
- **Fusion:** Cross-Attention
- **Loss:** Log-Cosh

Cấu hình này sẽ được chạy tại notebook `EXP_060A_bestsequential_full_configuration.ipynb` ở Phase 6.

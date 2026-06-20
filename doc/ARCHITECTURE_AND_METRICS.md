# Quyết định Thiết kế và Đánh giá (Architecture & Metrics)

Tài liệu này giải thích chi tiết các quyết định đằng sau việc lựa chọn mô hình, hàm Loss và hệ thống đánh giá trong mã nguồn.

## 1. Kiến trúc Mô hình (Model Architecture)

Mô hình hiện tại sử dụng phương pháp **Intermediate Fusion** (Kết hợp trung gian): Text và Image được xử lý song song bởi hai Backbone độc lập để trích xuất đặc trưng, rồi gộp lại qua Fusion MLP.

### Text Backbone: XLM-RoBERTa
- **Lý do lựa chọn:** Dữ liệu có thể chứa ngôn ngữ đa dạng (đánh giá nhà hàng bằng tiếng Việt, Anh, v.v.). XLM-RoBERTa là mô hình ngôn ngữ đa ngữ (Multilingual) hàng đầu hiện nay, vượt trội hơn so với mBERT truyền thống, đặc biệt trong việc hiểu ngữ cảnh cảm xúc của câu.
- **Vai trò:** Trích xuất đặc trưng ngữ nghĩa từ câu đánh giá. Sử dụng `pooler_output` nếu model cung cấp, ngược lại lấy `last_hidden_state[:, 0, :]` (first-token pooling).
- **Chiều đầu ra encoder:** 768 (với `xlm-roberta-base`)

### Image Backbone: ConvNeXt
- **Lý do lựa chọn:** ConvNeXt là kiến trúc lai giữa sức mạnh của Transformer và tốc độ của CNN. Nó đem lại hiệu suất vượt trội trên ảnh tự nhiên so với ResNet cũ, đồng thời tiết kiệm tài nguyên tính toán hơn so với ViT thuần túy.
- **Vai trò:** Trích xuất đặc trưng thị giác từ ảnh món ăn/nhà hàng (ví dụ: màu sắc món ăn, độ sáng không gian). Hỗ trợ **nhiều ảnh mỗi mẫu** (max 4 ảnh) với Average Pooling có mask.
- **Chiều đầu ra encoder:** 1024 (với `convnext_base_in22k`)

### Cơ chế Kết hợp (Fusion Mechanism)

Pipeline huấn luyện gồm 3 giai đoạn:

1. **Giai đoạn 1 — Train Text** (`--mode train_text`):
   ```
   Text Encoder → pooled feature (768) → FC(768→256) → ReLU → Dropout(0.2) → Linear(256→5 scores)
   ```
2. **Giai đoạn 2 — Train Image** (`--mode train_image`):
   ```
   Image Encoder → pooled feature (1024) → FC(1024→256) → ReLU → Dropout(0.2) → Linear(256→5 scores)
   ```
3. **Giai đoạn 3 — Train Fusion** (`--mode train_fusion`):
   ```
   Freeze Text & Image encoders
   → trích raw encoder features (768 + 1024 = 1792)
   → concat → FC(1792→512) → ReLU → Dropout(0.2) → FC(512→256) → ReLU → Linear(256→5 scores)
   ```

**Lưu ý quan trọng:** Fusion dùng **raw encoder features** (768-d text + 1024-d image), **không** dùng 256-d projection từ các nhánh unimodal. Đóng băng encoder nhưng có thể "tan băng" (`unfreeze_text_layers`, `unfreeze_image_layers`) để fine-tune chọn lọc.

### Multi-Image Handling

Mỗi mẫu có thể có nhiều ảnh (tối đa 4). `ImageModel` xử lý tensor 5D `[B, N, C, H, W]`:
- Encode từng ảnh riêng biệt qua encoder
- Average Pooling có mask (chỉ tính trung bình trên ảnh thực tế, bỏ qua ảnh đệm đen)

---

## 2. Hàm Mất mát (Loss Function)

### Loss hiện tại trong code

Mã nguồn hiện tại sử dụng **plain vector MSE** trên cả 5 đầu ra:

```python
criterion = nn.MSELoss()
loss = criterion(pred_factors, true_factors)  # 5 scores: food, price, atmos, service, overall
```

Đây là MSE trung bình đều trên cả 5 targets (food_score, price_score, atmosphere_score, service_score, overall_satisfaction), mỗi target được weighted như nhau.

### Joint Loss (dự kiến, chưa implement)

Thiết kế dự kiến trong tài liệu nghiên cứu là Joint Loss kết hợp điểm tổng quan và điểm thành phần:

```text
Loss = α × MSE_overall + (1 - α) × mean(MSE_factor_i)
```

Trong đó `α` là hệ số cân bằng (mặc định 0.5), `MSE_overall` là loss trên `overall_satisfaction`, và `mean(MSE_factor_i)` là trung bình MSE trên 4 điểm khía cạnh (food, price, atmos, service).

**Trạng thái:** Chưa được triển khai trong code. Hiện tại cả 5 targets đều được tối ưu hoá bằng MSE đều nhau. Đây là hướng mở rộng có thể implement trong giai đoạn tiếp theo.

### Tại sao Joint Loss có giá trị

Việc ép mô hình học dự đoán chính xác cả điểm thành phần giúp mô hình không bị "lười biếng" chỉ nhìn vào một khía cạnh. Mối tương quan logic (ví dụ: Món ăn ngon + Giá rẻ → Tổng quan cao) sẽ được mạng nơ-ron tự động khám phá thông qua cấu trúc Loss này.

---

## 3. Hệ thống Đo lường (Evaluation Metrics)

Kết quả mô hình được đo lường bằng 3 hệ metric chuẩn của bài toán Hồi quy, giúp cung cấp góc nhìn đa chiều về sai số.

### MAE (Mean Absolute Error)
- **Ý nghĩa:** Khoảng cách lệch trung bình tuyệt đối giữa điểm dự đoán và điểm thực tế.
- **Giá trị thực tiễn:** Nếu MAE = 1.0, nghĩa là trung bình mô hình chấm sai lệch khoảng 1 điểm (ví dụ thật là 7, dự đoán 8). Đây là metric thân thiện và dễ hiểu nhất.

### RMSE (Root Mean Square Error) & MSE (Mean Square Error)
- **Ý nghĩa:** Bình phương sai số. RMSE/MSE "trừng phạt" các dự đoán sai lệch quá lớn. Nếu đa số dự đoán lệch 0.5 điểm nhưng có vài mẫu lệch 4 điểm, MAE vẫn thấp nhưng RMSE vọt lên.
- **Giá trị thực tiễn:** Công bố cả MAE và RMSE giúp chứng minh độ ổn định (Robustness) của mô hình trước các đánh giá Outliers.

### 5 đầu ra đánh giá

Mô hình dự đoán đồng thời 5 điểm trên thang 0-10:

| Đầu ra | Ý nghĩa |
|--------|----------|
| `food_score` | Chất lượng đồ ăn |
| `price_score` | Mức độ phù hợp về giá |
| `atmosphere_score` | Không gian/bầu không khí |
| `service_score` | Chất lượng phục vụ |
| `overall_satisfaction` | Mức độ hài lòng tổng thể (sinh từ rule engine) |

---

## 4. Chi tiết Training Pipeline

### Optimizer
- **AdamW** với `weight_decay=1e-2`
- Chỉ tối ưu các parameter có `requires_grad=True`

### Scheduler
- **Cosine with Warmup** (`get_cosine_schedule_with_warmup`)
- `warmup_ratio=0.1` (mặc định)
- Tính `total_steps` dựa trên số batch và `grad_accum_steps`

### Gradient Accumulation
- Mặc định `grad_accum_steps=1`, có thể tăng để xử lý batch lớn trên GPU ít VRAM
- Gradient clip `max_norm=1.0` trước mỗi optimizer step

### Early Stopping
- `patience=3` (mặc định): dừng nếu val loss không cải thiện sau 3 epoch
- Lưu best checkpoint khi val loss tốt hơn

### Unfreeze Layers (Fusion stage)
- `--unfreeze_text_layers`: Số layer cuối của XLM-RoBERTa được "tan băng" (mặc định: 0)
- `--unfreeze_image_layers`: Số block cuối của ConvNeXt được "tan băng" (mặc định: 0)
- Cho phép fine-tune chọn lọc khi fusion, thay vì đóng băng toàn bộ encoder

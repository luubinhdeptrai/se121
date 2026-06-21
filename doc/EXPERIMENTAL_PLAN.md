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
Mỗi combo được thiết kế theo tư duy **multimodal-first**: text encoder và image encoder phải **tương thích với nhau** khi fusion, không chỉ tốt riêng lẻ. Tiêu chí lựa chọn gồm: (1) phù hợp domain (tiếng Việt social media), (2) feature space có thể kết hợp tốt, và (3) tính khả thi triển khai.

---

- **Thử nghiệm 1.1 — "RoBERTa + CLIP" (Đúng như paper):** `roberta-base` (Text) + `CLIP ViT-B/32` (Image)
  - **Lý do:** Đây là combo được test trực tiếp trong paper MDPI Applied Sciences 2026 trên bài toán multimodal sentiment. RoBERTa text encoder kết hợp với CLIP ViT-B/32 qua cross-attention — đạt **79.62% test accuracy, F1=79.42** trên MVSA-Single. CLIP ViT-B/32 được chọn vì visual features của nó **đã được align với ngôn ngữ** từ pre-training, giảm modality gap khi fusion. **`[3]`**
  - **Lưu ý kỹ thuật:** CLIP tokenizer giới hạn 77 token — chỉ dùng **visual encoder của CLIP** để lấy ảnh features, tokenizer của text model dùng riêng.
  - **HuggingFace ID:** `FacebookAI/roberta-base` | `openai/clip-vit-base-patch32` (visual encoder; timm: `vit_base_patch32_clip_224.openai`)

---

- **Thử nghiệm 1.2 — "Vietnamese Equivalent" (Thích nghi tiếng Việt):** `ViSoBERT` (Text) + `EfficientNet-B3` (Image)
  - **Lý do chọn Text:** Combo gốc từ IJACSA 2024 dùng RoBERTa — tại đây thay bằng **ViSoBERT** (`uitnlp/visobert`, EMNLP 2023) là VN equivalent phù hợp nhất vì cùng kiến trúc XLM-R và được pre-train trực tiếp trên văn bản mạng xã hội tiếng Việt (Facebook, YouTube, TikTok) — đúng domain của dữ liệu Foody/ShopeeFood. **`[1]`**
  - **Lý do chọn Image:** IJACSA 2024 thử nghiệm trực tiếp nhiều cặp fusion, **EfficientNet-B3 + RoBERTa đạt accuracy cao nhất (75%), vượt ResNet-50 và MobileNetV2**. Feature vector 1536-dim của EfficientNet-B3 dễ concat với hidden_size 768 của ViSoBERT. **`[5]`**
  - **HuggingFace/Timm ID:** `uitnlp/visobert` | `efficientnet_b3` (timm)

---

- **Thử nghiệm 1.3 — "DeBERTa-v3 + SigLIP2" (Đúng như paper):** `microsoft/deberta-v3-base` (Text) + `SigLIP2-base-patch16` (Image)
  - **Lý do:** Đây là combo được test trong SINC-V1 (HuggingFace, 2025) — multimodal product classifier kết hợp DeBERTa-v3 (text) với SigLIP2 (image encoder) qua Concatenation + MLP, đạt **92.46% test accuracy** trên 59,789 mẫu e-commerce. Đây là trường hợp duy nhất SigLIP2 hoạt động tốt trong late-fusion vì được paired với DeBERTa-v3 (cùng mức disentangled attention). **`[6]`**
  - **Lưu ý:** SigLIP2 trong standalone fusion với BERT/RoBERTa thông thường bị thua CLIP (benchmark MMHS150K: SigLIP F1=0.507 vs CLIP F1=0.566). Combo này chỉ hợp lý khi dùng đúng với DeBERTa-v3.
  - **HuggingFace/Timm ID:** `microsoft/deberta-v3-base` | `vit_base_patch16_siglip2_256` (timm)

### Nhóm 2: Thử nghiệm Hàm Mất Mát (Loss Functions)
Giữ nguyên kiến trúc mạng Baseline (ViSoBERT + ConvNeXt), chỉ thay đổi hàm tính Loss ở khâu cuối để cải thiện khả năng học:
- **Thử nghiệm 2.0 (Baseline):** Sử dụng **Joint MSE Loss** (trung bình cộng MSE của 5 nhãn). Đây là loss mặc định đã dùng ở Nhóm 1, dễ cài đặt nhưng bị hạn chế khi gặp review ngoại lai và không tự động cân bằng các nhãn.
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

---

## 4. Tài liệu Tham khảo — Nhóm 1

> Các nguồn dưới đây là cơ sở khoa học cho việc lựa chọn backbone trong Nhóm 1. Ký hiệu `[n]` tương ứng với số thứ tự trong phần thử nghiệm.

**[1]** Nguyen, N., Phan, T., Nguyen, D.-V., & Nguyen, K. (2023). **ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing**. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*, pp. 5191–5207. Association for Computational Linguistics, Singapore.
- 🔗 https://aclanthology.org/2023.emnlp-main.315
- 📦 HuggingFace: https://huggingface.co/uitnlp/visobert
- *(Dùng cho: 1.2 — VN equivalent thay RoBERTa)*

**[2]** Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**. *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*, pp. 8748–8763. PMLR.
- 🔗 https://proceedings.mlr.press/v139/radford21a.html
- 📦 HuggingFace: https://huggingface.co/openai/clip-vit-base-patch32
- *(Dùng cho: 1.1 — background cho CLIP visual encoder)*

**[3]** Gîrlea, M., et al. (2026). **Text-Anchored Residual Cross-Modal Fusion for Multimodal Sentiment Analysis: A Unified and Protocol-Aware Evaluation on MVSA-Single**. *Applied Sciences*, 16(9), 4514. MDPI. *(Combo RoBERTa + CLIP ViT-B/32 cross-attention đạt 79.62% test accuracy, F1=79.42 trên MVSA-Single — tốt nhất trong tất cả baselines được thử nghiệm.)*
- 🔗 https://doi.org/10.3390/app16094514
- *(Dùng cho: 1.1 — nguồn gốc trực tiếp của combo)*

**[4]** *(Đã xóa — ViDeBERTa không còn được dùng trong Nhóm 1)*

**[5]** Habib, M. B., Hafiz, M. F. B., Khan, N. A., & Hossain, S. (2024). **Multimodal Sentiment Analysis using Deep Learning Fusion Techniques and Transformers**. *International Journal of Advanced Computer Science and Applications (IJACSA)*, 15(6). The Science and Information Organization. *(Benchmark trực tiếp EfficientNet-B3 + RoBERTa đạt accuracy 75%, F1 74.9% — tốt nhất trong các combo text+image được thử nghiệm, vượt ResNet-50 và MobileNetV2.)*
- 🔗 http://dx.doi.org/10.14569/IJACSA.2024.0150686
- *(Dùng cho: 1.2 — nguồn gốc trực tiếp của combo image encoder)*

**[6]** Bangotra, M. (2025). **SINC-V1: Multimodal Product Classifier — DeBERTa-v3-small + SigLIP2-base-patch16**. *Hugging Face Model Hub*. Concatenation + MLP fusion, đạt **92.46% test accuracy / F1=0.73** trên 59,789 mẫu e-commerce. *(Combo DeBERTa-v3 + SigLIP2 được thực nghiệm và công bố kết quả tường minh.)*
- 📦 https://huggingface.co/manavbangotra/SINC-V1-SIGLIP2-KEE-SPEED-small
- *(Dùng cho: 1.3 — nguồn gốc trực tiếp của combo)*

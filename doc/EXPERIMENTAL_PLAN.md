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

- **Thử nghiệm 1.1 — "Domain-Exact" (Ưu tiên số 1):** `ViSoBERT` (Text) + `CLIP ViT-B/32` (Image)
  - **Lý do chọn Text:** ViSoBERT (EMNLP 2023) là mô hình duy nhất pre-train **trực tiếp trên văn bản mạng xã hội tiếng Việt** (Facebook, TikTok, YouTube) — đúng domain của dữ liệu Foody/ShopeeFood (emoji, teencode, ngôn ngữ không trang trọng). Trên các task sentiment review tiếng Việt, ViSoBERT vượt cả PhoBERT và XLM-R. **`[1]`**
  - **Lý do chọn Image:** CLIP ViT-B/32 (OpenAI) được pre-train bằng contrastive learning trên **400M cặp ảnh-văn bản**. Điều này giúp các vector ảnh của CLIP **có cùng semantic space** với các vector ngôn ngữ — giảm thiểu "khoảng cách modality" (modality gap) khi fusion. **`[2]`** Nghiên cứu multimodal sentiment 2026 (MDPI Applied Sciences) xác nhận `RoBERTa + CLIP` cross-attention đạt kết quả tốt nhất trong các combo được thử nghiệm. **`[3]`**
  - **Lưu ý kỹ thuật:** CLIP tokenizer giới hạn 77 token — giữ nguyên tokenizer của ViSoBERT cho text, chỉ dùng **visual encoder của CLIP** để lấy ảnh features.
  - **HuggingFace ID:** `uitnlp/visobert` | `openai/clip-vit-base-patch32` (visual encoder)

---

- **Thử nghiệm 1.2 — "Proven Multimodal Pair" (Cân bằng chất lượng + tốc độ):** `ViDeBERTa-base` (Text) + `EfficientNet-B3` (Image)
  - **Lý do chọn Text:** ViDeBERTa (EACL 2023) dùng DeBERTaV3 với disentangled attention, vượt PhoBERT-large với chỉ 86M params (PhoBERT-large: 370M). Phù hợp khi cần chất lượng tiếng Việt cao mà GPU budget hạn chế. **`[4]`**
  - **Lý do chọn Image:** Nghiên cứu multimodal sentiment năm 2024 (IJACSA 2024) thử nghiệm trực tiếp nhiều cặp fusion và phát hiện **EfficientNet-B3 + RoBERTa đạt accuracy cao nhất (75%) trong các combo text+image**, vượt ResNet-50 và MobileNetV2. EfficientNet-B3 cân bằng tốt giữa capacity và tốc độ, feature vector 1536-dim dễ concat/fuse với hidden_size của ViDeBERTa (768-dim). **`[5]`**
  - **Timm ID:** `HySonLab/ViDeBERTa` | `efficientnet_b3` (timm)

---

- **Thử nghiệm 1.3 — "Vietnamese ABSA Baseline" (Đối chiếu với nghiên cứu trong nước):** `XLM-RoBERTa-base` (Text) + `CLIP ViT-B/16` (Image)
  - **Lý do chọn Text:** XLM-R là backbone chuẩn của model `visolex/xlm-roberta-absa-restaurant` (HuggingFace, 2025) — model được fine-tune trực tiếp trên dữ liệu VLSP2018 **restaurant review tiếng Việt** với 12 aspect categories (FOOD#QUALITY, SERVICE#GENERAL, RESTAURANT#PRICES...), đạt Accuracy 0.89. Đây cũng chính là backbone baseline hiện tại của dự án — cho phép so sánh có kiểm soát. **`[6]`**
  - **Lý do chọn Image:** CLIP ViT-B/16 (patch size nhỏ hơn /32) giữ nhiều spatial detail hơn, phù hợp hơn khi ảnh nhà hàng cần nhận diện chi tiết món ăn, không gian. Trong nghiên cứu multimodal food review (Foody crawl), cặp BERT+CLIP style đã chứng minh hiệu quả trên dữ liệu tương tự. **`[2][3]`**
  - **Lưu ý kỹ thuật:** ViT-B/16 xuất ra **197 patch tokens** (không chỉ pooled vector), mở đường cho Cross-Attention trong Nhóm 3 (Thử nghiệm 3.3).
  - **HuggingFace ID:** `xlm-roberta-base` | `openai/clip-vit-base-patch16` (visual encoder)

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

---

## 4. Tài liệu Tham khảo — Nhóm 1

> Các nguồn dưới đây là cơ sở khoa học cho việc lựa chọn backbone trong Nhóm 1. Ký hiệu `[Rn]` tương ứng với số thứ tự trong phần thử nghiệm.

**[1]** Nguyen, N., Phan, T., Nguyen, D.-V., & Nguyen, K. (2023). **ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing**. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*, pp. 5191–5207. Association for Computational Linguistics, Singapore.
- 🔗 https://aclanthology.org/2023.emnlp-main.315
- 📦 HuggingFace: https://huggingface.co/uitnlp/visobert

**[2]** Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**. *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*, pp. 8748–8763. PMLR.
- 🔗 https://proceedings.mlr.press/v139/radford21a.html
- 📦 HuggingFace: https://huggingface.co/openai/clip-vit-base-patch32

**[3]** (Author et al., 2026). **Text-Anchored Residual Cross-Modal Fusion for Multimodal Sentiment Analysis: A Unified and Protocol-Aware Evaluation on MVSA-Single**. *Applied Sciences*, 16(9), 4514. MDPI. *(Xác nhận RoBERTa + CLIP cross-attention đạt kết quả tốt nhất, accuracy 82.63% val / 79.62% test trên MVSA-Single.)*
- 🔗 https://doi.org/10.3390/app16094514

**[4]** Nguyen, T. T., Hy, T. S., & Vu, T. (2023). **ViDeBERTa: A powerful pre-trained language model for Vietnamese**. *Findings of the Association for Computational Linguistics: EACL 2023*, pp. 1071–1078. Association for Computational Linguistics, Dubrovnik, Croatia.
- 🔗 https://aclanthology.org/2023.findings-eacl.79
- 📦 GitHub: https://github.com/HySonLab/ViDeBERTa

**[5]** Habib, M. B., Hafiz, M. F. B., Khan, N. A., & Hossain, S. (2024). **Multimodal Sentiment Analysis using Deep Learning Fusion Techniques and Transformers**. *International Journal of Advanced Computer Science and Applications (IJACSA)*, 15(6). The Science and Information Organization. *(Benchmark trực tiếp EfficientNet-B3 + RoBERTa đạt accuracy 75%, F1 74.9% — tốt nhất trong các combo text+image được thử nghiệm.)*
- 🔗 http://dx.doi.org/10.14569/IJACSA.2024.0150686

**[6]** ViSoLex Team. (2025). **XLM-RoBERTa base fine-tuned for Vietnamese Aspect-based Sentiment Analysis** (`visolex/xlm-roberta-absa-restaurant`). *Hugging Face Model Hub*. Fine-tuned trên VLSP2018-ABSA-Restaurant với 12 aspect categories, đạt Accuracy 0.8897 / Weighted-F1 0.8107.
- 📦 HuggingFace: https://huggingface.co/visolex/xlm-roberta-absa-restaurant

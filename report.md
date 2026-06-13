# BÁO CÁO TIẾN ĐỘ DỰ ÁN

## Hệ thống học sâu đa phương thức có khả năng giải thích cho đánh giá chất lượng trải nghiệm ăn uống từ ảnh và văn bản

**Môn học:** SE365  
**Dữ liệu nghiên cứu:** Đánh giá nhà hàng/quán ăn từ Foody.vn  
**Loại bài toán:** Hồi quy điểm đánh giá đa tiêu chí từ ảnh và bình luận  
**Ngày báo cáo:** 13/06/2026  
**Phiên bản báo cáo:** Progress Report

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)  
2. [Đóng góp của dự án](#2-đóng-góp-của-dự-án)  
3. [Công việc đã hoàn thành](#3-công-việc-đã-hoàn-thành)  
4. [Kết quả thực nghiệm hiện tại](#4-kết-quả-thực-nghiệm-hiện-tại)  
5. [Khó khăn đã gặp](#5-khó-khăn-đã-gặp)  
6. [Công việc còn lại và theo dõi tiến độ](#6-công-việc-còn-lại-và-theo-dõi-tiến-độ)  
7. [Cột mốc tiếp theo](#7-cột-mốc-tiếp-theo)  

---

# 1. Tổng quan dự án

## 1.1 Bối cảnh và động lực nghiên cứu

Trong các nền tảng đánh giá trực tuyến về ăn uống, người dùng thường dựa đồng thời vào hình ảnh món ăn, không gian/quán và nội dung bình luận để đánh giá chất lượng trải nghiệm. Tuy nhiên, nhiều hướng tiếp cận dự đoán điểm đánh giá chỉ sử dụng một nguồn thông tin đơn lẻ, chẳng hạn chỉ dùng văn bản hoặc chỉ dùng ảnh. Cách tiếp cận này dễ bỏ sót các tín hiệu quan trọng: ảnh có thể phản ánh hình thức món ăn hoặc bối cảnh trải nghiệm, trong khi bình luận thể hiện cảm nhận ngữ nghĩa về đồ ăn, giá cả, phục vụ và không gian.

Dự án hướng đến xây dựng một hệ thống học sâu đa phương thức có khả năng dự đoán điểm đánh giá từ cả ảnh và văn bản. Bên cạnh hiệu năng dự đoán, dự án cũng đặt trọng tâm vào tính minh bạch của mô hình. Trong bối cảnh hệ thống đề xuất và đánh giá tự động ngày càng ảnh hưởng đến quyết định của người dùng, khả năng giải thích vì sao một dự đoán được đưa ra là yếu tố quan trọng để tăng độ tin cậy và giá trị ứng dụng.

## 1.2 Bài toán nghiên cứu

Bài toán hiện tại được triển khai dưới dạng hồi quy đa đầu ra. Với mỗi cặp dữ liệu gồm bình luận đã làm sạch và ảnh đánh giá, hệ thống dự đoán bốn điểm khía cạnh:

| Đầu ra | Ý nghĩa |
|---|---|
| `food_score` | Chất lượng đồ ăn |
| `price_score` | Mức độ phù hợp về giá |
| `atmosphere_score` | Không gian/bầu không khí |
| `service_score` | Chất lượng phục vụ |

Ngoài bốn nhãn trên, dự án đã xây dựng thêm nhãn `overall_satisfaction` nhằm phản ánh mức độ hài lòng tổng thể dựa trên điểm trung bình bốn khía cạnh và tín hiệu ngôn ngữ toàn cục trong bình luận. Tuy nhiên, trong mã huấn luyện hiện tại, mô hình chính mới sử dụng bốn điểm khía cạnh làm nhãn huấn luyện; `overall_satisfaction` đã được tạo và lưu trong dữ liệu xử lý, nhưng chưa được tích hợp vào loss của pipeline huấn luyện chính.

## 1.3 Hướng giải quyết

Hệ thống hiện tại gồm ba nhánh mô hình:

| Thành phần | Triển khai hiện tại |
|---|---|
| Mô hình văn bản | `TextModel` dùng `AutoModel`, mặc định `xlm-roberta-base`, trích đặc trưng từ `pooler_output` hoặc token đầu tiên |
| Mô hình ảnh | `ImageModel` dùng `timm.create_model`, mặc định `convnext_base_in22k`, lấy vector đặc trưng trước classifier |
| Mô hình hợp nhất | `FusionModel` đóng băng hai nhánh text/image, nối đặc trưng văn bản và ảnh, sau đó đưa qua MLP `fusion_size -> 512 -> 256 -> 4` |

Pipeline huấn luyện sử dụng PyTorch, `MSELoss` cho hồi quy đa đầu ra, AdamW, gradient clipping và đánh giá bằng MSE, RMSE, MAE theo từng tiêu chí. Đây là nền tảng phù hợp để tiếp tục mở rộng sang giải thích mô hình bằng Grad-CAM cho ảnh, attention visualization cho văn bản và phân tích đóng góp đa phương thức bằng SHAP/LIME.

# 2. Đóng góp của dự án

## 2.1 Đóng góp đã được triển khai

| Mã | Đóng góp | Bằng chứng triển khai |
|---|---|---|
| C1 | Xây dựng bộ dữ liệu đánh giá Foody đa phương thức tiếng Việt | Dữ liệu đã làm sạch gồm 9.946 review hợp lệ, 22.150 cặp review-ảnh; dữ liệu crawl ban đầu gồm 300 nhà hàng/quán |
| C2 | Xây dựng pipeline làm sạch và chuẩn bị dữ liệu cho học máy | Có notebook crawl/clean, `preprocess_data.py`, `download_images.py`, dữ liệu `data_raw/` và `data_processed/` |
| C3 | Thiết kế nhãn hài lòng tổng thể có bằng chứng giải thích | `overall_satisfaction` được sinh bằng rule engine từ 14 nhóm luật, có `overall_evidence` cho từng review |
| C4 | Triển khai mô hình hồi quy đa phương thức ảnh-văn bản | Có `TextModel`, `ImageModel`, `FusionModel`, `Trainer.py`, `main.py`, `test.py` |
| C5 | Thực nghiệm so sánh ít nhất hai cấu hình encoder | Notebook đã chạy XLM-RoBERTa + ConvNeXt và mDeBERTa + SigLIP trên tập 4.000/500/500 |

## 2.2 Đóng góp đang ở mức định hướng hoặc chưa hoàn tất

Các kỹ thuật Grad-CAM, attention visualization, SHAP và LIME đã được xác định trong tài liệu nghiên cứu và có mô tả phương pháp, nhưng chưa có module XAI tích hợp trực tiếp vào pipeline huấn luyện/đánh giá hiện tại. Vì vậy, trong giai đoạn báo cáo này, phần XAI đã hoàn thành chủ yếu là giải thích nhãn `overall_satisfaction` bằng luật và bằng chứng văn bản; XAI cho mô hình học sâu vẫn là công việc cần hoàn thiện.

# 3. Công việc đã hoàn thành

## 3.1 Thu thập và làm sạch dữ liệu

| Hạng mục | Trạng thái | Kết quả hiện tại |
|---|---|---|
| Crawl nhà hàng, review và ảnh từ Foody | Hoàn thành | 300 nhà hàng, 11.111 review thô, 24.599 ảnh thô |
| Làm sạch kỹ thuật và nội dung | Hoàn thành | 9.946 review hợp lệ sau lọc nội dung và rating |
| Ghép dữ liệu ảnh-văn bản | Hoàn thành | 22.150 cặp review-ảnh, tương ứng 6.082 review có ảnh |
| Đánh giá độ phủ ảnh | Hoàn thành | 61,15% review hợp lệ có ảnh |
| Sinh dữ liệu tăng cường | Hoàn thành | `reviews_clean_enhanced.csv/json` gồm 9.946 dòng và 47 cột |

Trong tập review sạch, có 298 `restaurant_id` duy nhất xuất hiện sau khi lọc review hợp lệ. Con số 300 phản ánh số nhà hàng đã thu thập ở dữ liệu crawl ban đầu.

## 3.2 Sinh nhãn hài lòng tổng thể

Pipeline tạo nhãn `overall_satisfaction` đã được hoàn thành trong notebook `01_generate_overall_satisfaction.ipynb`. Nhãn được tính theo công thức:

```text
avg_rating = mean(food_score, service_score, atmosphere_score, price_score)
overall_satisfaction = clip(avg_rating + overall_adjustment, 0, 10)
```

Trong đó `position_score` được giữ lại để truy vết nhưng không dùng trong nhãn chất lượng chính, vì đây là tín hiệu về vị trí/khả năng tiếp cận, không trực tiếp quan sát được từ ảnh món ăn và nội dung review.

| Chỉ số | Giá trị |
|---|---:|
| Số review được sinh nhãn | 9.946 |
| Số nhóm luật | 14 |
| Nhóm luật tích cực | 8 |
| Nhóm luật tiêu cực | 6 |
| Review có điều chỉnh khác 0 | 3.263 |
| Review có điều chỉnh tích cực | 2.058 |
| Review có điều chỉnh tiêu cực | 1.205 |
| Review không kích hoạt luật | 6.643 |

## 3.3 Phát triển mô hình và pipeline huấn luyện

| Thành phần | Trạng thái | Mô tả |
|---|---|---|
| Dataset loader | Hoàn thành | `MultimodalDataset` đọc CSV, tokenize `comment_clean`, tải ảnh local theo MD5 URL hoặc tải từ URL dự phòng |
| Text branch | Hoàn thành | XLM-RoBERTa/mô hình HuggingFace bất kỳ qua `AutoModel`, head hồi quy 4 đầu ra |
| Image branch | Hoàn thành | ConvNeXt hoặc mô hình `timm`, head hồi quy 4 đầu ra |
| Fusion branch | Hoàn thành | Nối đặc trưng text-image, MLP hồi quy 4 điểm |
| Training loop | Hoàn thành | Huấn luyện theo mode `train_text`, `train_image`, `train_fusion` |
| Evaluation script | Hoàn thành | Đánh giá MSE, RMSE, MAE trên tập test độc lập |
| Lưu checkpoint | Hoàn thành trong pipeline | Lưu `best_model_train_text/image/fusion.pth` khi validation loss tốt hơn |

Lưu ý: trong workspace hiện tại không có thư mục `data/text`, `data/image` hoặc `checkpoints`; các thực nghiệm đầy đủ được ghi nhận trong notebook chạy trên Kaggle/Colab với dữ liệu ngoài.

## 3.4 Tổng hợp trạng thái chung

| Nhóm công việc | Trạng thái | Nhận xét |
|---|---|---|
| Dữ liệu thô và dữ liệu sạch | Hoàn thành | Có đầy đủ file CSV/JSON trong `data_raw/` và `data_processed/` |
| Nhãn `overall_satisfaction` | Hoàn thành | Có luật, bằng chứng và báo cáo phân tích |
| Mô hình text/image/fusion | Hoàn thành bản baseline | Đã có code huấn luyện và đánh giá |
| Thực nghiệm baseline | Hoàn thành một phần | Có kết quả cho hai cấu hình encoder chính |
| XAI cho mô hình học sâu | Chưa hoàn thành | Chưa tích hợp Grad-CAM, attention, SHAP/LIME vào code chạy chính |
| Báo cáo/thesis | Đang thực hiện | Đã có tài liệu kỹ thuật, cần chuẩn hóa thành nội dung học thuật cuối cùng |

# 4. Kết quả thực nghiệm hiện tại

## 4.1 Thiết lập thực nghiệm

Các notebook thực nghiệm gần nhất sử dụng tập dữ liệu đa phương thức gồm 5.000 mẫu sau khi lọc ảnh khả dụng:

| Tập dữ liệu | Số mẫu |
|---|---:|
| Train | 4.000 |
| Validation | 500 |
| Test | 500 |

Mô hình được huấn luyện theo ba giai đoạn: huấn luyện nhánh văn bản, huấn luyện nhánh ảnh, sau đó huấn luyện mô hình fusion với hai encoder đã được nạp trọng số và đóng băng.

## 4.2 Kết quả trên tập test

| Mô hình | Food MAE | Price MAE | Atmos MAE | Service MAE | Nhận xét |
|---|---:|---:|---:|---:|---|
| XLM-RoBERTa + ConvNeXt | **1.0098** | **1.0483** | **1.0135** | **1.0330** | Kết quả tốt nhất hiện tại |
| mDeBERTa + SigLIP | 1.1016 | 1.0694 | 1.0547 | 1.1185 | Kém hơn ở cả bốn tiêu chí |

| Mô hình | Food RMSE | Price RMSE | Atmos RMSE | Service RMSE |
|---|---:|---:|---:|---:|
| XLM-RoBERTa + ConvNeXt | **1.3870** | **1.4537** | **1.3673** | **1.3925** |
| mDeBERTa + SigLIP | 1.4941 | 1.4820 | 1.4252 | 1.5198 |

| Mô hình | Food MSE | Price MSE | Atmos MSE | Service MSE |
|---|---:|---:|---:|---:|
| XLM-RoBERTa + ConvNeXt | **1.9238** | **2.1133** | **1.8695** | **1.9391** |
| mDeBERTa + SigLIP | 2.2324 | 2.1963 | 2.0311 | 2.3099 |

## 4.3 Nhận xét thực nghiệm

Kết quả hiện tại cho thấy cấu hình XLM-RoBERTa + ConvNeXt đạt sai số MAE khoảng 1,01 đến 1,05 điểm trên thang 0-10 cho bốn tiêu chí đánh giá. Đây là kết quả khả quan đối với dữ liệu review người dùng có nhiễu, nhiều phong cách diễn đạt và độ lệch nhãn tự nhiên.

So với mDeBERTa + SigLIP, cấu hình XLM-RoBERTa + ConvNeXt ổn định hơn trong thực nghiệm hiện có. Kết quả này chưa đủ để kết luận tuyệt đối về ưu thế kiến trúc, nhưng là cơ sở hợp lý để chọn XLM-RoBERTa + ConvNeXt làm baseline chính cho giai đoạn tiếp theo.

Các kết quả hiện tại chỉ phản ánh bốn đầu ra khía cạnh. Thực nghiệm dự đoán trực tiếp `overall_satisfaction` chưa được tích hợp nhất quán vào pipeline hiện tại, do đó chưa nên trình bày như kết quả chính thức của mô hình. Notebook mBERT + ResNet50 hiện mới có lệnh chạy nhưng chưa có output kết quả, vì vậy không được đưa vào bảng so sánh định lượng.

# 5. Khó khăn đã gặp

## 5.1 Nhiễu và mất cân bằng trong dữ liệu review

Dữ liệu người dùng có nhiều dạng nhiễu: bình luận ngắn, ngôn ngữ không chuẩn, emoji, tiếng Việt không dấu/viết tắt, bình luận pha tiếng Anh và nhận xét cảm tính mạnh. Mặc dù pipeline cleaning đã loại bỏ một phần nhiễu, phân phối điểm vẫn thiên về các mức 6-10, khiến mô hình có nguy cơ học thiên lệch về các đánh giá tích cực.

## 5.2 Liên kết ảnh-văn bản không luôn hoàn hảo

Một review có thể có nhiều ảnh, và mỗi ảnh không nhất thiết phản ánh đầy đủ nội dung bình luận. Pipeline hiện tại biểu diễn mỗi cặp review-ảnh thành một dòng dữ liệu, giúp tăng số mẫu đa phương thức nhưng cũng tạo ra rủi ro: cùng một bình luận có thể được ghép với nhiều ảnh có mức độ liên quan khác nhau.

## 5.3 Hợp nhất đa phương thức còn ở mức baseline

Mô hình fusion hiện tại dùng nối vector đặc trưng và MLP. Cách này đơn giản, dễ kiểm soát và phù hợp cho baseline, nhưng chưa học tương tác sâu giữa token văn bản và vùng ảnh. Các phương pháp như cross-attention hoặc gated fusion có thể cải thiện năng lực biểu diễn nhưng cần thêm thực nghiệm và kiểm soát overfitting.

## 5.4 XAI cho mô hình học sâu chưa hoàn thiện

Dự án đã có giải thích rule-based cho nhãn `overall_satisfaction`, nhưng chưa có triển khai chạy được cho Grad-CAM, attention visualization, SHAP hoặc LIME trong pipeline chính. Đây là phần quan trọng cần hoàn thiện để dự án thực sự đạt mục tiêu “explainable” ở cấp độ mô hình dự đoán.

## 5.5 Ràng buộc tính toán và quản lý artifact

Các thực nghiệm huấn luyện encoder lớn cần GPU và dữ liệu ảnh local. Workspace hiện tại không lưu trực tiếp `data/image`, `data/text` và checkpoint mô hình, trong khi notebook thực nghiệm tham chiếu dữ liệu Kaggle/Drive. Điều này cần được chuẩn hóa để tái lập kết quả khi bảo vệ hoặc nộp báo cáo cuối kỳ.

# 6. Công việc còn lại và theo dõi tiến độ

| Công việc | Trạng thái | Tiến độ |
|---|---|---:|
| Crawl và làm sạch dữ liệu Foody | Hoàn thành | 100% |
| Tạo dataset đa phương thức review-ảnh | Hoàn thành | 100% |
| Sinh nhãn `overall_satisfaction` và bằng chứng luật | Hoàn thành | 100% |
| Triển khai mô hình Text/Image/Fusion baseline | Hoàn thành | 100% |
| Huấn luyện và đánh giá XLM-RoBERTa + ConvNeXt | Hoàn thành | 90% |
| So sánh với mDeBERTa + SigLIP | Hoàn thành một phần | 75% |
| Tái lập dữ liệu train/val/test trong repo hoặc hướng dẫn artifact | Đang thực hiện | 60% |
| Tích hợp dự đoán `overall_satisfaction` vào pipeline chính | Chưa hoàn thành | 35% |
| Grad-CAM cho nhánh ảnh | Chưa hoàn thành | 20% |
| Attention visualization cho nhánh văn bản | Chưa hoàn thành | 20% |
| SHAP/LIME cho đóng góp đa phương thức | Chưa hoàn thành | 10% |
| Phân tích lỗi định tính | Chưa hoàn thành | 25% |
| Viết báo cáo/thesis cuối cùng | Đang thực hiện | 45% |

# 7. Cột mốc tiếp theo

## 7.1 Mục tiêu ngắn hạn

Cột mốc tiếp theo là biến hệ thống hiện tại từ baseline dự đoán thành hệ thống có khả năng đánh giá và giải thích đầy đủ hơn. Trọng tâm trước mắt gồm:

1. Chuẩn hóa lại pipeline dữ liệu cuối cùng để đảm bảo có thể tái lập chính xác tập train/validation/test và đường dẫn ảnh.
2. Chạy lại cấu hình XLM-RoBERTa + ConvNeXt với checkpoint được lưu và quản lý rõ ràng.
3. Tích hợp thêm đầu ra `overall_satisfaction` hoặc thiết kế bài toán multi-task nếu phù hợp.
4. Cài đặt Grad-CAM cho ConvNeXt để trực quan hóa vùng ảnh ảnh hưởng đến từng điểm dự đoán.
5. Cài đặt attention/token visualization cho XLM-RoBERTa để xác định cụm từ quan trọng trong bình luận.
6. Thực hiện phân tích lỗi theo nhóm: dự đoán sai cao, review mâu thuẫn ảnh-văn bản, review nhiều ảnh và review có ngôn ngữ cảm xúc mạnh.

## 7.2 Kết quả kỳ vọng ở mốc tiếp theo

Sau cột mốc tiếp theo, dự án cần có một bộ kết quả hoàn chỉnh hơn gồm: bảng so sánh baseline, biểu đồ/ảnh minh họa XAI, ví dụ giải thích cho từng dự đoán và phân tích lỗi định tính. Đây sẽ là cơ sở trực tiếp cho báo cáo cuối kỳ hoặc chương thực nghiệm trong luận văn.

## 7.3 Kết luận tiến độ

Dự án đã hoàn thành phần nền tảng quan trọng: dữ liệu Foody đa phương thức, nhãn hài lòng tổng thể có bằng chứng, pipeline huấn luyện PyTorch và kết quả thực nghiệm baseline. Kết quả XLM-RoBERTa + ConvNeXt hiện là cấu hình mạnh nhất với MAE khoảng 1 điểm trên bốn tiêu chí. Phần còn lại chủ yếu tập trung vào chuẩn hóa tái lập thực nghiệm, mở rộng nhãn tổng thể vào mô hình và hoàn thiện Explainable AI cho mô hình học sâu. Với khối lượng đã hoàn thành, các bước còn lại là thực tế và có thể triển khai theo từng module trong giai đoạn tiếp theo.

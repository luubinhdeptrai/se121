# Tiki Review Crawler (Review-First)

Script chính: `Crawl_data_tiki.py`

Mục tiêu chính của script này là **crawl review** và (tuỳ chọn) **tải ảnh review**. Luồng theo category vẫn còn, nhưng khuyến nghị dùng mode `--review-only` để tránh crawl lan rộng.

## 1) Chuẩn bị

Yêu cầu:
- Python 3.9+
- Internet để gọi Tiki API

Cài nhanh (tuỳ chọn venv):
```bash
cd /Users/abc/Documents/SE
python3 -m venv .venv
. .venv/bin/activate
pip install requests pandas beautifulsoup4 tenacity pytz
```

## 2) Chạy đúng mục tiêu review (khuyến nghị)

Chạy 1 sản phẩm, chỉ lấy review, giới hạn số trang review:
```bash
python3 Crawl_data_tiki.py \
  --review-only \
  --product-id 279291402 \
  --output-folder review-one \
  --max-review-pages 1 \
  --download-review-images
```

Giải thích:
- `--review-only`: chỉ chạy pipeline review, không duyệt category.
- `--product-id`: ID sản phẩm cần crawl review.
- `--output-folder`: thư mục con trong `./data/` để lưu output.
- `--max-review-pages`: giới hạn số trang review mỗi sản phẩm.
- `--download-review-images`: chỉ giữ review có ảnh và tải ảnh về local.

## 3) Cấu trúc output (review-first)

Ví dụ với `--output-folder review-one`:
- `./data/review-one/reviews.csv`
- `./data/review-one/customers.csv`
- `./data/review-one/buy_historys.csv`
- `./data/review-one/reviews_checkpoint.json`
- `./data/review-one/review_images/{product_id}/{review_id}/*.jpg`

## 4) Schema chính của `reviews.csv`

Các cột quan trọng:
- `id`: review id
- `product_id`: id sản phẩm
- `content`: nội dung review
- `images`: danh sách URL ảnh review (JSON string)
- `review_image_count`: số ảnh trong review
- `first_review_image_url`: ảnh đầu tiên
- `has_review_images`: cờ có ảnh hay không
- `images_raw`: payload ảnh gốc từ API

Ghi chú:
- Khi bật `--download-review-images`, file `reviews.csv` chỉ chứa review có ảnh.
- Một review có thể có nhiều ảnh.

## 5) Chạy mode category (không khuyến nghị cho test nhanh)

Mode này có thể crawl nhiều nhánh category con:
```bash
python3 Crawl_data_tiki.py \
  --category-name nha-sach-tiki \
  --category-id 8322 \
  --max-products 2 \
  --max-review-pages 1 \
  --download-review-images
```

## 6) Lỗi thường gặp

- Không có `reviews.csv`: sản phẩm không có review có ảnh (khi bật `--download-review-images`).
- Crawl quá nhiều: bạn đang chạy mode category; chuyển sang `--review-only`.
- Warning `LibreSSL` từ `urllib3`: thường không chặn chạy, chỉ là cảnh báo môi trường SSL.

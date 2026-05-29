import requests
import json
import csv
from pathlib import Path
import random
import time
import argparse
from datetime import datetime
import pytz
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import logging

# Cấu hình logging
logging.basicConfig(
    filename="crawl_tiki.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

product_url = "https://tiki.vn/api/v2/products/{}"
headers = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"
}

# Hàm request với retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.HTTPError)),
    after=lambda retry_state: logging.warning(
        f"Thu lai that bai sau {retry_state.attempt_number} lan cho URL: {retry_state.args[0]}"
    )
)
def safe_request(url, headers, timeout=20):
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def normalize_review_image_urls(images):
    urls = []
    if not isinstance(images, list):
        return urls
    for image in images:
        if isinstance(image, str) and image.strip():
            urls.append(image.strip())
            continue
        if not isinstance(image, dict):
            continue
        for key in ("full_path", "large_url", "medium_url", "small_url", "thumbnail_url", "base_url"):
            val = image.get(key)
            if isinstance(val, str) and val.strip():
                urls.append(val.strip())
                break
    # Giữ thứ tự ban đầu, loại trùng.
    return list(dict.fromkeys(urls))


def download_review_images(reviews, output_root):
    root = Path(output_root)
    total_downloaded = 0
    for review in reviews:
        urls = review.get("review_image_urls", [])
        if not urls:
            continue
        product_id = review.get("product_id")
        review_id = review.get("id")
        if not product_id or not review_id:
            continue
        image_dir = root / str(product_id) / str(review_id)
        image_dir.mkdir(parents=True, exist_ok=True)
        for idx, image_url in enumerate(urls, start=1):
            try:
                response = requests.get(image_url, headers=headers, timeout=20)
                response.raise_for_status()
                ext = Path(image_url.split("?")[0]).suffix.lower()
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    ext = ".jpg"
                image_path = image_dir / f"{idx:02d}{ext}"
                image_path.write_bytes(response.content)
                total_downloaded += 1
            except Exception as exc:
                logging.warning(f"Khong tai duoc anh review {review_id} ({image_url}): {exc}")
    return total_downloaded

# Hàm lưu và tải checkpoint
def save_checkpoint(crawled_ids, checkpoint_file):
    with open(checkpoint_file, "w") as f:
        json.dump(list(crawled_ids), f)

def load_checkpoint(checkpoint_file):
    try:
        with open(checkpoint_file, "r") as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def fetch_product_ids(url):
    """
    dầu vào: url là API của một danh mục sản phẩm
    dầu ra: danh sách chứa ID sản phẩm và số trang
    """
    product_list = []
    i = 1
    while True:
        logging.info(f"Crawl danh muc, trang {i}: {url.format(i)}")
        try:
            response = safe_request(url.format(i), headers=headers)
            products = json.loads(response.text)["data"]
            if not products:
                logging.info(f"Het du lieu danh mục, trang {i}")
                break
            for product in products:
                product_id = str(product["id"])
                product_list.append(product_id)
            i += 1
            time.sleep(random.uniform(0.5, 1))
        except Exception as e:
            logging.error(f"Loi crawl danh muc, trang {i}: {e}")
            break
    logging.info(f"Crawl dược {len(product_list)} ID san pham")
    return product_list, i

def fetch_product_details(product_list=[]):
    """
    Lấy thông tin chi tiết sản phẩm từ Tiki API.
    """
    product_detail_list = []
    for product_id in product_list:
        logging.info(f"Crawl chi tiet san pham {product_id}")
        try:
            response = safe_request(product_url.format(product_id), headers=headers)
            product_detail_list.append(response.text)
            time.sleep(random.uniform(0.5, 1))
        except Exception as e:
            logging.error(f"Lỗi crawl sản phẩm {product_id} sau 3 lần thử: {e}")
            continue  # Chuyển sang sản phẩm tiếp theo
    logging.info(f"Crawl duoc {len(product_detail_list)} chi tiết san pham")
    return product_detail_list

def normalize_product_data(product):
    """
    Chuẩn hóa dữ liệu sản phẩm từ JSON.
    """
    if not product:
        return None
    try:
        e = json.loads(product)
    except json.JSONDecodeError as err:
        logging.error(f"Loi JSON: {err}")
        return None
    if not e or not e.get("id"):
        logging.warning(f"Khong tim thay 'id' trong JSON: {e}")
        return None

    soup = BeautifulSoup(e.get("description", "") or "", "html.parser")
    images = e.get("images", []) if isinstance(e.get("images", []), list) else []
    stock_item = e.get("stock_item") or {}

    data_product = {
        "id": e["id"],
        "name": e["name"],
        "price": e.get("price", e.get("list_price", e.get("original_price", None))),
        "short_description": e.get("short_description", None),
        "description": soup.get_text(separator=" ", strip=True),
        "rating_average": e.get("rating_average"),
        "review_count": e.get("review_count", 0),
        "image_base_url": ", ".join(img.get("base_url", "N/A") for img in images),
        "image_large_url": ", ".join(img.get("large_url", "N/A") for img in images),
        "image_medium_url": ", ".join(img.get("medium_url", "N/A") for img in images),
        "image_small_url": ", ".join(img.get("small_url", "N/A") for img in images),
        "image_thumbnail_url": ", ".join(img.get("thumbnail_url", "N/A") for img in images),
        "stock": stock_item.get("qty", None),
        "quantity_sold": e.get("review_count", 0),
    }
    return data_product, e["id"]

def save_product_id(product_id_file, product_list=[]):
    with open(product_id_file, "w") as file:
        file.write("\n".join(product_list))
    logging.info(f"Lưu file: {product_id_file}")

def save_raw_product(product_data_file, product_detail_list=[]):
    with open(product_data_file, "w", encoding="utf-8") as file:
        file.write("\n".join(product_detail_list))
    logging.info(f"Lưu file: {product_data_file}")

def save_product_list_incremental(product_file, product_json):
    """
    Lưu sản phẩm tăng dần vào file CSV.
    """
    df = pd.DataFrame([product_json])
    if not Path(product_file).exists():
        df.to_csv(product_file, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(product_file, mode="a", header=False, index=False, encoding="utf-8-sig")
    logging.info(f"da them 1 san pham vào {product_file}")

def load_raw_product(product_data_file):
    with open(product_data_file, "r") as file:
        return file.readlines()

def map_json_to_customers(json_data):
    customers = {}
    for review in json_data.get("data", []):
        created_by = review.get("created_by") or {}
        contribute_info = (created_by.get("contribute_info") or {}).get("summary") or {}
        customer_id = created_by.get("id")
        if not customer_id:
            continue
        if customer_id not in customers:
            purchased_at = created_by.get("purchased_at")
            customers[customer_id] = {
                "id": customer_id,
                "full_name": created_by.get("full_name", ""),
                "avatar_url": created_by.get("avatar_url", ""),
                "joined_time": contribute_info.get("joined_time", ""),
                "total_review": contribute_info.get("total_review", 0),
                "total_thank": contribute_info.get("total_thank", 0),
                "purchased": created_by.get("purchased", False),
                "purchased_at": datetime.fromtimestamp(purchased_at, pytz.UTC) if isinstance(purchased_at, (int, float)) else None,
                "group_id": created_by.get("group_id")
            }
    return list(customers.values())

def map_json_to_review(json_data):
    reviews = []
    for review in json_data.get("data", []):
        created_by = review.get("created_by") or {}
        timeline = review.get("timeline") or {}
        created_at = timeline.get("review_created_date")
        try:
            created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC) if created_at else None
        except ValueError:
            created_at = None
        review_image_urls = normalize_review_image_urls(review.get("images", []))
        review_data = {
            "id": review.get("id"),
            "customer_id": created_by.get("id"),
            "product_id": review.get("product_id"),
            "rating": review.get("rating", 0),
            "title": review.get("title", ""),
            "content": review.get("content", ""),
            "status": review.get("status", "unknown"),
            "thank_count": review.get("thank_count", 0),
            "comment_count": review.get("comment_count", 0),
            "images": json.dumps(review_image_urls, ensure_ascii=False),
            "images_raw": json.dumps(review.get("images", []), ensure_ascii=False),
            "review_image_urls": json.dumps(review_image_urls, ensure_ascii=False),
            "review_images_flat": " | ".join(review_image_urls),
            "review_image_count": len(review_image_urls),
            "has_review_images": len(review_image_urls) > 0,
            "first_review_image_url": review_image_urls[0] if review_image_urls else "",
            "seller_id": review.get("seller", {}).get("id"),
            "seller_name": review.get("seller", {}).get("name", ""),
            "delivery_rating": review.get("delivery_rating", [])
        }
        reviews.append(review_data)
    return reviews

def map_json_to_buy_history(json_data):
    buy_histories = []
    for review in json_data.get("data", []):
        timeline = review.get("timeline") or {}
        customer_info = review.get("created_by") or {}
        purchased_at = customer_info.get("purchased_at")
        delivery_date = timeline.get("delivery_date")
        try:
            purchased_at = datetime.fromtimestamp(purchased_at, pytz.UTC) if isinstance(purchased_at, (int, float)) else None
        except ValueError:
            purchased_at = None
        try:
            delivery_date = datetime.strptime(delivery_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC) if delivery_date else None
        except ValueError:
            delivery_date = None
        buy_history_data = {
            "customer_id": customer_info.get("id"),
            "product_id": review.get("product_id"),
            "order_id": None,
            "seller_id": review.get("seller", {}).get("id"),
            "seller_name": review.get("seller", {}).get("name", ""),
            "quantity": random.randint(1, 4),
            "price_at_purchase": None,
            "total_price": None,
            "purchased_at": purchased_at,
            "delivery_date": delivery_date,
            "review_id": review.get("id")
        }
        buy_histories.append(buy_history_data)
    return buy_histories

def fetch_user_reviews_data(folder_parent_path, product_id_list, max_review_pages=None, download_images=False):
    """
    Thu thập dữ liệu dánh giá, thông tin khách hàng và lịch sử mua hàng.
    """
    API_cmt = "https://tiki.vn/api/v2/reviews?product_id={}&page={}&limit=20"
    reviews = []
    customers = []
    buy_historys = []

    reviews_file = f"./data/{folder_parent_path}/reviews.csv"
    customers_file = f"./data/{folder_parent_path}/customers.csv"
    buy_historys_file = f"./data/{folder_parent_path}/buy_historys.csv"
    Path(f"./data/{folder_parent_path}").mkdir(parents=True, exist_ok=True)

    # Checkpoint cho dánh giá
    checkpoint_file = f"./data/{folder_parent_path}/reviews_checkpoint.json"
    crawled_reviews = load_checkpoint(checkpoint_file)

    for product_id in product_id_list:
        if product_id in crawled_reviews:
            logging.info(f"Bo qua san pham da crawl danh gia: {product_id}")
            continue
        i = 1
        while True:
            if isinstance(max_review_pages, int) and max_review_pages > 0 and i > max_review_pages:
                logging.info(f"Dung theo gioi han max_review_pages={max_review_pages} cho san pham {product_id}")
                break
            logging.info(f"Crawl danh gia san pham {product_id}, trang {i}")
            try:
                response = safe_request(API_cmt.format(product_id, i), headers=headers)
                e = json.loads(response.text)
                if e["reviews_count"] == 0 or (i-1) * 20 > e["reviews_count"]:
                    logging.info(f"Hoan thanh crawl san pham {product_id}: {e['reviews_count']} danh gia")
                    crawled_reviews.add(product_id)
                    save_checkpoint(crawled_reviews, checkpoint_file)
                    break
                reviews += map_json_to_review(e)
                customers += map_json_to_customers(e)
                buy_historys += map_json_to_buy_history(e)
                i += 1
                time.sleep(random.uniform(0.5, 1))
            except Exception as e:
                logging.error(f"Loi crawl san pham {product_id}, trang {i} sau 3 lan thu: {e}")
                time.sleep(random.uniform(1, 2))
                continue
        crawled_reviews.add(product_id)
        save_checkpoint(crawled_reviews, checkpoint_file)

    if download_images:
        # Chỉ giữ review có ảnh khi bật chế độ tải ảnh review.
        reviews = [r for r in reviews if (r.get("review_image_count") or 0) > 0]

    if reviews:
        pd.DataFrame(reviews).to_csv(reviews_file, index=False, encoding="utf-8-sig")
    if reviews and download_images:
        rows_for_download = []
        for review in reviews:
            urls = []
            try:
                urls = json.loads(review.get("review_image_urls", "[]"))
            except Exception:
                urls = []
            rows_for_download.append({**review, "review_image_urls": urls})
        images_root = f"./data/{folder_parent_path}/review_images"
        total_downloaded = download_review_images(rows_for_download, images_root)
        logging.info(f"Da tai {total_downloaded} anh review vao {images_root}")
    if customers:
        if download_images:
            customer_ids = {r.get("customer_id") for r in reviews if r.get("customer_id") is not None}
            customers = [c for c in customers if c.get("id") in customer_ids]
        pd.DataFrame(customers).to_csv(customers_file, index=False, encoding="utf-8-sig")
    if buy_historys:
        if download_images:
            review_ids = {r.get("id") for r in reviews if r.get("id") is not None}
            buy_historys = [b for b in buy_historys if b.get("review_id") in review_ids]
        pd.DataFrame(buy_historys).to_csv(buy_historys_file, index=False, encoding="utf-8-sig")

    logging.info(f"Hoan thanh crawl: {len(reviews)} danh gia, {len(customers)} khach hang, {len(buy_historys)} lich su mua")

def fetch_and_save_product_data(folder, product_list_id):
    """
    Lấy dữ liệu chi tiết sản phẩm và lưu vào file.
    """
    product_id_file = f"./data/{folder}/product-id.txt"
    product_data_file = f"./data/{folder}/product.txt"
    product_file = f"./data/{folder}/product.csv"
    checkpoint_file = f"./data/{folder}/product_checkpoint.json"

    crawled_ids = load_checkpoint(checkpoint_file)
    product_list_id = [pid for pid in product_list_id if int(pid) not in crawled_ids]

    product_list = fetch_product_details(product_list_id)
    logging.info(f"Crawl duoc {len(product_list)} san pham")

    error_query = []
    for i, product in enumerate(product_list):
        temp = normalize_product_data(product)
        if temp is not None:
            product_json, product_id = temp
            save_product_list_incremental(product_file, product_json)
            crawled_ids.add(product_id)
            save_checkpoint(crawled_ids, checkpoint_file)
        else:
            error_query.append(i)

    logging.info(f"Co {len(error_query)} loi query")
    product_list_id = [val for i, val in enumerate(product_list_id) if i not in error_query]
    product_list = [val for i, val in enumerate(product_list) if i not in error_query]

    save_product_id(product_id_file, product_list_id)
    save_raw_product(product_data_file, product_list)

def fetch_category_data(parent, data, check=False, max_products=None, max_review_pages=None, download_images=False):
    """
    Thu thập dữ liệu sản phẩm từ một danh mục.
    """
    categors.append({
        "id": data.get("id", ""),
        "name": data.get("url_key", ""),
        "parent": data.get("parent_id", "")
    })
    folder_parent_path = parent if check else f"{parent}/{data['url_key'].strip()}"
    folder_path = Path(f"data/{folder_parent_path}")
    folder_path.mkdir(parents=True, exist_ok=True)

    id_childen_list = []
    i = 1
    logging.info(f"Crawl danh muc {data['url_key']} (ID: {data['id']})")
    while True:
        try:
            response = safe_request(
                f"https://tiki.vn/api/personalish/v1/blocks/listings?limit=10&page={i}&urlKey={data['url_key']}&category={data['id']}",
                headers=headers
            )
            products = json.loads(response.text)["data"]
            if not products:
                logging.info(f"Het du lieu danh muc {data['url_key']}, trang {i}")
                break
            for product in products:
                product_id = str(product["id"])
                id_childen_list.append(product_id)
                if isinstance(max_products, int) and max_products > 0 and len(id_childen_list) >= max_products:
                    break
            if isinstance(max_products, int) and max_products > 0 and len(id_childen_list) >= max_products:
                logging.info(f"Dung theo gioi han max_products={max_products} cho danh muc {data['url_key']}")
                break
            i += 1
            time.sleep(random.uniform(0.5, 1))
        except Exception as e:
            logging.error(f"Loi crawl danh muc {data['url_key']}, trang {i} sau 3 lan thu: {e}")
            break

    fetch_and_save_product_data(folder_parent_path, id_childen_list)
    fetch_user_reviews_data(
        folder_parent_path,
        id_childen_list,
        max_review_pages=max_review_pages,
        download_images=download_images,
    )
    logging.info(f"Hoan thanh danh muc {data['url_key']}")
    time.sleep(random.uniform(0.5, 1))

def fetch_and_traverse_categories(name, category_id, parent=None, max_products=None, max_review_pages=None, download_images=False):
    """
    Duyệt dệ quy các danh mục con.
    """
    checkpoint_file = f"data/{name}/checkpoint.json"
    crawled_ids = load_checkpoint(checkpoint_file)
    if str(category_id) in crawled_ids:
        logging.info(f"Bo qua danh muc da crawl: {name}")
        return

    url = f"https://tiki.vn/api/v2/categories?include=children&parent_id={category_id}"
    try:
        response = safe_request(url, headers=headers)
        data = response.json()
    except Exception as e:
        logging.error(f"Loi crawl danh muc {name} sau 3 lan thu: {e}")
        return

    if "data" not in data or not isinstance(data["data"], list):
        logging.warning(f"Khong tim thay danh sach danh muc con: {url}")
        return
    categors.append({
        "id": category_id,
        "name": name,
        "parent": parent
    })
    e = data["data"]
    if not e:
        fetch_category_data(
            name,
            {"url_key": name, "id": category_id},
            True,
            max_products=max_products,
            max_review_pages=max_review_pages,
            download_images=download_images,
        )
    else:
        for data in e:
            if "children" in data:
                fetch_and_traverse_categories(
                    f"{name}/{data['url_key'].strip()}",
                    data["id"],
                    category_id,
                    max_products=max_products,
                    max_review_pages=max_review_pages,
                    download_images=download_images,
                )
            else:
                fetch_category_data(
                    name,
                    data,
                    max_products=max_products,
                    max_review_pages=max_review_pages,
                    download_images=download_images,
                )

    crawled_ids.add(str(category_id))
    save_checkpoint(crawled_ids, checkpoint_file)

def fetch_and_save_categories(category_name, category_id, output_folder="./data", max_products=None, max_review_pages=None, download_images=False):
    """
    Thu thập danh mục sản phẩm và lưu vào file CSV.
    """
    global categors
    categors = []
    fetch_and_traverse_categories(
        category_name,
        category_id,
        max_products=max_products,
        max_review_pages=max_review_pages,
        download_images=download_images,
    )
    df = pd.DataFrame(categors)
    category_csv_path = f"{output_folder}/{category_name}/category.csv"
    df.to_csv(category_csv_path, index=False, encoding="utf-8-sig")
    logging.info(f"Luu file danh muc: {category_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category-name", default="nha-sach-tiki")
    parser.add_argument("--category-id", default="8322")
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--max-review-pages", type=int, default=None)
    parser.add_argument("--download-review-images", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--product-id", default=None)
    parser.add_argument("--output-folder", default=None)
    args = parser.parse_args()

    if args.review_only:
        if not args.product_id:
            raise ValueError("Can truyen --product-id khi dung --review-only")
        target_folder = args.output_folder or "review-only"
        fetch_user_reviews_data(
            target_folder,
            [str(args.product_id)],
            max_review_pages=args.max_review_pages,
            download_images=args.download_review_images,
        )
    else:
        fetch_and_save_categories(
            args.category_name,
            args.category_id,
            max_products=args.max_products,
            max_review_pages=args.max_review_pages,
            download_images=args.download_review_images,
        )

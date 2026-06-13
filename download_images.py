import pandas as pd
import requests
import os
from tqdm import tqdm
import concurrent.futures

def download_image(args):
    url, image_id, save_folder = args
    save_path = os.path.join(save_folder, f"{image_id}.jpg")
    
    # Nếu ảnh đã tồn tại thì bỏ qua
    if os.path.exists(save_path):
        return
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
    except:
        pass # Bỏ qua nếu lỗi

def main():
    # Thư mục lưu ảnh
    save_folder = './data/image'
    os.makedirs(save_folder, exist_ok=True)
    
    # Đọc tất cả các file data
    files = [('./data/text/train.csv', 4000), 
             ('./data/text/val.csv', 500), 
             ('./data/text/test.csv', 500)]
    dfs = []
    for f, target_size in files:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
            
    if not dfs:
        print("Không tìm thấy các file CSV!")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Bắt đầu tải {len(df)} ảnh (bao gồm dự phòng)...")
    
    import hashlib
    download_args = []
    for _, row in df.iterrows():
        url = row['image_url']
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        download_args.append((url, url_hash, save_folder))
        
    # Tải song song
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(download_image, download_args), total=len(download_args)))
        
    print(f"Đã tải xong ảnh vào {save_folder}")
    
    # Lọc lại các file CSV và lấy đúng số lượng
    print("Đang rà soát, cắt đúng 5000 dòng và xoá ảnh rác...")
    valid_hashes = set()
    for f, target_size in files:
        if os.path.exists(f):
            df_curr = pd.read_csv(f)
            valid_rows = []
            for _, row in df_curr.iterrows():
                url = row['image_url']
                url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
                img_path = os.path.join(save_folder, f"{url_hash}.jpg")
                valid_rows.append(os.path.exists(img_path))
            
            df_valid = df_curr[valid_rows]
            
            # Cắt ĐÚNG target_size
            df_final = df_valid.head(target_size)
            df_final.to_csv(f, index=False)
            print(f"[V] Đã tạo {f} với chính xác {len(df_final)} dòng (yêu cầu: {target_size}).")
            
            # Lưu lại hash hợp lệ để xoá ảnh rác
            for _, row in df_final.iterrows():
                valid_hashes.add(hashlib.md5(row['image_url'].encode('utf-8')).hexdigest())
            
    # Xoá các file ảnh thừa (do tải dư từ tập dự phòng)
    deleted_images = 0
    for img_file in os.listdir(save_folder):
        if img_file.endswith('.jpg'):
            img_hash = img_file.replace('.jpg', '')
            if img_hash not in valid_hashes:
                os.remove(os.path.join(save_folder, img_file))
                deleted_images += 1
    print(f"Đã xoá {deleted_images} ảnh thừa/dự phòng khỏi thư mục.")

if __name__ == '__main__':
    main()

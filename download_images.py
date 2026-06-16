import pandas as pd
import requests
import os
from tqdm import tqdm
import concurrent.futures
import ast
import hashlib

def download_image(args):
    url, image_id, save_folder = args
    save_path = os.path.join(save_folder, f"{image_id}.jpg")
    
    if os.path.exists(save_path):
        return
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
    except:
        pass

def main():
    save_folder = './data/image'
    os.makedirs(save_folder, exist_ok=True)
    
    # Kích thước 4800/600/600
    files = [('./data/text/train.csv', 4800), 
             ('./data/text/val.csv', 600), 
             ('./data/text/test.csv', 600)]
    dfs = []
    for f, target_size in files:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
            
    if not dfs:
        print("Không tìm thấy các file CSV!")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Bắt đầu tải ảnh từ {len(df)} mẫu review...")
    
    download_args = []
    for _, row in df.iterrows():
        try:
            urls = ast.literal_eval(row['image_url'])
        except:
            urls = [row['image_url']]
            
        for url in urls:
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
            download_args.append((url, url_hash, save_folder))
        
    # Loại bỏ các URL trùng lặp để tải nhanh hơn
    download_args = list(set(download_args))
    
    # Tải song song
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(download_image, download_args), total=len(download_args)))
        
    print(f"Đã tải xong ảnh vào {save_folder}")
    
    # Lọc lại các file CSV và lấy đúng số lượng
    print("Đang rà soát, giữ những mẫu có ảnh...")
    valid_hashes = set()
    for f, target_size in files:
        if os.path.exists(f):
            df_curr = pd.read_csv(f)
            valid_rows = []
            for _, row in df_curr.iterrows():
                try:
                    urls = ast.literal_eval(row['image_url'])
                except:
                    urls = [row['image_url']]
                
                # Check if AT LEAST ONE image downloaded successfully
                has_valid_image = False
                for url in urls:
                    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
                    img_path = os.path.join(save_folder, f"{url_hash}.jpg")
                    if os.path.exists(img_path):
                        has_valid_image = True
                        valid_hashes.add(url_hash)
                
                valid_rows.append(has_valid_image)
            
            df_valid = df_curr[valid_rows]
            
            # Cắt ĐÚNG target_size
            df_final = df_valid.head(target_size)
            df_final.to_csv(f, index=False)
            print(f"[V] Đã tạo {f} với {len(df_final)} dòng (yêu cầu: {target_size}).")
            
    # Xoá các file ảnh thừa
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

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
        response = requests.get(url, timeout=5)
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
    files = ['./data/text/train.csv', './data/text/val.csv', './data/text/test.csv']
    dfs = []
    for f in files:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
            
    if not dfs:
        print("Không tìm thấy các file CSV trong thư mục ./data/text/")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Bắt đầu tải {len(df)} ảnh...")
    
    import hashlib
    download_args = []
    for _, row in df.iterrows():
        url = row['image_url']
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        download_args.append((url, url_hash, save_folder))
        
    # Tải song song cho nhanh (ví dụ dùng 20 luồng)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(download_image, download_args), total=len(download_args)))
        
    print(f"Đã tải xong ảnh vào {save_folder}")

if __name__ == '__main__':
    main()

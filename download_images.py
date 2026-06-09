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
    save_folder = './data/images'
    os.makedirs(save_folder, exist_ok=True)
    
    # Gom cả 3 file train, val, test để tải ảnh
    dfs = []
    for f in ['./data/train.csv', './data/val.csv', './data/test.csv']:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
    
    if not dfs:
        print("Lỗi: Không tìm thấy các file CSV!")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Bắt đầu tải {len(df)} ảnh...")
    
    # Tạo danh sách arguments cho multi-threading
    # Sử dụng index của dataframe làm tên file ảnh để dễ map lại
    download_args = []
    for idx, row in df.iterrows():
        download_args.append((row['image_url'], idx, save_folder))
        
    # Tải song song cho nhanh (ví dụ dùng 20 luồng)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(download_image, download_args), total=len(download_args)))
        
    print(f"Đã tải xong ảnh vào {save_folder}")

if __name__ == '__main__':
    main()

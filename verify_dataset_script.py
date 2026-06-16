import os
import pandas as pd

def verify():
    # Kiểm tra số lượng file ảnh
    img_dir = './data/image'
    if os.path.exists(img_dir):
        imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        print(f"Tổng số ảnh thực tế trong thư mục data/image: {len(imgs)}")
    else:
        print("Không tìm thấy thư mục ảnh!")
    
    # Đọc số lượng dòng trong các file CSV
    files = ['./data/text/train.csv', './data/text/val.csv', './data/text/test.csv']
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print(f"[{f}] có {len(df)} review.")
            
if __name__ == '__main__':
    verify()

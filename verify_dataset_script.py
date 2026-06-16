import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoImageProcessor
from src.dataset import MultimodalDataset
import ast

def verify():
    # Kiểm tra số lượng file ảnh
    img_dir = './data/image'
    imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print(f"Tổng số ảnh thực tế trong thư mục data/image: {len(imgs)}")
    
    # Đọc số lượng dòng trong các file CSV
    files = ['./data/text/train.csv', './data/text/val.csv', './data/text/test.csv']
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print(f"[{f}] có {len(df)} review.")
            
    print("-" * 50)
    print("Test load thử MultimodalDataset (trong Trainer)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        image_processor = AutoImageProcessor.from_pretrained('facebook/convnext-base-224-22k')
        
        train_dataset = MultimodalDataset('./data/text/train.csv', tokenizer, image_processor, max_images=4)
        sample = train_dataset[0]
        
        print("Load sample đầu tiên thành công!")
        print(f"Kích thước input_ids: {sample['input_ids'].shape}")
        print(f"Kích thước pixel_values (4 ảnh, padding): {sample['pixel_values'].shape}")
        print(f"Số ảnh thực tế (num_images): {sample['num_images']}")
        print(f"Nhãn factor_scores: {sample['factor_scores']}")
        
        # Test Dataloader
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        print("\nTest DataLoader (batch_size=2):")
        print(f"Batch pixel_values shape: {batch['pixel_values'].shape} -> [B, N, C, H, W]")
        print(f"Batch num_images: {batch['num_images']}")
        
    except Exception as e:
        print(f"Lỗi khi load Dataset: {e}")

if __name__ == '__main__':
    verify()

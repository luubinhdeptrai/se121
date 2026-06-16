import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import requests
from PIL import Image
from io import BytesIO
import ast

class MultimodalDataset(Dataset):
    """
    Dataset cho Thử nghiệm 1 & 2 (Baseline & Loss).
    Đã được nâng cấp để hỗ trợ Average Pooling (Strong Baseline):
    - Lấy tối đa max_images=4 ảnh.
    - Trả về số lượng ảnh thực tế (num_images).
    """
    def __init__(self, csv_file, text_tokenizer, image_processor, max_length=128, max_images=4, image_dir='./data/image'):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.max_images = max_images
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def _load_image(self, url):
        import hashlib
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        local_path = os.path.join(self.image_dir, f"{url_hash}.jpg")
        if os.path.exists(local_path):
            return Image.open(local_path).convert("RGB")
            
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img
        except Exception as e:
            return Image.new('RGB', (224, 224), color='black')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text
        text = str(row['comment_clean'])
        text_inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # Image (Strong Baseline: lấy tối đa max_images, đệm ảnh đen nếu thiếu)
        try:
            image_urls = ast.literal_eval(row['image_url'])
        except:
            image_urls = [row['image_url']]
            
        num_images = min(len(image_urls), self.max_images)
        images = []
        
        for i in range(self.max_images):
            if i < len(image_urls):
                img = self._load_image(image_urls[i])
            else:
                img = Image.new('RGB', (224, 224), color='black')
            images.append(img)
            
        image_inputs = self.image_processor(images, return_tensors="pt")['pixel_values'] # [max_images, 3, 224, 224]
        
        # Labels
        factor_scores = torch.tensor([
            row['food_score'],
            row['price_score'],
            row['atmosphere_score'],
            row['service_score'],
            row['overall_satisfaction']
        ], dtype=torch.float)
        
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': image_inputs, # [max_images, 3, 224, 224]
            'num_images': torch.tensor(num_images, dtype=torch.long),
            'factor_scores': factor_scores
        }

class AdvancedMultimodalDataset(Dataset):
    """
    Dataset dùng cho Thử nghiệm Nhóm 3 (Cross-Attention / Gom nhóm ảnh).
    Đọc từ file CSV và load toàn bộ danh sách ảnh của mỗi bài review.
    """
    def __init__(self, csv_file, text_tokenizer, image_processor, max_length=128, image_dir='./data/image'):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def _load_image(self, url):
        import hashlib
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        local_path = os.path.join(self.image_dir, f"{url_hash}.jpg")
        if os.path.exists(local_path):
            return Image.open(local_path).convert("RGB")
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img
        except Exception as e:
            return Image.new('RGB', (224, 224), color='black')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Text
        text = str(row['comment_clean'])
        text_inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # 2. Images (List)
        try:
            image_urls = ast.literal_eval(row['image_url'])
        except:
            image_urls = [row['image_url']]
            
        images = [self._load_image(url) for url in image_urls]
        
        # Process qua ImageProcessor (sẽ ra dạng [N, 3, 224, 224])
        # Lưu ý: Trainer cần viết thêm collate_fn để pad số lượng N cho đều trong 1 batch.
        image_inputs = self.image_processor(images, return_tensors="pt")['pixel_values']
        
        # 3. Labels
        factor_scores = torch.tensor([
            row['food_score'],
            row['price_score'],
            row['atmosphere_score'],
            row['service_score'],
            row['overall_satisfaction']
        ], dtype=torch.float)
        
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': image_inputs,  # Tensor 4 chiều [N, 3, 224, 224]
            'factor_scores': factor_scores
        }

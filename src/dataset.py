import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import requests
from PIL import Image
from io import BytesIO

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, text_tokenizer, image_processor, max_length=128, image_dir='./data/images'):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def _load_image(self, url, idx):
        import hashlib
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        # Đường dẫn ảnh local 
        local_path = os.path.join(self.image_dir, f"{url_hash}.jpg")
        if os.path.exists(local_path):
            return Image.open(local_path).convert("RGB")
            
        try:
            # Tải ảnh từ URL (dự phòng)
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img
        except Exception as e:
            # Ảnh đen nếu lỗi
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
        
        # Image
        image_url = row['image_url']
        image = self._load_image(image_url, idx)
        image_inputs = self.image_processor(image, return_tensors="pt")['pixel_values']
        
        # Labels
        factor_scores = torch.tensor([
            row['food_score'],
            row['price_score'],
            row['atmosphere_score'],
            row['service_score']
        ], dtype=torch.float)
        
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': image_inputs.squeeze(0),
            'factor_scores': factor_scores
        }

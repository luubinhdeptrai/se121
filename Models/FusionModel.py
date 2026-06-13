import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, text_model, image_model, num_factors=4):
        super(FusionModel, self).__init__()
        # Các mô hình nhánh đã được train
        self.text_model = text_model
        self.image_model = image_model
        
        # Đóng băng trọng số (Freeze) nếu cần, hoặc để fine-tune với learning rate thấp
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.image_model.parameters():
            param.requires_grad = False
            
        # Tự động lấy kích thước feature từ các encoder
        text_hidden_size = self.text_model.encoder.config.hidden_size
        image_hidden_size = self.image_model.encoder.num_features
        
        fusion_size = text_hidden_size + image_hidden_size
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # self.overall_head = nn.Linear(256, 1)
        self.factor_head = nn.Linear(256, num_factors)

    def forward(self, input_ids, attention_mask, pixel_values):
        # Lấy features từ text_model và image_model (chúng ta đã thiết kế model trả về features ở index 2)
        with torch.no_grad():
            _, _, text_features = self.text_model(input_ids, attention_mask)
            _, _, image_features = self.image_model(pixel_values)
            
        # Concatenation
        text_features = text_features.to(torch.float32)
        image_features = image_features.to(torch.float32)
        fused_features = torch.cat((text_features, image_features), dim=1)
        out = self.fusion_fc(fused_features)
        
        # overall_score = self.overall_head(out)
        factor_scores = self.factor_head(out)
        return None, factor_scores

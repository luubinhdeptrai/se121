import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, text_model, image_model, num_factors=5):
        super(FusionModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        
        # Đóng băng trọng số
        for param in list(self.text_model.parameters()) + list(self.image_model.parameters()):
            param.requires_grad = False
            
        fusion_size = self.text_model.encoder.config.hidden_size + self.image_model.encoder.num_features
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.factor_head = nn.Linear(256, num_factors)

    def forward(self, input_ids, attention_mask, pixel_values, num_images=None):
        with torch.no_grad():
            _, text_features = self.text_model(input_ids, attention_mask)
            # ImageModel đã tự xử lý 5D tensor và tính Average Pooling bên trong
            _, image_features = self.image_model(pixel_values, num_images=num_images)
            
        fused_features = torch.cat((text_features.to(torch.float32), image_features.to(torch.float32)), dim=1)
        return self.factor_head(self.fusion_fc(fused_features))

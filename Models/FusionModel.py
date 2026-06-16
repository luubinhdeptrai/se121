import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, text_model, image_model, num_factors=5, unfreeze_text_layers=0, unfreeze_image_layers=0):
        super(FusionModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        
        # Đóng băng trọng số mặc định
        for param in list(self.text_model.parameters()) + list(self.image_model.parameters()):
            param.requires_grad = False
            
        # Mở khóa TextModel layers (XLM-RoBERTa)
        if unfreeze_text_layers > 0:
            if hasattr(self.text_model.encoder, "encoder") and hasattr(self.text_model.encoder.encoder, "layer"):
                text_layers = self.text_model.encoder.encoder.layer
                for layer in text_layers[-unfreeze_text_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                        
        # Mở khóa ImageModel layers (ConvNeXt trong timm)
        if unfreeze_image_layers > 0:
            if hasattr(self.image_model.encoder, "stages"):
                last_stage_blocks = self.image_model.encoder.stages[-1].blocks
                for block in last_stage_blocks[-unfreeze_image_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
            
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
        # Không dùng torch.no_grad() để Autograd có thể truyền ngược qua các layer đã được "tan băng"
        _, text_features = self.text_model(input_ids, attention_mask)
        # ImageModel tự xử lý 5D tensor và tính Average Pooling
        _, image_features = self.image_model(pixel_values, num_images=num_images)
            
        fused_features = torch.cat((text_features.to(torch.float32), image_features.to(torch.float32)), dim=1)
        return self.factor_head(self.fusion_fc(fused_features))

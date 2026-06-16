import torch
import torch.nn as nn
import timm

class ImageModel(nn.Module):
    def __init__(self, model_name='convnext_base', num_factors=5):
        super(ImageModel, self).__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.factor_head = nn.Linear(256, num_factors)
        
    def forward(self, pixel_values, num_images=None):
        # Unify 4D to 5D for consistent processing
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
            
        B, N, C, H, W = pixel_values.shape
        features = self.encoder(pixel_values.view(B * N, C, H, W)).view(B, N, -1)
        
        # Average pooling
        if num_images is not None:
            mask = (torch.arange(N, device=pixel_values.device).expand(B, N) < num_images.unsqueeze(1)).float().unsqueeze(-1)
            features = (features * mask).sum(dim=1) / num_images.float().clamp(min=1).unsqueeze(1)
        else:
            features = features.mean(dim=1)
            
        features = features.to(torch.float32)
        return self.factor_head(self.fc(features)), features

import torch
import torch.nn as nn
import timm

class ImageModel(nn.Module):
    def __init__(self, model_name='convnext_base', num_factors=4):
        super(ImageModel, self).__init__()
        # num_classes=0 to get raw features
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        hidden_size = self.encoder.num_features
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.factor_head = nn.Linear(256, num_factors)
        
    def forward(self, pixel_values):
        features = self.encoder(pixel_values)
        features = features.to(torch.float32)
        out = self.fc(features)
        factor_scores = self.factor_head(out)
        return factor_scores, features

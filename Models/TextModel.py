import torch
import torch.nn as nn
from transformers import AutoModel

class TextModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_factors=5):
        super(TextModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name).float()
        
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.factor_head = nn.Linear(256, num_factors)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Rút trích feature an toàn không cần if-else rườm rà
        features = outputs.pooler_output if getattr(outputs, 'pooler_output', None) is not None else outputs.last_hidden_state[:, 0, :]
        features = features.to(torch.float32)
        
        return self.factor_head(self.fc(features)), features

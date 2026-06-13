import torch
import torch.nn as nn
from transformers import AutoModel

class TextModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_factors=3):
        super(TextModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # self.overall_head = nn.Linear(256, 1)
        self.factor_head = nn.Linear(256, num_factors)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Using pooler_output or mean of last_hidden_state
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]
            
        out = self.fc(features)
        # overall_score = self.overall_head(out)
        factor_scores = self.factor_head(out)
        return None, factor_scores, features

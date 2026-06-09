import torch
import torch.nn as nn
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        self.alpha = args.alpha

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch in loop:
            self.optimizer.zero_grad()
            
            # Forward based on mode
            if self.args.mode == 'train_text':
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pred_overall, pred_factors, _ = self.model(input_ids, attention_mask)
            elif self.args.mode == 'train_image':
                pixel_values = batch['pixel_values'].to(self.device)
                pred_overall, pred_factors, _ = self.model(pixel_values)
            else: # fusion
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                pred_overall, pred_factors = self.model(input_ids, attention_mask, pixel_values)
            
            # Labels
            true_overall = batch['overall_score'].to(self.device)
            true_factors = batch['factor_scores'].to(self.device)
            
            # Compute loss
            loss_overall = self.criterion(pred_overall, true_overall)
            loss_factors = self.criterion(pred_factors, true_factors)
            loss = self.alpha * loss_overall + (1 - self.alpha) * loss_factors
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                if self.args.mode == 'train_text':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    pred_overall, pred_factors, _ = self.model(input_ids, attention_mask)
                elif self.args.mode == 'train_image':
                    pixel_values = batch['pixel_values'].to(self.device)
                    pred_overall, pred_factors, _ = self.model(pixel_values)
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    pixel_values = batch['pixel_values'].to(self.device)
                    pred_overall, pred_factors = self.model(input_ids, attention_mask, pixel_values)
                
                true_overall = batch['overall_score'].to(self.device)
                true_factors = batch['factor_scores'].to(self.device)
                
                loss_overall = self.criterion(pred_overall, true_overall)
                loss_factors = self.criterion(pred_factors, true_factors)
                loss = self.alpha * loss_overall + (1 - self.alpha) * loss_factors
                val_loss += loss.item()
                
        return val_loss / len(self.val_loader)

    def run(self):
        best_val_loss = float('inf')
        os.makedirs(self.args.save_path, exist_ok=True)
        save_file = os.path.join(self.args.save_path, f'best_model_{self.args.mode}.pth')

        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_file)
                print(f"--> Saved best model to {save_file}")
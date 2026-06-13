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
        self.mae_criterion = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

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
                pred_factors, _ = self.model(input_ids, attention_mask)
            elif self.args.mode == 'train_image':
                pixel_values = batch['pixel_values'].to(self.device)
                pred_factors, _ = self.model(pixel_values)
            else: # fusion
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                pred_factors = self.model(input_ids, attention_mask, pixel_values)
            
            # Labels
            true_factors = batch['factor_scores'].to(self.device)
            
            # Compute loss
            loss = self.criterion(pred_factors, true_factors)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_loss_food = 0.0
        val_loss_price = 0.0
        val_loss_atmos = 0.0
        val_loss_service = 0.0
        
        val_mae_food = 0.0
        val_mae_price = 0.0
        val_mae_atmos = 0.0
        val_mae_service = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if self.args.mode == 'train_text':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    pred_factors, _ = self.model(input_ids, attention_mask)
                elif self.args.mode == 'train_image':
                    pixel_values = batch['pixel_values'].to(self.device)
                    pred_factors, _ = self.model(pixel_values)
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    pixel_values = batch['pixel_values'].to(self.device)
                    pred_factors = self.model(input_ids, attention_mask, pixel_values)
                
                true_factors = batch['factor_scores'].to(self.device)
                
                loss = self.criterion(pred_factors, true_factors)
                
                loss_food = self.criterion(pred_factors[:, 0], true_factors[:, 0])
                loss_price = self.criterion(pred_factors[:, 1], true_factors[:, 1])
                loss_atmos = self.criterion(pred_factors[:, 2], true_factors[:, 2])
                loss_service = self.criterion(pred_factors[:, 3], true_factors[:, 3])
                
                # Tính MAE (Mean Absolute Error)
                mae_food = self.mae_criterion(pred_factors[:, 0], true_factors[:, 0])
                mae_price = self.mae_criterion(pred_factors[:, 1], true_factors[:, 1])
                mae_atmos = self.mae_criterion(pred_factors[:, 2], true_factors[:, 2])
                mae_service = self.mae_criterion(pred_factors[:, 3], true_factors[:, 3])
                
                val_loss += loss.item()
                val_loss_food += loss_food.item()
                val_loss_price += loss_price.item()
                val_loss_atmos += loss_atmos.item()
                val_loss_service += loss_service.item()
                
                val_mae_food += mae_food.item()
                val_mae_price += mae_price.item()
                val_mae_atmos += mae_atmos.item()
                val_mae_service += mae_service.item()
                
        num_batches = len(self.val_loader)
        return {
            'loss': val_loss / num_batches,
            'mse_food': val_loss_food / num_batches,
            'mse_price': val_loss_price / num_batches,
            'mse_atmos': val_loss_atmos / num_batches,
            'mse_service': val_loss_service / num_batches,
            'mae_food': val_mae_food / num_batches,
            'mae_price': val_mae_price / num_batches,
            'mae_atmos': val_mae_atmos / num_batches,
            'mae_service': val_mae_service / num_batches
        }

    def run(self):
        best_val_loss = float('inf')
        os.makedirs(self.args.save_path, exist_ok=True)
        save_file = os.path.join(self.args.save_path, f'best_model_{self.args.mode}.pth')

        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            metrics = self.validate()
            val_loss = metrics['loss']
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  -> RMSE | Food: {metrics['mse_food']**0.5:.4f} | Price: {metrics['mse_price']**0.5:.4f} | Atmos: {metrics['mse_atmos']**0.5:.4f} | Service: {metrics['mse_service']**0.5:.4f}")
            print(f"  -> MAE  | Food: {metrics['mae_food']:.4f} | Price: {metrics['mae_price']:.4f} | Atmos: {metrics['mae_atmos']:.4f} | Service: {metrics['mae_service']:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_file)
                print(f"*** Saved best model to {save_file} ***\n")
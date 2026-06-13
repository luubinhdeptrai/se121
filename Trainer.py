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
            # loss_overall = self.criterion(pred_overall, true_overall)
            loss_factors = self.criterion(pred_factors, true_factors)
            loss = loss_factors # self.alpha * loss_overall + (1 - self.alpha) * loss_factors
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_loss_overall = 0.0
        val_loss_food = 0.0
        val_loss_price = 0.0
        val_loss_atmos = 0.0
        
        val_mae_overall = 0.0
        val_mae_food = 0.0
        val_mae_price = 0.0
        val_mae_atmos = 0.0
        
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
                
                # loss_overall = self.criterion(pred_overall, true_overall)
                loss_factors = self.criterion(pred_factors, true_factors)
                
                loss_food = self.criterion(pred_factors[:, 0], true_factors[:, 0])
                loss_price = self.criterion(pred_factors[:, 1], true_factors[:, 1])
                loss_atmos = self.criterion(pred_factors[:, 2], true_factors[:, 2])
                
                # Tính MAE (Mean Absolute Error)
                # mae_overall = self.mae_criterion(pred_overall, true_overall)
                mae_food = self.mae_criterion(pred_factors[:, 0], true_factors[:, 0])
                mae_price = self.mae_criterion(pred_factors[:, 1], true_factors[:, 1])
                mae_atmos = self.mae_criterion(pred_factors[:, 2], true_factors[:, 2])
                
                loss = loss_factors # self.alpha * loss_overall + (1 - self.alpha) * loss_factors
                val_loss += loss.item()
                # val_loss_overall += loss_overall.item()
                val_loss_food += loss_food.item()
                val_loss_price += loss_price.item()
                val_loss_atmos += loss_atmos.item()
                
                # val_mae_overall += mae_overall.item()
                val_mae_food += mae_food.item()
                val_mae_price += mae_price.item()
                val_mae_atmos += mae_atmos.item()
                
        num_batches = len(self.val_loader)
        return {
            'loss': val_loss / num_batches,
            'mse_overall': val_loss_overall / num_batches,
            'mse_food': val_loss_food / num_batches,
            'mse_price': val_loss_price / num_batches,
            'mse_atmos': val_loss_atmos / num_batches,
            'mae_overall': val_mae_overall / num_batches,
            'mae_food': val_mae_food / num_batches,
            'mae_price': val_mae_price / num_batches,
            'mae_atmos': val_mae_atmos / num_batches
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
            print(f"  -> RMSE | Overall: {metrics['mse_overall']**0.5:.4f} | Food: {metrics['mse_food']**0.5:.4f} | Price: {metrics['mse_price']**0.5:.4f} | Atmos: {metrics['mse_atmos']**0.5:.4f}")
            print(f"  -> MAE  | Overall: {metrics['mae_overall']:.4f} | Food: {metrics['mae_food']:.4f} | Price: {metrics['mae_price']:.4f} | Atmos: {metrics['mae_atmos']:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_file)
                print(f"*** Saved best model to {save_file} ***\n")
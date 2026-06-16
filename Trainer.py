import torch
import torch.nn as nn
from tqdm import tqdm
import os
from transformers import get_cosine_schedule_with_warmup

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        
        self.criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        
        # Setup Optimizer
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        
        # Setup Gradient Accumulation
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        
        # Setup LR Scheduler (Cosine with Warmup)
        total_steps = (len(train_loader) // self.grad_accum_steps) * args.epochs
        warmup_steps = int(getattr(args, 'warmup_ratio', 0.1) * total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

    def _prepare_inputs(self, batch):
        """Helper để tự động trích xuất đúng các tham số mà model cần, loại bỏ if-else"""
        if self.args.mode == 'train_text':
            keys = ['input_ids', 'attention_mask']
        elif self.args.mode == 'train_image':
            keys = ['pixel_values', 'num_images']
        else:
            keys = ['input_ids', 'attention_mask', 'pixel_values', 'num_images']
            
        return {k: batch[k].to(self.device) for k in keys if k in batch}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(loop):
            inputs = self._prepare_inputs(batch)
            output = self.model(**inputs)
            
            # TextModel và ImageModel trả về (factor_scores, features), FusionModel trả về factor_scores
            pred_factors = output[0] if isinstance(output, tuple) else output
            true_factors = batch['factor_scores'].to(self.device)
            
            loss = self.criterion(pred_factors, true_factors)
            
            # Gradient Accumulation
            loss = loss / self.grad_accum_steps
            loss.backward()
            
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.grad_accum_steps
            loop.set_postfix(loss=loss.item() * self.grad_accum_steps, lr=self.scheduler.get_last_lr()[0])
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_loss_factors = [0.0] * 5
        val_mae_factors = [0.0] * 5
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = self._prepare_inputs(batch)
                output = self.model(**inputs)
                
                pred_factors = output[0] if isinstance(output, tuple) else output
                true_factors = batch['factor_scores'].to(self.device)
                
                loss = self.criterion(pred_factors, true_factors)
                val_loss += loss.item()
                
                for i in range(5):
                    val_loss_factors[i] += self.criterion(pred_factors[:, i], true_factors[:, i]).item()
                    val_mae_factors[i] += self.mae_criterion(pred_factors[:, i], true_factors[:, i]).item()
                
        num_batches = len(self.val_loader)
        metrics = {'loss': val_loss / num_batches}
        factor_names = ['food', 'price', 'atmos', 'service', 'overall']
        
        for i, name in enumerate(factor_names):
            metrics[f'mse_{name}'] = val_loss_factors[i] / num_batches
            metrics[f'mae_{name}'] = val_mae_factors[i] / num_batches
            
        return metrics

    def run(self):
        best_val_loss = float('inf')
        patience_counter = 0
        patience = getattr(self.args, 'patience', 3)
        
        os.makedirs(self.args.save_path, exist_ok=True)
        save_file = os.path.join(self.args.save_path, f'best_model_{self.args.mode}.pth')

        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            metrics = self.validate()
            val_loss = metrics['loss']
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  -> RMSE | Food: {metrics['mse_food']**0.5:.4f} | Price: {metrics['mse_price']**0.5:.4f} | Atmos: {metrics['mse_atmos']**0.5:.4f} | Service: {metrics['mse_service']**0.5:.4f} | Overall: {metrics['mse_overall']**0.5:.4f}")
            print(f"  -> MAE  | Food: {metrics['mae_food']:.4f} | Price: {metrics['mae_price']:.4f} | Atmos: {metrics['mae_atmos']:.4f} | Service: {metrics['mae_service']:.4f} | Overall: {metrics['mae_overall']:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_file)
                print(f"*** Saved best model to {save_file} ***\n")
            else:
                patience_counter += 1
                print(f"--- Early Stopping Patience: {patience_counter}/{patience} ---\n")
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs!")
                    break
import os
import json
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class HomoscedasticUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, targets):
        mse = self.mse(preds, targets)
        precision = torch.exp(-self.log_vars)
        loss = precision * mse + self.log_vars
        return loss.mean()


class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args

        # ---- loss ----
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        loss_fn_str = getattr(args, 'loss_fn', 'mse')
        if loss_fn_str == 'huber':
            self.criterion = nn.HuberLoss()
        elif loss_fn_str == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif loss_fn_str == 'logcosh':
            self.criterion = LogCoshLoss()
        elif loss_fn_str == 'auto_weight':
            self.criterion = HomoscedasticUncertaintyLoss(num_tasks=5).to(device)
        else:
            self.criterion = nn.MSELoss()

        # ---- optimizer ----
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if loss_fn_str == 'auto_weight':
            params += list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay
        )

        # ---- scheduler ----
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        total_steps = (len(train_loader) // self.grad_accum_steps) * args.epochs
        warmup_steps = int(getattr(args, 'warmup_ratio', 0.1) * total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ---- AMP ----
        self.use_amp = getattr(args, 'use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ---- state defaults ----
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_mean_mae = float('inf')

        # ---- resume ----
        resume_path = getattr(args, 'resume', None)
        if resume_path and os.path.isfile(resume_path):
            print(f"[Resume] Loading checkpoint from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.best_mean_mae = ckpt.get('best_mean_mae', float('inf'))
            print(f"[Resume] Resuming from epoch {self.start_epoch} "
                  f"(best_mean_mae={self.best_mean_mae:.4f})")

    # ------------------------------------------------------------------
    def _prepare_inputs(self, batch):
        if self.args.mode == 'train_text':
            keys = ['input_ids', 'attention_mask']
        elif self.args.mode == 'train_image':
            keys = ['pixel_values', 'num_images']
        else:
            keys = ['input_ids', 'attention_mask', 'pixel_values', 'num_images']
        return {k: batch[k].to(self.device) for k in keys if k in batch}

    # ------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for step, batch in enumerate(loop):
            inputs = self._prepare_inputs(batch)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(**inputs)
                pred_factors = output[0] if isinstance(output, tuple) else output
                true_factors = batch['factor_scores'].to(self.device)
                loss = self.criterion(pred_factors, true_factors) / self.grad_accum_steps
            self.scaler.scale(loss).backward()
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            total_loss += loss.item() * self.grad_accum_steps
            loop.set_postfix(
                loss=loss.item() * self.grad_accum_steps,
                lr=self.scheduler.get_last_lr()[0],
            )
        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------
    def validate(self, return_predictions=False):
        self.model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0
        factor_names = ['food', 'price', 'atmos', 'service', 'overall']
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = self._prepare_inputs(batch)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = self.model(**inputs)
                    pred_factors = output[0] if isinstance(output, tuple) else output
                    true_factors = batch['factor_scores'].to(self.device)
                    loss = self.criterion(pred_factors, true_factors)
                val_loss += loss.item()
                all_preds.append(pred_factors.cpu())
                all_targets.append(true_factors.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()    # [N, 5]
        all_targets = torch.cat(all_targets, dim=0).numpy()  # [N, 5]

        metrics = {'loss': val_loss / len(self.val_loader)}
        mae_list = []
        for i, name in enumerate(factor_names):
            mae = float(np.mean(np.abs(all_preds[:, i] - all_targets[:, i])))
            rmse = float(np.sqrt(np.mean((all_preds[:, i] - all_targets[:, i]) ** 2)))
            ss_res = np.sum((all_targets[:, i] - all_preds[:, i]) ** 2)
            ss_tot = np.sum((all_targets[:, i] - np.mean(all_targets[:, i])) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            metrics[f'mae_{name}'] = mae
            metrics[f'rmse_{name}'] = rmse
            metrics[f'r2_{name}'] = r2
            mae_list.append(mae)

        metrics['mean_mae'] = float(np.mean(mae_list))
        metrics['overall_mae'] = metrics['mae_overall']
        aspect_names = ['food', 'price', 'atmos', 'service']
        metrics['aspect_mae'] = float(np.mean([metrics[f'mae_{n}'] for n in aspect_names]))

        if return_predictions:
            return metrics, all_preds, all_targets
        return metrics

    # ------------------------------------------------------------------
    def run(self):
        args = self.args
        factor_names = ['food', 'price', 'atmos', 'service', 'overall']

        # ---- experiment directory ----
        exp_dir = os.path.join(
            getattr(args, 'exp_dir', './experiments'),
            getattr(args, 'exp_id', 'EXP_000'),
        )
        os.makedirs(exp_dir, exist_ok=True)
        # Keep save_path for backward compat
        os.makedirs(args.save_path, exist_ok=True)

        checkpoint_path = os.path.join(exp_dir, f'best_model_{args.mode}.pth')

        # ---- logging (console + file) ----
        log_path = os.path.join(exp_dir, 'train.log')
        logger = logging.getLogger('trainer')
        logger.setLevel(logging.INFO)
        # avoid duplicate handlers on re-use
        if not logger.handlers:
            fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            fh.setFormatter(fmt)
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(fh)
            logger.addHandler(ch)

        def log(msg):
            logger.info(msg)

        log(f"====== START: {datetime.datetime.now().isoformat()} ======")
        log(f"Experiment: {getattr(args, 'exp_id', 'EXP_000')} | Mode: {args.mode} | exp_dir: {exp_dir}")

        # ---- save config ----
        config_path = os.path.join(exp_dir, 'config.yaml')
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)
        except ImportError:
            config_path = os.path.join(exp_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(vars(args), f, indent=2)

        patience = getattr(args, 'patience', 3)
        patience_counter = 0

        for epoch in range(self.start_epoch, args.epochs):
            train_loss = self.train_epoch(epoch)
            metrics = self.validate()
            val_loss = metrics['loss']
            mean_mae = metrics['mean_mae']

            log(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Mean MAE: {mean_mae:.4f}"
            )
            log(
                f"  -> MAE  | "
                + " | ".join(f"{n.capitalize()}: {metrics[f'mae_{n}']:.4f}" for n in factor_names)
            )
            log(
                f"  -> RMSE | "
                + " | ".join(f"{n.capitalize()}: {metrics[f'rmse_{n}']:.4f}" for n in factor_names)
            )
            log(
                f"  -> R2   | "
                + " | ".join(f"{n.capitalize()}: {metrics[f'r2_{n}']:.4f}" for n in factor_names)
            )

            # ---- best model selection by mean_mae ----
            if mean_mae < self.best_mean_mae:
                self.best_mean_mae = mean_mae
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_val_loss': self.best_val_loss,
                        'best_mean_mae': self.best_mean_mae,
                        'args': vars(args),
                    },
                    checkpoint_path,
                )
                
                # Backward compat: also save to args.save_path
                import shutil
                compat_path = os.path.join(args.save_path, f'best_model_{args.mode}.pth')
                shutil.copy(checkpoint_path, compat_path)
                
                log(f"*** Saved best model to {checkpoint_path} (mean_mae={mean_mae:.4f}) ***")
            else:
                patience_counter += 1
                log(f"No improvement for {patience_counter}/{patience} epoch(s).")
                if patience_counter >= patience:
                    log(f"Early stopping triggered after {epoch+1} epochs!")
                    break

        # ---- final evaluation with best checkpoint ----
        log("====== Final evaluation with best checkpoint ======")
        if os.path.isfile(checkpoint_path):
            best_ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(best_ckpt['model_state_dict'])
            log(f"Loaded best checkpoint from {checkpoint_path}")

        final_metrics, final_preds, final_targets = self.validate(return_predictions=True)
        log(f"Final Val Loss: {final_metrics['loss']:.4f} | Mean MAE: {final_metrics['mean_mae']:.4f}")

        # ---- save metrics.json ----
        metrics_path = os.path.join(exp_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        log(f"Metrics saved to {metrics_path}")

        # ---- save predictions.csv ----
        import csv
        pred_path = os.path.join(exp_dir, 'predictions.csv')
        header = ['index', 'split']
        for n in factor_names:
            header.append(f'y_true_{n}')
        for n in factor_names:
            header.append(f'y_pred_{n}')
        for n in factor_names:
            header.append(f'absolute_error_{n}')

        with open(pred_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx in range(len(final_preds)):
                row = [idx, 'val']
                row += [f'{final_targets[idx, j]:.6f}' for j in range(5)]
                row += [f'{final_preds[idx, j]:.6f}' for j in range(5)]
                row += [f'{abs(final_targets[idx, j] - final_preds[idx, j]):.6f}' for j in range(5)]
                writer.writerow(row)
        log(f"Predictions saved to {pred_path}")
        log(f"====== END: {datetime.datetime.now().isoformat()} ======")
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor

from Config import get_args
from src.dataset import MultimodalDataset
from Models.TextModel import TextModel
from Models.ImageModel import ImageModel
from Models.FusionModel import FusionModel
import os
import json


class TimmProcessor:
    def __init__(self, model_name):
        import timm
        data_config = timm.data.resolve_model_data_config(model_name)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    def __call__(self, images, return_tensors="pt"):
        import torch
        pixel_values = torch.stack([self.transform(img.convert('RGB')) for img in images])
        return {'pixel_values': pixel_values}

def load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"Loaded weights: {path}")


def test():
    args = get_args()
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    print(f"====== TESTING: {args.mode.upper()} ======")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    try:
        image_processor = AutoImageProcessor.from_pretrained(args.image_model_name)
    except Exception:
        if "google/siglip" in args.image_model_name:
            image_processor = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-256')
        else:
            image_processor = TimmProcessor(args.image_model_name)

    test_dataset = MultimodalDataset(
        args.test_path, tokenizer, image_processor,
        max_length=args.max_length, image_dir=args.image_dir,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    # ── Build model ────────────────────────────────────────────────────────────
    fusion_type = getattr(args, 'fusion_type', 'concat')

    if args.mode == 'train_text':
        model = TextModel(model_name=args.text_model_name)
        weight_path = os.path.join(args.save_path, 'best_model_train_text.pth')

    elif args.mode == 'train_image':
        model = ImageModel(model_name=args.image_model_name)
        weight_path = os.path.join(args.save_path, 'best_model_train_image.pth')

    else:  # train_fusion
        text_model  = TextModel(model_name=args.text_model_name)
        image_model = ImageModel(model_name=args.image_model_name)

        fusion_kwargs = dict(text_model=text_model, image_model=image_model)
        if fusion_type == 'gmu':
            from Models.GMUFusion import GMUFusion
            model = GMUFusion(**fusion_kwargs)
        elif fusion_type == 'gated_cross':
            from Models.GatedCrossModalFusion import GatedCrossModalFusion
            model = GatedCrossModalFusion(**fusion_kwargs)
        elif fusion_type == 'film':
            from Models.FiLMFusion import FiLMFusion
            model = FiLMFusion(**fusion_kwargs)
        elif fusion_type == 'cross_attention':
            from Models.CrossAttentionFusion import CrossAttentionFusion
            model = CrossAttentionFusion(**fusion_kwargs)
        else:
            model = FusionModel(**fusion_kwargs)

        weight_path = os.path.join(args.save_path, 'best_model_train_fusion.pth')

    if not os.path.exists(weight_path):
        print(f"ERROR: checkpoint not found at {weight_path}")
        return

    load_ckpt(model, weight_path, device)
    model.to(device)
    model.eval()

    # ── Inference ──────────────────────────────────────────────────────────────
    all_preds   = []
    all_targets = []

    use_amp = getattr(args, 'use_amp', False)

    with torch.no_grad():
        for batch in test_loader:
            if args.mode == 'train_text':
                inputs = {k: batch[k].to(device) for k in ['input_ids', 'attention_mask']}
            elif args.mode == 'train_image':
                inputs = {k: batch[k].to(device) for k in ['pixel_values', 'num_images']}
            else:
                inputs = {k: batch[k].to(device) for k in ['input_ids', 'attention_mask', 'pixel_values', 'num_images'] if k in batch}

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(**inputs)
                preds  = output[0] if isinstance(output, tuple) else output

            all_preds.append(preds.cpu())
            all_targets.append(batch['factor_scores'])

    all_preds   = torch.cat(all_preds,   dim=0).numpy()  # (N, 5)
    all_targets = torch.cat(all_targets, dim=0).numpy()  # (N, 5)

    # ── Sample-wise metrics ────────────────────────────────────────────────────
    factor_names = ['food', 'price', 'atmos', 'service', 'overall']
    metrics = {}
    mae_list = []

    for i, name in enumerate(factor_names):
        mae  = float(np.mean(np.abs(all_preds[:, i] - all_targets[:, i])))
        rmse = float(np.sqrt(np.mean((all_preds[:, i] - all_targets[:, i]) ** 2)))
        ss_res = np.sum((all_targets[:, i] - all_preds[:, i]) ** 2)
        ss_tot = np.sum((all_targets[:, i] - np.mean(all_targets[:, i])) ** 2)
        r2   = float(1 - ss_res / (ss_tot + 1e-10))
        metrics[f'mae_{name}']  = mae
        metrics[f'rmse_{name}'] = rmse
        metrics[f'r2_{name}']   = r2
        mae_list.append(mae)

    metrics['mean_mae']    = float(np.mean(mae_list))
    metrics['overall_mae'] = metrics['mae_overall']
    metrics['aspect_mae']  = float(np.mean([metrics[f'mae_{n}'] for n in ['food', 'price', 'atmos', 'service']]))

    # ── Print ──────────────────────────────────────────────────────────────────
    print(f"\n=== TEST SET RESULTS ===")
    print(f"             MAE      RMSE      R2")
    for name in factor_names:
        print(f"  {name:<8} : {metrics[f'mae_{name}']:.4f}   {metrics[f'rmse_{name}']:.4f}   {metrics[f'r2_{name}']:.4f}")
    print()
    print(f"  mean_mae   : {metrics['mean_mae']:.4f}")
    print(f"  aspect_mae : {metrics['aspect_mae']:.4f}")
    print(f"  overall_mae: {metrics['overall_mae']:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    exp_dir = os.path.join(getattr(args, 'exp_dir', './experiments'), args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # ---- 1. Save Metrics ----
    out_path = os.path.join(exp_dir, 'test_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved test metrics to {out_path}")

    # ---- 2. Save Predictions CSV ----
    import csv
    pred_path = os.path.join(exp_dir, 'test_predictions.csv')
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
        for idx in range(len(all_preds)):
            row = [idx, 'test']
            row += [f'{all_targets[idx, j]:.6f}' for j in range(5)]
            row += [f'{all_preds[idx, j]:.6f}' for j in range(5)]
            row += [f'{abs(all_targets[idx, j] - all_preds[idx, j]):.6f}' for j in range(5)]
            writer.writerow(row)
    print(f"Saved test predictions to {pred_path}")

    # ---- 3. Save Plots ----
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        # Plot 1: Bar chart of MAE per aspect
        plt.figure(figsize=(8, 5))
        maes = [metrics[f'mae_{n}'] for n in factor_names]
        sns.barplot(x=[n.capitalize() for n in factor_names], y=maes, palette='viridis')
        plt.title('Test Set - Mean Absolute Error (MAE) per Aspect')
        plt.ylabel('MAE')
        for i, v in enumerate(maes):
            plt.text(i, v + 0.02, f"{v:.3f}", color='black', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'test_mae_aspects.png'), dpi=150)
        plt.close()
        
        # Plot 2: Scatter plots of Pred vs True for each aspect
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for i, n in enumerate(factor_names):
            axes[i].scatter(all_targets[:, i], all_preds[:, i], alpha=0.3, color='steelblue')
            axes[i].plot([1, 5], [1, 5], 'r--', linewidth=2)  # Perfect prediction line
            axes[i].set_title(f'{n.capitalize()} (R²: {metrics[f"r2_{n}"]:.2f})', fontsize=14)
            axes[i].set_xlabel('True Score')
            axes[i].set_ylabel('Predicted Score')
            axes[i].set_xlim(0.5, 5.5)
            axes[i].set_ylim(0.5, 5.5)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'test_scatter_pred_vs_true.png'), dpi=150)
        plt.close()
        
        # Plot 3: Histogram of Errors
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for i, n in enumerate(factor_names):
            errors = all_preds[:, i] - all_targets[:, i]
            sns.histplot(errors, bins=20, kde=True, ax=axes[i], color='mediumpurple')
            axes[i].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[i].set_title(f'{n.capitalize()} Error Dist', fontsize=14)
            axes[i].set_xlabel('Prediction Error (Pred - True)')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'test_error_distributions.png'), dpi=150)
        plt.close()
        
        print(f"Saved test visualizations as PNGs in {exp_dir}")
    except ImportError:
        print("matplotlib or seaborn not installed, skipping visualizations. Install with: pip install matplotlib seaborn")

if __name__ == '__main__':
    test()

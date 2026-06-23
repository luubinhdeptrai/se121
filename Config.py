import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Multimodal Food Review Score Prediction")
    parser.add_argument('--mode', type=str, default='train_text', choices=['train_text', 'train_image', 'train_fusion'])
    parser.add_argument('--train_path', type=str, default='./data/text/train.csv')
    parser.add_argument('--val_path', type=str, default='./data/text/val.csv')
    parser.add_argument('--test_path', type=str, default='./data/text/test.csv')
    parser.add_argument('--image_dir', type=str, default='./data/image')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--text_model_name', type=str, default='xlm-roberta-base')
    parser.add_argument('--image_model_name', type=str, default='convnext_base_in22k')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--loss_fn', type=str, default='mse',
                        choices=['mse', 'huber', 'smoothl1', 'logcosh', 'auto_weight'])
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--unfreeze_text_layers', type=int, default=0)
    parser.add_argument('--unfreeze_image_layers', type=int, default=0)
    # New P0/P1 args
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--exp_id', type=str, default='EXP_000',
                        help='Experiment ID, used as the output folder name')
    parser.add_argument('--exp_dir', type=str, default='./experiments',
                        help='Root directory for all experiment outputs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint .pth file to resume training from')
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable Automatic Mixed Precision (AMP) training')
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'gmu', 'gated_cross', 'film', 'cross_attention'],
                        help='Fusion architecture: concat | gmu | gated_cross | film | cross_attention')
    args = parser.parse_args()
    return args
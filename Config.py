import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Multimodal Food Review Score Prediction")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train_text', 
                        choices=['train_text', 'train_image', 'train_fusion'],
                        help="Chọn chế độ train: text, image, hoặc fusion")
    
    # File paths
    parser.add_argument('--train_path', type=str, default='/kaggle/input/datasets/lchhong/multimodal/data/text/train.csv', help='Đường dẫn file CSV train')
    parser.add_argument('--val_path', type=str, default='/kaggle/input/datasets/lchhong/multimodal/data/text/val.csv', help='Đường dẫn file CSV val')
    parser.add_argument('--test_path', type=str, default='/kaggle/input/datasets/lchhong/multimodal/data/text/test.csv', help='Đường dẫn file CSV test')
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/datasets/lchhong/multimodal/data/image', help='Thư mục chứa ảnh đã tải')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Thư mục lưu model')
    
    # Model configuration
    parser.add_argument('--text_model_name', type=str, default='xlm-roberta-base', help='HuggingFace Text Model')
    parser.add_argument('--image_model_name', type=str, default='convnext_base_in22k', help='Timm Image Model')
    parser.add_argument('--max_length', type=int, default=128, help='Chiều dài tối đa chuỗi token text')
    parser.add_argument('--num_factors', type=int, default=3, help='Số lượng factor scores (ví dụ: food, price, atmosphere)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Số epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay cho AdamW')
    parser.add_argument('--alpha', type=float, default=0.5, help='Trọng số loss giữa Overall (alpha) và Factors (1-alpha)')
    
    args = parser.parse_args()
    return args
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
import torch.nn as nn

from Config import get_args
from src.dataset import MultimodalDataset
from Models.TextModel import TextModel
from Models.ImageModel import ImageModel
from Models.FusionModel import FusionModel
import os

def test():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    print(f"====== TESTING MODE: {args.mode.upper()} ======")
    
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    try:
        image_processor = AutoImageProcessor.from_pretrained(args.image_model_name)
    except:
        image_processor = AutoImageProcessor.from_pretrained('facebook/convnext-base-224-22k')

    print("Loading Test Dataset...")
    test_dataset = MultimodalDataset(args.test_path, tokenizer, image_processor, args.max_length, args.image_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Đã nạp {len(test_dataset)} mẫu Test")

    if args.mode == 'train_text':
        model = TextModel(model_name=args.text_model_name, num_factors=args.num_factors)
        weight_path = os.path.join(args.save_path, 'best_model_train_text.pth')
    elif args.mode == 'train_image':
        model = ImageModel(model_name=args.image_model_name, num_factors=args.num_factors)
        weight_path = os.path.join(args.save_path, 'best_model_train_image.pth')
    else:
        text_model = TextModel(model_name=args.text_model_name, num_factors=args.num_factors)
        image_model = ImageModel(model_name=args.image_model_name, num_factors=args.num_factors)
        model = FusionModel(text_model, image_model, num_factors=args.num_factors)
        weight_path = os.path.join(args.save_path, 'best_model_train_fusion.pth')

    if not os.path.exists(weight_path):
        print(f"LỖI: Không tìm thấy file trọng số {weight_path}. Bạn cần train mô hình này trước!")
        return

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    test_loss_overall = 0.0
    test_loss_factors = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            if args.mode == 'train_text':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pred_overall, pred_factors, _ = model(input_ids, attention_mask)
            elif args.mode == 'train_image':
                pixel_values = batch['pixel_values'].to(device)
                pred_overall, pred_factors, _ = model(pixel_values)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                pred_overall, pred_factors = model(input_ids, attention_mask, pixel_values)
            
            true_overall = batch['overall_score'].to(device)
            true_factors = batch['factor_scores'].to(device)
            
            test_loss_overall += criterion(pred_overall, true_overall).item()
            test_loss_factors += criterion(pred_factors, true_factors).item()
            
    mse_overall = test_loss_overall / len(test_loader)
    mse_factors = test_loss_factors / len(test_loader)
    
    print("\n[KẾT QUẢ TEST TRÊN TẬP DỮ LIỆU ĐỘC LẬP]")
    print(f"MSE Overall Score: {mse_overall:.4f}")
    print(f"MSE Factor Scores: {mse_factors:.4f}")
    print(f"Sai số tuyệt đối trung bình (MAE) ước lượng: {mse_overall**0.5:.4f} điểm / 10 điểm")

if __name__ == '__main__':
    test()

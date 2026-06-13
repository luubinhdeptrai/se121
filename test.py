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
    mae_criterion = nn.L1Loss()
    
    test_loss_overall = 0.0
    test_loss_factors = 0.0
    
    # Biến lưu loss từng phần
    test_loss_food = 0.0
    test_loss_price = 0.0
    test_loss_atmosphere = 0.0
    
    test_mae_overall = 0.0
    test_mae_food = 0.0
    test_mae_price = 0.0
    test_mae_atmosphere = 0.0
    
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
            
            # test_loss_overall += criterion(pred_overall, true_overall).item()
            test_loss_factors += criterion(pred_factors, true_factors).item()
            
            # Tính riêng từng tiêu chí cho MSE
            test_loss_food += criterion(pred_factors[:, 0], true_factors[:, 0]).item()
            test_loss_price += criterion(pred_factors[:, 1], true_factors[:, 1]).item()
            test_loss_atmosphere += criterion(pred_factors[:, 2], true_factors[:, 2]).item()
            
            # Tính MAE
            # test_mae_overall += mae_criterion(pred_overall, true_overall).item()
            test_mae_food += mae_criterion(pred_factors[:, 0], true_factors[:, 0]).item()
            test_mae_price += mae_criterion(pred_factors[:, 1], true_factors[:, 1]).item()
            test_mae_atmosphere += mae_criterion(pred_factors[:, 2], true_factors[:, 2]).item()
            
    mse_overall = test_loss_overall / len(test_loader)
    mse_factors = test_loss_factors / len(test_loader)
    mse_food = test_loss_food / len(test_loader)
    mse_price = test_loss_price / len(test_loader)
    mse_atmosphere = test_loss_atmosphere / len(test_loader)
    
    mae_overall = test_mae_overall / len(test_loader)
    mae_food = test_mae_food / len(test_loader)
    mae_price = test_mae_price / len(test_loader)
    mae_atmosphere = test_mae_atmosphere / len(test_loader)
    
    print("\n[EVALUATION METRICS ON INDEPENDENT TEST SET]")
    print(f"MSE  (Overall): {mse_overall:.4f} | (Food): {mse_food:.4f} | (Price): {mse_price:.4f} | (Atmos): {mse_atmosphere:.4f}")
    print(f"RMSE (Overall): {mse_overall**0.5:.4f} | (Food): {mse_food**0.5:.4f} | (Price): {mse_price**0.5:.4f} | (Atmos): {mse_atmosphere**0.5:.4f}")
    print(f"MAE  (Overall): {mae_overall:.4f} | (Food): {mae_food:.4f} | (Price): {mae_price:.4f} | (Atmos): {mae_atmosphere:.4f}")

if __name__ == '__main__':
    test()

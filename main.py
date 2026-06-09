import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoImageProcessor

from Config import get_args
from src.dataset import MultimodalDataset
from Models.TextModel import TextModel
from Models.ImageModel import ImageModel
from Models.FusionModel import FusionModel
from Trainer import Trainer
import os

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"====== MODE: {args.mode.upper()} ======")
    print(f"Using device: {device}")

    # Processors
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained('facebook/convnext-base-224-22k')

    # Data
    print("Loading Dataset...")
    dataset = MultimodalDataset(args.data_path, tokenizer, image_processor, args.max_length)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model Initialization
    print("Initializing Model...")
    if args.mode == 'train_text':
        model = TextModel(model_name=args.text_model_name, num_factors=args.num_factors)
    elif args.mode == 'train_image':
        model = ImageModel(model_name=args.image_model_name, num_factors=args.num_factors)
    elif args.mode == 'train_fusion':
        # Khởi tạo và load trọng số đã train
        text_model = TextModel(model_name=args.text_model_name, num_factors=args.num_factors)
        image_model = ImageModel(model_name=args.image_model_name, num_factors=args.num_factors)
        
        text_weights = os.path.join(args.save_path, 'best_model_train_text.pth')
        image_weights = os.path.join(args.save_path, 'best_model_train_image.pth')
        
        if os.path.exists(text_weights):
            text_model.load_state_dict(torch.load(text_weights, map_location=device))
            print(f"Loaded Text Model weights from {text_weights}")
        else:
            print("WARNING: Text Model weights not found! Training fusion with untrained text features.")
            
        if os.path.exists(image_weights):
            image_model.load_state_dict(torch.load(image_weights, map_location=device))
            print(f"Loaded Image Model weights from {image_weights}")
        else:
            print("WARNING: Image Model weights not found! Training fusion with untrained image features.")
            
        model = FusionModel(text_model, image_model, num_factors=args.num_factors)

    model.to(device)

    # Training
    trainer = Trainer(model, train_loader, val_loader, device, args)
    trainer.run()

if __name__ == '__main__':
    main()
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
from Config import get_args
from src.dataset import MultimodalDataset
from Models.TextModel import TextModel
from Models.ImageModel import ImageModel
from Models.FusionModel import FusionModel
from Trainer import Trainer
import os


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    args = get_args()

    # ---- reproducibility ----
    set_seed(args.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    print(f"====== MODE: {args.mode.upper()} ======")
    print(f"Using device: {device}")
    print(f"Seed: {args.seed} | Experiment: {args.exp_id}")

    # ---- tokenizer / image processor ----
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    try:
        image_processor = AutoImageProcessor.from_pretrained(args.image_model_name)
    except Exception:
        image_processor = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-256')

    # ---- datasets ----
    train_dataset = MultimodalDataset(
        args.train_path, tokenizer, image_processor,
        max_length=args.max_length, image_dir=args.image_dir,
    )
    val_dataset = MultimodalDataset(
        args.val_path, tokenizer, image_processor,
        max_length=args.max_length, image_dir=args.image_dir,
    )

    # ---- reproducible data loaders ----
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # ---- model ----
    if args.mode == 'train_text':
        model = TextModel(model_name=args.text_model_name)
    elif args.mode == 'train_image':
        model = ImageModel(model_name=args.image_model_name)
    elif args.mode == 'train_fusion':
        text_model = TextModel(model_name=args.text_model_name)
        image_model = ImageModel(model_name=args.image_model_name)
        text_weights = os.path.join(args.save_path, 'best_model_train_text.pth')
        image_weights = os.path.join(args.save_path, 'best_model_train_image.pth')

        if os.path.exists(text_weights):
            ckpt = torch.load(text_weights, map_location=device)
            text_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        if os.path.exists(image_weights):
            ckpt = torch.load(image_weights, map_location=device)
            image_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

        fusion_kwargs = dict(
            text_model=text_model,
            image_model=image_model,
            unfreeze_text_layers=args.unfreeze_text_layers,
            unfreeze_image_layers=args.unfreeze_image_layers,
        )
        fusion_type = getattr(args, 'fusion_type', 'concat')
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
        else:  # concat (default)
            model = FusionModel(**fusion_kwargs)

    model.to(device)
    trainer = Trainer(model, train_loader, val_loader, device, args)
    trainer.run()


if __name__ == '__main__':
    main()
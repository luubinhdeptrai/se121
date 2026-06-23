import torch
import torch.nn as nn
import traceback

from Models.FusionModel import FusionModel
from Models.GMUFusion import GMUFusion
from Models.GatedCrossModalFusion import GatedCrossModalFusion
from Models.FiLMFusion import FiLMFusion
from Models.CrossAttentionFusion import CrossAttentionFusion

class MockConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

class MockEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_features=1024):
        super().__init__()
        self.config = MockConfig(hidden_size)
        self.num_features = num_features

class MockTextModel(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.encoder = MockEncoder(hidden_size=hidden_size)
        self.dummy_param = nn.Parameter(torch.zeros(1))
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        # TextModel returns _, text_features (shape: [B, hidden_size])
        return None, torch.randn(batch_size, self.encoder.config.hidden_size)

class MockImageModel(nn.Module):
    def __init__(self, num_features=1024):
        super().__init__()
        self.encoder = MockEncoder(num_features=num_features)
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
    def forward(self, pixel_values, num_images=None):
        batch_size = pixel_values.shape[0]
        # ImageModel returns _, image_features (shape: [B, num_features])
        return None, torch.randn(batch_size, self.encoder.num_features)

def test_fusions():
    print("="*50)
    print("🚀 BẮT ĐẦU UNIT TEST 5 KIẾN TRÚC FUSION 🚀")
    print("="*50)
    
    text_model = MockTextModel(hidden_size=768)
    image_model = MockImageModel(num_features=1024)
    
    models = {
        "1. Basic Concat Fusion": FusionModel,
        "2. GMU Fusion": GMUFusion,
        "3. Gated Cross-Modal Fusion": GatedCrossModalFusion,
        "4. FiLM Fusion": FiLMFusion,
        "5. Cross-Attention Fusion": CrossAttentionFusion
    }
    
    batch_size = 4
    seq_len = 128
    
    # Tạo dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    pixel_values = torch.randn(batch_size, 3, 3, 224, 224) 
    num_images = torch.tensor([3, 2, 1, 3])
    
    passed_count = 0
    for name, ModelClass in models.items():
        print(f"\\nĐang test {name}...")
        try:
            # 1. Khởi tạo model
            model = ModelClass(text_model, image_model, num_factors=5, unfreeze_text_layers=0, unfreeze_image_layers=0)
            
            # 2. Chạy Forward Pass
            output = model(input_ids, attention_mask, pixel_values, num_images)
            
            # 3. Check Shape
            if output.shape == (batch_size, 5):
                print(f"  ✅ [PASS] Đầu ra chuẩn: {output.shape}")
            else:
                print(f"  ❌ [FAIL] Sai chiều. Expected {(batch_size, 5)}, got {output.shape}")
                continue
                
            # 4. Check Backward Pass (xem có lỗi autograd không)
            loss = output.sum()
            loss.backward()
            print(f"  ✅ [PASS] Lan truyền ngược (Backward) thành công.")
            passed_count += 1
            
        except Exception as e:
            print(f"  ❌ [CRASH] Lỗi Runtime:")
            traceback.print_exc()

    print("\\n" + "="*50)
    print(f"🎯 KẾT QUẢ: {passed_count}/{len(models)} KIẾN TRÚC VƯỢT QUA BÀI TEST!")
    print("="*50)

if __name__ == "__main__":
    test_fusions()

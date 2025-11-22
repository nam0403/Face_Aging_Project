import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os

# Import các module tự viết
from utils import SiameseWrapper, ContrastiveLoss, CACDPairDataset, train_one_epoch
from models.simple_cnn import SimpleCNNBackbone
from models.facenet_model import FaceNetBackbone
from models.arcface_model import ArcFaceResNetBackbone

# --- CẤU HÌNH ---
CONFIG = {
    "data_dir": "data/CACD_Cropped_112", # <--- SỬA ĐƯỜNG DẪN CỦA BẠN
    "img_size": 112,
    "embedding_size": 512,
    "batch_size": 32, # Tăng lên 64 nếu GPU mạnh
    "epochs": 20,
    "lr": 0.001,
    "margin": 1.0,
    "pairs_per_epoch": 5000 # Số lượng cặp train mỗi epoch
}

def get_model(model_name):
    """Factory function để lấy model dựa trên tên"""
    if model_name == 'simple_cnn':
        print("Đang khởi tạo: Simple CNN")
        backbone = SimpleCNNBackbone(CONFIG['embedding_size'])
    elif model_name == 'facenet':
        print("Đang khởi tạo: FaceNet (InceptionResnetV1)")
        backbone = FaceNetBackbone(CONFIG['embedding_size'])
    elif model_name == 'arcface':
        print("Đang khởi tạo: ArcFace (ResNet50)")
        backbone = ArcFaceResNetBackbone(CONFIG['embedding_size'])
    else:
        raise ValueError(f"Model {model_name} không hỗ trợ.")
    
    # Bọc trong Siamese Wrapper
    return SiameseWrapper(backbone)

def main():
    # 1. Parse tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Train Face Recognition Models for Temporal Robustness")
    parser.add_argument('--model', type=str, required=True, choices=['simple_cnn', 'facenet', 'arcface'], 
                        help='Chọn model để train: simple_cnn, facenet, hoặc arcface')
    parser.add_argument('--subset', type=float, default=0.4, help='Tỷ lệ dữ liệu sử dụng (0.1 - 1.0)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Device: {device}")

    # 2. Prepare Data
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    print("📂 Đang load Dataset...")
    dataset = CACDPairDataset(
        root_dir=CONFIG['data_dir'],
        transform=train_transform,
        subset_ratio=args.subset,
        num_pairs=CONFIG['pairs_per_epoch']
    )
    
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)

    # 3. Prepare Model
    model = get_model(args.model).to(device)

    # 4. Loss & Optimizer
    criterion = ContrastiveLoss(margin=CONFIG['margin'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 5. Training Loop
    print(f"\n🚀 BẮT ĐẦU TRAIN MODEL: {args.model.upper()}")
    print(f"   - Epochs: {CONFIG['epochs']}")
    print(f"   - Batch size: {CONFIG['batch_size']}")
    
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(CONFIG['epochs']):
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, device, epoch+1)
        
        print(f"✨ Epoch {epoch+1}/{CONFIG['epochs']} - Avg Loss: {avg_loss:.4f}")
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(save_dir, f"{args.model}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Đã lưu model: {save_path}")

    print("\n✅ HUẤN LUYỆN HOÀN TẤT!")

if __name__ == '__main__':
    main()
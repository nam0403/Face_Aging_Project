import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np

# Import model & dataset
from models.ada_face import AdaFaceNet
from utils import CACDClassificationDataset_IdentitySplit

# ===================== CẤU HÌNH AN TOÀN TUYỆT ĐỐI =====================
CONFIG = {
    "data_dir": "data/CACD_Cropped_112",
    "weight_path": "weights/InsightFace_Pytorch%2Bmodel_ir_se50.pth",
    "epochs": 20,
    "batch_size": 64,        # Giảm batch xuống vì tắt Autocast sẽ tốn VRAM hơn
    "lr": 1e-4,
    "subset": 0.5
}

# ===================== HÀM FREEZE BN (QUAN TRỌNG) =====================
def freeze_bn(model):
    """Giữ nguyên thống kê Mean/Var của Backbone, không cho update"""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

# ===================== SAFE ADAFACE LOSS (FLOAT32) =====================
class SafeAdaFaceLoss(nn.Module):
    def __init__(self, s=64.0, m=0.4, h=0.333):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.eps = 1e-4
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, cosine, norms, labels):
        # 1. Check Input NaN
        if torch.isnan(cosine).any() or torch.isnan(norms).any():
            print("⚠️ [Loss] Input Cosine or Norms contains NaN!")
            return None, None

        # 2. Adaptive Margin
        safe_norms = torch.clamp(norms, 0.001, 100)
        mean = safe_norms.mean()
        std = safe_norms.std()
        
        margin_scaler = (safe_norms - mean) / (std + self.eps)
        margin_scaler = torch.clamp(margin_scaler * self.h, -1, 1).detach()

        # 3. Cosine -> Theta -> Cosine + Margin
        safe_cosine = torch.clamp(cosine, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(safe_cosine)
        theta += self.m * margin_scaler
        final_cosine = torch.cos(theta)

        # 4. Logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1)

        logits = one_hot * final_cosine + (1-one_hot) * safe_cosine
        logits *= self.s

        loss = self.ce(logits, labels)
        return loss, logits

# ===================== LOAD WEIGHT =====================
def load_pretrained(model, path):
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy file: {path}")
        return
    
    # Kiểm tra kích thước file
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < 100:
        print(f"❌ CẢNH BÁO: File weight chỉ nặng {size_mb:.2f} MB. Đây là file lỗi (Git LFS pointer).")
        print("👉 Vui lòng tải lại file gốc nặng ~113MB.")
        raise RuntimeError("Weight file invalid")

    print(f"📥 Loading backbone ({size_mb:.1f} MB)...")
    try:
        sd = torch.load(path, map_location='cpu')
        new_sd = {k.replace("module.", ""):v for k,v in sd.items()}
        model.backbone.load_state_dict(new_sd, strict=False)
        print("✅ IR-SE50 Backbone loaded.")
    except Exception as e:
        print(f"❌ Error loading: {e}")

# ===================== MAIN =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Device: {device} | Batch: {CONFIG['batch_size']} | Precision: FLOAT32 (No AMP)")

    # 1. Data
    tf = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = CACDClassificationDataset_IdentitySplit(CONFIG['data_dir'], tf, subset_ratio=CONFIG['subset'], mode='train')
    val_ds = CACDClassificationDataset_IdentitySplit(CONFIG['data_dir'], tf, subset_ratio=CONFIG['subset'], mode='val')

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    # 2. Model
    model = AdaFaceNet(len(train_ds.classes)).to(device)
    load_pretrained(model, CONFIG['weight_path'])

    # 3. Setup
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = SafeAdaFaceLoss().to(device)

    best_loss = 999
    os.makedirs("ckpt_final_debug", exist_ok=True)

    # 4. Loop
    for epoch in range(CONFIG['epochs']):
        model.train()
        
        # --- QUAN TRỌNG: LUÔN FREEZE BN ---
        freeze_bn(model)
        
        total, correct, loss_sum = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for img, label in pbar:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            # --- FORWARD (FLOAT32 - NO AUTOCAST) ---
            cosine, norms = model(img, label)
            
            # Debug: Check output model trước khi vào loss
            if torch.isnan(cosine).any() or torch.isinf(cosine).any():
                print("⚠️ Model Output (Cosine) is NaN! Backbone bị lỗi.")
                continue

            loss, logits = criterion(cosine, norms, label)

            if loss is None or torch.isnan(loss):
                print("⚠️ Loss is NaN. Skipping.")
                continue

            # --- BACKWARD ---
            loss.backward()
            
            # Clip Gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()

            # Stats
            with torch.no_grad():
                preds = torch.argmax(logits, 1)
                correct += (preds==label).sum().item()
                total += label.size(0)
                loss_sum += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = loss_sum / len(train_loader) if len(train_loader) > 0 else 0
        avg_acc = 100 * correct / total if total > 0 else 0
        print(f"[Train] Ep {epoch+1} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")

        # Validation
        model.eval()
        vloss = 0
        with torch.no_grad():
            for img,label in val_loader:
                img,label = img.to(device),label.to(device)
                cosine, norms = model(img,label)
                loss, _ = criterion(cosine, norms, label)
                if loss is not None: vloss += loss.item()

        vloss /= len(val_loader)
        print(f"   >> Val Loss: {vloss:.4f}")

        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), "ckpt_final_debug/adaface_best.pth")
            print("   🔥 Best Model Saved")

        scheduler.step()

if __name__ == '__main__':
    main()
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# Models
from models.model_ir_se50 import IR_SE50
from models.facenet_model import FaceNetBackbone
from models.ada_face import AdaFaceNet

# Utils
from utils import (
    SiameseWrapper,
    ContrastiveLoss,
    CACDPairDataset,
    AdaFaceLoss,
    CACDClassificationDataset_IdentitySplit,
    run_epoch_adaface,
    CACDClassificationDataset_AgeGap
)

# ===================== CONFIG =====================
CONFIG = {
    "data_dir": "data/CACD_Cropped_112",
    "weight_path": "weights/InsightFace_Pytorch%2Bmodel_ir_se50.pth",
    "img_size": 112,
    "embedding_size": 512,
    "batch_size": 24,             # ↓ giảm batch
    "epochs": 50,
    "lr_siamese": 1e-4,
    "lr_adaface": 5e-5,           # ↓ giảm LR mạnh
    "margin": 1.0,
    "pairs_per_epoch": 8000,
    "val_pairs": 2000
}

# # ================= LOAD WEIGHTS =================
# def load_backbone_weights(model, path):
#     if not os.path.exists(path):
#         print(f"⚠️ Không thấy weights tại {path}")
#         return

#     print(f"📥 Loading IR-SE50 weights from {path}")
#     state_dict = torch.load(path, map_location='cpu')

#     clean = {k.replace('module.', ''): v for k, v in state_dict.items()}
#     model.load_state_dict(clean, strict=False)
#     print("✅ Weights loaded")


# ================= MODEL =================
def get_model(args, device, num_classes=None):

    if args.model == "adaface":
        model = AdaFaceNet(num_classes, CONFIG['embedding_size'], CONFIG['weight_path'])
        #load_backbone_weights(model.backbone, CONFIG['weight_path'])
        return model.to(device)

    if args.model == "facenet":
        backbone = FaceNetBackbone(CONFIG['embedding_size'])
    elif args.model == "arcface":
        backbone = IR_SE50(CONFIG['embedding_size'])
        #load_backbone_weights(backbone, CONFIG['weight_path'])
    else:
        raise ValueError("Invalid model")

    return SiameseWrapper(backbone).to(device)


# ================= FREEZE - 3 STAGE =================

def freeze_stage(model, stage):
    """
    Stage 1: only head
    Stage 2: head + output_layer
    Stage 3: + last 2 backbone blocks
    """

    if hasattr(model, 'backbone'):
        backbone = model.backbone
    else:
        backbone = model

    # Freeze ALL
    for p in model.parameters():
        p.requires_grad = False

    # Stage 1 – ONLY HEAD
    for p in model.head.parameters():
        p.requires_grad = True

    if stage >= 2:
        if hasattr(backbone, 'output_layer'):
            for p in backbone.output_layer.parameters():
                p.requires_grad = True

    if stage >= 3 and hasattr(backbone, 'body'):
        total_blocks = len(backbone.body)
        for i in range(total_blocks - 2, total_blocks):
            for p in backbone.body[i].parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ Stage {stage}: Trainable {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")


# ================= TRAIN LOOP – SIAMESE =================
def run_epoch_siamese(model, loader, criterion, optimizer, device, is_train=True):

    model.train() if is_train else model.eval()
    total_loss = 0
    scaler = GradScaler("cuda", enabled=True)

    with torch.set_grad_enabled(is_train):
        for img1, img2, label in tqdm(loader, leave=False):

            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            if is_train:
                optimizer.zero_grad()

            with autocast("cuda"):
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

    return total_loss / len(loader)


# ================= MAIN =================
# ... (Phần import và freeze_stage giữ nguyên) ...

# Đảm bảo đã import đủ các Dataset từ utils
from utils import (
    SiameseWrapper, ContrastiveLoss, CACDPairDataset, 
    AdaFaceLoss, CACDClassificationDataset_IdentitySplit, 
    CACDClassificationDataset_AgeGap, # <--- Cần thêm cái này
    run_epoch_adaface, run_epoch_siamese
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['facenet', 'arcface', 'adaface'])
    parser.add_argument('--subset', type=float, default=0.5)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ {device} | Model: {args.model} | Subset: {args.subset}")

    # Transforms
    tf_train = transforms.Compose([
        transforms.Resize((112, 112)),
        # Strong Augmentation giúp chống Overfitting
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tf_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    # --- KHỞI TẠO DATASET BAN ĐẦU (STAGE 1 & 2: Identity Split) ---
    print("\n📦 Loading Initial Dataset (Identity Split)...")
    
    if args.model == 'adaface':
        # Dataset Phân loại (Identity Split)
        ds_train = CACDClassificationDataset_IdentitySplit(
            root_dir=CONFIG["data_dir"],
            transform=tf_train,
            subset_ratio=args.subset,
            val_split=args.split,        
            mode="train",
            min_images_per_id=5          
        )

        ds_val = CACDClassificationDataset_IdentitySplit(
            root_dir=CONFIG["data_dir"],
            transform=tf_val,
            subset_ratio=args.subset,
            val_split=args.split,        
            mode="val",
            min_images_per_id=5          
        )
        
        num_classes = len(ds_train.classes)
        print(f"🎯 Classes: {num_classes}")
        
        model = get_model(args, device, num_classes)
        criterion = AdaFaceLoss(label_smoothing=0.1).to(device) # Thêm label smoothing
        runner = run_epoch_adaface
        
    else:
        # Siamese setup
        ds_train = CACDPairDataset(CONFIG['data_dir'], tf_train, args.subset, args.split, 'train')
        ds_val = CACDPairDataset(CONFIG['data_dir'], tf_val, args.subset, args.split, 'val')
        model = get_model(args, device)
        criterion = ContrastiveLoss().to(device)
        runner = run_epoch_siamese

    # DataLoader
    train_loader = DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    # Resume
    if args.resume and os.path.exists(args.resume):
        try:
            model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
            print("✅ Resumed checkpoint.")
        except Exception as e:
            print(f"⚠️ Lỗi load checkpoint: {e}")

    # --- BIẾN ĐIỀU KHIỂN ---
    optimizer = None
    scheduler = None
    dataset_switched = False 

    print("\n🚀 START TRAINING WITH PROGRESSIVE UNFREEZING...")
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        
        # ==========================================
        # 1. QUẢN LÝ DATASET & STAGE
        # ==========================================
        current_stage = 0
        lr = 0.001
        
        # Giai đoạn 1 (Ep 1-5): Warm-up Head
        if epoch <= 5:
            current_stage = 1
            lr = 0.0001 
            
        # Giai đoạn 2 (Ep 6-15): Mở Output Layer
        elif 6 <= epoch <= 15:
            current_stage = 2
            lr = 0.00002
            
        # Giai đoạn 3 (Ep 16+): Full Fine-tune | Dataset Khó
        else:
            current_stage = 3
            lr = 0.00005 
            
            # --- CHUYỂN ĐỔI DATASET ---
            if args.model == 'adaface' and not dataset_switched:
                print("\n🔄 SWITCHING TO HARD DATASET (Age-Gap Mining)...")
                # Thay min_gap=3 bằng 10 để tăng độ khó thực sự
                ds_train = CACDClassificationDataset_AgeGap(
                    CONFIG['data_dir'], tf_train, args.subset, args.split, 'train', 
                    min_gap=5, age_mode="both"
                )
                train_loader = DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
                dataset_switched = True
                print(f"✅ Dataset switched! New size: {len(ds_train)}")

        # ==========================================
        # 2. ÁP DỤNG FREEZE & OPTIMIZER
        # ==========================================
        if epoch in [1, 6, 16]: 
            print(f"\n--- Epoch {epoch}: Configuring Stage {current_stage} (LR={lr}) ---")
            
            freeze_stage(model, current_stage)
            
            params_to_update = filter(lambda p: p.requires_grad, model.parameters())
            
            if args.model == 'adaface':
                wd = 1e-3 if current_stage < 3 else 5e-4 
                optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9, weight_decay=wd)
            else:
                optimizer = optim.Adam(params_to_update, lr=lr)
                
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(CONFIG['epochs'] - epoch + 1))

        # ==========================================
        # 3. TRAINING LOOP
        # ==========================================
        if args.model == 'adaface':
            t_loss, t_acc = runner(model, train_loader, criterion, optimizer, device, True)
            print(f"Ep {epoch} | Train Loss: {t_loss:.4f} | Acc: {t_acc:.2f}%")
        else:
            t_loss = runner(model, train_loader, criterion, optimizer, device, True)
            print(f"Ep {epoch} | Train Loss: {t_loss:.4f}")

        # ==========================================
        # 4. VALIDATION
        # ==========================================
        if args.model == 'adaface':
            v_loss, v_acc = runner(model, val_loader, criterion, optimizer, device, False)
            print(f"   >> Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")
        else:
            v_loss = runner(model, val_loader, criterion, optimizer, device, False)
            print(f"   >> Val Loss: {v_loss:.4f}")

        # ==========================================
        # 5. SAVE MODEL
        # ==========================================
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model}_best.pth"))
            print("   🔥 Best Model Saved")
        
        if scheduler:
            scheduler.step()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model}_ep{epoch}.pth"))

    print("\n✅ TRAINING COMPLETED.")

def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='adaface')
    parser.add_argument('--subset', type=float, default=0.5) 
    parser.add_argument('--resume', type=str, required=True, help="Path to Epoch 7 checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 STARTING STAGE 3 (HARD MINING) DIRECTLY | Resume: {args.resume}")

    # Config cho Stage 3
    LR_STAGE_3 = 1e-5  # Rất nhỏ, an toàn
    BATCH_SIZE = 64    # Giữ nguyên như cũ
    EPOCHS = 20        # Train thêm 20 epoch nữa là đủ

    # Transforms
    tf_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tf_val = transforms.Compose([
        transforms.Resize((112, 112)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # --- 1. DATASET: DÙNG HARD MINING NGAY LẬP TỨC ---
    print("\n📦 Loading Hard Mining Dataset (Age Gap > 10)...")
    
    # Train trên tập khó
    ds_train = CACDClassificationDataset_IdentitySplit(
        CONFIG['data_dir'], tf_train, args.subset, 0.2, 'train', 
        min_images_per_id=5 # Lọc kỹ hơn chút
    )
    
    # Val vẫn trên tập Identity Split (để so sánh chuẩn)
    ds_val = CACDClassificationDataset_IdentitySplit(
        CONFIG['data_dir'], tf_val, args.subset, 0.2, 'val', min_images_per_id=5
    )
    
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- 2. MODEL & RESUME ---
    # Lưu ý: Hard Mining dataset sẽ lọc bớt class, nên số class có thể ít hơn Identity Split
    # Nhưng ta phải init model với số class CŨ (của file checkpoint) để load được weight
    # Mẹo: Init đại 2000 class (hoặc số class lúc train Identity), load weight, 
    # phần thừa ở Head sẽ không được update (không sao cả).
    num_classes_dummy = 800 
    
    model = get_model(args, device, num_classes_dummy)
    
    if os.path.exists(args.resume):
        print(f"📥 Loading weights from {args.resume}...")
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError("Phải cung cấp file checkpoint tốt (Epoch 7) để chạy Stage 3!")

    # --- 3. FREEZE & OPTIMIZER ---
    print("\n❄️ Configuring Stage 3 (Unfreeze Last Blocks)...")
    freeze_stage(model, stage=3) # Mở khóa Layer 4, Output, Head
    
    # Lấy params cần train
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    
    # Optimizer LR nhỏ
    optimizer = optim.SGD(params_to_update, lr=LR_STAGE_3, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion = AdaFaceLoss(label_smoothing=0.1).to(device)
    
    # --- 4. TRAINING LOOP ---
    best_val_loss = float('inf')
    
    # Cần định nghĩa lại runner để chèn freeze_bn vào
    def runner_stage3(model, loader, criterion, optimizer, device, is_train):
        model.train() if is_train else model.eval()
        # QUAN TRỌNG: Freeze BN ngay cả khi train
        if is_train:
            freeze_bn(model)
            
        total_loss = 0.0
        correct = 0; total = 0
        scaler = GradScaler('cuda', enabled=True)
        
        desc = "HardTrain" if is_train else "Val"
        with torch.set_grad_enabled(is_train):
            pbar = tqdm(loader, desc=desc, leave=False)
            for img, label in pbar:
                img, label = img.to(device), label.to(device)
                
                # Check label range (phòng hờ label của tập mới vượt quá num_classes_dummy)
                # Nếu label >= num_classes_dummy, bỏ qua (để tránh lỗi)
                if label.max() >= num_classes_dummy: continue

                if is_train: optimizer.zero_grad()
                
                with autocast('cuda', enabled=False): # Safe float32
                    cosine, norms = model(img, label)
                    loss, logits = criterion(cosine, norms, label)
                
                if is_train:
                    if torch.isnan(loss): continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    scaler.step(optimizer)
                    scaler.update()
                
                _, preds = torch.max(logits, 1)
                correct += (preds == label).sum().item()
                total += label.size(0)
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        return total_loss / len(loader), (correct / total * 100) if total else 0

    print("\n🔥 STARTING HARD MINING FINE-TUNING...")
    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = runner_stage3(model, train_loader, criterion, optimizer, device, True)
        print(f"Ep {epoch} | Hard Loss: {t_loss:.4f} | Hard Acc: {t_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        v_loss, v_acc = runner_stage3(model, val_loader, criterion, optimizer, device, False)
        print(f"   >> Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join("checkpoints", f"adaface_stage3_best.pth"))
            print("   🔥 Best Stage 3 Model Saved")
            
        scheduler.step()

def freeze_bn(model):
    """
    Đóng băng các lớp Batch Normalization.
    Chuyển chúng sang chế độ eval() để không update running mean/var.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    main1()
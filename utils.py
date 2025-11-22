import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from tqdm import tqdm

# Cho phép load ảnh bị lỗi nhẹ
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 1. SIAMESE WRAPPER (Dùng chung cho cả 3 Model)
# ==========================================
class SiameseWrapper(nn.Module):
    def __init__(self, backbone_model):
        super(SiameseWrapper, self).__init__()
        self.backbone = backbone_model
        
    def forward_one(self, x):
        x = self.backbone(x)
        # Quan trọng: Normalize vector về mặt cầu đơn vị
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_one(input1)
        out2 = self.forward_one(input2)
        return out1, out2

# ==========================================
# 2. CONTRASTIVE LOSS
# ==========================================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Label: 0 (Same), 1 (Diff) -> Lưu ý Dataset trả về 0/1
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# ==========================================
# 3. DATASET (Với Age-Gap Mining)
# ==========================================
class CACDPairDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset_ratio=1.0, num_pairs=5000):
        self.root_dir = root_dir
        self.transform = transform
        self.num_pairs = num_pairs
        self.data_map = {} 
        self.classes = []
        
        if not os.path.exists(root_dir):
            raise RuntimeError(f"❌ Không tìm thấy thư mục: {root_dir}")

        # Lấy danh sách thư mục
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Cắt giảm dữ liệu nếu cần (subset)
        num_take = int(len(all_classes) * subset_ratio)
        self.classes = all_classes[:num_take]
        
        print(f"🔄 Đang index {len(self.classes)} người...")
        
        valid_indices = []
        for idx, cls_name in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls_name)
            temp_imgs = []
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        age = int(img_name.split('_')[0]) # Parse tuổi
                    except:
                        age = -1
                    temp_imgs.append((os.path.join(cls_folder, img_name), age))
            
            if len(temp_imgs) >= 2:
                self.data_map[idx] = temp_imgs
                valid_indices.append(idx)
        
        self.valid_indices = valid_indices
        print(f"✅ Đã load xong index cho {len(self.valid_indices)} người hợp lệ.")

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, index):
        # 50% Same, 50% Diff
        should_get_same = random.randint(0, 1)
        
        idx1 = random.choice(self.valid_indices)
        img1_info = random.choice(self.data_map[idx1])
        
        if should_get_same: 
            # --- POSITIVE PAIR (HARD MINING) ---
            candidates = self.data_map[idx1]
            # Tìm ảnh lệch tuổi > 10
            hard_candidates = [x for x in candidates if abs(x[1] - img1_info[1]) > 10]
            
            if hard_candidates:
                img2_info = random.choice(hard_candidates)
            else:
                img2_info = random.choice(candidates)
                # Tránh lấy trùng ảnh chính nó
                while img2_info[0] == img1_info[0] and len(candidates) > 1:
                    img2_info = random.choice(candidates)
            
            label = torch.tensor([0], dtype=torch.float32)
            
        else:
            # --- NEGATIVE PAIR ---
            idx2 = random.choice(self.valid_indices)
            while idx2 == idx1: idx2 = random.choice(self.valid_indices)
            img2_info = random.choice(self.data_map[idx2])
            label = torch.tensor([1], dtype=torch.float32)

        try:
            img1 = Image.open(img1_info[0]).convert('RGB')
            img2 = Image.open(img2_info[0]).convert('RGB')
            img1.load(); img2.load() # Check corrupt
        except:
            return self.__getitem__(random.randint(0, self.num_pairs - 1))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label

# ==========================================
# 4. GENERIC TRAIN ENGINE
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx}", unit="batch")
    for img1, img2, label in pbar:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, label)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(loader)
    return avg_loss
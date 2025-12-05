import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.amp import GradScaler, autocast

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# PHẦN 1: CORE MODULES & LOSSES
# ============================================================

class SiameseWrapper(nn.Module):
    def __init__(self, backbone_model):
        super(SiameseWrapper, self).__init__()
        self.backbone = backbone_model

    def forward_one(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, input1, input2):
        return self.forward_one(input1), self.forward_one(input2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean(
            (1-label) * torch.pow(dist, 2) +
            (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss

# ============================================================
# PHẦN 2: DATASETS
# ============================================================

class CACDPairDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset_ratio=0.5, val_split=0.2,
                 mode='train', num_pairs=5000):

        self.root_dir = root_dir
        self.transform = transform
        self.num_pairs = num_pairs
        self.data_map = {}
        self.valid_indices = []

        if not os.path.exists(root_dir):
            raise RuntimeError(f"❌ Không tìm thấy: {root_dir}")

        all_classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])

        subset_len = int(len(all_classes) * subset_ratio)
        subset_classes = all_classes[:subset_len]

        split_idx = int(len(subset_classes) * (1 - val_split))
        self.classes = subset_classes[:split_idx] if mode == 'train' else subset_classes[split_idx:]

        print(f"[PAIR-{mode.upper()}] {len(self.classes)} classes")

        for idx, cls_name in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls_name)
            temp_imgs = []
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.jpg', '.png')):
                    try:
                        age = int(img_name.split('_')[0])
                    except:
                        age = -1
                    temp_imgs.append((os.path.join(cls_folder, img_name), age))

            if len(temp_imgs) >= 2:
                self.data_map[idx] = temp_imgs
                self.valid_indices.append(idx)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, index):

        should_same = random.choice([True, False])
        idx1 = random.choice(self.valid_indices)
        img1_inf = random.choice(self.data_map[idx1])

        if should_same:
            cands = self.data_map[idx1]
            hard = [x for x in cands if abs(x[1] - img1_inf[1]) > 10]
            img2_inf = random.choice(hard) if hard else random.choice(cands)
            label = torch.tensor([0.0])
        else:
            idx2 = random.choice(self.valid_indices)
            while idx2 == idx1:
                idx2 = random.choice(self.valid_indices)
            img2_inf = random.choice(self.data_map[idx2])
            label = torch.tensor([1.0])

        try:
            img1 = Image.open(img1_inf[0]).convert('RGB')
            img2 = Image.open(img2_inf[0]).convert('RGB')

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, label

        except:
            return self.__getitem__(0)


class CACDClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset_ratio=0.5,
                 val_split=0.2, mode='train'):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        all_classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])

        subset_len = int(len(all_classes) * subset_ratio)
        subset_classes = all_classes[:subset_len]

        split_idx = int(len(subset_classes) * (1 - val_split))
        self.classes = subset_classes[:split_idx] if mode == 'train' else subset_classes[split_idx:]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        print(f"[CLASS-{mode.upper()}] {len(self.classes)} classes")

        for cls in self.classes:
            path = os.path.join(root_dir, cls)
            for f in os.listdir(path):
                if f.lower().endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(path, f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

        except:
            return self.__getitem__(0)

class CACDClassificationDataset_IdentitySplit(Dataset):
    """
    Dataset chia theo Identity chuẩn: 
    - Train trên tập người A.
    - Val trên tập người B (người lạ).
    Dùng để train nền tảng (Base Training).
    """
    def __init__(self, root_dir, transform=None, subset_ratio=0.5, val_split=0.2, 
                 mode="train", min_images_per_id=3, shuffle=True):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        if not os.path.exists(root_dir):
            raise RuntimeError(f"❌ Không tìm thấy thư mục: {root_dir}")

        # 1. Lấy danh sách toàn bộ thư mục (Identities)
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # 2. Shuffle trước khi lọc để đảm bảo tính ngẫu nhiên nếu dùng subset
        if shuffle:
            random.seed(42) # Cố định seed để Train/Val không bị lệch nhau mỗi lần chạy lại
            random.shuffle(all_classes)

        # 3. Lấy subset (Ví dụ dùng 50% dữ liệu)
        subset_len = int(len(all_classes) * subset_ratio)
        subset_classes = all_classes[:subset_len]

        # 4. Chia Train/Val theo Identity
        split_idx = int(len(subset_classes) * (1 - val_split))
        target_classes = subset_classes[:split_idx] if mode == "train" else subset_classes[split_idx:]
        
        # Map class name -> index (Dùng index cục bộ 0..N-1 cho AdaFace Head)
        self.classes = []
        self.class_to_idx = {}
        
        print(f"🔄 [IdentitySplit-{mode.upper()}] Scanning {len(target_classes)} candidates...")
        
        valid_idx = 0
        for cls_name in target_classes:
            cls_folder = os.path.join(root_dir, cls_name)
            
            # Lấy danh sách ảnh hợp lệ
            images = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Lọc: Phải đủ số lượng ảnh tối thiểu
            if len(images) < min_images_per_id:
                continue
                
            # Nếu đạt chuẩn thì thêm vào danh sách chính thức
            self.classes.append(cls_name)
            self.class_to_idx[cls_name] = valid_idx
            
            for img_name in images:
                self.samples.append((os.path.join(cls_folder, img_name), valid_idx))
            
            valid_idx += 1

        print(f"✅ Final: {len(self.classes)} identities, {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            return self.__getitem__(random.randint(0, len(self.samples)-1))

class CACDClassificationDataset_AgeGap(Dataset):
    """
    Dataset nâng cao: Chỉ lấy các ID có độ lệch tuổi lớn và chọn lọc ảnh cực trị (Trẻ nhất/Già nhất).
    """
    def __init__(self, root_dir, transform=None, subset_ratio=0.5, val_split=0.2, mode="train", 
                 min_gap=3, age_mode="both"): # age_mode: "young", "old", "both", "all"
        
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        if not os.path.exists(root_dir): raise RuntimeError(f"❌ Không tìm thấy: {root_dir}")

        # 1. Lấy tất cả ID
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # 2. Split ID cho Train/Val trước (Để đảm bảo Val không bị lẫn ID train)
        subset_len = int(len(all_classes) * subset_ratio)
        subset_classes = all_classes[:subset_len]
        
        split_idx = int(len(subset_classes) * (1 - val_split))
        target_classes = subset_classes[:split_idx] if mode == "train" else subset_classes[split_idx:]
        
        # Map class name -> index (Dùng chung index toàn cục để nhất quán nếu cần)
        # Lưu ý: Ở đây ta map lại index từ 0 -> len(target_classes) - 1 cho gọn
        self.class_to_idx = {cls: i for i, cls in enumerate(target_classes)}
        self.classes = target_classes

        print(f"🔄 [AdvancedDataset-{mode.upper()}] Scanning {len(self.classes)} IDs with Gap > {min_gap}...")
        
        valid_id_count = 0
        
        for cls_name in tqdm(self.classes, desc="Filtering"):
            folder = os.path.join(root_dir, cls_name)
            img_age_list = []
            
            # Quét ảnh và tuổi
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.png')):
                    try:
                        age = int(f.split('_')[0])
                        img_age_list.append((os.path.join(folder, f), age))
                    except: pass
            
            if len(img_age_list) < 2: continue

            # Lọc theo Age Gap
            ages = [x[1] for x in img_age_list]
            if (max(ages) - min(ages)) < min_gap:
                continue # Bỏ qua người này vì không đủ thách thức
            
            valid_id_count += 1
            
            # Chọn lọc ảnh theo mode
            if age_mode == "all":
                final_imgs = [x[0] for x in img_age_list]
            else:
                mid_age = (min(ages) + max(ages)) / 2
                # Chia nhóm
                young_group = [x[0] for x in img_age_list if x[1] < mid_age]
                old_group = [x[0] for x in img_age_list if x[1] >= mid_age]
                
                if age_mode == "young": final_imgs = young_group
                elif age_mode == "old": final_imgs = old_group
                elif age_mode == "both": final_imgs = young_group + old_group # Lấy 2 cực, bỏ khúc giữa (nếu có logic lọc kỹ hơn)
                # Thực tế code của bạn lấy young < mid - gap/2 và old > mid + gap/2 là chuẩn bài Hard Mining
                # Ta sẽ dùng logic đó:
                else: # Default logic
                    final_imgs = [x[0] for x in img_age_list]

            # Add vào list mẫu
            label = self.class_to_idx[cls_name]
            for p in final_imgs:
                self.samples.append((p, label))

        print(f"✅ Filtered: {valid_id_count}/{len(self.classes)} IDs thỏa mãn gap. Tổng ảnh: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, label
        except: return self.__getitem__(random.randint(0, len(self.samples)-1))
# ============================================================
# PHẦN 3: TRAINING LOOPS (FIXED)
# ============================================================

def run_epoch_siamese(model, loader, criterion, optimizer, device, is_train=True):

    model.train() if is_train else model.eval()
    total_loss = 0.0

    scaler = GradScaler(enabled=(device.type == "cuda"))
    desc = "Train" if is_train else "Val"

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, desc=f"Siamese {desc}", leave=False)

        for img1, img2, label in pbar:

            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            label = label.to(device).view(-1, 1)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=str(device), enabled=(device.type == "cuda")):
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)

            if is_train and not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def run_epoch_adaface(model, loader, criterion, optimizer, device, is_train=True):

    model.train() if is_train else model.eval()
    if is_train:
        freeze_bn(model)
    total_loss = 0
    correct = 0
    total = 0

    scaler = GradScaler(enabled=(device.type == "cuda"))
    desc = "Train" if is_train else "Val"

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, desc=f"AdaFace {desc}", leave=False)

        for img, label in pbar:

            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            cosine, norms = model(img, label)
            loss, logits = criterion(cosine, norms, label)

            if is_train and not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                _, preds = torch.max(logits, 1)
                correct += (preds == label).sum().item()
                total += label.size(0)

            total_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(100 * correct / max(1, total)):.2f}%"
            })

    return total_loss / len(loader), (correct / total) * 100

class TemporalPairDataset(Dataset):
    """
    Temporal Pair dataset for CACD
    Folder structure:
        root/
            Person_1/
                29_50_img_001.jpg
                40_50_img_002.jpg
            Person_2/
                33_xx_xxx.jpg
    """

    def __init__(
        self,
        root_dir,
        min_age_gap=2,
        max_age_gap=30,
        transform=None,
        mode="train",      # train | val | test
        negative_ratio=1.0 # negative pairs = positive * ratio
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.min_age_gap = min_age_gap
        self.max_age_gap = max_age_gap
        self.mode = mode
        self.negative_ratio = negative_ratio

        self.identity_images = self._load_identities()
        self.identities = list(self.identity_images.keys())

        self.positive_pairs = self._build_positive_pairs()
        self.negative_pairs = self._build_negative_pairs()

        self.pairs = self.positive_pairs + self.negative_pairs
        random.shuffle(self.pairs)

        print(f"[{mode}] Positive pairs: {len(self.positive_pairs)}")
        print(f"[{mode}] Negative pairs: {len(self.negative_pairs)}")
        print(f"[{mode}] Total pairs: {len(self.pairs)}")

    # -----------------------------
    # LOAD IMAGES BY IDENTITY
    # -----------------------------
    def _load_identities(self):
        identity_images = {}

        for person in os.listdir(self.root_dir):
            person_path = os.path.join(self.root_dir, person)
            if not os.path.isdir(person_path):
                continue

            imgs = []
            for file in os.listdir(person_path):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(person_path, file)

                    try:
                        # age = first number before "_"
                        age = int(file.split("_")[0])
                        imgs.append((img_path, age))
                    except:
                        continue

            if len(imgs) >= 2:
                imgs.sort(key=lambda x: x[1])  # sort by age
                identity_images[person] = imgs

        return identity_images

    # -----------------------------
    # BUILD POSITIVE (same person, different age)
    # -----------------------------
    def _build_positive_pairs(self):
        pairs = []

        for person, images in self.identity_images.items():
            n = len(images)

            for i in range(n):
                for j in range(i + 1, n):
                    img1, age1 = images[i]
                    img2, age2 = images[j]

                    age_gap = abs(age1 - age2)

                    if self.min_age_gap <= age_gap <= self.max_age_gap:
                        pairs.append((img1, img2, 1, age_gap))  # label = 1

        return pairs

    # -----------------------------
    # BUILD NEGATIVE (different persons)
    # -----------------------------
    def _build_negative_pairs(self):
        neg_pairs = []
        num_neg = int(len(self.positive_pairs) * self.negative_ratio)

        attempts = 0
        while len(neg_pairs) < num_neg and attempts < num_neg * 10:
            id1, id2 = random.sample(self.identities, 2)

            img1, age1 = random.choice(self.identity_images[id1])
            img2, age2 = random.choice(self.identity_images[id2])

            age_gap = abs(age1 - age2)

            neg_pairs.append((img1, img2, 0, age_gap))
            attempts += 1

        return neg_pairs

    # -----------------------------
    # LOAD IMAGE
    # -----------------------------
    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    # -----------------------------
    # GET ITEM
    # -----------------------------
    def __getitem__(self, index):
        img1_path, img2_path, label, age_gap = self.pairs[index]

        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        return {
            "img1": img1,
            "img2": img2,
            "label": label,
            "age_gap": age_gap,
            "path1": img1_path,
            "path2": img2_path
        }

    def __len__(self):
        return len(self.pairs)
    
def freeze_bn(model):
    """
    Đóng băng các lớp Batch Normalization.
    Chuyển chúng sang chế độ eval() để không update running mean/var.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            # Tùy chọn: Khóa cả affine parameters (gamma, beta) nếu muốn cực kỳ an toàn
            # for param in module.parameters():
            #     param.requires_grad = False
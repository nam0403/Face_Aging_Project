import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from dataset.dataset import AgeGapDataset
import net
from tqdm import tqdm
# ================= CẤU HÌNH =================
CONFIG = {
    'pretrained_path': 'weights/adaface.ckpt',
    'train_csv_path': 'cacd_train.csv', 
    'val_csv_path': 'cacd_val.csv',
    'root_dir': 'data/CACD_Cropped_112',
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 200,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'triplet_margin': 0.2,
    'warmup_epochs': 5,
    'unfreeze_blocks': 2,
    'eval_every': 1,
    'lr-head' : 1e-2
}
# Trọng số Loss (Đã điều chỉnh hợp lý hơn)
W_ADA = 1.0
W_TRIP = 2.0        # Tăng lên để ép học Age-Invariant
W_PRES = 0.5        # Tăng nhẹ để giữ ổn định
W_DIV = 0.1         # Giữ diversity

class AdaFaceHead(nn.Module):
    def __init__(self, embedding_size, num_classes, m=0.4, h=0.333, s=64., t_alpha=0.01):
        super(AdaFaceHead, self).__init__()
        self.s = s
        self.m = m
        self.epsilon = 1e-3
        self.h = h
        self.t_alpha = t_alpha

        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, embeddings, labels, norms):
        kernel_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, kernel_norm)
        cosine = torch.clamp(cosine, -1+self.epsilon, 1-self.epsilon) 

        # --- AdaFace Logic ---
        safe_norms = torch.clamp(norms, min=0.001, max=100).clone().detach()
        batch_avg_norm = safe_norms.mean()
        norm_ratio = safe_norms / (batch_avg_norm + 1e-6)
        adaptive_m = self.m * norm_ratio
        adaptive_m = torch.clamp(adaptive_m, 0.0, 1.0)

        labels = labels.to(cosine.device)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        theta = torch.acos(cosine)
        theta_m = theta + adaptive_m * one_hot
        output = self.s * torch.cos(theta_m)
        
        return output
# ================= TRIPLET LOSS (Giữ nguyên) =================
class AdaptiveTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative, age_gap):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        adaptive_margin = self.margin * (1.0 + age_gap * 0.3)
        losses = F.relu(pos_dist - neg_dist + adaptive_margin)
        hard_mask = (losses > 0.0)
        if hard_mask.sum() > 0:
            return losses[hard_mask].mean()
        else:
            return losses.mean()

# ================= REGULARIZATION (Giữ nguyên) =================
def diversity_loss(embeddings):
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized, normalized.t())
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, 0)
    avg_similarity = similarity_matrix.abs().sum() / (similarity_matrix.numel() - similarity_matrix.size(0))
    return avg_similarity

class FeaturePreservationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
    def forward(self, current_emb, original_emb):
        cos_sim = F.cosine_similarity(current_emb, original_emb, dim=1)
        return (1.0 - cos_sim).mean() * self.alpha

# ================= LOAD MODEL (Giữ nguyên logic) =================
def load_pretrained_model(model, weight_path):
    print(f"🔄 Loading weights from {weight_path} ...")
    try:
        statedict = torch.load(weight_path, map_location=CONFIG['device'], weights_only=True)['state_dict']
    except Exception:
        print(f"⚠️  weights_only=True failed, trying weights_only=False...")
        statedict = torch.load(weight_path, map_location=CONFIG['device'], weights_only=False)['state_dict']

    new_state = {k[6:]: v for k, v in statedict.items() if k.startswith('model.')}
    model.load_state_dict(new_state, strict=False)
    return model

def selective_unfreeze(model, num_blocks=2):
    print(f"❄️ Freezing backbone, unfreezing last {num_blocks} blocks...")
    for param in model.parameters():
        param.requires_grad = False
    
    unfrozen = []
    total_blocks = 0
    for name, _ in model.named_parameters():
        if "body" in name.split('.'):
            parts = name.split('.')
            try:
                idx = parts.index("body")
                block = int(parts[idx + 1])
                total_blocks = max(total_blocks, block)
            except: pass
    
    threshold = total_blocks - num_blocks + 1
    for name, param in model.named_parameters():
        parts = name.split('.')
        if "output_layer" in name:
            param.requires_grad = True
            unfrozen.append(name)
            continue
        if "body" in parts:
            try:
                idx = parts.index("body")
                block = int(parts[idx + 1])
                if block >= threshold:
                    param.requires_grad = True
                    unfrozen.append(name)
            except: pass
            
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)")
    return unfrozen

def check_collapse(model, loader, device, num_batches=5):
    """Check if model has collapsed (all embeddings similar)"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i, (anchor, _, _, _,_) in enumerate(loader):
            if i >= num_batches:
                break
                
            anchor = anchor.to(device)
            emb = model(anchor)
            
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Compute pairwise similarities
    sim_matrix = torch.mm(all_embeddings, all_embeddings.t())
    
    # Remove diagonal
    mask = torch.eye(sim_matrix.size(0), device=device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, 0)
    
    avg_sim = sim_matrix.abs().sum() / (sim_matrix.numel() - sim_matrix.size(0))
    std_emb = all_embeddings.std(dim=0).mean()
    
    collapsed = avg_sim > 0.9 or std_emb < 0.01
    
    return {
        'collapsed': collapsed,
        'avg_similarity': avg_sim.item(),
        'embedding_std': std_emb.item()
    }

# ================= VALIDATION FUNCTION (MỚI) =================
def validate(model, val_loader, device, triplet_criterion):
    model.eval()
    val_loss = 0
    val_pos_sim = 0
    val_neg_sim = 0
    count = 0
    
    # Bọc loader bằng tqdm
    pbar = tqdm(val_loader, desc="⏳ Validating", leave=False)
    
    with torch.no_grad():
        for anchor, pos, neg, gap, label in pbar:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            gap = gap.to(device)

            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)

            if isinstance(emb_a, (tuple, list)):
                emb_a, emb_p, emb_n = emb_a[0], emb_p[0], emb_n[0]

            emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_p = F.normalize(emb_p, p=2, dim=1)
            emb_n = F.normalize(emb_n, p=2, dim=1)

            loss = triplet_criterion(emb_a, emb_p, emb_n, gap)
            val_loss += loss.item()

            val_pos_sim += F.cosine_similarity(emb_a, emb_p).mean().item()
            val_neg_sim += F.cosine_similarity(emb_a, emb_n).mean().item()
            count += 1
            
    return {
        'loss': val_loss / count,
        'pos_sim': val_pos_sim / count,
        'neg_sim': val_neg_sim / count,
        'separation': (val_pos_sim / count) - (val_neg_sim / count)
    }

# ================ MAIN ========================
def main():
    device = torch.device(CONFIG['device'])
    print("\n🏗️ Building IR-SE-101...")
    model = net.build_model("ir_101")
    model = load_pretrained_model(model, CONFIG['pretrained_path'])
    
    original_model = net.build_model("ir_101")
    original_model = load_pretrained_model(original_model, CONFIG['pretrained_path'])
    original_model.to(device).eval()
    for param in original_model.parameters(): param.requires_grad = False
    
    model.to(device)
    _ = selective_unfreeze(model, CONFIG['unfreeze_blocks'])

    # ================= DATA LOADING (THAY ĐỔI) =================
    # Train Transform: Có Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Val Transform: Chỉ resize, giữ nguyên chất lượng gốc
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    print(f"📂 Loading Train Set: {CONFIG['train_csv_path']}")
    train_dataset = AgeGapDataset(CONFIG['train_csv_path'], CONFIG['root_dir'], train_transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    print(f"📂 Loading Val Set: {CONFIG['val_csv_path']}")
    val_dataset = AgeGapDataset(CONFIG['val_csv_path'], CONFIG['root_dir'], val_transform)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True) # Shuffle False cho Val

    # ================= LOSS & OPTIMIZER =================
    num_classes = len(train_dataset.ids) 

    head = AdaFaceHead(embedding_size=512, num_classes=num_classes).to(device)

    optimizer = optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': CONFIG['lr']},
        {'params': head.parameters(), 'lr': CONFIG['lr-head']} 
    ], weight_decay=1e-4)

    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.01).to(device)
    triplet_criterion = AdaptiveTripletLoss(margin=CONFIG['triplet_margin']).to(device)
    preservation_criterion = FeaturePreservationLoss(alpha=0.3).to(device)

    def get_lr_multiplier(epoch):
        if epoch < CONFIG['warmup_epochs']: return (epoch + 1) / CONFIG['warmup_epochs']
        return 0.5 * (1 + np.cos(np.pi * (epoch - CONFIG['warmup_epochs']) / (CONFIG['epochs'] - CONFIG['warmup_epochs'])))

    def set_bn_eval(module):
        if isinstance(module, nn.BatchNorm2d): module.eval()

    # ================= TRAIN LOOP =================
    print("\n🚀 Training started")
    best_val_separation = -1.0 

    for epoch in range(CONFIG['epochs']):

        model.train()
        model.apply(set_bn_eval)
        
        lr_mult = get_lr_multiplier(epoch)
        optimizer.param_groups[0]['lr'] = CONFIG['lr'] * lr_mult
        optimizer.param_groups[1]['lr'] = CONFIG['lr-head'] * lr_mult
        print(f"Epoch {epoch+1}: Backbone LR={optimizer.param_groups[0]['lr']:.2e}, Head LR={optimizer.param_groups[1]['lr']:.2e}")
        train_loss = 0
        batch_count = 0
        collapse_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=True)

        for idx, (anchor, pos, neg, gap, label) in enumerate(pbar):
            anchor, pos, neg, gap, label = anchor.to(device), pos.to(device), neg.to(device), gap.to(device), label.to(device)

            optimizer.zero_grad()

            emb_a_norm, norm_a, feat_a = model(anchor)
            emb_p_norm, _, feat_p = model(pos)
            emb_n_norm, _, feat_n = model(neg)

            logits = head(emb_a_norm, label, norm_a)
            
            loss_ada = criterion_ce(logits, label)

            loss_triplet = triplet_criterion(emb_a_norm, emb_p_norm, emb_n_norm, gap)

            with torch.no_grad():
                orig = original_model(anchor)
                if isinstance(orig, (tuple, list)): orig = orig[0]
                orig = F.normalize(orig, p=2, dim=1)

            loss_pres = preservation_criterion(emb_a_norm, orig) * W_PRES

            loss_div = diversity_loss(emb_a_norm) * W_DIV
            
            loss = loss_ada + 2.0 * loss_triplet + loss_pres + loss_div

            optimizer.zero_grad() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 0.5)
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{lr_mult*CONFIG['lr']:.1e}"})

        avg_train_loss = train_loss / batch_count
        
        # --- VALIDATION PHASE ---
        val_metrics = validate(model, val_loader, device, triplet_criterion)
        print(f"\n📊 Summary Epoch {epoch+1}: "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Sep: {val_metrics['separation']:.4f} (Pos: {val_metrics['pos_sim']:.2f} - Neg: {val_metrics['neg_sim']:.2f})")

        # --- CHECK COLLAPSE & SAVE ---
        if (epoch + 1) % CONFIG['eval_every'] == 0:
            collapse_info = check_collapse(model, train_loader, device) 
            if collapse_info['collapsed']:
                print(f"⚠️  COLLAPSE WARNING! Avg Sim: {collapse_info['avg_similarity']:.4f}")
                collapse_count += 1
                if collapse_count >= 2: break
            else:
                collapse_count = 0

        if val_metrics['separation'] > best_val_separation:
            best_val_separation = val_metrics['separation']
            save_path = f"ir_se_101_temporal_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': CONFIG
            }, save_path)
            print(f"💾 Saved Best Model (Sep: {best_val_separation:.4f})")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"ir_se_101_temporal_ep{epoch+1}.pth")

    print("\n✅ Training completed!")

if __name__ == "__main__":
    main()
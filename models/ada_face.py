import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_ir_se50 import IR_SE50

# =================================================================
# 1. INTERNAL LOSS MODULE (Tích hợp sẵn bên trong)
# =================================================================
class AdaFaceLossInternal(nn.Module):
    def __init__(self, s=64.0, m=0.4, h=0.333, label_smoothing=0.1):
        super(AdaFaceLossInternal, self).__init__()
        self.s = s
        self.m = m
        self.h = h
        self.eps = 1e-4
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, cosine, norms, label):
        # --- SAFE OPS (Chống NaN) ---
        safe_norms = torch.clamp(norms, min=0.001, max=100)
        
        # Tính thống kê trực tiếp trên batch (Không dùng buffer cũ để tránh lỗi)
        mean = safe_norms.mean()
        std = safe_norms.std()

        # Tính Adaptive Margin
        # QUAN TRỌNG: Thêm .detach() để ngắt gradient, ngăn bùng nổ số
        margin_scaler = (safe_norms - mean) / (std + self.eps)
        margin_scaler = torch.clamp(margin_scaler * self.h, -1, 1).detach()

        # Tính Cosine an toàn
        safe_cosine = torch.clamp(cosine, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(safe_cosine)
        
        # Cộng Margin
        target_theta = theta + (self.m * margin_scaler)
        final_cosine = torch.cos(target_theta)

        # Tạo One-hot và tính Logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logits = (one_hot * final_cosine) + ((1.0 - one_hot) * safe_cosine)
        logits *= self.s

        # Trả về Loss và Logits
        loss = self.ce_loss(logits, label)
        return loss, logits

# =================================================================
# 2. MAIN MODEL (Tích hợp tất cả)
# =================================================================
class AdaFaceNet(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(AdaFaceNet, self).__init__()
        
        # 1. Backbone
        self.backbone = IR_SE50(embedding_size)
        
        # 2. Head (Linear Layer để tính Cosine)
        # bias=False là bắt buộc cho các mô hình dạng ArcFace/AdaFace
        self.head = nn.Linear(embedding_size, num_classes, bias=False)
        
        # 3. Loss Function (Nằm ngay trong model)
        self.loss_fn = AdaFaceLossInternal(s=64.0, m=0.4, h=0.333)

    def forward(self, x, label=None):
        # --- FEATURE EXTRACTION ---
        embeddings = self.backbone(x)
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_features = embeddings / (norms + 1e-5)

        # --- TRAINING MODE (Có nhãn) ---
        if label is not None:
            # Chuẩn hóa trọng số của Head
            normalized_weights = F.normalize(self.head.weight, p=2, dim=1)
            
            # Tính Cosine Similarity (x * W^T)
            cosine = F.linear(normalized_features, normalized_weights)
            
            # Gọi hàm Loss nội bộ
            loss, logits = self.loss_fn(cosine, norms, label)
            return loss, logits

        # --- INFERENCE MODE (Không nhãn) ---
        return normalized_features
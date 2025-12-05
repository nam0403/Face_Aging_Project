import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

# =================================================================
# 1. INTERNAL LOSS MODULE (Contrastive Loss)
# =================================================================
class ContrastiveLossInternal(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLossInternal, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        label: 1 nếu khác người (Different), 0 nếu cùng người (Same)
        Lưu ý: Kiểm tra kỹ dataset của bạn quy định 0 hay 1 là giống nhau.
        Code này giả định: 
           - label=0 (Same): Kéo gần lại -> Loss = dist^2
           - label=1 (Diff): Đẩy xa ra  -> Loss = max(0, margin - dist)^2
        """
        # Tính khoảng cách Euclidean giữa 2 vector
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Công thức Contrastive Loss chuẩn
        # Nếu label 0 (Same): (1-0) * dist^2 + 0... = dist^2
        # Nếu label 1 (Diff): (1-1) * dist^2 + 1 * clamp(...) = clamp(...)^2
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive, euclidean_distance

# =================================================================
# 2. MAIN MODEL (Siamese Network)
# =================================================================
class SiameseFaceNet(nn.Module):
    def __init__(self, embedding_size=512, margin=1.0):
        super(SiameseFaceNet, self).__init__()
        
        print("🏗️ Init Siamese FaceNet (InceptionResnetV1)...")
        
        # 1. Backbone: Dùng InceptionResnetV1 chuẩn của FaceNet
        # classify=False nghĩa là chỉ lấy embedding, không lấy lớp phân loại cuối
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        
        # FaceNet gốc ra 512, nếu muốn custom size có thể thêm Linear
        # Nhưng thường InceptionResnetV1 đã ra 512 rồi.
        
        # 2. Loss Function (Nằm bên trong model)
        self.loss_fn = ContrastiveLossInternal(margin=margin)

    def forward_one(self, x):
        """Chạy 1 nhánh (dùng cho Inference hoặc nhánh con)"""
        x = self.backbone(x)
        # Quan trọng: FaceNet luôn cần L2 Normalize vector đầu ra
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2=None, label=None):
        # --- INFERENCE MODE (1 ảnh) ---
        if x2 is None:
            return self.forward_one(x1)

        # --- TRAINING MODE (2 ảnh + nhãn) ---
        # 1. Trích xuất đặc trưng cho cả 2 ảnh (Siamese: chung trọng số backbone)
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # 2. Tính Loss
        if label is not None:
            loss, dist = self.loss_fn(feat1, feat2, label)
            return loss
        
        # Trường hợp test cặp đôi nhưng không cần loss (trả về khoảng cách)
        dist = F.pairwise_distance(feat1, feat2)
        return dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.model_ir_se50 import IR_SE50

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Ma trận trọng số W (Class Centers)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Các hằng số toán học (pre-compute để nhanh hơn)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, label):
        # 1. Chuẩn hóa Weights và Inputs
        # W = W / ||W||
        W = F.normalize(self.weight)
        # x = x / ||x||
        x = F.normalize(embedding)

        # 2. Tính Cosine Theta = x * W^T
        cosine = F.linear(x, W)
        # Kẹp giá trị để tránh NaN khi tính acos/sqrt
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # 3. Tính Cosine(Theta + m) dùng công thức lượng giác
        # cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # 4. Xử lý ổn định (nếu theta + m > pi)
        if self.training:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        else:
            phi = cosine # Khi test không cần margin (hoặc giữ nguyên tùy chiến lược)

        # 5. One-hot Encoding: Chỉ cộng margin vào đúng class thật (Ground Truth)
        # Tạo mask one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Output = (Scale) * [ one_hot * phi + (1 - one_hot) * cosine ]
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class ArcFaceNet(nn.Module):
    def __init__(self, num_classes, embedding_size=512, pretrained_path=None):
        super(ArcFaceNet, self).__init__()
        
        print(f"🏗️ Init ArcFaceNet with {num_classes} classes...")
        
        # --- PHẦN 1: BACKBONE (IR-SE50) ---
        self.backbone = IR_SE50((112, 112))
        
        # Load weight backbone nếu có
        if pretrained_path:
            self.load_backbone_weights(pretrained_path)
            
        # --- PHẦN 2: HEAD (ARCFACE) ---
        self.head = ArcFaceHead(in_features=embedding_size, out_features=num_classes)

    def load_backbone_weights(self, path):
        print(f"📥 Loading Backbone Weights from {path}")
        try:
            state_dict = torch.load(path, map_location='cpu')
            clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(clean_state, strict=False)
            print("✅ Backbone Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading backbone: {e}")

    def forward(self, x, label=None):
        # 1. Trích xuất đặc trưng (Luôn chạy)
        features = self.backbone(x) # [Batch, 512]
        
        # 2. Logic rẽ nhánh Train/Eval
        if label is not None:
            # --- TRAINING MODE ---
            # Trả về Logits đã có Margin để tính CrossEntropyLoss
            logits = self.head(features, label)
            return logits
        else:
            # --- INFERENCE MODE ---
            # Trả về Normalized Embedding để tính Cosine Similarity
            return F.normalize(features, p=2, dim=1)